"""Headless fixed figure generation from validated summaries and raw runs."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from itertools import repeat
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Callable

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
import matplotlib.pyplot as plt
import numpy as np

from .experiment import (
    JobSpec,
    canonical_json,
)
from .records import RunAggregate, aggregate_rows, iter_job_rows
from .inventory import validate_report_inventory
from .provenance import ExecutionProvenance
from .stats import comparison_table_rows, load_summary
from .workflows import effective_worker_count


FIGURE_NAMES = (
    "backoff-convergence",
    "delay-cdf",
    "scaling",
    "dynamic-adaptation",
    "held-out-utility",
    "fairness-delay-airtime-tradeoff",
    "arm-heatmap",
    "ablation-forest",
)

_TRACE_SCENARIO = "trace-static-4x4"
_TRACE_LIMIT = 512
_MAIN_POLICIES = frozenset(
    {
        "random_lbt",
        "primary_db_lbt",
        "tmc_db_lbt",
        "fixed_oracle",
        "adaptive_db_lbt",
    }
)


@dataclass(frozen=True, slots=True)
class PlotRun:
    job: JobSpec
    backoff_points: tuple[tuple[int, int], ...]
    delay_samples: tuple[int, ...]
    dynamic_points: tuple[tuple[int, int], ...]
    decision_points: tuple[tuple[int, int, float | None, bool], ...]
    arm_counts: np.ndarray
    execution_provenance: ExecutionProvenance = field(
        default_factory=lambda: ExecutionProvenance(mode="baseline_builtin")
    )


class _BoundedSampler:
    def __init__(self, limit: int = _TRACE_LIMIT) -> None:
        self.limit = limit
        self.stride = 1
        self.items: list[tuple[int, object]] = []

    def add(self, index: int, value: object) -> None:
        if index % self.stride == 0:
            self.items.append((index, value))
        if len(self.items) > self.limit:
            self.stride *= 2
            self.items = [
                item for item in self.items if item[0] % self.stride == 0
            ]

    def values(self) -> tuple[object, ...]:
        return tuple(value for _, value in self.items)


class _PlotAccumulator:
    def __init__(self, job: JobSpec) -> None:
        self.job = job
        self.row_index = 0
        self.delay_index = 0
        self.decision_index = 0
        self.backoff = _BoundedSampler()
        self.delays = _BoundedSampler()
        self.dynamic = _BoundedSampler()
        self.decisions = _BoundedSampler()
        self.counts = np.zeros((24, 4), dtype=np.int64)

    def observe(self, row: dict[str, object]) -> None:
        round_id = int(row["round_id"])
        senders = row["senders"]
        if senders:
            sender = senders[0]
            self.backoff.add(
                self.row_index,
                (round_id, int(sender["selected_backoff_before"])),
            )
        for sender in senders:
            if sender["delay_us"] is not None:
                self.delays.add(self.delay_index, int(sender["delay_us"]))
                self.delay_index += 1
        if self.job.scenario.join_interval_rounds is not None:
            self.dynamic.add(
                self.row_index,
                (round_id, len(row["active_node_ids"])),
            )
        for decision in row["decisions"]:
            arm = int(decision["arm"])
            bin_index = min(3, (4 * round_id) // self.job.rounds)
            self.counts[arm, bin_index] += 1
            self.decisions.add(
                self.decision_index,
                (
                    round_id,
                    arm,
                    (
                        None
                        if decision["reward"] is None
                        else float(decision["reward"])
                    ),
                    decision["previous_arm"] is not None
                    and decision["previous_arm"] != decision["arm"],
                ),
            )
            self.decision_index += 1
        self.row_index += 1

    def finish(
        self,
        aggregate: RunAggregate,
        execution_provenance: ExecutionProvenance,
    ) -> PlotRun:
        if self.decision_index != aggregate.decision_count:
            raise RuntimeError("decision reduction count mismatch")
        return PlotRun(
            job=self.job,
            backoff_points=self.backoff.values(),
            delay_samples=tuple(sorted(self.delays.values())),
            dynamic_points=self.dynamic.values(),
            decision_points=self.decisions.values(),
            arm_counts=self.counts,
            execution_provenance=execution_provenance,
        )


def _load_job(path: Path) -> JobSpec:
    raw = path.read_bytes()
    value = json.loads(raw.decode("ascii"))
    job = JobSpec.model_validate(value)
    if raw != (canonical_json(job) + "\n").encode("ascii"):
        raise ValueError("plot config input is not canonical JSON")
    return job


def _validated_plot_run(
    manifest_path: Path,
    root: Path,
    declared: dict[str, object],
) -> PlotRun:
    job = _load_job(root / "configs" / f"{manifest_path.stem}.json")
    if job.run_id != manifest_path.stem:
        raise ValueError("plot config run id does not match manifest")
    accumulator = _PlotAccumulator(job)
    row_stream = iter_job_rows(job, root)
    aggregate = aggregate_rows(row_stream, observer=accumulator.observe)
    expected = {
        "matrix": job.matrix,
        "scenario_id": job.scenario.id,
        "policy": job.policy,
        "seed": job.seed,
        "ablation": job.ablation,
        "arm_id": job.arm_id,
        "wifi_nodes": job.scenario.wifi_nodes,
        "nru_nodes": job.scenario.nru_nodes,
        "traffic": job.scenario.traffic,
        "interference_interval_ms": job.scenario.interference_interval_ms,
        "interruption_std": job.scenario.interruption_std,
        "join_interval_rounds": job.scenario.join_interval_rounds,
        "lifetime_rounds": job.scenario.lifetime_rounds,
        "config_hash": job.config_hash,
        "rounds": aggregate.rounds,
        "elapsed_us": aggregate.elapsed_us,
        "successes": aggregate.successes,
        "collisions": aggregate.collisions,
        "collision_probability": aggregate.collision_probability,
        "effective_airtime": aggregate.effective_airtime,
        "mean_delay_us": aggregate.mean_delay_us,
        "p95_delay_us": aggregate.p95_delay_us,
        "jain_fairness": aggregate.fairness,
        "evaluation_utility": aggregate.evaluation_utility,
        "decision_count": aggregate.decision_count,
        "switch_count": aggregate.switch_count,
        "training_sample_count": aggregate.training_sample_count,
    }
    if any(declared[field] != value for field, value in expected.items()):
        raise ValueError("summary metrics or config do not match canonical raw aggregate")
    return accumulator.finish(aggregate, row_stream.execution_provenance)


def validated_plot_inputs(
    summary_path: Path,
    manifest_dir: Path,
    *,
    workers: int = 1,
) -> tuple[list[dict[str, object]], list[PlotRun]]:
    max_workers = effective_worker_count(workers)
    summary = load_summary(summary_path)
    if not manifest_dir.is_dir():
        raise ValueError("manifest-dir must be an existing directory")
    manifest_paths = sorted(manifest_dir.glob("*.json"))
    summary_ids = {str(row["run_id"]) for row in summary}
    manifest_ids = {path.stem for path in manifest_paths}
    if manifest_ids != summary_ids or len(manifest_paths) != len(summary):
        raise ValueError("summary and manifest run ids do not match")
    root = manifest_dir.parent
    summary_by_id = {str(row["run_id"]): row for row in summary}
    declared_rows = [summary_by_id[path.stem] for path in manifest_paths]
    if max_workers == 1:
        runs = [
            _validated_plot_run(path, root, declared)
            for path, declared in zip(
                manifest_paths, declared_rows, strict=True
            )
        ]
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            runs = list(
                executor.map(
                    _validated_plot_run,
                    manifest_paths,
                    repeat(root),
                    declared_rows,
                )
            )
    return summary, runs


def _empty(ax: plt.Axes, label: str) -> None:
    ax.text(0.5, 0.5, label, ha="center", va="center", transform=ax.transAxes)


def _select_trace_runs(runs: list[PlotRun]) -> tuple[list[PlotRun], str]:
    candidates = [
        run
        for run in runs
        if run.job.matrix == "reproduction"
        and run.job.scenario.id == _TRACE_SCENARIO
    ]
    source = "reproduction"
    if not candidates:
        smoke = [run for run in runs if run.job.matrix == "smoke"]
        scenarios = {run.job.scenario.id for run in smoke}
        if len(scenarios) != 1:
            return [], "trace source unavailable"
        candidates = smoke
        source = "smoke fallback"
    seed = min(run.job.seed for run in candidates)
    selected = sorted(
        (run for run in candidates if run.job.seed == seed),
        key=lambda run: (run.job.policy, run.job.run_id),
    )
    scenario = selected[0].job.scenario.id
    return selected, f"{source}, {scenario}, seed {seed}"


def _backoff(ax: plt.Axes, runs: list[PlotRun]) -> None:
    runs, source = _select_trace_runs(runs)
    observed = False
    for policy in sorted({run.job.policy for run in runs}):
        values = [point for run in runs if run.job.policy == policy for point in run.backoff_points]
        if values:
            observed = True
            ax.scatter(*zip(*values, strict=True), s=9, alpha=0.45, label=policy)
    if observed:
        ax.legend(fontsize=7)
    else:
        _empty(ax, "No sender backoff observations")
    ax.set(
        xlabel="Contention round",
        ylabel="Selected backoff",
        title=f"Backoff convergence ({source})",
    )


def _delay_cdf(ax: plt.Axes, runs: list[PlotRun]) -> None:
    runs, source = _select_trace_runs(runs)
    observed = False
    for policy in sorted({run.job.policy for run in runs}):
        delays = sorted(sample for run in runs if run.job.policy == policy for sample in run.delay_samples)
        if delays:
            observed = True
            ax.step(delays, np.arange(1, len(delays) + 1) / len(delays), where="post", label=policy)
    if observed:
        ax.legend(fontsize=7)
    else:
        _empty(ax, "No completed access delays")
    ax.set(
        xlabel="Access delay (us)",
        ylabel="Empirical CDF",
        title=f"Delay distribution ({source})",
    )


def _scaling(ax: plt.Axes, summary: list[dict[str, object]]) -> None:
    summary = [
        row
        for row in summary
        if row["matrix"] == "reproduction"
        and str(row["scenario_id"]).startswith("static-symmetric-")
    ]
    for policy in sorted({str(row["policy"]) for row in summary}):
        grouped: dict[int, list[float]] = {}
        for row in summary:
            if row["policy"] != policy:
                continue
            nodes = int(row["wifi_nodes"]) + int(row["nru_nodes"])
            grouped.setdefault(nodes, []).append(float(row["evaluation_utility"]))
        keys = sorted(grouped)
        ax.plot(keys, [float(np.mean(grouped[key])) for key in keys], marker="o", label=policy)
    if summary:
        ax.legend(fontsize=7)
    else:
        _empty(ax, "Reproduction scaling evidence not available")
    ax.set(xlabel="Contending nodes", ylabel="Mean evaluation utility", title="Scaling")


def _select_dynamic_runs(runs: list[PlotRun]) -> tuple[list[PlotRun], str]:
    for matrix in ("reproduction", "heldout", "smoke"):
        matrix_runs = [
            run
            for run in runs
            if run.job.matrix == matrix
            and run.job.ablation is None
            and run.dynamic_points
        ]
        if not matrix_runs:
            continue
        scenario = min(run.job.scenario.id for run in matrix_runs)
        scenario_runs = [
            run for run in matrix_runs if run.job.scenario.id == scenario
        ]
        seed = min(run.job.seed for run in scenario_runs)
        selected = sorted(
            (run for run in scenario_runs if run.job.seed == seed),
            key=lambda run: (run.job.policy, run.job.run_id),
        )
        return selected, f"{matrix}, {scenario}, seed {seed}"
    return [], "main dynamic condition unavailable"


def _dynamic(ax: plt.Axes, runs: list[PlotRun]) -> None:
    dynamic_runs, source = _select_dynamic_runs(runs)
    if dynamic_runs:
        adaptive_runs = [
            run for run in dynamic_runs if run.job.policy == "adaptive_db_lbt"
        ]
        displayed_runs = adaptive_runs or [dynamic_runs[0]]
        arm_axis = ax.twinx()
        reward_axis = ax.twinx()
        reward_axis.spines["right"].set_position(("outward", 55))
        points = displayed_runs[0].dynamic_points
        ax.step(
            *zip(*points, strict=True),
            where="post",
            color="0.35",
            linewidth=1.2,
            label="Active nodes",
        )
        decisions = [
            point for run in displayed_runs for point in run.decision_points
        ]
        if decisions:
            arm_axis.scatter(
                [point[0] for point in decisions],
                [point[1] for point in decisions],
                color="tab:blue",
                marker="x",
                s=14,
                label="Selected arm",
            )
            rewarded = [point for point in decisions if point[2] is not None]
            if rewarded:
                reward_axis.scatter(
                    [point[0] for point in rewarded],
                    [float(point[2]) for point in rewarded],
                    color="tab:orange",
                    marker=".",
                    s=12,
                    label="Local reward",
                )
        handles: list[object] = []
        labels: list[str] = []
        for plot_axis in (ax, arm_axis, reward_axis):
            axis_handles, axis_labels = plot_axis.get_legend_handles_labels()
            handles.extend(axis_handles)
            labels.extend(axis_labels)
        ax.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=3,
            frameon=False,
            fontsize=7,
        )
        arm_axis.set_ylabel("Selected arm")
        arm_axis.set_ylim(-0.5, 23.5)
        arm_axis.tick_params(axis="y", colors="tab:blue")
        reward_axis.set_ylabel("Local reward")
        reward_axis.tick_params(axis="y", colors="tab:orange")
    else:
        _empty(ax, "No dynamic lifecycle observations")
    ax.set(xlabel="Contention round", ylabel="Active nodes")
    ax.set_title(f"Dynamic adaptation ({source})", pad=30)


def _utility(ax: plt.Axes, summary: list[dict[str, object]]) -> None:
    summary = [row for row in summary if row["matrix"] == "heldout"]
    if not summary:
        _empty(ax, "Held-out evidence not available")
        ax.set(ylabel="Mean evaluation utility", title="Held-out utility")
        return
    grouped: dict[str, list[float]] = {}
    for row in summary:
        grouped.setdefault(str(row["policy"]), []).append(float(row["evaluation_utility"]))
    policy_order = (
        "random_lbt",
        "primary_db_lbt",
        "tmc_db_lbt",
        "adaptive_db_lbt",
        "fixed_oracle",
    )
    policy_labels = {
        "random_lbt": "Random",
        "primary_db_lbt": "Primary",
        "tmc_db_lbt": "TMC",
        "adaptive_db_lbt": "Adaptive",
        "fixed_oracle": "Oracle",
    }
    policies = [policy for policy in policy_order if policy in grouped]
    policies.extend(sorted(set(grouped) - set(policies)))
    values = [float(np.mean(grouped[policy])) for policy in policies]
    positions = np.arange(len(policies))
    bars = ax.bar(positions, values)
    ax.set_xticks(
        positions,
        [policy_labels.get(policy, policy.replace("_", " ").title()) for policy in policies],
    )
    span = max(values) - min(values)
    margin = max(0.02, 0.15 * span)
    ax.set_ylim(max(0.0, min(values) - margin), min(1.0, max(values) + margin))
    ax.bar_label(bars, fmt="%.4f", padding=2, fontsize=7)
    ax.grid(axis="y", alpha=0.2)
    ax.set(ylabel="Mean evaluation utility", title="Held-out utility")


def _tradeoff(ax: plt.Axes, summary: list[dict[str, object]]) -> None:
    summary = [
        row
        for row in summary
        if row["matrix"] == "heldout"
        and row["ablation"] is None
        and row["policy"] in _MAIN_POLICIES
    ]
    for policy in sorted({str(row["policy"]) for row in summary}):
        rows = [row for row in summary if row["policy"] == policy]
        ax.scatter(
            [float(row["p95_delay_us"]) for row in rows],
            [float(row["jain_fairness"]) for row in rows],
            s=[25 + 80 * float(row["effective_airtime"]) for row in rows],
            label=policy,
            alpha=0.7,
        )
    if summary:
        ax.legend(fontsize=7)
    else:
        _empty(ax, "Held-out main-policy evidence not available")
    ax.set(xlabel="P95 delay (us)", ylabel="Jain fairness", title="Fairness-delay-airtime tradeoff")


def _heatmap(ax: plt.Axes, runs: list[PlotRun]) -> None:
    selected = [
        run
        for run in runs
        if run.job.matrix == "heldout"
        and run.job.policy == "adaptive_db_lbt"
        and run.job.ablation is None
        and run.job.arm_id is None
    ]
    counts = sum(
        (run.arm_counts for run in selected),
        start=np.zeros((24, 4), dtype=np.int64),
    )
    image = ax.imshow(counts, aspect="auto", origin="lower", interpolation="nearest")
    ax.figure.colorbar(image, ax=ax, label="Selections")
    ax.set(xlabel="Run quartile", ylabel="Arm", title="Arm selection heatmap")


def _ablation(ax: plt.Axes, comparisons: list[dict[str, object]]) -> None:
    rows = [row for row in comparisons if row["scope"] == "ablation"]
    if not rows:
        _empty(ax, "Ablation evidence not available")
        ax.set(xlabel="Paired utility difference vs full", title="Ablation forest")
        return
    rows.sort(key=lambda row: str(row["comparison_id"]))
    labels = [str(row["candidate_ablation"]) for row in rows]
    means = [float(row["paired_difference"]) for row in rows]
    errors = np.array(
        [
            [
                mean - float(row["lower_95"])
                for mean, row in zip(means, rows, strict=True)
            ],
            [
                float(row["upper_95"]) - mean
                for mean, row in zip(means, rows, strict=True)
            ],
        ]
    )
    positions = np.arange(len(labels))
    ax.errorbar(means, positions, xerr=errors, fmt="o", capsize=3)
    ax.set_yticks(positions, labels)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(xlabel="Paired utility difference vs full", title="Ablation forest")


_PLOTTERS: tuple[
    tuple[str, Callable[[plt.Axes, list[dict[str, object]], list[PlotRun]], None]],
    ...,
] = (
    ("backoff-convergence", lambda ax, summary, runs: _backoff(ax, runs)),
    ("delay-cdf", lambda ax, summary, runs: _delay_cdf(ax, runs)),
    ("scaling", lambda ax, summary, runs: _scaling(ax, summary)),
    ("dynamic-adaptation", lambda ax, summary, runs: _dynamic(ax, runs)),
    ("held-out-utility", lambda ax, summary, runs: _utility(ax, summary)),
    ("fairness-delay-airtime-tradeoff", lambda ax, summary, runs: _tradeoff(ax, summary)),
    ("arm-heatmap", lambda ax, summary, runs: _heatmap(ax, runs)),
    (
        "ablation-forest",
        lambda ax, summary, runs: _ablation(
            ax,
            (
                comparison_table_rows(summary)
                if {str(row["matrix"]) for row in summary}
                >= {"heldout", "ablation"}
                else []
            ),
        ),
    ),
)


def _save_atomic(fig: plt.Figure, path: Path) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.", suffix=path.suffix, dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
        fig.savefig(temporary, format=path.suffix[1:], dpi=160, bbox_inches="tight")
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _publish_tree(staging: Path, output: Path) -> None:
    """Atomically publish one complete directory tree with rollback."""
    output.parent.mkdir(parents=True, exist_ok=True)
    had_output = output.exists()
    if had_output and not output.is_dir():
        raise ValueError("report output must be a directory")
    if not had_output:
        output.mkdir()
    backup = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.backup.", dir=output.parent)
    )
    backup.rmdir()
    os.replace(output, backup)
    try:
        os.replace(staging, output)
    except BaseException:
        if output.exists():
            shutil.rmtree(output)
        os.replace(backup, output)
        if not had_output:
            shutil.rmtree(output)
        raise
    else:
        try:
            shutil.rmtree(backup)
        except OSError:
            pass


def _validate_output_tree(
    output: Path,
    *,
    run_root: Path,
    protected_files: tuple[Path, ...],
) -> None:
    """Reject any publication path that can replace protected inputs."""
    if (
        output == run_root
        or output.is_relative_to(run_root)
        or run_root.is_relative_to(output)
    ):
        raise ValueError("report output overlaps the protected run artifact tree")
    for protected in protected_files:
        if protected == output or protected.is_relative_to(output):
            raise ValueError("report output contains a protected input file")
    project = Path.cwd().resolve(strict=False)
    if output == project or project.is_relative_to(output):
        raise ValueError("report output cannot replace the current project tree")


def generate_figures(
    summary_path: str | Path,
    output_dir: str | Path,
    *,
    manifest_dir: str | Path | None,
    run_validator: Callable[[list[PlotRun]], None] | None = None,
    workers: int = 1,
) -> list[Path]:
    """Generate the fixed eight PDF and PNG figures from validated inputs."""
    if manifest_dir is None:
        raise ValueError("manifest-dir is required for canonical raw-backed figures")
    summary_source = Path(summary_path).resolve(strict=False)
    manifests = Path(manifest_dir).resolve(strict=False)
    summary, runs = validated_plot_inputs(
        summary_source, manifests, workers=workers
    )
    if run_validator is not None:
        run_validator(runs)
    output = Path(output_dir).resolve(strict=False)
    root = manifests.parent
    _validate_output_tree(
        output,
        run_root=root,
        protected_files=(summary_source,),
    )
    targets = [
        output / f"{name}.{suffix}"
        for name in FIGURE_NAMES
        for suffix in ("pdf", "png")
    ]
    if summary_source in {target.resolve(strict=False) for target in targets}:
        raise ValueError("figure output cannot overwrite summary input")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    try:
        for name, plotter in _PLOTTERS:
            fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
            try:
                plotter(ax, summary, runs)
                for suffix in ("pdf", "png"):
                    _save_atomic(fig, staging / f"{name}.{suffix}")
            finally:
                plt.close(fig)
        _publish_tree(staging, output)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return targets


def generate_report(
    summary_path: str | Path,
    output_dir: str | Path,
    manifest_dir: str | Path,
    model_path: str | Path | None,
    oracle_arm_file: str | Path | None = None,
    *,
    workers: int = 1,
) -> list[Path]:
    """Generate the fixed figure and table report from validated inputs."""
    from .stats import generate_tables, measure_model_overhead

    output = Path(output_dir).resolve(strict=False)
    summary_source = Path(summary_path).resolve(strict=False)
    manifests = Path(manifest_dir).resolve(strict=False)
    protected_files = [summary_source]
    if model_path is not None:
        protected_files.append(Path(model_path).resolve(strict=False))
    if oracle_arm_file is not None:
        protected_files.append(Path(oracle_arm_file).resolve(strict=False))
    _validate_output_tree(
        output,
        run_root=manifests.parent,
        protected_files=tuple(protected_files),
    )
    summary_rows = load_summary(summary_source)
    manifest_ids = {path.stem for path in manifests.glob("*.json")}
    mode = validate_report_inventory(summary_rows, manifest_ids)
    run_validator: Callable[[list[PlotRun]], None] | None = None
    if mode == "formal":
        if model_path is None:
            raise ValueError("formal report mode requires --model")
        if oracle_arm_file is None:
            raise ValueError("formal report mode requires --oracle-arm-file")
        from .audit import _audit_model_provenance

        model = Path(model_path).resolve(strict=False)
        oracle = Path(oracle_arm_file).resolve(strict=False)

        def validate_run_provenance(runs: list[PlotRun]) -> None:
            _audit_model_provenance(
                model, runs, oracle_arm_file=oracle
            )

        run_validator = validate_run_provenance
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.report.", dir=output.parent)
    )
    try:
        if run_validator is None:
            if workers == 1:
                figures = generate_figures(
                    summary_source, staging, manifest_dir=manifests
                )
            else:
                figures = generate_figures(
                    summary_source,
                    staging,
                    manifest_dir=manifests,
                    workers=workers,
                )
        else:
            if workers == 1:
                figures = generate_figures(
                    summary_source,
                    staging,
                    manifest_dir=manifests,
                    run_validator=run_validator,
                )
            else:
                figures = generate_figures(
                    summary_source,
                    staging,
                    manifest_dir=manifests,
                    run_validator=run_validator,
                    workers=workers,
                )
        tables: list[Path] = []
        if mode == "formal":
            overhead = measure_model_overhead(model_path)
            tables = generate_tables(summary_source, staging / "tables", overhead)
        relative_paths = [path.relative_to(staging) for path in [*figures, *tables]]
        _publish_tree(staging, output)
        return [output / path for path in relative_paths]
    finally:
        shutil.rmtree(staging, ignore_errors=True)
