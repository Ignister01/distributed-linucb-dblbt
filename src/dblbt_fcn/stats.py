"""Preregistered paired statistics and hypothesis decisions."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import io
import math
from numbers import Real
import os
from os import PathLike
from pathlib import Path
import tempfile
import time
from typing import Callable, Iterable, Literal, Mapping

import numpy as np

from .training import HELD_OUT_SEEDS
from .linucb import LinUCB
from .provenance import file_sha256


BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260715
HYPOTHESIS_STATUSES = frozenset(
    {"pass", "fail", "inconclusive", "not_evaluated"}
)


@dataclass(frozen=True, slots=True)
class PairedBootstrapResult:
    baseline_mean: float
    adaptive_mean: float
    paired_difference: float
    relative_difference: float | None
    lower_95: float
    upper_95: float
    decision: Literal["improvement", "degradation", "inconclusive"]
    resamples: int
    bootstrap_seed: int


@dataclass(frozen=True, slots=True)
class HypothesisResult:
    hypothesis: Literal["H1", "H2", "H3", "H4", "H5"]
    status: Literal["pass", "fail", "inconclusive", "not_evaluated"]
    threshold: float | None
    paired_difference: float | None
    lower_95: float | None
    upper_95: float | None


@dataclass(frozen=True, slots=True)
class ModelOverhead:
    model_path: str
    model_state_bytes: int
    model_sha256: str
    action_grid_hash: str
    warmup_calls: int
    measurement_calls: int
    median_us: float
    p95_us: float


def _paired_values(
    values: Iterable[tuple[int, float]], label: str
) -> dict[int, float]:
    paired: dict[int, float] = {}
    for item in values:
        if type(item) is not tuple or len(item) != 2:
            raise ValueError(f"{label} pairs must be (seed, value) tuples")
        seed, raw_value = item
        if type(seed) is not int:
            raise ValueError(f"{label} seed must be an exact integer")
        if seed in paired:
            raise ValueError(f"duplicate {label} seed: {seed}")
        if isinstance(raw_value, bool) or not isinstance(raw_value, Real):
            raise ValueError(f"{label} value must be finite")
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f"{label} value must be finite")
        paired[seed] = value
    if set(paired) != set(HELD_OUT_SEEDS):
        raise ValueError(f"{label} seeds must match the ten held-out seeds")
    return paired


def paired_bootstrap(
    baseline: Iterable[tuple[int, float]],
    adaptive: Iterable[tuple[int, float]],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> PairedBootstrapResult:
    """Return the fixed ten-pair bootstrap comparison."""
    if resamples != BOOTSTRAP_RESAMPLES or bootstrap_seed != BOOTSTRAP_SEED:
        raise ValueError(
            "registered bootstrap requires 10,000 resamples and seed 20260715"
        )
    baseline_by_seed = _paired_values(baseline, "baseline")
    adaptive_by_seed = _paired_values(adaptive, "adaptive")
    seeds = tuple(sorted(HELD_OUT_SEEDS))
    baseline_values = np.array(
        [baseline_by_seed[seed] for seed in seeds], dtype=np.float64
    )
    adaptive_values = np.array(
        [adaptive_by_seed[seed] for seed in seeds], dtype=np.float64
    )
    differences = adaptive_values - baseline_values
    generator = np.random.default_rng(bootstrap_seed)
    indices = generator.integers(
        0, len(seeds), size=(resamples, len(seeds)), endpoint=False
    )
    bootstrap_means = differences[indices].mean(axis=1)
    lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
    difference = float(differences.mean())
    baseline_mean = float(baseline_values.mean())
    adaptive_mean = float(adaptive_values.mean())
    if lower > 0:
        decision = "improvement"
    elif upper < 0:
        decision = "degradation"
    else:
        decision = "inconclusive"
    return PairedBootstrapResult(
        baseline_mean=baseline_mean,
        adaptive_mean=adaptive_mean,
        paired_difference=difference,
        relative_difference=(
            None if baseline_mean == 0.0 else difference / baseline_mean
        ),
        lower_95=float(lower),
        upper_95=float(upper),
        decision=decision,
        resamples=resamples,
        bootstrap_seed=bootstrap_seed,
    )


def _threshold_status(
    evidence: PairedBootstrapResult, threshold: float
) -> Literal["pass", "fail", "inconclusive"]:
    tolerance = 1e-12
    if evidence.lower_95 >= threshold - tolerance:
        return "pass"
    if evidence.upper_95 < threshold - tolerance:
        return "fail"
    return "inconclusive"


def evaluate_preregistered_hypotheses(
    evidence: Mapping[str, PairedBootstrapResult],
    *,
    ns3_available: bool,
    h5_direction_evidence: tuple[bool, bool, bool] | None = None,
) -> list[HypothesisResult]:
    """Evaluate H1-H5 without dropping negative or unavailable results."""
    if set(evidence) != {"H1", "H2", "H3", "H4"}:
        raise ValueError("hypothesis evidence must contain exactly H1 through H4")
    if ns3_available and h5_direction_evidence is None:
        raise ValueError(
            "H5 requires three-scenario direction evidence when ns-3 is available"
        )
    if h5_direction_evidence is not None and (
        type(h5_direction_evidence) is not tuple
        or len(h5_direction_evidence) != 3
        or any(type(item) is not bool for item in h5_direction_evidence)
    ):
        raise ValueError(
            "H5 direction evidence must be a tuple of exactly three booleans"
        )
    if not ns3_available and h5_direction_evidence is not None:
        raise ValueError("H5 direction evidence requires ns-3 availability")
    thresholds = {
        "H1": -0.02 * evidence["H1"].baseline_mean,
        "H2": 0.10 * evidence["H2"].baseline_mean,
        "H3": -0.01,
        "H4": 0.0,
    }
    rows: list[HypothesisResult] = []
    for hypothesis in ("H1", "H2", "H3", "H4"):
        result = evidence[hypothesis]
        threshold = thresholds[hypothesis]
        if hypothesis == "H4" and result.lower_95 <= 0 <= result.upper_95:
            status = "inconclusive"
        else:
            status = _threshold_status(result, threshold)
        rows.append(
            HypothesisResult(
                hypothesis=hypothesis,  # type: ignore[arg-type]
                status=status,
                threshold=threshold,
                paired_difference=result.paired_difference,
                lower_95=result.lower_95,
                upper_95=result.upper_95,
            )
        )
    h5_status: Literal["pass", "fail", "not_evaluated"] = "not_evaluated"
    if h5_direction_evidence is not None:
        h5_status = (
            "pass" if sum(h5_direction_evidence) >= 2 else "fail"
        )
    rows.append(
        HypothesisResult(
            hypothesis="H5",
            status=h5_status,
            threshold=None,
            paired_difference=None,
            lower_95=None,
            upper_95=None,
        )
    )
    return rows


def measure_model_overhead(
    model_path: str | PathLike[str],
    *,
    timer: Callable[[], int] = time.perf_counter_ns,
    selector: Callable[[LinUCB, object], int] | None = None,
) -> ModelOverhead:
    """Measure the frozen model file and preregistered select latency."""
    from .workflows import action_grid_hash

    path = Path(model_path)
    if not path.is_file():
        raise ValueError(f"model provenance input must be a regular file: {path}")
    grid_hash = action_grid_hash()
    agent = LinUCB.load(path, expected_action_grid_hash=grid_hash)
    choose = (
        (lambda value, context: value.select(context))
        if selector is None
        else selector
    )
    context = np.zeros(agent.context_dim, dtype=np.float64)
    for _ in range(100):
        choose(agent, context)
    durations = np.empty(10_000, dtype=np.float64)
    for index in range(10_000):
        started = timer()
        choose(agent, context)
        ended = timer()
        if type(started) is not int or type(ended) is not int or ended < started:
            raise ValueError("latency timer must return nondecreasing integer nanoseconds")
        durations[index] = (ended - started) / 1_000
    return ModelOverhead(
        model_path=str(path.resolve()),
        model_state_bytes=path.stat().st_size,
        model_sha256=file_sha256(path),
        action_grid_hash=grid_hash,
        warmup_calls=100,
        measurement_calls=10_000,
        median_us=float(np.median(durations)),
        p95_us=float(np.quantile(durations, 0.95)),
    )


SUMMARY_FIELDS = (
    "run_id",
    "matrix",
    "scenario_id",
    "policy",
    "seed",
    "ablation",
    "arm_id",
    "wifi_nodes",
    "nru_nodes",
    "traffic",
    "interference_interval_ms",
    "interruption_std",
    "join_interval_rounds",
    "lifetime_rounds",
    "config_hash",
    "rounds",
    "elapsed_us",
    "successes",
    "collisions",
    "collision_probability",
    "effective_airtime",
    "mean_delay_us",
    "p95_delay_us",
    "jain_fairness",
    "evaluation_utility",
    "decision_count",
    "switch_count",
    "training_sample_count",
)
_INTEGER_SUMMARY_FIELDS = {
    "seed",
    "wifi_nodes",
    "nru_nodes",
    "rounds",
    "elapsed_us",
    "successes",
    "collisions",
    "decision_count",
    "switch_count",
    "training_sample_count",
}
_NULLABLE_INTEGER_SUMMARY_FIELDS = {
    "arm_id",
    "interference_interval_ms",
    "join_interval_rounds",
    "lifetime_rounds",
}
_FLOAT_SUMMARY_FIELDS = {
    "interruption_std",
    "collision_probability",
    "effective_airtime",
    "mean_delay_us",
    "p95_delay_us",
    "jain_fairness",
    "evaluation_utility",
}


def load_summary(path: str | PathLike[str]) -> list[dict[str, object]]:
    """Load the exact canonical summary schema with finite numeric values."""
    source = Path(path).resolve(strict=False)
    if not source.is_file():
        raise ValueError("summary input must be an existing regular file")
    try:
        text = source.read_text(encoding="ascii")
    except UnicodeError as error:
        raise ValueError("summary must be ASCII CSV") from error
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if reader.fieldnames != list(SUMMARY_FIELDS):
        raise ValueError("summary CSV schema is invalid")
    rows: list[dict[str, object]] = []
    seen_runs: set[str] = set()
    seen_pairs: set[tuple[str, str, str, int, object, object]] = set()
    for raw in reader:
        if None in raw or any(value is None for value in raw.values()):
            raise ValueError("summary CSV row schema is invalid")
        row: dict[str, object] = dict(raw)
        for field in _INTEGER_SUMMARY_FIELDS:
            value = raw[field]
            try:
                numeric = int(value)
            except ValueError as error:
                raise ValueError(f"summary {field} must be an integer") from error
            if str(numeric) != value or numeric < 0:
                raise ValueError(f"summary {field} must be a canonical nonnegative integer")
            row[field] = numeric
        for field in _FLOAT_SUMMARY_FIELDS:
            try:
                numeric = float(raw[field])
            except ValueError as error:
                raise ValueError(f"summary {field} must be finite") from error
            if not math.isfinite(numeric):
                raise ValueError(f"summary {field} must be finite")
            row[field] = numeric
        for field in _NULLABLE_INTEGER_SUMMARY_FIELDS:
            value = raw[field]
            if value == "":
                row[field] = None
                continue
            try:
                numeric = int(value)
            except ValueError as error:
                raise ValueError(
                    f"summary {field} must be an integer or empty"
                ) from error
            if str(numeric) != value or numeric < 0:
                raise ValueError(
                    f"summary {field} must be canonical and nonnegative"
                )
            row[field] = numeric
        if raw["ablation"] == "":
            row["ablation"] = None
        run_id = str(row["run_id"])
        if run_id in seen_runs:
            raise ValueError(f"duplicate summary run_id: {run_id}")
        seen_runs.add(run_id)
        key = (
            str(row["matrix"]),
            str(row["scenario_id"]),
            str(row["policy"]),
            int(row["seed"]),
            row["ablation"],
            row["arm_id"],
        )
        if key in seen_pairs:
            raise ValueError("duplicate summary scenario/policy/seed row")
        seen_pairs.add(key)
        rows.append(row)
    if not rows:
        raise ValueError("summary CSV contains no rows")
    return rows


def _is_nonideal(scenario_id: str) -> bool:
    return scenario_id.startswith(("poisson-", "periodic-", "perturb-", "dynamic-"))


def _aggregate_pairs(
    rows: list[dict[str, object]],
    *,
    metric: str,
    scenario_filter: Callable[[str], bool],
) -> PairedBootstrapResult:
    relevant = [row for row in rows if scenario_filter(str(row["scenario_id"]))]
    scenarios = sorted({str(row["scenario_id"]) for row in relevant})
    if not scenarios:
        raise ValueError("paired comparison has no registered scenario rows")
    values: dict[str, dict[int, list[float]]] = {
        "tmc_db_lbt": {},
        "adaptive_db_lbt": {},
    }
    for scenario in scenarios:
        for policy in values:
            selected = [
                row
                for row in relevant
                if row["scenario_id"] == scenario and row["policy"] == policy
            ]
            if {int(row["seed"]) for row in selected} != set(HELD_OUT_SEEDS):
                raise ValueError(
                    f"paired seed completeness failed for {scenario}/{policy}"
                )
            for row in selected:
                values[policy].setdefault(int(row["seed"]), []).append(
                    float(row[metric])
                )
    baseline = [
        (seed, float(np.mean(values["tmc_db_lbt"][seed])))
        for seed in sorted(HELD_OUT_SEEDS)
    ]
    adaptive = [
        (seed, float(np.mean(values["adaptive_db_lbt"][seed])))
        for seed in sorted(HELD_OUT_SEEDS)
    ]
    return paired_bootstrap(baseline, adaptive)


def comparison_evidence(
    rows: list[dict[str, object]],
) -> dict[str, PairedBootstrapResult]:
    """Build H1-H4 evidence from all registered paired summary rows."""
    heldout = [row for row in rows if row["matrix"] == "heldout"]
    if not heldout:
        raise ValueError("hypothesis evidence requires heldout matrix rows")
    return {
        "H1": _aggregate_pairs(
            heldout,
            metric="evaluation_utility",
            scenario_filter=lambda scenario: not _is_nonideal(scenario),
        ),
        "H2": _aggregate_pairs(
            heldout,
            metric="evaluation_utility",
            scenario_filter=_is_nonideal,
        ),
        "H3": _aggregate_pairs(
            heldout,
            metric="jain_fairness",
            scenario_filter=lambda scenario: True,
        ),
        "H4": _aggregate_pairs(
            heldout,
            metric="evaluation_utility",
            scenario_filter=lambda scenario: True,
        ),
    }


COMPARISON_FIELDS = [
    "comparison_id",
    "scope",
    "hypothesis",
    "scenario_id",
    "baseline_policy",
    "candidate_policy",
    "baseline_ablation",
    "candidate_ablation",
    "metric",
    "direction",
    "baseline_mean",
    "candidate_mean",
    "paired_difference",
    "relative_difference",
    "lower_95",
    "upper_95",
    "decision",
    "resamples",
    "bootstrap_seed",
]


def _comparison_row(
    *,
    comparison_id: str,
    scope: str,
    hypothesis: str = "",
    scenario_id: str = "",
    baseline_policy: str = "",
    candidate_policy: str = "",
    baseline_ablation: str = "",
    candidate_ablation: str = "",
    metric: str,
    result: PairedBootstrapResult,
) -> dict[str, object]:
    return {
        "comparison_id": comparison_id,
        "scope": scope,
        "hypothesis": hypothesis,
        "scenario_id": scenario_id,
        "baseline_policy": baseline_policy,
        "candidate_policy": candidate_policy,
        "baseline_ablation": baseline_ablation,
        "candidate_ablation": candidate_ablation,
        "metric": metric,
        "direction": "higher_is_better",
        "baseline_mean": result.baseline_mean,
        "candidate_mean": result.adaptive_mean,
        "paired_difference": result.paired_difference,
        "relative_difference": (
            "" if result.relative_difference is None else result.relative_difference
        ),
        "lower_95": result.lower_95,
        "upper_95": result.upper_95,
        "decision": result.decision,
        "resamples": result.resamples,
        "bootstrap_seed": result.bootstrap_seed,
    }


def _policy_pairs(
    rows: list[dict[str, object]],
    scenario_id: str,
    baseline_policy: str,
    metric: str,
) -> PairedBootstrapResult:
    def values(policy: str) -> list[tuple[int, float]]:
        selected = [
            row
            for row in rows
            if row["matrix"] == "heldout"
            and row["scenario_id"] == scenario_id
            and row["policy"] == policy
            and row["ablation"] is None
        ]
        return [(int(row["seed"]), float(row[metric])) for row in selected]

    return paired_bootstrap(values(baseline_policy), values("adaptive_db_lbt"))


def _ablation_pairs(
    rows: list[dict[str, object]], condition: str
) -> PairedBootstrapResult:
    def values(ablation: str) -> list[tuple[int, float]]:
        selected = [
            row
            for row in rows
            if row["matrix"] == "ablation" and row["ablation"] == ablation
        ]
        return [
            (int(row["seed"]), float(row["evaluation_utility"]))
            for row in selected
        ]

    return paired_bootstrap(values("full"), values(condition))


def comparison_table_rows(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Build all machine-auditable paired comparison rows."""
    evidence = comparison_evidence(rows)
    metrics = {
        "H1": "evaluation_utility",
        "H2": "evaluation_utility",
        "H3": "jain_fairness",
        "H4": "evaluation_utility",
    }
    output = [
        _comparison_row(
            comparison_id=f"hypothesis:{key}",
            scope="hypothesis",
            hypothesis=key,
            baseline_policy="tmc_db_lbt",
            candidate_policy="adaptive_db_lbt",
            metric=metrics[key],
            result=evidence[key],
        )
        for key in ("H1", "H2", "H3", "H4")
    ]
    heldout = [row for row in rows if row["matrix"] == "heldout"]
    scenarios = sorted({str(row["scenario_id"]) for row in heldout})
    for scenario in scenarios:
        available = {
            str(row["policy"])
            for row in heldout
            if row["scenario_id"] == scenario
        }
        for baseline in (
            "random_lbt",
            "primary_db_lbt",
            "tmc_db_lbt",
            "fixed_oracle",
        ):
            if baseline not in available or "adaptive_db_lbt" not in available:
                continue
            output.append(
                _comparison_row(
                    comparison_id=(
                        f"heldout:{scenario}:adaptive_db_lbt-vs-{baseline}"
                    ),
                    scope="heldout_scenario",
                    scenario_id=scenario,
                    baseline_policy=baseline,
                    candidate_policy="adaptive_db_lbt",
                    metric="evaluation_utility",
                    result=_policy_pairs(
                        rows, scenario, baseline, "evaluation_utility"
                    ),
                )
            )
    ablation_values = {
        str(row["ablation"])
        for row in rows
        if row["matrix"] == "ablation"
    }
    conditions = (
        sorted(ablation_values - {"full"}) if "full" in ablation_values else []
    )
    for condition in conditions:
        output.append(
            _comparison_row(
                comparison_id=f"ablation:{condition}-vs-full",
                scope="ablation",
                scenario_id="dynamic-combined-4x4",
                baseline_policy="adaptive_db_lbt",
                candidate_policy="adaptive_db_lbt",
                baseline_ablation="full",
                candidate_ablation=condition,
                metric="evaluation_utility",
                result=_ablation_pairs(rows, condition),
            )
        )
    return output


def _csv_bytes(fields: list[str], rows: list[dict[str, object]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output, fieldnames=fields, extrasaction="raise", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("ascii")


def _latex_bytes(fields: list[str], rows: list[dict[str, object]]) -> bytes:
    def escape(value: object) -> str:
        return str(value).replace("_", r"\_").replace("%", r"\%")

    alignment = "l" * len(fields)
    lines = [
        f"\\begin{{tabular}}{{{alignment}}}",
        " & ".join(map(escape, fields)) + " \\\\",
    ]
    lines.extend(
        " & ".join(escape(row[field]) for field in fields) + " \\\\"
        for row in rows
    )
    lines.append(r"\end{tabular}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{path.name}.",
            suffix=".partial",
            dir=path.parent,
            delete=False,
        ) as destination:
            temporary = Path(destination.name)
            destination.write(payload)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def build_table_definitions(
    rows: list[dict[str, object]], overhead: ModelOverhead
) -> dict[str, tuple[list[str], list[dict[str, object]]]]:
    """Return the canonical in-memory definitions for all evidence tables."""
    if not isinstance(overhead, ModelOverhead):
        raise TypeError("overhead must be ModelOverhead")
    evidence = comparison_evidence(rows)
    hypotheses = evaluate_preregistered_hypotheses(
        evidence, ns3_available=False
    )
    per_seed_rows = sorted(
        rows,
        key=lambda row: (
            str(row["matrix"]),
            str(row["scenario_id"]),
            str(row["policy"]),
            "" if row["ablation"] is None else str(row["ablation"]),
            -1 if row["arm_id"] is None else int(row["arm_id"]),
            int(row["seed"]),
            str(row["run_id"]),
        ),
    )
    comparison_rows = comparison_table_rows(rows)
    overhead_fields = [
        "model_path",
        "model_state_bytes",
        "model_sha256",
        "action_grid_hash",
        "warmup_calls",
        "measurement_calls",
        "median_us",
        "p95_us",
    ]
    overhead_rows = [
        {field: getattr(overhead, field) for field in overhead_fields}
    ]
    hypothesis_fields = [
        "hypothesis",
        "status",
        "threshold",
        "paired_difference",
        "lower_95",
        "upper_95",
    ]
    hypothesis_rows = [
        {
            field: (
                "" if getattr(row, field) is None else getattr(row, field)
            )
            for field in hypothesis_fields
        }
        for row in hypotheses
    ]
    return {
        "per-seed-metrics": (list(SUMMARY_FIELDS), per_seed_rows),
        "paired-comparisons": (COMPARISON_FIELDS, comparison_rows),
        "overhead": (overhead_fields, overhead_rows),
        "hypotheses": (hypothesis_fields, hypothesis_rows),
    }


def table_payloads(
    rows: list[dict[str, object]], overhead: ModelOverhead
) -> dict[str, bytes]:
    """Serialize every canonical evidence table without writing files."""
    definitions = build_table_definitions(rows, overhead)
    payloads: dict[str, bytes] = {}
    for name, (fields, table_rows) in definitions.items():
        payloads[f"{name}.csv"] = _csv_bytes(fields, table_rows)
        payloads[f"{name}.tex"] = _latex_bytes(fields, table_rows)
    return payloads


def generate_tables(
    summary_path: str | PathLike[str],
    output_dir: str | PathLike[str],
    overhead: ModelOverhead,
) -> list[Path]:
    """Write the four fixed CSV and LaTeX evidence tables."""
    if not isinstance(overhead, ModelOverhead):
        raise TypeError("overhead must be ModelOverhead")
    summary = Path(summary_path).resolve(strict=False)
    output = Path(output_dir).resolve(strict=False)
    rows = load_summary(summary)
    payload_by_name = table_payloads(rows, overhead)
    targets = [
        output / name for name in payload_by_name
    ]
    protected = {summary, Path(overhead.model_path).resolve(strict=False)}
    if any(target.resolve(strict=False) in protected for target in targets):
        raise ValueError("table output cannot overwrite a protected input")
    for path in targets:
        _atomic_write(path, payload_by_name[path.name])
    return targets
