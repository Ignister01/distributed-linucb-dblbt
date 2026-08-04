"""Reduce the frozen comments0720 confirmation matrices into paper evidence."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from statistics import fmean
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dblbt_fcn.config import adaptive_arms
from dblbt_fcn.experiment import load_job
from dblbt_fcn.provenance import file_sha256
from dblbt_fcn.records import aggregate_rows, iter_job_rows
from dblbt_fcn.regime_evidence import split_phase_segments


BOOTSTRAP_SEED = 20260804
BOOTSTRAP_RESAMPLES = 10_000
EXPECTED_ROWS = {
    "fixed-arm-confirmation-summary.csv": 1_536,
    "adaptive-confirmation-summary.csv": 128,
    "multiphase-confirmation-summary.csv": 64,
    "multiphase-fixed-confirmation-summary.csv": 64,
}
METRICS = (
    "evaluation_utility",
    "collision_probability",
    "effective_airtime",
    "p95_delay_us",
    "jain_fairness",
)


@dataclass(frozen=True)
class Comparison:
    scope: str
    scenario_id: str
    candidate: str
    baseline: str
    seed_count: int
    utility_difference: float
    utility_lower_95: float
    utility_upper_95: float
    positive_seeds: int
    collision_difference: float
    effective_airtime_difference: float
    p95_delay_difference_us: float
    fairness_difference: float


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="ascii", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = EXPECTED_ROWS.get(path.name)
    if expected is not None and len(rows) != expected:
        raise ValueError(f"{path.name}: expected {expected} rows, found {len(rows)}")
    return rows


def _index(
    rows: Iterable[dict[str, str]],
    *,
    key_field: str,
) -> dict[tuple[str, str, int], dict[str, str]]:
    indexed: dict[tuple[str, str, int], dict[str, str]] = {}
    for row in rows:
        key = (row["scenario_id"], row[key_field], int(row["seed"]))
        if key in indexed:
            raise ValueError(f"duplicate result row: {key}")
        indexed[key] = row
    return indexed


def _paired_interval(values: list[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("paired interval requires values")
    samples = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    indices = generator.integers(
        0,
        len(samples),
        size=(BOOTSTRAP_RESAMPLES, len(samples)),
        endpoint=False,
    )
    means = samples[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _comparison(
    *,
    scope: str,
    scenario_id: str,
    candidate: str,
    baseline: str,
    candidate_rows: dict[int, dict[str, str]],
    baseline_rows: dict[int, dict[str, str]],
) -> Comparison:
    seeds = sorted(set(candidate_rows) & set(baseline_rows))
    if not seeds or set(candidate_rows) != set(baseline_rows):
        raise ValueError(f"unpaired seeds for {candidate} versus {baseline}")
    differences = {
        metric: [
            float(candidate_rows[seed][metric])
            - float(baseline_rows[seed][metric])
            for seed in seeds
        ]
        for metric in METRICS
    }
    utility = differences["evaluation_utility"]
    lower, upper = _paired_interval(utility)
    return Comparison(
        scope=scope,
        scenario_id=scenario_id,
        candidate=candidate,
        baseline=baseline,
        seed_count=len(seeds),
        utility_difference=fmean(utility),
        utility_lower_95=lower,
        utility_upper_95=upper,
        positive_seeds=sum(value > 0 for value in utility),
        collision_difference=fmean(differences["collision_probability"]),
        effective_airtime_difference=fmean(differences["effective_airtime"]),
        p95_delay_difference_us=fmean(differences["p95_delay_us"]),
        fairness_difference=fmean(differences["jain_fairness"]),
    )


def _rows_for(
    indexed: dict[tuple[str, str, int], dict[str, str]],
    scenario: str,
    key: str,
) -> dict[int, dict[str, str]]:
    return {
        seed: row
        for (scenario_id, row_key, seed), row in indexed.items()
        if scenario_id == scenario and row_key == key
    }


def _comparisons(results: Path) -> list[Comparison]:
    fixed = _index(
        _read_csv(results / "fixed-arm-confirmation-summary.csv"),
        key_field="arm_id",
    )
    stationary = _index(
        _read_csv(results / "adaptive-confirmation-summary.csv"),
        key_field="policy",
    )
    phase = _index(
        _read_csv(results / "multiphase-confirmation-summary.csv"),
        key_field="policy",
    )
    phase_fixed = _index(
        _read_csv(results / "multiphase-fixed-confirmation-summary.csv"),
        key_field="arm_id",
    )

    low = "poisson-n04-p025"
    high = "poisson-n06-p045"
    repeated = "repeated-poisson-n04-p025-n06-p045"
    comparisons: list[Comparison] = []
    for scenario, matched_arm in ((low, "4"), (high, "20")):
        comparisons.append(
            _comparison(
                scope="stationary",
                scenario_id=scenario,
                candidate="adaptive_db_lbt",
                baseline="tmc_db_lbt",
                candidate_rows=_rows_for(
                    stationary, scenario, "adaptive_db_lbt"
                ),
                baseline_rows=_rows_for(stationary, scenario, "tmc_db_lbt"),
            )
        )
        comparisons.append(
            _comparison(
                scope="stationary",
                scenario_id=scenario,
                candidate="adaptive_db_lbt",
                baseline=f"fixed_arm_{matched_arm}",
                candidate_rows=_rows_for(
                    stationary, scenario, "adaptive_db_lbt"
                ),
                baseline_rows=_rows_for(fixed, scenario, matched_arm),
            )
        )

    comparisons.extend(
        [
            _comparison(
                scope="fixed-profile-conflict",
                scenario_id=low,
                candidate="fixed_arm_4",
                baseline="fixed_arm_20",
                candidate_rows=_rows_for(fixed, low, "4"),
                baseline_rows=_rows_for(fixed, low, "20"),
            ),
            _comparison(
                scope="fixed-profile-conflict",
                scenario_id=high,
                candidate="fixed_arm_20",
                baseline="fixed_arm_4",
                candidate_rows=_rows_for(fixed, high, "20"),
                baseline_rows=_rows_for(fixed, high, "4"),
            ),
        ]
    )
    for baseline, source, key in (
        ("tmc_db_lbt", phase, "tmc_db_lbt"),
        ("fixed_arm_4", phase_fixed, "4"),
        ("fixed_arm_20", phase_fixed, "20"),
    ):
        comparisons.append(
            _comparison(
                scope="repeated-phase",
                scenario_id=repeated,
                candidate="adaptive_db_lbt",
                baseline=baseline,
                candidate_rows=_rows_for(phase, repeated, "adaptive_db_lbt"),
                baseline_rows=_rows_for(source, repeated, key),
            )
        )
    return comparisons


def _write_comparisons(path: Path, comparisons: list[Comparison]) -> None:
    fields = tuple(Comparison.__dataclass_fields__)
    with path.open("w", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(vars(row) for row in comparisons)


def _phase_decisions(results: Path) -> list[dict[str, object]]:
    run_root = results / "multiphase-confirmation-runs"
    counts: dict[tuple[str, int], int] = {}
    total_by_phase: dict[str, int] = {}
    for manifest in sorted((run_root / "manifests").glob("*.json")):
        job = load_job(run_root / "configs" / manifest.name)
        if job.policy != "adaptive_db_lbt":
            continue
        for row in iter_job_rows(job, run_root):
            phase = str(row["phase_id"])
            for decision in row["decisions"]:
                arm = int(decision["arm"])
                counts[(phase, arm)] = counts.get((phase, arm), 0) + 1
                total_by_phase[phase] = total_by_phase.get(phase, 0) + 1
    profiles = adaptive_arms()
    rows: list[dict[str, object]] = []
    for (phase, arm), count in sorted(counts.items()):
        profile = profiles[arm]
        rows.append(
            {
                "phase_id": phase,
                "arm_id": arm,
                "kappa": profile.kappa,
                "beta": profile.beta,
                "m": profile.m,
                "b_init": profile.b_init,
                "decision_count": count,
                "decision_share": count / total_by_phase[phase],
            }
        )
    return rows


def _stationary_decisions(results: Path) -> list[dict[str, object]]:
    run_root = results / "adaptive-confirmation-runs"
    counts: dict[tuple[str, int], int] = {}
    total_by_scenario: dict[str, int] = {}
    for manifest in sorted((run_root / "manifests").glob("*.json")):
        job = load_job(run_root / "configs" / manifest.name)
        if job.policy != "adaptive_db_lbt":
            continue
        for row in iter_job_rows(job, run_root):
            scenario = job.scenario.id
            for decision in row["decisions"]:
                arm = int(decision["arm"])
                counts[(scenario, arm)] = counts.get((scenario, arm), 0) + 1
                total_by_scenario[scenario] = total_by_scenario.get(scenario, 0) + 1
    profiles = adaptive_arms()
    rows: list[dict[str, object]] = []
    for (scenario, arm), count in sorted(counts.items()):
        profile = profiles[arm]
        rows.append(
            {
                "scenario_id": scenario,
                "arm_id": arm,
                "kappa": profile.kappa,
                "beta": profile.beta,
                "m": profile.m,
                "b_init": profile.b_init,
                "decision_count": count,
                "decision_share": count / total_by_scenario[scenario],
            }
        )
    return rows


def _write_phase_decisions(path: Path, rows: list[dict[str, object]]) -> None:
    fields = (
        "phase_id",
        "arm_id",
        "kappa",
        "beta",
        "m",
        "b_init",
        "decision_count",
        "decision_share",
    )
    with path.open("w", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_stationary_decisions(
    path: Path, rows: list[dict[str, object]]
) -> None:
    fields = (
        "scenario_id",
        "arm_id",
        "kappa",
        "beta",
        "m",
        "b_init",
        "decision_count",
        "decision_share",
    )
    with path.open("w", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _phase_segment_rows(
    results: Path,
) -> list[dict[str, object]]:
    return phase_segment_rows_from_roots(
        (
            results / "multiphase-confirmation-runs",
            results / "multiphase-fixed-confirmation-runs",
        )
    )


def phase_segment_rows_from_roots(
    roots: Iterable[Path],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for run_root in roots:
        for manifest in sorted((run_root / "manifests").glob("*.json")):
            job = load_job(run_root / "configs" / manifest.name)
            source_segments = split_phase_segments(
                list(iter_job_rows(job, run_root))
            )
            previous_end_us = 0
            for phase_index, source_rows in enumerate(source_segments):
                rows: list[dict[str, object]] = []
                for round_id, source_row in enumerate(source_rows):
                    row = dict(source_row)
                    row["round_id"] = round_id
                    row["round_end_us"] = (
                        int(source_row["round_end_us"]) - previous_end_us
                    )
                    rows.append(row)
                aggregate = aggregate_rows(rows)
                output.append(
                    {
                        "run_id": job.run_id,
                        "seed": job.seed,
                        "policy": job.policy,
                        "arm_id": "" if job.arm_id is None else job.arm_id,
                        "phase_index": phase_index,
                        "phase_id": rows[0]["phase_id"],
                        "rounds": aggregate.rounds,
                        "evaluation_utility": aggregate.evaluation_utility,
                        "collision_probability": aggregate.collision_probability,
                        "effective_airtime": aggregate.effective_airtime,
                        "p95_delay_us": aggregate.p95_delay_us,
                        "jain_fairness": aggregate.fairness,
                    }
                )
                previous_end_us = int(source_rows[-1]["round_end_us"])
    return output


def _write_phase_segments(path: Path, rows: list[dict[str, object]]) -> None:
    fields = (
        "run_id",
        "seed",
        "policy",
        "arm_id",
        "phase_index",
        "phase_id",
        "rounds",
        *METRICS,
    )
    with path.open("w", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _phase_segment_effects(rows: list[dict[str, object]]) -> list[Comparison]:
    grouped: dict[tuple[str, str, int], list[dict[str, object]]] = {}
    for row in rows:
        key = str(row["policy"])
        if key == "pretrain_arm":
            key = f"fixed_arm_{row['arm_id']}"
        grouped.setdefault((str(row["phase_id"]), key, int(row["seed"])), []).append(row)

    averaged: dict[tuple[str, str, int], dict[str, str]] = {}
    for key, values in grouped.items():
        averaged[key] = {
            metric: str(fmean(float(row[metric]) for row in values))
            for metric in METRICS
        }
    effects: list[Comparison] = []
    for phase_id in ("low-n04-p025", "high-n06-p045"):
        candidate = {
            seed: row
            for (phase, policy, seed), row in averaged.items()
            if phase == phase_id and policy == "adaptive_db_lbt"
        }
        for baseline in ("tmc_db_lbt", "fixed_arm_4", "fixed_arm_20"):
            baseline_rows = {
                seed: row
                for (phase, policy, seed), row in averaged.items()
                if phase == phase_id and policy == baseline
            }
            effects.append(
                _comparison(
                    scope="phase-segment",
                    scenario_id=phase_id,
                    candidate="adaptive_db_lbt",
                    baseline=baseline,
                    candidate_rows=candidate,
                    baseline_rows=baseline_rows,
                )
            )
    return effects


def _figure(
    path: Path,
    comparisons: list[Comparison],
    adaptation_rows: list[dict[str, str]],
) -> None:
    matplotlib.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(7.16, 2.35), constrained_layout=True)
    stationary = [
        row
        for row in comparisons
        if row.scope == "stationary" and row.baseline == "tmc_db_lbt"
    ]
    fixed = [row for row in comparisons if row.scope == "fixed-profile-conflict"]

    def effect_panel(axis: plt.Axes, rows: list[Comparison], title: str) -> None:
        positions = np.arange(len(rows))
        means = np.asarray([row.utility_difference for row in rows])
        lower = np.asarray([row.utility_lower_95 for row in rows])
        upper = np.asarray([row.utility_upper_95 for row in rows])
        axis.bar(positions, means, width=0.62, color="#0072B2")
        axis.errorbar(
            positions,
            means,
            yerr=np.vstack((means - lower, upper - means)),
            fmt="none",
            color="#222222",
            linewidth=0.8,
            capsize=2,
        )
        axis.axhline(0, color="#666666", linewidth=0.7)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.45)
        axis.set(
            title=title,
            ylabel="Paired utility difference",
            xticks=positions,
            xticklabels=["Low load", "High load"],
        )

    effect_panel(axes[0], stationary, "Adaptive - fixed TMC")
    effect_panel(axes[1], fixed, "Matched-profile advantage")

    by_phase: dict[str, list[int]] = {}
    for row in adaptation_rows:
        by_phase.setdefault(row["phase_id"], []).append(int(row["recovery_rounds"]))
    labels = ["Low load", "High load"]
    values = [
        by_phase["low-n04-p025"],
        by_phase["high-n06-p045"],
    ]
    axes[2].boxplot(
        values,
        tick_labels=labels,
        widths=0.5,
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "#009E73", "edgecolor": "#222222"},
        medianprops={"color": "#222222"},
        whiskerprops={"color": "#222222"},
        capprops={"color": "#222222"},
    )
    axes[2].axhline(4096, color="#D55E00", linestyle="--", linewidth=0.8)
    axes[2].grid(axis="y", color="#DDDDDD", linewidth=0.45)
    axes[2].set(title="Local-reward recovery", ylabel="Recovery rounds")
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(__file__).with_name("results"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("results") / "confirmation-analysis",
    )
    args = parser.parse_args()
    results = args.results.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    comparisons = _comparisons(results)
    comparison_path = output / "policy-comparisons.csv"
    _write_comparisons(comparison_path, comparisons)

    decision_rows = _phase_decisions(results)
    decision_path = output / "phase-decision-arm-share.csv"
    _write_phase_decisions(decision_path, decision_rows)

    stationary_decision_rows = _stationary_decisions(results)
    _write_stationary_decisions(
        output / "stationary-decision-arm-share.csv",
        stationary_decision_rows,
    )

    phase_segments = _phase_segment_rows(results)
    _write_phase_segments(output / "phase-segment-summary.csv", phase_segments)
    phase_effects = _phase_segment_effects(phase_segments)
    _write_comparisons(output / "phase-segment-effects.csv", phase_effects)

    with (results / "adaptation-analysis" / "adaptation-transitions.csv").open(
        encoding="ascii", newline=""
    ) as handle:
        adaptation_rows = list(csv.DictReader(handle))
    figure_path = output / "formal-confirmation.pdf"
    _figure(figure_path, comparisons, adaptation_rows)

    inputs = [
        results / name for name in EXPECTED_ROWS
    ] + [results / "adaptation-analysis" / "adaptation-transitions.csv"]
    audit = {
        "schema_version": 1,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "inputs": {path.name: file_sha256(path) for path in inputs},
        "comparison_rows": len(comparisons),
        "phase_decision_rows": len(decision_rows),
        "stationary_decision_rows": len(stationary_decision_rows),
        "phase_segment_rows": len(phase_segments),
        "phase_segment_effect_rows": len(phase_effects),
    }
    (output / "analysis-audit.json").write_text(
        json.dumps(audit, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    print(
        f"comparisons={len(comparisons)} decisions={len(decision_rows)} "
        f"phase_segments={len(phase_segments)}"
    )


if __name__ == "__main__":
    main()
