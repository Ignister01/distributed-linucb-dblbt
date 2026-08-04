"""Fixed-profile conflict and empirical adaptation-time evidence."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import io
import json
import math
from numbers import Real
import os
from pathlib import Path
from statistics import fmean, median
import tempfile
from typing import Sequence

import numpy as np

from .experiment import canonical_json, load_job
from .metrics import nearest_rank_p95
from .provenance import file_sha256
from .records import iter_job_rows


BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260803
RECOVERY_WINDOW = 8
RECOVERY_PERSISTENCE = 3
RECOVERY_FRACTION = 0.90


@dataclass(frozen=True, slots=True)
class ScenarioArmRanking:
    scenario_id: str
    best_arm: int
    runner_up_arm: int
    best_mean: float
    paired_margin: float
    lower_95: float


@dataclass(frozen=True, slots=True)
class FixedArmConflict:
    scenario_a: str
    scenario_b: str
    best_arm_a: int
    best_arm_b: int
    margin_a_over_b: float
    lower_a_over_b: float
    margin_b_over_a: float
    lower_b_over_a: float


@dataclass(frozen=True, slots=True)
class FixedArmAggregate:
    arm_id: int
    mean_utility: float
    worst_scenario_utility: float


@dataclass(frozen=True, slots=True)
class FixedArmAnalysis:
    rankings: tuple[ScenarioArmRanking, ...]
    conflicts: tuple[FixedArmConflict, ...]
    arm_aggregates: tuple[FixedArmAggregate, ...]
    best_global_arm: int
    minimax_arm: int
    scenario_count: int
    arm_count: int
    seed_count: int


@dataclass(frozen=True, slots=True)
class AdaptationTransition:
    run_id: str
    phase_id: str
    change_round: int
    recovery_rounds: int | None
    dwell_rounds: int
    censored: bool


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


def _csv_payload(fields: Sequence[str], rows: Sequence[dict[str, object]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=fields,
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("ascii")


def _json_payload(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be finite")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{label} must be finite")
    return numeric


def _paired_lower(
    differences: Sequence[float],
    *,
    resamples: int,
    bootstrap_seed: int,
) -> float:
    values = np.asarray(differences, dtype=np.float64)
    generator = np.random.default_rng(bootstrap_seed)
    indices = generator.integers(
        0, len(values), size=(resamples, len(values)), endpoint=False
    )
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025))


def rank_fixed_arms(
    rows: Sequence[dict[str, object]],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> FixedArmAnalysis:
    """Rank fixed profiles and test whether scenario optima conflict."""
    if type(resamples) is not int or resamples < 1:
        raise ValueError("resamples must be a positive exact integer")
    if type(bootstrap_seed) is not int or bootstrap_seed < 0:
        raise ValueError("bootstrap_seed must be a nonnegative exact integer")
    if not rows:
        raise ValueError("fixed-arm analysis requires rows")

    grouped: dict[str, dict[int, dict[int, float]]] = {}
    for row in rows:
        scenario = row.get("scenario_id")
        arm = row.get("arm_id")
        seed = row.get("seed")
        if type(scenario) is not str or not scenario:
            raise ValueError("scenario_id must be nonempty")
        if row.get("policy") != "pretrain_arm" or row.get("ablation") is not None:
            raise ValueError("fixed-arm analysis requires unablated pretrain_arm rows")
        if type(arm) is not int or not 0 <= arm < 24:
            raise ValueError("arm_id must be an exact integer in 0..23")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be an exact nonnegative integer")
        utility = _finite(row.get("evaluation_utility"), "evaluation_utility")
        by_seed = grouped.setdefault(scenario, {}).setdefault(arm, {})
        if seed in by_seed:
            raise ValueError(
                f"duplicate fixed-arm row for {scenario}/arm-{arm}/seed-{seed}"
            )
        by_seed[seed] = utility

    scenario_ids = tuple(sorted(grouped))
    arm_sets = {scenario: set(grouped[scenario]) for scenario in scenario_ids}
    common_arms = arm_sets[scenario_ids[0]]
    if len(common_arms) < 2 or any(
        arms != common_arms for arms in arm_sets.values()
    ):
        raise ValueError("fixed-arm scenarios must cover the same two or more arms")
    seed_sets: list[set[int]] = []
    for scenario in scenario_ids:
        seeds_for_arms = [
            set(grouped[scenario][arm]) for arm in sorted(common_arms)
        ]
        if not seeds_for_arms[0] or any(
            seeds != seeds_for_arms[0] for seeds in seeds_for_arms[1:]
        ):
            raise ValueError(f"paired seeds do not match for {scenario}")
        seed_sets.append(seeds_for_arms[0])
    if any(seeds != seed_sets[0] for seeds in seed_sets[1:]):
        raise ValueError("paired seeds do not match across scenarios")

    means = {
        scenario: {
            arm: fmean(grouped[scenario][arm].values())
            for arm in sorted(common_arms)
        }
        for scenario in scenario_ids
    }
    rankings: list[ScenarioArmRanking] = []
    for scenario in scenario_ids:
        ordered = sorted(common_arms, key=lambda arm: (-means[scenario][arm], arm))
        best, runner_up = ordered[:2]
        seeds = sorted(grouped[scenario][best])
        differences = [
            grouped[scenario][best][seed]
            - grouped[scenario][runner_up][seed]
            for seed in seeds
        ]
        rankings.append(
            ScenarioArmRanking(
                scenario_id=scenario,
                best_arm=best,
                runner_up_arm=runner_up,
                best_mean=means[scenario][best],
                paired_margin=fmean(differences),
                lower_95=_paired_lower(
                    differences,
                    resamples=resamples,
                    bootstrap_seed=bootstrap_seed,
                ),
            )
        )

    best_by_scenario = {row.scenario_id: row.best_arm for row in rankings}
    conflicts: list[FixedArmConflict] = []
    for first_index, scenario_a in enumerate(scenario_ids):
        for scenario_b in scenario_ids[first_index + 1 :]:
            arm_a = best_by_scenario[scenario_a]
            arm_b = best_by_scenario[scenario_b]
            if arm_a == arm_b:
                continue
            seeds = sorted(grouped[scenario_a][arm_a])
            differences_a = [
                grouped[scenario_a][arm_a][seed]
                - grouped[scenario_a][arm_b][seed]
                for seed in seeds
            ]
            differences_b = [
                grouped[scenario_b][arm_b][seed]
                - grouped[scenario_b][arm_a][seed]
                for seed in seeds
            ]
            conflicts.append(
                FixedArmConflict(
                    scenario_a=scenario_a,
                    scenario_b=scenario_b,
                    best_arm_a=arm_a,
                    best_arm_b=arm_b,
                    margin_a_over_b=fmean(differences_a),
                    lower_a_over_b=_paired_lower(
                        differences_a,
                        resamples=resamples,
                        bootstrap_seed=bootstrap_seed,
                    ),
                    margin_b_over_a=fmean(differences_b),
                    lower_b_over_a=_paired_lower(
                        differences_b,
                        resamples=resamples,
                        bootstrap_seed=bootstrap_seed,
                    ),
                )
            )

    aggregates = tuple(
        FixedArmAggregate(
            arm_id=arm,
            mean_utility=fmean(means[scenario][arm] for scenario in scenario_ids),
            worst_scenario_utility=min(
                means[scenario][arm] for scenario in scenario_ids
            ),
        )
        for arm in sorted(common_arms)
    )
    best_global = min(
        aggregates, key=lambda row: (-row.mean_utility, row.arm_id)
    ).arm_id
    minimax = min(
        aggregates,
        key=lambda row: (-row.worst_scenario_utility, row.arm_id),
    ).arm_id
    return FixedArmAnalysis(
        rankings=tuple(rankings),
        conflicts=tuple(conflicts),
        arm_aggregates=aggregates,
        best_global_arm=best_global,
        minimax_arm=minimax,
        scenario_count=len(scenario_ids),
        arm_count=len(common_arms),
        seed_count=len(seed_sets[0]),
    )


def split_phase_segments(
    rows: Sequence[dict[str, object]],
) -> tuple[tuple[dict[str, object], ...], ...]:
    """Split contiguous raw rows at explicit phase-change boundaries."""
    if not rows:
        raise ValueError("phase segmentation requires rows")
    segments: list[list[dict[str, object]]] = []
    for expected_round, row in enumerate(rows):
        if row.get("round_id") != expected_round:
            raise ValueError("phase rows must be contiguous from round zero")
        change_point = row.get("change_point")
        phase_id = row.get("phase_id")
        if type(change_point) is not bool:
            raise ValueError("change_point must be an exact boolean")
        if type(phase_id) is not str or not phase_id:
            raise ValueError("phase_id must be nonempty")
        if change_point:
            segments.append([])
        if not segments:
            raise ValueError("the first phase row must be a change point")
        if segments[-1] and segments[-1][0]["phase_id"] != phase_id:
            raise ValueError("phase_id changed without a change point")
        segments[-1].append(row)
    return tuple(tuple(segment) for segment in segments)


def adaptation_transitions(
    rows: Sequence[dict[str, object]],
    *,
    window: int = RECOVERY_WINDOW,
    persistence: int = RECOVERY_PERSISTENCE,
    recovery_fraction: float = RECOVERY_FRACTION,
) -> tuple[AdaptationTransition, ...]:
    """Measure persistent reward recovery after explicit phase changes."""
    if type(window) is not int or window < 1:
        raise ValueError("window must be a positive exact integer")
    if type(persistence) is not int or persistence < 1:
        raise ValueError("persistence must be a positive exact integer")
    if not 0.0 < recovery_fraction <= 1.0:
        raise ValueError("recovery_fraction must be in (0, 1]")
    if not rows:
        raise ValueError("adaptation analysis requires rows")

    segments: list[list[dict[str, object]]] = []
    run_ids: set[str] = set()
    for expected_round, row in enumerate(rows):
        run_id = row.get("run_id")
        if type(run_id) is not str or not run_id:
            raise ValueError("run_id must be nonempty")
        run_ids.add(run_id)
        if row.get("round_id") != expected_round:
            raise ValueError("adaptation rows must be contiguous from round zero")
        change_point = row.get("change_point")
        if type(change_point) is not bool:
            raise ValueError("change_point must be an exact boolean")
        if change_point:
            segments.append([])
        if not segments:
            raise ValueError("the first adaptation row must be a change point")
        segments[-1].append(row)
    if len(run_ids) != 1:
        raise ValueError("adaptation rows must contain exactly one run_id")

    transitions: list[AdaptationTransition] = []
    for segment in segments[1:]:
        change_round = int(segment[0]["round_id"])
        phase_id = segment[0].get("phase_id")
        if type(phase_id) is not str or not phase_id:
            raise ValueError("phase_id must be nonempty")
        decision_points: list[tuple[int, float]] = []
        for row in segment:
            decisions = row.get("decisions")
            if type(decisions) is not list:
                raise ValueError("decisions must be a list")
            rewards = [
                _finite(decision.get("reward"), "decision reward")
                for decision in decisions
                if type(decision) is dict and decision.get("reward") is not None
            ]
            if rewards:
                decision_points.append(
                    (
                        int(row["round_id"]) + 1 - change_round,
                        fmean(rewards),
                    )
                )
        if len(decision_points) < window + persistence:
            raise ValueError("phase has too few rewarded decision windows")

        rewards = [reward for _, reward in decision_points]
        immediate = fmean(rewards[:window])
        steady_count = max(1, len(rewards) // 4)
        steady = fmean(rewards[-steady_count:])
        recovery_rounds: int | None = None
        if steady <= immediate:
            recovery_rounds = 0
        else:
            threshold = immediate + recovery_fraction * (steady - immediate)
            rolling = [
                fmean(rewards[index : index + window])
                for index in range(len(rewards) - window + 1)
            ]
            qualifies = [value >= threshold for value in rolling]
            for index in range(len(qualifies) - persistence + 1):
                if all(qualifies[index : index + persistence]):
                    recovery_rounds = decision_points[index + window - 1][0]
                    break
        transitions.append(
            AdaptationTransition(
                run_id=next(iter(run_ids)),
                phase_id=phase_id,
                change_round=change_round,
                recovery_rounds=recovery_rounds,
                dwell_rounds=len(segment),
                censored=recovery_rounds is None,
            )
        )
    if not transitions:
        raise ValueError("adaptation analysis requires at least one transition")
    return tuple(transitions)


def write_fixed_arm_report(
    analysis: FixedArmAnalysis,
    output_dir: str | Path,
    *,
    input_path: str | Path,
) -> tuple[Path, ...]:
    """Write deterministic fixed-profile rankings and their source audit."""
    if not isinstance(analysis, FixedArmAnalysis):
        raise TypeError("analysis must be FixedArmAnalysis")
    source = Path(input_path)
    if not source.is_file():
        raise ValueError("fixed-arm input must be an existing regular file")
    root = Path(output_dir)
    rankings = root / "scenario-rankings.csv"
    conflicts = root / "conflicting-optima.csv"
    aggregates = root / "fixed-arm-aggregates.csv"
    audit = root / "regime-rank-audit.json"
    _atomic_write(
        rankings,
        _csv_payload(
            tuple(ScenarioArmRanking.__dataclass_fields__),
            [asdict(row) for row in analysis.rankings],
        ),
    )
    _atomic_write(
        conflicts,
        _csv_payload(
            tuple(FixedArmConflict.__dataclass_fields__),
            [asdict(row) for row in analysis.conflicts],
        ),
    )
    _atomic_write(
        aggregates,
        _csv_payload(
            tuple(FixedArmAggregate.__dataclass_fields__),
            [asdict(row) for row in analysis.arm_aggregates],
        ),
    )
    _atomic_write(
        audit,
        _json_payload(
            {
                "schema_version": 1,
                "input_file": source.name,
                "input_sha256": file_sha256(source),
                "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "scenario_count": analysis.scenario_count,
                "arm_count": analysis.arm_count,
                "seed_count": analysis.seed_count,
                "conflict_count": len(analysis.conflicts),
                "best_global_arm": analysis.best_global_arm,
                "minimax_arm": analysis.minimax_arm,
            }
        ),
    )
    return rankings, conflicts, aggregates, audit


def write_adaptation_manifest_report(
    manifest_dir: str | Path, output_dir: str | Path
) -> tuple[Path, ...]:
    """Validate phased adaptive runs and write transition-time evidence."""
    manifests = Path(manifest_dir).resolve(strict=False)
    if not manifests.is_dir():
        raise ValueError("manifest-dir must be an existing directory")
    entries = sorted(manifests.glob("*.json"), key=lambda path: path.name)
    if not entries:
        raise ValueError("manifest-dir contains no JSON manifests")
    run_root = manifests.parent
    transitions: list[AdaptationTransition] = []
    inputs: list[dict[str, str]] = []
    seeds: set[int] = set()
    revisions: set[str] = set()
    for manifest_path in entries:
        config_path = run_root / "configs" / f"{manifest_path.stem}.json"
        if not config_path.is_file():
            raise ValueError("adaptation manifest is missing its job config")
        job = load_job(config_path)
        if job.run_id != manifest_path.stem:
            raise ValueError("adaptation manifest filename does not match config")
        if config_path.read_bytes() != (canonical_json(job) + "\n").encode("ascii"):
            raise ValueError("adaptation job config is not canonical JSON")
        if job.policy != "adaptive_db_lbt" or not job.scenario.phases:
            continue
        stream = iter_job_rows(job, run_root)
        rows = list(stream)
        transitions.extend(adaptation_transitions(rows))
        seeds.add(job.seed)
        revisions.add(stream.manifest.git_revision)
        inputs.append(
            {
                "run_id": job.run_id,
                "config_sha256": file_sha256(config_path),
                "manifest_sha256": file_sha256(manifest_path),
                "record_sha256": stream.manifest.record_hash,
            }
        )
    if not transitions:
        raise ValueError("no phased adaptive runs were found")

    recovered = [
        row.recovery_rounds
        for row in transitions
        if row.recovery_rounds is not None
    ]
    dwell = [row.dwell_rounds for row in transitions]
    censored_fraction = sum(row.censored for row in transitions) / len(transitions)
    p95 = None if not recovered else nearest_rank_p95(recovered)
    minimum_dwell = min(dwell)
    summary = {
        "schema_version": 1,
        "transition_count": len(transitions),
        "recovered_count": len(recovered),
        "censored_count": len(transitions) - len(recovered),
        "censored_fraction": censored_fraction,
        "median_recovery_rounds": (
            None if not recovered else float(median(recovered))
        ),
        "p95_recovery_rounds": p95,
        "min_dwell_rounds": minimum_dwell,
        "p95_t_adapt_lt_t_dwell": p95 is not None and p95 < minimum_dwell,
        "effective_adaptation": (
            censored_fraction <= 0.10
            and p95 is not None
            and p95 < minimum_dwell
        ),
    }
    root = Path(output_dir)
    transition_path = root / "adaptation-transitions.csv"
    summary_path = root / "adaptation-summary.json"
    audit_path = root / "adaptation-audit.json"
    transition_rows = []
    for row in transitions:
        values = asdict(row)
        if values["recovery_rounds"] is None:
            values["recovery_rounds"] = ""
        transition_rows.append(values)
    _atomic_write(
        transition_path,
        _csv_payload(
            tuple(AdaptationTransition.__dataclass_fields__), transition_rows
        ),
    )
    _atomic_write(summary_path, _json_payload(summary))
    _atomic_write(
        audit_path,
        _json_payload(
            {
                "schema_version": 1,
                "recovery_window_decisions": RECOVERY_WINDOW,
                "persistence_windows": RECOVERY_PERSISTENCE,
                "recovery_fraction": RECOVERY_FRACTION,
                "seed_count": len(seeds),
                "git_revisions": sorted(revisions),
                "inputs": inputs,
            }
        ),
    )
    return transition_path, summary_path, audit_path
