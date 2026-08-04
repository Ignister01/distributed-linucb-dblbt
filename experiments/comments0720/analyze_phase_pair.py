"""Compare two repeated-phase policies without mixing active populations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean

from analyze_confirmation import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    METRICS,
    _comparison,
    _write_comparisons,
    _write_phase_segments,
    phase_segment_rows_from_roots,
)
from dblbt_fcn.experiment import load_job
from dblbt_fcn.provenance import file_sha256
from dblbt_fcn.records import iter_job_rows


def _averaged(
    rows: list[dict[str, object]], phase_id: str | None
) -> dict[tuple[str, int], dict[str, str]]:
    grouped: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in rows:
        if phase_id is not None and row["phase_id"] != phase_id:
            continue
        grouped.setdefault((str(row["policy"]), int(row["seed"])), []).append(row)
    return {
        key: {
            metric: str(fmean(float(row[metric]) for row in values))
            for metric in METRICS
        }
        for key, values in grouped.items()
    }


def _decision_counts(run_root: Path) -> list[dict[str, object]]:
    counts: dict[tuple[str, int], int] = {}
    for manifest in sorted((run_root / "manifests").glob("*.json")):
        job = load_job(run_root / "configs" / manifest.name)
        for row in iter_job_rows(job, run_root):
            for decision in row["decisions"]:
                arm = int(decision["arm"])
                if arm not in (4, 20):
                    raise ValueError(
                        f"restricted-profile run selected forbidden arm {arm}"
                    )
                key = (str(row["phase_id"]), arm)
                counts[key] = counts.get(key, 0) + 1
    return [
        {"phase_id": phase_id, "arm_id": arm, "decision_count": count}
        for (phase_id, arm), count in sorted(counts.items())
    ]


def _summary_by_seed(path: Path) -> dict[int, dict[str, str]]:
    with path.open(encoding="ascii", newline="") as handle:
        rows = list(csv.DictReader(handle))
    indexed: dict[int, dict[str, str]] = {}
    for row in rows:
        seed = int(row["seed"])
        if seed in indexed:
            raise ValueError(f"duplicate summary seed: {seed}")
        for metric in METRICS:
            float(row[metric])
        indexed[seed] = row
    if not indexed:
        raise ValueError("summary contains no rows")
    return indexed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-run-root", required=True, type=Path)
    parser.add_argument("--baseline-run-root", required=True, type=Path)
    parser.add_argument("--candidate-summary", required=True, type=Path)
    parser.add_argument("--baseline-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    rows = phase_segment_rows_from_roots(
        (args.candidate_run_root.resolve(), args.baseline_run_root.resolve())
    )
    _write_phase_segments(output / "phase-segment-summary.csv", rows)
    effects = []
    for phase_id in ("low-n04-p025", "high-n06-p045", None):
        averaged = _averaged(rows, phase_id)
        candidate = {
            seed: values
            for (policy, seed), values in averaged.items()
            if policy == "adaptive_db_lbt"
        }
        baseline = {
            seed: values
            for (policy, seed), values in averaged.items()
            if policy == "tmc_db_lbt"
        }
        effects.append(
            _comparison(
                scope=("time-averaged-phase" if phase_id is None else "phase-segment"),
                scenario_id=("all-phases" if phase_id is None else phase_id),
                candidate="adaptive_db_lbt",
                baseline="tmc_db_lbt",
                candidate_rows=candidate,
                baseline_rows=baseline,
            )
        )
    _write_comparisons(output / "phase-effects.csv", effects)

    whole_run_effect = _comparison(
        scope="whole-run-diagnostic",
        scenario_id="all-phases-mixed-active-set",
        candidate="adaptive_db_lbt",
        baseline="tmc_db_lbt",
        candidate_rows=_summary_by_seed(args.candidate_summary.resolve()),
        baseline_rows=_summary_by_seed(args.baseline_summary.resolve()),
    )
    _write_comparisons(output / "whole-run-effect.csv", [whole_run_effect])

    decision_rows = _decision_counts(args.candidate_run_root.resolve())
    with (output / "decision-arm-membership.csv").open(
        "w", encoding="ascii", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("phase_id", "arm_id", "decision_count"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(decision_rows)
    audit = {
        "schema_version": 1,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "candidate_summary_sha256": file_sha256(args.candidate_summary.resolve()),
        "baseline_summary_sha256": file_sha256(args.baseline_summary.resolve()),
        "phase_segment_rows": len(rows),
        "phase_effect_rows": len(effects),
        "decision_rows": len(decision_rows),
    }
    (output / "analysis-audit.json").write_text(
        json.dumps(audit, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    print(f"segments={len(rows)} effects={len(effects)}")


if __name__ == "__main__":
    main()
