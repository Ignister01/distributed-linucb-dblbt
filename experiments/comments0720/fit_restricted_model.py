"""Fit the two-profile LinUCB warm start from validated discovery samples."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from dblbt_fcn.experiment import load_job
from dblbt_fcn.linucb import LinUCB
from dblbt_fcn.provenance import file_sha256
from dblbt_fcn.records import iter_job_rows
from dblbt_fcn.workflows import action_grid_hash


SCENARIOS = ("poisson-n04-p025", "poisson-n06-p045")
ARMS = (4, 20)
SEEDS = (9103, 9113, 9127)


def _selected_runs(summary: Path) -> list[dict[str, str]]:
    with summary.open(encoding="ascii", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = [
        row
        for row in rows
        if row["scenario_id"] in SCENARIOS
        and int(row["arm_id"]) in ARMS
        and int(row["seed"]) in SEEDS
    ]
    expected = {
        (scenario, arm, seed)
        for scenario in SCENARIOS
        for arm in ARMS
        for seed in SEEDS
    }
    actual = {
        (row["scenario_id"], int(row["arm_id"]), int(row["seed"]))
        for row in selected
    }
    if actual != expected or len(selected) != len(expected):
        raise ValueError("discovery summary does not contain the frozen training set")
    return sorted(
        selected,
        key=lambda row: (row["scenario_id"], int(row["arm_id"]), int(row["seed"])),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--model-output", required=True, type=Path)
    parser.add_argument("--audit-output", required=True, type=Path)
    args = parser.parse_args()

    summary = args.summary.resolve()
    run_root = args.run_root.resolve()
    selected = _selected_runs(summary)
    agent = LinUCB(
        24,
        11,
        ridge=1.0,
        exploration=0.5,
        action_grid_hash=action_grid_hash(),
    )
    counts: dict[tuple[str, int], int] = {}
    reward_sums: dict[tuple[str, int], float] = {}
    inputs: list[dict[str, object]] = []
    provenance: set[tuple[str, int]] = set()

    for selected_row in selected:
        run_id = selected_row["run_id"]
        config = run_root / "configs" / f"{run_id}.json"
        manifest = run_root / "manifests" / f"{run_id}.json"
        job = load_job(config)
        expected_key = (
            selected_row["scenario_id"],
            int(selected_row["arm_id"]),
            int(selected_row["seed"]),
        )
        if (job.scenario.id, job.arm_id, job.seed) != expected_key:
            raise ValueError("training config does not match the summary row")
        stream = iter_job_rows(job, run_root)
        sample_count = 0
        for row in stream:
            for sample in row["training_samples"]:
                arm = sample["arm"]
                context = sample["context"]
                reward = sample["local_reward"]
                sample_key = (sample["node_id"], sample["local_sequence"])
                if (
                    arm != job.arm_id
                    or arm not in ARMS
                    or sample["pretraining_seed"] != job.seed
                    or not sample["node_id"].startswith(f"{run_id}:")
                    or sample_key in provenance
                    or len(context) != 11
                    or not math.isfinite(float(reward))
                ):
                    raise ValueError("invalid restricted-model training sample")
                provenance.add(sample_key)
                agent.update(arm, context, reward)
                key = (job.scenario.id, arm)
                counts[key] = counts.get(key, 0) + 1
                reward_sums[key] = reward_sums.get(key, 0.0) + float(reward)
                sample_count += 1
        if sample_count != int(selected_row["training_sample_count"]):
            raise ValueError("training sample count does not match summary")
        inputs.append(
            {
                "run_id": run_id,
                "scenario_id": job.scenario.id,
                "arm_id": job.arm_id,
                "seed": job.seed,
                "config_sha256": file_sha256(config),
                "manifest_sha256": file_sha256(manifest),
            }
        )

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    agent.save(args.model_output)
    loaded = LinUCB.load(
        args.model_output,
        expected_action_grid_hash=action_grid_hash(),
    )
    statistics = [
        {
            "scenario_id": scenario,
            "arm_id": arm,
            "sample_count": counts[(scenario, arm)],
            "mean_local_reward": reward_sums[(scenario, arm)] / counts[(scenario, arm)],
        }
        for scenario in SCENARIOS
        for arm in ARMS
    ]
    audit = {
        "schema_version": 1,
        "summary_sha256": file_sha256(summary),
        "model_sha256": file_sha256(args.model_output),
        "action_grid_hash": loaded.action_grid_hash,
        "ridge": loaded.ridge,
        "exploration": loaded.exploration,
        "sample_count": sum(counts.values()),
        "statistics": statistics,
        "inputs": inputs,
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(
        json.dumps(audit, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    print(
        f"samples={audit['sample_count']} model_sha256={audit['model_sha256']}"
    )


if __name__ == "__main__":
    main()
