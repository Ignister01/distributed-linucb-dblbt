"""Fixed-profile conflict and empirical adaptation-time evidence."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from dblbt_fcn.regime_evidence import (
    adaptation_transitions,
    rank_fixed_arms,
    split_phase_segments,
)
from dblbt_fcn.stats import SUMMARY_FIELDS


def _fixed_row(
    scenario: str, arm: int, seed: int, utility: float
) -> dict[str, object]:
    return {
        "scenario_id": scenario,
        "policy": "pretrain_arm",
        "seed": seed,
        "ablation": None,
        "arm_id": arm,
        "evaluation_utility": utility,
        "collision_probability": 0.1,
        "effective_airtime": 0.6,
        "p95_delay_us": 100.0,
        "jain_fairness": 0.95,
    }


def _fixed_rows() -> list[dict[str, object]]:
    means = {
        "regime-a": (0.91, 0.70, 0.60, 0.50),
        "regime-b": (0.60, 0.96, 0.65, 0.55),
        "regime-c": (0.80, 0.79, 0.70, 0.60),
    }
    offsets = (-0.01, 0.0, 0.01)
    return [
        _fixed_row(
            scenario, arm, seed, arm_means[arm] + offsets[seed - 1]
        )
        for scenario, arm_means in means.items()
        for arm, mean in enumerate(arm_means)
        for seed in (1, 2, 3)
    ]


def test_fixed_arm_ranking_detects_conflicting_scenario_optima() -> None:
    analysis = rank_fixed_arms(_fixed_rows())
    rankings = {row.scenario_id: row for row in analysis.rankings}

    assert rankings["regime-a"].best_arm == 0
    assert rankings["regime-a"].runner_up_arm == 1
    assert rankings["regime-a"].best_mean == pytest.approx(0.91)
    assert rankings["regime-a"].paired_margin == pytest.approx(0.21)
    assert rankings["regime-a"].lower_95 > 0
    assert rankings["regime-b"].best_arm == 1
    assert rankings["regime-c"].best_arm == 0
    assert analysis.best_global_arm == 1
    assert analysis.minimax_arm == 1
    assert {
        (row.scenario_a, row.scenario_b)
        for row in analysis.conflicts
    } == {("regime-a", "regime-b"), ("regime-b", "regime-c")}
    assert all(row.lower_a_over_b > 0 for row in analysis.conflicts)
    assert all(row.lower_b_over_a > 0 for row in analysis.conflicts)


def test_fixed_arm_ranking_is_deterministic_and_rejects_unpaired_rows() -> None:
    rows = _fixed_rows()

    assert rank_fixed_arms(rows) == rank_fixed_arms(list(reversed(rows)))
    with pytest.raises(ValueError, match="duplicate"):
        rank_fixed_arms([*rows, rows[0]])
    with pytest.raises(ValueError, match="paired seeds"):
        rank_fixed_arms(rows[:-1])


def _phase_rows(
    second_phase_rewards: list[float], *, run_id: str
) -> list[dict[str, object]]:
    first_phase_rewards = [1.0] * len(second_phase_rewards)
    rows: list[dict[str, object]] = []
    for phase_index, (phase_id, rewards) in enumerate(
        (("first", first_phase_rewards), ("second", second_phase_rewards))
    ):
        for phase_round in range(32 * len(rewards)):
            reward_index = phase_round // 32
            decisions: list[dict[str, object]] = []
            if phase_round % 32 == 31:
                decisions = [{"reward": rewards[reward_index]}]
            rows.append(
                {
                    "run_id": run_id,
                    "round_id": len(rows),
                    "phase_id": phase_id,
                    "phase_index": phase_index,
                    "phase_round": phase_round,
                    "change_point": phase_round == 0,
                    "decisions": decisions,
                }
            )
    return rows


def test_phase_segments_split_repeated_phase_indices_at_change_points() -> None:
    rows = [
        {
            "round_id": round_id,
            "phase_id": phase_id,
            "phase_index": phase_index,
            "change_point": change_point,
        }
        for round_id, phase_id, phase_index, change_point in (
            (0, "low", 0, True),
            (1, "low", 0, False),
            (2, "high", 1, True),
            (3, "high", 1, False),
            (4, "low", 0, True),
            (5, "low", 0, False),
        )
    ]

    segments = split_phase_segments(rows)

    assert [segment[0]["phase_id"] for segment in segments] == [
        "low",
        "high",
        "low",
    ]
    assert [len(segment) for segment in segments] == [2, 2, 2]


def test_adaptation_time_finds_persistent_ninety_percent_recovery() -> None:
    rows = _phase_rows([0.0] * 8 + [1.0] * 12, run_id="a" * 16)

    transition = adaptation_transitions(rows)[0]

    assert transition.run_id == "a" * 16
    assert transition.phase_id == "second"
    assert transition.change_round == 640
    assert transition.recovery_rounds == 512
    assert transition.dwell_rounds == 640
    assert transition.censored is False


def test_adaptation_time_marks_nonrecovery_as_censored() -> None:
    rows = _phase_rows([0.0] * 15 + [1.0] * 5, run_id="b" * 16)

    transition = adaptation_transitions(rows)[0]

    assert transition.recovery_rounds is None
    assert transition.censored is True


def _summary_row(
    source: dict[str, object], index: int
) -> dict[str, object]:
    values: dict[str, object] = {
        "run_id": f"{index:016x}",
        "matrix": "fixed-discovery",
        "scenario_id": source["scenario_id"],
        "policy": source["policy"],
        "seed": source["seed"],
        "ablation": "",
        "arm_id": source["arm_id"],
        "wifi_nodes": 2,
        "nru_nodes": 2,
        "traffic": "poisson",
        "interference_interval_ms": "",
        "interruption_std": 0.0,
        "join_interval_rounds": "",
        "lifetime_rounds": "",
        "config_hash": f"{index:064x}",
        "rounds": 20_000,
        "elapsed_us": 1_000_000,
        "successes": 18_000,
        "collisions": 2_000,
        "collision_probability": source["collision_probability"],
        "effective_airtime": source["effective_airtime"],
        "mean_delay_us": 80.0,
        "p95_delay_us": source["p95_delay_us"],
        "jain_fairness": source["jain_fairness"],
        "evaluation_utility": source["evaluation_utility"],
        "decision_count": 100,
        "switch_count": 0,
        "training_sample_count": 100,
    }
    return values


def test_regime_rank_cli_writes_rankings_conflicts_and_audit(
    tmp_path: Path,
) -> None:
    from dblbt_fcn import cli

    summary = tmp_path / "fixed.csv"
    with summary.open("w", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(
            _summary_row(row, index)
            for index, row in enumerate(_fixed_rows(), start=1)
        )
    output = tmp_path / "rank"

    result = CliRunner().invoke(
        cli.app,
        ["regime-rank", str(summary), "--output-dir", str(output)],
    )

    assert result.exit_code == 0, result.output
    assert {path.name for path in output.iterdir()} == {
        "scenario-rankings.csv",
        "conflicting-optima.csv",
        "fixed-arm-aggregates.csv",
        "regime-rank-audit.json",
    }
    audit = json.loads(
        (output / "regime-rank-audit.json").read_text(encoding="ascii")
    )
    assert audit["scenario_count"] == 3
    assert audit["best_global_arm"] == 1
    assert len(audit["input_sha256"]) == 64


def test_adaptation_report_cli_reads_validated_phased_manifests(
    tmp_path: Path,
) -> None:
    from dblbt_fcn import cli
    from dblbt_fcn.experiment import MatrixSpec, expand_matrix
    from dblbt_fcn.simulation import run_job

    matrix = MatrixSpec.model_validate(
        {
            "version": 1,
            "name": "adaptation-test",
            "rounds": 1_280,
            "seeds": [17],
            "policies": ["adaptive_db_lbt"],
            "scenarios": [
                {
                    "id": "switching",
                    "wifi_nodes": 1,
                    "nru_nodes": 1,
                    "traffic": "poisson",
                    "poisson_rate_packets_ms": 0.02,
                    "phases": [
                        {
                            "id": "light",
                            "duration_rounds": 640,
                            "active_wifi_nodes": 1,
                            "active_nru_nodes": 1,
                            "poisson_rate_packets_ms": 0.015,
                        },
                        {
                            "id": "busy",
                            "duration_rounds": 640,
                            "active_wifi_nodes": 1,
                            "active_nru_nodes": 1,
                            "poisson_rate_packets_ms": 0.035,
                        },
                    ],
                }
            ],
        }
    )
    root = tmp_path / "runs"
    run_job(expand_matrix(matrix)[0], root)
    output = tmp_path / "adaptation"

    result = CliRunner().invoke(
        cli.app,
        [
            "adaptation-report",
            str(root / "manifests"),
            "--output-dir",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert {path.name for path in output.iterdir()} == {
        "adaptation-transitions.csv",
        "adaptation-summary.json",
        "adaptation-audit.json",
    }
    summary = json.loads(
        (output / "adaptation-summary.json").read_text(encoding="ascii")
    )
    assert summary["transition_count"] == 1
    assert summary["min_dwell_rounds"] == 640
