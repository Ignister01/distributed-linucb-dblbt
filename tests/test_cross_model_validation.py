from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from dblbt_fcn.cross_validation import (
    cross_model_consistency,
    load_ns3_scenario_metrics,
    write_cross_model_evidence,
)


SCENARIOS = ("static-4x4", "dynamic-4x4", "nonideal-6x6-300ms")
SEEDS = (410, 523, 631)


def _event_rows(
    directions: dict[str, str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scenario in SCENARIOS:
        direction = directions[scenario]
        for seed in SEEDS:
            for policy in ("tmc_db_lbt", "adaptive_db_lbt"):
                adaptive = policy == "adaptive_db_lbt"
                sign = 0.0
                if adaptive and direction == "improvement":
                    sign = 1.0
                elif adaptive and direction == "degradation":
                    sign = -1.0
                rows.append(
                    {
                        "matrix": "ns3-cross-validation",
                        "scenario_id": scenario,
                        "policy": policy,
                        "seed": seed,
                        "effective_airtime": 0.5 + 0.1 * sign,
                        "mean_delay_us": 100.0 - 10.0 * sign,
                        "collision_probability": 0.2 - 0.02 * sign,
                    }
                )
    return rows


def _ns3_rows(
    directions: dict[str, str],
    *,
    occupancy_difference: float = 0.0,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scenario in SCENARIOS:
        direction = directions[scenario]
        sign = 0.0
        if direction == "improvement":
            sign = 1.0
        elif direction == "degradation":
            sign = -1.0
        for technology in ("wifi", "nru"):
            for metric in (
                "throughput_mbps",
                "mean_delay_us",
                "collision_probability",
                "channel_occupancy",
            ):
                if metric == "throughput_mbps":
                    difference = sign
                elif metric in ("mean_delay_us", "collision_probability"):
                    difference = -sign
                else:
                    difference = occupancy_difference
                rows.append(
                    {
                        "scenario": scenario,
                        "technology": technology,
                        "metric": metric,
                        "seed_count": 3,
                        "paired_difference": difference,
                    }
                )
    return rows


def test_cross_model_h5_passes_with_two_matching_scenario_directions() -> None:
    report = cross_model_consistency(
        _event_rows(
            {
                "static-4x4": "improvement",
                "dynamic-4x4": "degradation",
                "nonideal-6x6-300ms": "improvement",
            }
        ),
        _ns3_rows(
            {
                "static-4x4": "improvement",
                "dynamic-4x4": "improvement",
                "nonideal-6x6-300ms": "improvement",
            }
        ),
    )

    assert report.agreements == (True, False, True)
    assert report.h5_status == "pass"


def test_cross_model_h5_preserves_inconclusive_scenarios() -> None:
    report = cross_model_consistency(
        _event_rows(
            {
                "static-4x4": "inconclusive",
                "dynamic-4x4": "improvement",
                "nonideal-6x6-300ms": "inconclusive",
            }
        ),
        _ns3_rows({scenario: "degradation" for scenario in SCENARIOS}),
    )

    assert report.agreements == (None, False, None)
    assert report.h5_status == "inconclusive"


def test_cross_model_h5_fails_with_two_opposite_scenario_directions() -> None:
    report = cross_model_consistency(
        _event_rows({scenario: "improvement" for scenario in SCENARIOS}),
        _ns3_rows(
            {
                "static-4x4": "degradation",
                "dynamic-4x4": "degradation",
                "nonideal-6x6-300ms": "improvement",
            }
        ),
    )

    assert report.agreements == (False, False, True)
    assert report.h5_status == "fail"


def test_channel_occupancy_is_reported_but_does_not_set_h5_direction() -> None:
    event = _event_rows({scenario: "improvement" for scenario in SCENARIOS})
    baseline = cross_model_consistency(
        event,
        _ns3_rows({scenario: "improvement" for scenario in SCENARIOS}),
    )
    changed = cross_model_consistency(
        event,
        _ns3_rows(
            {scenario: "improvement" for scenario in SCENARIOS},
            occupancy_difference=-1_000_000.0,
        ),
    )

    assert changed.agreements == baseline.agreements
    assert changed.h5_status == baseline.h5_status
    assert (
        changed.scenarios[0].ns3_metric_differences
        != baseline.scenarios[0].ns3_metric_differences
    )


def test_cross_model_evidence_outputs_are_deterministic(tmp_path: Path) -> None:
    report = cross_model_consistency(
        _event_rows(
            {
                "static-4x4": "inconclusive",
                "dynamic-4x4": "improvement",
                "nonideal-6x6-300ms": "inconclusive",
            }
        ),
        _ns3_rows({scenario: "degradation" for scenario in SCENARIOS}),
    )
    hypotheses = tmp_path / "hypotheses.csv"
    hypotheses.write_text(
        "hypothesis,status,threshold,paired_difference,lower_95,upper_95\n"
        "H1,pass,-0.1,0.0,-0.1,0.1\n"
        "H2,fail,0.1,0.01,0.0,0.02\n"
        "H3,pass,-0.01,0.0,-0.01,0.0\n"
        "H4,pass,0.0,0.01,0.0,0.02\n"
        "H5,not_evaluated,,,,\n",
        encoding="ascii",
    )
    event_summary = tmp_path / "event.csv"
    ns3_metrics = tmp_path / "ns3.csv"
    reduction = tmp_path / "reduction.json"
    for path, payload in (
        (event_summary, b"event\n"),
        (ns3_metrics, b"ns3\n"),
        (reduction, b"{}\n"),
    ):
        path.write_bytes(payload)

    first = write_cross_model_evidence(
        report,
        tmp_path / "output",
        event_summary_path=event_summary,
        ns3_metrics_path=ns3_metrics,
        ns3_reduction_path=reduction,
        event_hypotheses_path=hypotheses,
    )
    first_bytes = {path.name: path.read_bytes() for path in first}
    second = write_cross_model_evidence(
        report,
        tmp_path / "output",
        event_summary_path=event_summary,
        ns3_metrics_path=ns3_metrics,
        ns3_reduction_path=reduction,
        event_hypotheses_path=hypotheses,
    )

    assert [path.name for path in first] == [
        "cross-model-scenarios.csv",
        "final-hypotheses.csv",
        "final-hypotheses.tex",
        "cross-model-audit.json",
    ]
    assert {path.name: path.read_bytes() for path in second} == first_bytes
    final_hypotheses = first[1].read_text(encoding="ascii")
    assert "H5,inconclusive,,,," in final_hypotheses


def test_ns3_scenario_metrics_are_bound_to_reduction_hash(tmp_path: Path) -> None:
    metrics = tmp_path / "scenario-metrics.csv"
    fields = [
        "scenario",
        "technology",
        "metric",
        "seed_count",
        "baseline_policy",
        "adaptive_policy",
        "baseline_mean",
        "adaptive_mean",
        "paired_difference",
    ]
    with metrics.open("w", newline="", encoding="ascii") as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in _ns3_rows({scenario: "improvement" for scenario in SCENARIOS}):
            difference = float(row["paired_difference"])
            writer.writerow(
                {
                    **row,
                    "baseline_policy": "tmc",
                    "adaptive_policy": "adaptive",
                    "baseline_mean": 10.0,
                    "adaptive_mean": 10.0 + difference,
                }
            )
    reduction = tmp_path / "reduction.json"
    reduction.write_text(
        json.dumps(
            {
                "audited": 27,
                "scenario_rows": 24,
                "scenario_metrics_sha256": hashlib.sha256(
                    metrics.read_bytes()
                ).hexdigest(),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="ascii",
    )

    assert len(load_ns3_scenario_metrics(metrics, reduction)) == 24
    metrics.write_bytes(metrics.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="hash"):
        load_ns3_scenario_metrics(metrics, reduction)
