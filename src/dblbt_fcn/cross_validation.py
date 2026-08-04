"""Auditable cross-model direction evidence for preregistered H5."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import io
import json
import math
from numbers import Real
import os
from os import PathLike
from pathlib import Path
import tempfile
from typing import Literal

import numpy as np


Direction = Literal["improvement", "degradation", "inconclusive"]
H5Status = Literal["pass", "fail", "inconclusive"]

_SCENARIOS = ("static-4x4", "dynamic-4x4", "nonideal-6x6-300ms")
_SEEDS = (410, 523, 631)
_EVENT_METRICS = (
    "effective_airtime",
    "mean_delay_us",
    "collision_probability",
)
_NS3_METRICS = (
    "throughput_mbps",
    "mean_delay_us",
    "packet_loss_ratio",
    "simultaneous_access_collision_rate",
    "channel_occupancy",
)


@dataclass(frozen=True, slots=True)
class ScenarioConsistency:
    scenario: str
    event_direction: Direction
    ns3_direction: Direction
    agreement: bool | None
    event_metric_differences: tuple[tuple[str, float], ...]
    ns3_metric_differences: tuple[tuple[str, str, float], ...]


@dataclass(frozen=True, slots=True)
class CrossModelReport:
    scenarios: tuple[ScenarioConsistency, ...]
    agreements: tuple[bool | None, bool | None, bool | None]
    h5_status: H5Status


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be finite")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{label} must be finite")
    return numeric


def _finite_text(value: str, label: str) -> float:
    try:
        numeric = float(value)
    except ValueError as error:
        raise ValueError(f"{label} must be finite") from error
    if not math.isfinite(numeric):
        raise ValueError(f"{label} must be finite")
    return numeric


def _vote(difference: float, *, higher_is_better: bool) -> int:
    if abs(difference) <= 1e-12:
        return 0
    improved = difference > 0 if higher_is_better else difference < 0
    return 1 if improved else -1


def _direction(votes: list[int]) -> Direction:
    score = sum(votes)
    if score > 0:
        return "improvement"
    if score < 0:
        return "degradation"
    return "inconclusive"


def _event_differences(
    rows: list[dict[str, object]], scenario: str
) -> tuple[tuple[str, float], ...]:
    differences: list[tuple[str, float]] = []
    for metric in _EVENT_METRICS:
        values: dict[str, dict[int, float]] = {
            "tmc_db_lbt": {},
            "adaptive_db_lbt": {},
        }
        for row in rows:
            if row.get("scenario_id") != scenario:
                continue
            policy = row.get("policy")
            if policy not in values:
                raise ValueError("event cross-validation policy is invalid")
            seed = row.get("seed")
            if type(seed) is not int or seed not in _SEEDS:
                raise ValueError("event cross-validation seed is invalid")
            if seed in values[policy]:
                raise ValueError("duplicate event cross-validation pair")
            values[policy][seed] = _finite(row.get(metric), f"event {metric}")
        if any(set(policy_values) != set(_SEEDS) for policy_values in values.values()):
            raise ValueError("event cross-validation pairs are incomplete")
        baseline = np.mean([values["tmc_db_lbt"][seed] for seed in _SEEDS])
        adaptive = np.mean([values["adaptive_db_lbt"][seed] for seed in _SEEDS])
        differences.append((metric, float(adaptive - baseline)))
    return tuple(differences)


def _ns3_differences(
    rows: list[dict[str, object]], scenario: str
) -> tuple[tuple[str, str, float], ...]:
    selected: dict[tuple[str, str], float] = {}
    for row in rows:
        if row.get("scenario") != scenario:
            continue
        technology = row.get("technology")
        metric = row.get("metric")
        if technology not in ("wifi", "nru") or metric not in _NS3_METRICS:
            raise ValueError("ns-3 scenario metric key is invalid")
        if row.get("seed_count") != 3:
            raise ValueError("ns-3 scenario metric seed count is invalid")
        key = (technology, metric)
        if key in selected:
            raise ValueError("duplicate ns-3 scenario metric")
        selected[key] = _finite(
            row.get("paired_difference"), "ns-3 paired difference"
        )
    expected = {
        (technology, metric)
        for technology in ("wifi", "nru")
        for metric in _NS3_METRICS
    }
    if set(selected) != expected:
        raise ValueError("ns-3 scenario metrics are incomplete")
    return tuple(
        (technology, metric, selected[(technology, metric)])
        for technology in ("wifi", "nru")
        for metric in _NS3_METRICS
    )


def cross_model_consistency(
    event_rows: list[dict[str, object]],
    ns3_scenario_rows: list[dict[str, object]],
) -> CrossModelReport:
    """Compare adaptive-versus-TMC directions in the three fixed scenarios."""
    if len(event_rows) != 18 or any(
        row.get("matrix") != "ns3-cross-validation" for row in event_rows
    ):
        raise ValueError("event cross-validation requires exactly 18 rows")
    if len(ns3_scenario_rows) != 30:
        raise ValueError("ns-3 cross-validation requires exactly 30 rows")

    scenarios: list[ScenarioConsistency] = []
    agreements: list[bool | None] = []
    for scenario in _SCENARIOS:
        event_differences = _event_differences(event_rows, scenario)
        event_votes = [
            _vote(value, higher_is_better=metric == "effective_airtime")
            for metric, value in event_differences
        ]
        ns3_differences = _ns3_differences(ns3_scenario_rows, scenario)
        ns3_votes = [
            _vote(value, higher_is_better=metric == "throughput_mbps")
            for _, metric, value in ns3_differences
            if metric
            not in {"packet_loss_ratio", "channel_occupancy"}
        ]
        event_direction = _direction(event_votes)
        ns3_direction = _direction(ns3_votes)
        agreement = (
            None
            if "inconclusive" in (event_direction, ns3_direction)
            else event_direction == ns3_direction
        )
        agreements.append(agreement)
        scenarios.append(
            ScenarioConsistency(
                scenario=scenario,
                event_direction=event_direction,
                ns3_direction=ns3_direction,
                agreement=agreement,
                event_metric_differences=event_differences,
                ns3_metric_differences=ns3_differences,
            )
        )

    matches = sum(agreement is True for agreement in agreements)
    opposites = sum(agreement is False for agreement in agreements)
    h5_status: H5Status
    if matches >= 2:
        h5_status = "pass"
    elif opposites >= 2:
        h5_status = "fail"
    else:
        h5_status = "inconclusive"
    return CrossModelReport(
        scenarios=tuple(scenarios),
        agreements=tuple(agreements),  # type: ignore[arg-type]
        h5_status=h5_status,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_ns3_scenario_metrics(
    metrics_path: str | PathLike[str], reduction_path: str | PathLike[str]
) -> list[dict[str, object]]:
    """Load the exact scenario reductions bound by the ns-3 audit metadata."""
    metrics = Path(metrics_path).resolve(strict=True)
    reduction = Path(reduction_path).resolve(strict=True)
    try:
        metadata = json.loads(reduction.read_text(encoding="ascii"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("ns-3 reduction metadata is invalid") from error
    if (
        type(metadata) is not dict
        or metadata.get("audited") != 27
        or metadata.get("scenario_rows") != 30
        or metadata.get("scenario_metrics_sha256") != _sha256(metrics)
    ):
        raise ValueError("ns-3 scenario metric hash or audit metadata is invalid")
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
    try:
        with metrics.open(newline="", encoding="ascii") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames != fields:
                raise ValueError("ns-3 scenario metric schema is invalid")
            raw_rows = list(reader)
    except UnicodeError as error:
        raise ValueError("ns-3 scenario metrics must be ASCII") from error
    rows: list[dict[str, object]] = []
    for raw in raw_rows:
        if None in raw or any(value is None for value in raw.values()):
            raise ValueError("ns-3 scenario metric row schema is invalid")
        if raw["baseline_policy"] != "tmc" or raw["adaptive_policy"] != "adaptive":
            raise ValueError("ns-3 scenario metric policies are invalid")
        try:
            seed_count = int(raw["seed_count"])
        except ValueError as error:
            raise ValueError("ns-3 scenario metric seed count is invalid") from error
        baseline = _finite_text(raw["baseline_mean"], "ns-3 baseline mean")
        adaptive = _finite_text(raw["adaptive_mean"], "ns-3 adaptive mean")
        difference = _finite_text(
            raw["paired_difference"], "ns-3 paired difference"
        )
        if not math.isclose(
            adaptive - baseline, difference, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("ns-3 paired difference does not match means")
        rows.append(
            {
                "scenario": raw["scenario"],
                "technology": raw["technology"],
                "metric": raw["metric"],
                "seed_count": seed_count,
                "paired_difference": difference,
            }
        )
    if len(rows) != 30:
        raise ValueError("ns-3 scenario metric row count is invalid")
    return rows


def _csv_bytes(fields: tuple[str, ...], rows: list[dict[str, object]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output, fieldnames=fields, extrasaction="raise", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("ascii")


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


def _final_hypothesis_rows(path: Path, h5_status: H5Status) -> list[dict[str, str]]:
    with path.open(newline="", encoding="ascii") as source:
        reader = csv.DictReader(source)
        fields = (
            "hypothesis",
            "status",
            "threshold",
            "paired_difference",
            "lower_95",
            "upper_95",
        )
        if reader.fieldnames != list(fields):
            raise ValueError("event hypothesis table schema is invalid")
        rows = list(reader)
    if [row["hypothesis"] for row in rows] != ["H1", "H2", "H3", "H4", "H5"]:
        raise ValueError("event hypothesis table order is invalid")
    if rows[-1] != {
        "hypothesis": "H5",
        "status": "not_evaluated",
        "threshold": "",
        "paired_difference": "",
        "lower_95": "",
        "upper_95": "",
    }:
        raise ValueError("event hypothesis H5 row is invalid")
    rows[-1]["status"] = h5_status
    return rows


def _latex_bytes(fields: tuple[str, ...], rows: list[dict[str, str]]) -> bytes:
    def escape(value: str) -> str:
        return value.replace("_", r"\_").replace("%", r"\%")

    lines = [
        f"\\begin{{tabular}}{{{'l' * len(fields)}}}",
        " & ".join(fields) + r" \\",
    ]
    lines.extend(
        " & ".join(escape(row[field]) for field in fields) + r" \\" for row in rows
    )
    lines.append(r"\end{tabular}")
    return ("\n".join(lines) + "\n").encode("ascii")


def write_cross_model_evidence(
    report: CrossModelReport,
    output_dir: str | PathLike[str],
    *,
    event_summary_path: str | PathLike[str],
    ns3_metrics_path: str | PathLike[str],
    ns3_reduction_path: str | PathLike[str],
    event_hypotheses_path: str | PathLike[str],
) -> tuple[Path, Path, Path, Path]:
    """Write deterministic H5 scenario, final hypothesis, and audit artifacts."""
    if not isinstance(report, CrossModelReport):
        raise TypeError("report must be a CrossModelReport")
    output = Path(output_dir).resolve(strict=False)
    event_summary = Path(event_summary_path).resolve(strict=True)
    ns3_metrics = Path(ns3_metrics_path).resolve(strict=True)
    ns3_reduction = Path(ns3_reduction_path).resolve(strict=True)
    event_hypotheses = Path(event_hypotheses_path).resolve(strict=True)
    scenario_fields = (
        "scenario",
        "event_direction",
        "ns3_direction",
        "agreement",
        "event_effective_airtime_difference",
        "event_mean_delay_us_difference",
        "event_collision_probability_difference",
        "ns3_wifi_throughput_mbps_difference",
        "ns3_wifi_mean_delay_us_difference",
        "ns3_wifi_packet_loss_ratio_difference",
        "ns3_wifi_simultaneous_access_collision_rate_difference",
        "ns3_wifi_channel_occupancy_difference",
        "ns3_nru_throughput_mbps_difference",
        "ns3_nru_mean_delay_us_difference",
        "ns3_nru_packet_loss_ratio_difference",
        "ns3_nru_simultaneous_access_collision_rate_difference",
        "ns3_nru_channel_occupancy_difference",
    )
    scenario_rows: list[dict[str, object]] = []
    for scenario in report.scenarios:
        event = dict(scenario.event_metric_differences)
        ns3 = {
            (technology, metric): value
            for technology, metric, value in scenario.ns3_metric_differences
        }
        scenario_rows.append(
            {
                "scenario": scenario.scenario,
                "event_direction": scenario.event_direction,
                "ns3_direction": scenario.ns3_direction,
                "agreement": (
                    "" if scenario.agreement is None else str(scenario.agreement).lower()
                ),
                "event_effective_airtime_difference": event["effective_airtime"],
                "event_mean_delay_us_difference": event["mean_delay_us"],
                "event_collision_probability_difference": event[
                    "collision_probability"
                ],
                "ns3_wifi_throughput_mbps_difference": ns3[
                    ("wifi", "throughput_mbps")
                ],
                "ns3_wifi_mean_delay_us_difference": ns3[
                    ("wifi", "mean_delay_us")
                ],
                "ns3_wifi_packet_loss_ratio_difference": ns3[
                    ("wifi", "packet_loss_ratio")
                ],
                "ns3_wifi_simultaneous_access_collision_rate_difference": ns3[
                    ("wifi", "simultaneous_access_collision_rate")
                ],
                "ns3_wifi_channel_occupancy_difference": ns3[
                    ("wifi", "channel_occupancy")
                ],
                "ns3_nru_throughput_mbps_difference": ns3[
                    ("nru", "throughput_mbps")
                ],
                "ns3_nru_mean_delay_us_difference": ns3[
                    ("nru", "mean_delay_us")
                ],
                "ns3_nru_packet_loss_ratio_difference": ns3[
                    ("nru", "packet_loss_ratio")
                ],
                "ns3_nru_simultaneous_access_collision_rate_difference": ns3[
                    ("nru", "simultaneous_access_collision_rate")
                ],
                "ns3_nru_channel_occupancy_difference": ns3[
                    ("nru", "channel_occupancy")
                ],
            }
        )
    scenario_payload = _csv_bytes(scenario_fields, scenario_rows)
    hypothesis_fields = (
        "hypothesis",
        "status",
        "threshold",
        "paired_difference",
        "lower_95",
        "upper_95",
    )
    hypothesis_rows = _final_hypothesis_rows(event_hypotheses, report.h5_status)
    hypothesis_payload = _csv_bytes(hypothesis_fields, hypothesis_rows)
    hypothesis_latex = _latex_bytes(hypothesis_fields, hypothesis_rows)
    audit_payload = (
        json.dumps(
            {
                "schema_version": 2,
                "h5_status": report.h5_status,
                "agreements": list(report.agreements),
                "event_summary_sha256": _sha256(event_summary),
                "ns3_metrics_sha256": _sha256(ns3_metrics),
                "ns3_reduction_sha256": _sha256(ns3_reduction),
                "event_hypotheses_sha256": _sha256(event_hypotheses),
                "scenario_table_sha256": hashlib.sha256(
                    scenario_payload
                ).hexdigest(),
                "final_hypotheses_sha256": hashlib.sha256(
                    hypothesis_payload
                ).hexdigest(),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")
    paths = (
        output / "cross-model-scenarios.csv",
        output / "final-hypotheses.csv",
        output / "final-hypotheses.tex",
        output / "cross-model-audit.json",
    )
    for path, payload in zip(
        paths,
        (scenario_payload, hypothesis_payload, hypothesis_latex, audit_payload),
        strict=True,
    ):
        _atomic_write(path, payload)
    return paths
