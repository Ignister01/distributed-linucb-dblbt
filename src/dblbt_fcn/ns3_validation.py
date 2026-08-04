"""Strict model interchange and audit helpers for ns-3 validation."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from os import PathLike
from pathlib import Path
import tempfile
from typing import Literal
import sqlite3

import numpy as np

from .config import adaptive_arms
from .linucb import LinUCB
from .workflows import action_grid_hash


_MODEL_MAGIC = "dblbt_ns3_model 1"
_MODEL_SCHEMA_VERSION = 1
_NUM_ARMS = 24
_CONTEXT_DIM = 11
_DECISION_COLUMNS = (
    "node_id",
    "decision_round",
    "arm_id",
    "kappa",
    "alpha",
    "beta",
    "m",
    "b_init",
    "reward",
    *(f"context_{index}" for index in range(_CONTEXT_DIM)),
)

Ns3Policy = Literal["random", "tmc", "adaptive"]
Ns3MetricName = Literal[
    "throughput_mbps",
    "mean_delay_us",
    "packet_loss_ratio",
    "simultaneous_access_collision_rate",
    "channel_occupancy",
]
Ns3Scenario = Literal[
    "static-4x4",
    "dynamic-4x4",
    "nonideal-6x6-300ms",
    "smoke-tmc",
    "smoke-adaptive",
]

_RAW_TABLE_SCHEMAS = {
    "channel_occupancy_": (
        ("TECHNOLOGY", "STRING", 1, 0),
        ("VALUE", "DOUBLE", 1, 0),
        ("SEED", "INT", 1, 0),
        ("RUN", "INT", 1, 0),
    ),
    "simultaneous_tx_": (
        ("UID", "INT", 1, 0),
        ("SIMULTANEOUS_TX_SAME_TECH", "INT", 1, 0),
        ("SIMULTANEOUS_TX_OTHER_TECH", "INT", 1, 0),
        ("TOTALTX", "INT", 1, 0),
        ("SEED", "INT", 1, 0),
        ("RUN", "INT", 1, 0),
    ),
    "mac_data_tx_failed_": (
        ("UID", "INT", 1, 0),
        ("NUMBER", "INT", 1, 0),
        ("BYTES", "INT", 1, 0),
        ("SEED", "INT", 1, 0),
        ("RUN", "INT", 1, 0),
    ),
    "sinr_results_": (
        ("UID", "INT", 1, 0),
        ("SINR", "DOUBLE", 1, 0),
        ("SEED", "INT", 1, 0),
        ("RUN", "INT", 1, 0),
    ),
    "e2e_": (
        ("TECHNOLOGY", "STRING", 1, 0),
        ("THROUGHPUT_MBPS", "DOUBLE", 1, 0),
        ("TXBYTES", "INT", 1, 0),
        ("RXBYTES", "INT", 1, 0),
        ("LATENCY_US", "DOUBLE", 1, 0),
        ("JITTER_US", "DOUBLE", 1, 0),
        ("ADDR", "STRING", 1, 0),
        ("SEED", "INT", 1, 0),
        ("RUN", "INT", 1, 0),
    ),
}


@dataclass(frozen=True, slots=True)
class Ns3ModelExport:
    """Validated fixed model state consumed by the ns-3 controller."""

    schema_version: int
    num_arms: int
    context_dim: int
    ridge: float
    exploration: float
    action_grid_hash: str
    source_model_sha256: str
    profiles: tuple[tuple[int, int, int, int, int], ...]
    A: np.ndarray
    b: np.ndarray


@dataclass(frozen=True, slots=True)
class Ns3ValidationJob:
    """One immutable member of the paired packet-level matrix."""

    policy: Ns3Policy
    scenario: Ns3Scenario
    seed: int
    run_id: int
    wifi_aps: int
    nru_gnbs: int
    sim_time_s: float
    app_start_s: float
    dynamic_load_change: bool
    interference_interval_ms: int | None
    interference_duration_ms: int | None
    shadowing_enabled: bool
    node_rate_bps: int = 2_000_000
    traffic_mode: str = "aggregate-saturated-cbr"
    alpha: int = 11
    cold_start_attempts: int = 8
    decision_interval: int = 32
    context_dim: int = 11
    num_arms: int = 24

    @property
    def job_id(self) -> str:
        return f"{self.scenario}__seed-{self.seed}__{self.policy}"

    @property
    def database_name(self) -> str:
        return f"{self.job_id}.db"


@dataclass(frozen=True, slots=True)
class Ns3AuditReport:
    """Read-only evidence returned after all formal databases are accepted."""

    audited: int
    model_sha256: str
    model_export_sha256: str
    adaptive_decisions: int
    database_hashes: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class Ns3PairedMetric:
    """One TMC-versus-adaptive packet-level metric pair."""

    scenario: str
    seed: int
    technology: Literal["wifi", "nru"]
    metric: Ns3MetricName
    baseline_value: float
    adaptive_value: float
    paired_difference: float


@dataclass(frozen=True, slots=True)
class Ns3ScenarioMetric:
    """Mean paired packet-level metric for one validation scenario."""

    scenario: str
    technology: Literal["wifi", "nru"]
    metric: Ns3MetricName
    seed_count: int
    baseline_mean: float
    adaptive_mean: float
    paired_difference: float


@dataclass(frozen=True, slots=True)
class Ns3ReductionReport:
    """Deterministic packet-level reductions derived from audited databases."""

    audited: int
    model_sha256: str
    model_export_sha256: str
    database_hashes: tuple[tuple[str, str], ...]
    per_seed: tuple[Ns3PairedMetric, ...]
    scenario_means: tuple[Ns3ScenarioMetric, ...]


def validation_jobs() -> tuple[Ns3ValidationJob, ...]:
    """Return the preregistered 27 jobs in deterministic order."""
    jobs: list[Ns3ValidationJob] = []
    for scenario in (
        "static-4x4",
        "dynamic-4x4",
        "nonideal-6x6-300ms",
    ):
        nonideal = scenario == "nonideal-6x6-300ms"
        for seed in (410, 523, 631):
            for policy in ("random", "tmc", "adaptive"):
                jobs.append(
                    Ns3ValidationJob(
                        policy=policy,
                        scenario=scenario,
                        seed=seed,
                        run_id=1,
                        wifi_aps=6 if nonideal else 4,
                        nru_gnbs=6 if nonideal else 4,
                        sim_time_s=2.0,
                        app_start_s=0.2,
                        dynamic_load_change=scenario == "dynamic-4x4",
                        interference_interval_ms=300 if nonideal else None,
                        interference_duration_ms=2 if nonideal else None,
                        shadowing_enabled=True,
                    )
                )
    return tuple(jobs)


def _job_for_artifact(job_id: str) -> Ns3ValidationJob:
    for job in validation_jobs():
        if job.job_id == job_id:
            return job
    if job_id in {"tmc", "adaptive"}:
        return Ns3ValidationJob(
            policy=job_id,
            scenario=f"smoke-{job_id}",
            seed=410,
            run_id=1,
            wifi_aps=1,
            nru_gnbs=1,
            sim_time_s=0.8,
            app_start_s=0.2,
            dynamic_load_change=False,
            interference_interval_ms=None,
            interference_duration_ms=None,
            shadowing_enabled=False,
        )
    raise ValueError(f"unknown ns-3 validation job: {job_id}")


def _validation_inputs(
    model_path: str | PathLike[str] | None,
    expected_patch_sha256: str | None,
    expected_scenario_sha256: str | None,
    expected_build_profile: str | None,
) -> tuple[Ns3ModelExport, str, str, str]:
    if model_path is None:
        raise ValueError("ns-3 validation audit requires the exported model")
    if expected_patch_sha256 is None or expected_scenario_sha256 is None:
        raise ValueError("ns-3 validation audit requires source hashes")
    if expected_build_profile != "optimized":
        raise ValueError("ns-3 validation requires the optimized build profile")
    exported_path = _path(model_path).resolve(strict=True)
    return (
        read_ns3_model(exported_path),
        _sha256(exported_path),
        _hash("patch_sha256", expected_patch_sha256),
        _hash("scenario_sha256", expected_scenario_sha256),
    )


def validate_ns3_job_database(
    database: str | PathLike[str],
    *,
    job_id: str,
    model_path: str | PathLike[str] | None = None,
    expected_patch_sha256: str | None = None,
    expected_scenario_sha256: str | None = None,
    expected_build_profile: str | None = None,
) -> int:
    """Strictly validate one formal or smoke database without modifying it."""
    model, export_hash, patch_hash, scenario_hash = _validation_inputs(
        model_path,
        expected_patch_sha256,
        expected_scenario_sha256,
        expected_build_profile,
    )
    return _validate_ns3_database(
        _path(database).resolve(strict=True),
        _job_for_artifact(job_id),
        model,
        export_hash,
        patch_hash,
        scenario_hash,
    )


def _finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _integer(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int)


def _official_rows(
    connection: sqlite3.Connection,
    prefix: str,
    job_id: str,
    *,
    allow_empty: bool = False,
) -> list[tuple[object, ...]]:
    names = [
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name GLOB ? ORDER BY name",
            (f"{prefix}*",),
        ).fetchall()
    ]
    if len(names) != 1:
        raise ValueError(
            f"official raw evidence schema failed for {job_id}: {prefix}"
        )
    table = names[0]
    quoted = '"' + table.replace('"', '""') + '"'
    schema = tuple(
        (row[1], row[2], row[3], row[5])
        for row in connection.execute(f"PRAGMA table_info({quoted})").fetchall()
    )
    if schema != _RAW_TABLE_SCHEMAS[prefix]:
        raise ValueError(
            f"official raw evidence schema failed for {job_id}: {table}"
        )
    rows = connection.execute(f"SELECT * FROM {quoted}").fetchall()
    if not rows and not allow_empty:
        raise ValueError(
            f"official raw evidence values failed for {job_id}: {table}"
        )
    return rows


def _validate_official_raw_evidence(
    connection: sqlite3.Connection,
    job: Ns3ValidationJob,
) -> None:
    official_uids = set(range(2 * (job.wifi_aps + job.nru_gnbs)))
    occupancy = _official_rows(connection, "channel_occupancy_", job.job_id)
    if {row[0] for row in occupancy} != {"wifi", "nru"} or any(
        not _finite_number(value)
        or not 0 <= float(value) <= 1
        or not _integer(seed)
        or seed != job.seed
        or not _integer(run)
        or run != job.run_id
        for _, value, seed, run in occupancy
    ):
        raise ValueError(
            f"official raw evidence values failed for {job.job_id}"
        )

    simultaneous = _official_rows(connection, "simultaneous_tx_", job.job_id)
    simultaneous_uids = {row[0] for row in simultaneous}
    if not simultaneous_uids <= official_uids or any(
        not all(
            _integer(value) for value in (uid, same, other, total, seed, run)
        )
        or uid not in official_uids
        or min(same, other, total) < 0
        or same > total
        or other > total
        or seed != job.seed
        or run != job.run_id
        for uid, same, other, total, seed, run in simultaneous
    ):
        raise ValueError(
            f"official raw evidence values failed for {job.job_id}"
        )

    failures = _official_rows(
        connection,
        "mac_data_tx_failed_",
        job.job_id,
        allow_empty=True,
    )
    if not {row[0] for row in failures} <= official_uids or any(
        not all(
            _integer(value) for value in (uid, number, byte_count, seed, run)
        )
        or uid not in official_uids
        or number < 0
        or byte_count < 0
        or seed != job.seed
        or run != job.run_id
        for uid, number, byte_count, seed, run in failures
    ):
        raise ValueError(
            f"official raw evidence values failed for {job.job_id}"
        )

    sinr = _official_rows(connection, "sinr_results_", job.job_id)
    sinr_uids = {row[0] for row in sinr}
    if not sinr_uids <= official_uids or any(
        not _integer(uid)
        or uid not in official_uids
        or not _finite_number(value)
        or not _integer(seed)
        or seed != job.seed
        or not _integer(run)
        or run != job.run_id
        for uid, value, seed, run in sinr
    ):
        raise ValueError(
            f"official raw evidence values failed for {job.job_id}"
        )

    e2e = _official_rows(connection, "e2e_", job.job_id)
    if {row[0] for row in e2e} != {"wifi", "nru"} or any(
        technology not in {"wifi", "nru"}
        or not _finite_number(throughput)
        or float(throughput) < 0
        or not _integer(tx_bytes)
        or not _integer(rx_bytes)
        or tx_bytes < 0
        or not 0 <= rx_bytes <= tx_bytes
        or not _finite_number(latency)
        or float(latency) < 0
        or not _finite_number(jitter)
        or float(jitter) < 0
        or not isinstance(address, str)
        or not address
        or not _integer(seed)
        or seed != job.seed
        or not _integer(run)
        or run != job.run_id
        for (
            technology,
            throughput,
            tx_bytes,
            rx_bytes,
            latency,
            jitter,
            address,
            seed,
            run,
        ) in e2e
    ):
        raise ValueError(
            f"official raw evidence values failed for {job.job_id}"
        )


def _validate_ns3_database(
    database: Path,
    job: Ns3ValidationJob,
    model: Ns3ModelExport,
    export_hash: str,
    patch_hash: str,
    scenario_hash: str,
) -> int:
    uri = f"file:{database.as_posix()}?mode=ro&immutable=1"
    try:
        connection = sqlite3.connect(uri, uri=True)
        decision_columns = tuple(
            row[1]
            for row in connection.execute(
                'PRAGMA table_info("dblbt_decisions")'
            ).fetchall()
        )
        if decision_columns != _DECISION_COLUMNS:
            raise ValueError(
                f"invalid dblbt_decisions schema for {job.job_id}"
            )
        metadata = connection.execute(
            "SELECT schema_version, job_id, policy, scenario, seed, "
            "run_id, wifi_aps, nru_gnbs, node_rate_bps, traffic_mode, "
            "sim_time_s, shadowing_enabled, srs_enabled, alpha, "
            "cold_start_attempts, "
            "decision_interval, context_dim, num_arms, model_sha256, "
            "model_export_sha256, action_grid_hash, ns3_commit, "
            "nr_commit, nru_commit, patch_sha256, scenario_sha256 "
            "FROM validation_metadata"
        ).fetchall()
        expected_metadata = (
            2,
            job.job_id,
            job.policy,
            job.scenario,
            job.seed,
            job.run_id,
            job.wifi_aps,
            job.nru_gnbs,
            job.node_rate_bps,
            job.traffic_mode,
            job.sim_time_s,
            int(job.shadowing_enabled),
            0,
            job.alpha,
            job.cold_start_attempts,
            job.decision_interval,
            job.context_dim,
            job.num_arms,
            model.source_model_sha256,
            export_hash,
            model.action_grid_hash,
            "ac88b75eac1818c673cf2c939a96ac3005b1f051",
            "fe0a1d2a5fb7d1547e46042041288a684893ba9e",
            "75a45143b1cd382326876a9597e856338673039a",
            patch_hash,
            scenario_hash,
        )
        if metadata != [expected_metadata]:
            raise ValueError(
                f"ns-3 metadata violates frozen job {job.job_id}"
            )
        nodes = connection.execute(
            "SELECT node_id, technology, state_id "
            "FROM dblbt_nodes ORDER BY node_id"
        ).fetchall()
        node_ids = {row[0] for row in nodes}
        technologies = [row[1] for row in nodes]
        state_ids = [row[2] for row in nodes]
        if (
            len(nodes) != job.wifi_aps + job.nru_gnbs
            or technologies.count("wifi") != job.wifi_aps
            or technologies.count("nru") != job.nru_gnbs
            or any(
                not isinstance(state_id, str) or not state_id
                for state_id in state_ids
            )
            or len(state_ids) != len(set(state_ids))
        ):
            raise ValueError(
                f"ns-3 database violates per-node local states for {job.job_id}"
            )
        decisions = connection.execute(
            "SELECT node_id, decision_round, arm_id, kappa, alpha, "
            "beta, m, b_init, reward, "
            + ", ".join(f"context_{index}" for index in range(_CONTEXT_DIM))
            + " FROM dblbt_decisions ORDER BY node_id, decision_round"
        ).fetchall()
        decision_rounds: dict[int, list[int]] = {}
        decision_valid = True
        for decision in decisions:
            (
                node_id,
                decision_round,
                arm_id,
                kappa,
                alpha,
                beta,
                m,
                b_init,
                reward,
                *context,
            ) = decision
            decision_rounds.setdefault(node_id, []).append(decision_round)
            if (
                node_id not in node_ids
                or not _integer(decision_round)
                or decision_round < job.cold_start_attempts
                or decision_round < job.decision_interval
                or decision_round % job.decision_interval != 0
                or not _integer(arm_id)
                or not 0 <= arm_id < _NUM_ARMS
                or (kappa, alpha, beta, m, b_init) != model.profiles[arm_id]
                or alpha != 11
                or beta >= kappa
                or not _finite_number(reward)
                or len(context) != _CONTEXT_DIM
                or any(
                    not _finite_number(value) or not 0 <= float(value) <= 1
                    for value in context
                )
            ):
                decision_valid = False
                break
        if job.policy == "adaptive":
            attempt_counts = dict(
                connection.execute(
                    "SELECT node_id, COUNT(*) FROM dblbt_attempts "
                    "GROUP BY node_id"
                ).fetchall()
            )
            if (
                not decision_valid
                or any(
                    decision_rounds.get(node_id, [])
                    != list(
                        range(
                            job.decision_interval,
                            attempt_counts.get(node_id, 0) + 1,
                            job.decision_interval,
                        )
                    )
                    for node_id in node_ids
                )
            ):
                raise ValueError(
                    f"ns-3 adaptive decision contract failed for {job.job_id}"
                )
        elif decisions:
            raise ValueError(
                f"non-adaptive ns-3 job contains decisions for {job.job_id}"
            )
        metrics = connection.execute(
            "SELECT technology, throughput_mbps, mean_delay_us, "
            "packet_loss_ratio, simultaneous_access_collision_rate, "
            "channel_occupancy "
            "FROM validation_metrics ORDER BY technology"
        ).fetchall()
        if len(metrics) != 2 or {row[0] for row in metrics} != {"wifi", "nru"}:
            raise ValueError(
                f"ns-3 database lacks paired technology metrics for {job.job_id}"
            )
        for (
            _,
            throughput,
            delay,
            packet_loss,
            access_collisions,
            occupancy,
        ) in metrics:
            if any(
                not _finite_number(value)
                for value in (
                    throughput,
                    delay,
                    packet_loss,
                    access_collisions,
                    occupancy,
                )
            ) or not (
                throughput >= 0
                and delay >= 0
                and 0 <= packet_loss <= 1
                and 0 <= access_collisions <= 1
                and 0 <= occupancy <= 1
            ):
                raise ValueError("ns-3 databases require finite numeric metrics")
        _validate_official_raw_evidence(connection, job)
        return len(decisions) if job.policy == "adaptive" else 0
    except sqlite3.DatabaseError as error:
        raise ValueError(f"invalid ns-3 database for {job.job_id}: {error}") from error
    finally:
        if "connection" in locals():
            connection.close()


def audit_ns3_validation(
    root: str | PathLike[str],
    *,
    model_path: str | PathLike[str] | None = None,
    expected_patch_sha256: str | None = None,
    expected_scenario_sha256: str | None = None,
    expected_build_profile: str | None = None,
) -> Ns3AuditReport:
    """Audit the complete immutable packet-level validation directory."""
    output_root = _path(root).resolve(strict=False)
    database_root = output_root / "databases"
    missing = [
        job.database_name
        for job in validation_jobs()
        if not (database_root / job.database_name).is_file()
    ]
    if missing:
        raise ValueError(
            f"missing {len(missing)} ns-3 validation databases"
        )
    model, export_hash, patch_hash, scenario_hash = _validation_inputs(
        model_path,
        expected_patch_sha256,
        expected_scenario_sha256,
        expected_build_profile,
    )
    manifest_root = output_root / "manifests"
    adaptive_decisions = 0
    database_hashes: list[tuple[str, str]] = []

    for job in validation_jobs():
        database = (database_root / job.database_name).resolve(strict=True)
        manifest_path = manifest_root / f"{job.job_id}.json"
        if not manifest_path.is_file():
            raise ValueError(f"missing ns-3 manifest for {job.job_id}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="ascii"))
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(
                f"invalid ns-3 manifest for {job.job_id}"
            ) from error
        database_hash = _sha256(database)
        expected_manifest = {
            "schema_version": 2,
            "status": "complete",
            "exit_code": 0,
            "job_id": job.job_id,
            "database": f"databases/{job.database_name}",
            "database_bytes": database.stat().st_size,
            "database_sha256": database_hash,
            "model_sha256": model.source_model_sha256,
            "model_export_sha256": export_hash,
            "action_grid_hash": model.action_grid_hash,
            "patch_sha256": patch_hash,
            "scenario_sha256": scenario_hash,
            "node_rate_bps": job.node_rate_bps,
            "traffic_mode": job.traffic_mode,
            "sim_time_s": job.sim_time_s,
            "shadowing_enabled": job.shadowing_enabled,
            "srs_enabled": False,
            "build_profile": expected_build_profile,
        }
        if manifest != expected_manifest:
            raise ValueError(
                f"ns-3 manifest does not match artifacts for {job.job_id}"
            )

        adaptive_decisions += _validate_ns3_database(
            database,
            job,
            model,
            export_hash,
            patch_hash,
            scenario_hash,
        )
        database_hashes.append((job.job_id, database_hash))

    return Ns3AuditReport(
        audited=len(database_hashes),
        model_sha256=model.source_model_sha256,
        model_export_sha256=export_hash,
        adaptive_decisions=adaptive_decisions,
        database_hashes=tuple(database_hashes),
    )


_NS3_METRICS: tuple[Ns3MetricName, ...] = (
    "throughput_mbps",
    "mean_delay_us",
    "packet_loss_ratio",
    "simultaneous_access_collision_rate",
    "channel_occupancy",
)


def reduce_ns3_validation(
    root: str | PathLike[str],
    *,
    model_path: str | PathLike[str] | None = None,
    expected_patch_sha256: str | None = None,
    expected_scenario_sha256: str | None = None,
    expected_build_profile: str | None = None,
) -> Ns3ReductionReport:
    """Reduce TMC/adaptive metrics only after the full audit succeeds."""
    audit = audit_ns3_validation(
        root,
        model_path=model_path,
        expected_patch_sha256=expected_patch_sha256,
        expected_scenario_sha256=expected_scenario_sha256,
        expected_build_profile=expected_build_profile,
    )
    database_root = _path(root).resolve(strict=True) / "databases"
    values: dict[
        tuple[str, int, Ns3Policy, str], tuple[float, ...]
    ] = {}
    for job in validation_jobs():
        if job.policy == "random":
            continue
        database = (database_root / job.database_name).resolve(strict=True)
        uri = f"{database.as_uri()}?mode=ro"
        with sqlite3.connect(uri, uri=True) as connection:
            rows = connection.execute(
                "SELECT technology, throughput_mbps, mean_delay_us, "
                "packet_loss_ratio, simultaneous_access_collision_rate, "
                "channel_occupancy "
                "FROM validation_metrics ORDER BY technology"
            ).fetchall()
        for technology, *raw_metrics in rows:
            values[(job.scenario, job.seed, job.policy, technology)] = tuple(
                float(value) for value in raw_metrics
            )  # type: ignore[assignment]

    per_seed: list[Ns3PairedMetric] = []
    for scenario in (
        "static-4x4",
        "dynamic-4x4",
        "nonideal-6x6-300ms",
    ):
        for seed in (410, 523, 631):
            for technology in ("wifi", "nru"):
                baseline = values[(scenario, seed, "tmc", technology)]
                adaptive = values[(scenario, seed, "adaptive", technology)]
                for index, metric in enumerate(_NS3_METRICS):
                    per_seed.append(
                        Ns3PairedMetric(
                            scenario=scenario,
                            seed=seed,
                            technology=technology,
                            metric=metric,
                            baseline_value=baseline[index],
                            adaptive_value=adaptive[index],
                            paired_difference=adaptive[index] - baseline[index],
                        )
                    )

    scenario_means: list[Ns3ScenarioMetric] = []
    for scenario in (
        "static-4x4",
        "dynamic-4x4",
        "nonideal-6x6-300ms",
    ):
        for technology in ("wifi", "nru"):
            for metric in _NS3_METRICS:
                selected = [
                    row
                    for row in per_seed
                    if row.scenario == scenario
                    and row.technology == technology
                    and row.metric == metric
                ]
                baseline_mean = float(
                    np.mean([row.baseline_value for row in selected])
                )
                adaptive_mean = float(
                    np.mean([row.adaptive_value for row in selected])
                )
                scenario_means.append(
                    Ns3ScenarioMetric(
                        scenario=scenario,
                        technology=technology,
                        metric=metric,
                        seed_count=len(selected),
                        baseline_mean=baseline_mean,
                        adaptive_mean=adaptive_mean,
                        paired_difference=adaptive_mean - baseline_mean,
                    )
                )

    return Ns3ReductionReport(
        audited=audit.audited,
        model_sha256=audit.model_sha256,
        model_export_sha256=audit.model_export_sha256,
        database_hashes=audit.database_hashes,
        per_seed=tuple(per_seed),
        scenario_means=tuple(scenario_means),
    )


def _metric_csv_bytes(
    fields: tuple[str, ...], rows: list[dict[str, object]]
) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output, fieldnames=fields, extrasaction="raise", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("ascii")


def _atomic_write_bytes(target: Path, payload: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as output:
            temporary = Path(output.name)
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def write_ns3_reduction(
    report: Ns3ReductionReport, output_dir: str | PathLike[str]
) -> tuple[Path, Path, Path]:
    """Write deterministic packet-level CSV and provenance JSON artifacts."""
    output = _path(output_dir).resolve(strict=False)
    paired_fields = (
        "scenario",
        "seed",
        "technology",
        "metric",
        "baseline_policy",
        "adaptive_policy",
        "baseline_value",
        "adaptive_value",
        "paired_difference",
    )
    paired_rows = [
        {
            "scenario": row.scenario,
            "seed": row.seed,
            "technology": row.technology,
            "metric": row.metric,
            "baseline_policy": "tmc",
            "adaptive_policy": "adaptive",
            "baseline_value": _decimal(row.baseline_value),
            "adaptive_value": _decimal(row.adaptive_value),
            "paired_difference": _decimal(row.paired_difference),
        }
        for row in report.per_seed
    ]
    scenario_fields = (
        "scenario",
        "technology",
        "metric",
        "seed_count",
        "baseline_policy",
        "adaptive_policy",
        "baseline_mean",
        "adaptive_mean",
        "paired_difference",
    )
    scenario_rows = [
        {
            "scenario": row.scenario,
            "technology": row.technology,
            "metric": row.metric,
            "seed_count": row.seed_count,
            "baseline_policy": "tmc",
            "adaptive_policy": "adaptive",
            "baseline_mean": _decimal(row.baseline_mean),
            "adaptive_mean": _decimal(row.adaptive_mean),
            "paired_difference": _decimal(row.paired_difference),
        }
        for row in report.scenario_means
    ]
    paired_payload = _metric_csv_bytes(paired_fields, paired_rows)
    scenario_payload = _metric_csv_bytes(scenario_fields, scenario_rows)
    metadata_payload = (
        json.dumps(
            {
                "schema_version": 2,
                "audited": report.audited,
                "model_sha256": report.model_sha256,
                "model_export_sha256": report.model_export_sha256,
                "database_hashes": dict(report.database_hashes),
                "paired_rows": len(report.per_seed),
                "scenario_rows": len(report.scenario_means),
                "paired_metrics_sha256": hashlib.sha256(
                    paired_payload
                ).hexdigest(),
                "scenario_metrics_sha256": hashlib.sha256(
                    scenario_payload
                ).hexdigest(),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")
    paths = (
        output / "paired-metrics.csv",
        output / "scenario-metrics.csv",
        output / "reduction.json",
    )
    for path, payload in zip(
        paths, (paired_payload, scenario_payload, metadata_payload), strict=True
    ):
        _atomic_write_bytes(path, payload)
    return paths


def _path(value: str | PathLike[str]) -> Path:
    try:
        return Path(value)
    except TypeError as error:
        raise ValueError("path must be a filesystem path") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decimal(value: float) -> str:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("model values must be finite")
    return format(numeric, ".17g")


def export_ns3_model(
    source: str | PathLike[str], destination: str | PathLike[str]
) -> Ns3ModelExport:
    """Export a validated canonical LinUCB NPZ to deterministic ASCII."""
    source_path = _path(source).resolve(strict=True)
    if not source_path.is_file():
        raise ValueError("source model must be a regular file")
    target = _path(destination).resolve(strict=False)
    grid_hash = action_grid_hash()
    model = LinUCB.load(
        source_path, expected_action_grid_hash=grid_hash
    )
    if model.num_arms != _NUM_ARMS or model.context_dim != _CONTEXT_DIM:
        raise ValueError("ns-3 model must contain 24 arms and 11 features")

    profiles = tuple(
        (arm.kappa, arm.alpha, arm.beta, arm.m, arm.b_init)
        for arm in adaptive_arms()
    )
    lines = [
        _MODEL_MAGIC,
        f"schema_version {_MODEL_SCHEMA_VERSION}",
        f"num_arms {_NUM_ARMS}",
        f"context_dim {_CONTEXT_DIM}",
        f"ridge {_decimal(model.ridge)}",
        f"exploration {_decimal(model.exploration)}",
        f"action_grid_hash {grid_hash}",
        f"source_model_sha256 {_sha256(source_path)}",
        f"profiles {len(profiles)}",
    ]
    lines.extend(
        f"profile {index} {kappa} {alpha} {beta} {m} {b_init}"
        for index, (kappa, alpha, beta, m, b_init) in enumerate(profiles)
    )
    lines.append(f"A_rows {_NUM_ARMS * _CONTEXT_DIM}")
    for arm in range(_NUM_ARMS):
        for row in range(_CONTEXT_DIM):
            values = " ".join(
                _decimal(value) for value in model.A[arm, row]
            )
            lines.append(f"A {arm} {row} {values}")
    lines.append(f"b_rows {_NUM_ARMS}")
    for arm in range(_NUM_ARMS):
        values = " ".join(_decimal(value) for value in model.b[arm])
        lines.append(f"b {arm} {values}")
    lines.append("end")
    payload = ("\n".join(lines) + "\n").encode("ascii")

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as output:
            temporary = Path(output.name)
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        read_ns3_model(temporary)
        os.replace(temporary, target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return read_ns3_model(target)


def _field(line: str, expected: str) -> str:
    parts = line.split(" ")
    if len(parts) != 2 or parts[0] != expected or not parts[1]:
        raise ValueError(f"invalid ns-3 model field: {expected}")
    return parts[1]


def _positive_int(name: str, value: str) -> int:
    try:
        parsed = int(value, 10)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _finite(name: str, value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"{name} must be numeric") from error
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _hash(name: str, value: str) -> str:
    normalized = value.lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{name} must be a SHA-256 hex string")
    return normalized


def read_ns3_model(path: str | PathLike[str]) -> Ns3ModelExport:
    """Read the exact deterministic ASCII model interchange schema."""
    source = _path(path).resolve(strict=True)
    if not source.is_file():
        raise ValueError("ns-3 model must be a regular file")
    try:
        text = source.read_bytes().decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError("ns-3 model must be ASCII") from error
    if not text.endswith("\n"):
        raise ValueError("ns-3 model must end with a newline")
    lines = text.splitlines()
    minimum_lines = 12 + _NUM_ARMS + _NUM_ARMS * _CONTEXT_DIM + _NUM_ARMS
    if len(lines) != minimum_lines:
        raise ValueError("ns-3 model has an invalid line count")
    cursor = 0
    if lines[cursor] != _MODEL_MAGIC:
        raise ValueError("unsupported ns-3 model magic")
    cursor += 1
    schema_version = _positive_int(
        "schema_version", _field(lines[cursor], "schema_version")
    )
    cursor += 1
    num_arms = _positive_int(
        "num_arms", _field(lines[cursor], "num_arms")
    )
    cursor += 1
    context_dim = _positive_int(
        "context_dim", _field(lines[cursor], "context_dim")
    )
    cursor += 1
    ridge = _finite("ridge", _field(lines[cursor], "ridge"))
    cursor += 1
    exploration = _finite(
        "exploration", _field(lines[cursor], "exploration")
    )
    cursor += 1
    grid_hash = _hash(
        "action_grid_hash", _field(lines[cursor], "action_grid_hash")
    )
    cursor += 1
    model_hash = _hash(
        "source_model_sha256",
        _field(lines[cursor], "source_model_sha256"),
    )
    cursor += 1
    profile_count = _positive_int(
        "profiles", _field(lines[cursor], "profiles")
    )
    cursor += 1
    if (
        schema_version != _MODEL_SCHEMA_VERSION
        or num_arms != _NUM_ARMS
        or context_dim != _CONTEXT_DIM
        or profile_count != _NUM_ARMS
        or ridge <= 0
        or exploration < 0
        or grid_hash != action_grid_hash()
    ):
        raise ValueError("ns-3 model metadata violates the frozen contract")

    profiles: list[tuple[int, int, int, int, int]] = []
    for expected_index in range(_NUM_ARMS):
        parts = lines[cursor].split(" ")
        cursor += 1
        if len(parts) != 7 or parts[:2] != ["profile", str(expected_index)]:
            raise ValueError("ns-3 model profile order is invalid")
        try:
            profile = tuple(int(value, 10) for value in parts[2:])
        except ValueError as error:
            raise ValueError("ns-3 model profile is invalid") from error
        profiles.append(profile)
    expected_profiles = tuple(
        (arm.kappa, arm.alpha, arm.beta, arm.m, arm.b_init)
        for arm in adaptive_arms()
    )
    if tuple(profiles) != expected_profiles:
        raise ValueError("ns-3 model action grid is invalid")

    A_rows = _positive_int("A_rows", _field(lines[cursor], "A_rows"))
    cursor += 1
    if A_rows != _NUM_ARMS * _CONTEXT_DIM:
        raise ValueError("ns-3 model A row count is invalid")
    A = np.empty((_NUM_ARMS, _CONTEXT_DIM, _CONTEXT_DIM), dtype=np.float64)
    for arm in range(_NUM_ARMS):
        for row in range(_CONTEXT_DIM):
            parts = lines[cursor].split(" ")
            cursor += 1
            if len(parts) != 3 + _CONTEXT_DIM or parts[:3] != [
                "A",
                str(arm),
                str(row),
            ]:
                raise ValueError("ns-3 model A row order is invalid")
            A[arm, row] = [
                _finite("A value", value) for value in parts[3:]
            ]

    b_rows = _positive_int("b_rows", _field(lines[cursor], "b_rows"))
    cursor += 1
    if b_rows != _NUM_ARMS:
        raise ValueError("ns-3 model b row count is invalid")
    b = np.empty((_NUM_ARMS, _CONTEXT_DIM), dtype=np.float64)
    for arm in range(_NUM_ARMS):
        parts = lines[cursor].split(" ")
        cursor += 1
        if len(parts) != 2 + _CONTEXT_DIM or parts[:2] != ["b", str(arm)]:
            raise ValueError("ns-3 model b row order is invalid")
        b[arm] = [_finite("b value", value) for value in parts[2:]]
    if lines[cursor] != "end":
        raise ValueError("ns-3 model terminator is invalid")

    for arm in range(_NUM_ARMS):
        if not np.array_equal(A[arm], A[arm].T):
            raise ValueError(f"ns-3 model A for arm {arm} is not symmetric")
        try:
            np.linalg.cholesky(A[arm])
        except np.linalg.LinAlgError as error:
            raise ValueError(
                f"ns-3 model A for arm {arm} is not positive definite"
            ) from error
    return Ns3ModelExport(
        schema_version=schema_version,
        num_arms=num_arms,
        context_dim=context_dim,
        ridge=ridge,
        exploration=exploration,
        action_grid_hash=grid_hash,
        source_model_sha256=model_hash,
        profiles=tuple(profiles),
        A=A,
        b=b,
    )
