from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np
import pytest

from dblbt_fcn.config import adaptive_arms
from dblbt_fcn.linucb import LinUCB
from dblbt_fcn.ns3_validation import (
    audit_ns3_validation,
    export_ns3_model,
    read_ns3_model,
    reduce_ns3_validation,
    validate_ns3_job_database,
    validation_jobs,
    write_ns3_reduction,
)
from dblbt_fcn.workflows import action_grid_hash


PATCH_HASH = "1" * 64
SCENARIO_HASH = "2" * 64
SCENARIO_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "ns3"
    / "scenarios"
    / "dblbt-nru-wifi-validation.cc"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _export_model(tmp_path: Path) -> tuple[Path, str, str]:
    grid_hash = action_grid_hash()
    source = tmp_path / "formal-model.npz"
    exported = tmp_path / "formal-model.txt"
    LinUCB(24, 11, action_grid_hash=grid_hash).save(source)
    export_ns3_model(source, exported)
    return exported, _sha256(source), _sha256(exported)


def _audit(root: Path, model_path: Path) -> object:
    return audit_ns3_validation(
        root,
        model_path=model_path,
        expected_patch_sha256=PATCH_HASH,
        expected_scenario_sha256=SCENARIO_HASH,
        expected_build_profile="optimized",
    )


def _reduce(root: Path, model_path: Path) -> object:
    return reduce_ns3_validation(
        root,
        model_path=model_path,
        expected_patch_sha256=PATCH_HASH,
        expected_scenario_sha256=SCENARIO_HASH,
        expected_build_profile="optimized",
    )


def test_scenario_counts_access_overlap_once_and_keeps_packet_loss_separate() -> None:
    source = SCENARIO_SOURCE.read_text(encoding="ascii")

    assert "double packetLossRatio {0.0};" in source
    assert "double simultaneousAccessCollisionRate {0.0};" in source
    assert "const bool overlap = own.Active () || other.Active ();" in source
    assert "tracker.transmissions += 1;" in source
    assert "tracker.overlaps += overlap ? 1 : 0;" in source
    assert "packet_loss_ratio REAL NOT NULL" in source
    assert "simultaneous_access_collision_rate REAL NOT NULL" in source
    assert "sim_time_s REAL NOT NULL" in source
    assert "shadowing_enabled INTEGER NOT NULL" in source


def _write_valid_database(
    root: Path,
    job: object,
    source_model_hash: str,
    export_hash: str,
) -> Path:
    databases = root / "databases"
    manifests = root / "manifests"
    databases.mkdir(parents=True, exist_ok=True)
    manifests.mkdir(parents=True, exist_ok=True)
    database = databases / job.database_name
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE validation_metadata (
          schema_version INTEGER NOT NULL,
          job_id TEXT NOT NULL,
          policy TEXT NOT NULL,
          scenario TEXT NOT NULL,
          seed INTEGER NOT NULL,
          run_id INTEGER NOT NULL,
          wifi_aps INTEGER NOT NULL,
          nru_gnbs INTEGER NOT NULL,
          node_rate_bps INTEGER NOT NULL,
          traffic_mode TEXT NOT NULL,
          sim_time_s REAL NOT NULL,
          shadowing_enabled INTEGER NOT NULL,
          srs_enabled INTEGER NOT NULL,
          alpha INTEGER NOT NULL,
          cold_start_attempts INTEGER NOT NULL,
          decision_interval INTEGER NOT NULL,
          context_dim INTEGER NOT NULL,
          num_arms INTEGER NOT NULL,
          model_sha256 TEXT NOT NULL,
          model_export_sha256 TEXT NOT NULL,
          action_grid_hash TEXT NOT NULL,
          ns3_commit TEXT NOT NULL,
          nr_commit TEXT NOT NULL,
          nru_commit TEXT NOT NULL,
          patch_sha256 TEXT NOT NULL,
          scenario_sha256 TEXT NOT NULL
        );
        CREATE TABLE dblbt_nodes (
          node_id INTEGER PRIMARY KEY,
          technology TEXT NOT NULL,
          state_id TEXT NOT NULL UNIQUE
        );
        CREATE TABLE dblbt_attempts (
          node_id INTEGER NOT NULL,
          attempt_id INTEGER NOT NULL,
          outcome TEXT NOT NULL,
          elapsed_us REAL NOT NULL,
          busy_us REAL NOT NULL,
          interruptions INTEGER NOT NULL,
          access_delay_us REAL NOT NULL,
          queue_occupancy REAL NOT NULL,
          arrivals INTEGER NOT NULL,
          retries INTEGER NOT NULL,
          effective_data_us REAL NOT NULL,
          selected_backoff INTEGER NOT NULL,
          PRIMARY KEY (node_id, attempt_id)
        );
        CREATE TABLE dblbt_decisions (
          node_id INTEGER NOT NULL,
          decision_round INTEGER NOT NULL,
          arm_id INTEGER NOT NULL,
          kappa INTEGER NOT NULL,
          alpha INTEGER NOT NULL,
          beta INTEGER NOT NULL,
          m INTEGER NOT NULL,
          b_init INTEGER NOT NULL,
          reward REAL NOT NULL,
          context_0 REAL NOT NULL,
          context_1 REAL NOT NULL,
          context_2 REAL NOT NULL,
          context_3 REAL NOT NULL,
          context_4 REAL NOT NULL,
          context_5 REAL NOT NULL,
          context_6 REAL NOT NULL,
          context_7 REAL NOT NULL,
          context_8 REAL NOT NULL,
          context_9 REAL NOT NULL,
          context_10 REAL NOT NULL,
          PRIMARY KEY (node_id, decision_round)
        );
        CREATE TABLE validation_metrics (
          technology TEXT PRIMARY KEY,
          throughput_mbps REAL NOT NULL,
          mean_delay_us REAL NOT NULL,
          packet_loss_ratio REAL NOT NULL,
          simultaneous_access_collision_rate REAL NOT NULL,
          channel_occupancy REAL NOT NULL
        );
        CREATE TABLE channel_occupancy_5 (
          TECHNOLOGY STRING NOT NULL, VALUE DOUBLE NOT NULL,
          SEED INT NOT NULL, RUN INT NOT NULL
        );
        CREATE TABLE simultaneous_tx_5 (
          UID INT NOT NULL, SIMULTANEOUS_TX_SAME_TECH INT NOT NULL,
          SIMULTANEOUS_TX_OTHER_TECH INT NOT NULL, TOTALTX INT NOT NULL,
          SEED INT NOT NULL, RUN INT NOT NULL
        );
        CREATE TABLE mac_data_tx_failed_5 (
          UID INT NOT NULL, NUMBER INT NOT NULL, BYTES INT NOT NULL,
          SEED INT NOT NULL, RUN INT NOT NULL
        );
        CREATE TABLE sinr_results_5 (
          UID INT NOT NULL, SINR DOUBLE NOT NULL,
          SEED INT NOT NULL, RUN INT NOT NULL
        );
        CREATE TABLE e2e_5 (
          TECHNOLOGY STRING NOT NULL, THROUGHPUT_MBPS DOUBLE NOT NULL,
          TXBYTES INT NOT NULL, RXBYTES INT NOT NULL,
          LATENCY_US DOUBLE NOT NULL, JITTER_US DOUBLE NOT NULL,
          ADDR STRING NOT NULL, SEED INT NOT NULL, RUN INT NOT NULL
        );
        """
    )
    connection.execute(
        "INSERT INTO validation_metadata VALUES "
        "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
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
            source_model_hash,
            export_hash,
            action_grid_hash(),
            "ac88b75eac1818c673cf2c939a96ac3005b1f051",
            "fe0a1d2a5fb7d1547e46042041288a684893ba9e",
            "75a45143b1cd382326876a9597e856338673039a",
            PATCH_HASH,
            SCENARIO_HASH,
        ),
    )
    controller_ids = [
        *range(job.wifi_aps),
        *range(2 * job.wifi_aps, 2 * job.wifi_aps + job.nru_gnbs),
    ]
    for index, node_id in enumerate(controller_ids):
        technology = "wifi" if index < job.wifi_aps else "nru"
        connection.execute(
            "INSERT INTO dblbt_nodes VALUES (?,?,?)",
            (node_id, technology, f"{job.policy}-{job.seed}-{node_id}"),
        )
        if job.policy != "random":
            for attempt_id in range(1, 33):
                collision = attempt_id % 5 == 0
                connection.execute(
                    "INSERT INTO dblbt_attempts VALUES "
                    "(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        node_id,
                        attempt_id,
                        "collision" if collision else "success",
                        1000.0,
                        200.0,
                        1,
                        300.0,
                        0.5,
                        1,
                        1 if collision else 0,
                        700.0 if not collision else 0.0,
                        11,
                    ),
                )
        if job.policy == "adaptive":
            connection.execute(
                "INSERT INTO dblbt_decisions VALUES "
                "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (node_id, 32, 0, 5, 11, 2, 4, 15, 0.5, *([0.1] * 11)),
            )
    for uid in range(2 * (job.wifi_aps + job.nru_gnbs)):
        connection.execute(
            "INSERT INTO simultaneous_tx_5 VALUES (?,?,?,?,?,?)",
            (uid, 1, 1, 10, job.seed, job.run_id),
        )
        connection.execute(
            "INSERT INTO mac_data_tx_failed_5 VALUES (?,?,?,?,?)",
            (uid, 1, 1000, job.seed, job.run_id),
        )
        connection.execute(
            "INSERT INTO sinr_results_5 VALUES (?,?,?,?)",
            (uid, 12.0, job.seed, job.run_id),
        )
    for technology in ("wifi", "nru"):
        connection.execute(
            "INSERT INTO validation_metrics VALUES (?,?,?,?,?,?)",
            (technology, 10.0, 500.0, 0.1, 0.04, 0.4),
        )
        connection.execute(
            "INSERT INTO channel_occupancy_5 VALUES (?,?,?,?)",
            (technology, 0.4, job.seed, job.run_id),
        )
        connection.execute(
            "INSERT INTO e2e_5 VALUES (?,?,?,?,?,?,?,?,?)",
            (
                technology,
                10.0,
                10000,
                9000,
                500.0,
                50.0,
                f"10.0.0.{1 if technology == 'wifi' else 2}",
                job.seed,
                job.run_id,
            ),
        )
    connection.commit()
    connection.close()
    manifest = {
        "schema_version": 2,
        "status": "complete",
        "exit_code": 0,
        "job_id": job.job_id,
        "database": f"databases/{job.database_name}",
        "database_bytes": database.stat().st_size,
        "database_sha256": _sha256(database),
        "model_sha256": source_model_hash,
        "model_export_sha256": export_hash,
        "action_grid_hash": action_grid_hash(),
        "patch_sha256": PATCH_HASH,
        "scenario_sha256": SCENARIO_HASH,
        "node_rate_bps": job.node_rate_bps,
        "traffic_mode": job.traffic_mode,
        "sim_time_s": job.sim_time_s,
        "shadowing_enabled": job.shadowing_enabled,
        "srs_enabled": False,
        "build_profile": "optimized",
    }
    manifest_path = manifests / f"{job.job_id}.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    return database


def _refresh_manifest(root: Path, job: object) -> None:
    database = root / "databases" / job.database_name
    manifest_path = root / "manifests" / f"{job.job_id}.json"
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    manifest["database_bytes"] = database.stat().st_size
    manifest["database_sha256"] = _sha256(database)
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )


def _validate_database(
    database: Path, job: object, model_path: Path
) -> int:
    return validate_ns3_job_database(
        database,
        job_id=job.job_id,
        model_path=model_path,
        expected_patch_sha256=PATCH_HASH,
        expected_scenario_sha256=SCENARIO_HASH,
        expected_build_profile="optimized",
    )


def _replace_adaptive_schedule(
    database: Path, attempt_counts: dict[int, int]
) -> None:
    arm = adaptive_arms()[0]
    with sqlite3.connect(database) as connection:
        connection.execute("DELETE FROM dblbt_attempts")
        connection.execute("DELETE FROM dblbt_decisions")
        for node_id, count in attempt_counts.items():
            for attempt_id in range(1, count + 1):
                collision = attempt_id % 5 == 0
                connection.execute(
                    "INSERT INTO dblbt_attempts VALUES "
                    "(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        node_id,
                        attempt_id,
                        "collision" if collision else "success",
                        1000.0,
                        200.0,
                        1,
                        300.0,
                        0.5,
                        1,
                        1 if collision else 0,
                        700.0 if not collision else 0.0,
                        11,
                    ),
                )
            for decision_round in range(32, count + 1, 32):
                connection.execute(
                    "INSERT INTO dblbt_decisions VALUES "
                    "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        node_id,
                        decision_round,
                        0,
                        arm.kappa,
                        arm.alpha,
                        arm.beta,
                        arm.m,
                        arm.b_init,
                        0.5,
                        *([0.1] * 11),
                    ),
                )


def test_model_export_is_deterministic_and_complete(tmp_path: Path) -> None:
    grid_hash = action_grid_hash()
    agent = LinUCB(24, 11, action_grid_hash=grid_hash)
    for arm in range(24):
        context = np.linspace(0.01, 0.99, 11, dtype=np.float64)
        agent.update(arm, context, (arm - 12) / 24)

    source = tmp_path / "model.npz"
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    agent.save(source)

    export_ns3_model(source, first)
    export_ns3_model(source, second)

    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes().isascii()
    exported = read_ns3_model(first)
    assert exported.schema_version == 1
    assert exported.num_arms == 24
    assert exported.context_dim == 11
    assert exported.action_grid_hash == grid_hash
    assert exported.source_model_sha256 == _sha256(source)
    assert exported.profiles == tuple(
        (arm.kappa, arm.alpha, arm.beta, arm.m, arm.b_init)
        for arm in adaptive_arms()
    )
    np.testing.assert_array_equal(exported.A, agent.A)
    np.testing.assert_array_equal(exported.b, agent.b)


def test_validation_jobs_are_the_exact_frozen_27() -> None:
    jobs = validation_jobs()

    assert len(jobs) == 27
    assert len({job.job_id for job in jobs}) == 27
    assert {
        (job.policy, job.scenario, job.seed) for job in jobs
    } == {
        (policy, scenario, seed)
        for policy in ("random", "tmc", "adaptive")
        for scenario in (
            "static-4x4",
            "dynamic-4x4",
            "nonideal-6x6-300ms",
        )
        for seed in (410, 523, 631)
    }
    for job in jobs:
        assert job.run_id == 1
        assert job.sim_time_s == 2.0
        assert job.app_start_s == 0.2
        assert job.node_rate_bps == 2_000_000
        assert job.traffic_mode == "aggregate-saturated-cbr"
        assert job.shadowing_enabled is True
        assert job.alpha == 11
        assert job.cold_start_attempts == 8
        assert job.decision_interval == 32
        assert job.context_dim == 11
        assert job.num_arms == 24
        if job.scenario == "nonideal-6x6-300ms":
            assert (job.wifi_aps, job.nru_gnbs) == (6, 6)
            assert job.interference_interval_ms == 300
            assert job.interference_duration_ms == 2
        else:
            assert (job.wifi_aps, job.nru_gnbs) == (4, 4)
            assert job.interference_interval_ms is None
            assert job.interference_duration_ms is None
        assert job.dynamic_load_change is (
            job.scenario == "dynamic-4x4"
        )


def test_audit_rejects_all_missing_formal_databases(tmp_path: Path) -> None:
    with pytest.raises(
        ValueError, match="missing 27 ns-3 validation databases"
    ):
        audit_ns3_validation(tmp_path)


def test_audit_accepts_exact_27_and_is_read_only(tmp_path: Path) -> None:
    model_path, source_hash, export_hash = _export_model(tmp_path)
    root = tmp_path / "formal"
    databases = [
        _write_valid_database(root, job, source_hash, export_hash)
        for job in validation_jobs()
    ]
    before = {
        path: (_sha256(path), path.stat().st_mtime_ns) for path in databases
    }

    report = _audit(root, model_path)

    assert report.audited == 27
    assert report.model_sha256 == source_hash
    assert report.model_export_sha256 == export_hash
    assert report.adaptive_decisions == sum(
        job.wifi_aps + job.nru_gnbs
        for job in validation_jobs()
        if job.policy == "adaptive"
    )
    assert {
        path: (_sha256(path), path.stat().st_mtime_ns) for path in databases
    } == before


def test_reduction_pairs_all_tmc_and_adaptive_ns3_metrics(
    tmp_path: Path,
) -> None:
    model_path, source_hash, export_hash = _export_model(tmp_path)
    root = tmp_path / "formal"
    databases = [
        _write_valid_database(root, job, source_hash, export_hash)
        for job in validation_jobs()
    ]
    for job in validation_jobs():
        if job.policy == "random":
            continue
        database = root / "databases" / job.database_name
        offset = 1.0 if job.policy == "adaptive" else 0.0
        with sqlite3.connect(database) as connection:
            connection.execute(
                "UPDATE validation_metrics SET "
                "throughput_mbps = throughput_mbps + ?, "
                "mean_delay_us = mean_delay_us - ?, "
                "packet_loss_ratio = packet_loss_ratio - ?, "
                "simultaneous_access_collision_rate = "
                "simultaneous_access_collision_rate - ?, "
                "channel_occupancy = channel_occupancy + ?",
                (
                    offset,
                    50.0 * offset,
                    0.01 * offset,
                    0.02 * offset,
                    0.05 * offset,
                ),
            )
        _refresh_manifest(root, job)
    before = {
        path: (_sha256(path), path.stat().st_mtime_ns) for path in databases
    }

    report = _reduce(root, model_path)

    assert report.audited == 27
    assert len(report.per_seed) == 90
    assert len(report.scenario_means) == 30
    row = next(
        row
        for row in report.per_seed
        if (
            row.scenario,
            row.seed,
            row.technology,
            row.metric,
        )
        == ("static-4x4", 410, "wifi", "throughput_mbps")
    )
    assert (row.baseline_value, row.adaptive_value, row.paired_difference) == (
        10.0,
        11.0,
        1.0,
    )
    packet_loss = next(
        row
        for row in report.per_seed
        if (
            row.scenario,
            row.seed,
            row.technology,
            row.metric,
        ) == ("static-4x4", 410, "wifi", "packet_loss_ratio")
    )
    access_collision = next(
        row
        for row in report.per_seed
        if (
            row.scenario,
            row.seed,
            row.technology,
            row.metric,
        )
        == (
            "static-4x4",
            410,
            "wifi",
            "simultaneous_access_collision_rate",
        )
    )
    assert packet_loss.paired_difference == pytest.approx(-0.01)
    assert access_collision.paired_difference == pytest.approx(-0.02)
    assert {
        path: (_sha256(path), path.stat().st_mtime_ns) for path in databases
    } == before


def test_reduction_outputs_are_deterministic_and_separate_from_raw_data(
    tmp_path: Path,
) -> None:
    model_path, source_hash, export_hash = _export_model(tmp_path)
    root = tmp_path / "formal"
    for job in validation_jobs():
        _write_valid_database(root, job, source_hash, export_hash)
    report = _reduce(root, model_path)
    output = tmp_path / "derived"

    first = write_ns3_reduction(report, output)
    first_bytes = {path.name: path.read_bytes() for path in first}
    second = write_ns3_reduction(report, output)

    assert [path.name for path in first] == [
        "paired-metrics.csv",
        "scenario-metrics.csv",
        "reduction.json",
    ]
    assert {path.name: path.read_bytes() for path in second} == first_bytes
    assert not any(path.suffix == ".db" for path in output.iterdir())


def test_audit_rejects_nonfinite_metric_with_valid_hash(tmp_path: Path) -> None:
    model_path, source_hash, export_hash = _export_model(tmp_path)
    root = tmp_path / "formal"
    jobs = validation_jobs()
    for job in jobs:
        _write_valid_database(root, job, source_hash, export_hash)
    corrupted = jobs[0]
    database = root / "databases" / corrupted.database_name
    connection = sqlite3.connect(database)
    connection.execute(
        "UPDATE validation_metrics SET throughput_mbps = ? "
        "WHERE technology = 'wifi'",
        (float("inf"),),
    )
    connection.commit()
    connection.close()
    _refresh_manifest(root, corrupted)

    with pytest.raises(ValueError, match="finite numeric metrics"):
        _audit(root, model_path)


def test_audit_rejects_global_context_column(tmp_path: Path) -> None:
    model_path, source_hash, export_hash = _export_model(tmp_path)
    root = tmp_path / "formal"
    jobs = validation_jobs()
    for job in jobs:
        _write_valid_database(root, job, source_hash, export_hash)
    corrupted = next(job for job in jobs if job.policy == "adaptive")
    database = root / "databases" / corrupted.database_name
    connection = sqlite3.connect(database)
    connection.execute(
        "ALTER TABLE dblbt_decisions ADD COLUMN global_jain REAL"
    )
    connection.commit()
    connection.close()
    _refresh_manifest(root, corrupted)

    with pytest.raises(ValueError, match="dblbt_decisions schema"):
        _audit(root, model_path)


def test_audit_rejects_missing_per_node_state(tmp_path: Path) -> None:
    model_path, source_hash, export_hash = _export_model(tmp_path)
    root = tmp_path / "formal"
    jobs = validation_jobs()
    for job in jobs:
        _write_valid_database(root, job, source_hash, export_hash)
    corrupted = jobs[0]
    database = root / "databases" / corrupted.database_name
    connection = sqlite3.connect(database)
    connection.execute("DELETE FROM dblbt_nodes WHERE node_id = 0")
    connection.commit()
    connection.close()
    _refresh_manifest(root, corrupted)

    with pytest.raises(ValueError, match="per-node local states"):
        _audit(root, model_path)


def test_audit_rejects_invalid_adaptive_decision(tmp_path: Path) -> None:
    model_path, source_hash, export_hash = _export_model(tmp_path)
    root = tmp_path / "formal"
    jobs = validation_jobs()
    for job in jobs:
        _write_valid_database(root, job, source_hash, export_hash)
    corrupted = next(job for job in jobs if job.policy == "adaptive")
    database = root / "databases" / corrupted.database_name
    connection = sqlite3.connect(database)
    connection.execute("UPDATE dblbt_decisions SET alpha = 12 LIMIT 1")
    connection.commit()
    connection.close()
    _refresh_manifest(root, corrupted)

    with pytest.raises(ValueError, match="adaptive decision contract"):
        _audit(root, model_path)


def test_audit_rejects_wrong_frozen_offered_load(tmp_path: Path) -> None:
    model_path, source_hash, export_hash = _export_model(tmp_path)
    root = tmp_path / "formal"
    jobs = validation_jobs()
    for job in jobs:
        _write_valid_database(root, job, source_hash, export_hash)
    corrupted = jobs[0]
    database = root / "databases" / corrupted.database_name
    connection = sqlite3.connect(database)
    connection.execute(
        "UPDATE validation_metadata SET node_rate_bps = 1000000"
    )
    connection.commit()
    connection.close()
    _refresh_manifest(root, corrupted)

    with pytest.raises(ValueError, match="frozen job"):
        _audit(root, model_path)


def test_audit_rejects_malformed_official_raw_schema(tmp_path: Path) -> None:
    model_path, source_hash, export_hash = _export_model(tmp_path)
    root = tmp_path / "formal"
    jobs = validation_jobs()
    for job in jobs:
        _write_valid_database(root, job, source_hash, export_hash)
    corrupted = jobs[0]
    database = root / "databases" / corrupted.database_name
    with sqlite3.connect(database) as connection:
        connection.execute(
            "ALTER TABLE channel_occupancy_5 ADD COLUMN EXTRA DOUBLE"
        )
    _refresh_manifest(root, corrupted)

    with pytest.raises(ValueError, match="official raw evidence schema"):
        _audit(root, model_path)


def test_audit_rejects_nonfinite_official_raw_value(tmp_path: Path) -> None:
    model_path, source_hash, export_hash = _export_model(tmp_path)
    root = tmp_path / "formal"
    jobs = validation_jobs()
    for job in jobs:
        _write_valid_database(root, job, source_hash, export_hash)
    corrupted = jobs[0]
    database = root / "databases" / corrupted.database_name
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE sinr_results_5 SET SINR = ? WHERE UID = 0",
            (float("inf"),),
        )
    _refresh_manifest(root, corrupted)

    with pytest.raises(ValueError, match="official raw evidence values"):
        _audit(root, model_path)


def test_validator_accepts_sparse_official_transmitter_uids(
    tmp_path: Path,
) -> None:
    model_path, source_hash, export_hash = _export_model(tmp_path)
    job = next(job for job in validation_jobs() if job.policy == "random")
    database = _write_valid_database(
        tmp_path / "formal", job, source_hash, export_hash
    )
    with sqlite3.connect(database) as connection:
        connection.execute(
            "DELETE FROM simultaneous_tx_5 WHERE UID IN (4, 6, 7)"
        )
        connection.execute(
            "DELETE FROM sinr_results_5 WHERE UID BETWEEN 8 AND 11"
        )
        connection.execute("DELETE FROM mac_data_tx_failed_5")

    assert _validate_database(database, job, model_path) == 0


def test_validator_accepts_adaptive_schedule_from_per_node_attempt_counts(
    tmp_path: Path,
) -> None:
    model_path, source_hash, export_hash = _export_model(tmp_path)
    job = next(job for job in validation_jobs() if job.policy == "adaptive")
    database = _write_valid_database(
        tmp_path / "formal", job, source_hash, export_hash
    )
    attempt_counts = {
        0: 65,
        1: 0,
        2: 63,
        3: 32,
        8: 31,
        9: 8,
        10: 7,
        11: 0,
    }
    _replace_adaptive_schedule(database, attempt_counts)

    assert _validate_database(database, job, model_path) == 4


@pytest.mark.parametrize("schedule_error", ["missing", "beyond_attempts"])
def test_validator_rejects_nonexact_adaptive_decision_schedule(
    tmp_path: Path, schedule_error: str
) -> None:
    model_path, source_hash, export_hash = _export_model(tmp_path)
    job = next(job for job in validation_jobs() if job.policy == "adaptive")
    database = _write_valid_database(
        tmp_path / "formal", job, source_hash, export_hash
    )
    attempt_counts = {
        0: 64 if schedule_error == "missing" else 32,
        1: 32,
        2: 32,
        3: 32,
        8: 32,
        9: 32,
        10: 32,
        11: 32,
    }
    _replace_adaptive_schedule(database, attempt_counts)
    with sqlite3.connect(database) as connection:
        if schedule_error == "missing":
            connection.execute(
                "DELETE FROM dblbt_decisions "
                "WHERE node_id = 0 AND decision_round = 64"
            )
        else:
            arm = adaptive_arms()[0]
            connection.execute(
                "INSERT INTO dblbt_decisions VALUES "
                "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    0,
                    64,
                    0,
                    arm.kappa,
                    arm.alpha,
                    arm.beta,
                    arm.m,
                    arm.b_init,
                    0.5,
                    *([0.1] * 11),
                ),
            )

    with pytest.raises(ValueError, match="adaptive decision contract"):
        _validate_database(database, job, model_path)


def test_validation_scenario_declares_frozen_local_interface() -> None:
    scenario = (
        Path(__file__).parents[1]
        / "ns3"
        / "scenarios"
        / "dblbt-nru-wifi-validation.cc"
    )
    text = scenario.read_text(encoding="ascii")

    for token in (
        "WIFI_STANDARD_80211ax_5GHZ",
        "5.2e9",
        "20e6",
        "NrDbLbtAccessManager",
        "WaveformGenerator",
        "interferenceIntervalMs",
        'AddValue ("policy"',
        'AddValue ("scenario"',
        'AddValue ("wifiAps"',
        'AddValue ("nruGnbs"',
        'AddValue ("modelPath"',
        'AddValue ("outputDb"',
        "validation_metadata",
        "dblbt_nodes",
        "dblbt_attempts",
        "dblbt_decisions",
        "validation_metrics",
        "context_10",
        "aggregate-saturated-cbr",
        "EnableSrsInFSlots",
        "EnableSrsInUlSlots",
    ):
        assert token in text
    assert "global_jain" not in text.lower()
    assert "jain" not in text.lower()
    assert 'std::string nodeRate = "2Mbps"' in text
    assert 'nodeRate != "2Mbps"' in text


def test_validation_scenario_routes_in_backhaul_creation_order() -> None:
    scenario = (
        Path(__file__).parents[1]
        / "ns3"
        / "scenarios"
        / "dblbt-nru-wifi-validation.cc"
    )
    text = scenario.read_text(encoding="ascii")

    assert text.index("setup->ConnectToRemotes") < text.index(
        "nrSetup->ConnectToRemotes"
    )
    assert 'Ipv4Address ("7.0.0.0"), Ipv4Mask ("255.0.0.0"),\n    1 + wifiAps' in text
    assert "Ipv4Mask (\"255.255.255.0\"), 1 + index" in text
