"""CLI workflow orchestration without formal-size experiment runs."""

from __future__ import annotations

import os
from datetime import UTC, datetime
import json
from pathlib import Path
import csv
import hashlib

import pytest

from dblbt_fcn.experiment import (
    MatrixSpec,
    artifact_paths,
    canonical_json,
    expand_matrix,
    job_is_complete,
)
from dblbt_fcn.io import RunManifest, write_jsonl_gz, write_manifest
from dblbt_fcn.provenance import ExecutionProvenance, execution_provenance
from dblbt_fcn.config import adaptive_arms


def tiny_matrix(
    *,
    policies: list[str] | None = None,
    seeds: list[int] | None = None,
    arm_ids: list[int] | None = None,
    rounds: int = 2,
    name: str = "tiny-sweep",
) -> MatrixSpec:
    return MatrixSpec.model_validate(
        {
            "version": 1,
            "name": name,
            "rounds": rounds,
            "alpha": 11,
            "timing": {
                "slot_us": 1,
                "tx_us": 2_000,
                "wifi_ack_us": 0,
                "nru_sync_us": 250,
            },
            "seeds": [410] if seeds is None else seeds,
            "policies": ["random_lbt"] if policies is None else policies,
            "conditions": [],
            "arm_ids": [] if arm_ids is None else arm_ids,
            "scenarios": [
                {
                    "id": "tiny-1x1",
                    "wifi_nodes": 1,
                    "nru_nodes": 1,
                    "legacy_ap_nodes": 0,
                    "legacy_sta_nodes": 0,
                    "traffic": "saturated",
                    "poisson_rate_packets_ms": None,
                    "interference_interval_ms": None,
                    "interference_duration_us": None,
                    "interruption_std": 0.0,
                    "join_interval_rounds": None,
                    "lifetime_rounds": None,
                    "trace": False,
                }
            ],
        }
    )


def canonical_row(
    job: object,
    round_id: int,
    *,
    training_sample: bool = False,
    prior_decision: bool = False,
    local_sequence: int | None = None,
) -> dict[str, object]:
    success = round_id % 2 == 0
    node_id = "wifi-000" if success else "nru-000"
    node_ids = [node_id] if success else ["wifi-000", "nru-000"]
    technologies = ["wifi"] if success else ["wifi", "nru"]
    sample = []
    decisions = []
    if training_sample or prior_decision:
        context = [float(index) / 10.0 for index in range(11)]
        decisions = [
            {
                "round_id": round_id + 1,
                "node_id": "wifi-000",
                "previous_arm": None if prior_decision else job.arm_id,
                "arm": job.arm_id,
                "profile": adaptive_arms()[job.arm_id].model_dump(mode="json"),
                "context": context,
                "reward": None if prior_decision else 0.25,
                "components": None if prior_decision else {
                    "airtime_utility": 0.5,
                    "delay_utility": 0.5,
                    "share_utility": 0.5,
                    "reward": 0.25,
                },
            }
        ]
    if training_sample:
        sample = [
            {
                "context": [float(index) / 10.0 for index in range(11)],
                "arm": job.arm_id,
                "local_reward": 0.25,
                "local_sequence": (
                    round_id if local_sequence is None else local_sequence
                ),
                "node_id": f"{job.run_id}:wifi-000",
                "pretraining_seed": job.seed,
            }
        ]
    return {
        "schema_version": 1,
        "record_type": "contention_round",
        "run_id": job.run_id,
        "scenario_id": job.scenario.id,
        "policy": job.policy,
        "seed": job.seed,
        "config_hash": job.config_hash,
        "round_id": round_id,
        "tx_start_us": round_id * 2_000,
        "round_end_us": (round_id + 1) * 2_000,
        "kind": "success" if success else "collision",
        "node_ids": node_ids,
        "technologies": technologies,
        "collision_size": 0 if success else 2,
        "reservation_us": 0,
        "effective_data_us": 2_000 if success else 0,
        "background_busy_us": 0,
        "active_node_ids": ["wifi-000", "nru-000"],
        "backlogged_node_ids": ["wifi-000", "nru-000"],
        "senders": [
            {
                "node_id": sender_id,
                "technology": technologies[index],
                "selected_backoff_before": 0,
                "next_selected_backoff": 0,
                "interruptions_before": 0,
                "retries_after": 0,
                "db_initialized": True,
                "deterministic_countdown": False,
                "delay_us": 100 + round_id if success else None,
                "effective_data_us": 2_000 if success else 0,
            }
            for index, sender_id in enumerate(node_ids)
        ],
        "decisions": decisions,
        "training_samples": sample,
    }


def write_complete_job(output: Path, job: object) -> None:
    paths = artifact_paths(job, output)
    rows = [
        canonical_row(
            job,
            round_id,
            training_sample=(
                job.policy == "pretrain_arm" and round_id == 63
            ),
            prior_decision=(
                job.policy == "pretrain_arm" and round_id == 31
            ),
            local_sequence=0,
        )
        for round_id in range(job.rounds)
    ]
    metadata = write_jsonl_gz(paths.raw, rows)
    now = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    write_manifest(
        paths.manifest,
        RunManifest(
            run_id=job.run_id,
            scenario_id=job.scenario.id,
            policy=job.policy,
            seed=job.seed,
            config_hash=job.config_hash,
            git_revision="a" * 40,
            dependency_versions={"python": "3.12"},
            host="test-worker",
            started_at_utc=now,
            ended_at_utc=now,
            elapsed_seconds=0.0,
            record_path=str(paths.raw),
            record_hash=metadata.sha256,
            row_count=metadata.row_count,
            exit_code=0,
            status="complete",
            execution_provenance=execution_provenance(job),
        ),
    )


def write_complete_rows(
    output: Path,
    job: object,
    rows: object,
    *,
    execution: object | None = None,
) -> None:
    paths = artifact_paths(job, output)
    metadata = write_jsonl_gz(paths.raw, rows)
    now = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    write_manifest(
        paths.manifest,
        RunManifest(
            run_id=job.run_id,
            scenario_id=job.scenario.id,
            policy=job.policy,
            seed=job.seed,
            config_hash=job.config_hash,
            git_revision="a" * 40,
            dependency_versions={"python": "3.12"},
            host="test-worker",
            started_at_utc=now,
            ended_at_utc=now,
            elapsed_seconds=0.0,
            record_path=str(paths.raw),
            record_hash=metadata.sha256,
            row_count=metadata.row_count,
            exit_code=0,
            status="complete",
            execution_provenance=(
                execution_provenance(job) if execution is None else execution
            ),
        ),
    )


def write_job_config(output: Path, job: object) -> None:
    path = output / "configs" / f"{job.run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        (canonical_json(job) + "\n").encode("ascii")
    )


def write_oracle_inputs(
    root: Path, *, arm: int = 9, source_name: str = "oracle-source"
) -> tuple[Path, Path]:
    from dblbt_fcn.linucb import LinUCB
    from dblbt_fcn.workflows import action_grid_hash

    root.mkdir(parents=True, exist_ok=True)
    source = tiny_matrix(
        policies=["pretrain_arm"],
        seeds=[1103, 2207, 3301],
        arm_ids=list(range(24)),
        name=source_name,
    )
    model = root / "model.npz"
    LinUCB(24, 11, action_grid_hash=action_grid_hash()).save(model)
    oracle = root / "oracle.json"
    oracle.write_bytes(
        (
            canonical_json(
                {
                    "schema_version": 1,
                    "arm": arm,
                    "action_grid_hash": action_grid_hash(),
                    "source_matrix": source.model_dump(mode="json"),
                    "source_matrix_hash": hashlib.sha256(
                        canonical_json(source).encode("ascii")
                    ).hexdigest(),
                    "model_sha256": hashlib.sha256(
                        model.read_bytes()
                    ).hexdigest(),
                }
            )
            + "\n"
        ).encode("ascii")
    )
    return model, oracle


@pytest.mark.parametrize("value", [0, -1, True, 1.0, "1"])
def test_worker_count_rejects_non_positive_or_non_exact_integers(
    value: object,
) -> None:
    from dblbt_fcn.workflows import effective_worker_count

    with pytest.raises(ValueError, match="workers"):
        effective_worker_count(value)


def test_worker_count_accepts_one_and_caps_requests_at_24() -> None:
    from dblbt_fcn.workflows import effective_worker_count

    assert effective_worker_count(1) == 1
    assert effective_worker_count(25) == 24


def test_oracle_loader_requires_complete_provenance(tmp_path: Path) -> None:
    from dblbt_fcn.workflows import action_grid_hash, load_oracle_arm

    path = tmp_path / "oracle.json"
    path.write_bytes(
        (
        canonical_json(
            {
                "schema_version": 1,
                "arm": 3,
                "action_grid_hash": action_grid_hash(),
            }
        )
        + "\n"
        ).encode("ascii")
    )

    with pytest.raises((TypeError, ValueError), match="fields|provenance|model"):
        load_oracle_arm(path, model_path=tmp_path / "missing.npz")


@pytest.mark.parametrize("forgery", ["model", "source"])
def test_oracle_loader_rejects_forged_model_or_source_hash(
    tmp_path: Path, forgery: str
) -> None:
    from dblbt_fcn.linucb import LinUCB
    from dblbt_fcn.workflows import action_grid_hash, load_oracle_arm

    matrix = tiny_matrix(
        policies=["pretrain_arm"],
        seeds=[1103, 2207, 3301],
        arm_ids=list(range(24)),
        name="oracle-source",
    )
    model = tmp_path / "model.npz"
    LinUCB(24, 11, action_grid_hash=action_grid_hash()).save(model)
    value = {
        "schema_version": 1,
        "arm": 3,
        "action_grid_hash": action_grid_hash(),
        "source_matrix": matrix.model_dump(mode="json"),
        "source_matrix_hash": hashlib.sha256(
            canonical_json(matrix).encode("ascii")
        ).hexdigest(),
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
    }
    if forgery == "model":
        value["model_sha256"] = "f" * 64
    else:
        value["source_matrix_hash"] = "f" * 64
    oracle = tmp_path / "oracle.json"
    oracle.write_bytes((canonical_json(value) + "\n").encode("ascii"))

    with pytest.raises(ValueError, match=forgery):
        load_oracle_arm(oracle, model_path=model)


@pytest.mark.parametrize(
    ("source", "match"),
    [
        ("heldout", "pretrain"),
        ("wrong_seed", "PRETRAINING_SEEDS"),
        ("missing_arm", "arms"),
        ("wrong_policy", "pretrain_arm"),
    ],
)
def test_oracle_loader_rejects_non_pretraining_source_matrix(
    tmp_path: Path, source: str, match: str
) -> None:
    from dblbt_fcn.linucb import LinUCB
    from dblbt_fcn.workflows import action_grid_hash, load_oracle_arm

    if source == "heldout":
        matrix = tiny_matrix(
            policies=["random_lbt"], seeds=[410], name="heldout"
        )
    elif source == "wrong_seed":
        matrix = tiny_matrix(
            policies=["pretrain_arm"],
            seeds=[410, 2207, 3301],
            arm_ids=list(range(24)),
            name="wrong-seed-source",
        )
    elif source == "missing_arm":
        matrix = tiny_matrix(
            policies=["pretrain_arm"],
            seeds=[1103, 2207, 3301],
            arm_ids=list(range(23)),
            name="missing-arm-source",
        )
    else:
        matrix = tiny_matrix(
            policies=["random_lbt"],
            seeds=[1103, 2207, 3301],
            name="wrong-policy-source",
        )
    model = tmp_path / "model.npz"
    LinUCB(24, 11, action_grid_hash=action_grid_hash()).save(model)
    value = {
        "schema_version": 1,
        "arm": 3,
        "action_grid_hash": action_grid_hash(),
        "source_matrix": matrix.model_dump(mode="json"),
        "source_matrix_hash": hashlib.sha256(
            canonical_json(matrix).encode("ascii")
        ).hexdigest(),
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
    }
    oracle = tmp_path / "oracle.json"
    oracle.write_bytes((canonical_json(value) + "\n").encode("ascii"))

    with pytest.raises(ValueError, match=match):
        load_oracle_arm(oracle, model_path=model)


def test_sweep_submits_one_future_per_job_in_stable_order_and_caps_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dblbt_fcn.workflows as workflows

    matrix = tiny_matrix(
        policies=["random_lbt", "tmc_db_lbt"], seeds=[7, 8]
    )
    expected_jobs = expand_matrix(matrix)
    output = tmp_path / "runs"
    observed: dict[str, object] = {"submitted": []}

    class Future:
        def __init__(self, job: object) -> None:
            self.job = job

        def result(self) -> dict[str, object]:
            write_complete_job(output, self.job)
            manifest = workflows.load_completed_job_manifest(self.job, output)
            return manifest.model_dump(mode="json")

    class Executor:
        def __init__(self, *, max_workers: int) -> None:
            observed["max_workers"] = max_workers

        def __enter__(self) -> "Executor":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def submit(self, function: object, job: object, *args: object) -> Future:
            assert function is workflows._run_sweep_job
            observed["submitted"].append((job, args))  # type: ignore[union-attr]
            return Future(job)

        def shutdown(self, **kwargs: object) -> None:
            observed["shutdown"] = kwargs

    monkeypatch.setattr(workflows, "ProcessPoolExecutor", Executor)
    monkeypatch.setattr(
        workflows,
        "wait",
        lambda futures, return_when: (set(futures), set()),
    )

    results = workflows.run_sweep(
        matrix, output, workers=25
    )

    assert observed["max_workers"] == 24
    assert observed["shutdown"] == {"wait": True, "cancel_futures": False}
    submitted = [item[0] for item in observed["submitted"]]  # type: ignore[index]
    assert submitted == expected_jobs
    assert [result.run_id for result in results] == [
        job.run_id for job in expected_jobs
    ]


@pytest.mark.parametrize("result_kind", ["nondict", "wrong-job", "missing-artifact"])
def test_sweep_rejects_untrusted_worker_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    result_kind: str,
) -> None:
    import dblbt_fcn.workflows as workflows

    matrix = tiny_matrix(rounds=1)
    job = expand_matrix(matrix)[0]
    output = tmp_path / "runs"

    class Future:
        def result(self) -> object:
            if result_kind == "nondict":
                return object()
            if result_kind == "missing-artifact":
                manifest: dict[str, object] = {
                    "run_id": job.run_id,
                    "scenario_id": job.scenario.id,
                    "policy": job.policy,
                    "seed": job.seed,
                    "config_hash": job.config_hash,
                    "git_revision": "a" * 40,
                    "dependency_versions": {"python": "3.12"},
                    "host": "test-worker",
                    "started_at_utc": "2026-01-02T03:04:05Z",
                    "ended_at_utc": "2026-01-02T03:04:05Z",
                    "elapsed_seconds": 0.0,
                    "record_path": str(artifact_paths(job, output).raw),
                    "record_hash": "0" * 64,
                    "row_count": job.rounds,
                    "exit_code": 0,
                    "status": "complete",
                }
            else:
                write_complete_job(output, job)
                manifest = workflows.load_completed_job_manifest(
                    job, output
                ).model_dump(mode="json")
            if result_kind == "wrong-job":
                manifest["run_id"] = "f" * 16
            return manifest

    class Executor:
        def __init__(self, *, max_workers: int) -> None:
            pass

        def __enter__(self) -> "Executor":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def submit(self, *args: object) -> Future:
            return Future()

        def shutdown(self, **kwargs: object) -> None:
            pass

    monkeypatch.setattr(workflows, "ProcessPoolExecutor", Executor)
    monkeypatch.setattr(
        workflows,
        "wait",
        lambda futures, return_when: (set(futures), set()),
    )

    with pytest.raises((TypeError, ValueError, FileNotFoundError)):
        workflows.run_sweep(matrix, output, workers=1)


def test_sweep_first_failure_with_one_worker_never_submits_later_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dblbt_fcn.workflows as workflows

    matrix = tiny_matrix(seeds=[1, 2, 3], rounds=1)
    submitted: list[str] = []

    class Future:
        def result(self) -> object:
            raise RuntimeError("first worker failed")

        def cancel(self) -> bool:
            return True

    class Executor:
        def __init__(self, *, max_workers: int) -> None:
            assert max_workers == 1

        def __enter__(self) -> "Executor":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def submit(self, function: object, job: object, *args: object) -> Future:
            submitted.append(job.run_id)  # type: ignore[union-attr]
            return Future()

        def shutdown(self, **kwargs: object) -> None:
            assert kwargs.get("cancel_futures") is True

    monkeypatch.setattr(workflows, "ProcessPoolExecutor", Executor)
    monkeypatch.setattr(
        workflows,
        "wait",
        lambda futures, return_when: (set(futures), set()),
    )

    with pytest.raises(RuntimeError, match="first worker failed"):
        workflows.run_sweep(matrix, tmp_path / "runs", workers=1)

    assert len(submitted) == 1


def test_sweep_preserves_canonical_config_for_skipped_complete_job(
    tmp_path: Path,
) -> None:
    from dblbt_fcn.simulation import run_job
    from dblbt_fcn.workflows import run_sweep

    matrix = tiny_matrix(rounds=1)
    job = expand_matrix(matrix)[0]
    output = tmp_path / "runs"
    run_job(job, output)
    config = output / "configs" / f"{job.run_id}.json"
    before = config.read_bytes()

    assert run_sweep(matrix, output, workers=1) == []
    assert config.read_bytes() == before


@pytest.mark.skipif(os.name != "posix", reason="safe sidecar backfill requires WSL/POSIX")
def test_sweep_backfills_missing_sidecar_without_rewriting_complete_job(
    tmp_path: Path,
) -> None:
    from dblbt_fcn.experiment import canonical_json
    from dblbt_fcn.simulation import run_job
    from dblbt_fcn.workflows import run_sweep

    matrix = tiny_matrix(rounds=1)
    job = expand_matrix(matrix)[0]
    output = tmp_path / "runs"
    run_job(job, output)
    paths = artifact_paths(job, output)
    before = {
        path: path.read_bytes()
        for path in (paths.raw, paths.marker, paths.manifest)
    }
    config = output / "configs" / f"{job.run_id}.json"
    config.unlink()

    assert run_sweep(matrix, output, workers=1) == []
    assert config.read_bytes() == (canonical_json(job) + "\n").encode("ascii")
    assert all(path.read_bytes() == payload for path, payload in before.items())


@pytest.mark.skipif(os.name != "posix", reason="real sweep lock recovery requires WSL/POSIX")
def test_real_process_sweep_skips_complete_and_reruns_corrupt_job(
    tmp_path: Path,
) -> None:
    from dblbt_fcn.workflows import run_sweep

    matrix = tiny_matrix(rounds=3)
    job = expand_matrix(matrix)[0]
    output = tmp_path / "runs"

    first = run_sweep(matrix, output, workers=1)
    paths = artifact_paths(job, output)
    before = {path: path.read_bytes() for path in (paths.raw, paths.marker, paths.manifest)}
    skipped = run_sweep(matrix, output, workers=1)

    assert [item.run_id for item in first] == [job.run_id]
    assert skipped == []
    assert {path: path.read_bytes() for path in before} == before

    paths.raw.write_bytes(before[paths.raw][:8])
    rerun = run_sweep(matrix, output, workers=1)

    assert [item.run_id for item in rerun] == [job.run_id]
    assert paths.raw.read_bytes() == before[paths.raw]
    assert job_is_complete(job, output)


@pytest.mark.skipif(os.name != "posix", reason="real Oracle resume requires WSL/POSIX")
def test_sweep_reruns_only_fixed_job_when_oracle_artifact_changes(
    tmp_path: Path,
) -> None:
    from dblbt_fcn.workflows import run_sweep

    matrix = tiny_matrix(
        policies=["random_lbt", "fixed_oracle"], rounds=1
    )
    output = tmp_path / "runs"
    model, oracle = write_oracle_inputs(tmp_path, source_name="source-a")
    jobs = expand_matrix(matrix)
    fixed = next(job for job in jobs if job.policy == "fixed_oracle")
    baseline = next(job for job in jobs if job.policy == "random_lbt")

    first = run_sweep(
        matrix,
        output,
        workers=1,
        model_path=model,
        oracle_arm_path=oracle,
    )
    fixed_paths = artifact_paths(fixed, output)
    baseline_paths = artifact_paths(baseline, output)
    first_fixed_fingerprint = next(
        item.execution_fingerprint
        for item in first
        if item.policy == "fixed_oracle"
    )
    before = {
        path: path.read_bytes()
        for path in (
            fixed_paths.raw,
            fixed_paths.marker,
            fixed_paths.manifest,
            baseline_paths.raw,
            baseline_paths.marker,
            baseline_paths.manifest,
        )
    }

    assert run_sweep(
        matrix,
        output,
        workers=1,
        model_path=model,
        oracle_arm_path=oracle,
    ) == []
    assert all(path.read_bytes() == payload for path, payload in before.items())

    replacement = json.loads(oracle.read_text(encoding="ascii"))
    replacement["source_matrix"]["name"] = "source-b"
    replacement_source = MatrixSpec.model_validate(
        replacement["source_matrix"]
    )
    replacement["source_matrix_hash"] = hashlib.sha256(
        canonical_json(replacement_source).encode("ascii")
    ).hexdigest()
    oracle.write_bytes(
        (canonical_json(replacement) + "\n").encode("ascii")
    )
    changed = run_sweep(
        matrix,
        output,
        workers=1,
        model_path=model,
        oracle_arm_path=oracle,
    )

    assert [item.run_id for item in changed] == [fixed.run_id]
    assert changed[0].execution_fingerprint != first_fixed_fingerprint
    assert all(
        path.read_bytes() == before[path]
        for path in (
            baseline_paths.raw,
            baseline_paths.marker,
            baseline_paths.manifest,
        )
    )


@pytest.mark.skipif(os.name != "posix", reason="Oracle provenance requires WSL/POSIX")
def test_run_job_rejects_fixed_oracle_execution_with_wrong_arm(
    tmp_path: Path,
) -> None:
    from dblbt_fcn.linucb import LinUCB
    from dblbt_fcn.provenance import execution_provenance, file_sha256
    from dblbt_fcn.simulation import run_job
    from dblbt_fcn.workflows import action_grid_hash, load_oracle_arm

    job = expand_matrix(
        tiny_matrix(policies=["fixed_oracle"], rounds=1)
    )[0]
    model, oracle_path = write_oracle_inputs(tmp_path, arm=9)
    oracle = load_oracle_arm(oracle_path, model_path=model)
    agent = LinUCB.load(model, expected_action_grid_hash=action_grid_hash())
    stale = execution_provenance(
        job,
        initial_agent=agent,
        model_path=model,
        oracle_arm=8,
        oracle_artifact_sha256=file_sha256(oracle_path),
        oracle_model_sha256=oracle.model_sha256,
        source_matrix_sha256=oracle.source_matrix_hash,
    )

    with pytest.raises(ValueError, match="execution provenance"):
        run_job(
            job,
            tmp_path / "runs",
            initial_agent=agent,
            oracle_arm=9,
            model_path=model,
            oracle_artifact_path=oracle_path,
            execution=stale,
        )

    assert not (tmp_path / "runs").exists()


@pytest.mark.skipif(os.name != "posix", reason="real raw conformance requires WSL/POSIX")
@pytest.mark.parametrize("policy", ["random_lbt", "tmc_db_lbt"])
def test_real_baseline_raw_sender_boolean_conforms_to_reader(
    tmp_path: Path, policy: str
) -> None:
    from dblbt_fcn.records import read_job_rows
    from dblbt_fcn.simulation import run_job

    job = expand_matrix(tiny_matrix(policies=[policy], rounds=3))[0]
    output = tmp_path / "runs"

    run_job(job, output)
    rows = read_job_rows(job, output)

    assert len(rows) == 3
    assert all(
        type(sender["deterministic_countdown"]) is bool
        for row in rows
        for sender in row["senders"]
    )


@pytest.mark.skipif(os.name != "posix", reason="real raw conformance requires WSL/POSIX")
def test_summary_reads_real_random_tmc_and_adaptive_runs(tmp_path: Path) -> None:
    from dblbt_fcn.reporting import summarize_manifests
    from dblbt_fcn.workflows import run_sweep

    matrix = tiny_matrix(
        policies=["random_lbt", "tmc_db_lbt", "adaptive_db_lbt"],
        rounds=64,
        name="smoke",
    )
    output = tmp_path / "runs"
    jobs = expand_matrix(matrix)

    run_sweep(matrix, output, workers=2)
    rows = summarize_manifests(
        output / "manifests", tmp_path / "summary.csv"
    )

    assert [row["run_id"] for row in rows] == sorted(job.run_id for job in jobs)
    assert {row["policy"] for row in rows} == {
        "random_lbt",
        "tmc_db_lbt",
        "adaptive_db_lbt",
    }
    adaptive = next(row for row in rows if row["policy"] == "adaptive_db_lbt")
    assert adaptive["decision_count"] > 0


def test_iter_job_rows_and_one_shot_aggregate_preserve_reader_semantics(
    tmp_path: Path,
) -> None:
    from dblbt_fcn.records import aggregate_rows, iter_job_rows, read_job_rows
    from dblbt_fcn.simulation import run_job

    job = expand_matrix(tiny_matrix(policies=["random_lbt"], rounds=8))[0]
    run_job(job, tmp_path)
    expected = aggregate_rows(read_job_rows(job, tmp_path))
    stream = iter_job_rows(job, tmp_path)

    assert iter(stream) is stream
    assert aggregate_rows(stream) == expected


def test_validated_job_rows_close_is_idempotent_and_releases_raw_handle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dblbt_fcn.records import iter_job_rows
    from dblbt_fcn.simulation import run_job

    job = expand_matrix(tiny_matrix(policies=["random_lbt"], rounds=8))[0]
    run_job(job, tmp_path)
    raw = artifact_paths(job, tmp_path).raw
    original_open = Path.open
    opened: list[object] = []

    def observed_open(
        path: Path, *args: object, **kwargs: object
    ) -> object:
        handle = original_open(path, *args, **kwargs)
        if path == raw and args and args[0] == "rb":
            opened.append(handle)
        return handle

    monkeypatch.setattr(Path, "open", observed_open)
    stream = iter_job_rows(job, tmp_path)

    assert not stream.closed
    next(stream)
    assert opened and not opened[-1].closed  # type: ignore[attr-defined]

    stream.close()
    stream.close()

    assert stream.closed
    assert opened[-1].closed  # type: ignore[attr-defined]
    with pytest.raises(StopIteration):
        next(stream)


def test_validated_job_rows_context_manager_closes_partial_stream(
    tmp_path: Path,
) -> None:
    from dblbt_fcn.records import iter_job_rows
    from dblbt_fcn.simulation import run_job

    job = expand_matrix(tiny_matrix(policies=["random_lbt"], rounds=8))[0]
    run_job(job, tmp_path)
    stream = iter_job_rows(job, tmp_path)

    with stream as rows:
        next(rows)
        assert not rows.closed

    assert stream.closed
    with pytest.raises(StopIteration):
        next(stream)


def test_aggregate_rows_rejects_partial_and_reordered_round_sequences(
    tmp_path: Path,
) -> None:
    from dblbt_fcn.records import aggregate_rows, iter_job_rows, read_job_rows
    from dblbt_fcn.simulation import run_job

    job = expand_matrix(tiny_matrix(policies=["random_lbt"], rounds=8))[0]
    run_job(job, tmp_path)
    partial = iter_job_rows(job, tmp_path)
    next(partial)
    rows = read_job_rows(job, tmp_path)
    reordered = [rows[0], rows[2], rows[1], *rows[3:]]

    with pytest.raises(ValueError, match="round|contiguous|start"):
        aggregate_rows(partial)
    assert partial.closed
    with pytest.raises(ValueError, match="round|contiguous|start"):
        aggregate_rows(reordered)


def test_aggregate_rows_closes_stream_on_success_and_observer_error(
    tmp_path: Path,
) -> None:
    from dblbt_fcn.records import aggregate_rows, iter_job_rows
    from dblbt_fcn.simulation import run_job

    job = expand_matrix(tiny_matrix(policies=["random_lbt"], rounds=8))[0]
    run_job(job, tmp_path)
    successful = iter_job_rows(job, tmp_path)

    assert aggregate_rows(successful).rounds == 8
    assert successful.closed

    failed = iter_job_rows(job, tmp_path)

    def fail(_row: dict[str, object]) -> None:
        raise RuntimeError("observer failed")

    with pytest.raises(RuntimeError, match="observer failed"):
        aggregate_rows(failed, observer=fail)

    assert failed.closed


def test_summary_streams_rows_without_read_job_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dblbt_fcn.records as records
    from dblbt_fcn.reporting import summarize_manifests
    from dblbt_fcn.simulation import run_job

    job = expand_matrix(tiny_matrix(policies=["random_lbt"], rounds=8))[0]
    run_job(job, tmp_path)
    monkeypatch.setattr(
        records,
        "read_job_rows",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("summary must stream raw rows")
        ),
    )

    output = tmp_path / "summary.csv"
    summarize_manifests(tmp_path / "manifests", output)

    assert output.is_file()


def test_reporting_decompresses_each_job_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import gzip

    from dblbt_fcn.reporting import summarize_manifests
    from dblbt_fcn.simulation import run_job

    jobs = expand_matrix(
        tiny_matrix(
            policies=["random_lbt", "tmc_db_lbt", "adaptive_db_lbt"],
            rounds=8,
            name="smoke",
        )
    )
    root = tmp_path / "runs"
    for job in jobs:
        run_job(job, root)
    original = gzip.GzipFile
    reads = 0

    def observed(*args: object, **kwargs: object):
        nonlocal reads
        mode = kwargs.get("mode", args[1] if len(args) > 1 else None)
        if mode in {"rb", "r"}:
            reads += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(gzip, "GzipFile", observed)

    summarize_manifests(root / "manifests", tmp_path / "summary.csv")

    assert reads == len(jobs)


@pytest.mark.skipif(os.name != "posix", reason="real raw conformance requires WSL/POSIX")
def test_real_adaptive_completed_boundaries_conform_to_reader(
    tmp_path: Path,
) -> None:
    from dblbt_fcn.records import read_job_rows
    from dblbt_fcn.simulation import run_job

    matrix = tiny_matrix(
        policies=["adaptive_db_lbt"], rounds=64, name="smoke"
    )
    job = expand_matrix(matrix)[0]
    output = tmp_path / "runs"

    run_job(job, output)
    rows = read_job_rows(job, output)
    boundaries = [
        (row["round_id"], decision["round_id"])
        for row in rows
        for decision in row["decisions"]
    ]

    assert boundaries
    assert {decision_round for _, decision_round in boundaries} == {32, 64}
    assert all(decision_round == row_id + 1 for row_id, decision_round in boundaries)
    assert all(decision_round % 32 == 0 for _, decision_round in boundaries)


@pytest.mark.skipif(os.name != "posix", reason="real raw conformance requires WSL/POSIX")
def test_real_fixed_oracle_rows_conform_without_training_samples(
    tmp_path: Path,
) -> None:
    from dblbt_fcn.linucb import LinUCB
    from dblbt_fcn.records import read_job_rows
    from dblbt_fcn.simulation import run_job
    from dblbt_fcn.workflows import action_grid_hash

    matrix = tiny_matrix(policies=["fixed_oracle"], rounds=64)
    job = expand_matrix(matrix)[0]
    output = tmp_path / "runs"
    model, oracle = write_oracle_inputs(tmp_path)
    agent = LinUCB.load(model, expected_action_grid_hash=action_grid_hash())

    run_job(
        job,
        output,
        initial_agent=agent,
        oracle_arm=9,
        model_path=model,
        oracle_artifact_path=oracle,
    )
    rows = read_job_rows(job, output)

    assert any(row["decisions"] for row in rows)
    assert all(not row["training_samples"] for row in rows)
    assert {
        decision["arm"]
        for row in rows
        for decision in row["decisions"]
    } == {9}


@pytest.mark.skipif(os.name != "posix", reason="real raw conformance requires WSL/POSIX")
def test_real_pretrain_boundaries_and_samples_conform_to_reader(
    tmp_path: Path,
) -> None:
    from dblbt_fcn.records import read_job_rows
    from dblbt_fcn.simulation import run_job

    matrix = tiny_matrix(
        policies=["pretrain_arm"],
        seeds=[1103],
        arm_ids=[7],
        rounds=96,
        name="tiny-pretrain-reader",
    )
    job = expand_matrix(matrix)[0]
    output = tmp_path / "runs"

    run_job(job, output)
    rows = read_job_rows(job, output)
    decisions = [decision for row in rows for decision in row["decisions"]]
    samples = [sample for row in rows for sample in row["training_samples"]]

    assert {decision["round_id"] for decision in decisions} == {32, 64, 96}
    assert samples
    assert all(sample["arm"] == 7 for sample in samples)
    assert all(sample["pretraining_seed"] == 1103 for sample in samples)


@pytest.mark.parametrize("damage", ["topology", "decision", "training"])
def test_reader_rejects_semantically_tampered_but_integrity_valid_raw(
    tmp_path: Path, damage: str
) -> None:
    from dblbt_fcn.records import read_job_rows
    from dblbt_fcn.simulation import simulate_job_records

    if damage == "topology":
        matrix = tiny_matrix(policies=["random_lbt"], rounds=3)
    elif damage == "decision":
        matrix = tiny_matrix(
            policies=["adaptive_db_lbt"], rounds=64, name="smoke"
        )
    else:
        matrix = tiny_matrix(
            policies=["pretrain_arm"],
            seeds=[1103],
            arm_ids=[7],
            rounds=96,
            name="semantic-pretrain",
        )
    job = expand_matrix(matrix)[0]
    rows = list(simulate_job_records(job))
    if damage == "topology":
        rows[0]["active_node_ids"].append(rows[0]["active_node_ids"][0])
    elif damage == "decision":
        decision = next(
            item for row in rows for item in row["decisions"]
        )
        decision["profile"]["beta"] = 99
    else:
        sample = next(
            item for row in rows for item in row["training_samples"]
        )
        sample["context"] = [0.75] * 11
    write_complete_rows(tmp_path, job, rows)

    with pytest.raises(ValueError, match="active|profile|training|context"):
        read_job_rows(job, tmp_path)


@pytest.mark.parametrize(
    ("damage", "match"),
    [
        ("fixed_arm", "Oracle arm"),
        ("pretrain_arm", "pretrain arm"),
        ("static_active", "active"),
        ("impossible_reward", "reward"),
    ],
)
def test_reader_rejects_policy_topology_and_reward_semantic_forgery(
    tmp_path: Path, damage: str, match: str
) -> None:
    from dblbt_fcn.records import read_job_rows
    from dblbt_fcn.simulation import simulate_job_records

    if damage == "fixed_arm":
        job = expand_matrix(
            tiny_matrix(policies=["fixed_oracle"], rounds=64)
        )[0]
        rows = list(simulate_job_records(job, oracle_arm=9))
        execution = ExecutionProvenance(
            mode="fixed_oracle",
            agent_state_sha256="a" * 64,
            model_file_sha256="b" * 64,
            oracle_arm=9,
            oracle_artifact_sha256="c" * 64,
            oracle_model_sha256="b" * 64,
            source_matrix_sha256="d" * 64,
        )
        decision = next(item for row in rows for item in row["decisions"])
        decision["arm"] = 8
        decision["profile"] = adaptive_arms()[8].model_dump(mode="json")
    elif damage in {"pretrain_arm", "impossible_reward"}:
        job = expand_matrix(
            tiny_matrix(
                policies=["pretrain_arm"],
                seeds=[1103],
                arm_ids=[7],
                rounds=96,
                name="raw-policy-binding",
            )
        )[0]
        rows = list(simulate_job_records(job))
        execution = execution_provenance(job)
        if damage == "pretrain_arm":
            for row in rows:
                for decision in row["decisions"]:
                    decision["arm"] = 8
                    if decision["previous_arm"] is not None:
                        decision["previous_arm"] = 8
                    decision["profile"] = adaptive_arms()[8].model_dump(
                        mode="json"
                    )
                for sample in row["training_samples"]:
                    sample["arm"] = 8
        else:
            rewarded = next(
                item
                for row in rows
                for item in row["decisions"]
                if item["reward"] is not None
            )
            rewarded["reward"] = 999.0
            rewarded["components"]["reward"] = 999.0
            sample = next(
                item for row in rows for item in row["training_samples"]
            )
            sample["local_reward"] = 999.0
    else:
        job = expand_matrix(tiny_matrix(rounds=2))[0]
        rows = list(simulate_job_records(job))
        execution = execution_provenance(job)
        removable = next(
            node_id
            for node_id in rows[0]["active_node_ids"]
            if node_id not in rows[0]["node_ids"]
        )
        rows[0]["active_node_ids"].remove(removable)
        rows[0]["backlogged_node_ids"].remove(removable)
    write_complete_rows(tmp_path, job, rows, execution=execution)

    with pytest.raises(ValueError, match=match):
        read_job_rows(job, tmp_path)


def test_reader_accepts_exact_dynamic_active_windows(tmp_path: Path) -> None:
    from dblbt_fcn.records import read_job_rows
    from dblbt_fcn.simulation import simulate_job_records

    matrix = tiny_matrix(rounds=64)
    value = matrix.model_dump(mode="json")
    value["scenarios"][0]["join_interval_rounds"] = 10
    value["scenarios"][0]["lifetime_rounds"] = 40
    dynamic = MatrixSpec.model_validate(value)
    job = expand_matrix(dynamic)[0]
    write_complete_rows(tmp_path, job, simulate_job_records(job))

    rows = read_job_rows(job, tmp_path)

    assert len(rows) == job.rounds


def test_reader_requires_policy_provenance_before_first_decision(
    tmp_path: Path,
) -> None:
    from dblbt_fcn.records import read_job_rows
    from dblbt_fcn.simulation import simulate_job_records

    matrix = tiny_matrix(
        policies=["pretrain_arm"],
        seeds=[1103],
        arm_ids=[7],
        rounds=1,
    )
    job = expand_matrix(matrix)[0]
    rows = simulate_job_records(job)
    write_complete_rows(
        tmp_path,
        job,
        rows,
        execution=ExecutionProvenance(mode="baseline_builtin"),
    )

    with pytest.raises(ValueError, match="provenance"):
        read_job_rows(job, tmp_path)


@pytest.mark.parametrize(
    ("policy", "ablation", "mode"),
    [
        ("random_lbt", None, "adaptive_blank"),
        ("primary_db_lbt", None, "adaptive_blank"),
        ("tmc_db_lbt", None, "adaptive_blank"),
        ("adaptive_db_lbt", None, "baseline_builtin"),
        ("adaptive_db_lbt", "context_free_ucb", "adaptive_blank"),
    ],
)
def test_reader_rejects_execution_mode_that_conflicts_with_job(
    tmp_path: Path, policy: str, ablation: str | None, mode: str
) -> None:
    from dblbt_fcn.records import read_job_rows
    from dblbt_fcn.simulation import simulate_job_records

    matrix = tiny_matrix(policies=[policy], rounds=1, name="mode-binding")
    value = matrix.model_dump(mode="json")
    value["conditions"] = [] if ablation is None else [ablation]
    job = expand_matrix(MatrixSpec.model_validate(value))[0]
    execution = (
        ExecutionProvenance(mode=mode, agent_state_sha256="a" * 64)
        if mode == "adaptive_blank"
        else ExecutionProvenance(mode=mode)
    )
    write_complete_rows(
        tmp_path, job, simulate_job_records(job), execution=execution
    )

    with pytest.raises(ValueError, match="mode|provenance|policy"):
        read_job_rows(job, tmp_path)


@pytest.mark.skipif(os.name != "posix", reason="real worker cleanup requires WSL/POSIX")
def test_real_worker_failure_propagates_without_completion(tmp_path: Path) -> None:
    from dblbt_fcn.workflows import run_sweep

    matrix = tiny_matrix(policies=["fixed_oracle"])
    job = expand_matrix(matrix)[0]
    output = tmp_path / "runs"

    with pytest.raises(ValueError, match="oracle"):
        run_sweep(matrix, output, workers=1)

    paths = artifact_paths(job, output)
    assert not paths.manifest.exists()
    assert not paths.raw.exists()
    assert not paths.marker.exists()
    assert not job_is_complete(job, output)


def test_pretraining_builds_loadable_model_and_lowest_tied_oracle(
    tmp_path: Path,
) -> None:
    from dblbt_fcn.linucb import LinUCB
    from dblbt_fcn.workflows import action_grid_hash, build_pretraining

    matrix = tiny_matrix(
        policies=["pretrain_arm"],
        seeds=[1103, 2207, 3301],
        arm_ids=list(range(24)),
        rounds=64,
        name="tiny-pretrain",
    )
    run_dir = tmp_path / "runs"
    for job in expand_matrix(matrix):
        write_complete_job(run_dir, job)
    model_path = tmp_path / "models" / "initial.npz"
    oracle_path = tmp_path / "models" / "fixed-oracle-arm.json"

    model, oracle_arm = build_pretraining(
        matrix,
        run_dir,
        model_path,
        oracle_path,
        workers=1,
    )

    assert oracle_arm == 0
    expected_hash = action_grid_hash()
    loaded = LinUCB.load(
        model_path, expected_action_grid_hash=expected_hash
    )
    assert loaded.num_arms == 24
    assert loaded.context_dim == 11
    assert loaded.ridge == 1.0
    assert loaded.exploration == 0.5
    assert all(loaded.b[arm].any() for arm in range(24))
    assert not loaded.A.flags.writebackifcopy
    assert model is not loaded
    oracle = json.loads(oracle_path.read_text(encoding="ascii"))
    assert oracle["schema_version"] == 1
    assert oracle["arm"] == 0
    assert oracle["action_grid_hash"] == expected_hash
    assert len(oracle["source_matrix_hash"]) == 64
    assert len(oracle["model_sha256"]) == 64


def test_pretraining_consumes_real_simulation_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dblbt_fcn.workflows as workflows
    from dblbt_fcn.simulation import simulate_job_records

    matrix = tiny_matrix(
        policies=["pretrain_arm"],
        seeds=[1103, 2207, 3301],
        arm_ids=list(range(24)),
        rounds=64,
        name="tiny-real-pretrain",
    )
    run_dir = tmp_path / "runs"
    jobs = expand_matrix(matrix)
    for job in jobs:
        write_complete_rows(
            run_dir, job, simulate_job_records(job)
        )
    model_path = tmp_path / "model.npz"
    oracle_path = tmp_path / "oracle.json"
    monkeypatch.setattr(workflows, "run_sweep", lambda *args, **kwargs: [])

    trained, arm = workflows.build_pretraining(
        matrix,
        run_dir,
        model_path,
        oracle_path,
        workers=1,
    )

    assert 0 <= arm < 24
    assert model_path.is_file()
    assert oracle_path.is_file()
    assert any(trained.b[arm_index].any() for arm_index in range(24))


def test_pretraining_model_publish_failure_preserves_existing_oracle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dblbt_fcn.workflows as workflows

    matrix = tiny_matrix(
        policies=["pretrain_arm"],
        seeds=[1103, 2207, 3301],
        arm_ids=list(range(24)),
        rounds=64,
        name="tiny-pretrain",
    )
    run_dir = tmp_path / "runs"
    for job in expand_matrix(matrix):
        write_complete_job(run_dir, job)
    model_path = tmp_path / "models" / "initial.npz"
    oracle_path = tmp_path / "models" / "fixed-oracle-arm.json"
    oracle_path.parent.mkdir(parents=True)
    original_oracle = b"existing-oracle\n"
    oracle_path.write_bytes(original_oracle)
    real_replace = workflows.os.replace

    def fail_model_publish(source: object, destination: object) -> None:
        if Path(destination) == model_path:
            raise OSError("model publish failed")
        real_replace(source, destination)

    monkeypatch.setattr(workflows.os, "replace", fail_model_publish)

    with pytest.raises(OSError, match="model publish failed"):
        workflows.build_pretraining(
            matrix,
            run_dir,
            model_path,
            oracle_path,
            workers=1,
        )

    assert oracle_path.read_bytes() == original_oracle
    assert not model_path.exists()


@pytest.mark.parametrize("failure", ["heldout", "missing", "corrupt"])
def test_pretraining_rejects_leakage_missing_or_corrupt_inputs_without_outputs(
    tmp_path: Path,
    failure: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dblbt_fcn.workflows as workflows

    seeds = [410] if failure == "heldout" else [1103, 2207, 3301]
    matrix = tiny_matrix(
        policies=["pretrain_arm"],
        seeds=seeds,
        arm_ids=list(range(24)),
        rounds=1,
        name="tiny-pretrain",
    )
    run_dir = tmp_path / "runs"
    jobs = expand_matrix(matrix)
    if failure != "heldout":
        for job in jobs:
            write_complete_job(run_dir, job)
        damaged = artifact_paths(jobs[0], run_dir)
        if failure == "missing":
            damaged.manifest.unlink()
        else:
            damaged.raw.write_bytes(b"not gzip")
    model_path = tmp_path / "models" / "initial.npz"
    oracle_path = tmp_path / "models" / "fixed-oracle-arm.json"
    if failure != "heldout":
        monkeypatch.setattr(workflows, "run_sweep", lambda *args, **kwargs: [])

    with pytest.raises((ValueError, OSError, RuntimeError)):
        workflows.build_pretraining(
            matrix,
            run_dir,
            model_path,
            oracle_path,
            workers=1,
        )

    assert not model_path.exists()
    assert not oracle_path.exists()
    assert not model_path.with_name(model_path.name + ".partial").exists()
    assert not oracle_path.with_name(oracle_path.name + ".partial").exists()


@pytest.mark.parametrize("failure", ["all-empty", "one-empty", "sequence-gap"])
def test_pretraining_rejects_missing_or_gapped_job_samples(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    import dblbt_fcn.workflows as workflows

    matrix = tiny_matrix(
        policies=["pretrain_arm"],
        seeds=[1103, 2207, 3301],
        arm_ids=list(range(24)),
        rounds=2,
        name="tiny-pretrain",
    )
    run_dir = tmp_path / "runs"
    jobs = expand_matrix(matrix)
    for index, job in enumerate(jobs):
        paths = artifact_paths(job, run_dir)
        empty = failure == "all-empty" or (
            failure == "one-empty" and index == 0
        )
        gap = failure == "sequence-gap" and index == 0
        rows = [
            canonical_row(
                job,
                round_id,
                prior_decision=(not empty and round_id == 31),
                training_sample=(not empty and round_id == 63),
                local_sequence=1 if gap else 0,
            )
            for round_id in range(job.rounds)
        ]
        metadata = write_jsonl_gz(paths.raw, rows)
        now = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
        write_manifest(
            paths.manifest,
            RunManifest(
                run_id=job.run_id,
                scenario_id=job.scenario.id,
                policy=job.policy,
                seed=job.seed,
                config_hash=job.config_hash,
                git_revision="a" * 40,
                dependency_versions={"python": "3.12"},
                host="test-worker",
                started_at_utc=now,
                ended_at_utc=now,
                elapsed_seconds=0.0,
                record_path=str(paths.raw),
                record_hash=metadata.sha256,
                row_count=metadata.row_count,
                exit_code=0,
                status="complete",
                execution_provenance=execution_provenance(job),
            ),
        )
    model_path = tmp_path / "model.npz"
    oracle_path = tmp_path / "oracle.json"
    monkeypatch.setattr(workflows, "run_sweep", lambda *args, **kwargs: [])

    with pytest.raises(ValueError, match="sample|sequence"):
        workflows.build_pretraining(
            matrix,
            run_dir,
            model_path,
            oracle_path,
            workers=1,
        )

    assert not model_path.exists()
    assert not oracle_path.exists()


def test_summary_is_numeric_sorted_and_byte_stable(tmp_path: Path) -> None:
    from dblbt_fcn.reporting import summarize_manifests

    matrix = tiny_matrix(
        policies=["random_lbt", "tmc_db_lbt"], rounds=2
    )
    run_dir = tmp_path / "runs"
    jobs = expand_matrix(matrix)
    for job in reversed(jobs):
        write_complete_job(run_dir, job)
        write_job_config(run_dir, job)
    output = tmp_path / "summary.csv"

    summarize_manifests(run_dir / "manifests", output)
    first = output.read_bytes()
    summarize_manifests(run_dir / "manifests", output)

    assert output.read_bytes() == first
    rows = list(csv.DictReader(output.read_text(encoding="ascii").splitlines()))
    assert [row["run_id"] for row in rows] == sorted(job.run_id for job in jobs)
    assert set(rows[0]) == {
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
    }
    assert rows[0]["rounds"] == "2"
    assert rows[0]["elapsed_us"] == "4000"
    assert rows[0]["successes"] == "1"
    assert rows[0]["collisions"] == "1"
    assert rows[0]["collision_probability"] == "0.5"
    assert rows[0]["effective_airtime"] == "0.5"
    assert rows[0]["mean_delay_us"] == "100.0"
    assert rows[0]["p95_delay_us"] == "100.0"
    assert rows[0]["jain_fairness"] == "0.5"
    assert rows[0]["evaluation_utility"] == "0.5416"
    assert rows[0]["decision_count"] == "0"
    assert rows[0]["switch_count"] == "0"
    assert rows[0]["training_sample_count"] == "0"


def test_parallel_summary_is_byte_identical_to_sequential(tmp_path: Path) -> None:
    from dblbt_fcn.reporting import summarize_manifests

    matrix = tiny_matrix(
        policies=["random_lbt", "tmc_db_lbt"], rounds=4
    )
    run_dir = tmp_path / "runs"
    jobs = expand_matrix(matrix)
    for job in reversed(jobs):
        write_complete_job(run_dir, job)
        write_job_config(run_dir, job)
    sequential = tmp_path / "sequential.csv"
    parallel = tmp_path / "parallel.csv"

    sequential_rows = summarize_manifests(
        run_dir / "manifests", sequential, workers=1
    )
    parallel_rows = summarize_manifests(
        run_dir / "manifests", parallel, workers=2
    )

    assert parallel_rows == sequential_rows
    assert parallel.read_bytes() == sequential.read_bytes()


def test_summary_rejects_invalid_worker_count_without_output(
    tmp_path: Path,
) -> None:
    from dblbt_fcn.reporting import summarize_manifests

    job = expand_matrix(tiny_matrix(rounds=1))[0]
    run_dir = tmp_path / "runs"
    write_complete_job(run_dir, job)
    write_job_config(run_dir, job)
    output = tmp_path / "summary.csv"

    with pytest.raises(ValueError, match="workers"):
        summarize_manifests(
            run_dir / "manifests", output, workers=0
        )

    assert not output.exists()


@pytest.mark.parametrize("failure", ["corrupt", "duplicate", "nan"])
def test_summary_rejects_corrupt_duplicate_or_nonfinite_input_without_output(
    tmp_path: Path, failure: str
) -> None:
    from dblbt_fcn.reporting import summarize_manifests

    matrix = tiny_matrix(rounds=1)
    job = expand_matrix(matrix)[0]
    run_dir = tmp_path / "runs"
    write_complete_job(run_dir, job)
    write_job_config(run_dir, job)
    paths = artifact_paths(job, run_dir)
    if failure == "corrupt":
        paths.raw.write_bytes(b"broken")
    elif failure == "duplicate":
        duplicate = paths.manifest.with_name(f"copy-{paths.manifest.name}")
        duplicate.write_bytes(paths.manifest.read_bytes())
    else:
        payload = json.loads(paths.manifest.read_text(encoding="utf-8"))
        payload["elapsed_seconds"] = float("nan")
        paths.manifest.write_text(
            json.dumps(payload, allow_nan=True), encoding="ascii"
        )
    output = tmp_path / "summary.csv"

    with pytest.raises((ValueError, OSError, RuntimeError)):
        summarize_manifests(run_dir / "manifests", output)

    assert not output.exists()
    assert not output.with_name(output.name + ".partial").exists()


def test_summary_refuses_to_overwrite_raw_input(tmp_path: Path) -> None:
    from dblbt_fcn.reporting import summarize_manifests

    job = expand_matrix(tiny_matrix(rounds=1))[0]
    run_dir = tmp_path / "runs"
    write_complete_job(run_dir, job)
    write_job_config(run_dir, job)
    raw = artifact_paths(job, run_dir).raw
    before = raw.read_bytes()

    with pytest.raises(ValueError, match="raw"):
        summarize_manifests(run_dir / "manifests", raw)

    assert raw.read_bytes() == before


@pytest.mark.parametrize(
    "target_kind", ["manifest", "config", "marker", "raw-dir-new", "alias"]
)
def test_pretraining_outputs_cannot_enter_or_alias_input_artifact_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_kind: str,
) -> None:
    import dblbt_fcn.workflows as workflows

    matrix = tiny_matrix(
        policies=["pretrain_arm"],
        seeds=[1103, 2207, 3301],
        arm_ids=list(range(24)),
        rounds=64,
        name="protected-pretrain",
    )
    run_dir = tmp_path / "runs"
    jobs = expand_matrix(matrix)
    for job in jobs:
        write_complete_job(run_dir, job)
        write_job_config(run_dir, job)
    first = artifact_paths(jobs[0], run_dir)
    protected = {
        path: path.read_bytes()
        for path in (first.raw, first.marker, first.manifest)
    }
    config = run_dir / "configs" / f"{jobs[0].run_id}.json"
    protected[config] = config.read_bytes()
    oracle = tmp_path / "oracle.json"
    if target_kind == "manifest":
        model = first.manifest
    elif target_kind == "config":
        model = config
    elif target_kind == "marker":
        model = first.marker
    elif target_kind == "raw-dir-new":
        model = run_dir / "raw" / "new-model.npz"
    else:
        alias = tmp_path / "raw-alias"
        try:
            alias.symlink_to(run_dir / "raw", target_is_directory=True)
        except OSError:
            pytest.skip("symlink privilege unavailable")
        model = alias / "new-model.npz"
    monkeypatch.setattr(workflows, "run_sweep", lambda *args, **kwargs: [])

    with pytest.raises(ValueError, match="artifact|input|output|alias"):
        workflows.build_pretraining(
            matrix, run_dir, model, oracle, workers=1
        )

    assert {path: path.read_bytes() for path in protected} == protected
    assert all(job_is_complete(job, run_dir) for job in jobs)


def test_summary_rejects_noncanonical_job_sidecar(tmp_path: Path) -> None:
    from dblbt_fcn.reporting import summarize_manifests

    job = expand_matrix(tiny_matrix(rounds=1))[0]
    run_dir = tmp_path / "runs"
    write_complete_job(run_dir, job)
    config = run_dir / "configs" / f"{job.run_id}.json"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        json.dumps(job.model_dump(mode="json"), indent=2) + "\n",
        encoding="ascii",
    )
    output = tmp_path / "summary.csv"

    with pytest.raises(ValueError, match="canonical"):
        summarize_manifests(run_dir / "manifests", output)

    assert not output.exists()


@pytest.mark.skipif(os.name != "posix", reason="manifest recovery requires WSL/POSIX")
def test_sweep_reruns_noncanonical_manifest(tmp_path: Path) -> None:
    from dblbt_fcn.workflows import run_sweep

    matrix = tiny_matrix(rounds=2)
    job = expand_matrix(matrix)[0]
    output = tmp_path / "runs"
    first = run_sweep(matrix, output, workers=1)
    paths = artifact_paths(job, output)
    value = json.loads(paths.manifest.read_bytes())
    paths.manifest.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    rerun = run_sweep(matrix, output, workers=1)

    assert [item.run_id for item in first] == [job.run_id]
    assert [item.run_id for item in rerun] == [job.run_id]
    rerun_bytes = paths.manifest.read_bytes()
    rerun_value = json.loads(rerun_bytes)
    assert rerun_bytes == (
        json.dumps(
            rerun_value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    assert job_is_complete(job, output)
