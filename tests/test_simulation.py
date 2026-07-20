"""Canonical single-job simulation and artifact orchestration."""

from __future__ import annotations

import gzip
import json
import math
import os
from pathlib import Path
import tempfile
from concurrent.futures import ProcessPoolExecutor
import time

import numpy as np
import pytest

from dblbt_fcn.experiment import JobSpec, ScenarioSpec, TimingSpec
from dblbt_fcn.linucb import LinUCB


def make_job(
    policy: str = "random_lbt",
    *,
    rounds: int = 12,
    seed: int = 410,
    arm_id: int | None = None,
    ablation: str | None = None,
    **scenario_updates: object,
) -> JobSpec:
    scenario: dict[str, object] = {
        "id": "unit-2x2",
        "wifi_nodes": 2,
        "nru_nodes": 2,
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
    scenario.update(scenario_updates)
    values: dict[str, object] = {
        "matrix": "unit",
        "rounds": rounds,
        "alpha": 11,
        "timing": TimingSpec(),
        "scenario": ScenarioSpec.model_validate(scenario),
        "policy": policy,
        "seed": seed,
        "arm_id": arm_id,
        "ablation": ablation,
    }
    if policy == "pretrain_arm" and arm_id is None:
        values["arm_id"] = 7
    return JobSpec.model_validate(values)


def assert_finite_builtins(value: object) -> None:
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        assert math.isfinite(value)
        return
    if type(value) is list:
        for item in value:
            assert_finite_builtins(item)
        return
    if type(value) is dict:
        assert all(type(key) is str for key in value)
        for item in value.values():
            assert_finite_builtins(item)
        return
    pytest.fail(f"non-built-in record value: {type(value)!r}")


def _retry_run_job(job: JobSpec, output: str) -> str:
    from dblbt_fcn.simulation import run_job

    for _ in range(100):
        try:
            return run_job(job, output).run_id
        except BlockingIOError:
            time.sleep(0.01)
    raise RuntimeError("job lock did not become available")


def test_simulation_module_exposes_single_job_entry_points() -> None:
    from dblbt_fcn.simulation import run_job, simulate_job_records

    assert callable(simulate_job_records)
    assert callable(run_job)


@pytest.mark.parametrize(
    ("policy", "seed", "oracle_arm"),
    [
        ("random_lbt", 410, None),
        ("primary_db_lbt", 410, None),
        ("tmc_db_lbt", 410, None),
        ("adaptive_db_lbt", 410, None),
        ("fixed_oracle", 410, 7),
        ("pretrain_arm", 1103, None),
    ],
)
def test_small_jobs_emit_exact_finite_contention_round_rows(
    policy: str, seed: int, oracle_arm: int | None
) -> None:
    from dblbt_fcn.simulation import simulate_job_records

    job = make_job(policy, seed=seed)
    rows = list(simulate_job_records(job, oracle_arm=oracle_arm))

    assert len(rows) == job.rounds
    assert [row["round_id"] for row in rows] == list(range(job.rounds))
    for row in rows:
        assert_finite_builtins(row)
        json.dumps(row, allow_nan=False)
        assert row["schema_version"] == 1
        assert row["record_type"] == "contention_round"
        assert row["run_id"] == job.run_id
        assert row["scenario_id"] == job.scenario.id
        assert row["policy"] == policy
        assert row["seed"] == job.seed
        assert row["config_hash"] == job.config_hash
        assert row["kind"] in {"success", "collision"}
        assert row["round_end_us"] > row["tx_start_us"]
        assert type(row["decisions"]) is list
        assert type(row["training_samples"]) is list
        for sender in row["senders"]:
            assert set(sender) == {
                "node_id",
                "technology",
                "selected_backoff_before",
                "next_selected_backoff",
                "interruptions_before",
                "retries_after",
                "db_initialized",
                "deterministic_countdown",
                "delay_us",
                "effective_data_us",
            }


def test_same_job_repeats_records_and_paired_policies_share_initial_state() -> None:
    from dblbt_fcn.simulation import simulate_job_records

    random_rows = list(simulate_job_records(make_job("random_lbt")))
    repeated = list(simulate_job_records(make_job("random_lbt")))
    tmc_rows = list(simulate_job_records(make_job("tmc_db_lbt")))

    assert random_rows == repeated
    assert random_rows[0]["node_ids"] == tmc_rows[0]["node_ids"]
    assert [
        (sender["node_id"], sender["selected_backoff_before"])
        for sender in random_rows[0]["senders"]
    ] == [
        (sender["node_id"], sender["selected_backoff_before"])
        for sender in tmc_rows[0]["senders"]
    ]


def test_adaptive_decisions_use_legal_boundaries_and_preserve_initial_model() -> None:
    from dblbt_fcn.simulation import simulate_job_records

    initial = LinUCB(24, 11, ridge=1.0, exploration=0.5)
    before_A = initial.A.tobytes()
    before_b = initial.b.tobytes()
    rows = list(
        simulate_job_records(
            make_job("adaptive_db_lbt", rounds=96),
            initial_agent=initial,
        )
    )
    decisions = [decision for row in rows for decision in row["decisions"]]

    assert decisions
    assert all(decision["round_id"] % 32 == 0 for decision in decisions)
    assert all(0 <= decision["arm"] < 24 for decision in decisions)
    assert all(decision["profile"]["alpha"] == 11 for decision in decisions)
    assert all(len(decision["context"]) == 11 for decision in decisions)
    assert initial.A.tobytes() == before_A
    assert initial.b.tobytes() == before_b


def test_runner_reads_only_incremental_adaptive_decisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dblbt_fcn.adaptive import AdaptiveController
    from dblbt_fcn.simulation import simulate_job_records

    def forbid_full_history(controller: AdaptiveController):
        raise AssertionError("runner copied complete decision history")

    monkeypatch.setattr(
        AdaptiveController, "decisions", property(forbid_full_history)
    )

    rows = list(
        simulate_job_records(make_job("adaptive_db_lbt", rounds=64))
    )

    assert any(row["decisions"] for row in rows)


def test_pretrain_samples_use_previous_decision_context_and_exact_fields() -> None:
    from dblbt_fcn.simulation import simulate_job_records

    job = make_job("pretrain_arm", rounds=96, seed=1103, arm_id=7)
    rows = list(simulate_job_records(job))
    previous_context: dict[str, list[float]] = {}
    samples: list[dict[str, object]] = []
    for row in rows:
        row_samples = row["training_samples"]
        samples.extend(row_samples)
        for sample in row_samples:
            local_node_id = sample["node_id"].split(":", 1)[1]
            assert sample["context"] == previous_context[local_node_id]
        for decision in row["decisions"]:
            assert decision["arm"] == 7
            previous_context[decision["node_id"]] = decision["context"]

    assert samples
    assert all(
        set(sample)
        == {
            "context",
            "arm",
            "local_reward",
            "local_sequence",
            "node_id",
            "pretraining_seed",
        }
        for sample in samples
    )
    assert all(sample["arm"] == 7 for sample in samples)
    assert all(sample["pretraining_seed"] == 1103 for sample in samples)
    assert all(
        sample["node_id"].startswith(f"{job.run_id}:")
        for sample in samples
    )


def test_fixed_oracle_requires_arm_and_context_free_uses_ucb1_selector() -> None:
    from dblbt_fcn.simulation import simulate_job_records

    fixed = make_job("fixed_oracle", rounds=96)
    with pytest.raises(ValueError, match="oracle_arm"):
        list(simulate_job_records(fixed))

    fixed_rows = list(simulate_job_records(fixed, oracle_arm=9))
    fixed_arms = [
        decision["arm"]
        for row in fixed_rows
        for decision in row["decisions"]
    ]
    assert fixed_arms and set(fixed_arms) == {9}

    context_free_rows = list(
        simulate_job_records(
            make_job(
                "adaptive_db_lbt",
                rounds=96,
                ablation="context_free_ucb",
            )
        )
    )
    per_node: dict[str, list[int]] = {}
    for row in context_free_rows:
        for decision in row["decisions"]:
            per_node.setdefault(decision["node_id"], []).append(
                decision["arm"]
            )
    assert per_node
    assert all(arms[:3] == [0, 1, 2] for arms in per_node.values())


def test_no_queue_ablation_masks_only_declared_context_positions() -> None:
    from dblbt_fcn.simulation import simulate_job_records

    rows = list(
        simulate_job_records(
            make_job("adaptive_db_lbt", rounds=32, ablation="no_queue")
        )
    )
    contexts = [
        decision["context"] for row in rows for decision in row["decisions"]
    ]

    assert contexts
    assert all(
        np.asarray(context)[[6, 7]].tolist() == [0.0, 0.0]
        for context in contexts
    )


def test_poisson_queues_pause_and_reactivate_deterministically() -> None:
    from dblbt_fcn.simulation import simulate_job_records

    job = make_job(
        rounds=40,
        traffic="poisson",
        poisson_rate_packets_ms=0.02,
    )
    first = list(simulate_job_records(job))
    second = list(simulate_job_records(job))

    assert first == second
    assert len(first) == job.rounds
    assert any(
        len(row["backlogged_node_ids"]) < 4 for row in first
    )
    assert all(row["backlogged_node_ids"] for row in first)


def test_paired_poisson_policies_use_the_same_initial_countdowns() -> None:
    from dblbt_fcn.simulation import simulate_job_records

    scenario = {
        "rounds": 1,
        "traffic": "poisson",
        "poisson_rate_packets_ms": 0.02,
    }
    random_row = next(
        simulate_job_records(make_job("random_lbt", **scenario))
    )
    primary_row = next(
        simulate_job_records(make_job("primary_db_lbt", **scenario))
    )

    assert random_row["node_ids"] == primary_row["node_ids"]
    assert [
        (sender["node_id"], sender["selected_backoff_before"])
        for sender in random_row["senders"]
    ] == [
        (sender["node_id"], sender["selected_backoff_before"])
        for sender in primary_row["senders"]
    ]


def test_dynamic_windows_change_active_sets_without_losing_rounds() -> None:
    from dblbt_fcn.simulation import simulate_job_records

    rows = list(
        simulate_job_records(
            make_job(
                rounds=240,
                join_interval_rounds=10,
                lifetime_rounds=200,
            )
        )
    )

    assert len(rows) == 240
    assert rows[0]["active_node_ids"] == ["wifi-000"]
    assert rows[9]["active_node_ids"] == ["wifi-000"]
    assert rows[10]["active_node_ids"] == ["wifi-000", "wifi-001"]
    assert "wifi-000" not in rows[200]["active_node_ids"]
    assert rows[230]["active_node_ids"] == ["wifi-000"]


def test_dynamic_poisson_waits_for_real_source_arrival_before_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dblbt_fcn.simulation as simulation

    arrivals = iter((0, 0, 1, 0))
    elapsed_calls: list[int] = []

    def sequenced_arrivals(source: object, elapsed_us: int) -> int:
        elapsed_calls.append(elapsed_us)
        return next(arrivals)

    monkeypatch.setattr(
        simulation.PoissonTraffic, "arrivals", sequenced_arrivals
    )
    job = make_job(
        rounds=1,
        traffic="poisson",
        poisson_rate_packets_ms=0.02,
        join_interval_rounds=10,
        lifetime_rounds=200,
    )

    row = next(simulation.simulate_job_records(job))

    assert elapsed_calls[:3] == [job.timing.tx_us] * 3
    assert row["tx_start_us"] >= 3 * job.timing.tx_us
    assert row["kind"] == "success"


def test_dynamic_poisson_all_zero_source_hits_idle_limit_without_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dblbt_fcn.simulation as simulation

    monkeypatch.setattr(simulation, "_MAX_IDLE_TICKS", 3)
    monkeypatch.setattr(
        simulation.PoissonTraffic,
        "arrivals",
        lambda source, elapsed_us: 0,
    )
    job = make_job(
        rounds=1,
        traffic="poisson",
        poisson_rate_packets_ms=0.02,
        join_interval_rounds=10,
        lifetime_rounds=200,
    )

    with pytest.raises(RuntimeError, match="idle tick limit"):
        list(simulation.simulate_job_records(job))


def test_periodic_busy_is_recorded_between_rounds_without_extra_rows() -> None:
    from dblbt_fcn.simulation import simulate_job_records

    job = make_job(
        rounds=8,
        interference_interval_ms=1,
        interference_duration_us=100,
    )
    rows = list(simulate_job_records(job))

    assert [row["round_id"] for row in rows] == list(range(8))
    assert all(row["background_busy_us"] >= 100 for row in rows)
    assert all(
        row["round_end_us"] - row["tx_start_us"] == job.timing.tx_us
        for row in rows
    )


def test_interruption_perturbation_changes_only_local_observations() -> None:
    import random

    from dblbt_fcn.adaptive import AdaptiveController
    from dblbt_fcn.channel import Channel, Node
    from dblbt_fcn.nonideal import InterruptionPerturbation
    from dblbt_fcn.types import PolicyKind, Technology

    def run(sigma: float) -> tuple[list[object], list[int], int]:
        node = Node(
            "adaptive",
            Technology.WIFI,
            PolicyKind.ADAPTIVE,
            selected=0,
            remaining=0,
        )
        controller = AdaptiveController(
            Channel([node], seed=91),
            LinUCB(24, 11),
            interruption_perturbations={
                "adaptive": InterruptionPerturbation(
                    sigma, random.Random(91)
                )
            },
        )
        results = [controller.step() for _ in range(12)]
        assert controller.last_attempt("adaptive") is not None
        observed = [
            attempt.interruptions
            for attempt in controller.state("adaptive").window.attempts
        ]
        return results, observed, node.db_state.interruptions

    control_results, control_observed, control_real = run(0.0)
    perturbed_results, perturbed_observed, perturbed_real = run(2.0)
    repeated_results, repeated_observed, repeated_real = run(2.0)

    assert perturbed_results == control_results == repeated_results
    assert perturbed_real == control_real == repeated_real
    assert perturbed_observed != control_observed
    assert perturbed_observed == repeated_observed


def test_controller_last_attempt_is_lightweight_immutable_observation() -> None:
    from dblbt_fcn.adaptive import AdaptiveController
    from dblbt_fcn.channel import Channel, Node
    from dblbt_fcn.observation import AttemptRecord
    from dblbt_fcn.types import PolicyKind, Technology

    node = Node(
        "adaptive",
        Technology.WIFI,
        PolicyKind.ADAPTIVE,
        selected=0,
        remaining=0,
    )
    controller = AdaptiveController(Channel([node], seed=7), LinUCB(24, 11))

    assert controller.last_attempt("adaptive") is None
    controller.step()

    attempt = controller.last_attempt("adaptive")
    assert isinstance(attempt, AttemptRecord)
    assert attempt is controller.last_attempt("adaptive")
    with pytest.raises(ValueError, match="unknown adaptive"):
        controller.last_attempt("missing")


def test_run_job_writes_complete_artifacts_and_valid_resume_is_byte_stable(
    tmp_path: Path,
) -> None:
    from dblbt_fcn.experiment import artifact_paths, job_is_complete
    from dblbt_fcn.io import RunManifest, validate_jsonl_gz
    from dblbt_fcn.simulation import run_job

    job = make_job(rounds=10)
    paths = artifact_paths(job, tmp_path)
    first = run_job(job, tmp_path)

    assert isinstance(first, RunManifest)
    assert first.status == "complete"
    assert first.run_id == job.run_id
    assert first.config_hash == job.config_hash
    assert first.row_count == job.rounds
    assert first.record_path == str(paths.raw)
    assert paths.manifest.is_file()
    assert paths.raw.is_file()
    assert paths.marker.is_file()
    config_path = tmp_path / "configs" / f"{job.run_id}.json"
    assert config_path.is_file()
    assert json.loads(config_path.read_text(encoding="ascii")) == (
        job.model_dump(mode="json")
    )
    assert validate_jsonl_gz(paths.raw, require_marker=True).row_count == 10
    assert job_is_complete(job, tmp_path)

    before = {
        path: path.read_bytes()
        for path in (paths.raw, paths.marker, paths.manifest, config_path)
    }
    second = run_job(job, tmp_path)

    assert second == first
    assert all(path.read_bytes() == payload for path, payload in before.items())


def test_run_job_rejects_mismatched_existing_config_sidecar(
    tmp_path: Path,
) -> None:
    from dblbt_fcn.simulation import run_job

    job = make_job(rounds=2)
    config_path = tmp_path / "configs" / f"{job.run_id}.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("{}\n", encoding="ascii")

    with pytest.raises(ValueError, match="config sidecar"):
        run_job(job, tmp_path)

    assert config_path.read_bytes() == b"{}\n"


def test_run_job_reruns_when_adaptive_initial_state_changes(
    tmp_path: Path,
) -> None:
    from dblbt_fcn.experiment import artifact_paths
    from dblbt_fcn.simulation import run_job

    job = make_job("adaptive_db_lbt", rounds=64)
    first_agent = LinUCB(24, 11, action_grid_hash="a" * 64)
    second_agent = LinUCB(24, 11, action_grid_hash="a" * 64)
    second_agent.update(7, [1.0] * 11, 0.75)

    first = run_job(job, tmp_path, initial_agent=first_agent)
    paths = artifact_paths(job, tmp_path)
    first_raw = paths.raw.read_bytes()
    second = run_job(job, tmp_path, initial_agent=second_agent)

    assert first.execution_fingerprint != second.execution_fingerprint
    assert paths.raw.read_bytes() != first_raw


@pytest.mark.skipif(os.name != "posix", reason="resume cleanup requires WSL/POSIX")
@pytest.mark.parametrize("forgery", ["agent", "mode", "model"])
def test_run_job_rejects_execution_that_does_not_match_actual_inputs(
    tmp_path: Path, forgery: str
) -> None:
    from dblbt_fcn.experiment import artifact_paths
    from dblbt_fcn.provenance import execution_provenance
    from dblbt_fcn.simulation import run_job

    job = make_job("adaptive_db_lbt", rounds=2)
    first_agent = LinUCB(24, 11, action_grid_hash="a" * 64)
    second_agent = LinUCB(24, 11, action_grid_hash="a" * 64)
    second_agent.update(7, [1.0] * 11, 0.75)
    first_model = tmp_path / "first.npz"
    second_model = tmp_path / "second.npz"
    first_agent.save(first_model)
    second_model.write_bytes(first_model.read_bytes() + b"different-model")
    execution = execution_provenance(
        job, initial_agent=first_agent, model_path=first_model
    )
    if forgery == "mode":
        execution = execution.model_copy(
            update={"mode": "adaptive_blank", "fingerprint": ""}
        )
    first = run_job(
        job,
        tmp_path,
        initial_agent=first_agent,
        model_path=first_model,
        execution=(
            execution_provenance(
                job, initial_agent=first_agent, model_path=first_model
            )
            if forgery == "mode"
            else execution
        ),
    )
    paths = artifact_paths(job, tmp_path)
    before = {
        path: path.read_bytes()
        for path in (paths.raw, paths.marker, paths.manifest)
    }

    with pytest.raises(ValueError, match="execution provenance"):
        run_job(
            job,
            tmp_path,
            initial_agent=(second_agent if forgery == "agent" else first_agent),
            model_path=(second_model if forgery == "model" else first_model),
            execution=execution,
        )

    assert all(path.read_bytes() == payload for path, payload in before.items())
    assert first.execution_provenance == execution_provenance(
        job, initial_agent=first_agent, model_path=first_model
    )


@pytest.mark.skipif(os.name != "posix", reason="resume cleanup requires WSL/POSIX")
def test_run_job_model_file_bytes_participate_in_execution_identity(
    tmp_path: Path,
) -> None:
    from dblbt_fcn.simulation import run_job

    job = make_job("adaptive_db_lbt", rounds=2)
    agent = LinUCB(24, 11, action_grid_hash="a" * 64)
    first_model = tmp_path / "first.npz"
    second_model = tmp_path / "second.npz"
    agent.save(first_model)
    second_model.write_bytes(first_model.read_bytes() + b"trailing-provenance")

    first = run_job(
        job, tmp_path / "runs", initial_agent=agent, model_path=first_model
    )
    second = run_job(
        job, tmp_path / "runs", initial_agent=agent, model_path=second_model
    )

    assert first.execution_fingerprint != second.execution_fingerprint
    assert first.execution_provenance.agent_state_sha256 == (
        second.execution_provenance.agent_state_sha256
    )


@pytest.mark.skipif(os.name != "posix", reason="sidecar no-follow requires WSL/POSIX")
def test_run_job_refuses_configs_symlink_outside_root(tmp_path: Path) -> None:
    from dblbt_fcn.simulation import run_job

    job = make_job(rounds=2)
    root = tmp_path / "runs"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "configs").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="config|symlink|directory"):
        run_job(job, root)

    assert list(outside.iterdir()) == []


@pytest.mark.skipif(os.name != "posix", reason="sidecar concurrency requires WSL/POSIX")
def test_concurrent_same_job_sidecar_is_byte_stable(tmp_path: Path) -> None:
    from dblbt_fcn.experiment import canonical_json

    job = make_job(rounds=2)
    root = tmp_path / "runs"

    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(_retry_run_job, job, str(root))
            for _ in range(2)
        ]
        run_ids = [future.result() for future in futures]

    config = root / "configs" / f"{job.run_id}.json"
    assert run_ids == [job.run_id, job.run_id]
    assert config.read_bytes() == (canonical_json(job) + "\n").encode("ascii")
    assert list(config.parent.glob("*.partial")) == []


@pytest.mark.skipif(
    not Path("/mnt/d").exists(),
    reason="portable Windows/WSL path test requires /mnt/d",
)
def test_run_job_returns_portable_completed_manifest_without_foreign_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dblbt_fcn.io as io_module
    import dblbt_fcn.simulation as simulation
    from dblbt_fcn.experiment import artifact_paths, job_is_complete

    repository_root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(
        prefix="portable-resume-", dir=repository_root
    ) as directory:
        output_dir = Path(directory)
        job = make_job(rounds=2)
        simulation.run_job(job, output_dir)
        paths = artifact_paths(job, output_dir)
        manifest_value = json.loads(paths.manifest.read_text(encoding="utf-8"))
        assert paths.raw.parts[:3] == ("/", "mnt", "d")
        manifest_value["record_path"] = "D:\\" + "\\".join(
            paths.raw.parts[3:]
        )
        paths.manifest.write_text(
            json.dumps(
                manifest_value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        assert job_is_complete(job, output_dir)
        before = {
            path: path.read_bytes()
            for path in (paths.raw, paths.marker, paths.manifest)
        }

        real_validate = io_module.validate_jsonl_gz
        validated_paths: list[Path] = []

        def validate_expected_path(
            path: str | Path, *args: object, **kwargs: object
        ):
            native_path = Path(path).resolve(strict=False)
            validated_paths.append(native_path)
            assert native_path == paths.raw
            return real_validate(path, *args, **kwargs)

        monkeypatch.setattr(
            io_module, "validate_jsonl_gz", validate_expected_path
        )

        resumed = simulation.run_job(job, output_dir)

        assert resumed.record_path == str(paths.raw)
        assert validated_paths
        assert all(
            path.read_bytes() == payload for path, payload in before.items()
        )


def test_run_job_raw_bytes_match_across_output_directories(
    tmp_path: Path,
) -> None:
    from dblbt_fcn.experiment import artifact_paths
    from dblbt_fcn.simulation import run_job

    job = make_job(rounds=15)
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"

    run_job(job, first_root)
    run_job(job, second_root)

    assert artifact_paths(job, first_root).raw.read_bytes() == artifact_paths(
        job, second_root
    ).raw.read_bytes()


def test_run_job_clears_corrupt_expected_artifacts_and_reruns(
    tmp_path: Path,
) -> None:
    from dblbt_fcn.experiment import artifact_paths, job_is_complete
    from dblbt_fcn.simulation import run_job

    job = make_job(rounds=9)
    run_job(job, tmp_path)
    paths = artifact_paths(job, tmp_path)
    expected_raw = paths.raw.read_bytes()
    paths.raw.write_bytes(expected_raw[:8])
    assert not job_is_complete(job, tmp_path)

    rerun = run_job(job, tmp_path)

    assert rerun.status == "complete"
    assert paths.raw.read_bytes() == expected_raw
    assert job_is_complete(job, tmp_path)


def test_run_job_rejects_semantically_truncated_complete_artifacts(
    tmp_path: Path,
) -> None:
    from dblbt_fcn.experiment import artifact_paths, job_is_complete
    from dblbt_fcn.io import write_jsonl_gz, write_manifest
    from dblbt_fcn.simulation import run_job

    job = make_job(rounds=2)
    original = run_job(job, tmp_path)
    paths = artifact_paths(job, tmp_path)
    with gzip.open(paths.raw, "rt", encoding="utf-8") as handle:
        first_row = json.loads(next(handle))

    paths.manifest.unlink()
    paths.marker.unlink()
    paths.raw.unlink()
    metadata = write_jsonl_gz(paths.raw, [first_row])
    truncated = original.model_copy(
        update={
            "record_hash": metadata.sha256,
            "row_count": metadata.row_count,
        }
    )
    write_manifest(paths.manifest, truncated)

    assert not job_is_complete(job, tmp_path)

    rerun = run_job(job, tmp_path)
    assert rerun.row_count == job.rounds
    assert job_is_complete(job, tmp_path)


def test_run_job_uses_writer_lock_and_fault_cleanup_leaves_no_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from contextlib import contextmanager

    import dblbt_fcn.simulation as simulation
    from dblbt_fcn.experiment import artifact_paths, job_is_complete

    job = make_job(rounds=5)
    paths = artifact_paths(job, tmp_path)
    real_lock = simulation.job_artifact_lock
    lock_entries = 0

    @contextmanager
    def tracking_lock(job_value: JobSpec, output_value: Path):
        nonlocal lock_entries
        lock_entries += 1
        with real_lock(job_value, output_value) as root_fd:
            yield root_fd

    def fail_manifest(*args: object, **kwargs: object) -> None:
        raise OSError("manifest fault")

    monkeypatch.setattr(simulation, "job_artifact_lock", tracking_lock)
    monkeypatch.setattr(simulation, "write_manifest", fail_manifest)

    with pytest.raises(OSError, match="manifest fault"):
        simulation.run_job(job, tmp_path)

    assert lock_entries == 2
    assert not job_is_complete(job, tmp_path)
    assert not paths.manifest.exists()
    assert not paths.raw.exists()
    assert not paths.marker.exists()


def test_run_job_provenance_failure_never_installs_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dblbt_fcn.simulation as simulation
    from dblbt_fcn.experiment import artifact_paths

    job = make_job(rounds=5)
    paths = artifact_paths(job, tmp_path)

    def fail_git() -> str:
        raise RuntimeError("git provenance fault")

    monkeypatch.setattr(simulation, "_git_revision", fail_git)

    with pytest.raises(RuntimeError, match="git provenance fault"):
        simulation.run_job(job, tmp_path)

    assert not paths.manifest.exists()
    assert not paths.raw.exists()
    assert not paths.marker.exists()
