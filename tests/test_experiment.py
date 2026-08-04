"""Deterministic experiment matrices, identities, and resume behavior."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

import dblbt_fcn.experiment as experiment_module
import dblbt_fcn.io as io_module
from dblbt_fcn.experiment import (
    JobSpec,
    MatrixSpec,
    ScenarioSpec,
    artifact_paths,
    canonical_json,
    clear_invalid_job_artifacts,
    derive_stream_seed,
    expand_matrix,
    job_is_complete,
    jobs_to_run,
    load_matrix,
    pair_key,
    paired_exogenous_seed,
)
from dblbt_fcn.io import (
    RunManifest,
    completion_marker_path,
    write_jsonl_gz,
    write_manifest,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MATRIX_DIRECTORY = REPOSITORY_ROOT / "configs" / "matrices"
FORMAL_SEEDS = {410, 523, 631, 742, 859, 967, 1081, 1193, 1307, 1429}


def symlink_or_skip(
    link: Path, target: Path, *, target_is_directory: bool = False
) -> None:
    try:
        link.symlink_to(target, target_is_directory=target_is_directory)
    except OSError as error:
        if os.name == "nt" and getattr(error, "winerror", None) == 1314:
            pytest.skip("Windows symlink privilege is unavailable")
        raise


def scenario_values(**updates: object) -> dict[str, object]:
    values: dict[str, object] = {
        "id": "static-2x2",
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
    values.update(updates)
    return values


def matrix_values(**updates: object) -> dict[str, object]:
    values: dict[str, object] = {
        "version": 1,
        "name": "unit",
        "rounds": 100,
        "alpha": 11,
        "timing": {
            "slot_us": 1,
            "tx_us": 2000,
            "wifi_ack_us": 0,
            "nru_sync_us": 250,
        },
        "seeds": [7],
        "policies": ["random_lbt"],
        "conditions": [],
        "arm_ids": [],
        "scenarios": [scenario_values()],
    }
    values.update(updates)
    return values


def one_job() -> JobSpec:
    return expand_matrix(MatrixSpec.model_validate(matrix_values()))[0]


def write_complete_job(tmp_path: Path, job: JobSpec) -> None:
    paths = artifact_paths(job, tmp_path)
    metadata = write_jsonl_gz(
        paths.raw, ({"round": round_id} for round_id in range(job.rounds))
    )
    now = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    manifest = RunManifest(
        run_id=job.run_id,
        scenario_id=job.scenario.id,
        policy=job.policy,
        seed=job.seed,
        config_hash=job.config_hash,
        git_revision="abc123",
        dependency_versions={"dblbt-fcn": "0.1.0"},
        host="worker",
        started_at_utc=now,
        ended_at_utc=now,
        elapsed_seconds=0.0,
        record_path=str(paths.raw),
        record_hash=metadata.sha256,
        row_count=metadata.row_count,
        exit_code=0,
        status="complete",
    )
    write_manifest(paths.manifest, manifest)


def test_models_are_strict_immutable_and_forbid_unknown_fields() -> None:
    matrix = MatrixSpec.model_validate(matrix_values())

    with pytest.raises(ValidationError):
        matrix.rounds = 5
    with pytest.raises(ValidationError):
        MatrixSpec.model_validate({**matrix_values(), "unknown": True})
    with pytest.raises(ValidationError):
        ScenarioSpec.model_validate({**scenario_values(), "unknown": True})
    with pytest.raises(ValidationError):
        MatrixSpec.model_validate({**matrix_values(), "rounds": "100"})
    with pytest.raises(ValidationError):
        ScenarioSpec.model_validate({**scenario_values(), "trace": 0})


def test_job_is_immutable_and_excludes_execution_environment_fields() -> None:
    job = one_job()

    assert {"output_dir", "host", "started_at_utc"}.isdisjoint(
        JobSpec.model_fields
    )
    with pytest.raises(ValidationError):
        job.policy = "tmc_db_lbt"
    with pytest.raises(ValidationError):
        JobSpec.model_validate({**job.model_dump(), "output_dir": "results"})


def test_job_rejects_ambiguous_policy_dimensions() -> None:
    payload = one_job().model_dump()

    with pytest.raises(ValidationError):
        JobSpec.model_validate(
            {
                **payload,
                "policy": "pretrain_arm",
                "arm_id": 3,
                "ablation": "no_queue",
            }
        )


@pytest.mark.parametrize(
    ("target", "updates"),
    [
        ("matrix", {"policies": ["random_lbtt"]}),
        (
            "matrix",
            {
                "policies": ["adaptive_db_lbt"],
                "conditions": ["no_queu"],
            },
        ),
        ("job", {"policy": "random_lbtt"}),
        (
            "job",
            {"policy": "adaptive_db_lbt", "ablation": "no_queu"},
        ),
    ],
)
def test_policy_and_ablation_vocabulary_rejects_typos(
    target: str, updates: dict[str, object]
) -> None:
    with pytest.raises(ValidationError):
        if target == "matrix":
            MatrixSpec.model_validate(matrix_values(**updates))
        else:
            JobSpec.model_validate({**one_job().model_dump(), **updates})


def test_load_matrix_rejects_unknown_and_duplicate_yaml_keys(
    tmp_path: Path,
) -> None:
    unknown = tmp_path / "unknown.yaml"
    unknown.write_text(
        (MATRIX_DIRECTORY / "smoke.yaml").read_text(encoding="utf-8")
        + "unknown: true\n",
        encoding="utf-8",
    )
    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text(
        "version: 1\nversion: 1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_matrix(unknown)
    with pytest.raises(ValueError, match="duplicate YAML mapping key"):
        load_matrix(duplicate)


@pytest.mark.parametrize(
    "updates",
    [
        {"seeds": []},
        {"seeds": [7, 7]},
        {"policies": []},
        {"policies": ["random_lbt", "random_lbt"]},
        {"conditions": ["full", "full"]},
        {"scenarios": [scenario_values(), scenario_values()]},
        {
            "scenarios": [
                scenario_values(),
                scenario_values(id="static-2x2", wifi_nodes=3),
            ]
        },
    ],
)
def test_matrix_rejects_empty_or_duplicate_dimensions(
    updates: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        MatrixSpec.model_validate(matrix_values(**updates))


@pytest.mark.parametrize(
    "updates",
    [
        {"wifi_nodes": -1},
        {
            "wifi_nodes": 0,
            "nru_nodes": 0,
            "legacy_ap_nodes": 0,
            "legacy_sta_nodes": 0,
        },
        {"traffic": "saturated", "poisson_rate_packets_ms": 0.02},
        {"traffic": "poisson", "poisson_rate_packets_ms": None},
        {"traffic": "poisson", "poisson_rate_packets_ms": float("nan")},
        {"traffic": "poisson", "poisson_rate_packets_ms": float("inf")},
        {"interference_interval_ms": 10},
        {"interference_duration_us": 2000},
        {"interruption_std": -0.1},
        {"interruption_std": float("nan")},
        {"interruption_std": float("inf")},
        {"join_interval_rounds": 10},
        {"lifetime_rounds": 200},
    ],
)
def test_scenario_rejects_invalid_ranges_and_field_combinations(
    updates: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        ScenarioSpec.model_validate(scenario_values(**updates))


def test_scenario_accepts_explicit_repeated_poisson_phases() -> None:
    scenario = ScenarioSpec.model_validate(
        scenario_values(
            traffic="poisson",
            poisson_rate_packets_ms=0.015,
            phases=[
                {
                    "id": "light",
                    "duration_rounds": 64,
                    "active_wifi_nodes": 1,
                    "active_nru_nodes": 1,
                    "poisson_rate_packets_ms": 0.015,
                },
                {
                    "id": "dense",
                    "duration_rounds": 96,
                    "active_wifi_nodes": 2,
                    "active_nru_nodes": 2,
                    "poisson_rate_packets_ms": 0.03,
                },
            ],
            phase_repetitions=3,
        )
    )

    assert [phase.id for phase in scenario.phases] == ["light", "dense"]
    assert [phase.duration_rounds for phase in scenario.phases] == [64, 96]
    assert scenario.phase_repetitions == 3


def test_unphased_scenario_keeps_legacy_canonical_identity() -> None:
    scenario = ScenarioSpec.model_validate(scenario_values())

    assert canonical_json(scenario) == (
        '{"id":"static-2x2","interference_duration_us":null,'
        '"interference_interval_ms":null,"interruption_std":0.0,'
        '"join_interval_rounds":null,"legacy_ap_nodes":0,'
        '"legacy_sta_nodes":0,"lifetime_rounds":null,"nru_nodes":2,'
        '"poisson_rate_packets_ms":null,"trace":false,'
        '"traffic":"saturated","wifi_nodes":2}'
    )


@pytest.mark.parametrize(
    "updates",
    [
        {
            "traffic": "poisson",
            "poisson_rate_packets_ms": 0.02,
            "phases": [
                {
                    "id": "empty",
                    "duration_rounds": 32,
                    "active_wifi_nodes": 0,
                    "active_nru_nodes": 0,
                    "poisson_rate_packets_ms": 0.02,
                }
            ],
        },
        {
            "traffic": "poisson",
            "poisson_rate_packets_ms": 0.02,
            "phases": [
                {
                    "id": "too-many-wifi",
                    "duration_rounds": 32,
                    "active_wifi_nodes": 3,
                    "active_nru_nodes": 1,
                    "poisson_rate_packets_ms": 0.02,
                }
            ],
        },
        {
            "traffic": "poisson",
            "poisson_rate_packets_ms": 0.02,
            "phases": [
                {
                    "id": "too-many-nru",
                    "duration_rounds": 32,
                    "active_wifi_nodes": 1,
                    "active_nru_nodes": 3,
                    "poisson_rate_packets_ms": 0.02,
                }
            ],
        },
        {
            "traffic": "poisson",
            "poisson_rate_packets_ms": 0.02,
            "phases": [
                {
                    "id": "same",
                    "duration_rounds": 32,
                    "active_wifi_nodes": 1,
                    "active_nru_nodes": 1,
                    "poisson_rate_packets_ms": 0.02,
                },
                {
                    "id": "same",
                    "duration_rounds": 64,
                    "active_wifi_nodes": 2,
                    "active_nru_nodes": 2,
                    "poisson_rate_packets_ms": 0.03,
                },
            ],
        },
        {
            "traffic": "poisson",
            "poisson_rate_packets_ms": 0.02,
            "join_interval_rounds": 10,
            "lifetime_rounds": 200,
            "phases": [
                {
                    "id": "mixed-schedulers",
                    "duration_rounds": 32,
                    "active_wifi_nodes": 1,
                    "active_nru_nodes": 1,
                    "poisson_rate_packets_ms": 0.02,
                }
            ],
        },
        {
            "traffic": "poisson",
            "poisson_rate_packets_ms": 0.02,
            "legacy_ap_nodes": 1,
            "phases": [
                {
                    "id": "legacy-ambiguous",
                    "duration_rounds": 32,
                    "active_wifi_nodes": 1,
                    "active_nru_nodes": 1,
                    "poisson_rate_packets_ms": 0.02,
                }
            ],
        },
        {"phase_repetitions": 2},
    ],
)
def test_scenario_rejects_invalid_phase_combinations(
    updates: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        ScenarioSpec.model_validate(scenario_values(**updates))


def test_phased_job_rounds_must_equal_the_complete_schedule() -> None:
    scenario = ScenarioSpec.model_validate(
        scenario_values(
            traffic="poisson",
            poisson_rate_packets_ms=0.02,
            phases=[
                {
                    "id": "light",
                    "duration_rounds": 32,
                    "active_wifi_nodes": 1,
                    "active_nru_nodes": 1,
                    "poisson_rate_packets_ms": 0.02,
                },
                {
                    "id": "dense",
                    "duration_rounds": 64,
                    "active_wifi_nodes": 2,
                    "active_nru_nodes": 2,
                    "poisson_rate_packets_ms": 0.03,
                },
            ],
            phase_repetitions=2,
        )
    )
    payload = {
        **one_job().model_dump(),
        "scenario": scenario,
        "rounds": 191,
    }

    with pytest.raises(ValidationError, match="phase schedule"):
        JobSpec.model_validate(payload)

    assert JobSpec.model_validate({**payload, "rounds": 192}).rounds == 192


def test_interruption_negative_zero_has_one_canonical_representation() -> None:
    positive = ScenarioSpec.model_validate(
        scenario_values(interruption_std=0.0)
    )
    negative = ScenarioSpec.model_validate(
        scenario_values(interruption_std=-0.0)
    )

    assert negative.interruption_std == 0.0
    assert canonical_json(negative) == canonical_json(positive)


def test_timing_defaults_are_frozen_and_explicit() -> None:
    values = matrix_values()
    del values["timing"]
    matrix = MatrixSpec.model_validate(values)

    assert matrix.timing.model_dump() == {
        "slot_us": 1,
        "tx_us": 2000,
        "wifi_ack_us": 0,
        "nru_sync_us": 250,
    }


def test_shared_alpha_is_explicit_common_and_fixed() -> None:
    for path in sorted(MATRIX_DIRECTORY.glob("*.yaml")):
        matrix = load_matrix(path)

        assert matrix.alpha == 11
        assert {job.alpha for job in expand_matrix(matrix)} == {11}

    with pytest.raises(ValidationError):
        MatrixSpec.model_validate(matrix_values(alpha=12))
    with pytest.raises(ValidationError):
        MatrixSpec.model_validate(matrix_values(alpha="11"))


@pytest.mark.parametrize(
    ("name", "scenario_count", "job_count", "rounds", "seeds"),
    [
        ("smoke", 1, 3, 1_000, {410}),
        ("reproduction", 9, 360, 100_000, FORMAL_SEEDS),
        ("pretrain", 11, 792, 100_000, {1103, 2207, 3301}),
        ("heldout", 10, 500, 100_000, FORMAL_SEEDS),
        ("ablation", 1, 80, 100_000, FORMAL_SEEDS),
        ("ns3-cross-validation", 3, 18, 100_000, {410, 523, 631}),
    ],
)
def test_approved_matrices_have_exact_sizes_and_seed_sets(
    name: str,
    scenario_count: int,
    job_count: int,
    rounds: int,
    seeds: set[int],
) -> None:
    matrix = load_matrix(MATRIX_DIRECTORY / f"{name}.yaml")
    jobs = expand_matrix(matrix)

    assert matrix.name == name
    assert matrix.rounds == rounds
    assert len(matrix.scenarios) == scenario_count
    assert set(matrix.seeds) == seeds
    assert len(jobs) == job_count


def test_ns3_cross_validation_matrix_matches_packet_level_scenarios() -> None:
    matrix = load_matrix(MATRIX_DIRECTORY / "ns3-cross-validation.yaml")
    scenarios = {scenario.id: scenario for scenario in matrix.scenarios}

    assert tuple(matrix.policies) == ("tmc_db_lbt", "adaptive_db_lbt")
    assert set(scenarios) == {
        "static-4x4",
        "dynamic-4x4",
        "nonideal-6x6-300ms",
    }
    assert (scenarios["static-4x4"].wifi_nodes, scenarios["static-4x4"].nru_nodes) == (4, 4)
    assert scenarios["dynamic-4x4"].join_interval_rounds == 10
    assert scenarios["dynamic-4x4"].lifetime_rounds == 200
    nonideal = scenarios["nonideal-6x6-300ms"]
    assert (nonideal.wifi_nodes, nonideal.nru_nodes) == (6, 6)
    assert nonideal.interference_interval_ms == 300
    assert nonideal.interference_duration_us == 2_000


def test_pretrain_crosses_every_scenario_seed_and_registered_arm() -> None:
    jobs = expand_matrix(load_matrix(MATRIX_DIRECTORY / "pretrain.yaml"))

    assert {job.policy for job in jobs} == {"pretrain_arm"}
    assert {job.arm_id for job in jobs} == set(range(24))
    assert all(job.ablation is None for job in jobs)
    combinations = {
        (job.scenario.id, job.seed, job.arm_id)
        for job in jobs
    }
    assert len(combinations) == 11 * 3 * 24


def test_ablation_conditions_map_to_adaptive_policy() -> None:
    jobs = expand_matrix(load_matrix(MATRIX_DIRECTORY / "ablation.yaml"))
    expected = {
        "full",
        "no_queue",
        "no_cca_interrupt",
        "no_delay",
        "frozen_online",
        "context_free_ucb",
        "collision_weight_0.125",
        "collision_weight_0.5",
    }

    assert {job.policy for job in jobs} == {"adaptive_db_lbt"}
    assert {job.ablation for job in jobs} == expected
    assert all(job.arm_id is None for job in jobs)


def test_restricted_profiles_condition_is_a_valid_adaptive_job() -> None:
    matrix = MatrixSpec.model_validate(
        matrix_values(
            policies=["adaptive_db_lbt"],
            conditions=["restricted_profiles"],
        )
    )

    jobs = expand_matrix(matrix)

    assert len(jobs) == 1
    assert jobs[0].policy == "adaptive_db_lbt"
    assert jobs[0].ablation == "restricted_profiles"


def test_pretrain_is_one_factor_at_a_time_around_static_4x4() -> None:
    matrix = load_matrix(MATRIX_DIRECTORY / "pretrain.yaml")
    scenarios = {scenario.id: scenario for scenario in matrix.scenarios}
    baseline = scenarios["topology-4x4"].model_dump()

    expected_changed_fields = {
        "topology-2x2": {"id", "wifi_nodes", "nru_nodes"},
        "topology-8x8": {"id", "wifi_nodes", "nru_nodes"},
        "poisson-0.02": {"id", "traffic", "poisson_rate_packets_ms"},
        "poisson-0.05": {"id", "traffic", "poisson_rate_packets_ms"},
        "poisson-0.08": {"id", "traffic", "poisson_rate_packets_ms"},
        "periodic-10ms": {
            "id",
            "interference_interval_ms",
            "interference_duration_us",
        },
        "periodic-100ms": {
            "id",
            "interference_interval_ms",
            "interference_duration_us",
        },
        "periodic-1000ms": {
            "id",
            "interference_interval_ms",
            "interference_duration_us",
        },
        "perturb-0.4": {"id", "interruption_std"},
        "perturb-1.0": {"id", "interruption_std"},
    }
    assert set(scenarios) == {"topology-4x4", *expected_changed_fields}
    for scenario_id, expected in expected_changed_fields.items():
        actual = scenarios[scenario_id].model_dump()
        changed = {key for key in baseline if baseline[key] != actual[key]}
        assert changed == expected


def test_heldout_dynamic_combines_all_registered_nonideal_factors() -> None:
    matrix = load_matrix(MATRIX_DIRECTORY / "heldout.yaml")
    scenario = next(
        item for item in matrix.scenarios if item.id == "dynamic-combined-4x4"
    )

    assert scenario.traffic == "poisson"
    assert scenario.poisson_rate_packets_ms == 0.065
    assert scenario.interference_interval_ms == 300
    assert scenario.interference_duration_us == 2000
    assert scenario.join_interval_rounds == 10
    assert scenario.lifetime_rounds == 200


def test_expand_matrix_returns_sorted_unique_run_ids() -> None:
    jobs = expand_matrix(load_matrix(MATRIX_DIRECTORY / "reproduction.yaml"))
    run_ids = [job.run_id for job in jobs]

    assert run_ids == sorted(run_ids)
    assert len(run_ids) == len(set(run_ids))


def test_duplicate_canonical_jobs_are_rejected_immediately() -> None:
    matrix = MatrixSpec.model_construct(
        **{
            **MatrixSpec.model_validate(matrix_values()).model_dump(),
            "policies": ("random_lbt", "random_lbt"),
        }
    )

    with pytest.raises(ValueError, match="duplicate canonical job"):
        expand_matrix(matrix)


def test_job_hash_and_id_use_independently_computed_canonical_json() -> None:
    job = one_job()
    payload = json.dumps(
        job.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    expected_hash = hashlib.sha256(payload.encode("ascii")).hexdigest()

    assert canonical_json(job) == payload
    assert job.config_hash == expected_hash
    assert job.run_id == expected_hash[:16]
    assert len(job.config_hash) == 64


def test_canonical_json_normalizes_signed_zero_recursively() -> None:
    class ZeroCarrier(BaseModel):
        value: float
        nested: tuple[float, ...]

    negative = {
        "mapping": {"value": -0.0},
        "list": [-0.0, {"model": ZeroCarrier(value=-0.0, nested=(-0.0,))}],
        "tuple": (-0.0, 1.5),
    }
    positive = {
        "mapping": {"value": 0.0},
        "list": [0.0, {"model": ZeroCarrier(value=0.0, nested=(0.0,))}],
        "tuple": (0.0, 1.5),
    }

    negative_json = canonical_json(negative)
    positive_json = canonical_json(positive)

    assert "-0.0" not in negative_json
    assert negative_json == positive_json
    assert hashlib.sha256(negative_json.encode("ascii")).digest() == (
        hashlib.sha256(positive_json.encode("ascii")).digest()
    )


@pytest.mark.parametrize(
    "value",
    [
        {1: -0.0},
        {-0.0: "value"},
        {"nested": [({2: "value"},)]},
    ],
)
def test_canonical_json_rejects_non_string_mapping_keys(
    value: object,
) -> None:
    with pytest.raises(
        ValueError, match="canonical JSON mapping keys must be exact strings"
    ):
        canonical_json(value)


@pytest.mark.parametrize(
    ("value", "error"),
    [
        ({"value": {1, 2}}, TypeError),
        ({"value": b"bytes"}, TypeError),
        ({"value": Decimal("1.0")}, TypeError),
        ({"value": float("nan")}, ValueError),
    ],
)
def test_canonical_json_keeps_rejecting_non_json_values(
    value: object, error: type[Exception]
) -> None:
    with pytest.raises(error):
        canonical_json(value)


def test_canonical_json_rejects_non_json_types_inside_models() -> None:
    class MappingCarrier(BaseModel):
        value: dict[int, str]

    class NestedMappingCarrier(BaseModel):
        value: list[dict[int, str]]

    class DecimalCarrier(BaseModel):
        value: Decimal

    class SetCarrier(BaseModel):
        value: set[int]

    invalid_models = [
        MappingCarrier(value={1: "one"}),
        NestedMappingCarrier(value=[{2: "two"}]),
        DecimalCarrier(value=Decimal("1.25")),
        SetCarrier(value={3, 4}),
    ]

    for model in invalid_models:
        with pytest.raises((TypeError, ValueError)):
            canonical_json(model)


def test_canonical_json_accepts_strict_model_keys_and_signed_zero() -> None:
    class LegalCarrier(BaseModel):
        value: dict[str, tuple[float, ...]]

    negative = LegalCarrier(value={"nested": (-0.0, 1.5)})
    positive = LegalCarrier(value={"nested": (0.0, 1.5)})

    assert canonical_json(negative) == canonical_json(positive)
    assert "-0.0" not in canonical_json(negative)


def test_registered_matrix_run_ids_remain_stable() -> None:
    expected = {
        "ablation": (
            80,
            "0204d3386a87d32e",
            "fbd7e611939423b9",
            "503fa6a5ff27ae5d1f3c39962d6af81a587b24305162dc6972888df652ff2125",
            "535ea88b2006891883d9901125eac22f391484c1a8c5555fc5a41cbf61de152a",
        ),
        "heldout": (
            500,
            "001cffbfae3b2569",
            "fe0d978ed6999f51",
            "5315159b4b3a791902e5b3ee3affd55f45184e40455c974587c9fa05bb189521",
            "8dbfac696398980a19d28665872b3cf6e40e5ddab94de9d6092c420f19354659",
        ),
        "pretrain": (
            792,
            "01607417dc6a3fb0",
            "fee3cd25d24e09a1",
            "9bbacbfa2a705efa36828fa7501e3a0c65e6b13ef3a0ab24ac50410557bd378d",
            "d3e63c9490062ae9365e4fadf848dd777e8444d3ba06aca849a1bbe189f4978f",
        ),
        "reproduction": (
            360,
            "019ac8e23675daac",
            "ff0172b97317a077",
            "400abe21a32693614b14e73d6c4356e021c43bd7d7c94c6c70cc789aecb3063a",
            "6f989ced40fc81c6241fae6cbd18571d1f57169192db50aa794b2080f2d9326d",
        ),
        "smoke": (
            3,
            "130dda82637fae04",
            "93ad32e3f4a03223",
            "f91dcd3d685640e25304722a89ec129a242c5ac282af94ad77478cdcaef1af80",
            "c6e6c94b097ad697c60b181a8c94807f44c9a5dac36a99d7d5e5fb69bf1521a9",
        ),
    }

    total_jobs = 0
    all_run_ids: list[str] = []
    for name, (
        count,
        first_run_id,
        last_run_id,
        expected_snapshot,
        expected_run_id_snapshot,
    ) in expected.items():
        jobs = expand_matrix(load_matrix(MATRIX_DIRECTORY / f"{name}.yaml"))
        payload = [job.model_dump(mode="json") for job in jobs]
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")

        assert len(jobs) == count
        assert jobs[0].run_id == first_run_id
        assert jobs[-1].run_id == last_run_id
        assert hashlib.sha256(encoded).hexdigest() == expected_snapshot
        run_id_snapshot = hashlib.sha256(
            json.dumps(
                [job.run_id for job in jobs],
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
        ).hexdigest()
        assert run_id_snapshot == expected_run_id_snapshot
        total_jobs += len(jobs)
        all_run_ids.extend(job.run_id for job in jobs)

    assert total_jobs == 1_735
    assert len(set(all_run_ids)) == 1_735
    global_run_id_snapshot = hashlib.sha256(
        json.dumps(
            sorted(all_run_ids),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    assert global_run_id_snapshot == (
        "094475c2083fe625634481a635bdf7a4581bec55a8339c9fe0561d874d76611b"
    )


def test_paired_policies_share_exogenous_seed_but_scenarios_and_seeds_do_not() -> None:
    jobs = expand_matrix(load_matrix(MATRIX_DIRECTORY / "smoke.yaml"))
    assert len({paired_exogenous_seed(job) for job in jobs}) == 1

    matrix = MatrixSpec.model_validate(
        matrix_values(
            seeds=[7, 8],
            policies=["random_lbt", "tmc_db_lbt"],
            scenarios=[scenario_values(), scenario_values(id="static-3x3", wifi_nodes=3, nru_nodes=3)],
        )
    )
    expanded = expand_matrix(matrix)
    by_pair = {
        (job.scenario.id, job.seed): paired_exogenous_seed(job)
        for job in expanded
    }
    assert len(by_pair) == 4
    assert len(set(by_pair.values())) == 4


def test_identical_scenario_and_seed_pair_across_matrices() -> None:
    heldout_jobs = expand_matrix(
        load_matrix(MATRIX_DIRECTORY / "heldout.yaml")
    )
    ablation_jobs = expand_matrix(
        load_matrix(MATRIX_DIRECTORY / "ablation.yaml")
    )
    heldout = next(
        job
        for job in heldout_jobs
        if job.scenario.id == "dynamic-combined-4x4"
        and job.seed == 410
        and job.policy == "adaptive_db_lbt"
    )
    ablation = next(
        job
        for job in ablation_jobs
        if job.scenario.id == "dynamic-combined-4x4"
        and job.seed == 410
        and job.ablation == "full"
    )
    different_seed = heldout.model_copy(update={"seed": 523})
    scenario_data = heldout.scenario.model_dump()
    scenario_data["interference_interval_ms"] = 301
    different_scenario = heldout.model_copy(
        update={"scenario": ScenarioSpec.model_validate(scenario_data)}
    )

    assert heldout.scenario.model_dump(mode="json") == (
        ablation.scenario.model_dump(mode="json")
    )
    assert heldout.matrix != ablation.matrix
    assert heldout.policy == ablation.policy
    assert heldout.ablation != ablation.ablation
    assert pair_key(heldout) == pair_key(ablation)
    assert paired_exogenous_seed(heldout) == paired_exogenous_seed(ablation)
    assert pair_key(different_seed) != pair_key(heldout)
    assert pair_key(different_scenario) != pair_key(heldout)


def test_pair_seed_excludes_policy_arm_and_ablation() -> None:
    base = one_job()
    payload = base.model_dump()
    variants = [
        JobSpec.model_validate({**payload, "policy": "tmc_db_lbt"}),
        JobSpec.model_validate(
            {**payload, "policy": "pretrain_arm", "arm_id": 3}
        ),
        JobSpec.model_validate(
            {
                **payload,
                "policy": "adaptive_db_lbt",
                "ablation": "no_queue",
            }
        ),
    ]

    assert all(
        paired_exogenous_seed(item) == paired_exogenous_seed(base)
        for item in variants
    )


def test_stream_seed_is_fixed_sha256_prefix_big_endian() -> None:
    payload = json.dumps(
        [410, 7, "backoff"],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    expected = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")

    assert expected == 4_092_240_447_075_252_226
    assert derive_stream_seed(410, 7, "backoff") == expected
    assert derive_stream_seed(410, 7, "backoff") == expected
    assert len(
        {
            derive_stream_seed(410, 7, "backoff"),
            derive_stream_seed(411, 7, "backoff"),
            derive_stream_seed(410, 8, "backoff"),
            derive_stream_seed(410, 7, "traffic"),
        }
    ) == 4


def test_stream_seed_accepts_stable_trimmed_string_node_identity() -> None:
    payload = json.dumps(
        [410, "w1", "backoff"],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    expected = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")

    assert expected == 17_010_646_038_799_652_591
    assert derive_stream_seed(410, "w1", "backoff") == expected
    assert derive_stream_seed(410, "w1", "backoff") != derive_stream_seed(
        410, "n1", "backoff"
    )


@pytest.mark.parametrize(
    "args",
    [
        (-1, 0, "stream"),
        (True, 0, "stream"),
        (1.0, 0, "stream"),
        (1, -1, "stream"),
        (1, True, "stream"),
        (1, "", "stream"),
        (1, " ", "stream"),
        (1, " w1", "stream"),
        (1, "w1 ", "stream"),
        (1, 1.0, "stream"),
        (1, 0, ""),
        (1, 0, " "),
        (1, 0, 3),
    ],
)
def test_stream_seed_rejects_ambiguous_or_coerced_inputs(
    args: tuple[object, object, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        derive_stream_seed(*args)  # type: ignore[arg-type]


def test_experiment_source_never_calls_builtin_hash() -> None:
    path = REPOSITORY_ROOT / "src" / "dblbt_fcn" / "experiment.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))

    forbidden = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "hash"
    ]
    assert forbidden == []


def test_artifact_paths_are_deterministic_under_output_directory(
    tmp_path: Path,
) -> None:
    job = one_job()
    paths = artifact_paths(job, tmp_path)

    assert paths.raw == tmp_path.resolve() / "raw" / f"{job.run_id}.jsonl.gz"
    assert paths.manifest == tmp_path.resolve() / "manifests" / f"{job.run_id}.json"
    assert paths.marker == completion_marker_path(paths.raw)
    assert paths.raw_partial.name == f"{job.run_id}.jsonl.gz.partial"
    assert paths.marker_partial.name == f"{job.run_id}.jsonl.gz.complete.partial"
    assert paths.manifest_partial.name == f"{job.run_id}.json.partial"
    assert paths.cleanup_lock == (
        tmp_path.resolve() / f".{job.run_id}.cleanup.lock"
    )


@pytest.mark.parametrize(
    ("declared", "expected", "equivalent"),
    [
        ("/tmp/results/raw/run.jsonl.gz", "/tmp/results/raw/run.jsonl.gz", True),
        (r"D:\results\raw\run.jsonl.gz", "/mnt/d/results/raw/run.jsonl.gz", True),
        ("/mnt/D/results/raw/run.jsonl.gz", r"d:\results\raw\run.jsonl.gz", True),
        ("results/raw/run.jsonl.gz", "/tmp/results/raw/run.jsonl.gz", False),
        ("/tmp/results/../raw/run.jsonl.gz", "/tmp/raw/run.jsonl.gz", False),
        (r"D:\results\..\raw\run.jsonl.gz", "/mnt/d/raw/run.jsonl.gz", False),
        ("/outside/raw/run.jsonl.gz", "/tmp/results/raw/run.jsonl.gz", False),
        (r"C:\results\raw\run.jsonl.gz", "/mnt/d/results/raw/run.jsonl.gz", False),
        ("/mnt/dd/results/raw/run.jsonl.gz", r"D:\results\raw\run.jsonl.gz", False),
        (r"D:\results\\raw\run.jsonl.gz", r"D:\results\raw\run.jsonl.gz", False),
        ("D:\\results\\raw\\run.jsonl.gz\\", r"D:\results\raw\run.jsonl.gz", False),
    ],
)
def test_portable_record_path_identity_is_strict_and_lexical(
    declared: str, expected: str, equivalent: bool
) -> None:
    assert (
        experiment_module._portable_record_paths_equal(declared, expected)
        is equivalent
    )


def test_valid_complete_job_is_skipped_and_never_cleared(tmp_path: Path) -> None:
    job = one_job()
    write_complete_job(tmp_path, job)
    paths = artifact_paths(job, tmp_path)

    assert job_is_complete(job, tmp_path)
    assert jobs_to_run([job], tmp_path) == []
    assert clear_invalid_job_artifacts(job, tmp_path) is False
    assert paths.raw.exists()
    assert paths.marker.exists()
    assert paths.manifest.exists()


def test_noncanonical_complete_manifest_is_invalid(tmp_path: Path) -> None:
    from dblbt_fcn.experiment import load_completed_job_manifest

    job = one_job()
    write_complete_job(tmp_path, job)
    paths = artifact_paths(job, tmp_path)
    value = json.loads(paths.manifest.read_text(encoding="utf-8"))
    paths.manifest.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="canonical"):
        load_completed_job_manifest(job, tmp_path)
    assert not job_is_complete(job, tmp_path)


def test_completed_manifest_loader_rejects_mismatched_job_identity(
    tmp_path: Path,
) -> None:
    from dblbt_fcn.experiment import load_completed_job_manifest

    job = one_job()
    write_complete_job(tmp_path, job)
    paths = artifact_paths(job, tmp_path)
    manifest_value = json.loads(paths.manifest.read_text(encoding="utf-8"))
    manifest_value["scenario_id"] = "different-scenario"
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

    with pytest.raises(ValueError, match="does not match job"):
        load_completed_job_manifest(job, tmp_path)


@pytest.mark.parametrize(
    "damage",
    ["missing", "partial", "record", "marker", "config", "failed"],
)
def test_invalid_or_incomplete_job_is_rerun_and_can_be_cleaned(
    tmp_path: Path,
    damage: str,
) -> None:
    job = one_job()
    paths = artifact_paths(job, tmp_path)
    if damage == "missing":
        pass
    elif damage == "partial":
        paths.raw_partial.parent.mkdir(parents=True)
        paths.raw_partial.write_bytes(b"partial")
        paths.manifest_partial.parent.mkdir(parents=True)
        paths.manifest_partial.write_bytes(b"partial")
    elif damage in {"record", "marker", "config"}:
        write_complete_job(tmp_path, job)
        if damage == "record":
            paths.raw.write_bytes(b"corrupt")
        elif damage == "marker":
            paths.marker.write_text("{}\n", encoding="utf-8")
        else:
            manifest_data = json.loads(paths.manifest.read_text(encoding="utf-8"))
            manifest_data["config_hash"] = "0" * 64
            paths.manifest.write_text(json.dumps(manifest_data), encoding="utf-8")
    else:
        now = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
        manifest = RunManifest(
            run_id=job.run_id,
            scenario_id=job.scenario.id,
            policy=job.policy,
            seed=job.seed,
            config_hash=job.config_hash,
            git_revision="abc123",
            dependency_versions={},
            host="worker",
            started_at_utc=now,
            ended_at_utc=now,
            elapsed_seconds=0.0,
            record_path=None,
            record_hash=None,
            row_count=None,
            exit_code=1,
            status="failed",
        )
        write_manifest(paths.manifest, manifest)

    assert not job_is_complete(job, tmp_path)
    assert jobs_to_run([job], tmp_path) == [job]
    if os.name == "nt":
        with pytest.raises(RuntimeError, match="requires WSL/POSIX"):
            clear_invalid_job_artifacts(job, tmp_path)
        return
    assert clear_invalid_job_artifacts(job, tmp_path) is True
    assert not any(
        path.exists()
        for path in (
            paths.raw,
            paths.marker,
            paths.manifest,
            paths.raw_partial,
            paths.marker_partial,
            paths.manifest_partial,
        )
    )


def test_jobs_to_run_preserves_run_id_order(tmp_path: Path) -> None:
    jobs = expand_matrix(
        MatrixSpec.model_validate(
            matrix_values(
                seeds=[7, 8],
                policies=["random_lbt", "tmc_db_lbt"],
            )
        )
    )
    write_complete_job(tmp_path, jobs[1])

    pending = jobs_to_run(reversed(jobs), tmp_path)

    assert pending == [job for job in jobs if job != jobs[1]]


def test_cleanup_ignores_untrusted_manifest_record_path(tmp_path: Path) -> None:
    job = one_job()
    paths = artifact_paths(job, tmp_path)
    outside = tmp_path.parent / f"outside-{job.run_id}.jsonl.gz"
    outside.write_bytes(b"must remain")
    paths.manifest.parent.mkdir(parents=True)
    paths.manifest.write_text(
        json.dumps(
            {
                "run_id": job.run_id,
                "scenario_id": job.scenario.id,
                "policy": job.policy,
                "seed": job.seed,
                "config_hash": job.config_hash,
                "git_revision": "abc123",
                "dependency_versions": {},
                "host": "worker",
                "started_at_utc": "2026-01-02T03:04:05Z",
                "ended_at_utc": "2026-01-02T03:04:05Z",
                "elapsed_seconds": 0.0,
                "record_path": str(outside),
                "record_hash": "0" * 64,
                "row_count": 0,
                "exit_code": 0,
                "status": "complete",
            }
        ),
        encoding="utf-8",
    )

    if os.name == "nt":
        with pytest.raises(RuntimeError, match="requires WSL/POSIX"):
            clear_invalid_job_artifacts(job, tmp_path)
    else:
        assert clear_invalid_job_artifacts(job, tmp_path) is True
    assert outside.read_bytes() == b"must remain"
    assert paths.manifest.exists() is (os.name == "nt")


def test_resume_rejects_external_record_path_before_manifest_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = one_job()
    paths = artifact_paths(job, tmp_path)
    outside = tmp_path.parent / f"outside-record-{job.run_id}.jsonl.gz"
    outside.write_bytes(b"must not be read")
    paths.manifest.parent.mkdir(parents=True)
    paths.manifest.write_text(
        json.dumps({"record_path": str(outside)}),
        encoding="utf-8",
    )
    called = False

    def unexpected_validation(*args: object, **kwargs: object) -> RunManifest:
        nonlocal called
        called = True
        raise RuntimeError("manifest validation must not run")

    monkeypatch.setattr(
        experiment_module.RunManifest,
        "model_validate_json",
        staticmethod(unexpected_validation),
    )

    assert not job_is_complete(job, tmp_path)
    assert not called


@pytest.mark.parametrize("operation", ["inspect", "cleanup"])
def test_resume_propagates_manifest_permission_errors_without_deleting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    job = one_job()
    write_complete_job(tmp_path, job)
    paths = artifact_paths(job, tmp_path)
    artifacts = (paths.raw, paths.marker, paths.manifest)
    original_read_bytes = Path.read_bytes
    before = {path: original_read_bytes(path) for path in artifacts}

    def deny_manifest_read(path: Path) -> bytes:
        if path == paths.manifest:
            raise PermissionError("simulated manifest permission failure")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", deny_manifest_read)

    with pytest.raises(PermissionError):
        if operation == "inspect":
            job_is_complete(job, tmp_path)
        else:
            clear_invalid_job_artifacts(job, tmp_path)

    assert {path: original_read_bytes(path) for path in artifacts} == before


@pytest.mark.parametrize("operation", ["inspect", "cleanup"])
@pytest.mark.parametrize("error_type", [RuntimeError, OSError])
def test_resume_propagates_operational_validation_errors_without_deleting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    error_type: type[Exception],
) -> None:
    job = one_job()
    write_complete_job(tmp_path, job)
    paths = artifact_paths(job, tmp_path)
    artifacts = (paths.raw, paths.marker, paths.manifest)
    before = {path: path.read_bytes() for path in artifacts}

    def fail_validation(*args: object, **kwargs: object) -> None:
        raise error_type("simulated record validation failure")

    monkeypatch.setattr(io_module, "validate_jsonl_gz", fail_validation)

    with pytest.raises(error_type):
        if operation == "inspect":
            job_is_complete(job, tmp_path)
        else:
            clear_invalid_job_artifacts(job, tmp_path)

    assert {path: path.read_bytes() for path in artifacts} == before


def test_resume_rejects_external_completion_marker_symlink(
    tmp_path: Path,
) -> None:
    job = one_job()
    write_complete_job(tmp_path, job)
    paths = artifact_paths(job, tmp_path)
    outside = tmp_path.parent / f"outside-marker-{job.run_id}"
    outside.write_bytes(paths.marker.read_bytes())
    paths.marker.unlink()
    symlink_or_skip(paths.marker, outside)

    assert not job_is_complete(job, tmp_path)


def test_cleanup_rejects_a_parent_symlink_outside_output(
    tmp_path: Path,
) -> None:
    job = one_job()
    paths = artifact_paths(job, tmp_path)
    outside = tmp_path.parent / f"outside-dir-{job.run_id}"
    outside.mkdir()
    sentinel = outside / "sentinel"
    sentinel.write_bytes(b"must remain")
    symlink_or_skip(
        paths.raw.parent, outside, target_is_directory=True
    )

    if os.name == "nt":
        with pytest.raises(RuntimeError, match="requires WSL/POSIX"):
            clear_invalid_job_artifacts(job, tmp_path)
    else:
        with pytest.raises(ValueError, match="escapes output directory"):
            clear_invalid_job_artifacts(job, tmp_path)

    assert sentinel.read_bytes() == b"must remain"


def test_cleanup_recovers_from_a_stale_lock_file(tmp_path: Path) -> None:
    if os.name != "posix":
        pytest.skip("stale lock recovery is a POSIX cleanup contract")
    job = one_job()
    paths = artifact_paths(job, tmp_path)
    paths.raw_partial.parent.mkdir(parents=True)
    paths.raw_partial.write_bytes(b"partial")
    paths.cleanup_lock.write_bytes(b"stale")

    assert clear_invalid_job_artifacts(job, tmp_path) is True

    assert not paths.raw_partial.exists()
    assert paths.cleanup_lock.read_bytes() == b"stale"


def test_writer_artifact_lock_excludes_cleanup(tmp_path: Path) -> None:
    if os.name != "posix":
        pytest.skip("shared artifact locks are a POSIX runtime contract")
    job = one_job()
    paths = artifact_paths(job, tmp_path)
    paths.raw_partial.parent.mkdir(parents=True)
    paths.raw_partial.write_bytes(b"preserve")

    with experiment_module.job_artifact_lock(job, tmp_path):
        assert paths.cleanup_lock.exists()
        with pytest.raises(BlockingIOError, match="artifact lock is already held"):
            clear_invalid_job_artifacts(job, tmp_path)

    assert paths.cleanup_lock.exists()
    assert paths.raw_partial.read_bytes() == b"preserve"


@pytest.mark.parametrize("operation", ["writer", "cleanup"])
def test_external_artifact_lock_blocks_until_owner_process_exits(
    tmp_path: Path, operation: str
) -> None:
    if os.name != "posix":
        pytest.skip("kernel artifact locks are a POSIX runtime contract")
    job = one_job()
    paths = artifact_paths(job, tmp_path)
    paths.raw_partial.parent.mkdir(parents=True)
    paths.raw_partial.write_bytes(b"preserve")
    script = """
import fcntl
import os
import sys

fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
print("locked", flush=True)
sys.stdin.read()
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script, os.fspath(paths.cleanup_lock)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        assert process.stdout.readline() == "locked\n"
        with pytest.raises(
            BlockingIOError, match="artifact lock is already held"
        ):
            if operation == "writer":
                with experiment_module.job_artifact_lock(job, tmp_path):
                    pass
            else:
                clear_invalid_job_artifacts(job, tmp_path)
        assert paths.raw_partial.read_bytes() == b"preserve"
    finally:
        assert process.stdin is not None
        process.stdin.close()
        assert process.wait(timeout=5) == 0
        process.stdout.close()

    assert clear_invalid_job_artifacts(job, tmp_path) is True
    assert not paths.raw_partial.exists()
    assert paths.cleanup_lock.exists()


def test_writer_artifact_lock_releases_when_context_raises(
    tmp_path: Path,
) -> None:
    if os.name != "posix":
        pytest.skip("kernel artifact locks are a POSIX runtime contract")
    job = one_job()
    paths = artifact_paths(job, tmp_path)

    with pytest.raises(RuntimeError, match="writer failed"):
        with experiment_module.job_artifact_lock(job, tmp_path):
            raise RuntimeError("writer failed")

    with experiment_module.job_artifact_lock(job, tmp_path):
        assert paths.cleanup_lock.exists()


def test_cleanup_uses_directory_fd_for_artifact_unlinks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if os.name != "posix":
        pytest.skip("dir-fd deletion is a POSIX safety contract")
    job = one_job()
    paths = artifact_paths(job, tmp_path)
    paths.raw_partial.parent.mkdir(parents=True)
    paths.raw_partial.write_bytes(b"partial")
    original_unlink = os.unlink
    calls: list[tuple[str, int | None]] = []

    def tracked_unlink(
        path: str | bytes | os.PathLike[str],
        *,
        dir_fd: int | None = None,
    ) -> None:
        calls.append((os.fsdecode(path), dir_fd))
        original_unlink(path, dir_fd=dir_fd)

    monkeypatch.setattr(experiment_module.os, "unlink", tracked_unlink)

    assert clear_invalid_job_artifacts(job, tmp_path) is True
    assert any(
        name == paths.raw_partial.name and dir_fd is not None
        for name, dir_fd in calls
    )


def test_cleanup_refuses_without_safe_posix_dirfd_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = one_job()
    paths = artifact_paths(job, tmp_path)
    paths.raw_partial.parent.mkdir(parents=True)
    paths.raw_partial.write_bytes(b"preserve")
    monkeypatch.setattr(
        experiment_module, "_SAFE_POSIX_CLEANUP", False, raising=False
    )

    with pytest.raises(RuntimeError, match="requires WSL/POSIX"):
        clear_invalid_job_artifacts(job, tmp_path)

    assert paths.raw_partial.read_bytes() == b"preserve"


@pytest.mark.skipif(os.name != "nt", reason="Windows-only cleanup contract")
def test_windows_cleanup_refuses_destructive_fallback(tmp_path: Path) -> None:
    job = one_job()
    paths = artifact_paths(job, tmp_path)
    paths.raw_partial.parent.mkdir(parents=True)
    paths.raw_partial.write_bytes(b"preserve")

    with pytest.raises(RuntimeError, match="requires WSL/POSIX"):
        clear_invalid_job_artifacts(job, tmp_path)

    assert paths.raw_partial.read_bytes() == b"preserve"


def test_cleanup_parent_swap_cannot_reach_external_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if os.name != "posix":
        pytest.skip("parent swap injection exercises POSIX dir-fd cleanup")
    job = one_job()
    paths = artifact_paths(job, tmp_path)
    paths.raw.parent.mkdir(parents=True)
    paths.raw.write_bytes(b"inside")
    outside = tmp_path.parent / f"outside-swap-{job.run_id}"
    outside.mkdir()
    outside_raw = outside / paths.raw.name
    outside_raw.write_bytes(b"must remain")
    sentinel = outside / "sentinel"
    sentinel.write_bytes(b"must remain")
    original_raw_parent = tmp_path / "raw-before-swap"
    original_open = os.open
    swapped = False

    def swap_before_open(
        path: str | bytes | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if os.fsdecode(path) == "raw" and dir_fd is not None and not swapped:
            paths.raw.parent.rename(original_raw_parent)
            symlink_or_skip(
                paths.raw.parent, outside, target_is_directory=True
            )
            swapped = True
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(experiment_module.os, "open", swap_before_open)

    with pytest.raises((OSError, RuntimeError, ValueError)):
        clear_invalid_job_artifacts(job, tmp_path)

    assert swapped
    assert outside_raw.read_bytes() == b"must remain"
    assert sentinel.read_bytes() == b"must remain"
    assert paths.cleanup_lock.exists()


def test_cleanup_release_failure_still_closes_root_fd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if os.name != "posix":
        pytest.skip("descriptor lifecycle is a POSIX runtime contract")
    job = one_job()
    paths = artifact_paths(job, tmp_path)
    paths.raw_partial.parent.mkdir(parents=True)
    paths.raw_partial.write_bytes(b"partial")
    original_open = os.open
    original_close = os.close
    root_fd: int | None = None
    lock_fd: int | None = None

    def capture_root_open(
        path: str | bytes | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal lock_fd, root_fd
        fd = original_open(path, flags, mode, dir_fd=dir_fd)
        if dir_fd is None and Path(path) == tmp_path.resolve():
            root_fd = fd
        elif os.fsdecode(path) == paths.cleanup_lock.name:
            lock_fd = fd
        return fd

    def fail_lock_close(fd: int) -> None:
        original_close(fd)
        if fd == lock_fd:
            raise OSError("simulated lock release failure")

    monkeypatch.setattr(experiment_module.os, "open", capture_root_open)
    monkeypatch.setattr(experiment_module.os, "close", fail_lock_close)

    try:
        with pytest.raises(OSError, match="simulated lock release failure"):
            clear_invalid_job_artifacts(job, tmp_path)
        assert root_fd is not None
        with pytest.raises(OSError):
            os.fstat(root_fd)
    finally:
        if root_fd is not None:
            try:
                os.close(root_fd)
            except OSError:
                pass


def test_cleanup_release_failure_does_not_mask_primary_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if os.name != "posix":
        pytest.skip("descriptor lifecycle is a POSIX runtime contract")
    job = one_job()
    paths = artifact_paths(job, tmp_path)
    original_open = os.open
    original_close = os.close
    lock_fd: int | None = None

    def fail_cleanup(*args: object, **kwargs: object) -> None:
        raise RuntimeError("primary cleanup failure")

    def capture_lock_open(
        path: str | bytes | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal lock_fd
        fd = original_open(path, flags, mode, dir_fd=dir_fd)
        if os.fsdecode(path) == paths.cleanup_lock.name:
            lock_fd = fd
        return fd

    def fail_lock_close(fd: int) -> None:
        original_close(fd)
        if fd == lock_fd:
            raise OSError("secondary lock release failure")

    monkeypatch.setattr(
        experiment_module, "_safe_cleanup_artifacts", fail_cleanup
    )
    monkeypatch.setattr(experiment_module.os, "open", capture_lock_open)
    monkeypatch.setattr(experiment_module.os, "close", fail_lock_close)

    with pytest.raises(RuntimeError, match="primary cleanup failure") as error:
        clear_invalid_job_artifacts(job, tmp_path)

    assert any(
        "secondary lock release failure" in note
        for note in getattr(error.value, "__notes__", ())
    )
