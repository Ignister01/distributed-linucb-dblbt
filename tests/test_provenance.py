"""Strict execution provenance mode and field contracts."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from dblbt_fcn.io import RunManifest
from dblbt_fcn.provenance import ExecutionProvenance


HASH_A = "a" * 64
HASH_B = "b" * 64


@pytest.mark.parametrize(
    "value",
    [
        {"mode": "baseline_builtin", "agent_state_sha256": HASH_A},
        {"mode": "adaptive_blank"},
        {
            "mode": "adaptive_blank",
            "agent_state_sha256": HASH_A,
            "model_file_sha256": HASH_B,
        },
        {"mode": "adaptive_model", "agent_state_sha256": HASH_A},
        {"mode": "context_free_builtin", "model_file_sha256": HASH_A},
        {"mode": "pretrain_arm", "oracle_arm": 7},
        {
            "mode": "fixed_oracle",
            "agent_state_sha256": HASH_A,
            "model_file_sha256": HASH_A,
            "oracle_arm": 7,
            "oracle_artifact_sha256": HASH_A,
            "oracle_model_sha256": HASH_B,
            "source_matrix_sha256": HASH_A,
        },
    ],
)
def test_execution_provenance_rejects_invalid_mode_field_combinations(
    value: dict[str, object],
) -> None:
    with pytest.raises(ValidationError, match="mode|provenance|field|hash"):
        ExecutionProvenance.model_validate(value)


def test_manifest_json_rejects_invalid_execution_mode_field_combination() -> None:
    value = {
        "run_id": "run-001",
        "scenario_id": "scenario-001",
        "policy": "random_lbt",
        "seed": 7,
        "config_hash": "1" * 64,
        "git_revision": "a" * 40,
        "dependency_versions": {"python": "3.12"},
        "host": "worker-01",
        "started_at_utc": "2026-01-02T03:04:05Z",
        "ended_at_utc": "2026-01-02T03:04:05Z",
        "elapsed_seconds": 0.0,
        "record_path": None,
        "record_hash": None,
        "row_count": None,
        "exit_code": 1,
        "status": "failed",
        "execution_provenance": {
            "mode": "baseline_builtin",
            "model_file_sha256": HASH_A,
        },
    }

    with pytest.raises(ValidationError, match="mode|provenance|field"):
        RunManifest.model_validate_json(json.dumps(value))


@pytest.mark.parametrize(
    "value",
    [
        {"mode": "baseline_builtin"},
        {"mode": "adaptive_blank", "agent_state_sha256": HASH_A},
        {
            "mode": "adaptive_model",
            "agent_state_sha256": HASH_A,
            "model_file_sha256": HASH_B,
        },
        {"mode": "context_free_builtin"},
        {"mode": "pretrain_arm"},
        {
            "mode": "fixed_oracle",
            "agent_state_sha256": HASH_A,
            "model_file_sha256": HASH_B,
            "oracle_arm": 7,
            "oracle_artifact_sha256": HASH_A,
            "oracle_model_sha256": HASH_B,
            "source_matrix_sha256": HASH_A,
        },
    ],
)
def test_execution_provenance_accepts_exact_mode_field_combinations(
    value: dict[str, object],
) -> None:
    provenance = ExecutionProvenance.model_validate(value)

    assert len(provenance.fingerprint) == 64


def test_context_free_provenance_does_not_read_unused_model_path() -> None:
    from dblbt_fcn.experiment import JobSpec, ScenarioSpec, TimingSpec
    from dblbt_fcn.provenance import execution_provenance

    job = JobSpec(
        matrix="unit",
        rounds=1,
        alpha=11,
        timing=TimingSpec(),
        scenario=ScenarioSpec(
            id="unit",
            wifi_nodes=1,
            nru_nodes=1,
        ),
        policy="adaptive_db_lbt",
        seed=410,
        ablation="context_free_ucb",
    )

    provenance = execution_provenance(job, model_path="missing-model.npz")

    assert provenance.mode == "context_free_builtin"
    assert provenance.model_file_sha256 is None
