"""Canonical execution-input provenance for resumable experiment runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from .linucb import LinUCB


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _sha256(value: object, label: str) -> str | None:
    if value is None:
        return None
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 or None")
    return value


def file_sha256(path: str | Path) -> str:
    """Hash a regular file from its actual bytes."""
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"provenance input must be a regular file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def linucb_state_sha256(agent: LinUCB) -> str:
    """Hash validated LinUCB metadata and numeric state without trusting callers."""
    if not isinstance(agent, LinUCB):
        raise TypeError("agent must be a LinUCB")
    A = np.ascontiguousarray(agent.A, dtype="<f8")
    b = np.ascontiguousarray(agent.b, dtype="<f8")
    if A.shape != (agent.num_arms, agent.context_dim, agent.context_dim):
        raise ValueError("LinUCB A shape conflicts with metadata")
    if b.shape != (agent.num_arms, agent.context_dim):
        raise ValueError("LinUCB b shape conflicts with metadata")
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)):
        raise ValueError("LinUCB provenance state must be finite")
    metadata = _canonical_json(
        {
            "num_arms": agent.num_arms,
            "context_dim": agent.context_dim,
            "ridge": agent.ridge,
            "exploration": agent.exploration,
            "action_grid_hash": agent.action_grid_hash,
            "A_shape": list(A.shape),
            "b_shape": list(b.shape),
            "dtype": "float64-le",
        }
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(len(metadata).to_bytes(8, "big"))
    digest.update(metadata)
    digest.update(A.tobytes(order="C"))
    digest.update(b.tobytes(order="C"))
    return digest.hexdigest()


class ExecutionProvenance(BaseModel):
    """Frozen canonical identity of every non-JobSpec execution input."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    schema_version: Literal[1] = 1
    mode: Literal[
        "baseline_builtin",
        "adaptive_blank",
        "adaptive_model",
        "context_free_builtin",
        "pretrain_arm",
        "fixed_oracle",
    ]
    agent_state_sha256: str | None = None
    model_file_sha256: str | None = None
    oracle_arm: int | None = None
    oracle_artifact_sha256: str | None = None
    oracle_model_sha256: str | None = None
    source_matrix_sha256: str | None = None
    fingerprint: str = ""

    @field_validator(
        "agent_state_sha256",
        "model_file_sha256",
        "oracle_artifact_sha256",
        "oracle_model_sha256",
        "source_matrix_sha256",
    )
    @classmethod
    def validate_hashes(cls, value: str | None, info: object) -> str | None:
        return _sha256(value, "execution provenance hash")

    @field_validator("oracle_arm")
    @classmethod
    def validate_arm(cls, value: int | None) -> int | None:
        if value is not None and (type(value) is not int or not 0 <= value < 24):
            raise ValueError("oracle_arm must be an exact integer in range 0..23")
        return value

    @model_validator(mode="after")
    def validate_fingerprint(self) -> Self:
        populated = {
            field
            for field in (
                "agent_state_sha256",
                "model_file_sha256",
                "oracle_arm",
                "oracle_artifact_sha256",
                "oracle_model_sha256",
                "source_matrix_sha256",
            )
            if getattr(self, field) is not None
        }
        required = {
            "baseline_builtin": set(),
            "adaptive_blank": {"agent_state_sha256"},
            "adaptive_model": {
                "agent_state_sha256",
                "model_file_sha256",
            },
            "context_free_builtin": set(),
            "pretrain_arm": set(),
            "fixed_oracle": {
                "agent_state_sha256",
                "model_file_sha256",
                "oracle_arm",
                "oracle_artifact_sha256",
                "oracle_model_sha256",
                "source_matrix_sha256",
            },
        }[self.mode]
        if populated != required:
            raise ValueError(
                f"execution provenance fields do not match mode {self.mode}"
            )
        if (
            self.mode == "fixed_oracle"
            and self.oracle_model_sha256 != self.model_file_sha256
        ):
            raise ValueError(
                "fixed_oracle model hashes must identify the same file"
            )
        payload = self.model_dump(mode="json", exclude={"fingerprint"})
        expected = hashlib.sha256(
            _canonical_json(payload).encode("ascii")
        ).hexdigest()
        if self.fingerprint:
            if _sha256(self.fingerprint, "fingerprint") != expected:
                raise ValueError("execution provenance fingerprint mismatch")
        else:
            object.__setattr__(self, "fingerprint", expected)
        return self


def execution_provenance(
    job: object,
    *,
    initial_agent: LinUCB | None = None,
    model_path: str | Path | None = None,
    oracle_arm: int | None = None,
    oracle_artifact_sha256: str | None = None,
    oracle_model_sha256: str | None = None,
    source_matrix_sha256: str | None = None,
) -> ExecutionProvenance:
    """Derive execution provenance from actual job inputs."""
    policy = getattr(job, "policy", None)
    ablation = getattr(job, "ablation", None)
    if type(policy) is not str:
        raise TypeError("job must expose a validated policy")
    if policy == "adaptive_db_lbt":
        if ablation == "context_free_ucb":
            mode = "context_free_builtin"
            state_hash = None
            model_hash = None
        else:
            agent = initial_agent or LinUCB(24, 11, ridge=1.0, exploration=0.5)
            mode = "adaptive_blank" if model_path is None else "adaptive_model"
            state_hash = linucb_state_sha256(agent)
            model_hash = None if model_path is None else file_sha256(model_path)
    elif policy == "fixed_oracle":
        mode = "fixed_oracle"
        state_hash = (
            None if initial_agent is None else linucb_state_sha256(initial_agent)
        )
        model_hash = None if model_path is None else file_sha256(model_path)
    elif policy == "pretrain_arm":
        mode = "pretrain_arm"
        state_hash = None
        model_hash = None
    else:
        mode = "baseline_builtin"
        state_hash = None
        model_hash = None
    return ExecutionProvenance(
        mode=mode,
        agent_state_sha256=state_hash,
        model_file_sha256=model_hash,
        oracle_arm=oracle_arm,
        oracle_artifact_sha256=oracle_artifact_sha256,
        oracle_model_sha256=oracle_model_sha256,
        source_matrix_sha256=source_matrix_sha256,
    )
