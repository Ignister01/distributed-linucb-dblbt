"""Process-level experiment workflow orchestration."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from .config import adaptive_arms
from .experiment import (
    JobSpec,
    MatrixSpec,
    canonical_json,
    ensure_job_config,
    expand_matrix,
    jobs_to_run,
    load_completed_job_manifest,
)
from .io import RunManifest
from .linucb import LinUCB
from .provenance import (
    ExecutionProvenance,
    execution_provenance,
    file_sha256,
)
from .simulation import run_job
from .records import aggregate_rows, read_job_rows
from .training import (
    LocalSample,
    OracleSample,
    PRETRAINING_SEEDS,
    fit_fixed_oracle,
    pretrain_linucb,
)


_SHA256_LENGTH = 64


def action_grid_hash() -> str:
    """Return the SHA-256 of the canonical preregistered 24-arm grid."""
    payload = canonical_json(
        [arm.model_dump(mode="json") for arm in adaptive_arms()]
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def effective_worker_count(workers: object) -> int:
    """Validate a requested process count and apply the 24-worker cap."""
    if type(workers) is not int or workers <= 0:
        raise ValueError("workers must be an exact positive integer")
    return min(workers, 24)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def _reject_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate Oracle artifact key: {key}")
        value[key] = item
    return value


def _sha256_value(label: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


class FixedOracleArtifact(BaseModel):
    """Strict canonical fixed-Oracle choice and its training provenance."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    schema_version: int
    arm: int
    action_grid_hash: str
    source_matrix: MatrixSpec
    source_matrix_hash: str
    model_sha256: str

    @field_validator("schema_version")
    @classmethod
    def validate_version(cls, value: int) -> int:
        if type(value) is not int or value != 1:
            raise ValueError("unsupported Oracle schema_version")
        return value

    @field_validator("arm")
    @classmethod
    def validate_arm(cls, value: int) -> int:
        if type(value) is not int or not 0 <= value < 24:
            raise ValueError("Oracle arm must be in range 0..23")
        return value

    @field_validator("action_grid_hash", "source_matrix_hash", "model_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        return _sha256_value("Oracle provenance hash", value)

    @model_validator(mode="after")
    def validate_source(self) -> "FixedOracleArtifact":
        expected = hashlib.sha256(
            canonical_json(self.source_matrix).encode("ascii")
        ).hexdigest()
        if self.source_matrix_hash != expected:
            raise ValueError("Oracle source matrix hash mismatch")
        if self.action_grid_hash != action_grid_hash():
            raise ValueError("Oracle action_grid_hash mismatch")
        return self


def load_oracle_arm(
    path: str | Path, *, model_path: str | Path
) -> FixedOracleArtifact:
    """Load an Oracle artifact and bind it to the actual deployed model."""
    source = Path(path)
    raw = source.read_bytes()
    value = json.loads(
        raw.decode("utf-8"),
        parse_constant=_reject_json_constant,
        object_pairs_hook=_reject_duplicate_keys,
    )
    if type(value) is not dict:
        raise ValueError("Oracle artifact root must be an object")
    if raw != (canonical_json(value) + "\n").encode("ascii"):
        raise ValueError("Oracle artifact is not canonical JSON")
    artifact = FixedOracleArtifact.model_validate(value)
    _pretraining_jobs(artifact.source_matrix)
    model = LinUCB.load(
        model_path, expected_action_grid_hash=artifact.action_grid_hash
    )
    if model.num_arms != 24 or model.context_dim != 11:
        raise ValueError("Oracle model dimensions are invalid")
    if artifact.model_sha256 != file_sha256(model_path):
        raise ValueError("Oracle model hash mismatch")
    return artifact


def _run_sweep_job(
    job: JobSpec,
    output_dir: str,
    model_path: str | None,
    oracle_arm_path: str | None,
) -> dict[str, object]:
    """Run one complete job in a process without shared mutable state."""
    initial_agent: LinUCB | None = None
    oracle_arm: int | None = None
    oracle: FixedOracleArtifact | None = None
    if job.policy == "adaptive_db_lbt":
        if model_path is None and job.matrix != "smoke":
            raise ValueError("non-smoke adaptive jobs require a model")
        if model_path is not None:
            initial_agent = LinUCB.load(
                model_path, expected_action_grid_hash=action_grid_hash()
            )
    if job.policy == "fixed_oracle":
        if oracle_arm_path is None:
            raise ValueError("fixed_oracle jobs require an Oracle arm file")
        if model_path is None:
            raise ValueError("fixed_oracle jobs require a model file")
        initial_agent = LinUCB.load(
            model_path, expected_action_grid_hash=action_grid_hash()
        )
        oracle = load_oracle_arm(oracle_arm_path, model_path=model_path)
        oracle_arm = oracle.arm
    execution = execution_provenance(
        job,
        initial_agent=initial_agent,
        model_path=(
            model_path
            if job.policy in {"adaptive_db_lbt", "fixed_oracle"}
            else None
        ),
        oracle_arm=oracle_arm,
        oracle_artifact_sha256=(
            file_sha256(oracle_arm_path)
            if job.policy == "fixed_oracle" and oracle_arm_path is not None
            else None
        ),
        oracle_model_sha256=(None if oracle is None else oracle.model_sha256),
        source_matrix_sha256=(
            None if oracle is None else oracle.source_matrix_hash
        ),
    )
    manifest = run_job(
        job,
        output_dir,
        initial_agent=initial_agent,
        oracle_arm=oracle_arm,
        model_path=(
            model_path
            if job.policy in {"adaptive_db_lbt", "fixed_oracle"}
            else None
        ),
        oracle_artifact_path=(
            oracle_arm_path if job.policy == "fixed_oracle" else None
        ),
        execution=execution,
    )
    return manifest.model_dump(mode="json")


def _validated_worker_result(
    job: JobSpec,
    value: object,
    output_dir: Path,
    expected_execution: ExecutionProvenance,
) -> RunManifest:
    if type(value) is not dict:
        raise TypeError("sweep worker result must be an exact dictionary")
    manifest = RunManifest.model_validate_json(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    )
    if not (
        manifest.run_id == job.run_id
        and manifest.scenario_id == job.scenario.id
        and manifest.policy == job.policy
        and manifest.seed == job.seed
        and manifest.config_hash == job.config_hash
        and manifest.row_count == job.rounds
        and manifest.execution_provenance == expected_execution
    ):
        raise ValueError("sweep worker result does not match submitted job")
    completed = load_completed_job_manifest(
        job, output_dir, expected_execution
    )
    if manifest != completed:
        raise ValueError("sweep worker result does not match completed artifact")
    return completed


def run_sweep(
    matrix: MatrixSpec,
    output_dir: str | Path,
    *,
    workers: object,
    model_path: str | Path | None = None,
    oracle_arm_path: str | Path | None = None,
) -> list[RunManifest]:
    """Resume and run matrix jobs, one whole job per process future."""
    if not isinstance(matrix, MatrixSpec):
        raise TypeError("matrix must be a MatrixSpec")
    max_workers = effective_worker_count(workers)
    root = Path(output_dir).resolve(strict=False)
    all_jobs = expand_matrix(matrix)
    model_value = None if model_path is None else str(Path(model_path).resolve())
    oracle_value = (
        None
        if oracle_arm_path is None
        else str(Path(oracle_arm_path).resolve())
    )
    parent_agent: LinUCB | None = None
    oracle: FixedOracleArtifact | None = None
    if model_value is not None:
        parent_agent = LinUCB.load(
            model_value, expected_action_grid_hash=action_grid_hash()
        )
    if any(job.policy == "fixed_oracle" for job in all_jobs):
        if model_value is None or oracle_value is None:
            raise ValueError("fixed_oracle sweep requires model and Oracle files")
        oracle = load_oracle_arm(oracle_value, model_path=model_value)
    expected_executions = {
        job.run_id: execution_provenance(
            job,
            initial_agent=(
                parent_agent
                if job.policy in {"adaptive_db_lbt", "fixed_oracle"}
                else None
            ),
            model_path=(
                model_value
                if job.policy in {"adaptive_db_lbt", "fixed_oracle"}
                else None
            ),
            oracle_arm=(oracle.arm if job.policy == "fixed_oracle" and oracle else None),
            oracle_artifact_sha256=(
                file_sha256(oracle_value)
                if job.policy == "fixed_oracle" and oracle_value
                else None
            ),
            oracle_model_sha256=(
                oracle.model_sha256
                if job.policy == "fixed_oracle" and oracle
                else None
            ),
            source_matrix_sha256=(
                oracle.source_matrix_hash
                if job.policy == "fixed_oracle" and oracle
                else None
            ),
        )
        for job in all_jobs
    }
    for job in all_jobs:
        ensure_job_config(job, root)
    pending = jobs_to_run(all_jobs, root, expected_executions)
    if not pending:
        return []
    executor = ProcessPoolExecutor(max_workers=max_workers)
    in_flight: dict[Future[dict[str, object]], JobSpec] = {}
    results: list[RunManifest] = []
    iterator = iter(pending)

    def submit_next() -> bool:
        try:
            job = next(iterator)
        except StopIteration:
            return False
        future = executor.submit(
            _run_sweep_job,
            job,
            str(root),
            model_value,
            oracle_value,
        )
        in_flight[future] = job
        return True

    try:
        for _ in range(min(max_workers, len(pending))):
            submit_next()
        while in_flight:
            completed, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in sorted(
                completed, key=lambda item: in_flight[item].run_id
            ):
                job = in_flight.pop(future)
                results.append(
                    _validated_worker_result(
                        job,
                        future.result(),
                        root,
                        expected_executions[job.run_id],
                    )
                )
                submit_next()
    except BaseException:
        for future in in_flight:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True, cancel_futures=False)
    return sorted(results, key=lambda manifest: manifest.run_id)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
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


def _stage_bytes(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    return temporary


def _restore_path(path: Path, previous: bytes | None) -> None:
    if previous is None:
        path.unlink(missing_ok=True)
    else:
        _write_bytes_atomic(path, previous)


def _matrix_hash(matrix: MatrixSpec) -> str:
    return hashlib.sha256(canonical_json(matrix).encode("ascii")).hexdigest()


def _pretraining_jobs(matrix: MatrixSpec) -> list[JobSpec]:
    if matrix.policies != ("pretrain_arm",):
        raise ValueError("pretrain requires only the pretrain_arm policy")
    if set(matrix.seeds) != set(PRETRAINING_SEEDS):
        raise ValueError("pretrain matrix must use exactly PRETRAINING_SEEDS")
    if matrix.arm_ids != tuple(range(24)):
        raise ValueError("pretrain matrix must register all arms 0..23 in order")
    if matrix.conditions:
        raise ValueError("pretrain matrix conditions must be empty")
    jobs = expand_matrix(matrix)
    expected = {
        (scenario.id, seed, arm)
        for scenario in matrix.scenarios
        for seed in PRETRAINING_SEEDS
        for arm in range(24)
    }
    actual = {
        (job.scenario.id, job.seed, job.arm_id) for job in jobs
    }
    if actual != expected or len(jobs) != len(expected):
        raise ValueError("pretrain matrix has missing or duplicate jobs")
    return jobs


def _resolved_output_identity(path: Path) -> Path:
    missing: list[str] = []
    cursor = path
    while not cursor.exists() and not cursor.is_symlink():
        missing.append(cursor.name)
        cursor = cursor.parent
    resolved = cursor.resolve(strict=True)
    for name in reversed(missing):
        resolved /= name
    return resolved


def _validate_pretraining_outputs(
    root: Path, model_path: Path, oracle_path: Path
) -> None:
    root_identity = root.resolve(strict=False)
    model_identity = _resolved_output_identity(model_path)
    oracle_identity = _resolved_output_identity(oracle_path)
    for label, identity in (
        ("model", model_identity),
        ("Oracle", oracle_identity),
    ):
        if identity == root_identity or identity.is_relative_to(root_identity):
            raise ValueError(
                f"pretraining {label} output cannot enter input artifact tree"
            )
    if model_identity == oracle_identity:
        raise ValueError("model and Oracle outputs cannot alias each other")


def _load_pretraining_inputs(
    jobs: list[JobSpec], output_dir: Path
) -> tuple[list[LocalSample], list[OracleSample]]:
    samples: list[LocalSample] = []
    sequences_by_node: dict[tuple[int, str], list[int]] = {}
    by_seed_arm_scenario: dict[tuple[int, int], dict[str, float]] = {}
    for job in jobs:
        if job.arm_id is None:
            raise ValueError("pretraining job is missing arm_id")
        load_completed_job_manifest(job, output_dir)
        rows = read_job_rows(job, output_dir)
        aggregate = aggregate_rows(rows)
        scenario_utilities = by_seed_arm_scenario.setdefault(
            (job.seed, job.arm_id), {}
        )
        if job.scenario.id in scenario_utilities:
            raise ValueError("duplicate pretraining scenario utility")
        scenario_utilities[job.scenario.id] = aggregate.evaluation_utility
        job_samples: list[LocalSample] = []
        for row in rows:
            for value in row["training_samples"]:
                if (
                    value["pretraining_seed"] != job.seed
                    or value["arm"] != job.arm_id
                    or not value["node_id"].startswith(f"{job.run_id}:")
                ):
                    raise ValueError("training sample provenance mismatch")
                sample = LocalSample(**value)
                job_samples.append(sample)
                sequences_by_node.setdefault(
                    (sample.pretraining_seed, sample.node_id), []
                ).append(sample.local_sequence)
        if not job_samples:
            raise ValueError(
                f"pretraining job {job.run_id} has no training samples"
            )
        samples.extend(job_samples)

    for key, sequences in sequences_by_node.items():
        if sorted(sequences) != list(range(len(sequences))):
            raise ValueError(
                "pretraining local sequence must start at zero and be "
                f"continuous for seed/node {key!r}"
            )

    scenario_ids = {job.scenario.id for job in jobs}
    oracle_rows: list[OracleSample] = []
    for (seed, arm), utilities in sorted(by_seed_arm_scenario.items()):
        if set(utilities) != scenario_ids:
            raise ValueError("Oracle aggregation is missing pretrain scenarios")
        exact_total = sum(
            (Fraction.from_float(utilities[key]) for key in sorted(utilities)),
            Fraction(),
        )
        utility = float(exact_total / len(utilities))
        oracle_rows.append(OracleSample(arm=arm, utility=utility, seed=seed))
    if len(oracle_rows) != len(PRETRAINING_SEEDS) * 24:
        raise ValueError("Oracle rows are missing seed-arm combinations")
    return samples, oracle_rows


def build_pretraining(
    matrix: MatrixSpec,
    output_dir: str | Path,
    model_output: str | Path,
    oracle_output: str | Path,
    *,
    workers: object,
) -> tuple[LinUCB, int]:
    """Resume pretraining jobs and atomically publish model and Oracle arm."""
    jobs = _pretraining_jobs(matrix)
    root = Path(output_dir).resolve(strict=False)
    model_path = Path(model_output).resolve(strict=False)
    oracle_path = Path(oracle_output).resolve(strict=False)
    _validate_pretraining_outputs(root, model_path, oracle_path)

    run_sweep(matrix, root, workers=workers)
    local_samples, oracle_rows = _load_pretraining_inputs(jobs, root)
    grid_hash = action_grid_hash()
    initial = LinUCB(
        24,
        11,
        ridge=1.0,
        exploration=0.5,
        action_grid_hash=grid_hash,
    )
    trained = pretrain_linucb(local_samples, initial)
    oracle_arm = fit_fixed_oracle(oracle_rows, PRETRAINING_SEEDS)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    oracle_path.parent.mkdir(parents=True, exist_ok=True)
    staged_model: Path | None = None
    staged_oracle: Path | None = None
    previous_model = model_path.read_bytes() if model_path.exists() else None
    previous_oracle = oracle_path.read_bytes() if oracle_path.exists() else None
    model_installed = False
    oracle_installed = False
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{model_path.name}.",
            suffix=".partial.npz",
            dir=model_path.parent,
            delete=False,
        ) as destination:
            staged_model = Path(destination.name)
        staged_model.unlink()
        trained.save(staged_model)
        model_hash = _file_sha256(staged_model)
        oracle_payload = (
            canonical_json(
                {
                    "schema_version": 1,
                    "arm": oracle_arm,
                    "action_grid_hash": grid_hash,
                    "source_matrix": matrix.model_dump(mode="json"),
                    "source_matrix_hash": _matrix_hash(matrix),
                    "model_sha256": model_hash,
                }
            )
            + "\n"
        ).encode("ascii")
        staged_oracle = _stage_bytes(oracle_path, oracle_payload)
        try:
            os.replace(staged_model, model_path)
            model_installed = True
            os.replace(staged_oracle, oracle_path)
            oracle_installed = True
        except BaseException as error:
            try:
                if oracle_installed:
                    _restore_path(oracle_path, previous_oracle)
                if model_installed:
                    _restore_path(model_path, previous_model)
            except BaseException as rollback_error:
                error.add_note(
                    f"failed to roll back pretraining outputs: {rollback_error}"
                )
            raise
    finally:
        if staged_model is not None and staged_model.exists():
            staged_model.unlink()
        if staged_oracle is not None and staged_oracle.exists():
            staged_oracle.unlink()
    return trained, oracle_arm
