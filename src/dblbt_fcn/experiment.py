"""Deterministic experiment matrix expansion and resume helpers."""

from __future__ import annotations

import errno
import gzip
import hashlib
import json
import os
import re
import stat
import sys

try:
    import fcntl
except ImportError:  # pragma: no cover - unavailable on native Windows
    fcntl = None  # type: ignore[assignment]

from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from .io import RunManifest, RunManifestMetadata, completion_marker_path
from .provenance import ExecutionProvenance


class _StrictModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


PolicyName = Literal[
    "random_lbt",
    "primary_db_lbt",
    "tmc_db_lbt",
    "adaptive_db_lbt",
    "fixed_oracle",
    "pretrain_arm",
]
AblationName = Literal[
    "full",
    "no_queue",
    "no_cca_interrupt",
    "no_delay",
    "frozen_online",
    "context_free_ucb",
    "collision_weight_0.125",
    "collision_weight_0.5",
]


def _nonempty_identifier(value: str, *, label: str) -> str:
    if not value.strip():
        raise ValueError(f"{label} must be nonempty")
    if value != value.strip():
        raise ValueError(f"{label} must not have surrounding whitespace")
    return value


def _tuple_input(value: object, *, label: str) -> tuple[object, ...]:
    if type(value) not in (list, tuple):
        raise ValueError(f"{label} must be a list")
    return tuple(value)


class TimingSpec(_StrictModel):
    """Shared slot timing for every job in a matrix."""

    slot_us: int = Field(default=1, ge=1)
    tx_us: int = Field(default=2_000, ge=1)
    wifi_ack_us: int = Field(default=0, ge=0)
    nru_sync_us: int = Field(default=250, ge=1)


class ScenarioSpec(_StrictModel):
    """A fully explicit topology and exogenous channel condition."""

    id: str
    wifi_nodes: int = Field(ge=0)
    nru_nodes: int = Field(ge=0)
    legacy_ap_nodes: int = Field(default=0, ge=0)
    legacy_sta_nodes: int = Field(default=0, ge=0)
    traffic: Literal["saturated", "poisson"] = "saturated"
    poisson_rate_packets_ms: float | None = Field(
        default=None, gt=0, allow_inf_nan=False
    )
    interference_interval_ms: int | None = Field(default=None, gt=0)
    interference_duration_us: int | None = Field(default=None, gt=0)
    interruption_std: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    join_interval_rounds: int | None = Field(default=None, gt=0)
    lifetime_rounds: int | None = Field(default=None, gt=0)
    trace: bool = False

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return _nonempty_identifier(value, label="scenario id")

    @field_validator("interruption_std")
    @classmethod
    def normalize_zero_std(cls, value: float) -> float:
        return 0.0 if value == 0.0 else value

    @model_validator(mode="after")
    def validate_combinations(self) -> Self:
        node_count = (
            self.wifi_nodes
            + self.nru_nodes
            + self.legacy_ap_nodes
            + self.legacy_sta_nodes
        )
        if node_count == 0:
            raise ValueError("scenario must contain at least one node")

        if self.traffic == "poisson":
            if self.poisson_rate_packets_ms is None:
                raise ValueError("poisson traffic requires a packet rate")
        elif self.poisson_rate_packets_ms is not None:
            raise ValueError("saturated traffic cannot specify a packet rate")

        periodic_fields = (
            self.interference_interval_ms,
            self.interference_duration_us,
        )
        if any(value is not None for value in periodic_fields) and not all(
            value is not None for value in periodic_fields
        ):
            raise ValueError(
                "periodic interference interval and duration must appear together"
            )

        dynamic_fields = (
            self.join_interval_rounds,
            self.lifetime_rounds,
        )
        if any(value is not None for value in dynamic_fields) and not all(
            value is not None for value in dynamic_fields
        ):
            raise ValueError(
                "dynamic join interval and lifetime must appear together"
            )
        return self


class MatrixSpec(_StrictModel):
    """A validated, auditable one-factor experiment matrix."""

    version: Literal[1] = 1
    name: str
    rounds: int = Field(ge=1)
    alpha: Literal[11] = 11
    timing: TimingSpec = Field(default_factory=TimingSpec)
    seeds: tuple[int, ...]
    policies: tuple[PolicyName, ...]
    conditions: tuple[AblationName, ...] = ()
    arm_ids: tuple[int, ...] = ()
    scenarios: tuple[ScenarioSpec, ...]

    @field_validator("seeds", "policies", "conditions", "arm_ids", "scenarios", mode="before")
    @classmethod
    def freeze_sequences(cls, value: object, info: Any) -> tuple[object, ...]:
        return _tuple_input(value, label=info.field_name)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return _nonempty_identifier(value, label="matrix name")

    @field_validator("policies", "conditions")
    @classmethod
    def validate_identifiers(
        cls, values: tuple[str, ...], info: Any
    ) -> tuple[str, ...]:
        return tuple(
            _nonempty_identifier(value, label=info.field_name)
            for value in values
        )

    @field_validator("seeds")
    @classmethod
    def validate_seeds(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        if not values:
            raise ValueError("seeds must be nonempty")
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("seeds must be exact nonnegative integers")
        if len(values) != len(set(values)):
            raise ValueError("seeds must be unique")
        return values

    @field_validator("policies")
    @classmethod
    def validate_policies(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if not values:
            raise ValueError("policies must be nonempty")
        if len(values) != len(set(values)):
            raise ValueError("policies must be unique")
        return values

    @field_validator("conditions")
    @classmethod
    def validate_conditions(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("conditions must be unique")
        return values

    @field_validator("arm_ids")
    @classmethod
    def validate_arm_ids(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        if any(
            type(value) is not int or not 0 <= value < 24 for value in values
        ):
            raise ValueError("arm ids must be exact integers in range 0..23")
        if len(values) != len(set(values)):
            raise ValueError("arm ids must be unique")
        return values

    @model_validator(mode="after")
    def validate_scenarios_and_dimensions(self) -> Self:
        if not self.scenarios:
            raise ValueError("scenarios must be nonempty")
        scenario_ids = [scenario.id for scenario in self.scenarios]
        if len(scenario_ids) != len(set(scenario_ids)):
            raise ValueError("scenario ids must be unique")
        scenario_json = [canonical_json(scenario) for scenario in self.scenarios]
        if len(scenario_json) != len(set(scenario_json)):
            raise ValueError("scenarios must be unique")
        if self.conditions and self.arm_ids:
            raise ValueError("conditions and arm ids cannot be crossed")
        if self.arm_ids and self.policies != ("pretrain_arm",):
            raise ValueError("arm ids require the pretrain_arm policy")
        if "pretrain_arm" in self.policies and not self.arm_ids:
            raise ValueError("pretrain_arm requires explicit arm ids")
        if self.conditions and self.policies != ("adaptive_db_lbt",):
            raise ValueError("ablation conditions require adaptive_db_lbt")
        return self


class JobSpec(_StrictModel):
    """Immutable scientific configuration for one runnable job."""

    matrix: str
    rounds: int = Field(ge=1)
    alpha: Literal[11] = 11
    timing: TimingSpec
    scenario: ScenarioSpec
    policy: PolicyName
    seed: int = Field(ge=0)
    arm_id: int | None = Field(default=None, ge=0, lt=24)
    ablation: AblationName | None = None

    @field_validator("matrix", "policy")
    @classmethod
    def validate_required_identifiers(cls, value: str, info: Any) -> str:
        return _nonempty_identifier(value, label=info.field_name)

    @field_validator("ablation")
    @classmethod
    def validate_ablation(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _nonempty_identifier(value, label="ablation")

    @model_validator(mode="after")
    def validate_policy_dimensions(self) -> Self:
        if self.arm_id is not None and self.ablation is not None:
            raise ValueError("arm id and ablation are mutually exclusive")
        if self.arm_id is not None and self.policy != "pretrain_arm":
            raise ValueError("arm id requires the pretrain_arm policy")
        if self.policy == "pretrain_arm" and self.arm_id is None:
            raise ValueError("pretrain_arm requires an arm id")
        if self.ablation is not None and self.policy != "adaptive_db_lbt":
            raise ValueError("ablation requires adaptive_db_lbt")
        return self

    @property
    def formal_seed(self) -> int:
        """Return the preregistered scenario seed."""
        return self.seed

    @property
    def config_hash(self) -> str:
        """Return the full SHA-256 of the canonical job configuration."""
        return hashlib.sha256(canonical_json(self).encode("ascii")).hexdigest()

    @property
    def run_id(self) -> str:
        """Return the stable 16-hex run identifier."""
        return self.config_hash[:16]

    @property
    def exogenous_seed(self) -> int:
        """Return the policy-independent paired exogenous seed."""
        return paired_exogenous_seed(self)


def _normalize_json_value(value: object) -> object:
    if isinstance(value, BaseModel):
        return _normalize_json_value(value.model_dump(mode="python"))
    if isinstance(value, float) and value == 0.0:
        return 0.0
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError(
                    "canonical JSON mapping keys must be exact strings"
                )
            normalized[key] = _normalize_json_value(item)
        return normalized
    if isinstance(value, list):
        return [_normalize_json_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_normalize_json_value(item) for item in value)
    return value


def canonical_json(value: object) -> str:
    """Serialize a model or JSON value using the experiment canonical form."""
    return json.dumps(
        _normalize_json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


class _UniqueKeyLoader(yaml.SafeLoader):
    """YAML loader that refuses silent duplicate-key replacement."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[object, object]:
    loader.flatten_mapping(node)
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as error:
            raise ValueError("YAML mapping keys must be hashable") from error
        if duplicate:
            raise ValueError(f"duplicate YAML mapping key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _load_config_mapping(path: str | Path) -> dict[str, object]:
    """Load a UTF-8 YAML/JSON mapping without accepting duplicate keys."""
    with Path(path).open(encoding="utf-8") as handle:
        value = yaml.load(handle, Loader=_UniqueKeyLoader)
    if type(value) is not dict:
        raise ValueError(f"config root must be a mapping: {path}")
    if not all(type(key) is str for key in value):
        raise ValueError("config root keys must be exact strings")
    return value


def load_matrix(path: str | Path) -> MatrixSpec:
    """Load a UTF-8 matrix YAML/JSON with strict keys and value types."""
    return MatrixSpec.model_validate(_load_config_mapping(path))


def load_job(path: str | Path) -> JobSpec:
    """Load one UTF-8 job YAML/JSON with strict keys and value types."""
    return JobSpec.model_validate(_load_config_mapping(path))


def expand_matrix(matrix: MatrixSpec) -> list[JobSpec]:
    """Expand a matrix into canonical jobs sorted by unique run id."""
    if not isinstance(matrix, MatrixSpec):
        raise TypeError("matrix must be a MatrixSpec")

    arm_ids: tuple[int | None, ...] = matrix.arm_ids or (None,)
    conditions: tuple[str | None, ...] = matrix.conditions or (None,)
    jobs: list[JobSpec] = []
    canonical_jobs: set[str] = set()
    run_ids: set[str] = set()
    for scenario in matrix.scenarios:
        for seed in matrix.seeds:
            for policy in matrix.policies:
                for arm_id in arm_ids:
                    for condition in conditions:
                        job = JobSpec(
                            matrix=matrix.name,
                            rounds=matrix.rounds,
                            alpha=matrix.alpha,
                            timing=matrix.timing,
                            scenario=scenario,
                            policy=policy,
                            seed=seed,
                            arm_id=arm_id,
                            ablation=condition,
                        )
                        serialized = canonical_json(job)
                        if serialized in canonical_jobs:
                            raise ValueError("duplicate canonical job in matrix")
                        canonical_jobs.add(serialized)
                        if job.run_id in run_ids:
                            raise ValueError("run id collision in matrix")
                        run_ids.add(job.run_id)
                        jobs.append(job)
    return sorted(jobs, key=lambda job: job.run_id)


def pair_key(job: JobSpec) -> str:
    """Return the canonical identity shared by paired policy conditions."""
    if not isinstance(job, JobSpec):
        raise TypeError("job must be a JobSpec")
    return canonical_json([job.scenario.model_dump(mode="json"), job.seed])


def paired_exogenous_seed(job: JobSpec) -> int:
    """Derive a fixed 64-bit seed that excludes policy, arm, and ablation."""
    digest = hashlib.sha256(pair_key(job).encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big")


def derive_stream_seed(
    scenario_seed: int, node_id: int | str, stream_name: str
) -> int:
    """Derive a stable 64-bit per-node stream seed using SHA-256."""
    if type(scenario_seed) is not int or scenario_seed < 0:
        raise ValueError("scenario_seed must be an exact nonnegative integer")
    if type(node_id) is int:
        if node_id < 0:
            raise ValueError("integer node_id must be nonnegative")
    elif type(node_id) is str:
        if not node_id.strip():
            raise ValueError("string node_id must be nonempty")
        if node_id != node_id.strip():
            raise ValueError(
                "string node_id must not have surrounding whitespace"
            )
    else:
        raise ValueError(
            "node_id must be a nonnegative integer or trimmed string"
        )
    if type(stream_name) is not str or not stream_name.strip():
        raise ValueError("stream_name must be a nonempty string")
    if stream_name != stream_name.strip():
        raise ValueError("stream_name must not have surrounding whitespace")
    payload = canonical_json([scenario_seed, node_id, stream_name]).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


@dataclass(frozen=True, slots=True)
class ArtifactPaths:
    """Expected files for one job beneath a caller-owned output directory."""

    raw: Path
    manifest: Path
    marker: Path
    raw_partial: Path
    manifest_partial: Path
    marker_partial: Path
    cleanup_lock: Path


def _partial_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.partial")


def artifact_paths(job: JobSpec, output_dir: str | Path) -> ArtifactPaths:
    """Return deterministic artifact paths without creating them."""
    if not isinstance(job, JobSpec):
        raise TypeError("job must be a JobSpec")
    root = Path(output_dir).resolve(strict=False)
    raw = root / "raw" / f"{job.run_id}.jsonl.gz"
    manifest = root / "manifests" / f"{job.run_id}.json"
    marker = completion_marker_path(raw)
    return ArtifactPaths(
        raw=raw,
        manifest=manifest,
        marker=marker,
        raw_partial=_partial_path(raw),
        manifest_partial=_partial_path(manifest),
        marker_partial=_partial_path(marker),
        cleanup_lock=root / f".{job.run_id}.cleanup.lock",
    )


_WINDOWS_DRIVE_PATH = re.compile(r"\A([A-Za-z]):[\\/](.*)\Z")


def _portable_path_identity(value: str) -> tuple[str, ...] | None:
    if type(value) is not str or not value or "\x00" in value:
        return None

    windows_match = _WINDOWS_DRIVE_PATH.fullmatch(value)
    if windows_match is not None:
        parts = tuple(re.split(r"[\\/]", windows_match.group(2)))
        if not parts or any(part in {"", ".", ".."} for part in parts):
            return None
        return ("drive", windows_match.group(1).lower(), *parts)

    if not value.startswith("/"):
        return None
    parts = tuple(value.split("/")[1:])
    if not parts or any(part in {"", ".", ".."} for part in parts):
        return None
    if (
        len(parts) >= 2
        and parts[0].lower() == "mnt"
        and len(parts[1]) == 1
        and parts[1].isalpha()
    ):
        return ("drive", parts[1].lower(), *parts[2:])
    return ("posix", *parts)


def _portable_record_paths_equal(declared: str, expected: str) -> bool:
    declared_identity = _portable_path_identity(declared)
    expected_identity = _portable_path_identity(expected)
    return (
        declared_identity is not None
        and declared_identity == expected_identity
    )


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def _reject_json_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate manifest key: {key}")
        value[key] = item
    return value


def _read_manifest_value(manifest_path: Path) -> dict[str, object]:
    raw = manifest_path.read_bytes()
    value = json.loads(
        raw.decode("utf-8"),
        parse_constant=_reject_json_constant,
        object_pairs_hook=_reject_json_duplicate_keys,
    )
    if type(value) is not dict:
        raise ValueError("manifest root must be an object")
    canonical = (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if raw != canonical:
        raise ValueError("completed manifest is not canonical JSON")
    return value


def _operational_validation_error(
    error: ValidationError,
) -> OSError | RuntimeError | None:
    stack: list[BaseException] = []
    for detail in error.errors(include_url=False):
        nested = detail.get("ctx", {}).get("error")
        if isinstance(nested, BaseException):
            stack.append(nested)

    seen: set[int] = set()
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, (FileNotFoundError, gzip.BadGzipFile, EOFError)):
            continue
        if isinstance(current, (OSError, RuntimeError)):
            return current
        if current.__cause__ is not None:
            stack.append(current.__cause__)
        if current.__context__ is not None:
            stack.append(current.__context__)
    return None


def load_completed_job_manifest(
    job: JobSpec,
    output_dir: str | Path,
    expected_provenance: ExecutionProvenance | None = None,
) -> RunManifest:
    """Load one complete job manifest using its expected native record path."""
    if not isinstance(job, JobSpec):
        raise TypeError("job must be a JobSpec")
    root = Path(output_dir).resolve(strict=False)
    paths = artifact_paths(job, root)
    for path in (paths.manifest, paths.raw, paths.marker):
        _validate_artifact_path(path, root)

    manifest_value = _read_manifest_value(paths.manifest)
    declared_record_path = manifest_value.get("record_path")
    if type(declared_record_path) is not str or not declared_record_path.strip():
        raise ValueError("completed manifest has no valid record_path")
    if not _portable_record_paths_equal(
        declared_record_path, str(paths.raw)
    ):
        raise ValueError(
            "completed manifest record_path does not match job artifact"
        )

    native_manifest_value = dict(manifest_value)
    native_manifest_value["record_path"] = str(paths.raw)
    try:
        manifest = RunManifest.model_validate_json(
            json.dumps(
                native_manifest_value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
        )
    except ValidationError as error:
        operational_error = _operational_validation_error(error)
        if operational_error is not None:
            raise operational_error
        raise ValueError("completed job manifest is invalid") from error

    if not (
        manifest.status == "complete"
        and manifest.exit_code == 0
        and manifest.run_id == job.run_id
        and manifest.scenario_id == job.scenario.id
        and manifest.policy == job.policy
        and manifest.seed == job.seed
        and manifest.config_hash == job.config_hash
        and manifest.row_count == job.rounds
        and (
            expected_provenance is None
            or manifest.execution_provenance == expected_provenance
        )
    ):
        raise ValueError("completed manifest does not match job")
    return manifest


def _load_completed_job_manifest_metadata(
    job: JobSpec,
    output_dir: str | Path,
    expected_provenance: ExecutionProvenance | None = None,
) -> RunManifestMetadata:
    """Load and bind completed manifest metadata without reading raw records."""
    if not isinstance(job, JobSpec):
        raise TypeError("job must be a JobSpec")
    root = Path(output_dir).resolve(strict=False)
    paths = artifact_paths(job, root)
    for path in (paths.manifest, paths.raw, paths.marker):
        _validate_artifact_path(path, root)

    manifest_value = _read_manifest_value(paths.manifest)
    declared_record_path = manifest_value.get("record_path")
    if type(declared_record_path) is not str or not declared_record_path.strip():
        raise ValueError("completed manifest has no valid record_path")
    if not _portable_record_paths_equal(declared_record_path, str(paths.raw)):
        raise ValueError(
            "completed manifest record_path does not match job artifact"
        )

    native_manifest_value = dict(manifest_value)
    native_manifest_value["record_path"] = str(paths.raw)
    try:
        manifest = RunManifestMetadata.model_validate_json(
            json.dumps(
                native_manifest_value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
        )
    except ValidationError as error:
        raise ValueError("completed job manifest is invalid") from error

    if not (
        manifest.status == "complete"
        and manifest.exit_code == 0
        and manifest.run_id == job.run_id
        and manifest.scenario_id == job.scenario.id
        and manifest.policy == job.policy
        and manifest.seed == job.seed
        and manifest.config_hash == job.config_hash
        and manifest.row_count == job.rounds
        and (
            expected_provenance is None
            or manifest.execution_provenance == expected_provenance
        )
    ):
        raise ValueError("completed manifest does not match job")
    return manifest


def job_is_complete(
    job: JobSpec,
    output_dir: str | Path,
    expected_provenance: ExecutionProvenance | None = None,
) -> bool:
    """Return whether a job has a matching, fully validated manifest."""
    try:
        load_completed_job_manifest(job, output_dir, expected_provenance)
        return True
    except FileNotFoundError:
        return False
    except (gzip.BadGzipFile, EOFError, UnicodeError, ValueError, TypeError):
        return False


def jobs_to_run(
    jobs: Iterable[JobSpec],
    output_dir: str | Path,
    expected_provenance: Mapping[str, ExecutionProvenance] | None = None,
) -> list[JobSpec]:
    """Return incomplete jobs in stable run-id order."""
    ordered = sorted(jobs, key=lambda job: job.run_id)
    return [
        job
        for job in ordered
        if not job_is_complete(
            job,
            output_dir,
            None if expected_provenance is None else expected_provenance[job.run_id],
        )
    ]


def _validate_artifact_path(path: Path, root: Path) -> None:
    if not path.is_absolute() or not path.is_relative_to(root):
        raise ValueError("artifact cleanup path escapes output directory")
    resolved_parent = path.parent.resolve(strict=False)
    if not resolved_parent.is_relative_to(root):
        raise ValueError("artifact cleanup parent escapes output directory")
    if os.path.lexists(path) and path.resolve(strict=False) != path:
        resolved = path.resolve(strict=False)
        if not resolved.is_relative_to(root):
            raise ValueError("artifact cleanup target escapes output directory")


_SAFE_POSIX_CLEANUP = (
    os.name == "posix"
    and fcntl is not None
    and hasattr(os, "O_DIRECTORY")
    and hasattr(os, "O_NOFOLLOW")
    and os.unlink in os.supports_dir_fd
    and Path("/proc/self/fd").is_dir()
)


def _directory_open_flags() -> int:
    return (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )


def _resolved_directory_fd(fd: int) -> Path:
    return Path(os.readlink(f"/proc/self/fd/{fd}")).resolve(strict=False)


def _open_cleanup_root(root: Path) -> int:
    fd = os.open(root, _directory_open_flags())
    try:
        if _resolved_directory_fd(fd) != root:
            raise ValueError("cleanup root changed during directory open")
        return fd
    except BaseException:
        os.close(fd)
        raise


@contextmanager
def _locked_job_root(root: Path, lock_name: str) -> Iterator[int]:
    root_fd = _open_cleanup_root(root)
    lock_fd: int | None = None
    try:
        if fcntl is None:
            raise RuntimeError("POSIX advisory locking is unavailable")
        lock_flags = (
            os.O_RDWR
            | os.O_CREAT
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0)
        )
        lock_fd = os.open(
            lock_name,
            lock_flags,
            0o600,
            dir_fd=root_fd,
        )
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            if error.errno not in {errno.EACCES, errno.EAGAIN}:
                raise
            raise BlockingIOError(
                errno.EWOULDBLOCK,
                f"artifact lock is already held: {lock_name}",
            ) from error
        yield root_fd
    finally:
        primary_error = sys.exc_info()[1]
        release_errors: list[OSError] = []
        if lock_fd is not None:
            try:
                os.close(lock_fd)
            except OSError as error:
                release_errors.append(error)
        try:
            os.close(root_fd)
        except OSError as error:
            release_errors.append(error)

        if release_errors:
            if primary_error is None:
                raise release_errors[0]
            for error in release_errors:
                primary_error.add_note(f"artifact lock release failed: {error}")


@contextmanager
def job_artifact_lock(
    job: JobSpec, output_dir: str | Path
) -> Iterator[int]:
    """Hold the cooperative per-job lock used by writers and cleanup."""
    if not isinstance(job, JobSpec):
        raise TypeError("job must be a JobSpec")
    if not _SAFE_POSIX_CLEANUP:
        raise RuntimeError(
            "safe artifact locking requires WSL/POSIX dir-fd support"
        )
    root = Path(output_dir).resolve(strict=False)
    paths = artifact_paths(job, root)
    with _locked_job_root(root, paths.cleanup_lock.name) as root_fd:
        yield root_fd


def ensure_job_config_sidecar(
    job: JobSpec, output_dir: str | Path, root_fd: int
) -> Path:
    """Safely create or verify a canonical config beneath a locked root fd."""
    if not isinstance(job, JobSpec):
        raise TypeError("job must be a JobSpec")
    if not _SAFE_POSIX_CLEANUP:
        raise RuntimeError("safe config sidecar writes require WSL/POSIX")
    root = Path(output_dir).resolve(strict=False)
    if _resolved_directory_fd(root_fd) != root:
        raise ValueError("config sidecar root changed")
    try:
        os.mkdir("configs", mode=0o700, dir_fd=root_fd)
    except FileExistsError:
        pass
    config_fd = _open_cleanup_child(root_fd, root, "configs")
    if config_fd is None:
        raise ValueError("config sidecar directory is unavailable")
    payload = (canonical_json(job) + "\n").encode("ascii")
    name = f"{job.run_id}.json"
    temporary = f".{name}.partial"
    try:
        try:
            existing_fd = os.open(
                name,
                os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
                dir_fd=config_fd,
            )
        except FileNotFoundError:
            existing_fd = None
        if existing_fd is not None:
            try:
                with os.fdopen(existing_fd, "rb", closefd=False) as handle:
                    existing = handle.read()
            finally:
                os.close(existing_fd)
            if existing != payload:
                raise ValueError("existing config sidecar does not match job")
            return root / "configs" / name
        try:
            os.unlink(temporary, dir_fd=config_fd)
        except FileNotFoundError:
            pass
        temporary_fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=config_fd,
        )
        try:
            with os.fdopen(temporary_fd, "wb", closefd=False) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            os.close(temporary_fd)
        os.rename(
            temporary,
            name,
            src_dir_fd=config_fd,
            dst_dir_fd=config_fd,
        )
        os.fsync(config_fd)
        return root / "configs" / name
    finally:
        try:
            os.unlink(temporary, dir_fd=config_fd)
        except FileNotFoundError:
            pass
        os.close(config_fd)


def ensure_job_config(job: JobSpec, output_dir: str | Path) -> Path:
    """Create or verify one canonical sidecar under the job artifact lock."""
    root = Path(output_dir).resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True)
    with job_artifact_lock(job, root) as root_fd:
        return ensure_job_config_sidecar(job, root, root_fd)


def _open_cleanup_child(
    root_fd: int, root: Path, name: str
) -> int | None:
    try:
        fd = os.open(name, _directory_open_flags(), dir_fd=root_fd)
    except FileNotFoundError:
        return None
    except OSError as error:
        if error.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise ValueError(
                "artifact cleanup parent escapes output directory"
            ) from error
        raise
    try:
        resolved = _resolved_directory_fd(fd)
        if not resolved.is_relative_to(root):
            raise ValueError(
                "artifact cleanup parent escapes output directory"
            )
        return fd
    except BaseException:
        os.close(fd)
        raise


def _safe_cleanup_artifacts(
    root_fd: int, root: Path, paths: ArtifactPaths
) -> None:
    groups = {
        "manifests": (paths.manifest.name, paths.manifest_partial.name),
        "raw": (
            paths.raw.name,
            paths.marker.name,
            paths.raw_partial.name,
            paths.marker_partial.name,
        ),
    }
    directory_fds: dict[str, int] = {}
    targets: list[tuple[int, str]] = []
    try:
        for directory in groups:
            fd = _open_cleanup_child(root_fd, root, directory)
            if fd is not None:
                directory_fds[directory] = fd

        for directory, names in groups.items():
            fd = directory_fds.get(directory)
            if fd is None:
                continue
            for name in names:
                try:
                    metadata = os.stat(name, dir_fd=fd, follow_symlinks=False)
                except FileNotFoundError:
                    continue
                if not (
                    stat.S_ISREG(metadata.st_mode)
                    or stat.S_ISLNK(metadata.st_mode)
                ):
                    raise ValueError(
                        "artifact cleanup target must be a regular file or symlink"
                    )
                targets.append((fd, name))

        for fd, name in targets:
            os.unlink(name, dir_fd=fd)
    finally:
        for fd in directory_fds.values():
            os.close(fd)


def clear_invalid_job_artifacts(
    job: JobSpec,
    output_dir: str | Path,
    expected_provenance: ExecutionProvenance | None = None,
) -> bool:
    """Remove only expected invalid files, preserving valid completed jobs."""
    if job_is_complete(job, output_dir, expected_provenance):
        return False

    if not _SAFE_POSIX_CLEANUP:
        raise RuntimeError(
            "safe destructive cleanup requires WSL/POSIX dir-fd support"
        )

    root = Path(output_dir).resolve(strict=False)
    if not root.exists():
        return True
    paths = artifact_paths(job, root)
    with _locked_job_root(root, paths.cleanup_lock.name) as root_fd:
        if job_is_complete(job, root, expected_provenance):
            return False
        _safe_cleanup_artifacts(root_fd, root, paths)
        return True
