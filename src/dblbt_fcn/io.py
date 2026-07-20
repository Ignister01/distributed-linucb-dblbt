"""Canonical persistence helpers for experiment records and manifests."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from .provenance import ExecutionProvenance


@dataclass(frozen=True)
class RecordMetadata:
    """Integrity metadata for one compressed JSON Lines record."""

    sha256: str
    row_count: int

    def __post_init__(self) -> None:
        _validate_sha256(self.sha256, label="sha256")
        _validate_row_count(self.row_count, label="row_count")


_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_CANONICAL_GZIP_HEADER = b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x02\xff"
_ELAPSED_TIME_TOLERANCE_SECONDS = 1.0


def completion_marker_path(path: str | Path) -> Path:
    """Return the completion marker adjacent to a record file."""
    target = Path(path)
    return target.with_name(f"{target.name}.complete")


def _partial_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.partial")


def _path_exists(path: Path) -> bool:
    return os.path.lexists(path)


def _remove_stale_partial(path: Path) -> None:
    if _path_exists(path):
        path.unlink()


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_bytes_fsync(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _validate_sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be exactly 64 lowercase hexadecimal digits")
    return value


def _validate_row_count(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be an exact nonnegative integer")
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key: {key}")
        value[key] = item
    return value


def write_jsonl_gz(
    path: str | Path, rows: Iterable[Mapping[str, Any]]
) -> RecordMetadata:
    """Atomically write immutable deterministic gzip-compressed JSON Lines.

    The fixed sibling partial path assumes one worker owns a record path at a
    time. Cross-process locking is intentionally outside this helper.
    """
    target = Path(path)
    marker = completion_marker_path(target)
    record_partial = _partial_path(target)
    marker_partial = _partial_path(marker)
    target.parent.mkdir(parents=True, exist_ok=True)
    if _path_exists(target):
        raise FileExistsError(f"record already exists: {target}")
    if _path_exists(marker):
        raise FileExistsError(f"completion marker already exists: {marker}")

    _remove_stale_partial(record_partial)
    _remove_stale_partial(marker_partial)
    installed_target = False
    try:
        row_count = 0
        with record_partial.open("xb") as raw:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw,
                compresslevel=9,
                mtime=0,
            ) as compressed:
                for row in rows:
                    if not isinstance(row, Mapping):
                        raise TypeError("record row must be a mapping")
                    if not all(type(key) is str for key in row):
                        raise TypeError("record root keys must be exact strings")
                    compressed.write(_canonical_json_bytes(dict(row)))
                    row_count += 1
            raw.flush()
            os.fsync(raw.fileno())

        metadata = validate_jsonl_gz(
            record_partial, expected_row_count=row_count
        )
        marker_payload = _canonical_json_bytes(
            {"sha256": metadata.sha256, "row_count": metadata.row_count}
        )
        _write_bytes_fsync(marker_partial, marker_payload)

        os.replace(record_partial, target)
        installed_target = True
        os.replace(marker_partial, marker)
        return metadata
    except BaseException:
        for partial in (record_partial, marker_partial):
            if _path_exists(partial):
                partial.unlink()
        if installed_target and not _path_exists(marker) and _path_exists(target):
            target.unlink()
        raise


def validate_jsonl_gz(
    path: str | Path,
    expected_sha256: str | None = None,
    expected_row_count: int | None = None,
    require_marker: bool = False,
) -> RecordMetadata:
    """Validate a compressed JSON Lines record and return its metadata."""
    stream = validated_jsonl_gz_stream(
        path,
        expected_sha256=expected_sha256,
        expected_row_count=expected_row_count,
        require_marker=require_marker,
    )
    with stream as rows:
        for _ in rows:
            pass
    if stream.metadata is None:
        raise RuntimeError("record stream completed without integrity metadata")
    return stream.metadata


class _HashingReader:
    def __init__(self, handle: Any) -> None:
        self.handle = handle
        self.digest = hashlib.sha256()
        self.prefix = bytearray()

    def read(self, size: int = -1) -> bytes:
        payload = self.handle.read(size)
        self.digest.update(payload)
        missing = len(_CANONICAL_GZIP_HEADER) - len(self.prefix)
        if missing > 0:
            self.prefix.extend(payload[:missing])
        return payload

    def __getattr__(self, name: str) -> Any:
        return getattr(self.handle, name)


def _install_single_member_guard(stream: gzip.GzipFile) -> None:
    """Reject a second gzip member while retaining stdlib CRC checks."""
    reader = stream._buffer.raw
    original = reader._read_gzip_header
    member_count = 0

    def read_one_header() -> bool:
        nonlocal member_count
        found = original()
        if found:
            member_count += 1
            if member_count > 1:
                raise ValueError("record contains multiple gzip members")
        return found

    reader._read_gzip_header = read_one_header


class ValidatedJsonlGzStream(Iterator[dict[str, Any]]):
    """Single-pass canonical JSONL/gzip/hash/marker validation stream."""

    def __init__(
        self,
        path: str | Path,
        *,
        expected_sha256: str | None,
        expected_row_count: int | None,
        require_marker: bool,
    ) -> None:
        if type(require_marker) is not bool:
            raise ValueError("require_marker must be an exact boolean")
        if expected_sha256 is not None:
            _validate_sha256(expected_sha256, label="expected_sha256")
        if expected_row_count is not None:
            _validate_row_count(expected_row_count, label="expected_row_count")
        self.target = Path(path)
        self.expected_sha256 = expected_sha256
        self.expected_row_count = expected_row_count
        self.require_marker = require_marker
        self.metadata: RecordMetadata | None = None
        self._raw: Any = None
        self._hashing: _HashingReader | None = None
        self._gzip: gzip.GzipFile | None = None
        self._row_count = 0
        self._entered = False

    def __enter__(self) -> Self:
        if self._entered:
            raise RuntimeError("validated record stream cannot be re-entered")
        self._raw = self.target.open("rb")
        self._hashing = _HashingReader(self._raw)
        self._gzip = gzip.GzipFile(fileobj=self._hashing, mode="rb")
        _install_single_member_guard(self._gzip)
        self._entered = True
        return self

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> dict[str, Any]:
        if not self._entered or self._gzip is None:
            raise RuntimeError("validated record stream is not open")
        if self.metadata is not None:
            raise StopIteration
        line = self._gzip.readline()
        if not line:
            self._finish()
            raise StopIteration
        if not line.strip():
            raise ValueError("record contains a blank line")
        value = json.loads(
            line.decode("utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
        if not isinstance(value, dict):
            raise ValueError("record row must be a mapping")
        if not all(type(key) is str for key in value):
            raise ValueError("record root keys must be exact strings")
        if line != _canonical_json_bytes(value):
            raise ValueError("record row is not canonical JSON Lines")
        self._row_count += 1
        return value

    def _finish(self) -> None:
        if self._hashing is None:
            raise RuntimeError("validated record stream has no hashing reader")
        for _ in iter(lambda: self._hashing.read(1024 * 1024), b""):
            pass
        if bytes(self._hashing.prefix) != _CANONICAL_GZIP_HEADER:
            raise ValueError("record gzip header is not canonical")
        metadata = RecordMetadata(
            sha256=self._hashing.digest.hexdigest(), row_count=self._row_count
        )
        if (
            self.expected_sha256 is not None
            and metadata.sha256 != self.expected_sha256
        ):
            raise ValueError("record SHA-256 mismatch")
        if (
            self.expected_row_count is not None
            and metadata.row_count != self.expected_row_count
        ):
            raise ValueError("record row count mismatch")
        if self.require_marker:
            marker_bytes = completion_marker_path(self.target).read_bytes()
            marker_value = json.loads(
                marker_bytes.decode("utf-8"),
                parse_constant=_reject_json_constant,
                object_pairs_hook=_reject_duplicate_keys,
            )
            if type(marker_value) is not dict or set(marker_value) != {
                "sha256",
                "row_count",
            }:
                raise ValueError("completion marker has invalid fields")
            marker_metadata = RecordMetadata(
                sha256=_validate_sha256(
                    marker_value["sha256"], label="completion marker sha256"
                ),
                row_count=_validate_row_count(
                    marker_value["row_count"],
                    label="completion marker row_count",
                ),
            )
            if marker_bytes != _canonical_json_bytes(marker_value):
                raise ValueError("completion marker is not canonical JSON")
            if marker_metadata != metadata:
                raise ValueError("completion marker mismatch")
        self.metadata = metadata

    def __exit__(
        self,
        error_type: type[BaseException] | None,
        error: BaseException | None,
        traceback: object,
    ) -> bool:
        try:
            if error_type is None and self.metadata is None:
                raise ValueError("validated record stream must be fully consumed")
        finally:
            if self._gzip is not None:
                self._gzip.close()
            if self._raw is not None:
                self._raw.close()
        return False


def validated_jsonl_gz_stream(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    expected_row_count: int | None = None,
    require_marker: bool = False,
) -> ValidatedJsonlGzStream:
    """Create a stream that validates compressed integrity while iterating."""
    return ValidatedJsonlGzStream(
        path,
        expected_sha256=expected_sha256,
        expected_row_count=expected_row_count,
        require_marker=require_marker,
    )


class RunManifestMetadata(BaseModel):
    """Strict immutable provenance for one experiment run."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    run_id: str
    scenario_id: str
    policy: str
    seed: int = Field(ge=0)
    config_hash: str
    git_revision: str
    dependency_versions: Mapping[str, str]
    host: str
    started_at_utc: datetime
    ended_at_utc: datetime
    elapsed_seconds: float = Field(ge=0, allow_inf_nan=False)
    record_path: str | None
    record_hash: str | None
    row_count: int | None = Field(ge=0)
    exit_code: int
    status: Literal["complete", "failed"]
    execution_provenance: ExecutionProvenance = Field(
        default_factory=lambda: ExecutionProvenance(mode="baseline_builtin")
    )

    @property
    def execution_fingerprint(self) -> str:
        """Return the canonical execution-input fingerprint."""
        return self.execution_provenance.fingerprint

    @field_validator(
        "run_id", "scenario_id", "policy", "git_revision", "host"
    )
    @classmethod
    def validate_nonempty_identifiers(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("identifier and host fields must be nonempty")
        return value

    @field_validator("config_hash")
    @classmethod
    def validate_config_hash(cls, value: str) -> str:
        return _validate_sha256(value, label="config_hash")

    @field_validator("record_hash")
    @classmethod
    def validate_record_hash(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_sha256(value, label="record_hash")

    @field_validator("record_path", mode="before")
    @classmethod
    def normalize_record_path(cls, value: object) -> str | None:
        if value is None:
            return None
        if type(value) is not str:
            raise ValueError("record_path must be a string or None")
        if not value.strip():
            raise ValueError("record_path must be nonblank when present")
        return str(Path(value).resolve(strict=False))

    @field_validator("dependency_versions", mode="before")
    @classmethod
    def copy_and_validate_dependencies(cls, value: object) -> dict[str, str]:
        if not isinstance(value, Mapping):
            raise ValueError("dependency_versions must be a mapping")
        copied: dict[str, str] = {}
        for key, version in value.items():
            if type(key) is not str or not key.strip():
                raise ValueError("dependency names must be nonempty strings")
            if type(version) is not str or not version.strip():
                raise ValueError("dependency versions must be nonempty strings")
            copied[key] = version
        return copied

    @field_validator("dependency_versions")
    @classmethod
    def freeze_dependencies(
        cls, value: Mapping[str, str]
    ) -> Mapping[str, str]:
        return MappingProxyType(dict(value))

    @field_validator("started_at_utc", "ended_at_utc")
    @classmethod
    def validate_utc_datetime(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ValueError("manifest timestamps must be UTC-aware")
        return value

    @field_serializer("dependency_versions")
    def serialize_dependencies(self, value: Mapping[str, str]) -> dict[str, str]:
        return dict(value)

    @field_serializer("started_at_utc", "ended_at_utc")
    def serialize_utc_datetime(self, value: datetime) -> str:
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")

    @model_validator(mode="after")
    def validate_run_state(self) -> Self:
        if self.ended_at_utc < self.started_at_utc:
            raise ValueError("ended_at_utc must not precede started_at_utc")
        measured_elapsed = (
            self.ended_at_utc - self.started_at_utc
        ).total_seconds()
        if (
            abs(self.elapsed_seconds - measured_elapsed)
            > _ELAPSED_TIME_TOLERANCE_SECONDS
        ):
            raise ValueError("elapsed_seconds is inconsistent with UTC timestamps")

        record_fields_present = (
            self.record_path is not None,
            self.record_hash is not None,
            self.row_count is not None,
        )
        if any(record_fields_present) and not all(record_fields_present):
            raise ValueError("record metadata fields must be all absent or all present")
        return self


class RunManifest(RunManifestMetadata):
    """Public manifest that strongly revalidates complete record artifacts."""

    @model_validator(mode="after")
    def validate_complete_record(self) -> Self:
        if self.status == "complete":
            if self.exit_code != 0:
                raise ValueError("complete runs must have exit_code 0")
            if not self.record_path:
                raise ValueError("complete runs require a nonempty record_path")
            if self.record_hash is None or self.row_count is None:
                raise ValueError("complete runs require record integrity metadata")
            try:
                validate_jsonl_gz(
                    self.record_path,
                    expected_sha256=self.record_hash,
                    expected_row_count=self.row_count,
                    require_marker=True,
                )
            except Exception as error:
                raise ValueError(
                    f"complete run record validation failed: {error}"
                ) from error
        return self


def write_manifest(path: str | Path, manifest: RunManifest) -> None:
    """Atomically write a canonical immutable run manifest."""
    if not isinstance(manifest, RunManifest):
        raise TypeError("manifest must be a RunManifest")
    if manifest.status == "complete":
        if (
            not manifest.record_path
            or manifest.record_hash is None
            or manifest.row_count is None
        ):
            raise ValueError("complete manifest is missing record metadata")
        try:
            validate_jsonl_gz(
                manifest.record_path,
                expected_sha256=manifest.record_hash,
                expected_row_count=manifest.row_count,
                require_marker=True,
            )
        except Exception as error:
            raise ValueError(
                f"complete run record validation failed before write: {error}"
            ) from error
    target = Path(path)
    partial = _partial_path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if _path_exists(target):
        raise FileExistsError(f"manifest already exists: {target}")

    _remove_stale_partial(partial)
    payload = _canonical_json_bytes(manifest.model_dump(mode="json"))
    try:
        _write_bytes_fsync(partial, payload)
        os.replace(partial, target)
    except BaseException:
        if _path_exists(partial):
            partial.unlink()
        raise


def load_manifest(path: str | Path) -> RunManifest:
    """Strictly load a manifest, revalidating a complete record."""
    return RunManifest.model_validate_json(Path(path).read_bytes())
