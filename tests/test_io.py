"""Canonical record and run-manifest persistence."""

import gzip
import hashlib
import json
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

import dblbt_fcn.io as io_module
from dblbt_fcn.io import (
    RecordMetadata,
    RunManifest,
    completion_marker_path,
    load_manifest,
    validate_jsonl_gz,
    write_manifest,
    write_jsonl_gz,
)


def partial_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.partial")


def test_record_bytes_are_identical_for_the_same_rows(tmp_path: Path) -> None:
    rows = [{"z": 1, "label": "\u5171\u5b58"}, {"value": 2.5}]
    first = tmp_path / "first" / "records.jsonl.gz"
    second = tmp_path / "second" / "records.jsonl.gz"

    write_jsonl_gz(first, rows)
    write_jsonl_gz(second, rows)

    assert first.read_bytes() == second.read_bytes()


def test_writer_returns_metadata_and_creates_a_valid_completion_marker(
    tmp_path: Path,
) -> None:
    path = tmp_path / "records.jsonl.gz"

    metadata = write_jsonl_gz(path, [{"value": 1}])

    assert isinstance(metadata, RecordMetadata)
    assert validate_jsonl_gz(path, require_marker=True) == metadata
    assert completion_marker_path(path).read_bytes() == (
        b'{"row_count":1,"sha256":"'
        + metadata.sha256.encode("ascii")
        + b'"}\n'
    )


@pytest.mark.parametrize(
    ("sha256", "row_count"),
    [
        ("A" * 64, 1),
        ("0" * 63, 1),
        (1, 1),
        ("0" * 64, True),
        ("0" * 64, "1"),
        ("0" * 64, -1),
    ],
)
def test_record_metadata_rejects_invalid_values(
    sha256: object, row_count: object
) -> None:
    with pytest.raises((TypeError, ValueError)):
        RecordMetadata(sha256=sha256, row_count=row_count)  # type: ignore[arg-type]


def test_writer_emits_canonical_utf8_json_and_consumes_generator_once(
    tmp_path: Path,
) -> None:
    iterations = 0

    def rows():
        nonlocal iterations
        iterations += 1
        if iterations > 1:
            raise AssertionError("rows were consumed more than once")
        yield {"z": 2, "a": "\u5171\u5b58"}

    path = tmp_path / "records.jsonl.gz"
    metadata = write_jsonl_gz(path, rows())

    with gzip.open(path, "rb") as handle:
        assert handle.read() == b'{"a":"\xe5\x85\xb1\xe5\xad\x98","z":2}\n'
    assert metadata.row_count == 1
    assert iterations == 1


def test_writer_accepts_empty_rows(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl.gz"

    metadata = write_jsonl_gz(path, iter(()))

    assert metadata.row_count == 0
    assert validate_jsonl_gz(path, require_marker=True) == metadata


@pytest.mark.parametrize(
    "invalid_row",
    [
        ["not", "a", "mapping"],
        {"value": float("nan")},
        {"value": float("inf")},
        {"value": float("-inf")},
        {"value": {1, 2}},
    ],
)
def test_writer_rejects_invalid_rows_without_leaving_artifacts(
    tmp_path: Path, invalid_row: object
) -> None:
    path = tmp_path / "records.jsonl.gz"
    marker = completion_marker_path(path)

    with pytest.raises((TypeError, ValueError)):
        write_jsonl_gz(path, iter([invalid_row]))  # type: ignore[list-item]

    assert not path.exists()
    assert not marker.exists()
    assert not partial_path(path).exists()
    assert not partial_path(marker).exists()


@pytest.mark.parametrize("existing", ["target", "marker"])
def test_writer_never_overwrites_a_record_or_marker(
    tmp_path: Path, existing: str
) -> None:
    path = tmp_path / "records.jsonl.gz"
    marker = completion_marker_path(path)
    existing_path = {"target": path, "marker": marker}[existing]
    existing_path.write_bytes(b"preserve me")

    with pytest.raises(FileExistsError):
        write_jsonl_gz(path, [{"value": 1}])

    assert existing_path.read_bytes() == b"preserve me"
    assert not partial_path(path).exists()
    assert not partial_path(marker).exists()


def test_writer_replaces_a_stale_record_partial(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl.gz"
    partial = partial_path(path)
    partial.write_bytes(b"stale")

    metadata = write_jsonl_gz(path, [{"value": 1}])

    assert validate_jsonl_gz(path, require_marker=True) == metadata
    assert not partial.exists()
    assert not partial_path(completion_marker_path(path)).exists()


def write_unchecked_gzip(
    path: Path,
    payload: bytes,
    *,
    filename: str = "",
    mtime: int = 0,
    compresslevel: int = 9,
) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(
            filename=filename,
            mode="wb",
            fileobj=raw,
            mtime=mtime,
            compresslevel=compresslevel,
        ) as out:
            out.write(payload)


def write_matching_marker(path: Path, row_count: int) -> RecordMetadata:
    metadata = RecordMetadata(
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        row_count=row_count,
    )
    completion_marker_path(path).write_bytes(
        (
            json.dumps(
                {"sha256": metadata.sha256, "row_count": metadata.row_count},
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    )
    return metadata


@pytest.mark.parametrize(
    "payload",
    [
        b'{"z":1,"a":2}\n',
        b'{"a": 2,"z":1}\n',
        b'{"a":1,"a":2}\n',
        b'{"a":1}',
    ],
)
def test_validator_and_complete_manifest_reject_noncanonical_record_bytes(
    tmp_path: Path, payload: bytes
) -> None:
    path = tmp_path / "noncanonical.jsonl.gz"
    write_unchecked_gzip(path, payload)
    metadata = write_matching_marker(path, row_count=1)

    with pytest.raises(ValueError):
        validate_jsonl_gz(path, require_marker=True)

    values = manifest_values(
        record_path=str(path),
        record_hash=metadata.sha256,
        row_count=metadata.row_count,
    )
    with pytest.raises(ValidationError):
        RunManifest(**values)


@pytest.mark.parametrize(
    "gzip_options",
    [
        {"mtime": 1},
        {"filename": "source.jsonl"},
        {"compresslevel": 1},
    ],
)
def test_validator_and_complete_manifest_reject_noncanonical_gzip_header(
    tmp_path: Path, gzip_options: dict[str, object]
) -> None:
    path = tmp_path / "noncanonical-header.jsonl.gz"
    write_unchecked_gzip(path, b'{"a":1}\n', **gzip_options)  # type: ignore[arg-type]
    metadata = write_matching_marker(path, row_count=1)

    with pytest.raises(ValueError):
        validate_jsonl_gz(path, require_marker=True)

    values = manifest_values(
        record_path=str(path),
        record_hash=metadata.sha256,
        row_count=metadata.row_count,
    )
    with pytest.raises(ValidationError):
        RunManifest(**values)


def test_validator_rejects_concatenated_canonical_gzip_members(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.jsonl.gz"
    second = tmp_path / "second.jsonl.gz"
    target = tmp_path / "concatenated.jsonl.gz"
    write_jsonl_gz(first, [{"value": 1}])
    write_jsonl_gz(second, [{"value": 2}])
    target.write_bytes(first.read_bytes() + second.read_bytes())
    write_matching_marker(target, row_count=2)

    with pytest.raises(ValueError, match="canonical|gzip|member"):
        validate_jsonl_gz(target, require_marker=True)


def test_writer_rejects_non_string_root_keys_without_artifacts(
    tmp_path: Path,
) -> None:
    path = tmp_path / "records.jsonl.gz"

    with pytest.raises((TypeError, ValueError)):
        write_jsonl_gz(path, [{1: "not a string key"}])  # type: ignore[dict-item]

    assert not path.exists()
    assert not completion_marker_path(path).exists()
    assert not partial_path(path).exists()


@pytest.mark.parametrize(
    "payload",
    [
        b"\n",
        b"[]\n",
        b'{"value":1}\n\n',
        b'{"value":NaN}\n',
        b"\xff\n",
    ],
)
def test_validator_rejects_invalid_json_lines(
    tmp_path: Path, payload: bytes
) -> None:
    path = tmp_path / "invalid.jsonl.gz"
    write_unchecked_gzip(path, payload)

    with pytest.raises((UnicodeDecodeError, ValueError)):
        validate_jsonl_gz(path)


def test_validated_stream_rejects_early_successful_exit(tmp_path: Path) -> None:
    from dblbt_fcn.io import validated_jsonl_gz_stream

    path = tmp_path / "records.jsonl.gz"
    metadata = write_jsonl_gz(path, [{"value": 1}, {"value": 2}])

    with pytest.raises(ValueError, match="fully consumed|complete"):
        with validated_jsonl_gz_stream(
            path,
            expected_sha256=metadata.sha256,
            expected_row_count=metadata.row_count,
            require_marker=True,
        ) as rows:
            next(rows)


@pytest.mark.parametrize(
    "payload",
    [
        b'{"value":1,"value":2}\n',
        b'{"value": 1}\n',
        b'{"value":NaN}\n',
    ],
)
def test_validated_stream_rejects_duplicate_noncanonical_or_constant_json(
    tmp_path: Path, payload: bytes
) -> None:
    from dblbt_fcn.io import validated_jsonl_gz_stream

    path = tmp_path / "stream-invalid.jsonl.gz"
    write_unchecked_gzip(path, payload)
    metadata = write_matching_marker(path, row_count=1)

    with pytest.raises(ValueError):
        with validated_jsonl_gz_stream(
            path,
            expected_sha256=metadata.sha256,
            expected_row_count=metadata.row_count,
            require_marker=True,
        ) as rows:
            list(rows)


@pytest.mark.parametrize("damage", ["compressed_hash", "marker"])
def test_validated_stream_rejects_hash_or_marker_tamper(
    tmp_path: Path, damage: str
) -> None:
    from dblbt_fcn.io import validated_jsonl_gz_stream

    path = tmp_path / "stream-tampered.jsonl.gz"
    metadata = write_jsonl_gz(path, [{"value": 1}])
    expected_sha256 = metadata.sha256
    if damage == "compressed_hash":
        expected_sha256 = "0" * 64
    else:
        completion_marker_path(path).write_bytes(
            b'{"row_count":1,"sha256":"' + b"0" * 64 + b'"}\n'
        )

    with pytest.raises(ValueError, match="SHA-256|marker|mismatch"):
        with validated_jsonl_gz_stream(
            path,
            expected_sha256=expected_sha256,
            expected_row_count=metadata.row_count,
            require_marker=True,
        ) as rows:
            list(rows)


def test_validator_rejects_corrupt_gzip(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.jsonl.gz"
    path.write_bytes(b"not gzip")

    with pytest.raises((gzip.BadGzipFile, EOFError)):
        validate_jsonl_gz(path)


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("expected_sha256", "A" * 64),
        ("expected_sha256", "0" * 63),
        ("expected_row_count", True),
        ("expected_row_count", "1"),
        ("expected_row_count", -1),
    ],
)
def test_validator_rejects_malformed_expected_metadata(
    tmp_path: Path, keyword: str, value: object
) -> None:
    path = tmp_path / "records.jsonl.gz"
    write_jsonl_gz(path, [{"value": 1}])

    with pytest.raises((TypeError, ValueError)):
        validate_jsonl_gz(path, **{keyword: value})  # type: ignore[arg-type]


@pytest.mark.parametrize("require_marker", [0, 1, "true", None])
def test_validator_requires_an_exact_boolean_marker_flag(
    tmp_path: Path, require_marker: object
) -> None:
    path = tmp_path / "records.jsonl.gz"
    write_jsonl_gz(path, [{"value": 1}])

    with pytest.raises((TypeError, ValueError)):
        validate_jsonl_gz(path, require_marker=require_marker)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "marker_value",
    [
        {"sha256": "0" * 64, "row_count": True},
        {"sha256": "0" * 64, "row_count": 1, "extra": "forbidden"},
        {"sha256": "A" * 64, "row_count": 1},
        {"sha256": "0" * 63, "row_count": 1},
    ],
)
def test_validator_rejects_malformed_completion_marker(
    tmp_path: Path, marker_value: dict[str, object]
) -> None:
    path = tmp_path / "records.jsonl.gz"
    write_jsonl_gz(path, [{"value": 1}])
    completion_marker_path(path).write_text(
        json.dumps(marker_value) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError):
        validate_jsonl_gz(path, require_marker=True)


STARTED_AT = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
ENDED_AT = datetime(2026, 1, 2, 3, 4, 7, tzinfo=UTC)


def manifest_values(
    *,
    record_path: str | None,
    record_hash: str | None,
    row_count: int | None,
    status: str = "complete",
) -> dict[str, object]:
    return {
        "run_id": "run-001",
        "scenario_id": "scenario-001",
        "policy": "random_lbt",
        "seed": 7,
        "config_hash": "1" * 64,
        "git_revision": "abc123",
        "dependency_versions": {"numpy": "2.0.0", "pydantic": "2.8.0"},
        "host": "worker-01",
        "started_at_utc": STARTED_AT,
        "ended_at_utc": ENDED_AT,
        "elapsed_seconds": 2.0,
        "record_path": record_path,
        "record_hash": record_hash,
        "row_count": row_count,
        "exit_code": 0,
        "status": status,
    }


def complete_manifest_values(tmp_path: Path) -> tuple[dict[str, object], Path]:
    record_path = tmp_path / "records.jsonl.gz"
    metadata = write_jsonl_gz(record_path, [{"value": 1}])
    return (
        manifest_values(
            record_path=str(record_path),
            record_hash=metadata.sha256,
            row_count=metadata.row_count,
        ),
        record_path,
    )


def test_complete_manifest_validates_the_finished_record(tmp_path: Path) -> None:
    values, _ = complete_manifest_values(tmp_path)

    manifest = RunManifest(**values)

    assert manifest.status == "complete"
    assert manifest.exit_code == 0
    assert manifest.row_count == 1


def test_complete_manifest_requires_zero_exit_code(tmp_path: Path) -> None:
    values, _ = complete_manifest_values(tmp_path)
    values["exit_code"] = 1

    with pytest.raises(ValidationError):
        RunManifest(**values)


@pytest.mark.parametrize("missing_field", ["record_path", "record_hash", "row_count"])
def test_complete_manifest_requires_all_record_metadata(
    tmp_path: Path, missing_field: str
) -> None:
    values, _ = complete_manifest_values(tmp_path)
    values[missing_field] = None

    with pytest.raises(ValidationError):
        RunManifest(**values)


def test_complete_manifest_rejects_an_empty_record_path(tmp_path: Path) -> None:
    values, _ = complete_manifest_values(tmp_path)
    values["record_path"] = ""

    with pytest.raises(ValidationError):
        RunManifest(**values)


def test_complete_manifest_rejects_a_missing_record(tmp_path: Path) -> None:
    path = tmp_path / "missing.jsonl.gz"
    values = manifest_values(
        record_path=str(path), record_hash="0" * 64, row_count=1
    )

    with pytest.raises((ValidationError, ValueError)):
        RunManifest(**values)


def test_complete_manifest_rejects_a_corrupt_record(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.jsonl.gz"
    path.write_bytes(b"not gzip")
    values = manifest_values(
        record_path=str(path), record_hash="0" * 64, row_count=1
    )

    with pytest.raises((ValidationError, ValueError)):
        RunManifest(**values)


@pytest.mark.parametrize("mismatch", ["hash", "row_count"])
def test_complete_manifest_rejects_record_metadata_mismatch(
    tmp_path: Path, mismatch: str
) -> None:
    values, _ = complete_manifest_values(tmp_path)
    if mismatch == "hash":
        values["record_hash"] = "0" * 64
    else:
        values["row_count"] = 2

    with pytest.raises(ValidationError):
        RunManifest(**values)


def test_complete_manifest_rejects_completion_marker_mismatch(
    tmp_path: Path,
) -> None:
    values, path = complete_manifest_values(tmp_path)
    completion_marker_path(path).write_bytes(
        (
            json.dumps(
                {"sha256": values["record_hash"], "row_count": 2},
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    )

    with pytest.raises(ValidationError):
        RunManifest(**values)


def test_failed_manifest_allows_no_record_even_with_zero_exit_code(
    tmp_path: Path,
) -> None:
    values = manifest_values(
        record_path=None, record_hash=None, row_count=None, status="failed"
    )

    manifest = RunManifest(**values)

    assert manifest.record_path is None
    assert manifest.record_hash is None
    assert manifest.row_count is None
    assert manifest.exit_code == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("run_id", ""),
        ("scenario_id", " "),
        ("policy", ""),
        ("git_revision", ""),
        ("host", " "),
    ],
)
def test_manifest_rejects_empty_identifiers_and_host(
    field: str, value: str
) -> None:
    values = manifest_values(
        record_path=None, record_hash=None, row_count=None, status="failed"
    )
    values[field] = value

    with pytest.raises(ValidationError):
        RunManifest(**values)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seed", True),
        ("seed", "7"),
        ("row_count", True),
        ("row_count", "1"),
        ("exit_code", True),
        ("exit_code", "0"),
        ("elapsed_seconds", True),
        ("elapsed_seconds", "2.0"),
    ],
)
def test_manifest_rejects_coerced_numeric_values(
    field: str, value: object
) -> None:
    values = manifest_values(
        record_path=None, record_hash=None, row_count=None, status="failed"
    )
    values[field] = value

    with pytest.raises(ValidationError):
        RunManifest(**values)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seed", -1),
        ("row_count", -1),
        ("elapsed_seconds", -1.0),
        ("elapsed_seconds", float("nan")),
        ("elapsed_seconds", float("inf")),
    ],
)
def test_manifest_rejects_invalid_numeric_ranges(
    field: str, value: object
) -> None:
    values = manifest_values(
        record_path=None, record_hash=None, row_count=None, status="failed"
    )
    values[field] = value

    with pytest.raises(ValidationError):
        RunManifest(**values)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("config_hash", "A" * 64),
        ("config_hash", "0" * 63),
        ("record_hash", "A" * 64),
        ("record_hash", "0" * 63),
    ],
)
def test_manifest_rejects_malformed_hashes(field: str, value: str) -> None:
    values = manifest_values(
        record_path=None, record_hash=None, row_count=None, status="failed"
    )
    values[field] = value

    with pytest.raises(ValidationError):
        RunManifest(**values)


@pytest.mark.parametrize(
    "started_at",
    [
        datetime(2026, 1, 2, 3, 4, 5),
        datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone(timedelta(hours=8))),
        "2026-01-02T03:04:05Z",
    ],
)
def test_manifest_requires_strict_utc_datetimes(started_at: object) -> None:
    values = manifest_values(
        record_path=None, record_hash=None, row_count=None, status="failed"
    )
    values["started_at_utc"] = started_at

    with pytest.raises(ValidationError):
        RunManifest(**values)


def test_manifest_rejects_end_before_start() -> None:
    values = manifest_values(
        record_path=None, record_hash=None, row_count=None, status="failed"
    )
    values["ended_at_utc"] = STARTED_AT - timedelta(seconds=1)

    with pytest.raises(ValidationError):
        RunManifest(**values)


@pytest.mark.parametrize("elapsed_seconds", [1.0, 3.0])
def test_manifest_accepts_elapsed_time_at_consistency_tolerance_boundary(
    elapsed_seconds: float,
) -> None:
    values = manifest_values(
        record_path=None, record_hash=None, row_count=None, status="failed"
    )
    values["elapsed_seconds"] = elapsed_seconds

    assert RunManifest(**values).elapsed_seconds == elapsed_seconds


@pytest.mark.parametrize("elapsed_seconds", [0.999, 3.001, 999.0])
def test_manifest_rejects_elapsed_time_outside_consistency_tolerance(
    elapsed_seconds: float,
) -> None:
    values = manifest_values(
        record_path=None, record_hash=None, row_count=None, status="failed"
    )
    values["elapsed_seconds"] = elapsed_seconds

    with pytest.raises(ValidationError):
        RunManifest(**values)


def test_complete_manifest_normalizes_record_path_and_loads_after_cwd_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record_path = tmp_path / "record.jsonl.gz"
    metadata = write_jsonl_gz(record_path, [{"value": 1}])
    monkeypatch.chdir(tmp_path)
    values = manifest_values(
        record_path=record_path.name,
        record_hash=metadata.sha256,
        row_count=metadata.row_count,
    )

    manifest = RunManifest(**values)

    assert manifest.record_path == str(record_path.resolve())
    manifest_path = tmp_path / "manifest.json"
    write_manifest(manifest_path, manifest)
    other_directory = tmp_path / "other"
    other_directory.mkdir()
    monkeypatch.chdir(other_directory)
    assert load_manifest(manifest_path) == manifest


@pytest.mark.parametrize(
    ("record_path", "record_hash", "row_count"),
    [
        ("diagnostic.jsonl.gz", None, None),
        (None, "0" * 64, None),
        (None, None, 1),
        ("diagnostic.jsonl.gz", "0" * 64, None),
        ("diagnostic.jsonl.gz", None, 1),
        (None, "0" * 64, 1),
    ],
)
def test_failed_manifest_rejects_partial_record_metadata(
    record_path: str | None,
    record_hash: str | None,
    row_count: int | None,
) -> None:
    values = manifest_values(
        record_path=record_path,
        record_hash=record_hash,
        row_count=row_count,
        status="failed",
    )

    with pytest.raises(ValidationError):
        RunManifest(**values)


def test_failed_manifest_allows_all_diagnostic_record_metadata() -> None:
    values = manifest_values(
        record_path="diagnostic.jsonl.gz",
        record_hash="0" * 64,
        row_count=1,
        status="failed",
    )

    manifest = RunManifest(**values)

    assert manifest.record_hash == "0" * 64
    assert manifest.row_count == 1
    assert manifest.record_path is not None
    assert Path(manifest.record_path).is_absolute()


@pytest.mark.parametrize("record_path", ["", "   "])
def test_failed_manifest_rejects_blank_diagnostic_record_path(
    record_path: str,
) -> None:
    values = manifest_values(
        record_path=record_path,
        record_hash="0" * 64,
        row_count=1,
        status="failed",
    )

    with pytest.raises(ValidationError):
        RunManifest(**values)


@pytest.mark.parametrize(
    "dependency_versions",
    [
        {"numpy": ""},
        {"": "2.0.0"},
        {1: "2.0.0"},
        {"numpy": 2},
    ],
)
def test_manifest_rejects_invalid_dependency_versions(
    dependency_versions: object,
) -> None:
    values = manifest_values(
        record_path=None, record_hash=None, row_count=None, status="failed"
    )
    values["dependency_versions"] = dependency_versions

    with pytest.raises(ValidationError):
        RunManifest(**values)


def test_manifest_is_immutable_and_deep_copies_dependencies() -> None:
    dependencies = {"numpy": "2.0.0"}
    values = manifest_values(
        record_path=None, record_hash=None, row_count=None, status="failed"
    )
    values["dependency_versions"] = dependencies
    manifest = RunManifest(**values)

    dependencies["numpy"] = "changed"

    assert manifest.dependency_versions["numpy"] == "2.0.0"
    with pytest.raises(TypeError):
        manifest.dependency_versions["numpy"] = "changed"  # type: ignore[index]
    with pytest.raises(ValidationError):
        manifest.host = "changed"


def test_manifest_forbids_extra_fields_and_unknown_status() -> None:
    values = manifest_values(
        record_path=None, record_hash=None, row_count=None, status="failed"
    )

    with pytest.raises(ValidationError):
        RunManifest(**values, unknown=True)
    values["status"] = "running"
    with pytest.raises(ValidationError):
        RunManifest(**values)


def test_manifest_bytes_are_canonical_and_deterministic(tmp_path: Path) -> None:
    values, _ = complete_manifest_values(tmp_path)
    manifest = RunManifest(**values)
    first = tmp_path / "first" / "manifest.json"
    second = tmp_path / "second" / "manifest.json"

    write_manifest(first, manifest)
    write_manifest(second, manifest)

    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes().endswith(b"\n")
    assert b'"started_at_utc":"2026-01-02T03:04:05Z"' in first.read_bytes()


def test_manifest_round_trips_and_load_revalidates_record(
    tmp_path: Path,
) -> None:
    values, record_path = complete_manifest_values(tmp_path)
    manifest = RunManifest(**values)
    path = tmp_path / "manifest.json"
    write_manifest(path, manifest)

    assert load_manifest(path) == manifest

    record_path.write_bytes(b"corrupt after manifest write")
    with pytest.raises((ValidationError, ValueError)):
        load_manifest(path)


def test_write_manifest_refuses_to_overwrite(tmp_path: Path) -> None:
    values = manifest_values(
        record_path=None, record_hash=None, row_count=None, status="failed"
    )
    manifest = RunManifest(**values)
    path = tmp_path / "manifest.json"
    path.write_bytes(b"preserve me")

    with pytest.raises(FileExistsError):
        write_manifest(path, manifest)

    assert path.read_bytes() == b"preserve me"


def test_write_manifest_cleans_partial_when_atomic_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = manifest_values(
        record_path=None, record_hash=None, row_count=None, status="failed"
    )
    manifest = RunManifest(**values)
    path = tmp_path / "manifest.json"

    def fail_replace(source: Path, target: Path) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(io_module.os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        write_manifest(path, manifest)

    assert not path.exists()
    assert not partial_path(path).exists()


@pytest.mark.parametrize("damaged", ["record", "marker"])
def test_write_complete_manifest_revalidates_record_before_writing(
    tmp_path: Path, damaged: str
) -> None:
    values, record_path = complete_manifest_values(tmp_path)
    manifest = RunManifest(**values)
    path = tmp_path / "manifest.json"
    if damaged == "record":
        record_path.write_bytes(b"corrupt after construction")
    else:
        completion_marker_path(record_path).write_bytes(
            b'{"row_count":2,"sha256":"'
            + str(values["record_hash"]).encode("ascii")
            + b'"}\n'
        )

    with pytest.raises(ValueError):
        write_manifest(path, manifest)

    assert not path.exists()
    assert not partial_path(path).exists()


def test_load_manifest_strictly_rejects_extra_fields(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    values = manifest_values(
        record_path=None, record_hash=None, row_count=None, status="failed"
    )
    values["unknown"] = True
    path.write_text(json.dumps(values, default=str), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_manifest(path)
