"""Focused tests for the pinned official ns-3 build gate."""

from __future__ import annotations

import hashlib
import shlex
import sqlite3
import subprocess
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
VERSIONS = REPOSITORY_ROOT / "ns3" / "VERSIONS"
README = REPOSITORY_ROOT / "ns3" / "README.md"
GATE_SCRIPT = REPOSITORY_ROOT / "scripts" / "run_ns3_gate.sh"
GIT_ATTRIBUTES = REPOSITORY_ROOT / ".gitattributes"
REQUIRED_TABLES = (
    "sinr_results_10",
    "mac_data_tx_failed_10",
    "channel_occupancy_10",
    "simultaneous_tx_10",
    "e2e_10",
)


def parse_assignments(path: Path) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for raw_line in path.read_text(encoding="ascii").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        assert separator == "=", f"invalid assignment in {path}: {raw_line!r}"
        assert key and key not in assignments
        parsed_value = shlex.split(value)
        assert len(parsed_value) == 1
        assignments[key] = parsed_value[0]
    return assignments


def run_gate_function(
    function: str, *arguments: str | Path
) -> subprocess.CompletedProcess[str]:
    assert GATE_SCRIPT.is_file(), f"missing gate runner: {GATE_SCRIPT}"
    command = " ".join(
        (
            shlex.quote(function),
            *(shlex.quote(str(argument)) for argument in arguments),
        )
    )
    return subprocess.run(
        [
            "bash",
            "-c",
            f"source {shlex.quote(str(GATE_SCRIPT))}\n{command}",
        ],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def git(repository: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        text=True,
        capture_output=True,
        check=False,
    )


def make_git_origin(tmp_path: Path) -> tuple[Path, str, str]:
    origin = tmp_path / "origin"
    origin.mkdir()
    assert git(origin, "init", "-b", "main").returncode == 0
    assert git(origin, "config", "user.name", "Gate Test").returncode == 0
    assert git(origin, "config", "user.email", "gate@example.invalid").returncode == 0

    payload = origin / "payload.txt"
    payload.write_text("first\n", encoding="ascii")
    assert git(origin, "add", "payload.txt").returncode == 0
    assert git(origin, "commit", "-m", "first").returncode == 0
    first = git(origin, "rev-parse", "HEAD").stdout.strip()
    assert git(origin, "tag", "pinned", first).returncode == 0

    payload.write_text("second\n", encoding="ascii")
    assert git(origin, "commit", "-am", "second").returncode == 0
    second = git(origin, "rev-parse", "HEAD").stdout.strip()
    return origin, first, second


def create_database(path: Path, tables: tuple[str, ...]) -> None:
    path.unlink(missing_ok=True)
    with sqlite3.connect(path) as connection:
        for table in tables:
            connection.execute(f'CREATE TABLE "{table}" (value INTEGER)')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def prepare_run(
    tmp_path: Path, *, stage: str, log: bytes
) -> tuple[Path, Path]:
    run_dir = tmp_path / "attempt"
    metadata = run_dir / "metadata"
    metadata.mkdir(parents=True)
    (run_dir / "stage").write_text(f"{stage}\n", encoding="ascii")
    (run_dir / "gate.log").write_bytes(log)
    (metadata / "started_at_utc").write_text(
        "2026-07-17T12:00:00Z\n", encoding="ascii"
    )
    return run_dir, metadata


def test_versions_pin_the_official_stack() -> None:
    versions = parse_assignments(VERSIONS)

    assert versions == {
        "ns3_repo": "https://gitlab.com/nsnam/ns-3-dev.git",
        "ns3_ref": "ns-3.35",
        "ns3_tag_object": "020c5f533253c98ee805b715d3efbd559a0ac7b4",
        "ns3_commit": "ac88b75eac1818c673cf2c939a96ac3005b1f051",
        "nr_repo": "https://gitlab.com/cttc-lena/nr.git",
        "nr_release": "5g-lena-v1.2.y",
        "nr_ref": "fe0a1d2a5fb7d1547e46042041288a684893ba9e",
        "nru_repo": "https://gitlab.com/cttc-lena/nr-u.git",
        "nru_ref": "75a45143b1cd382326876a9597e856338673039a",
        "official_example": "cttc-nr-wifi-interference",
    }


def test_versions_file_is_forced_to_lf_for_wsl_bash() -> None:
    attributes = GIT_ATTRIBUTES.read_text(encoding="ascii").splitlines()

    assert "ns3/VERSIONS text eol=lf" in attributes


def test_readme_declares_gate_evidence_and_failure_policy() -> None:
    text = README.read_text(encoding="ascii")

    for required in (
        "WSL2 Ubuntu 24.04",
        "four-hour",
        "gate-status.env",
        "not_evaluated",
        "must not",
        "cttc-nr-wifi-interference",
        "GCC 11",
        "--simTime=0.7",
        "--seed=410",
        "--runId=1",
        "--enableNr=true",
        "--enableWifi=true",
        "--wifiStandard=11ax",
    ):
        assert required in text


def test_ensure_repo_is_idempotent_and_detaches_exact_commit(
    tmp_path: Path,
) -> None:
    origin, first, second = make_git_origin(tmp_path)
    checkout = tmp_path / "checkout"

    for _ in range(2):
        result = run_gate_function(
            "ensure_repo",
            "fixture",
            origin,
            checkout,
            "refs/tags/pinned",
            first,
            first,
        )
        assert result.returncode == 0, result.stderr
        assert git(checkout, "rev-parse", "HEAD").stdout.strip() == first
        assert git(checkout, "symbolic-ref", "-q", "HEAD").returncode != 0

    assert first != second


def test_ensure_repo_distinguishes_annotated_tag_object_from_commit(
    tmp_path: Path,
) -> None:
    origin, first, _ = make_git_origin(tmp_path)
    assert git(
        origin, "tag", "-a", "release", "-m", "release", first
    ).returncode == 0
    tag_object = git(origin, "rev-parse", "refs/tags/release").stdout.strip()
    assert tag_object != first
    checkout = tmp_path / "checkout"

    accepted = run_gate_function(
        "ensure_repo",
        "fixture",
        origin,
        checkout,
        "refs/tags/release",
        tag_object,
        first,
    )
    rejected = run_gate_function(
        "ensure_repo",
        "fixture",
        origin,
        checkout,
        "refs/tags/release",
        first,
        first,
    )

    assert accepted.returncode == 0, accepted.stderr
    assert git(checkout, "rev-parse", "HEAD").stdout.strip() == first
    assert rejected.returncode != 0
    assert "object" in rejected.stderr


def test_ensure_repo_rejects_a_declared_ref_that_moved(tmp_path: Path) -> None:
    origin, first, second = make_git_origin(tmp_path)

    result = run_gate_function(
        "ensure_repo",
        "fixture",
        origin,
        tmp_path / "checkout",
        "main",
        first,
        first,
    )

    assert first != second
    assert result.returncode != 0
    assert "resolved" in result.stderr


@pytest.mark.parametrize("missing_table", REQUIRED_TABLES)
def test_validate_output_database_requires_every_official_table_family(
    tmp_path: Path, missing_table: str
) -> None:
    database = tmp_path / "official.db"
    create_database(database, REQUIRED_TABLES)

    accepted = run_gate_function("validate_output_database", database)

    assert accepted.returncode == 0, accepted.stderr

    incomplete = tuple(table for table in REQUIRED_TABLES if table != missing_table)
    create_database(database, incomplete)

    rejected = run_gate_function("validate_output_database", database)

    assert rejected.returncode != 0


def test_finalize_failure_records_stage_log_hash_and_h5(tmp_path: Path) -> None:
    run_dir, _ = prepare_run(tmp_path, stage="build", log=b"compiler failed\n")
    declared = run_gate_function("record_declared_versions", run_dir)
    assert declared.returncode == 0, declared.stderr

    result = run_gate_function("finalize_run", run_dir, "17")

    assert result.returncode == 0, result.stderr
    status = parse_assignments(run_dir / "gate-status.env")
    assert status["status"] == "failed"
    assert status["stage"] == "build"
    assert status["exit_code"] == "17"
    assert status["timed_out"] == "false"
    assert status["h5_status"] == "not_evaluated"
    assert status["log_sha256"] == sha256(run_dir / "gate.log")
    assert status["output_database_sha256"] == ""
    assert status["declared_ns3_commit"] == (
        "ac88b75eac1818c673cf2c939a96ac3005b1f051"
    )
    assert status["declared_ns3_tag_object"] == (
        "020c5f533253c98ee805b715d3efbd559a0ac7b4"
    )
    assert status["declared_nr_commit"] == (
        "fe0a1d2a5fb7d1547e46042041288a684893ba9e"
    )
    assert status["declared_nru_commit"] == (
        "75a45143b1cd382326876a9597e856338673039a"
    )


def test_finalize_success_records_exact_provenance_and_database_hash(
    tmp_path: Path,
) -> None:
    run_dir, metadata = prepare_run(tmp_path, stage="complete", log=b"gate ok\n")
    database = run_dir / "example" / "official.db"
    database.parent.mkdir()
    create_database(database, REQUIRED_TABLES)
    values = {
        "ns3_ref": "ns-3.35",
        "ns3_commit": "ac88b75eac1818c673cf2c939a96ac3005b1f051",
        "nr_release": "5g-lena-v1.2.y",
        "nr_commit": "fe0a1d2a5fb7d1547e46042041288a684893ba9e",
        "nru_commit": "75a45143b1cd382326876a9597e856338673039a",
        "compiler_path": "/usr/bin/g++",
        "compiler_version": "g++ (Ubuntu 13.3.0) 13.3.0",
        "output_database": str(database),
    }
    for key, value in values.items():
        (metadata / key).write_text(f"{value}\n", encoding="ascii")

    result = run_gate_function("finalize_run", run_dir, "0")

    assert result.returncode == 0, result.stderr
    status = parse_assignments(run_dir / "gate-status.env")
    assert status["status"] == "passed"
    assert status["stage"] == "complete"
    assert status["h5_status"] == "pending_ns3_validation"
    for key, value in values.items():
        assert status[key] == value
    assert status["log_sha256"] == sha256(run_dir / "gate.log")
    assert status["output_database_sha256"] == sha256(database)


def test_finalize_refuses_success_without_a_retained_log(tmp_path: Path) -> None:
    run_dir, metadata = prepare_run(tmp_path, stage="complete", log=b"")
    (run_dir / "gate.log").unlink()
    database = run_dir / "example" / "official.db"
    database.parent.mkdir()
    create_database(database, REQUIRED_TABLES)
    values = {
        "ns3_ref": "ns-3.35",
        "ns3_commit": "ac88b75eac1818c673cf2c939a96ac3005b1f051",
        "nr_release": "5g-lena-v1.2.y",
        "nr_commit": "fe0a1d2a5fb7d1547e46042041288a684893ba9e",
        "nru_commit": "75a45143b1cd382326876a9597e856338673039a",
        "compiler_path": "/usr/bin/g++",
        "compiler_version": "g++ (Ubuntu 13.3.0) 13.3.0",
        "output_database": str(database),
    }
    for key, value in values.items():
        (metadata / key).write_text(f"{value}\n", encoding="ascii")

    result = run_gate_function("finalize_run", run_dir, "0")

    assert result.returncode == 0, result.stderr
    status = parse_assignments(run_dir / "gate-status.env")
    assert status["status"] == "failed"
    assert status["exit_code"] == "70"
    assert status["h5_status"] == "not_evaluated"
    assert status["log_sha256"] == sha256(run_dir / "gate.log")


def test_supervisor_times_out_and_finalizes_last_stage(tmp_path: Path) -> None:
    run_dir = tmp_path / "attempt"
    worker = tmp_path / "worker.sh"
    worker.write_text(
        '#!/usr/bin/env bash\necho build > "$1/stage"\nsleep 3\n',
        encoding="ascii",
    )
    worker.chmod(0o755)

    result = run_gate_function(
        "run_supervisor", run_dir, "1s", worker, run_dir
    )

    assert result.returncode == 124
    status = parse_assignments(run_dir / "gate-status.env")
    assert status["status"] == "failed"
    assert status["timed_out"] == "true"
    assert status["stage"] == "build"
    assert status["h5_status"] == "not_evaluated"
    assert status["log_sha256"] == sha256(run_dir / "gate.log")


def test_declared_exact_commits_are_recorded_before_checkout(tmp_path: Path) -> None:
    run_dir = tmp_path / "attempt"
    (run_dir / "metadata").mkdir(parents=True)

    result = run_gate_function("record_declared_versions", run_dir)

    assert result.returncode == 0, result.stderr
    assert (run_dir / "metadata" / "declared_ns3_commit").read_text(
        encoding="ascii"
    ).strip() == "ac88b75eac1818c673cf2c939a96ac3005b1f051"
    assert (run_dir / "metadata" / "declared_ns3_tag_object").read_text(
        encoding="ascii"
    ).strip() == "020c5f533253c98ee805b715d3efbd559a0ac7b4"
    assert (run_dir / "metadata" / "declared_nr_commit").read_text(
        encoding="ascii"
    ).strip() == "fe0a1d2a5fb7d1547e46042041288a684893ba9e"
    assert (run_dir / "metadata" / "declared_nru_commit").read_text(
        encoding="ascii"
    ).strip() == "75a45143b1cd382326876a9597e856338673039a"


def test_runner_declares_the_complete_official_gate_contract() -> None:
    text = GATE_SCRIPT.read_text(encoding="ascii")
    example = (
        "cttc-nr-wifi-interference --simTime=0.7 --seed=410 --runId=1 "
        "--enableNr=true --enableWifi=true --wifiStandard=11ax"
    )

    assert 'GATE_TIMEOUT="4h"' in text
    assert 'GATE_JOBS="8"' in text
    for package in (
        "build-essential",
        "g++-11",
        "git",
        "python3",
        "sqlite3",
        "libsqlite3-dev",
        "libc6-dev",
        "pkg-config",
    ):
        assert package in text
    for command in (
        "dpkg-query",
        "apt-get update",
        "apt-get install",
        "./waf clean",
        'CC="$GATE_CC" CXX="$GATE_CXX" ./waf configure --enable-examples --enable-tests',
        "./waf configure --enable-examples --enable-tests",
        './waf -j"$GATE_JOBS" build',
        'ensure_repo ns3 "$ns3_repo" "$source_root" "$ns3_ref" "$ns3_tag_object" "$ns3_commit"',
        'ensure_repo nr "$nr_repo" "$source_root/contrib/nr" "$nr_release" "$nr_ref" "$nr_ref"',
        'ensure_repo nr-u "$nru_repo" "$source_root/contrib/nr-u" "$nru_ref" "$nru_ref" "$nru_ref"',
    ):
        assert command in text
    assert text.count(example) == 1
    assert 'GATE_CC="gcc-11"' in text
    assert 'GATE_CXX="g++-11"' in text
    for stage in (
        "dependencies",
        "checkout_ns3",
        "checkout_nr",
        "checkout_nru",
        "configure",
        "build",
        "example",
        "validate_database",
        "complete",
    ):
        assert f'set_stage "$run_dir" {stage}' in text
    assert "gate-runs" in text
    assert "dblbt-fcn" not in text
    assert "run_smoke" not in text


def test_runner_entry_point_follows_every_function_definition() -> None:
    text = GATE_SCRIPT.read_text(encoding="ascii")
    entry_point = text.index('if [[ "${BASH_SOURCE[0]}" == "$0" ]]')

    assert text.index("ensure_repo()") < entry_point
    assert text.index("validate_output_database()") < entry_point
    assert "() {" not in text[entry_point:]


def test_worker_records_provenance_as_each_stage_becomes_available() -> None:
    text = GATE_SCRIPT.read_text(encoding="ascii")
    worker = text[text.index("gate_worker()") : text.index("main()")]

    dependencies = worker.index("ensure_dependencies")
    compiler = worker.index('write_metadata "$run_dir" compiler_path')
    checkout_ns3 = worker.index("ensure_repo ns3 ")
    record_ns3 = worker.index('write_metadata "$run_dir" ns3_commit')
    checkout_nr = worker.index("ensure_repo nr ")
    record_nr = worker.index('write_metadata "$run_dir" nr_commit')
    checkout_nru = worker.index("ensure_repo nr-u ")
    record_nru = worker.index('write_metadata "$run_dir" nru_commit')

    assert dependencies < compiler < checkout_ns3 < record_ns3
    assert record_ns3 < checkout_nr < record_nr
    assert record_nr < checkout_nru < record_nru


@pytest.mark.parametrize("missing_module", ["nr", "nr-u"])
def test_built_module_check_requires_nr_and_nru(
    tmp_path: Path, missing_module: str
) -> None:
    library_dir = tmp_path / "build" / "lib"
    library_dir.mkdir(parents=True)
    libraries = {
        "nr": library_dir / "libns3.35-nr-default.so",
        "nr-u": library_dir / "libns3.35-nr-u-default.so",
    }
    for library in libraries.values():
        library.touch()

    accepted = run_gate_function("assert_built_modules", library_dir)

    assert accepted.returncode == 0, accepted.stderr

    libraries[missing_module].unlink()

    rejected = run_gate_function("assert_built_modules", library_dir)

    assert rejected.returncode != 0
    assert missing_module in rejected.stderr
