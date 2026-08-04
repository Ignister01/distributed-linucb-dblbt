"""Contract tests for the resumable formal ns-3 validation runner."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_ns3_validation.sh"
LOCK = ROOT / "ns3" / "validation.env"


def _assignments(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="ascii").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        assert separator == "=" and key and key not in values
        parsed = shlex.split(value)
        assert len(parsed) == 1
        values[key] = parsed[0]
    return values


def _run(
    function: str,
    *arguments: str | Path,
    prelude: str = "",
) -> subprocess.CompletedProcess[str]:
    assert RUNNER.is_file(), f"missing validation runner: {RUNNER}"
    command = " ".join(
        (function, *(shlex.quote(str(argument)) for argument in arguments))
    )
    return subprocess.run(
        [
            "bash",
            "-c",
            f"source {shlex.quote(str(RUNNER))}\n{prelude}\n{command}",
        ],
        cwd=ROOT,
        env={**os.environ, "DBLBT_VALIDATION_PYTHON": sys.executable},
        text=True,
        capture_output=True,
        check=False,
    )


def _git_repository(path: Path) -> str:
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Test"], check=True
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    (path / "README").write_text("pinned source\n", encoding="ascii")
    subprocess.run(["git", "-C", str(path), "add", "README"], check=True)
    subprocess.run(
        ["git", "-C", str(path), "commit", "-q", "-m", "fixture"], check=True
    )
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE validation_metadata (
              schema_version INTEGER NOT NULL, job_id TEXT NOT NULL,
              policy TEXT NOT NULL, scenario TEXT NOT NULL,
              seed INTEGER NOT NULL, run_id INTEGER NOT NULL,
              wifi_aps INTEGER NOT NULL, nru_gnbs INTEGER NOT NULL,
              node_rate_bps INTEGER NOT NULL, traffic_mode TEXT NOT NULL,
              sim_time_s REAL NOT NULL, shadowing_enabled INTEGER NOT NULL,
              srs_enabled INTEGER NOT NULL, alpha INTEGER NOT NULL,
              cold_start_attempts INTEGER NOT NULL,
              decision_interval INTEGER NOT NULL,
              context_dim INTEGER NOT NULL, num_arms INTEGER NOT NULL,
              model_sha256 TEXT NOT NULL, model_export_sha256 TEXT NOT NULL,
              action_grid_hash TEXT NOT NULL, ns3_commit TEXT NOT NULL,
              nr_commit TEXT NOT NULL, nru_commit TEXT NOT NULL,
              patch_sha256 TEXT NOT NULL, scenario_sha256 TEXT NOT NULL
            );
            CREATE TABLE dblbt_nodes (
              node_id INTEGER PRIMARY KEY, technology TEXT NOT NULL,
              state_id TEXT NOT NULL UNIQUE
            );
            CREATE TABLE dblbt_attempts (
              node_id INTEGER NOT NULL, attempt_id INTEGER NOT NULL,
              outcome TEXT NOT NULL, elapsed_us REAL NOT NULL,
              busy_us REAL NOT NULL, interruptions INTEGER NOT NULL,
              access_delay_us REAL NOT NULL, queue_occupancy REAL NOT NULL,
              arrivals INTEGER NOT NULL, retries INTEGER NOT NULL,
              effective_data_us REAL NOT NULL, selected_backoff INTEGER NOT NULL,
              PRIMARY KEY (node_id, attempt_id)
            );
            CREATE TABLE dblbt_decisions (
              node_id INTEGER NOT NULL, decision_round INTEGER NOT NULL,
              arm_id INTEGER NOT NULL, kappa INTEGER NOT NULL,
              alpha INTEGER NOT NULL, beta INTEGER NOT NULL,
              m INTEGER NOT NULL, b_init INTEGER NOT NULL,
              reward REAL NOT NULL, context_0 REAL NOT NULL,
              context_1 REAL NOT NULL, context_2 REAL NOT NULL,
              context_3 REAL NOT NULL, context_4 REAL NOT NULL,
              context_5 REAL NOT NULL, context_6 REAL NOT NULL,
              context_7 REAL NOT NULL, context_8 REAL NOT NULL,
              context_9 REAL NOT NULL, context_10 REAL NOT NULL,
              PRIMARY KEY (node_id, decision_round)
            );
            CREATE TABLE validation_metrics (
              technology TEXT PRIMARY KEY, throughput_mbps REAL NOT NULL,
              mean_delay_us REAL NOT NULL, packet_loss_ratio REAL NOT NULL,
              simultaneous_access_collision_rate REAL NOT NULL,
              channel_occupancy REAL NOT NULL
            );
            CREATE TABLE channel_occupancy_5 (
              TECHNOLOGY STRING NOT NULL, VALUE DOUBLE NOT NULL,
              SEED INT NOT NULL, RUN INT NOT NULL
            );
            CREATE TABLE simultaneous_tx_5 (
              UID INT NOT NULL, SIMULTANEOUS_TX_SAME_TECH INT NOT NULL,
              SIMULTANEOUS_TX_OTHER_TECH INT NOT NULL, TOTALTX INT NOT NULL,
              SEED INT NOT NULL, RUN INT NOT NULL
            );
            CREATE TABLE mac_data_tx_failed_5 (
              UID INT NOT NULL, NUMBER INT NOT NULL, BYTES INT NOT NULL,
              SEED INT NOT NULL, RUN INT NOT NULL
            );
            CREATE TABLE sinr_results_5 (
              UID INT NOT NULL, SINR DOUBLE NOT NULL,
              SEED INT NOT NULL, RUN INT NOT NULL
            );
            CREATE TABLE e2e_5 (
              TECHNOLOGY STRING NOT NULL, THROUGHPUT_MBPS DOUBLE NOT NULL,
              TXBYTES INT NOT NULL, RXBYTES INT NOT NULL,
              LATENCY_US DOUBLE NOT NULL, JITTER_US DOUBLE NOT NULL,
              ADDR STRING NOT NULL, SEED INT NOT NULL, RUN INT NOT NULL
            );
            """
        )
        connection.execute(
            "INSERT INTO validation_metadata VALUES "
            "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                2,
                "static-4x4__seed-410__random",
                "random",
                "static-4x4",
                410,
                1,
                4,
                4,
                2_000_000,
                "aggregate-saturated-cbr",
                2.0,
                1,
                0,
                11,
                8,
                32,
                11,
                24,
                "70611e9712e3a8e4b0f35fc8e0e616f4fde9a20b17c844f1b6406da2833301e6",
                "f82d68db38cdc66a88cb144d9337a22b9e43b44f45643fbbaf8e42b8e9e8efd9",
                "558da7340dfa32d8cc484ba68a05951314936d7aff34a145cc34ea051c07707c",
                "ac88b75eac1818c673cf2c939a96ac3005b1f051",
                "fe0a1d2a5fb7d1547e46042041288a684893ba9e",
                "75a45143b1cd382326876a9597e856338673039a",
                "8a729fa68489bc0bc724f6e6d925b88aa25b574423606f7a36f57065d6e284b9",
                "f4a8ed1a4eec64b9b3a534b4545c5a925a914e22696a6ecbe13183859bd84565",
            ),
        )
        controller_ids = [*range(4), *range(8, 12)]
        for index, node_id in enumerate(controller_ids):
            technology = "wifi" if index < 4 else "nru"
            connection.execute(
                "INSERT INTO dblbt_nodes VALUES (?,?,?)",
                (node_id, technology, f"random-410-{node_id}"),
            )
        for uid in range(16):
            connection.execute(
                "INSERT INTO simultaneous_tx_5 VALUES (?,?,?,?,?,?)",
                (uid, 1, 1, 10, 410, 1),
            )
            connection.execute(
                "INSERT INTO mac_data_tx_failed_5 VALUES (?,?,?,?,?)",
                (uid, 1, 1000, 410, 1),
            )
            connection.execute(
                "INSERT INTO sinr_results_5 VALUES (?,?,?,?)",
                (uid, 12.0, 410, 1),
            )
        for technology in ("wifi", "nru"):
            connection.execute(
                "INSERT INTO validation_metrics VALUES (?,?,?,?,?,?)",
                (technology, 10.0, 500.0, 0.1, 0.04, 0.4),
            )
            connection.execute(
                "INSERT INTO channel_occupancy_5 VALUES (?,?,?,?)",
                (technology, 0.4, 410, 1),
            )
            connection.execute(
                "INSERT INTO e2e_5 VALUES (?,?,?,?,?,?,?,?,?)",
                (technology, 10.0, 10000, 9000, 500.0, 50.0,
                 f"10.0.0.{1 if technology == 'wifi' else 2}", 410, 1),
            )


def _refresh_manifest(path: Path, database: Path) -> None:
    payload = json.loads(path.read_text(encoding="ascii"))
    payload["database_bytes"] = database.stat().st_size
    payload["database_sha256"] = hashlib.sha256(database.read_bytes()).hexdigest()
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )


def test_validation_lock_freezes_formal_sources_and_runtime() -> None:
    assert RUNNER.is_file()
    assert LOCK.is_file()
    assert _assignments(LOCK) == {
        "schema_version": "2",
        "ns3_commit": "ac88b75eac1818c673cf2c939a96ac3005b1f051",
        "nr_commit": "fe0a1d2a5fb7d1547e46042041288a684893ba9e",
        "nru_commit": "75a45143b1cd382326876a9597e856338673039a",
        "wifi_patch_sha256": "4ab70d716af146046117420724b7a4a593f8044c5fbd85d99c3337826e33e7d6",
        "nr_patch_sha256": "5f95f6e2a6bf717ee177fd00352b49507ea9d2c98ac96b4777e6da64221634a6",
        "nru_patch_sha256": "c792aac7d05561fc354043f62521d36adf137bb6e00fac0f4156688d8eddaeb9",
        "patch_bundle_sha256": "8a729fa68489bc0bc724f6e6d925b88aa25b574423606f7a36f57065d6e284b9",
        "scenario_sha256": "f4a8ed1a4eec64b9b3a534b4545c5a925a914e22696a6ecbe13183859bd84565",
        "model_sha256": "70611e9712e3a8e4b0f35fc8e0e616f4fde9a20b17c844f1b6406da2833301e6",
        "model_export_sha256": "f82d68db38cdc66a88cb144d9337a22b9e43b44f45643fbbaf8e42b8e9e8efd9",
        "action_grid_hash": "558da7340dfa32d8cc484ba68a05951314936d7aff34a145cc34ea051c07707c",
        "node_rate_bps": "2000000",
        "traffic_mode": "aggregate-saturated-cbr",
        "build_profile": "optimized",
        "max_workers": "8",
    }


def test_runner_enforces_paths_workers_and_toolchain() -> None:
    assert _run("assert_descendant", ROOT, ROOT / "ns3").returncode == 0
    assert _run("assert_descendant", ROOT, ROOT.parent).returncode != 0
    for workers in (1, 4, 8):
        assert _run("validate_workers", str(workers)).returncode == 0
    for workers in (0, 9, "x"):
        assert _run("validate_workers", str(workers)).returncode != 0
    text = RUNNER.read_text(encoding="ascii")
    assert 'VALIDATION_CC="gcc-11"' in text
    assert 'VALIDATION_CXX="g++-11"' in text
    assert "--build-profile=optimized" in text
    smoke_body = text.split("run_smoke_policy() {", 1)[1].split(
        "\n}\n\nrun_smoke()", 1
    )[0]
    formal_body = text.split("run_formal_job() {", 1)[1].split(
        "\n}\n\nrun_formal_matrix()", 1
    )[0]
    assert "--shadowingEnabled=false" in smoke_body
    assert "--shadowingEnabled=true" in formal_body


@pytest.mark.skipif(
    not Path("/mnt/d").exists(),
    reason="WSL drvfs check requires /mnt/d",
)
def test_runner_requires_ext4_cache_and_runtime_paths() -> None:
    ext4_cache = "/root/.cache/dblbt-fcn"
    ext4_runtime = "/root/.cache/dblbt-fcn/runtime"
    mounted_workspace = "/mnt/d"

    filesystem = _run("filesystem_type", ext4_cache)
    assert filesystem.returncode == 0, filesystem.stderr
    assert filesystem.stdout.strip() == "ext4"

    for cache, runtime, expected in (
        (ext4_cache, ext4_runtime, 0),
        (mounted_workspace, ext4_runtime, 1),
        (ext4_cache, "/mnt/d/dblbt-runtime", 1),
    ):
        script = f"""
source {shlex.quote(str(RUNNER))}
CACHE_PARENT={shlex.quote(cache)}
RUNTIME_ROOT={shlex.quote(runtime)}
require_runtime_filesystems
"""
        result = subprocess.run(
            ["bash", "-e", "-c", script],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert (result.returncode == 0) is (expected == 0)
        if expected:
            assert "requires ext4" in result.stderr.lower()


def test_prepare_runtime_checks_ext4_before_source_or_build_work() -> None:
    text = RUNNER.read_text(encoding="ascii")
    body = text.split("prepare_runtime() {", 1)[1].split("\n}\n\nrun_binary()", 1)[0]

    assert "findmnt -T" in text
    assert body.index("require_runtime_filesystems") < body.index(
        "verify_validation_sources"
    )


def test_run_audit_uses_configured_validation_python() -> None:
    text = RUNNER.read_text(encoding="ascii")
    body = text.split("run_audit() {", 1)[1].split("\n}\n\nmain()", 1)[0]

    assert 'PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"' in body
    assert '"$VALIDATION_PYTHON" -' in body
    assert '"$ROOT/.venv/bin/python"' not in body


def test_run_reduce_uses_audited_ns3_reduction_contract() -> None:
    text = RUNNER.read_text(encoding="ascii")
    body = text.split("run_reduce() {", 1)[1].split("\n}\n\nmain()", 1)[0]

    assert 'PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"' in body
    assert '"$VALIDATION_PYTHON" -' in body
    assert "reduce_ns3_validation" in body
    assert "write_ns3_reduction" in body
    all_body = text.split("all)", 1)[1].split(";;", 1)[0]
    assert all_body.index("run_audit") < all_body.index("run_reduce")


def test_runner_verifies_all_frozen_source_hashes(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    ns3_commit = _git_repository(source_root)
    nr_commit = _git_repository(source_root / "contrib" / "nr")
    nru_commit = _git_repository(source_root / "contrib" / "nr-u")
    fixture_lock = tmp_path / "fixture.env"
    fixture_lock.write_text(
        LOCK.read_text(encoding="ascii")
        .replace(
            "ns3_commit=ac88b75eac1818c673cf2c939a96ac3005b1f051",
            f"ns3_commit={ns3_commit}",
        )
        .replace(
            "nr_commit=fe0a1d2a5fb7d1547e46042041288a684893ba9e",
            f"nr_commit={nr_commit}",
        )
        .replace(
            "nru_commit=75a45143b1cd382326876a9597e856338673039a",
            f"nru_commit={nru_commit}",
        ),
        encoding="ascii",
    )
    prelude = f"SOURCE_ROOT={shlex.quote(str(source_root))}"
    assert (
        _run(
            "verify_validation_sources", ROOT, fixture_lock, prelude=prelude
        ).returncode
        == 0
    )
    bad_lock = tmp_path / "bad.env"
    bad_lock.write_text(
        fixture_lock.read_text(encoding="ascii").replace(
            "scenario_sha256=f4a8ed1a", "scenario_sha256=04a8ed1a"
        ),
        encoding="ascii",
    )
    result = _run(
        "verify_validation_sources", ROOT, bad_lock, prelude=prelude
    )
    assert result.returncode != 0
    assert "scenario hash mismatch" in result.stderr.lower()


def test_runner_manifests_are_atomic_and_reject_changed_database(
    tmp_path: Path,
) -> None:
    output = tmp_path / "formal"
    databases = output / "databases"
    databases.mkdir(parents=True)
    job_id = "static-4x4__seed-410__random"
    database = databases / f"{job_id}.db"
    _database(database)

    result = _run("write_job_manifest", output, job_id)
    assert result.returncode == 0, result.stderr
    manifest = output / "manifests" / f"{job_id}.json"
    assert manifest.is_file()
    assert not list((output / "manifests").glob("*.tmp.*"))
    assert _run("job_is_complete", output, job_id).returncode == 0

    with database.open("ab") as stream:
        stream.write(b"changed")
    assert _run("job_is_complete", output, job_id).returncode != 0


def test_resume_rejects_valid_manifest_with_wrong_frozen_metadata(
    tmp_path: Path,
) -> None:
    output = tmp_path / "formal"
    databases = output / "databases"
    databases.mkdir(parents=True)
    job_id = "static-4x4__seed-410__random"
    database = databases / f"{job_id}.db"
    _database(database)
    assert _run("write_job_manifest", output, job_id).returncode == 0

    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE validation_metadata SET node_rate_bps = 1000000"
        )
    manifest = output / "manifests" / f"{job_id}.json"
    _refresh_manifest(manifest, database)

    assert _run("job_is_complete", output, job_id).returncode != 0


def test_runner_requires_both_audited_smoke_policies(tmp_path: Path) -> None:
    smoke = tmp_path / "smoke"
    databases = smoke / "databases"
    databases.mkdir(parents=True)
    for policy in ("tmc", "adaptive"):
        _database(databases / f"{policy}.db")

    bypass = "strict_validate_database() { :; }"
    assert _run("require_smoke_gate", smoke, prelude=bypass).returncode != 0
    assert _run(
        "write_smoke_marker", smoke, "tmc", prelude=bypass
    ).returncode == 0
    assert _run("require_smoke_gate", smoke, prelude=bypass).returncode != 0
    assert _run(
        "write_smoke_marker", smoke, "adaptive", prelude=bypass
    ).returncode == 0
    assert _run("require_smoke_gate", smoke, prelude=bypass).returncode == 0


def test_runtime_provenance_rejects_stale_compiler_profile_and_commit(
    tmp_path: Path,
) -> None:
    values = _assignments(LOCK)
    expected = {
        "patch_bundle_sha256": values["patch_bundle_sha256"],
        "scenario_sha256": values["scenario_sha256"],
        "compiler": "pinned-compiler",
        "build_profile": values["build_profile"],
        "ns3_commit": values["ns3_commit"],
        "nr_commit": values["nr_commit"],
        "nru_commit": values["nru_commit"],
    }

    def matches(payload: dict[str, str]) -> bool:
        runtime_env = tmp_path / "runtime.env"
        runtime_env.write_text(
            "".join(f"{key}={value}\n" for key, value in payload.items()),
            encoding="ascii",
        )
        script = f"""
source {shlex.quote(str(RUNNER))}
validation_compiler_identity() {{ printf 'pinned-compiler\\n'; }}
runtime_provenance_matches {shlex.quote(str(runtime_env))}
"""
        return subprocess.run(
            ["bash", "-e", "-c", script],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        ).returncode == 0

    assert matches(expected)
    for key in (
        "compiler",
        "build_profile",
        "ns3_commit",
        "nr_commit",
        "nru_commit",
    ):
        stale = dict(expected)
        stale[key] = f"stale-{key}"
        assert not matches(stale)


def test_runner_declares_per_job_logs_scratch_staging_and_bounded_waits() -> None:
    text = RUNNER.read_text(encoding="ascii")
    for token in (
        'logs/${job_id}.log',
        "mktemp -d",
        "mv -fT",
        "wait -n",
        "DBLBT_NS3_WORKERS",
        "DBLBT_NS3_SCRATCH_ROOT",
        '/dev/shm/dblbt-fcn/ns3-validation-$patch_bundle_sha256',
        '"$RUNTIME_ROOT/tmp"',
        "require_smoke_gate",
        "job_is_complete",
    ):
        assert token in text
    assert "test.py" not in text
    assert "nvidia" not in text.lower()


@pytest.mark.parametrize("workers", (1, 6))
def test_formal_matrix_completes_all_jobs_under_errexit(
    tmp_path: Path,
    workers: int,
) -> None:
    completed = tmp_path / "completed.txt"
    script = f"""
source {shlex.quote(str(RUNNER))}
require_smoke_gate() {{ :; }}
run_formal_job() {{
  printf '%s|%s|%s\\n' "$1" "$2" "$3" >> {shlex.quote(str(completed))}
}}
OUTPUT_ROOT={shlex.quote(str(tmp_path / "output"))}
DBLBT_NS3_WORKERS={workers}
run_formal_matrix
"""
    result = subprocess.run(
        ["bash", "-e", "-c", script],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    jobs = completed.read_text(encoding="ascii").splitlines()
    assert len(jobs) == 27
    assert len(set(jobs)) == 27


def test_explicit_scratch_root_rejects_insufficient_capacity(
    tmp_path: Path,
) -> None:
    scratch = tmp_path / "scratch"
    script = f"""
source {shlex.quote(str(RUNNER))}
scratch_available_kib() {{ printf '1\\n'; }}
DBLBT_NS3_SCRATCH_ROOT={shlex.quote(str(scratch))}
DBLBT_NS3_WORKERS=2
select_scratch_root
"""
    result = subprocess.run(
        ["bash", "-e", "-c", script],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "insufficient scratch capacity" in result.stderr.lower()


def test_smoke_and_formal_jobs_use_isolated_configured_scratch(
    tmp_path: Path,
) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    sentinel = scratch / "caller-owned.txt"
    sentinel.write_text("keep\n", encoding="ascii")
    captured = tmp_path / "captured.txt"
    output = tmp_path / "output"
    script = f"""
source {shlex.quote(str(RUNNER))}
DBLBT_NS3_SCRATCH_ROOT={shlex.quote(str(scratch))}
DBLBT_NS3_WORKERS=1
job_is_complete() {{ return 1; }}
run_binary() {{
  local argument
  for argument in "$@"; do
    case "$argument" in
      --outputDb=*) printf '%s\\n' "${{argument#--outputDb=}}" >> {shlex.quote(str(captured))} ;;
    esac
  done
}}
validate_smoke_database() {{ :; }}
validate_database_contract() {{ :; }}
copy_database_atomic() {{ :; }}
write_smoke_marker() {{ :; }}
write_job_manifest() {{ :; }}
run_smoke_policy {shlex.quote(str(output / "smoke"))} tmc
run_formal_job static-4x4 410 random {shlex.quote(str(output / "formal"))}
"""
    result = subprocess.run(
        ["bash", "-e", "-c", script],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    databases = [Path(line) for line in captured.read_text(encoding="ascii").splitlines()]
    assert len(databases) == 2
    assert len({database.parent for database in databases}) == 2
    assert all(scratch.resolve() in database.resolve().parents for database in databases)
    assert all(not database.parent.exists() for database in databases)
    assert sentinel.read_text(encoding="ascii") == "keep\n"


def test_failed_job_retains_its_scratch_child(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    captured = tmp_path / "captured.txt"
    output = tmp_path / "output"
    script = f"""
source {shlex.quote(str(RUNNER))}
DBLBT_NS3_SCRATCH_ROOT={shlex.quote(str(scratch))}
DBLBT_NS3_WORKERS=1
job_is_complete() {{ return 1; }}
run_binary() {{
  local argument
  for argument in "$@"; do
    case "$argument" in
      --outputDb=*) printf '%s\\n' "${{argument#--outputDb=}}" > {shlex.quote(str(captured))} ;;
    esac
  done
  return 1
}}
run_formal_job static-4x4 410 random {shlex.quote(str(output / "formal"))}
"""
    result = subprocess.run(
        ["bash", "-e", "-c", script],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    database = Path(captured.read_text(encoding="ascii").strip())
    assert database.parent.is_dir()
    assert scratch.resolve() in database.resolve().parents


def test_cleanup_uses_original_scratch_root_when_selector_changes(
    tmp_path: Path,
) -> None:
    original = tmp_path / "scratch-a"
    changed = tmp_path / "scratch-b"
    original.mkdir()
    changed.mkdir()
    job = original / "static-4x4__seed-410__random.Abc123"
    job.mkdir()
    ordinary = original / "caller-owned"
    ordinary.mkdir()
    script = f"""
source {shlex.quote(str(RUNNER))}
select_scratch_root() {{ printf '%s\\n' {shlex.quote(str(changed))}; }}
remove_successful_job_scratch {shlex.quote(str(original))} {shlex.quote(str(job))}
"""
    result = subprocess.run(
        ["bash", "-e", "-c", script],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert not job.exists()
    assert original.is_dir()
    assert changed.is_dir()
    assert ordinary.is_dir()
