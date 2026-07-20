"""Behavioral tests for the shell bootstrap safety helpers."""

from __future__ import annotations

import hashlib
import re
import shlex
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SAFETY_HELPER = REPOSITORY_ROOT / "scripts" / "bootstrap_safety.sh"


@pytest.fixture
def sandbox() -> Iterator[Path]:
    with tempfile.TemporaryDirectory(
        prefix="dblbt-bootstrap-safety-", dir="/tmp"
    ) as directory:
        yield Path(directory)


def run_safety_function(
    function: str, *arguments: str | Path
) -> subprocess.CompletedProcess[str]:
    """Source the helper and invoke one public function with quoted arguments."""
    assert SAFETY_HELPER.is_file(), f"missing shell helper: {SAFETY_HELPER}"

    quoted_helper = shlex.quote(str(SAFETY_HELPER))
    quoted_function = shlex.quote(function)
    quoted_arguments = " ".join(shlex.quote(str(argument)) for argument in arguments)
    command = "\n".join(
        (
            "set -uo pipefail",
            f"source {quoted_helper} || exit 126",
            f"declare -F {quoted_function} >/dev/null || exit 127",
            f"{quoted_function} {quoted_arguments}".rstrip(),
        )
    )
    result = subprocess.run(
        ["bash", "-c", command],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode not in {126, 127}, (
        f"could not load {function} from {SAFETY_HELPER}: {result.stderr}"
    )
    return result


def assert_succeeded(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode == 0, result.stderr


def assert_rejected(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode != 0, result.stdout


def canonical(path: Path) -> str:
    return str(path.resolve())


def is_wsl() -> bool:
    osrelease = Path("/proc/sys/kernel/osrelease")
    if not osrelease.is_file():
        return False
    release = osrelease.read_text(encoding="utf-8").lower()
    return "microsoft" in release or "wsl" in release


def assert_sha_project_key(root: Path, key: str) -> None:
    assert key.startswith(root.name)
    suffix = key[len(root.name) :].lstrip("-_.")
    assert re.fullmatch(r"[0-9a-f]{8,64}", suffix)

    digest = hashlib.sha256(canonical(root).encode("utf-8")).hexdigest()
    assert digest.startswith(suffix)


def test_bootstrap_safety_helper_exists() -> None:
    assert SAFETY_HELPER.is_file(), f"missing shell helper: {SAFETY_HELPER}"


def test_project_key_is_stable_and_separates_same_named_roots(sandbox: Path) -> None:
    first_root = sandbox / "checkout-a" / "shared-project"
    second_root = sandbox / "checkout-b" / "shared-project"
    first_root.mkdir(parents=True)
    second_root.mkdir(parents=True)

    first = run_safety_function("dblbt_project_key", first_root)
    repeated = run_safety_function("dblbt_project_key", first_root)
    second = run_safety_function("dblbt_project_key", second_root)
    for result in (first, repeated, second):
        assert_succeeded(result)
        assert len(result.stdout.splitlines()) == 1

    first_key = first.stdout.strip()
    repeated_key = repeated.stdout.strip()
    second_key = second.stdout.strip()
    assert first_key == repeated_key
    assert first_key != second_key
    assert_sha_project_key(first_root, first_key)
    assert_sha_project_key(second_root, second_key)


def make_validation_paths(sandbox: Path) -> tuple[Path, Path]:
    root = sandbox / "workspace" / "project"
    home = sandbox / "users" / "alice"
    root.mkdir(parents=True)
    home.mkdir(parents=True)
    return root, home


@pytest.mark.parametrize("unsafe_path", ["/", "root", "home"])
def test_validate_venv_path_rejects_root_home_and_filesystem_root(
    sandbox: Path, unsafe_path: str
) -> None:
    root, home = make_validation_paths(sandbox)
    paths = {"/": Path("/"), "root": root, "home": home}

    result = run_safety_function(
        "dblbt_validate_venv_path", root, home, paths[unsafe_path]
    )

    assert_rejected(result)


@pytest.mark.parametrize("ancestor_of", ["root", "home"])
def test_validate_venv_path_rejects_ancestors(
    sandbox: Path, ancestor_of: str
) -> None:
    root, home = make_validation_paths(sandbox)
    unsafe_path = {"root": root.parent, "home": home.parent}[ancestor_of]

    result = run_safety_function(
        "dblbt_validate_venv_path", root, home, unsafe_path
    )

    assert_rejected(result)


def test_validate_venv_path_accepts_unrelated_target(sandbox: Path) -> None:
    root, home = make_validation_paths(sandbox)
    venv = sandbox / "cache" / "venv"

    result = run_safety_function("dblbt_validate_venv_path", root, home, venv)

    assert_succeeded(result)
    assert not venv.exists()


def test_assert_owned_or_empty_accepts_absent_target(sandbox: Path) -> None:
    root = sandbox / "project"
    root.mkdir()
    venv = sandbox / "absent-venv"

    result = run_safety_function("dblbt_assert_owned_or_empty", root, venv)

    assert_succeeded(result)
    assert not venv.exists()


def test_assert_owned_or_empty_accepts_empty_target(sandbox: Path) -> None:
    root = sandbox / "project"
    venv = sandbox / "empty-venv"
    root.mkdir()
    venv.mkdir()

    result = run_safety_function("dblbt_assert_owned_or_empty", root, venv)

    assert_succeeded(result)
    assert list(venv.iterdir()) == []


def test_assert_owned_or_empty_accepts_exact_owner_marker(sandbox: Path) -> None:
    root = sandbox / "project"
    venv = sandbox / "owned-venv"
    root.mkdir()
    venv.mkdir()
    marker = venv / ".dblbt-fcn-owner"
    sentinel = venv / "sentinel.txt"
    marker.write_text(f"{canonical(root)}\n", encoding="utf-8")
    sentinel.write_text("keep me", encoding="utf-8")

    result = run_safety_function("dblbt_assert_owned_or_empty", root, venv)

    assert_succeeded(result)
    assert marker.read_text(encoding="utf-8").splitlines() == [canonical(root)]
    assert sentinel.read_text(encoding="utf-8") == "keep me"


def test_assert_owned_or_empty_rejects_unowned_nonempty_target_without_deleting(
    sandbox: Path,
) -> None:
    root = sandbox / "project"
    venv = sandbox / "unowned-venv"
    root.mkdir()
    venv.mkdir()
    sentinel = venv / "sentinel.txt"
    sentinel.write_text("must survive", encoding="utf-8")

    result = run_safety_function("dblbt_assert_owned_or_empty", root, venv)

    assert_rejected(result)
    assert venv.is_dir()
    assert sentinel.read_text(encoding="utf-8") == "must survive"


def test_assert_owned_or_empty_rejects_mismatched_marker(sandbox: Path) -> None:
    root = sandbox / "project"
    other_root = sandbox / "different-project"
    venv = sandbox / "wrong-owner-venv"
    root.mkdir()
    other_root.mkdir()
    venv.mkdir()
    marker = venv / ".dblbt-fcn-owner"
    sentinel = venv / "sentinel.txt"
    marker.write_text(f"{canonical(other_root)}\n", encoding="utf-8")
    sentinel.write_text("must also survive", encoding="utf-8")

    result = run_safety_function("dblbt_assert_owned_or_empty", root, venv)

    assert_rejected(result)
    assert marker.read_text(encoding="utf-8").splitlines() == [canonical(other_root)]
    assert sentinel.read_text(encoding="utf-8") == "must also survive"


def test_mark_owner_writes_exact_canonical_root(sandbox: Path) -> None:
    root = sandbox / "real" / "project"
    root.mkdir(parents=True)
    root_alias = sandbox / "project-link"
    root_alias.symlink_to(root, target_is_directory=True)
    venv = sandbox / "venv"
    venv.mkdir()

    result = run_safety_function("dblbt_mark_owner", root_alias, venv)

    assert_succeeded(result)
    marker = venv / ".dblbt-fcn-owner"
    assert marker.read_text(encoding="utf-8").splitlines() == [canonical(root)]


@pytest.mark.skipif(not is_wsl(), reason="WSL filesystem check")
def test_native_filesystem_accepts_missing_target_on_native_ext4(
    sandbox: Path,
) -> None:
    venv = sandbox / "not-created" / "venv"

    result = run_safety_function("dblbt_assert_native_filesystem", venv)

    assert_succeeded(result)
    assert not venv.exists()


@pytest.mark.skipif(not is_wsl(), reason="WSL filesystem check")
def test_native_filesystem_rejects_drvfs_target(sandbox: Path) -> None:
    drvfs_root = Path("/mnt/d")
    if not drvfs_root.is_dir():
        pytest.skip("/mnt/d is not mounted")
    venv = drvfs_root / f".{sandbox.name}" / "venv"
    assert not venv.parent.exists()

    result = run_safety_function("dblbt_assert_native_filesystem", venv)

    assert_rejected(result)
    assert not venv.exists()


def test_native_project_venv_rejects_symlink_without_touching_target(
    sandbox: Path,
) -> None:
    target = sandbox / "external-venv"
    target.mkdir()
    sentinel = target / "sentinel.txt"
    sentinel.write_text("do not clear", encoding="utf-8")
    project_venv = sandbox / "project" / ".venv"
    project_venv.parent.mkdir()
    project_venv.symlink_to(target, target_is_directory=True)

    result = run_safety_function(
        "dblbt_assert_native_project_venv", project_venv
    )

    assert_rejected(result)
    assert project_venv.is_symlink()
    assert project_venv.resolve() == target.resolve()
    assert sentinel.read_text(encoding="utf-8") == "do not clear"
