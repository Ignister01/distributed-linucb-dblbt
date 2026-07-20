"""Complete canonical smoke-gate workflow."""

from __future__ import annotations

import math
from pathlib import Path
import stat

from typer.testing import CliRunner


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SMOKE_MATRIX = REPOSITORY_ROOT / "configs" / "matrices" / "smoke.yaml"


def test_smoke_runner_declares_the_complete_gate() -> None:
    script = REPOSITORY_ROOT / "scripts" / "run_smoke.sh"

    assert script.is_file()
    assert script.stat().st_mode & stat.S_IXUSR
    payload = script.read_text(encoding="ascii")
    commands = ["pytest", "sweep", "summarize", "plot", "audit"]
    positions = [payload.index(command) for command in commands]
    assert positions == sorted(positions)
    assert "set -euo pipefail" in payload


def test_overnight_runner_declares_frozen_resumable_formal_order() -> None:
    script = REPOSITORY_ROOT / "scripts" / "run_overnight.sh"

    assert script.is_file()
    payload = script.read_text(encoding="ascii")
    commands = [
        'bash "$ROOT/scripts/run_smoke.sh"',
        '"$CLI" pretrain',
        "sha256sum models/",
        "reproduction.yaml",
        "heldout.yaml",
        "ablation.yaml",
        "summarize",
        "plot",
        "audit",
    ]
    positions = [payload.index(command) for command in commands]
    assert positions == sorted(positions)
    assert "set -euo pipefail" in payload
    assert "DBLBT_WORKERS" in payload
    assert "40" in payload
    assert "--output-dir \"$FORMAL_ROOT\"" in payload


def test_canonical_smoke_cli_pipeline(tmp_path: Path) -> None:
    from dblbt_fcn import cli
    from dblbt_fcn.experiment import (
        expand_matrix,
        load_completed_job_manifest,
        load_matrix,
    )
    from dblbt_fcn.plotting import FIGURE_NAMES
    from dblbt_fcn.stats import load_summary

    root = tmp_path / "runs" / "smoke"
    summary = tmp_path / "results" / "tables" / "smoke.csv"
    report = tmp_path / "results" / "figures" / "smoke"
    runner = CliRunner()
    commands = [
        [
            "sweep",
            "--matrix",
            str(SMOKE_MATRIX),
            "--workers",
            "3",
            "--output-dir",
            str(root),
        ],
        [
            "summarize",
            "--manifest-dir",
            str(root / "manifests"),
            "--output",
            str(summary),
        ],
        [
            "plot",
            "--summary",
            str(summary),
            "--output-dir",
            str(report),
            "--manifest-dir",
            str(root / "manifests"),
        ],
        [
            "audit",
            "--manifest-dir",
            str(root / "manifests"),
            "--summary",
            str(summary),
            "--output-dir",
            str(report),
        ],
    ]
    for command in commands:
        result = runner.invoke(cli.app, command)
        assert result.exit_code == 0, result.output

    jobs = expand_matrix(load_matrix(SMOKE_MATRIX))
    assert len(list((root / "manifests").glob("*.json"))) == 3
    manifests = [load_completed_job_manifest(job, root) for job in jobs]
    assert all(manifest.status == "complete" for manifest in manifests)
    assert all(manifest.record_hash for manifest in manifests)
    rows = load_summary(summary)
    assert len(rows) == 3
    assert {str(row["policy"]) for row in rows} == {
        "random_lbt",
        "tmc_db_lbt",
        "adaptive_db_lbt",
    }
    for row in rows:
        for field in (
            "collision_probability",
            "effective_airtime",
            "mean_delay_us",
            "p95_delay_us",
            "jain_fairness",
            "evaluation_utility",
        ):
            assert math.isfinite(float(row[field]))
    figures = [
        report / f"{name}.{suffix}"
        for name in FIGURE_NAMES
        for suffix in ("pdf", "png")
    ]
    assert len(figures) == 16
    assert all(path.is_file() and path.stat().st_size > 100 for path in figures)
