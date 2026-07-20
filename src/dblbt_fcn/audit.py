"""Read-only fail-closed audit of Task 12 evidence outputs."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import math
from pathlib import Path

from .linucb import LinUCB
from .inventory import validate_report_inventory
from .plotting import FIGURE_NAMES, PlotRun, validated_plot_inputs
from .provenance import file_sha256, linucb_state_sha256
from .stats import (
    ModelOverhead,
    table_payloads,
)
from .training import HELD_OUT_SEEDS
from .workflows import action_grid_hash, load_oracle_arm


@dataclass(frozen=True, slots=True)
class AuditResult:
    run_count: int
    figure_count: int
    table_count: int


def _csv_rows(path: Path, expected_fields: list[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"missing table: {path.name}")
    try:
        rows = list(csv.DictReader(path.open(encoding="ascii", newline="")))
    except UnicodeError as error:
        raise ValueError(f"table is not ASCII: {path.name}") from error
    if not rows or list(rows[0]) != expected_fields:
        raise ValueError(f"table schema is invalid: {path.name}")
    return rows


def _float(value: str, label: str) -> float:
    try:
        numeric = float(value)
    except ValueError as error:
        raise ValueError(f"{label} must be finite") from error
    if not math.isfinite(numeric):
        raise ValueError(f"{label} must be finite")
    return numeric


def _audit_pair_completeness(summary: list[dict[str, object]]) -> None:
    expected = set(HELD_OUT_SEEDS)
    heldout = [row for row in summary if row["matrix"] == "heldout"]
    if not heldout:
        raise ValueError("audit requires heldout matrix evidence")
    scenarios = sorted({str(row["scenario_id"]) for row in heldout})
    for scenario in scenarios:
        for policy in ("tmc_db_lbt", "adaptive_db_lbt"):
            seeds = [
                int(row["seed"])
                for row in heldout
                if row["scenario_id"] == scenario and row["policy"] == policy
            ]
            if len(seeds) != len(expected) or set(seeds) != expected:
                raise ValueError(
                    f"paired seed completeness failed for {scenario}/{policy}"
                )


def _audit_model_provenance(
    model_path: Path,
    runs: list[PlotRun],
    *,
    oracle_arm_file: Path | None = None,
) -> tuple[int, str, str]:
    if not model_path.is_file():
        raise ValueError("model provenance input is missing")
    grid_hash = action_grid_hash()
    agent = LinUCB.load(model_path, expected_action_grid_hash=grid_hash)
    model_hash = file_sha256(model_path)
    state_hash = linucb_state_sha256(agent)
    oracle = None
    oracle_hash = None
    if oracle_arm_file is not None:
        oracle = load_oracle_arm(oracle_arm_file, model_path=model_path)
        oracle_hash = file_sha256(oracle_arm_file)
    for run in runs:
        job = run.job
        if job.policy not in {"adaptive_db_lbt", "fixed_oracle"}:
            continue
        provenance = run.execution_provenance
        if job.policy == "adaptive_db_lbt":
            if job.ablation == "context_free_ucb":
                if provenance.mode != "context_free_builtin":
                    raise ValueError(
                        "context_free_ucb evidence requires "
                        "context_free_builtin provenance"
                    )
                continue
            if job.matrix not in {"smoke", "pretrain"}:
                if provenance.mode != "adaptive_model":
                    raise ValueError(
                        "formal adaptive evidence requires adaptive_model provenance"
                    )
            elif provenance.mode == "adaptive_blank":
                continue
            if provenance.mode != "adaptive_model":
                raise ValueError("adaptive evidence has invalid provenance mode")
        elif provenance.mode != "fixed_oracle":
            raise ValueError("fixed Oracle evidence has invalid provenance mode")
        if (
            provenance.model_file_sha256 != model_hash
            or provenance.agent_state_sha256 != state_hash
        ):
            raise ValueError("run model provenance does not match audited model")
        if (
            job.policy == "fixed_oracle"
            and provenance.oracle_model_sha256 != model_hash
        ):
            raise ValueError("fixed Oracle model provenance mismatch")
        if job.policy == "fixed_oracle":
            if oracle is None or oracle_hash is None:
                raise ValueError("fixed Oracle evidence requires an Oracle artifact")
            if (
                provenance.oracle_artifact_sha256 != oracle_hash
                or provenance.source_matrix_sha256
                != oracle.source_matrix_hash
                or provenance.oracle_arm != oracle.arm
                or provenance.oracle_model_sha256 != oracle.model_sha256
            ):
                raise ValueError(
                    "fixed Oracle provenance does not match audited artifact"
                )
    return model_path.stat().st_size, model_hash, grid_hash


def audit_report(
    manifest_dir: str | Path,
    summary_path: str | Path,
    output_dir: str | Path,
    model_path: str | Path | None,
    oracle_arm_file: str | Path | None = None,
    *,
    workers: int = 1,
) -> AuditResult:
    """Audit all source artifacts, paired evidence, figures, and tables read-only."""
    manifests = Path(manifest_dir).resolve(strict=False)
    summary_source = Path(summary_path).resolve(strict=False)
    if workers == 1:
        summary, runs = validated_plot_inputs(summary_source, manifests)
    else:
        summary, runs = validated_plot_inputs(
            summary_source, manifests, workers=workers
        )
    mode = validate_report_inventory(
        summary, {path.stem for path in manifests.glob("*.json")}
    )
    if mode == "formal" and model_path is None:
        raise ValueError("formal audit mode requires --model")
    if mode == "formal" and oracle_arm_file is None:
        raise ValueError("formal audit mode requires --oracle-arm-file")
    output = Path(output_dir).resolve(strict=False)
    figure_paths = [
        output / f"{name}.{suffix}"
        for name in FIGURE_NAMES
        for suffix in ("pdf", "png")
    ]
    for path in figure_paths:
        if not path.is_file() or path.stat().st_size <= 100:
            raise ValueError(f"missing or empty figure: {path.name}")
        payload = path.read_bytes()[:8]
        if path.suffix == ".pdf" and not payload.startswith(b"%PDF-"):
            raise ValueError(f"corrupt PDF figure: {path.name}")
        if path.suffix == ".png" and payload != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"corrupt PNG figure: {path.name}")
    if mode == "smoke":
        return AuditResult(
            run_count=len(summary), figure_count=len(figure_paths), table_count=0
        )
    _audit_pair_completeness(summary)
    model = Path(model_path).resolve(strict=False)
    oracle_path = Path(oracle_arm_file).resolve(strict=False)
    model_bytes, model_hash, grid_hash = _audit_model_provenance(
        model, runs, oracle_arm_file=oracle_path
    )
    tables = output / "tables"
    overhead_fields = [
        "model_path", "model_state_bytes", "model_sha256",
        "action_grid_hash", "warmup_calls", "measurement_calls",
        "median_us", "p95_us",
    ]
    overhead = _csv_rows(tables / "overhead.csv", overhead_fields)
    if len(overhead) != 1:
        raise ValueError("overhead table must contain one row")
    row = overhead[0]
    median_us = _float(row["median_us"], "median latency")
    p95_us = _float(row["p95_us"], "P95 latency")
    if (
        Path(row["model_path"]).resolve(strict=False) != model
        or int(row["model_state_bytes"]) != model_bytes
        or row["model_sha256"] != model_hash
        or row["action_grid_hash"] != grid_hash
        or int(row["warmup_calls"]) != 100
        or int(row["measurement_calls"]) != 10_000
        or median_us < 0
        or p95_us < median_us
    ):
        raise ValueError("overhead model provenance is invalid")
    expected_overhead = ModelOverhead(
        model_path=str(model),
        model_state_bytes=model_bytes,
        model_sha256=model_hash,
        action_grid_hash=grid_hash,
        warmup_calls=100,
        measurement_calls=10_000,
        median_us=median_us,
        p95_us=p95_us,
    )
    expected_payloads = table_payloads(summary, expected_overhead)
    for name, payload in expected_payloads.items():
        path = tables / name
        if not path.is_file() or path.read_bytes() != payload:
            raise ValueError(f"table bytes do not match audited evidence: {name}")
    return AuditResult(
        run_count=len(summary), figure_count=len(figure_paths), table_count=8
    )
