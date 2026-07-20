"""Canonical smoke and shared-root formal report inventories."""

from __future__ import annotations

from collections.abc import Mapping, Sequence, Set
from importlib.resources import as_file, files
from pathlib import Path

from .experiment import JobSpec, MatrixSpec, expand_matrix, load_matrix


_SOURCE_MATRIX_DIRECTORY = (
    Path(__file__).resolve().parents[2] / "configs" / "matrices"
)
_FORMAL_MATRICES = ("reproduction", "heldout", "ablation")


def _load_canonical_matrix(name: str) -> MatrixSpec:
    resource = files("dblbt_fcn").joinpath("matrices", f"{name}.yaml")
    if resource.is_file():
        with as_file(resource) as path:
            return load_matrix(path)
    source = _SOURCE_MATRIX_DIRECTORY / f"{name}.yaml"
    if source.is_file():
        return load_matrix(source)
    raise FileNotFoundError(f"canonical report matrix is unavailable: {name}")


def canonical_report_inventory() -> dict[str, tuple[JobSpec, ...]]:
    """Load the versioned canonical jobs allowed in report roots."""
    smoke = tuple(expand_matrix(_load_canonical_matrix("smoke")))
    formal = tuple(
        sorted(
            (
                job
                for matrix in _FORMAL_MATRICES
                for job in expand_matrix(_load_canonical_matrix(matrix))
            ),
            key=lambda job: job.run_id,
        )
    )
    if len(smoke) != 3 or len(formal) != 940:
        raise RuntimeError("canonical report matrix inventory has changed")
    return {"smoke": smoke, "formal": formal}


def validate_report_inventory(
    summary_rows: Sequence[Mapping[str, object]], manifest_ids: Set[str]
) -> str:
    """Return the report mode after exact summary/manifest inventory checks."""
    matrices = {str(row["matrix"]) for row in summary_rows}
    if matrices == {"smoke"}:
        mode = "smoke"
    elif matrices == set(_FORMAL_MATRICES):
        mode = "formal"
    else:
        raise ValueError(
            "report matrix set must be smoke only or one shared formal root "
            "containing reproduction, heldout, and ablation"
        )
    expected = {
        job.run_id for job in canonical_report_inventory()[mode]
    }
    summary_ids = {str(row["run_id"]) for row in summary_rows}
    missing = expected - summary_ids
    extra = summary_ids - expected
    if missing or extra:
        raise ValueError(
            f"{mode} report inventory mismatch: "
            f"missing={len(missing)} extra={len(extra)}"
        )
    if set(manifest_ids) != expected:
        raise ValueError(
            f"{mode} manifest inventory must match the complete shared root"
        )
    return mode
