"""Validated stable per-run CSV reporting."""

from __future__ import annotations

import csv
from concurrent.futures import ProcessPoolExecutor
import io
from itertools import repeat
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from .experiment import JobSpec, artifact_paths, canonical_json
from .records import aggregate_rows, iter_job_rows
from .workflows import effective_worker_count


_RUN_ID = re.compile(r"[0-9a-f]{16}\Z")
_FIELDS = [
    "run_id",
    "matrix",
    "scenario_id",
    "policy",
    "seed",
    "ablation",
    "arm_id",
    "wifi_nodes",
    "nru_nodes",
    "traffic",
    "interference_interval_ms",
    "interruption_std",
    "join_interval_rounds",
    "lifetime_rounds",
    "config_hash",
    "rounds",
    "elapsed_us",
    "successes",
    "collisions",
    "collision_probability",
    "effective_airtime",
    "mean_delay_us",
    "p95_delay_us",
    "jain_fairness",
    "evaluation_utility",
    "decision_count",
    "switch_count",
    "training_sample_count",
]


def _reject_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _load_job_config(path: Path) -> JobSpec:
    raw = path.read_bytes()
    value = json.loads(
        raw.decode("utf-8"),
        parse_constant=_reject_constant,
        object_pairs_hook=_reject_duplicate_keys,
    )
    if type(value) is not dict:
        raise ValueError("job config root must be an object")
    job = JobSpec.model_validate(value)
    if raw != (canonical_json(job) + "\n").encode("ascii"):
        raise ValueError("job config sidecar is not canonical JSON")
    return job


def _atomic_write(path: Path, payload: bytes) -> None:
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


def _summarize_manifest(
    manifest_path: Path, run_root: Path
) -> tuple[dict[str, object], tuple[Path, Path, Path, Path]]:
    run_id = manifest_path.stem
    if _RUN_ID.fullmatch(run_id) is None:
        raise ValueError(f"manifest filename is not a run_id: {manifest_path.name}")
    raw_manifest = json.loads(
        manifest_path.read_text(encoding="utf-8"),
        parse_constant=_reject_constant,
        object_pairs_hook=_reject_duplicate_keys,
    )
    if type(raw_manifest) is not dict or raw_manifest.get("run_id") != run_id:
        raise ValueError("manifest filename and run_id do not match")
    config_path = run_root / "configs" / f"{run_id}.json"
    job = _load_job_config(config_path)
    if job.run_id != run_id:
        raise ValueError("job config run_id does not match manifest")
    expected = artifact_paths(job, run_root)
    if expected.manifest != manifest_path:
        raise ValueError("manifest is not at its expected sibling path")
    aggregate = aggregate_rows(iter_job_rows(job, run_root))
    return (
        {
            "run_id": job.run_id,
            "matrix": job.matrix,
            "scenario_id": job.scenario.id,
            "policy": job.policy,
            "seed": job.seed,
            "ablation": "" if job.ablation is None else job.ablation,
            "arm_id": "" if job.arm_id is None else job.arm_id,
            "wifi_nodes": job.scenario.wifi_nodes,
            "nru_nodes": job.scenario.nru_nodes,
            "traffic": job.scenario.traffic,
            "interference_interval_ms": (
                ""
                if job.scenario.interference_interval_ms is None
                else job.scenario.interference_interval_ms
            ),
            "interruption_std": job.scenario.interruption_std,
            "join_interval_rounds": (
                ""
                if job.scenario.join_interval_rounds is None
                else job.scenario.join_interval_rounds
            ),
            "lifetime_rounds": (
                ""
                if job.scenario.lifetime_rounds is None
                else job.scenario.lifetime_rounds
            ),
            "config_hash": job.config_hash,
            "rounds": aggregate.rounds,
            "elapsed_us": aggregate.elapsed_us,
            "successes": aggregate.successes,
            "collisions": aggregate.collisions,
            "collision_probability": aggregate.collision_probability,
            "effective_airtime": aggregate.effective_airtime,
            "mean_delay_us": aggregate.mean_delay_us,
            "p95_delay_us": aggregate.p95_delay_us,
            "jain_fairness": aggregate.fairness,
            "evaluation_utility": aggregate.evaluation_utility,
            "decision_count": aggregate.decision_count,
            "switch_count": aggregate.switch_count,
            "training_sample_count": aggregate.training_sample_count,
        },
        (expected.raw, expected.marker, expected.manifest, config_path),
    )


def summarize_manifests(
    manifest_dir: str | Path,
    output: str | Path,
    *,
    workers: int = 1,
) -> list[dict[str, object]]:
    """Validate complete runs and atomically write a run-id-sorted CSV."""
    max_workers = effective_worker_count(workers)
    manifests = Path(manifest_dir).resolve(strict=False)
    if not manifests.is_dir():
        raise ValueError("manifest-dir must be an existing directory")
    run_root = manifests.parent
    entries = sorted(manifests.glob("*.json"), key=lambda path: path.name)
    if not entries:
        raise ValueError("manifest-dir contains no JSON manifests")

    if max_workers == 1:
        results = (
            _summarize_manifest(manifest_path, run_root)
            for manifest_path in entries
        )
        collected = list(results)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            collected = list(
                executor.map(
                    _summarize_manifest,
                    entries,
                    repeat(run_root),
                )
            )

    rows: list[dict[str, object]] = []
    protected_inputs: set[Path] = set()
    for row, protected in collected:
        rows.append(row)
        protected_inputs.update(protected)

    rows.sort(key=lambda row: str(row["run_id"]))
    text = io.StringIO(newline="")
    writer = csv.DictWriter(
        text,
        fieldnames=_FIELDS,
        lineterminator="\n",
        extrasaction="raise",
    )
    writer.writeheader()
    writer.writerows(rows)
    target = Path(output).resolve(strict=False)
    if target in protected_inputs:
        raise ValueError("summary output cannot overwrite raw or provenance inputs")
    _atomic_write(target, text.getvalue().encode("ascii"))
    return rows
