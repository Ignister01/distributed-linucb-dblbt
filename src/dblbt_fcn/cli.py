"""Command-line entry point for the DB-LBT experiment package."""

from pathlib import Path

import typer

from .experiment import artifact_paths, expand_matrix, load_job, load_matrix
from .linucb import LinUCB
from .provenance import execution_provenance, file_sha256
from .simulation import run_job
from .reporting import summarize_manifests
from .plotting import generate_report
from .audit import audit_report
from .cross_validation import (
    cross_model_consistency,
    load_ns3_scenario_metrics,
    write_cross_model_evidence,
)
from .stats import load_summary
from .workflows import (
    action_grid_hash,
    build_pretraining,
    effective_worker_count,
    load_oracle_arm,
    run_sweep,
)

app = typer.Typer(help="Run reproducible DB-LBT experiments.")


@app.callback()
def main() -> None:
    """Run reproducible DB-LBT experiments."""


def _fail(error: Exception) -> None:
    typer.echo(f"error: {error}", err=True)
    raise typer.Exit(code=1) from error


@app.command("validate-config")
def validate_config(path: Path) -> None:
    """Strictly validate one matrix or single-job configuration."""
    try:
        matrix_error: Exception | None = None
        try:
            matrix = load_matrix(path)
        except Exception as error:
            matrix_error = error
        else:
            typer.echo(
                f"type=matrix name={matrix.name} jobs={len(expand_matrix(matrix))}"
            )
            return

        try:
            job = load_job(path)
        except Exception as job_error:
            if "matrix" not in str(job_error).lower() and matrix_error is not None:
                job_error.add_note(f"matrix validation also failed: {matrix_error}")
            raise job_error
        typer.echo(f"type=job run_id={job.run_id} jobs=1")
    except Exception as error:
        _fail(error)


@app.command("simulate")
def simulate(
    config: Path = typer.Option(..., "--config"),
    output_dir: Path = typer.Option(..., "--output-dir"),
    model: Path | None = typer.Option(None, "--model"),
    oracle_arm_file: Path | None = typer.Option(None, "--oracle-arm-file"),
) -> None:
    """Run one strictly validated JobSpec."""
    try:
        job = load_job(config)
        initial_agent: LinUCB | None = None
        if job.policy == "adaptive_db_lbt":
            if model is None and job.matrix != "smoke":
                raise ValueError(
                    "non-smoke adaptive jobs require --model"
                )
            if model is not None:
                initial_agent = LinUCB.load(
                    model,
                    expected_action_grid_hash=action_grid_hash(),
                )
        if job.policy == "fixed_oracle" and oracle_arm_file is None:
            raise ValueError("fixed_oracle jobs require --oracle-arm-file")
        if job.policy == "fixed_oracle" and model is None:
            raise ValueError("fixed_oracle jobs require --model")
        oracle = (
            None
            if oracle_arm_file is None
            else load_oracle_arm(oracle_arm_file, model_path=model)
        )
        oracle_arm = None if oracle is None else oracle.arm
        if job.policy == "fixed_oracle" and model is not None:
            initial_agent = LinUCB.load(
                model, expected_action_grid_hash=action_grid_hash()
            )
        execution = execution_provenance(
            job,
            initial_agent=initial_agent,
            model_path=model,
            oracle_arm=oracle_arm,
            oracle_artifact_sha256=(
                file_sha256(oracle_arm_file)
                if oracle is not None and oracle_arm_file is not None
                else None
            ),
            oracle_model_sha256=(
                oracle.model_sha256 if oracle is not None else None
            ),
            source_matrix_sha256=(
                oracle.source_matrix_hash if oracle is not None else None
            ),
        )
        manifest = run_job(
            job,
            output_dir,
            initial_agent=initial_agent,
            oracle_arm=oracle_arm,
            model_path=model,
            oracle_artifact_path=oracle_arm_file,
            execution=execution,
        )
        typer.echo(
            f"run_id={manifest.run_id} "
            f"manifest={artifact_paths(job, output_dir).manifest}"
        )
    except Exception as error:
        _fail(error)


@app.command("sweep")
def sweep(
    matrix: Path = typer.Option(..., "--matrix"),
    workers: int = typer.Option(..., "--workers"),
    output_dir: Path | None = typer.Option(None, "--output-dir"),
    model: Path | None = typer.Option(None, "--model"),
    oracle_arm_file: Path | None = typer.Option(None, "--oracle-arm-file"),
) -> None:
    """Run an experiment matrix with process-level parallelism."""
    try:
        specification = load_matrix(matrix)
        effective_worker_count(workers)
        root = (
            Path("runs") / specification.name
            if output_dir is None
            else output_dir
        )
        results = run_sweep(
            specification,
            root,
            workers=workers,
            model_path=model,
            oracle_arm_path=oracle_arm_file,
        )
        for manifest in sorted(results, key=lambda item: item.run_id):
            typer.echo(f"run_id={manifest.run_id}")
        typer.echo(f"completed={len(results)}")
    except Exception as error:
        _fail(error)


@app.command("pretrain")
def pretrain(
    matrix: Path = typer.Option(..., "--matrix"),
    output: Path = typer.Option(
        Path("models/linucb-initial.npz"), "--output"
    ),
    workers: int = typer.Option(1, "--workers"),
    output_dir: Path | None = typer.Option(None, "--output-dir"),
    oracle_output: Path | None = typer.Option(None, "--oracle-output"),
) -> None:
    """Build the initial LinUCB model and fixed Oracle arm."""
    try:
        specification = load_matrix(matrix)
        effective_worker_count(workers)
        root = (
            Path("runs") / specification.name
            if output_dir is None
            else output_dir
        )
        oracle_path = (
            output.with_name("fixed-oracle-arm.json")
            if oracle_output is None
            else oracle_output
        )
        _, arm = build_pretraining(
            specification,
            root,
            output,
            oracle_path,
            workers=workers,
        )
        typer.echo(f"model={output} oracle={oracle_path} arm={arm}")
    except Exception as error:
        _fail(error)


@app.command("summarize")
def summarize(
    manifest_dir: Path = typer.Option(..., "--manifest-dir"),
    output: Path = typer.Option(..., "--output"),
    workers: int = typer.Option(1, "--workers"),
) -> None:
    """Create a validated basic per-run CSV summary."""
    try:
        rows = summarize_manifests(manifest_dir, output, workers=workers)
        typer.echo(f"output={output} rows={len(rows)}")
    except Exception as error:
        _fail(error)


@app.command("plot")
def plot(
    summary: Path = typer.Option(..., "--summary"),
    output_dir: Path = typer.Option(..., "--output-dir"),
    manifest_dir: Path = typer.Option(..., "--manifest-dir"),
    model: Path | None = typer.Option(None, "--model"),
    oracle_arm_file: Path | None = typer.Option(None, "--oracle-arm-file"),
    workers: int = typer.Option(1, "--workers"),
) -> None:
    """Generate the fixed validated figure and evidence-table report."""
    try:
        if workers == 1:
            outputs = generate_report(
                summary, output_dir, manifest_dir, model, oracle_arm_file
            )
        else:
            outputs = generate_report(
                summary,
                output_dir,
                manifest_dir,
                model,
                oracle_arm_file,
                workers=workers,
            )
        typer.echo(f"output-dir={output_dir} files={len(outputs)}")
    except Exception as error:
        _fail(error)


@app.command("audit")
def audit(
    manifest_dir: Path = typer.Option(..., "--manifest-dir"),
    summary: Path = typer.Option(..., "--summary"),
    output_dir: Path = typer.Option(..., "--output-dir"),
    model: Path | None = typer.Option(None, "--model"),
    oracle_arm_file: Path | None = typer.Option(None, "--oracle-arm-file"),
    workers: int = typer.Option(1, "--workers"),
) -> None:
    """Audit complete source artifacts and Task 12 evidence read-only."""
    try:
        if workers == 1:
            result = audit_report(
                manifest_dir,
                summary,
                output_dir,
                model,
                oracle_arm_file,
            )
        else:
            result = audit_report(
                manifest_dir,
                summary,
                output_dir,
                model,
                oracle_arm_file,
                workers=workers,
            )
        typer.echo(f"audited={result.run_count}")
    except Exception as error:
        _fail(error)


@app.command("cross-validate")
def cross_validate(
    event_summary: Path = typer.Option(..., "--event-summary"),
    ns3_metrics: Path = typer.Option(..., "--ns3-metrics"),
    ns3_reduction: Path = typer.Option(..., "--ns3-reduction"),
    event_hypotheses: Path = typer.Option(..., "--event-hypotheses"),
    output_dir: Path = typer.Option(..., "--output-dir"),
) -> None:
    """Evaluate and write audited event-versus-ns-3 H5 evidence."""
    try:
        event_rows = load_summary(event_summary)
        ns3_rows = load_ns3_scenario_metrics(ns3_metrics, ns3_reduction)
        report = cross_model_consistency(event_rows, ns3_rows)
        outputs = write_cross_model_evidence(
            report,
            output_dir,
            event_summary_path=event_summary,
            ns3_metrics_path=ns3_metrics,
            ns3_reduction_path=ns3_reduction,
            event_hypotheses_path=event_hypotheses,
        )
        typer.echo(f"h5_status={report.h5_status} files={len(outputs)}")
    except Exception as error:
        _fail(error)
