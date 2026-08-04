"""Command-line interface tests."""

from importlib import import_module
import json
from pathlib import Path
import re

import pytest
from typer.testing import CliRunner

from dblbt_fcn.experiment import JobSpec
from dblbt_fcn.linucb import LinUCB


COMMANDS = {
    "validate-config",
    "simulate",
    "sweep",
    "pretrain",
    "summarize",
    "regime-report",
    "regime-rank",
    "adaptation-report",
    "regime-confirmation",
    "plot",
    "audit",
    "cross-validate",
}

_ANSI_SGR = re.compile(r"\x1b\[[0-9;]*m")


def _job_payload(*, matrix: str = "smoke", policy: str = "random_lbt") -> dict[str, object]:
    return {
        "matrix": matrix,
        "rounds": 2,
        "alpha": 11,
        "timing": {
            "slot_us": 1,
            "tx_us": 2_000,
            "wifi_ack_us": 0,
            "nru_sync_us": 250,
        },
        "scenario": {
            "id": "tiny",
            "wifi_nodes": 1,
            "nru_nodes": 1,
            "legacy_ap_nodes": 0,
            "legacy_sta_nodes": 0,
            "traffic": "saturated",
            "poisson_rate_packets_ms": None,
            "interference_interval_ms": None,
            "interference_duration_us": None,
            "interruption_std": 0.0,
            "join_interval_rounds": None,
            "lifetime_rounds": None,
            "trace": False,
        },
        "policy": policy,
        "seed": 410,
        "arm_id": None,
        "ablation": None,
    }


def _matrix_payload() -> dict[str, object]:
    job = _job_payload()
    return {
        "version": 1,
        "name": "tiny-matrix",
        "rounds": job["rounds"],
        "alpha": job["alpha"],
        "timing": job["timing"],
        "seeds": [job["seed"]],
        "policies": [job["policy"]],
        "conditions": [],
        "arm_ids": [],
        "scenarios": [job["scenario"]],
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_cli_help_exits_successfully() -> None:
    cli = import_module("dblbt_fcn.cli")

    result = CliRunner().invoke(cli.app, ["--help"])

    assert result.exit_code == 0, result.output
    listed = {command.name for command in cli.app.registered_commands}
    assert listed == COMMANDS
    assert all(command in result.output for command in COMMANDS)


@pytest.mark.parametrize("kind", ["matrix", "job"])
def test_validate_config_reports_stable_identity(
    tmp_path: Path, kind: str
) -> None:
    cli = import_module("dblbt_fcn.cli")
    path = tmp_path / f"{kind}.json"
    if kind == "matrix":
        _write_json(path, _matrix_payload())
    else:
        _write_json(path, _job_payload())

    result = CliRunner().invoke(cli.app, ["validate-config", str(path)])

    assert result.exit_code == 0, result.output
    if kind == "matrix":
        assert result.output == "type=matrix name=tiny-matrix jobs=1\n"
    else:
        expected = JobSpec.model_validate(_job_payload())
        assert result.output == (
            f"type=job run_id={expected.run_id} jobs=1\n"
        )


@pytest.mark.parametrize(
    "payload",
    [
        {**_job_payload(), "rounds": "2"},
        {**_job_payload(), "unknown": True},
    ],
)
def test_validate_config_rejects_coercion_and_unknown_keys(
    tmp_path: Path, payload: dict[str, object]
) -> None:
    cli = import_module("dblbt_fcn.cli")
    path = tmp_path / "bad.json"
    _write_json(path, payload)

    result = CliRunner().invoke(cli.app, ["validate-config", str(path)])

    assert result.exit_code != 0
    assert "error:" in result.output.lower()
    assert result.exception is not None


def test_validate_config_rejects_duplicate_yaml_keys(tmp_path: Path) -> None:
    cli = import_module("dblbt_fcn.cli")
    path = tmp_path / "duplicate.yaml"
    path.write_text("matrix: smoke\nmatrix: smoke\n", encoding="utf-8")

    result = CliRunner().invoke(cli.app, ["validate-config", str(path)])

    assert result.exit_code != 0
    assert "duplicate YAML mapping key" in result.output


def test_simulate_routes_a_smoke_job_and_prints_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = import_module("dblbt_fcn.cli")
    config = tmp_path / "job.json"
    output = tmp_path / "runs"
    _write_json(config, _job_payload())
    observed: dict[str, object] = {}

    class Manifest:
        run_id = JobSpec.model_validate(_job_payload()).run_id

    def fake_run_job(job: JobSpec, output_dir: Path, **kwargs: object) -> Manifest:
        observed.update(job=job, output_dir=output_dir, kwargs=kwargs)
        return Manifest()

    monkeypatch.setattr(cli, "run_job", fake_run_job)

    result = CliRunner().invoke(
        cli.app,
        ["simulate", "--config", str(config), "--output-dir", str(output)],
    )

    assert result.exit_code == 0, result.output
    assert observed["output_dir"] == output
    assert result.output == (
        f"run_id={Manifest.run_id} manifest={output.resolve() / 'manifests' / (Manifest.run_id + '.json')}\n"
    )


def test_simulate_requires_model_for_non_smoke_adaptive_job(
    tmp_path: Path,
) -> None:
    cli = import_module("dblbt_fcn.cli")
    config = tmp_path / "formal.json"
    _write_json(
        config,
        _job_payload(matrix="heldout", policy="adaptive_db_lbt"),
    )

    result = CliRunner().invoke(
        cli.app,
        ["simulate", "--config", str(config), "--output-dir", str(tmp_path / "runs")],
    )

    assert result.exit_code != 0
    assert "--model" in result.output


def test_simulate_loads_model_with_grid_hash_and_oracle_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = import_module("dblbt_fcn.cli")
    config = tmp_path / "fixed.json"
    payload = _job_payload(policy="fixed_oracle")
    _write_json(config, payload)
    output = tmp_path / "runs"
    oracle = tmp_path / "oracle.json"
    oracle.write_text("unused", encoding="ascii")
    model = tmp_path / "model.npz"
    model.write_bytes(b"model")
    observed: dict[str, object] = {}

    def fake_oracle(path: Path, *, model_path: Path) -> object:
        from dblbt_fcn.provenance import file_sha256

        observed.update(oracle_path=path, oracle_model_path=model_path)
        return type(
            "Oracle",
            (),
            {
                "arm": 9,
                "model_sha256": file_sha256(model_path),
                "source_matrix_hash": "b" * 64,
            },
        )()

    def fake_run_job(job: JobSpec, output_dir: Path, **kwargs: object):
        observed.update(kwargs)
        return type("Manifest", (), {"run_id": job.run_id})()

    monkeypatch.setattr(cli, "load_oracle_arm", fake_oracle)
    monkeypatch.setattr(cli.LinUCB, "load", lambda *args, **kwargs: LinUCB(24, 11))
    monkeypatch.setattr(cli, "run_job", fake_run_job)

    result = CliRunner().invoke(
        cli.app,
        [
            "simulate",
            "--config",
            str(config),
            "--output-dir",
            str(output),
            "--oracle-arm-file",
            str(oracle),
            "--model",
            str(model),
        ],
    )

    assert result.exit_code == 0, result.output
    assert observed["oracle_path"] == oracle
    assert observed["oracle_model_path"] == model
    assert observed["oracle_arm"] == 9
    assert observed["oracle_artifact_path"] == oracle
    assert observed["model_path"] == model
    assert observed["execution"].oracle_arm == 9
    assert observed["execution"].oracle_artifact_sha256 is not None
    assert observed["execution"].oracle_model_sha256 == (
        observed["execution"].model_file_sha256
    )
    assert observed["execution"].source_matrix_sha256 == "b" * 64


def test_simulate_fixed_oracle_requires_model(tmp_path: Path) -> None:
    cli = import_module("dblbt_fcn.cli")
    config = tmp_path / "fixed.json"
    _write_json(config, _job_payload(policy="fixed_oracle"))
    oracle = tmp_path / "oracle.json"
    oracle.write_text("unused", encoding="ascii")

    result = CliRunner().invoke(
        cli.app,
        [
            "simulate",
            "--config",
            str(config),
            "--output-dir",
            str(tmp_path / "runs"),
            "--oracle-arm-file",
            str(oracle),
        ],
    )

    assert result.exit_code != 0
    assert "--model" in result.output


def test_simulate_fixed_oracle_records_actual_provenance(tmp_path: Path) -> None:
    from dblbt_fcn.experiment import canonical_json, load_completed_job_manifest
    from dblbt_fcn.provenance import file_sha256
    from dblbt_fcn.workflows import action_grid_hash

    cli = import_module("dblbt_fcn.cli")
    payload = _job_payload(policy="fixed_oracle")
    config = tmp_path / "fixed.json"
    _write_json(config, payload)
    job = JobSpec.model_validate(payload)
    source = {
        "version": 1,
        "name": "cli-oracle-source",
        "rounds": 1,
        "alpha": 11,
        "timing": payload["timing"],
        "seeds": [1103, 2207, 3301],
        "policies": ["pretrain_arm"],
        "conditions": [],
        "arm_ids": list(range(24)),
        "scenarios": [payload["scenario"]],
    }
    model = tmp_path / "model.npz"
    LinUCB(24, 11, action_grid_hash=action_grid_hash()).save(model)
    import hashlib

    source_hash = hashlib.sha256(
        canonical_json(source).encode("ascii")
    ).hexdigest()
    oracle = tmp_path / "oracle.json"
    oracle.write_bytes(
        (
            canonical_json(
                {
                    "schema_version": 1,
                    "arm": 9,
                    "action_grid_hash": action_grid_hash(),
                    "source_matrix": source,
                    "source_matrix_hash": source_hash,
                    "model_sha256": file_sha256(model),
                }
            )
            + "\n"
        ).encode("ascii")
    )
    output = tmp_path / "runs"

    result = CliRunner().invoke(
        cli.app,
        [
            "simulate",
            "--config",
            str(config),
            "--output-dir",
            str(output),
            "--model",
            str(model),
            "--oracle-arm-file",
            str(oracle),
        ],
    )

    assert result.exit_code == 0, result.output
    manifest = load_completed_job_manifest(job, output)
    provenance = manifest.execution_provenance
    assert provenance.oracle_arm == 9
    assert provenance.model_file_sha256 == file_sha256(model)
    assert provenance.oracle_model_sha256 == file_sha256(model)
    assert provenance.oracle_artifact_sha256 == file_sha256(oracle)
    assert provenance.source_matrix_sha256 == source_hash


def test_simulate_adaptive_model_load_uses_expected_grid_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = import_module("dblbt_fcn.cli")
    config = tmp_path / "adaptive.json"
    _write_json(
        config,
        _job_payload(matrix="heldout", policy="adaptive_db_lbt"),
    )
    model = tmp_path / "model.npz"
    model.write_bytes(b"model")
    observed: dict[str, object] = {}

    def fake_load(path: Path, **kwargs: object) -> LinUCB:
        observed.update(path=path, kwargs=kwargs)
        return LinUCB(24, 11)

    monkeypatch.setattr(cli.LinUCB, "load", fake_load)
    monkeypatch.setattr(
        cli,
        "run_job",
        lambda job, output_dir, **kwargs: type(
            "Manifest", (), {"run_id": job.run_id}
        )(),
    )

    result = CliRunner().invoke(
        cli.app,
        [
            "simulate",
            "--config",
            str(config),
            "--output-dir",
            str(tmp_path / "runs"),
            "--model",
            str(model),
        ],
    )

    assert result.exit_code == 0, result.output
    assert observed["path"] == model
    assert observed["kwargs"] == {
        "expected_action_grid_hash": cli.action_grid_hash()
    }


def test_cli_failure_boundary_does_not_swallow_keyboard_interrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = import_module("dblbt_fcn.cli")
    config = tmp_path / "job.json"
    _write_json(config, _job_payload())
    monkeypatch.setattr(
        cli, "load_matrix", lambda path: (_ for _ in ()).throw(KeyboardInterrupt())
    )

    result = CliRunner().invoke(cli.app, ["validate-config", str(config)])

    assert result.exit_code == 130
    assert "error:" not in result.output.lower()


def test_sweep_cli_uses_matrix_default_output_and_stable_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = import_module("dblbt_fcn.cli")
    matrix_path = tmp_path / "matrix.json"
    _write_json(matrix_path, _matrix_payload())
    observed: dict[str, object] = {}

    def fake_sweep(matrix: object, output: Path, **kwargs: object):
        observed.update(matrix=matrix, output=output, kwargs=kwargs)
        return [
            type("Manifest", (), {"run_id": "bbbbbbbbbbbbbbbb"})(),
            type("Manifest", (), {"run_id": "aaaaaaaaaaaaaaaa"})(),
        ]

    monkeypatch.setattr(cli, "run_sweep", fake_sweep, raising=False)

    result = CliRunner().invoke(
        cli.app, ["sweep", "--matrix", str(matrix_path), "--workers", "25"]
    )

    assert result.exit_code == 0, result.output
    assert observed["output"] == Path("runs") / "tiny-matrix"
    assert observed["kwargs"] == {
        "workers": 25,
        "model_path": None,
        "oracle_arm_path": None,
    }
    assert result.output == (
        "run_id=aaaaaaaaaaaaaaaa\nrun_id=bbbbbbbbbbbbbbbb\ncompleted=2\n"
    )


def test_sweep_cli_rejects_zero_workers(tmp_path: Path) -> None:
    cli = import_module("dblbt_fcn.cli")
    matrix_path = tmp_path / "matrix.json"
    _write_json(matrix_path, _matrix_payload())

    result = CliRunner().invoke(
        cli.app, ["sweep", "--matrix", str(matrix_path), "--workers", "0"]
    )

    assert result.exit_code != 0
    assert "workers" in result.output


def test_pretrain_cli_routes_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cli = import_module("dblbt_fcn.cli")
    matrix_path = tmp_path / "pretrain.json"
    payload = _matrix_payload()
    payload.update(
        name="tiny-pretrain",
        seeds=[1103, 2207, 3301],
        policies=["pretrain_arm"],
        arm_ids=list(range(24)),
    )
    _write_json(matrix_path, payload)
    observed: dict[str, object] = {}

    def fake_build(*args: object, **kwargs: object):
        observed.update(args=args, kwargs=kwargs)
        return LinUCB(24, 11), 3

    monkeypatch.setattr(cli, "build_pretraining", fake_build, raising=False)

    result = CliRunner().invoke(
        cli.app,
        ["pretrain", "--matrix", str(matrix_path), "--workers", "1"],
    )

    assert result.exit_code == 0, result.output
    assert observed["args"][1:] == (
        Path("runs") / "tiny-pretrain",
        Path("models/linucb-initial.npz"),
        Path("models/fixed-oracle-arm.json"),
    )
    assert observed["kwargs"] == {"workers": 1}
    assert result.output == (
        f"model={Path('models/linucb-initial.npz')} "
        f"oracle={Path('models/fixed-oracle-arm.json')} arm=3\n"
    )


def test_summarize_cli_routes_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = import_module("dblbt_fcn.cli")
    manifests = tmp_path / "manifests"
    output = tmp_path / "summary.csv"
    observed: dict[str, object] = {}

    def fake_summary(manifest_dir: Path, target: Path, *, workers: int):
        observed.update(
            manifest_dir=manifest_dir, target=target, workers=workers
        )
        return [{"run_id": "a"}, {"run_id": "b"}]

    monkeypatch.setattr(cli, "summarize_manifests", fake_summary, raising=False)

    result = CliRunner().invoke(
        cli.app,
        [
            "summarize",
            "--manifest-dir",
            str(manifests),
            "--output",
            str(output),
            "--workers",
            "2",
        ],
    )

    assert result.exit_code == 0, result.output
    assert observed == {
        "manifest_dir": manifests,
        "target": output,
        "workers": 2,
    }
    assert result.output == f"output={output} rows=2\n"


def test_regime_report_cli_writes_effects_and_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = import_module("dblbt_fcn.cli")
    summary = tmp_path / "summary.csv"
    effects = tmp_path / "effects.csv"
    selection = tmp_path / "selected.txt"
    observed: dict[str, object] = {}
    effect_rows = [object(), object()]

    monkeypatch.setattr(cli, "load_summary", lambda path: [str(path)])
    monkeypatch.setattr(
        cli, "scenario_effects", lambda rows: effect_rows, raising=False
    )
    monkeypatch.setattr(
        cli,
        "write_effects_csv",
        lambda rows, path: observed.update(effects=(rows, path)),
        raising=False,
    )
    monkeypatch.setattr(
        cli,
        "select_confirmation_scenarios",
        lambda rows: ("load-a",),
        raising=False,
    )
    monkeypatch.setattr(
        cli,
        "write_selected_scenarios",
        lambda rows, path: observed.update(selection=(rows, path)),
        raising=False,
    )

    result = CliRunner().invoke(
        cli.app,
        [
            "regime-report",
            "--summary",
            str(summary),
            "--effects-output",
            str(effects),
            "--selection-output",
            str(selection),
        ],
    )

    assert result.exit_code == 0, result.output
    assert observed == {
        "effects": (effect_rows, effects),
        "selection": (("load-a",), selection),
    }
    assert result.output == "effects=2 selected=1\n"


def test_regime_confirmation_cli_writes_generated_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = import_module("dblbt_fcn.cli")
    pilot_path = tmp_path / "pilot.yaml"
    selection_path = tmp_path / "selected.txt"
    output = tmp_path / "confirmation.yaml"
    observed: dict[str, object] = {}
    pilot = object()
    generated = type("Generated", (), {"scenarios": (object(), object())})()

    monkeypatch.setattr(cli, "load_matrix", lambda path: pilot)
    monkeypatch.setattr(
        cli,
        "load_selected_scenarios",
        lambda path: ("load-a", "occupancy-a"),
        raising=False,
    )
    def fake_confirmation(matrix: object, selected: object, **options: object):
        observed["confirmation"] = (matrix, selected, options)
        return generated

    monkeypatch.setattr(cli, "confirmation_matrix", fake_confirmation, raising=False)
    monkeypatch.setattr(
        cli,
        "write_confirmation_matrix",
        lambda matrix, path: observed.update(write=(matrix, path)),
        raising=False,
    )
    monkeypatch.setattr(cli, "expand_matrix", lambda matrix: [object()] * 60)

    result = CliRunner().invoke(
        cli.app,
        [
            "regime-confirmation",
            "--pilot-matrix",
            str(pilot_path),
            "--selection",
            str(selection_path),
            "--output",
            str(output),
            "--name",
            "large-effect-confirmation",
            "--rounds",
            "120000",
            "--seed",
            "6101",
            "--seed",
            "6113",
        ],
    )

    assert result.exit_code == 0, result.output
    assert observed == {
        "confirmation": (
            pilot,
            ("load-a", "occupancy-a"),
            {
                "name": "large-effect-confirmation",
                "rounds": 120_000,
                "seeds": (6101, 6113),
            },
        ),
        "write": (generated, output),
    }
    assert result.output == "output=" + str(output) + " scenarios=2 jobs=60\n"


def test_plot_and_audit_cli_forward_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = import_module("dblbt_fcn.cli")
    summary = tmp_path / "summary.csv"
    manifests = tmp_path / "manifests"
    output = tmp_path / "report"
    observed: dict[str, int] = {}

    def fake_report(*args: object, workers: int, **kwargs: object):
        observed["plot"] = workers
        return [output / "figure.png"]

    class AuditResult:
        run_count = 3

    def fake_audit(*args: object, workers: int, **kwargs: object):
        observed["audit"] = workers
        return AuditResult()

    monkeypatch.setattr(cli, "generate_report", fake_report, raising=False)
    monkeypatch.setattr(cli, "audit_report", fake_audit, raising=False)

    common = [
        "--summary",
        str(summary),
        "--output-dir",
        str(output),
        "--manifest-dir",
        str(manifests),
        "--workers",
        "2",
    ]
    plot = CliRunner().invoke(cli.app, ["plot", *common])
    audit = CliRunner().invoke(cli.app, ["audit", *common])

    assert plot.exit_code == 0, plot.output
    assert audit.exit_code == 0, audit.output
    assert observed == {"plot": 2, "audit": 2}


def test_cross_validate_cli_routes_audited_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = import_module("dblbt_fcn.cli")
    event = tmp_path / "event.csv"
    ns3_metrics = tmp_path / "scenario-metrics.csv"
    reduction = tmp_path / "reduction.json"
    hypotheses = tmp_path / "hypotheses.csv"
    output = tmp_path / "output"
    observed: dict[str, object] = {}

    monkeypatch.setattr(cli, "load_summary", lambda path: [str(path)])
    monkeypatch.setattr(
        cli,
        "load_ns3_scenario_metrics",
        lambda metrics, metadata: [str(metrics), str(metadata)],
    )

    class Report:
        h5_status = "inconclusive"

    def fake_consistency(event_rows: object, ns3_rows: object) -> Report:
        observed["rows"] = (event_rows, ns3_rows)
        return Report()

    def fake_write(report: object, target: Path, **inputs: object) -> list[Path]:
        observed["write"] = (report, target, inputs)
        return [target / "cross-model-audit.json"]

    monkeypatch.setattr(cli, "cross_model_consistency", fake_consistency)
    monkeypatch.setattr(cli, "write_cross_model_evidence", fake_write)
    result = CliRunner().invoke(
        cli.app,
        [
            "cross-validate",
            "--event-summary",
            str(event),
            "--ns3-metrics",
            str(ns3_metrics),
            "--ns3-reduction",
            str(reduction),
            "--event-hypotheses",
            str(hypotheses),
            "--output-dir",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert observed["rows"] == (
        [str(event)],
        [str(ns3_metrics), str(reduction)],
    )
    assert result.output == "h5_status=inconclusive files=1\n"


@pytest.mark.parametrize(
    ("command", "arguments"),
    [
        ("plot", ["--summary", "summary.csv", "--output-dir", "figures"]),
        (
            "audit",
            ["--manifest-dir", "runs/manifests", "--summary", "summary.csv"],
        ),
    ],
)
def test_task12_commands_fail_closed(command: str, arguments: list[str]) -> None:
    cli = import_module("dblbt_fcn.cli")

    result = CliRunner().invoke(cli.app, [command, *arguments])

    assert result.exit_code != 0
    assert "Task12 not implemented" not in result.output
    assert "Missing option" in result.output


def test_task12_model_option_is_optional_for_smoke_mode() -> None:
    cli = import_module("dblbt_fcn.cli")

    plot = CliRunner().invoke(cli.app, ["plot", "--help"], terminal_width=120)
    audit = CliRunner().invoke(cli.app, ["audit", "--help"], terminal_width=120)

    assert plot.exit_code == 0
    assert audit.exit_code == 0
    assert "required" not in next(
        line
        for line in _ANSI_SGR.sub("", plot.output).splitlines()
        if "--model" in line
    ).lower()
    assert "required" not in next(
        line
        for line in _ANSI_SGR.sub("", audit.output).splitlines()
        if "--model" in line
    ).lower()
