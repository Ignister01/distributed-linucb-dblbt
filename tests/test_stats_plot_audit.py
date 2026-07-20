"""Task 12 paired evidence, report output, and audit contracts."""

from __future__ import annotations

import csv
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path

import pytest

from dblbt_fcn.training import HELD_OUT_SEEDS


SEEDS = tuple(sorted(HELD_OUT_SEEDS))


def test_plotting_uses_ieee_compatible_embedded_fonts() -> None:
    import matplotlib
    import dblbt_fcn.plotting  # noqa: F401

    assert matplotlib.rcParams["pdf.fonttype"] == 42
    assert matplotlib.rcParams["ps.fonttype"] == 42


def pairs(values: list[float]) -> list[tuple[int, float]]:
    return [(SEEDS[index], value) for index, value in enumerate(values)]


def test_paired_bootstrap_is_deterministic_and_reports_registered_fields() -> None:
    from dblbt_fcn.stats import paired_bootstrap

    baseline = pairs([1.0 + index / 100 for index in range(10)])
    adaptive = pairs([value + 0.1 for _, value in baseline])

    first = paired_bootstrap(baseline, adaptive)
    second = paired_bootstrap(baseline, adaptive)

    assert first == second
    assert first.baseline_mean == pytest.approx(1.045)
    assert first.adaptive_mean == pytest.approx(1.145)
    assert first.paired_difference == pytest.approx(0.1)
    assert first.relative_difference == pytest.approx(0.1 / 1.045)
    assert first.lower_95 == pytest.approx(0.1)
    assert first.upper_95 == pytest.approx(0.1)
    assert first.decision == "improvement"
    assert first.resamples == 10_000
    assert first.bootstrap_seed == 20260715


@pytest.mark.parametrize(
    ("baseline", "adaptive", "match"),
    [
        (pairs([1.0] * 9), pairs([1.1] * 10), "seed"),
        (
            pairs([1.0] * 10) + [(SEEDS[0], 1.0)],
            pairs([1.1] * 10),
            "duplicate",
        ),
        (pairs([1.0] * 10) + [(9999, 1.0)], pairs([1.1] * 10), "seed"),
        (pairs([1.0] * 10), pairs([1.1] * 9), "seed"),
        (
            pairs([1.0] * 9 + [math.nan]),
            pairs([1.1] * 10),
            "finite",
        ),
    ],
)
def test_paired_bootstrap_rejects_invalid_pairs(
    baseline: list[tuple[int, float]],
    adaptive: list[tuple[int, float]],
    match: str,
) -> None:
    from dblbt_fcn.stats import paired_bootstrap

    with pytest.raises(ValueError, match=match):
        paired_bootstrap(baseline, adaptive)


@pytest.mark.parametrize(
    "kwargs",
    [{"resamples": 9_999}, {"bootstrap_seed": 7}],
)
def test_paired_bootstrap_rejects_nonregistered_parameters(
    kwargs: dict[str, int],
) -> None:
    from dblbt_fcn.stats import paired_bootstrap

    with pytest.raises(ValueError, match="10,000|20260715|registered"):
        paired_bootstrap(
            pairs([1.0] * 10), pairs([1.1] * 10), **kwargs
        )


def test_paired_bootstrap_marks_interval_crossing_zero_inconclusive() -> None:
    from dblbt_fcn.stats import paired_bootstrap

    baseline = pairs([1.0] * 10)
    adaptive = pairs(
        [0.7, 0.8, 0.9, 1.0, 1.0, 1.0, 1.1, 1.1, 1.2, 1.3]
    )

    result = paired_bootstrap(baseline, adaptive)

    assert result.lower_95 < 0 < result.upper_95
    assert result.decision == "inconclusive"


def test_paired_bootstrap_uses_none_for_zero_baseline_relative_difference() -> None:
    from dblbt_fcn.stats import paired_bootstrap

    result = paired_bootstrap(pairs([0.0] * 10), pairs([0.1] * 10))

    assert result.baseline_mean == 0.0
    assert result.relative_difference is None
    assert result.decision == "improvement"


def test_preregistered_hypothesis_threshold_boundaries_and_statuses() -> None:
    from dblbt_fcn.stats import (
        HYPOTHESIS_STATUSES,
        evaluate_preregistered_hypotheses,
        paired_bootstrap,
    )

    evidence = {
        "H1": paired_bootstrap(pairs([1.0] * 10), pairs([0.98] * 10)),
        "H2": paired_bootstrap(pairs([1.0] * 10), pairs([1.10] * 10)),
        "H3": paired_bootstrap(pairs([0.9] * 10), pairs([0.89] * 10)),
        "H4": paired_bootstrap(pairs([1.0] * 10), pairs([1.01] * 10)),
    }

    rows = evaluate_preregistered_hypotheses(
        evidence, ns3_available=False
    )

    assert [row.hypothesis for row in rows] == ["H1", "H2", "H3", "H4", "H5"]
    assert [row.status for row in rows] == [
        "pass",
        "pass",
        "pass",
        "pass",
        "not_evaluated",
    ]
    assert HYPOTHESIS_STATUSES == frozenset(
        {"pass", "fail", "inconclusive", "not_evaluated"}
    )


def test_hypothesis_evaluation_preserves_fail_and_inconclusive_results() -> None:
    from dblbt_fcn.stats import (
        evaluate_preregistered_hypotheses,
        paired_bootstrap,
    )

    crossing = paired_bootstrap(
        pairs([1.0] * 10),
        pairs([0.7, 0.8, 0.9, 1.0, 1.0, 1.0, 1.1, 1.1, 1.2, 1.3]),
    )
    evidence = {
        "H1": paired_bootstrap(pairs([1.0] * 10), pairs([0.9] * 10)),
        "H2": crossing,
        "H3": paired_bootstrap(pairs([0.9] * 10), pairs([0.8] * 10)),
        "H4": crossing,
    }

    rows = evaluate_preregistered_hypotheses(evidence, ns3_available=False)

    assert [row.status for row in rows] == [
        "fail",
        "inconclusive",
        "fail",
        "inconclusive",
        "not_evaluated",
    ]


def test_h5_requires_direction_evidence_when_ns3_is_available() -> None:
    from dblbt_fcn.stats import (
        evaluate_preregistered_hypotheses,
        paired_bootstrap,
    )

    evidence = {
        key: paired_bootstrap(pairs([1.0] * 10), pairs([1.01] * 10))
        for key in ("H1", "H2", "H3", "H4")
    }

    with pytest.raises(ValueError, match="H5|direction|ns-3"):
        evaluate_preregistered_hypotheses(evidence, ns3_available=True)


@pytest.mark.parametrize(
    "directions",
    [
        (True, True),
        (1, 1, 0),
        [True, True, False],
        (True, True, False, False),
    ],
)
def test_h5_rejects_noncanonical_runtime_direction_evidence(
    directions: object,
) -> None:
    from dblbt_fcn.stats import (
        evaluate_preregistered_hypotheses,
        paired_bootstrap,
    )

    evidence = {
        key: paired_bootstrap(pairs([1.0] * 10), pairs([1.01] * 10))
        for key in ("H1", "H2", "H3", "H4")
    }

    with pytest.raises(ValueError, match="H5|tuple|boolean|three"):
        evaluate_preregistered_hypotheses(
            evidence,
            ns3_available=True,
            h5_direction_evidence=directions,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("directions", "status"),
    [
        ((True, True, False), "pass"),
        ((True, False, False), "fail"),
    ],
)
def test_h5_accepts_exactly_three_booleans(
    directions: tuple[bool, bool, bool], status: str
) -> None:
    from dblbt_fcn.stats import (
        evaluate_preregistered_hypotheses,
        paired_bootstrap,
    )

    evidence = {
        key: paired_bootstrap(pairs([1.0] * 10), pairs([1.01] * 10))
        for key in ("H1", "H2", "H3", "H4")
    }

    rows = evaluate_preregistered_hypotheses(
        evidence,
        ns3_available=True,
        h5_direction_evidence=directions,
    )

    assert rows[-1].status == status


def test_canonical_report_inventory_has_only_approved_modes_and_counts() -> None:
    from dblbt_fcn.inventory import canonical_report_inventory

    inventory = canonical_report_inventory()

    assert set(inventory) == {"smoke", "formal"}
    assert len(inventory["smoke"]) == 3
    assert len(inventory["formal"]) == 940


@pytest.mark.parametrize(
    ("label", "predicate"),
    [
        ("Random", lambda job: job.policy == "random_lbt"),
        ("Primary", lambda job: job.policy == "primary_db_lbt"),
        ("Oracle", lambda job: job.policy == "fixed_oracle"),
        ("reproduction", lambda job: job.matrix == "reproduction"),
        ("ablation", lambda job: job.matrix == "ablation"),
    ],
)
def test_formal_inventory_rejects_missing_registered_job_family(
    label: str, predicate: object
) -> None:
    from dblbt_fcn.inventory import (
        canonical_report_inventory,
        validate_report_inventory,
    )

    jobs = canonical_report_inventory()["formal"]
    removed = next(job for job in jobs if predicate(job))
    rows = [
        {"matrix": job.matrix, "run_id": job.run_id}
        for job in jobs
        if job.run_id != removed.run_id
    ]

    with pytest.raises(ValueError, match="inventory|shared formal root|missing"):
        validate_report_inventory(rows, {str(row["run_id"]) for row in rows})


def test_formal_inventory_rejects_extra_run_and_unsupported_matrix_mix() -> None:
    from dblbt_fcn.inventory import (
        canonical_report_inventory,
        validate_report_inventory,
    )

    jobs = canonical_report_inventory()["formal"]
    rows = [{"matrix": job.matrix, "run_id": job.run_id} for job in jobs]
    rows.append({"matrix": "heldout", "run_id": "f" * 16})
    with pytest.raises(ValueError, match="inventory|extra"):
        validate_report_inventory(rows, {str(row["run_id"]) for row in rows})

    with pytest.raises(ValueError, match="mode|matrix|shared formal root"):
        validate_report_inventory(
            [
                {"matrix": "smoke", "run_id": "a" * 16},
                {"matrix": "heldout", "run_id": "b" * 16},
            ],
            {"a" * 16, "b" * 16},
        )


def write_model(path: Path) -> None:
    from dblbt_fcn.linucb import LinUCB
    from dblbt_fcn.workflows import action_grid_hash

    LinUCB(24, 11, action_grid_hash=action_grid_hash()).save(path)


def write_oracle(path: Path, model: Path, *, arm: int = 9) -> None:
    from dblbt_fcn.experiment import canonical_json, load_matrix
    from dblbt_fcn.provenance import file_sha256
    from dblbt_fcn.workflows import action_grid_hash

    matrix_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "matrices"
        / "pretrain.yaml"
    )
    matrix = load_matrix(matrix_path)
    value = {
        "schema_version": 1,
        "arm": arm,
        "action_grid_hash": action_grid_hash(),
        "source_matrix": matrix.model_dump(mode="json"),
        "source_matrix_hash": hashlib.sha256(
            canonical_json(matrix).encode("ascii")
        ).hexdigest(),
        "model_sha256": file_sha256(model),
    }
    path.write_bytes((canonical_json(value) + "\n").encode("ascii"))


def test_model_overhead_uses_real_file_and_excludes_exact_warmup(
    tmp_path: Path,
) -> None:
    from dblbt_fcn.provenance import file_sha256
    from dblbt_fcn.stats import measure_model_overhead
    from dblbt_fcn.workflows import action_grid_hash

    model = tmp_path / "model.npz"
    write_model(model)
    calls = {"select": 0, "timer": 0}
    clock = 0

    def selector(agent: object, context: object) -> int:
        calls["select"] += 1
        return 0

    def timer() -> int:
        nonlocal clock
        calls["timer"] += 1
        clock += 1_000
        return clock

    result = measure_model_overhead(
        model, timer=timer, selector=selector
    )

    assert calls == {"select": 10_100, "timer": 20_000}
    assert result.warmup_calls == 100
    assert result.measurement_calls == 10_000
    assert result.model_state_bytes == model.stat().st_size
    assert result.model_sha256 == file_sha256(model)
    assert result.action_grid_hash == action_grid_hash()
    assert result.median_us == 1.0
    assert result.p95_us == 1.0


def test_model_overhead_rejects_missing_or_wrong_grid_model(
    tmp_path: Path,
) -> None:
    from dblbt_fcn.linucb import LinUCB
    from dblbt_fcn.stats import measure_model_overhead

    with pytest.raises(ValueError, match="file|model|provenance"):
        measure_model_overhead(tmp_path / "missing.npz")

    wrong = tmp_path / "wrong.npz"
    LinUCB(24, 11, action_grid_hash="a" * 64).save(wrong)
    with pytest.raises(ValueError, match="action_grid_hash"):
        measure_model_overhead(wrong)


SUMMARY_FIELDS = [
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


def summary_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scenario in ("symmetric-6x6", "dynamic-combined-4x4"):
        for policy in ("tmc_db_lbt", "adaptive_db_lbt"):
            for seed in reversed(SEEDS):
                dynamic = scenario.startswith("dynamic")
                baseline_utility = 0.5 if dynamic else 0.6
                adaptive_utility = 0.6 if dynamic else 0.594
                rows.append(
                    {
                        "run_id": f"{len(rows):016x}",
                        "matrix": "heldout",
                        "scenario_id": scenario,
                        "policy": policy,
                        "seed": seed,
                        "ablation": "",
                        "arm_id": "",
                        "wifi_nodes": 4 if dynamic else 6,
                        "nru_nodes": 4 if dynamic else 6,
                        "traffic": "poisson" if dynamic else "saturated",
                        "interference_interval_ms": 300 if dynamic else "",
                        "interruption_std": 0.0,
                        "join_interval_rounds": 10 if dynamic else "",
                        "lifetime_rounds": 200 if dynamic else "",
                        "config_hash": f"{len(rows):064x}",
                        "rounds": 64,
                        "elapsed_us": 128_000,
                        "successes": 48,
                        "collisions": 16,
                        "collision_probability": 0.25,
                        "effective_airtime": 0.75,
                        "mean_delay_us": 100.0,
                        "p95_delay_us": 200.0,
                        "jain_fairness": 0.9,
                        "evaluation_utility": (
                            adaptive_utility
                            if policy == "adaptive_db_lbt"
                            else baseline_utility
                        ),
                        "decision_count": 2 if policy == "adaptive_db_lbt" else 0,
                        "switch_count": 1 if policy == "adaptive_db_lbt" else 0,
                        "training_sample_count": 0,
                    }
                )
    return rows


def complete_comparison_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    policies = (
        "random_lbt",
        "primary_db_lbt",
        "tmc_db_lbt",
        "fixed_oracle",
        "adaptive_db_lbt",
    )
    for scenario_index, scenario in enumerate(
        ("symmetric-6x6", "dynamic-combined-4x4")
    ):
        for policy_index, policy in enumerate(policies):
            for seed_index, seed in enumerate(SEEDS):
                row = dict(summary_rows()[0])
                row.update(
                    run_id=f"{len(rows):016x}",
                    matrix="heldout",
                    scenario_id=scenario,
                    policy=policy,
                    seed=seed,
                    ablation=None,
                    config_hash=f"{len(rows):064x}",
                    evaluation_utility=(
                        0.4
                        + scenario_index / 10
                        + policy_index / 100
                        + seed_index / 10_000
                    ),
                    jain_fairness=0.8 + policy_index / 100,
                )
                rows.append(row)
    for condition_index, condition in enumerate(
        (
            "full",
            "no_queue",
            "no_cca_interrupt",
            "no_delay",
            "frozen_online",
            "context_free_ucb",
            "collision_weight_0.125",
            "collision_weight_0.5",
        )
    ):
        for seed_index, seed in enumerate(SEEDS):
            row = dict(summary_rows()[0])
            row.update(
                run_id=f"{len(rows):016x}",
                matrix="ablation",
                scenario_id="dynamic-combined-4x4",
                policy="adaptive_db_lbt",
                seed=seed,
                ablation=condition,
                config_hash=f"{len(rows):064x}",
                evaluation_utility=(
                    0.7 + condition_index / 100 + seed_index / 10_000
                ),
            )
            rows.append(row)
    return rows


def test_comparison_table_rows_cover_hypotheses_scenarios_and_ablations() -> None:
    from dblbt_fcn.stats import comparison_table_rows

    rows = comparison_table_rows(complete_comparison_rows())

    assert len(rows) == 19
    assert {row["scope"] for row in rows} == {
        "hypothesis",
        "heldout_scenario",
        "ablation",
    }
    assert len({row["comparison_id"] for row in rows}) == len(rows)
    assert {row["direction"] for row in rows} == {"higher_is_better"}
    heldout = [row for row in rows if row["scope"] == "heldout_scenario"]
    assert len(heldout) == 8
    assert {row["baseline_policy"] for row in heldout} == {
        "random_lbt",
        "primary_db_lbt",
        "tmc_db_lbt",
        "fixed_oracle",
    }
    ablations = [row for row in rows if row["scope"] == "ablation"]
    assert len(ablations) == 7
    assert {row["baseline_ablation"] for row in ablations} == {"full"}
    assert "full" not in {row["candidate_ablation"] for row in ablations}


def test_ablation_forest_consumes_comparison_rows_without_hidden_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import matplotlib.pyplot as plt

    import dblbt_fcn.plotting as plotting
    from dblbt_fcn.stats import comparison_table_rows

    comparisons = comparison_table_rows(complete_comparison_rows())
    monkeypatch.setattr(
        plotting,
        "paired_bootstrap",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("forest must not compute hidden bootstrap evidence")
        ),
        raising=False,
    )
    figure, axis = plt.subplots()
    try:
        plotting._ablation(axis, comparisons)
        assert len(axis.get_yticklabels()) == 7
    finally:
        plt.close(figure)


def write_summary(path: Path, rows: list[dict[str, object]] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary_rows() if rows is None else rows)


def test_generate_tables_writes_fixed_schemas_sorted_and_stable(
    tmp_path: Path,
) -> None:
    from dblbt_fcn.stats import generate_tables, measure_model_overhead

    summary = tmp_path / "summary.csv"
    write_summary(summary)
    model = tmp_path / "model.npz"
    write_model(model)
    clock = iter(range(0, 20_001_000, 1_000))
    overhead = measure_model_overhead(
        model,
        timer=lambda: next(clock),
        selector=lambda agent, context: 0,
    )
    output = tmp_path / "tables"

    paths = generate_tables(summary, output, overhead)
    first = {path: path.read_bytes() for path in paths}
    paths_again = generate_tables(summary, output, overhead)

    assert set(path.name for path in paths) == {
        "per-seed-metrics.csv",
        "per-seed-metrics.tex",
        "paired-comparisons.csv",
        "paired-comparisons.tex",
        "overhead.csv",
        "overhead.tex",
        "hypotheses.csv",
        "hypotheses.tex",
    }
    assert paths_again == paths
    assert {path: path.read_bytes() for path in paths} == first
    per_seed = list(csv.DictReader((output / "per-seed-metrics.csv").open()))
    assert [(row["scenario_id"], row["policy"], int(row["seed"])) for row in per_seed] == sorted(
        (row["scenario_id"], row["policy"], int(row["seed"]))
        for row in per_seed
    )
    comparisons = list(
        csv.DictReader((output / "paired-comparisons.csv").open())
    )
    assert [row["hypothesis"] for row in comparisons] == [
        "H1", "H2", "H3", "H4", "", ""
    ]
    assert set(comparisons[0]) == {
        "comparison_id",
        "scope",
        "hypothesis",
        "scenario_id",
        "baseline_policy",
        "candidate_policy",
        "baseline_ablation",
        "candidate_ablation",
        "metric",
        "direction",
        "baseline_mean",
        "candidate_mean",
        "paired_difference",
        "relative_difference",
        "lower_95",
        "upper_95",
        "decision",
        "resamples",
        "bootstrap_seed",
    }
    hypotheses = list(csv.DictReader((output / "hypotheses.csv").open()))
    assert [row["hypothesis"] for row in hypotheses] == ["H1", "H2", "H3", "H4", "H5"]
    assert {row["status"] for row in hypotheses} <= {
        "pass", "fail", "inconclusive", "not_evaluated"
    }
    overhead_rows = list(csv.DictReader((output / "overhead.csv").open()))
    assert len(overhead_rows) == 1
    assert int(overhead_rows[0]["model_state_bytes"]) == model.stat().st_size
    assert all(path.stat().st_size > 0 for path in paths)


@pytest.mark.parametrize("failure", ["duplicate", "missing_pair", "nan"])
def test_generate_tables_fails_closed_on_invalid_summary(
    tmp_path: Path, failure: str
) -> None:
    from dblbt_fcn.stats import generate_tables, measure_model_overhead

    rows = summary_rows()
    if failure == "duplicate":
        rows.append(dict(rows[0]))
    elif failure == "missing_pair":
        rows.pop()
    else:
        rows[0]["evaluation_utility"] = math.nan
    summary = tmp_path / "summary.csv"
    write_summary(summary, rows)
    model = tmp_path / "model.npz"
    write_model(model)
    clock = iter(range(0, 20_001_000, 1_000))
    overhead = measure_model_overhead(
        model,
        timer=lambda: next(clock),
        selector=lambda agent, context: 0,
    )
    output = tmp_path / "tables"

    with pytest.raises(ValueError, match="duplicate|pair|seed|finite"):
        generate_tables(summary, output, overhead)

    assert not output.exists() or list(output.iterdir()) == []


def test_generate_tables_refuses_to_overwrite_summary_input(tmp_path: Path) -> None:
    from dblbt_fcn.stats import generate_tables, measure_model_overhead

    output = tmp_path / "tables"
    summary = output / "per-seed-metrics.csv"
    write_summary(summary)
    before = summary.read_bytes()
    model = tmp_path / "model.npz"
    write_model(model)
    clock = iter(range(0, 20_001_000, 1_000))
    overhead = measure_model_overhead(
        model,
        timer=lambda: next(clock),
        selector=lambda agent, context: 0,
    )

    with pytest.raises(ValueError, match="input|overwrite|protected"):
        generate_tables(summary, output, overhead)

    assert summary.read_bytes() == before


def test_hypothesis_tables_ignore_nonheldout_ablation_rows(
    tmp_path: Path,
) -> None:
    from dblbt_fcn.stats import generate_tables, measure_model_overhead

    rows = summary_rows()
    ablation = dict(rows[0])
    ablation.update(
        run_id="e" * 16,
        matrix="ablation",
        policy="adaptive_db_lbt",
        ablation="no_delay",
        seed=SEEDS[0],
    )
    rows.append(ablation)
    summary = tmp_path / "summary.csv"
    write_summary(summary, rows)
    model = tmp_path / "model.npz"
    write_model(model)
    clock = iter(range(0, 20_001_000, 1_000))
    overhead = measure_model_overhead(
        model,
        timer=lambda: next(clock),
        selector=lambda agent, context: 0,
    )

    paths = generate_tables(summary, tmp_path / "tables", overhead)

    assert len(paths) == 8
    per_seed = list(
        csv.DictReader((tmp_path / "tables" / "per-seed-metrics.csv").open())
    )
    assert any(row["matrix"] == "ablation" for row in per_seed)


def test_per_seed_table_bytes_are_stable_across_multimatrix_input_order(
    tmp_path: Path,
) -> None:
    from dblbt_fcn.stats import generate_tables, measure_model_overhead

    rows = summary_rows()
    ablation = dict(rows[0])
    ablation.update(
        run_id="d" * 16,
        matrix="ablation",
        policy="adaptive_db_lbt",
        ablation="full",
    )
    rows.append(ablation)
    first_summary = tmp_path / "first.csv"
    second_summary = tmp_path / "second.csv"
    write_summary(first_summary, rows)
    write_summary(second_summary, list(reversed(rows)))
    model = tmp_path / "model.npz"
    write_model(model)

    def overhead():
        clock = iter(range(0, 20_001_000, 1_000))
        return measure_model_overhead(
            model,
            timer=lambda: next(clock),
            selector=lambda agent, context: 0,
        )

    generate_tables(first_summary, tmp_path / "a", overhead())
    generate_tables(second_summary, tmp_path / "b", overhead())

    assert (tmp_path / "a" / "per-seed-metrics.csv").read_bytes() == (
        tmp_path / "b" / "per-seed-metrics.csv"
    ).read_bytes()


FIGURE_NAMES = {
    "backoff-convergence",
    "delay-cdf",
    "scaling",
    "dynamic-adaptation",
    "held-out-utility",
    "fairness-delay-airtime-tradeoff",
    "arm-heatmap",
    "ablation-forest",
}


def make_plot_inputs(
    tmp_path: Path,
    *,
    rounds: int = 64,
    nodes_per_technology: int = 2,
    dynamic: bool = True,
) -> tuple[Path, Path, Path]:
    from dblbt_fcn.experiment import JobSpec, ScenarioSpec, TimingSpec
    from dblbt_fcn.reporting import summarize_manifests
    from dblbt_fcn.simulation import run_job

    job = JobSpec(
        matrix="smoke",
        rounds=rounds,
        alpha=11,
        timing=TimingSpec(),
        scenario=ScenarioSpec(
            id=(
                f"dynamic-combined-{nodes_per_technology}x{nodes_per_technology}"
                if dynamic
                else f"static-{nodes_per_technology}x{nodes_per_technology}"
            ),
            wifi_nodes=nodes_per_technology,
            nru_nodes=nodes_per_technology,
            join_interval_rounds=10 if dynamic else None,
            lifetime_rounds=40 if dynamic else None,
        ),
        policy="adaptive_db_lbt",
        seed=410,
    )
    root = tmp_path / "runs"
    run_job(job, root)
    summary = tmp_path / "summary.csv"
    summarize_manifests(root / "manifests", summary)
    return summary, root / "manifests", root


def make_canonical_smoke_inputs(tmp_path: Path) -> tuple[Path, Path]:
    from dblbt_fcn.experiment import expand_matrix, load_matrix
    from dblbt_fcn.reporting import summarize_manifests
    from dblbt_fcn.simulation import run_job

    matrix_path = Path(__file__).resolve().parents[1] / "configs" / "matrices" / "smoke.yaml"
    root = tmp_path / "smoke-runs"
    for job in expand_matrix(load_matrix(matrix_path)):
        run_job(job, root)
    summary = tmp_path / "smoke-summary.csv"
    summarize_manifests(root / "manifests", summary)
    return summary, root / "manifests"


def test_smoke_report_and_audit_succeed_without_model_or_formal_tables(
    tmp_path: Path,
) -> None:
    from dblbt_fcn.audit import audit_report
    from dblbt_fcn.plotting import generate_report

    summary, manifests = make_canonical_smoke_inputs(tmp_path)
    output = tmp_path / "smoke-report"

    paths = generate_report(summary, output, manifests, None)
    result = audit_report(manifests, summary, output, None)

    assert len(paths) == 16
    assert result.run_count == 3
    assert result.figure_count == 16
    assert result.table_count == 0
    assert not (output / "tables").exists()


def test_audit_decompresses_each_job_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import gzip

    from dblbt_fcn.audit import audit_report
    from dblbt_fcn.plotting import generate_report

    summary, manifests = make_canonical_smoke_inputs(tmp_path)
    output = tmp_path / "smoke-report"
    generate_report(summary, output, manifests, None)
    original = gzip.GzipFile
    reads = 0

    def observed(*args: object, **kwargs: object):
        nonlocal reads
        mode = kwargs.get("mode", args[1] if len(args) > 1 else None)
        if mode in {"rb", "r"}:
            reads += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(gzip, "GzipFile", observed)

    audit_report(manifests, summary, output, None)

    assert reads == 3


@pytest.mark.parametrize("existing", [False, True])
def test_generate_report_rolls_back_second_final_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing: bool,
) -> None:
    import dblbt_fcn.plotting as plotting

    summary, manifests = make_canonical_smoke_inputs(tmp_path)
    output = tmp_path / "smoke-report"
    before: dict[str, bytes] = {}
    if existing:
        plotting.generate_report(summary, output, manifests, None)
        before = {
            str(path.relative_to(output)): path.read_bytes()
            for path in output.rglob("*")
            if path.is_file()
        }
    original_replace = plotting.os.replace
    injected = False

    def fail_second_final(source: object, target: object) -> None:
        nonlocal injected
        target_path = Path(target)
        if target_path == output and not injected:
            injected = True
            raise OSError("injected second final replace failure")
        original_replace(source, target)

    monkeypatch.setattr(plotting.os, "replace", fail_second_final)

    with pytest.raises(OSError, match="second final replace"):
        plotting.generate_report(summary, output, manifests, None)

    after = (
        {
            str(path.relative_to(output)): path.read_bytes()
            for path in output.rglob("*")
            if path.is_file()
        }
        if output.exists()
        else {}
    )
    assert after == before


@pytest.mark.parametrize("unsafe_target", ["run_root", "project_root"])
def test_generate_report_rejects_destructive_output_before_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unsafe_target: str,
) -> None:
    import dblbt_fcn.plotting as plotting

    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    summary = project / "summary.csv"
    summary.write_bytes(b"protected summary\n")
    manifests = project / "runs" / "manifests"
    manifests.mkdir(parents=True)
    (manifests / "protected.json").write_bytes(b"protected manifest\n")
    raw = manifests.parent / "raw" / "protected.jsonl.gz"
    raw.parent.mkdir()
    raw.write_bytes(b"protected raw\n")
    output = manifests.parent if unsafe_target == "run_root" else project
    before = {
        path.relative_to(project): path.read_bytes()
        for path in project.rglob("*")
        if path.is_file()
    }
    monkeypatch.setattr(
        plotting,
        "load_summary",
        lambda path: [{"matrix": "smoke", "run_id": "protected"}],
    )
    monkeypatch.setattr(
        plotting,
        "validate_report_inventory",
        lambda rows, manifest_ids: "smoke",
    )
    monkeypatch.setattr(
        plotting.tempfile,
        "mkdtemp",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("unsafe output must be rejected before staging")
        ),
    )

    with pytest.raises(ValueError, match="output|protected|project|run"):
        plotting.generate_report(summary, output, manifests, None)

    assert {
        path.relative_to(project): path.read_bytes()
        for path in project.rglob("*")
        if path.is_file()
    } == before


def test_generate_report_ignores_committed_backup_cleanup_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dblbt_fcn.plotting as plotting

    summary = tmp_path / "summary.csv"
    summary.write_bytes(b"summary\n")
    manifests = tmp_path / "runs" / "manifests"
    manifests.mkdir(parents=True)
    (manifests / "protected.json").write_bytes(b"manifest\n")
    output = tmp_path / "smoke-report"
    monkeypatch.setattr(
        plotting,
        "load_summary",
        lambda path: [{"matrix": "smoke", "run_id": "protected"}],
    )
    monkeypatch.setattr(
        plotting,
        "validate_report_inventory",
        lambda rows, manifest_ids: "smoke",
    )

    def fake_figures(
        summary_path: object,
        destination: object,
        *,
        manifest_dir: object,
    ) -> list[Path]:
        target = Path(destination)
        paths = [target / f"figure-{index}.png" for index in range(16)]
        for path in paths:
            path.write_bytes(b"figure")
        return paths

    monkeypatch.setattr(plotting, "generate_figures", fake_figures)
    plotting.generate_report(summary, output, manifests, None)
    obsolete = output / "obsolete.txt"
    obsolete.write_bytes(b"old report")
    original_rmtree = plotting.shutil.rmtree

    def fail_backup_cleanup(path: object, *args: object, **kwargs: object) -> None:
        if ".backup." in Path(path).name:
            raise PermissionError("injected backup cleanup failure")
        original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(plotting.shutil, "rmtree", fail_backup_cleanup)

    paths = plotting.generate_report(summary, output, manifests, None)

    assert len(paths) == 16
    assert not obsolete.exists()
    assert all(path.is_file() for path in paths)


def test_generate_report_protects_model_and_oracle_before_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dblbt_fcn.plotting as plotting

    summary = tmp_path / "summary.csv"
    summary.write_bytes(b"summary\n")
    manifests = tmp_path / "runs" / "manifests"
    manifests.mkdir(parents=True)
    model = tmp_path / "inputs" / "model.npz"
    model.parent.mkdir()
    model.write_bytes(b"model")
    oracle = model.parent / "fixed-oracle-arm.json"
    oracle.write_bytes(b"oracle")
    before = {path: path.read_bytes() for path in (model, oracle)}
    monkeypatch.setattr(
        plotting,
        "load_summary",
        lambda path: [{"matrix": "heldout", "run_id": "protected"}],
    )
    monkeypatch.setattr(
        plotting,
        "validate_report_inventory",
        lambda rows, manifest_ids: "formal",
    )
    monkeypatch.setattr(
        plotting.tempfile,
        "mkdtemp",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("protected inputs must be rejected before staging")
        ),
    )

    with pytest.raises(ValueError, match="output|protected|model|Oracle"):
        plotting.generate_report(
            summary, model.parent, manifests, model, oracle
        )

    assert {path: path.read_bytes() for path in before} == before


def test_formal_report_validates_run_provenance_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dblbt_fcn.audit as audit
    import dblbt_fcn.plotting as plotting

    summary = tmp_path / "summary.csv"
    summary.write_bytes(b"summary\n")
    manifests = tmp_path / "runs" / "manifests"
    manifests.mkdir(parents=True)
    model = tmp_path / "model.npz"
    model.write_bytes(b"model")
    oracle = tmp_path / "fixed-oracle-arm.json"
    oracle.write_bytes(b"oracle")
    output = tmp_path / "report"
    monkeypatch.setattr(
        plotting,
        "load_summary",
        lambda path: [{"matrix": "heldout", "run_id": "protected"}],
    )
    monkeypatch.setattr(
        plotting,
        "validate_report_inventory",
        lambda rows, manifest_ids: "formal",
    )

    def fake_figures(
        summary_path: object,
        destination: object,
        *,
        manifest_dir: object,
        run_validator: object,
    ) -> list[Path]:
        assert callable(run_validator)
        run_validator([])
        raise AssertionError("provenance validator must reject the runs")

    monkeypatch.setattr(plotting, "generate_figures", fake_figures)
    monkeypatch.setattr(
        audit,
        "_audit_model_provenance",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("forged Oracle provenance")
        ),
    )

    with pytest.raises(ValueError, match="forged Oracle provenance"):
        plotting.generate_report(
            summary, output, manifests, model, oracle
        )

    assert not output.exists()


def test_formal_figures_and_tables_share_one_publication_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dblbt_fcn.plotting as plotting

    summary, manifests, model, oracle = make_formal_report_inputs(tmp_path)
    output = tmp_path / "formal-report"
    monkeypatch.setattr(
        plotting,
        "validate_report_inventory",
        lambda rows, manifest_ids: "formal",
    )
    original_replace = plotting.os.replace
    injected = False

    def fail_publication(source: object, target: object) -> None:
        nonlocal injected
        if Path(target) == output and not injected:
            injected = True
            raise OSError("injected formal publication failure")
        original_replace(source, target)

    monkeypatch.setattr(plotting.os, "replace", fail_publication)

    with pytest.raises(OSError, match="formal publication"):
        plotting.generate_report(summary, output, manifests, model, oracle)

    assert not output.exists()


def test_generate_figures_writes_exact_pdf_png_set_with_valid_magic(
    tmp_path: Path,
) -> None:
    from dblbt_fcn.plotting import generate_figures

    summary, manifests, _ = make_plot_inputs(tmp_path)
    output = tmp_path / "figures"

    paths = generate_figures(summary, output, manifest_dir=manifests)

    assert {path.stem for path in paths} == FIGURE_NAMES
    assert {path.suffix for path in paths} == {".pdf", ".png"}
    assert len(paths) == 16
    for path in paths:
        payload = path.read_bytes()
        assert len(payload) > 100
        if path.suffix == ".pdf":
            assert payload.startswith(b"%PDF-")
        else:
            assert payload.startswith(b"\x89PNG\r\n\x1a\n")


def test_generate_report_rejects_incomplete_inventory_before_publication(
    tmp_path: Path,
) -> None:
    from dblbt_fcn.plotting import generate_report

    summary, manifests, _ = make_plot_inputs(tmp_path)
    model = tmp_path / "model.npz"
    write_model(model)
    output = tmp_path / "report"

    with pytest.raises(ValueError, match="inventory|shared formal root|missing"):
        generate_report(summary, output, manifests, model)

    assert not output.exists()


def _allow_component_inventory(monkeypatch: pytest.MonkeyPatch) -> None:
    import dblbt_fcn.audit as audit

    monkeypatch.setattr(
        audit,
        "validate_report_inventory",
        lambda summary, manifest_ids: "formal",
    )


def test_formal_audit_requires_frozen_oracle_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dblbt_fcn.audit as audit

    model = tmp_path / "model.npz"
    write_model(model)
    monkeypatch.setattr(
        audit,
        "validated_plot_inputs",
        lambda summary, manifests: ([{"matrix": "heldout"}], []),
    )
    monkeypatch.setattr(
        audit,
        "validate_report_inventory",
        lambda summary, manifest_ids: "formal",
    )

    with pytest.raises(ValueError, match="Oracle|oracle"):
        audit.audit_report(
            tmp_path / "manifests",
            tmp_path / "summary.csv",
            tmp_path / "report",
            model,
            oracle_arm_file=None,
        )


def test_audit_report_rejects_incomplete_inventory_before_evidence_audit(
    tmp_path: Path,
) -> None:
    from dblbt_fcn.audit import audit_report

    summary, manifests, _ = make_plot_inputs(tmp_path)
    model = tmp_path / "model.npz"
    write_model(model)

    with pytest.raises(ValueError, match="inventory|shared formal root|missing"):
        audit_report(manifests, summary, tmp_path / "missing-report", model)


@pytest.mark.parametrize("failure", ["missing", "corrupt", "mismatch"])
def test_generate_figures_fails_closed_on_invalid_raw_inputs(
    tmp_path: Path, failure: str
) -> None:
    from dblbt_fcn.plotting import generate_figures

    summary, manifests, root = make_plot_inputs(tmp_path)
    manifest = next(manifests.glob("*.json"))
    if failure == "missing":
        manifest.unlink()
    elif failure == "corrupt":
        raw = next((root / "raw").glob("*.jsonl.gz"))
        raw.write_bytes(b"corrupt")
    else:
        rows = list(csv.DictReader(summary.open()))
        rows[0]["run_id"] = "f" * 16
        write_summary(summary, rows)
    output = tmp_path / "figures"

    with pytest.raises((ValueError, OSError, RuntimeError)):
        generate_figures(summary, output, manifest_dir=manifests)

    assert not output.exists() or list(output.iterdir()) == []


def test_generate_figures_refuses_artifact_tree_output(tmp_path: Path) -> None:
    from dblbt_fcn.plotting import generate_figures

    summary, manifests, root = make_plot_inputs(tmp_path)
    raw_before = {
        path: path.read_bytes() for path in (root / "raw").iterdir()
    }

    with pytest.raises(ValueError, match="artifact|input|output|protected"):
        generate_figures(summary, root / "raw", manifest_dir=manifests)

    assert {path: path.read_bytes() for path in raw_before} == raw_before


def test_generate_figures_failure_leaves_no_partial_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dblbt_fcn.plotting as plotting

    summary, manifests, _ = make_plot_inputs(tmp_path)
    output = tmp_path / "figures"
    original = plotting._PLOTTERS

    def fail(ax: object, summary_rows: object, runs: object) -> None:
        raise RuntimeError("plot failed")

    monkeypatch.setattr(
        plotting,
        "_PLOTTERS",
        (original[0], (original[1][0], fail), *original[2:]),
    )

    with pytest.raises(RuntimeError, match="plot failed"):
        plotting.generate_figures(summary, output, manifest_dir=manifests)

    assert not output.exists() or list(output.iterdir()) == []


def test_plot_validation_binds_summary_metrics_to_canonical_raw(
    tmp_path: Path,
) -> None:
    from dblbt_fcn.plotting import generate_figures

    summary, manifests, _ = make_plot_inputs(tmp_path)
    rows = list(csv.DictReader(summary.open()))
    rows[0]["evaluation_utility"] = str(
        float(rows[0]["evaluation_utility"]) + 0.1
    )
    write_summary(summary, rows)

    with pytest.raises(ValueError, match="summary|raw|metric|aggregate"):
        generate_figures(summary, tmp_path / "figures", manifest_dir=manifests)


def test_plot_validation_returns_bounded_trace_after_validating_long_raw(
    tmp_path: Path,
) -> None:
    from dblbt_fcn.plotting import validated_plot_inputs

    summary, manifests, _ = make_plot_inputs(
        tmp_path,
        rounds=600,
        nodes_per_technology=32,
        dynamic=False,
    )
    summary_rows, runs = validated_plot_inputs(summary, manifests)

    assert runs
    assert all(len(run.backoff_points) <= 512 for run in runs)
    assert all(len(run.delay_samples) <= 512 for run in runs)
    assert all(len(run.dynamic_points) <= 512 for run in runs)
    assert all(len(run.decision_points) <= 512 for run in runs)
    assert all(int(row["decision_count"]) > 512 for row in summary_rows)
    assert [int(run.arm_counts.sum()) for run in runs] == [
        int(row["decision_count"]) for row in summary_rows
    ]
    assert all(not hasattr(run, "rows") for run in runs)


def test_parallel_plot_validation_matches_sequential(tmp_path: Path) -> None:
    import numpy as np

    from dblbt_fcn.plotting import validated_plot_inputs

    summary, manifests = make_canonical_smoke_inputs(tmp_path)

    sequential_rows, sequential_runs = validated_plot_inputs(
        summary, manifests, workers=1
    )
    parallel_rows, parallel_runs = validated_plot_inputs(
        summary, manifests, workers=2
    )

    assert parallel_rows == sequential_rows
    assert [run.job.run_id for run in parallel_runs] == [
        run.job.run_id for run in sequential_runs
    ]
    for sequential, parallel in zip(
        sequential_runs, parallel_runs, strict=True
    ):
        assert parallel.job == sequential.job
        assert parallel.backoff_points == sequential.backoff_points
        assert parallel.delay_samples == sequential.delay_samples
        assert parallel.dynamic_points == sequential.dynamic_points
        assert parallel.decision_points == sequential.decision_points
        assert np.array_equal(parallel.arm_counts, sequential.arm_counts)
        assert parallel.execution_provenance == sequential.execution_provenance


def test_plot_streams_validated_rows_without_read_job_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dblbt_fcn.records as records
    from dblbt_fcn.plotting import validated_plot_inputs

    summary, manifests, _ = make_plot_inputs(tmp_path)
    monkeypatch.setattr(
        records,
        "read_job_rows",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("plot must stream raw rows")
        ),
    )

    rows, runs = validated_plot_inputs(summary, manifests)

    assert len(rows) == len(runs) == 1
    assert len(runs[0].decision_points) <= 512


def test_plot_validation_decompresses_each_job_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import gzip

    from dblbt_fcn.plotting import validated_plot_inputs

    summary, manifests, _ = make_plot_inputs(tmp_path)
    original = gzip.GzipFile
    reads = 0

    def observed(*args: object, **kwargs: object):
        nonlocal reads
        mode = kwargs.get("mode", args[1] if len(args) > 1 else None)
        if mode in {"rb", "r"}:
            reads += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(gzip, "GzipFile", observed)

    validated_plot_inputs(summary, manifests)

    assert reads == 1


def test_save_atomic_fsyncs_a_writable_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import matplotlib.pyplot as plt

    import dblbt_fcn.plotting as plotting

    modes: list[str] = []
    original_open = Path.open

    def observed_open(path: Path, mode: str = "r", *args: object, **kwargs: object):
        if path.parent == tmp_path and path.name.startswith(".figure.png"):
            modes.append(mode)
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", observed_open)
    figure, _ = plt.subplots()
    try:
        plotting._save_atomic(figure, tmp_path / "figure.png")
    finally:
        plt.close(figure)

    assert "r+b" in modes


def test_formal_context_free_provenance_is_model_independent_in_audit(
    tmp_path: Path,
) -> None:
    from dblbt_fcn.audit import _audit_model_provenance
    from dblbt_fcn.experiment import JobSpec, ScenarioSpec, TimingSpec
    from dblbt_fcn.plotting import validated_plot_inputs
    from dblbt_fcn.reporting import summarize_manifests
    from dblbt_fcn.simulation import run_job

    model = tmp_path / "model.npz"
    write_model(model)
    job = JobSpec(
        matrix="ablation",
        rounds=64,
        alpha=11,
        timing=TimingSpec(),
        scenario=ScenarioSpec(id="dynamic-combined-4x4", wifi_nodes=2, nru_nodes=2),
        policy="adaptive_db_lbt",
        seed=410,
        ablation="context_free_ucb",
    )
    root = tmp_path / "runs"
    run_job(job, root)
    summary = tmp_path / "summary.csv"
    summarize_manifests(root / "manifests", summary)
    _, runs = validated_plot_inputs(summary, root / "manifests")

    _audit_model_provenance(model, runs)


def test_context_free_provenance_rejects_hashes_and_wrong_adaptive_mode(
    tmp_path: Path,
) -> None:
    from dblbt_fcn.audit import _audit_model_provenance
    from dblbt_fcn.experiment import JobSpec, ScenarioSpec, TimingSpec
    from dblbt_fcn.plotting import PlotRun
    from dblbt_fcn.provenance import ExecutionProvenance

    with pytest.raises(ValueError, match="context_free_builtin|fields|mode"):
        ExecutionProvenance(
            mode="context_free_builtin", model_file_sha256="a" * 64
        )

    model = tmp_path / "model.npz"
    write_model(model)
    job = JobSpec(
        matrix="ablation",
        rounds=1,
        alpha=11,
        timing=TimingSpec(),
        scenario=ScenarioSpec(id="dynamic-combined-4x4", wifi_nodes=1, nru_nodes=1),
        policy="adaptive_db_lbt",
        seed=410,
        ablation="context_free_ucb",
    )
    run = PlotRun(
        job=job,
        execution_provenance=ExecutionProvenance(
            mode="adaptive_blank", agent_state_sha256="b" * 64
        ),
        backoff_points=(),
        delay_samples=(),
        dynamic_points=(),
        decision_points=(),
        arm_counts=__import__("numpy").zeros((24, 4), dtype=int),
    )

    with pytest.raises(ValueError, match="context|mode|provenance"):
        _audit_model_provenance(model, [run])


@pytest.mark.parametrize(
    "forgery",
    ["oracle_artifact_sha256", "source_matrix_sha256", "oracle_arm"],
)
def test_fixed_oracle_audit_binds_exact_frozen_artifact(
    tmp_path: Path, forgery: str
) -> None:
    import numpy as np

    from dblbt_fcn.audit import _audit_model_provenance
    from dblbt_fcn.experiment import JobSpec, ScenarioSpec, TimingSpec
    from dblbt_fcn.linucb import LinUCB
    from dblbt_fcn.plotting import PlotRun
    from dblbt_fcn.provenance import execution_provenance, file_sha256
    from dblbt_fcn.workflows import action_grid_hash, load_oracle_arm

    model = tmp_path / "model.npz"
    oracle_path = tmp_path / "fixed-oracle-arm.json"
    write_model(model)
    write_oracle(oracle_path, model)
    oracle = load_oracle_arm(oracle_path, model_path=model)
    agent = LinUCB.load(model, expected_action_grid_hash=action_grid_hash())
    job = JobSpec(
        matrix="heldout",
        rounds=1,
        alpha=11,
        timing=TimingSpec(),
        scenario=ScenarioSpec(id="symmetric-6x6", wifi_nodes=1, nru_nodes=1),
        policy="fixed_oracle",
        seed=SEEDS[0],
    )
    values = {
        "oracle_arm": oracle.arm,
        "oracle_artifact_sha256": file_sha256(oracle_path),
        "oracle_model_sha256": oracle.model_sha256,
        "source_matrix_sha256": oracle.source_matrix_hash,
    }
    values[forgery] = 8 if forgery == "oracle_arm" else "f" * 64
    run = PlotRun(
        job=job,
        execution_provenance=execution_provenance(
            job,
            initial_agent=agent,
            model_path=model,
            **values,
        ),
        backoff_points=(),
        delay_samples=(),
        dynamic_points=(),
        decision_points=(),
        arm_counts=np.zeros((24, 4), dtype=np.int64),
    )

    with pytest.raises(ValueError, match="Oracle|oracle|artifact|source|arm"):
        _audit_model_provenance(
            model, [run], oracle_arm_file=oracle_path
        )


def _plot_run(
    *,
    matrix: str,
    scenario_id: str,
    policy: str,
    seed: int,
    value: int,
    dynamic: bool = False,
    ablation: str | None = None,
):
    import numpy as np

    from dblbt_fcn.experiment import JobSpec, ScenarioSpec, TimingSpec
    from dblbt_fcn.plotting import PlotRun

    return PlotRun(
        job=JobSpec(
            matrix=matrix,
            rounds=10,
            alpha=11,
            timing=TimingSpec(),
            scenario=ScenarioSpec(
                id=scenario_id,
                wifi_nodes=2,
                nru_nodes=2,
                join_interval_rounds=2 if dynamic else None,
                lifetime_rounds=5 if dynamic else None,
                trace=scenario_id == "trace-static-4x4",
            ),
            policy=policy,
            seed=seed,
            arm_id=0 if policy == "pretrain_arm" else None,
            ablation=ablation,
        ),
        backoff_points=((1, value),),
        delay_samples=(value * 10,),
        dynamic_points=((1, 4),) if dynamic else (),
        decision_points=((1, value % 24, 0.5, False),),
        arm_counts=np.full((24, 4), value, dtype=np.int64),
    )


def test_trace_figures_select_lowest_seed_reproduction_trace_by_policy() -> None:
    import matplotlib.pyplot as plt

    from dblbt_fcn.plotting import _backoff, _delay_cdf

    runs = [
        _plot_run(
            matrix="smoke",
            scenario_id="static-2x2",
            policy="adaptive_db_lbt",
            seed=410,
            value=77,
        ),
        _plot_run(
            matrix="reproduction",
            scenario_id="static-symmetric-4x4",
            policy="tmc_db_lbt",
            seed=410,
            value=88,
        ),
        _plot_run(
            matrix="reproduction",
            scenario_id="trace-static-4x4",
            policy="adaptive_db_lbt",
            seed=523,
            value=99,
        ),
        _plot_run(
            matrix="reproduction",
            scenario_id="trace-static-4x4",
            policy="tmc_db_lbt",
            seed=410,
            value=42,
        ),
        _plot_run(
            matrix="reproduction",
            scenario_id="trace-static-4x4",
            policy="adaptive_db_lbt",
            seed=410,
            value=41,
        ),
    ]
    backoff_figure, backoff_axis = plt.subplots()
    delay_figure, delay_axis = plt.subplots()
    try:
        _backoff(backoff_axis, list(reversed(runs)))
        _delay_cdf(delay_axis, runs)

        assert {
            int(offset[1])
            for collection in backoff_axis.collections
            for offset in collection.get_offsets()
        } == {41, 42}
        assert {
            int(value)
            for line in delay_axis.lines
            for value in line.get_xdata()
        } == {410, 420}
        assert backoff_axis.get_legend_handles_labels()[1] == [
            "adaptive_db_lbt",
            "tmc_db_lbt",
        ]
        for axis in (backoff_axis, delay_axis):
            assert "reproduction" in axis.get_title()
            assert "trace-static-4x4" in axis.get_title()
            assert "seed 410" in axis.get_title()
    finally:
        plt.close(backoff_figure)
        plt.close(delay_figure)


def test_trace_figures_use_declared_unique_smoke_fallback() -> None:
    import matplotlib.pyplot as plt

    from dblbt_fcn.plotting import _backoff

    runs = [
        _plot_run(
            matrix="smoke",
            scenario_id="static-2x2",
            policy=policy,
            seed=410,
            value=value,
        )
        for policy, value in (
            ("tmc_db_lbt", 12),
            ("adaptive_db_lbt", 11),
        )
    ]
    figure, axis = plt.subplots()
    try:
        _backoff(axis, runs)

        assert "smoke fallback" in axis.get_title()
        assert "static-2x2" in axis.get_title()
        assert "seed 410" in axis.get_title()
    finally:
        plt.close(figure)


def test_dynamic_figure_separates_active_arm_and_reward_axes() -> None:
    import matplotlib.pyplot as plt

    from dblbt_fcn.plotting import _dynamic

    run = _plot_run(
        matrix="heldout",
        scenario_id="dynamic-combined-4x4",
        policy="adaptive_db_lbt",
        seed=410,
        value=7,
        dynamic=True,
    )
    figure, axis = plt.subplots()
    try:
        _dynamic(axis, [run])

        assert {item.get_ylabel() for item in figure.axes} == {
            "Active nodes",
            "Selected arm",
            "Local reward",
        }
        assert {
            item.get_text() for item in axis.get_legend().get_texts()
        } == {"Active nodes", "Selected arm", "Local reward"}
    finally:
        plt.close(figure)


def test_heldout_utility_uses_readable_labels_values_and_scale() -> None:
    import matplotlib.pyplot as plt

    from dblbt_fcn.plotting import _utility

    rows = [
        {
            "matrix": "heldout",
            "policy": "adaptive_db_lbt",
            "evaluation_utility": 0.928023,
        },
        {
            "matrix": "heldout",
            "policy": "tmc_db_lbt",
            "evaluation_utility": 0.926467,
        },
        {
            "matrix": "heldout",
            "policy": "random_lbt",
            "evaluation_utility": 0.745797,
        },
    ]
    figure, axis = plt.subplots()
    try:
        _utility(axis, rows)

        assert [item.get_text() for item in axis.get_xticklabels()] == [
            "Random",
            "TMC",
            "Adaptive",
        ]
        assert axis.get_ylim()[0] > 0.0
        assert {item.get_text() for item in axis.texts} == {
            "0.7458",
            "0.9265",
            "0.9280",
        }
    finally:
        plt.close(figure)


def test_scaling_uses_only_reproduction_static_symmetric_rows() -> None:
    import matplotlib.pyplot as plt

    from dblbt_fcn.plotting import _scaling

    def row(
        matrix: str,
        scenario: str,
        policy: str,
        nodes: int,
        utility: float,
    ) -> dict[str, object]:
        return {
            "matrix": matrix,
            "scenario_id": scenario,
            "policy": policy,
            "wifi_nodes": nodes,
            "nru_nodes": nodes,
            "evaluation_utility": utility,
        }

    rows = [
        row("reproduction", "static-symmetric-1x1", "tmc_db_lbt", 1, 0.1),
        row("reproduction", "static-symmetric-2x2", "tmc_db_lbt", 2, 0.2),
        row("reproduction", "trace-static-4x4", "tmc_db_lbt", 4, 9.1),
        row("heldout", "symmetric-6x6", "tmc_db_lbt", 6, 9.2),
        row("ablation", "dynamic-combined-4x4", "adaptive_db_lbt", 4, 9.3),
    ]
    figure, axis = plt.subplots()
    try:
        _scaling(axis, rows)

        assert axis.get_legend_handles_labels()[1] == ["tmc_db_lbt"]
        assert list(axis.lines[0].get_xdata()) == [2, 4]
        assert list(axis.lines[0].get_ydata()) == [0.1, 0.2]
    finally:
        plt.close(figure)


def test_tradeoff_uses_only_unablated_heldout_main_policies() -> None:
    import matplotlib.pyplot as plt

    from dblbt_fcn.plotting import _tradeoff

    def row(
        matrix: str,
        policy: str,
        delay: float,
        *,
        ablation: str | None = None,
    ) -> dict[str, object]:
        return {
            "matrix": matrix,
            "policy": policy,
            "ablation": ablation,
            "p95_delay_us": delay,
            "jain_fairness": delay / 1_000,
            "effective_airtime": 0.5,
        }

    rows = [
        row("heldout", "tmc_db_lbt", 100.0),
        row("heldout", "adaptive_db_lbt", 200.0),
        row("ablation", "adaptive_db_lbt", 900.0, ablation="full"),
        row("reproduction", "random_lbt", 800.0),
        row("heldout", "pretrain_arm", 700.0),
    ]
    figure, axis = plt.subplots()
    try:
        _tradeoff(axis, rows)

        assert axis.get_legend_handles_labels()[1] == [
            "adaptive_db_lbt",
            "tmc_db_lbt",
        ]
        assert {
            float(offset[0])
            for collection in axis.collections
            for offset in collection.get_offsets()
        } == {100.0, 200.0}
    finally:
        plt.close(figure)


def test_arm_heatmap_uses_only_unablated_adaptive_heldout_runs() -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    from dblbt_fcn.plotting import _heatmap

    runs = [
        _plot_run(
            matrix="heldout",
            scenario_id="symmetric-6x6",
            policy="adaptive_db_lbt",
            seed=410,
            value=1,
        ),
        _plot_run(
            matrix="heldout",
            scenario_id="symmetric-6x6",
            policy="fixed_oracle",
            seed=410,
            value=2,
        ),
        _plot_run(
            matrix="ablation",
            scenario_id="dynamic-combined-4x4",
            policy="adaptive_db_lbt",
            seed=410,
            value=4,
            dynamic=True,
            ablation="full",
        ),
        _plot_run(
            matrix="pretrain",
            scenario_id="static-2x2",
            policy="pretrain_arm",
            seed=410,
            value=8,
        ),
    ]
    figure, axis = plt.subplots()
    try:
        _heatmap(axis, runs)

        assert np.array_equal(axis.images[0].get_array(), np.ones((24, 4)))
    finally:
        plt.close(figure)


def test_dynamic_figure_selects_one_main_condition_and_lowest_seed() -> None:
    import matplotlib.pyplot as plt

    from dblbt_fcn.plotting import _dynamic

    runs = [
        _plot_run(
            matrix="heldout",
            scenario_id="dynamic-combined-4x4",
            policy="adaptive_db_lbt",
            seed=410,
            value=19,
            dynamic=True,
        ),
        _plot_run(
            matrix="ablation",
            scenario_id="dynamic-combined-4x4",
            policy="adaptive_db_lbt",
            seed=410,
            value=18,
            dynamic=True,
            ablation="no_delay",
        ),
        _plot_run(
            matrix="reproduction",
            scenario_id="dynamic-4x4",
            policy="adaptive_db_lbt",
            seed=523,
            value=17,
            dynamic=True,
        ),
        _plot_run(
            matrix="reproduction",
            scenario_id="dynamic-4x4",
            policy="tmc_db_lbt",
            seed=410,
            value=12,
            dynamic=True,
        ),
        _plot_run(
            matrix="reproduction",
            scenario_id="dynamic-4x4",
            policy="adaptive_db_lbt",
            seed=410,
            value=11,
            dynamic=True,
        ),
    ]
    figure, axis = plt.subplots()
    try:
        _dynamic(axis, list(reversed(runs)))

        arm_axis = next(
            item for item in figure.axes if item.get_ylabel() == "Selected arm"
        )
        assert {
            int(offset[1])
            for collection in arm_axis.collections
            for offset in collection.get_offsets()
        } == {11}
        assert "reproduction" in axis.get_title()
        assert "dynamic-4x4" in axis.get_title()
        assert "seed 410" in axis.get_title()
    finally:
        plt.close(figure)


def make_formal_report_inputs(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path]:
    from dblbt_fcn.experiment import (
        JobSpec,
        ScenarioSpec,
        TimingSpec,
        artifact_paths,
        canonical_json,
    )
    from dblbt_fcn.io import RunManifest, write_jsonl_gz, write_manifest
    from dblbt_fcn.provenance import execution_provenance
    from dblbt_fcn.reporting import summarize_manifests
    from dblbt_fcn.simulation import simulate_job_records

    root = tmp_path / "formal-runs"
    model = tmp_path / "model.npz"
    write_model(model)
    oracle = tmp_path / "fixed-oracle-arm.json"
    write_oracle(oracle, model)
    from dblbt_fcn.linucb import LinUCB
    from dblbt_fcn.workflows import action_grid_hash

    agent = LinUCB.load(model, expected_action_grid_hash=action_grid_hash())
    scenarios = [
        ScenarioSpec(id="symmetric-6x6", wifi_nodes=1, nru_nodes=1),
        ScenarioSpec(
            id="dynamic-combined-4x4",
            wifi_nodes=1,
            nru_nodes=1,
            join_interval_rounds=10,
            lifetime_rounds=40,
        ),
    ]
    now = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    for scenario in scenarios:
        for policy in ("tmc_db_lbt", "adaptive_db_lbt"):
            for seed in SEEDS:
                job = JobSpec(
                    matrix="heldout",
                    rounds=1,
                    alpha=11,
                    timing=TimingSpec(),
                    scenario=scenario,
                    policy=policy,
                    seed=seed,
                )
                paths = artifact_paths(job, root)
                metadata = write_jsonl_gz(
                    paths.raw,
                    simulate_job_records(
                        job,
                        initial_agent=(
                            agent if policy == "adaptive_db_lbt" else None
                        ),
                    ),
                )
                write_manifest(
                    paths.manifest,
                    RunManifest(
                        run_id=job.run_id,
                        scenario_id=job.scenario.id,
                        policy=job.policy,
                        seed=job.seed,
                        config_hash=job.config_hash,
                        git_revision="a" * 40,
                        dependency_versions={"python": "3.12"},
                        host="audit-test",
                        started_at_utc=now,
                        ended_at_utc=now,
                        elapsed_seconds=0.0,
                        record_path=str(paths.raw),
                        record_hash=metadata.sha256,
                        row_count=metadata.row_count,
                        exit_code=0,
                        status="complete",
                        execution_provenance=execution_provenance(
                            job,
                            initial_agent=(
                                agent if policy == "adaptive_db_lbt" else None
                            ),
                            model_path=(
                                model if policy == "adaptive_db_lbt" else None
                            ),
                        ),
                    ),
                )
                config = root / "configs" / f"{job.run_id}.json"
                config.parent.mkdir(parents=True, exist_ok=True)
                config.write_bytes((canonical_json(job) + "\n").encode("ascii"))
    summary = tmp_path / "formal-summary.csv"
    summarize_manifests(root / "manifests", summary)
    return summary, root / "manifests", model, oracle


def create_report_outputs(
    summary: Path, manifests: Path, model: Path, output: Path
) -> None:
    from dblbt_fcn.plotting import generate_figures
    from dblbt_fcn.stats import generate_tables, measure_model_overhead

    generate_figures(summary, output, manifest_dir=manifests)
    clock = iter(range(0, 20_001_000, 1_000))
    overhead = measure_model_overhead(
        model,
        timer=lambda: next(clock),
        selector=lambda agent, context: 0,
    )
    generate_tables(summary, output / "tables", overhead)


def test_audit_report_accepts_complete_read_only_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dblbt_fcn.records as records
    from dblbt_fcn.audit import audit_report

    _allow_component_inventory(monkeypatch)
    summary, manifests, model, oracle = make_formal_report_inputs(tmp_path)
    output = tmp_path / "report"
    create_report_outputs(summary, manifests, model, output)
    protected = {
        path: path.read_bytes()
        for path in [summary, model, *manifests.glob("*.json")]
    }
    monkeypatch.setattr(
        records,
        "read_job_rows",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("audit must stream raw rows")
        ),
    )

    result = audit_report(manifests, summary, output, model, oracle)

    assert result.run_count == 40
    assert result.figure_count == 16
    assert result.table_count == 8
    assert {path: path.read_bytes() for path in protected} == protected


@pytest.mark.parametrize(
    "failure",
    [
        "missing_figure",
        "corrupt_figure",
        "missing_pair",
        "bad_table",
        "wrong_model",
        "blank_formal",
        "tampered_summary_metric",
        "tampered_ci",
        "tampered_per_seed",
        "tampered_latex",
    ],
)
def test_audit_report_fails_closed_on_missing_or_corrupt_evidence(
    tmp_path: Path, failure: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dblbt_fcn.audit import audit_report

    _allow_component_inventory(monkeypatch)
    summary, manifests, model, oracle = make_formal_report_inputs(tmp_path)
    output = tmp_path / "report"
    create_report_outputs(summary, manifests, model, output)
    if failure == "missing_figure":
        (output / "delay-cdf.pdf").unlink()
    elif failure == "corrupt_figure":
        (output / "delay-cdf.png").write_bytes(b"not png")
    elif failure == "missing_pair":
        rows = list(csv.DictReader(summary.open()))
        write_summary(summary, rows[:-1])
    else:
        if failure == "bad_table":
            (output / "tables" / "hypotheses.csv").write_text(
                "hypothesis,status\nH1,hidden\n", encoding="ascii"
            )
        elif failure == "wrong_model":
            from dblbt_fcn.linucb import LinUCB
            from dblbt_fcn.workflows import action_grid_hash

            changed = LinUCB(24, 11, action_grid_hash=action_grid_hash())
            changed.update(3, [1.0] * 11, 0.5)
            changed.save(model)
        elif failure == "blank_formal":
            from dblbt_fcn.experiment import JobSpec, canonical_json
            from dblbt_fcn.provenance import ExecutionProvenance

            manifest = next(
                path
                for path in manifests.glob("*.json")
                if json.loads(path.read_text())["policy"] == "adaptive_db_lbt"
            )
            config = manifests.parent / "configs" / manifest.name
            job = JobSpec.model_validate_json(config.read_bytes())
            value = json.loads(manifest.read_text())
            state_hash = value["execution_provenance"]["agent_state_sha256"]
            value["execution_provenance"] = ExecutionProvenance(
                mode="adaptive_blank", agent_state_sha256=state_hash
            ).model_dump(mode="json")
            manifest.write_bytes(
                (canonical_json(value) + "\n").encode("ascii")
            )
        elif failure == "tampered_summary_metric":
            rows = list(csv.DictReader(summary.open()))
            rows[0]["evaluation_utility"] = str(
                float(rows[0]["evaluation_utility"]) + 0.01
            )
            write_summary(summary, rows)
        elif failure == "tampered_ci":
            path = output / "tables" / "paired-comparisons.csv"
            rows = list(csv.DictReader(path.open()))
            rows[0]["lower_95"] = str(float(rows[0]["lower_95"]) + 0.01)
            with path.open("w", encoding="ascii", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=list(rows[0]), lineterminator="\n"
                )
                writer.writeheader()
                writer.writerows(rows)
        elif failure == "tampered_latex":
            (output / "tables" / "hypotheses.tex").write_text(
                "\\begin{tabular}TAMPERED\n", encoding="ascii"
            )
        else:
            path = output / "tables" / "per-seed-metrics.csv"
            rows = list(csv.DictReader(path.open()))
            rows[0]["evaluation_utility"] = str(
                float(rows[0]["evaluation_utility"]) + 0.01
            )
            with path.open("w", encoding="ascii", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=list(rows[0]), lineterminator="\n"
                )
                writer.writeheader()
                writer.writerows(rows)

    with pytest.raises((ValueError, OSError, RuntimeError)):
        audit_report(manifests, summary, output, model, oracle)


def test_task12_cli_routes_real_plot_and_audit_and_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typer.testing import CliRunner

    import dblbt_fcn.cli as cli

    observed: dict[str, tuple[object, ...]] = {}
    summary = tmp_path / "summary.csv"
    manifests = tmp_path / "manifests"
    model = tmp_path / "model.npz"
    oracle = tmp_path / "fixed-oracle-arm.json"
    output = tmp_path / "report"
    monkeypatch.setattr(
        cli,
        "generate_report",
        lambda *args: observed.setdefault("plot", args) or [],
        raising=False,
    )
    def fake_audit(*args: object) -> object:
        observed["audit"] = args
        return type("Audit", (), {"run_count": 40})()

    monkeypatch.setattr(
        cli,
        "audit_report",
        fake_audit,
        raising=False,
    )

    plot = CliRunner().invoke(
        cli.app,
        [
            "plot",
            "--summary",
            str(summary),
            "--output-dir",
            str(output),
            "--manifest-dir",
            str(manifests),
            "--model",
            str(model),
            "--oracle-arm-file",
            str(oracle),
        ],
    )
    audit = CliRunner().invoke(
        cli.app,
        [
            "audit",
            "--manifest-dir",
            str(manifests),
            "--summary",
            str(summary),
            "--output-dir",
            str(output),
            "--model",
            str(model),
            "--oracle-arm-file",
            str(oracle),
        ],
    )

    assert plot.exit_code == 0, plot.output
    assert audit.exit_code == 0, audit.output
    assert observed["plot"] == (summary, output, manifests, model, oracle)
    assert observed["audit"] == (manifests, summary, output, model, oracle)

    monkeypatch.setattr(
        cli,
        "audit_report",
        lambda *args: (_ for _ in ()).throw(ValueError("missing figure")),
    )
    failed = CliRunner().invoke(
        cli.app,
        [
            "audit",
            "--manifest-dir",
            str(manifests),
            "--summary",
            str(summary),
            "--output-dir",
            str(output),
            "--model",
            str(model),
        ],
    )
    assert failed.exit_code != 0
    assert "missing figure" in failed.output
