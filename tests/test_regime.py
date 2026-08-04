from dataclasses import replace
from pathlib import Path

import pytest

import dblbt_fcn.regime_plotting as regime_plotting
from dblbt_fcn.experiment import MatrixSpec, load_matrix
from dblbt_fcn.regime import (
    CONFIRMATION_SEEDS,
    confirmation_matrix,
    scenario_effects,
    select_confirmation_scenarios,
    write_confirmation_matrix,
    write_effects_csv,
)
from dblbt_fcn.regime_plotting import (
    _confirmation_figure,
    generate_regime_figures,
)


def _row(
    scenario: str,
    policy: str,
    seed: int,
    *,
    utility: float,
    collision: float = 0.2,
    airtime: float = 0.7,
    delay: float = 100.0,
    fairness: float = 0.9,
) -> dict[str, object]:
    return {
        "scenario_id": scenario,
        "policy": policy,
        "seed": seed,
        "ablation": None,
        "arm_id": None,
        "evaluation_utility": utility,
        "collision_probability": collision,
        "effective_airtime": airtime,
        "p95_delay_us": delay,
        "jain_fairness": fairness,
    }


def _paired_rows(
    scenario: str = "load-p025",
    *,
    baseline_utility: tuple[float, ...] = (0.70, 0.72, 0.71),
    candidate_utility: tuple[float, ...] = (0.72, 0.74, 0.73),
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed, baseline, candidate in zip(
        (1, 2, 3), baseline_utility, candidate_utility, strict=True
    ):
        rows.append(
            _row(
                scenario,
                "tmc_db_lbt",
                seed,
                utility=baseline,
                collision=0.20,
                airtime=0.70,
                delay=120.0,
                fairness=0.91,
            )
        )
        rows.append(
            _row(
                scenario,
                "adaptive_db_lbt",
                seed,
                utility=candidate,
                collision=0.18,
                airtime=0.73,
                delay=100.0,
                fairness=0.90,
            )
        )
    return rows


def test_scenario_effects_require_matching_policy_seeds() -> None:
    rows = _paired_rows()
    rows.pop()

    with pytest.raises(ValueError, match="paired seeds"):
        scenario_effects(rows)


def test_scenario_effects_decompose_candidate_minus_baseline() -> None:
    effect = scenario_effects(_paired_rows())[0]

    assert effect.scenario_id == "load-p025"
    assert effect.family == "load"
    assert effect.seed_count == 3
    assert effect.baseline_mean == pytest.approx(0.71)
    assert effect.candidate_mean == pytest.approx(0.73)
    assert effect.utility_difference == pytest.approx(0.02)
    assert effect.relative_difference == pytest.approx(0.02 / 0.71)
    assert effect.lower_95 > 0
    assert effect.upper_95 > 0
    assert effect.positive_seeds == 3
    assert effect.collision_difference == pytest.approx(-0.02)
    assert effect.effective_airtime_difference == pytest.approx(0.03)
    assert effect.p95_delay_difference == pytest.approx(-20.0)
    assert effect.fairness_difference == pytest.approx(-0.01)


def test_selection_keeps_one_eligible_scenario_per_family() -> None:
    load_a = scenario_effects(_paired_rows("load-a"))[0]
    load_b = scenario_effects(
        _paired_rows(
            "load-b",
            baseline_utility=(0.70, 0.70, 0.70),
            candidate_utility=(0.73, 0.73, 0.73),
        )
    )[0]
    occupancy = scenario_effects(_paired_rows("occupancy-a"))[0]
    unfair = replace(
        scenario_effects(_paired_rows("turnover-a"))[0],
        fairness_difference=-0.02,
    )
    inconsistent = replace(
        scenario_effects(_paired_rows("sensing-a"))[0],
        positive_seeds=2,
    )

    selected = select_confirmation_scenarios(
        (load_a, load_b, occupancy, unfair, inconsistent)
    )

    assert selected == ("load-b", "occupancy-a")


def _pilot_matrix() -> MatrixSpec:
    base = {
        "wifi_nodes": 4,
        "nru_nodes": 4,
        "traffic": "poisson",
        "poisson_rate_packets_ms": 0.025,
    }
    return MatrixSpec.model_validate(
        {
            "version": 1,
            "name": "pilot",
            "rounds": 20_000,
            "seeds": [1, 2, 3],
            "policies": ["tmc_db_lbt", "adaptive_db_lbt"],
            "scenarios": [
                {"id": "load-a", **base},
                {
                    "id": "occupancy-a",
                    **base,
                    "interference_interval_ms": 10,
                    "interference_duration_us": 500,
                },
            ],
        }
    )


def test_confirmation_matrix_uses_untouched_seeds_and_exact_scenarios() -> None:
    matrix = confirmation_matrix(_pilot_matrix(), ("occupancy-a",))

    assert matrix.name == "linucb-regime-confirmation"
    assert matrix.rounds == 100_000
    assert matrix.seeds == CONFIRMATION_SEEDS
    assert matrix.policies == (
        "primary_db_lbt",
        "tmc_db_lbt",
        "adaptive_db_lbt",
    )
    assert tuple(scenario.id for scenario in matrix.scenarios) == ("occupancy-a",)
    assert matrix.scenarios[0] == _pilot_matrix().scenarios[1]


def test_confirmation_matrix_rejects_unknown_or_duplicate_selection() -> None:
    with pytest.raises(ValueError, match="not present"):
        confirmation_matrix(_pilot_matrix(), ("missing",))
    with pytest.raises(ValueError, match="duplicate"):
        confirmation_matrix(_pilot_matrix(), ("load-a", "load-a"))


def test_confirmation_matrix_accepts_isolated_name_rounds_and_seeds() -> None:
    matrix = confirmation_matrix(
        _pilot_matrix(),
        ("load-a",),
        name="large-effect-confirmation",
        rounds=120_000,
        seeds=(6101, 6113),
    )

    assert matrix.name == "large-effect-confirmation"
    assert matrix.rounds == 120_000
    assert matrix.seeds == (6101, 6113)


def test_confirmation_matrix_rejects_invalid_custom_seed_contract() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        confirmation_matrix(_pilot_matrix(), ("load-a",), seeds=())
    with pytest.raises(ValueError, match="unique"):
        confirmation_matrix(_pilot_matrix(), ("load-a",), seeds=(6101, 6101))


def test_effect_and_confirmation_writers_are_reloadable(tmp_path: Path) -> None:
    effect = scenario_effects(_paired_rows())[0]
    effects_path = tmp_path / "effects.csv"
    matrix_path = tmp_path / "confirmation.yaml"

    write_effects_csv((effect,), effects_path)
    write_confirmation_matrix(
        confirmation_matrix(_pilot_matrix(), ("load-a",)), matrix_path
    )

    header, row = effects_path.read_text(encoding="ascii").splitlines()
    assert header.startswith("scenario_id,family,seed_count")
    assert row.startswith("load-p025,load,3")
    assert load_matrix(matrix_path).name == "linucb-regime-confirmation"


def test_generate_regime_figures_writes_png_and_pdf_outputs(
    tmp_path: Path,
) -> None:
    pilot_rows: list[object] = []
    for scenario, difference in (
        ("load-p015", 0.016),
        ("load-p055", 0.003),
        ("load-p065", -0.001),
        ("occupancy-i010-d0500", 0.011),
        ("occupancy-i010-d5000", -0.001),
        ("combined-a", 0.018),
        ("combined-f", -0.007),
    ):
        effect = scenario_effects(_paired_rows(scenario))[0]
        pilot_rows.append(
            replace(
                effect,
                utility_difference=difference,
                lower_95=difference - 0.001,
                upper_95=difference + 0.001,
            )
        )
    confirmation_rows = [
        replace(
            scenario_effects(_paired_rows("load-p025"))[0],
            utility_difference=0.018,
            lower_95=0.017,
            upper_95=0.019,
            collision_difference=-0.07,
        ),
        replace(
            scenario_effects(_paired_rows("turnover-j10-l200"))[0],
            utility_difference=0.021,
            lower_95=0.020,
            upper_95=0.022,
            collision_difference=-0.05,
        ),
    ]
    pilot = tmp_path / "pilot.csv"
    confirmation = tmp_path / "confirmation.csv"
    write_effects_csv(pilot_rows, pilot)
    write_effects_csv(confirmation_rows, confirmation)

    outputs = generate_regime_figures(pilot, confirmation, tmp_path / "figures")

    assert {path.name for path in outputs} == {
        "pilot-regime-map.pdf",
        "pilot-regime-map.png",
        "confirmed-gains.pdf",
        "confirmed-gains.png",
    }
    assert all(path.stat().st_size > 1_000 for path in outputs)


def test_generate_regime_figures_supports_density_load_search(
    tmp_path: Path,
) -> None:
    pilot_rows: list[object] = []
    for scenario, difference in (
        ("load-n03-p015", 0.012),
        ("load-n06-p015", 0.019),
        ("turnover-n03-p020-j10-l200", 0.020),
        ("turnover-n06-p030-j10-l200", 0.210),
        ("combined-n03-p020", 0.015),
        ("combined-n06-p030", 0.209),
    ):
        effect = scenario_effects(_paired_rows(scenario))[0]
        pilot_rows.append(
            replace(
                effect,
                utility_difference=difference,
                lower_95=difference - 0.001,
                upper_95=difference + 0.001,
            )
        )
    confirmation_rows = [
        replace(
            scenario_effects(_paired_rows("combined-n06-p030"))[0],
            utility_difference=0.209,
            lower_95=0.208,
            upper_95=0.211,
            collision_difference=-0.036,
        ),
        replace(
            scenario_effects(
                _paired_rows("turnover-n06-p030-j10-l200")
            )[0],
            utility_difference=0.214,
            lower_95=0.212,
            upper_95=0.215,
            collision_difference=-0.043,
        ),
    ]
    pilot = tmp_path / "pilot.csv"
    confirmation = tmp_path / "confirmation.csv"
    write_effects_csv(pilot_rows, pilot)
    write_effects_csv(confirmation_rows, confirmation)

    outputs = generate_regime_figures(pilot, confirmation, tmp_path / "figures")

    assert {path.name for path in outputs} == {
        "pilot-regime-map.pdf",
        "pilot-regime-map.png",
        "confirmed-gains.pdf",
        "confirmed-gains.png",
    }
    assert all(path.stat().st_size > 1_000 for path in outputs)


def test_confirmation_figure_leaves_room_for_value_labels() -> None:
    effect = replace(
        scenario_effects(_paired_rows("combined-n06-p030"))[0],
        utility_difference=0.210,
        lower_95=0.208,
        upper_95=0.212,
        collision_difference=-0.036,
    )

    figure = _confirmation_figure((effect,))

    assert figure.axes[0].get_xlim()[1] >= 0.210 * 1.15
    assert figure.axes[1].get_xlim()[1] >= 0.036 * 1.15


def test_confirmation_figure_uses_latency_for_large_delay_effects() -> None:
    effect = replace(
        scenario_effects(_paired_rows("combined-n06-p030"))[0],
        p95_delay_difference=-468_600.0,
    )

    figure = _confirmation_figure((effect,))

    assert figure.axes[1].get_title() == "Latency mechanism"
    assert figure.axes[1].get_xlabel() == "P95-delay reduction (ms)"


def test_confirmation_figure_uses_readable_large_effect_labels() -> None:
    effect = scenario_effects(
        _paired_rows("turnover-n06-p030-j10-l200")
    )[0]

    figure = _confirmation_figure((effect,))

    labels = [label.get_text() for label in figure.axes[0].get_yticklabels()]
    assert labels == ["Turnover (6+6, rate 0.030)"]


def test_effect_label_normalizes_rounded_negative_zero() -> None:
    assert regime_plotting._effect_label(-0.0001) == "0.000"
    assert regime_plotting._effect_label(-0.001) == "-0.001"
