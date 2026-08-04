from pathlib import Path

from dblbt_fcn.experiment import expand_matrix, load_matrix
from dblbt_fcn.training import HELD_OUT_SEEDS, PRETRAINING_SEEDS


MATRIX = Path("experiments/linucb-large-effect-search/pilot.yaml")
SEARCH_SEEDS = (5101, 5107, 5113)
CONFIRMATION_SEEDS = (
    6101,
    6113,
    6121,
    6131,
    6133,
    6143,
    6151,
    6163,
    6173,
    6197,
)


def test_large_effect_matrix_has_fixed_factor_contract() -> None:
    matrix = load_matrix(MATRIX)

    assert matrix.name == "linucb-large-effect-pilot"
    assert matrix.rounds == 20_000
    assert matrix.seeds == SEARCH_SEEDS
    assert matrix.policies == ("tmc_db_lbt", "adaptive_db_lbt")
    assert not set(SEARCH_SEEDS) & set(CONFIRMATION_SEEDS)
    assert not set(SEARCH_SEEDS) & set(PRETRAINING_SEEDS)
    assert not set(SEARCH_SEEDS) & set(HELD_OUT_SEEDS)

    family_counts: dict[str, int] = {}
    for scenario in matrix.scenarios:
        family = scenario.id.split("-", 1)[0]
        family_counts[family] = family_counts.get(family, 0) + 1
        assert scenario.wifi_nodes == scenario.nru_nodes
        assert 3 <= scenario.wifi_nodes <= 6
        assert scenario.traffic == "poisson"
    assert family_counts == {"load": 16, "turnover": 24, "combined": 16}
    assert len(matrix.scenarios) == 56
    assert len(expand_matrix(matrix)) == 336


def test_large_effect_channel_controls_remain_exogenous() -> None:
    matrix = load_matrix(MATRIX)

    load = [scenario for scenario in matrix.scenarios if scenario.id.startswith("load-")]
    turnover = [
        scenario for scenario in matrix.scenarios if scenario.id.startswith("turnover-")
    ]
    combined = [
        scenario for scenario in matrix.scenarios if scenario.id.startswith("combined-")
    ]
    assert all(scenario.interference_interval_ms is None for scenario in load)
    assert all(scenario.join_interval_rounds is None for scenario in load)
    assert all(scenario.join_interval_rounds is not None for scenario in turnover)
    assert all(scenario.interference_interval_ms is None for scenario in turnover)
    assert all(scenario.interference_interval_ms == 30 for scenario in combined)
    assert all(scenario.interference_duration_us == 2_000 for scenario in combined)
    assert all(scenario.interruption_std == 0.4 for scenario in combined)
    assert all(scenario.join_interval_rounds == 10 for scenario in combined)
    assert all(scenario.lifetime_rounds == 200 for scenario in combined)
