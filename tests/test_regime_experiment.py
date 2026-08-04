from pathlib import Path

from dblbt_fcn.experiment import expand_matrix, load_matrix
from dblbt_fcn.training import HELD_OUT_SEEDS, PRETRAINING_SEEDS


PILOT_MATRIX = Path("experiments/linucb-regime-discovery/pilot.yaml")


def test_regime_pilot_matrix_is_isolated_and_factor_complete() -> None:
    matrix = load_matrix(PILOT_MATRIX)

    assert matrix.name == "linucb-regime-pilot"
    assert matrix.rounds == 20_000
    assert matrix.seeds == (1709, 1871, 1999)
    assert matrix.policies == ("tmc_db_lbt", "adaptive_db_lbt")
    assert not set(matrix.seeds) & set(PRETRAINING_SEEDS)
    assert not set(matrix.seeds) & set(HELD_OUT_SEEDS)

    family_counts: dict[str, int] = {}
    for scenario in matrix.scenarios:
        family = scenario.id.split("-", 1)[0]
        family_counts[family] = family_counts.get(family, 0) + 1
    assert family_counts == {
        "load": 8,
        "occupancy": 8,
        "turnover": 6,
        "sensing": 6,
        "combined": 8,
    }
    assert len(matrix.scenarios) == 36
    assert len(expand_matrix(matrix)) == 216


def test_regime_pilot_uses_only_supported_channel_controls() -> None:
    matrix = load_matrix(PILOT_MATRIX)

    occupancy = [
        scenario
        for scenario in matrix.scenarios
        if scenario.id.startswith(("occupancy-", "combined-"))
    ]
    assert occupancy
    assert all(
        scenario.interference_interval_ms is not None
        and scenario.interference_duration_us is not None
        for scenario in occupancy
    )
    assert all(scenario.wifi_nodes > 0 for scenario in matrix.scenarios)
    assert all(scenario.nru_nodes > 0 for scenario in matrix.scenarios)
