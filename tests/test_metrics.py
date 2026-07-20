import math

import pytest

from dblbt_fcn.channel import RoundResult


def _success(
    node_id: str,
    effective_data_us: int = 100,
    round_id: int = 0,
    now_us: int | None = None,
) -> RoundResult:
    return RoundResult(
        round_id=round_id,
        now_us=round_id * 2_000 if now_us is None else now_us,
        kind="success",
        node_ids=[node_id],
        technologies=["wifi"],
        collision_size=0,
        reservation_us=0,
        effective_data_us=effective_data_us,
    )


def _collision(round_id: int = 0) -> RoundResult:
    return RoundResult(
        round_id=round_id,
        now_us=round_id * 2_000,
        kind="collision",
        node_ids=["wifi-0", "nru-0"],
        technologies=["wifi", "nru"],
        collision_size=2,
        reservation_us=0,
        effective_data_us=0,
    )


def test_jain_equal_shares_is_one() -> None:
    from dblbt_fcn.metrics import jain

    assert jain([0.25, 0.25, 0.25, 0.25]) == 1.0


def test_jain_handles_zero_unequal_and_single_use_generator() -> None:
    from dblbt_fcn.metrics import jain

    assert jain([0, 0, 0]) == 0.0
    assert jain([1, 2]) == pytest.approx(0.9)
    assert jain(value for value in [2.0, 2.0, 2.0]) == 1.0


@pytest.mark.parametrize(
    "values",
    [[], [-1.0, 1.0], [True], [math.nan], [math.inf], [-math.inf], ["1"]],
)
def test_jain_rejects_invalid_values(values: list[object]) -> None:
    from dblbt_fcn.metrics import jain

    with pytest.raises(ValueError):
        jain(values)


def test_collision_probability_counts_collision_rounds() -> None:
    from dblbt_fcn.metrics import collision_probability

    assert collision_probability(
        [_success("wifi-0"), _collision(1), _success("nru-0", round_id=2)]
    ) == pytest.approx(1.0 / 3.0)


@pytest.mark.parametrize("results", [[], [object()]])
def test_collision_probability_rejects_invalid_results(
    results: list[object],
) -> None:
    from dblbt_fcn.metrics import collision_probability

    with pytest.raises(ValueError):
        collision_probability(results)


@pytest.mark.parametrize(
    "results",
    [
        [
            _success("wifi-0", round_id=1, now_us=100),
            _success("wifi-0", round_id=1, now_us=200),
        ],
        [
            _success("wifi-0", round_id=2, now_us=100),
            _success("wifi-0", round_id=1, now_us=200),
        ],
        [
            _success("wifi-0", round_id=1, now_us=100),
            _success("wifi-0", round_id=2, now_us=100),
        ],
        [
            _success("wifi-0", round_id=1, now_us=200),
            _success("wifi-0", round_id=2, now_us=100),
        ],
    ],
    ids=[
        "duplicate-round-id",
        "decreasing-round-id",
        "duplicate-now-us",
        "decreasing-now-us",
    ],
)
@pytest.mark.parametrize(
    "reducer",
    ["collision_probability", "normalized_airtime", "per_node_airtime"],
)
def test_result_reducers_reject_non_increasing_rounds(
    results: list[RoundResult], reducer: str
) -> None:
    from dblbt_fcn.metrics import (
        collision_probability,
        normalized_effective_airtime,
        per_node_effective_airtime,
    )

    with pytest.raises(ValueError, match="strictly increasing"):
        if reducer == "collision_probability":
            collision_probability(results)
        elif reducer == "normalized_airtime":
            normalized_effective_airtime(results, 1_000)
        else:
            per_node_effective_airtime(results, ["wifi-0"], 1_000)


@pytest.mark.parametrize(
    "reducer",
    ["collision_probability", "normalized_airtime", "per_node_airtime"],
)
def test_result_reducers_consume_valid_gapped_generator_once(
    reducer: str,
) -> None:
    from dblbt_fcn.metrics import (
        collision_probability,
        normalized_effective_airtime,
        per_node_effective_airtime,
    )

    source = [
        _success("wifi-0", round_id=10, now_us=100),
        _success("wifi-0", round_id=12, now_us=300),
    ]
    consumed = []

    def generate_results():
        for result in source:
            consumed.append((result.round_id, result.now_us))
            yield result

    results = generate_results()
    if reducer == "collision_probability":
        assert collision_probability(results) == 0.0
    elif reducer == "normalized_airtime":
        assert normalized_effective_airtime(results, 1_000) == 0.2
    else:
        assert per_node_effective_airtime(
            results, ["wifi-0"], 1_000
        ) == {"wifi-0": 0.2}
    assert consumed == [(10, 100), (12, 300)]


def test_normalized_effective_airtime_counts_only_successes() -> None:
    from dblbt_fcn.metrics import normalized_effective_airtime

    assert normalized_effective_airtime(
        [_success("wifi-0", 80), _collision(1), _success("nru-0", 70, 2)],
        300,
    ) == pytest.approx(0.5)


def test_normalized_effective_airtime_rejects_inconsistent_input() -> None:
    from dblbt_fcn.metrics import normalized_effective_airtime

    with pytest.raises(ValueError, match="airtime"):
        normalized_effective_airtime([_success("wifi-0", 101)], 100)


@pytest.mark.parametrize("total_elapsed_us", [0, -1, True, math.nan, math.inf])
def test_normalized_effective_airtime_rejects_invalid_elapsed(
    total_elapsed_us: object,
) -> None:
    from dblbt_fcn.metrics import normalized_effective_airtime

    with pytest.raises(ValueError, match="total_elapsed_us"):
        normalized_effective_airtime([_success("wifi-0")], total_elapsed_us)


def test_nearest_rank_percentile_uses_one_based_ceiling_rank() -> None:
    from dblbt_fcn.metrics import nearest_rank_percentile

    assert nearest_rank_percentile([4, 1, 3, 2], 50) == 2
    assert nearest_rank_percentile((value for value in [1, 2, 3, 4]), 75) == 3


def test_nearest_rank_p95_wraps_nearest_rank_percentile() -> None:
    from dblbt_fcn.metrics import nearest_rank_p95

    assert nearest_rank_p95(range(1, 21)) == 19


@pytest.mark.parametrize(
    ("values", "percentile"),
    [
        ([], 95),
        ([-1], 95),
        ([True], 95),
        ([math.nan], 95),
        ([math.inf], 95),
        ([1], 0),
        ([1], -1),
        ([1], 101),
        ([1], True),
        ([1], math.nan),
        ([1], math.inf),
    ],
)
def test_nearest_rank_percentile_rejects_invalid_input(
    values: list[object], percentile: object
) -> None:
    from dblbt_fcn.metrics import nearest_rank_percentile

    with pytest.raises(ValueError):
        nearest_rank_percentile(values, percentile)


def test_per_node_effective_airtime_preserves_declared_nodes_and_order() -> None:
    from dblbt_fcn.metrics import jain, per_node_effective_airtime

    airtime = per_node_effective_airtime(
        [_success("nru-0", 50), _collision(1)],
        ["wifi-0", "nru-0", "wifi-1"],
        200,
    )

    assert airtime == {"wifi-0": 0.0, "nru-0": 0.25, "wifi-1": 0.0}
    assert list(airtime) == ["wifi-0", "nru-0", "wifi-1"]
    assert jain(airtime.values()) == pytest.approx(1.0 / 3.0)


@pytest.mark.parametrize("node_ids", [[], ["wifi-0", "wifi-0"], [""], [True]])
def test_per_node_effective_airtime_rejects_invalid_node_ids(
    node_ids: list[object],
) -> None:
    from dblbt_fcn.metrics import per_node_effective_airtime

    with pytest.raises(ValueError, match="node_ids"):
        per_node_effective_airtime([_success("wifi-0")], node_ids, 100)


def test_per_node_effective_airtime_rejects_undeclared_result_node() -> None:
    from dblbt_fcn.metrics import per_node_effective_airtime

    with pytest.raises(ValueError, match="declared"):
        per_node_effective_airtime(
            [_success("nru-0")], ["wifi-0"], 100
        )


def test_evaluation_utility_matches_spec() -> None:
    from dblbt_fcn.metrics import evaluation_utility

    value = evaluation_utility(0.9, 100.0, 1.0, 0.1)

    assert value == pytest.approx((0.9 + 0.8 + 1.0) / 3 - 0.025)


def test_evaluation_utility_clips_airtime_and_delay_terms() -> None:
    from dblbt_fcn.metrics import evaluation_utility

    assert evaluation_utility(2.0, 1_000.0, 0.5, 0.2) == pytest.approx(
        (1.0 + 0.0 + 0.5) / 3.0 - 0.05
    )
    assert evaluation_utility(-1.0, 0.0, 0.5, 0.0) == pytest.approx(0.5)


@pytest.mark.parametrize(
    "arguments",
    [
        (math.nan, 100.0, 1.0, 0.1),
        (math.inf, 100.0, 1.0, 0.1),
        (True, 100.0, 1.0, 0.1),
        (0.9, -1.0, 1.0, 0.1),
        (0.9, math.nan, 1.0, 0.1),
        (0.9, 100.0, -0.1, 0.1),
        (0.9, 100.0, 1.1, 0.1),
        (0.9, 100.0, True, 0.1),
        (0.9, 100.0, 1.0, -0.1),
        (0.9, 100.0, 1.0, 1.1),
        (0.9, 100.0, 1.0, math.inf),
    ],
)
def test_evaluation_utility_rejects_invalid_input(
    arguments: tuple[object, object, object, object],
) -> None:
    from dblbt_fcn.metrics import evaluation_utility

    with pytest.raises(ValueError):
        evaluation_utility(*arguments)
