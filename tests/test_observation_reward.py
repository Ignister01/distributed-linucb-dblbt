"""Local observation and reward contracts."""

from dataclasses import FrozenInstanceError
from decimal import Decimal
import math

import numpy as np
import pytest


def _attempt(**changes: object):
    from dblbt_fcn.observation import AttemptRecord

    values: dict[str, object] = {
        "outcome": "success",
        "elapsed_us": 1_000,
        "busy_us": 250,
        "interruptions": 0,
        "access_delay_us": None,
        "queue_occupancy_ratio": 0.25,
        "arrivals": 0,
        "retries": 0,
        "effective_data_us": 500,
    }
    values.update(changes)
    return AttemptRecord(**values)


def test_empty_context_has_eleven_zero_features() -> None:
    from dblbt_fcn.observation import LocalWindow

    context = LocalWindow(max_attempts=64).context()

    assert context.shape == (11,)
    assert context.dtype == np.float64
    np.testing.assert_array_equal(context, np.zeros(11))


def test_attempt_record_is_immutable() -> None:
    attempt = _attempt()

    with pytest.raises(FrozenInstanceError):
        attempt.outcome = "collision"


def test_attempt_record_accepts_and_normalizes_numpy_numeric_scalars() -> None:
    from dblbt_fcn.observation import AttemptRecord, LocalWindow

    attempt = AttemptRecord(
        outcome="success",
        elapsed_us=np.float64(1_000),
        busy_us=np.int64(250),
        interruptions=np.int64(1),
        access_delay_us=np.float64(100),
        queue_occupancy_ratio=np.float64(0.5),
        arrivals=np.int64(2),
        retries=np.int64(3),
        effective_data_us=np.int64(500),
    )

    assert type(attempt.elapsed_us) is float
    assert type(attempt.busy_us) is float
    assert type(attempt.interruptions) is int
    assert type(attempt.access_delay_us) is float
    assert type(attempt.queue_occupancy_ratio) is float
    assert type(attempt.arrivals) is int
    assert type(attempt.retries) is int
    assert type(attempt.effective_data_us) is float
    window = LocalWindow()
    window.record(attempt)
    assert np.all(np.isfinite(window.context()))


def test_window_becomes_ready_at_eight_attempts_and_keeps_latest_64() -> None:
    from dblbt_fcn.observation import LocalWindow

    window = LocalWindow()
    for _ in range(7):
        window.record(_attempt())
    assert not window.ready

    window.record(_attempt())
    assert window.ready

    for _ in range(56):
        window.record(_attempt())
    window.record(_attempt(outcome="collision"))

    assert len(window) == 64
    assert window.context()[0] == pytest.approx(1 / 64)
    assert window.context()[1] == pytest.approx(63 / 64)


@pytest.mark.parametrize("max_attempts", [7, 65, True, 8.0])
def test_window_rejects_invalid_capacity(max_attempts: object) -> None:
    from dblbt_fcn.observation import LocalWindow

    with pytest.raises(ValueError, match="max_attempts"):
        LocalWindow(max_attempts=max_attempts)


def test_context_uses_local_rolling_totals_and_means_in_strict_order() -> None:
    from dblbt_fcn.observation import LocalWindow

    window = LocalWindow()
    window.record(
        _attempt(
            outcome="success",
            elapsed_us=1_000,
            busy_us=200,
            interruptions=1,
            access_delay_us=100,
            queue_occupancy_ratio=0.2,
            arrivals=1,
            retries=0,
            effective_data_us=400,
        )
    )
    window.record(
        _attempt(
            outcome="collision",
            elapsed_us=3_000,
            busy_us=1_200,
            interruptions=3,
            access_delay_us=300,
            queue_occupancy_ratio=0.6,
            arrivals=2,
            retries=2,
            effective_data_us=600,
        )
    )

    estimated_contenders = 3.0
    delay_capacity_us = 8 * 2_000 * estimated_contenders
    expected = np.array(
        [
            0.5,
            0.5,
            1_400 / 4_000,
            2 / 31,
            140 / delay_capacity_us,
            300 / delay_capacity_us,
            0.4,
            1.0,
            0.5,
            1_000 / 4_000,
            2 / 31,
        ]
    )

    np.testing.assert_allclose(window.context(), expected)


def test_delay_statistics_use_current_window_samples_only() -> None:
    from dblbt_fcn.observation import LocalWindow

    window = LocalWindow()
    window.record(_attempt(access_delay_us=10_000))
    for _ in range(63):
        window.record(_attempt(access_delay_us=100))
    window.record(_attempt(access_delay_us=300))

    context = window.context()
    assert context[4] == pytest.approx(140 / 16_000)
    assert context[5] == pytest.approx(100 / 16_000)


def test_delay_p95_uses_nearest_rank_and_missing_delays_are_zero() -> None:
    from dblbt_fcn.observation import LocalWindow

    sampled = LocalWindow()
    for delay_us in range(1, 21):
        sampled.record(_attempt(access_delay_us=delay_us))

    missing = LocalWindow()
    missing.record(_attempt(access_delay_us=None))

    assert sampled.context()[5] == pytest.approx(19 / 16_000)
    assert missing.context()[4] == 0.0
    assert missing.context()[5] == 0.0


def test_context_clips_interruption_estimate_and_normalized_features() -> None:
    from dblbt_fcn.observation import LocalWindow

    window = LocalWindow()
    window.record(
        _attempt(
            interruptions=99,
            access_delay_us=16_000,
            arrivals=100,
            retries=99,
        )
    )

    context = window.context()
    assert context[3] == 1.0
    assert context[4] == pytest.approx(16_000 / (16_000 * 32))
    assert context[7] == 1.0
    assert context[8] == pytest.approx(99 / 100)
    assert context[10] == 1.0
    assert np.all((0.0 <= context) & (context <= 1.0))


def test_window_rejects_non_attempt_records() -> None:
    from dblbt_fcn.observation import LocalWindow

    with pytest.raises(ValueError, match="AttemptRecord"):
        LocalWindow().record(object())


@pytest.mark.parametrize(
    ("changes", "field"),
    [
        ({"outcome": "timeout"}, "outcome"),
        ({"outcome": 1}, "outcome"),
        ({"elapsed_us": 0}, "elapsed_us"),
        ({"elapsed_us": -1}, "elapsed_us"),
        ({"elapsed_us": math.nan}, "elapsed_us"),
        ({"elapsed_us": math.inf}, "elapsed_us"),
        ({"busy_us": -1}, "busy_us"),
        ({"busy_us": math.nan}, "busy_us"),
        ({"busy_us": math.inf}, "busy_us"),
        ({"busy_us": 1_001}, "busy_us"),
        ({"interruptions": -1}, "interruptions"),
        ({"interruptions": math.nan}, "interruptions"),
        ({"access_delay_us": -1}, "access_delay_us"),
        ({"access_delay_us": math.inf}, "access_delay_us"),
        ({"queue_occupancy_ratio": -0.1}, "queue_occupancy_ratio"),
        ({"queue_occupancy_ratio": 1.1}, "queue_occupancy_ratio"),
        ({"queue_occupancy_ratio": math.nan}, "queue_occupancy_ratio"),
        ({"arrivals": -1}, "arrivals"),
        ({"arrivals": math.inf}, "arrivals"),
        ({"retries": -1}, "retries"),
        ({"retries": math.nan}, "retries"),
        ({"effective_data_us": -1}, "effective_data_us"),
        ({"effective_data_us": math.inf}, "effective_data_us"),
        ({"effective_data_us": 1_001}, "effective_data_us"),
    ],
)
def test_attempt_record_rejects_invalid_inputs(
    changes: dict[str, object], field: str
) -> None:
    with pytest.raises(ValueError, match=field):
        _attempt(**changes)


@pytest.mark.parametrize(
    "value",
    [
        True,
        np.float64(math.nan),
        np.float64(math.inf),
        1 + 0j,
        "1.0",
        Decimal("1.0"),
    ],
)
def test_attempt_record_rejects_non_finite_or_non_real_values(
    value: object,
) -> None:
    with pytest.raises(ValueError, match="elapsed_us"):
        _attempt(elapsed_us=value)


@pytest.mark.parametrize(
    "value",
    [True, np.float64(1.0), 1.0, 1 + 0j, "1", Decimal("1")],
)
def test_attempt_record_integer_counts_require_non_boolean_integrals(
    value: object,
) -> None:
    with pytest.raises(ValueError, match="interruptions"):
        _attempt(interruptions=value)


def test_reward_formula_preserves_component_api() -> None:
    from dblbt_fcn.reward import local_reward

    assert local_reward(0.9, 0.8, 0.7, 0.2, 0.25) == pytest.approx(0.75)


def test_component_reward_accepts_finite_numpy_reals() -> None:
    from dblbt_fcn.reward import local_reward

    reward = local_reward(
        np.float64(0.9),
        np.float64(0.8),
        np.float64(0.7),
        np.float64(0.2),
        np.float64(0.25),
    )

    assert type(reward) is float
    assert reward == pytest.approx(0.75)


def test_raw_reward_accepts_numpy_context_scalars() -> None:
    from dblbt_fcn.observation import LocalWindow
    from dblbt_fcn.reward import local_reward_components

    window = LocalWindow()
    window.record(_attempt(interruptions=1))
    context = window.context()
    estimated_contenders = context[10] * 31 + 1

    components = local_reward_components(
        estimated_contenders,
        context[9],
        np.int64(100),
        context[9],
        context[0],
    )

    assert all(
        type(value) is float
        for value in (
            components.airtime_utility,
            components.delay_utility,
            components.share_utility,
            components.reward,
        )
    )


def test_interval_reward_accepts_finite_numpy_reals_and_integrals() -> None:
    from dblbt_fcn.reward import local_reward_from_interval

    components = local_reward_from_interval(
        np.float64(2),
        np.int64(200),
        np.float64(1_000),
        np.int64(100),
        np.float64(0.5),
        np.float64(0.1),
    )

    assert type(components.reward) is float
    assert math.isfinite(components.reward)


@pytest.mark.parametrize(
    "value",
    [
        True,
        np.float64(math.nan),
        np.float64(math.inf),
        1 + 0j,
        "0.5",
        Decimal("0.5"),
    ],
)
def test_reward_apis_reject_non_finite_or_non_real_values(
    value: object,
) -> None:
    from dblbt_fcn.reward import (
        local_reward,
        local_reward_components,
        local_reward_from_interval,
    )

    with pytest.raises(ValueError, match="airtime_utility"):
        local_reward(value, 0.5, 0.5, 0.1)
    with pytest.raises(ValueError, match="estimated_contenders"):
        local_reward_components(value, 0.2, 100, 0.5, 0.1)
    with pytest.raises(ValueError, match="elapsed_us"):
        local_reward_from_interval(2, 200, value, 100, 0.5, 0.1)


def test_reward_components_follow_local_raw_formulas() -> None:
    from dblbt_fcn.reward import local_reward_components

    components = local_reward_components(
        estimated_contenders=4,
        local_airtime=0.2,
        delay_p95_us=32_000,
        local_share=0.125,
        collision_probability=0.4,
    )

    assert components.airtime_utility == pytest.approx(0.8)
    assert components.delay_utility == pytest.approx(0.5)
    assert components.share_utility == pytest.approx(0.5)
    assert components.reward == pytest.approx(0.5)


def test_reward_components_clip_raw_utilities() -> None:
    from dblbt_fcn.reward import local_reward_components

    components = local_reward_components(2, 0.8, 1_000_000, 1.0, 0.0)

    assert components.airtime_utility == 1.0
    assert components.delay_utility == 0.0
    assert components.share_utility == 0.0
    assert components.reward == pytest.approx(1 / 3)


def test_interval_reward_derives_local_airtime_without_dividing_by_zero() -> None:
    from dblbt_fcn.reward import (
        local_reward_components,
        local_reward_from_interval,
    )

    from_interval = local_reward_from_interval(
        estimated_contenders=2,
        effective_data_us=200,
        elapsed_us=1_000,
        delay_p95_us=1_000,
        local_share=0.5,
        collision_probability=0.1,
    )
    from_ratio = local_reward_components(2, 0.2, 1_000, 0.5, 0.1)

    assert from_interval == from_ratio
    with pytest.raises(ValueError, match="elapsed_us"):
        local_reward_from_interval(2, 0, 0, 0, 0.5, 0)


@pytest.mark.parametrize(
    ("arguments", "field"),
    [
        ((0, 0.2, 100, 0.5, 0.1), "estimated_contenders"),
        ((33, 0.2, 100, 0.5, 0.1), "estimated_contenders"),
        ((math.nan, 0.2, 100, 0.5, 0.1), "estimated_contenders"),
        ((math.inf, 0.2, 100, 0.5, 0.1), "estimated_contenders"),
        ((2, -0.1, 100, 0.5, 0.1), "local_airtime"),
        ((2, 1.1, 100, 0.5, 0.1), "local_airtime"),
        ((2, math.nan, 100, 0.5, 0.1), "local_airtime"),
        ((2, 0.2, -1, 0.5, 0.1), "delay_p95_us"),
        ((2, 0.2, math.inf, 0.5, 0.1), "delay_p95_us"),
        ((2, 0.2, 100, -0.1, 0.1), "local_share"),
        ((2, 0.2, 100, 1.1, 0.1), "local_share"),
        ((2, 0.2, 100, math.nan, 0.1), "local_share"),
        ((2, 0.2, 100, 0.5, -0.1), "collision_probability"),
        ((2, 0.2, 100, 0.5, 1.1), "collision_probability"),
        ((2, 0.2, 100, 0.5, math.inf), "collision_probability"),
    ],
)
def test_reward_components_reject_invalid_raw_inputs(
    arguments: tuple[object, ...], field: str
) -> None:
    from dblbt_fcn.reward import local_reward_components

    with pytest.raises(ValueError, match=field):
        local_reward_components(*arguments)


@pytest.mark.parametrize("collision_weight", [-0.1, math.nan, math.inf])
def test_reward_rejects_invalid_collision_weight(
    collision_weight: object,
) -> None:
    from dblbt_fcn.reward import local_reward, local_reward_components

    with pytest.raises(ValueError, match="collision_weight"):
        local_reward(0.5, 0.5, 0.5, 0.5, collision_weight)
    with pytest.raises(ValueError, match="collision_weight"):
        local_reward_components(2, 0.2, 100, 0.5, 0.1, collision_weight)


@pytest.mark.parametrize(
    ("arguments", "field"),
    [
        ((2, 0, -1, 0, 0.5, 0), "elapsed_us"),
        ((2, 0, math.nan, 0, 0.5, 0), "elapsed_us"),
        ((2, 0, math.inf, 0, 0.5, 0), "elapsed_us"),
        ((2, -1, 1_000, 0, 0.5, 0), "effective_data_us"),
        ((2, math.nan, 1_000, 0, 0.5, 0), "effective_data_us"),
        ((2, 1_001, 1_000, 0, 0.5, 0), "effective_data_us"),
    ],
)
def test_interval_reward_rejects_invalid_times(
    arguments: tuple[object, ...], field: str
) -> None:
    from dblbt_fcn.reward import local_reward_from_interval

    with pytest.raises(ValueError, match=field):
        local_reward_from_interval(*arguments)


@pytest.mark.parametrize(
    ("arguments", "field"),
    [
        ((-0.1, 0.5, 0.5, 0.1), "airtime_utility"),
        ((0.5, 1.1, 0.5, 0.1), "delay_utility"),
        ((0.5, 0.5, math.nan, 0.1), "share_utility"),
        ((0.5, 0.5, 0.5, math.inf), "collision_probability"),
    ],
)
def test_component_reward_rejects_invalid_inputs(
    arguments: tuple[object, ...], field: str
) -> None:
    from dblbt_fcn.reward import local_reward

    with pytest.raises(ValueError, match=field):
        local_reward(*arguments)
