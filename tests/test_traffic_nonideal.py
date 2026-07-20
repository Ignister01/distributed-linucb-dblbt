"""Traffic sources and explicit non-ideal channel events."""

from dataclasses import FrozenInstanceError
from importlib import import_module
import math
import random
import sys

import pytest

from dblbt_fcn.channel import Channel, Node
from dblbt_fcn.types import PolicyKind, Technology


def _traffic() -> object:
    return import_module("dblbt_fcn.traffic")


def _nonideal() -> object:
    return import_module("dblbt_fcn.nonideal")


@pytest.mark.parametrize("module_name", ["dblbt_fcn.traffic", "dblbt_fcn.nonideal"])
def test_public_module_is_available(module_name: str) -> None:
    assert import_module(module_name) is not None


def test_active_window_uses_half_open_round_interval() -> None:
    window = _traffic().ActiveWindow(start_round=10, lifetime_rounds=200)

    assert window.active(9) is False
    assert window.active(10) is True
    assert window.active(209) is True
    assert window.active(210) is False


@pytest.mark.parametrize(
    ("changes", "field"),
    [
        ({"start_round": -1}, "start_round"),
        ({"start_round": True}, "start_round"),
        ({"start_round": 1.0}, "start_round"),
        ({"lifetime_rounds": 0}, "lifetime_rounds"),
        ({"lifetime_rounds": -1}, "lifetime_rounds"),
        ({"lifetime_rounds": True}, "lifetime_rounds"),
        ({"lifetime_rounds": 1.0}, "lifetime_rounds"),
    ],
)
def test_active_window_rejects_invalid_configuration(
    changes: dict[str, object], field: str
) -> None:
    values: dict[str, object] = {"start_round": 0, "lifetime_rounds": 1}
    values.update(changes)

    with pytest.raises(ValueError, match=field):
        _traffic().ActiveWindow(**values)


@pytest.mark.parametrize("round_id", [-1, True, 1.0])
def test_active_window_rejects_invalid_round_id(round_id: object) -> None:
    window = _traffic().ActiveWindow(start_round=0, lifetime_rounds=1)

    with pytest.raises(ValueError, match="round_id"):
        window.active(round_id)


@pytest.mark.parametrize("queue_depth", [0, 1, 1_000_000])
def test_saturated_traffic_is_always_backlogged(queue_depth: int) -> None:
    assert _traffic().SaturatedTraffic().backlogged(queue_depth) is True


@pytest.mark.parametrize("queue_depth", [-1, True, 1.0])
def test_saturated_traffic_rejects_invalid_queue_depth(
    queue_depth: object,
) -> None:
    with pytest.raises(ValueError, match="queue_depth"):
        _traffic().SaturatedTraffic().backlogged(queue_depth)


def test_poisson_traffic_has_a_fixed_seeded_call_sequence() -> None:
    source = _traffic().PoissonTraffic(
        rate_packets_per_ms=1.25, rng=random.Random(1729)
    )

    assert [source.arrivals(elapsed) for elapsed in [0, 1000, 2500, 1000, 4000]] == [
        0,
        0,
        2,
        1,
        6,
    ]


def test_poisson_traffic_seed_controls_the_entire_sequence() -> None:
    durations = [1000, 2500, 1000, 4000]

    first = _traffic().PoissonTraffic(1.25, random.Random(1729))
    repeated = _traffic().PoissonTraffic(1.25, random.Random(1729))
    different = _traffic().PoissonTraffic(1.25, random.Random(1730))

    first_sequence = [first.arrivals(elapsed) for elapsed in durations]
    assert first_sequence == [repeated.arrivals(elapsed) for elapsed in durations]
    assert first_sequence != [different.arrivals(elapsed) for elapsed in durations]


def test_zero_elapsed_poisson_interval_does_not_consume_rng() -> None:
    rng = random.Random(88)
    source = _traffic().PoissonTraffic(1.0, rng)
    before = rng.getstate()

    assert source.arrivals(0) == 0
    assert rng.getstate() == before


def test_poisson_traffic_mean_is_consistent_with_configured_rate() -> None:
    source = _traffic().PoissonTraffic(4.0, random.Random(2026))

    mean = sum(source.arrivals(1000) for _ in range(2_000)) / 2_000

    assert 3.6 <= mean <= 4.4


def test_large_mean_poisson_matches_mean_and_variance_across_seeds() -> None:
    samples = [
        _traffic().PoissonTraffic(100.0, random.Random(seed)).arrivals(1000)
        for seed in range(10_000)
    ]
    mean = sum(samples) / len(samples)
    variance = sum((sample - mean) ** 2 for sample in samples) / len(samples)

    assert 98.5 <= mean <= 101.5
    assert 95.0 <= variance <= 105.0


class _BoundedWorkRandom(random.Random):
    def __init__(self, seed: int, maximum_calls: int) -> None:
        super().__init__(seed)
        self.maximum_calls = maximum_calls
        self.random_calls = 0
        self.exponential_calls = 0

    def random(self) -> float:
        self.random_calls += 1
        if self.random_calls > self.maximum_calls:
            raise AssertionError("Poisson sampling exceeded its work bound")
        return super().random()

    def expovariate(self, lambd: float) -> float:
        self.exponential_calls += 1
        if self.exponential_calls > self.maximum_calls:
            raise AssertionError("Poisson sampling performed linear work")
        return super().expovariate(lambd)


def test_large_poisson_mean_is_deterministic_and_bounded_work() -> None:
    first_rng = _BoundedWorkRandom(99, maximum_calls=1_000)
    second_rng = _BoundedWorkRandom(99, maximum_calls=1_000)
    first = _traffic().PoissonTraffic(1_000_000.0, first_rng)
    second = _traffic().PoissonTraffic(1_000_000.0, second_rng)

    first_value = first.arrivals(1000)
    second_value = second.arrivals(1000)

    assert type(first_value) is int
    assert first_value == second_value
    for rng in (first_rng, second_rng):
        assert rng.random_calls <= rng.maximum_calls
        assert rng.exponential_calls <= rng.maximum_calls


class _NoSamplingRandom(random.Random):
    def random(self) -> float:
        raise AssertionError("RNG was consumed before rejecting the mean")


class _SequenceRandom(random.Random):
    def __init__(self, values: list[float]) -> None:
        super().__init__(0)
        self.values = iter(values)

    def random(self) -> float:
        return next(self.values)


def test_ptrs_acceptance_log_does_not_underflow_for_positive_uniform() -> None:
    rng = _SequenceRandom([0.95, math.ulp(0.0)])
    source = _traffic().PoissonTraffic(100.0, rng)

    assert type(source.arrivals(1000)) is int


@pytest.mark.parametrize(
    ("rate", "elapsed_us"),
    [
        (1_000_000_001.0, 1000),
        (sys.float_info.max, sys.maxsize),
    ],
)
def test_oversized_poisson_mean_is_rejected_before_rng_use(
    rate: float, elapsed_us: int
) -> None:
    assert _traffic().MAX_POISSON_MEAN == 1_000_000_000.0
    rng = _NoSamplingRandom(77)
    source = _traffic().PoissonTraffic(rate, rng)
    before = rng.getstate()

    with pytest.raises(ValueError, match="mean"):
        source.arrivals(elapsed_us)

    assert rng.getstate() == before


@pytest.mark.parametrize(
    "rate",
    [
        0.0,
        -1.0,
        math.nan,
        math.inf,
        -math.inf,
        True,
        math.ulp(0.0),
        10**400,
    ],
)
def test_poisson_traffic_rejects_invalid_rate(rate: object) -> None:
    with pytest.raises(ValueError, match="rate_packets_per_ms"):
        _traffic().PoissonTraffic(rate, random.Random(1))


def test_poisson_traffic_requires_explicit_random_instance() -> None:
    with pytest.raises(ValueError, match="rng"):
        _traffic().PoissonTraffic(1.0, object())


@pytest.mark.parametrize(
    ("field", "replacement"),
    [("rate_packets_per_ms", math.nan), ("rng", object())],
)
def test_poisson_traffic_configuration_cannot_be_replaced(
    field: str, replacement: object
) -> None:
    source = _traffic().PoissonTraffic(1.0, random.Random(1))

    with pytest.raises(FrozenInstanceError):
        setattr(source, field, replacement)


@pytest.mark.parametrize("elapsed_us", [-1, True, 1.0])
def test_poisson_traffic_rejects_invalid_elapsed_us(
    elapsed_us: object,
) -> None:
    source = _traffic().PoissonTraffic(1.0, random.Random(1))

    with pytest.raises(ValueError, match="elapsed_us"):
        source.arrivals(elapsed_us)


def test_periodic_busy_process_only_fires_on_period_boundaries() -> None:
    process = _nonideal().PeriodicBusyProcess(
        period_us=100, busy_us=17, start_us=25
    )

    assert process.busy_duration_at(24) == 0
    assert process.busy_duration_at(25) == 17
    assert process.busy_duration_at(26) == 0
    assert process.busy_duration_at(125) == 17
    assert process.busy_duration_at(225) == 17


def test_periodic_busy_process_reports_next_start_at_or_after_time() -> None:
    process = _nonideal().PeriodicBusyProcess(
        period_us=100, busy_us=17, start_us=25
    )

    assert process.next_start_at_or_after(0) == 25
    assert process.next_start_at_or_after(25) == 25
    assert process.next_start_at_or_after(26) == 125
    assert process.next_start_at_or_after(125) == 125


@pytest.mark.parametrize(
    ("changes", "field"),
    [
        ({"period_us": 0}, "period_us"),
        ({"period_us": -1}, "period_us"),
        ({"period_us": True}, "period_us"),
        ({"period_us": 1.0}, "period_us"),
        ({"busy_us": 0}, "busy_us"),
        ({"busy_us": -1}, "busy_us"),
        ({"busy_us": True}, "busy_us"),
        ({"busy_us": 1.0}, "busy_us"),
        ({"start_us": -1}, "start_us"),
        ({"start_us": True}, "start_us"),
        ({"start_us": 1.0}, "start_us"),
    ],
)
def test_periodic_busy_process_rejects_invalid_configuration(
    changes: dict[str, object], field: str
) -> None:
    values: dict[str, object] = {
        "period_us": 100,
        "busy_us": 10,
        "start_us": 0,
    }
    values.update(changes)

    with pytest.raises(ValueError, match=field):
        _nonideal().PeriodicBusyProcess(**values)


@pytest.mark.parametrize("now_us", [-1, True, 1.0])
def test_periodic_busy_process_rejects_invalid_query_time(
    now_us: object,
) -> None:
    process = _nonideal().PeriodicBusyProcess(period_us=100, busy_us=10)

    with pytest.raises(ValueError, match="now_us"):
        process.busy_duration_at(now_us)
    with pytest.raises(ValueError, match="now_us"):
        process.next_start_at_or_after(now_us)


def test_interruption_perturbation_applies_seeded_positive_noise() -> None:
    perturbation = _nonideal().InterruptionPerturbation(
        sigma=2.0, rng=random.Random(0)
    )

    assert perturbation.apply(10) == 12


def test_interruption_perturbation_applies_negative_noise_and_clamps() -> None:
    perturbation = _nonideal().InterruptionPerturbation(
        sigma=2.0, rng=random.Random(5)
    )

    assert perturbation.apply(1) == 0


def test_interruption_perturbation_has_a_reproducible_sequence() -> None:
    first = _nonideal().InterruptionPerturbation(2.0, random.Random(91))
    second = _nonideal().InterruptionPerturbation(2.0, random.Random(91))

    first_sequence = [first.apply(10) for _ in range(6)]

    assert first_sequence == [11, 11, 10, 9, 14, 9]
    assert first_sequence == [second.apply(10) for _ in range(6)]


def test_zero_sigma_returns_input_without_consuming_rng() -> None:
    rng = random.Random(91)
    perturbation = _nonideal().InterruptionPerturbation(0.0, rng)
    before = rng.getstate()

    assert perturbation.apply(12) == 12
    assert rng.getstate() == before


def test_non_finite_gaussian_noise_restores_rng_before_rejecting() -> None:
    rng = random.Random(2)
    perturbation = _nonideal().InterruptionPerturbation(1e308, rng)
    before = rng.getstate()

    with pytest.raises(ValueError, match="noise"):
        perturbation.apply(10)

    assert rng.getstate() == before


@pytest.mark.parametrize(
    "sigma", [-1.0, math.nan, math.inf, -math.inf, True, 10**400]
)
def test_interruption_perturbation_rejects_invalid_sigma(
    sigma: object,
) -> None:
    with pytest.raises(ValueError, match="sigma"):
        _nonideal().InterruptionPerturbation(sigma, random.Random(1))


def test_interruption_perturbation_requires_explicit_random_instance() -> None:
    with pytest.raises(ValueError, match="rng"):
        _nonideal().InterruptionPerturbation(1.0, object())


@pytest.mark.parametrize(
    ("field", "replacement"),
    [("sigma", math.nan), ("rng", object())],
)
def test_interruption_perturbation_configuration_cannot_be_replaced(
    field: str, replacement: object
) -> None:
    perturbation = _nonideal().InterruptionPerturbation(
        1.0, random.Random(1)
    )

    with pytest.raises(FrozenInstanceError):
        setattr(perturbation, field, replacement)


@pytest.mark.parametrize("interruptions", [-1, True, 1.0])
def test_interruption_perturbation_rejects_invalid_interruptions(
    interruptions: object,
) -> None:
    perturbation = _nonideal().InterruptionPerturbation(
        1.0, random.Random(1)
    )

    with pytest.raises(ValueError, match="interruptions"):
        perturbation.apply(interruptions)


def _node(
    node_id: str,
    *,
    active: bool = True,
    backlogged: bool = True,
) -> Node:
    return Node(
        node_id,
        Technology.WIFI,
        PolicyKind.RANDOM,
        selected=4,
        remaining=4,
        active=active,
        backlogged=backlogged,
    )


def test_background_busy_advances_time_and_interrupts_each_waiter_once() -> None:
    first = _node("first")
    second = _node("second")
    channel = Channel(nodes=[first, second], seed=1)

    assert channel.apply_background_busy(17) == 17

    assert channel.now_us == 17
    assert channel.contention_round == 0
    assert (first.selected, first.remaining) == (4, 4)
    assert (second.selected, second.remaining) == (4, 4)
    assert first.db_state.interruptions == 1
    assert second.db_state.interruptions == 1


def test_each_background_busy_event_adds_one_interruption() -> None:
    node = _node("waiter")
    channel = Channel(nodes=[node], seed=1)

    channel.apply_background_busy(5)
    channel.apply_background_busy(7)

    assert channel.now_us == 12
    assert node.db_state.interruptions == 2


def test_background_busy_excludes_inactive_and_non_backlogged_nodes() -> None:
    eligible = _node("eligible")
    inactive = _node("inactive", active=False)
    empty = _node("empty", backlogged=False)
    channel = Channel(nodes=[eligible, inactive, empty], seed=1)

    channel.apply_background_busy(10)

    assert eligible.db_state.interruptions == 1
    assert inactive.db_state.interruptions == 0
    assert empty.db_state.interruptions == 0


def test_background_busy_leaves_round_for_following_step() -> None:
    node = Node(
        "ready", Technology.WIFI, PolicyKind.RANDOM, selected=0, remaining=0
    )
    channel = Channel(nodes=[node], seed=9)

    channel.apply_background_busy(17)
    result = channel.step()

    assert result.round_id == 0
    assert result.now_us == 17
    assert channel.now_us == 2_017
    assert channel.contention_round == 1


def test_background_busy_does_not_consume_node_rng() -> None:
    busy_node = Node(
        "same", Technology.WIFI, PolicyKind.RANDOM, selected=0, remaining=0
    )
    control_node = Node(
        "same", Technology.WIFI, PolicyKind.RANDOM, selected=0, remaining=0
    )
    busy_channel = Channel(nodes=[busy_node], seed=31)
    control_channel = Channel(nodes=[control_node], seed=31)

    busy_channel.apply_background_busy(17)
    busy_channel.step()
    control_channel.step()

    assert busy_node.selected == control_node.selected
    assert busy_node.remaining == control_node.remaining


def test_background_busy_without_eligible_nodes_is_atomic() -> None:
    inactive = _node("inactive", active=False)
    empty = _node("empty", backlogged=False)
    channel = Channel(nodes=[inactive, empty], seed=1)

    with pytest.raises(ValueError, match="eligible"):
        channel.apply_background_busy(17)

    assert channel.now_us == 0
    assert channel.contention_round == 0
    assert inactive.db_state.interruptions == 0
    assert empty.db_state.interruptions == 0


@pytest.mark.parametrize("duration_us", [0, -1, True, 1.0])
def test_background_busy_rejects_invalid_duration_atomically(
    duration_us: object,
) -> None:
    node = _node("waiter")
    channel = Channel(nodes=[node], seed=1)

    with pytest.raises(ValueError, match="duration_us"):
        channel.apply_background_busy(duration_us)

    assert channel.now_us == 0
    assert channel.contention_round == 0
    assert node.db_state.interruptions == 0
    assert (node.selected, node.remaining) == (4, 4)


def test_background_busy_revalidates_nodes_before_any_mutation() -> None:
    node = _node("waiter")
    control_node = _node("waiter")
    channel = Channel(nodes=[node], seed=31)
    control = Channel(nodes=[control_node], seed=31)
    node.remaining = -1

    with pytest.raises(ValueError, match="remaining"):
        channel.apply_background_busy(17)

    assert channel.now_us == 0
    assert channel.contention_round == 0
    assert node.db_state.interruptions == 0

    node.remaining = 4
    channel.step()
    control.step()
    assert node.selected == control_node.selected


def test_background_busy_rejects_future_success_end_atomically() -> None:
    node = _node("waiter")
    control_node = _node("waiter")
    channel = Channel(nodes=[node], seed=31)
    control = Channel(nodes=[control_node], seed=31)
    node.last_success_end_us = channel.now_us + 1

    with pytest.raises(ValueError, match="last_success_end_us"):
        channel.apply_background_busy(17)

    assert channel.now_us == 0
    assert channel.contention_round == 0
    assert node.db_state.interruptions == 0
    assert (node.selected, node.remaining) == (4, 4)

    node.last_success_end_us = None
    channel.step()
    control.step()
    assert node.selected == control_node.selected
    assert node.remaining == control_node.remaining
