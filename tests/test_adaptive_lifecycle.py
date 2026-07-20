"""Per-node adaptive DB-LBT lifecycle integration contracts."""

import builtins
from dataclasses import FrozenInstanceError
from importlib import import_module
import math
import random

import numpy as np
import pytest

from dblbt_fcn.channel import Channel, Node
from dblbt_fcn.config import adaptive_arms
from dblbt_fcn.experiment import derive_stream_seed
from dblbt_fcn.linucb import LinUCB
from dblbt_fcn.policies import DbState
from dblbt_fcn.types import PolicyKind, RecoveryProfile, Technology


FIXED_PROFILE = RecoveryProfile(kappa=7, beta=3, m=6, b_init=15)


def _adaptive_node(
    node_id: str,
    *,
    technology: Technology = Technology.WIFI,
    selected: int = 0,
    remaining: int | None = None,
    active: bool = True,
    backlogged: bool = True,
) -> Node:
    return Node(
        node_id,
        technology,
        PolicyKind.ADAPTIVE,
        selected=selected,
        remaining=selected if remaining is None else remaining,
        active=active,
        backlogged=backlogged,
    )


def _controller(
    nodes: list[Node],
    *,
    seed: int = 410,
    feature_mask: object = None,
    online_updates: bool = True,
    collision_weight: float = 0.25,
):
    from dblbt_fcn.adaptive import AdaptiveController

    return AdaptiveController(
        Channel(nodes, seed=seed),
        LinUCB(24, 11),
        feature_mask=feature_mask,
        online_updates=online_updates,
        collision_weight=collision_weight,
    )


def test_adaptive_module_exposes_lifecycle_contract_types() -> None:
    module = import_module("dblbt_fcn.adaptive")

    assert {
        "AdaptiveController",
        "AdaptiveNodeState",
        "DecisionRecord",
        "LocalStepInput",
    } <= set(dir(module))


def test_channel_uses_sha_seeded_stable_identity_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_hash(*args: object, **kwargs: object) -> int:
        raise AssertionError("built-in hash must not seed backoff")

    first = Node("a", Technology.WIFI, PolicyKind.RANDOM, 0, 0)
    second = Node(
        "b",
        Technology.NRU,
        PolicyKind.RANDOM,
        0,
        0,
        active=False,
    )
    repeated = Node("a", Technology.WIFI, PolicyKind.RANDOM, 0, 0)

    with monkeypatch.context() as scoped:
        scoped.setattr(builtins, "hash", fail_hash)
        Channel([first, second], seed=410).step()
        Channel([repeated], seed=410).step()

    expected = random.Random(
        derive_stream_seed(410, "a", "backoff")
    ).randint(0, 15)
    assert first.selected == repeated.selected == expected
    assert derive_stream_seed(410, "a", "backoff") != derive_stream_seed(
        410, "b", "backoff"
    )


def test_channel_profile_change_preserves_active_countdown_state() -> None:
    node = _adaptive_node("w1", selected=9, remaining=4)
    channel = Channel([node], seed=410)
    profile = adaptive_arms()[-1]
    before = (
        node.selected,
        node.remaining,
        node.db_state.interruptions,
        node.db_state.retries,
        node.db_initialized,
        node.deterministic_countdown,
    )

    assert channel.recovery_profile("w1") == FIXED_PROFILE
    channel.set_recovery_profile("w1", profile)

    assert channel.recovery_profile("w1") == profile
    assert (
        node.selected,
        node.remaining,
        node.db_state.interruptions,
        node.db_state.retries,
        node.db_initialized,
        node.deterministic_countdown,
    ) == before


def test_channel_reactivation_uses_current_profile_without_cross_node_state() -> None:
    node = _adaptive_node("w1", selected=8, remaining=5)
    node.db_initialized = True
    node.db_state = DbState(interruptions=2, retries=1)
    other = _adaptive_node("n1", selected=6, remaining=3)
    channel = Channel([node, other], seed=410)
    profile = RecoveryProfile(kappa=5, beta=2, m=10, b_init=31)
    other_before = (
        other.active,
        other.backlogged,
        other.selected,
        other.remaining,
        other.db_initialized,
        other.deterministic_countdown,
        other.db_state.interruptions,
        other.db_state.retries,
    )

    channel.set_backlogged("w1", False)
    assert not node.backlogged
    assert (node.selected, node.remaining) == (8, 5)
    channel.set_recovery_profile("w1", profile)
    channel.set_backlogged("w1", True)

    assert node.backlogged
    assert node.deterministic_countdown
    assert (node.selected, node.remaining) == (13, 13)
    assert (node.db_state.interruptions, node.db_state.retries) == (0, 1)
    assert (
        other.active,
        other.backlogged,
        other.selected,
        other.remaining,
        other.db_initialized,
        other.deterministic_countdown,
        other.db_state.interruptions,
        other.db_state.retries,
    ) == other_before


def test_uninitialized_reactivation_draws_current_b_init_and_keeps_invariant() -> None:
    node = _adaptive_node("w1", selected=4)
    channel = Channel([node], seed=523)
    profile = RecoveryProfile(kappa=5, beta=2, m=4, b_init=31)

    channel.set_backlogged("w1", False)
    channel.set_recovery_profile("w1", profile)
    channel.set_backlogged("w1", True)

    expected = random.Random(
        derive_stream_seed(523, "w1", "backoff")
    ).randint(0, 31)
    assert (node.selected, node.remaining) == (expected, expected)
    assert not node.db_initialized
    assert not node.deterministic_countdown
    assert (node.db_state.interruptions, node.db_state.retries) == (0, 0)


@pytest.mark.parametrize("policy", [PolicyKind.RANDOM, PolicyKind.TMC_DB])
def test_channel_recovery_profile_rejects_nonadaptive_nodes(
    policy: PolicyKind,
) -> None:
    node = Node("w1", Technology.WIFI, policy, 0, 0)
    channel = Channel([node], seed=1)

    with pytest.raises(ValueError, match="adaptive"):
        channel.recovery_profile("w1")
    with pytest.raises(ValueError, match="adaptive"):
        channel.set_recovery_profile("w1", FIXED_PROFILE)


def test_channel_lifecycle_apis_validate_ids_values_and_profiles() -> None:
    channel = Channel([_adaptive_node("w1")], seed=1)

    with pytest.raises(ValueError, match="node_id"):
        channel.recovery_profile("missing")
    with pytest.raises(ValueError, match="RecoveryProfile"):
        channel.set_recovery_profile("w1", object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="boolean"):
        channel.set_backlogged("w1", 1)  # type: ignore[arg-type]


def test_only_actual_adaptive_senders_grow_their_local_windows() -> None:
    sender = _adaptive_node("w1", selected=0)
    waiter = _adaptive_node(
        "n1", technology=Technology.NRU, selected=10
    )
    controller = _controller([sender, waiter])

    result = controller.step()

    assert result.node_ids == ("w1",)
    assert len(controller.state("w1").window) == 1
    assert len(controller.state("n1").window) == 0
    assert controller.state("n1").local_busy_us == 2_000


def test_cold_start_waits_for_strict_global_round_32_boundary() -> None:
    node = _adaptive_node("w1")
    controller = _controller([node])

    for _ in range(8):
        controller.step()
    assert controller.state("w1").window.ready
    assert controller.channel.recovery_profile("w1") == FIXED_PROFILE
    assert controller.decisions == ()

    for _ in range(23):
        controller.step()
    before = (
        node.selected,
        node.remaining,
        node.db_state.interruptions,
        node.db_state.retries,
        node.db_initialized,
        node.deterministic_countdown,
    )
    assert controller.channel.contention_round == 31
    assert controller.decisions == ()

    controller.step()

    decision = controller.decisions[-1]
    assert decision.round_id == 32
    assert decision.previous_arm is None
    assert decision.new_arm == 0
    assert decision.reward is None
    assert decision.profile == adaptive_arms()[0]
    assert decision.profile in adaptive_arms()
    assert decision.profile.alpha == 11
    assert controller.channel.recovery_profile("w1") == adaptive_arms()[0]
    assert (
        node.selected,
        node.remaining,
        node.db_state.interruptions,
        node.db_state.retries,
        node.db_initialized,
        node.deterministic_countdown,
    ) == before


def test_second_boundary_updates_only_node_with_interval_attempts() -> None:
    w1 = _adaptive_node("w1")
    n1 = _adaptive_node("n1", technology=Technology.NRU)
    initial = LinUCB(24, 11)
    initial_A = initial.A.tobytes()
    initial_b = initial.b.tobytes()

    from dblbt_fcn.adaptive import AdaptiveController

    controller = AdaptiveController(Channel([w1, n1], seed=410), initial)
    for _ in range(32):
        w1.selected = w1.remaining = 0
        n1.selected = n1.remaining = 0
        controller.step()
    assert len(controller.decisions) == 2

    n1.selected = n1.remaining = 1_000
    for _ in range(32):
        w1.remaining = 0
        controller.step()

    w1_state = controller.state("w1")
    n1_state = controller.state("n1")
    w1_decision = [
        item for item in controller.decisions if item.node_id == "w1"
    ][-1]
    n1_decision = [
        item for item in controller.decisions if item.node_id == "n1"
    ][-1]
    assert w1_decision.round_id == n1_decision.round_id == 64
    assert w1_decision.reward_components is not None
    assert w1_decision.reward == pytest.approx(
        w1_decision.reward_components.reward
    )
    assert w1_decision.reward == pytest.approx(1.0)
    assert n1_decision.reward is None
    assert n1_decision.reward_components is None
    assert w1_state.agent.A.tobytes() != initial_A
    assert w1_state.agent.b.tobytes() != initial_b
    assert n1_state.agent.A.tobytes() == initial_A
    assert n1_state.agent.b.tobytes() == initial_b
    assert initial.A.tobytes() == initial_A
    assert initial.b.tobytes() == initial_b


def test_attempt_records_exact_local_cca_snapshot_delay_and_retry_fields() -> None:
    from dblbt_fcn.adaptive import LocalStepInput

    adaptive = _adaptive_node("w1", selected=5)
    other = Node("other", Technology.WIFI, PolicyKind.RANDOM, 0, 0)
    controller = _controller([adaptive, other])
    controller.apply_background_busy(7)
    controller.step()
    assert len(controller.state("w1").window) == 0

    adaptive.selected = adaptive.remaining = 0
    other.selected = other.remaining = 0
    controller.step({"w1": LocalStepInput(0.25, 2)})
    collision = controller.state("w1").window.attempts[-1]
    assert collision.outcome == "collision"
    assert collision.elapsed_us == 4_007
    assert collision.busy_us == 2_007
    assert collision.interruptions == 0
    assert collision.access_delay_us is None
    assert collision.queue_occupancy_ratio == 0.25
    assert collision.arrivals == 2
    assert collision.retries == 0
    assert collision.effective_data_us == 0

    other.active = False
    adaptive.db_initialized = True
    adaptive.deterministic_countdown = True
    adaptive.db_state.interruptions = 3
    adaptive.db_state.retries = 2
    adaptive.selected = adaptive.remaining = 0
    controller.step()
    first_success = controller.state("w1").window.attempts[-1]
    assert first_success.elapsed_us == 2_000
    assert first_success.busy_us == 0
    assert first_success.interruptions == 3
    assert first_success.retries == 0
    assert first_success.access_delay_us is None
    assert first_success.effective_data_us == 2_000

    controller.apply_background_busy(9)
    adaptive.remaining = 0
    controller.step()
    delayed_success = controller.state("w1").window.attempts[-1]
    assert delayed_success.elapsed_us == 2_009
    assert delayed_success.busy_us == 9
    assert delayed_success.interruptions == 1
    assert delayed_success.access_delay_us == 9


def _run_weighted_interval(
    collision_weight: float,
    *,
    feature_mask: object = None,
    online_updates: bool = True,
):
    adaptive = _adaptive_node("w1")
    other = Node("other", Technology.NRU, PolicyKind.RANDOM, 0, 0)
    controller = _controller(
        [adaptive, other],
        collision_weight=collision_weight,
        feature_mask=feature_mask,
        online_updates=online_updates,
    )
    for _ in range(32):
        adaptive.selected = adaptive.remaining = 0
        other.selected = other.remaining = 0
        controller.step()
    for _ in range(16):
        adaptive.selected = adaptive.remaining = 0
        other.selected = other.remaining = 0
        controller.step()
    other.active = False
    for _ in range(16):
        adaptive.remaining = 0
        controller.step()
    return controller


def test_feature_mask_zeroes_select_and_update_context_copies() -> None:
    controller = _run_weighted_interval(0.25, feature_mask=[0, 6])
    state = controller.state("w1")
    first, second = [
        item for item in controller.decisions if item.node_id == "w1"
    ]

    assert state.window.context()[0] > 0
    assert state.window.context()[6] == 1
    assert first.context[0] == first.context[6] == 0
    assert second.context[0] == second.context[6] == 0
    assert state.agent.b[0, 0] == state.agent.b[0, 6] == 0
    assert state.agent.A[0, 0, 0] == 1
    assert state.agent.A[0, 6, 6] == 1


def test_frozen_updates_and_collision_weight_ablation_hooks() -> None:
    light = _run_weighted_interval(0.125)
    heavy = _run_weighted_interval(0.5)
    frozen = _run_weighted_interval(0.25, online_updates=False)
    light_record = [
        item for item in light.decisions if item.node_id == "w1"
    ][-1]
    heavy_record = [
        item for item in heavy.decisions if item.node_id == "w1"
    ][-1]
    frozen_record = [
        item for item in frozen.decisions if item.node_id == "w1"
    ][-1]

    assert light_record.reward is not None
    assert heavy_record.reward is not None
    assert light_record.reward - heavy_record.reward == pytest.approx(
        (0.5 - 0.125) * 0.5
    )
    assert frozen_record.reward is not None
    np.testing.assert_array_equal(
        frozen.state("w1").agent.A, LinUCB(24, 11).A
    )
    np.testing.assert_array_equal(
        frozen.state("w1").agent.b, LinUCB(24, 11).b
    )


def test_interval_with_no_sensed_or_successful_busy_defines_zero_share() -> None:
    adaptive = _adaptive_node("w1")
    other = Node("other", Technology.NRU, PolicyKind.RANDOM, 0, 0)
    controller = _controller([adaptive, other])
    for _ in range(64):
        adaptive.selected = adaptive.remaining = 0
        other.selected = other.remaining = 0
        controller.step()

    record = [
        item for item in controller.decisions if item.node_id == "w1"
    ][-1]
    assert record.reward_components is not None
    assert record.reward_components.share_utility == 0
    assert record.reward == pytest.approx(1 / 3 - 0.25)


def test_records_and_local_inputs_are_immutable_and_validated() -> None:
    from dblbt_fcn.adaptive import DecisionRecord, LocalStepInput

    local = LocalStepInput()
    assert (local.queue_occupancy_ratio, local.arrivals) == (1.0, 0)
    with pytest.raises(FrozenInstanceError):
        local.arrivals = 1
    with pytest.raises(ValueError, match="queue_occupancy_ratio"):
        LocalStepInput(1.1, 0)
    with pytest.raises(ValueError, match="arrivals"):
        LocalStepInput(1, True)
    assert DecisionRecord.__dataclass_params__.frozen


@pytest.mark.parametrize(
    ("changes", "field"),
    [
        ({"agent": LinUCB(23, 11)}, "num_arms"),
        ({"agent": LinUCB(24, 10)}, "context_dim"),
        ({"feature_mask": [11]}, "feature_mask"),
        ({"feature_mask": [True] * 10}, "feature_mask"),
        ({"online_updates": 1}, "online_updates"),
        ({"collision_weight": -0.1}, "collision_weight"),
        ({"collision_weight": math.inf}, "collision_weight"),
        ({"collision_weight": 10**10_000}, "collision_weight"),
    ],
)
def test_controller_rejects_invalid_construction(
    changes: dict[str, object], field: str
) -> None:
    from dblbt_fcn.adaptive import AdaptiveController

    values: dict[str, object] = {
        "channel": Channel([_adaptive_node("w1")], seed=1),
        "agent": LinUCB(24, 11),
    }
    values.update(changes)

    with pytest.raises(ValueError, match=field):
        AdaptiveController(**values)  # type: ignore[arg-type]


def test_controller_validates_runtime_node_and_local_inputs() -> None:
    controller = _controller([_adaptive_node("w1")])

    with pytest.raises(ValueError, match="node_id"):
        controller.state("missing")
    with pytest.raises(ValueError, match="node_id"):
        controller.set_backlogged("missing", False)
    with pytest.raises(ValueError, match="node_id"):
        controller.set_active("missing", False)
    with pytest.raises(ValueError, match="boolean"):
        controller.set_active("w1", 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="LocalStepInput"):
        controller.step({"w1": object()})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="adaptive"):
        controller.step(
            {"other": import_module("dblbt_fcn.adaptive").LocalStepInput()}
        )


def test_local_input_normalizes_numeric_overflow_to_field_error() -> None:
    from dblbt_fcn.adaptive import LocalStepInput

    with pytest.raises(ValueError, match="queue_occupancy_ratio"):
        LocalStepInput(10**10_000, 0)


def test_initially_nonbacklogged_attempt_starts_measurement_at_reactivation() -> None:
    adaptive = _adaptive_node("w1", backlogged=False)
    driver = Node("driver", Technology.NRU, PolicyKind.RANDOM, 0, 0)
    controller = _controller([adaptive, driver])

    for _ in range(3):
        driver.remaining = 0
        controller.step()
    state = controller.state("w1")
    assert len(state.window) == 0
    assert state.local_busy_us == 0
    decisions_before = controller.decisions

    controller.set_backlogged("w1", True)
    driver.active = False
    adaptive.remaining = 0
    controller.step()

    state = controller.state("w1")
    attempt = state.window.attempts[-1]
    assert len(state.window) == 1
    assert attempt.elapsed_us == controller.channel.tx_us
    assert attempt.busy_us == 0
    assert controller.decisions == decisions_before


def test_queue_pause_clears_pending_busy_and_preserves_learning_state() -> None:
    adaptive = _adaptive_node("w1")
    driver = Node(
        "driver", Technology.NRU, PolicyKind.RANDOM, 0, 0, active=False
    )
    controller = _controller([adaptive, driver])
    for _ in range(32):
        adaptive.remaining = 0
        controller.step()

    driver.active = True
    adaptive.selected = adaptive.remaining = 10
    driver.selected = driver.remaining = 0
    controller.step()
    state = controller.state("w1")
    assert state.local_busy_us == controller.channel.tx_us
    interval_other_busy = state.interval.other_busy_us
    window_before = state.window.attempts
    arm_before = state.current_arm
    A_before = state.agent.A.tobytes()
    b_before = state.agent.b.tobytes()
    decisions_before = controller.decisions

    controller.set_backlogged("w1", False)
    assert controller.state("w1").local_busy_us == 0
    for _ in range(3):
        driver.remaining = 0
        controller.step()
    controller.set_backlogged("w1", True)

    state = controller.state("w1")
    assert state.window.attempts == window_before
    assert state.current_arm == arm_before
    assert state.agent.A.tobytes() == A_before
    assert state.agent.b.tobytes() == b_before
    assert state.interval.other_busy_us == interval_other_busy
    assert controller.decisions == decisions_before

    driver.active = False
    adaptive.remaining = 0
    controller.step()
    state = controller.state("w1")
    attempt = state.window.attempts[-1]
    assert len(state.window) == len(window_before) + 1
    assert attempt.elapsed_us == controller.channel.tx_us
    assert attempt.busy_us == 0


def test_inactive_gap_is_paused_out_of_interval_airtime_reward() -> None:
    adaptive = _adaptive_node("w1")
    driver = Node("driver", Technology.NRU, PolicyKind.RANDOM, 0, 0)
    controller = _controller([adaptive, driver])
    for _ in range(32):
        adaptive.selected = adaptive.remaining = 0
        driver.selected = driver.remaining = 0
        controller.step()

    driver.active = False
    adaptive.remaining = 0
    controller.step()
    assert controller.state("w1").interval.attempts == 1

    controller.set_active("w1", False)
    driver.active = True
    for _ in range(30):
        driver.remaining = 0
        controller.step()
    controller.set_active("w1", True)
    driver.remaining = 0
    controller.step()

    decisions = [
        record for record in controller.decisions if record.node_id == "w1"
    ]
    assert len(decisions) == 2
    assert decisions[-1].round_id == 64
    assert decisions[-1].reward_components is not None
    assert decisions[-1].reward_components.airtime_utility == pytest.approx(
        0.5
    )
    assert decisions[-1].reward == pytest.approx(2 / 3)


def test_eligibility_pauses_once_across_active_and_backlogged_dimensions() -> None:
    adaptive = _adaptive_node("w1", active=False)
    driver = Node("driver", Technology.NRU, PolicyKind.RANDOM, 0, 0)
    controller = _controller([adaptive, driver])
    state = controller.state("w1")
    initial_A = state.agent.A.tobytes()
    initial_b = state.agent.b.tobytes()

    controller.set_backlogged("w1", False)
    driver.remaining = 0
    controller.step()
    controller.set_active("w1", True)
    driver.remaining = 0
    controller.step()
    assert controller.state("w1").inactive_since_us == 0

    controller.set_backlogged("w1", True)
    state = controller.state("w1")
    assert state.inactive_since_us is None
    assert state.last_attempt_end_us == controller.channel.now_us
    assert state.interval.start_us == controller.channel.now_us

    controller.set_backlogged("w1", False)
    paused_at = controller.channel.now_us
    controller.set_active("w1", False)
    driver.remaining = 0
    controller.step()
    controller.set_backlogged("w1", True)
    assert controller.state("w1").inactive_since_us == paused_at
    controller.set_active("w1", True)

    state = controller.state("w1")
    assert state.inactive_since_us is None
    assert state.last_attempt_end_us == controller.channel.now_us
    assert state.interval.start_us == controller.channel.now_us
    assert len(state.window) == 0
    assert state.current_arm is None
    assert state.agent.A.tobytes() == initial_A
    assert state.agent.b.tobytes() == initial_b
    assert controller.decisions == ()


def test_public_state_is_frozen_and_detached_from_controller_learning() -> None:
    adaptive = _adaptive_node("w1")
    controller = _controller([adaptive])
    for _ in range(32):
        adaptive.remaining = 0
        controller.step()

    snapshot = controller.state("w1")
    window_before = snapshot.window.attempts
    agent_A_before = snapshot.agent.A.tobytes()
    agent_b_before = snapshot.agent.b.tobytes()
    with pytest.raises(FrozenInstanceError):
        snapshot.current_arm = 23
    snapshot.window.record(snapshot.window.attempts[-1])
    snapshot.agent.update(23, np.ones(11), 100)
    assert snapshot.decision_context is not None
    with pytest.raises(ValueError):
        snapshot.decision_context[0] = 1

    actual = controller.state("w1")
    assert actual.current_arm == 0
    assert actual.window.attempts == window_before
    assert actual.agent.A.tobytes() == agent_A_before
    assert actual.agent.b.tobytes() == agent_b_before
    assert controller.channel.recovery_profile("w1") == adaptive_arms()[0]

    for _ in range(32):
        adaptive.remaining = 0
        controller.step()
    assert controller.decisions[-1].previous_arm == 0


def _run_sparse_pause(*, direct_channel: bool) -> float:
    adaptive = _adaptive_node("w1")
    driver = Node("driver", Technology.NRU, PolicyKind.RANDOM, 0, 0)
    controller = _controller([adaptive, driver])
    for _ in range(32):
        adaptive.selected = adaptive.remaining = 0
        driver.selected = driver.remaining = 0
        controller.step()

    driver.active = False
    adaptive.remaining = 0
    controller.step()
    if direct_channel:
        controller.channel.set_backlogged("w1", False)
    else:
        controller.set_backlogged("w1", False)
    driver.active = True
    for _ in range(30):
        driver.remaining = 0
        controller.step()
    if direct_channel:
        controller.channel.set_backlogged("w1", True)
    else:
        controller.set_backlogged("w1", True)
    driver.active = False
    adaptive.remaining = 0
    controller.step()

    components = controller.decisions[-1].reward_components
    assert components is not None
    return components.airtime_utility


def test_direct_channel_queue_pause_is_synchronized_before_measurement() -> None:
    controlled = _run_sparse_pause(direct_channel=False)
    bypass = _run_sparse_pause(direct_channel=True)

    assert controlled == bypass == pytest.approx(1.0)


def test_external_recovery_profile_change_fails_closed_before_step() -> None:
    adaptive = _adaptive_node("w1")
    controller = _controller([adaptive])
    for _ in range(32):
        adaptive.remaining = 0
        controller.step()
    before = controller.state("w1")
    controller.channel.set_recovery_profile("w1", adaptive_arms()[-1])

    with pytest.raises(RuntimeError, match="recovery profile"):
        controller.step()

    after = controller.state("w1")
    assert controller.channel.contention_round == 32
    assert after.current_arm == before.current_arm == 0
    assert after.agent.A.tobytes() == before.agent.A.tobytes()
    assert after.agent.b.tobytes() == before.agent.b.tobytes()


def test_active_empty_queue_still_settles_and_selects_at_boundaries() -> None:
    from dblbt_fcn.adaptive import LocalStepInput

    adaptive = _adaptive_node("w1")
    driver = Node("driver", Technology.NRU, PolicyKind.RANDOM, 0, 0)
    controller = _controller([adaptive, driver])
    for _ in range(32):
        adaptive.selected = adaptive.remaining = 0
        driver.selected = driver.remaining = 0
        controller.step()

    driver.active = False
    adaptive.remaining = 0
    controller.step({"w1": LocalStepInput(0.0, 0)})
    controller.set_backlogged("w1", False)
    driver.active = True
    for _ in range(31):
        driver.remaining = 0
        controller.step()

    decisions = [
        record for record in controller.decisions if record.node_id == "w1"
    ]
    assert [record.round_id for record in decisions] == [32, 64]
    assert decisions[-1].previous_arm == decisions[0].new_arm
    assert decisions[-1].reward == pytest.approx(1.0)
    assert controller.channel.recovery_profile("w1") == decisions[-1].profile
    state = controller.state("w1")
    assert state.interval.attempts == 0
    assert state.inactive_since_us == controller.channel.now_us
    A_after_reward = state.agent.A.tobytes()
    b_after_reward = state.agent.b.tobytes()

    for _ in range(32):
        driver.remaining = 0
        controller.step()
    decisions = [
        record for record in controller.decisions if record.node_id == "w1"
    ]
    assert [record.round_id for record in decisions] == [32, 64, 96]
    assert decisions[-1].reward is None
    state = controller.state("w1")
    assert state.agent.A.tobytes() == A_after_reward
    assert state.agent.b.tobytes() == b_after_reward
    assert state.interval.attempts == 0
    assert state.inactive_since_us == controller.channel.now_us


def test_waiter_accumulates_arrivals_and_latest_occupancy_until_attempt() -> None:
    from dblbt_fcn.adaptive import LocalStepInput

    waiter = _adaptive_node("w1", selected=10)
    sender = _adaptive_node("n1", technology=Technology.NRU)
    controller = _controller([waiter, sender])
    for arrivals, occupancy in ((2, 0.2), (3, 0.4), (5, 0.6)):
        sender.remaining = 0
        controller.step(
            {
                "w1": LocalStepInput(occupancy, arrivals),
                "n1": LocalStepInput(0.9, 1),
            }
        )

    controller.set_active("n1", False)
    waiter.remaining = 0
    controller.step({"w1": LocalStepInput(0.8, 7)})

    w1 = controller.state("w1")
    n1 = controller.state("n1")
    assert w1.window.attempts[-1].arrivals == 17
    assert w1.window.attempts[-1].queue_occupancy_ratio == 0.8
    assert w1.pending_arrivals == 0
    assert w1.latest_queue_occupancy == 0.8
    assert [attempt.arrivals for attempt in n1.window.attempts] == [1, 1, 1]


def test_pending_arrivals_survive_ineligible_gap_until_next_attempt() -> None:
    from dblbt_fcn.adaptive import LocalStepInput

    adaptive = _adaptive_node("w1", selected=10)
    driver = Node("driver", Technology.NRU, PolicyKind.RANDOM, 0, 0)
    controller = _controller([adaptive, driver])
    driver.remaining = 0
    controller.step({"w1": LocalStepInput(0.25, 4)})
    controller.set_backlogged("w1", False)
    for _ in range(3):
        driver.remaining = 0
        controller.step()
    controller.set_backlogged("w1", True)
    driver.active = False
    adaptive.remaining = 0
    controller.step({"w1": LocalStepInput(0.75, 6)})

    state = controller.state("w1")
    assert state.window.attempts[-1].arrivals == 10
    assert state.window.attempts[-1].queue_occupancy_ratio == 0.75
    assert state.pending_arrivals == 0


def test_arrival_overflow_is_atomic_across_local_nodes() -> None:
    from dblbt_fcn.adaptive import LocalStepInput

    w1 = _adaptive_node("w1", selected=10)
    n1 = _adaptive_node("n1", selected=10)
    driver = Node("driver", Technology.NRU, PolicyKind.RANDOM, 0, 0)
    controller = _controller([w1, n1, driver])
    driver.remaining = 0
    controller.step(
        {
            "w1": LocalStepInput(0.2, 3),
            "n1": LocalStepInput(0.4, 4),
        }
    )
    before_w1 = controller.state("w1")
    before_n1 = controller.state("n1")
    round_before = controller.channel.contention_round

    with pytest.raises(ValueError, match="arrivals"):
        controller.step(
            {
                "w1": LocalStepInput(0.7, 2),
                "n1": LocalStepInput(0.9, 10**10_000),
            }
        )

    after_w1 = controller.state("w1")
    after_n1 = controller.state("n1")
    assert controller.channel.contention_round == round_before
    assert after_w1.pending_arrivals == before_w1.pending_arrivals == 3
    assert after_n1.pending_arrivals == before_n1.pending_arrivals == 4
    assert after_w1.latest_queue_occupancy == 0.2
    assert after_n1.latest_queue_occupancy == 0.4


def test_context_free_ucb_visits_untried_arms_then_balances_ucb() -> None:
    from dblbt_fcn.adaptive import ContextFreeUCB

    agent = ContextFreeUCB(24, exploration=1.0)
    selected = []
    for arm in range(24):
        selected.append(agent.select())
        agent.update(arm, 1.0 if arm == 0 else 0.0)

    assert selected == list(range(24))
    assert agent.select() == 0
    agent.update(0, 0.0)
    assert agent.select() == 1
    np.testing.assert_array_equal(
        agent.counts, np.array([2] + [1] * 23, dtype=np.int64)
    )


def test_context_free_ucb_uses_standard_two_log_confidence_factor() -> None:
    from dblbt_fcn.adaptive import ContextFreeUCB

    agent = ContextFreeUCB(2, exploration=1.0)
    agent.counts[:] = [10, 1]
    agent.reward_sums[:] = [10.0, -0.2]

    assert agent.select() == 1


def test_context_free_ucb_clone_and_validation_are_independent() -> None:
    from dblbt_fcn.adaptive import ContextFreeUCB

    agent = ContextFreeUCB(2, exploration=0.5)
    agent.update(0, 0.75)
    cloned = agent.clone()
    cloned.update(1, -0.25)

    np.testing.assert_array_equal(agent.counts, [1, 0])
    np.testing.assert_array_equal(cloned.counts, [1, 1])
    assert not np.shares_memory(agent.counts, cloned.counts)
    assert not np.shares_memory(agent.reward_sums, cloned.reward_sums)
    with pytest.raises(ValueError, match="num_arms"):
        ContextFreeUCB(True)
    with pytest.raises(ValueError, match="exploration"):
        ContextFreeUCB(2, exploration=math.inf)
    with pytest.raises(ValueError, match="arm"):
        agent.update(True, 0.0)
    with pytest.raises(ValueError, match="reward"):
        agent.update(0, math.nan)


def test_context_free_controller_updates_only_attempting_node_counts() -> None:
    from dblbt_fcn.adaptive import AdaptiveController, ContextFreeUCB

    w1 = _adaptive_node("w1")
    n1 = _adaptive_node("n1", technology=Technology.NRU)
    initial = ContextFreeUCB(24)
    controller = AdaptiveController(Channel([w1, n1], seed=410), initial)
    for _ in range(32):
        w1.selected = w1.remaining = 0
        n1.selected = n1.remaining = 0
        controller.step()
    n1.selected = n1.remaining = 1_000
    for _ in range(32):
        w1.remaining = 0
        controller.step()

    w1_state = controller.state("w1")
    n1_state = controller.state("n1")
    np.testing.assert_array_equal(
        w1_state.agent.counts, np.array([1] + [0] * 23)
    )
    np.testing.assert_array_equal(n1_state.agent.counts, np.zeros(24))
    np.testing.assert_array_equal(initial.counts, np.zeros(24))
    assert controller.decisions[-2].new_arm == 1
    assert controller.decisions[-1].new_arm == 0
    assert len(controller.decisions[-1].context) == 11


def test_context_free_controller_cycles_first_twenty_four_decisions() -> None:
    from dblbt_fcn.adaptive import AdaptiveController, ContextFreeUCB

    node = _adaptive_node("w1")
    controller = AdaptiveController(
        Channel([node], seed=410), ContextFreeUCB(24)
    )
    for _ in range(24 * 32):
        node.remaining = 0
        controller.step()

    assert [record.new_arm for record in controller.decisions] == list(
        range(24)
    )
    state = controller.state("w1")
    np.testing.assert_array_equal(
        state.agent.counts, np.array([1] * 23 + [0])
    )


def test_channel_step_failure_does_not_commit_staged_local_inputs() -> None:
    from dblbt_fcn.adaptive import LocalStepInput

    node = _adaptive_node("w1", active=False)
    controller = _controller([node])
    measurement = LocalStepInput(0.25, 5)

    with pytest.raises(ValueError, match="eligible"):
        controller.step({"w1": measurement})

    failed = controller.state("w1")
    assert controller.channel.contention_round == 0
    assert failed.pending_arrivals == 0
    assert failed.latest_queue_occupancy == 1.0

    controller.set_active("w1", True)
    node.remaining = 0
    controller.step({"w1": measurement})

    attempt = controller.state("w1").window.attempts[-1]
    assert attempt.arrivals == 5
    assert attempt.queue_occupancy_ratio == 0.25


@pytest.mark.parametrize("selector_kind", ["linucb", "context_free"])
def test_boundary_profile_failure_discards_candidate_update(
    selector_kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dblbt_fcn.adaptive import AdaptiveController, ContextFreeUCB

    node = _adaptive_node("w1")
    selector = (
        LinUCB(24, 11)
        if selector_kind == "linucb"
        else ContextFreeUCB(24)
    )
    controller = AdaptiveController(Channel([node], seed=410), selector)
    for _ in range(32):
        node.remaining = 0
        controller.step()
    for _ in range(31):
        node.remaining = 0
        controller.step()

    before = controller.state("w1")
    decisions_before = controller.decisions
    original_setter = controller.channel.set_recovery_profile
    observed: dict[str, object] = {}

    def fail_profile(node_id: str, profile: RecoveryProfile) -> None:
        observed["interval"] = controller.state(node_id).interval
        raise RuntimeError("profile write failed")

    monkeypatch.setattr(
        controller.channel, "set_recovery_profile", fail_profile
    )
    node.remaining = 0
    with pytest.raises(RuntimeError, match="profile write failed"):
        controller.step()

    after_failure = controller.state("w1")
    assert controller.channel.contention_round == 64
    assert after_failure.current_arm == before.current_arm == 0
    assert after_failure.previous_arm == before.previous_arm
    np.testing.assert_array_equal(
        after_failure.decision_context, before.decision_context
    )
    assert after_failure.interval == observed["interval"]
    assert controller.decisions == decisions_before
    if selector_kind == "linucb":
        np.testing.assert_array_equal(after_failure.agent.A, before.agent.A)
        np.testing.assert_array_equal(after_failure.agent.b, before.agent.b)
    else:
        np.testing.assert_array_equal(
            after_failure.agent.counts, before.agent.counts
        )
        np.testing.assert_array_equal(
            after_failure.agent.reward_sums, before.agent.reward_sums
        )

    monkeypatch.setattr(
        controller.channel, "set_recovery_profile", original_setter
    )
    for _ in range(32):
        node.remaining = 0
        controller.step()

    final = controller.state("w1")
    assert [record.round_id for record in controller.decisions] == [32, 96]
    assert controller.decisions[-1].previous_arm == 0
    if selector_kind == "linucb":
        expected_A = before.agent.A.copy()
        expected_A[0] += np.outer(
            before.decision_context, before.decision_context
        )
        np.testing.assert_array_equal(final.agent.A, expected_A)
    else:
        np.testing.assert_array_equal(
            final.agent.counts, np.array([1] + [0] * 23)
        )


@pytest.mark.parametrize("selector_kind", ["linucb", "context_free"])
def test_boundary_clone_failure_never_calls_profile_setter(
    selector_kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dblbt_fcn.adaptive import AdaptiveController, ContextFreeUCB

    node = _adaptive_node("w1")
    selector = (
        LinUCB(24, 11)
        if selector_kind == "linucb"
        else ContextFreeUCB(24)
    )
    controller = AdaptiveController(Channel([node], seed=410), selector)
    for _ in range(63):
        node.remaining = 0
        controller.step()

    original_setter = controller.channel.set_recovery_profile
    setter_calls: list[tuple[str, RecoveryProfile]] = []

    def record_profile(node_id: str, profile: RecoveryProfile) -> None:
        setter_calls.append((node_id, profile))
        original_setter(node_id, profile)

    def fail_clone(agent: object) -> object:
        raise RuntimeError("clone failure")

    monkeypatch.setattr(
        controller.channel, "set_recovery_profile", record_profile
    )
    monkeypatch.setattr(type(selector), "clone", fail_clone)
    node.remaining = 0
    with pytest.raises(RuntimeError, match="clone failure"):
        controller.step()

    assert controller.channel.contention_round == 64
    assert setter_calls == []
