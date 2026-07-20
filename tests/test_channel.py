"""Deterministic contention-round behavior."""

from dataclasses import FrozenInstanceError
import random

import pytest

from dblbt_fcn.channel import Channel, Node, RoundResult
from dblbt_fcn.experiment import derive_stream_seed
from dblbt_fcn.policies import DbState
from dblbt_fcn.types import PolicyKind, Technology


def test_single_zero_is_success() -> None:
    nodes = [
        Node(
            "w1",
            Technology.WIFI,
            PolicyKind.RANDOM,
            selected=0,
            remaining=0,
        ),
        Node(
            "n1",
            Technology.NRU,
            PolicyKind.RANDOM,
            selected=3,
            remaining=3,
        ),
    ]

    result = Channel(nodes=nodes, seed=1).step()

    assert result.kind == "success"
    assert result.node_ids == ("w1",)


def test_equal_zeros_collide() -> None:
    nodes = [
        Node(
            "w1",
            Technology.WIFI,
            PolicyKind.RANDOM,
            selected=0,
            remaining=0,
        ),
        Node(
            "n1",
            Technology.NRU,
            PolicyKind.RANDOM,
            selected=0,
            remaining=0,
        ),
    ]

    result = Channel(nodes=nodes, seed=1).step()

    assert result.kind == "collision"
    assert result.node_ids == ("w1", "n1")


def test_minimum_countdown_advances_time_and_decrements_eligible_nodes() -> None:
    sender = Node(
        "w1", Technology.WIFI, PolicyKind.RANDOM, selected=2, remaining=2
    )
    waiter = Node(
        "n1", Technology.NRU, PolicyKind.RANDOM, selected=5, remaining=5
    )
    channel = Channel(nodes=[sender, waiter], seed=1, slot_us=3)

    result = channel.step()

    assert result.now_us == 6
    assert channel.now_us == 2_006
    assert waiter.remaining == 3


def test_waiter_records_one_interruption_for_entire_busy_period() -> None:
    sender = Node(
        "w1", Technology.WIFI, PolicyKind.RANDOM, selected=0, remaining=0
    )
    waiter = Node(
        "n1",
        Technology.NRU,
        PolicyKind.TMC_DB,
        selected=2_005,
        remaining=2_005,
        db_initialized=True,
        deterministic_countdown=True,
    )

    Channel(nodes=[sender, waiter], seed=1).step()

    assert waiter.remaining == 2_005
    assert waiter.db_state.interruptions == 1


def test_inactive_and_non_backlogged_nodes_are_excluded() -> None:
    inactive = Node(
        "off",
        Technology.WIFI,
        PolicyKind.RANDOM,
        selected=0,
        remaining=0,
        active=False,
    )
    empty = Node(
        "empty",
        Technology.NRU,
        PolicyKind.RANDOM,
        selected=0,
        remaining=0,
        backlogged=False,
    )
    eligible = Node(
        "ready",
        Technology.WIFI,
        PolicyKind.RANDOM,
        selected=4,
        remaining=4,
    )

    result = Channel(nodes=[inactive, empty, eligible], seed=1).step()

    assert result.node_ids == ("ready",)
    assert result.now_us == 4
    assert inactive.remaining == 0
    assert empty.remaining == 0
    assert inactive.db_state.interruptions == 0
    assert empty.db_state.interruptions == 0


def test_step_rejects_channel_without_eligible_nodes() -> None:
    node = Node(
        "off",
        Technology.WIFI,
        PolicyKind.RANDOM,
        selected=0,
        remaining=0,
        active=False,
    )

    with pytest.raises(ValueError, match="eligible"):
        Channel(nodes=[node], seed=1).step()


def test_random_collision_grows_window_before_rearming() -> None:
    first = Node(
        "w1", Technology.WIFI, PolicyKind.RANDOM, selected=0, remaining=0
    )
    second = Node(
        "w2", Technology.WIFI, PolicyKind.RANDOM, selected=0, remaining=0
    )

    channel = Channel(nodes=[first, second], seed=3)
    channel.step()

    assert 15 < first.selected <= 31
    assert first.remaining == first.selected
    assert 15 < second.selected <= 31
    assert second.remaining == second.selected


def test_primary_db_collision_rearms_with_recovery_rule() -> None:
    nodes = [
        Node(
            "a",
            Technology.WIFI,
            PolicyKind.PRIMARY_DB,
            selected=0,
            remaining=0,
        ),
        Node(
            "b",
            Technology.NRU,
            PolicyKind.PRIMARY_DB,
            selected=0,
            remaining=0,
        ),
    ]

    Channel(nodes=nodes, seed=1).step()

    for node in nodes:
        assert node.db_state.retries == 1
        assert node.selected == 11
        assert node.remaining == 11
        assert node.deterministic_countdown


@pytest.mark.parametrize(
    "policy_kind",
    [PolicyKind.TMC_DB, PolicyKind.ADAPTIVE],
)
def test_tmc_initial_collision_retries_random_access(
    policy_kind: PolicyKind,
) -> None:
    nodes = [
        Node("a", Technology.WIFI, policy_kind, selected=0, remaining=0),
        Node("b", Technology.NRU, policy_kind, selected=0, remaining=0),
    ]

    Channel(nodes=nodes, seed=1).step()

    for node in nodes:
        assert node.db_state.retries == 0
        assert 0 <= node.selected <= 15
        assert node.remaining == node.selected
        assert not node.db_initialized
        assert not node.deterministic_countdown


def test_primary_db_success_resets_to_deterministic_recovery() -> None:
    node = Node(
        "only",
        Technology.WIFI,
        PolicyKind.PRIMARY_DB,
        selected=0,
        remaining=0,
        db_state=DbState(interruptions=3, retries=7),
    )

    Channel(nodes=[node], seed=1).step()

    assert node.db_state.retries == 0
    assert node.db_state.interruptions == 0
    assert node.selected == 14
    assert node.remaining == 14
    assert node.deterministic_countdown


def test_primary_random_countdown_still_records_interruption() -> None:
    sender = Node("sender", Technology.WIFI, PolicyKind.RANDOM, 0, 0)
    waiting = Node(
        "waiting",
        Technology.NRU,
        PolicyKind.PRIMARY_DB,
        selected=5,
        remaining=5,
        db_state=DbState(interruptions=4, retries=3),
        deterministic_countdown=False,
    )

    Channel(nodes=[sender, waiting], seed=1).step()

    assert waiting.db_state.interruptions == 5
    assert (waiting.selected, waiting.remaining) == (5, 5)
    assert not waiting.deterministic_countdown


@pytest.mark.parametrize(
    "policy_kind", [PolicyKind.TMC_DB, PolicyKind.ADAPTIVE]
)
def test_tmc_success_enters_deterministic_saturated_schedule(
    policy_kind: PolicyKind,
) -> None:
    node = Node(
        "only",
        Technology.WIFI,
        policy_kind,
        selected=0,
        remaining=0,
    )

    Channel(nodes=[node], seed=1).step()

    assert node.db_state.retries == 0
    assert node.db_initialized
    assert node.deterministic_countdown
    assert node.selected == 11
    assert node.remaining == 11


@pytest.mark.parametrize(
    "policy_kind", [PolicyKind.TMC_DB, PolicyKind.ADAPTIVE]
)
def test_tmc_nodes_start_in_initial_random_mode(
    policy_kind: PolicyKind,
) -> None:
    node = Node(
        "initial", Technology.WIFI, policy_kind, selected=4, remaining=4
    )

    assert not node.db_initialized
    assert not node.deterministic_countdown


@pytest.mark.parametrize(
    "policy_kind", [PolicyKind.TMC_DB, PolicyKind.ADAPTIVE]
)
@pytest.mark.parametrize(
    ("field", "changes"),
    [
        ("deterministic_countdown", {"deterministic_countdown": True}),
        ("retries", {"db_state": DbState(retries=1)}),
        ("interruptions", {"db_state": DbState(interruptions=1)}),
    ],
)
def test_uninitialized_tmc_rejects_non_initial_lifecycle_state(
    policy_kind: PolicyKind,
    field: str,
    changes: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match=field):
        Node(
            "invalid",
            Technology.WIFI,
            policy_kind,
            selected=4,
            remaining=4,
            **changes,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "policy_kind", [PolicyKind.TMC_DB, PolicyKind.ADAPTIVE]
)
def test_tmc_deterministic_success_uses_only_mode_interruptions(
    policy_kind: PolicyKind,
) -> None:
    node = Node(
        "only", Technology.WIFI, policy_kind, selected=0, remaining=0
    )
    channel = Channel(nodes=[node], seed=1)

    channel.step()
    assert node.db_initialized
    assert node.deterministic_countdown
    assert node.selected == 11

    for duration_us in (3, 5, 7):
        channel.apply_background_busy(duration_us)

    assert node.db_state.interruptions == 3
    assert (node.selected, node.remaining) == (11, 11)

    node.remaining = 0
    channel.step()

    assert node.selected == 14
    assert node.remaining == 14
    assert node.db_state.interruptions == 0
    assert node.deterministic_countdown


@pytest.mark.parametrize(
    "policy_kind", [PolicyKind.TMC_DB, PolicyKind.ADAPTIVE]
)
def test_tmc_initial_random_wait_does_not_increment_interruptions(
    policy_kind: PolicyKind,
) -> None:
    sender = Node("sender", Technology.WIFI, PolicyKind.RANDOM, 0, 0)
    waiting = Node(
        "waiting", Technology.NRU, policy_kind, selected=5, remaining=5
    )

    Channel(nodes=[sender, waiting], seed=1).step()

    assert waiting.db_state.interruptions == 0
    assert (waiting.selected, waiting.remaining) == (5, 5)
    assert not waiting.db_initialized
    assert not waiting.deterministic_countdown


@pytest.mark.parametrize(
    "policy_kind", [PolicyKind.TMC_DB, PolicyKind.ADAPTIVE]
)
def test_tmc_intermediate_random_wait_preserves_interruptions(
    policy_kind: PolicyKind,
) -> None:
    sender = Node("sender", Technology.WIFI, PolicyKind.RANDOM, 0, 0)
    waiting = Node(
        "waiting",
        Technology.NRU,
        policy_kind,
        selected=5,
        remaining=5,
        db_state=DbState(interruptions=4, retries=3),
        db_initialized=True,
        deterministic_countdown=False,
    )

    Channel(nodes=[sender, waiting], seed=1).step()

    assert waiting.db_state.interruptions == 4
    assert (waiting.selected, waiting.remaining) == (5, 5)
    assert not waiting.deterministic_countdown


@pytest.mark.parametrize(
    "policy_kind", [PolicyKind.TMC_DB, PolicyKind.ADAPTIVE]
)
def test_initialized_tmc_collision_enters_deterministic_mode(
    policy_kind: PolicyKind,
) -> None:
    recovering = Node(
        "recovering",
        Technology.NRU,
        policy_kind,
        selected=0,
        remaining=0,
        db_state=DbState(interruptions=2, retries=0),
        db_initialized=True,
        deterministic_countdown=False,
    )
    other = Node("other", Technology.WIFI, PolicyKind.RANDOM, 0, 0)

    Channel(nodes=[recovering, other], seed=1).step()

    assert recovering.db_state.retries == 1
    assert recovering.db_state.interruptions == 0
    assert recovering.selected == 13
    assert recovering.remaining == 13
    assert recovering.deterministic_countdown


@pytest.mark.parametrize(
    "policy_kind", [PolicyKind.TMC_DB, PolicyKind.ADAPTIVE]
)
def test_initialized_tmc_collision_random_mode_preserves_interruptions(
    policy_kind: PolicyKind,
) -> None:
    recovering = Node(
        "recovering",
        Technology.NRU,
        policy_kind,
        selected=0,
        remaining=0,
        db_state=DbState(interruptions=4, retries=2),
        db_initialized=True,
        deterministic_countdown=True,
    )
    other = Node("other", Technology.WIFI, PolicyKind.RANDOM, 0, 0)

    Channel(nodes=[recovering, other], seed=1).step()

    assert recovering.db_state.retries == 3
    assert recovering.db_state.interruptions == 4
    assert 0 <= recovering.selected <= 6
    assert recovering.remaining == recovering.selected
    assert not recovering.deterministic_countdown


def test_run_rearms_sender_and_continues_round_ids() -> None:
    node = Node(
        "w1", Technology.WIFI, PolicyKind.RANDOM, selected=0, remaining=0
    )
    channel = Channel(nodes=[node], seed=1)

    results = channel.run(3)

    assert [result.round_id for result in results] == [0, 1, 2]
    assert [result.kind for result in results] == ["success"] * 3
    assert node.remaining == node.selected
    assert channel.contention_round == 3
    assert channel.now_us == results[-1].now_us + channel.tx_us


def test_round_result_is_deeply_immutable() -> None:
    node_ids = ["w1"]
    technologies = ["wifi"]
    result = RoundResult(
        round_id=0,
        now_us=0,
        kind="success",
        node_ids=node_ids,
        technologies=technologies,
        collision_size=0,
        reservation_us=0,
        effective_data_us=2_000,
    )

    node_ids.append("later")
    technologies.append("nru")

    assert result.node_ids == ("w1",)
    assert result.technologies == ("wifi",)
    with pytest.raises(FrozenInstanceError):
        result.kind = "collision"  # type: ignore[misc]
    with pytest.raises(TypeError):
        result.node_ids[0] = "other"  # type: ignore[index]


@pytest.mark.parametrize(
    ("changes", "field"),
    [
        ({"kind": "other"}, "kind"),
        ({"kind": 1}, "kind"),
        ({"node_ids": [["w1"]]}, "node_ids"),
        ({"technologies": [["wifi"]]}, "technologies"),
    ],
)
def test_round_result_rejects_invalid_public_inputs(
    changes: dict[str, object], field: str
) -> None:
    values: dict[str, object] = {
        "round_id": 0,
        "now_us": 0,
        "kind": "success",
        "node_ids": ["w1"],
        "technologies": ["wifi"],
        "collision_size": 0,
        "reservation_us": 0,
        "effective_data_us": 2_000,
    }
    values.update(changes)

    with pytest.raises(ValueError, match=field):
        RoundResult(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("changes", "field"),
    [
        ({"technologies": []}, "technologies"),
        (
            {
                "kind": "collision",
                "node_ids": ["same", "same"],
                "technologies": ["wifi", "nru"],
                "collision_size": 2,
                "effective_data_us": 0,
            },
            "node_ids",
        ),
        ({"node_ids": [""]}, "node_ids"),
        ({"node_ids": ["   "]}, "node_ids"),
        ({"technologies": ["bluetooth"]}, "technologies"),
        ({"node_ids": [], "technologies": []}, "success"),
        (
            {
                "node_ids": ["w1", "n1"],
                "technologies": ["wifi", "nru"],
            },
            "success",
        ),
        ({"collision_size": 1}, "collision_size"),
        (
            {
                "kind": "collision",
                "collision_size": 1,
                "effective_data_us": 0,
            },
            "collision",
        ),
        (
            {
                "kind": "collision",
                "node_ids": ["w1", "n1"],
                "technologies": ["wifi", "nru"],
                "collision_size": 1,
                "effective_data_us": 0,
            },
            "collision_size",
        ),
        (
            {
                "kind": "collision",
                "node_ids": ["w1", "n1"],
                "technologies": ["wifi", "nru"],
                "collision_size": 2,
                "reservation_us": 1,
                "effective_data_us": 0,
            },
            "reservation_us",
        ),
        (
            {
                "kind": "collision",
                "node_ids": ["w1", "n1"],
                "technologies": ["wifi", "nru"],
                "collision_size": 2,
                "effective_data_us": 1,
            },
            "effective_data_us",
        ),
    ],
)
def test_round_result_rejects_cross_field_inconsistencies(
    changes: dict[str, object], field: str
) -> None:
    values: dict[str, object] = {
        "round_id": 0,
        "now_us": 0,
        "kind": "success",
        "node_ids": ["w1"],
        "technologies": ["wifi"],
        "collision_size": 0,
        "reservation_us": 0,
        "effective_data_us": 2_000,
    }
    values.update(changes)

    with pytest.raises(ValueError, match=field):
        RoundResult(**values)  # type: ignore[arg-type]


def test_collision_records_no_effective_data() -> None:
    nodes = [
        Node(
            "w1", Technology.WIFI, PolicyKind.RANDOM, selected=0, remaining=0
        ),
        Node(
            "n1", Technology.NRU, PolicyKind.RANDOM, selected=0, remaining=0
        ),
    ]

    result = Channel(nodes=nodes, seed=1).step()

    assert result.collision_size == 2
    assert result.reservation_us == 0
    assert result.effective_data_us == 0


def test_nru_success_reserves_until_next_sync_boundary() -> None:
    node = Node(
        "n1", Technology.NRU, PolicyKind.RANDOM, selected=125, remaining=125
    )
    channel = Channel(nodes=[node], seed=1)

    result = channel.step()

    assert result.now_us == 125
    assert result.reservation_us == 125
    assert result.effective_data_us == 1_875
    assert channel.now_us == 2_125


def test_nru_success_on_sync_boundary_has_no_reservation() -> None:
    node = Node(
        "n1", Technology.NRU, PolicyKind.RANDOM, selected=250, remaining=250
    )

    result = Channel(nodes=[node], seed=1).step()

    assert result.reservation_us == 0
    assert result.effective_data_us == 2_000


def test_wifi_success_excludes_ack_from_effective_data() -> None:
    node = Node(
        "w1", Technology.WIFI, PolicyKind.RANDOM, selected=0, remaining=0
    )

    result = Channel(nodes=[node], seed=1, wifi_ack_us=100).step()

    assert result.reservation_us == 0
    assert result.effective_data_us == 1_900


def test_access_delay_starts_after_first_success_end() -> None:
    node = Node(
        "w1", Technology.WIFI, PolicyKind.RANDOM, selected=0, remaining=0
    )
    channel = Channel(nodes=[node], seed=1)

    first = channel.step()
    assert first.now_us == 0
    assert node.access_delays_us == []
    assert node.last_success_end_us == 2_000

    node.selected = 125
    node.remaining = 125
    second = channel.step()

    assert second.now_us == 2_125
    assert node.access_delays_us == [125]
    assert node.last_success_end_us == 4_125


def test_channel_resumes_after_latest_recorded_success_end() -> None:
    node = Node(
        "w1",
        Technology.WIFI,
        PolicyKind.RANDOM,
        selected=125,
        remaining=125,
        last_success_end_us=2_000,
    )

    result = Channel(nodes=[node], seed=1).step()

    assert result.now_us == 2_125
    assert node.access_delays_us == [125]
    assert node.last_success_end_us == 4_125


def test_future_success_end_is_rejected_without_negative_delay() -> None:
    node = Node(
        "w1", Technology.WIFI, PolicyKind.RANDOM, selected=0, remaining=0
    )
    channel = Channel(nodes=[node], seed=1)
    node.last_success_end_us = 1

    with pytest.raises(ValueError, match="last_success_end_us"):
        channel.step()

    assert node.access_delays_us == []


def test_future_success_end_is_rejected_for_collision_without_mutation() -> None:
    nodes = [
        Node("w1", Technology.WIFI, PolicyKind.RANDOM, 0, 0),
        Node("n1", Technology.NRU, PolicyKind.RANDOM, 0, 0),
    ]
    channel = Channel(nodes=nodes, seed=1)
    nodes[0].last_success_end_us = 1

    with pytest.raises(ValueError, match="last_success_end_us"):
        channel.step()

    assert channel.now_us == 0
    assert channel.contention_round == 0
    assert [node.remaining for node in nodes] == [0, 0]
    assert [node.db_state.retries for node in nodes] == [0, 0]


def test_collision_does_not_record_success_data_or_delay() -> None:
    nodes = [
        Node(
            "w1", Technology.WIFI, PolicyKind.RANDOM, selected=0, remaining=0
        ),
        Node(
            "n1", Technology.NRU, PolicyKind.RANDOM, selected=0, remaining=0
        ),
    ]

    result = Channel(nodes=nodes, seed=1).step()

    assert result.effective_data_us == 0
    for node in nodes:
        assert node.last_success_end_us is None
        assert node.access_delays_us == []


def test_waiting_node_keeps_selection_until_it_transmits() -> None:
    sender = Node(
        "w1", Technology.WIFI, PolicyKind.RANDOM, selected=2, remaining=2
    )
    waiter = Node(
        "n1", Technology.NRU, PolicyKind.TMC_DB, selected=8, remaining=5
    )

    Channel(nodes=[sender, waiter], seed=1).step()

    assert waiter.selected == 8
    assert waiter.remaining == 3


def test_interruption_is_consumed_by_next_deterministic_recovery() -> None:
    first = Node(
        "w1", Technology.WIFI, PolicyKind.RANDOM, selected=0, remaining=0
    )
    recovering = Node(
        "n1",
        Technology.NRU,
        PolicyKind.TMC_DB,
        selected=5,
        remaining=5,
        db_initialized=True,
        deterministic_countdown=True,
    )
    channel = Channel(nodes=[first, recovering], seed=1)
    channel.step()

    first.selected = 5
    first.remaining = 5
    result = channel.step()

    assert result.kind == "collision"
    assert recovering.selected == 12
    assert recovering.remaining == 12
    assert recovering.db_state.interruptions == 0


def test_random_success_resets_grown_window_before_drawing() -> None:
    first = Node(
        "w1", Technology.WIFI, PolicyKind.RANDOM, selected=0, remaining=0
    )
    second = Node(
        "w2", Technology.WIFI, PolicyKind.RANDOM, selected=0, remaining=0
    )
    channel = Channel(nodes=[first, second], seed=1)
    channel.step()
    second.active = False
    first.remaining = 0

    channel.step()

    assert 0 <= first.selected <= 15
    assert first.remaining == first.selected


def test_seeded_node_streams_are_reproducible() -> None:
    def build() -> tuple[Channel, list[Node]]:
        nodes = [
            Node(
                "w1",
                Technology.WIFI,
                PolicyKind.RANDOM,
                selected=0,
                remaining=0,
            ),
            Node(
                "n1",
                Technology.NRU,
                PolicyKind.TMC_DB,
                selected=0,
                remaining=0,
            ),
        ]
        return Channel(nodes=nodes, seed=91), nodes

    left, left_nodes = build()
    right, right_nodes = build()

    assert left.run(8) == right.run(8)
    assert [node.selected for node in left_nodes] == [
        node.selected for node in right_nodes
    ]
    assert [node.remaining for node in left_nodes] == [
        node.remaining for node in right_nodes
    ]


def test_node_rng_is_independent_of_input_position() -> None:
    first = Node(
        "w1", Technology.WIFI, PolicyKind.RANDOM, selected=0, remaining=0
    )
    shifted = Node(
        "w1", Technology.WIFI, PolicyKind.RANDOM, selected=0, remaining=0
    )
    inactive = Node(
        "other",
        Technology.NRU,
        PolicyKind.RANDOM,
        selected=0,
        remaining=0,
        active=False,
    )

    Channel(nodes=[first], seed=12).step()
    Channel(nodes=[inactive, shifted], seed=12).step()

    expected = random.Random(
        derive_stream_seed(12, "w1", "backoff")
    ).randint(0, 15)
    assert first.selected == shifted.selected == expected


def test_node_rng_identity_is_distinct_and_policy_independent() -> None:
    random_node = Node(
        "same",
        Technology.WIFI,
        PolicyKind.RANDOM,
        selected=0,
        remaining=0,
        backlogged=False,
    )
    tmc_node = Node(
        "same",
        Technology.WIFI,
        PolicyKind.TMC_DB,
        selected=0,
        remaining=0,
        backlogged=False,
    )
    random_channel = Channel([random_node], seed=91)
    tmc_channel = Channel([tmc_node], seed=91)

    random_channel.set_backlogged("same", True)
    tmc_channel.set_backlogged("same", True)

    expected = random.Random(
        derive_stream_seed(91, "same", "backoff")
    ).randint(0, 15)
    assert random_node.selected == tmc_node.selected == expected
    assert derive_stream_seed(91, "same", "backoff") != derive_stream_seed(
        91, "different", "backoff"
    )


def test_channel_does_not_use_module_level_random(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: object, **kwargs: object) -> int:
        raise AssertionError("module-level RNG was used")

    monkeypatch.setattr(random, "randint", fail)
    node = Node(
        "w1", Technology.WIFI, PolicyKind.RANDOM, selected=0, remaining=0
    )

    Channel(nodes=[node], seed=1).step()


@pytest.mark.parametrize(
    ("changes", "field"),
    [
        ({"node_id": ""}, "node_id"),
        ({"node_id": "   "}, "node_id"),
        ({"node_id": 1}, "node_id"),
        ({"technology": "wifi"}, "technology"),
        ({"policy_kind": "random_lbt"}, "policy_kind"),
        ({"selected": -1}, "selected"),
        ({"selected": True}, "selected"),
        ({"selected": 1.0}, "selected"),
        ({"remaining": -1}, "remaining"),
        ({"remaining": True}, "remaining"),
        ({"remaining": 1.0}, "remaining"),
        ({"active": 1}, "active"),
        ({"backlogged": 1}, "backlogged"),
        ({"db_state": object()}, "db_state"),
        ({"db_initialized": 1}, "db_initialized"),
        ({"db_initialized": 0.0}, "db_initialized"),
        ({"deterministic_countdown": 1}, "deterministic_countdown"),
        ({"deterministic_countdown": 0.0}, "deterministic_countdown"),
        ({"last_success_end_us": -1}, "last_success_end_us"),
        ({"last_success_end_us": True}, "last_success_end_us"),
        ({"access_delays_us": [-1]}, "access_delays_us"),
        ({"access_delays_us": [True]}, "access_delays_us"),
    ],
)
def test_node_rejects_invalid_inputs(
    changes: dict[str, object], field: str
) -> None:
    values: dict[str, object] = {
        "node_id": "w1",
        "technology": Technology.WIFI,
        "policy_kind": PolicyKind.RANDOM,
        "selected": 0,
        "remaining": 0,
    }
    values.update(changes)

    with pytest.raises(ValueError, match=field):
        Node(**values)  # type: ignore[arg-type]


def test_node_rejects_remaining_above_selected() -> None:
    with pytest.raises(ValueError, match="remaining"):
        Node(
            "w1",
            Technology.WIFI,
            PolicyKind.RANDOM,
            selected=2,
            remaining=3,
        )


@pytest.mark.parametrize(
    ("changes", "field"),
    [
        ({"seed": -1}, "seed"),
        ({"seed": True}, "seed"),
        ({"seed": 1.0}, "seed"),
        ({"slot_us": 0}, "slot_us"),
        ({"slot_us": 1.0}, "slot_us"),
        ({"tx_us": 0}, "tx_us"),
        ({"wifi_ack_us": -1}, "wifi_ack_us"),
        ({"wifi_ack_us": 2_001}, "wifi_ack_us"),
        ({"nru_sync_us": 0}, "nru_sync_us"),
        ({"nru_sync_us": 2_001}, "nru_sync_us"),
    ],
)
def test_channel_rejects_invalid_inputs(
    changes: dict[str, object], field: str
) -> None:
    values: dict[str, object] = {
        "nodes": [
            Node(
                "w1",
                Technology.WIFI,
                PolicyKind.RANDOM,
                selected=0,
                remaining=0,
            )
        ],
        "seed": 1,
    }
    values.update(changes)

    with pytest.raises(ValueError, match=field):
        Channel(**values)  # type: ignore[arg-type]


def test_channel_rejects_duplicate_node_ids() -> None:
    nodes = [
        Node("same", Technology.WIFI, PolicyKind.RANDOM, 0, 0),
        Node("same", Technology.NRU, PolicyKind.RANDOM, 0, 0),
    ]

    with pytest.raises(ValueError, match="node_id"):
        Channel(nodes=nodes, seed=1)


def test_channel_rejects_nodes_that_share_db_state_before_collision() -> None:
    shared_state = DbState()
    nodes = [
        Node(
            "w1",
            Technology.WIFI,
            PolicyKind.TMC_DB,
            0,
            0,
            db_state=shared_state,
        ),
        Node(
            "n1",
            Technology.NRU,
            PolicyKind.TMC_DB,
            0,
            0,
            db_state=shared_state,
        ),
    ]

    with pytest.raises(ValueError, match="db_state"):
        Channel(nodes=nodes, seed=1)

    assert shared_state.retries == 0
    assert shared_state.interruptions == 0
    assert [node.remaining for node in nodes] == [0, 0]


def test_step_rejects_technology_replacement_without_side_effects() -> None:
    node = Node("w1", Technology.WIFI, PolicyKind.RANDOM, 0, 0)
    control_node = Node("w1", Technology.WIFI, PolicyKind.RANDOM, 0, 0)
    channel = Channel(nodes=[node], seed=31)
    control = Channel(nodes=[control_node], seed=31)
    node.technology = Technology.NRU

    with pytest.raises(ValueError, match="technology"):
        channel.step()

    assert channel.now_us == 0
    assert channel.contention_round == 0
    assert node.selected == 0
    assert node.remaining == 0
    assert node.last_success_end_us is None
    assert node.access_delays_us == []

    node.technology = Technology.WIFI
    assert channel.step() == control.step()
    assert node.selected == control_node.selected


def test_step_rejects_db_state_replacement_without_side_effects() -> None:
    original_state = DbState()
    replacement_state = DbState(interruptions=2, retries=3)
    node = Node(
        "w1",
        Technology.WIFI,
        PolicyKind.PRIMARY_DB,
        0,
        0,
        db_state=original_state,
    )
    control_node = Node(
        "w1", Technology.WIFI, PolicyKind.PRIMARY_DB, 0, 0
    )
    channel = Channel(nodes=[node], seed=31)
    control = Channel(nodes=[control_node], seed=31)
    node.db_state = replacement_state

    with pytest.raises(ValueError, match="db_state"):
        channel.step()

    assert channel.now_us == 0
    assert channel.contention_round == 0
    assert (original_state.interruptions, original_state.retries) == (0, 0)
    assert (replacement_state.interruptions, replacement_state.retries) == (
        2,
        3,
    )
    assert node.selected == 0
    assert node.remaining == 0
    assert node.last_success_end_us is None

    node.db_state = original_state
    assert channel.step() == control.step()
    assert node.selected == control_node.selected


@pytest.mark.parametrize(
    ("mutation", "field"),
    [
        (lambda node: setattr(node, "remaining", -1), "remaining"),
        (lambda node: setattr(node, "node_id", "changed"), "node_id"),
        (
            lambda node: setattr(node, "policy_kind", PolicyKind.TMC_DB),
            "policy_kind",
        ),
        (lambda node: setattr(node.db_state, "retries", -1), "retries"),
        (
            lambda node: setattr(node, "db_initialized", 1),
            "db_initialized",
        ),
        (
            lambda node: setattr(node, "deterministic_countdown", 0.0),
            "deterministic_countdown",
        ),
    ],
)
def test_step_revalidates_mutable_node_before_any_mutation(
    mutation: object, field: str
) -> None:
    node = Node(
        "w1", Technology.WIFI, PolicyKind.RANDOM, selected=0, remaining=0
    )
    channel = Channel(nodes=[node], seed=1)
    mutation(node)  # type: ignore[operator]

    with pytest.raises(ValueError, match=field):
        channel.step()

    assert channel.now_us == 0
    assert channel.contention_round == 0
    assert node.last_success_end_us is None


@pytest.mark.parametrize(
    "mutation",
    [
        lambda node: setattr(node, "remaining", 3),
        lambda node: setattr(node, "selected", 1),
    ],
)
def test_step_rejects_remaining_above_selected_without_mutation(
    mutation: object,
) -> None:
    node = Node(
        "w1", Technology.WIFI, PolicyKind.RANDOM, selected=2, remaining=2
    )
    channel = Channel(nodes=[node], seed=1)
    mutation(node)  # type: ignore[operator]

    with pytest.raises(ValueError, match="remaining"):
        channel.step()

    assert channel.now_us == 0
    assert channel.contention_round == 0
    assert node.last_success_end_us is None
    assert node.access_delays_us == []


@pytest.mark.parametrize(
    "policy_kind", [PolicyKind.TMC_DB, PolicyKind.ADAPTIVE]
)
@pytest.mark.parametrize(
    "field", ["deterministic_countdown", "retries", "interruptions"]
)
@pytest.mark.parametrize("operation", ["step", "background_busy"])
def test_channel_operation_rejects_corrupted_uninitialized_tmc_atomically(
    policy_kind: PolicyKind,
    field: str,
    operation: str,
) -> None:
    node = Node(
        "same", Technology.WIFI, policy_kind, selected=4, remaining=4
    )
    control_node = Node(
        "same", Technology.WIFI, policy_kind, selected=4, remaining=4
    )
    channel = Channel(nodes=[node], seed=31)
    control = Channel(nodes=[control_node], seed=31)

    if field == "deterministic_countdown":
        node.deterministic_countdown = True
    elif field == "retries":
        node.db_state.retries = 1
    else:
        node.db_state.interruptions = 1
    corrupted_db_state = (
        node.db_state.interruptions,
        node.db_state.retries,
    )

    with pytest.raises(ValueError, match=field):
        if operation == "step":
            channel.step()
        else:
            channel.apply_background_busy(7)

    assert channel.now_us == 0
    assert channel.contention_round == 0
    assert (node.selected, node.remaining) == (4, 4)
    assert node.db_initialized is False
    assert (
        node.db_state.interruptions,
        node.db_state.retries,
    ) == corrupted_db_state
    assert node.last_success_end_us is None
    assert node.access_delays_us == []

    node.deterministic_countdown = False
    node.db_state.interruptions = 0
    node.db_state.retries = 0
    assert channel.step() == control.step()
    assert node.selected == control_node.selected
    assert node.remaining == control_node.remaining


@pytest.mark.parametrize("rounds", [-1, True, 1.0])
def test_run_rejects_invalid_round_count(rounds: object) -> None:
    node = Node(
        "w1", Technology.WIFI, PolicyKind.RANDOM, selected=0, remaining=0
    )

    with pytest.raises(ValueError, match="rounds"):
        Channel(nodes=[node], seed=1).run(rounds)  # type: ignore[arg-type]


def test_legacy_sta_random_window_can_exceed_63_and_caps_at_1023() -> None:
    sta = Node(
        "sta", Technology.LEGACY_STA, PolicyKind.RANDOM, selected=0, remaining=0
    )
    ap = Node(
        "ap", Technology.LEGACY_AP, PolicyKind.RANDOM, selected=0, remaining=0
    )
    channel = Channel([sta, ap], seed=17)
    sta_draws: list[int] = []
    ap_draws: list[int] = []

    for _ in range(12):
        sta.remaining = 0
        ap.remaining = 0
        channel.step()
        sta_draws.append(sta.selected)
        ap_draws.append(ap.selected)

    assert any(value > 63 for value in sta_draws)
    assert max(sta_draws) <= 1023
    assert max(ap_draws) <= 63


def test_access_delay_history_rejects_invalid_inplace_mutation() -> None:
    node = Node(
        "w1", Technology.WIFI, PolicyKind.RANDOM, selected=0, remaining=0
    )

    with pytest.raises(ValueError, match="access_delays_us"):
        node.access_delays_us.append(-1)

    assert node.access_delays_us == []


def test_access_delay_history_cannot_bypass_validation_with_list_descriptor() -> None:
    node = Node(
        "w1", Technology.WIFI, PolicyKind.RANDOM, selected=0, remaining=0
    )

    assert not isinstance(node.access_delays_us, list)
    with pytest.raises(TypeError):
        list.append(node.access_delays_us, -1)  # type: ignore[arg-type]

    assert node.access_delays_us == []


def test_access_delay_history_validates_mutable_sequence_operations() -> None:
    source = [1, 2]
    node = Node(
        "w1",
        Technology.WIFI,
        PolicyKind.RANDOM,
        selected=0,
        remaining=0,
        access_delays_us=source,
    )
    history = node.access_delays_us
    source.append(99)

    history.append(3)
    history.insert(0, 0)
    history[1] = 4
    history[2:3] = [5, 6]
    del history[0]

    assert history == [4, 5, 6, 3]
    assert list(history) == [4, 5, 6, 3]
    assert len(history) == 4
    assert history[-1] == 3
    with pytest.raises(ValueError, match="access_delays_us"):
        history.append(-1)
    with pytest.raises(ValueError, match="access_delays_us"):
        history[0] = True
    with pytest.raises(ValueError, match="access_delays_us"):
        history[1:3] = [7, -1]
    assert history == [4, 5, 6, 3]


def test_access_delay_history_extend_is_atomic_on_invalid_value() -> None:
    node = Node(
        "w1",
        Technology.WIFI,
        PolicyKind.RANDOM,
        selected=0,
        remaining=0,
        access_delays_us=[1, 2],
    )

    with pytest.raises(ValueError, match="access_delays_us"):
        node.access_delays_us.extend([7, -1])

    assert node.access_delays_us == [1, 2]


def test_access_delay_history_iadd_is_atomic_on_invalid_value() -> None:
    node = Node(
        "w1",
        Technology.WIFI,
        PolicyKind.RANDOM,
        selected=0,
        remaining=0,
        access_delays_us=[1, 2],
    )
    history = node.access_delays_us

    with pytest.raises(ValueError, match="access_delays_us"):
        history += [9, -1]

    assert history is node.access_delays_us
    assert history == [1, 2]


def test_channel_rejects_access_delay_history_replacement_atomically() -> None:
    node = Node(
        "w1", Technology.WIFI, PolicyKind.RANDOM, selected=0, remaining=0
    )
    channel = Channel([node], seed=7)
    original = node.access_delays_us
    node.access_delays_us = []

    with pytest.raises(ValueError, match="access_delays_us cannot change"):
        channel.step()

    assert channel.now_us == 0
    assert channel.contention_round == 0
    node.access_delays_us = original
    assert channel.step().round_id == 0
