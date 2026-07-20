from dataclasses import dataclass
import random

from dblbt_fcn.channel import Channel, Node, RoundResult
from dblbt_fcn.metrics import collision_probability, nearest_rank_p95
from dblbt_fcn.types import PolicyKind, Technology


INITIAL_SEED = 410
CHANNEL_SEED = 410
PAIRED_ROUNDS = 5_000


@dataclass(frozen=True)
class _Run:
    initial_selected: tuple[int, ...]
    selected_trace: tuple[tuple[int, ...], ...]
    results: tuple[RoundResult, ...]
    access_delays_us: tuple[int, ...]


def _saturated_nodes(policy_kind: PolicyKind) -> list[Node]:
    rng = random.Random(INITIAL_SEED)
    identities = [
        *((f"wifi-{index}", Technology.WIFI) for index in range(4)),
        *((f"nru-{index}", Technology.NRU) for index in range(4)),
    ]
    nodes = []
    for node_id, technology in identities:
        selected = rng.randint(0, 15)
        nodes.append(
            Node(
                node_id=node_id,
                technology=technology,
                policy_kind=policy_kind,
                selected=selected,
                remaining=selected,
                active=True,
                backlogged=True,
            )
        )
    return nodes


def _run(policy_kind: PolicyKind, rounds: int) -> _Run:
    nodes = _saturated_nodes(policy_kind)
    initial_selected = tuple(node.selected for node in nodes)
    channel = Channel(nodes, seed=CHANNEL_SEED)
    results = []
    selected_trace = []
    for _ in range(rounds):
        results.append(channel.step())
        selected_trace.append(tuple(node.selected for node in nodes))
    access_delays_us = tuple(
        delay for node in nodes for delay in node.access_delays_us
    )
    return _Run(
        initial_selected=initial_selected,
        selected_trace=tuple(selected_trace),
        results=tuple(results),
        access_delays_us=access_delays_us,
    )


def test_paired_topology_has_stable_ids_and_identical_initial_state() -> None:
    random_nodes = _saturated_nodes(PolicyKind.RANDOM)
    tmc_nodes = _saturated_nodes(PolicyKind.TMC_DB)

    assert [node.node_id for node in random_nodes] == [
        "wifi-0",
        "wifi-1",
        "wifi-2",
        "wifi-3",
        "nru-0",
        "nru-1",
        "nru-2",
        "nru-3",
    ]
    assert [node.selected for node in random_nodes] == [
        node.selected for node in tmc_nodes
    ]
    assert all(0 <= node.selected <= 15 for node in random_nodes)
    assert all(node.active and node.backlogged for node in random_nodes)


def test_tmc_fixed_stable_trace_reaches_selected_backoff_18() -> None:
    run = _run(PolicyKind.TMC_DB, PAIRED_ROUNDS)

    assert any(18 in selected for selected in run.selected_trace)


def test_tmc_fixed_improves_paired_collision_and_p95_delay() -> None:
    random_run = _run(PolicyKind.RANDOM, PAIRED_ROUNDS)
    tmc_run = _run(PolicyKind.TMC_DB, PAIRED_ROUNDS)

    assert random_run.initial_selected == tmc_run.initial_selected
    assert random_run.access_delays_us
    assert tmc_run.access_delays_us
    random_collision = collision_probability(random_run.results)
    tmc_collision = collision_probability(tmc_run.results)
    random_p95 = nearest_rank_p95(random_run.access_delays_us)
    tmc_p95 = nearest_rank_p95(tmc_run.access_delays_us)

    assert tmc_collision < random_collision, {
        "random_collision": random_collision,
        "tmc_collision": tmc_collision,
    }
    assert tmc_p95 < random_p95, {
        "random_p95_us": random_p95,
        "tmc_p95_us": tmc_p95,
    }


def test_same_seed_repeats_trace_and_metrics() -> None:
    first = _run(PolicyKind.TMC_DB, PAIRED_ROUNDS)
    second = _run(PolicyKind.TMC_DB, PAIRED_ROUNDS)

    assert first.selected_trace == second.selected_trace
    assert first.results == second.results
    assert first.access_delays_us == second.access_delays_us
    assert collision_probability(first.results) == collision_probability(
        second.results
    )
    assert nearest_rank_p95(first.access_delays_us) == nearest_rank_p95(
        second.access_delays_us
    )
