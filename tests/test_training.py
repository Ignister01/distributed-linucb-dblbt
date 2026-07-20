"""Leakage-resistant Oracle fitting and local-only policy training."""

import ast
from dataclasses import FrozenInstanceError, fields
from itertools import permutations
import math
from pathlib import Path
import sys

import numpy as np
import pytest

from dblbt_fcn.linucb import LinUCB
from dblbt_fcn.training import (
    HELD_OUT_SEEDS,
    PRETRAINING_SEEDS,
    LocalSample,
    OracleSample,
    deploy_independent,
    fit_fixed_oracle,
    pretrain_linucb,
)


LOCAL_FIELDS = (
    "context",
    "arm",
    "local_reward",
    "local_sequence",
    "node_id",
    "pretraining_seed",
)


def local_sample(**changes: object) -> LocalSample:
    values: dict[str, object] = {
        "context": [1.0, 0.0],
        "arm": 0,
        "local_reward": 1.0,
        "local_sequence": 0,
        "node_id": "node-a",
        "pretraining_seed": 1103,
    }
    values.update(changes)
    return LocalSample(**values)


def test_seed_partitions_are_fixed_immutable_and_disjoint() -> None:
    assert PRETRAINING_SEEDS == frozenset({1103, 2207, 3301})
    assert HELD_OUT_SEEDS == frozenset(
        {410, 523, 631, 742, 859, 967, 1081, 1193, 1307, 1429}
    )
    assert isinstance(PRETRAINING_SEEDS, frozenset)
    assert isinstance(HELD_OUT_SEEDS, frozenset)
    assert PRETRAINING_SEEDS.isdisjoint(HELD_OUT_SEEDS)


def test_local_sample_has_only_the_immutable_local_schema() -> None:
    sample = local_sample(
        context=np.array([1, 2], dtype=np.int16),
        arm=np.int64(1),
        local_reward=np.float32(0.25),
        local_sequence=np.int32(3),
        pretraining_seed=np.int64(2207),
    )

    assert tuple(field.name for field in fields(LocalSample)) == LOCAL_FIELDS
    assert tuple(LocalSample.__annotations__) == LOCAL_FIELDS
    assert sample.context == (1.0, 2.0)
    assert sample.arm == 1
    assert sample.local_reward == 0.25
    assert sample.local_sequence == 3
    assert sample.pretraining_seed == 2207
    with pytest.raises(FrozenInstanceError):
        sample.arm = 0  # type: ignore[misc]


@pytest.mark.parametrize(
    "extra_name",
    [
        "global_utility",
        "jain_index",
        "other_queue",
        "technology_totals",
        "true_node_count",
        "heldout_seed",
    ],
)
def test_local_sample_rejects_every_extra_field(extra_name: str) -> None:
    values = {
        "context": [1.0, 0.0],
        "arm": 0,
        "local_reward": 1.0,
        "local_sequence": 0,
        "node_id": "node-a",
        "pretraining_seed": 1103,
        extra_name: 1,
    }

    with pytest.raises(TypeError):
        LocalSample(**values)


@pytest.mark.parametrize(
    "context",
    [
        1.0,
        "12",
        [[1.0, 2.0]],
        [1.0, math.nan],
        [1.0, math.inf],
        [1.0, 2.0 + 0j],
        [True, False],
        [True, 1.0],
        np.array([1.0, object()], dtype=object),
    ],
)
def test_local_sample_rejects_invalid_context(context: object) -> None:
    with pytest.raises(ValueError, match="context"):
        local_sample(context=context)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"arm": -1}, "arm"),
        ({"arm": True}, "arm"),
        ({"arm": 1.0}, "arm"),
        ({"local_reward": True}, "local_reward"),
        ({"local_reward": math.nan}, "local_reward"),
        ({"local_reward": math.inf}, "local_reward"),
        ({"local_sequence": -1}, "local_sequence"),
        ({"local_sequence": True}, "local_sequence"),
        ({"local_sequence": 1.0}, "local_sequence"),
        ({"node_id": ""}, "node_id"),
        ({"node_id": " node-a"}, "node_id"),
        ({"node_id": "node-a "}, "node_id"),
        ({"node_id": 1}, "node_id"),
        ({"pretraining_seed": True}, "pretraining_seed"),
        ({"pretraining_seed": 410}, "pretraining_seed"),
        ({"pretraining_seed": 9999}, "pretraining_seed"),
    ],
)
def test_local_sample_validates_each_field(
    change: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        local_sample(**change)


def test_oracle_sample_is_minimal_immutable_and_normalized() -> None:
    row = OracleSample(
        arm=np.int64(2), utility=np.float32(0.5), seed=np.int32(1103)
    )

    assert tuple(field.name for field in fields(OracleSample)) == (
        "arm",
        "utility",
        "seed",
    )
    assert row == OracleSample(arm=2, utility=0.5, seed=1103)
    with pytest.raises(FrozenInstanceError):
        row.utility = 1.0  # type: ignore[misc]


def test_oracle_sample_seed_requires_only_an_integer() -> None:
    row = OracleSample(arm=0, utility=1.0, seed=-1)

    assert row.seed == -1
    with pytest.raises(ValueError, match="row.*seed"):
        fit_fixed_oracle([row], {1103})


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"arm": -1, "utility": 1.0, "seed": 1103}, "arm"),
        ({"arm": True, "utility": 1.0, "seed": 1103}, "arm"),
        ({"arm": 1.0, "utility": 1.0, "seed": 1103}, "arm"),
        ({"arm": 0, "utility": True, "seed": 1103}, "utility"),
        ({"arm": 0, "utility": math.nan, "seed": 1103}, "utility"),
        ({"arm": 0, "utility": math.inf, "seed": 1103}, "utility"),
        ({"arm": 0, "utility": 1.0, "seed": True}, "seed"),
        ({"arm": 0, "utility": 1.0, "seed": 1103.0}, "seed"),
    ],
)
def test_oracle_sample_validates_each_field(
    values: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        OracleSample(**values)


def test_fixed_oracle_chooses_highest_mean_and_lowest_exact_tie() -> None:
    rows = [
        OracleSample(arm=2, utility=0.0, seed=1103),
        OracleSample(arm=1, utility=0.4, seed=1103),
        OracleSample(arm=2, utility=1.0, seed=2207),
        OracleSample(arm=1, utility=0.6, seed=2207),
        OracleSample(arm=0, utility=0.5, seed=3301),
    ]

    assert fit_fixed_oracle(rows, PRETRAINING_SEEDS) == 0


def test_fixed_oracle_uses_means_instead_of_totals() -> None:
    rows = [
        OracleSample(arm=0, utility=0.8, seed=1103),
        OracleSample(arm=1, utility=0.6, seed=1103),
        OracleSample(arm=1, utility=0.6, seed=2207),
    ]

    assert fit_fixed_oracle(rows, {1103, 2207}) == 0


def test_fixed_oracle_ranks_extreme_finite_means_exactly() -> None:
    maximum = sys.float_info.max
    previous = math.nextafter(maximum, 0.0)
    rows = [
        OracleSample(arm=0, utility=maximum, seed=1103),
        OracleSample(arm=0, utility=previous, seed=2207),
        OracleSample(arm=1, utility=maximum, seed=3301),
    ]

    assert fit_fixed_oracle(rows, PRETRAINING_SEEDS) == 1


def test_fixed_oracle_choice_is_invariant_to_row_order() -> None:
    rows = [
        OracleSample(arm=0, utility=0.0, seed=1103),
        OracleSample(arm=0, utility=1.0, seed=2207),
        OracleSample(arm=1, utility=0.5, seed=3301),
    ]

    choices = {
        fit_fixed_oracle(order, PRETRAINING_SEEDS)
        for order in permutations(rows)
    }

    assert choices == {0}


@pytest.mark.parametrize("bad_seed", [410, 9999, True])
def test_fixed_oracle_rejects_invalid_allowed_seed(bad_seed: object) -> None:
    rows = [OracleSample(arm=0, utility=1.0, seed=1103)]

    with pytest.raises(ValueError, match="allowed_seeds"):
        fit_fixed_oracle(rows, {1103, bad_seed})


def test_fixed_oracle_rejects_empty_allowed_seeds_and_rows() -> None:
    rows = [OracleSample(arm=0, utility=1.0, seed=1103)]

    with pytest.raises(ValueError, match="allowed_seeds"):
        fit_fixed_oracle(rows, set())
    with pytest.raises(ValueError, match="rows"):
        fit_fixed_oracle([], {1103})


@pytest.mark.parametrize("row_seed", [410, 9999])
def test_fixed_oracle_rejects_rows_outside_allowed_seeds(
    row_seed: int,
) -> None:
    rows = [OracleSample(arm=0, utility=1.0, seed=row_seed)]

    with pytest.raises(ValueError, match="row.*seed"):
        fit_fixed_oracle(rows, {1103})


def test_fixed_oracle_rejects_disallowed_training_rows() -> None:
    rows = [OracleSample(arm=0, utility=1.0, seed=2207)]

    with pytest.raises(ValueError, match="row.*seed"):
        fit_fixed_oracle(rows, {1103})


def test_fixed_oracle_rejects_duplicate_seed_arm_rows() -> None:
    rows = [
        OracleSample(arm=0, utility=1.0, seed=1103),
        OracleSample(arm=0, utility=0.5, seed=1103),
    ]

    with pytest.raises(ValueError, match="duplicate"):
        fit_fixed_oracle(rows, {1103})


def test_pretraining_uses_only_local_reward_and_preserves_initial() -> None:
    initial = LinUCB(2, 2, ridge=2.0, exploration=0.25)
    initial_A = initial.A.tobytes()
    initial_b = initial.b.tobytes()

    trained = pretrain_linucb(
        [local_sample(context=[2.0, 3.0], arm=1, local_reward=0.5)],
        initial,
    )

    assert trained is not initial
    assert initial.A.tobytes() == initial_A
    assert initial.b.tobytes() == initial_b
    np.testing.assert_array_equal(
        trained.A[1], np.array([[6.0, 6.0], [6.0, 11.0]])
    )
    np.testing.assert_array_equal(trained.b[1], np.array([1.0, 1.5]))


def test_pretraining_empty_input_returns_an_independent_exact_clone() -> None:
    initial = LinUCB(2, 2)

    trained = pretrain_linucb([], initial)

    assert trained is not initial
    assert trained.A.tobytes() == initial.A.tobytes()
    assert trained.b.tobytes() == initial.b.tobytes()
    assert not np.shares_memory(trained.A, initial.A)
    assert not np.shares_memory(trained.b, initial.b)


def test_pretraining_canonical_order_is_byte_exact() -> None:
    samples = [
        local_sample(
            context=[1.0e8, 1.0],
            arm=0,
            local_reward=1.0e-8,
            local_sequence=2,
            node_id="node-b",
            pretraining_seed=3301,
        ),
        local_sample(
            context=[1.0, 2.0],
            arm=1,
            local_reward=-0.25,
            local_sequence=1,
            node_id="node-a",
            pretraining_seed=1103,
        ),
        local_sample(
            context=[1.0e-8, 3.0],
            arm=0,
            local_reward=1.0e8,
            local_sequence=0,
            node_id="node-c",
            pretraining_seed=2207,
        ),
    ]

    forward = pretrain_linucb(samples, LinUCB(2, 2))
    reverse = pretrain_linucb(reversed(samples), LinUCB(2, 2))

    assert forward.A.tobytes() == reverse.A.tobytes()
    assert forward.b.tobytes() == reverse.b.tobytes()


def test_pretraining_rejects_duplicate_local_sequence_provenance() -> None:
    samples = [
        local_sample(arm=0),
        local_sample(arm=1, local_reward=0.5),
    ]

    with pytest.raises(ValueError, match="duplicate"):
        pretrain_linucb(samples, LinUCB(2, 2))


@pytest.mark.parametrize(
    "sample",
    [
        local_sample(context=[1.0]),
        local_sample(context=[1.0, 2.0, 3.0]),
        local_sample(arm=2),
    ],
)
def test_pretraining_rejects_agent_incompatible_samples(
    sample: LocalSample,
) -> None:
    with pytest.raises(ValueError):
        pretrain_linucb([sample], LinUCB(2, 2))


def test_deployment_creates_independent_per_node_agents() -> None:
    initial = LinUCB(2, 2)
    initial.update(1, [1.0, 2.0], 0.5)
    initial_A = initial.A.tobytes()
    initial_b = initial.b.tobytes()

    deployed = deploy_independent(initial, ["node-a", "node-b"])

    assert set(deployed) == {"node-a", "node-b"}
    for agent in deployed.values():
        assert agent.A.tobytes() == initial_A
        assert agent.b.tobytes() == initial_b
        assert not np.shares_memory(agent.A, initial.A)
        assert not np.shares_memory(agent.b, initial.b)
    assert not np.shares_memory(deployed["node-a"].A, deployed["node-b"].A)
    assert not np.shares_memory(deployed["node-a"].b, deployed["node-b"].b)

    deployed["node-a"].update(0, [2.0, 1.0], 1.0)
    assert deployed["node-b"].A.tobytes() == initial_A
    assert deployed["node-b"].b.tobytes() == initial_b
    assert initial.A.tobytes() == initial_A
    assert initial.b.tobytes() == initial_b


@pytest.mark.parametrize(
    "node_ids",
    [
        ["node-a", "node-a"],
        [""],
        [" node-a"],
        ["node-a "],
        [1],
    ],
)
def test_deployment_rejects_invalid_node_ids(node_ids: list[object]) -> None:
    with pytest.raises(ValueError, match="node"):
        deploy_independent(LinUCB(1, 1), node_ids)


def test_deployment_accepts_no_nodes() -> None:
    assert deploy_independent(LinUCB(1, 1), []) == {}


def test_training_module_has_no_forbidden_locality_identifiers() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "dblbt_fcn"
        / "training.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    identifiers = [
        node.id if isinstance(node, ast.Name) else node.attr
        for node in ast.walk(tree)
        if isinstance(node, (ast.Name, ast.Attribute))
    ]
    forbidden = (
        "global",
        "jain",
        "other_queue",
        "true_node_count",
        "technology_totals",
    )

    assert not {
        identifier
        for identifier in identifiers
        if any(token in identifier.lower() for token in forbidden)
    }
