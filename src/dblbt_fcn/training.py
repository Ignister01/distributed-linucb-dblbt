"""Fixed-reference fitting and leakage-resistant local policy training."""

from dataclasses import dataclass
from fractions import Fraction
import math
from numbers import Integral, Real
from typing import Final, Iterable

import numpy as np

from dblbt_fcn.linucb import LinUCB


PRETRAINING_SEEDS: Final[frozenset[int]] = frozenset({1103, 2207, 3301})
HELD_OUT_SEEDS: Final[frozenset[int]] = frozenset(
    {410, 523, 631, 742, 859, 967, 1081, 1193, 1307, 1429}
)


def _integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _nonnegative_integer(name: str, value: object) -> int:
    normalized = _integer(name, value)
    if normalized < 0:
        raise ValueError(f"{name} must be non-negative")
    return normalized


def _finite_number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number")
    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _context_tuple(value: object) -> tuple[float, ...]:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "context must be a one-dimensional real sequence"
        ) from error
    if raw.ndim != 1:
        raise ValueError("context must be a one-dimensional real sequence")
    if raw.dtype.kind not in "iuf":
        raise ValueError("context must contain only real numeric values")

    try:
        source_values = tuple(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError(
            "context must be a one-dimensional real sequence"
        ) from error
    values = tuple(
        _finite_number("context", item) for item in source_values
    )
    return values


def _node_identifier(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("node_id must be a string")
    if not value or value != value.strip():
        raise ValueError("node_id must be non-empty without outer whitespace")
    return value


@dataclass(frozen=True, slots=True)
class LocalSample:
    """One node's local observation and reward from a training run."""

    context: tuple[float, ...]
    arm: int
    local_reward: float
    local_sequence: int
    node_id: str
    pretraining_seed: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "context", _context_tuple(self.context))
        object.__setattr__(
            self, "arm", _nonnegative_integer("arm", self.arm)
        )
        object.__setattr__(
            self,
            "local_reward",
            _finite_number("local_reward", self.local_reward),
        )
        object.__setattr__(
            self,
            "local_sequence",
            _nonnegative_integer("local_sequence", self.local_sequence),
        )
        object.__setattr__(self, "node_id", _node_identifier(self.node_id))
        seed = _nonnegative_integer(
            "pretraining_seed", self.pretraining_seed
        )
        if seed not in PRETRAINING_SEEDS:
            raise ValueError(
                "pretraining_seed must belong to PRETRAINING_SEEDS"
            )
        object.__setattr__(self, "pretraining_seed", seed)


@dataclass(frozen=True, slots=True)
class OracleSample:
    """Minimal non-deployable evaluation row used by the fixed Oracle."""

    arm: int
    utility: float
    seed: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "arm", _nonnegative_integer("arm", self.arm)
        )
        object.__setattr__(
            self, "utility", _finite_number("utility", self.utility)
        )
        object.__setattr__(
            self, "seed", _integer("seed", self.seed)
        )


def _materialize(name: str, values: object) -> tuple[object, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be an iterable of values")
    try:
        return tuple(values)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError(f"{name} must be an iterable of values") from error


def _stable_mean(values: list[float]) -> float:
    exact_total = sum(
        (Fraction.from_float(value) for value in values), Fraction()
    )
    return float(exact_total / len(values))


def fit_fixed_oracle(
    rows: Iterable[OracleSample], allowed_seeds: Iterable[Integral]
) -> int:
    """Select the lowest arm with the highest allowed-run mean utility."""
    raw_allowed = _materialize("allowed_seeds", allowed_seeds)
    if not raw_allowed:
        raise ValueError("allowed_seeds must be non-empty")

    normalized_allowed: set[int] = set()
    for value in raw_allowed:
        if (
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or int(value) not in PRETRAINING_SEEDS
        ):
            raise ValueError(
                "allowed_seeds must contain only PRETRAINING_SEEDS"
            )
        normalized_allowed.add(int(value))

    raw_rows = _materialize("rows", rows)
    if not raw_rows:
        raise ValueError("rows must be non-empty")

    utilities: dict[int, list[float]] = {}
    provenance: set[tuple[int, int]] = set()
    for row in raw_rows:
        if not isinstance(row, OracleSample):
            raise ValueError("rows must contain only OracleSample values")
        if row.seed not in normalized_allowed:
            raise ValueError("row seed is not in allowed_seeds")
        key = (row.seed, row.arm)
        if key in provenance:
            raise ValueError("duplicate (seed, arm) Oracle row")
        provenance.add(key)
        utilities.setdefault(row.arm, []).append(row.utility)

    means = {
        arm: _stable_mean(values) for arm, values in utilities.items()
    }
    return min(means, key=lambda arm: (-means[arm], arm))


def pretrain_linucb(
    samples: Iterable[LocalSample], initial: LinUCB
) -> LinUCB:
    """Train an independent clone from canonicalized local samples only."""
    if not isinstance(initial, LinUCB):
        raise ValueError("initial must be a LinUCB agent")
    raw_samples = _materialize("samples", samples)
    validated: list[LocalSample] = []
    provenance: set[tuple[int, str, int]] = set()
    for sample in raw_samples:
        if not isinstance(sample, LocalSample):
            raise ValueError("samples must contain only LocalSample values")
        if len(sample.context) != initial.context_dim:
            raise ValueError(
                "sample context length must equal initial.context_dim"
            )
        if sample.arm >= initial.num_arms:
            raise ValueError("sample arm must be less than initial.num_arms")
        key = (
            sample.pretraining_seed,
            sample.node_id,
            sample.local_sequence,
        )
        if key in provenance:
            raise ValueError("duplicate local sample sequence provenance")
        provenance.add(key)
        validated.append(sample)

    validated.sort(
        key=lambda sample: (
            sample.pretraining_seed,
            sample.node_id,
            sample.local_sequence,
            sample.arm,
        )
    )
    trained = initial.clone()
    for sample in validated:
        trained.update(sample.arm, sample.context, sample.local_reward)
    return trained


def deploy_independent(
    initial: LinUCB, node_ids: Iterable[str]
) -> dict[str, LinUCB]:
    """Clone one isolated policy state for each validated node identifier."""
    if not isinstance(initial, LinUCB):
        raise ValueError("initial must be a LinUCB agent")
    raw_node_ids = _materialize("node_ids", node_ids)
    normalized: list[str] = []
    seen: set[str] = set()
    for value in raw_node_ids:
        node_id = _node_identifier(value)
        if node_id in seen:
            raise ValueError("node_ids must not contain duplicates")
        seen.add(node_id)
        normalized.append(node_id)

    return {node_id: initial.clone() for node_id in normalized}
