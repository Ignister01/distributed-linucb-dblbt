"""Pure metric reducers for DB-LBT experiment results."""

from collections.abc import Iterable
import math

from .channel import RoundResult


Numeric = int | float


def _require_finite_number(name: str, value: object) -> Numeric:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be an integer or float")
    if type(value) is float and not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _materialize(name: str, values: object) -> list[object]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be an iterable")
    try:
        return list(values)
    except TypeError as error:
        raise ValueError(f"{name} must be an iterable") from error


def _non_negative_values(name: str, values: object) -> list[Numeric]:
    samples = _materialize(name, values)
    if not samples:
        raise ValueError(f"{name} must not be empty")
    validated: list[Numeric] = []
    for value in samples:
        numeric = _require_finite_number(name, value)
        if numeric < 0:
            raise ValueError(f"{name} must contain only non-negative values")
        validated.append(numeric)
    return validated


def _positive_number(name: str, value: object) -> Numeric:
    numeric = _require_finite_number(name, value)
    if numeric <= 0:
        raise ValueError(f"{name} must be positive")
    return numeric


def _round_results(results: object) -> list[RoundResult]:
    values = _materialize("results", results)
    rounds: list[RoundResult] = []
    for value in values:
        if not isinstance(value, RoundResult):
            raise ValueError("results must contain only RoundResult instances")
        rounds.append(value)
    for previous, current in zip(rounds, rounds[1:]):
        if current.round_id <= previous.round_id:
            raise ValueError("round_id values must be strictly increasing")
        if current.now_us <= previous.now_us:
            raise ValueError("now_us values must be strictly increasing")
    return rounds


def _validate_effective_airtime(
    results: Iterable[RoundResult], total_elapsed_us: Numeric
) -> int:
    effective_data_us = sum(
        result.effective_data_us
        for result in results
        if result.kind == "success"
    )
    if effective_data_us > total_elapsed_us:
        raise ValueError(
            "effective airtime exceeds total_elapsed_us; inputs are inconsistent"
        )
    return effective_data_us


def jain(values: object) -> float:
    """Return Jain's fairness index for finite non-negative values."""
    samples = _non_negative_values("values", values)
    scale = max(samples)
    if scale == 0:
        return 0.0
    scaled = [value / scale for value in samples]
    total = math.fsum(scaled)
    squares = math.fsum(value * value for value in scaled)
    return total * total / (len(scaled) * squares)


def collision_probability(results: object) -> float:
    """Return the fraction of contention rounds that collided."""
    rounds = _round_results(results)
    if not rounds:
        raise ValueError("results must not be empty")
    collisions = sum(result.kind == "collision" for result in rounds)
    return collisions / len(rounds)


def normalized_effective_airtime(
    results: object, total_elapsed_us: object
) -> float:
    """Return successful effective data airtime divided by elapsed time."""
    elapsed = _positive_number("total_elapsed_us", total_elapsed_us)
    rounds = _round_results(results)
    effective_data_us = _validate_effective_airtime(rounds, elapsed)
    return effective_data_us / elapsed


def nearest_rank_percentile(
    values: object, percentile: object
) -> Numeric:
    """Return a nearest-rank percentile using a one-based ceiling rank."""
    samples = _non_negative_values("values", values)
    requested = _require_finite_number("percentile", percentile)
    if not 0 < requested <= 100:
        raise ValueError("percentile must be greater than zero and at most 100")
    rank = math.ceil(requested / 100 * len(samples))
    return sorted(samples)[rank - 1]


def nearest_rank_p95(values: object) -> Numeric:
    """Return the nearest-rank 95th percentile."""
    return nearest_rank_percentile(values, 95)


def per_node_effective_airtime(
    results: object,
    node_ids: object,
    total_elapsed_us: object,
) -> dict[str, float]:
    """Return normalized successful effective airtime for declared nodes."""
    declared_values = _materialize("node_ids", node_ids)
    if not declared_values:
        raise ValueError("node_ids must not be empty")
    if not all(
        type(node_id) is str and bool(node_id.strip())
        for node_id in declared_values
    ):
        raise ValueError("node_ids must contain only non-empty strings")
    declared = list(declared_values)
    if len(declared) != len(set(declared)):
        raise ValueError("node_ids must be unique")

    elapsed = _positive_number("total_elapsed_us", total_elapsed_us)
    rounds = _round_results(results)
    _validate_effective_airtime(rounds, elapsed)
    declared_set = set(declared)
    totals = {node_id: 0 for node_id in declared}
    for result in rounds:
        if any(node_id not in declared_set for node_id in result.node_ids):
            raise ValueError("result node_ids must all be declared")
        if result.kind == "success":
            totals[result.node_ids[0]] += result.effective_data_us
    return {
        node_id: effective_data_us / elapsed
        for node_id, effective_data_us in totals.items()
    }


def evaluation_utility(
    a_total: object,
    d95_ms: object,
    fairness: object,
    collision_probability: object,
) -> float:
    """Return the preregistered scalar evaluation utility."""
    airtime_value = _require_finite_number("a_total", a_total)
    delay_value = _require_finite_number("d95_ms", d95_ms)
    fairness_value = _require_finite_number("fairness", fairness)
    collision_value = _require_finite_number(
        "collision_probability", collision_probability
    )
    if delay_value < 0:
        raise ValueError("d95_ms must be non-negative")
    if not 0 <= fairness_value <= 1:
        raise ValueError("fairness must be between zero and one")
    if not 0 <= collision_value <= 1:
        raise ValueError(
            "collision_probability must be between zero and one"
        )

    airtime = min(max(airtime_value, 0), 1)
    delay_ratio = min(delay_value, 500) / 500
    delay_utility = 1 - delay_ratio
    return (
        (airtime + delay_utility + fairness_value) / 3
        - 0.25 * collision_value
    )
