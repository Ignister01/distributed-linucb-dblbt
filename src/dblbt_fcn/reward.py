"""Reward utilities computed from local controller measurements."""

from dataclasses import dataclass
import math
from numbers import Real


Numeric = Real
_BASE_DELAY_CAPACITY_US = 8 * 2_000


def _finite_number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number")
    try:
        numeric = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _non_negative_number(name: str, value: object) -> float:
    numeric = _finite_number(name, value)
    if numeric < 0:
        raise ValueError(f"{name} must be non-negative")
    return numeric


def _positive_number(name: str, value: object) -> float:
    numeric = _finite_number(name, value)
    if numeric <= 0:
        raise ValueError(f"{name} must be positive")
    return numeric


def _ratio(name: str, value: object) -> float:
    numeric = _finite_number(name, value)
    if not 0 <= numeric <= 1:
        raise ValueError(f"{name} must be between zero and one")
    return numeric


def _clip_ratio(value: float) -> float:
    return float(min(max(value, 0), 1))


@dataclass(frozen=True, slots=True)
class RewardComponents:
    """Exact utility components and their collision-penalized reward."""

    airtime_utility: float
    delay_utility: float
    share_utility: float
    reward: float


def local_reward(
    airtime_utility: Numeric,
    delay_utility: Numeric,
    share_utility: Numeric,
    collision_probability: Numeric,
    collision_weight: Numeric = 0.25,
) -> float:
    """Combine three normalized utilities and a local collision penalty."""
    airtime = _ratio("airtime_utility", airtime_utility)
    delay = _ratio("delay_utility", delay_utility)
    share = _ratio("share_utility", share_utility)
    collisions = _ratio("collision_probability", collision_probability)
    weight = _non_negative_number("collision_weight", collision_weight)
    return (
        (airtime + delay + share) / 3
        - weight * collisions
    )


def local_reward_components(
    estimated_contenders: Numeric,
    local_airtime: Numeric,
    delay_p95_us: Numeric,
    local_share: Numeric,
    collision_probability: Numeric,
    collision_weight: Numeric = 0.25,
) -> RewardComponents:
    """Derive exact reward components from validated local measurements."""
    contenders = _finite_number(
        "estimated_contenders", estimated_contenders
    )
    if not 1 <= contenders <= 32:
        raise ValueError("estimated_contenders must be between 1 and 32")
    airtime_ratio = _ratio("local_airtime", local_airtime)
    delay_us = _non_negative_number("delay_p95_us", delay_p95_us)
    share_ratio = _ratio("local_share", local_share)
    collisions = _ratio("collision_probability", collision_probability)
    weight = _non_negative_number("collision_weight", collision_weight)

    airtime_utility = _clip_ratio(contenders * airtime_ratio)
    delay_capacity_us = _BASE_DELAY_CAPACITY_US * contenders
    delay_utility = 1 - _clip_ratio(delay_us / delay_capacity_us)
    target_share = 1 / contenders
    share_utility = 1 - _clip_ratio(
        abs(share_ratio - target_share) / target_share
    )
    reward = local_reward(
        airtime_utility,
        delay_utility,
        share_utility,
        collisions,
        weight,
    )
    return RewardComponents(
        airtime_utility=airtime_utility,
        delay_utility=delay_utility,
        share_utility=share_utility,
        reward=reward,
    )


def local_reward_from_interval(
    estimated_contenders: Numeric,
    effective_data_us: Numeric,
    elapsed_us: Numeric,
    delay_p95_us: Numeric,
    local_share: Numeric,
    collision_probability: Numeric,
    collision_weight: Numeric = 0.25,
) -> RewardComponents:
    """Derive local airtime from a finite, positive measurement interval."""
    elapsed = _positive_number("elapsed_us", elapsed_us)
    effective = _non_negative_number(
        "effective_data_us", effective_data_us
    )
    if effective > elapsed:
        raise ValueError("effective_data_us must not exceed elapsed_us")
    return local_reward_components(
        estimated_contenders,
        effective / elapsed,
        delay_p95_us,
        local_share,
        collision_probability,
        collision_weight,
    )
