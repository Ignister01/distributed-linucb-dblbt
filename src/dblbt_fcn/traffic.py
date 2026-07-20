"""Deterministic and explicitly seeded traffic models."""

from dataclasses import dataclass
import math
import random


# Keep PTRS log-probability cancellation within a numerically stable scale.
MAX_POISSON_MEAN = 1_000_000_000.0

_PTRS_MIN_MEAN = 30.0


def _require_non_negative_int(name: str, value: object) -> None:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class ActiveWindow:
    """A half-open interval of active contention rounds."""

    start_round: int
    lifetime_rounds: int

    def __post_init__(self) -> None:
        _require_non_negative_int("start_round", self.start_round)
        _require_non_negative_int("lifetime_rounds", self.lifetime_rounds)
        if self.lifetime_rounds == 0:
            raise ValueError("lifetime_rounds must be positive")

    def active(self, round_id: int) -> bool:
        """Return whether round_id lies in this active window."""
        _require_non_negative_int("round_id", round_id)
        return self.start_round <= round_id < (
            self.start_round + self.lifetime_rounds
        )


class SaturatedTraffic:
    """A traffic policy that always has another packet available."""

    def backlogged(self, queue_depth: int) -> bool:
        """Validate queue state and report permanent backlog."""
        _require_non_negative_int("queue_depth", queue_depth)
        return True


@dataclass(frozen=True)
class PoissonTraffic:
    """Sample independent Poisson increments using an injected RNG."""

    rate_packets_per_ms: float
    rng: random.Random

    def __post_init__(self) -> None:
        rate = self.rate_packets_per_ms
        if type(rate) not in (int, float):
            raise ValueError(
                "rate_packets_per_ms must be positive and finite"
            )
        try:
            finite_rate = float(rate)
        except OverflowError as error:
            raise ValueError(
                "rate_packets_per_ms must be positive and finite"
            ) from error
        if (
            not math.isfinite(finite_rate)
            or finite_rate <= 0
            or finite_rate / 1_000.0 <= 0
        ):
            raise ValueError("rate_packets_per_ms must be positive and finite")
        if not isinstance(self.rng, random.Random):
            raise ValueError("rng must be a random.Random instance")

    def arrivals(self, elapsed_us: int) -> int:
        """Sample arrivals in one non-overlapping interval."""
        _require_non_negative_int("elapsed_us", elapsed_us)
        if elapsed_us == 0:
            return 0

        rate_per_us = self.rate_packets_per_ms / 1_000.0
        try:
            mean = rate_per_us * elapsed_us
        except OverflowError as error:
            raise ValueError("Poisson mean exceeds MAX_POISSON_MEAN") from error
        if not math.isfinite(mean) or mean > MAX_POISSON_MEAN:
            raise ValueError("Poisson mean exceeds MAX_POISSON_MEAN")
        if mean >= _PTRS_MIN_MEAN:
            return self._ptrs(mean)

        arrival_time_us = 0.0
        arrivals = 0
        while True:
            arrival_time_us += self.rng.expovariate(rate_per_us)
            if arrival_time_us > elapsed_us:
                return arrivals
            arrivals += 1

    def _ptrs(self, mean: float) -> int:
        """Sample an exact large-mean Poisson variate with Hormann PTRS."""
        sqrt_mean = math.sqrt(mean)
        log_mean = math.log(mean)
        b = 0.931 + 2.53 * sqrt_mean
        a = -0.059 + 0.02483 * b
        inverse_alpha = 1.1239 + 1.1328 / (b - 3.4)
        squeeze = 0.9277 - 3.6224 / (b - 2.0)

        while True:
            u = self.rng.random() - 0.5
            v = self.rng.random()
            distance = 0.5 - abs(u)
            if distance == 0:
                continue
            candidate = math.floor(
                (2.0 * a / distance + b) * u + mean + 0.43
            )
            if candidate < 0:
                continue
            if distance >= 0.07 and v <= squeeze:
                return candidate
            if distance < 0.013 and v > distance:
                continue
            if v == 0:
                return candidate

            log_acceptance = (
                math.log(v)
                + math.log(inverse_alpha)
                - math.log(a / (distance * distance) + b)
            )
            log_probability = (
                -mean
                + candidate * log_mean
                - math.lgamma(candidate + 1)
            )
            if log_acceptance <= log_probability:
                return candidate
