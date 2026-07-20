"""Explicit, orchestrator-driven non-ideal channel processes."""

from dataclasses import dataclass
import math
import random


def _require_non_negative_int(name: str, value: object) -> None:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _require_positive_int(name: str, value: object) -> None:
    _require_non_negative_int(name, value)
    if value == 0:
        raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class PeriodicBusyProcess:
    """A deterministic periodic source of background busy events."""

    period_us: int
    busy_us: int
    start_us: int = 0

    def __post_init__(self) -> None:
        _require_positive_int("period_us", self.period_us)
        _require_positive_int("busy_us", self.busy_us)
        _require_non_negative_int("start_us", self.start_us)

    def busy_duration_at(self, now_us: int) -> int:
        """Return busy duration only at an exact scheduled start."""
        _require_non_negative_int("now_us", now_us)
        if now_us < self.start_us:
            return 0
        if (now_us - self.start_us) % self.period_us == 0:
            return self.busy_us
        return 0

    def next_start_at_or_after(self, now_us: int) -> int:
        """Return the first scheduled start no earlier than now_us."""
        _require_non_negative_int("now_us", now_us)
        if now_us <= self.start_us:
            return self.start_us
        elapsed = now_us - self.start_us
        periods = (elapsed + self.period_us - 1) // self.period_us
        return self.start_us + periods * self.period_us


@dataclass(frozen=True)
class InterruptionPerturbation:
    """Apply explicitly seeded Gaussian noise to interruption counts."""

    sigma: float
    rng: random.Random

    def __post_init__(self) -> None:
        sigma = self.sigma
        if type(sigma) not in (int, float):
            raise ValueError("sigma must be non-negative and finite")
        try:
            finite_sigma = float(sigma)
        except OverflowError as error:
            raise ValueError(
                "sigma must be non-negative and finite"
            ) from error
        if (
            not math.isfinite(finite_sigma)
            or finite_sigma < 0
        ):
            raise ValueError("sigma must be non-negative and finite")
        if not isinstance(self.rng, random.Random):
            raise ValueError("rng must be a random.Random instance")

    def apply(self, interruptions: int) -> int:
        """Perturb and clamp one exact interruption count."""
        _require_non_negative_int("interruptions", interruptions)
        if self.sigma == 0:
            return interruptions
        rng_state = self.rng.getstate()
        noise_sample = self.rng.gauss(0, 1.0) * self.sigma
        if not math.isfinite(noise_sample):
            self.rng.setstate(rng_state)
            raise ValueError("Gaussian noise must be finite")
        noise = round(noise_sample)
        return max(0, interruptions + noise)
