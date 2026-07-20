"""Backoff policies and state transitions from the DB-LBT papers."""

from dataclasses import dataclass
import random

from .types import RecoveryProfile


def _require_int(name: str, value: object) -> None:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")


@dataclass
class DbState:
    """Mutable collision-recovery state for a DB-LBT contender."""

    interruptions: int = 0
    retries: int = 0

    def __post_init__(self) -> None:
        _require_int("interruptions", self.interruptions)
        _require_int("retries", self.retries)
        if self.interruptions < 0:
            raise ValueError("interruptions must be non-negative")
        if self.retries < 0:
            raise ValueError("retries must be non-negative")

    def success(self) -> None:
        """Clear the retransmission counter after a success."""
        self.retries = 0

    def collision(self) -> None:
        """Record one additional collision."""
        self.retries += 1


class PrimaryDbLbt:
    """Primary-paper DB-LBT collision-recovery rule."""

    def __init__(self, alpha: int, m: int, beta: int) -> None:
        _require_int("alpha", alpha)
        _require_int("m", m)
        _require_int("beta", beta)
        if alpha != 11:
            raise ValueError("alpha must be 11")
        if m <= 0:
            raise ValueError("m must be positive")
        if not 0 < beta < m:
            raise ValueError("beta must be between zero and m")
        self.alpha = alpha
        self.m = m
        self.beta = beta

    def next_backoff(self, state: DbState, rng: random.Random) -> int:
        """Return the next backoff and update deterministic-branch state."""
        if state.retries % self.m < self.beta:
            value = self.alpha + state.interruptions
            state.interruptions = 0
            return value
        return rng.randint(0, self.m - 1)


class TmcDbLbt:
    """TMC-paper DB-LBT initial and collision-recovery rules."""

    def __init__(self, profile: RecoveryProfile) -> None:
        self.profile = profile

    def initial_backoff(self, rng: random.Random) -> int:
        """Draw the inclusive initial backoff range."""
        return rng.randint(0, self.profile.b_init)

    def next_backoff(self, state: DbState, rng: random.Random) -> int:
        """Return the next backoff and update deterministic-branch state."""
        if state.retries % self.profile.kappa < self.profile.beta:
            value = self.profile.alpha + state.interruptions
            state.interruptions = 0
            return value
        return rng.randint(0, self.profile.m)


class RandomLbt:
    """Random LBT with bounded binary exponential contention growth."""

    def __init__(self, cw_min: int, cw_max: int) -> None:
        _require_int("cw_min", cw_min)
        _require_int("cw_max", cw_max)
        if cw_min < 0:
            raise ValueError("cw_min must be non-negative")
        if cw_max < cw_min:
            raise ValueError("cw_max must be at least cw_min")
        self.cw_min = cw_min
        self.cw_max = cw_max
        self.cw = cw_min

    def draw(self, rng: random.Random) -> int:
        """Draw inclusively from zero through the current window."""
        return rng.randint(0, self.cw)

    def collision(self) -> None:
        """Grow the window after a collision, capped at the maximum."""
        self.cw = min(2 * self.cw + 1, self.cw_max)

    def success(self) -> None:
        """Reset the window after a success."""
        self.cw = self.cw_min
