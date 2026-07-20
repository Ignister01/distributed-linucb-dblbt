"""Local attempt history and normalized controller context."""

from collections import deque
from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Literal

import numpy as np

from .metrics import nearest_rank_p95


Numeric = Real
Outcome = Literal["success", "collision"]

_CONTEXT_SIZE = 11
_MIN_READY_ATTEMPTS = 8
_MAX_ATTEMPTS = 64
_DELAY_EWMA_ALPHA = 0.2
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


def _non_negative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    numeric = int(value)
    if numeric < 0:
        raise ValueError(f"{name} must be non-negative")
    return numeric


def _ratio(name: str, value: object) -> float:
    numeric = _finite_number(name, value)
    if not 0 <= numeric <= 1:
        raise ValueError(f"{name} must be between zero and one")
    return numeric


@dataclass(frozen=True, slots=True)
class AttemptRecord:
    """Immutable measurements available after one local access attempt."""

    outcome: Outcome
    elapsed_us: Numeric
    busy_us: Numeric
    interruptions: Integral
    access_delay_us: Numeric | None
    queue_occupancy_ratio: Numeric
    arrivals: Integral
    retries: Integral
    effective_data_us: Numeric

    def __post_init__(self) -> None:
        if type(self.outcome) is not str or self.outcome not in (
            "success",
            "collision",
        ):
            raise ValueError("outcome must be success or collision")
        elapsed_us = _positive_number("elapsed_us", self.elapsed_us)
        busy_us = _non_negative_number("busy_us", self.busy_us)
        if busy_us > elapsed_us:
            raise ValueError("busy_us must not exceed elapsed_us")
        interruptions = _non_negative_int(
            "interruptions", self.interruptions
        )
        if self.access_delay_us is not None:
            access_delay_us = _non_negative_number(
                "access_delay_us", self.access_delay_us
            )
        else:
            access_delay_us = None
        queue_occupancy_ratio = _ratio(
            "queue_occupancy_ratio", self.queue_occupancy_ratio
        )
        arrivals = _non_negative_int("arrivals", self.arrivals)
        retries = _non_negative_int("retries", self.retries)
        effective_data_us = _non_negative_number(
            "effective_data_us", self.effective_data_us
        )
        if effective_data_us > elapsed_us:
            raise ValueError("effective_data_us must not exceed elapsed_us")
        object.__setattr__(self, "elapsed_us", elapsed_us)
        object.__setattr__(self, "busy_us", busy_us)
        object.__setattr__(self, "interruptions", interruptions)
        object.__setattr__(self, "access_delay_us", access_delay_us)
        object.__setattr__(
            self, "queue_occupancy_ratio", queue_occupancy_ratio
        )
        object.__setattr__(self, "arrivals", arrivals)
        object.__setattr__(self, "retries", retries)
        object.__setattr__(self, "effective_data_us", effective_data_us)


class LocalWindow:
    """Retain recent local attempts and expose their normalized context."""

    def __init__(self, max_attempts: Integral = _MAX_ATTEMPTS) -> None:
        capacity = _non_negative_int("max_attempts", max_attempts)
        if not _MIN_READY_ATTEMPTS <= capacity <= _MAX_ATTEMPTS:
            raise ValueError("max_attempts must be between 8 and 64")
        self._attempts: deque[AttemptRecord] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._attempts)

    @property
    def ready(self) -> bool:
        """Return whether enough local attempts exist for controller use."""
        return len(self) >= _MIN_READY_ATTEMPTS

    @property
    def attempts(self) -> tuple[AttemptRecord, ...]:
        """Return an immutable snapshot of the current local records."""
        return tuple(self._attempts)

    def record(self, attempt: AttemptRecord) -> None:
        """Append one validated local attempt, evicting the oldest if full."""
        if not isinstance(attempt, AttemptRecord):
            raise ValueError("attempt must be an AttemptRecord")
        self._attempts.append(attempt)

    def context(self) -> np.ndarray:
        """Return the current 11-feature context in its contract order."""
        if not self._attempts:
            return np.zeros(_CONTEXT_SIZE, dtype=np.float64)

        samples = tuple(self._attempts)
        attempt_count = len(samples)
        elapsed_us = math.fsum(sample.elapsed_us for sample in samples)
        mean_interruptions = math.fsum(
            sample.interruptions for sample in samples
        ) / attempt_count
        estimated_contenders = min(
            max(1.0 + mean_interruptions, 1.0), 32.0
        )
        delay_capacity_us = (
            _BASE_DELAY_CAPACITY_US * estimated_contenders
        )
        delays_us = [
            sample.access_delay_us
            for sample in samples
            if sample.access_delay_us is not None
        ]
        delay_ewma_us = self._delay_ewma(delays_us)
        delay_p95_us = (
            float(nearest_rank_p95(delays_us)) if delays_us else 0.0
        )
        mean_retries = math.fsum(
            sample.retries for sample in samples
        ) / attempt_count

        features = np.array(
            [
                sum(
                    sample.outcome == "collision" for sample in samples
                )
                / attempt_count,
                sum(sample.outcome == "success" for sample in samples)
                / attempt_count,
                math.fsum(sample.busy_us for sample in samples)
                / elapsed_us,
                mean_interruptions / 31,
                delay_ewma_us / delay_capacity_us,
                delay_p95_us / delay_capacity_us,
                math.fsum(
                    sample.queue_occupancy_ratio for sample in samples
                )
                / attempt_count,
                (
                    math.fsum(sample.arrivals for sample in samples)
                    / (elapsed_us / 1_000)
                    / 0.1
                ),
                mean_retries / (1 + mean_retries),
                math.fsum(
                    sample.effective_data_us for sample in samples
                )
                / elapsed_us,
                (estimated_contenders - 1) / 31,
            ],
            dtype=np.float64,
        )
        return np.clip(features, 0.0, 1.0)

    @staticmethod
    def _delay_ewma(delays_us: list[float]) -> float:
        if not delays_us:
            return 0.0
        value = float(delays_us[0])
        for sample in delays_us[1:]:
            value = (
                (1 - _DELAY_EWMA_ALPHA) * value
                + _DELAY_EWMA_ALPHA * sample
            )
        return value
