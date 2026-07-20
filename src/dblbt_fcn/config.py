"""Configuration models and loading helpers."""

from itertools import product
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .types import RecoveryProfile


class SimulationConfig(BaseModel):
    """Validated timing and reproducibility settings for a simulation."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    rounds: int = Field(default=100_000, ge=1)
    slot_us: int = Field(default=1, ge=1)
    tx_us: int = Field(default=2_000, ge=1)
    wifi_ack_us: int = Field(default=0, ge=0)
    nru_sync_us: int = Field(default=250, ge=1)
    seed: int = Field(ge=0)


def adaptive_arms() -> list[RecoveryProfile]:
    """Return the deterministic 24-arm adaptive recovery grid."""
    return [
        RecoveryProfile(kappa=kappa, beta=beta, m=m, b_init=b_init)
        for kappa, beta, m, b_init in product(
            (5, 7), (2, 3), (4, 6, 10), (15, 31)
        )
    ]


def load_yaml(path: str | Path) -> dict[str, object]:
    """Load a UTF-8 YAML configuration whose document root is a mapping."""
    with Path(path).open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)

    if not isinstance(value, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"config root keys must be strings: {path}")
    return value
