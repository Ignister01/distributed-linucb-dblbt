"""Shared domain types for DB-LBT experiments."""

from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Technology(StrEnum):
    """Technology identifiers used in experiment configuration."""

    WIFI = "wifi"
    NRU = "nru"
    LEGACY_AP = "legacy_ap"
    LEGACY_STA = "legacy_sta"


class PolicyKind(StrEnum):
    """Policy identifiers used in experiment configuration."""

    RANDOM = "random_lbt"
    PRIMARY_DB = "primary_db_lbt"
    TMC_DB = "tmc_db_lbt"
    ADAPTIVE = "adaptive_db_lbt"


class RecoveryProfile(BaseModel):
    """Immutable recovery parameters with a shared, fixed alpha."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    alpha: Literal[11] = 11
    kappa: int = Field(gt=0)
    beta: int = Field(gt=0)
    m: int = Field(gt=0)
    b_init: int = Field(gt=0)

    @field_validator("alpha", mode="before")
    @classmethod
    def validate_alpha_type(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("alpha must be the integer 11")
        return value

    @model_validator(mode="after")
    def validate_thresholds(self) -> Self:
        if self.beta >= self.kappa:
            raise ValueError("beta must be lower than kappa")
        return self
