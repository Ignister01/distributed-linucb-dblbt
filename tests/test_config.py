"""Validated configuration invariants."""

from decimal import Decimal
from itertools import product
from pathlib import Path

import pytest
from pydantic import ValidationError

from dblbt_fcn.config import SimulationConfig, adaptive_arms, load_yaml
from dblbt_fcn.types import PolicyKind, RecoveryProfile, Technology


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_alpha_is_common_and_fixed() -> None:
    with pytest.raises(ValidationError):
        RecoveryProfile(alpha=12, kappa=7, beta=3, m=6, b_init=15)


@pytest.mark.parametrize("alpha", [True, "11", 11.0, Decimal("11")])
def test_alpha_rejects_coerced_values(alpha: object) -> None:
    with pytest.raises(ValidationError):
        RecoveryProfile(alpha=alpha, kappa=7, beta=3, m=6, b_init=15)


def test_beta_must_be_lower_than_kappa() -> None:
    with pytest.raises(ValidationError):
        RecoveryProfile(alpha=11, kappa=3, beta=3, m=6, b_init=15)


def test_action_grid_has_24_unique_arms() -> None:
    arms = adaptive_arms()
    expected = list(product((5, 7), (2, 3), (4, 6, 10), (15, 31)))
    actual = [
        (arm.kappa, arm.beta, arm.m, arm.b_init)
        for arm in arms
    ]

    assert len(arms) == 24
    assert len(set(actual)) == 24
    assert actual == expected
    assert all(arm.alpha == 11 and arm.beta < arm.kappa for arm in arms)


def test_enum_values_match_config_vocabulary() -> None:
    assert [technology.value for technology in Technology] == [
        "wifi",
        "nru",
        "legacy_ap",
        "legacy_sta",
    ]
    assert [policy.value for policy in PolicyKind] == [
        "random_lbt",
        "primary_db_lbt",
        "tmc_db_lbt",
        "adaptive_db_lbt",
    ]


def test_recovery_profile_is_immutable_and_forbids_extra_fields() -> None:
    profile = RecoveryProfile(kappa=7, beta=3, m=6, b_init=15)

    with pytest.raises(ValidationError):
        profile.kappa = 5
    with pytest.raises(ValidationError):
        RecoveryProfile(kappa=7, beta=3, m=6, b_init=15, unknown=1)


@pytest.mark.parametrize("field", ["kappa", "beta", "m", "b_init"])
def test_recovery_parameters_must_be_positive(field: str) -> None:
    values = {"kappa": 7, "beta": 3, "m": 6, "b_init": 15}
    values[field] = 0

    with pytest.raises(ValidationError):
        RecoveryProfile(**values)


@pytest.mark.parametrize(
    ("field", "valid_value"),
    [("kappa", 7), ("beta", 3), ("m", 6), ("b_init", 15)],
)
@pytest.mark.parametrize("as_string", [False, True])
def test_recovery_profile_rejects_coerced_integer_inputs(
    field: str, valid_value: int, as_string: bool
) -> None:
    values = {"alpha": 11, "kappa": 7, "beta": 3, "m": 6, "b_init": 15}
    values[field] = str(valid_value) if as_string else True

    with pytest.raises(ValidationError) as error:
        RecoveryProfile(**values)
    assert error.value.errors()[0]["loc"] == (field,)
    assert error.value.errors()[0]["type"] == "int_type"


def test_simulation_config_has_exact_defaults_and_requires_seed() -> None:
    config = SimulationConfig(seed=0)

    assert config.model_dump() == {
        "rounds": 100_000,
        "slot_us": 1,
        "tx_us": 2_000,
        "wifi_ack_us": 0,
        "nru_sync_us": 250,
        "seed": 0,
    }
    with pytest.raises(ValidationError):
        SimulationConfig()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rounds", 0),
        ("slot_us", 0),
        ("tx_us", 0),
        ("wifi_ack_us", -1),
        ("nru_sync_us", 0),
        ("seed", -1),
    ],
)
def test_simulation_config_rejects_values_below_bounds(
    field: str, value: int
) -> None:
    values = {"seed": 0, field: value}

    with pytest.raises(ValidationError) as error:
        SimulationConfig(**values)
    assert error.value.errors()[0]["loc"] == (field,)
    assert error.value.errors()[0]["type"] == "greater_than_equal"


@pytest.mark.parametrize(
    "field", ["rounds", "slot_us", "tx_us", "wifi_ack_us", "nru_sync_us", "seed"]
)
@pytest.mark.parametrize("value", [True, "1"])
def test_simulation_config_rejects_coerced_integer_inputs(
    field: str, value: bool | str
) -> None:
    values = {"seed": 0, field: value}

    with pytest.raises(ValidationError) as error:
        SimulationConfig(**values)
    assert error.value.errors()[0]["loc"] == (field,)
    assert error.value.errors()[0]["type"] == "int_type"


def test_simulation_config_is_immutable() -> None:
    config = SimulationConfig(seed=0)

    with pytest.raises(ValidationError):
        config.rounds = 1


def test_simulation_config_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        SimulationConfig(seed=0, unknown=1)


def test_load_yaml_reads_utf8_mapping(tmp_path: Path) -> None:
    path = tmp_path / "mapping.yaml"
    path.write_text("label: \u5171\u5b58\n", encoding="utf-8")

    assert load_yaml(path) == {"label": "\u5171\u5b58"}


def test_load_yaml_rejects_non_mapping_root(tmp_path: Path) -> None:
    path = tmp_path / "sequence.yaml"
    path.write_text("- one\n- two\n", encoding="utf-8")

    with pytest.raises(ValueError, match="config root must be a mapping"):
        load_yaml(path)


def test_load_yaml_rejects_non_string_root_key(tmp_path: Path) -> None:
    path = tmp_path / "non-string-key.yaml"
    path.write_text("1: value\n", encoding="utf-8")

    with pytest.raises(ValueError, match="config root keys must be strings"):
        load_yaml(path)


def test_base_simulation_mapping_validates_with_seed() -> None:
    base = load_yaml(REPOSITORY_ROOT / "configs" / "base.yaml")
    simulation = base["simulation"]
    assert isinstance(simulation, dict)

    config = SimulationConfig.model_validate({**simulation, "seed": 0})

    assert config.model_dump() == {
        "rounds": 100_000,
        "slot_us": 1,
        "tx_us": 2_000,
        "wifi_ack_us": 0,
        "nru_sync_us": 250,
        "seed": 0,
    }


def test_base_yaml_contains_exact_declared_defaults() -> None:
    assert load_yaml(REPOSITORY_ROOT / "configs" / "base.yaml") == {
        "simulation": {
            "rounds": 100_000,
            "slot_us": 1,
            "tx_us": 2_000,
            "wifi_ack_us": 0,
            "nru_sync_us": 250,
        },
        "contention_windows": {
            "ap": {"cw_min": 15, "cw_max": 63},
            "gnb": {"cw_min": 15, "cw_max": 63},
            "legacy_station": {"cw_max": 1_023},
        },
        "channel": {"sensing": "ideal", "hidden_nodes": False},
        "traffic": {"kind": "saturated"},
    }


def test_policies_yaml_contains_fixed_policies_and_exact_adaptive_grid() -> None:
    assert load_yaml(REPOSITORY_ROOT / "configs" / "policies.yaml") == {
        "policies": {
            "random_lbt": {
                "kind": "random_lbt",
                "cw_min": 15,
                "cw_max": 63,
            },
            "primary_db_lbt": {
                "kind": "primary_db_lbt",
                "alpha": 11,
                "m": 4,
                "beta": 3,
            },
            "tmc_db_lbt": {
                "kind": "tmc_db_lbt",
                "alpha": 11,
                "kappa": 7,
                "beta": 3,
                "m": 6,
                "b_init": 15,
            },
            "adaptive_db_lbt": {
                "kind": "adaptive_db_lbt",
                "alpha": 11,
                "grid": {
                    "kappa": [5, 7],
                    "beta": [2, 3],
                    "m": [4, 6, 10],
                    "b_init": [15, 31],
                },
            },
        }
    }
