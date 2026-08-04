"""Paired analysis and confirmation setup for LinUCB regime discovery."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import io
import math
from numbers import Real
import os
from pathlib import Path
import tempfile
from typing import Iterable, Sequence

import numpy as np
import yaml

from .experiment import MatrixSpec


BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260722
CONFIRMATION_SEEDS = (
    4001,
    4003,
    4007,
    4013,
    4019,
    4021,
    4027,
    4049,
    4051,
    4057,
)
FACTOR_FAMILIES = frozenset(
    {"load", "occupancy", "turnover", "sensing", "combined"}
)


@dataclass(frozen=True, slots=True)
class ScenarioEffect:
    scenario_id: str
    family: str
    seed_count: int
    baseline_mean: float
    candidate_mean: float
    utility_difference: float
    relative_difference: float | None
    lower_95: float
    upper_95: float
    positive_seeds: int
    collision_difference: float
    effective_airtime_difference: float
    p95_delay_difference: float
    fairness_difference: float


EFFECT_FIELDS = tuple(ScenarioEffect.__dataclass_fields__)
_METRICS = (
    "evaluation_utility",
    "collision_probability",
    "effective_airtime",
    "p95_delay_us",
    "jain_fairness",
)


def _finite(row: dict[str, object], field: str) -> float:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be finite")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{field} must be finite")
    return numeric


def _policy_rows(
    rows: Iterable[dict[str, object]],
    policy: str,
) -> dict[str, dict[int, dict[str, object]]]:
    grouped: dict[str, dict[int, dict[str, object]]] = {}
    for row in rows:
        if row.get("policy") != policy:
            continue
        if row.get("ablation") is not None or row.get("arm_id") is not None:
            raise ValueError("regime analysis requires unablated policy rows")
        scenario = row.get("scenario_id")
        seed = row.get("seed")
        if type(scenario) is not str or not scenario:
            raise ValueError("scenario_id must be nonempty")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be an exact nonnegative integer")
        by_seed = grouped.setdefault(scenario, {})
        if seed in by_seed:
            raise ValueError(f"duplicate {policy} seed for {scenario}: {seed}")
        for metric in _METRICS:
            _finite(row, metric)
        by_seed[seed] = row
    return grouped


def _mean_difference(
    baseline: dict[int, dict[str, object]],
    candidate: dict[int, dict[str, object]],
    metric: str,
) -> float:
    seeds = sorted(baseline)
    return float(
        np.mean(
            [
                _finite(candidate[seed], metric) - _finite(baseline[seed], metric)
                for seed in seeds
            ],
            dtype=np.float64,
        )
    )


def scenario_effects(
    rows: Sequence[dict[str, object]],
    *,
    baseline_policy: str = "tmc_db_lbt",
    candidate_policy: str = "adaptive_db_lbt",
    resamples: int = BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> tuple[ScenarioEffect, ...]:
    """Compute candidate-minus-baseline effects using exact paired seeds."""
    if type(resamples) is not int or resamples < 1:
        raise ValueError("resamples must be a positive exact integer")
    if type(bootstrap_seed) is not int or bootstrap_seed < 0:
        raise ValueError("bootstrap_seed must be a nonnegative exact integer")
    baseline = _policy_rows(rows, baseline_policy)
    candidate = _policy_rows(rows, candidate_policy)
    if not baseline or not candidate:
        raise ValueError("regime analysis requires both comparison policies")
    if set(baseline) != set(candidate):
        raise ValueError("comparison policies must cover the same scenarios")

    effects: list[ScenarioEffect] = []
    for scenario in sorted(baseline):
        baseline_rows = baseline[scenario]
        candidate_rows = candidate[scenario]
        if set(baseline_rows) != set(candidate_rows):
            raise ValueError(f"paired seeds do not match for {scenario}")
        seeds = tuple(sorted(baseline_rows))
        if not seeds:
            raise ValueError(f"paired seeds are empty for {scenario}")
        baseline_values = np.array(
            [
                _finite(baseline_rows[seed], "evaluation_utility")
                for seed in seeds
            ],
            dtype=np.float64,
        )
        candidate_values = np.array(
            [
                _finite(candidate_rows[seed], "evaluation_utility")
                for seed in seeds
            ],
            dtype=np.float64,
        )
        differences = candidate_values - baseline_values
        generator = np.random.default_rng(bootstrap_seed)
        indices = generator.integers(
            0, len(seeds), size=(resamples, len(seeds)), endpoint=False
        )
        bootstrap_means = differences[indices].mean(axis=1)
        lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
        baseline_mean = float(baseline_values.mean())
        candidate_mean = float(candidate_values.mean())
        difference = float(differences.mean())
        effects.append(
            ScenarioEffect(
                scenario_id=scenario,
                family=scenario.split("-", 1)[0],
                seed_count=len(seeds),
                baseline_mean=baseline_mean,
                candidate_mean=candidate_mean,
                utility_difference=difference,
                relative_difference=(
                    None if baseline_mean == 0 else difference / baseline_mean
                ),
                lower_95=float(lower),
                upper_95=float(upper),
                positive_seeds=int(np.count_nonzero(differences > 0)),
                collision_difference=_mean_difference(
                    baseline_rows, candidate_rows, "collision_probability"
                ),
                effective_airtime_difference=_mean_difference(
                    baseline_rows, candidate_rows, "effective_airtime"
                ),
                p95_delay_difference=_mean_difference(
                    baseline_rows, candidate_rows, "p95_delay_us"
                ),
                fairness_difference=_mean_difference(
                    baseline_rows, candidate_rows, "jain_fairness"
                ),
            )
        )
    return tuple(effects)


def select_confirmation_scenarios(
    effects: Sequence[ScenarioEffect],
) -> tuple[str, ...]:
    """Select at most one preregistered pilot candidate per factor family."""
    selected: dict[str, ScenarioEffect] = {}
    for effect in effects:
        if effect.family not in FACTOR_FAMILIES:
            continue
        if not (
            effect.positive_seeds == effect.seed_count
            and effect.utility_difference >= 0.002
            and effect.fairness_difference >= -0.01 - 1e-12
        ):
            continue
        current = selected.get(effect.family)
        if current is None or (
            effect.utility_difference,
            effect.scenario_id,
        ) > (
            current.utility_difference,
            current.scenario_id,
        ):
            selected[effect.family] = effect
    return tuple(selected[family].scenario_id for family in sorted(selected))


def confirmation_matrix(
    pilot: MatrixSpec,
    selected_scenarios: Sequence[str],
    *,
    name: str = "linucb-regime-confirmation",
    rounds: int = 100_000,
    seeds: Sequence[int] = CONFIRMATION_SEEDS,
) -> MatrixSpec:
    """Build the independent full-length confirmation matrix."""
    confirmation_seeds = tuple(seeds)
    if not confirmation_seeds:
        raise ValueError("confirmation seeds must be nonempty")
    if any(type(seed) is not int or seed < 0 for seed in confirmation_seeds):
        raise ValueError("confirmation seeds must be nonnegative exact integers")
    if len(confirmation_seeds) != len(set(confirmation_seeds)):
        raise ValueError("confirmation seeds must be unique")
    if type(rounds) is not int or rounds < 1:
        raise ValueError("confirmation rounds must be a positive exact integer")
    if not selected_scenarios:
        raise ValueError("confirmation selection must be nonempty")
    if len(selected_scenarios) != len(set(selected_scenarios)):
        raise ValueError("confirmation selection contains duplicate scenarios")
    families = [scenario.split("-", 1)[0] for scenario in selected_scenarios]
    if len(families) != len(set(families)):
        raise ValueError("confirmation selection contains duplicate families")
    available = {scenario.id: scenario for scenario in pilot.scenarios}
    missing = [scenario for scenario in selected_scenarios if scenario not in available]
    if missing:
        raise ValueError(
            "selected scenarios not present in pilot matrix: " + ", ".join(missing)
        )
    return MatrixSpec.model_validate(
        {
            "version": 1,
            "name": name,
            "rounds": rounds,
            "alpha": pilot.alpha,
            "timing": pilot.timing.model_dump(mode="python"),
            "seeds": list(confirmation_seeds),
            "policies": [
                "primary_db_lbt",
                "tmc_db_lbt",
                "adaptive_db_lbt",
            ],
            "conditions": [],
            "arm_ids": [],
            "scenarios": [
                available[scenario].model_dump(mode="python")
                for scenario in selected_scenarios
            ],
        }
    )


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{path.name}.",
            suffix=".partial",
            dir=path.parent,
            delete=False,
        ) as destination:
            temporary = Path(destination.name)
            destination.write(payload)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def write_effects_csv(
    effects: Sequence[ScenarioEffect], path: str | Path
) -> None:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=EFFECT_FIELDS,
        lineterminator="\n",
        extrasaction="raise",
    )
    writer.writeheader()
    writer.writerows(asdict(effect) for effect in effects)
    _atomic_write(Path(path), output.getvalue().encode("ascii"))


def write_selected_scenarios(
    scenarios: Sequence[str], path: str | Path
) -> None:
    if len(scenarios) != len(set(scenarios)):
        raise ValueError("selected scenarios contain duplicates")
    if any(type(scenario) is not str or not scenario for scenario in scenarios):
        raise ValueError("selected scenario ids must be nonempty strings")
    payload = "".join(f"{scenario}\n" for scenario in scenarios).encode("ascii")
    _atomic_write(Path(path), payload)


def load_selected_scenarios(path: str | Path) -> tuple[str, ...]:
    source = Path(path)
    if not source.is_file():
        raise ValueError("selection input must be an existing regular file")
    lines = source.read_text(encoding="ascii").splitlines()
    if not lines or any(not line or line != line.strip() for line in lines):
        raise ValueError("selection must contain trimmed nonempty scenario ids")
    if len(lines) != len(set(lines)):
        raise ValueError("selection contains duplicate scenarios")
    return tuple(lines)


def write_confirmation_matrix(matrix: MatrixSpec, path: str | Path) -> None:
    payload = yaml.safe_dump(
        matrix.model_dump(mode="json"),
        sort_keys=False,
        allow_unicode=False,
        default_flow_style=False,
    ).encode("ascii")
    _atomic_write(Path(path), payload)
