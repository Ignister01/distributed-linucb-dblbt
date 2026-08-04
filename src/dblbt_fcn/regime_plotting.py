"""Compact figures for the LinUCB regime-discovery experiment."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import re
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np

from .regime import EFFECT_FIELDS, ScenarioEffect


_BLUE = "#0072B2"
_ORANGE = "#D55E00"
_GREEN = "#009E73"
_PURPLE = "#CC79A7"
_YELLOW = "#E69F00"
_GRAY = "#666666"


def _load_effects(path: str | Path) -> tuple[ScenarioEffect, ...]:
    source = Path(path)
    with source.open(encoding="ascii", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(EFFECT_FIELDS):
            raise ValueError("effect CSV schema is invalid")
        effects: list[ScenarioEffect] = []
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("effect CSV row schema is invalid")
            relative = row["relative_difference"]
            effect = ScenarioEffect(
                scenario_id=row["scenario_id"],
                family=row["family"],
                seed_count=int(row["seed_count"]),
                baseline_mean=float(row["baseline_mean"]),
                candidate_mean=float(row["candidate_mean"]),
                utility_difference=float(row["utility_difference"]),
                relative_difference=(None if relative == "" else float(relative)),
                lower_95=float(row["lower_95"]),
                upper_95=float(row["upper_95"]),
                positive_seeds=int(row["positive_seeds"]),
                collision_difference=float(row["collision_difference"]),
                effective_airtime_difference=float(
                    row["effective_airtime_difference"]
                ),
                p95_delay_difference=float(row["p95_delay_difference"]),
                fairness_difference=float(row["fairness_difference"]),
            )
            numeric = [
                effect.baseline_mean,
                effect.candidate_mean,
                effect.utility_difference,
                effect.lower_95,
                effect.upper_95,
                effect.collision_difference,
                effect.effective_airtime_difference,
                effect.p95_delay_difference,
                effect.fairness_difference,
            ]
            if effect.relative_difference is not None:
                numeric.append(effect.relative_difference)
            if not all(math.isfinite(value) for value in numeric):
                raise ValueError("effect CSV values must be finite")
            effects.append(effect)
    if not effects:
        raise ValueError("effect CSV contains no rows")
    return tuple(effects)


def _reference_lines(axis: plt.Axes) -> None:
    axis.axhline(0.0, color=_GRAY, linewidth=0.8)
    axis.axhline(
        0.005,
        color=_GRAY,
        linewidth=0.8,
        linestyle="--",
    )
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.5)


def _effect_label(value: float) -> str:
    rounded = round(value, 3)
    return f"{0.0 if rounded == 0 else rounded:.3f}"


def _density_load_pilot_figure(
    effects: Sequence[ScenarioEffect],
) -> plt.Figure:
    patterns = {
        "load": re.compile(r"load-n(\d+)-p(\d+)\Z"),
        "turnover": re.compile(
            r"turnover-n(\d+)-p(\d+)-j10-l200\Z"
        ),
        "combined": re.compile(r"combined-n(\d+)-p(\d+)\Z"),
    }
    titles = {
        "load": "Static load",
        "turnover": "Active-set turnover",
        "combined": "Combined dynamics",
    }
    points: dict[str, list[tuple[int, float, ScenarioEffect]]] = {
        family: [] for family in patterns
    }
    for effect in effects:
        pattern = patterns.get(effect.family)
        if pattern is None:
            continue
        match = pattern.fullmatch(effect.scenario_id)
        if match is not None:
            points[effect.family].append(
                (int(match.group(1)), int(match.group(2)) / 1_000, effect)
            )
    if any(not family_points for family_points in points.values()):
        raise ValueError("pilot effects contain an incomplete density-load search")

    values = [
        effect.utility_difference
        for family_points in points.values()
        for _, _, effect in family_points
    ]
    minimum = min(values)
    maximum = max(values)
    if minimum < 0 < maximum:
        norm: Normalize = TwoSlopeNorm(vmin=minimum, vcenter=0, vmax=maximum)
        color_map = "RdYlBu"
    else:
        norm = Normalize(vmin=min(0, minimum), vmax=maximum)
        color_map = "YlGnBu"

    figure, axes = plt.subplots(
        1, 3, figsize=(7.2, 2.55), constrained_layout=True
    )
    image = None
    for axis, family in zip(axes, patterns, strict=True):
        family_points = points[family]
        nodes = sorted({node_count for node_count, _, _ in family_points})
        rates = sorted({rate for _, rate, _ in family_points})
        node_index = {node_count: index for index, node_count in enumerate(nodes)}
        rate_index = {rate: index for index, rate in enumerate(rates)}
        grid = np.full((len(nodes), len(rates)), np.nan)
        for node_count, rate, effect in family_points:
            grid[node_index[node_count], rate_index[rate]] = (
                effect.utility_difference
            )
        image = axis.imshow(
            np.ma.masked_invalid(grid),
            aspect="auto",
            origin="lower",
            cmap=color_map,
            norm=norm,
        )
        axis.set(
            title=titles[family],
            xlabel="Poisson rate (packets/ms/node)",
            ylabel="Nodes per technology",
            xticks=np.arange(len(rates)),
            xticklabels=[f"{rate:.3f}" for rate in rates],
            yticks=np.arange(len(nodes)),
            yticklabels=nodes,
        )
        axis.tick_params(axis="x", rotation=35)
        threshold = norm.vmin + 0.58 * (norm.vmax - norm.vmin)
        for row, column in np.ndindex(grid.shape):
            value = grid[row, column]
            if np.isfinite(value):
                axis.text(
                    column,
                    row,
                    _effect_label(float(value)),
                    ha="center",
                    va="center",
                    color="white" if value >= threshold else "#222222",
                    fontsize=6.5,
                )
    if image is None:
        raise RuntimeError("density-load figure has no image")
    color_bar = figure.colorbar(image, ax=axes, shrink=0.82, pad=0.02)
    color_bar.set_label("Adaptive - fixed TMC utility")
    return figure


def _pilot_figure(effects: Sequence[ScenarioEffect]) -> plt.Figure:
    if any(effect.scenario_id.startswith("load-n") for effect in effects):
        return _density_load_pilot_figure(effects)

    figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.45), constrained_layout=True)

    load_pattern = re.compile(r"load-p(\d+)\Z")
    load_rows = []
    for effect in effects:
        match = load_pattern.fullmatch(effect.scenario_id)
        if match is not None:
            load_rows.append((int(match.group(1)) / 1_000, effect))
    load_rows.sort()
    if not load_rows:
        raise ValueError("pilot effects contain no load sweep")
    axes[0].plot(
        [rate for rate, _ in load_rows],
        [effect.utility_difference for _, effect in load_rows],
        color=_BLUE,
        marker="o",
        markersize=3.5,
        linewidth=1.2,
    )
    _reference_lines(axes[0])
    axes[0].set(
        title="Offered load",
        xlabel="Poisson rate (packets/ms/node)",
        ylabel="Adaptive - fixed TMC utility",
    )
    axes[0].annotate(
        "Material threshold",
        xy=(load_rows[-1][0], 0.005),
        xytext=(0, 3),
        textcoords="offset points",
        ha="right",
        va="bottom",
        color=_GRAY,
    )

    occupancy_pattern = re.compile(r"occupancy-i(\d+)-d(\d+)\Z")
    occupancy: dict[int, list[tuple[float, ScenarioEffect]]] = {}
    for effect in effects:
        match = occupancy_pattern.fullmatch(effect.scenario_id)
        if match is None:
            continue
        interval_ms = int(match.group(1))
        duration_us = int(match.group(2))
        duty_percent = 100 * duration_us / (interval_ms * 1_000)
        occupancy.setdefault(interval_ms, []).append((duty_percent, effect))
    if not occupancy:
        raise ValueError("pilot effects contain no occupancy sweep")
    colors = (_BLUE, _ORANGE, _GREEN, _PURPLE)
    for color, interval_ms in zip(colors, sorted(occupancy), strict=False):
        points = sorted(occupancy[interval_ms])
        axes[1].plot(
            [duty for duty, _ in points],
            [effect.utility_difference for _, effect in points],
            color=color,
            marker="o",
            markersize=3.5,
            linewidth=1.0,
            label=f"{interval_ms} ms period",
        )
    _reference_lines(axes[1])
    axes[1].set(
        title="Periodic channel occupancy",
        xlabel="External busy duty cycle (%)",
        ylabel="Adaptive - fixed TMC utility",
    )
    axes[1].legend(frameon=False, ncol=1, handlelength=1.5)

    combined = sorted(
        (effect for effect in effects if effect.family == "combined"),
        key=lambda effect: effect.scenario_id,
    )
    if not combined:
        raise ValueError("pilot effects contain no combined sweep")
    positions = np.arange(len(combined))
    axes[2].bar(
        positions,
        [effect.utility_difference for effect in combined],
        color=[
            _BLUE if effect.utility_difference >= 0 else _ORANGE
            for effect in combined
        ],
        width=0.68,
    )
    _reference_lines(axes[2])
    axes[2].set(
        title="Combined dynamics",
        xlabel="Scenario",
        ylabel="Adaptive - fixed TMC utility",
        xticks=positions,
        xticklabels=[effect.scenario_id.removeprefix("combined-") for effect in combined],
    )
    return figure


def _confirmation_figure(effects: Sequence[ScenarioEffect]) -> plt.Figure:
    ordered = sorted(effects, key=lambda effect: effect.utility_difference)
    labels = [_confirmation_label(effect.scenario_id) for effect in ordered]
    positions = np.arange(len(ordered))
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), constrained_layout=True)

    gains = np.array([effect.utility_difference for effect in ordered])
    lower = np.array([effect.lower_95 for effect in ordered])
    upper = np.array([effect.upper_95 for effect in ordered])
    axes[0].barh(positions, gains, color=_BLUE, height=0.62)
    axes[0].errorbar(
        gains,
        positions,
        xerr=np.vstack((gains - lower, upper - gains)),
        color="#222222",
        fmt="none",
        capsize=2,
        linewidth=0.8,
    )
    axes[0].axvline(0.005, color=_GRAY, linewidth=0.8, linestyle="--")
    axes[0].grid(axis="x", color="#D9D9D9", linewidth=0.5)
    axes[0].set(
        title="Independent confirmation (10 seeds)",
        xlabel="Utility gain with paired 95% CI",
        yticks=positions,
        yticklabels=labels,
    )
    gain_limit = max(float(np.max(upper)), float(np.max(gains)), 0.001)
    axes[0].set_xlim(
        left=min(0.0, float(np.min(lower)) * 1.05),
        right=gain_limit * 1.35,
    )
    for position, value in zip(positions, gains, strict=True):
        axes[0].text(
            value + gain_limit * 0.050,
            position,
            _effect_label(float(value)),
            va="center",
        )

    delay_reductions = np.array(
        [-effect.p95_delay_difference / 1_000 for effect in ordered]
    )
    if float(np.max(delay_reductions)) >= 10:
        reductions = delay_reductions
        mechanism_title = "Latency mechanism"
        mechanism_label = "P95-delay reduction (ms)"
        mechanism_precision = 1
    else:
        reductions = np.array(
            [-effect.collision_difference for effect in ordered]
        )
        mechanism_title = "Collision mechanism"
        mechanism_label = "Collision-probability reduction"
        mechanism_precision = 3
    axes[1].barh(positions, reductions, color=_GREEN, height=0.62)
    axes[1].grid(axis="x", color="#D9D9D9", linewidth=0.5)
    axes[1].set(
        title=mechanism_title,
        xlabel=mechanism_label,
        yticks=positions,
        yticklabels=labels,
    )
    mechanism_limit = max(float(np.max(reductions)), 0.001)
    axes[1].set_xlim(
        left=min(0.0, float(np.min(reductions)) * 1.05),
        right=mechanism_limit * 1.30,
    )
    for position, value in zip(positions, reductions, strict=True):
        axes[1].text(
            value + mechanism_limit * 0.025,
            position,
            f"{value:.{mechanism_precision}f}",
            va="center",
        )
    return figure


def _save_pair(figure: plt.Figure, stem: Path) -> tuple[Path, Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    pdf = stem.with_suffix(".pdf")
    png = stem.with_suffix(".png")
    figure.savefig(pdf, bbox_inches="tight")
    figure.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return pdf, png


def _confirmation_label(scenario_id: str) -> str:
    patterns = (
        (re.compile(r"load-n(\d+)-p(\d+)\Z"), "Static load"),
        (
            re.compile(r"turnover-n(\d+)-p(\d+)-j10-l200\Z"),
            "Turnover",
        ),
        (re.compile(r"combined-n(\d+)-p(\d+)\Z"), "Combined"),
    )
    for pattern, name in patterns:
        match = pattern.fullmatch(scenario_id)
        if match is not None:
            nodes = int(match.group(1))
            rate = int(match.group(2)) / 1_000
            return f"{name} ({nodes}+{nodes}, rate {rate:.3f})"
    return scenario_id


def generate_regime_figures(
    pilot_effects: str | Path,
    confirmation_effects: str | Path,
    output_dir: str | Path,
) -> tuple[Path, ...]:
    """Generate the pilot boundary and independent-confirmation figures."""
    pilot = _load_effects(pilot_effects)
    confirmation = _load_effects(confirmation_effects)
    root = Path(output_dir)
    return (
        *_save_pair(_pilot_figure(pilot), root / "pilot-regime-map"),
        *_save_pair(_confirmation_figure(confirmation), root / "confirmed-gains"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-effects", type=Path, required=True)
    parser.add_argument("--confirmation-effects", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    for output in generate_regime_figures(
        arguments.pilot_effects,
        arguments.confirmation_effects,
        arguments.output_dir,
    ):
        print(output)


if __name__ == "__main__":
    main()
