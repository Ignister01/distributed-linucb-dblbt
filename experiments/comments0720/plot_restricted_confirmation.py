"""Plot the independent restricted-profile confirmation in IEEE dimensions."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "low": "#0072B2",
    "high": "#D55E00",
    "average": "#009E73",
    "diagnostic": "#777777",
    "arm4": "#0072B2",
    "arm20": "#E69F00",
}


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="ascii", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-effects", required=True, type=Path)
    parser.add_argument("--whole-run-effect", required=True, type=Path)
    parser.add_argument("--decisions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    phase = _rows(args.phase_effects)
    whole = _rows(args.whole_run_effect)
    decisions = _rows(args.decisions)
    if len(phase) != 3 or len(whole) != 1 or len(decisions) != 4:
        raise ValueError("restricted confirmation inputs have unexpected rows")
    keyed = {row["scenario_id"]: row for row in phase}
    ordered = [
        keyed["low-n04-p025"],
        keyed["high-n06-p045"],
        keyed["all-phases"],
        whole[0],
    ]
    labels = ["Low 4+4", "High 6+6", "Phase avg.", "Mixed-set\ndiag."]
    colors = [
        COLORS["low"],
        COLORS["high"],
        COLORS["average"],
        COLORS["diagnostic"],
    ]
    means = np.asarray([float(row["utility_difference"]) for row in ordered])
    lower = np.asarray([float(row["utility_lower_95"]) for row in ordered])
    upper = np.asarray([float(row["utility_upper_95"]) for row in ordered])

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
    figure, axes = plt.subplots(
        1, 2, figsize=(7.16, 2.45), constrained_layout=True
    )
    positions = np.arange(len(ordered))
    axes[0].bar(positions, means, width=0.64, color=colors)
    axes[0].errorbar(
        positions,
        means,
        yerr=np.vstack((means - lower, upper - means)),
        fmt="none",
        color="#222222",
        linewidth=0.8,
        capsize=2,
    )
    axes[0].axhline(0, color="#444444", linewidth=0.7)
    axes[0].grid(axis="y", color="#DDDDDD", linewidth=0.5)
    axes[0].set(
        title="Paired Adaptive - TMC utility (32 seeds)",
        ylabel="Utility difference with 95% CI",
        xticks=positions,
        xticklabels=labels,
    )
    for position, value in zip(positions, means, strict=True):
        offset = 0.00055 if value >= 0 else -0.00055
        axes[0].text(
            position,
            value + offset,
            f"{value:+.4f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=6.8,
        )
    axes[0].set_ylim(-0.0032, 0.0184)

    counts = {
        (row["phase_id"], int(row["arm_id"])): int(row["decision_count"])
        for row in decisions
    }
    phases = ("low-n04-p025", "high-n06-p045")
    arm4 = np.asarray(
        [counts[(phase_id, 4)] for phase_id in phases], dtype=np.float64
    )
    arm20 = np.asarray(
        [counts[(phase_id, 20)] for phase_id in phases], dtype=np.float64
    )
    totals = arm4 + arm20
    share4 = 100 * arm4 / totals
    share20 = 100 * arm20 / totals
    phase_positions = np.arange(2)
    axes[1].bar(
        phase_positions,
        share4,
        width=0.58,
        color=COLORS["arm4"],
        label=r"Arm 4: $(5,2,10,15)$",
    )
    axes[1].bar(
        phase_positions,
        share20,
        bottom=share4,
        width=0.58,
        color=COLORS["arm20"],
        label=r"Arm 20: $(7,3,6,15)$",
    )
    for position, first, second in zip(
        phase_positions, share4, share20, strict=True
    ):
        axes[1].text(
            position,
            first / 2,
            f"{first:.1f}%",
            ha="center",
            va="center",
            color="white" if first > 18 else "#222222",
            fontsize=7,
        )
        axes[1].text(
            position,
            101.5 if second < 5 else first + second / 2,
            f"{second:.1f}%",
            ha="center",
            va="bottom" if second < 5 else "center",
            color="#222222",
            fontsize=7,
            clip_on=False,
        )
    axes[1].set(
        title="Distributed LinUCB decisions",
        ylabel="Decision share (%)",
        xticks=phase_positions,
        xticklabels=("Low 4+4", "High 6+6"),
        ylim=(0, 100),
    )
    axes[1].set_title("Distributed LinUCB decisions", pad=27)
    axes[1].legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=2,
        columnspacing=0.9,
        handletextpad=0.4,
    )
    axes[1].grid(axis="y", color="#DDDDDD", linewidth=0.5)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight")
    figure.savefig(args.output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
