"""Generate IEEE-style comparison figures for the PoC results.

Compares two approaches on the suicidality risk classification task:
  1. End-to-end few-shot classification.
  2. Structured cue extraction + rule mapping.

Outputs two PNG figures next to this script:
  - poc_overall_metrics.png
  - poc_per_class_recall.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Paths
END_TO_END_METRICS = "end_to_end/outputs/metrics.json"
STRUCTURED_METRICS = "structured_cue_extraction/outputs/metrics.json"
END_TO_END_CM = "end_to_end/outputs/confusion_matrix.csv"
STRUCTURED_CM = "structured_cue_extraction/outputs/confusion_matrix.csv"

OUT_DIR = Path("figures")
OVERALL_PNG = OUT_DIR / "poc_overall_metrics.png"
PER_CLASS_PNG = OUT_DIR / "poc_per_class_recall.png"

# Display configuration.
APPROACHES = [
    ("End-to-end few-shot", END_TO_END_METRICS),
    ("Structured cue + rules", STRUCTURED_METRICS),
]

OVERALL_METRIC_KEYS = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "explicit_risk_recall",
    "parse_error_rate",
]
OVERALL_METRIC_LABELS = [
    "Accuracy",
    "Bal. accuracy",
    "Macro F1",
    "Explicit-risk recall",
    "Parse-error rate",
]

# Internal keys vs. report-facing display labels.
CLASS_KEYS = ["attempt", "behavior", "ideation", "indicator", "safe"]
CLASS_DISPLAY = ["attempt", "behaviour", "ideation", "indicator", "safe"]

BAR_COLORS = ("#0072B2", "#E69F00")

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def load_metrics(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def annotate_bars(ax: plt.Axes, bars, fmt: str = "{:.2f}") -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=6.5,
        )


def grouped_bar(
    ax: plt.Axes,
    categories: list[str],
    series: list[tuple[str, list[float]]],
    ylabel: str,
    ylim: tuple[float, float] = (0.0, 1.1),
) -> None:
    n_groups = len(categories)
    n_series = len(series)
    width = 0.8 / n_series
    x = np.arange(n_groups)

    for i, (label, values) in enumerate(series):
        offset = (i - (n_series - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            label=label,
            color=BAR_COLORS[i % len(BAR_COLORS)],
            edgecolor="black",
            linewidth=0.4,
        )
        annotate_bars(ax, bars)

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=n_series,
    )


def plot_overall(metrics_by_approach: list[tuple[str, dict]], out_path: Path) -> None:
    series = []
    for name, metrics in metrics_by_approach:
        values = [float(metrics.get(k, 0.0)) for k in OVERALL_METRIC_KEYS]
        series.append((name, values))

    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    grouped_bar(ax, OVERALL_METRIC_LABELS, series, ylabel="Score")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_recall(
    metrics_by_approach: list[tuple[str, dict]], out_path: Path
) -> None:
    series = []
    for name, metrics in metrics_by_approach:
        per_class = metrics.get("per_class", {})
        values = [float(per_class.get(k, {}).get("recall", 0.0)) for k in CLASS_KEYS]
        series.append((name, values))

    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    grouped_bar(ax, CLASS_DISPLAY, series, ylabel="Recall")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics_by_approach = [(name, load_metrics(path)) for name, path in APPROACHES]

    plot_overall(metrics_by_approach, OVERALL_PNG)
    plot_per_class_recall(metrics_by_approach, PER_CLASS_PNG)

    print(f"Wrote {OVERALL_PNG}")
    print(f"Wrote {PER_CLASS_PNG}")


if __name__ == "__main__":
    main()
