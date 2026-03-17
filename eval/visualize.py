"""Generate evaluation report visualizations."""
from __future__ import annotations

import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def generate_report(summary: dict, output_path: str) -> str:
    """Generate a composite evaluation report image.

    Contains:
    - Per-task IoU bar chart (horizontal, color-coded by quality)
    - Accuracy at IoU thresholds bar chart
    - Summary stats text panel

    Returns the path to the saved image.
    """
    per_task = summary["per_task"]
    tasks = [t["task"] for t in per_task]
    labels = [t["label"] for t in per_task]
    ious = [t["iou"] for t in per_task]
    scores = [t["pred_score"] for t in per_task]

    display_names = [f"{task}\n({label})" for task, label in zip(tasks, labels)]

    fig, axes = plt.subplots(1, 3, figsize=(20, 8), gridspec_kw={"width_ratios": [3, 1.2, 1.2]})
    fig.suptitle("GroundingDINO Minecraft Evaluation Report", fontsize=16, fontweight="bold", y=0.98)

    _plot_iou_bars(axes[0], display_names, ious)
    _plot_accuracy_bars(axes[1], summary["accuracy"])
    _plot_summary_panel(axes[2], summary)

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def _iou_color(iou: float) -> str:
    if iou >= 0.75:
        return "#2ecc71"
    if iou >= 0.5:
        return "#f39c12"
    if iou >= 0.25:
        return "#e67e22"
    return "#e74c3c"


def _plot_iou_bars(ax, names: List[str], ious: List[float]) -> None:
    y_pos = np.arange(len(names))
    colors = [_iou_color(v) for v in ious]

    bars = ax.barh(y_pos, ious, color=colors, edgecolor="white", height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("IoU", fontsize=11)
    ax.set_title("Per-Task IoU", fontsize=13, fontweight="bold")
    ax.invert_yaxis()

    for bar, iou in zip(bars, ious):
        ax.text(
            bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
            f"{iou:.3f}", va="center", fontsize=9,
        )

    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.axvline(x=0.75, color="gray", linestyle=":", alpha=0.4, linewidth=0.8)

    legend_patches = [
        mpatches.Patch(color="#2ecc71", label="IoU ≥ 0.75"),
        mpatches.Patch(color="#f39c12", label="0.50 ≤ IoU < 0.75"),
        mpatches.Patch(color="#e67e22", label="0.25 ≤ IoU < 0.50"),
        mpatches.Patch(color="#e74c3c", label="IoU < 0.25"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8)


def _plot_accuracy_bars(ax, accuracy: dict) -> None:
    thresholds = sorted(accuracy.keys(), key=float)
    values = [accuracy[t] * 100 for t in thresholds]
    x_labels = [f"IoU≥{float(t):.2f}" for t in thresholds]

    colors = ["#3498db", "#2980b9", "#1f618d"][:len(thresholds)]
    bars = ax.bar(x_labels, values, color=colors, edgecolor="white", width=0.6)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("Accuracy @ IoU Thresholds", fontsize=13, fontweight="bold")

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
            f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold",
        )


def _plot_summary_panel(ax, summary: dict) -> None:
    ax.axis("off")
    ax.set_title("Summary", fontsize=13, fontweight="bold")

    lines = [
        ("Total Samples", f"{summary['total']}"),
        ("Detection Rate", f"{summary['detection_rate']:.1%}"),
        ("Mean IoU", f"{summary['mean_iou']:.4f}"),
        ("", ""),
    ]

    for t in sorted(summary["accuracy"].keys(), key=float):
        lines.append((f"Acc@IoU≥{float(t):.2f}", f"{summary['accuracy'][t]:.1%}"))

    ious_all = [t["iou"] for t in summary["per_task"]]
    if ious_all:
        best_idx = int(np.argmax(ious_all))
        worst_idx = int(np.argmin(ious_all))
        lines.append(("", ""))
        lines.append(("Best Task", f"{summary['per_task'][best_idx]['task']}"))
        lines.append(("  IoU", f"{ious_all[best_idx]:.3f}"))
        lines.append(("Worst Task", f"{summary['per_task'][worst_idx]['task']}"))
        lines.append(("  IoU", f"{ious_all[worst_idx]:.3f}"))

    y_start = 0.92
    for i, (key, val) in enumerate(lines):
        y = y_start - i * 0.065
        if key == "":
            continue
        ax.text(0.05, y, key + ":", fontsize=10, transform=ax.transAxes,
                fontweight="bold", color="#2c3e50")
        ax.text(0.95, y, val, fontsize=10, transform=ax.transAxes,
                ha="right", color="#34495e")
