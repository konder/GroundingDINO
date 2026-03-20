"""Generate evaluation report visualizations."""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image


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


def _draw_box(ax, bbox_norm: Sequence[float], img_w: int, img_h: int,
              color: str, label: str, linestyle: str = "-") -> None:
    """Draw a normalized [x1,y1,x2,y2] bbox on an axes."""
    x1, y1, x2, y2 = bbox_norm
    px1, py1 = x1 * img_w, y1 * img_h
    pw, ph = (x2 - x1) * img_w, (y2 - y1) * img_h
    rect = mpatches.FancyBboxPatch(
        (px1, py1), pw, ph,
        linewidth=2, edgecolor=color, facecolor="none",
        linestyle=linestyle,
        boxstyle="square,pad=0",
    )
    ax.add_patch(rect)
    ax.text(px1, py1 - 2, label, fontsize=6, color=color,
            fontweight="bold", va="bottom",
            bbox=dict(facecolor="black", alpha=0.5, pad=1, edgecolor="none"))


def generate_visual_grid(
    results: list,
    annotations: list,
    image_dir: str,
    output_path: str,
    cols: int = 4,
) -> str:
    """Generate a grid of test images with GT (green) and predicted (red) bboxes.

    Args:
        results: list of EvalResult from evaluator.
        annotations: list of annotation dicts (with 'image', 'bbox', 'label', 'task').
        image_dir: directory containing the test images.
        output_path: where to save the grid image.
        cols: number of columns in the grid.

    Returns:
        Path to the saved grid image.
    """
    n = len(results)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    if rows == 1:
        axes = [axes] if cols == 1 else list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    for i, (res, ann) in enumerate(zip(results, annotations)):
        ax = axes[i]
        img_path = os.path.join(image_dir, ann["image"])
        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
            img_arr = np.array(img)
            ax.imshow(img_arr)
            img_w, img_h = img.size
        else:
            ax.text(0.5, 0.5, "IMAGE\nMISSING", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14, color="red")
            ax.set_facecolor("#1a1a1a")
            img_w, img_h = 640, 360

        _draw_box(ax, ann["bbox"], img_w, img_h,
                  color="#00ff00", label=f"GT: {ann['label']}")

        if res.pred_bbox is not None:
            _draw_box(ax, res.pred_bbox, img_w, img_h,
                      color="#ff3333", label=f"Pred ({res.pred_score:.2f})",
                      linestyle="--")

        iou_color = "#2ecc71" if res.iou >= 0.5 else "#e67e22" if res.iou >= 0.25 else "#e74c3c"
        ax.set_title(f"{ann['task']}  |  IoU={res.iou:.3f}",
                     fontsize=9, fontweight="bold", color=iou_color)
        ax.axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle("GroundingDINO Detection Results  (Green=GT, Red=Pred)",
                 fontsize=14, fontweight="bold", y=1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path
