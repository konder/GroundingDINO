"""Evaluate fine-tuning dataset quality.

Generates:
  1. Per-category sample grids with bbox overlays
  2. Statistical summary (bbox sizes, center bias, coverage)
  3. Quality flags for suspicious annotations

Usage:
    python scripts/evaluate_data_quality.py \
        --data-dir data/processed/finetune_v1 \
        --output-dir data/processed/finetune_v1/quality_report \
        --samples-per-cat 6
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


@dataclass
class BboxStats:
    widths: List[float] = field(default_factory=list)
    heights: List[float] = field(default_factory=list)
    areas: List[float] = field(default_factory=list)
    coverage_ratios: List[float] = field(default_factory=list)
    center_offsets: List[float] = field(default_factory=list)
    center_x_offsets: List[float] = field(default_factory=list)
    center_y_offsets: List[float] = field(default_factory=list)


def load_coco(data_dir: str) -> dict:
    ann_path = os.path.join(data_dir, "annotations.json")
    with open(ann_path) as f:
        return json.load(f)


def build_lookups(coco: dict):
    id_to_img = {img["id"]: img for img in coco["images"]}
    id_to_cat = {cat["id"]: cat["name"] for cat in coco["categories"]}

    img_to_anns: Dict[int, list] = defaultdict(list)
    cat_to_anns: Dict[str, list] = defaultdict(list)

    for ann in coco["annotations"]:
        img_to_anns[ann["image_id"]].append(ann)
        cat_name = id_to_cat[ann["category_id"]]
        cat_to_anns[cat_name].append(ann)

    return id_to_img, id_to_cat, img_to_anns, cat_to_anns


def compute_bbox_stats(
    coco: dict,
    id_to_img: dict,
    id_to_cat: dict,
) -> Tuple[BboxStats, Dict[str, BboxStats]]:
    """Compute global and per-category bbox statistics."""
    global_stats = BboxStats()
    cat_stats: Dict[str, BboxStats] = defaultdict(BboxStats)

    for ann in coco["annotations"]:
        img = id_to_img[ann["image_id"]]
        iw, ih = img["width"], img["height"]
        x, y, w, h = ann["bbox"]
        cat_name = id_to_cat[ann["category_id"]]

        area_ratio = (w * h) / (iw * ih) if iw * ih > 0 else 0

        bbox_cx = x + w / 2
        bbox_cy = y + h / 2
        img_cx = iw / 2
        img_cy = ih / 2
        dx = (bbox_cx - img_cx) / iw
        dy = (bbox_cy - img_cy) / ih
        center_offset = (dx**2 + dy**2) ** 0.5

        for stats in [global_stats, cat_stats[cat_name]]:
            stats.widths.append(w / iw)
            stats.heights.append(h / ih)
            stats.areas.append(w * h)
            stats.coverage_ratios.append(area_ratio)
            stats.center_offsets.append(center_offset)
            stats.center_x_offsets.append(dx)
            stats.center_y_offsets.append(dy)

    return global_stats, cat_stats


def flag_suspicious(coco: dict, id_to_img: dict, id_to_cat: dict) -> List[dict]:
    """Flag annotations that look suspicious."""
    flags = []
    for ann in coco["annotations"]:
        img = id_to_img[ann["image_id"]]
        iw, ih = img["width"], img["height"]
        x, y, w, h = ann["bbox"]
        cat = id_to_cat[ann["category_id"]]
        problems = []

        coverage = (w * h) / (iw * ih)
        if coverage > 0.5:
            problems.append(f"very_large_bbox({coverage:.0%})")
        if w / iw > 0.8:
            problems.append(f"full_width({w/iw:.0%})")
        if h / ih > 0.8:
            problems.append(f"full_height({h/ih:.0%})")
        if w < 10 or h < 10:
            problems.append(f"tiny_bbox({w:.0f}x{h:.0f})")

        bbox_cx = x + w / 2
        bbox_cy = y + h / 2
        dist_from_center = ((bbox_cx - iw/2)**2 + (bbox_cy - ih/2)**2)**0.5
        max_dist = ((iw/2)**2 + (ih/2)**2)**0.5
        if dist_from_center / max_dist > 0.6:
            problems.append("far_from_center")

        if y < 30 and h < ih * 0.3:
            problems.append("possible_gui_top")
        if y + h > ih - 50 and h < ih * 0.3:
            problems.append("possible_gui_bottom")

        if problems:
            flags.append({
                "ann_id": ann["id"],
                "image": img["file_name"],
                "category": cat,
                "bbox": ann["bbox"],
                "problems": problems,
                "event": ann.get("event_type", ""),
            })
    return flags


def draw_bbox_on_image(
    img: np.ndarray,
    bbox: List[float],
    label: str,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw bbox and label on image (OpenCV)."""
    out = img.copy()
    x, y, w, h = [int(v) for v in bbox]
    cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)

    font_scale = 0.5
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    cv2.rectangle(out, (x, y - th - baseline - 4), (x + tw + 4, y), color, -1)
    cv2.putText(out, label, (x + 2, y - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)

    ih, iw = out.shape[:2]
    cx, cy = iw // 2, ih // 2
    cv2.drawMarker(out, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 20, 1)

    return out


def generate_category_grid(
    data_dir: str,
    output_dir: str,
    coco: dict,
    cat_to_anns: Dict[str, list],
    id_to_img: dict,
    id_to_cat: dict,
    samples_per_cat: int = 6,
):
    """Generate per-category sample grids with bbox overlays."""
    if not HAS_MPL:
        print("  matplotlib not available, skipping grid generation")
        return

    img_dir = os.path.join(data_dir, "images")
    os.makedirs(output_dir, exist_ok=True)

    for cat_name, anns in sorted(cat_to_anns.items()):
        np.random.seed(42)
        sample_anns = np.random.choice(
            anns, size=min(samples_per_cat, len(anns)), replace=False
        ).tolist()

        n = len(sample_anns)
        cols = min(3, n)
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
        if rows == 1 and cols == 1:
            axes = np.array([axes])
        axes = np.atleast_2d(axes)

        fig.suptitle(f"{cat_name} ({len(anns)} annotations)", fontsize=14, fontweight="bold")

        for idx, ann in enumerate(sample_anns):
            r, c = divmod(idx, cols)
            ax = axes[r, c]

            img_info = id_to_img[ann["image_id"]]
            img_path = os.path.join(img_dir, img_info["file_name"])

            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = draw_bbox_on_image(img, ann["bbox"], cat_name, (0, 255, 0))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "IMAGE MISSING", ha="center", va="center",
                        transform=ax.transAxes, fontsize=12)

            x, y, w, h = ann["bbox"]
            iw, ih = img_info["width"], img_info["height"]
            coverage = (w * h) / (iw * ih) * 100
            event = ann.get("event_type", "")
            ax.set_title(f"{img_info['file_name']}\n{event}\n"
                         f"bbox={int(w)}x{int(h)} cov={coverage:.1f}%",
                         fontsize=8)
            ax.axis("off")

        for idx in range(n, rows * cols):
            r, c = divmod(idx, cols)
            axes[r, c].axis("off")

        plt.tight_layout()
        out_path = os.path.join(output_dir, f"cat_{cat_name}.png")
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

    print(f"  Saved category grids to {output_dir}")


def generate_stats_report(
    output_dir: str,
    global_stats: BboxStats,
    cat_stats: Dict[str, BboxStats],
    flags: List[dict],
    coco: dict,
    id_to_cat: dict,
):
    """Generate text + visual stats report."""
    os.makedirs(output_dir, exist_ok=True)

    lines = []
    lines.append("=" * 70)
    lines.append("  FINE-TUNING DATA QUALITY REPORT")
    lines.append("=" * 70)
    lines.append("")

    total_anns = len(coco["annotations"])
    total_imgs = len(coco["images"])
    lines.append(f"Total images:       {total_imgs}")
    lines.append(f"Total annotations:  {total_anns}")
    lines.append(f"Annotations/image:  {total_anns/total_imgs:.2f}")
    lines.append("")

    cat_counts = defaultdict(int)
    for ann in coco["annotations"]:
        cat_counts[id_to_cat[ann["category_id"]]] += 1
    lines.append("Category distribution:")
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        bar = "#" * min(cnt, 50)
        lines.append(f"  {cat:20s} {cnt:4d}  {bar}")
    lines.append("")

    def _fmt_stats(s: BboxStats, label: str) -> List[str]:
        out = [f"  [{label}]"]
        if not s.widths:
            out.append("    (no data)")
            return out
        out.append(f"    Width (rel):    mean={np.mean(s.widths):.3f}  "
                   f"median={np.median(s.widths):.3f}  "
                   f"std={np.std(s.widths):.3f}")
        out.append(f"    Height (rel):   mean={np.mean(s.heights):.3f}  "
                   f"median={np.median(s.heights):.3f}  "
                   f"std={np.std(s.heights):.3f}")
        out.append(f"    Coverage:       mean={np.mean(s.coverage_ratios):.3f}  "
                   f"median={np.median(s.coverage_ratios):.3f}")
        out.append(f"    Center offset:  mean={np.mean(s.center_offsets):.3f}  "
                   f"median={np.median(s.center_offsets):.3f}")
        return out

    lines.append("Bbox statistics:")
    lines.extend(_fmt_stats(global_stats, "GLOBAL"))
    lines.append("")
    for cat_name in sorted(cat_stats):
        lines.extend(_fmt_stats(cat_stats[cat_name], cat_name))
    lines.append("")

    suspicious_rate = len(flags) / total_anns * 100 if total_anns else 0
    lines.append(f"Suspicious annotations: {len(flags)}/{total_anns} ({suspicious_rate:.1f}%)")
    if flags:
        problem_counts = defaultdict(int)
        for f in flags:
            for p in f["problems"]:
                problem_name = p.split("(")[0]
                problem_counts[problem_name] += 1
        lines.append("  Problem breakdown:")
        for prob, cnt in sorted(problem_counts.items(), key=lambda x: -x[1]):
            lines.append(f"    {prob:25s} {cnt:4d}")
    lines.append("")

    center_bias = np.mean(global_stats.center_offsets)
    large_coverage = sum(1 for r in global_stats.coverage_ratios if r > 0.3)
    lines.append("Quality indicators:")
    if center_bias < 0.15:
        lines.append(f"  [GOOD] Bboxes are well-centered (mean offset={center_bias:.3f})")
    elif center_bias < 0.25:
        lines.append(f"  [OK]   Moderate center bias (mean offset={center_bias:.3f})")
    else:
        lines.append(f"  [WARN] Bboxes far from center (mean offset={center_bias:.3f})")

    if large_coverage / total_anns < 0.1:
        lines.append(f"  [GOOD] Few oversized bboxes ({large_coverage}/{total_anns})")
    else:
        lines.append(f"  [WARN] Many oversized bboxes ({large_coverage}/{total_anns})")

    if suspicious_rate < 10:
        lines.append(f"  [GOOD] Low suspicious rate ({suspicious_rate:.1f}%)")
    elif suspicious_rate < 25:
        lines.append(f"  [OK]   Moderate suspicious rate ({suspicious_rate:.1f}%)")
    else:
        lines.append(f"  [WARN] High suspicious rate ({suspicious_rate:.1f}%)")

    lines.append("")
    lines.append("=" * 70)

    report_text = "\n".join(lines)
    print(report_text)

    with open(os.path.join(output_dir, "quality_report.txt"), "w") as f:
        f.write(report_text)

    if flags:
        with open(os.path.join(output_dir, "suspicious_annotations.json"), "w") as f:
            json.dump(flags, f, indent=2)

    if HAS_MPL:
        _plot_stats(output_dir, global_stats, cat_stats, cat_counts)


def _plot_stats(
    output_dir: str,
    global_stats: BboxStats,
    cat_stats: Dict[str, BboxStats],
    cat_counts: Dict[str, int],
):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Dataset Quality Statistics", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    ax.hist(global_stats.coverage_ratios, bins=30, color="steelblue", edgecolor="white")
    ax.axvline(np.median(global_stats.coverage_ratios), color="red", ls="--", label="median")
    ax.set_xlabel("Coverage Ratio (bbox area / image area)")
    ax.set_ylabel("Count")
    ax.set_title("Bbox Coverage Distribution")
    ax.legend()

    ax = axes[0, 1]
    ax.hist(global_stats.center_offsets, bins=30, color="coral", edgecolor="white")
    ax.axvline(np.median(global_stats.center_offsets), color="red", ls="--", label="median")
    ax.set_xlabel("Center Offset (normalized)")
    ax.set_ylabel("Count")
    ax.set_title("Bbox Center Offset Distribution")
    ax.legend()

    ax = axes[0, 2]
    ax.scatter(global_stats.center_x_offsets, global_stats.center_y_offsets,
               alpha=0.3, s=10, c="steelblue")
    ax.axhline(0, color="gray", ls="--", lw=0.5)
    ax.axvline(0, color="gray", ls="--", lw=0.5)
    circle = plt.Circle((0, 0), 0.15, fill=False, color="red", ls="--")
    ax.add_patch(circle)
    ax.set_xlabel("X offset from center")
    ax.set_ylabel("Y offset from center")
    ax.set_title("Bbox Center Positions")
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.6, 0.6)
    ax.set_aspect("equal")

    ax = axes[1, 0]
    cats_sorted = sorted(cat_counts.items(), key=lambda x: -x[1])
    names = [c[0] for c in cats_sorted]
    counts = [c[1] for c in cats_sorted]
    bars = ax.barh(names, counts, color="steelblue")
    ax.set_xlabel("Annotation Count")
    ax.set_title("Category Distribution")
    ax.invert_yaxis()
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                str(cnt), va="center", fontsize=8)

    ax = axes[1, 1]
    cat_coverages = {}
    for cat_name, stats in sorted(cat_stats.items()):
        cat_coverages[cat_name] = np.mean(stats.coverage_ratios)
    sorted_cats = sorted(cat_coverages.items(), key=lambda x: -x[1])
    ax.barh([c[0] for c in sorted_cats], [c[1] for c in sorted_cats], color="coral")
    ax.set_xlabel("Mean Coverage Ratio")
    ax.set_title("Mean Bbox Coverage per Category")
    ax.invert_yaxis()

    ax = axes[1, 2]
    cat_centers = {}
    for cat_name, stats in sorted(cat_stats.items()):
        cat_centers[cat_name] = np.mean(stats.center_offsets)
    sorted_cats = sorted(cat_centers.items(), key=lambda x: -x[1])
    colors = ["coral" if v > 0.2 else "steelblue" for _, v in sorted_cats]
    ax.barh([c[0] for c in sorted_cats], [c[1] for c in sorted_cats], color=colors)
    ax.axvline(0.15, color="green", ls="--", alpha=0.5, label="good threshold")
    ax.set_xlabel("Mean Center Offset")
    ax.set_title("Mean Center Offset per Category")
    ax.invert_yaxis()
    ax.legend()

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "stats_overview.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved stats overview to {output_dir}/stats_overview.png")


def generate_flagged_samples(
    data_dir: str,
    output_dir: str,
    flags: List[dict],
    id_to_img: dict,
    max_samples: int = 20,
):
    """Visualize the most suspicious annotations."""
    if not HAS_MPL or not flags:
        return

    img_dir = os.path.join(data_dir, "images")
    flag_dir = os.path.join(output_dir, "flagged")
    os.makedirs(flag_dir, exist_ok=True)

    for i, f in enumerate(flags[:max_samples]):
        img_path = os.path.join(img_dir, f["image"])
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = draw_bbox_on_image(img, f["bbox"], f["category"], (0, 0, 255))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{f['image']} | {f['category']}\n"
                     f"Problems: {', '.join(f['problems'])}\n"
                     f"Event: {f['event']}",
                     fontsize=9)
        ax.axis("off")
        fig.savefig(os.path.join(flag_dir, f"flag_{i:03d}_{f['category']}.png"),
                    dpi=100, bbox_inches="tight")
        plt.close(fig)

    print(f"  Saved {min(len(flags), max_samples)} flagged samples to {flag_dir}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuning data quality")
    parser.add_argument("--data-dir", default="data/processed/finetune_v1",
                        help="Dataset directory with annotations.json and images/")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for report (default: <data-dir>/quality_report)")
    parser.add_argument("--samples-per-cat", type=int, default=6,
                        help="Samples per category in grid visualization")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.data_dir, "quality_report")

    print("Loading annotations ...")
    coco = load_coco(args.data_dir)
    id_to_img, id_to_cat, img_to_anns, cat_to_anns = build_lookups(coco)

    img_dir = os.path.join(args.data_dir, "images")
    exist_count = sum(
        1 for img in coco["images"]
        if os.path.exists(os.path.join(img_dir, img["file_name"]))
    )
    print(f"  {exist_count}/{len(coco['images'])} images found on disk")

    print("Computing statistics ...")
    global_stats, cat_stats = compute_bbox_stats(coco, id_to_img, id_to_cat)

    print("Flagging suspicious annotations ...")
    flags = flag_suspicious(coco, id_to_img, id_to_cat)

    print("Generating report ...")
    cat_counts = defaultdict(int)
    for ann in coco["annotations"]:
        cat_counts[id_to_cat[ann["category_id"]]] += 1
    generate_stats_report(args.output_dir, global_stats, cat_stats, flags, coco, id_to_cat)

    print("Generating category sample grids ...")
    generate_category_grid(
        args.data_dir, args.output_dir, coco,
        cat_to_anns, id_to_img, id_to_cat,
        samples_per_cat=args.samples_per_cat,
    )

    print("Generating flagged sample visualizations ...")
    generate_flagged_samples(args.data_dir, args.output_dir, flags, id_to_img)

    print(f"\nDone! Full report at {args.output_dir}/")


if __name__ == "__main__":
    main()
