#!/usr/bin/env python3
"""Generate label-aligned test annotations and run comparative evaluation.

Creates a modified annotations.json where prompts use training-set style labels
(simple names like "coal ore" instead of "Point to the coal ore"), and maps
test labels to training-set equivalents where they differ.

Usage:
    python scripts/eval_label_aligned.py \
        --checkpoint weights/checkpoint_epoch005.pth \
        --device cpu
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LABEL_MAP = {
    "tree_trunk": "oak_log",
    "ground": "dirt",
    "portal_frame": "obsidian",
    "ocean": "water",
    "cliff_edge": "stone",
}

PROMPT_STYLES = {
    "original": lambda ann: ann["prompt"],
    "simple": lambda ann: ann["label"].replace("_", " "),
    "aligned": lambda ann: LABEL_MAP.get(ann["label"], ann["label"]).replace("_", " "),
}


def build_aligned_annotations(src_path: str) -> dict:
    """Create three annotation variants: original, simple, aligned."""
    with open(src_path) as f:
        data = json.load(f)

    variants = {}
    for style_name, prompt_fn in PROMPT_STYLES.items():
        variant = copy.deepcopy(data)
        for key, ann in variant["annotations"].items():
            ann["prompt"] = prompt_fn(ann)
            if style_name == "aligned":
                ann["label"] = LABEL_MAP.get(ann["label"], ann["label"])
        variants[style_name] = variant
    return variants


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--checkpoint", default="weights/checkpoint_epoch005.pth")
    parser.add_argument("--annotations", default="data/test/annotations.json")
    parser.add_argument("--image-dir", default="data/test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="logs")
    args = parser.parse_args()

    import torch
    from groundingdino.util.inference import load_model
    from eval.evaluator import run_evaluation, load_annotations
    from eval.metrics import summarize_results
    from eval.visualize import generate_visual_grid

    if args.device == "auto":
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"

    print(f"Device: {args.device}")
    print(f"Loading model: {args.checkpoint}")
    model = load_model(args.config, args.checkpoint, device=args.device)

    variants = build_aligned_annotations(args.annotations)
    os.makedirs(args.output_dir, exist_ok=True)

    all_results = {}
    for style_name, variant_data in variants.items():
        tmp_path = os.path.join(args.output_dir, f"_tmp_ann_{style_name}.json")
        with open(tmp_path, "w") as f:
            json.dump(variant_data, f)

        print(f"\n{'='*60}")
        print(f"  Style: {style_name}")
        print(f"{'='*60}")

        sample_anns = list(variant_data["annotations"].values())
        for ann in sample_anns[:3]:
            print(f"  {ann['task']:20s} prompt=\"{ann['prompt']}\"")
        print(f"  ...")

        results = run_evaluation(
            model=model,
            annotation_path=tmp_path,
            image_dir=args.image_dir,
            device=args.device,
        )
        summary = summarize_results(results)
        all_results[style_name] = (results, summary, sample_anns)

        grid_path = os.path.join(args.output_dir, f"label_test_{style_name}_grid.png")
        annotations_list = load_annotations(tmp_path)
        generate_visual_grid(results, annotations_list, args.image_dir, grid_path)
        print(f"  Grid: {grid_path}")

        os.remove(tmp_path)

    print(f"\n{'='*70}")
    print(f"  COMPARISON: original vs simple vs aligned prompts")
    print(f"{'='*70}")
    print(f"  {'':25s} {'original':>10s} {'simple':>10s} {'aligned':>10s}")
    print(f"  {'-'*55}")
    print(f"  {'Mean IoU':25s}", end="")
    for s in ["original", "simple", "aligned"]:
        print(f" {all_results[s][1]['mean_iou']:>10.3f}", end="")
    print()
    print(f"  {'Detection Rate':25s}", end="")
    for s in ["original", "simple", "aligned"]:
        print(f" {all_results[s][1]['detection_rate']:>10.1%}", end="")
    print()
    for t in ["0.25", "0.5", "0.75"]:
        print(f"  {f'Acc@IoU>={t}':25s}", end="")
        for s in ["original", "simple", "aligned"]:
            print(f" {all_results[s][1]['accuracy'][float(t)]:>10.1%}", end="")
        print()

    print(f"\n  Per-task IoU comparison:")
    print(f"  {'Task':25s} {'Label':18s} {'original':>10s} {'simple':>10s} {'aligned':>10s} {'Δ(align)':>10s}")
    print(f"  {'-'*93}")

    orig_results = all_results["original"][0]
    simple_results = all_results["simple"][0]
    aligned_results = all_results["aligned"][0]

    for i in range(len(orig_results)):
        orig = orig_results[i]
        simp = simple_results[i]
        alig = aligned_results[i]
        delta = alig.iou - orig.iou
        marker = ""
        if delta > 0.05:
            marker = " ↑"
        elif delta < -0.05:
            marker = " ↓"
        aligned_ann = list(all_results["aligned"][2])[i]
        print(f"  {orig.task:25s} {aligned_ann['label']:18s}"
              f" {orig.iou:>10.3f} {simp.iou:>10.3f} {alig.iou:>10.3f}"
              f" {delta:>+10.3f}{marker}")

    print(f"{'='*70}")


if __name__ == "__main__":
    main()
