#!/usr/bin/env python3
"""Subsample a COCO dataset: filter categories and cap per-category samples.

Operates purely on annotations.json — no image processing needed.
Images are referenced by filename (assumed to exist in the same images/ dir).

Usage:
    python scripts/subsample_coco.py \
        --input data/processed/lmdb_finetune_all/annotations.json \
        --output data/processed/finetune_train/annotations.json \
        --min-per-cat 100 \
        --max-per-cat 2000 \
        --top-k 0

    # Quick first experiment: top 50 categories, 1000 each
    python scripts/subsample_coco.py \
        --input data/processed/lmdb_finetune_all/annotations.json \
        --output data/processed/finetune_train/annotations.json \
        --max-per-cat 1000 --top-k 50
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path


def subsample(
    input_path: str,
    output_path: str,
    min_per_cat: int = 0,
    max_per_cat: int = 0,
    top_k: int = 0,
    seed: int = 42,
) -> dict:
    """Filter and subsample a COCO annotations file.

    Args:
        min_per_cat: Drop categories with fewer than this many annotations.
        max_per_cat: Cap each category at this many annotations (0=no cap).
        top_k: Keep only the top-K categories by count (0=all).
        seed: Random seed for reproducible subsampling.
    """
    random.seed(seed)

    with open(input_path) as f:
        coco = json.load(f)

    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    img_id_to_info = {img["id"]: img for img in coco["images"]}

    # Group annotations by category
    by_cat: dict[int, list] = defaultdict(list)
    for ann in coco["annotations"]:
        by_cat[ann["category_id"]].append(ann)

    # Sort categories by count
    sorted_cats = sorted(by_cat.items(), key=lambda x: -len(x[1]))

    # Filter by min count
    if min_per_cat > 0:
        sorted_cats = [(cid, anns) for cid, anns in sorted_cats
                        if len(anns) >= min_per_cat]

    # Keep top-K
    if top_k > 0:
        sorted_cats = sorted_cats[:top_k]

    # Subsample per category
    kept_anns = []
    kept_cat_ids = set()
    stats = {}

    for cat_id, anns in sorted_cats:
        cat_name = cat_id_to_name.get(cat_id, f"id_{cat_id}")
        original = len(anns)

        if max_per_cat > 0 and len(anns) > max_per_cat:
            anns = random.sample(anns, max_per_cat)

        kept_anns.extend(anns)
        kept_cat_ids.add(cat_id)
        stats[cat_name] = {"original": original, "kept": len(anns)}

    # Collect referenced images
    used_img_ids = set(ann["image_id"] for ann in kept_anns)
    kept_images = [img_id_to_info[iid] for iid in used_img_ids
                   if iid in img_id_to_info]

    # Remap IDs for clean output
    new_cat_map = {}
    new_categories = []
    for new_id, (cat_id, _) in enumerate(sorted_cats, 1):
        new_cat_map[cat_id] = new_id
        new_categories.append({
            "id": new_id,
            "name": cat_id_to_name[cat_id],
            "supercategory": "minecraft",
        })

    new_img_map = {}
    new_images = []
    for new_id, img in enumerate(kept_images, 1):
        new_img_map[img["id"]] = new_id
        new_images.append({
            "id": new_id,
            "file_name": img["file_name"],
            "width": img["width"],
            "height": img["height"],
        })

    new_anns = []
    for new_id, ann in enumerate(kept_anns, 1):
        if ann["image_id"] not in new_img_map:
            continue
        new_anns.append({
            "id": new_id,
            "image_id": new_img_map[ann["image_id"]],
            "category_id": new_cat_map[ann["category_id"]],
            "bbox": ann["bbox"],
            "area": ann["area"],
            "iscrowd": 0,
        })

    output_coco = {
        "images": new_images,
        "annotations": new_anns,
        "categories": new_categories,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_coco, f)

    summary = {
        "input": input_path,
        "output": output_path,
        "original_images": len(coco["images"]),
        "original_annotations": len(coco["annotations"]),
        "original_categories": len(coco["categories"]),
        "kept_images": len(new_images),
        "kept_annotations": len(new_anns),
        "kept_categories": len(new_categories),
        "min_per_cat": min_per_cat,
        "max_per_cat": max_per_cat,
        "top_k": top_k,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Subsample COCO dataset: filter categories + cap samples")
    parser.add_argument("--input", required=True, help="Input annotations.json")
    parser.add_argument("--output", required=True, help="Output annotations.json")
    parser.add_argument("--min-per-cat", type=int, default=0,
                        help="Drop categories with fewer annotations (default: 0)")
    parser.add_argument("--max-per-cat", type=int, default=0,
                        help="Cap per-category annotations (0=no cap)")
    parser.add_argument("--top-k", type=int, default=0,
                        help="Keep only top-K categories by count (0=all)")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    summary = subsample(
        args.input, args.output,
        min_per_cat=args.min_per_cat,
        max_per_cat=args.max_per_cat,
        top_k=args.top_k,
        seed=args.seed,
    )

    print(f"\n{'='*50}")
    print(f"Original: {summary['original_images']} images, "
          f"{summary['original_annotations']} annotations, "
          f"{summary['original_categories']} categories")
    print(f"Filtered: {summary['kept_images']} images, "
          f"{summary['kept_annotations']} annotations, "
          f"{summary['kept_categories']} categories")
    print(f"{'='*50}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
