"""Merge multiple COCO-format datasets into one.

Each input directory should contain:
  - annotations.json  (COCO format)
  - images/           (image files)

The merge:
  - Unifies categories across all datasets (same name → same ID)
  - Re-maps image/annotation IDs to be globally unique
  - Prefixes filenames with source directory name to avoid collisions
  - Copies (or symlinks) images into a single output directory

Usage:
    python scripts/merge_coco_datasets.py \\
        --inputs data/processed/finetune_6xx data/processed/finetune_7xx \\
        --output data/processed/finetune_merged
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional


def merge_coco_datasets(
    input_dirs: List[str],
    output_dir: str,
    use_symlinks: bool = False,
) -> dict:
    """Merge multiple COCO datasets.

    Returns merged summary dict.
    """
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

    global_categories: Dict[str, int] = {}
    next_cat_id = 1
    all_images = []
    all_annotations = []
    next_img_id = 1
    next_ann_id = 1

    stats = {"sources": [], "total_images": 0, "total_annotations": 0}

    for input_dir in input_dirs:
        ann_path = os.path.join(input_dir, "annotations.json")
        if not os.path.isfile(ann_path):
            print(f"[WARN] Skipping {input_dir}: no annotations.json", flush=True)
            continue

        src_name = Path(input_dir).name
        with open(ann_path) as f:
            data = json.load(f)

        # --- Unify categories ---
        old_cat_to_new: Dict[int, int] = {}
        for cat in data.get("categories", []):
            name = cat["name"]
            if name not in global_categories:
                global_categories[name] = next_cat_id
                next_cat_id += 1
            old_cat_to_new[cat["id"]] = global_categories[name]

        # --- Re-map images ---
        old_img_to_new: Dict[int, int] = {}
        n_imgs = 0
        for img in data.get("images", []):
            old_id = img["id"]
            new_id = next_img_id
            next_img_id += 1
            old_img_to_new[old_id] = new_id

            old_fname = img["file_name"]
            new_fname = f"{src_name}_{old_fname}"

            src_path = os.path.join(input_dir, "images", old_fname)
            dst_path = os.path.join(output_dir, "images", new_fname)

            if os.path.isfile(src_path) and not os.path.exists(dst_path):
                if use_symlinks:
                    os.symlink(os.path.abspath(src_path), dst_path)
                else:
                    shutil.copy2(src_path, dst_path)

            new_img = dict(img)
            new_img["id"] = new_id
            new_img["file_name"] = new_fname
            all_images.append(new_img)
            n_imgs += 1

        # --- Re-map annotations ---
        n_anns = 0
        for ann in data.get("annotations", []):
            new_ann = dict(ann)
            new_ann["id"] = next_ann_id
            next_ann_id += 1
            new_ann["image_id"] = old_img_to_new.get(ann["image_id"], ann["image_id"])
            new_ann["category_id"] = old_cat_to_new.get(ann["category_id"], ann["category_id"])
            all_annotations.append(new_ann)
            n_anns += 1

        print(f"  [{src_name}] {n_imgs} images, {n_anns} annotations", flush=True)
        stats["sources"].append({"name": src_name, "images": n_imgs, "annotations": n_anns})

    categories_list = [
        {"id": cid, "name": name}
        for name, cid in sorted(global_categories.items(), key=lambda x: x[1])
    ]

    merged = {
        "images": all_images,
        "annotations": all_annotations,
        "categories": categories_list,
    }

    out_path = os.path.join(output_dir, "annotations.json")
    with open(out_path, "w") as f:
        json.dump(merged, f)

    stats["total_images"] = len(all_images)
    stats["total_annotations"] = len(all_annotations)
    stats["categories"] = len(categories_list)

    summary_path = os.path.join(output_dir, "merge_summary.json")
    with open(summary_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nMerged: {stats['total_images']} images, "
          f"{stats['total_annotations']} annotations, "
          f"{stats['categories']} categories", flush=True)
    print(f"Output: {out_path}", flush=True)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Merge multiple COCO datasets")
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="Input dataset directories (each with annotations.json + images/)")
    parser.add_argument("--output", required=True, help="Output merged directory")
    parser.add_argument("--symlinks", action="store_true",
                        help="Use symlinks instead of copying images (saves disk space)")
    args = parser.parse_args()

    merge_coco_datasets(
        input_dirs=args.inputs,
        output_dir=args.output,
        use_symlinks=args.symlinks,
    )


if __name__ == "__main__":
    main()
