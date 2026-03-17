"""Sample aligned data from MineStudio LMDB for fine-tuning development.

Unlike sample_minestudio.py (which samples each DB independently), this script
ensures segmentation and image entries are aligned: for each segmentation entry
that contains event annotations, it also extracts the matching image chunk.

Usage:
    python scripts/sample_aligned.py \
        /mnt/nas/rocket2_train/dataset_6xx \
        -o data/raw/minestudio_aligned \
        -n 50 --events-only

Requires: pip install lmdb
"""
from __future__ import annotations

import argparse
import json
import lmdb
import os
import pickle
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


def find_partition_pairs(data_root: str) -> List[dict]:
    """Find corresponding segmentation / video (or image) partition pairs.

    Prefers video/ (640x360) over image/ (224x224).
    MineStudio naming: seg/part-949 ↔ video/video-950, seg/part-1898 ↔ video/video-1900, etc.
    """
    root = Path(data_root)
    seg_dir = root / "segmentation"

    if not seg_dir.exists():
        print(f"ERROR: {seg_dir} not found")
        sys.exit(1)

    # Prefer video/ (640x360) over image/ (224x224)
    vid_dir = root / "video"
    img_dir = root / "image"

    if vid_dir.exists():
        source_dir = vid_dir
        source_type = "video"
        print(f"Using video/ directory (640x360 original resolution)")
    elif img_dir.exists():
        source_dir = img_dir
        source_type = "image"
        print(f"Using image/ directory (224x224 resized)")
    else:
        print(f"ERROR: Need either {vid_dir} or {img_dir}")
        sys.exit(1)

    seg_parts = sorted([
        d for d in seg_dir.iterdir()
        if d.is_dir() and (d / "data.mdb").exists()
    ], key=lambda p: _part_num(p.name))

    src_parts = sorted([
        d for d in source_dir.iterdir()
        if d.is_dir() and (d / "data.mdb").exists()
    ], key=lambda p: _part_num(p.name))

    src_nums = {_part_num(p.name): p for p in src_parts}

    pairs = []
    for sp in seg_parts:
        sp_num = _part_num(sp.name)
        closest_num = min(src_nums.keys(), key=lambda n: abs(n - sp_num))
        pairs.append({
            "seg_path": str(sp),
            "img_path": str(src_nums[closest_num]),
            "seg_part": sp.name,
            "img_part": src_nums[closest_num].name,
            "source_type": source_type,
        })

    return pairs


def _part_num(name: str) -> int:
    """Extract numeric part from 'part-NNN'."""
    try:
        return int(name.split("-")[-1])
    except ValueError:
        return 0


def find_event_keys(
    seg_path: str,
    max_keys: int = 0,
    events_only: bool = True,
) -> List[Tuple[bytes, int, str]]:
    """Scan segmentation DB for keys that contain event annotations.

    Returns list of (key_bytes, frame_index_in_chunk, event_description).
    """
    env = lmdb.open(seg_path, readonly=True, lock=False,
                    readahead=False, map_size=1024**3 * 100)
    results = []
    seen_keys = set()

    with env.begin() as txn:
        for key_raw, val_raw in txn.cursor():
            if max_keys and len(seen_keys) >= max_keys:
                break

            if key_raw in seen_keys:
                continue

            val = pickle.loads(val_raw)
            has_event = False

            for fi, frame_dict in enumerate(val):
                for event_key, event_val in frame_dict.items():
                    if not isinstance(event_key, tuple) or len(event_key) < 2:
                        continue
                    rle = event_val.get("rle_mask", "")
                    if events_only and (not rle or not rle.strip()):
                        continue
                    event_name = event_val.get("event", "unknown")
                    skip = {"right_click", "landmark", "attack"}
                    if event_name in skip:
                        continue
                    has_event = True
                    break
                if has_event:
                    break

            if has_event or not events_only:
                seen_keys.add(key_raw)
                results.append((key_raw, 0, "has_events"))

    env.close()
    return results


def sample_aligned(
    data_root: str,
    output_dir: str,
    num_samples: int = 50,
    events_only: bool = True,
) -> dict:
    """Create aligned sample with matched segmentation + video/image data."""
    pairs = find_partition_pairs(data_root)
    print(f"Found {len(pairs)} segmentation partition pairs")

    output_root = Path(output_dir)
    os.makedirs(output_root / "segmentation", exist_ok=True)

    source_type = pairs[0]["source_type"] if pairs else "video"
    os.makedirs(output_root / source_type, exist_ok=True)

    manifest = {
        "source": data_root,
        "source_type": source_type,
        "num_samples_per_partition": num_samples,
        "events_only": events_only,
        "partitions": [],
    }

    total_seg = 0
    total_img = 0

    for pair in pairs:
        seg_path = pair["seg_path"]
        img_path = pair["img_path"]
        seg_part = pair["seg_part"]
        img_part = pair["img_part"]
        stype = pair["source_type"]

        print(f"\n--- {seg_part} ↔ {img_part} ({stype}) ---")

        event_keys = find_event_keys(seg_path, max_keys=num_samples,
                                     events_only=events_only)
        if not event_keys:
            print(f"  No event keys found, skipping")
            continue

        key_bytes_list = [ek[0] for ek in event_keys]
        print(f"  Found {len(key_bytes_list)} keys with events")

        dst_seg = str(output_root / "segmentation" / seg_part)
        if os.path.exists(dst_seg):
            shutil.rmtree(dst_seg)
        seg_copied = _copy_keys(seg_path, dst_seg, key_bytes_list)
        print(f"  Segmentation: copied {seg_copied} entries")
        total_seg += seg_copied

        dst_img = str(output_root / stype / img_part)
        if os.path.exists(dst_img):
            shutil.rmtree(dst_img)
        img_copied = _copy_keys(img_path, dst_img, key_bytes_list)
        print(f"  {stype.capitalize()}: copied {img_copied}/{len(key_bytes_list)} entries")
        total_img += img_copied

        manifest["partitions"].append({
            "seg_part": seg_part,
            "img_part": img_part,
            "source_type": stype,
            "event_keys": len(key_bytes_list),
            "seg_copied": seg_copied,
            "img_copied": img_copied,
        })

    # Also copy event DB (small, copy first N entries)
    event_src = os.path.join(data_root, "event")
    if os.path.isdir(event_src) and (Path(event_src) / "data.mdb").exists():
        dst_event = str(output_root / "event")
        if os.path.exists(dst_event):
            shutil.rmtree(dst_event)
        ev_copied = _copy_first_n(event_src, dst_event, num_samples * 10)
        print(f"\nEvent DB: copied {ev_copied} entries")

    manifest_path = output_root / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    total_size = sum(
        f.stat().st_size for f in output_root.rglob("*") if f.is_file()
    ) / (1024 * 1024)

    print(f"\n{'='*60}")
    print(f"Aligned sample created: {output_root}")
    print(f"  Segmentation entries: {total_seg}")
    print(f"  Image entries: {total_img}")
    print(f"  Total size: {total_size:.1f} MB")
    print(f"  Manifest: {manifest_path}")
    print(f"{'='*60}")

    return manifest


_METADATA_KEYS = [
    b"__chunk_infos__",
    b"__chunk_size__",
    b"__num_episodes__",
    b"__num_total_frames__",
]


def _copy_keys(
    src_dir: str,
    dst_dir: str,
    keys: List[bytes],
) -> int:
    """Copy specific keys + MineStudio metadata from source LMDB to destination."""
    src_env = lmdb.open(src_dir, readonly=True, lock=False,
                        readahead=False, map_size=1024**3 * 100)
    os.makedirs(dst_dir, exist_ok=True)
    dst_env = lmdb.open(dst_dir, map_size=1024**3 * 10)

    copied = 0

    with src_env.begin() as src_txn, dst_env.begin(write=True) as dst_txn:
        for mk in _METADATA_KEYS:
            val = src_txn.get(mk)
            if val is not None:
                dst_txn.put(mk, val)

        for key in keys:
            val = src_txn.get(key)
            if val is not None:
                dst_txn.put(key, val)
                copied += 1

    src_env.close()
    dst_env.close()
    return copied


def _copy_first_n(src_dir: str, dst_dir: str, n: int) -> int:
    """Copy first N entries from LMDB."""
    src_env = lmdb.open(src_dir, readonly=True, lock=False,
                        readahead=False, map_size=1024**3 * 100)
    os.makedirs(dst_dir, exist_ok=True)
    dst_env = lmdb.open(dst_dir, map_size=1024**3 * 10)

    copied = 0
    with src_env.begin() as src_txn, dst_env.begin(write=True) as dst_txn:
        for key, val in src_txn.cursor():
            if copied >= n:
                break
            dst_txn.put(key, val)
            copied += 1

    src_env.close()
    dst_env.close()
    return copied


def main():
    parser = argparse.ArgumentParser(
        description="Sample aligned segmentation+image data from MineStudio"
    )
    parser.add_argument("dataset_dir",
                        help="Root directory of MineStudio dataset")
    parser.add_argument("-o", "--output", default="data/raw/minestudio_aligned",
                        help="Output directory")
    parser.add_argument("-n", "--num-samples", type=int, default=50,
                        help="Max event keys per partition")
    parser.add_argument("--events-only", action="store_true", default=True,
                        help="Only sample entries with event annotations (default)")
    parser.add_argument("--all-entries", action="store_true",
                        help="Sample all entries, not just those with events")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing output")

    args = parser.parse_args()

    if os.path.exists(args.output) and not args.force:
        print(f"Output {args.output} exists. Use --force to overwrite.")
        sys.exit(1)

    events_only = not args.all_entries

    sample_aligned(
        data_root=args.dataset_dir,
        output_dir=args.output,
        num_samples=args.num_samples,
        events_only=events_only,
    )


if __name__ == "__main__":
    main()
