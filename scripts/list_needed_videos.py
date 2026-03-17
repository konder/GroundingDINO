"""List raw MP4 video files needed for the fine-tuning dataset.

Run on the production machine where both the aligned sample LMDB
and the raw video directory are accessible.

Usage:
    python scripts/list_needed_videos.py \
        --data-root data/raw/minestudio_aligned \
        --raw-video-dir /mnt/nas/vpt_raw_video \
        -o needed_videos.txt
"""
from __future__ import annotations

import argparse
import json
import lmdb
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, Set


def main():
    parser = argparse.ArgumentParser(description="List raw MP4 files needed for fine-tuning")
    parser.add_argument("--data-root", required=True,
                        help="Aligned MineStudio sample (with segmentation/, image/)")
    parser.add_argument("--raw-video-dir", required=True,
                        help="Root of raw VPT video files")
    parser.add_argument("-o", "--output", default="needed_videos.txt",
                        help="Output file for video paths")
    args = parser.parse_args()

    data_root = args.data_root

    # 1. Find episode IDs used in segmentation
    seg_episode_ids: Set[int] = set()
    seg_dir = Path(data_root) / "segmentation"
    for part in sorted(seg_dir.iterdir()):
        if not part.is_dir() or not (part / "data.mdb").exists():
            continue
        env = lmdb.open(str(part), readonly=True, lock=False,
                        readahead=False, map_size=1024**3 * 100)
        with env.begin() as txn:
            for key_raw, _ in txn.cursor():
                key_str = key_raw.decode()
                if key_str.startswith("__"):
                    continue
                ep_id = eval(key_str)[0]
                seg_episode_ids.add(ep_id)
        env.close()
    print(f"Episode IDs in segmentation: {sorted(seg_episode_ids)}")

    # 2. Read __chunk_infos__ from ALL source LMDBs to get episode_idx → name
    id_to_name: Dict[int, str] = {}
    for subdir in ["video", "image", "segmentation"]:
        src_dir = Path(data_root) / subdir
        if not src_dir.exists():
            continue
        for part in sorted(src_dir.iterdir()):
            if not part.is_dir() or not (part / "data.mdb").exists():
                continue
            env = lmdb.open(str(part), readonly=True, lock=False,
                            readahead=False, map_size=1024**3 * 100)
            with env.begin() as txn:
                raw = txn.get(b"__chunk_infos__")
                if raw:
                    for info in pickle.loads(raw):
                        idx = info.get("episode_idx", info.get("idx"))
                        name = info.get("episode", "")
                        if idx is not None and name:
                            id_to_name[idx] = name
            env.close()

    print(f"Episode mappings found: {len(id_to_name)}")

    needed_names = sorted(set(
        id_to_name[eid] for eid in seg_episode_ids if eid in id_to_name
    ))
    print(f"Episodes needing raw video: {len(needed_names)}")

    if not needed_names:
        print("ERROR: No episode name mappings found. "
              "Re-run sample_aligned.py with the latest code to include __chunk_infos__.")
        sys.exit(1)

    # 3. Scan raw video dir to find full paths
    print(f"Scanning {args.raw_video_dir} for MP4 files ...")
    raw_index: Dict[str, str] = {}
    for mp4 in Path(args.raw_video_dir).rglob("*.mp4"):
        if mp4.stem not in raw_index:
            raw_index[mp4.stem] = str(mp4)
    print(f"  Found {len(raw_index)} MP4 files total")

    # 4. Match and output
    found = []
    missing = []
    for name in needed_names:
        path = raw_index.get(name)
        if path:
            found.append(path)
        else:
            missing.append(name)

    with open(args.output, "w") as f:
        for p in found:
            f.write(p + "\n")

    print(f"\nResult: {len(found)} videos found, {len(missing)} missing")
    print(f"Saved to {args.output}")

    if missing:
        print(f"\nMissing videos:")
        for m in missing:
            print(f"  {m}.mp4")

    # Also save a JSON with details
    json_out = args.output.replace(".txt", ".json")
    with open(json_out, "w") as f:
        json.dump({
            "episode_ids": sorted(seg_episode_ids),
            "episode_names": needed_names,
            "found_paths": found,
            "missing": missing,
        }, f, indent=2)
    print(f"Details saved to {json_out}")


if __name__ == "__main__":
    main()
