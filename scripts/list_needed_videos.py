"""List raw MP4 video files needed for the fine-tuning dataset.

Uses SEGMENTATION partition's __chunk_infos__ (not image's) to resolve
episode names, since segmentation and image partitions contain different
episodes under the same episode_id.

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
from typing import Dict, Set, Tuple


def read_seg_episode_mapping(seg_path: str) -> Dict[int, str]:
    """Read episode_idx → episode_name from a segmentation LMDB's __chunk_infos__."""
    env = lmdb.open(seg_path, readonly=True, lock=False,
                    readahead=False, map_size=1024**3 * 100)
    mapping = {}
    with env.begin() as txn:
        raw = txn.get(b"__chunk_infos__")
        if raw:
            for info in pickle.loads(raw):
                idx = info.get("episode_idx", info.get("idx"))
                name = info.get("episode", "")
                if idx is not None and name:
                    mapping[idx] = name
    env.close()
    return mapping


def main():
    parser = argparse.ArgumentParser(description="List raw MP4 files needed for fine-tuning")
    parser.add_argument("--data-root", required=True,
                        help="Aligned MineStudio sample (with segmentation/)")
    parser.add_argument("--raw-video-dir", required=True,
                        help="Root of raw VPT video files")
    parser.add_argument("-o", "--output", default="needed_videos.txt",
                        help="Output file for video paths")
    args = parser.parse_args()

    seg_dir = Path(args.data_root) / "segmentation"
    if not seg_dir.exists():
        print(f"ERROR: {seg_dir} not found")
        sys.exit(1)

    # 1. For each segmentation partition, find which episode IDs are used
    #    and resolve them via THAT partition's __chunk_infos__
    needed_names: Set[str] = set()
    per_partition_info = []

    for part in sorted(seg_dir.iterdir()):
        if not part.is_dir() or not (part / "data.mdb").exists():
            continue

        mapping = read_seg_episode_mapping(str(part))

        ep_ids_in_data: Set[int] = set()
        env = lmdb.open(str(part), readonly=True, lock=False,
                        readahead=False, map_size=1024**3 * 100)
        with env.begin() as txn:
            for key_raw, _ in txn.cursor():
                key_str = key_raw.decode()
                if key_str.startswith("__"):
                    continue
                ep_id = eval(key_str)[0]
                ep_ids_in_data.add(ep_id)
        env.close()

        resolved = {}
        missing_ids = []
        for eid in sorted(ep_ids_in_data):
            name = mapping.get(eid)
            if name:
                resolved[eid] = name
                needed_names.add(name)
            else:
                missing_ids.append(eid)

        info = {
            "partition": part.name,
            "chunk_infos_count": len(mapping),
            "episode_ids_in_data": sorted(ep_ids_in_data),
            "resolved": resolved,
            "missing_ids": missing_ids,
        }
        per_partition_info.append(info)

        print(f"  {part.name}: {len(ep_ids_in_data)} episodes in data, "
              f"{len(mapping)} in __chunk_infos__, "
              f"{len(resolved)} resolved, {len(missing_ids)} missing")
        for eid, name in sorted(resolved.items()):
            print(f"    ep_id={eid} → {name}")
        for eid in missing_ids:
            print(f"    ep_id={eid} → ??? (not in __chunk_infos__)")

    print(f"\nTotal unique episodes needed: {len(needed_names)}")
    if not needed_names:
        print("ERROR: No episode names resolved. Check __chunk_infos__ in segmentation LMDBs.")
        sys.exit(1)

    # 2. Scan raw video dir
    print(f"\nScanning {args.raw_video_dir} for MP4 files ...")
    raw_index: Dict[str, str] = {}
    for mp4 in Path(args.raw_video_dir).rglob("*.mp4"):
        if mp4.stem not in raw_index:
            raw_index[mp4.stem] = str(mp4)
    print(f"  Found {len(raw_index)} MP4 files total")

    # 3. Match
    found = []
    missing = []
    for name in sorted(needed_names):
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

    if found:
        print(f"\nFound videos:")
        for p in found:
            print(f"  {p}")

    json_out = args.output.replace(".txt", ".json")
    with open(json_out, "w") as f:
        json.dump({
            "needed_episodes": sorted(needed_names),
            "found_paths": found,
            "missing": missing,
            "per_partition": per_partition_info,
        }, f, indent=2)
    print(f"Details saved to {json_out}")


if __name__ == "__main__":
    main()
