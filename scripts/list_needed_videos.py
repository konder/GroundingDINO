"""List raw MP4 video files needed for the fine-tuning dataset.

VPT splits recordings into ~60-second chunks.  MineStudio stitches these
into episodes with continuous frame numbering.  This script identifies
ALL chunks needed per episode (not just the first one) by reading the
segmentation LMDB's __chunk_infos__ to get episode names and num_frames.

Usage:
    python scripts/list_needed_videos.py \
        --data-root data/raw/minestudio_aligned \
        --raw-video-dir data/raw/videos \
        -o needed_videos.txt

    # With VPT index JSON for download URLs:
    python scripts/list_needed_videos.py \
        --data-root data/raw/minestudio_aligned \
        --vpt-index ~/Downloads/all_6xx_Jun_29.json \
        -o needed_videos.txt
"""
from __future__ import annotations

import argparse
import json
import lmdb
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


FRAMES_PER_CHUNK = 1201


def read_seg_episode_mapping(seg_path: str) -> List[dict]:
    """Read full __chunk_infos__ from a segmentation LMDB partition."""
    env = lmdb.open(seg_path, readonly=True, lock=False,
                    readahead=False, map_size=1024**3 * 100)
    infos = []
    with env.begin() as txn:
        raw = txn.get(b"__chunk_infos__")
        if raw:
            infos = pickle.loads(raw)
    env.close()
    return infos


def _parse_episode_prefix(episode_name: str) -> Tuple[str, str]:
    """Split 'Player122-f153ac423f61-20211217-140509' → (prefix, start_ts)."""
    parts = episode_name.rsplit("-", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0], parts[1]
    return episode_name, ""


def find_episode_chunks_in_index(
    vpt_relpaths: List[str],
    basedir: str,
    episode_name: str,
    num_chunks_needed: int,
) -> List[str]:
    """Find all chunk URLs for an episode from the VPT index JSON."""
    prefix, start_ts = _parse_episode_prefix(episode_name)

    candidates = []
    for path in vpt_relpaths:
        fname = path.split("/")[-1].replace(".mp4", "")
        if fname.startswith(prefix + "-"):
            ts = fname.rsplit("-", 1)[-1]
            if ts >= start_ts:
                candidates.append((ts, basedir + path))

    candidates.sort()
    return [url for _, url in candidates[:num_chunks_needed]]


def find_episode_chunks_in_dir(
    raw_video_dir: str,
    episode_name: str,
    num_chunks_needed: int,
) -> List[str]:
    """Find all chunk paths for an episode in a local directory."""
    prefix, start_ts = _parse_episode_prefix(episode_name)

    candidates = []
    for mp4 in Path(raw_video_dir).rglob("*.mp4"):
        stem = mp4.stem
        if stem.startswith(prefix + "-"):
            ts = stem.rsplit("-", 1)[-1]
            if ts >= start_ts:
                candidates.append((ts, str(mp4)))

    candidates.sort()
    return [path for _, path in candidates[:num_chunks_needed]]


def main():
    parser = argparse.ArgumentParser(
        description="List ALL raw MP4 chunks needed for fine-tuning (multi-chunk episodes)")
    parser.add_argument("--data-root", required=True,
                        help="Aligned MineStudio sample (with segmentation/)")
    parser.add_argument("--raw-video-dir", default=None,
                        help="Root of raw VPT video files (for local matching)")
    parser.add_argument("--vpt-index", default=None,
                        help="VPT index JSON (e.g. all_6xx_Jun_29.json) for download URLs")
    parser.add_argument("-o", "--output", default="needed_videos.txt",
                        help="Output file")
    args = parser.parse_args()

    seg_dir = Path(args.data_root) / "segmentation"
    if not seg_dir.exists():
        print(f"ERROR: {seg_dir} not found")
        sys.exit(1)

    vpt_data = None
    if args.vpt_index:
        with open(args.vpt_index) as f:
            vpt_data = json.load(f)

    # 1. Collect needed episodes with their num_frames
    needed: Dict[str, int] = {}  # episode_name → num_frames

    for part in sorted(seg_dir.iterdir()):
        if not part.is_dir() or not (part / "data.mdb").exists():
            continue

        infos = read_seg_episode_mapping(str(part))

        ep_ids_in_data: Set[int] = set()
        env = lmdb.open(str(part), readonly=True, lock=False,
                        readahead=False, map_size=1024**3 * 100)
        with env.begin() as txn:
            for key_raw, _ in txn.cursor():
                key_str = key_raw.decode()
                if key_str.startswith("__"):
                    continue
                ep_ids_in_data.add(eval(key_str)[0])
        env.close()

        info_map = {}
        for info in infos:
            idx = info.get("episode_idx", info.get("idx"))
            if idx is not None:
                info_map[idx] = info

        print(f"  {part.name}: {len(ep_ids_in_data)} episodes in data")
        for eid in sorted(ep_ids_in_data):
            info = info_map.get(eid)
            if info:
                name = info["episode"]
                nf = info.get("num_frames", 0)
                needed[name] = max(needed.get(name, 0), nf)
                n_chunks = (nf + FRAMES_PER_CHUNK - 1) // FRAMES_PER_CHUNK
                print(f"    ep_id={eid} → {name} ({nf} frames, ~{n_chunks} chunks)")
            else:
                print(f"    ep_id={eid} → ??? (not in __chunk_infos__)")

    print(f"\nTotal unique episodes: {len(needed)}")

    # 2. For each episode, find all needed chunks
    all_files: List[str] = []
    episode_details = []

    for ep_name in sorted(needed.keys()):
        nf = needed[ep_name]
        n_chunks = (nf + FRAMES_PER_CHUNK - 1) // FRAMES_PER_CHUNK

        if vpt_data:
            chunks = find_episode_chunks_in_index(
                vpt_data["relpaths"], vpt_data["basedir"],
                ep_name, n_chunks)
        elif args.raw_video_dir:
            chunks = find_episode_chunks_in_dir(
                args.raw_video_dir, ep_name, n_chunks)
        else:
            chunks = [f"{ep_name}.mp4"]

        detail = {
            "episode": ep_name,
            "num_frames": nf,
            "chunks_needed": n_chunks,
            "chunks_found": len(chunks),
            "files": chunks,
        }
        episode_details.append(detail)
        all_files.extend(chunks)

        status = "OK" if len(chunks) >= n_chunks else f"INCOMPLETE ({len(chunks)}/{n_chunks})"
        print(f"  {ep_name}: {nf} frames → {n_chunks} chunks [{status}]")

    # 3. Write output
    with open(args.output, "w") as f:
        for p in all_files:
            f.write(p + "\n")
    print(f"\nTotal files: {len(all_files)}")
    print(f"Saved to {args.output}")

    # Generate download script if using VPT index
    if vpt_data:
        sh_path = args.output.replace(".txt", "_download.sh")
        with open(sh_path, "w") as f:
            f.write("#!/bin/bash\n")
            f.write("# Download all VPT video chunks needed for fine-tuning\n")
            f.write("mkdir -p data/raw/videos\ncd data/raw/videos\n\n")
            for url in all_files:
                fname = url.split("/")[-1]
                f.write(f'[ -f "{fname}" ] || wget -q --show-progress "{url}"\n')
            f.write(f'\necho "Done. {len(all_files)} files needed."\n')
        print(f"Download script: {sh_path}")

    json_out = args.output.replace(".txt", ".json")
    with open(json_out, "w") as f:
        json.dump({
            "episodes": episode_details,
            "total_files": len(all_files),
        }, f, indent=2)
    print(f"Details: {json_out}")


if __name__ == "__main__":
    main()
