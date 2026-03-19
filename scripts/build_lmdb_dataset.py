#!/usr/bin/env python3
"""Build GroundingDINO fine-tuning dataset directly from MineStudio LMDBs.

Assumes MineStudio's segmentation LMDB is the source of truth:
  - Each event has a tracking window (frame_range) with per-frame RLE masks.
  - The image LMDB stores matching frames as 32-frame MP4 chunks at 224x224.
  - Masks are at 640x360. Bboxes are scaled to the image resolution.

No raw video needed — everything comes from the aligned LMDBs.

Usage (single dataset):
    python scripts/build_lmdb_dataset.py \\
        --data-root /mnt/nas/rocket2_train/dataset_6xx \\
        --output-dir data/processed/lmdb_finetune_6xx

Usage (multiple datasets):
    python scripts/build_lmdb_dataset.py \\
        --data-root /mnt/nas/rocket2_train/dataset_6xx \\
                    /mnt/nas/rocket2_train/dataset_7xx \\
        --output-dir data/processed/lmdb_finetune_all

Pipeline:
    1. Build episode maps (seg ↔ image) across all datasets
    2. Scan segmentation for events → extract masks → bboxes
    3. Select best training frames per event (early, intact block)
    4. Decode frames from image LMDB (224x224)
    5. Output COCO-format JSON + frame PNGs
"""
from __future__ import annotations

import argparse
import cv2
import hashlib
import json
import lmdb
import numpy as np
import os
import pickle
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Event label parsing
# ---------------------------------------------------------------------------

SKIP_ACTIONS = {"custom", "open_chest", "craft"}


def parse_event_label(event_str: str) -> Optional[str]:
    """Extract category label from Minecraft event string.

    'mine_block:coal_ore' → 'coal_ore'
    'craft:crafting_table' → None
    """
    if not event_str:
        return None
    s = event_str.replace("minecraft.", "")
    if ":" in s:
        action, target = s.split(":", 1)
        if action in SKIP_ACTIONS:
            return None
        return target.strip() or None
    if s in ("right_click", "landmark", "attack", "craft"):
        return None
    return s.strip() or None


# ---------------------------------------------------------------------------
# RLE mask utilities
# ---------------------------------------------------------------------------

def rle_to_mask(rle_str: str, height: int, width: int) -> np.ndarray:
    """Decode MineStudio RLE to binary mask."""
    if not rle_str or not rle_str.strip():
        return np.zeros((height, width), dtype=np.uint8)
    parts = list(map(int, rle_str.split()))
    total = height * width
    flat = np.zeros(total, dtype=np.uint8)
    for i in range(0, len(parts), 2):
        start = parts[i]
        length = parts[i + 1] if i + 1 < len(parts) else 1
        end = min(start + length, total)
        if start < total:
            flat[start:end] = 1
    return flat.reshape(height, width)


def mask_to_bbox(mask: np.ndarray) -> Optional[List[int]]:
    """Binary mask → [x, y, w, h] or None if empty."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return [int(cmin), int(rmin), int(cmax - cmin + 1), int(rmax - rmin + 1)]


def filter_mask_by_point(
    mask: np.ndarray, point: Optional[Tuple[int, int]]
) -> np.ndarray:
    """Keep only the connected component containing the point."""
    if point is None or mask.sum() == 0:
        return mask
    row, col = point
    row = max(0, min(row, mask.shape[0] - 1))
    col = max(0, min(col, mask.shape[1] - 1))
    num_labels, labels = cv2.connectedComponents(mask)
    target = labels[row, col]
    if target == 0:
        ys, xs = np.where(mask > 0)
        if len(ys) == 0:
            return mask
        dists = (ys - row) ** 2 + (xs - col) ** 2
        target = labels[ys[dists.argmin()], xs[dists.argmin()]]
    return (labels == target).astype(np.uint8)


def compute_mask_area(rle_str: str) -> int:
    """Count foreground pixels from RLE without full decode."""
    if not rle_str or not rle_str.strip():
        return 0
    parts = list(map(int, rle_str.split()))
    return sum(parts[i + 1] for i in range(0, len(parts), 2) if i + 1 < len(parts))


# ---------------------------------------------------------------------------
# Frame selection
# ---------------------------------------------------------------------------

def select_training_frames(
    frame_start: int,
    frame_end: int,
    n_frames: int = 4,
    skip_tail: int = 4,
) -> List[int]:
    """Select training frames from the early part of an event's tracking window.

    For mine_block events, the last frame is when the block breaks.
    We want early frames where the block is still intact and visible.
    skip_tail removes frames near the end (mining cracks / break animation).

    Returns up to n_frames frame indices, evenly spaced from the usable range.
    """
    usable_end = max(frame_start, frame_end - skip_tail)
    total_usable = usable_end - frame_start + 1
    if total_usable <= 0:
        return []
    actual_n = min(n_frames, total_usable)
    if actual_n == 1:
        return [frame_start]
    step = (total_usable - 1) / (actual_n - 1)
    return [frame_start + int(round(i * step)) for i in range(actual_n)]


# ---------------------------------------------------------------------------
# Bbox scaling
# ---------------------------------------------------------------------------

def scale_bbox_to_image(
    bbox: List[int],
    mask_hw: Tuple[int, int],
    image_hw: Tuple[int, int],
) -> List[float]:
    """Scale bbox from mask resolution to image resolution."""
    sx = image_hw[1] / mask_hw[1]
    sy = image_hw[0] / mask_hw[0]
    return [bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy]


# ---------------------------------------------------------------------------
# COCO output
# ---------------------------------------------------------------------------

def build_coco_output(annotations: List[Dict]) -> Dict:
    """Convert flat annotation list to COCO detection format.

    Each annotation dict has: image_file, category, bbox, image_width, image_height
    """
    categories: Dict[str, int] = {}
    cat_id = 1
    images: Dict[str, Dict] = {}
    coco_anns = []
    ann_id = 1

    for ann in annotations:
        cat = ann["category"]
        if cat not in categories:
            categories[cat] = cat_id
            cat_id += 1

        fname = ann["image_file"]
        if fname not in images:
            images[fname] = {
                "id": len(images) + 1,
                "file_name": fname,
                "width": ann["image_width"],
                "height": ann["image_height"],
            }

        bbox = ann["bbox"]
        coco_anns.append({
            "id": ann_id,
            "image_id": images[fname]["id"],
            "category_id": categories[cat],
            "bbox": [round(v, 2) for v in bbox],
            "area": round(bbox[2] * bbox[3], 2),
            "iscrowd": 0,
        })
        ann_id += 1

    return {
        "images": list(images.values()),
        "annotations": coco_anns,
        "categories": [
            {"id": cid, "name": name, "supercategory": "minecraft"}
            for name, cid in sorted(categories.items(), key=lambda x: x[1])
        ],
    }


# ---------------------------------------------------------------------------
# Image LMDB decode
# ---------------------------------------------------------------------------

def decode_image_chunk(env: lmdb.Environment, ep_idx: int,
                       chunk_offset: int) -> Optional[List[np.ndarray]]:
    """Decode 32 frames from an image LMDB MP4 chunk.

    Uses a shared tempfile to avoid per-call tempfile creation overhead.
    """
    key = f"({ep_idx}, {chunk_offset})"
    with env.begin() as txn:
        raw = txn.get(key.encode())
        if raw is None:
            return None

    # Reuse a single tempfile path per thread to reduce filesystem overhead
    tmp_path = getattr(decode_image_chunk, "_tmp_path", None)
    if tmp_path is None:
        fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        decode_image_chunk._tmp_path = tmp_path

    with open(tmp_path, "wb") as f:
        f.write(raw)

    frames = []
    cap = cv2.VideoCapture(tmp_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames if frames else None


# ---------------------------------------------------------------------------
# GUI detection
# ---------------------------------------------------------------------------

def detect_gui_frame(frame: np.ndarray, threshold: float = 0.20) -> bool:
    """Detect if a frame has an open Minecraft GUI overlay."""
    h, w = frame.shape[:2]
    roi = frame[int(h * 0.1):int(h * 0.7), int(w * 0.2):int(w * 0.8)]
    gray_diff = np.max(roi.astype(np.int16), axis=2) - np.min(roi.astype(np.int16), axis=2)
    brightness = np.mean(roi, axis=2)
    gray_mask = (gray_diff < 20) & (brightness > 130) & (brightness < 220)
    return bool(gray_mask.sum() / gray_mask.size >= threshold)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def _find_lmdb_parts(base_dir: str) -> List[str]:
    """Find all LMDB partition directories."""
    base = Path(base_dir)
    if not base.exists():
        return []
    parts = []
    if (base / "data.mdb").exists():
        parts.append(str(base))
    for child in sorted(base.iterdir()):
        if child.is_dir() and (child / "data.mdb").exists():
            parts.append(str(child))
    return parts


def _build_episode_maps(data_roots: List[str]):
    """Build bidirectional episode maps across all data roots.

    Returns:
        seg_ep_map: {(seg_part_path, ep_idx): ep_name}
        img_ep_map: {ep_name: (img_part_path, ep_idx)}
    """
    seg_ep_map = {}
    img_ep_map = {}

    for ri, root in enumerate(data_roots):
        root_name = Path(root).name
        seg_dir = os.path.join(root, "segmentation")
        img_dir = os.path.join(root, "image")

        seg_parts = _find_lmdb_parts(seg_dir)
        img_parts = _find_lmdb_parts(img_dir)
        _print(f"  [{ri+1}/{len(data_roots)}] {root_name}: "
               f"{len(seg_parts)} seg parts, {len(img_parts)} img parts")

        for pi, part_path in enumerate(seg_parts):
            if (pi + 1) % 10 == 0 or pi == 0 or pi == len(seg_parts) - 1:
                _print(f"    seg partition {pi+1}/{len(seg_parts)} "
                       f"({len(seg_ep_map)} episodes)")
            env = lmdb.open(part_path, readonly=True, lock=False,
                            readahead=False, map_size=1024**3 * 100)
            with env.begin() as txn:
                raw = txn.get("__chunk_infos__".encode())
                if raw:
                    for info in pickle.loads(raw):
                        seg_ep_map[(part_path, info["episode_idx"])] = info["episode"]
            env.close()

        for pi, part_path in enumerate(img_parts):
            if (pi + 1) % 10 == 0 or pi == 0 or pi == len(img_parts) - 1:
                _print(f"    img partition {pi+1}/{len(img_parts)} "
                       f"({len(img_ep_map)} episodes)")
            env = lmdb.open(part_path, readonly=True, lock=False,
                            readahead=False, map_size=1024**3 * 100)
            with env.begin() as txn:
                raw = txn.get("__chunk_infos__".encode())
                if raw:
                    for info in pickle.loads(raw):
                        img_ep_map[info["episode"]] = (part_path, info["episode_idx"])
            env.close()

    return seg_ep_map, img_ep_map


@dataclass
class EventRecord:
    """A single event tracking window found in segmentation LMDB."""
    event_name: str
    label: str
    ep_name: str
    seg_part: str
    seg_ep_idx: int
    img_part: str
    img_ep_idx: int
    frame_start: int
    frame_end: int


def _scan_events(
    data_roots: List[str],
    seg_ep_map: Dict,
    img_ep_map: Dict,
    min_mask_area: int = 3000,
    max_mask_area_frac: float = 0.35,
) -> List[EventRecord]:
    """Scan segmentation LMDBs for events that have matching image data.

    Only keeps events where:
    - Event label is parseable (not craft/custom)
    - Episode exists in image LMDB
    - Event has a frame_range (tracking window)
    """
    events: List[EventRecord] = []
    seen = set()

    # Pre-build set of ep_idx values that have image data per partition
    matched_eps: Dict[str, Set[int]] = defaultdict(set)
    for (part_path, ep_idx), ep_name in seg_ep_map.items():
        if ep_name in img_ep_map:
            matched_eps[part_path].add(ep_idx)

    all_seg_parts = []
    for root in data_roots:
        seg_dir = os.path.join(root, "segmentation")
        all_seg_parts.extend(_find_lmdb_parts(seg_dir))

    t_scan = time.time()
    for part_i, part_path in enumerate(all_seg_parts):
        part_matched = matched_eps.get(part_path, set())
        if not part_matched:
            _print(f"    [{part_i+1}/{len(all_seg_parts)}] "
                   f"{Path(part_path).name}: 0 matched episodes, skipping")
            continue

        env = lmdb.open(part_path, readonly=True, lock=False,
                        readahead=False, map_size=1024**3 * 100)
        with env.begin() as txn:
            n_entries = txn.stat()["entries"]
            chunk_count = 0
            skipped_ep = 0
            for key_raw, val_raw in txn.cursor():
                ks = key_raw.decode()
                if ks.startswith("__"):
                    continue

                chunk_count += 1
                if chunk_count % 5000 == 0:
                    elapsed = time.time() - t_scan
                    _print(f"      chunk {chunk_count}/{n_entries} "
                           f"({len(events)} events, {elapsed:.0f}s)")

                chunk_key = eval(ks)
                ep_idx, frame_offset = chunk_key

                if ep_idx not in part_matched:
                    skipped_ep += 1
                    continue

                ep_name = seg_ep_map[(part_path, ep_idx)]
                frames = pickle.loads(val_raw)
                for fi, fd in enumerate(frames):
                    gf = frame_offset + fi
                    for ek, ev in fd.items():
                        if not isinstance(ev, dict):
                            continue
                        event_name = ev.get("event", "")
                        label = parse_event_label(event_name)
                        if label is None:
                            continue
                        fr = ev.get("frame_range")
                        if not fr:
                            continue
                        if gf != fr[1]:
                            continue

                        dedup_key = (ep_name, event_name, fr[0], fr[1])
                        if dedup_key in seen:
                            continue
                        seen.add(dedup_key)

                        img_part, img_eidx = img_ep_map[ep_name]
                        events.append(EventRecord(
                            event_name=event_name,
                            label=label,
                            ep_name=ep_name,
                            seg_part=part_path,
                            seg_ep_idx=ep_idx,
                            img_part=img_part,
                            img_ep_idx=img_eidx,
                            frame_start=fr[0],
                            frame_end=fr[1],
                        ))
        env.close()
        _print(f"    [{part_i+1}/{len(all_seg_parts)}] "
               f"{Path(part_path).name}: {chunk_count} chunks "
               f"(skipped {skipped_ep} unmatched), "
               f"{len(events)} events total")

    return events


_env_cache: Dict[str, lmdb.Environment] = {}


def _get_env(path: str) -> lmdb.Environment:
    """Get or create a cached LMDB environment (avoids repeated open/close)."""
    if path not in _env_cache:
        _env_cache[path] = lmdb.open(
            path, readonly=True, lock=False,
            readahead=False, map_size=1024**3 * 100)
    return _env_cache[path]


def _close_all_envs():
    """Close all cached LMDB environments."""
    for env in _env_cache.values():
        try:
            env.close()
        except Exception:
            pass
    _env_cache.clear()


def _collect_masks_for_frames(
    seg_part: str,
    seg_ep_idx: int,
    target_event: str,
    frame_indices: List[int],
    mask_h: int = 360,
    mask_w: int = 640,
    min_mask_area: int = 3000,
    max_mask_area_frac: float = 0.35,
) -> Dict[int, Tuple[List[int], Optional[Tuple[int, int]]]]:
    """Read segmentation masks for specific frames.

    Returns: {global_frame: (bbox_xywh, point)} for frames with valid masks.
    """
    max_pixels = int(mask_h * mask_w * max_mask_area_frac)
    frame_set = set(frame_indices)
    chunks_needed = set()
    for gf in frame_indices:
        chunks_needed.add((gf // 32) * 32)

    results = {}
    env = _get_env(seg_part)
    with env.begin() as txn:
        for chunk_off in sorted(chunks_needed):
            key = f"({seg_ep_idx}, {chunk_off})"
            raw = txn.get(key.encode())
            if raw is None:
                continue
            frames = pickle.loads(raw)
            for fi, fd in enumerate(frames):
                gf = chunk_off + fi
                if gf not in frame_set:
                    continue
                for ek, ev in fd.items():
                    if not isinstance(ev, dict):
                        continue
                    if ev.get("event") != target_event:
                        continue
                    rle = ev.get("rle_mask", "")
                    if not rle:
                        continue
                    area = compute_mask_area(rle)
                    if area < min_mask_area or area > max_pixels:
                        continue
                    point = ev.get("point")
                    pt_tuple = tuple(point) if point else None
                    mask = rle_to_mask(rle, mask_h, mask_w)
                    mask = filter_mask_by_point(mask, pt_tuple)
                    bbox = mask_to_bbox(mask)
                    if bbox:
                        results[gf] = (bbox, pt_tuple)
    return results


def _decode_frames_from_image_lmdb(
    img_part: str,
    img_ep_idx: int,
    frame_indices: List[int],
) -> Dict[int, np.ndarray]:
    """Decode specific frames from image LMDB."""
    frame_set = set(frame_indices)
    chunks_needed = set()
    for gf in frame_indices:
        chunks_needed.add((gf // 32) * 32)

    decoded = {}
    env = _get_env(img_part)
    for chunk_off in sorted(chunks_needed):
        chunk_frames = decode_image_chunk(env, img_ep_idx, chunk_off)
        if chunk_frames:
            for i, frame in enumerate(chunk_frames):
                gf = chunk_off + i
                if gf in frame_set:
                    decoded[gf] = frame
    return decoded


def _print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


# ---------------------------------------------------------------------------
# Multiprocess worker for Step 3
# ---------------------------------------------------------------------------

def _process_events_worker(args):
    """Self-contained worker: process a batch of events, return annotations.

    Each worker maintains its own LMDB env cache and image chunk LRU cache
    to avoid cross-process sharing issues. On completion, saves a checkpoint
    file so interrupted runs can resume.
    """
    (events_batch, img_dir, mask_hw, n_frames, skip_tail,
     min_mask_area, max_mask_area_frac, filter_gui,
     worker_id, total_workers, output_dir) = args

    os.environ["OPENCV_LOG_LEVEL"] = "FATAL"
    os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"

    envs: Dict[str, lmdb.Environment] = {}
    img_chunk_cache: Dict[Tuple, Optional[List[np.ndarray]]] = {}
    IMG_CACHE_MAX = 48

    fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)

    def get_env(path):
        if path not in envs:
            envs[path] = lmdb.open(
                path, readonly=True, lock=False,
                readahead=False, map_size=1024**3 * 100)
        return envs[path]

    def decode_chunk_cached(img_part, ep_idx, chunk_off):
        cache_key = (img_part, ep_idx, chunk_off)
        if cache_key in img_chunk_cache:
            return img_chunk_cache[cache_key]

        lmdb_key = f"({ep_idx}, {chunk_off})"
        env = get_env(img_part)
        with env.begin() as txn:
            raw = txn.get(lmdb_key.encode())
            if raw is None:
                img_chunk_cache[cache_key] = None
                return None

        with open(tmp_path, "wb") as f:
            f.write(raw)

        frames = []
        cap = cv2.VideoCapture(tmp_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        result = frames if frames else None

        if len(img_chunk_cache) >= IMG_CACHE_MAX:
            img_chunk_cache.pop(next(iter(img_chunk_cache)))
        img_chunk_cache[cache_key] = result
        return result

    annotations = []
    stats = defaultdict(int)
    max_pixels = int(mask_hw[0] * mask_hw[1] * max_mask_area_frac)
    t0 = time.time()

    for i, ev in enumerate(events_batch):
        if (i + 1) % 2000 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            _print(f"    [W{worker_id}] {i+1}/{len(events_batch)} "
                   f"({stats['frames_exported']} frames, {rate:.1f} ev/s)")

        candidate_frames = select_training_frames(
            ev.frame_start, ev.frame_end, n_frames, skip_tail)
        if not candidate_frames:
            stats["events_no_valid_frames"] += 1
            continue

        # --- Collect masks from segmentation LMDB ---
        frame_set = set(candidate_frames)
        seg_chunks = set((gf // 32) * 32 for gf in candidate_frames)
        masks = {}

        seg_env = get_env(ev.seg_part)
        with seg_env.begin() as txn:
            for chunk_off in seg_chunks:
                raw = txn.get(f"({ev.seg_ep_idx}, {chunk_off})".encode())
                if raw is None:
                    continue
                frames_data = pickle.loads(raw)
                for fi, fd in enumerate(frames_data):
                    gf = chunk_off + fi
                    if gf not in frame_set:
                        continue
                    for ek, ev_data in fd.items():
                        if not isinstance(ev_data, dict):
                            continue
                        if ev_data.get("event") != ev.event_name:
                            continue
                        rle = ev_data.get("rle_mask", "")
                        if not rle:
                            continue
                        area = compute_mask_area(rle)
                        if area < min_mask_area or area > max_pixels:
                            continue
                        point = ev_data.get("point")
                        pt = tuple(point) if point else None
                        mask = rle_to_mask(rle, mask_hw[0], mask_hw[1])
                        mask = filter_mask_by_point(mask, pt)
                        bbox = mask_to_bbox(mask)
                        if bbox:
                            masks[gf] = (bbox, pt)

        valid_frames = [f for f in candidate_frames if f in masks]
        if not valid_frames:
            stats["frames_no_mask"] += len(candidate_frames)
            stats["events_no_valid_frames"] += 1
            continue

        # --- Decode image frames (with LRU cache) ---
        vf_set = set(valid_frames)
        img_chunks = set((gf // 32) * 32 for gf in valid_frames)
        decoded = {}
        for chunk_off in sorted(img_chunks):
            chunk_frames = decode_chunk_cached(
                ev.img_part, ev.img_ep_idx, chunk_off)
            if chunk_frames:
                for fi, frame in enumerate(chunk_frames):
                    gf = chunk_off + fi
                    if gf in vf_set:
                        decoded[gf] = frame

        # --- Filter + write ---
        has_any = False
        for gf in valid_frames:
            if gf not in decoded:
                stats["frames_no_image"] += 1
                continue
            frame = decoded[gf]
            if filter_gui and detect_gui_frame(frame):
                stats["frames_gui_filtered"] += 1
                continue

            h_img, w_img = frame.shape[:2]
            bbox_scaled = scale_bbox_to_image(
                masks[gf][0], mask_hw, (h_img, w_img))
            ds_name = Path(ev.seg_part).parent.parent.name
            seg_part_name = Path(ev.seg_part).name
            img_filename = f"{ds_name}_{seg_part_name}_ep{ev.seg_ep_idx}_f{gf:06d}.png"
            img_path = os.path.join(img_dir, img_filename)
            if not os.path.exists(img_path):
                cv2.imwrite(img_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])

            annotations.append({
                "image_file": img_filename,
                "category": ev.label,
                "bbox": bbox_scaled,
                "image_width": w_img,
                "image_height": h_img,
            })
            stats["frames_exported"] += 1
            has_any = True

        if not has_any:
            stats["events_no_valid_frames"] += 1
        stats["events_processed"] += 1

    # Cleanup
    for e in envs.values():
        try:
            e.close()
        except Exception:
            pass
    try:
        os.unlink(tmp_path)
    except OSError:
        pass

    result_stats = dict(stats)
    _save_checkpoint(output_dir, worker_id, annotations, result_stats)
    _print(f"    [W{worker_id}] Done: {len(annotations)} annotations saved "
           f"to checkpoint")

    return annotations, result_stats


def _checkpoint_path(output_dir: str, worker_id: int) -> str:
    return os.path.join(output_dir, f".checkpoint_w{worker_id}.json")


def _save_checkpoint(output_dir: str, worker_id: int,
                     annotations: List[Dict], stats: Dict) -> None:
    path = _checkpoint_path(output_dir, worker_id)
    try:
        with open(path, "w") as f:
            json.dump({"annotations": annotations, "stats": stats}, f)
    except IOError:
        pass


def _load_checkpoint(output_dir: str, worker_id: int
                     ) -> Optional[Tuple[List[Dict], Dict]]:
    path = _checkpoint_path(output_dir, worker_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        return data["annotations"], data["stats"]
    except (json.JSONDecodeError, KeyError, IOError):
        return None


def _clean_checkpoints(output_dir: str, n_workers: int) -> None:
    for wi in range(n_workers):
        path = _checkpoint_path(output_dir, wi)
        if os.path.exists(path):
            os.remove(path)


def build_dataset(
    data_roots: List[str],
    output_dir: str,
    n_frames: int = 4,
    skip_tail: int = 4,
    min_mask_area: int = 3000,
    max_mask_area_frac: float = 0.35,
    filter_gui: bool = True,
    max_visualize: int = 0,
    n_workers: int = 0,
) -> Dict:
    """Main pipeline: build GroundingDINO training data from MineStudio LMDBs.

    Steps:
    1. Build episode maps across all dataset roots
    2. Scan segmentation for events with image data
    3. For each event, select training frames and extract masks→bboxes
    4. Decode frames from image LMDB
    5. Output COCO JSON + PNGs
    """
    os.makedirs(output_dir, exist_ok=True)
    img_dir = os.path.join(output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    mask_hw = (360, 640)

    # --- Step 1: Episode maps ---
    _print(f"[1/5] Building episode maps across {len(data_roots)} dataset(s)...")
    seg_ep_map, img_ep_map = _build_episode_maps(data_roots)
    _print(f"  Segmentation episodes: {len(seg_ep_map)}")
    _print(f"  Image episodes: {len(img_ep_map)}")
    matched = sum(1 for (_, _), name in seg_ep_map.items() if name in img_ep_map)
    _print(f"  Matched (seg→img): {matched}")

    # --- Step 2: Scan events ---
    _print(f"[2/5] Scanning segmentation for events...")

    cache_key = hashlib.md5(
        f"{sorted(data_roots)}|{min_mask_area}|{max_mask_area_frac}".encode()
    ).hexdigest()[:12]
    event_cache_path = f"/tmp/lmdb_events_{cache_key}.json"

    events = None
    if os.path.exists(event_cache_path):
        try:
            with open(event_cache_path) as f:
                cached = json.load(f)
            events = [EventRecord(**e) for e in cached]
            _print(f"  Loaded {len(events)} events from cache")
        except (json.JSONDecodeError, TypeError):
            events = None

    if events is None:
        events = _scan_events(
            data_roots, seg_ep_map, img_ep_map,
            min_mask_area=min_mask_area,
            max_mask_area_frac=max_mask_area_frac,
        )
        try:
            with open(event_cache_path, "w") as f:
                json.dump([vars(e) for e in events], f)
            _print(f"  Cached {len(events)} events to {event_cache_path}")
        except IOError:
            pass

    _print(f"  Found {len(events)} events")
    label_counts = defaultdict(int)
    for ev in events:
        label_counts[ev.label] += 1
    for lbl, cnt in sorted(label_counts.items(), key=lambda x: -x[1])[:20]:
        _print(f"    {lbl}: {cnt}")
    if len(label_counts) > 20:
        _print(f"    ... and {len(label_counts) - 20} more categories")

    # --- Step 3 & 4: Extract frames + masks (multiprocess with checkpoint) ---
    _close_all_envs()

    if n_workers <= 0:
        n_workers = min(os.cpu_count() or 4, 16)
    _print(f"[3/5] Extracting frames and masks (n_frames={n_frames}, "
           f"skip_tail={skip_tail}, workers={n_workers})...")

    # Sort by (seg_part, img_part) for LMDB locality within each worker
    events_sorted = sorted(events, key=lambda e: (e.seg_part, e.img_part))

    # Determine actual number of batches
    actual_workers = max(1, min(n_workers, len(events_sorted)))
    if len(events_sorted) <= 100:
        actual_workers = 1
    chunk_size = (len(events_sorted) + actual_workers - 1) // actual_workers

    # Check existing checkpoints for resume
    loaded_anns: Dict[int, List[Dict]] = {}
    loaded_stats: Dict[int, Dict] = {}
    pending_workers = []

    for wi in range(actual_workers):
        start = wi * chunk_size
        end = min(start + chunk_size, len(events_sorted))
        if start >= len(events_sorted):
            break
        cached = _load_checkpoint(output_dir, wi)
        if cached is not None:
            loaded_anns[wi] = cached[0]
            loaded_stats[wi] = cached[1]
        else:
            pending_workers.append(wi)

    if loaded_anns:
        _print(f"  Resumed {len(loaded_anns)}/{actual_workers} workers "
               f"from checkpoint "
               f"({sum(len(a) for a in loaded_anns.values())} annotations)")

    all_annotations = []
    stats = defaultdict(int)
    t_start = time.time()

    # Merge already-completed workers
    for wi in sorted(loaded_anns):
        all_annotations.extend(loaded_anns[wi])
        for k, v in loaded_stats[wi].items():
            stats[k] += v

    if not pending_workers:
        _print(f"  All workers completed from checkpoint, skipping Step 3")
    elif actual_workers > 1:
        batches = []
        for wi in pending_workers:
            start = wi * chunk_size
            end = min(start + chunk_size, len(events_sorted))
            batches.append((
                events_sorted[start:end], img_dir, mask_hw,
                n_frames, skip_tail, min_mask_area, max_mask_area_frac,
                filter_gui, wi, actual_workers, output_dir,
            ))

        _print(f"  Dispatching {sum(len(b[0]) for b in batches)} events to "
               f"{len(batches)} workers "
               f"(~{chunk_size} events each)")

        import multiprocessing as mp
        ctx = mp.get_context("fork")
        with ctx.Pool(len(batches)) as pool:
            results = pool.map(_process_events_worker, batches)

        for worker_anns, worker_stats in results:
            all_annotations.extend(worker_anns)
            for k, v in worker_stats.items():
                stats[k] += v
    else:
        _print(f"  Single-process mode ({len(events_sorted)} events)")
        result = _process_events_worker((
            events_sorted, img_dir, mask_hw,
            n_frames, skip_tail, min_mask_area, max_mask_area_frac,
            filter_gui, 0, 1, output_dir,
        ))
        all_annotations.extend(result[0])
        for k, v in result[1].items():
            stats[k] += v

    elapsed_total = time.time() - t_start
    total_ev = len(events_sorted)
    rate = total_ev / elapsed_total if elapsed_total > 0 else 0
    _print(f"  Done in {elapsed_total:.1f}s ({rate:.1f} ev/s overall)")
    _print(f"  Processed {stats.get('events_processed', 0)} events")
    _print(f"  Exported {stats.get('frames_exported', 0)} frames")
    _print(f"  No valid frames: {stats.get('events_no_valid_frames', 0)} events")
    _print(f"  GUI filtered: {stats.get('frames_gui_filtered', 0)} frames")
    _print(f"  No mask: {stats.get('frames_no_mask', 0)} frames")
    _print(f"  No image data: {stats.get('frames_no_image', 0)} frames")

    # --- Step 4: COCO output ---
    _print(f"[4/5] Building COCO annotations...")
    coco = build_coco_output(all_annotations)
    coco_path = os.path.join(output_dir, "annotations.json")
    with open(coco_path, "w") as f:
        json.dump(coco, f, indent=2)
    _print(f"  {len(coco['images'])} images, "
           f"{len(coco['annotations'])} annotations, "
           f"{len(coco['categories'])} categories")

    # Category breakdown
    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    cat_counts = defaultdict(int)
    for ann in coco["annotations"]:
        cat_counts[cat_id_to_name[ann["category_id"]]] += 1
    for name, cnt in sorted(cat_counts.items(), key=lambda x: -x[1])[:30]:
        _print(f"    {name}: {cnt}")

    # --- Visualization ---
    if max_visualize > 0:
        _print(f"[5/5] Generating {max_visualize} visualizations...")
        vis_dir = os.path.join(output_dir, "vis")
        os.makedirs(vis_dir, exist_ok=True)
        count = 0
        for ann in all_annotations:
            if count >= max_visualize:
                break
            img_path = os.path.join(img_dir, ann["image_file"])
            if not os.path.exists(img_path):
                continue
            img = cv2.imread(img_path)
            if img is None:
                continue
            x, y, w, h = [int(v) for v in ann["bbox"]]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, ann["category"], (x, max(y - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            vis_path = os.path.join(vis_dir, f"vis_{count:04d}_{ann['category']}.png")
            cv2.imwrite(vis_path, img)
            count += 1
        _print(f"  Saved {count} visualizations to {vis_dir}")
    else:
        _print("[5/5] Skipping visualization (use --visualize N)")

    summary = {
        "data_roots": data_roots,
        "output_dir": output_dir,
        "n_frames_per_event": n_frames,
        "skip_tail": skip_tail,
        "min_mask_area": min_mask_area,
        "max_mask_area_frac": max_mask_area_frac,
        "total_events": len(events),
        "total_images": len(coco["images"]),
        "total_annotations": len(coco["annotations"]),
        "total_categories": len(coco["categories"]),
        "category_counts": dict(cat_counts),
        "stats": dict(stats),
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # All steps done successfully — clean up checkpoint files
    _clean_checkpoints(output_dir, actual_workers)
    _print("  Cleaned up checkpoint files")

    return summary


# ---------------------------------------------------------------------------
# Preview / quality report
# ---------------------------------------------------------------------------

def generate_quality_report(output_dir: str, all_annotations: List[Dict],
                            max_per_cat: int = 6) -> str:
    """Generate a visual quality report: grid of sample images per category.

    Returns path to the saved report image.
    """
    img_dir = os.path.join(output_dir, "images")

    # Group annotations by category
    by_cat: Dict[str, List[Dict]] = defaultdict(list)
    for ann in all_annotations:
        by_cat[ann["category"]].append(ann)

    cats = sorted(by_cat.keys(), key=lambda c: -len(by_cat[c]))
    n_cats = min(len(cats), 20)
    if n_cats == 0:
        return ""

    cols = max_per_cat
    rows = n_cats
    cell_w, cell_h = 240, 240
    margin = 120  # left margin for category labels
    grid = np.zeros((rows * cell_h, margin + cols * cell_w, 3), dtype=np.uint8)

    for ri, cat in enumerate(cats[:n_cats]):
        # Category label
        y_text = ri * cell_h + cell_h // 2
        label = f"{cat} ({len(by_cat[cat])})"
        cv2.putText(grid, label, (5, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        samples = by_cat[cat][:max_per_cat]
        for ci, ann in enumerate(samples):
            img_path = os.path.join(img_dir, ann["image_file"])
            if not os.path.exists(img_path):
                continue
            img = cv2.imread(img_path)
            if img is None:
                continue

            # Draw bbox
            x, y, w, h = [int(v) for v in ann["bbox"]]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Resize to cell
            thumb = cv2.resize(img, (cell_w, cell_h),
                               interpolation=cv2.INTER_AREA)
            # Bbox size info
            bw, bh = ann["bbox"][2], ann["bbox"][3]
            info = f"{int(bw)}x{int(bh)}"
            cv2.putText(thumb, info, (5, cell_h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

            grid[ri * cell_h:(ri + 1) * cell_h,
                 margin + ci * cell_w:margin + (ci + 1) * cell_w] = thumb

    report_path = os.path.join(output_dir, "quality_report.png")
    cv2.imwrite(report_path, grid)

    # Text summary
    bbox_widths = [a["bbox"][2] for a in all_annotations]
    bbox_heights = [a["bbox"][3] for a in all_annotations]
    summary_lines = [
        f"Total: {len(all_annotations)} annotations, "
        f"{len(set(a['image_file'] for a in all_annotations))} images, "
        f"{len(by_cat)} categories",
        f"Bbox size: width {np.min(bbox_widths):.0f}-{np.max(bbox_widths):.0f} "
        f"(median {np.median(bbox_widths):.0f}), "
        f"height {np.min(bbox_heights):.0f}-{np.max(bbox_heights):.0f} "
        f"(median {np.median(bbox_heights):.0f})",
        f"Top categories: " + ", ".join(
            f"{c}({len(by_cat[c])})" for c in cats[:10]),
    ]
    report_txt = os.path.join(output_dir, "quality_report.txt")
    with open(report_txt, "w") as f:
        f.write("\n".join(summary_lines))

    return report_path


def preview_dataset(
    data_roots: List[str],
    output_dir: str,
    n_events_per_cat: int = 5,
    n_frames: int = 4,
    skip_tail: int = 4,
    min_mask_area: int = 3000,
    max_mask_area_frac: float = 0.35,
    filter_gui: bool = True,
) -> None:
    """Quick preview mode: sample a few events per category to evaluate quality.

    Runs single-process on a small subset for fast parameter tuning.
    Generates a visual report grid.
    """
    os.makedirs(output_dir, exist_ok=True)
    img_dir = os.path.join(output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    mask_hw = (360, 640)

    _print(f"=== PREVIEW MODE ({n_events_per_cat} events/category) ===")
    _print(f"  Params: n_frames={n_frames}, skip_tail={skip_tail}, "
           f"min_mask_area={min_mask_area}, "
           f"max_mask_area={max_mask_area_frac:.0%}")

    # Step 1-2: reuse normal pipeline
    _print("[1/3] Building maps and scanning events...")
    seg_ep_map, img_ep_map = _build_episode_maps(data_roots)

    cache_key = hashlib.md5(
        f"{sorted(data_roots)}|{min_mask_area}|{max_mask_area_frac}".encode()
    ).hexdigest()[:12]
    event_cache_path = f"/tmp/lmdb_events_{cache_key}.json"

    events = None
    if os.path.exists(event_cache_path):
        try:
            with open(event_cache_path) as f:
                events = [EventRecord(**e) for e in json.load(f)]
            _print(f"  Loaded {len(events)} events from cache")
        except (json.JSONDecodeError, TypeError):
            events = None
    if events is None:
        events = _scan_events(
            data_roots, seg_ep_map, img_ep_map,
            min_mask_area=min_mask_area,
            max_mask_area_frac=max_mask_area_frac)
        try:
            with open(event_cache_path, "w") as f:
                json.dump([vars(e) for e in events], f)
        except IOError:
            pass

    # Sample events: up to n_events_per_cat per category
    import random
    by_cat: Dict[str, List[EventRecord]] = defaultdict(list)
    for ev in events:
        by_cat[ev.label].append(ev)
    sampled = []
    for cat, cat_events in sorted(by_cat.items(), key=lambda x: -len(x[1])):
        sample = random.sample(cat_events, min(n_events_per_cat, len(cat_events)))
        sampled.extend(sample)
    _print(f"  Sampled {len(sampled)} events from {len(by_cat)} categories")

    # Step 2: Process sampled events (single process)
    _print(f"[2/3] Processing {len(sampled)} events...")
    _close_all_envs()
    result = _process_events_worker((
        sampled, img_dir, mask_hw,
        n_frames, skip_tail, min_mask_area, max_mask_area_frac,
        filter_gui, 0, 1, output_dir,
    ))
    all_annotations, rstats = result

    _print(f"  Exported {rstats.get('frames_exported', 0)} frames")
    _print(f"  GUI filtered: {rstats.get('frames_gui_filtered', 0)}")
    _print(f"  No mask: {rstats.get('frames_no_mask', 0)}")
    _print(f"  No image: {rstats.get('frames_no_image', 0)}")

    # Step 3: Generate quality report
    _print("[3/3] Generating quality report...")
    if all_annotations:
        report_path = generate_quality_report(
            output_dir, all_annotations, max_per_cat=6)
        _print(f"\n  Report saved: {report_path}")
        _print(f"  Open this image to visually inspect bbox quality.")

        # Bbox statistics
        widths = [a["bbox"][2] for a in all_annotations]
        heights = [a["bbox"][3] for a in all_annotations]
        areas = [w * h for w, h in zip(widths, heights)]
        _print(f"\n  Bbox stats:")
        _print(f"    Width:  min={min(widths):.0f}  median={np.median(widths):.0f}  "
               f"max={max(widths):.0f}")
        _print(f"    Height: min={min(heights):.0f}  median={np.median(heights):.0f}  "
               f"max={max(heights):.0f}")
        _print(f"    Area:   min={min(areas):.0f}  median={np.median(areas):.0f}  "
               f"max={max(areas):.0f}")

        ann_by_cat = defaultdict(int)
        for a in all_annotations:
            ann_by_cat[a["category"]] += 1
        _print(f"\n  Per-category breakdown:")
        for cat, cnt in sorted(ann_by_cat.items(), key=lambda x: -x[1]):
            _print(f"    {cat}: {cnt} annotations")
    else:
        _print("  WARNING: No annotations generated. Check parameters.")

    # Cleanup preview checkpoint
    cp = _checkpoint_path(output_dir, 0)
    if os.path.exists(cp):
        os.remove(cp)


def main():
    parser = argparse.ArgumentParser(
        description="Build GroundingDINO fine-tuning dataset from MineStudio LMDBs"
    )
    parser.add_argument("--data-root", nargs="+", required=True,
                        help="One or more MineStudio dataset roots "
                             "(each with image/ and segmentation/ subdirs)")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for COCO dataset")
    parser.add_argument("--n-frames", type=int, default=4,
                        help="Frames to select per event (default: 4)")
    parser.add_argument("--skip-tail", type=int, default=4,
                        help="Skip N frames at end of tracking window (default: 4)")
    parser.add_argument("--min-mask-area", type=int, default=3000,
                        help="Min mask area in pixels (default: 3000)")
    parser.add_argument("--max-mask-area", type=float, default=0.35,
                        help="Max mask area as fraction of image (default: 0.35)")
    parser.add_argument("--no-gui-filter", action="store_true",
                        help="Disable GUI frame detection")
    parser.add_argument("--visualize", type=int, default=0,
                        help="Number of visualization samples (default: 0)")
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of parallel workers (0=auto, 1=single-process)")
    parser.add_argument("--preview", type=int, default=0, metavar="N",
                        help="Preview mode: sample N events per category for "
                             "quick quality evaluation (e.g. --preview 5)")

    args = parser.parse_args()

    if args.preview > 0:
        preview_dataset(
            data_roots=args.data_root,
            output_dir=args.output_dir,
            n_events_per_cat=args.preview,
            n_frames=args.n_frames,
            skip_tail=args.skip_tail,
            min_mask_area=args.min_mask_area,
            max_mask_area_frac=args.max_mask_area,
            filter_gui=not args.no_gui_filter,
        )
        return

    summary = build_dataset(
        data_roots=args.data_root,
        output_dir=args.output_dir,
        n_frames=args.n_frames,
        skip_tail=args.skip_tail,
        min_mask_area=args.min_mask_area,
        max_mask_area_frac=args.max_mask_area,
        filter_gui=not args.no_gui_filter,
        max_visualize=args.visualize,
        n_workers=args.workers,
    )

    _print("\n" + "=" * 60)
    _print("Dataset build complete!")
    _print(f"  Images:      {summary['total_images']}")
    _print(f"  Annotations: {summary['total_annotations']}")
    _print(f"  Categories:  {summary['total_categories']}")
    _print("=" * 60)


if __name__ == "__main__":
    os.environ["OPENCV_LOG_LEVEL"] = "FATAL"
    os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
    main()
