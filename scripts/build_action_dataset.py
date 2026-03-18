"""Build fine-tuning dataset using frame backtracking from block-break events.

Core hypothesis:
  mine_block events fire when a block is destroyed.  At that moment the
  player's crosshair (screen center) points at the block's position.
  Going back N frames (≈ mining animation duration) the block is still
  intact and visible at screen center — exactly the image we need for
  object detection training.

Supported frame-selection modes:
  backtrack   (NEW, recommended)
      Go back from the event frame using *original* video coordinates
      (ori_frame_range).  No action LMDB required.
  early_range
      First N frames of the SAM2 tracking window (frame_range).
  preattack
      N frames before the continuous attack=1 sequence (needs action LMDB).
"""
from __future__ import annotations

import argparse
import json
import lmdb
import numpy as np
import os
import pickle
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.build_finetune_dataset import (
    parse_event_label,
    build_raw_video_index,
    build_episode_chunk_info,
    decode_multichunk_frame,
    detect_gui_frame,
    mask_to_bbox,
    filter_mask_by_point,
    rle_to_mask,
    scale_bbox,
    SKIP_ACTIONS,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EventInfo:
    """A single mining/use event extracted from segmentation LMDB."""
    label: str
    event_name: str
    episode_name: str
    episode_idx: int
    seg_partition: str
    event_frame: int
    frame_range: Tuple[int, int]
    ori_frame_range: Optional[Tuple[int, int]] = None


@dataclass
class AnnotatedFrame:
    """A selected pre-attack frame with its annotation."""
    image_id: int
    frame_id: int
    episode_name: str
    label: str
    event_name: str
    bbox: Optional[List[int]] = None
    mask_area: int = 0
    img_filename: str = ""


# ---------------------------------------------------------------------------
# Segmentation mask lookup (reuse existing MineStudio masks)
# ---------------------------------------------------------------------------

def lookup_seg_mask(
    seg_partition: str,
    episode_idx: int,
    frame_id: int,
    event_name: str,
    mask_height: int = 360,
    mask_width: int = 640,
    chunk_size: int = 32,
) -> Optional[np.ndarray]:
    """Look up an existing RLE mask from segmentation LMDB for a specific frame.

    For early_range mode, the selected frames are within the SAM2 tracking
    window (frame_range), so masks already exist in the segmentation data.
    Returns the decoded binary mask or None if not found.
    """
    chunk_offset = (frame_id // chunk_size) * chunk_size
    local_idx = frame_id - chunk_offset

    env = lmdb.open(seg_partition, readonly=True, lock=False,
                    readahead=False, map_size=1024**3 * 100)
    try:
        key = f"({episode_idx}, {chunk_offset})"
        with env.begin() as txn:
            raw = txn.get(key.encode())
            if raw is None:
                return None
            frames = pickle.loads(raw)
            if local_idx >= len(frames):
                return None
            frame_dict = frames[local_idx]
            for event_key, event_val in frame_dict.items():
                if not isinstance(event_val, dict):
                    continue
                ev_name = event_val.get("event", "")
                if ev_name == event_name or event_name in str(event_key):
                    rle = event_val.get("rle_mask", "")
                    if rle:
                        mask = rle_to_mask(rle, mask_height, mask_width)
                        point = event_val.get("point")
                        if point is not None:
                            mask = filter_mask_by_point(mask, point)
                        return mask
    finally:
        env.close()
    return None


# ---------------------------------------------------------------------------
# Action index
# ---------------------------------------------------------------------------

def build_action_index(action_root: str) -> Dict[str, Tuple[str, int]]:
    """Map episode_name → (lmdb_path, episode_idx) across all action partitions.

    Action partitions use different numbering than segmentation partitions,
    so we must search all partitions by episode name.
    """
    index: Dict[str, Tuple[str, int]] = {}
    action_dir = Path(action_root)
    if not action_dir.exists():
        return index

    for part_dir in sorted(action_dir.iterdir()):
        if not part_dir.is_dir() or not (part_dir / "data.mdb").exists():
            continue
        env = lmdb.open(str(part_dir), readonly=True, lock=False,
                        readahead=False, map_size=1024**3 * 10)
        try:
            with env.begin() as txn:
                raw = txn.get("__chunk_infos__".encode())
                if raw is None:
                    continue
                infos = pickle.loads(raw)
                for info in infos:
                    ep_name = info["episode"]
                    ep_idx = info["episode_idx"]
                    index[ep_name] = (str(part_dir), ep_idx)
        finally:
            env.close()
    return index


def read_action_chunk(
    lmdb_path: str,
    episode_idx: int,
    frame_offset: int,
) -> Optional[Dict[str, np.ndarray]]:
    """Read a single 32-frame action chunk from LMDB."""
    env = lmdb.open(lmdb_path, readonly=True, lock=False,
                    readahead=False, map_size=1024**3 * 10)
    try:
        key = f"({episode_idx}, {frame_offset})"
        with env.begin() as txn:
            raw = txn.get(key.encode())
            if raw is None:
                return None
            return pickle.loads(raw)
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Attack tracing
# ---------------------------------------------------------------------------

def find_attack_start(
    lmdb_path: str,
    episode_idx: int,
    trace_from: int,
    chunk_size: int = 32,
    max_lookback: int = 2000,
) -> Optional[int]:
    """Trace backwards from trace_from to find where continuous attack=1 began.

    Returns the global frame number where attack first became 1, or None
    if no action data is available.
    """
    env = lmdb.open(lmdb_path, readonly=True, lock=False,
                    readahead=False, map_size=1024**3 * 10)
    try:
        start_chunk_offset = (trace_from // chunk_size) * chunk_size
        min_frame = max(0, trace_from - max_lookback)

        with env.begin() as txn:
            for chunk_offset in range(start_chunk_offset, min_frame - chunk_size, -chunk_size):
                if chunk_offset < 0:
                    break
                key = f"({episode_idx}, {chunk_offset})"
                raw = txn.get(key.encode())
                if raw is None:
                    return None

                data = pickle.loads(raw)
                attack = data.get("attack")
                if attack is None:
                    return None

                for i in range(min(len(attack) - 1, trace_from - chunk_offset), -1, -1):
                    global_frame = chunk_offset + i
                    if global_frame > trace_from:
                        continue
                    if attack[i] == 0:
                        return global_frame + 1

            return 0
    finally:
        env.close()


def find_attack_start_for_event(
    lmdb_path: str,
    episode_idx: int,
    event_frame: int,
    frame_range: Tuple[int, int],
    chunk_size: int = 32,
    max_gap: int = 50,
) -> Optional[int]:
    """Find attack start relevant to THIS specific event.

    For continuous mining chains, the raw attack_start may be far back
    (aimed at a different block). We use frame_range[0] as the anchor:
    - Trace backwards from frame_range[0] (when SAM2 started tracking this block)
    - If the gap between attack_start and frame_range[0] is reasonable (<=max_gap),
      use that attack_start
    - Otherwise, use frame_range[0] as the reference (the block just became the target)
    """
    fr_start = frame_range[0]

    attack_start = find_attack_start(
        lmdb_path, episode_idx, fr_start, chunk_size)

    if attack_start is None:
        return None

    if attack_start > fr_start:
        return fr_start

    if (fr_start - attack_start) <= max_gap:
        return attack_start

    return fr_start


def select_preattack_frames(
    attack_start: int,
    n_before: int = 8,
) -> List[int]:
    """Select N frames immediately before the attack/tracking started.

    These frames have:
    - No GUI open (must be closed to mine)
    - Crosshair pointing at target (player aiming)
    - Block intact (no mining cracks)
    """
    start = max(0, attack_start - n_before)
    return list(range(start, attack_start))


def select_early_range_frames(
    frame_range: Tuple[int, int],
    n_frames: int = 8,
) -> List[int]:
    """Select first N frames from the SAM2 tracking window.

    frame_range[0] is when SAM2 first tracked the target object (confirmed
    visible at crosshair). Early frames in this window have:
    - Target block visible and at crosshair center
    - No or minimal mining cracks (mining just started)
    - No GUI (player is in gameplay)

    More reliable than pre-attack frames for continuous mining chains
    where the player doesn't release attack between blocks.
    """
    fr_start, fr_end = frame_range
    end = min(fr_start + n_frames, fr_end)
    return list(range(fr_start, end))


def select_backtrack_frames(
    ori_frame_range: Tuple[int, int],
    n_frames: int = 4,
    skip_tail: int = 8,
) -> List[int]:
    """Select frames by backtracking from the block-break event.

    ori_frame_range uses original video coordinates:
      [0] = earliest frame where SAM2 saw the block
      [1] = the frame when the block broke (mine_block event)

    Strategy: go back from the event frame, skipping the tail frames
    (which show mining cracks / break animation) and selecting N frames
    where the block should be intact.

    With default skip_tail=8 (~0.4s at 20fps) and n_frames=4:
      event at frame 613 → select frames 601..604
      (skip 609-612 = break/crack frames)
    """
    event_frame = ori_frame_range[1]
    tracking_start = ori_frame_range[0]

    end = max(event_frame - skip_tail, tracking_start)
    start = max(end - n_frames, tracking_start)

    if start >= end:
        return []
    return list(range(start, end))


# ---------------------------------------------------------------------------
# Event extraction from segmentation
# ---------------------------------------------------------------------------

def extract_events_from_segmentation(
    data_root: str,
) -> List[EventInfo]:
    """Extract unique events from segmentation LMDB.

    For each event, we take the LAST occurrence (highest frame) as the
    definitive event frame, since that's when the block actually breaks.
    """
    seg_dir = Path(data_root) / "segmentation"
    if not seg_dir.exists():
        return []

    episode_maps: Dict[str, Dict[int, str]] = {}
    events_by_key: Dict[Tuple[str, str, int], EventInfo] = {}

    for part_dir in sorted(seg_dir.iterdir()):
        if not part_dir.is_dir() or not (part_dir / "data.mdb").exists():
            continue
        part_path = str(part_dir)

        env = lmdb.open(part_path, readonly=True, lock=False,
                        readahead=False, map_size=1024**3 * 100)
        try:
            with env.begin() as txn:
                raw = txn.get("__chunk_infos__".encode())
                if raw is None:
                    continue
                infos = pickle.loads(raw)
                ep_map = {info["episode_idx"]: info["episode"] for info in infos}
                episode_maps[part_path] = ep_map

                for key_raw, val_raw in txn.cursor():
                    key_str = key_raw.decode()
                    if key_str.startswith("__"):
                        continue

                    chunk_key = eval(key_str)
                    episode_idx = chunk_key[0]
                    episode_name = ep_map.get(episode_idx)
                    if episode_name is None:
                        continue

                    frames = pickle.loads(val_raw)
                    for fi, frame_dict in enumerate(frames):
                        for event_key, event_val in frame_dict.items():
                            if not isinstance(event_key, tuple):
                                continue
                            if not isinstance(event_val, dict):
                                continue

                            event_name = event_val.get("event", "")
                            label = parse_event_label(event_name)
                            if label is None:
                                continue

                            frame_range = event_val.get("frame_range")
                            if frame_range is None:
                                continue

                            ori_frame_range = event_val.get("ori_frame_range")
                            event_frame = frame_range[1]
                            dedup_key = (episode_name, event_name, event_frame)

                            if dedup_key not in events_by_key:
                                events_by_key[dedup_key] = EventInfo(
                                    label=label,
                                    event_name=event_name,
                                    episode_name=episode_name,
                                    episode_idx=episode_idx,
                                    seg_partition=part_path,
                                    event_frame=event_frame,
                                    frame_range=frame_range,
                                    ori_frame_range=tuple(ori_frame_range) if ori_frame_range else None,
                                )
        finally:
            env.close()

    events = sorted(events_by_key.values(),
                    key=lambda e: (e.episode_name, e.event_frame))
    return events


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_action_dataset(
    data_root: str,
    output_dir: str,
    raw_video_dir: str,
    action_root: Optional[str] = None,
    n_frames: int = 4,
    frame_mode: str = "backtrack",
    max_visualize: int = 0,
    sam2_model: Optional[str] = None,
    **kwargs,
) -> dict:
    """Build fine-tuning dataset using action-based frame selection.

    Steps:
    1. Extract events from segmentation LMDB
    2. Build action index and trace attack start for each event
    3. Select pre-attack frames
    4. Decode video frames
    5. (Optional) Run SAM2 for mask generation
    6. Output COCO-format dataset
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

    if action_root is None:
        action_root = os.path.join(data_root, "action")

    # Step 1: Extract events
    print("[1/6] Extracting events from segmentation ...")
    events = extract_events_from_segmentation(data_root)
    print(f"  Found {len(events)} unique events")
    cat_counts = defaultdict(int)
    for ev in events:
        cat_counts[ev.label] += 1
    print(f"  Categories: {', '.join(f'{k}({v})' for k, v in sorted(cat_counts.items()))}")

    if not events:
        print("  No events found!")
        return {"total_events": 0}

    # Step 2: Build action index (only needed for preattack mode)
    action_index: Dict[str, Tuple[str, int]] = {}
    if frame_mode == "preattack":
        print("[2/6] Building action index ...")
        action_index = build_action_index(action_root)
        print(f"  Action episodes: {len(action_index)}")
        matched = sum(1 for ev in events if ev.episode_name in action_index)
        print(f"  Events with action data: {matched}/{len(events)}")
    else:
        print("[2/6] Action index not needed for mode={frame_mode}, skipping")

    # Step 3: Build raw video index
    print("[3/6] Building raw video index ...")
    raw_video_index = build_raw_video_index(raw_video_dir)
    print(f"  Found {len(raw_video_index)} raw MP4 files")

    # Step 4: For each event, select frames
    print(f"[4/6] Selecting frames (mode={frame_mode}, n={n_frames}) ...")
    annotated_frames: List[AnnotatedFrame] = []
    stats = {
        "no_action_data": 0,
        "no_attack_found": 0,
        "no_ori_range": 0,
        "no_frames": 0,
        "selected": 0,
        "decode_ok": 0,
        "decode_fail": 0,
        "gui_filtered": 0,
        "no_mask": 0,
        "mask_too_small": 0,
        "mask_too_large": 0,
    }
    ann_id = 0

    skip_tail = kwargs.get("skip_tail", 8)

    for ev in events:
        if frame_mode == "backtrack":
            if ev.ori_frame_range is None:
                stats["no_ori_range"] += 1
                continue
            selected = select_backtrack_frames(
                ev.ori_frame_range, n_frames, skip_tail)
        elif frame_mode == "early_range":
            selected = select_early_range_frames(ev.frame_range, n_frames)
        elif frame_mode == "preattack":
            act_entry = action_index.get(ev.episode_name)
            if act_entry is None:
                stats["no_action_data"] += 1
                continue
            act_lmdb_path, act_ep_idx = act_entry
            attack_start = find_attack_start_for_event(
                act_lmdb_path, act_ep_idx, ev.event_frame, ev.frame_range)
            if attack_start is None:
                stats["no_attack_found"] += 1
                continue
            selected = select_preattack_frames(attack_start, n_frames)
        else:
            raise ValueError(f"Unknown frame_mode: {frame_mode}")

        if not selected:
            stats["no_frames"] += 1
            continue

        stats["selected"] += len(selected)

        for fid in selected:
            img_filename = (f"{ev.label}_{ev.episode_name}_"
                           f"f{fid:06d}_ev{ev.event_frame}.png")
            img_path = os.path.join(output_dir, "images", img_filename)

            frame = decode_multichunk_frame(
                raw_video_index, ev.episode_name, fid)
            if frame is None:
                stats["decode_fail"] += 1
                continue

            cv2.imwrite(img_path, frame)
            stats["decode_ok"] += 1

            if detect_gui_frame(img_path):
                os.remove(img_path)
                stats["gui_filtered"] += 1
                continue

            bbox = None
            mask_area = 0
            img_h, img_w = frame.shape[:2]
            total_pixels = img_h * img_w

            if frame_mode == "early_range":
                mask = lookup_seg_mask(
                    ev.seg_partition, ev.episode_idx, fid,
                    ev.event_name)
                if mask is not None and mask.any():
                    bbox = mask_to_bbox(mask)
                    mask_area = int(mask.sum())
                    area_ratio = mask_area / total_pixels
                    if mask_area < 500:
                        bbox = None
                        stats["mask_too_small"] += 1
                        os.remove(img_path)
                        continue
                    if area_ratio > 0.35:
                        bbox = None
                        stats["mask_too_large"] += 1
                        os.remove(img_path)
                        continue

            if bbox is None and frame_mode == "early_range":
                stats["no_mask"] += 1
                os.remove(img_path)
                continue

            af = AnnotatedFrame(
                image_id=ann_id,
                frame_id=fid,
                episode_name=ev.episode_name,
                label=ev.label,
                event_name=ev.event_name,
                img_filename=img_filename,
                bbox=bbox,
                mask_area=mask_area,
            )
            annotated_frames.append(af)
            ann_id += 1

    if stats["no_ori_range"]:
        print(f"  Events without ori_frame_range: {stats['no_ori_range']}")
    if stats["no_action_data"]:
        print(f"  Events without action data: {stats['no_action_data']}")
    if stats["no_attack_found"]:
        print(f"  Events without attack found: {stats['no_attack_found']}")
    print(f"  Events with no frames: {stats['no_frames']}")
    print(f"  Selected frames: {stats['selected']}")
    print(f"  Decoded OK: {stats['decode_ok']}")
    print(f"  Decode failed: {stats['decode_fail']}")
    print(f"  GUI filtered: {stats['gui_filtered']}")
    if frame_mode == "early_range":
        print(f"  No mask found: {stats.get('no_mask', 0)}")
        print(f"  Mask too small: {stats.get('mask_too_small', 0)}")
        print(f"  Mask too large: {stats.get('mask_too_large', 0)}")
    print(f"  Final annotations: {len(annotated_frames)}")

    # Step 5: Generate bboxes (from masks already loaded, or SAM2/placeholder)
    if frame_mode == "early_range":
        print("[5/6] Masks loaded from segmentation LMDB (no SAM2 needed)")
    elif sam2_model:
        print("[5/6] Running SAM2 ...")
        _run_sam2_inference(output_dir, annotated_frames, sam2_ckpt=sam2_model)
    else:
        crop = kwargs.get("crop_fraction", 0.25)
        print(f"[5/6] Generating center-crop bboxes ({crop:.0%} of image)")
        _generate_center_bboxes(output_dir, annotated_frames, crop_fraction=crop)

    # Step 6: Generate COCO JSON
    print("[6/6] Generating COCO JSON ...")
    coco = _to_coco(annotated_frames)
    coco_path = os.path.join(output_dir, "annotations.json")
    with open(coco_path, "w") as f:
        json.dump(coco, f, indent=2)

    cat_summary = defaultdict(int)
    for af in annotated_frames:
        cat_summary[af.label] += 1

    print(f"  Saved to {coco_path}")
    print(f"  {len(coco['images'])} images, {len(coco['annotations'])} annotations, "
          f"{len(coco['categories'])} categories")

    # Visualization and quality report
    if max_visualize > 0:
        _visualize_samples(output_dir, annotated_frames, max_visualize)
    _generate_quality_report(output_dir, annotated_frames)

    summary = {
        "total_events": len(events),
        "total_annotations": len(annotated_frames),
        "categories": dict(cat_summary),
        **stats,
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# ---------------------------------------------------------------------------
# Bbox / mask helpers
# ---------------------------------------------------------------------------

def _generate_center_bboxes(
    output_dir: str,
    frames: List[AnnotatedFrame],
    crop_fraction: float = 0.3,
):
    """Generate placeholder center-crop bboxes (used when SAM2 is not available).

    This provides a reasonable initial bbox centered on the crosshair.
    """
    for af in frames:
        img_path = os.path.join(output_dir, "images", af.img_filename)
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        bw = int(w * crop_fraction)
        bh = int(h * crop_fraction)
        x = (w - bw) // 2
        y = (h - bh) // 2
        af.bbox = [x, y, bw, bh]


def _run_sam2_inference(
    output_dir: str,
    frames: List[AnnotatedFrame],
    sam2_ckpt: str,
    sam2_cfg: str = "configs/sam2.1/sam2.1_hiera_b+.yaml",
    min_mask_area: int = 500,
    max_mask_ratio: float = 0.35,
):
    """Run SAM2 point-prompt inference on each frame.

    For Minecraft scenes, SAM2's multi-mask output behaves:
    - mask[0]: Largest region (wall/surface), low score but most useful
    - mask[1-2]: Tiny (just the crosshair pixel), high score but useless

    Strategy: Always use mask[0], then filter by connected components
    (keeping only the component touching the center point).
    """
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError:
        print("  WARNING: sam2 not installed, falling back to center bboxes")
        _generate_center_bboxes(output_dir, frames)
        return

    import torch, time
    device = "mps"
    print(f"  Loading SAM2 on {device} ...")
    model = build_sam2(sam2_cfg, sam2_ckpt, device=device)
    predictor = SAM2ImagePredictor(model)

    processed = 0
    skipped_small = 0
    skipped_large = 0
    kept = 0
    total = len(frames)
    t0 = time.time()

    to_remove = []

    for idx, af in enumerate(frames):
        if idx % 50 == 0 and idx > 0:
            elapsed = time.time() - t0
            fps = idx / elapsed
            eta = (total - idx) / fps
            print(f"  [{idx}/{total}] {fps:.1f} img/s, ETA {eta:.0f}s | "
                  f"kept={kept} small={skipped_small} large={skipped_large}",
                  flush=True)

        img_path = os.path.join(output_dir, "images", af.img_filename)
        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        total_pixels = h * w
        cx, cy = w // 2 + 2, h // 2 + 2

        predictor.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        masks, scores, _ = predictor.predict(
            point_coords=np.array([[cx, cy]], dtype=np.float32),
            point_labels=np.array([1], dtype=np.int32),
            multimask_output=True,
        )

        mask = masks[0].astype(np.uint8)
        mask = filter_mask_by_point(mask, (cy, cx))
        area = int(mask.sum())
        bbox = mask_to_bbox(mask)
        processed += 1

        if bbox is None or area < min_mask_area:
            skipped_small += 1
            to_remove.append(af)
            continue

        if area / total_pixels > max_mask_ratio:
            skipped_large += 1
            to_remove.append(af)
            continue

        af.bbox = bbox
        af.mask_area = area
        kept += 1

    for af in to_remove:
        frames.remove(af)
        img_path = os.path.join(output_dir, "images", af.img_filename)
        if os.path.exists(img_path):
            os.remove(img_path)

    elapsed = time.time() - t0
    print(f"  SAM2 done in {elapsed:.0f}s ({processed/elapsed:.1f} img/s): "
          f"kept {kept}, small {skipped_small}, large {skipped_large}")


# ---------------------------------------------------------------------------
# COCO output
# ---------------------------------------------------------------------------

def _to_coco(frames: List[AnnotatedFrame]) -> dict:
    """Convert annotated frames to COCO format."""
    categories = {}
    images = []
    annotations = []
    seen_images = set()

    for af in frames:
        if af.bbox is None:
            continue

        if af.label not in categories:
            categories[af.label] = len(categories) + 1

        if af.img_filename not in seen_images:
            seen_images.add(af.img_filename)
            images.append({
                "id": af.image_id,
                "file_name": af.img_filename,
                "width": 640,
                "height": 360,
            })

        x, y, w, h = af.bbox
        annotations.append({
            "id": af.image_id,
            "image_id": af.image_id,
            "category_id": categories[af.label],
            "bbox": [x, y, w, h],
            "area": w * h,
            "iscrowd": 0,
        })

    cat_list = [{"id": v, "name": k} for k, v in
                sorted(categories.items(), key=lambda x: x[1])]

    return {
        "images": images,
        "annotations": annotations,
        "categories": cat_list,
    }


def _visualize_samples(
    output_dir: str,
    frames: List[AnnotatedFrame],
    max_count: int,
):
    """Generate visualization images with bbox overlays."""
    vis_dir = os.path.join(output_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)
    count = 0

    for af in frames:
        if count >= max_count:
            break
        if af.bbox is None:
            continue

        img_path = os.path.join(output_dir, "images", af.img_filename)
        img = cv2.imread(img_path)
        if img is None:
            continue

        x, y, w, h = af.bbox
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        label_text = f"{af.label} (f{af.frame_id})"
        cv2.putText(img, label_text, (x, max(y - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw crosshair
        ih, iw = img.shape[:2]
        cx, cy = iw // 2, ih // 2
        cv2.drawMarker(img, (cx, cy), (0, 0, 255),
                      cv2.MARKER_CROSS, 15, 2)

        vis_path = os.path.join(vis_dir, f"vis_{count:03d}_{af.label}.png")
        cv2.imwrite(vis_path, img)
        count += 1

    print(f"  Saved {count} visualizations to {vis_dir}")


def _generate_quality_report(
    output_dir: str,
    frames: List[AnnotatedFrame],
    samples_per_cat: int = 9,
):
    """Generate per-category grid of annotated samples for quality review."""
    report_dir = os.path.join(output_dir, "quality_report")
    os.makedirs(report_dir, exist_ok=True)

    by_cat: Dict[str, List[AnnotatedFrame]] = defaultdict(list)
    for af in frames:
        if af.bbox is not None:
            by_cat[af.label].append(af)

    for cat, cat_frames in sorted(by_cat.items()):
        n = min(len(cat_frames), samples_per_cat)
        step = max(1, len(cat_frames) // n)
        sampled = cat_frames[::step][:n]

        cols = min(3, n)
        rows = (n + cols - 1) // cols
        cell_w, cell_h = 320, 180
        grid = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)

        for idx, af in enumerate(sampled):
            r, c = divmod(idx, cols)
            img_path = os.path.join(output_dir, "images", af.img_filename)
            img = cv2.imread(img_path)
            if img is None:
                continue
            x, y, w, h = af.bbox
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            ih, iw = img.shape[:2]
            cv2.drawMarker(img, (iw // 2, ih // 2), (0, 0, 255),
                          cv2.MARKER_CROSS, 12, 1)
            info = f"{af.label} f{af.frame_id} area={af.mask_area}"
            cv2.putText(img, info, (5, 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            thumb = cv2.resize(img, (cell_w, cell_h))
            grid[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w] = thumb

        out_path = os.path.join(report_dir, f"cat_{cat}.png")
        cv2.imwrite(out_path, grid)

    print(f"  Quality report: {len(by_cat)} categories → {report_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build fine-tuning dataset using action-based frame selection"
    )
    parser.add_argument("--data-root", required=True,
                        help="MineStudio data root (with segmentation/, action/)")
    parser.add_argument("--output-dir", default="data/processed/finetune_action",
                        help="Output directory")
    parser.add_argument("--raw-video-dir", required=True,
                        help="Path to raw VPT MP4 video files")
    parser.add_argument("--action-root", default=None,
                        help="Action LMDB root (default: data_root/action/)")
    parser.add_argument("--n-frames", type=int, default=4,
                        help="Number of frames to select per event (default: 4)")
    parser.add_argument("--frame-mode",
                        choices=["backtrack", "early_range", "preattack"],
                        default="backtrack",
                        help="Frame selection: backtrack (go back from event, "
                             "recommended), early_range, or preattack")
    parser.add_argument("--skip-tail", type=int, default=8,
                        help="Frames to skip from event (mining crack animation, "
                             "default: 8 ≈ 0.4s at 20fps). Only for backtrack mode.")
    parser.add_argument("--crop-fraction", type=float, default=0.25,
                        help="Center-crop bbox size as fraction of image "
                             "(default: 0.25). Only used when --sam2-model is not set.")
    parser.add_argument("--visualize", type=int, default=0,
                        help="Number of samples to visualize")
    parser.add_argument("--sam2-model", default=None,
                        help="SAM2 model checkpoint path (omit for center-crop bbox)")

    args = parser.parse_args()

    summary = build_action_dataset(
        data_root=args.data_root,
        output_dir=args.output_dir,
        raw_video_dir=args.raw_video_dir,
        action_root=args.action_root,
        n_frames=args.n_frames,
        frame_mode=args.frame_mode,
        max_visualize=args.visualize,
        sam2_model=args.sam2_model,
        skip_tail=args.skip_tail,
        crop_fraction=args.crop_fraction,
    )

    print("\n" + "=" * 60)
    print("Action-based dataset build complete!")
    print(f"  Events:      {summary.get('total_events', 0)}")
    print(f"  Annotations: {summary.get('total_annotations', 0)}")
    print(f"  Categories:  {summary.get('categories', {})}")
    print("=" * 60)


if __name__ == "__main__":
    main()
