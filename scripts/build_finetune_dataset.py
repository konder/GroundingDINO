"""Build GroundingDINO fine-tuning dataset from MineStudio LMDB data.

Pipeline:
    segmentation DB  →  extract events with RLE masks
                     →  decode RLE → bounding box
    image DB         →  decode MP4 chunk → extract frame
    output           →  COCO-format JSON + individual frame images

The segmentation masks in MineStudio are generated via SAM2 with event-triggered
point prompts (player crosshair). We convert these masks to bounding boxes for
GroundingDINO object detection fine-tuning.

Usage:
    python scripts/build_finetune_dataset.py \
        --data-root data/raw/minestudio_sample \
        --output-dir data/processed/finetune_v1 \
        --visualize 5
"""
from __future__ import annotations

import argparse
import cv2
import json
import lmdb
import numpy as np
import os
import pickle
import re
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DetectionAnnotation:
    """Single detection annotation in COCO style."""
    image_id: int
    category: str
    bbox: List[float]          # [x, y, w, h] in pixels
    point: Optional[List[int]] # SAM2 prompt point (x, y)
    event_type: str            # raw event string
    episode_id: int
    frame_id: int              # global frame index in episode
    seg_partition: str = ""    # segmentation LMDB partition path


# ---------------------------------------------------------------------------
# RLE mask → bounding box
# ---------------------------------------------------------------------------

def rle_to_mask(rle_str: str, height: int, width: int) -> np.ndarray:
    """Decode MineStudio RLE string to binary mask.

    RLE format: space-separated pairs of (start, run_length) on a flattened image.
    """
    if not rle_str or not rle_str.strip():
        return np.zeros((height, width), dtype=np.uint8)

    parts = list(map(int, rle_str.split()))
    total_pixels = height * width
    flat = np.zeros(total_pixels, dtype=np.uint8)

    for i in range(0, len(parts), 2):
        start = parts[i]
        length = parts[i + 1] if i + 1 < len(parts) else 1
        end = min(start + length, total_pixels)
        if start < total_pixels:
            flat[start:end] = 1

    return flat.reshape(height, width)


def mask_to_bbox(mask: np.ndarray) -> Optional[List[int]]:
    """Convert binary mask to [x, y, w, h] bounding box.

    Returns None if mask is empty.
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return None

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return [int(cmin), int(rmin), int(cmax - cmin + 1), int(rmax - rmin + 1)]


def filter_mask_by_point(
    mask: np.ndarray,
    point: Optional[Tuple[int, int]],
) -> np.ndarray:
    """Keep only the connected component of mask that contains the point.

    SAM2 masks can cover large areas (entire walls) or have scattered pixels.
    This filters to only the connected region touching the point prompt,
    producing tighter bounding boxes for object detection.

    Args:
        mask: Binary mask (H, W).
        point: (row, col) point prompt. If None, returns mask unchanged.
    """
    if point is None or mask.sum() == 0:
        return mask

    row, col = point
    row = max(0, min(row, mask.shape[0] - 1))
    col = max(0, min(col, mask.shape[1] - 1))

    num_labels, labels = cv2.connectedComponents(mask)

    target_label = labels[row, col]
    if target_label == 0:
        # Point not on mask; find nearest component
        ys, xs = np.where(mask > 0)
        if len(ys) == 0:
            return mask
        dists = (ys - row) ** 2 + (xs - col) ** 2
        nearest_idx = dists.argmin()
        target_label = labels[ys[nearest_idx], xs[nearest_idx]]

    return (labels == target_label).astype(np.uint8)


def rle_to_bbox(
    rle_str: str,
    height: int,
    width: int,
    point: Optional[Tuple[int, int]] = None,
) -> Optional[List[int]]:
    """Convert RLE string to bbox [x, y, w, h].

    If point is provided, filters mask to the connected component containing
    that point before computing the bounding box.
    """
    mask = rle_to_mask(rle_str, height, width)
    mask = filter_mask_by_point(mask, point)
    return mask_to_bbox(mask)


def infer_mask_resolution(rle_str: str) -> Tuple[int, int]:
    """Infer the image resolution from RLE max pixel index.

    MineStudio uses 640x360 for the original Minecraft recording.
    """
    if not rle_str or not rle_str.strip():
        return (360, 640)

    parts = list(map(int, rle_str.split()))
    max_pixel = 0
    for i in range(0, len(parts), 2):
        start = parts[i]
        length = parts[i + 1] if i + 1 < len(parts) else 1
        max_pixel = max(max_pixel, start + length)

    if max_pixel <= 224 * 224:
        return (224, 224)
    elif max_pixel <= 360 * 640:
        return (360, 640)
    else:
        return (360, 640)


# ---------------------------------------------------------------------------
# Event name → category label
# ---------------------------------------------------------------------------

EVENT_LABEL_MAP = {
    "mine_block": lambda item: item.replace("minecraft.", ""),
    "break_item": lambda item: item.replace("minecraft.", ""),
    "use_item": lambda item: item.replace("minecraft.", ""),
    "kill_entity": lambda item: item.replace("minecraft.", ""),
    "entity_killed_by": lambda item: item.replace("minecraft.", ""),
    "craft_item": lambda item: item.replace("minecraft.", ""),
    "pickup": lambda item: item.replace("minecraft.", ""),
    "drop": lambda item: item.replace("minecraft.", ""),
}


def parse_event_label(event_str: str) -> Optional[str]:
    """Extract a human-readable category label from a Minecraft event string.

    Examples:
        'mine_block:coal_ore'       → 'coal_ore'
        'minecraft.mine_block:minecraft.coal_ore' → 'coal_ore'
        'custom:open_chest'         → 'chest'
        'right_click'               → None (too generic)
        'landmark'                  → None (no specific object)
    """
    if not event_str:
        return None

    s = event_str.replace("minecraft.", "")

    if ":" in s:
        parts = s.split(":")
        action = parts[0]
        target = parts[-1]

        skip_actions = {"custom", "open_chest"}
        if action in skip_actions:
            return None

        return target.strip()

    if s in ("right_click", "landmark", "attack"):
        return None

    return s.strip() or None


# ---------------------------------------------------------------------------
# LMDB episode metadata → raw video mapping
# ---------------------------------------------------------------------------

def read_episode_mapping(lmdb_path: str) -> Dict[int, str]:
    """Read __chunk_infos__ from LMDB to get episode_idx → episode_name mapping.

    MineStudio stores metadata in each LMDB partition:
        __chunk_infos__ = [{'episode': 'Player249-...', 'episode_idx': 0, ...}, ...]
    """
    env = lmdb.open(lmdb_path, readonly=True, lock=False,
                    readahead=False, map_size=1024**3 * 100)
    mapping = {}
    with env.begin() as txn:
        raw = txn.get("__chunk_infos__".encode())
        if raw:
            chunk_infos = pickle.loads(raw)
            for info in chunk_infos:
                idx = info.get("episode_idx", info.get("idx"))
                name = info.get("episode", "")
                if idx is not None and name:
                    mapping[idx] = name
    env.close()
    return mapping


def build_episode_index(data_root: str, source: str = "segmentation") -> Dict[str, Dict[int, str]]:
    """Build episode_idx → episode_name mapping for LMDB partitions.

    Reads from segmentation by default, since segmentation episode IDs
    may differ from image/video episode IDs (different data partitions).

    Returns: {lmdb_path: {episode_idx: episode_name, ...}, ...}
    """
    index = {}
    for part_path in find_lmdb_parts(data_root, source):
        mapping = read_episode_mapping(part_path)
        if mapping:
            index[part_path] = mapping
    return index


def _raw_video_cache_path(raw_video_dir: str) -> str:
    """Deterministic cache file path in /tmp based on raw_video_dir."""
    import hashlib
    h = hashlib.md5(raw_video_dir.encode()).hexdigest()[:12]
    return f"/tmp/raw_video_index_{h}.json"


def build_raw_video_index(raw_video_dir: str) -> Dict[str, str]:
    """Scan raw_video_dir recursively and build stem → full_path index.

    Handles multi-level structures like:
        raw_video_dir/all_6xx_Jun_29/data/6.0/Player249-xxx.mp4
        raw_video_dir/all_6xx_Jun_29/data/6.13/Player100-yyy.mp4

    The index is cached to /tmp so subsequent runs skip the scan.

    Returns: {"Player249-xxx": "/full/path/Player249-xxx.mp4", ...}
    """
    cache_file = _raw_video_cache_path(raw_video_dir)

    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                index = json.load(f)
            print(f"  Loaded cached index from {cache_file} ({len(index)} entries)")
            return index
        except (json.JSONDecodeError, IOError):
            pass

    raw_root = Path(raw_video_dir)
    index: Dict[str, str] = {}
    for mp4 in raw_root.rglob("*.mp4"):
        stem = mp4.stem
        if stem not in index:
            index[stem] = str(mp4)

    try:
        with open(cache_file, "w") as f:
            json.dump(index, f)
        print(f"  Saved index cache to {cache_file}")
    except IOError:
        pass

    return index


def find_raw_video(
    raw_video_dir: str,
    episode_name: str,
    _index_cache: Dict[str, Dict[str, str]] = {},
) -> Optional[str]:
    """Find the raw MP4 file for an episode name using pre-built index.

    The index is built once per raw_video_dir and cached for subsequent calls.
    """
    if raw_video_dir not in _index_cache:
        _index_cache[raw_video_dir] = build_raw_video_index(raw_video_dir)
    return _index_cache[raw_video_dir].get(episode_name)


def decode_raw_video_frame(
    video_path: str,
    frame_index: int,
    _cap_cache: Dict[str, cv2.VideoCapture] = {},
) -> Optional[np.ndarray]:
    """Extract a specific frame from a raw MP4 video file.

    Returns BGR numpy array at original resolution (typically 640x360).
    """
    if video_path not in _cap_cache:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        _cap_cache[video_path] = cap

    cap = _cap_cache[video_path]
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    return frame if ret else None


# ---------------------------------------------------------------------------
# Video chunk → frame extraction
# ---------------------------------------------------------------------------

def decode_video_frame(video_bytes: bytes, frame_index: int) -> Optional[np.ndarray]:
    """Decode a specific frame from an MP4 video chunk stored as bytes.

    Returns BGR numpy array or None on failure.
    """
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
    try:
        with os.fdopen(tmp_fd, "wb") as f:
            f.write(video_bytes)

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return None

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        cap.release()

        return frame if ret else None
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def find_lmdb_parts(data_root: str, subdir: str) -> List[str]:
    """Find all LMDB part directories under data_root/subdir/."""
    base = Path(data_root) / subdir
    if not base.exists():
        return []

    parts = []
    if (base / "data.mdb").exists():
        parts.append(str(base))
    for child in sorted(base.iterdir()):
        if child.is_dir() and (child / "data.mdb").exists():
            parts.append(str(child))
    return parts


def _part_num(name: str) -> int:
    """Extract numeric part number from directory name like 'part-950'."""
    try:
        return int(name.split("-")[-1])
    except (ValueError, IndexError):
        return 0


def detect_video_source(data_root: str) -> str:
    """Detect whether to use video/ (640x360) or image/ (224x224)."""
    if find_lmdb_parts(data_root, "video"):
        return "video"
    return "image"


def find_partition_mapping(data_root: str) -> Dict[str, str]:
    """Map segmentation partition paths to closest video/image partition paths.

    MineStudio uses nearly-matching part numbers across data types:
    seg/part-949 ↔ video/video-950, seg/part-1898 ↔ video/video-1900, etc.
    """
    source = detect_video_source(data_root)
    seg_parts = find_lmdb_parts(data_root, "segmentation")
    src_parts = find_lmdb_parts(data_root, source)

    if not src_parts:
        return {}

    src_by_num = {_part_num(Path(p).name): p for p in src_parts}
    mapping = {}

    for sp in seg_parts:
        sp_num = _part_num(Path(sp).name)
        closest_num = min(src_by_num.keys(), key=lambda n: abs(n - sp_num))
        mapping[sp] = src_by_num[closest_num]

    return mapping


def build_image_index(data_root: str) -> Dict[str, List[str]]:
    """Build an index: key_str → [lmdb_path, ...] for fast lookup.

    Prefers video/ (640x360) over image/ (224x224).
    """
    source = detect_video_source(data_root)
    index: Dict[str, List[str]] = defaultdict(list)
    for part_path in find_lmdb_parts(data_root, source):
        env = lmdb.open(part_path, readonly=True, lock=False,
                        readahead=False, map_size=1024**3 * 100)
        with env.begin() as txn:
            for key_raw, _ in txn.cursor():
                k = key_raw.decode()
                if not k.startswith("__"):
                    index[k].append(part_path)
        env.close()
    return dict(index)


def _point_near_center(
    point: Optional[Tuple[int, int]],
    center: Tuple[int, int],
    threshold: float,
    half_diag: float,
) -> bool:
    """Check if SAM2 tracking point is within threshold of screen center."""
    if point is None:
        return False
    row, col = point
    dist = ((row - center[0]) ** 2 + (col - center[1]) ** 2) ** 0.5
    return dist / half_diag <= threshold


def _find_event_boundaries(frames: List[dict]) -> Dict[str, Tuple[int, int]]:
    """Find first/last frame index for each event type within a chunk.

    Returns: {event_label: (first_frame_idx, last_frame_idx)}
    """
    boundaries: Dict[str, List[int]] = defaultdict(list)
    for fi, frame_dict in enumerate(frames):
        for event_key, event_val in frame_dict.items():
            if not isinstance(event_key, tuple) or len(event_key) < 2:
                continue
            event_name = event_val.get("event", "")
            label = parse_event_label(event_name)
            if label is not None:
                boundaries[label].append(fi)

    return {
        label: (min(indices), max(indices))
        for label, indices in boundaries.items()
    }


def extract_annotations_from_segmentation(
    data_root: str,
    image_index: Dict[str, List[str]],
    mask_height: int = 360,
    mask_width: int = 640,
    center_threshold: float = 0.20,
    event_window: int = 8,
) -> Tuple[List[DetectionAnnotation], Dict[str, set]]:
    """Walk segmentation DB, extract high-quality annotations.

    Dual filtering strategy:
    1. Temporal: only keep frames within `event_window` frames before each
       event's last occurrence in the chunk (when the player is actively
       targeting the object, right before it's mined/used/consumed).
    2. Spatial: only keep frames where the SAM2 tracking point is within
       `center_threshold` of screen center (player crosshair on target).
    """
    annotations: List[DetectionAnnotation] = []
    category_set: Dict[str, set] = defaultdict(set)
    ann_id = 0
    skipped_off_center = 0
    skipped_temporal = 0

    cy, cx = mask_height / 2, mask_width / 2
    half_diag = (cy ** 2 + cx ** 2) ** 0.5

    for seg_path in find_lmdb_parts(data_root, "segmentation"):
        env = lmdb.open(seg_path, readonly=True, lock=False,
                        readahead=False, map_size=1024**3 * 100)
        with env.begin() as txn:
            for key_raw, val_raw in txn.cursor():
                key_str = key_raw.decode()

                if key_str.startswith("__"):
                    continue

                chunk_key = eval(key_str)  # (episode_id, frame_offset)
                episode_id = chunk_key[0]
                frame_offset = chunk_key[1]
                frames = pickle.loads(val_raw)

                event_bounds = _find_event_boundaries(frames)

                for fi, frame_dict in enumerate(frames):
                    for event_key, event_val in frame_dict.items():
                        if not isinstance(event_key, tuple) or len(event_key) < 2:
                            continue

                        event_name = event_val.get("event", "")
                        rle_str = event_val.get("rle_mask", "")
                        point = event_val.get("point")

                        label = parse_event_label(event_name)
                        if label is None:
                            continue

                        if not rle_str or not rle_str.strip():
                            continue

                        _, last_fi = event_bounds.get(label, (0, 31))
                        if fi < last_fi - event_window or fi > last_fi:
                            skipped_temporal += 1
                            continue

                        if not _point_near_center(point, (cy, cx),
                                                  center_threshold, half_diag):
                            skipped_off_center += 1
                            continue

                        h, w = mask_height, mask_width
                        bbox = rle_to_bbox(rle_str, h, w, point=point)
                        if bbox is None:
                            continue

                        global_frame = frame_offset + fi

                        ann = DetectionAnnotation(
                            image_id=ann_id,
                            category=label,
                            bbox=bbox,
                            point=list(point) if point else None,
                            event_type=event_name,
                            episode_id=episode_id,
                            frame_id=global_frame,
                            seg_partition=seg_path,
                        )
                        annotations.append(ann)
                        category_set[label].add(ann_id)
                        ann_id += 1

        env.close()

    print(f"  Skipped {skipped_temporal} annotations (outside event window)")
    print(f"  Skipped {skipped_off_center} annotations (point too far from center)")
    return annotations, category_set


def export_frame_image(
    image_index: Dict[str, List[str]],
    chunk_key_str: str,
    frame_in_chunk: int,
    output_path: str,
    _env_cache: Dict[str, lmdb.Environment] = {},
) -> bool:
    """Extract and save a single frame from the image LMDB."""
    paths = image_index.get(chunk_key_str, [])
    for img_path in paths:
        if img_path not in _env_cache:
            _env_cache[img_path] = lmdb.open(
                img_path, readonly=True, lock=False,
                readahead=False, map_size=1024**3 * 100)

        env = _env_cache[img_path]
        with env.begin() as txn:
            val = txn.get(chunk_key_str.encode())
            if val is None:
                continue
            frame = decode_video_frame(val, frame_in_chunk)
            if frame is not None:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                cv2.imwrite(output_path, frame)
                return True
    return False


def scale_bbox(bbox: List[int], src_hw: Tuple[int, int],
               dst_hw: Tuple[int, int]) -> List[float]:
    """Scale bbox [x, y, w, h] from source to destination resolution."""
    sx = dst_hw[1] / src_hw[1]
    sy = dst_hw[0] / src_hw[0]
    return [bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy]


# ---------------------------------------------------------------------------
# COCO format output
# ---------------------------------------------------------------------------

def to_coco_format(
    annotations: List[DetectionAnnotation],
    image_hw: Tuple[int, int],
    mask_hw: Tuple[int, int] = (360, 640),
) -> dict:
    """Convert annotations to COCO detection format JSON."""
    categories = {}
    cat_id = 1
    images = {}
    coco_anns = []
    ann_id = 1

    for ann in annotations:
        if ann.category not in categories:
            categories[ann.category] = cat_id
            cat_id += 1

        seg_part_name = Path(ann.seg_partition).name if ann.seg_partition else "unk"
        img_filename = f"{seg_part_name}_ep{ann.episode_id}_f{ann.frame_id:06d}.png"
        img_key = (ann.seg_partition, ann.episode_id, ann.frame_id)

        if img_key not in images:
            img_id = len(images) + 1
            images[img_key] = {
                "id": img_id,
                "file_name": img_filename,
                "width": image_hw[1],
                "height": image_hw[0],
            }

        scaled_bbox = scale_bbox(ann.bbox, mask_hw, image_hw)

        coco_anns.append({
            "id": ann_id,
            "image_id": images[img_key]["id"],
            "category_id": categories[ann.category],
            "bbox": [round(v, 2) for v in scaled_bbox],
            "area": round(scaled_bbox[2] * scaled_bbox[3], 2),
            "iscrowd": 0,
            "event_type": ann.event_type,
        })
        ann_id += 1

    coco = {
        "images": list(images.values()),
        "annotations": coco_anns,
        "categories": [
            {"id": cid, "name": name, "supercategory": "minecraft"}
            for name, cid in sorted(categories.items(), key=lambda x: x[1])
        ],
    }
    return coco


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_annotation(
    image_path: str,
    bbox_scaled: List[float],
    label: str,
    output_path: str,
) -> None:
    """Draw bbox and label on an image and save."""
    img = cv2.imread(image_path)
    if img is None:
        return

    x, y, w, h = [int(v) for v in bbox_scaled]
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(img, label, (x, max(y - 5, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite(output_path, img)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_dataset(
    data_root: str,
    output_dir: str,
    mask_height: int = 360,
    mask_width: int = 640,
    max_visualize: int = 0,
    raw_video_dir: Optional[str] = None,
    center_threshold: float = 0.20,
    event_window: int = 8,
) -> dict:
    """Main pipeline: build fine-tuning dataset from MineStudio LMDB.

    Frames are extracted from raw MP4 files matching the segmentation
    episode mapping (not the image LMDB, which may contain different episodes).
    center_threshold + event_window jointly filter for high-quality annotations.
    """
    source = detect_video_source(data_root)
    use_raw = raw_video_dir is not None

    print(f"[1/5] Building index ...")
    image_index = build_image_index(data_root)
    print(f"  Source: {source}/ ({len(image_index)} chunk keys)")

    seg_episode_index = build_episode_index(data_root, source="segmentation")
    seg_total = sum(len(m) for m in seg_episode_index.values())
    print(f"  Segmentation episode mappings: {seg_total} episodes across "
          f"{len(seg_episode_index)} partitions")

    raw_video_index: Dict[str, str] = {}
    if use_raw:
        print(f"  Raw video dir: {raw_video_dir}")
        print(f"  Scanning for MP4 files ...")
        raw_video_index = build_raw_video_index(raw_video_dir)
        print(f"  Found {len(raw_video_index)} raw MP4 files")
        needed_names = set(n for m in seg_episode_index.values() for n in m.values())
        matched = sum(1 for n in needed_names if n in raw_video_index)
        print(f"  Episodes with raw video: {matched}/{len(needed_names)}")

    print(f"[2/5] Extracting annotations from segmentation "
          f"(center={center_threshold:.0%}, window={event_window}) ...")
    annotations, cat_stats = extract_annotations_from_segmentation(
        data_root, image_index, mask_height, mask_width,
        center_threshold=center_threshold,
        event_window=event_window,
    )
    print(f"  Extracted {len(annotations)} annotations")
    print(f"  Categories: {', '.join(f'{k}({len(v)})' for k, v in sorted(cat_stats.items()))}")

    if not annotations:
        print("  WARNING: No annotations found. Check data alignment.")
        return {"annotations": 0}

    print(f"[3/5] Exporting frame images to {output_dir}/images/ ...")
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    exported = 0
    skipped = 0
    seen_frames = set()
    missing_videos: List[str] = []

    for ann in annotations:
        frame_key = (ann.seg_partition, ann.episode_id, ann.frame_id)
        if frame_key in seen_frames:
            continue
        seen_frames.add(frame_key)

        seg_part_name = Path(ann.seg_partition).name if ann.seg_partition else "unk"
        img_filename = f"{seg_part_name}_ep{ann.episode_id}_f{ann.frame_id:06d}.png"
        img_path = os.path.join(output_dir, "images", img_filename)

        if use_raw:
            part_mapping = seg_episode_index.get(ann.seg_partition, {})
            ep_name = part_mapping.get(ann.episode_id)
            if not ep_name:
                msg = (f"ep_id={ann.episode_id} in {seg_part_name} "
                       f"(no mapping in __chunk_infos__)")
                if msg not in missing_videos:
                    missing_videos.append(msg)
                    print(f"  MISS: {msg}")
                skipped += 1
                continue

            mp4_path = find_raw_video(raw_video_dir, ep_name)
            if not mp4_path:
                msg = f"{ep_name}.mp4 (ep_id={ann.episode_id} in {seg_part_name})"
                if msg not in missing_videos:
                    missing_videos.append(msg)
                    print(f"  MISS: {msg}")
                skipped += 1
                continue

            frame = decode_raw_video_frame(mp4_path, ann.frame_id)
            if frame is not None:
                cv2.imwrite(img_path, frame)
                exported += 1
            else:
                print(f"  MISS: frame {ann.frame_id} decode failed in {ep_name}.mp4")
                skipped += 1
        else:
            chunk_offset = (ann.frame_id // 32) * 32
            frame_in_chunk = ann.frame_id % 32
            chunk_key_str = f"({ann.episode_id}, {chunk_offset})"
            if export_frame_image(image_index, chunk_key_str, frame_in_chunk, img_path):
                exported += 1
            else:
                skipped += 1

    print(f"  Exported {exported}/{len(seen_frames)} frames")
    if skipped > 0:
        print(f"  Skipped: {skipped} frames")
    if missing_videos:
        print(f"  Missing videos ({len(missing_videos)}):")
        for mv in missing_videos:
            print(f"    - {mv}")

    # Detect actual image resolution from first exported image
    sample_img_path = os.path.join(output_dir, "images",
                                   f"ep{annotations[0].episode_id}_f{annotations[0].frame_id:06d}.png")
    if os.path.exists(sample_img_path):
        sample_img = cv2.imread(sample_img_path)
        image_hw = (sample_img.shape[0], sample_img.shape[1])
    else:
        image_hw = (360, 640) if use_raw else (224, 224)
    print(f"  Image resolution: {image_hw[1]}x{image_hw[0]}")

    print(f"[4/5] Generating COCO JSON ...")
    coco = to_coco_format(annotations, image_hw, (mask_height, mask_width))
    coco_path = os.path.join(output_dir, "annotations.json")
    with open(coco_path, "w") as f:
        json.dump(coco, f, indent=2)
    print(f"  Saved to {coco_path}")
    print(f"  {len(coco['images'])} images, {len(coco['annotations'])} annotations, "
          f"{len(coco['categories'])} categories")

    if max_visualize > 0:
        print(f"[5/5] Generating {max_visualize} visualization samples ...")
        vis_dir = os.path.join(output_dir, "vis")
        os.makedirs(vis_dir, exist_ok=True)
        count = 0
        for ann in annotations[:max_visualize]:
            img_filename = f"ep{ann.episode_id}_f{ann.frame_id:06d}.png"
            img_path = os.path.join(output_dir, "images", img_filename)
            if not os.path.exists(img_path):
                continue
            scaled = scale_bbox(ann.bbox, (mask_height, mask_width), image_hw)
            vis_path = os.path.join(vis_dir, f"vis_{count:03d}_{ann.category}.png")
            visualize_annotation(img_path, scaled, ann.category, vis_path)
            count += 1
        print(f"  Saved {count} visualizations to {vis_dir}")
    else:
        print("[5/5] Skipping visualization (use --visualize N to enable)")

    summary = {
        "data_root": data_root,
        "output_dir": output_dir,
        "source": "raw_video" if use_raw else source,
        "raw_video_dir": raw_video_dir,
        "mask_resolution": f"{mask_width}x{mask_height}",
        "image_resolution": f"{image_hw[1]}x{image_hw[0]}",
        "total_annotations": len(annotations),
        "total_images": len(coco["images"]),
        "categories": {c["name"]: sum(1 for a in coco["annotations"] if a["category_id"] == c["id"])
                       for c in coco["categories"]},
        "exported_frames": exported,
        "skipped_frames": skipped,
        "missing_videos": missing_videos if use_raw else [],
    }
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Build GroundingDINO fine-tuning dataset from MineStudio LMDB"
    )
    parser.add_argument("--data-root", required=True,
                        help="MineStudio data root (with image/, segmentation/ subdirs)")
    parser.add_argument("--output-dir", default="data/processed/finetune_v1",
                        help="Output directory for COCO dataset")
    parser.add_argument("--mask-height", type=int, default=360,
                        help="Mask resolution height (default: 360)")
    parser.add_argument("--mask-width", type=int, default=640,
                        help="Mask resolution width (default: 640)")
    parser.add_argument("--visualize", type=int, default=0,
                        help="Number of samples to visualize (0 = none)")
    parser.add_argument("--raw-video-dir", default=None,
                        help="Path to raw VPT MP4 video files for 640x360 frame extraction")
    parser.add_argument("--center-threshold", type=float, default=0.20,
                        help="Max normalized distance from screen center for SAM2 point "
                             "(0.15=strict, 0.20=default, 0.30=loose)")
    parser.add_argument("--event-window", type=int, default=8,
                        help="Take only the last N frames before each event's end in "
                             "the 32-frame chunk (default: 8)")

    args = parser.parse_args()

    summary = build_dataset(
        data_root=args.data_root,
        output_dir=args.output_dir,
        mask_height=args.mask_height,
        mask_width=args.mask_width,
        max_visualize=args.visualize,
        raw_video_dir=args.raw_video_dir,
        center_threshold=args.center_threshold,
        event_window=args.event_window,
    )

    print("\n" + "=" * 60)
    print("Dataset build complete!")
    print(f"  Images:      {summary.get('total_images', 0)}")
    print(f"  Annotations: {summary.get('total_annotations', 0)}")
    print(f"  Categories:  {summary.get('categories', {})}")
    print("=" * 60)


if __name__ == "__main__":
    main()
