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
import time
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


SKIP_ACTIONS = {"custom", "open_chest", "craft"}


def parse_event_label(event_str: str, exclude_actions: Optional[set] = None) -> Optional[str]:
    """Extract a human-readable category label from a Minecraft event string.

    Examples:
        'mine_block:coal_ore'       → 'coal_ore'
        'minecraft.mine_block:minecraft.coal_ore' → 'coal_ore'
        'craft:crafting_table'      → None (craft excluded by default)
        'custom:open_chest'         → None
        'right_click'               → None (too generic)
        'landmark'                  → None (no specific object)
    """
    if not event_str:
        return None

    skip = SKIP_ACTIONS | (exclude_actions or set())
    s = event_str.replace("minecraft.", "")

    if ":" in s:
        parts = s.split(":")
        action = parts[0]
        target = parts[-1]

        if action in skip:
            return None

        return target.strip()

    if s in ("right_click", "landmark", "attack", "craft"):
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
    parts = find_lmdb_parts(data_root, source)
    index = {}
    for i, part_path in enumerate(parts):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"    Scanning {source} partition {i + 1}/{len(parts)} ...",
                  flush=True)
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
    count = 0
    print(f"  Scanning {raw_video_dir} for MP4 files ...", flush=True)
    for mp4 in raw_root.rglob("*.mp4"):
        stem = mp4.stem
        if stem not in index:
            index[stem] = str(mp4)
        count += 1
        if count % 5000 == 0:
            print(f"    Scanned {count} files ({len(index)} unique) ...", flush=True)

    try:
        with open(cache_file, "w") as f:
            json.dump(index, f)
        print(f"  Saved index cache to {cache_file}", flush=True)
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
# Multi-chunk episode support
# ---------------------------------------------------------------------------

def _parse_episode_prefix(episode_name: str) -> Tuple[str, str]:
    """Parse episode name into (session_prefix, start_timestamp).

    VPT recordings are split into ~60-second chunks with names like:
        Player122-f153ac423f61-20211217-140509
    The session prefix is everything up to the last hyphen-separated
    timestamp, and the start timestamp identifies this session's first chunk.

    Returns ('Player122-f153ac423f61-20211217', '140509')
    """
    parts = episode_name.rsplit("-", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0], parts[1]
    return episode_name, ""


@dataclass
class EpisodeChunkInfo:
    """Ordered chunks that compose a multi-chunk VPT episode."""
    chunks: List[Tuple[str, int]]   # [(chunk_path, num_frames), ...]
    cumulative: List[int]           # [0, n1, n1+n2, ...] prefix sums

    @property
    def total_frames(self) -> int:
        return self.cumulative[-1] if len(self.cumulative) > 1 else 0

    def resolve_frame(self, global_frame: int) -> Optional[Tuple[str, int]]:
        """Map global frame index → (chunk_path, local_frame_index)."""
        if global_frame < 0 or global_frame >= self.total_frames:
            return None
        import bisect
        idx = bisect.bisect_right(self.cumulative, global_frame) - 1
        if idx < 0 or idx >= len(self.chunks):
            return None
        return self.chunks[idx][0], global_frame - self.cumulative[idx]


def _build_prefix_index(
    raw_video_index: Dict[str, str],
) -> Dict[str, List[Tuple[str, str, str]]]:
    """Pre-group raw videos by session prefix for O(1) lookup.

    Returns: {prefix: [(timestamp, stem, path), ...]} sorted by timestamp.
    """
    groups: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
    for stem, path in raw_video_index.items():
        prefix, ts = _parse_episode_prefix(stem)
        groups[prefix].append((ts, stem, path))
    for v in groups.values():
        v.sort()
    return dict(groups)


_prefix_index_cache: Dict[int, Dict[str, List[Tuple[str, str, str]]]] = {}


def _get_prefix_index(raw_video_index: Dict[str, str]):
    key = id(raw_video_index)
    if key not in _prefix_index_cache:
        _prefix_index_cache[key] = _build_prefix_index(raw_video_index)
    return _prefix_index_cache[key]


# Frame count cache on disk to avoid re-opening MP4s across runs
def _frame_count_cache_path(raw_video_dir_hash: str) -> str:
    return f"/tmp/frame_counts_{raw_video_dir_hash}.json"


_frame_count_cache: Dict[str, int] = {}
_frame_count_loaded = False


def _load_frame_count_cache():
    global _frame_count_loaded
    if _frame_count_loaded:
        return
    import glob
    for f in glob.glob("/tmp/frame_counts_*.json"):
        try:
            with open(f) as fh:
                _frame_count_cache.update(json.load(fh))
        except (json.JSONDecodeError, IOError):
            pass
    _frame_count_loaded = True


def _save_frame_count_cache():
    import hashlib
    h = hashlib.md5(str(sorted(_frame_count_cache.keys())[:5]).encode()).hexdigest()[:12]
    try:
        with open(f"/tmp/frame_counts_{h}.json", "w") as f:
            json.dump(_frame_count_cache, f)
    except IOError:
        pass


def _get_frame_count(path: str) -> int:
    """Get frame count for a video, using disk cache."""
    _load_frame_count_cache()
    if path in _frame_count_cache:
        return _frame_count_cache[path]
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        _frame_count_cache[path] = 0
        return 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    _frame_count_cache[path] = max(n, 0)
    return max(n, 0)


def build_episode_chunk_info(
    raw_video_index: Dict[str, str],
    episode_name: str,
    _cache: Dict[str, Optional[EpisodeChunkInfo]] = {},
) -> Optional[EpisodeChunkInfo]:
    """Build ordered chunk list for a multi-chunk VPT episode.

    VPT splits player sessions into ~60-second (1201 frame) MP4 chunks.
    MineStudio stitches these into a single episode with continuous
    frame numbering.  This function finds all chunks belonging to an
    episode, reads their actual frame counts, and builds a cumulative
    offset table for O(log n) frame-to-chunk resolution.
    """
    if episode_name in _cache:
        return _cache[episode_name]

    prefix, start_ts = _parse_episode_prefix(episode_name)
    pidx = _get_prefix_index(raw_video_index)
    all_candidates = pidx.get(prefix, [])

    candidates = [(ts, stem, path) for ts, stem, path in all_candidates
                  if ts >= start_ts]

    if not candidates:
        _cache[episode_name] = None
        return None

    chunks: List[Tuple[str, int]] = []
    cumulative = [0]
    for _, _, path in candidates:
        n = _get_frame_count(path)
        if n <= 0:
            continue
        chunks.append((path, n))
        cumulative.append(cumulative[-1] + n)

    if not chunks:
        _cache[episode_name] = None
        return None

    info = EpisodeChunkInfo(chunks=chunks, cumulative=cumulative)
    _cache[episode_name] = info
    return info


def decode_multichunk_frame(
    raw_video_index: Dict[str, str],
    episode_name: str,
    global_frame: int,
) -> Optional[np.ndarray]:
    """Decode a frame from a (possibly multi-chunk) VPT episode.

    Resolves the global frame index to the correct chunk file and local
    offset, then extracts the frame.  Chunk metadata is cached so
    subsequent calls for the same episode skip the scan.
    """
    info = build_episode_chunk_info(raw_video_index, episode_name)
    if info is None:
        return None

    resolved = info.resolve_frame(global_frame)
    if resolved is None:
        return None

    chunk_path, local_frame = resolved
    return decode_raw_video_frame(chunk_path, local_frame)


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


def detect_gui_frame(
    img_path: str,
    gui_threshold: float = 0.20,
) -> bool:
    """Detect if a Minecraft frame has an open GUI (inventory/crafting/furnace).

    Minecraft GUIs render a semi-transparent gray panel in the center of the
    screen. We check the center ROI for the ratio of gray pixels with
    characteristic brightness (130-220) and low color variance (<20).

    Returns True if the frame likely contains a GUI overlay.
    """
    img = cv2.imread(img_path)
    if img is None:
        return False
    h, w = img.shape[:2]
    roi = img[int(h * 0.1):int(h * 0.7), int(w * 0.2):int(w * 0.8)]
    gray_diff = np.max(roi.astype(np.int16), axis=2) - np.min(roi.astype(np.int16), axis=2)
    brightness = np.mean(roi, axis=2)
    gray_mask = (gray_diff < 20) & (brightness > 130) & (brightness < 220)
    gray_ratio = gray_mask.sum() / gray_mask.size
    return bool(gray_ratio >= gui_threshold)


def _compute_mask_area(rle_str: str) -> int:
    """Count the number of foreground pixels in an RLE mask."""
    if not rle_str or not rle_str.strip():
        return 0
    parts = list(map(int, rle_str.split()))
    return sum(parts[i + 1] for i in range(0, len(parts), 2) if i + 1 < len(parts))


def extract_annotations_from_segmentation(
    data_root: str,
    image_index: Dict[str, List[str]],
    mask_height: int = 360,
    mask_width: int = 640,
    min_mask_area: int = 5000,
    max_mask_area: float = 0.5,
    center_threshold: float = 0.0,
) -> Tuple[List[DetectionAnnotation], Dict[str, set]]:
    """Walk segmentation DB, extract annotations filtered by mask quality.

    Filtering strategy (replaces temporal window):
    1. Mask area: only keep frames where the SAM2 mask covers at least
       `min_mask_area` pixels (filters post-break residuals and GUI icons)
       and at most `max_mask_area` fraction of the image (filters runaway
       masks that cover the whole screen).
    2. Spatial (optional): if `center_threshold` > 0, also require the
       SAM2 tracking point to be within that fraction of screen center.
    """
    annotations: List[DetectionAnnotation] = []
    category_set: Dict[str, set] = defaultdict(set)
    ann_id = 0
    skipped_small_mask = 0
    skipped_large_mask = 0
    skipped_off_center = 0
    total_pixels = mask_height * mask_width
    max_pixels = int(total_pixels * max_mask_area)

    use_center_filter = center_threshold > 0
    cy, cx = mask_height / 2, mask_width / 2
    half_diag = (cy ** 2 + cx ** 2) ** 0.5

    seg_parts = find_lmdb_parts(data_root, "segmentation")
    for seg_i, seg_path in enumerate(seg_parts):
        print(f"    Processing partition {seg_i + 1}/{len(seg_parts)} "
              f"({len(annotations)} annotations so far) ...", flush=True)
        env = lmdb.open(seg_path, readonly=True, lock=False,
                        readahead=False, map_size=1024**3 * 100)
        with env.begin() as txn:
            n_keys_in_part = txn.stat()["entries"]
            chunk_count = 0
            for key_raw, val_raw in txn.cursor():
                key_str = key_raw.decode()

                if key_str.startswith("__"):
                    continue

                chunk_count += 1
                if chunk_count % 2000 == 0:
                    print(f"      chunk {chunk_count}/{n_keys_in_part} "
                          f"({len(annotations)} annotations) ...", flush=True)

                chunk_key = eval(key_str)  # (episode_id, frame_offset)
                episode_id = chunk_key[0]
                frame_offset = chunk_key[1]
                frames = pickle.loads(val_raw)

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

                        raw_area = _compute_mask_area(rle_str)
                        if raw_area < min_mask_area:
                            skipped_small_mask += 1
                            continue

                        if raw_area > max_pixels:
                            skipped_large_mask += 1
                            continue

                        if use_center_filter and not _point_near_center(
                                point, (cy, cx), center_threshold, half_diag):
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

    print(f"  Skipped {skipped_small_mask} annotations (mask < {min_mask_area}px)")
    print(f"  Skipped {skipped_large_mask} annotations (mask > {max_mask_area:.0%} of image)")
    if use_center_filter:
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
    min_mask_area: int = 5000,
    max_mask_area: float = 0.5,
    center_threshold: float = 0.0,
    filter_gui: bool = True,
) -> dict:
    """Main pipeline: build fine-tuning dataset from MineStudio LMDB.

    Frames are extracted from raw MP4 files matching the segmentation
    episode mapping (not the image LMDB, which may contain different episodes).
    Mask area filtering + GUI detection for quality control.
    """
    source = detect_video_source(data_root)
    use_raw = raw_video_dir is not None

    print(f"[1/5] Building index ...", flush=True)

    if use_raw:
        image_index: Dict[str, List[str]] = {}
        print(f"  Using raw video dir → skipping image/video LMDB index", flush=True)
    else:
        image_index = build_image_index(data_root)
        print(f"  Source: {source}/ ({len(image_index)} chunk keys)", flush=True)

    seg_episode_index = build_episode_index(data_root, source="segmentation")
    seg_total = sum(len(m) for m in seg_episode_index.values())
    print(f"  Segmentation episode mappings: {seg_total} episodes across "
          f"{len(seg_episode_index)} partitions", flush=True)

    raw_video_index: Dict[str, str] = {}
    if use_raw:
        print(f"  Raw video dir: {raw_video_dir}")
        print(f"  Scanning for MP4 files ...")
        raw_video_index = build_raw_video_index(raw_video_dir)
        print(f"  Found {len(raw_video_index)} raw MP4 files")
        needed_names = sorted(set(
            n for m in seg_episode_index.values() for n in m.values()
        ))
        matched = 0
        total_chunks = 0
        corrupted = 0
        print(f"  Matching {len(needed_names)} episodes to raw videos ...",
              flush=True)
        for i, n in enumerate(needed_names):
            if (i + 1) % 200 == 0 or i == 0:
                print(f"    Checking episode {i + 1}/{len(needed_names)} "
                      f"(matched={matched}) ...", flush=True)
            chunk_info = build_episode_chunk_info(raw_video_index, n)
            if chunk_info and chunk_info.total_frames > 0:
                matched += 1
                total_chunks += len(chunk_info.chunks)
            elif chunk_info and chunk_info.total_frames == 0:
                corrupted += 1
        build_episode_chunk_info.__defaults__[0].clear()
        _save_frame_count_cache()
        print(f"  Episodes with raw video: {matched}/{len(needed_names)} "
              f"({total_chunks} total chunks)", flush=True)
        if corrupted:
            print(f"  Corrupted video files (moov atom missing): {corrupted}",
                  flush=True)

    print(f"[2/5] Extracting annotations from segmentation "
          f"(mask_area={min_mask_area}-{max_mask_area:.0%}"
          f"{f', center={center_threshold:.0%}' if center_threshold > 0 else ''}) ...")
    annotations, cat_stats = extract_annotations_from_segmentation(
        data_root, image_index, mask_height, mask_width,
        min_mask_area=min_mask_area,
        max_mask_area=max_mask_area,
        center_threshold=center_threshold,
    )
    print(f"  Extracted {len(annotations)} annotations")
    print(f"  Categories: {', '.join(f'{k}({len(v)})' for k, v in sorted(cat_stats.items()))}")

    if not annotations:
        print("  WARNING: No annotations found. Check data alignment.")
        return {"annotations": 0}

    print(f"[3/5] Exporting frame images to {output_dir}/images/ ...", flush=True)
    img_dir = os.path.join(output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    exported = 0
    skipped = 0
    seen_frames = set()
    missing_videos: List[str] = []

    # Deduplicate and collect unique frames
    frame_tasks = []
    for ann in annotations:
        frame_key = (ann.seg_partition, ann.episode_id, ann.frame_id)
        if frame_key in seen_frames:
            continue
        seen_frames.add(frame_key)
        frame_tasks.append(ann)

    n_tasks = len(frame_tasks)
    print(f"  Unique frames to export: {n_tasks}", flush=True)
    t_export_start = time.time()

    if use_raw:
        # --- Batch-grouped export: group by chunk file, read sequentially ---
        # Resolve each frame to (chunk_path, local_frame) and group
        chunk_groups: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
        for ann in frame_tasks:
            seg_part_name = Path(ann.seg_partition).name if ann.seg_partition else "unk"
            img_filename = f"{seg_part_name}_ep{ann.episode_id}_f{ann.frame_id:06d}.png"
            img_path = os.path.join(img_dir, img_filename)

            if os.path.exists(img_path):
                exported += 1
                continue

            part_mapping = seg_episode_index.get(ann.seg_partition, {})
            ep_name = part_mapping.get(ann.episode_id)
            if not ep_name:
                skipped += 1
                continue

            info = build_episode_chunk_info(raw_video_index, ep_name)
            if info is None:
                skipped += 1
                continue

            resolved = info.resolve_frame(ann.frame_id)
            if resolved is None:
                skipped += 1
                continue

            chunk_path, local_frame = resolved
            chunk_groups[chunk_path].append((local_frame, img_path))

        # Sort frames within each chunk for sequential read
        for frames_list in chunk_groups.values():
            frames_list.sort()

        total_chunks = len(chunk_groups)
        total_frames_to_read = sum(len(v) for v in chunk_groups.values())
        print(f"  Grouped into {total_chunks} video chunks, "
              f"{total_frames_to_read} frames to read "
              f"(skipped {exported} existing, {skipped} missing)", flush=True)

        frames_done = 0
        for chunk_i, (chunk_path, frames_list) in enumerate(chunk_groups.items()):
            if (chunk_i + 1) % 100 == 0 or chunk_i == 0:
                elapsed = time.time() - t_export_start
                rate = frames_done / elapsed if elapsed > 0 else 0
                eta = (total_frames_to_read - frames_done) / rate if rate > 0 else 0
                print(f"    Chunk {chunk_i + 1}/{total_chunks} "
                      f"({frames_done}/{total_frames_to_read} frames, "
                      f"{rate:.1f} f/s, ETA {eta:.0f}s)", flush=True)

            cap = cv2.VideoCapture(chunk_path)
            if not cap.isOpened():
                skipped += len(frames_list)
                frames_done += len(frames_list)
                continue

            for local_frame, img_path in frames_list:
                cap.set(cv2.CAP_PROP_POS_FRAMES, local_frame)
                ret, frame = cap.read()
                if ret and frame is not None:
                    cv2.imwrite(img_path, frame,
                                [cv2.IMWRITE_PNG_COMPRESSION, 1])
                    exported += 1
                else:
                    skipped += 1
                frames_done += 1

            cap.release()
    else:
        for task_i, ann in enumerate(frame_tasks):
            if (task_i + 1) % 500 == 0:
                elapsed = time.time() - t_export_start
                rate = (task_i + 1) / elapsed if elapsed > 0 else 0
                eta = (n_tasks - task_i - 1) / rate if rate > 0 else 0
                print(f"    Frame {task_i + 1}/{n_tasks} "
                      f"(exported={exported}, skipped={skipped}, "
                      f"{rate:.1f} frames/s, ETA {eta:.0f}s) ...", flush=True)

            seg_part_name = Path(ann.seg_partition).name if ann.seg_partition else "unk"
            img_filename = f"{seg_part_name}_ep{ann.episode_id}_f{ann.frame_id:06d}.png"
            img_path = os.path.join(img_dir, img_filename)

            if os.path.exists(img_path):
                exported += 1
                continue

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
    image_hw = (360, 640) if use_raw else (224, 224)
    img_dir = os.path.join(output_dir, "images")
    exported_imgs = [f for f in os.listdir(img_dir) if f.endswith(".png")] if os.path.isdir(img_dir) else []
    if exported_imgs:
        sample_img = cv2.imread(os.path.join(img_dir, exported_imgs[0]))
        if sample_img is not None:
            image_hw = (sample_img.shape[0], sample_img.shape[1])
    print(f"  Image resolution: {image_hw[1]}x{image_hw[0]}")

    # GUI detection: remove frames with Minecraft GUI overlay
    gui_removed = 0
    if filter_gui:
        pre_count = len(annotations)
        kept = []
        gui_frame_keys = set()
        for ann in annotations:
            seg_pn = Path(ann.seg_partition).name if ann.seg_partition else "unk"
            img_filename = f"{seg_pn}_ep{ann.episode_id}_f{ann.frame_id:06d}.png"
            img_path = os.path.join(img_dir, img_filename)
            fk = (ann.seg_partition, ann.episode_id, ann.frame_id)
            if fk in gui_frame_keys:
                continue
            if os.path.exists(img_path) and detect_gui_frame(img_path):
                gui_frame_keys.add(fk)
                os.remove(img_path)
                exported -= 1
            else:
                kept.append(ann)
        gui_removed = pre_count - len(kept)
        annotations = kept
        cat_stats = defaultdict(set)
        for i, ann in enumerate(annotations):
            cat_stats[ann.category].add(i)
        print(f"  GUI filter removed {gui_removed} annotations "
              f"({len(gui_frame_keys)} frames deleted)")

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
        for ann in annotations:
            if count >= max_visualize:
                break
            seg_pn = Path(ann.seg_partition).name if ann.seg_partition else "unk"
            img_filename = f"{seg_pn}_ep{ann.episode_id}_f{ann.frame_id:06d}.png"
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
        "gui_removed": gui_removed,
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
    parser.add_argument("--min-mask-area", type=int, default=5000,
                        help="Minimum mask area in pixels to keep an annotation "
                             "(filters post-break residuals and GUI icons, default: 5000)")
    parser.add_argument("--max-mask-area", type=float, default=0.5,
                        help="Maximum mask area as fraction of image "
                             "(filters runaway full-screen masks, default: 0.5)")
    parser.add_argument("--center-threshold", type=float, default=0.0,
                        help="Optional: max normalized distance from screen center "
                             "for SAM2 point (0=disabled, 0.20=moderate, 0.30=loose)")
    parser.add_argument("--no-gui-filter", action="store_true",
                        help="Disable pixel-based GUI frame detection and removal")

    args = parser.parse_args()

    summary = build_dataset(
        data_root=args.data_root,
        output_dir=args.output_dir,
        mask_height=args.mask_height,
        mask_width=args.mask_width,
        max_visualize=args.visualize,
        raw_video_dir=args.raw_video_dir,
        min_mask_area=args.min_mask_area,
        max_mask_area=args.max_mask_area,
        center_threshold=args.center_threshold,
        filter_gui=not args.no_gui_filter,
    )

    print("\n" + "=" * 60)
    print("Dataset build complete!")
    print(f"  Images:      {summary.get('total_images', 0)}")
    print(f"  Annotations: {summary.get('total_annotations', 0)}")
    print(f"  Categories:  {summary.get('categories', {})}")
    print("=" * 60)


if __name__ == "__main__":
    import os as _os
    _os.environ["OPENCV_LOG_LEVEL"] = "FATAL"
    _os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
    main()
