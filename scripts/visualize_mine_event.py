#!/usr/bin/env python3
"""Visualize mine_block event frames from MineStudio LMDB data.

Reads frames from image LMDB (224x224 MP4 chunks) and overlays
segmentation masks. No raw video needed — all from aligned LMDB.

Usage:
    python scripts/visualize_mine_event.py \
        --data-root /path/to/dataset_6xx \
        --output-dir /tmp/frame_analysis
"""
import argparse
import cv2
import lmdb
import numpy as np
import os
import pickle
import sys
import tempfile
from collections import defaultdict
from pathlib import Path


def decode_image_chunk(env, ep_idx, chunk_offset):
    """Decode 32 frames from an image LMDB MP4 chunk."""
    key = f"({ep_idx}, {chunk_offset})"
    with env.begin() as txn:
        raw = txn.get(key.encode())
        if raw is None:
            return None
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.write(raw)
    tmp.close()
    frames = []
    cap = cv2.VideoCapture(tmp.name)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    os.unlink(tmp.name)
    return frames if frames else None


def rle_to_mask(rle_str, h, w):
    parts = list(map(int, rle_str.split()))
    mask = np.zeros(h * w, dtype=np.uint8)
    pos = 0
    for i in range(0, len(parts), 2):
        if i + 1 < len(parts):
            start = parts[i]
            length = parts[i + 1]
            if start + length <= len(mask):
                mask[start:start + length] = 1
            pos = start + length
    return mask.reshape(h, w)


def mask_to_bbox(mask):
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()),
            int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True,
                        help="MineStudio dataset root (e.g. /path/to/dataset_6xx)")
    parser.add_argument("--output-dir", default="/tmp/frame_analysis")
    parser.add_argument("--event-type", default="mine_block:stone")
    parser.add_argument("--n-frames", type=int, default=16,
                        help="Number of frames to show (default 16 = 4x4)")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_event = args.event_type

    seg_dir = data_root / "segmentation"
    img_dir = data_root / "image"

    if not seg_dir.exists():
        print(f"ERROR: {seg_dir} not found"); return
    if not img_dir.exists():
        print(f"ERROR: {img_dir} not found"); return

    # ── Step 1: build episode name maps for both seg and image ──
    print("[1/4] Building episode maps...", flush=True)

    seg_ep_map = {}  # (part_path, ep_idx) -> ep_name
    img_ep_map = {}  # ep_name -> (part_path, ep_idx)

    for part_dir in sorted(seg_dir.iterdir()):
        if not part_dir.is_dir() or not (part_dir / "data.mdb").exists():
            continue
        env = lmdb.open(str(part_dir), readonly=True, lock=False,
                        readahead=False, map_size=1024**3 * 100)
        with env.begin() as txn:
            raw = txn.get("__chunk_infos__".encode())
            if raw:
                for i in pickle.loads(raw):
                    seg_ep_map[(str(part_dir), i["episode_idx"])] = i["episode"]
        env.close()

    for part_dir in sorted(img_dir.iterdir()):
        if not part_dir.is_dir() or not (part_dir / "data.mdb").exists():
            continue
        env = lmdb.open(str(part_dir), readonly=True, lock=False,
                        readahead=False, map_size=1024**3 * 100)
        with env.begin() as txn:
            raw = txn.get("__chunk_infos__".encode())
            if raw:
                for i in pickle.loads(raw):
                    img_ep_map[i["episode"]] = (str(part_dir), i["episode_idx"])
        env.close()

    print(f"  Seg episodes: {len(seg_ep_map)}, Image episodes: {len(img_ep_map)}", flush=True)

    # ── Step 2: find a mine_block event where image data exists ──
    print(f"[2/4] Finding {target_event} event with image data...", flush=True)

    found_event = None
    event_masks = {}  # gf -> (rle, point)

    for part_dir in sorted(seg_dir.iterdir()):
        if found_event:
            break
        if not part_dir.is_dir() or not (part_dir / "data.mdb").exists():
            continue
        env = lmdb.open(str(part_dir), readonly=True, lock=False,
                        readahead=False, map_size=1024**3 * 100)
        with env.begin() as txn:
            for key_raw, val_raw in txn.cursor():
                if found_event:
                    break
                ks = key_raw.decode()
                if ks.startswith("__"):
                    continue
                ck = eval(ks)
                ep_idx, fo = ck
                ep_name = seg_ep_map.get((str(part_dir), ep_idx))
                if not ep_name:
                    continue
                if ep_name not in img_ep_map:
                    continue

                frames = pickle.loads(val_raw)
                for fi, fd in enumerate(frames):
                    gf = fo + fi
                    for ek, ev in fd.items():
                        if not isinstance(ev, dict):
                            continue
                        if ev.get("event") != target_event:
                            continue
                        fr = ev.get("frame_range")
                        if not fr:
                            continue
                        if gf != fr[1]:
                            continue

                        img_part, img_eidx = img_ep_map[ep_name]
                        # Verify image LMDB has chunks covering frame_range
                        img_env = lmdb.open(img_part, readonly=True, lock=False,
                                            readahead=False, map_size=1024**3 * 100)
                        has_data = False
                        with img_env.begin() as itxn:
                            chunk_off = (fr[0] // 32) * 32
                            test_key = f"({img_eidx}, {chunk_off})"
                            if itxn.get(test_key.encode()):
                                has_data = True
                        img_env.close()

                        if not has_data:
                            continue

                        found_event = {
                            "event": target_event,
                            "ep_name": ep_name,
                            "seg_part": str(part_dir),
                            "seg_ep_idx": ep_idx,
                            "img_part": img_part,
                            "img_ep_idx": img_eidx,
                            "frame_range": tuple(fr),
                            "ori_frame_range": tuple(ev["ori_frame_range"]) if ev.get("ori_frame_range") else None,
                        }
                        print(f"  Found: {ep_name}", flush=True)
                        print(f"    seg ep_idx={ep_idx}, img ep_idx={img_eidx}", flush=True)
                        print(f"    frame_range={fr}", flush=True)
                        print(f"    ori_frame_range={ev.get('ori_frame_range')}", flush=True)
                        break
        env.close()

    if not found_event:
        print("  ERROR: no matching event found with image data")
        return

    # Collect masks only for the frames we need to display
    fr_start, fr_end = found_event["frame_range"]
    n = args.n_frames
    show_start = max(fr_start, fr_end - n + 1)

    seg_env = lmdb.open(found_event["seg_part"], readonly=True, lock=False,
                        readahead=False, map_size=1024**3 * 100)
    chunks_to_read = set()
    for gf in range(show_start, fr_end + 1):
        chunks_to_read.add((gf // 32) * 32)

    with seg_env.begin() as txn:
        for chunk_off in sorted(chunks_to_read):
            key = f"({found_event['seg_ep_idx']}, {chunk_off})"
            raw = txn.get(key.encode())
            if raw is None:
                continue
            frames = pickle.loads(raw)
            for fi, fd in enumerate(frames):
                gf = chunk_off + fi
                if gf < show_start or gf > fr_end:
                    continue
                for ek, ev in fd.items():
                    if not isinstance(ev, dict):
                        continue
                    if ev.get("event") != target_event:
                        continue
                    rle = ev.get("rle_mask", "")
                    pt = ev.get("point")
                    if rle:
                        event_masks[gf] = (rle, pt)
    seg_env.close()
    print(f"  Masks: {len(event_masks)} frames", flush=True)

    # ── Step 3: read frames from image LMDB ──
    print(f"[3/4] Reading frames {show_start}..{fr_end} from image LMDB...", flush=True)

    img_env = lmdb.open(found_event["img_part"], readonly=True, lock=False,
                        readahead=False, map_size=1024**3 * 100)

    decoded_frames = {}  # gf -> frame (BGR)
    chunks_needed = set()
    for gf in range(show_start, fr_end + 1):
        chunks_needed.add((gf // 32) * 32)

    for chunk_off in sorted(chunks_needed):
        chunk_frames = decode_image_chunk(
            img_env, found_event["img_ep_idx"], chunk_off)
        if chunk_frames is None:
            # Try with seg ep_idx (in case frame coords are shared)
            chunk_frames = decode_image_chunk(
                img_env, found_event["seg_ep_idx"], chunk_off)
        if chunk_frames:
            for i, frame in enumerate(chunk_frames):
                decoded_frames[chunk_off + i] = frame

    img_env.close()
    print(f"  Decoded {len(decoded_frames)} frames", flush=True)

    # ── Step 4: generate 4x4 grid ──
    print("[4/4] Generating grid image...", flush=True)

    cols = 4
    rows = (n + cols - 1) // cols
    cell_w, cell_h = 280, 280  # 224->280 for readability
    grid = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)

    for i in range(n):
        gf = show_start + i
        r, c = divmod(i, cols)
        frame = decoded_frames.get(gf)
        if frame is None:
            continue

        fh, fw = frame.shape[:2]

        # Overlay mask
        has_mask = gf in event_masks
        if has_mask:
            rle, pt = event_masks[gf]
            mask = rle_to_mask(rle, 360, 640)
            mask_resized = cv2.resize(mask, (fw, fh),
                                      interpolation=cv2.INTER_NEAREST)
            overlay = frame.copy()
            overlay[mask_resized > 0] = [0, 200, 0]
            frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
            bbox = mask_to_bbox(mask_resized)
            if bbox:
                x, y, w, h = bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Crosshair
        cv2.drawMarker(frame, (fw // 2, fh // 2), (0, 0, 255),
                       cv2.MARKER_CROSS, 15, 1)

        # Label
        t = gf - fr_end
        if t < 0:
            lbl = f"f{gf} T{t}"
        elif t == 0:
            lbl = f"f{gf} T=0 EVENT"
        else:
            lbl = f"f{gf} T+{t}"
        if has_mask:
            lbl += " [M]"

        thumb = cv2.resize(frame, (cell_w, cell_h),
                           interpolation=cv2.INTER_NEAREST)
        color = (0, 0, 255) if t == 0 else (255, 255, 255)
        cv2.putText(thumb, lbl, (5, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        grid[r * cell_h:(r + 1) * cell_h,
             c * cell_w:(c + 1) * cell_w] = thumb

    event_short = target_event.replace(":", "_")
    out_path = str(out_dir / f"{event_short}_image_lmdb_T-{n-1}_to_T0.png")
    cv2.imwrite(out_path, grid)
    print(f"\nSaved: {out_path}", flush=True)
    print(f"  Event: {target_event}")
    print(f"  Episode: {found_event['ep_name']}")
    print(f"  frame_range(LMDB): {found_event['frame_range']}")
    print(f"  ori_frame_range: {found_event['ori_frame_range']}")
    print(f"  Frames shown: {show_start}..{fr_end}")
    print(f"  Masks overlaid: {sum(1 for gf in range(show_start, fr_end+1) if gf in event_masks)}")


if __name__ == "__main__":
    main()
