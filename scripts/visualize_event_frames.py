"""Visualize 16 frames before a mine_block event and an inventory pickup event.

Generates two 4x4 grid images for analysis:
1. coal_ore_break_T-15_to_T0.png  - frames before block destruction (with mask overlay)
2. coal_inventory_pickup_T-15_to_T0.png - frames before inventory coal increase
"""
import cv2
import glob
import json
import lmdb
import numpy as np
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.build_finetune_dataset import (
    build_raw_video_index, build_episode_chunk_info, decode_multichunk_frame,
    rle_to_mask, filter_mask_by_point, mask_to_bbox
)


def make_grid(frames_data, cell_w=320, cell_h=180):
    n = len(frames_data)
    cols = 4
    rows = (n + cols - 1) // cols
    grid = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
    for i, (frame, label) in enumerate(frames_data):
        if frame is None:
            continue
        r, c = divmod(i, cols)
        thumb = cv2.resize(frame, (cell_w, cell_h))
        cv2.putText(thumb, label, (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        grid[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w] = thumb
    return grid


def main():
    out_dir = Path("data/processed/frame_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Building video index...", flush=True)
    raw_idx = build_raw_video_index("data/raw/videos")
    print(f"  {len(raw_idx)} videos indexed", flush=True)

    ep_name = "Player122-f153ac423f61-20211217-140509"
    ori_fr = (398, 427)
    event_frame = 427

    # ── Collect masks from segmentation LMDB ──
    print("Scanning segmentation for masks...", flush=True)
    seg_masks = {}
    seg_dir = Path("data/raw/minestudio_aligned/segmentation")
    for part_dir in sorted(seg_dir.iterdir()):
        if not part_dir.is_dir() or not (part_dir / "data.mdb").exists():
            continue
        env = lmdb.open(str(part_dir), readonly=True, lock=False,
                        readahead=False, map_size=1024**3 * 100)
        txn_check = env.begin()
        raw_info = txn_check.get("__chunk_infos__".encode())
        if not raw_info:
            txn_check.abort()
            env.close()
            continue
        infos = pickle.loads(raw_info)
        if not any(i["episode"] == ep_name and i["episode_idx"] == 0
                   for i in infos):
            txn_check.abort()
            env.close()
            continue
        txn_check.abort()

        with env.begin() as txn:
            print(f"  Found partition: {part_dir.name}", flush=True)

            for key_raw, val_raw in txn.cursor():
                ks = key_raw.decode()
                if ks.startswith("__"):
                    continue
                ck = eval(ks)
                if ck[0] != 0:
                    continue
                fo = ck[1]
                frames = pickle.loads(val_raw)
                for fi, fd in enumerate(frames):
                    gf = fo + fi
                    for ek, ev in fd.items():
                        if not isinstance(ev, dict):
                            continue
                        if "coal_ore" not in ev.get("event", ""):
                            continue
                        ofr = ev.get("ori_frame_range")
                        fr = ev.get("frame_range")
                        if not ofr or tuple(ofr) != ori_fr:
                            continue
                        ori_f = gf + (ofr[0] - fr[0])
                        rle = ev.get("rle_mask", "")
                        pt = ev.get("point")
                        if rle:
                            mask = rle_to_mask(rle, 360, 640)
                            if pt:
                                mask = filter_mask_by_point(mask, pt)
                            bbox = mask_to_bbox(mask)
                            seg_masks[ori_f] = (mask, pt, bbox)
        env.close()
        break

    print(f"  Masks for {len(seg_masks)} ori frames: "
          f"{sorted(seg_masks.keys())[:8]}...", flush=True)

    # ── Image 1: mine_block event T-15 to T0 ──
    print(f"\nGenerating break event grid (ori frames {event_frame-15}..{event_frame})...",
          flush=True)
    frames_data = []
    for i in range(16):
        ori_f = event_frame - 15 + i
        frame = decode_multichunk_frame(raw_idx, ep_name, ori_f)
        if frame is None:
            print(f"  f{ori_f}: DECODE FAIL", flush=True)
            frames_data.append((None, ""))
            continue

        fh, fw = frame.shape[:2]

        if ori_f in seg_masks:
            mask, pt, bbox = seg_masks[ori_f]
            mask_r = cv2.resize(mask.astype(np.uint8), (fw, fh),
                                interpolation=cv2.INTER_NEAREST)
            overlay = frame.copy()
            overlay[mask_r > 0] = [0, 200, 0]
            frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
            if bbox:
                x, y, w, h = bbox
                sx, sy = fw / 640, fh / 360
                cv2.rectangle(frame,
                              (int(x * sx), int(y * sy)),
                              (int((x + w) * sx), int((y + h) * sy)),
                              (0, 255, 0), 2)

        cv2.drawMarker(frame, (fw // 2, fh // 2), (0, 0, 255),
                       cv2.MARKER_CROSS, 20, 2)

        t = ori_f - event_frame
        lbl = f"f{ori_f} T{t}"
        if t == 0:
            lbl = f"f{ori_f} T=0 BREAK"
        if ori_f in seg_masks:
            lbl += " [M]"
        frames_data.append((frame, lbl))

    grid1 = make_grid(frames_data)
    p1 = str(out_dir / "coal_ore_break_T-15_to_T0.png")
    cv2.imwrite(p1, grid1)
    print(f"  Saved: {p1}", flush=True)

    # ── Image 2: inventory coal pickup ──
    print("\nSearching JSONL for coal inventory increase...", flush=True)
    jsonl_files = sorted(glob.glob("data/raw/vpt_raw_video/**/*.jsonl",
                                   recursive=True))
    print(f"  {len(jsonl_files)} JSONL files", flush=True)

    for jf in jsonl_files:
        jf_ep = Path(jf).stem
        info = build_episode_chunk_info(raw_idx, jf_ep)
        if info is None:
            continue
        prev_coal = 0
        pickup_frame = None
        with open(jf) as f:
            for fi, line in enumerate(f):
                d = json.loads(line)
                inv = d.get("inventory", [])
                coal = sum(x["quantity"] for x in inv
                           if x.get("type") in ("coal", "charcoal"))
                if coal > prev_coal:
                    pickup_frame = fi
                    print(f"  Coal pickup in {jf_ep} at frame {fi}: "
                          f"{prev_coal} -> {coal}", flush=True)
                    break
                prev_coal = coal
        if pickup_frame is None:
            continue

        frames_data2 = []
        for i in range(16):
            ori_f = pickup_frame - 15 + i
            if ori_f < 0:
                frames_data2.append((None, ""))
                continue
            frame = decode_multichunk_frame(raw_idx, jf_ep, ori_f)
            if frame is None:
                frames_data2.append((None, ""))
                continue
            fh, fw = frame.shape[:2]
            cv2.drawMarker(frame, (fw // 2, fh // 2), (0, 0, 255),
                           cv2.MARKER_CROSS, 20, 2)
            t = ori_f - pickup_frame
            lbl = f"f{ori_f} T{t}" if t < 0 else f"f{ori_f} T=0 PICKUP"
            frames_data2.append((frame, lbl))

        grid2 = make_grid(frames_data2)
        p2 = str(out_dir / "coal_inventory_pickup_T-15_to_T0.png")
        cv2.imwrite(p2, grid2)
        print(f"  Saved: {p2}", flush=True)
        break
    else:
        print("  No coal pickup found in any JSONL", flush=True)

    print("\nDone!", flush=True)


if __name__ == "__main__":
    main()
