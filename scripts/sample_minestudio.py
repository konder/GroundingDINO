"""Sample a small subset from MineStudio LMDB dataset for local development.

Usage:
    # Step 1: Inspect the dataset structure (understand keys/values)
    python scripts/sample_minestudio.py inspect /mnt/nas/rocket2_train/dataset_6xx

    # Step 2: Sample N entries from each sub-database
    python scripts/sample_minestudio.py sample /mnt/nas/rocket2_train/dataset_6xx \
        --output data/raw/minestudio_sample --num-samples 10

Requires: pip install lmdb
"""
from __future__ import annotations

import argparse
import json
import lmdb
import os
import pickle
import shutil
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def find_lmdb_dirs(root: str) -> List[Path]:
    """Find all directories containing data.mdb files."""
    root_path = Path(root)
    results = []
    for mdb_file in sorted(root_path.rglob("data.mdb")):
        results.append(mdb_file.parent)
    return results


def open_lmdb_readonly(lmdb_dir: str) -> lmdb.Environment:
    return lmdb.open(
        str(lmdb_dir),
        readonly=True,
        lock=False,
        readahead=False,
        map_size=1024 * 1024 * 1024 * 100,  # 100GB map_size for large files
    )


def decode_value(raw: bytes) -> Any:
    """Try multiple decoding strategies for LMDB values."""
    # Try pickle first (most common in Python ML datasets)
    try:
        return pickle.loads(raw)
    except Exception:
        pass
    # Try JSON
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        pass
    # Return raw bytes info
    return f"<raw bytes, len={len(raw)}>"


def decode_key(raw: bytes) -> str:
    """Decode LMDB key to string."""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.hex()


def inspect_lmdb(lmdb_dir: str, max_keys: int = 5) -> Dict:
    """Inspect a single LMDB database: count entries, show sample keys/values."""
    env = open_lmdb_readonly(lmdb_dir)
    stat = env.stat()
    info = {
        "path": str(lmdb_dir),
        "entries": stat["entries"],
        "map_size_mb": env.info()["map_size"] / (1024 * 1024),
        "data_mdb_size_mb": os.path.getsize(os.path.join(lmdb_dir, "data.mdb")) / (1024 * 1024),
        "sample_keys": [],
        "sample_value_types": [],
    }

    with env.begin() as txn:
        cursor = txn.cursor()
        count = 0
        for key_raw, val_raw in cursor:
            if count >= max_keys:
                break
            key_str = decode_key(key_raw)
            val = decode_value(val_raw)
            val_summary = _summarize_value(val)
            info["sample_keys"].append(key_str)
            info["sample_value_types"].append(val_summary)
            count += 1

    env.close()
    return info


def _summarize_value(val: Any, max_depth: int = 2) -> Any:
    """Produce a human-readable summary of a decoded value."""
    if isinstance(val, dict):
        if max_depth <= 0:
            return f"{{dict, {len(val)} keys}}"
        return {k: _summarize_value(v, max_depth - 1) for k, v in list(val.items())[:10]}
    if isinstance(val, (list, tuple)):
        type_name = "list" if isinstance(val, list) else "tuple"
        if len(val) == 0:
            return f"[{type_name}, empty]"
        sample = _summarize_value(val[0], max_depth - 1)
        return f"[{type_name}, len={len(val)}, first={sample}]"
    if isinstance(val, bytes):
        return f"<bytes, len={len(val)}>"
    if isinstance(val, str):
        return val[:100] + ("..." if len(val) > 100 else "")
    if hasattr(val, "shape"):  # numpy array
        return f"<ndarray, shape={val.shape}, dtype={val.dtype}>"
    return repr(val)[:100]


def sample_lmdb(
    src_dir: str,
    dst_dir: str,
    num_samples: int,
    strategy: str = "first",
) -> int:
    """Copy a sample of entries from src LMDB to a new dst LMDB.

    Args:
        strategy: 'first' takes first N, 'uniform' takes evenly spaced entries.

    Returns number of entries copied.
    """
    src_env = open_lmdb_readonly(src_dir)
    stat = src_env.stat()
    total = stat["entries"]

    if total == 0:
        src_env.close()
        return 0

    actual_samples = min(num_samples, total)
    os.makedirs(dst_dir, exist_ok=True)

    dst_env = lmdb.open(str(dst_dir), map_size=1024 * 1024 * 1024 * 10)  # 10GB

    if strategy == "uniform" and total > actual_samples:
        step = total / actual_samples
        target_indices = {int(i * step) for i in range(actual_samples)}
    else:
        target_indices = None

    copied = 0
    with src_env.begin() as src_txn, dst_env.begin(write=True) as dst_txn:
        cursor = src_txn.cursor()
        for idx, (key, val) in enumerate(cursor):
            if target_indices is not None:
                if idx not in target_indices:
                    continue
            elif idx >= actual_samples:
                break

            dst_txn.put(key, val)
            copied += 1

            if copied >= actual_samples:
                break

    src_env.close()
    dst_env.close()
    return copied


# -------------------------------------------------------------------------
# CLI commands
# -------------------------------------------------------------------------

def cmd_inspect(args: argparse.Namespace) -> None:
    """Inspect all LMDB databases in the dataset directory."""
    lmdb_dirs = find_lmdb_dirs(args.dataset_dir)
    if not lmdb_dirs:
        print(f"No LMDB databases found in {args.dataset_dir}")
        sys.exit(1)

    print(f"Found {len(lmdb_dirs)} LMDB databases in {args.dataset_dir}\n")

    all_info = []
    for lmdb_dir in lmdb_dirs:
        rel = os.path.relpath(lmdb_dir, args.dataset_dir)
        print(f"--- {rel} ---")
        try:
            info = inspect_lmdb(str(lmdb_dir), max_keys=args.max_keys)
            print(f"  Entries: {info['entries']:,}")
            print(f"  File size: {info['data_mdb_size_mb']:.1f} MB")
            for i, (k, v) in enumerate(zip(info["sample_keys"], info["sample_value_types"])):
                print(f"  Key[{i}]: {k}")
                print(f"    Value: {v}")
            all_info.append(info)
        except Exception as e:
            print(f"  ERROR: {e}")
        print()

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_info, f, indent=2, default=str)
        print(f"Inspection results saved to {args.output}")


def cmd_sample(args: argparse.Namespace) -> None:
    """Sample entries from all LMDB databases."""
    lmdb_dirs = find_lmdb_dirs(args.dataset_dir)
    if not lmdb_dirs:
        print(f"No LMDB databases found in {args.dataset_dir}")
        sys.exit(1)

    output_root = Path(args.output)
    if output_root.exists() and not args.force:
        print(f"Output directory {output_root} already exists. Use --force to overwrite.")
        sys.exit(1)

    print(f"Sampling {args.num_samples} entries from each of {len(lmdb_dirs)} databases\n")

    manifest = {"source": args.dataset_dir, "num_samples": args.num_samples, "databases": []}

    for lmdb_dir in lmdb_dirs:
        rel = os.path.relpath(lmdb_dir, args.dataset_dir)
        dst_dir = output_root / rel
        print(f"  {rel}: ", end="", flush=True)

        try:
            if dst_dir.exists():
                shutil.rmtree(dst_dir)
            copied = sample_lmdb(
                str(lmdb_dir), str(dst_dir),
                num_samples=args.num_samples,
                strategy=args.strategy,
            )
            src_stat = open_lmdb_readonly(str(lmdb_dir))
            total = src_stat.stat()["entries"]
            src_stat.close()
            print(f"{copied}/{total} entries")
            manifest["databases"].append({
                "path": rel,
                "total_entries": total,
                "sampled_entries": copied,
            })
        except Exception as e:
            print(f"ERROR: {e}")
            manifest["databases"].append({"path": rel, "error": str(e)})

    manifest_path = output_root / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    total_size = sum(
        f.stat().st_size for f in output_root.rglob("*") if f.is_file()
    ) / (1024 * 1024)
    print(f"\nSample dataset created: {output_root}")
    print(f"Total size: {total_size:.1f} MB")
    print(f"Manifest: {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample MineStudio LMDB dataset for local development"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # inspect
    p_inspect = subparsers.add_parser("inspect", help="Inspect dataset structure")
    p_inspect.add_argument("dataset_dir", help="Root directory of MineStudio dataset")
    p_inspect.add_argument("--max-keys", type=int, default=3, help="Max sample keys per DB")
    p_inspect.add_argument("-o", "--output", help="Save inspection results to JSON")

    # sample
    p_sample = subparsers.add_parser("sample", help="Create a sample subset")
    p_sample.add_argument("dataset_dir", help="Root directory of MineStudio dataset")
    p_sample.add_argument("-o", "--output", default="data/raw/minestudio_sample",
                          help="Output directory for sample dataset")
    p_sample.add_argument("-n", "--num-samples", type=int, default=10,
                          help="Number of entries to sample per database")
    p_sample.add_argument("--strategy", choices=["first", "uniform"], default="uniform",
                          help="Sampling strategy: first N or uniformly spaced")
    p_sample.add_argument("--force", action="store_true", help="Overwrite existing output")

    args = parser.parse_args()

    if args.command == "inspect":
        cmd_inspect(args)
    elif args.command == "sample":
        cmd_sample(args)


if __name__ == "__main__":
    main()
