"""CLI entry point: python -m eval [options]"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import torch

from groundingdino.util.inference import load_model
from eval.evaluator import load_annotations, run_evaluation
from eval.metrics import summarize_results
from eval.visualize import generate_report, generate_visual_grid


def get_best_device() -> str:
    """Auto-detect the best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate GroundingDINO on Minecraft test set"
    )
    parser.add_argument(
        "-c", "--config",
        default="groundingdino/config/GroundingDINO_SwinT_OGC.py",
        help="Model config file path",
    )
    parser.add_argument(
        "-p", "--checkpoint",
        default="weights/groundingdino_swint_ogc.pth",
        help="Model checkpoint file path",
    )
    parser.add_argument(
        "-a", "--annotations",
        default="data/test/annotations.json",
        help="Annotations JSON file path",
    )
    parser.add_argument(
        "-i", "--image-dir",
        default="data/test",
        help="Directory containing test images",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to run inference on (default: auto-detect)",
    )
    parser.add_argument(
        "--box-threshold",
        type=float,
        default=0.1,
        help="Box confidence threshold (low to ensure detections)",
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=0.1,
        help="Text similarity threshold",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="logs",
        help="Directory to save results (default: logs/)",
    )
    parser.add_argument(
        "--tag",
        default="baseline",
        help="Tag for this evaluation run (used in filenames)",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip visualization report generation",
    )
    return parser.parse_args()


def print_report(summary: dict) -> None:
    print("\n" + "=" * 64)
    print("  GroundingDINO Minecraft Evaluation Results")
    print("=" * 64)
    print(f"  Total samples:    {summary['total']}")
    print(f"  Detection rate:   {summary['detection_rate']:.1%}")
    print(f"  Mean IoU:         {summary['mean_iou']:.4f}")
    print()
    print("  Accuracy @ IoU thresholds:")
    for threshold, acc in sorted(summary["accuracy"].items(), key=lambda x: float(x[0])):
        print(f"    IoU >= {float(threshold):.2f}:  {acc:.1%}")

    print()
    print("  Per-task breakdown:")
    print(f"  {'Task':<25s} {'Label':<18s} {'IoU':>6s} {'Score':>6s} {'Det?':>5s}")
    print("  " + "-" * 62)
    for t in summary["per_task"]:
        det = "Y" if t["has_prediction"] else "N"
        print(
            f"  {t['task']:<25s} {t['label']:<18s} "
            f"{t['iou']:>6.3f} {t['pred_score']:>6.3f} {det:>5s}"
        )
    print("=" * 64)


def main() -> None:
    args = parse_args()

    if args.device == "auto":
        args.device = get_best_device()

    timestamp = make_timestamp()
    os.makedirs(args.output_dir, exist_ok=True)

    json_path = os.path.join(args.output_dir, f"eval_{args.tag}_{timestamp}.json")
    report_path = os.path.join(args.output_dir, f"eval_{args.tag}_{timestamp}.png")

    print(f"Using device: {args.device}")
    print(f"Loading model from {args.checkpoint} ...")
    model = load_model(args.config, args.checkpoint, device=args.device)

    print(f"Running evaluation on {args.annotations} ...")
    results = run_evaluation(
        model=model,
        annotation_path=args.annotations,
        image_dir=args.image_dir,
        device=args.device,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )

    summary = summarize_results(results)
    summary_serializable = {
        "tag": args.tag,
        "timestamp": timestamp,
        "device": args.device,
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "mean_iou": summary["mean_iou"],
        "detection_rate": summary["detection_rate"],
        "accuracy": {str(k): v for k, v in summary["accuracy"].items()},
        "per_task": summary["per_task"],
        "total": summary["total"],
    }

    with open(json_path, "w") as f:
        json.dump(summary_serializable, f, indent=2, ensure_ascii=False)

    print_report(summary_serializable)
    print(f"\n  Results saved to: {json_path}")

    if not args.no_viz:
        generate_report(summary_serializable, report_path)
        print(f"  Report image:     {report_path}")

        grid_path = os.path.join(
            args.output_dir, f"eval_{args.tag}_{timestamp}_grid.png")
        annotations = load_annotations(args.annotations)
        generate_visual_grid(results, annotations, args.image_dir, grid_path)
        print(f"  Visual grid:      {grid_path}")

    print()


if __name__ == "__main__":
    main()
