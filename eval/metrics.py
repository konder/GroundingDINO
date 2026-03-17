"""Evaluation metrics for bounding box detection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence


def compute_iou(
    box_a: Sequence[float],
    box_b: Sequence[float],
) -> float:
    """Compute IoU between two boxes in [x1, y1, x2, y2] normalized format."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def compute_accuracy_at_thresholds(
    ious: Sequence[float],
    thresholds: Sequence[float] = (0.25, 0.5, 0.75),
) -> dict[float, float]:
    """Compute detection accuracy at various IoU thresholds."""
    if len(ious) == 0:
        return {t: 0.0 for t in thresholds}
    return {
        t: sum(1 for iou in ious if iou >= t) / len(ious)
        for t in thresholds
    }


def summarize_results(results: list) -> dict:
    """Produce a summary dict from a list of EvalResult objects.

    Returns dict with keys: mean_iou, accuracy, detection_rate, per_task, total.
    """
    if not results:
        return {
            "mean_iou": 0.0,
            "accuracy": {},
            "detection_rate": 0.0,
            "per_task": [],
            "total": 0,
        }

    ious = [r.iou for r in results]
    mean_iou = sum(ious) / len(ious)
    detection_rate = sum(1 for r in results if r.has_prediction) / len(results)
    accuracy = compute_accuracy_at_thresholds(ious)

    per_task = [
        {
            "task": r.task,
            "label": r.label,
            "iou": r.iou,
            "pred_score": r.pred_score,
            "has_prediction": r.has_prediction,
        }
        for r in results
    ]

    return {
        "mean_iou": mean_iou,
        "accuracy": accuracy,
        "detection_rate": detection_rate,
        "per_task": per_task,
        "total": len(results),
    }
