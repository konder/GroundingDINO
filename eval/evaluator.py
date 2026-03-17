"""Core evaluation logic: load annotations, run model, compare predictions."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from torchvision.ops import box_convert

from groundingdino.util.inference import load_image, predict as predict_fn  # noqa: F401
from eval.metrics import compute_iou


@dataclass
class EvalResult:
    task: str
    label: str
    iou: float
    gt_bbox: List[float]
    pred_bbox: Optional[List[float]]
    pred_score: float
    has_prediction: bool


def load_annotations(annotation_path: str) -> List[dict]:
    """Load annotations from the Minecraft test JSON file.

    Returns a flat list of annotation dicts, each with keys:
    task, image, prompt, interaction_type, bbox, label.
    """
    with open(annotation_path, "r") as f:
        data = json.load(f)
    return list(data["annotations"].values())


def select_top_prediction(
    boxes: torch.Tensor,
    logits: torch.Tensor,
) -> Tuple[Optional[List[float]], float]:
    """Select the prediction with the highest confidence score.

    Args:
        boxes: (N, 4) tensor in cxcywh normalized format.
        logits: (N,) tensor of confidence scores.

    Returns:
        (bbox_xyxy_normalized | None, score)
    """
    if boxes.numel() == 0:
        return None, 0.0

    top_idx = logits.argmax().item()
    score = logits[top_idx].item()
    box_cxcywh = boxes[top_idx].unsqueeze(0)
    box_xyxy = box_convert(box_cxcywh, in_fmt="cxcywh", out_fmt="xyxy").squeeze(0)
    return box_xyxy.tolist(), score


def evaluate_single(
    model,
    annotation: dict,
    image_dir: str,
    device: str = "cpu",
    box_threshold: float = 0.1,
    text_threshold: float = 0.1,
) -> EvalResult:
    """Run inference on one image and compare with ground truth.

    Uses low thresholds by default to ensure we get at least some detections,
    then selects the highest-scoring one.
    """
    image_path = os.path.join(image_dir, annotation["image"])
    image_source, image_tensor = load_image(image_path)

    boxes, logits, phrases = predict_fn(
        model=model,
        image=image_tensor,
        caption=annotation["prompt"],
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=device,
    )

    pred_bbox, pred_score = select_top_prediction(boxes, logits)
    gt_bbox = annotation["bbox"]

    if pred_bbox is not None:
        iou = compute_iou(gt_bbox, pred_bbox)
        has_prediction = True
    else:
        iou = 0.0
        has_prediction = False

    return EvalResult(
        task=annotation["task"],
        label=annotation["label"],
        iou=iou,
        gt_bbox=gt_bbox,
        pred_bbox=pred_bbox,
        pred_score=pred_score,
        has_prediction=has_prediction,
    )


def run_evaluation(
    model,
    annotation_path: str,
    image_dir: str,
    device: str = "cpu",
    box_threshold: float = 0.1,
    text_threshold: float = 0.1,
) -> List[EvalResult]:
    """Run evaluation on all annotations and return per-task results."""
    annotations = load_annotations(annotation_path)
    results = []
    for ann in annotations:
        result = evaluate_single(
            model=model,
            annotation=ann,
            image_dir=image_dir,
            device=device,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        results.append(result)
    return results
