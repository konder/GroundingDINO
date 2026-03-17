"""Tests for eval module: metrics computation and evaluator logic."""
import json
import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from eval.metrics import compute_iou, compute_accuracy_at_thresholds, summarize_results
from eval.evaluator import (
    load_annotations,
    select_top_prediction,
    evaluate_single,
    EvalResult,
)
from eval.visualize import generate_report


# ---------------------------------------------------------------------------
# metrics.compute_iou
# ---------------------------------------------------------------------------

class TestComputeIoU:
    def test_perfect_overlap(self):
        box = [0.1, 0.2, 0.5, 0.6]
        assert compute_iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self):
        box_a = [0.0, 0.0, 0.1, 0.1]
        box_b = [0.5, 0.5, 0.6, 0.6]
        assert compute_iou(box_a, box_b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        box_a = [0.0, 0.0, 0.4, 0.4]
        box_b = [0.2, 0.2, 0.6, 0.6]
        inter_area = 0.2 * 0.2  # 0.04
        union_area = 0.4 * 0.4 + 0.4 * 0.4 - inter_area  # 0.28
        expected_iou = inter_area / union_area
        assert compute_iou(box_a, box_b) == pytest.approx(expected_iou, abs=1e-6)

    def test_one_contains_other(self):
        outer = [0.0, 0.0, 1.0, 1.0]
        inner = [0.2, 0.2, 0.4, 0.4]
        inner_area = 0.2 * 0.2
        outer_area = 1.0
        expected = inner_area / outer_area
        assert compute_iou(outer, inner) == pytest.approx(expected, abs=1e-6)

    def test_zero_area_box(self):
        box_a = [0.5, 0.5, 0.5, 0.5]
        box_b = [0.0, 0.0, 1.0, 1.0]
        assert compute_iou(box_a, box_b) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# metrics.compute_accuracy_at_thresholds
# ---------------------------------------------------------------------------

class TestAccuracyAtThresholds:
    def test_all_above_threshold(self):
        ious = [0.8, 0.9, 0.7, 0.85]
        acc = compute_accuracy_at_thresholds(ious, thresholds=[0.5])
        assert acc[0.5] == pytest.approx(1.0)

    def test_none_above_threshold(self):
        ious = [0.1, 0.2, 0.3]
        acc = compute_accuracy_at_thresholds(ious, thresholds=[0.5])
        assert acc[0.5] == pytest.approx(0.0)

    def test_mixed(self):
        ious = [0.6, 0.3, 0.8, 0.1]
        acc = compute_accuracy_at_thresholds(ious, thresholds=[0.25, 0.5, 0.75])
        assert acc[0.25] == pytest.approx(3 / 4)
        assert acc[0.5] == pytest.approx(2 / 4)
        assert acc[0.75] == pytest.approx(1 / 4)

    def test_empty_list(self):
        acc = compute_accuracy_at_thresholds([], thresholds=[0.5])
        assert acc[0.5] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# metrics.summarize_results
# ---------------------------------------------------------------------------

class TestSummarizeResults:
    def test_summary_structure(self):
        results = [
            EvalResult(task="a", label="x", iou=0.8, gt_bbox=[0, 0, 1, 1],
                       pred_bbox=[0, 0, 1, 1], pred_score=0.9, has_prediction=True),
            EvalResult(task="b", label="y", iou=0.3, gt_bbox=[0, 0, 1, 1],
                       pred_bbox=[0.5, 0.5, 1, 1], pred_score=0.7, has_prediction=True),
            EvalResult(task="c", label="z", iou=0.0, gt_bbox=[0, 0, 1, 1],
                       pred_bbox=None, pred_score=0.0, has_prediction=False),
        ]
        summary = summarize_results(results)
        assert "mean_iou" in summary
        assert "accuracy" in summary
        assert "detection_rate" in summary
        assert "per_task" in summary
        assert summary["detection_rate"] == pytest.approx(2 / 3)
        assert summary["mean_iou"] == pytest.approx((0.8 + 0.3 + 0.0) / 3)
        assert len(summary["per_task"]) == 3


# ---------------------------------------------------------------------------
# evaluator.load_annotations
# ---------------------------------------------------------------------------

class TestLoadAnnotations:
    def test_load_from_file(self, tmp_path):
        ann_data = {
            "total": 1,
            "annotations": {
                "test_task": {
                    "task": "test_task",
                    "image": "test.png",
                    "prompt": "Find the block",
                    "interaction_type": "mine",
                    "bbox": [0.1, 0.2, 0.3, 0.4],
                    "label": "block",
                }
            },
        }
        ann_file = tmp_path / "annotations.json"
        ann_file.write_text(json.dumps(ann_data))
        annotations = load_annotations(str(ann_file))
        assert len(annotations) == 1
        assert annotations[0]["task"] == "test_task"
        assert annotations[0]["bbox"] == [0.1, 0.2, 0.3, 0.4]


# ---------------------------------------------------------------------------
# evaluator.select_top_prediction
# ---------------------------------------------------------------------------

class TestSelectTopPrediction:
    def test_selects_highest_score(self):
        import torch
        boxes = torch.tensor([[0.3, 0.3, 0.2, 0.2], [0.7, 0.7, 0.1, 0.1]])
        logits = torch.tensor([0.5, 0.9])
        bbox, score = select_top_prediction(boxes, logits)
        assert score == pytest.approx(0.9)
        assert bbox == pytest.approx([0.65, 0.65, 0.75, 0.75], abs=1e-4)

    def test_empty_predictions(self):
        import torch
        boxes = torch.zeros(0, 4)
        logits = torch.zeros(0)
        bbox, score = select_top_prediction(boxes, logits)
        assert bbox is None
        assert score == 0.0

    def test_single_prediction(self):
        import torch
        boxes = torch.tensor([[0.5, 0.5, 0.4, 0.4]])
        logits = torch.tensor([0.85])
        bbox, score = select_top_prediction(boxes, logits)
        assert score == pytest.approx(0.85)
        assert bbox == pytest.approx([0.3, 0.3, 0.7, 0.7], abs=1e-4)


# ---------------------------------------------------------------------------
# evaluator.evaluate_single
# ---------------------------------------------------------------------------

class TestEvaluateSingle:
    def test_with_prediction(self):
        import torch

        mock_model = MagicMock()
        predicted_boxes = torch.tensor([[0.5, 0.5, 0.4, 0.4]])
        predicted_logits = torch.tensor([0.9])
        predicted_phrases = ["cow"]

        annotation = {
            "task": "hunt_cowonly",
            "image": "hunt_cowonly_frame0.png",
            "prompt": "Point to the cow",
            "bbox": [0.3, 0.3, 0.7, 0.7],
            "label": "cow",
        }

        with patch("eval.evaluator.load_image") as mock_load, \
             patch("eval.evaluator.predict_fn") as mock_predict:
            mock_load.return_value = (np.zeros((360, 640, 3), dtype=np.uint8),
                                      torch.zeros(3, 360, 640))
            mock_predict.return_value = (predicted_boxes, predicted_logits, predicted_phrases)

            result = evaluate_single(
                model=mock_model,
                annotation=annotation,
                image_dir="/fake/dir",
                device="cpu",
            )

        assert isinstance(result, EvalResult)
        assert result.has_prediction is True
        assert result.iou == pytest.approx(1.0)
        assert result.pred_score == pytest.approx(0.9)

    def test_no_detection(self):
        import torch

        mock_model = MagicMock()

        annotation = {
            "task": "test",
            "image": "test.png",
            "prompt": "Find something",
            "bbox": [0.1, 0.1, 0.5, 0.5],
            "label": "something",
        }

        with patch("eval.evaluator.load_image") as mock_load, \
             patch("eval.evaluator.predict_fn") as mock_predict:
            mock_load.return_value = (np.zeros((360, 640, 3), dtype=np.uint8),
                                      torch.zeros(3, 360, 640))
            mock_predict.return_value = (torch.zeros(0, 4), torch.zeros(0), [])

            result = evaluate_single(
                model=mock_model,
                annotation=annotation,
                image_dir="/fake/dir",
                device="cpu",
            )

        assert result.has_prediction is False
        assert result.iou == pytest.approx(0.0)
        assert result.pred_bbox is None


# ---------------------------------------------------------------------------
# visualize.generate_report
# ---------------------------------------------------------------------------

class TestGenerateReport:
    def test_generates_png(self, tmp_path):
        summary = {
            "tag": "test",
            "timestamp": "20260317_120000",
            "mean_iou": 0.45,
            "detection_rate": 0.8,
            "accuracy": {"0.25": 0.6, "0.5": 0.4, "0.75": 0.2},
            "per_task": [
                {"task": "mine_coal", "label": "coal_ore", "iou": 0.8,
                 "pred_score": 0.9, "has_prediction": True},
                {"task": "hunt_cow", "label": "cow", "iou": 0.3,
                 "pred_score": 0.5, "has_prediction": True},
                {"task": "place_door", "label": "door", "iou": 0.0,
                 "pred_score": 0.0, "has_prediction": False},
            ],
            "total": 3,
        }
        out_path = str(tmp_path / "report.png")
        result = generate_report(summary, out_path)
        assert os.path.exists(result)
        assert result.endswith(".png")
        assert os.path.getsize(result) > 1000
