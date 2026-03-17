"""Tests for scripts/train_finetune.py"""
import json
import os
import numpy as np
import pytest
import torch
from PIL import Image

from scripts.train_finetune import (
    MinecraftCocoDataset,
    build_caption_and_spans,
    compute_loss,
)


class TestBuildCaptionAndSpans:
    def test_single_category(self):
        categories = [{"id": 1, "name": "stone"}]
        caption, id2span = build_caption_and_spans(categories)
        assert "stone" in caption
        assert caption.endswith(".")
        assert 1 in id2span

    def test_multiple_categories(self):
        categories = [
            {"id": 1, "name": "stone"},
            {"id": 2, "name": "coal_ore"},
            {"id": 3, "name": "granite"},
        ]
        caption, id2span = build_caption_and_spans(categories)
        for cat in categories:
            assert cat["name"].replace("_", " ") in caption or cat["name"] in caption
            assert cat["id"] in id2span

    def test_underscore_categories(self):
        categories = [{"id": 1, "name": "coal_ore"}]
        caption, id2span = build_caption_and_spans(categories)
        assert "coal ore" in caption or "coal_ore" in caption


class TestMinecraftCocoDataset:
    @pytest.fixture
    def sample_dataset(self, tmp_path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        # Create a small test image
        img = Image.new("RGB", (640, 360), color=(100, 150, 200))
        img.save(str(img_dir / "test_001.png"))
        img.save(str(img_dir / "test_002.png"))

        ann = {
            "images": [
                {"id": 1, "file_name": "test_001.png", "width": 640, "height": 360},
                {"id": 2, "file_name": "test_002.png", "width": 640, "height": 360},
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1,
                 "bbox": [100, 50, 200, 150], "area": 30000, "iscrowd": 0},
                {"id": 2, "image_id": 1, "category_id": 2,
                 "bbox": [300, 100, 100, 80], "area": 8000, "iscrowd": 0},
                {"id": 3, "image_id": 2, "category_id": 1,
                 "bbox": [50, 30, 150, 120], "area": 18000, "iscrowd": 0},
            ],
            "categories": [
                {"id": 1, "name": "stone"},
                {"id": 2, "name": "coal_ore"},
            ],
        }
        ann_path = str(tmp_path / "annotations.json")
        with open(ann_path, "w") as f:
            json.dump(ann, f)

        return MinecraftCocoDataset(ann_path, str(img_dir))

    def test_len(self, sample_dataset):
        assert len(sample_dataset) == 2

    def test_getitem_returns_dict(self, sample_dataset):
        item = sample_dataset[0]
        assert "image" in item
        assert "boxes" in item
        assert "labels" in item
        assert "caption" in item

    def test_getitem_boxes_normalized(self, sample_dataset):
        item = sample_dataset[0]
        boxes = item["boxes"]
        assert boxes.shape[1] == 4
        assert (boxes >= 0).all() and (boxes <= 1).all(), \
            f"Boxes should be normalized to [0,1]: {boxes}"

    def test_getitem_boxes_cxcywh(self, sample_dataset):
        """Boxes should be in cxcywh format, normalized to [0,1]."""
        item = sample_dataset[0]
        boxes = item["boxes"]
        # After Normalize transform: xyxy -> cxcywh, divided by new image size
        # Original bbox [100,50,200,150] xywh → xyxy [100,50,300,200]
        # cx_orig = 200/640 = 0.3125, cy_orig = 125/360 ≈ 0.347
        # RandomResize may scale but ratios are preserved after normalization
        assert abs(boxes[0][0].item() - 0.3125) < 0.02
        assert abs(boxes[0][1].item() - 125 / 360) < 0.02

    def test_getitem_positive_map(self, sample_dataset):
        item = sample_dataset[0]
        assert "positive_map" in item
        pm = item["positive_map"]
        assert pm.shape[0] == item["boxes"].shape[0]


class TestComputeLoss:
    def test_loss_returns_scalar(self):
        B, N, C = 1, 900, 256
        outputs = {
            "pred_logits": torch.randn(B, N, C),
            "pred_boxes": torch.sigmoid(torch.randn(B, N, 4)),
        }
        targets = [{
            "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
            "positive_map": torch.zeros(1, C),
        }]
        targets[0]["positive_map"][0, 10:15] = 0.2

        loss_dict = compute_loss(outputs, targets)
        assert "loss_ce" in loss_dict
        assert "loss_bbox" in loss_dict
        assert "loss_giou" in loss_dict
        for v in loss_dict.values():
            assert v.dim() == 0, "Loss should be scalar"
            assert not torch.isnan(v), f"Loss should not be NaN"
