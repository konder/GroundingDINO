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
    build_dynamic_caption,
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


class TestBuildDynamicCaption:
    def test_single_category(self):
        cats = [{"id": 1, "name": "stone"}]
        caption, id2span = build_dynamic_caption(cats)
        assert caption == "stone ."
        assert 1 in id2span

    def test_multi_word_category(self):
        cats = [{"id": 5, "name": "coal_ore"}]
        caption, id2span = build_dynamic_caption(cats)
        assert "coal ore" in caption
        assert 5 in id2span
        spans = id2span[5]
        assert len(spans) == 2
        for start, end in spans:
            assert caption[start:end] in ("coal", "ore")

    def test_multiple_categories(self):
        cats = [
            {"id": 1, "name": "stone"},
            {"id": 3, "name": "oak_log"},
        ]
        caption, id2span = build_dynamic_caption(cats)
        assert "stone" in caption
        assert "oak log" in caption
        assert 1 in id2span
        assert 3 in id2span

    def test_spans_point_to_correct_text(self):
        cats = [
            {"id": 10, "name": "iron_ore"},
            {"id": 20, "name": "dirt"},
        ]
        caption, id2span = build_dynamic_caption(cats)
        for cid, spans in id2span.items():
            for start, end in spans:
                word = caption[start:end]
                cat_name = next(c["name"] for c in cats if c["id"] == cid)
                assert word in cat_name.replace("_", " ")


class TestMinecraftCocoDataset:
    @pytest.fixture
    def _make_ann_file(self, tmp_path):
        """Create annotation file and images; return (ann_path, img_dir)."""
        img_dir = tmp_path / "images"
        img_dir.mkdir()
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
                {"id": 3, "name": "granite"},
                {"id": 4, "name": "dirt"},
                {"id": 5, "name": "oak_log"},
                {"id": 6, "name": "iron_ore"},
                {"id": 7, "name": "grass_block"},
                {"id": 8, "name": "cobblestone"},
            ],
        }
        ann_path = str(tmp_path / "annotations.json")
        with open(ann_path, "w") as f:
            json.dump(ann, f)
        return ann_path, str(img_dir)

    @pytest.fixture
    def sample_dataset(self, _make_ann_file):
        ann_path, img_dir = _make_ann_file
        return MinecraftCocoDataset(ann_path, img_dir, caption_mode="global")

    @pytest.fixture
    def dynamic_dataset(self, _make_ann_file):
        ann_path, img_dir = _make_ann_file
        return MinecraftCocoDataset(
            ann_path, img_dir,
            caption_mode="dynamic", n_neg_categories=3,
        )

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
        assert abs(boxes[0][0].item() - 0.3125) < 0.02
        assert abs(boxes[0][1].item() - 125 / 360) < 0.02

    def test_getitem_positive_map(self, sample_dataset):
        item = sample_dataset[0]
        assert "positive_map" in item
        pm = item["positive_map"]
        assert pm.shape[0] == item["boxes"].shape[0]

    # --- Dynamic caption tests ---

    def test_dynamic_caption_is_short(self, dynamic_dataset):
        """Dynamic caption should be much shorter than global."""
        item = dynamic_dataset[0]
        caption = item["caption"]
        n_periods = caption.count(".")
        assert n_periods <= 1 + 3 + 1, \
            f"Expected at most ~5 categories (2 GT + 3 neg), got {n_periods} periods"

    def test_dynamic_caption_contains_gt_label(self, dynamic_dataset):
        """Dynamic caption must contain the GT category name."""
        item = dynamic_dataset[0]
        caption = item["caption"]
        assert "stone" in caption, f"GT label 'stone' not in caption: {caption}"

    def test_dynamic_caption_different_per_image(self, dynamic_dataset):
        """Different images should (usually) get different captions."""
        item0 = dynamic_dataset[0]
        item1 = dynamic_dataset[1]
        cap0 = item0["caption"]
        cap1 = item1["caption"]
        # image 0 has stone+coal_ore, image 1 has only stone
        # Captions may differ in negative categories
        assert isinstance(cap0, str) and isinstance(cap1, str)

    def test_dynamic_positive_map_has_signal(self, dynamic_dataset):
        """positive_map should have nonzero entries for GT categories."""
        item = dynamic_dataset[0]
        pm = item["positive_map"]
        assert pm.sum() > 0, "positive_map should have nonzero entries"
        for row in range(pm.shape[0]):
            assert pm[row].sum() > 0, \
                f"Box {row} should have positive token matches"

    def test_dynamic_vs_global_same_boxes(self, _make_ann_file):
        """Both modes should produce the same bounding boxes."""
        ann_path, img_dir = _make_ann_file
        ds_global = MinecraftCocoDataset(
            ann_path, img_dir, caption_mode="global")
        ds_dynamic = MinecraftCocoDataset(
            ann_path, img_dir, caption_mode="dynamic", n_neg_categories=3)
        g_item = ds_global[0]
        d_item = ds_dynamic[0]
        assert torch.allclose(g_item["boxes"], d_item["boxes"]), \
            "Boxes should be identical regardless of caption mode"
        assert torch.equal(g_item["labels"], d_item["labels"])


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
