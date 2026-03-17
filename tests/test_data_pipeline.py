"""Tests for scripts/build_finetune_dataset.py"""
import json
import numpy as np
import os
import pickle
import pytest
import tempfile

from scripts.build_finetune_dataset import (
    rle_to_mask,
    mask_to_bbox,
    rle_to_bbox,
    infer_mask_resolution,
    parse_event_label,
    scale_bbox,
    DetectionAnnotation,
    to_coco_format,
)


# ---------------------------------------------------------------------------
# rle_to_mask
# ---------------------------------------------------------------------------

class TestRleToMask:
    def test_empty_string(self):
        mask = rle_to_mask("", 4, 4)
        assert mask.shape == (4, 4)
        assert mask.sum() == 0

    def test_simple_rle(self):
        # 4x4 image, mark pixels 0-3 (first row)
        mask = rle_to_mask("0 4", 4, 4)
        assert mask.shape == (4, 4)
        assert mask[0].sum() == 4
        assert mask[1:].sum() == 0

    def test_multiple_runs(self):
        # Two runs: pixels 0-1 and 4-5 in a 4x4 image
        mask = rle_to_mask("0 2 4 2", 4, 4)
        assert mask.sum() == 4
        assert mask[0, 0] == 1
        assert mask[0, 1] == 1
        assert mask[1, 0] == 1
        assert mask[1, 1] == 1

    def test_out_of_bounds_clipped(self):
        # Run extends beyond image, should be clipped
        mask = rle_to_mask("14 10", 4, 4)
        assert mask.sum() == 2  # only pixels 14, 15 within 4x4=16

    def test_whitespace_only(self):
        mask = rle_to_mask("   ", 4, 4)
        assert mask.sum() == 0


# ---------------------------------------------------------------------------
# mask_to_bbox
# ---------------------------------------------------------------------------

class TestMaskToBbox:
    def test_empty_mask(self):
        mask = np.zeros((4, 4), dtype=np.uint8)
        assert mask_to_bbox(mask) is None

    def test_single_pixel(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[3, 5] = 1
        bbox = mask_to_bbox(mask)
        assert bbox == [5, 3, 1, 1]

    def test_rectangle(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:5, 3:7] = 1
        bbox = mask_to_bbox(mask)
        assert bbox == [3, 2, 4, 3]

    def test_full_image(self):
        mask = np.ones((8, 6), dtype=np.uint8)
        bbox = mask_to_bbox(mask)
        assert bbox == [0, 0, 6, 8]


# ---------------------------------------------------------------------------
# rle_to_bbox
# ---------------------------------------------------------------------------

class TestRleToBbox:
    def test_empty_returns_none(self):
        assert rle_to_bbox("", 4, 4) is None

    def test_simple_block(self):
        # 8x8 image, fill row 2 cols 2-5 → rle start=18, len=4
        bbox = rle_to_bbox("18 4", 8, 8)
        assert bbox == [2, 2, 4, 1]


# ---------------------------------------------------------------------------
# parse_event_label
# ---------------------------------------------------------------------------

class TestParseEventLabel:
    def test_mine_block(self):
        assert parse_event_label("mine_block:coal_ore") == "coal_ore"

    def test_minecraft_prefix(self):
        assert parse_event_label("minecraft.mine_block:minecraft.coal_ore") == "coal_ore"

    def test_use_item(self):
        assert parse_event_label("use_item:torch") == "torch"

    def test_right_click_skipped(self):
        assert parse_event_label("right_click") is None

    def test_landmark_skipped(self):
        assert parse_event_label("landmark") is None

    def test_custom_open_chest(self):
        assert parse_event_label("custom:open_chest") is None

    def test_empty(self):
        assert parse_event_label("") is None

    def test_none(self):
        assert parse_event_label(None) is None


# ---------------------------------------------------------------------------
# scale_bbox
# ---------------------------------------------------------------------------

class TestScaleBbox:
    def test_identity(self):
        assert scale_bbox([10, 20, 30, 40], (360, 640), (360, 640)) == [10, 20, 30, 40]

    def test_half(self):
        result = scale_bbox([100, 100, 200, 100], (360, 640), (180, 320))
        assert result == [50, 50, 100, 50]

    def test_to_224(self):
        result = scale_bbox([320, 180, 64, 36], (360, 640), (224, 224))
        sx = 224 / 640
        sy = 224 / 360
        assert abs(result[0] - 320 * sx) < 0.01
        assert abs(result[1] - 180 * sy) < 0.01


# ---------------------------------------------------------------------------
# to_coco_format
# ---------------------------------------------------------------------------

class TestToCoco:
    def test_basic_conversion(self):
        anns = [
            DetectionAnnotation(
                image_id=0, category="coal_ore",
                bbox=[100, 50, 60, 40], point=[130, 70],
                event_type="mine_block:coal_ore",
                episode_id=0, frame_id=10016,
            ),
            DetectionAnnotation(
                image_id=1, category="stone",
                bbox=[200, 100, 80, 60], point=[240, 130],
                event_type="mine_block:stone",
                episode_id=0, frame_id=10017,
            ),
        ]
        coco = to_coco_format(anns, image_hw=(224, 224), mask_hw=(360, 640))

        assert len(coco["images"]) == 2
        assert len(coco["annotations"]) == 2
        assert len(coco["categories"]) == 2

        cat_names = {c["name"] for c in coco["categories"]}
        assert cat_names == {"coal_ore", "stone"}

        for img in coco["images"]:
            assert img["width"] == 224
            assert img["height"] == 224

    def test_same_frame_multiple_annotations(self):
        anns = [
            DetectionAnnotation(
                image_id=0, category="coal_ore",
                bbox=[100, 50, 60, 40], point=None,
                event_type="mine_block:coal_ore",
                episode_id=0, frame_id=10016,
            ),
            DetectionAnnotation(
                image_id=1, category="stone",
                bbox=[200, 100, 80, 60], point=None,
                event_type="mine_block:stone",
                episode_id=0, frame_id=10016,  # same frame
            ),
        ]
        coco = to_coco_format(anns, image_hw=(224, 224))
        assert len(coco["images"]) == 1  # same frame → one image entry
        assert len(coco["annotations"]) == 2


# ---------------------------------------------------------------------------
# infer_mask_resolution
# ---------------------------------------------------------------------------

class TestInferMaskResolution:
    def test_small_mask(self):
        h, w = infer_mask_resolution("100 5 200 3")
        assert (h, w) == (224, 224)

    def test_large_mask(self):
        h, w = infer_mask_resolution("100000 5")
        assert (h, w) == (360, 640)

    def test_empty(self):
        h, w = infer_mask_resolution("")
        assert (h, w) == (360, 640)
