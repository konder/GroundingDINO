"""Tests for scripts/build_lmdb_dataset.py"""
import json
import numpy as np
import os
import pickle
import pytest
import tempfile

from scripts.build_lmdb_dataset import (
    parse_event_label,
    rle_to_mask,
    mask_to_bbox,
    filter_mask_by_point,
    select_training_frames,
    scale_bbox_to_image,
    build_coco_output,
)


# ---------------------------------------------------------------------------
# parse_event_label
# ---------------------------------------------------------------------------

class TestParseEventLabel:
    def test_mine_block(self):
        assert parse_event_label("mine_block:stone") == "stone"

    def test_mine_block_namespaced(self):
        assert parse_event_label("minecraft.mine_block:minecraft.coal_ore") == "coal_ore"

    def test_craft_excluded(self):
        assert parse_event_label("craft:crafting_table") is None

    def test_custom_excluded(self):
        assert parse_event_label("custom:open_chest") is None

    def test_empty(self):
        assert parse_event_label("") is None

    def test_kill_entity(self):
        assert parse_event_label("kill_entity:zombie") == "zombie"

    def test_use_item(self):
        assert parse_event_label("use_item:diamond_pickaxe") == "diamond_pickaxe"


# ---------------------------------------------------------------------------
# RLE mask utilities
# ---------------------------------------------------------------------------

class TestRleMask:
    def test_simple_mask(self):
        rle = "0 5 10 3"
        mask = rle_to_mask(rle, 4, 5)
        assert mask.shape == (4, 5)
        assert mask.sum() == 8  # 5 + 3

    def test_empty_rle(self):
        mask = rle_to_mask("", 10, 10)
        assert mask.sum() == 0

    def test_bbox_from_mask(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:5, 3:7] = 1
        bbox = mask_to_bbox(mask)
        assert bbox == [3, 2, 4, 3]

    def test_bbox_empty_mask(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        assert mask_to_bbox(mask) is None


# ---------------------------------------------------------------------------
# filter_mask_by_point
# ---------------------------------------------------------------------------

class TestFilterMaskByPoint:
    def test_keeps_component_at_point(self):
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[2:5, 2:5] = 1    # component A
        mask[15:18, 15:18] = 1  # component B
        filtered = filter_mask_by_point(mask, (3, 3))
        assert filtered[3, 3] == 1
        assert filtered[16, 16] == 0

    def test_none_point_returns_original(self):
        mask = np.ones((5, 5), dtype=np.uint8)
        result = filter_mask_by_point(mask, None)
        assert np.array_equal(result, mask)


# ---------------------------------------------------------------------------
# select_training_frames
# ---------------------------------------------------------------------------

class TestSelectTrainingFrames:
    def test_selects_early_frames(self):
        frames = select_training_frames(
            frame_start=100, frame_end=120, n_frames=4, skip_tail=4)
        assert len(frames) == 4
        # Should pick from early part, avoiding tail
        for f in frames:
            assert 100 <= f <= 116

    def test_short_range(self):
        frames = select_training_frames(
            frame_start=10, frame_end=12, n_frames=4, skip_tail=0)
        assert len(frames) <= 3  # only 3 frames available (10,11,12)
        for f in frames:
            assert 10 <= f <= 12

    def test_skip_tail(self):
        frames = select_training_frames(
            frame_start=0, frame_end=20, n_frames=4, skip_tail=8)
        for f in frames:
            assert f <= 12  # 20 - 8 = 12


# ---------------------------------------------------------------------------
# scale_bbox_to_image
# ---------------------------------------------------------------------------

class TestScaleBbox:
    def test_identity(self):
        bbox = [10, 20, 30, 40]
        result = scale_bbox_to_image(bbox, (360, 640), (360, 640))
        assert result == [10.0, 20.0, 30.0, 40.0]

    def test_downscale(self):
        bbox = [320, 180, 64, 36]
        result = scale_bbox_to_image(bbox, (360, 640), (224, 224))
        assert abs(result[0] - 320 * 224 / 640) < 0.01
        assert abs(result[1] - 180 * 224 / 360) < 0.01


# ---------------------------------------------------------------------------
# build_coco_output
# ---------------------------------------------------------------------------

class TestBuildCocoOutput:
    def test_basic_structure(self):
        annotations = [
            {
                "image_file": "ep0_f000100.png",
                "category": "stone",
                "bbox": [10.0, 20.0, 30.0, 40.0],
                "image_width": 224,
                "image_height": 224,
            },
            {
                "image_file": "ep0_f000100.png",
                "category": "stone",
                "bbox": [50.0, 60.0, 20.0, 20.0],
                "image_width": 224,
                "image_height": 224,
            },
            {
                "image_file": "ep0_f000110.png",
                "category": "coal_ore",
                "bbox": [15.0, 25.0, 35.0, 45.0],
                "image_width": 224,
                "image_height": 224,
            },
        ]
        coco = build_coco_output(annotations)
        assert len(coco["images"]) == 2
        assert len(coco["annotations"]) == 3
        assert len(coco["categories"]) == 2
        cat_names = {c["name"] for c in coco["categories"]}
        assert "stone" in cat_names
        assert "coal_ore" in cat_names
