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
    filter_mask_by_point,
    infer_mask_resolution,
    parse_event_label,
    scale_bbox,
    DetectionAnnotation,
    to_coco_format,
    read_episode_mapping,
    build_raw_video_index,
    find_raw_video,
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

    def test_with_point_filters_component(self):
        # 10x10 image: two blobs
        # blob A: rows 1-2, cols 1-2 → starts at 11,12,21,22
        # blob B: rows 7-8, cols 7-8 → starts at 77,78,87,88
        rle = "11 2 21 2 77 2 87 2"
        # Without point → bbox covers both blobs
        bbox_full = rle_to_bbox(rle, 10, 10)
        assert bbox_full == [1, 1, 8, 8]
        # With point on blob B → bbox covers only blob B
        bbox_b = rle_to_bbox(rle, 10, 10, point=(7, 7))
        assert bbox_b == [7, 7, 2, 2]
        # With point on blob A → bbox covers only blob A
        bbox_a = rle_to_bbox(rle, 10, 10, point=(1, 1))
        assert bbox_a == [1, 1, 2, 2]


# ---------------------------------------------------------------------------
# filter_mask_by_point
# ---------------------------------------------------------------------------

class TestFilterMaskByPoint:
    def test_none_point_returns_unchanged(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:4, 2:4] = 1
        result = filter_mask_by_point(mask, None)
        assert np.array_equal(result, mask)

    def test_keeps_component_at_point(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[1:3, 1:3] = 1  # blob A
        mask[7:9, 7:9] = 1  # blob B
        result = filter_mask_by_point(mask, (7, 7))
        assert result[7, 7] == 1
        assert result[1, 1] == 0
        assert result.sum() == 4

    def test_nearest_component_when_point_off_mask(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[5:7, 5:7] = 1
        result = filter_mask_by_point(mask, (4, 4))
        assert result[5, 5] == 1
        assert result.sum() == 4

    def test_empty_mask(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        result = filter_mask_by_point(mask, (5, 5))
        assert result.sum() == 0


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


# ---------------------------------------------------------------------------
# read_episode_mapping
# ---------------------------------------------------------------------------

class TestReadEpisodeMapping:
    def test_reads_chunk_infos(self, tmp_path):
        """LMDB with __chunk_infos__ → episode_idx to episode_name mapping."""
        import lmdb as _lmdb

        lmdb_path = str(tmp_path / "test_lmdb")
        env = _lmdb.open(lmdb_path, map_size=10 * 1024 * 1024)
        chunk_infos = [
            {"episode": "Player100-abc123-20210101-120000", "episode_idx": 0, "num_frames": 5000},
            {"episode": "Player200-def456-20210202-130000", "episode_idx": 1, "num_frames": 3000},
        ]
        with env.begin(write=True) as txn:
            txn.put("__chunk_infos__".encode(), pickle.dumps(chunk_infos))
        env.close()

        mapping = read_episode_mapping(lmdb_path)
        assert mapping == {
            0: "Player100-abc123-20210101-120000",
            1: "Player200-def456-20210202-130000",
        }

    def test_empty_lmdb(self, tmp_path):
        """LMDB without __chunk_infos__ → empty mapping."""
        import lmdb as _lmdb

        lmdb_path = str(tmp_path / "empty_lmdb")
        env = _lmdb.open(lmdb_path, map_size=10 * 1024 * 1024)
        env.close()

        mapping = read_episode_mapping(lmdb_path)
        assert mapping == {}


# ---------------------------------------------------------------------------
# find_raw_video
# ---------------------------------------------------------------------------

class TestBuildRawVideoIndex:
    def test_flat_directory(self, tmp_path):
        (tmp_path / "Player100-abc.mp4").write_bytes(b"fake")
        (tmp_path / "Player200-def.mp4").write_bytes(b"fake")
        index = build_raw_video_index(str(tmp_path))
        assert "Player100-abc" in index
        assert "Player200-def" in index
        assert len(index) == 2

    def test_nested_directories(self, tmp_path):
        """Handles multi-level structures like all_6xx_Jun_29/data/6.0/xx.mp4"""
        deep = tmp_path / "all_6xx_Jun_29" / "data" / "6.0"
        deep.mkdir(parents=True)
        (deep / "Player100-abc.mp4").write_bytes(b"fake")

        deep2 = tmp_path / "all_6xx_Jun_29" / "data" / "6.13"
        deep2.mkdir(parents=True)
        (deep2 / "Player200-def.mp4").write_bytes(b"fake")

        index = build_raw_video_index(str(tmp_path))
        assert "Player100-abc" in index
        assert "Player200-def" in index
        assert "6.0" in index["Player100-abc"]
        assert "6.13" in index["Player200-def"]

    def test_empty_dir(self, tmp_path):
        index = build_raw_video_index(str(tmp_path))
        assert index == {}

    def test_ignores_non_mp4(self, tmp_path):
        (tmp_path / "data.txt").write_bytes(b"text")
        (tmp_path / "video.avi").write_bytes(b"avi")
        (tmp_path / "Player1.mp4").write_bytes(b"mp4")
        index = build_raw_video_index(str(tmp_path))
        assert len(index) == 1
        assert "Player1" in index


def _clear_raw_video_cache():
    """Clear the mutable default dict cache in find_raw_video."""
    cache = find_raw_video.__defaults__[0]
    cache.clear()


class TestFindRawVideo:
    def test_direct_match(self, tmp_path):
        mp4 = tmp_path / "Player100-abc.mp4"
        mp4.write_bytes(b"fake")
        _clear_raw_video_cache()
        result = find_raw_video(str(tmp_path), "Player100-abc")
        assert result == str(mp4)

    def test_deep_nested_match(self, tmp_path):
        deep = tmp_path / "all_6xx" / "data" / "6.0"
        deep.mkdir(parents=True)
        mp4 = deep / "Player100-abc.mp4"
        mp4.write_bytes(b"fake")
        _clear_raw_video_cache()
        result = find_raw_video(str(tmp_path), "Player100-abc")
        assert result == str(mp4)

    def test_no_match(self, tmp_path):
        _clear_raw_video_cache()
        result = find_raw_video(str(tmp_path), "nonexistent")
        assert result is None

    def test_index_cached(self, tmp_path):
        (tmp_path / "A.mp4").write_bytes(b"a")
        _clear_raw_video_cache()
        find_raw_video(str(tmp_path), "A")
        cache = find_raw_video.__defaults__[0]
        assert str(tmp_path) in cache


class TestPointNearCenter:
    """Test _point_near_center filter."""

    def test_exact_center(self):
        from scripts.build_finetune_dataset import _point_near_center
        import math
        center = (180, 320)
        half_diag = math.sqrt(180**2 + 320**2)
        assert _point_near_center((180, 320), center, 0.15, half_diag) is True

    def test_near_center(self):
        from scripts.build_finetune_dataset import _point_near_center
        import math
        center = (180, 320)
        half_diag = math.sqrt(180**2 + 320**2)
        assert _point_near_center((190, 330), center, 0.15, half_diag) is True

    def test_far_from_center(self):
        from scripts.build_finetune_dataset import _point_near_center
        import math
        center = (180, 320)
        half_diag = math.sqrt(180**2 + 320**2)
        assert _point_near_center((10, 10), center, 0.15, half_diag) is False

    def test_none_point(self):
        from scripts.build_finetune_dataset import _point_near_center
        assert _point_near_center(None, (180, 320), 0.15, 366.6) is False

    def test_threshold_boundary(self):
        from scripts.build_finetune_dataset import _point_near_center
        import math
        center = (180, 320)
        half_diag = math.sqrt(180**2 + 320**2)
        dist_20pct = half_diag * 0.20
        point = (180, 320 + int(dist_20pct) - 1)
        assert _point_near_center(point, center, 0.20, half_diag) is True
        point_far = (180, 320 + int(dist_20pct) + 5)
        assert _point_near_center(point_far, center, 0.20, half_diag) is False
