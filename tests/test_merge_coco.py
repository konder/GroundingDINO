"""Tests for scripts/merge_coco_datasets.py"""
import json
import os
import shutil
import pytest
from pathlib import Path

from scripts.merge_coco_datasets import merge_coco_datasets


@pytest.fixture
def two_datasets(tmp_path):
    """Create two small COCO datasets for testing."""
    for ds_name, cats, imgs, anns in [
        (
            "ds_a",
            [{"id": 1, "name": "stone"}, {"id": 2, "name": "coal_ore"}],
            [
                {"id": 1, "file_name": "img_001.png", "width": 640, "height": 360},
                {"id": 2, "file_name": "img_002.png", "width": 640, "height": 360},
            ],
            [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 100, 80], "area": 8000, "iscrowd": 0},
                {"id": 2, "image_id": 2, "category_id": 2, "bbox": [50, 60, 120, 90], "area": 10800, "iscrowd": 0},
            ],
        ),
        (
            "ds_b",
            [{"id": 1, "name": "stone"}, {"id": 3, "name": "granite"}],
            [
                {"id": 1, "file_name": "img_001.png", "width": 640, "height": 360},
                {"id": 2, "file_name": "img_003.png", "width": 640, "height": 360},
            ],
            [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [30, 40, 80, 60], "area": 4800, "iscrowd": 0},
                {"id": 2, "image_id": 2, "category_id": 3, "bbox": [100, 100, 150, 100], "area": 15000, "iscrowd": 0},
            ],
        ),
    ]:
        ds_dir = tmp_path / ds_name
        img_dir = ds_dir / "images"
        img_dir.mkdir(parents=True)
        for img_info in imgs:
            (img_dir / img_info["file_name"]).write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        with open(ds_dir / "annotations.json", "w") as f:
            json.dump({"images": imgs, "annotations": anns, "categories": cats}, f)

    return tmp_path


class TestMergeCoco:
    def test_basic_merge(self, two_datasets, tmp_path):
        out = tmp_path / "merged"
        merge_coco_datasets(
            input_dirs=[str(two_datasets / "ds_a"), str(two_datasets / "ds_b")],
            output_dir=str(out),
        )
        with open(out / "annotations.json") as f:
            data = json.load(f)

        assert len(data["images"]) == 4
        assert len(data["annotations"]) == 4

    def test_categories_unified(self, two_datasets, tmp_path):
        out = tmp_path / "merged"
        merge_coco_datasets(
            input_dirs=[str(two_datasets / "ds_a"), str(two_datasets / "ds_b")],
            output_dir=str(out),
        )
        with open(out / "annotations.json") as f:
            data = json.load(f)

        cat_names = {c["name"] for c in data["categories"]}
        assert cat_names == {"stone", "coal_ore", "granite"}

    def test_ids_unique(self, two_datasets, tmp_path):
        out = tmp_path / "merged"
        merge_coco_datasets(
            input_dirs=[str(two_datasets / "ds_a"), str(two_datasets / "ds_b")],
            output_dir=str(out),
        )
        with open(out / "annotations.json") as f:
            data = json.load(f)

        img_ids = [img["id"] for img in data["images"]]
        ann_ids = [ann["id"] for ann in data["annotations"]]
        assert len(img_ids) == len(set(img_ids)), "Image IDs must be unique"
        assert len(ann_ids) == len(set(ann_ids)), "Annotation IDs must be unique"

    def test_images_copied(self, two_datasets, tmp_path):
        out = tmp_path / "merged"
        merge_coco_datasets(
            input_dirs=[str(two_datasets / "ds_a"), str(two_datasets / "ds_b")],
            output_dir=str(out),
        )
        img_dir = out / "images"
        assert img_dir.exists()
        assert len(list(img_dir.iterdir())) == 4

    def test_filenames_no_collision(self, two_datasets, tmp_path):
        """ds_a and ds_b both have img_001.png; merged should prefix with source."""
        out = tmp_path / "merged"
        merge_coco_datasets(
            input_dirs=[str(two_datasets / "ds_a"), str(two_datasets / "ds_b")],
            output_dir=str(out),
        )
        with open(out / "annotations.json") as f:
            data = json.load(f)

        filenames = [img["file_name"] for img in data["images"]]
        assert len(filenames) == len(set(filenames)), f"Filenames must be unique: {filenames}"

    def test_symlink_mode(self, two_datasets, tmp_path):
        out = tmp_path / "merged_symlink"
        merge_coco_datasets(
            input_dirs=[str(two_datasets / "ds_a"), str(two_datasets / "ds_b")],
            output_dir=str(out),
            use_symlinks=True,
        )
        img_dir = out / "images"
        some_file = next(img_dir.iterdir())
        assert some_file.is_symlink()
