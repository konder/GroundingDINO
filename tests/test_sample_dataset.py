"""Tests for scripts/sample_minestudio.py utility functions."""
import json
import os
import pickle
import pytest
import lmdb

from scripts.sample_minestudio import (
    find_lmdb_dirs,
    inspect_lmdb,
    sample_lmdb,
    decode_key,
    decode_value,
    _summarize_value,
)


@pytest.fixture
def sample_lmdb_dir(tmp_path):
    """Create a small LMDB database for testing."""
    db_dir = tmp_path / "test_db"
    db_dir.mkdir()
    env = lmdb.open(str(db_dir), map_size=1024 * 1024 * 10)
    with env.begin(write=True) as txn:
        for i in range(20):
            key = f"episode_{i:04d}".encode()
            val = pickle.dumps({
                "frames": [f"frame_{j}" for j in range(5)],
                "actions": list(range(5)),
                "metadata": {"episode_id": i, "length": 5},
            })
            txn.put(key, val)
    env.close()
    return db_dir


@pytest.fixture
def dataset_tree(tmp_path, sample_lmdb_dir):
    """Create a directory tree mimicking MineStudio structure."""
    root = tmp_path / "dataset_6xx"

    for subdir in ["event", "video/video-100", "action/action-500"]:
        dst = root / subdir
        dst.mkdir(parents=True)
        env = lmdb.open(str(dst), map_size=1024 * 1024 * 10)
        with env.begin(write=True) as txn:
            for i in range(10):
                txn.put(f"key_{i}".encode(), pickle.dumps({"idx": i}))
        env.close()

    return root


class TestFindLmdbDirs:
    def test_finds_all_dbs(self, dataset_tree):
        dirs = find_lmdb_dirs(str(dataset_tree))
        assert len(dirs) == 3

    def test_empty_dir(self, tmp_path):
        dirs = find_lmdb_dirs(str(tmp_path))
        assert len(dirs) == 0


class TestDecodeKey:
    def test_utf8_key(self):
        assert decode_key(b"episode_0001") == "episode_0001"

    def test_binary_key(self):
        raw = bytes([0x80, 0x81, 0x82])
        result = decode_key(raw)
        assert result == raw.hex()


class TestDecodeValue:
    def test_pickle_value(self):
        data = {"a": 1, "b": [2, 3]}
        result = decode_value(pickle.dumps(data))
        assert result == data

    def test_json_value(self):
        data = {"x": "hello"}
        result = decode_value(json.dumps(data).encode())
        assert result == data

    def test_raw_bytes(self):
        result = decode_value(b"\x00\x01\x02")
        assert "raw bytes" in str(result)


class TestSummarizeValue:
    def test_dict(self):
        result = _summarize_value({"a": 1, "b": 2})
        assert isinstance(result, dict)

    def test_list(self):
        result = _summarize_value([1, 2, 3])
        assert "list" in result
        assert "len=3" in result

    def test_string(self):
        result = _summarize_value("hello")
        assert result == "hello"

    def test_nested_depth_limit(self):
        deep = {"a": {"b": {"c": {"d": 1}}}}
        result = _summarize_value(deep, max_depth=1)
        assert isinstance(result, dict)


class TestInspectLmdb:
    def test_basic_inspect(self, sample_lmdb_dir):
        info = inspect_lmdb(str(sample_lmdb_dir), max_keys=3)
        assert info["entries"] == 20
        assert len(info["sample_keys"]) == 3
        assert len(info["sample_value_types"]) == 3
        assert info["data_mdb_size_mb"] > 0

    def test_key_content(self, sample_lmdb_dir):
        info = inspect_lmdb(str(sample_lmdb_dir), max_keys=1)
        assert "episode_" in info["sample_keys"][0]


class TestSampleLmdb:
    def test_sample_first(self, sample_lmdb_dir, tmp_path):
        dst = tmp_path / "sampled"
        copied = sample_lmdb(str(sample_lmdb_dir), str(dst), num_samples=5, strategy="first")
        assert copied == 5

        env = lmdb.open(str(dst), readonly=True, lock=False)
        assert env.stat()["entries"] == 5
        env.close()

    def test_sample_uniform(self, sample_lmdb_dir, tmp_path):
        dst = tmp_path / "sampled_uniform"
        copied = sample_lmdb(str(sample_lmdb_dir), str(dst), num_samples=5, strategy="uniform")
        assert copied == 5

    def test_sample_more_than_available(self, sample_lmdb_dir, tmp_path):
        dst = tmp_path / "sampled_all"
        copied = sample_lmdb(str(sample_lmdb_dir), str(dst), num_samples=100)
        assert copied == 20

    def test_preserves_data(self, sample_lmdb_dir, tmp_path):
        dst = tmp_path / "sampled_verify"
        sample_lmdb(str(sample_lmdb_dir), str(dst), num_samples=3, strategy="first")

        env = lmdb.open(str(dst), readonly=True, lock=False)
        with env.begin() as txn:
            cursor = txn.cursor()
            for key_raw, val_raw in cursor:
                key = decode_key(key_raw)
                val = decode_value(val_raw)
                assert "episode_" in key
                assert "frames" in val
                assert "actions" in val
        env.close()
