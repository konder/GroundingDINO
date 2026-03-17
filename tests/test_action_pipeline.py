"""Tests for scripts/build_action_dataset.py"""
import cv2
import numpy as np
import os
import pickle
import pytest

from scripts.build_action_dataset import (
    build_action_index,
    read_action_chunk,
    find_attack_start,
    find_attack_start_for_event,
    select_preattack_frames,
    select_early_range_frames,
    EventInfo,
    extract_events_from_segmentation,
)


# ---------------------------------------------------------------------------
# build_action_index
# ---------------------------------------------------------------------------

class TestBuildActionIndex:
    def test_builds_index(self, tmp_path):
        import lmdb as _lmdb
        p = tmp_path / "action" / "part-100"
        p.mkdir(parents=True)
        env = _lmdb.open(str(p), map_size=10 * 1024 * 1024)
        infos = [
            {"episode": "PlayerA-abc-20210101-120000", "episode_idx": 0, "num_frames": 5000},
            {"episode": "PlayerB-def-20210202-130000", "episode_idx": 1, "num_frames": 3000},
        ]
        with env.begin(write=True) as txn:
            txn.put("__chunk_infos__".encode(), pickle.dumps(infos))
        env.close()

        index = build_action_index(str(tmp_path / "action"))
        assert "PlayerA-abc-20210101-120000" in index
        assert index["PlayerA-abc-20210101-120000"] == (str(p), 0)
        assert "PlayerB-def-20210202-130000" in index
        assert index["PlayerB-def-20210202-130000"] == (str(p), 1)

    def test_multiple_partitions(self, tmp_path):
        import lmdb as _lmdb
        for part_name, ep_name, ep_idx in [
            ("part-100", "PlayerA", 0),
            ("part-200", "PlayerB", 5),
        ]:
            p = tmp_path / part_name
            p.mkdir()
            env = _lmdb.open(str(p), map_size=10 * 1024 * 1024)
            with env.begin(write=True) as txn:
                txn.put("__chunk_infos__".encode(), pickle.dumps(
                    [{"episode": ep_name, "episode_idx": ep_idx, "num_frames": 1000}]
                ))
            env.close()

        index = build_action_index(str(tmp_path))
        assert len(index) == 2
        assert "PlayerA" in index
        assert "PlayerB" in index


# ---------------------------------------------------------------------------
# read_action_chunk
# ---------------------------------------------------------------------------

class TestReadActionChunk:
    def test_reads_attack(self, tmp_path):
        import lmdb as _lmdb
        p = str(tmp_path / "action_lmdb")
        env = _lmdb.open(p, map_size=10 * 1024 * 1024)
        chunk_data = {
            "attack": np.array([0, 0, 1, 1, 1, 0] + [0] * 26, dtype=np.int64),
            "use": np.zeros(32, dtype=np.int64),
        }
        with env.begin(write=True) as txn:
            txn.put("(0, 64)".encode(), pickle.dumps(chunk_data))
        env.close()

        result = read_action_chunk(p, 0, 64)
        assert result is not None
        assert "attack" in result
        np.testing.assert_array_equal(result["attack"][:6], [0, 0, 1, 1, 1, 0])

    def test_missing_chunk(self, tmp_path):
        import lmdb as _lmdb
        p = str(tmp_path / "empty_lmdb")
        env = _lmdb.open(p, map_size=10 * 1024 * 1024)
        env.close()

        result = read_action_chunk(p, 0, 999)
        assert result is None


# ---------------------------------------------------------------------------
# find_attack_start
# ---------------------------------------------------------------------------

class TestFindAttackStart:
    def test_finds_start_in_same_chunk(self, tmp_path):
        import lmdb as _lmdb
        p = str(tmp_path / "lmdb")
        env = _lmdb.open(p, map_size=10 * 1024 * 1024)
        # Chunk at offset 96: attack starts at frame 100 (index 4)
        attack = np.zeros(32, dtype=np.int64)
        attack[4:] = 1  # frames 100-127 have attack=1
        with env.begin(write=True) as txn:
            txn.put("(0, 96)".encode(), pickle.dumps({"attack": attack}))
        env.close()

        start = find_attack_start(p, 0, trace_from=110, chunk_size=32)
        assert start == 100

    def test_finds_start_across_chunks(self, tmp_path):
        import lmdb as _lmdb
        p = str(tmp_path / "lmdb")
        env = _lmdb.open(p, map_size=10 * 1024 * 1024)
        # Chunk at offset 64: all attack=1
        # Chunk at offset 32: attack starts at index 20 (frame 52)
        with env.begin(write=True) as txn:
            atk64 = np.ones(32, dtype=np.int64)
            txn.put("(0, 64)".encode(), pickle.dumps({"attack": atk64}))
            atk32 = np.zeros(32, dtype=np.int64)
            atk32[20:] = 1  # frames 52-63 have attack=1
            txn.put("(0, 32)".encode(), pickle.dumps({"attack": atk32}))
        env.close()

        start = find_attack_start(p, 0, trace_from=80, chunk_size=32)
        assert start == 52

    def test_attack_from_beginning(self, tmp_path):
        """If attack=1 from the very start of the episode, return frame 0."""
        import lmdb as _lmdb
        p = str(tmp_path / "lmdb")
        env = _lmdb.open(p, map_size=10 * 1024 * 1024)
        with env.begin(write=True) as txn:
            txn.put("(0, 0)".encode(), pickle.dumps(
                {"attack": np.ones(32, dtype=np.int64)}))
            txn.put("(0, 32)".encode(), pickle.dumps(
                {"attack": np.ones(32, dtype=np.int64)}))
        env.close()

        start = find_attack_start(p, 0, trace_from=50, chunk_size=32)
        assert start == 0

    def test_no_action_data(self, tmp_path):
        import lmdb as _lmdb
        p = str(tmp_path / "lmdb")
        env = _lmdb.open(p, map_size=10 * 1024 * 1024)
        env.close()

        start = find_attack_start(p, 0, trace_from=50, chunk_size=32)
        assert start is None


# ---------------------------------------------------------------------------
# select_preattack_frames
# ---------------------------------------------------------------------------

class TestFindAttackStartForEvent:
    def test_short_attack_uses_real_start(self, tmp_path):
        """When attack_start is close to frame_range[0], use the real start."""
        import lmdb as _lmdb
        p = str(tmp_path / "lmdb")
        env = _lmdb.open(p, map_size=10 * 1024 * 1024)
        with env.begin(write=True) as txn:
            # Chunk at offset 96: attack starts at index 2 (frame 98)
            atk = np.zeros(32, dtype=np.int64)
            atk[2:] = 1
            txn.put("(0, 96)".encode(), pickle.dumps({"attack": atk}))
        env.close()

        result = find_attack_start_for_event(
            p, 0, event_frame=120, frame_range=(100, 120), chunk_size=32)
        assert result == 98

    def test_long_chain_uses_frame_range_start(self, tmp_path):
        """When attack has been held for a long time, fall back to frame_range[0]."""
        import lmdb as _lmdb
        p = str(tmp_path / "lmdb")
        env = _lmdb.open(p, map_size=10 * 1024 * 1024)
        with env.begin(write=True) as txn:
            # All chunks have continuous attack=1
            for offset in range(0, 320, 32):
                txn.put(f"(0, {offset})".encode(),
                        pickle.dumps({"attack": np.ones(32, dtype=np.int64)}))
        env.close()

        result = find_attack_start_for_event(
            p, 0, event_frame=300, frame_range=(250, 300),
            chunk_size=32, max_gap=50)
        assert result == 250

    def test_attack_starts_after_frame_range(self, tmp_path):
        """When attack_start > frame_range[0] (e.g. use event), return frame_range[0]."""
        import lmdb as _lmdb
        p = str(tmp_path / "lmdb")
        env = _lmdb.open(p, map_size=10 * 1024 * 1024)
        with env.begin(write=True) as txn:
            # No attack in chunk containing frame_range[0]
            txn.put("(0, 96)".encode(),
                    pickle.dumps({"attack": np.zeros(32, dtype=np.int64)}))
            # Attack starts in next chunk at index 5 (frame 133)
            atk = np.zeros(32, dtype=np.int64)
            atk[5:] = 1
            txn.put("(0, 128)".encode(), pickle.dumps({"attack": atk}))
        env.close()

        result = find_attack_start_for_event(
            p, 0, event_frame=150, frame_range=(100, 150), chunk_size=32)
        assert result == 100

    def test_no_action_data(self, tmp_path):
        import lmdb as _lmdb
        p = str(tmp_path / "empty")
        env = _lmdb.open(p, map_size=10 * 1024 * 1024)
        env.close()

        result = find_attack_start_for_event(
            p, 0, event_frame=100, frame_range=(80, 100))
        assert result is None


class TestSelectPreattackFrames:
    def test_normal_case(self):
        frames = select_preattack_frames(attack_start=100, n_before=8)
        assert frames == list(range(92, 100))

    def test_clamp_to_zero(self):
        frames = select_preattack_frames(attack_start=3, n_before=8)
        assert frames == [0, 1, 2]
        assert len(frames) == 3

    def test_attack_at_zero(self):
        frames = select_preattack_frames(attack_start=0, n_before=8)
        assert frames == []


class TestSelectEarlyRangeFrames:
    def test_normal_case(self):
        frames = select_early_range_frames((100, 130), n_frames=8)
        assert frames == list(range(100, 108))

    def test_short_range(self):
        frames = select_early_range_frames((100, 103), n_frames=8)
        assert frames == [100, 101, 102]

    def test_single_frame_range(self):
        frames = select_early_range_frames((100, 101), n_frames=8)
        assert frames == [100]


# ---------------------------------------------------------------------------
# EventInfo / extract_events_from_segmentation
# ---------------------------------------------------------------------------

class TestExtractEvents:
    def test_extracts_events(self, tmp_path):
        import lmdb as _lmdb
        seg_dir = tmp_path / "segmentation" / "part-100"
        seg_dir.mkdir(parents=True)
        env = _lmdb.open(str(seg_dir), map_size=10 * 1024 * 1024)
        infos = [{"episode": "Player1-abc-20210101-120000", "episode_idx": 0, "num_frames": 5000}]
        frame_data = [{}] * 32
        frame_data[10] = {
            (42, "mine_block:stone"): {
                "event": "mine_block:stone",
                "rle_mask": "100 50",
                "point": (180, 320),
                "frame_id": 330,
                "frame_range": (300, 330),
            }
        }
        with env.begin(write=True) as txn:
            txn.put("__chunk_infos__".encode(), pickle.dumps(infos))
            txn.put("(0, 320)".encode(), pickle.dumps(frame_data))
        env.close()

        events = extract_events_from_segmentation(str(tmp_path))
        assert len(events) == 1
        ev = events[0]
        assert ev.label == "stone"
        assert ev.event_name == "mine_block:stone"
        assert ev.episode_name == "Player1-abc-20210101-120000"
        assert ev.event_frame == 330
        assert ev.frame_range == (300, 330)
