from pathlib import Path

import numpy as np
import pytest

from wildtriage_ml.adapters.sam3 import Sam3VideoTracker, merge_frame_outputs, split_frame_folder


def test_split_frame_folder_chunks_in_order():
    frames = [Path(f"{i}.jpg") for i in range(7)]
    chunks = split_frame_folder(frames, 3)
    assert [len(c) for c in chunks] == [3, 3, 1]
    assert chunks[1][0] == Path("3.jpg")


def test_split_frame_folder_rejects_bad_length():
    with pytest.raises(ValueError):
        split_frame_folder([], 0)


def test_merge_frame_outputs_builds_tracks_with_offsets():
    outputs = {
        0: {"out_obj_ids": np.array([1, 2]), "out_probs": np.array([0.9, 0.4]),
            "out_boxes_xywh": np.array([[0.1, 0.1, 0.2, 0.2], [0.5, 0.5, 0.1, 0.1]])},
        1: {"out_obj_ids": np.array([1]), "out_probs": np.array([0.95]),
            "out_boxes_xywh": np.array([[0.12, 0.1, 0.2, 0.2]])},
    }
    tracks = {t.track_key: t for t in merge_frame_outputs(outputs, frame_offset=180, key_offset=2000, label="animal")}
    assert set(tracks) == {2001, 2002}
    assert tracks[2001].frames == [180, 181]
    assert tracks[2001].score == pytest.approx(0.95)
    assert tracks[2002].label == "animal"
    assert len(tracks[2002].boxes) == 1


def test_merge_frame_outputs_handles_empty_frames():
    assert merge_frame_outputs({0: {"out_obj_ids": [], "out_boxes_xywh": []}}, 0, 0, None) == []


def test_tracker_requires_cuda_and_load_before_track():
    assert Sam3VideoTracker.info.devices == ("cuda",)
    pytest.importorskip("torch")
    with pytest.raises(RuntimeError):
        Sam3VideoTracker().track(Path("x"), ["animal"])
