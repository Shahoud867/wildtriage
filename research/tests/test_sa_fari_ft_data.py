import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.training.data import SaFariFrameLoader, build_subset, select_videos, split_locations  # noqa: E402

RLE = {"size": [100, 200], "counts": "abc"}


def _gt():
    videos = [{"id": i, "video_name": f"v{i}", "file_names": [f"v{i}/{f:05d}.jpg" for f in range(6)], "length": 6,
               "height": 100, "width": 200, "location_id": f"L{i % 4}"} for i in range(8)]
    # video 0: one animal, present in frames 0-2 only; video 1: a second positive phrase
    ann = [{"id": 0, "video_id": 0, "category_id": 10, "noun_phrase": "zebra",
            "segmentations": [RLE, RLE, RLE, None, None, None],
            "bboxes": [[20, 10, 40, 30], [20, 10, 40, 30], [20, 10, 40, 30], None, None, None],
            "areas": [1200, 1200, 1200, None, None, None]}]
    pairs = [{"video_id": 0, "category_id": 10, "noun_phrase": "zebra", "num_masklets": 1},
             {"video_id": 0, "category_id": 11, "noun_phrase": "lion", "num_masklets": 0},
             {"video_id": 0, "category_id": 12, "noun_phrase": "giraffe", "num_masklets": 0}]
    return {"videos": videos, "annotations": ann, "video_np_pairs": pairs,
            "categories": [{"id": 10, "name": "zebra"}, {"id": 11, "name": "lion"}, {"id": 12, "name": "giraffe"}]}


def test_location_split_is_disjoint_and_selection_spreads_over_sites():
    gt = _gt()
    train_locs, val_locs = split_locations(gt["videos"], 0.25, seed=0)
    assert train_locs.isdisjoint(val_locs) and train_locs | val_locs == {"L0", "L1", "L2", "L3"}
    chosen = select_videos(gt["videos"], train_locs, n=3, seed=0)
    assert len({v["location_id"] for v in chosen}) == min(3, len(train_locs))


def test_windowed_dataset_rotates_through_all_frames_and_resumes_identically(tmp_path):
    pytest.importorskip("sam3")
    from research.training.dataset import WindowedSam3ImageDataset

    gt = _gt()
    path = tmp_path / "train.json"
    path.write_text(json.dumps(build_subset(gt, [gt["videos"][0]], frame_stride=1, neg_per_frame=0, seed=0)))

    def make():
        return WindowedSam3ImageDataset(
            img_folder=str(tmp_path), ann_file=str(path), transforms=[], max_ann_per_img=100, multiplier=1,
            training=True, use_caching=False, coco_json_loader=SaFariFrameLoader, epoch_frames=4)

    ds = make()
    seen = []
    for epoch in range(3):  # 6 frames, windows of 4: wraps around without dropping or duplicating within an epoch
        ds.set_epoch(epoch)
        assert len(ds) == 4 and len(set(ds.ids.tolist())) == 4
        seen.append(ds.ids.tolist())
    assert set(seen[0] + seen[1]) == set(range(6))
    fresh = make()
    fresh.set_epoch(2)  # a resumed run lands on the same window
    assert fresh.ids.tolist() == seen[2]


def test_frame_loader_gives_normalised_boxes_masks_and_negatives(tmp_path):
    gt = _gt()
    subset = build_subset(gt, [gt["videos"][0]], frame_stride=1, neg_per_frame=1, seed=0)
    assert len(subset["images"]) == 6 and len(subset["annotations"]) == 3
    path = tmp_path / "train.json"
    path.write_text(json.dumps(subset))
    loader = SaFariFrameLoader(str(path))

    queries, anns = loader.loadQueriesAndAnnotationsFromDatapoint(0)  # animal in frame
    assert [q["query_text"] for q in queries][0] == "zebra" and len(queries) == 2  # + 1 negative phrase
    assert queries[0]["object_ids_output"] == [0] and queries[1]["object_ids_output"] == []
    assert anns[0]["bbox"] == [0.1, 0.1, 0.2, 0.3] and anns[0]["segmentation"] == RLE

    queries, anns = loader.loadQueriesAndAnnotationsFromDatapoint(4)  # animal left the frame
    assert anns == [] and queries[0]["query_text"] == "zebra" and queries[0]["object_ids_output"] == []
    assert loader.loadImagesFromDatapoint(4)[0]["file_name"] == "v0/00004.jpg"
