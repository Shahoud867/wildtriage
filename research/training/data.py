"""SA-FARI *train* subset -> frame-level phrase-grounding data for fine-tuning SAM 3's detector.

    python research/training/prepare.py --n-train 2000 --n-val 150 --out D:/wildtriage-data/sa_fari_ft

Each datapoint is one frame plus the video's noun phrases: a positive phrase has the frame's masks (none if the
animal is out of frame, which is a valid negative), and a few of the video's negative phrases are added as
empty queries. `SaFariFrameLoader` is the `coco_json_loader` that sam3.train.data.Sam3ImageDataset plugs in.

Train and validation videos are split by camera location, so validation never shares a site with training.
SA-FARI *test* is never touched here.
"""

from __future__ import annotations

import json
import random
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

FRAMES_URL = "https://storage.googleapis.com/cxl-public-camera-trap/sa_fari/sa_fari_train/JPEGImages_6fps"


def location(video: dict) -> str:
    """Some videos carry a NaN / missing location id; they all share one 'unknown' site."""
    loc = video.get("location_id")
    return loc if isinstance(loc, str) and loc else "unknown"


def split_locations(videos: list[dict], val_fraction: float, seed: int) -> tuple[set[str], set[str]]:
    locations = sorted({location(v) for v in videos})
    random.Random(seed).shuffle(locations)
    n_val = max(1, round(len(locations) * val_fraction))
    return set(locations[n_val:]), set(locations[:n_val])


def select_videos(videos: list[dict], locations: set[str], n: int, seed: int) -> list[dict]:
    """Round-robin over locations so the subset covers as many sites as possible."""
    by_loc: dict[str, list[dict]] = defaultdict(list)
    for v in videos:
        if location(v) in locations:
            by_loc[location(v)].append(v)
    rng = random.Random(seed)
    queues = [rng.sample(vs, len(vs)) for _, vs in sorted(by_loc.items())]
    rng.shuffle(queues)
    chosen: list[dict] = []
    while queues and len(chosen) < n:
        for q in list(queues):
            chosen.append(q.pop())
            if not q:
                queues.remove(q)
            if len(chosen) >= n:
                break
    return chosen


def build_subset(gt: dict, videos: list[dict], frame_stride: int, neg_per_frame: int, seed: int) -> dict:
    """Compact frame-level JSON (pixel xywh boxes + RLE masks) for the given videos."""
    rng = random.Random(seed)
    names = {c["id"]: c["name"] for c in gt["categories"]}
    masklets: dict[int, list[dict]] = defaultdict(list)
    for a in gt["annotations"]:
        masklets[a["video_id"]].append(a)
    pairs: dict[int, dict[str, list[int]]] = defaultdict(lambda: {"pos": [], "neg": []})
    for p in gt["video_np_pairs"]:
        pairs[p["video_id"]]["pos" if p.get("num_masklets", 0) > 0 else "neg"].append(p["category_id"])

    images, annotations, used_cats = [], [], set()
    for v in videos:
        pos, neg = pairs[v["id"]]["pos"], pairs[v["id"]]["neg"]
        start = rng.randrange(frame_stride)
        for fi in range(start, v["length"], frame_stride):
            image_id = len(images)
            images.append({
                "id": image_id, "file_name": v["file_names"][fi], "height": v["height"], "width": v["width"],
                "video_id": v["id"], "frame_index": fi, "pos": pos,
                "neg": rng.sample(neg, min(neg_per_frame, len(neg))),
            })
            for m in masklets[v["id"]]:
                seg, box = m["segmentations"][fi], m["bboxes"][fi]
                if seg is None or box is None:
                    continue
                annotations.append({"id": len(annotations), "image_id": image_id, "category_id": m["category_id"],
                                    "bbox": box, "area": m["areas"][fi], "segmentation": seg, "iscrowd": 0})
        used_cats.update(pos)
    for img in images:
        used_cats.update(img["neg"])
    return {"images": images, "annotations": annotations,
            "categories": [{"id": c, "name": names[c]} for c in sorted(used_cats)]}


def download_frames(images: list[dict], root: Path, workers: int = 16, retries: int = 5) -> int:
    """Fetch the frames anonymously and resumably; returns how many were newly downloaded.

    Each frame is retried with backoff (the bucket drops connections under load); frames that still fail are
    reported together at the end rather than aborting the whole download.
    """
    def get(name: str) -> int:
        dest = root / name
        if dest.exists() and dest.stat().st_size > 0:
            return 0
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_suffix(".part")
        for attempt in range(retries):
            try:
                urllib.request.urlretrieve(f"{FRAMES_URL}/{name}", part)
                part.replace(dest)
                return 1
            except OSError:  # URLError, RemoteDisconnected, timeouts
                time.sleep(2 ** attempt)
        raise OSError(name)

    def safe_get(name: str) -> tuple[int, str | None]:
        try:
            return get(name), None
        except OSError:
            return 0, name

    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(safe_get, [i["file_name"] for i in images]))
    failed = [name for _, name in results if name]
    if failed:
        raise RuntimeError(f"{len(failed)} frames failed after {retries} tries, e.g. {failed[:3]}; re-run to resume")
    return sum(n for n, _ in results)


class SaFariFrameLoader:
    """`coco_json_loader` for Sam3ImageDataset: one datapoint per frame, one query per phrase."""

    def __init__(self, annotation_file: str):
        with open(annotation_file, encoding="utf-8") as f:
            data = json.load(f)
        self._images = data["images"]
        self._names = {c["id"]: c["name"] for c in data["categories"]}
        self._anns: dict[int, list[dict]] = defaultdict(list)
        for a in data["annotations"]:
            self._anns[a["image_id"]].append(a)

    def getDatapointIds(self) -> list[int]:
        return list(range(len(self._images)))

    def loadImagesFromDatapoint(self, idx: int) -> list[dict]:
        img = self._images[idx]
        return [{"id": 0, "file_name": img["file_name"], "original_img_id": img["id"], "coco_img_id": img["id"]}]

    def loadQueriesAndAnnotationsFromDatapoint(self, idx: int) -> tuple[list[dict], list[dict]]:
        img = self._images[idx]
        w, h = img["width"], img["height"]
        by_cat: dict[int, list[dict]] = defaultdict(list)
        for a in self._anns[img["id"]]:
            by_cat[a["category_id"]].append(a)

        queries, annotations = [], []
        for cat in list(img["pos"]) + list(img["neg"]):
            out_ids = []
            for a in by_cat.get(cat, []):
                x, y, bw, bh = a["bbox"]
                annotations.append({
                    "id": len(annotations), "object_id": len(annotations), "image_id": 0, "is_crowd": 0,
                    "bbox": [x / w, y / h, bw / w, bh / h], "area": (bw / w) * (bh / h),
                    "segmentation": a["segmentation"],
                })
                out_ids.append(annotations[-1]["id"])
            queries.append({
                "id": len(queries), "original_cat_id": cat, "object_ids_output": out_ids,
                "query_text": self._names[cat], "query_processing_order": 0, "ptr_x_query_id": None,
                "ptr_y_query_id": None, "image_id": 0, "input_box": None, "input_box_label": None,
                "input_points": None, "is_exhaustive": True,
            })
        return queries, annotations
