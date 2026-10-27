"""A-06 smoke test: SAM 3 on a few SA-FARI *test* clips (needs gated HF access and a CUDA GPU).

    python research/smoke/run_sam3.py --n 3 --prompt animal --report research/results/smoke_sam3.md

Downloads the 6 fps JPEG frames from the public cxl-public-camera-trap bucket, runs Sam3VideoTracker, and
compares the predicted MaxN with the ground-truth MaxN from sa_fari_test.json. Test split only.
"""

from __future__ import annotations

import argparse
import json
import random
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from wildtriage_ml.adapters.sam3 import Sam3VideoTracker
from wildtriage_ml.device import pick_device
from wildtriage_ml.pipeline.counting import max_n

FRAMES_URL = "https://storage.googleapis.com/cxl-public-camera-trap/sa_fari/sa_fari_test/JPEGImages_6fps"


def gt_max_n(annotations: list[dict], video_id: int) -> int:
    """Most masklets present in any single frame (a box is None where the object is absent)."""
    per_frame: dict[int, int] = {}
    for a in annotations:
        if a["video_id"] != video_id:
            continue
        for i, box in enumerate(a["bboxes"]):
            if box:
                per_frame[i] = per_frame.get(i, 0) + 1
    return max(per_frame.values(), default=0)


def download_clip(video: dict, out_dir: Path) -> Path:
    clip_dir = out_dir / video["video_name"]
    clip_dir.mkdir(parents=True, exist_ok=True)

    def get(name: str) -> None:
        dest = clip_dir / Path(name).name
        if not dest.exists():
            urllib.request.urlretrieve(f"{FRAMES_URL}/{name}", dest)

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(get, video["file_names"]))
    return clip_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--prompt", default="animal")
    ap.add_argument("--checkpoint", default="facebook/sam3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-dir", default="research/smoke/sa_fari_cache")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    import torch
    from huggingface_hub import hf_hub_download

    device = pick_device("cuda")
    ann = json.load(open(hf_hub_download("facebook/SA-FARI", "annotation/sa_fari_test.json", repo_type="dataset"), encoding="utf-8"))
    positive = sorted({a["video_id"] for a in ann["annotations"]})
    videos = {v["id"]: v for v in ann["videos"]}
    picked = random.Random(args.seed).sample(positive, args.n)

    tracker = Sam3VideoTracker(checkpoint=args.checkpoint)
    t0 = time.perf_counter()
    tracker.load(device)
    print(f"SAM 3 loaded in {time.perf_counter() - t0:.1f}s on {torch.cuda.get_device_name(0)}")

    rows = []
    for vid in picked:
        video = videos[vid]
        clip_dir = download_clip(video, Path(args.cache_dir))
        t = time.perf_counter()
        tracks = tracker.track(clip_dir, [args.prompt])
        secs = time.perf_counter() - t
        row = (video["video_name"], video["length"], len(tracks), max_n(tracks), gt_max_n(ann["annotations"], vid),
               secs, tracker.peak_vram_gb)
        rows.append(row)
        print("{}: {} frames, {} masklets, MaxN pred={} gt={}, {:.1f}s, peak VRAM {:.1f} GB".format(*row))

    exact = sum(r[3] == r[4] for r in rows)
    summary = f"{len(rows)} clips, prompt {args.prompt!r}: MaxN exact on {exact}/{len(rows)}; mean abs error " \
              f"{sum(abs(r[3] - r[4]) for r in rows) / len(rows):.2f}; max peak VRAM {max(r[6] for r in rows):.1f} GB"
    print(summary)
    if args.report:
        lines = ["# Smoke test: SAM 3 on SA-FARI test clips", "", summary, "",
                 "| clip | frames | masklets | MaxN pred | MaxN gt | seconds | peak VRAM GB |", "|---|---|---|---|---|---|---|"]
        lines += [f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]:.1f} | {r[6]:.1f} |" for r in rows]
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    tracker.unload()


if __name__ == "__main__":
    main()
