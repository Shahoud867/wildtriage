"""Exp 2 (lab PC only): SAM 3 on a random SA-FARI *test* subset, scored with the official SA-Co/VEval evaluator,
plus per-clip counting error (MaxN).

    python research/experiments/exp2_video_counts.py run  --n-clips 150 --out research/results/exp2
    python research/experiments/exp2_video_counts.py eval --out research/results/exp2

`run` is resumable: each clip's predictions are cached in <out>/clips/<video_id>.json, and finished clips are
skipped. For every video it prompts SAM 3 with each of the video's (positive and negative) noun phrases, which
is the VEval protocol (Table 3: cgF1, pHOTA), and with the class-agnostic prompt "animal" (used for MaxN).
`eval` assembles the prediction file, runs the offline evaluator from third_party/sam3 on the subset's
ground truth, and writes <out>/exp2_summary.json (VEval metrics + MaxN MAE / RMSE / exact-match overall and for
the multi-animal subset), with the git commit if available.

Test split only: never evaluate SAM 3 on SA-FARI train. Differences from the paper (subset, checkpoint, fps) go
in the report; do not tune toward the paper's numbers.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

FRAMES_URL = "https://storage.googleapis.com/cxl-public-camera-trap/sa_fari/sa_fari_test/JPEGImages_6fps"
ANIMAL = "animal"


def load_gt() -> dict:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("facebook/SA-FARI", "annotation/sa_fari_test.json", repo_type="dataset")
    return json.load(open(path, encoding="utf-8"))


def pick_videos(gt: dict, n: int, seed: int) -> list[int]:
    ids = sorted(v["id"] for v in gt["videos"])
    return sorted(random.Random(seed).sample(ids, min(n, len(ids))))


def subset_gt(gt: dict, video_ids: list[int]) -> dict:
    keep = set(video_ids)
    return {
        "info": gt["info"],
        "videos": [v for v in gt["videos"] if v["id"] in keep],
        "annotations": [a for a in gt["annotations"] if a["video_id"] in keep],
        "categories": gt["categories"],
        "video_np_pairs": [p for p in gt["video_np_pairs"] if p["video_id"] in keep],
    }


def gt_max_n(annotations: list[dict], video_id: int) -> int:
    per_frame: dict[int, int] = {}
    for a in annotations:
        if a["video_id"] == video_id:
            for i, box in enumerate(a["bboxes"]):
                if box:
                    per_frame[i] = per_frame.get(i, 0) + 1
    return max(per_frame.values(), default=0)


def download_clip(video: dict, cache_dir: Path) -> Path:
    clip_dir = cache_dir / video["video_name"]
    clip_dir.mkdir(parents=True, exist_ok=True)

    def get(name: str) -> None:
        dest = clip_dir / Path(name).name
        if not dest.exists():
            urllib.request.urlretrieve(f"{FRAMES_URL}/{name}", dest.with_suffix(".part"))
            dest.with_suffix(".part").replace(dest)

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(get, video["file_names"]))
    return clip_dir


def tracks_to_preds(tracks, video: dict, category_id: int) -> list[dict]:
    """TrackResults of one prompt -> YTVIS-style prediction dicts (absolute boxes, one entry per frame)."""
    from pycocotools import mask as mask_utils

    preds = []
    for t in tracks:
        rle = t.extra.get("rle", {})
        segs, boxes, areas = [], [], []
        for f in range(video["length"]):
            r = rle.get(f)
            if r is None:
                segs.append(None), boxes.append(None), areas.append(None)
                continue
            seg = {"size": r["size"], "counts": r["counts"].encode("ascii")}
            segs.append(r)
            boxes.append([float(v) for v in mask_utils.toBbox(seg)])
            areas.append(float(mask_utils.area(seg)))
        preds.append({"video_id": video["id"], "category_id": category_id, "score": float(t.score),
                      "segmentations": segs, "bboxes": boxes, "areas": areas})
    return preds


def cmd_run(args) -> None:
    import torch

    from wildtriage_ml.adapters.sam3 import Sam3VideoTracker
    from wildtriage_ml.pipeline.counting import max_n

    out = Path(args.out)
    (out / "clips").mkdir(parents=True, exist_ok=True)
    gt = load_gt()
    ids = pick_videos(gt, args.n_clips, args.seed)
    json.dump({"seed": args.seed, "video_ids": ids}, open(out / "subset.json", "w"))
    videos = {v["id"]: v for v in gt["videos"]}
    pairs: dict[int, list[dict]] = {}
    for p in gt["video_np_pairs"]:
        pairs.setdefault(p["video_id"], []).append(p)

    tracker = Sam3VideoTracker(checkpoint=args.checkpoint, keep_masks=True)
    tracker.load("cuda")
    for n, vid in enumerate(ids, 1):
        cache = out / "clips" / f"{vid}.json"
        if cache.exists():
            continue
        video = videos[vid]
        clip_dir = download_clip(video, Path(args.frames_dir))
        prompts = [ANIMAL] + sorted({p["noun_phrase"] for p in pairs.get(vid, [])} - {ANIMAL})
        t0 = time.perf_counter()
        tracks = tracker.track(clip_dir, prompts)
        secs = time.perf_counter() - t0
        np_preds = []
        for p in pairs.get(vid, []):
            np_preds += tracks_to_preds([t for t in tracks if t.label == p["noun_phrase"]], video, p["category_id"])
        animal = [t for t in tracks if t.label == ANIMAL]
        record = {"video_id": vid, "seconds": secs, "peak_vram_gb": tracker.peak_vram_gb, "prompts": prompts,
                  "pred_max_n_animal": max_n(animal), "gt_max_n": gt_max_n(gt["annotations"], vid),
                  "n_animal_tracks": len(animal), "preds": np_preds}
        cache.write_text(json.dumps(record), encoding="utf-8")
        print(f"[{n}/{len(ids)}] {video['video_name']} prompts={len(prompts)} {secs:.1f}s "
              f"MaxN pred={record['pred_max_n_animal']} gt={record['gt_max_n']} peakVRAM={tracker.peak_vram_gb:.1f}GB", flush=True)
        del tracks, animal, np_preds, record
        gc.collect()
        torch.cuda.empty_cache()
    tracker.unload()


def rmse(xs):
    return math.sqrt(sum(x * x for x in xs) / len(xs)) if xs else float("nan")


def phrase_max_n(record: dict, positive_categories: set[int]) -> int:
    """MaxN from the noun-phrase predictions of the video's *positive* phrases (the phrase is given, as in the
    VEval protocol): the most simultaneously visible predicted masklets summed over those phrases."""
    per_frame: dict[int, int] = {}
    for p in record["preds"]:
        if p["category_id"] in positive_categories:
            for f, seg in enumerate(p["segmentations"]):
                if seg is not None:
                    per_frame[f] = per_frame.get(f, 0) + 1
    return max(per_frame.values(), default=0)


def count_summary(records: list[dict], key: str = "pred_max_n_animal") -> dict:
    errs = [r[key] - r["gt_max_n"] for r in records]
    return {"clips": len(records), "mae": sum(map(abs, errs)) / len(errs) if errs else float("nan"),
            "rmse": rmse(errs), "exact_match": sum(e == 0 for e in errs) / len(errs) if errs else float("nan"),
            "mean_signed_error": sum(errs) / len(errs) if errs else float("nan")}


def cmd_eval(args) -> None:
    out = Path(args.out)
    gt = load_gt()
    ids = json.load(open(out / "subset.json"))["video_ids"]
    records = [json.load(open(p, encoding="utf-8")) for p in sorted((out / "clips").glob("*.json"))]
    done = {r["video_id"] for r in records}
    done_ids = [i for i in ids if i in done]
    print(f"{len(done_ids)}/{len(ids)} clips finished")
    positive: dict[int, set[int]] = {}
    for p in gt["video_np_pairs"]:
        if p["num_masklets"] > 0:
            positive.setdefault(p["video_id"], set()).add(p["category_id"])
    for r in records:
        r["pred_max_n_phrase"] = phrase_max_n(r, positive.get(r["video_id"], set()))

    gt_sub = subset_gt(gt, done_ids)
    gt_file, pred_file, res_file = out / "gt_subset.json", out / "preds.json", out / "veval_res.json"
    json.dump(gt_sub, open(gt_file, "w"))
    json.dump([p for r in records for p in r["preds"]], open(pred_file, "w"))

    summary = {"clips_evaluated": len(done_ids), "git_commit": _git_commit(),
               "seconds_per_clip_mean": sum(r["seconds"] for r in records) / max(1, len(records)),
               "peak_vram_gb_max": max((r["peak_vram_gb"] or 0) for r in records) if records else None,
               # "animal" is a class-agnostic prompt; SAM 3 often returns nothing for it on camera-trap clips.
               "maxn_animal_prompt": count_summary(records),
               "maxn_animal_prompt_multi_animal_gt": count_summary([r for r in records if r["gt_max_n"] >= 2]),
               # The positive phrases are given (VEval protocol), so this is an oracle-prompt upper bound.
               "maxn_given_phrase": count_summary(records, "pred_max_n_phrase"),
               "maxn_given_phrase_multi_animal_gt": count_summary([r for r in records if r["gt_max_n"] >= 2], "pred_max_n_phrase")}
    try:
        from sam3.eval.saco_veval_eval import VEvalEvaluator

        res = VEvalEvaluator(gt_annot_file=str(gt_file), eval_res_file=str(res_file)).run_eval(pred_file=str(pred_file))
        summary["veval"] = {k: res["dataset_results"][k] for k in res["dataset_results"]
                            if k in ("video_mask_demo_cgf1_micro_50_95", "video_mask_all_phrase_HOTA")}
        summary["veval_all"] = res["dataset_results"]
    except Exception as exc:  # keep the counting results even if the evaluator environment is incomplete
        summary["veval_error"] = repr(exc)
        print("VEval failed:", repr(exc))
    (out / "exp2_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "veval_all"}, indent=2, default=float))


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--n-clips", type=int, default=150)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--checkpoint", default="facebook/sam3")
    r.add_argument("--out", default="research/results/exp2")
    r.add_argument("--frames-dir", default="D:/wildtriage-data/sa_fari_frames")
    r.set_defaults(fn=cmd_run)
    e = sub.add_parser("eval")
    e.add_argument("--out", default="research/results/exp2")
    e.set_defaults(fn=cmd_eval)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
