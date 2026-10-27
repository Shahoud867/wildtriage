"""Build the SA-FARI fine-tuning subset (location-disjoint train/val) and download its frames.

    python research/training/prepare.py --n-train 2000 --n-val 150 --out D:/wildtriage-data/sa_fari_ft

Needs the gated facebook/SA-FARI annotations (HF token). Resumable: re-running skips frames already on disk.
Writes <out>/train.json, <out>/val.json, <out>/subset_summary.json and frames under <out>/frames/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.training.data import build_subset, download_frames, location, select_videos, split_locations  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-val", type=int, default=150)
    ap.add_argument("--frame-stride", type=int, default=3, help="keep every k-th 6 fps frame")
    ap.add_argument("--neg-per-frame", type=int, default=2, help="negative phrases added to each frame")
    ap.add_argument("--val-location-fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-download", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download

    gt = json.load(open(hf_hub_download("facebook/SA-FARI", "annotation/sa_fari_train.json", repo_type="dataset"),
                        encoding="utf-8"))
    train_locs, val_locs = split_locations(gt["videos"], args.val_location_fraction, args.seed)
    splits = {"train": select_videos(gt["videos"], train_locs, args.n_train, args.seed),
              "val": select_videos(gt["videos"], val_locs, args.n_val, args.seed)}

    args.out.mkdir(parents=True, exist_ok=True)
    summary = {"args": {k: str(v) for k, v in vars(args).items()}}
    for name, videos in splits.items():
        subset = build_subset(gt, videos, args.frame_stride, args.neg_per_frame, args.seed)
        (args.out / f"{name}.json").write_text(json.dumps(subset), encoding="utf-8")
        summary[name] = {"videos": len(videos), "locations": len({location(v) for v in videos}),
                         "frames": len(subset["images"]), "masks": len(subset["annotations"]),
                         "phrases": len(subset["categories"])}
        print(name, summary[name], flush=True)
        if not args.no_download:
            print(f"  downloaded {download_frames(subset['images'], args.out / 'frames', args.workers)} new frames")
    (args.out / "subset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
