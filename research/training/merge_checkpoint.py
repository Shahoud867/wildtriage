"""Write fine-tuned detector weights into the full SAM 3 video checkpoint so Exp 2 can evaluate them.

    python research/training/merge_checkpoint.py --finetuned D:/wildtriage-data/runs/stage_a/checkpoints/checkpoint.pt \
        --out D:/wildtriage-data/runs/stage_a/sam3_video_stage_a.pt

The trainer saves the image model (`backbone.*`, `transformer.*`, ...) minus the frozen backbones; the video
checkpoint stores the same tensors under `detector.*` next to the untouched `tracker.*`. Every fine-tuned key
must already exist in the base checkpoint with the same shape, otherwise this stops instead of guessing.
Then: python research/experiments/exp2_video_counts.py run --checkpoint <out> ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def merge(base: dict, finetuned: dict) -> tuple[dict, int]:
    merged, replaced = dict(base), 0
    for key, tensor in finetuned.items():
        target = f"detector.{key}"
        if target not in base:
            raise KeyError(f"{target} is not in the base checkpoint")
        if base[target].shape != tensor.shape:
            raise ValueError(f"{target}: shape {tuple(tensor.shape)} != base {tuple(base[target].shape)}")
        merged[target] = tensor.to(base[target].dtype)
        replaced += 1
    return merged, replaced


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--finetuned", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base", type=Path, default=None, help="default: the facebook/sam3 download")
    args = ap.parse_args()

    from sam3.model_builder import download_ckpt_from_hf

    base_path = args.base or Path(download_ckpt_from_hf(version="sam3"))
    base = torch.load(base_path, map_location="cpu", weights_only=True)
    base = base["model"] if "model" in base and isinstance(base["model"], dict) else base
    tuned = torch.load(args.finetuned, map_location="cpu", weights_only=True)["model"]
    merged, replaced = merge(base, tuned)
    changed = sum(1 for k in tuned if not torch.equal(base[f"detector.{k}"], merged[f"detector.{k}"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, args.out)
    print(f"base {base_path.name}: {len(base)} tensors; replaced {replaced} detector tensors "
          f"({changed} differ from base); wrote {args.out}")


if __name__ == "__main__":
    main()
