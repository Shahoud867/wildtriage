"""Rescale a checkpoint's epoch counter after shrinking the epoch window, so a resume continues where it stopped.

    python research/training/rechunk_checkpoint.py --checkpoint <dir>/checkpoint.pt --factor 4

Going from epoch_frames=4000 to 1000 (factor 4) and max_epochs 6 -> 24 keeps everything the trainer derives from the
epoch counter identical: the data window (epoch * epoch_frames), the progress fraction epoch / max_epochs, and the
LR-scheduler step (epoch * iterations per epoch). The original is kept next to it as <name>.before_rechunk.pt.
`steps` (a logging counter) is left alone.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--factor", type=int, required=True, help="old epoch_frames / new epoch_frames")
    args = ap.parse_args()

    backup = args.checkpoint.with_suffix(".before_rechunk.pt")
    if backup.exists():
        raise SystemExit(f"{backup} exists: this checkpoint was already re-chunked")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    old = ckpt["epoch"]
    shutil.copy2(args.checkpoint, backup)
    ckpt["epoch"] = old * args.factor
    torch.save(ckpt, args.checkpoint)
    print(f"{args.checkpoint.name}: epoch {old} -> {ckpt['epoch']} (backup {backup.name})")


if __name__ == "__main__":
    main()
