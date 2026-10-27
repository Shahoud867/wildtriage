"""Exp 3 (P1, lab PC only): DepthSAM (Paper 2) reproduction on COD10K-test.

    python research/experiments/exp3_camouflage.py reproduce \
        --checkpoint D:/wildtriage-data/depthsam/Model_COD_gen.pth --out research/results/exp3_cod10k.json

Setup (ROADMAP A-08):
  1. Code: https://github.com/hjy0518/DepthSAM unpacked to third_party/DepthSAM (gitignored; record the commit
     in third_party/DepthSAM.lock). Weights: the Google Drive link in its README (CC-BY-NC-4.0 DA-V2-Large inside).
  2. Data: the 2026 COD10K test images + masks, here from the HF dataset `nobg/COD10K` (a repackaging of the
     SINet TestDataset; the CAMO and CHAMELEON portions are excluded). Downloaded to --data-dir.
  3. Preprocessing replicates DepthSAM's test.py: resize to 512x512, ImageNet normalisation, sigmoid output
     min-max normalised, bilinear upsampling to the GT size. The depth maps in test_dataset are not used by the
     forward pass, so they are not needed.

Metrics (py-sod-metrics): S-alpha, weighted F-beta, mean E-phi, MAE, overall and per COD10K super-class when the
file name carries it (COD10K-CAM-<n>-<Aquatic|Terrestrial|Flying|Amphibian|Other>-...). Compare with the paper's
Table 1 and state the difference; do not tune toward it.

Refinement on SA-FARI crops (A-11) is a later subcommand once A-10 has SAM 3 masks.

Safety: the checkpoint is loaded with weights_only=True, so it cannot execute code on load.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SUPERCLASS = re.compile(r"CAM-\d+-([A-Za-z]+)")


def load_cod10k_test(data_dir: Path):
    from huggingface_hub import snapshot_download
    import pyarrow.parquet as pq
    from PIL import Image

    root = snapshot_download("nobg/COD10K", repo_type="dataset", allow_patterns=["data/test-*"], local_dir=data_dir)
    for f in sorted(Path(root, "data").glob("test-*.parquet")):
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=16):
            for row in batch.to_pylist():
                name = row["image"].get("path") or ""
                yield (name, Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB"),
                       Image.open(io.BytesIO(row["mask"]["bytes"])).convert("L"))


def build_model(checkpoint: Path, device):
    import torch

    import functools
    import os

    sys.path.insert(0, str(REPO / "third_party" / "DepthSAM"))
    from segment_anything_training.build_DepthSAM import build_sam_DepthSAM

    # DepthSAM's constructor reads the relative path 'checkpoints/depth_anything_v2_vitl.pth' (Depth-Anything-V2-Large,
    # from HF depth-anything/Depth-Anything-V2-Large) with a bare torch.load. Run it from the checkpoint's folder
    # and force weights_only=True for that call.
    real_load, cwd = torch.load, os.getcwd()
    torch.load = functools.partial(real_load, weights_only=True, map_location="cpu")
    os.chdir(Path(checkpoint).parent)
    try:
        model = build_sam_DepthSAM()
    finally:
        os.chdir(cwd)
        torch.load = real_load
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = {k.removeprefix("module."): v for k, v in state.items()}
    print(model.load_state_dict(state))
    return model.to(device).eval()


def cmd_reproduce(args) -> None:
    import py_sod_metrics as sod
    import torch
    import torch.nn.functional as F
    import torchvision.transforms as T

    device = torch.device("cuda")
    model = build_model(Path(args.checkpoint), device)
    transform = T.Compose([T.Resize((args.size, args.size)), T.ToTensor(),
                           T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def new_metrics():
        return {"sm": sod.Smeasure(), "wfm": sod.WeightedFmeasure(), "em": sod.Emeasure(), "mae": sod.MAE()}

    groups: dict[str, dict] = {"all": new_metrics()}
    counts: dict[str, int] = {"all": 0}
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with torch.no_grad():
        for i, (name, img, gt_img) in enumerate(load_cod10k_test(Path(args.data_dir))):
            if args.limit and i >= args.limit:
                break
            x = transform(img).unsqueeze(0).to(device)
            # DepthSAM's test.py also builds a batched_input from the normalised tensor (only sizes are used).
            batched = [{"image": x[0].to(torch.uint8), "original_size": tuple(x.shape[-2:])}]
            res = model(batched, x)
            gt = np.asarray(gt_img, dtype=np.uint8)
            res = F.interpolate(res, size=gt.shape, mode="bilinear", align_corners=False).sigmoid().squeeze().cpu().numpy()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            pred, gt = (res * 255).astype(np.uint8), ((gt > 127) * 255).astype(np.uint8)
            m = SUPERCLASS.search(name)
            for key in ["all"] + ([m.group(1)] if m else []):
                if key not in groups:
                    groups[key], counts[key] = new_metrics(), 0
                counts[key] += 1
                for metric in groups[key].values():
                    metric.step(pred=pred, gt=gt)
            if i % 200 == 0:
                print(f"{i} images, {time.perf_counter() - t0:.0f}s", flush=True)

    results = {}
    for key, metrics in groups.items():
        em = metrics["em"].get_results()["em"]
        results[key] = {"images": counts[key], "S_alpha": metrics["sm"].get_results()["sm"],
                        "F_beta_w": metrics["wfm"].get_results()["wfm"], "E_phi_mean": float(np.mean(em["curve"])),
                        "MAE": metrics["mae"].get_results()["mae"]}
    out = {"dataset": "COD10K-test (nobg/COD10K, SINet TestDataset)", "checkpoint": str(args.checkpoint),
           "seconds": time.perf_counter() - t0, "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
           "results": results}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("reproduce")
    r.add_argument("--checkpoint", default="D:/wildtriage-data/depthsam/Model_COD_gen.pth")
    r.add_argument("--data-dir", default="D:/wildtriage-data/depthsam/cod10k_nobg")
    r.add_argument("--out", default="research/results/exp3_cod10k.json")
    r.add_argument("--size", type=int, default=512)
    r.add_argument("--limit", type=int, default=0, help="Only the first N images (smoke test)")
    r.set_defaults(fn=cmd_reproduce)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
