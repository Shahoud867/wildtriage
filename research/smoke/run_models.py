"""A-02 smoke test: MegaDetector + BioCLIP 2 on a few LILA images, through the app's adapters.

    python research/smoke/run_models.py --manifest research/manifests/lila_dev.parquet --n 50 --device auto

Prints boxes, top-5 species, latency and peak memory per image, then a summary, and optionally writes a
Markdown report (--report research/results/smoke_cuda.md). Images are streamed from LILA's public bucket
and cached under --cache-dir.
"""

from __future__ import annotations

import argparse
import statistics
import time
import urllib.request
from pathlib import Path

import numpy as np

from wildtriage_ml.adapters.bioclip import BioCLIP2Classifier
from wildtriage_ml.adapters.megadetector import MegaDetectorAdapter
from wildtriage_ml.device import device_kind, pick_device

GCS_BASE_URL = "https://storage.googleapis.com/public-datasets-lila/snapshotserengeti-unzipped"


def fetch(image_id: str, cache_dir: Path) -> Path | None:
    dest = cache_dir / (image_id.replace("/", "__") + ".JPG")
    if dest.exists():
        return dest
    req = urllib.request.Request(f"{GCS_BASE_URL}/{image_id}.JPG", headers={"User-Agent": "WildTriage/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            dest.write_bytes(resp.read())
    except Exception as exc:
        print(f"  download failed for {image_id}: {exc}")
        return None
    return dest


def peak_memory_gb(device: str) -> float | None:
    import torch

    if device_kind(device) == "cuda":
        return torch.cuda.max_memory_allocated() / 1024**3
    return None


def sync(device: str) -> None:
    import torch

    if device_kind(device) == "cuda":
        torch.cuda.synchronize()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="research/manifests/lila_dev.parquet")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-dir", default="research/smoke/cache")
    ap.add_argument("--report", default=None, help="Optional Markdown report path")
    args = ap.parse_args()

    import pandas as pd
    import torch

    device = pick_device(args.device)
    df = pd.read_parquet(args.manifest)
    sample = df.sample(n=min(args.n, len(df)), random_state=args.seed).reset_index(drop=True)
    labels = sorted(df["species"].unique())
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"device: {device} | torch {torch.__version__} | {len(sample)} images | {len(labels)} species")
    if device_kind(device) == "cuda":
        torch.cuda.reset_peak_memory_stats()
    detector, classifier = MegaDetectorAdapter(), BioCLIP2Classifier()
    t0 = time.perf_counter()
    detector.load(device)
    classifier.load(device)
    print(f"models loaded in {time.perf_counter() - t0:.1f}s; peak mem after load: {peak_memory_gb(device)}")

    rows, det_ms, cls_ms, top1_hits, with_box = [], [], [], 0, 0
    for i, rec in sample.iterrows():
        path = fetch(rec["image_id"], cache_dir)
        if path is None:
            continue
        t = time.perf_counter()
        boxes = detector.detect(path)
        sync(device)
        det_ms.append((time.perf_counter() - t) * 1000)

        animals = [b for b in boxes if b.category == "animal"]
        best = max(animals, key=lambda b: b.score) if animals else None
        t = time.perf_counter()
        scores = classifier.classify(path, best, labels)
        sync(device)
        cls_ms.append((time.perf_counter() - t) * 1000)

        top5 = np.argsort(-scores.logits)[:5]
        top5_names = [labels[j] for j in top5]
        with_box += best is not None
        top1_hits += top5_names[0] == rec["species"]
        rows.append((rec["image_id"], rec["species"], len(boxes), best.score if best else 0.0, top5_names))
        print(f"[{i + 1}/{len(sample)}] {rec['image_id']} truth={rec['species']} boxes={len(boxes)} "
              f"best={best.score if best else 0:.2f} top5={top5_names}")

    n = len(rows)
    if not n:
        raise SystemExit("no images could be processed")
    top5_hits = sum(r[1] in r[4] for r in rows)
    summary = [
        f"device: {device} ({torch.cuda.get_device_name(0) if device_kind(device) == 'cuda' else 'n/a'}), "
        f"dtype: {next(classifier._model.parameters()).dtype}",
        f"images processed: {n}/{len(sample)}; with at least one animal box: {with_box}",
        f"BioCLIP 2 zero-shot top-1: {top1_hits / n:.1%}, top-5: {top5_hits / n:.1%} (whole-image or best-box crop)",
        f"MegaDetector latency ms: median {statistics.median(det_ms):.0f}, mean {statistics.mean(det_ms):.0f} "
        f"(first image included, includes warm-up)",
        f"BioCLIP 2 latency ms: median {statistics.median(cls_ms):.0f}, mean {statistics.mean(cls_ms):.0f}",
        f"peak CUDA memory allocated (GB): {peak_memory_gb(device)}",
    ]
    print("\n" + "\n".join(summary))

    if args.report:
        lines = ["# Smoke test: MegaDetector + BioCLIP 2", "", *[f"- {s}" for s in summary], "",
                 "| image | truth | boxes | best score | top-5 |", "|---|---|---|---|---|"]
        lines += [f"| {r[0]} | {r[1]} | {r[2]} | {r[3]:.2f} | {', '.join(r[4])} |" for r in rows]
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"report written to {args.report}")


if __name__ == "__main__":
    main()
