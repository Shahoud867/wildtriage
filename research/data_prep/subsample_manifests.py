"""S-02: stratified subsample of the LILA site manifests (ROADMAP P1: 544k images is too many to stream).

Per site: keep the first `max_intervals` chronological intervals that have at least `min_images` images, and
from each take at most `per_interval` images, split evenly between day and night where both exist (the
remainder of a short stratum is filled from the other). Deterministic for a given seed.

    python research/data_prep/subsample_manifests.py

Limitation: the manifests were de-duplicated by image_id keeping one label, so multi-species images can no
longer be identified here; filtering them needs the raw annotations (see select_sites_lila.py).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def sample_interval(df: pd.DataFrame, per_interval: int, rng: np.random.Generator) -> pd.DataFrame:
    night, day = df[df["is_night"]], df[~df["is_night"]]
    half = per_interval // 2
    n_night = min(len(night), max(half, per_interval - len(day)))
    n_day = min(len(day), per_interval - n_night)
    parts = [g.iloc[np.sort(rng.choice(len(g), n, replace=False))] for g, n in ((night, n_night), (day, n_day)) if n]
    return pd.concat(parts) if parts else df.iloc[:0]


def subsample(df: pd.DataFrame, max_intervals: int = 18, min_images: int = 50, per_interval: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = []
    for _, site_df in df.groupby("site", sort=True):
        counts = site_df.groupby("interval").size()
        keep = counts[counts >= min_images].index.sort_values()[:max_intervals]
        for interval in keep:
            out.append(sample_interval(site_df[site_df["interval"] == interval], per_interval, rng))
    return pd.concat(out).sort_values(["site", "timestamp", "image_id"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest-dir", default="research/manifests")
    ap.add_argument("--max-intervals", type=int, default=18)
    ap.add_argument("--min-images", type=int, default=50)
    ap.add_argument("--per-interval", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.manifest_dir)
    summaries = []
    for split in ("dev", "test"):
        sub = subsample(pd.read_parquet(root / f"lila_{split}.parquet"), args.max_intervals, args.min_images, args.per_interval, args.seed)
        sub.to_parquet(root / f"lila_{split}_sub.parquet", index=False)
        summary = sub.groupby("site").agg(images=("image_id", "size"), species=("species", "nunique"),
                                          intervals=("interval", "nunique"), night_frac=("is_night", "mean")).round(3)
        summary.insert(0, "split", split)
        summaries.append(summary.reset_index())
        print(f"{split}: {len(sub)} images, {sub['site'].nunique()} sites, min intervals/site {summary['intervals'].min()}")
    pd.concat(summaries).to_csv(root / "lila_sub_summary.csv", index=False)


if __name__ == "__main__":
    main()
