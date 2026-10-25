"""Stream-parse Snapshot Serengeti COCO JSON, select camera sites, and create manifests.

Follows research/data_prep/streamtrap_mini.py protocol:
- Filter criteria: span >= 6 months, >= 1,000 animal images, >= 3 species.
- Disjoint split: 10 dev sites, 20 test sites (seed=0).
- Calculates chronological monthly intervals and day/night flags.
- Saves research/manifests/lila_dev.parquet and research/manifests/lila_test.parquet.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("select_sites_lila")

DATA_ROOT = Path("data/lila/snapshot_serengeti")
JSON_PATH = DATA_ROOT / "SnapshotSerengeti_S1-11_v2.1.json"
MANIFESTS_DIR = Path("research/manifests")


def parse_categories(f) -> dict[int, str]:
    """Extract category_id -> name from open file stream."""
    categories: dict[int, str] = {}
    for line in f:
        if '"categories":' in line:
            break

    for line in f:
        line_s = line.strip()
        if line_s.startswith("]"):
            break
        if "{" in line_s:
            block = [line_s]
            while not block[-1].endswith("}") and not block[-1].endswith("},"):
                block.append(next(f).strip())
            block_str = " ".join(block).rstrip(",")
            cat = json.loads(block_str)
            categories[cat["id"]] = cat["name"]
    return categories


def stream_annotations(f, categories: dict[int, str]):
    """Yield wildlife annotation records one by one without loading entire JSON into RAM."""
    for line in f:
        if '"annotations":' in line:
            break

    for line in f:
        line_s = line.strip()
        if line_s.startswith("]"):
            break
        if "{" in line_s:
            block = [line_s]
            while not block[-1].endswith("}") and not block[-1].endswith("},"):
                block.append(next(f).strip())
            block_str = " ".join(block).rstrip(",")
            try:
                ann = json.loads(block_str)
            except Exception:
                continue

            cat_id = ann.get("category_id")
            # 0 is empty, 1 is human; > 1 is wildlife species
            if cat_id is not None and cat_id > 1:
                loc = ann.get("location")
                dt = ann.get("datetime")
                img_id = ann.get("image_id")
                if loc and dt and img_id:
                    yield {
                        "site": loc,
                        "image_id": img_id,
                        "timestamp": dt,
                        "category_id": cat_id,
                        "species": categories.get(cat_id, str(cat_id)),
                        "season": ann.get("season", ""),
                    }


def select_qualifying_sites(
    records: list[dict],
    min_months: int = 6,
    min_images: int = 1000,
    min_species: int = 3,
) -> tuple[pd.DataFrame, list[str]]:
    """Convert parsed records to DataFrame and apply StreamTrap-mini site filter."""
    logger.info("Building DataFrame from %d wildlife records...", len(records))
    df = pd.DataFrame(records)
    # Deduplicate image_id per site if multiple annotations exist for one image
    df = df.drop_duplicates(subset=["site", "image_id"]).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()

    logger.info("Computing site-level statistics across %d unique sites...", df["site"].nunique())
    g = df.groupby("site")
    stats = pd.DataFrame({
        "num_images": g.size(),
        "num_species": g["species"].nunique(),
        "min_time": g["timestamp"].min(),
        "max_time": g["timestamp"].max(),
    })
    stats["span_months"] = (stats["max_time"] - stats["min_time"]).dt.days / 30.44

    qualifying = stats[
        (stats["span_months"] >= min_months)
        & (stats["num_images"] >= min_images)
        & (stats["num_species"] >= min_species)
    ].sort_values(by="num_images", ascending=False)

    logger.info("Found %d qualifying sites matching filter criteria.", len(qualifying))
    return df, qualifying.index.tolist()


def make_intervals(df: pd.DataFrame) -> pd.Series:
    """0-based chronological calendar-month index per site (months with no images are skipped)."""
    month = df["timestamp"].dt.year * 12 + df["timestamp"].dt.month
    return (month.groupby(df["site"]).rank(method="dense") - 1).astype(int)


def main():
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    if not JSON_PATH.exists():
        raise FileNotFoundError(f"Missing {JSON_PATH}. Run download first.")

    logger.info("Streaming categories from %s...", JSON_PATH)
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        categories = parse_categories(f)
    logger.info("Loaded %d categories (%d wildlife species).", len(categories), len(categories) - 2)

    logger.info("Streaming annotations from 5.5GB JSON...")
    records = []
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        for idx, rec in enumerate(stream_annotations(f, categories)):
            records.append(rec)
            if (idx + 1) % 500000 == 0:
                logger.info("Processed %d wildlife records...", idx + 1)

    logger.info("Total wildlife records collected: %d", len(records))

    df, qualifying_sites = select_qualifying_sites(records, min_months=6, min_images=1000, min_species=3)
    logger.info("Top qualifying sites count: %d", len(qualifying_sites))

    if len(qualifying_sites) < 30:
        raise ValueError(f"Need at least 30 sites for 10 dev / 20 test, but found {len(qualifying_sites)}")

    # Pick top 30 sites by data density, then randomly split into 10 dev and 20 test (seed=0)
    top_30_sites = qualifying_sites[:30]
    dev_sites = pd.Series(top_30_sites).sample(n=10, random_state=0).tolist()
    test_sites = [s for s in top_30_sites if s not in dev_sites]

    site_split = {s: "dev" for s in dev_sites}
    site_split.update({s: "test" for s in test_sites})

    # Filter dataframe to the 30 sites
    df_manifest = df[df["site"].isin(top_30_sites)].copy()
    df_manifest["split"] = df_manifest["site"].map(site_split)
    df_manifest["interval"] = make_intervals(df_manifest)
    # Day / night split based on local solar hour: night is 19:00 - 06:00
    df_manifest["hour"] = df_manifest["timestamp"].dt.hour
    df_manifest["is_night"] = (df_manifest["hour"] < 6) | (df_manifest["hour"] >= 19)
    df_manifest = df_manifest.sort_values(by=["site", "timestamp"]).reset_index(drop=True)

    df_dev = df_manifest[df_manifest["split"] == "dev"].reset_index(drop=True)
    df_test = df_manifest[df_manifest["split"] == "test"].reset_index(drop=True)

    dev_path = MANIFESTS_DIR / "lila_dev.parquet"
    test_path = MANIFESTS_DIR / "lila_test.parquet"
    summary_path = MANIFESTS_DIR / "lila_sites_summary.csv"

    df_dev.to_parquet(dev_path, index=False)
    df_test.to_parquet(test_path, index=False)

    # Save summary table
    summary = (
        df_manifest.groupby(["split", "site"])
        .agg(
            images=("image_id", "count"),
            species=("species", "nunique"),
            start_date=("timestamp", "min"),
            end_date=("timestamp", "max"),
            intervals=("interval", "nunique"),
        )
        .reset_index()
    )
    summary.to_csv(summary_path, index=False)

    logger.info("Saved %s (%d crops, %d sites)", dev_path, len(df_dev), len(dev_sites))
    logger.info("Saved %s (%d crops, %d sites)", test_path, len(df_test), len(test_sites))
    logger.info("Summary saved to %s", summary_path)
    print("\n=== SITE SELECTION COMPLETE ===")
    print(f"Dev split:  {len(dev_sites)} sites, {len(df_dev):,} image crops")
    print(f"Test split: {len(test_sites)} sites, {len(df_test):,} image crops")
    print(f"Dev Parquet:  {dev_path}")
    print(f"Test Parquet: {test_path}")


if __name__ == "__main__":
    main()
