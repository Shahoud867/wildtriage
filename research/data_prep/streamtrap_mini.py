"""Build "StreamTrap-mini" from LILA camera-trap metadata (Paper 3's protocol, rebuilt at small scale).

Steps (W1-W2):
  1. Download each chosen dataset's metadata (COCO Camera Traps JSON) and LILA's precomputed
     MegaDetector results (https://lila.science/megadetector-results-for-camera-trap-datasets/).
  2. Keep single-species images with an animal box >= md_min_conf; crop with 50% box expansion.
  3. select_sites(): sites with >= 6 months and >= 1,000 animal images (StreamTrap's filter), >= 3 species.
  4. make_intervals(): chronological monthly intervals per site.
  5. Split sites into dev/test (site-disjoint, stratified by source dataset); write a Parquet manifest
     with sha256 per crop. Embedding extraction then runs on the lab PC or T1000 (extract_embeddings.py, W2).

Record each dataset's license in the manifest and in docs/dataset_cards/.
"""

import pandas as pd


def select_sites(df: pd.DataFrame, min_months: int = 6, min_images: int = 1000, min_species: int = 3) -> list[str]:
    """df columns: site, timestamp (datetime64), species. One row per animal crop."""
    g = df.groupby("site")
    span_months = (g["timestamp"].max() - g["timestamp"].min()).dt.days / 30.44
    keep = (span_months >= min_months) & (g.size() >= min_images) & (g["species"].nunique() >= min_species)
    return sorted(keep[keep].index)


def make_intervals(df: pd.DataFrame) -> pd.Series:
    """0-based chronological calendar-month index per site (months with no images are skipped)."""
    month = df["timestamp"].dt.year * 12 + df["timestamp"].dt.month
    return (month.groupby(df["site"]).rank(method="dense") - 1).astype(int)


def split_sites(sites: list[str], n_dev: int, seed: int = 0) -> dict[str, str]:
    """Site-disjoint dev/test split. Stratify by source dataset when calling this per dataset."""
    shuffled = pd.Series(sorted(sites)).sample(frac=1.0, random_state=seed).tolist()
    return {s: ("dev" if i < n_dev else "test") for i, s in enumerate(shuffled)}
