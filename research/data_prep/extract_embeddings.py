"""Extract and cache BioCLIP 2 embeddings for camera trap sites from Parquet manifests.

Produces per-site .npz files conforming to the SiteStream contract expected by
research/experiments/exp1_triage.py:
  - embeddings: (n, d) float32 L2-normalised image embeddings
  - labels: (n,) int64 class indices
  - interval: (n,) int64 chronological monthly interval indices
  - is_night: (n,) bool day/night indicator
  - text_embeddings: (k, d) float32 zero-shot text prompt embeddings for all k classes
  - logit_scale: scalar float (from BioCLIP 2)
  - classes: (k,) string array of species names
  - image_ids: (n,) string array of image identifiers

Supports:
  - Resumable site processing (--skip-existing)
  - Local image directory or direct anonymous GCS streaming
  - Device selection via wildtriage_ml.device.resolve_device()
  - Smoke testing mode (--limit) for quick validation
"""

from __future__ import annotations

import argparse
import io
import logging
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from wildtriage_ml.device import pick_device, pick_dtype

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("extract_embeddings")

GCS_BASE_URL = "https://storage.googleapis.com/public-datasets-lila/snapshotserengeti-unzipped"
DEFAULT_MODEL = "hf-hub:imageomics/bioclip-2"


def fetch_image(
    image_id: str,
    images_dir: Path | None = None,
    base_url: str = GCS_BASE_URL,
    timeout: int = 15,
    mock: bool = False,
) -> Image.Image | None:
    """Fetch an image from local disk, over HTTP, or create a mock image."""
    if mock:
        return Image.new("RGB", (224, 224), color=(128, 128, 128))

    if images_dir is not None:
        for ext in [".JPG", ".jpg", ".JPEG", ".jpeg"]:
            cand = images_dir / f"{image_id}{ext}"
            if cand.exists():
                try:
                    return Image.open(cand).convert("RGB")
                except Exception as e:
                    logger.warning("Corrupt local image %s: %s", cand, e)
                    return None

    # Fallback to HTTP download
    url = f"{base_url}/{image_id}.JPG"
    req = urllib.request.Request(url, headers={"User-Agent": "WildTriage/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        logger.warning("Failed to download %s: %s", url, e)
        return None


class MockExtractor:
    """Deterministic mock extractor for offline testing and smoke verification."""

    def __init__(self, dim: int = 512, seed: int = 42):
        self.dim = dim
        self.logit_scale = 30.0
        self.rng = np.random.default_rng(seed)

    def encode_text(self, labels: list[str]) -> np.ndarray:
        feats = self.rng.normal(size=(len(labels), self.dim)).astype(np.float32)
        return feats / np.linalg.norm(feats, axis=-1, keepdims=True)

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        feats = self.rng.normal(size=(len(images), self.dim)).astype(np.float32)
        return feats / np.linalg.norm(feats, axis=-1, keepdims=True)


class BioCLIP2Extractor:
    """Wraps OpenCLIP model for batch image and text embedding extraction."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "auto",
        prompt_template: str = "a photo of {label}.",
    ):
        self.device = pick_device(device)
        self.dtype = pick_dtype(self.device)
        self.prompt_template = prompt_template

        logger.info("Loading %s onto device '%s' (dtype: %s)...", model_name, self.device, self.dtype)
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(model_name)
        self.model = model.to(device=self.device, dtype=self.dtype).eval()
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(model_name)

        with torch.no_grad():
            self.logit_scale = float(self.model.logit_scale.exp().cpu())

    def encode_text(self, labels: list[str]) -> np.ndarray:
        """Encode species text prompts into L2-normalised features (k, d)."""
        logger.info("Computing zero-shot text embeddings for %d species...", len(labels))
        prompts = [self.prompt_template.format(label=lbl) for lbl in labels]
        tokens = self.tokenizer(prompts).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.float().cpu().numpy()

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        """Encode batch of PIL Images into L2-normalised features (b, d)."""
        tensors = [self.preprocess(img) for img in images]
        batch = torch.stack(tensors).to(device=self.device, dtype=self.dtype)
        with torch.no_grad():
            feats = self.model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.float().cpu().numpy()


def is_valid_npz(path: Path) -> bool:
    """Check if an existing .npz has all required keys and non-empty data."""
    if not path.exists() or path.stat().st_size < 1024:
        return False
    try:
        with np.load(path) as z:
            required = ["embeddings", "labels", "interval", "is_night", "text_embeddings", "logit_scale"]
            return all(k in z and np.size(z[k]) > 0 for k in required)  # logit_scale is a 0-d scalar
    except Exception:
        return False


def process_site(
    site_id: str,
    df_site: pd.DataFrame,
    class_to_idx: dict[str, int],
    text_embeddings: np.ndarray,
    extractor: BioCLIP2Extractor,
    output_dir: Path,
    images_dir: Path | None,
    batch_size: int = 32,
    num_workers: int = 4,
    limit: int | None = None,
    skip_existing: bool = True,
    mock: bool = False,
) -> Path | None:
    """Extract and write embeddings for a single camera trap site."""
    out_file = output_dir / f"{site_id}.npz"
    if skip_existing and is_valid_npz(out_file):
        logger.info("Site %s already processed at %s (skipping)", site_id, out_file)
        return out_file

    if limit is not None:
        df_site = df_site.head(limit)

    total_images = len(df_site)
    logger.info("Processing site '%s': %d images across %d intervals...", site_id, total_images, df_site["interval"].nunique())

    image_ids = df_site["image_id"].tolist()
    labels = [class_to_idx[sp] for sp in df_site["species"]]
    intervals = df_site["interval"].tolist()
    is_nights = df_site["is_night"].tolist()

    valid_embs = []
    valid_labels = []
    valid_intervals = []
    valid_nights = []
    valid_image_ids = []

    # Process in batches
    for i in range(0, total_images, batch_size):
        chunk_ids = image_ids[i : i + batch_size]
        chunk_labels = labels[i : i + batch_size]
        chunk_intervals = intervals[i : i + batch_size]
        chunk_nights = is_nights[i : i + batch_size]

        # Download / load images concurrently
        if mock:
            imgs = [Image.new("RGB", (224, 224), (128, 128, 128)) for _ in chunk_ids]
        else:
            with ThreadPoolExecutor(max_workers=num_workers) as pool:
                imgs = list(pool.map(lambda i_id: fetch_image(i_id, images_dir=images_dir), chunk_ids))

        # Filter out failed downloads
        batch_imgs = []
        for j, img in enumerate(imgs):
            if img is not None:
                batch_imgs.append(img)
                valid_labels.append(chunk_labels[j])
                valid_intervals.append(chunk_intervals[j])
                valid_nights.append(chunk_nights[j])
                valid_image_ids.append(chunk_ids[j])

        if batch_imgs:
            embs = extractor.encode_images(batch_imgs)
            valid_embs.append(embs)

        if (i + batch_size) % (batch_size * 5) == 0 or (i + batch_size) >= total_images:
            logger.info("  Site %s progress: %d/%d images processed", site_id, min(i + batch_size, total_images), total_images)

    if not valid_embs:
        logger.error("Site %s had 0 valid embeddings extracted!", site_id)
        return None

    all_embs = np.vstack(valid_embs)
    all_labels = np.array(valid_labels, dtype=np.int64)
    all_intervals = np.array(valid_intervals, dtype=np.int64)
    all_nights = np.array(valid_nights, dtype=bool)
    all_image_ids = np.array(valid_image_ids)

    # Save compressed .npz
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_file,
        embeddings=all_embs,
        labels=all_labels,
        interval=all_intervals,
        is_night=all_nights,
        text_embeddings=text_embeddings,
        logit_scale=extractor.logit_scale,
        site=site_id,
        image_ids=all_image_ids,
        classes=np.array(list(class_to_idx.keys())),
    )
    logger.info("Saved %s (shape: %s, size: %.2f MB)", out_file, all_embs.shape, out_file.stat().st_size / (1024 * 1024))
    return out_file


def main():
    parser = argparse.ArgumentParser(description="Extract BioCLIP 2 embeddings for camera trap sites.")
    parser.add_argument("--manifest", type=str, default="research/manifests/lila_dev.parquet", help="Path to manifest parquet")
    parser.add_argument("--output-dir", type=str, default="research/embeddings/lila_dev", help="Directory for output .npz files")
    parser.add_argument("--images-dir", type=str, default=None, help="Optional directory with local images")
    parser.add_argument("--sites", nargs="*", default=None, help="Optional list of specific sites to process")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for BioCLIP 2 encoding")
    parser.add_argument("--num-workers", type=int, default=4, help="Parallel image downloads per batch")
    parser.add_argument("--device", type=str, default="auto", help="Execution device ('auto', 'cpu', 'cuda', 'mps')")
    parser.add_argument("--limit", type=int, default=None, help="Limit images per site (for smoke testing)")
    parser.add_argument("--mock", action="store_true", help="Use mock extractor for rapid offline verification")
    parser.add_argument("--skip-existing", action="store_true", default=True, help="Skip already extracted sites")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    logger.info("Loading manifest from %s...", manifest_path)
    df = pd.read_parquet(manifest_path)
    logger.info("Loaded %d total records across %d sites.", len(df), df["site"].nunique())

    # Build consistent global species index across the manifest
    unique_species = sorted(df["species"].unique())
    class_to_idx = {sp: idx for idx, sp in enumerate(unique_species)}
    logger.info("Global class vocabulary: %d species", len(unique_species))

    # Initialize extractor
    if args.mock:
        logger.info("Using MockExtractor for rapid verification (dim=512)...")
        extractor = MockExtractor()
    else:
        extractor = BioCLIP2Extractor(device=args.device)
    text_embeddings = extractor.encode_text(unique_species)

    output_dir = Path(args.output_dir)
    target_sites = sorted(df["site"].unique()) if args.sites is None else args.sites

    logger.info("Beginning extraction for %d sites: %s", len(target_sites), target_sites)
    for site_id in target_sites:
        df_site = df[df["site"] == site_id].sort_values(by="timestamp").reset_index(drop=True)
        if len(df_site) == 0:
            logger.warning("No records found for site %s", site_id)
            continue
        process_site(
            site_id=site_id,
            df_site=df_site,
            class_to_idx=class_to_idx,
            text_embeddings=text_embeddings,
            extractor=extractor,
            output_dir=output_dir,
            images_dir=Path(args.images_dir) if args.images_dir else None,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            limit=args.limit,
            skip_existing=args.skip_existing,
            mock=args.mock,
        )

    logger.info("All requested sites processed.")


if __name__ == "__main__":
    main()
