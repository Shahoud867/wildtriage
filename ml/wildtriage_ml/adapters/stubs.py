"""Deterministic CPU stand-ins for the real models.

They let the platform (API, queue, UI) be developed and tested without a GPU. Outputs are seeded
from the file's bytes, so the same file always gives the same predictions.
"""

import hashlib
from pathlib import Path

import numpy as np

from wildtriage_ml.adapters.base import BoxDetection, ClassScores, ModelInfo, TrackResult


def _rng_for(path: Path, salt: str) -> np.random.Generator:
    digest = hashlib.sha256(salt.encode() + Path(path).read_bytes()[:65536]).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


class _NoLoad:
    def load(self, device: str) -> None:
        pass

    def unload(self) -> None:
        pass


class StubDetector(_NoLoad):
    info = ModelInfo("stub-detector", "0", "detect", "Apache-2.0", True, vram_gb=0.0, framework="numpy")

    def detect(self, image_path: Path) -> list[BoxDetection]:
        rng = _rng_for(image_path, "det")
        n = int(rng.integers(0, 3))  # 0 boxes = an empty image
        dets = []
        for _ in range(n):
            w, h = rng.uniform(0.05, 0.4, size=2)
            x, y = rng.uniform(0, 1 - w), rng.uniform(0, 1 - h)
            dets.append(BoxDetection(float(x), float(y), float(w), float(h), float(rng.uniform(0.3, 0.99)), "animal"))
        return dets


class StubClassifier(_NoLoad):
    info = ModelInfo("stub-classifier", "0", "classify", "Apache-2.0", True, vram_gb=0.0, framework="numpy")

    def classify(self, image_path: Path, box: BoxDetection | None, labels: list[str]) -> ClassScores:
        rng = _rng_for(image_path, f"cls:{box}")
        logits = rng.normal(0, 1, size=len(labels))
        logits[int(rng.integers(0, len(labels)))] += rng.uniform(0, 5)
        return ClassScores(labels=list(labels), logits=logits)


class StubVideoTracker(_NoLoad):
    info = ModelInfo("stub-video-tracker", "0", "segment_track", "Apache-2.0", True, vram_gb=0.0, framework="numpy")

    def track(self, video_path: Path, prompts: list[str]) -> list[TrackResult]:
        rng = _rng_for(video_path, "trk")
        tracks = []
        for key in range(int(rng.integers(0, 4))):
            start = int(rng.integers(0, 60))
            frames = list(range(start, start + int(rng.integers(6, 60))))
            boxes = [(0.1, 0.1, 0.2, 0.2)] * len(frames)
            label = prompts[int(rng.integers(0, len(prompts)))] if prompts else None
            tracks.append(TrackResult(key, frames, boxes, label, float(rng.uniform(0.3, 0.99))))
        return tracks
