"""Typed interfaces every model adapter implements, so models can be swapped without touching the app."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class ModelInfo:
    name: str
    version: str
    task: str  # detect | classify | segment_track | camouflage | aerial_count | triage
    license: str
    commercial_ok: bool
    vram_gb: float  # estimate used by ModelRegistry's budget; replace with measured values in W1
    framework: str = "pytorch"
    # Device kinds the model is known to run on. ModelRegistry refuses others (e.g. SAM 3 on a Mac).
    devices: tuple[str, ...] = ("cuda", "mps", "cpu")


@dataclass
class BoxDetection:
    # Normalised [0, 1] top-left x, y, width, height (MegaDetector convention).
    x: float
    y: float
    w: float
    h: float
    score: float
    category: str  # animal | person | vehicle

    @property
    def area(self) -> float:
        return self.w * self.h


@dataclass
class ClassScores:
    labels: list[str]
    logits: np.ndarray  # shape (n_labels,), uncalibrated


@dataclass
class TrackResult:
    track_key: int
    frames: list[int]
    boxes: list[tuple[float, float, float, float]]
    label: str | None = None
    score: float = 0.0
    extra: dict = field(default_factory=dict)


class Adapter(Protocol):
    info: ModelInfo

    def load(self, device: str) -> None: ...

    def unload(self) -> None: ...


class Detector(Adapter, Protocol):
    def detect(self, image_path: Path) -> list[BoxDetection]: ...


class SpeciesClassifier(Adapter, Protocol):
    def classify(self, image_path: Path, box: BoxDetection | None, labels: list[str]) -> ClassScores: ...


class VideoTracker(Adapter, Protocol):
    def track(self, video_path: Path, prompts: list[str]) -> list[TrackResult]: ...


class CamouflageRefiner(Adapter, Protocol):
    def refine(self, image_path: Path, box: BoxDetection) -> np.ndarray: ...


class AerialCounter(Adapter, Protocol):
    def count(self, image_path: Path) -> list[tuple[float, float, float]]:
        """Returns (x, y, score) points in normalised image coordinates."""
        ...
