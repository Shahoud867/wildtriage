"""Adapters for P1/P2 modules. Interfaces are fixed now so the app can expose them later.

- DepthSAM (Paper 2, https://github.com/hjy0518/DepthSAM): whole-image camouflaged-object
  segmentation. Code Apache-2.0, but it depends on Depth-Anything-V2-Large (CC-BY-NC-4.0).
  Our use is an adaptation: run it on an enlarged crop around a low-confidence detection.
  Runs in its own environment and queue ("gpu-cod"); never resident together with SAM 3.
- SpeciesNet (https://github.com/google/cameratrapai, Apache-2.0): reference classifier, run via
  its CLI on folders. Trained on public LILA data, so it is not zero-shot on LILA.
- OWL / HerdNet (https://github.com/microsoft/MegaDetector-Overhead): point-based aerial counting.
  Code MIT; weights and data CC BY-NC-SA 4.0. Tiled inference on large aerial stills.
"""

from pathlib import Path

import numpy as np

from wildtriage_ml.adapters.base import BoxDetection, ModelInfo


class DepthSAMRefiner:
    # [A] treated as CUDA-only until its code is checked on MPS/CPU (training uses torchrun on 4 GPUs).
    info = ModelInfo("depthsam", "cvpr2026", "camouflage", "Apache-2.0 + CC-BY-NC-4.0 (DA-V2-L)", False, vram_gb=6.0,
                     devices=("cuda",))

    def load(self, device: str) -> None:
        raise NotImplementedError("P1 (W5): load DepthSAM from third_party/DepthSAM with its released weights")

    def unload(self) -> None:
        pass

    def refine(self, image_path: Path, box: BoxDetection) -> np.ndarray:
        raise NotImplementedError("P1 (W5): crop around the box, predict a camouflage mask, paste back")


class OWLAerialCounter:
    # OWL-C is documented to run on CPU; MPS untested.
    info = ModelInfo("owl-c", "megadetector-overhead", "aerial_count", "MIT + CC-BY-NC-SA-4.0", False, vram_gb=2.0,
                     devices=("cuda", "cpu"))

    def load(self, device: str) -> None:
        raise NotImplementedError("P1 (W6): load OWL-C/OWL-D from MegaDetector-Overhead")

    def unload(self) -> None:
        pass

    def count(self, image_path: Path) -> list[tuple[float, float, float]]:
        raise NotImplementedError("P1 (W6): tiled inference, merge points across tile overlaps")
