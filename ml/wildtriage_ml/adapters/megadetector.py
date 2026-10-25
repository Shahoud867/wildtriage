"""MegaDetector adapter (https://github.com/agentmorris/MegaDetector, `pip install megadetector`).

[A] The Python entry points below follow the package's run_detector module; confirm them against
the installed version in W1/W2 (the documented CLI is `python -m megadetector.detection.run_detector_batch`).
License: repository MIT; MDv5 weights build on YOLOv5 (AGPL-3.0 code). Check before a commercial release.
"""

from pathlib import Path

from wildtriage_ml.adapters.base import BoxDetection, ModelInfo

# MegaDetector output categories.
CATEGORIES = {"1": "animal", "2": "person", "3": "vehicle"}


class MegaDetectorAdapter:
    def __init__(self, model_name: str = "MDV5A", threshold: float = 0.1):
        self.model_name = model_name
        self.threshold = threshold
        self.info = ModelInfo(f"megadetector-{model_name.lower()}", model_name, "detect", "MIT (weights: verify)", False, vram_gb=1.5)
        self._detector = None

    def load(self, device: str) -> None:
        from megadetector.detection.run_detector import load_detector  # [A] verify import path

        # [A] load_detector picks its own device; check in W2 that it uses MPS on the Mac and the GPU on
        # CUDA machines. If MPS misbehaves, force CPU for this model (slower but reliable).
        self._detector = load_detector(self.model_name)

    def unload(self) -> None:
        self._detector = None

    def detect(self, image_path: Path) -> list[BoxDetection]:
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        result = self._detector.generate_detections_one_image(
            image, str(image_path), detection_threshold=self.threshold
        )  # [A] verify signature
        return [
            BoxDetection(*d["bbox"], score=float(d["conf"]), category=CATEGORIES.get(str(d["category"]), "animal"))
            for d in result.get("detections", [])
        ]
