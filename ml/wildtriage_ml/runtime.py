"""The inference pipeline a worker uses: one place that decides which adapters back each role."""

from wildtriage_ml.adapters.base import Detector, SpeciesClassifier, VideoTracker
from wildtriage_ml.registry import ModelRegistry
from wildtriage_ml.triage.calibration import TemperatureScaler
from wildtriage_ml.triage.policy import PolicyConfig, TriagePolicy


class Pipeline:
    """backend="stub": deterministic CPU models (laptop, CI). backend="real": MegaDetector, BioCLIP 2, SAM 3.

    device: "auto" (default) picks cuda, then mps, then cpu; or pass/set WT_DEVICE explicitly.
    Stub models always run on the CPU.
    """

    def __init__(self, backend: str = "stub", vram_budget_gb: float = 18.0, device: str | None = None):
        if backend not in ("stub", "real"):
            raise ValueError(f"unknown model backend {backend!r}")
        self.backend = backend
        self.registry = ModelRegistry(vram_budget_gb, "cpu" if backend == "stub" else device)
        # Replaced by the active project/site calibrator and policy from model_versions (W4-W5).
        self.calibrator = TemperatureScaler()
        self.policy = TriagePolicy(PolicyConfig())

    @property
    def detector(self) -> Detector:
        if self.backend == "stub":
            from wildtriage_ml.adapters.stubs import StubDetector

            return self.registry.get("detector", StubDetector)
        from wildtriage_ml.adapters.megadetector import MegaDetectorAdapter

        return self.registry.get("detector", MegaDetectorAdapter)

    @property
    def classifier(self) -> SpeciesClassifier:
        if self.backend == "stub":
            from wildtriage_ml.adapters.stubs import StubClassifier

            return self.registry.get("classifier", StubClassifier)
        from wildtriage_ml.adapters.bioclip import BioCLIP2Classifier

        return self.registry.get("classifier", BioCLIP2Classifier)

    @property
    def video_tracker(self) -> VideoTracker:
        """SAM 3 (real backend) needs CUDA and ~8 GB [A]: the registry refuses it on the 4 GB T1000 and on the Mac."""
        if self.backend == "stub":
            from wildtriage_ml.adapters.stubs import StubVideoTracker

            return self.registry.get("video_tracker", StubVideoTracker)
        from wildtriage_ml.adapters.sam3 import Sam3VideoTracker

        return self.registry.get("video_tracker", Sam3VideoTracker)
