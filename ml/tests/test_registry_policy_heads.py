import numpy as np
import pytest

from wildtriage_ml.adaptation.heads import LinearHead, fit_linear_head, wise_interpolate
from wildtriage_ml.adapters.base import ModelInfo
from wildtriage_ml.registry import ModelRegistry, VRAMBudgetError
from wildtriage_ml.triage.policy import PolicyConfig, TriagePolicy


class FakeModel:
    def __init__(self, name, gb, log):
        self.info = ModelInfo(name, "0", "detect", "MIT", True, vram_gb=gb)
        self.log = log

    def load(self, device):
        self.log.append(("load", self.info.name))

    def unload(self):
        self.log.append(("unload", self.info.name))


def test_registry_evicts_least_recently_used_to_fit_budget():
    log = []
    reg = ModelRegistry(vram_budget_gb=3.5, device="cpu")  # the T1000 budget
    reg.get("md", lambda: FakeModel("md", 1.5, log))
    reg.get("bioclip", lambda: FakeModel("bioclip", 2.0, log))
    reg.get("md", lambda: FakeModel("md", 1.5, log))  # touch: bioclip becomes LRU
    reg.get("speciesnet", lambda: FakeModel("speciesnet", 1.5, log))
    assert reg.loaded == ["md", "speciesnet"]
    assert ("unload", "bioclip") in log
    assert reg.used_gb <= 3.5


def test_registry_refuses_models_that_cannot_fit():
    reg = ModelRegistry(vram_budget_gb=3.5, device="cpu")
    with pytest.raises(VRAMBudgetError, match="gpu-heavy"):
        reg.get("sam3", lambda: FakeModel("sam3", 8.0, []))


def test_policy_accepts_confident_day_detections_and_reviews_the_rest():
    policy = TriagePolicy(PolicyConfig(audit_rate=0.0))
    day = {"is_night": False, "brightness": 0.5, "blur": 0.01}
    night = {**day, "is_night": True}
    assert policy.decide_detection(0.93, 0.03, 0.05, day, "a").outcome == "auto_accept"
    d = policy.decide_detection(0.93, 0.03, 0.05, night, "b")  # night needs 0.95
    assert d.outcome == "review" and {"low_confidence", "night"} <= set(d.reasons)
    d = policy.decide_detection(0.99, 0.0, 0.0001, day, "c")
    assert d.outcome == "review" and "tiny_object" in d.reasons
    dark = {**day, "brightness": 0.01}
    assert "too_dark" in policy.decide_detection(0.99, 0.0, 0.05, dark, "d").reasons


def test_policy_audit_sampling_is_deterministic():
    policy = TriagePolicy(PolicyConfig(audit_rate=0.5))
    day = {"is_night": False, "brightness": 0.5, "blur": 0.01}
    outcomes = [policy.decide_detection(0.99, 0.0, 0.05, day, f"k{i}").outcome for i in range(200)]
    assert outcomes == [policy.decide_detection(0.99, 0.0, 0.05, day, f"k{i}").outcome for i in range(200)]
    assert 60 < outcomes.count("review") < 140


def test_empty_decisions():
    policy = TriagePolicy()
    ok = {"brightness": 0.5, "blur": 0.01}
    assert policy.decide_empty(0.02, ok).outcome == "auto_empty"
    d = policy.decide_empty(0.3, ok)
    assert d.outcome == "review" and "possible_missed_animal" in d.reasons


def test_linear_head_starts_at_zero_shot_and_learns():
    rng = np.random.default_rng(0)
    k, dim, n = 3, 16, 600
    text = rng.normal(size=(k, dim))
    text /= np.linalg.norm(text, axis=1, keepdims=True)
    zs = LinearHead.from_zero_shot(text, logit_scale=10.0)
    centers = rng.normal(size=(k, dim)) * 2
    y = rng.integers(0, k, n)
    X = centers[y] + rng.normal(size=(n, dim)) * 0.5
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    head = fit_linear_head(X, y, zs, lr=0.5, epochs=300)
    acc = (head.logits(X).argmax(1) == y).mean()
    assert acc > 0.9
    assert np.allclose(wise_interpolate(zs, head, 0.0).weight, zs.weight)
    assert np.allclose(wise_interpolate(zs, head, 1.0).weight, head.weight)
