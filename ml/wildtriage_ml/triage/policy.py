"""The triage policy the app runs: auto-accept, send to review (with reasons), or auto-empty.

Thresholds come from research code (conformal fitting on past, reviewed data) and are stored with the
policy's model_version, so every decision is reproducible.
"""

import hashlib
from dataclasses import asdict, dataclass, field


@dataclass
class PolicyConfig:
    # Calibrated top-1 probability needed to auto-accept, per condition group ("Mondrian" by lighting).
    accept_threshold: dict[str, float] = field(default_factory=lambda: {"day": 0.90, "night": 0.95})
    min_margin: float = 0.1  # top-1 minus top-2 probability; below this the species is ambiguous
    detection_min_score: float = 0.2  # below this a box is ignored
    empty_max_score: float = 0.1  # media with no box above this may be auto-emptied
    tiny_box_area: float = 0.0025  # 5% x 5% of the frame
    min_brightness: float = 0.08
    min_blur: float = 1e-4  # Laplacian variance on [0, 1] grayscale; tune on the hard set
    audit_rate: float = 0.02  # share of auto-accepted items still sent to review, to measure realised error

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Decision:
    outcome: str  # auto_accept | review | auto_empty
    risk: float
    reasons: list[str]

    @property
    def priority(self) -> float:
        # Riskier items first; quality problems nudge items up the queue.
        return self.risk + 0.05 * len([r for r in self.reasons if r != "audit_sample"])


def _quality_reasons(cfg: PolicyConfig, quality: dict) -> list[str]:
    reasons = []
    if quality.get("brightness", 1.0) < cfg.min_brightness:
        reasons.append("too_dark")
    if quality.get("blur", 1.0) < cfg.min_blur:
        reasons.append("blurred")
    return reasons


def _in_audit_sample(key: str, rate: float) -> bool:
    """Deterministic pseudo-random audit sampling, so reruns make the same decisions."""
    h = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")
    return h / 2**64 < rate


class TriagePolicy:
    def __init__(self, config: PolicyConfig | None = None):
        self.config = config or PolicyConfig()

    def decide_detection(
        self, top1: float, top2: float, box_area: float, quality: dict, audit_key: str
    ) -> Decision:
        cfg = self.config
        group = "night" if quality.get("is_night") else "day"
        reasons = []
        if top1 < cfg.accept_threshold.get(group, max(cfg.accept_threshold.values())):
            reasons.append("low_confidence")
            if group == "night":
                reasons.append("night")
        if top1 - top2 < cfg.min_margin:
            reasons.append("ambiguous_species")
        if box_area < cfg.tiny_box_area:
            reasons.append("tiny_object")
        reasons += _quality_reasons(cfg, quality)
        risk = 1.0 - top1
        if reasons:
            return Decision("review", risk, reasons)
        if _in_audit_sample(audit_key, cfg.audit_rate):
            return Decision("review", risk, ["audit_sample"])
        return Decision("auto_accept", risk, [])

    def decide_empty(self, max_animal_score: float, quality: dict) -> Decision:
        cfg = self.config
        reasons = _quality_reasons(cfg, quality)
        if max_animal_score < cfg.empty_max_score and not reasons:
            return Decision("auto_empty", max_animal_score, [])
        return Decision("review", max(max_animal_score, 0.5), ["possible_missed_animal", *reasons])
