import numpy as np


def expand_box(box: tuple[float, float, float, float], factor: float) -> tuple[float, float, float, float]:
    """Grow a normalised (x, y, w, h) box by `factor` (0.5 = +50%) around its centre, clipped to the image."""
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    w2, h2 = w * (1 + factor), h * (1 + factor)
    x0, y0 = max(0.0, cx - w2 / 2), max(0.0, cy - h2 / 2)
    x1, y1 = min(1.0, cx + w2 / 2), min(1.0, cy + h2 / 2)
    return x0, y0, x1 - x0, y1 - y0


def aggregate_logits(logits: list[np.ndarray]) -> np.ndarray:
    """Default for bursts and clips: average the *logits*, then apply the softmax once.

    Dussert et al. (Remote Sensing in Ecology and Conservation 2025) found this the most effective way
    to get good calibration and accuracy at sequence level for camera-trap classifiers. Temperature
    scaling is then fitted on the aggregated logits, not on single images.
    """
    z = np.mean(np.stack(logits).astype(np.float64), axis=0)
    z -= z.max()
    e = np.exp(z)
    return e / e.sum()


def aggregate_sequence(probs: list[np.ndarray], method: str = "mean") -> np.ndarray:
    """Combine per-image (or per-frame) class *probabilities*. Kept for ablations against aggregate_logits."""
    stacked = np.stack(probs)
    if method == "mean":
        out = stacked.mean(axis=0)
    elif method == "max":
        out = stacked.max(axis=0)
    elif method == "log_mean":  # product of experts: sum of log-probabilities
        out = np.exp(np.log(np.clip(stacked, 1e-12, 1)).mean(axis=0))
    else:
        raise ValueError(f"unknown aggregation method: {method}")
    return out / out.sum()
