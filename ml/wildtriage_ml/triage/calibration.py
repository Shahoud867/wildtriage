"""Post-hoc calibration (temperature scaling) and calibration metrics."""

import numpy as np

_GOLDEN = (np.sqrt(5) - 1) / 2


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def nll(logits: np.ndarray, labels: np.ndarray, temperature: float = 1.0) -> float:
    labels = np.asarray(labels, dtype=int)
    p = softmax(logits, temperature)
    return float(-np.log(np.clip(p[np.arange(len(labels)), labels], 1e-12, 1.0)).mean())


def fit_temperature(
    logits: np.ndarray, labels: np.ndarray, t_min: float = 0.05, t_max: float = 20.0, iters: int = 80
) -> float:
    """Fit T by minimising NLL. NLL is convex in the inverse temperature, so golden-section search is exact."""
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    if labels.size == 0:
        return 1.0

    def f(beta: float) -> float:
        return nll(logits, labels, 1.0 / beta)

    lo, hi = 1.0 / t_max, 1.0 / t_min
    c, d = hi - _GOLDEN * (hi - lo), lo + _GOLDEN * (hi - lo)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - _GOLDEN * (hi - lo)
            fc = f(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + _GOLDEN * (hi - lo)
            fd = f(d)
    return float(2.0 / (lo + hi))


class TemperatureScaler:
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "TemperatureScaler":
        self.temperature = fit_temperature(logits, labels)
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        return softmax(logits, self.temperature)


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15, adaptive: bool = False
) -> float:
    """Top-label ECE. `adaptive=True` uses equal-mass bins (adaptive ECE)."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    if labels.size == 0:
        return 0.0
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(np.float64)
    if adaptive:
        edges = np.quantile(conf, np.linspace(0, 1, n_bins + 1))
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    bins = np.clip(np.searchsorted(edges, conf, side="right") - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)
