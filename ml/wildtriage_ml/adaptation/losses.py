"""Balanced Softmax (Ren et al., NeurIPS 2020): train with logits + log(class prior), predict with raw logits.
Counteracts the severe class imbalance of camera-trap streams."""

import numpy as np


def log_prior(class_counts: np.ndarray, eps: float = 1.0) -> np.ndarray:
    counts = np.asarray(class_counts, dtype=np.float64) + eps  # smoothing: unseen classes keep a small prior
    return np.log(counts / counts.sum())


def balanced_softmax_loss(logits: np.ndarray, labels: np.ndarray, class_counts: np.ndarray) -> float:
    z = np.asarray(logits, dtype=np.float64) + log_prior(class_counts)
    z = z - z.max(axis=1, keepdims=True)
    log_p = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return float(-log_p[np.arange(len(labels)), np.asarray(labels, dtype=int)].mean())
