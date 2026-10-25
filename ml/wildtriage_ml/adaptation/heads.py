"""Linear species heads on frozen (cached) BioCLIP 2 embeddings.

The head starts from the zero-shot text classifier (logit_scale * text embeddings), so with no data
it reproduces zero-shot predictions exactly; WiSE interpolation blends the trained head back toward it.
Cheap enough to retrain per site in seconds on a laptop CPU.
"""

from dataclasses import dataclass

import numpy as np

from wildtriage_ml.adaptation.losses import log_prior


@dataclass
class LinearHead:
    weight: np.ndarray  # (n_classes, dim)
    bias: np.ndarray  # (n_classes,)

    def logits(self, embeddings: np.ndarray) -> np.ndarray:
        return np.asarray(embeddings) @ self.weight.T + self.bias

    @classmethod
    def from_zero_shot(cls, text_embeddings: np.ndarray, logit_scale: float) -> "LinearHead":
        w = logit_scale * np.asarray(text_embeddings, dtype=np.float64)
        return cls(weight=w, bias=np.zeros(w.shape[0]))


def wise_interpolate(zero_shot: LinearHead, finetuned: LinearHead, alpha: float) -> LinearHead:
    """WiSE-FT style weight interpolation: alpha=0 -> zero-shot, alpha=1 -> fine-tuned."""
    return LinearHead(
        weight=(1 - alpha) * zero_shot.weight + alpha * finetuned.weight,
        bias=(1 - alpha) * zero_shot.bias + alpha * finetuned.bias,
    )


def fit_linear_head(
    embeddings: np.ndarray,
    labels: np.ndarray,
    init: LinearHead,
    class_counts: np.ndarray | None = None,
    balanced_softmax: bool = True,
    lr: float = 0.05,
    epochs: int = 200,
    weight_decay: float = 1e-4,
    anchor: float = 1e-3,
) -> LinearHead:
    """Full-batch gradient descent on (Balanced) softmax cross-entropy.

    `anchor` adds an L2 pull toward the initial (zero-shot) weights, which keeps tiny per-site label
    sets from overwriting what the foundation model already knows.
    """
    X = np.asarray(embeddings, dtype=np.float64)
    y = np.asarray(labels, dtype=int)
    W, b = init.weight.copy(), init.bias.copy()
    W0 = init.weight
    n, k = X.shape[0], W.shape[0]
    if n == 0:
        return LinearHead(W, b)
    counts = np.bincount(y, minlength=k) if class_counts is None else np.asarray(class_counts)
    adjust = log_prior(counts) if balanced_softmax else np.zeros(k)
    onehot = np.eye(k)[y]
    for _ in range(epochs):
        z = X @ W.T + b + adjust
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        g = (p - onehot) / n
        W -= lr * (g.T @ X + weight_decay * W + anchor * (W - W0))
        b -= lr * g.sum(axis=0)
    return LinearHead(W, b)
