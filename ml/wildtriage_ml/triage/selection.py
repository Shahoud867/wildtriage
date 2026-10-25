"""Which items humans should review, given a fixed budget (Exp 1 baselines plus our strategy)."""

import numpy as np

STRATEGIES = ("random", "least_confidence", "margin", "entropy", "kcenter", "balanced_uncertainty")


def uncertainty(probs: np.ndarray, kind: str = "least_confidence") -> np.ndarray:
    """Higher = more uncertain."""
    probs = np.asarray(probs, dtype=np.float64)
    if kind == "least_confidence":
        return 1.0 - probs.max(axis=1)
    if kind == "margin":
        top2 = np.sort(probs, axis=1)[:, -2:] if probs.shape[1] > 1 else np.hstack([np.zeros_like(probs), probs])
        return 1.0 - (top2[:, 1] - top2[:, 0])
    if kind == "entropy":
        return -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=1)
    raise ValueError(f"unknown uncertainty kind: {kind}")


def _kcenter(embeddings: np.ndarray, budget: int, labeled: np.ndarray | None, first: int) -> np.ndarray:
    """Greedy farthest-first traversal (k-center coreset)."""
    emb = np.asarray(embeddings, dtype=np.float64)
    if labeled is not None and len(labeled):
        dist = np.min(((emb[:, None, :] - np.asarray(labeled)[None, :, :]) ** 2).sum(-1), axis=1)
        chosen: list[int] = []
    else:
        chosen = [first]
        dist = ((emb - emb[first]) ** 2).sum(-1)
    while len(chosen) < budget:
        nxt = int(np.argmax(dist))
        chosen.append(nxt)
        dist = np.minimum(dist, ((emb - emb[nxt]) ** 2).sum(-1))
    return np.array(chosen[:budget], dtype=int)


def _balanced_uncertainty(probs: np.ndarray, budget: int) -> np.ndarray:
    """Split the budget across predicted classes (proportional to sqrt(count), at least one per class
    while budget allows), take the most uncertain items within each class, fill the rest globally.
    Protects rare species that pure uncertainty sampling would starve."""
    pred = probs.argmax(axis=1)
    unc = uncertainty(probs, "least_confidence")
    classes, counts = np.unique(pred, return_counts=True)
    order = np.argsort(-counts)
    classes, counts = classes[order], counts[order]
    weights = np.sqrt(counts)
    quota = np.floor(budget * weights / weights.sum()).astype(int)
    for i in range(len(classes)):
        if quota.sum() >= budget:
            break
        if quota[i] == 0:
            quota[i] = 1
    while quota.sum() > budget:
        quota[np.argmax(quota)] -= 1
    quota = np.minimum(quota, counts)
    chosen: list[int] = []
    for cls, q in zip(classes, quota):
        idx = np.flatnonzero(pred == cls)
        chosen.extend(idx[np.argsort(-unc[idx])][:q].tolist())
    if len(chosen) < budget:
        rest = np.setdiff1d(np.argsort(-unc), chosen, assume_unique=False)
        rest = rest[np.argsort(-unc[rest])]
        chosen.extend(rest[: budget - len(chosen)].tolist())
    return np.array(chosen[:budget], dtype=int)


def select_for_review(
    strategy: str,
    budget: int,
    probs: np.ndarray,
    embeddings: np.ndarray | None = None,
    labeled_embeddings: np.ndarray | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Return indices (into `probs`) of the items to send to human review."""
    probs = np.asarray(probs, dtype=np.float64)
    n = probs.shape[0]
    budget = int(min(max(budget, 0), n))
    if budget == 0:
        return np.array([], dtype=int)
    if strategy == "random":
        return np.random.default_rng(seed).choice(n, size=budget, replace=False)
    if strategy in ("least_confidence", "margin", "entropy"):
        return np.argsort(-uncertainty(probs, strategy), kind="stable")[:budget]
    if strategy == "kcenter":
        if embeddings is None:
            raise ValueError("kcenter needs embeddings")
        first = int(np.argmax(uncertainty(probs)))
        return _kcenter(embeddings, budget, labeled_embeddings, first)
    if strategy == "balanced_uncertainty":
        return _balanced_uncertainty(probs, budget)
    raise ValueError(f"unknown strategy {strategy!r}; expected one of {STRATEGIES}")
