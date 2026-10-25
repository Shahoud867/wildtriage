import numpy as np
import pytest

from wildtriage_ml.triage.selection import STRATEGIES, select_for_review


def _probs(n=200, k=4, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 2, size=(n, k))
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_strategies_return_unique_indices_within_budget(strategy):
    probs = _probs()
    emb = np.random.default_rng(1).normal(size=(200, 8))
    idx = select_for_review(strategy, 20, probs, embeddings=emb)
    assert len(idx) == 20
    assert len(set(idx.tolist())) == 20
    assert idx.min() >= 0 and idx.max() < 200


def test_budget_larger_than_pool_is_capped():
    assert len(select_for_review("least_confidence", 999, _probs(n=10))) == 10


def test_least_confidence_picks_lowest_max_probability():
    probs = np.array([[0.9, 0.1], [0.55, 0.45], [0.7, 0.3]])
    assert select_for_review("least_confidence", 1, probs).tolist() == [1]


def test_balanced_uncertainty_reaches_rare_predicted_class():
    # 95 confident-but-uncertain-ish items of class 0, 5 of class 1 that are less uncertain.
    probs = np.vstack([np.tile([0.6, 0.4], (95, 1)), np.tile([0.2, 0.8], (5, 1))])
    picked = select_for_review("balanced_uncertainty", 10, probs)
    assert any(i >= 95 for i in picked)
    assert not any(i >= 95 for i in select_for_review("least_confidence", 10, probs))


def test_random_is_reproducible():
    probs = _probs()
    a = select_for_review("random", 15, probs, seed=7)
    b = select_for_review("random", 15, probs, seed=7)
    assert a.tolist() == b.tolist()
