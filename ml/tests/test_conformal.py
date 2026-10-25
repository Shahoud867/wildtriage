from math import comb

import numpy as np

from wildtriage_ml.triage.conformal import (
    AdaptiveAlpha,
    binom_cdf,
    crc_accept_threshold,
    ltt_selective_threshold,
)


def _calibrated(n, rng):
    conf = rng.uniform(0, 1, n)
    correct = rng.uniform(0, 1, n) < conf  # P(correct | conf) = conf
    return conf, correct


def test_binom_cdf_matches_exact():
    for n, k, p in [(10, 3, 0.2), (25, 0, 0.05), (40, 12, 0.3)]:
        exact = sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))
        assert abs(binom_cdf(k, n, p) - exact) < 1e-12


def test_crc_controls_marginal_accepted_error_rate():
    rng = np.random.default_rng(0)
    alpha, risks = 0.05, []
    for _ in range(200):
        conf, correct = _calibrated(500, rng)
        lam = crc_accept_threshold(conf, correct, alpha)
        t_conf, t_correct = _calibrated(5000, rng)
        risks.append(np.mean((t_conf >= lam) & ~t_correct))
    assert np.mean(risks) <= alpha + 0.005


def test_crc_accepts_nothing_when_alpha_is_impossible():
    conf, correct = np.array([0.9, 0.8]), np.array([False, False])
    assert crc_accept_threshold(conf, correct, alpha=0.01) == float("inf")


def test_ltt_bounds_error_among_accepted_with_high_probability():
    rng = np.random.default_rng(1)
    alpha, delta, violations = 0.1, 0.1, 0
    trials = 100
    for _ in range(trials):
        conf, correct = _calibrated(2000, rng)
        lam = ltt_selective_threshold(conf, correct, alpha, delta)
        t_conf, t_correct = _calibrated(20000, rng)
        accepted = t_conf >= lam
        if accepted.any() and (~t_correct[accepted]).mean() > alpha:
            violations += 1
    assert violations / trials <= delta + 0.05


def test_aci_long_run_error_tracks_target_under_shift():
    rng = np.random.default_rng(2)
    target, gamma, T = 0.05, 0.01, 20000
    aci = AdaptiveAlpha(target=target, gamma=gamma)
    for t in range(T):
        scale = 1.0 + 2.0 * t / T  # the model gets steadily more miscalibrated
        p_err = float(np.clip(aci.alpha_t * scale, 0.0, 1.0))
        aci.update(float(rng.uniform() < p_err))
    bound = (max(target, 1 - target) + gamma) / (gamma * T)  # Gibbs & Candes, Prop. 4.1
    assert abs(np.mean(aci.history) - target) <= bound
