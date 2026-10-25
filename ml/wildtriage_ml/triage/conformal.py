"""Risk-controlled auto-acceptance thresholds.

- `crc_accept_threshold`: conformal risk control (Angelopoulos et al., ICLR 2024). Bounds the expected
  share of *all* items that are auto-accepted and wrong. Marginal guarantee under exchangeability.
- `ltt_selective_threshold`: Learn-then-Test (Angelopoulos et al., 2021) with fixed-sequence testing.
  With probability >= 1 - delta over the calibration set, bounds the error rate *among accepted* items.
- `AdaptiveAlpha`: adaptive conformal inference (Gibbs & Candes, NeurIPS 2021). Adjusts the working
  alpha online from audited outcomes, because temporal shift at a camera site breaks exchangeability.
"""

from dataclasses import dataclass, field
from math import exp, lgamma, log, log1p

import numpy as np


def crc_accept_threshold(conf: np.ndarray, correct: np.ndarray, alpha: float) -> float:
    """Smallest lambda with (n * R(lambda) + 1) / (n + 1) <= alpha, where R(lambda) is the fraction
    of calibration items with conf >= lambda that are wrong. Returns inf to mean 'accept nothing'."""
    conf = np.asarray(conf, dtype=np.float64)
    wrong = ~np.asarray(correct, dtype=bool)
    n = conf.size
    if n == 0:
        return float("inf")
    order = np.argsort(conf, kind="stable")
    c_sorted, w_sorted = conf[order], wrong[order]
    wrong_at_or_above = np.cumsum(w_sorted[::-1])[::-1]
    for i in range(n):
        if i > 0 and c_sorted[i] == c_sorted[i - 1]:
            continue
        if (wrong_at_or_above[i] + 1) / (n + 1) <= alpha:
            return float(c_sorted[i])
    return float("inf")


def binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p), computed in log space."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    logs = [lgamma(n + 1) - lgamma(i + 1) - lgamma(n - i + 1) + i * log(p) + (n - i) * log1p(-p) for i in range(k + 1)]
    m = max(logs)
    return min(1.0, exp(m) * sum(exp(v - m) for v in logs))


def ltt_selective_threshold(
    conf: np.ndarray,
    correct: np.ndarray,
    alpha: float,
    delta: float = 0.1,
    grid_size: int = 200,
    min_accept: int = 10,
) -> float:
    """Test thresholds on a fixed grid from strict to lenient. H0(lambda): error among accepted > alpha.
    Reject with the exact binomial p-value; stop at the first non-rejection (fixed-sequence testing)."""
    conf = np.asarray(conf, dtype=np.float64)
    wrong = ~np.asarray(correct, dtype=bool)
    best = float("inf")
    for lam in np.linspace(1.0, 0.0, grid_size + 1):
        accepted = conf >= lam
        m = int(accepted.sum())
        if m < min_accept:
            continue  # too few accepted items to test yet; the sequence starts once m >= min_accept
        k = int(wrong[accepted].sum())
        if binom_cdf(k, m, alpha) <= delta:
            best = float(lam)
        else:
            break
    return best


@dataclass
class AdaptiveAlpha:
    """alpha_{t+1} = alpha_t + gamma * (target - err_t). err_t is the realised error rate for step t,
    measured on a random audit sample of auto-accepted items. Not clipped (Gibbs & Candes): alpha_t <= 0
    means 'accept nothing', alpha_t >= 1 means 'accept everything'."""

    target: float
    gamma: float = 0.005
    alpha_t: float = field(default=float("nan"))
    history: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if np.isnan(self.alpha_t):
            self.alpha_t = self.target

    def update(self, err: float) -> float:
        self.history.append(err)
        self.alpha_t = self.alpha_t + self.gamma * (self.target - err)
        return self.alpha_t

    @property
    def effective_alpha(self) -> float:
        return float(np.clip(self.alpha_t, 0.0, 1.0))
