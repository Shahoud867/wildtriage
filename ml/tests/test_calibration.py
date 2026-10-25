import numpy as np

from wildtriage_ml.triage.calibration import (
    TemperatureScaler,
    expected_calibration_error,
    fit_temperature,
    softmax,
)


def _overconfident(n=20000, k=5, t_true=2.0, seed=0):
    rng = np.random.default_rng(seed)
    true_logits = rng.normal(0, 2.0, size=(n, k))
    p = softmax(true_logits)
    labels = np.array([rng.choice(k, p=row) for row in p])
    return true_logits * t_true, labels  # the "model" is overconfident by a factor t_true


def test_fit_temperature_recovers_known_temperature():
    logits, labels = _overconfident(t_true=2.0)
    assert abs(fit_temperature(logits, labels) - 2.0) < 0.1


def test_temperature_scaling_reduces_ece():
    logits, labels = _overconfident(t_true=2.5, seed=1)
    before = expected_calibration_error(softmax(logits), labels)
    after = expected_calibration_error(TemperatureScaler().fit(logits, labels).transform(logits), labels)
    assert after < before / 3


def test_ece_toy_cases():
    probs = np.tile([0.8, 0.2], (10, 1))
    labels = np.array([0] * 8 + [1] * 2)
    assert expected_calibration_error(probs, labels) < 1e-9
    probs = np.tile([1.0, 0.0], (10, 1))
    labels = np.array([0] * 5 + [1] * 5)
    assert abs(expected_calibration_error(probs, labels) - 0.5) < 1e-9
    assert abs(expected_calibration_error(probs, labels, adaptive=True) - 0.5) < 1e-9


def test_fit_temperature_empty_input():
    assert fit_temperature(np.zeros((0, 3)), np.zeros(0)) == 1.0
