import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from exp1_triage import PolicySpec, make_synthetic_site, simulate_site  # noqa: E402


def test_simulation_runs_and_respects_budget():
    stream = make_synthetic_site(0, n_intervals=4, per_interval=300)
    spec = PolicySpec(budget=0.1, audit_rate=0.0)
    recs = simulate_site(stream, spec)
    assert [r["interval"] for r in recs] == [0, 1, 2, 3]
    for r in recs:
        assert r["labels_used"] <= np.ceil(0.1 * r["n"])
        assert 0.0 <= r["coverage"] <= 1.0
        assert r["final_label_acc"] >= r["model_per_class_acc"] - 0.1  # reviews can only fix labels


def test_no_auto_accept_before_enough_calibration_data():
    recs = simulate_site(make_synthetic_site(1, n_intervals=2, per_interval=200), PolicySpec())
    assert recs[0]["coverage"] == 0.0  # cold start: nothing certified yet


def test_risk_control_keeps_accepted_error_near_alpha():
    errs = []
    for seed in range(3):
        for r in simulate_site(make_synthetic_site(seed), PolicySpec(alpha=0.10)):
            if not np.isnan(r["accepted_error"]):
                errs.append(r["accepted_error"])
    assert errs, "expected some auto-accepted items"
    assert np.mean(errs) <= 0.10 + 0.02
