"""Exp 1: review-budgeted, risk-controlled triage with site adaptation, simulated over a site's
chronological stream of cached embeddings (docs/PROJECT_PLAN.md §3, §9).

At each interval t the deployed model M_t (trained on reviews from intervals < t) predicts every item.
The review budget is split in two:
  * a random *calibration* slice: unbiased labels for the conformal thresholds (actively selected
    items are not exchangeable with the stream, so they must not be used for the guarantee);
  * a *targeted* slice chosen by the selection strategy: labels for learning.
Remaining items are auto-accepted if their calibrated confidence clears the group threshold
(day/night "Mondrian" groups), otherwise left unverified. A random audit sample of accepted items is
reviewed too, to measure realised error (and drive ACI). Then the head and temperature are updated.

    python research/experiments/exp1_triage.py --synthetic            # smoke run, any CPU
    python research/experiments/exp1_triage.py --sites embeddings/lila/*.npz
"""

import argparse
import glob
from dataclasses import asdict, dataclass

import numpy as np

from wildtriage_ml.adaptation.heads import LinearHead, fit_linear_head, wise_interpolate
from wildtriage_ml.triage.calibration import TemperatureScaler, expected_calibration_error
from wildtriage_ml.triage.conformal import AdaptiveAlpha, crc_accept_threshold, ltt_selective_threshold
from wildtriage_ml.triage.selection import select_for_review

MIN_CALIBRATION = 30  # below this, nothing is auto-accepted (cold start)


@dataclass
class SiteStream:
    site: str
    embeddings: np.ndarray  # (n, d), L2-normalised
    labels: np.ndarray  # (n,) class indices
    interval: np.ndarray  # (n,) chronological interval index
    is_night: np.ndarray  # (n,) bool
    text_embeddings: np.ndarray  # (k, d) zero-shot class prompts
    logit_scale: float = 100.0


@dataclass
class PolicySpec:
    selection: str = "balanced_uncertainty"
    acceptance: str = "ltt"  # none | threshold_raw | crc | ltt | aci
    update: str = "linear_head_bsm_wise"  # none | linear_head | linear_head_bsm | linear_head_bsm_wise
    budget: float = 0.10
    calib_fraction: float = 0.3
    alpha: float = 0.05
    delta: float = 0.1
    mondrian: bool = True
    audit_rate: float = 0.02
    wise_alpha: float = 0.5
    raw_threshold: float = 0.9
    seed: int = 0


def per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    classes = np.unique(y_true)
    return float(np.mean([(y_pred[y_true == c] == c).mean() for c in classes])) if classes.size else float("nan")


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    f1s = []
    for c in np.unique(np.concatenate([y_true, y_pred])):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp, fn = np.sum((y_pred == c) & (y_true != c)), np.sum((y_pred != c) & (y_true == c))
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return float(np.mean(f1s)) if f1s else float("nan")


def simulate_site(stream: SiteStream, spec: PolicySpec) -> list[dict]:
    rng = np.random.default_rng(spec.seed)
    zs = LinearHead.from_zero_shot(stream.text_embeddings, stream.logit_scale)
    head, scaler = zs, TemperatureScaler()
    groups = ["day", "night"] if spec.mondrian else ["all"]
    aci = {g: AdaptiveAlpha(spec.alpha, gamma=0.02) for g in groups}
    # Calibration buffer: predictions *as made at the time* on randomly sampled reviewed items.
    cal_logits, cal_y, cal_group = [], [], []
    train_X, train_y = [], []
    records = []

    for t in np.unique(stream.interval):
        idx = np.flatnonzero(stream.interval == t)
        X, y = stream.embeddings[idx], stream.labels[idx]
        group = np.where(stream.is_night[idx], "night", "day") if spec.mondrian else np.full(idx.size, "all")
        logits = head.logits(X)
        probs = scaler.transform(logits)
        conf, pred = probs.max(axis=1), probs.argmax(axis=1)
        n = idx.size

        # 1) thresholds from past calibration data only
        thr = {}
        for g in groups:
            if spec.acceptance == "none":
                thr[g] = np.inf
            elif spec.acceptance == "threshold_raw":
                thr[g] = spec.raw_threshold
            else:
                m = np.array(cal_group) == g
                if m.sum() < MIN_CALIBRATION:
                    thr[g] = np.inf
                    continue
                p_cal = scaler.transform(np.array(cal_logits)[m])
                c_conf, c_ok = p_cal.max(axis=1), p_cal.argmax(axis=1) == np.array(cal_y)[m]
                a = aci[g].effective_alpha if spec.acceptance == "aci" else spec.alpha
                thr[g] = crc_accept_threshold(c_conf, c_ok, a) if spec.acceptance == "crc" else \
                    ltt_selective_threshold(c_conf, c_ok, a, spec.delta)

        # 2) spend the review budget: random calibration slice + targeted slice
        budget = int(np.ceil(spec.budget * n))
        n_cal = int(round(spec.calib_fraction * budget))
        calib = rng.choice(n, size=min(n_cal, n), replace=False)
        rest = np.setdiff1d(np.arange(n), calib)
        targeted = rest[select_for_review(spec.selection, budget - calib.size, probs[rest],
                                          embeddings=X[rest], seed=spec.seed + int(t))]
        reviewed = np.zeros(n, bool)
        reviewed[calib] = reviewed[targeted] = True

        # 3) auto-accept, then audit a random sample of the accepted items
        thr_vec = np.array([thr[g] for g in group])
        accepted = ~reviewed & (conf >= thr_vec)
        audit = accepted & (rng.uniform(size=n) < spec.audit_rate)
        for g in groups:
            m = audit & (group == g)
            if spec.acceptance == "aci" and m.any():
                aci[g].update(float(np.mean(pred[m] != y[m])))

        labelled = reviewed | audit
        final = np.where(labelled, y, pred)
        records.append({
            "site": stream.site, "interval": int(t), "n": n,
            "model_per_class_acc": per_class_accuracy(y, pred),  # M_t on its (unseen) interval
            "model_macro_f1": macro_f1(y, pred),
            "ece": expected_calibration_error(probs, y),
            "labels_used": int(labelled.sum()),
            "coverage": float(accepted.mean()),
            "accepted_error": float(np.mean(pred[accepted] != y[accepted])) if accepted.any() else float("nan"),
            "unverified": float((~labelled & ~accepted).mean()),
            "final_label_acc": float(np.mean(final == y)),
            "thresholds": {g: float(v) for g, v in thr.items()},
        })

        # 4) learn from this interval's labels for the next one
        cal_logits.extend(logits[calib]); cal_y.extend(y[calib]); cal_group.extend(group[calib])
        train_X.extend(X[labelled]); train_y.extend(y[labelled])
        if spec.update != "none" and len(train_y) >= 10:
            ft = fit_linear_head(np.array(train_X), np.array(train_y), init=zs,
                                 class_counts=np.bincount(train_y, minlength=zs.weight.shape[0]),
                                 balanced_softmax="bsm" in spec.update)
            head = wise_interpolate(zs, ft, spec.wise_alpha) if "wise" in spec.update else ft
        if len(cal_y) >= MIN_CALIBRATION:
            scaler.fit(np.array(cal_logits), np.array(cal_y))
    return records


def make_synthetic_site(seed: int = 0, n_intervals: int = 8, per_interval: int = 1000, k: int = 6, d: int = 32,
                        site: str = "synthetic", text_noise: float = 1.2, day_noise: float = 0.9,
                        night_noise: float = 1.4) -> SiteStream:
    """Classes are Gaussian clusters; the zero-shot text prompts are noisy versions of the class centres;
    the class mix drifts over time and night items are noisier (lower SNR, like IR frames)."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(k, d))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    text = centres + rng.normal(scale=text_noise / np.sqrt(d), size=(k, d))
    text /= np.linalg.norm(text, axis=1, keepdims=True)
    X, y, interval, night = [], [], [], []
    for t in range(n_intervals):
        prior = rng.dirichlet(np.ones(k) * 0.7)  # seasonal shift in which species appear
        yt = rng.choice(k, size=per_interval, p=prior)
        nt = rng.uniform(size=per_interval) < 0.35
        noise = np.where(nt, night_noise, day_noise)[:, None]  # total noise norm relative to unit-norm class centres
        xt = centres[yt] + rng.normal(size=(per_interval, d)) * noise / np.sqrt(d)
        X.append(xt / np.linalg.norm(xt, axis=1, keepdims=True))
        y.append(yt); interval.append(np.full(per_interval, t)); night.append(nt)
    return SiteStream(site, np.vstack(X), np.concatenate(y), np.concatenate(interval), np.concatenate(night),
                      text, logit_scale=30.0)


def load_site(path: str) -> SiteStream:
    """npz with: embeddings (n, d), labels (n,), interval (n,), is_night (n,), text_embeddings (k, d), logit_scale."""
    z = np.load(path, allow_pickle=False)
    return SiteStream(path, z["embeddings"].astype(np.float64), z["labels"], z["interval"], z["is_night"].astype(bool),
                      z["text_embeddings"].astype(np.float64), float(z["logit_scale"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--sites", nargs="*", default=[])
    args = parser.parse_args()
    streams = [make_synthetic_site(s, site=f"synthetic-{s}") for s in range(3)] if args.synthetic else \
        [load_site(p) for pattern in args.sites for p in glob.glob(pattern)]
    specs = {
        "zero-shot, no review": PolicySpec(acceptance="none", update="none", budget=0.0),
        "random review + recipe": PolicySpec(selection="random"),
        "least-confidence + recipe": PolicySpec(selection="least_confidence"),
        "ours (balanced + LTT)": PolicySpec(),
        "ours, no day/night split": PolicySpec(mondrian=False),
        "ours, alpha=0.10": PolicySpec(alpha=0.10),
        "ours (balanced + ACI)": PolicySpec(acceptance="aci"),
    }
    print(f"{'policy':28s} {'model acc':>9s} {'final acc':>9s} {'coverage':>8s} {'acc. err':>8s} {'labels':>7s}")
    for name, spec in specs.items():
        recs = [r for s in streams for r in simulate_site(s, spec)]
        late = [r for r in recs if r["interval"] >= 3]
        errs = [r["accepted_error"] for r in late if not np.isnan(r["accepted_error"])]
        err = float(np.mean(errs)) if errs else float("nan")
        print(f"{name:28s} {np.mean([r['model_per_class_acc'] for r in late]):9.3f} "
              f"{np.mean([r['final_label_acc'] for r in late]):9.3f} {np.mean([r['coverage'] for r in late]):8.3f} "
              f"{err:8.3f} {sum(r['labels_used'] for r in recs):7d}")
    print("\nSpec defaults:", asdict(PolicySpec()))


if __name__ == "__main__":
    main()
