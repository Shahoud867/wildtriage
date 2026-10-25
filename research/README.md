# Research workspace

Offline experiments that use the same `wildtriage_ml` package as the app. Plan and protocol: `docs/PROJECT_PLAN.md` §3 and §9.

| Experiment | Question | Data | Where it runs | Status |
|---|---|---|---|---|
| `exp1_triage.py` (**core contribution**) | How many selected human labels close the gap to the oracle? Does risk-controlled auto-accept stay ≤ α on future intervals? | StreamTrap-mini (LILA, chronological per-site intervals) | Embedding extraction on the **lab PC** or **T1000**; the policy simulation on **any CPU** (incl. the laptop) | Simulator runs on synthetic data; real data from W2 |
| `exp2_video_counts.py` | SA-FARI reproduction (SAM 3, MegaDetector + ByteTrack) + per-clip MaxN error | SA-FARI **test** only (SAM 3 may have trained on train) | **Lab PC** only | W2–W3 |
| `exp3_camouflage.py` (P1) | DepthSAM reproduction on COD10K; refinement of SA-FARI small/challenging masks | COD10K/CAMO/NC4K/CHAMELEON; SA-FARI test | **Lab PC** only | W5 |
| `exp4_aerial.py` (P1) | Point-F1 and count error of OWL/HerdNet | One LILA aerial dataset | Lab PC (OWL-D) or T1000 (OWL-C) | W6 |

## Rules (leakage and reproducibility)

- Site-disjoint dev/test splits. Chronological intervals within each site. Bursts (`seq_id`) never split.
- Calibration, thresholds and adapters use **only past intervals** and **only reviewed items**.
- Hyperparameters are tuned on dev sites. The test sites run **once**, from a tagged commit.
- Every run writes its config, git commit and metrics (as an `evaluation_runs` row or `results/<run>/metrics.json`).
- Manifests (`manifests/*.parquet`: path, sha256, site, timestamp, seq_id, split) are committed. Raw data is not.

## Sharing between machines

Only the lab PC holds raw images and videos. It exports compact artifacts that the other machines use:

- `embeddings/<dataset>/<site>.npz`: BioCLIP 2 crop embeddings (fp16), labels, timestamps, seq_id, quality flags.
- `predictions/<experiment>/*.parquet`: per-item model outputs.

Put them on the team drive. Exp 1 then runs on any machine with `numpy`.
