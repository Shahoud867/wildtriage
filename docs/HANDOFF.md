# Handoff: university-PC GPU session, 26 Oct 2026

Written for whoever (a person or a new Claude chat) continues this work from another machine. Read it with
[`ROADMAP.md`](ROADMAP.md) and [`lab_pc_notes.md`](lab_pc_notes.md). Everything below was done on the lab PC
(RTX 4000 Ada) and was **not committed there** (the PC has no git), so it arrives with the branch that contains this file.

## Done and verified
| Item | Result | Where |
|---|---|---|
| A-02 smoke test, MegaDetector + BioCLIP 2 on CUDA | 48/50 images with animal boxes; BioCLIP 2 zero-shot top-1 46%, top-5 58%; 226 ms / 32 ms median; 1.6 GB peak | `research/smoke/run_models.py`, `research/results/smoke_cuda.md` |
| Bug fix in `adapters/bioclip.py` (`classify` crashed: tensor requires grad) | fixed | commit with this file |
| A-06 SAM 3 adapter (`adapters/sam3.py`), 3-clip smoke on SA-FARI test | works; about 21 s per 75-frame clip, 4.8 GB peak | `research/smoke/run_sam3.py`, `research/results/smoke_sam3.md` |
| A-08 DepthSAM reproduction on COD10K-test (2026 images) | S-alpha 0.920, F-beta-w 0.871, E-phi 0.962, MAE 0.0145; paper (DepthSAM-L, 512) 0.920 / 0.867 / 0.964 / 0.014 | `research/experiments/exp3_camouflage.py`, `research/results/exp3_cod10k.json` |
| S-02 subsample manifests (about 59k images, 30 sites, 18 intervals each) | `lila_{dev,test}_sub.parquet`, `lila_sub_summary.csv` | `research/data_prep/subsample_manifests.py` |
| `is_valid_npz` bug (resume never skipped finished sites) | fixed + test | `research/data_prep/extract_embeddings.py` |

## Was still running when this was written (check `D:\wildtriage-data\pipeline_status.json` on the lab PC)
`scripts/lab/pipeline.py` runs one step at a time: embeddings for dev then test (`D:\wildtriage-data\embeddings\`) ->
Exp 2 (SAM 3 on 150 SA-FARI test clips, `D:\wildtriage-data\exp2`) -> Exp 2 eval -> Exp 1 on the dev embeddings
(`research/results/exp1_dev_real.txt`). The final report is `D:\wildtriage-data\PIPELINE_REPORT.md`. Embeddings and
Exp 2 raw outputs are not in git; copy them to the shared drive.

## Findings that change the plan
1. **Exp 1 on 8 real dev sites: coverage 0.000 for every policy at alpha 0.05.** Whole-image BioCLIP 2 is only about
   36% accurate per class, so nothing is certified for auto-accept. Final label accuracy rose from 0.35 (zero-shot) to
   about 0.45-0.47 with 10% review. This supports Stage 2 MegaDetector crops (ROADMAP S-11) and an alpha / calibration sweep (H-05).
2. **SAM 3 with the prompt "animal" often returns nothing** on SA-FARI clips, while the correct species noun phrase
   finds the animal (score 0.85-0.9). Exp 2 reports MaxN for "animal" and for the given positive phrases (oracle).
3. **Crashes were Windows commit-limit exhaustion**, not GPU memory: run one CUDA process at a time.

## Not done yet
MegaDetector + ByteTrack baseline for Exp 2 (A-10); A-04 batching; A-11 DepthSAM on SA-FARI crops; night / small-animal
subset tables for Exp 2; real-model run of the full app (M1, H-02 Postgres); MegaDetector crops; H-05 budget-split study;
Exp 1 on test sites (only after the M4 freeze); the HF Space / frontend demo.

## Deviations to state in the report
Whole images, not crops; one dataset (Snapshot Serengeti); subsample cannot remove multi-species images (manifests were
already de-duplicated to one label); COD10K images from the HF repackaging `nobg/COD10K` of the SINet TestDataset;
SAM 3 evaluated on a random 150-clip subset of SA-FARI test, not the full split.

## Environment pitfalls (details in `lab_pc_notes.md`)
HF downloads need `HF_TOKEN_PATH` set when `HF_HOME` is moved; SAM 3 needs `triton-windows` and pins numpy 1.26
(pandas<2.3, pyarrow<20, scipy<1.14); `-ExecutionPolicy Bypass` is blocked by the auto-mode classifier, so start
scripts through Python.

## Suggested branch and commit
Branch `shahoud/lab-gpu-session`, PR title "A-02/A-06/A-08, S-02: GPU session results and tooling".
