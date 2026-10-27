# SA-FARI fine-tuning of SAM 3 (paper Table 3, rows 4 -> 5)

What the SA-FARI paper (CVPR 2026) does and what is available:

| Item | Status |
|---|---|
| Dataset (train 10,776 videos, test 833) | public, `facebook/SA-FARI` (gated) + `cxl-public-camera-trap` bucket |
| Fine-tuned SAM 3 weights ("SAM 3 FT (SA-FARI)") | **not released** (nothing on the dataset card, the project page or the SAM 3 repo) |
| Training recipe (LR, epochs, batch, frozen parts) | **not in the paper** (its "training details" reference is a broken `??`) |
| SAM 3 fine-tuning code | public (`third_party/sam3`, `README_TRAIN.md`), but only image configs; no video-training config |

So the fine-tune is a *re-creation*, not a reproduction of the authors' run. The comparison that stays meaningful is the
**gain** from fine-tuning on SA-FARI, measured against the public `facebook/sam3` checkpoint on the same test clips.

## Baseline caveat (read before comparing to the paper)

The public `facebook/sam3` already scores cgF1 40.3 / pHOTA 66.7 / TETA 59.4 on our 150 SA-FARI test clips
(`D:/wildtriage-data/exp2/veval_res.json`), close to the paper's *trained* rows (row 4: 39.0 / 63.1 / 52.1; row 5:
46.9 / 68.1 / 58.7), not its base row 3 (14.0 / 48.5 / 39.6). SA-FARI's masks came from the SAM 3 data engine, so the
public checkpoint has probably seen similar data. Row 3 is therefore not reproducible with released weights, and
headroom for fine-tuning is the row 4 -> 5 gap (about +8 cgF1, +5 pHOTA). This is an inference from the scores, not a
statement by the authors.

## Pipeline

```
prepare.py        subset of SA-FARI *train* (location-disjoint train/val), frame download     -> D:/wildtriage-data/sa_fari_ft
launch.py         SAM 3's own Trainer, single GPU, Hydra config configs/stage_a.yaml          -> D:/wildtriage-data/runs/<run>
merge_checkpoint  fine-tuned detector weights -> full video checkpoint (detector.* keys)
exp2_video_counts run --checkpoint <merged.pt>   then   eval        (same 150 clips, official VEval)
```

Stage A freezes both backbones (vision trunk 446 M, text encoder 353 M) and trains the neck, detector transformer, scoring
and mask heads (about 40 M parameters). The tracker is not trained. Stage B (unfreeze the last trunk blocks) only if A
shows a gain.

Deviations from the paper to state in the report: detector-only fine-tuning on frames (no video-level tracker training);
subset of the training split; the recipe is ours; evaluated on a random 150-clip subset of the test split; one
checkpoint (`facebook/sam3`), not the authors' pre-SA-FARI base.

SA-FARI **test** videos are never used for training or model selection. Train/val are split by camera location.

## Running it (Windows lab PC)

```
python research/training/prepare.py --n-train 2000 --n-val 150 --out D:/wildtriage-data/sa_fari_ft      # ~15 min, 63k frames
python research/training/supervise.py --retries 5 --env HF_HOME=... --env TEMP=... --env HF_TOKEN_PATH=... -- \
    --config stage_a --out D:/wildtriage-data/runs/stage_a --data D:/wildtriage-data/sa_fari_ft
python research/training/merge_checkpoint.py --finetuned D:/wildtriage-data/runs/stage_a/checkpoints/checkpoint.pt \
    --out D:/wildtriage-data/runs/stage_a/sam3_video_stage_a.pt
python research/experiments/exp2_video_counts.py run --checkpoint D:/wildtriage-data/runs/stage_a/sam3_video_stage_a.pt \
    --out D:/wildtriage-data/exp2_stage_a ; ... eval --out D:/wildtriage-data/exp2_stage_a
python research/training/compare.py --baseline D:/wildtriage-data/exp2 --finetuned D:/wildtriage-data/exp2_stage_a
```

Measured on the RTX 4000 Ada (20 GB): 40.5 M trainable / 800 M frozen parameters, 15-17 GB peak at 4 frames per
micro-batch, GPU 82-100% busy, about 5.4 s per optimizer step of 8 frames (about 1.4 frames/s). Stage A = 6 epochs x 4,000
frames = 3,000 steps, about 4.5 h. Re-running the same command resumes from `checkpoints/checkpoint.pt`.

Windows-specific changes needed to run SAM 3's trainer (none touch `third_party/sam3`):
- gloo instead of NCCL, one process; `Sam3LossWrapper.normalization: local` because gloo `all_reduce` on a CUDA tensor
  crashes the process (access violation) - identical result with one process.
- frozen modules run under `torch.no_grad()` (SAM 3's fused MLP kernel in the vision trunk refuses to run with autograd on).
- DDP `static_graph: true`: some neck / mask-head parameters take no gradient, and gradient accumulation breaks without it.
- the collator must be `collate_fn_api_with_chunking` when accumulating; the loader batch is micro-batch x accumulation.
- the template's `FilterEmptyTargets` is removed so negative phrases and empty frames are trained on.
- a detached process (Win32_Process.Create) does not inherit your shell's environment: pass `--env` (HF_HOME, TEMP, token).

Not yet done: a validation-loss or val-clip metric during training (val.json is built but unused; model selection is
therefore fixed in advance to the last checkpoint, and the test clips are scored once); confidence intervals on the VEval
metrics (VEval gives dataset-level numbers only); Stage B.
