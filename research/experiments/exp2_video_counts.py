"""Exp 2 (lab PC only): reproduce SA-FARI benchmark rows and measure per-clip counting error.

W2-W3 steps:
  1. Data: SA-FARI *test* annotations from HF `facebook/SA-FARI` (gated, CC-BY-NC-4.0) and 6 fps JPEG
     frames from the `cxl-public-camera-trap` bucket. Never evaluate SAM 3 on SA-FARI train: the
     paper's "SAM 3 (SA-FARI)" variant was trained with it, and the public checkpoint may have been too.
  2. SAM 3 / 3.1: prompts "animal" (Table 4: IDF1, HOTA, DetA, AssA) and species names
     (Table 3: cgF1, pHOTA, TETA) with the offline evaluator in sam3/scripts/eval/veval.
  3. Baseline: MegaDetector per frame + ByteTrack (the `supervision` package) -> HOTA/IDF1 via TrackEval.
  4. Counting: per-clip MaxN from predicted tracks vs ground-truth masklets (wildtriage_ml.pipeline.counting.max_n);
     report MAE, RMSE and exact-match % overall and on the small / night / multi-animal / challenging subsets.
  5. Log peak VRAM and seconds per clip (RTX 4000 Ada, bf16) for the efficiency table.

Report differences from the paper's numbers with the protocol difference that explains them; do not tune toward them.
"""


def main() -> None:
    raise NotImplementedError("W2-W3: see module docstring")


if __name__ == "__main__":
    main()
