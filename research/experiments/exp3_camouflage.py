"""Exp 3 (P1, lab PC only): DepthSAM (Paper 2) reproduction and camera-trap refinement.

W5 steps:
  1. Clone https://github.com/hjy0518/DepthSAM into third_party/ in its own venv; download the released
     weights (Google Drive/Baidu links in its README). Note: Depth-Anything-V2-Large is CC-BY-NC-4.0.
  2. Reproduce COD10K-test S-alpha, F-beta^w, E-phi, MAE (py-sod-metrics); also report the animal-only subset.
     Sanity check against the independent Jittor port's reported MAE values.
  3. Refinement (our adaptation, experimental): for SA-FARI test detections in the small / challenging
     subsets, run DepthSAM on a 50%-enlarged crop, paste the mask back, and compare IoU/Dice against
     SAM 3's mask using SA-FARI ground truth.
"""


def main() -> None:
    raise NotImplementedError("W5 (P1): see module docstring")


if __name__ == "__main__":
    main()
