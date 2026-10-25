"""Exp 4 (P1): aerial counting with MegaDetector-Overhead (OWL) / HerdNet on one public dataset.

W6 steps:
  1. Pick one dataset with point or box labels (LILA Izembek Lagoon Waterfowl or NOAA Arctic Seals; Delplanque
     et al. if its server and license check out). Record the license (most are NC / unclear: research only).
  2. Run OWL-C (light enough for the T1000) and OWL-D (lab PC; 3.5 GB checkpoint) with tiled inference.
  3. Metrics: point precision/recall/F1 within a pixel radius; count MAE/RMSE per image; tile-size ablation;
     images per second and peak VRAM.
Cut to P2 if it is not working by Wednesday of W6 (scope ladder).
"""


def main() -> None:
    raise NotImplementedError("W6 (P1): see module docstring")


if __name__ == "__main__":
    main()
