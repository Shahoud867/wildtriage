"""Side-by-side table: public SAM 3 vs the fine-tuned model on the same SA-FARI test clips, next to the paper.

    python research/training/compare.py --baseline D:/wildtriage-data/exp2 --finetuned D:/wildtriage-data/exp2_stage_a

Both directories come from `exp2_video_counts.py run` + `eval` with the same --seed / --n-clips. Metrics are the
official SA-Co/VEval mask metrics, scaled to 0-100 like the paper's Table 3.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# (cgF1, pHOTA, DetA, AssA, TETA), Table 3 of the SA-FARI paper (SAM 3 rows). The paper evaluates its own
# checkpoints on the full test split; ours is a 150-clip subset of the public checkpoint, so only gains compare.
PAPER = {
    "paper row 3: SAM 3": (14.0, 48.5, 28.4, 83.4, 39.6),
    "paper row 4: SAM 3 (SA-FARI in training mix)": (39.0, 63.1, 47.9, 83.9, 52.1),
    "paper row 5: SAM 3 FT (SA-FARI)": (46.9, 68.1, 55.4, 84.6, 58.7),
}
KEYS = ("video_mask_demo_cgf1_micro_50_95", "video_mask_all_phrase_HOTA", "video_mask_all_phrase_DetA",
        "video_mask_all_phrase_AssA", "video_mask_teta")


def veval_row(directory: Path) -> tuple[float, ...]:
    res = json.loads((directory / "veval_res.json").read_text(encoding="utf-8"))["dataset_results"]
    # phrase_* metrics are fractions, TETA is already 0-100
    return tuple(res[k] if k == "video_mask_teta" else 100 * res[k] for k in KEYS)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--finetuned", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None, help="also write the markdown here")
    args = ap.parse_args()

    base, tuned = veval_row(args.baseline), veval_row(args.finetuned)
    rows = [("public facebook/sam3 (ours, 150 clips)", base), ("fine-tuned, stage A (ours, 150 clips)", tuned),
            ("gain (ours)", tuple(t - b for t, b in zip(tuned, base))), *PAPER.items(),
            ("gain, paper row 4 -> 5", tuple(b - a for a, b in zip(PAPER["paper row 4: SAM 3 (SA-FARI in training mix)"],
                                                               PAPER["paper row 5: SAM 3 FT (SA-FARI)"])))]
    lines = ["| | cgF1 | pHOTA | DetA | AssA | TETA |", "|---|---|---|---|---|---|"]
    lines += [f"| {name} | " + " | ".join(f"{v:+.1f}" if name.startswith("gain") else f"{v:.1f}" for v in vals) + " |"
              for name, vals in rows]

    counts = []
    for tag, d in (("public", args.baseline), ("fine-tuned", args.finetuned)):
        s = json.loads((d / "exp2_summary.json").read_text(encoding="utf-8"))
        for prompt in ("maxn_given_phrase", "maxn_animal_prompt"):
            c = s[prompt]
            counts.append(f"| {tag} | {prompt} | {c['mae']:.3f} | {100 * c['exact_match']:.1f}% | {c['mean_signed_error']:+.2f} |")
    lines += ["", "| model | MaxN prompt | MAE | exact | mean signed err |", "|---|---|---|---|---|", *counts]

    text = "\n".join(lines)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
