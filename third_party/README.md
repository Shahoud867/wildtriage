# third_party/

Pinned clones of upstream code we reuse, kept out of git (see `.gitignore`). Record the commit you used.

| Repo | Used for | License | Notes |
|---|---|---|---|
| https://github.com/facebookresearch/sam3 | Paper 1 video expert + SA-FARI evaluator (`scripts/eval/veval`) | SAM License | Gated weights: request access to `facebook/sam3` / `facebook/sam3.1` on Hugging Face |
| https://github.com/hjy0518/DepthSAM | Paper 2 camouflage expert (P1) | Apache-2.0 code; DA-V2-Large weights CC-BY-NC-4.0 | Own virtual environment; weights via its README links |
| https://github.com/microsoft/MegaDetector-Overhead | Aerial counting (P1) | MIT code; CC BY-NC-SA 4.0 weights | Uses `uv sync` |

```
git clone https://github.com/facebookresearch/sam3.git third_party/sam3
git -C third_party/sam3 rev-parse HEAD   # write the commit into docs/model_cards/sam3.md
```
