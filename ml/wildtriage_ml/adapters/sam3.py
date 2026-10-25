"""SAM 3 / SAM 3.1 video adapter (https://github.com/facebookresearch/sam3, SAM License, gated weights).

Paper 1 (SA-FARI) benchmarks SAM 3 on camera-trap video. We reuse it as-is with text prompts
("animal" plus top-k species names) on frames sampled at 6 fps (SA-FARI's annotation rate).

To implement in W3, following the repo's video notebook:
  predictor = build_sam3_video_predictor()          # verified name in the README
  predictor.handle_request(start_session request)   # with the clip's frames
  predictor.handle_request(add_prompt request)      # text prompt + frame index
  then propagate through the clip and read response["outputs"] (masks, boxes, scores, object ids).
Chunk clips to <= 30 s and run in bf16 to stay inside the 20 GB VRAM budget; log peak memory.
"""

from pathlib import Path

from wildtriage_ml.adapters.base import ModelInfo, TrackResult


class Sam3VideoTracker:
    # CUDA only: the official repo needs Triton/CUDA. Apple-silicon support is an unmerged pull request
    # (facebookresearch/sam3#627, Sep 2026), so SAM 3 stays on the lab PC or molab ("gpu-heavy" queue).
    info = ModelInfo("sam3", "facebook/sam3", "segment_track", "SAM License", False, vram_gb=8.0,  # [A] measure in W1
                     devices=("cuda",))

    def __init__(self, checkpoint: str = "facebook/sam3", fps: float = 6.0, max_chunk_s: float = 30.0):
        self.checkpoint = checkpoint
        self.fps = fps
        self.max_chunk_s = max_chunk_s
        self._predictor = None

    def load(self, device: str) -> None:
        raise NotImplementedError("W3: build the SAM 3 video predictor (see module docstring)")

    def unload(self) -> None:
        self._predictor = None

    def track(self, video_path: Path, prompts: list[str]) -> list[TrackResult]:
        raise NotImplementedError("W3: run SAM 3 per chunk and merge object ids into TrackResults")
