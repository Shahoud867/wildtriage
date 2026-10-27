"""SAM 3 / SAM 3.1 video adapter (https://github.com/facebookresearch/sam3, SAM License, gated weights).

Paper 1 (SA-FARI) benchmarks SAM 3 on camera-trap video. We reuse it as-is with text prompts
("animal" plus top-k species names) on frames sampled at 6 fps (SA-FARI's annotation rate).

The predictor API follows the repo's examples/sam3_video_predictor_example.ipynb:
  build_sam3_video_predictor() -> handle_request(start_session | reset_session | add_prompt | close_session)
  and handle_stream_request(propagate_in_video), which yields {"frame_index", "outputs"} with outputs holding
  out_obj_ids, out_probs, out_boxes_xywh and out_binary_masks.

Clips are cut into chunks of at most `max_chunk_s` seconds (JPEG folders at `fps`) to stay inside the 20 GB
VRAM budget. Object ids are not linked across chunks: each chunk's tracks get their own track_key range.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from wildtriage_ml.adapters.base import ModelInfo, TrackResult

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _frame_sort_key(path: Path):
    return (0, int(path.stem)) if path.stem.isdigit() else (1, path.name)


def split_frame_folder(frames: list[Path], chunk_len: int) -> list[list[Path]]:
    """Split an ordered frame list into consecutive chunks of at most `chunk_len` frames."""
    if chunk_len < 1:
        raise ValueError("chunk_len must be >= 1")
    return [frames[i : i + chunk_len] for i in range(0, len(frames), chunk_len)]


def encode_rle(mask) -> dict:
    """COCO RLE (counts as str, JSON-safe) of one binary mask, as the SA-Co/VEval evaluator expects."""
    import numpy as np
    from pycocotools import mask as mask_utils

    rle = mask_utils.encode(np.asfortranarray(np.asarray(mask, dtype=np.uint8)))
    return {"size": [int(v) for v in rle["size"]], "counts": rle["counts"].decode("ascii")}


def merge_frame_outputs(
    outputs_per_frame: dict[int, dict], frame_offset: int, key_offset: int, label: str | None,
    keep_masks: bool = False,
) -> list[TrackResult]:
    """Turn one chunk's per-frame predictor outputs into TrackResults (boxes normalised xywh).

    With keep_masks, each track's `extra["rle"]` maps absolute frame index -> RLE.
    """
    tracks: dict[int, TrackResult] = {}
    for frame_idx in sorted(outputs_per_frame):
        out = outputs_per_frame[frame_idx]
        ids = out.get("out_obj_ids", [])
        boxes = out.get("out_boxes_xywh", [])
        probs = out.get("out_probs", [0.0] * len(ids))
        masks = out.get("out_binary_masks") if keep_masks else None
        for i, (obj_id, box, prob) in enumerate(zip(ids, boxes, probs)):
            key = key_offset + int(obj_id)
            track = tracks.setdefault(key, TrackResult(track_key=key, frames=[], boxes=[], label=label))
            track.frames.append(frame_offset + int(frame_idx))
            track.boxes.append(tuple(float(v) for v in box))
            track.score = max(track.score, float(prob))
            if masks is not None:
                track.extra.setdefault("rle", {})[frame_offset + int(frame_idx)] = encode_rle(masks[i])
    return list(tracks.values())


class Sam3VideoTracker:
    # CUDA only: the official repo needs Triton/CUDA. Apple-silicon support is an unmerged pull request
    # (facebookresearch/sam3#627, Sep 2026), so SAM 3 stays on the lab PC or molab ("gpu-heavy" queue).
    info = ModelInfo("sam3", "facebook/sam3", "segment_track", "SAM License", False, vram_gb=8.0,  # [A] measure on the lab PC
                     devices=("cuda",))

    def __init__(self, checkpoint: str = "facebook/sam3", fps: float = 6.0, max_chunk_s: float = 30.0,
                 keep_masks: bool = False):
        self.checkpoint = checkpoint
        self.keep_masks = keep_masks  # store per-frame RLE masks in TrackResult.extra["rle"]
        self.fps = fps
        self.max_chunk_s = max_chunk_s
        self._predictor = None
        self.peak_vram_gb: float | None = None

    def load(self, device: str) -> None:
        # build_sam3_video_predictor downloads the gated checkpoint, so `hf auth login` must have run.
        # It uses all visible GPUs by default; pin to one so the VRAM budget is predictable.
        import torch
        from sam3.model_builder import build_sam3_video_predictor

        index = int(device.split(":", 1)[1]) if ":" in device else torch.cuda.current_device()
        # A local file is a full video state dict (e.g. fine-tuned weights merged by research/training);
        # anything else (the default "facebook/sam3") means the gated Hugging Face download.
        kwargs = {"checkpoint_path": self.checkpoint} if Path(self.checkpoint).is_file() else {}
        self._predictor = build_sam3_video_predictor(gpus_to_use=[index], **kwargs)

    def unload(self) -> None:
        if self._predictor is not None and hasattr(self._predictor, "shutdown"):
            self._predictor.shutdown()
        self._predictor = None

    def _chunks(self, video_path: Path, workdir: Path) -> list[tuple[str, int]]:
        """Return (JPEG folder, first frame index) per chunk, sampling an mp4 at self.fps."""
        chunk_len = max(1, int(self.fps * self.max_chunk_s))
        if video_path.is_dir():
            frames = sorted((p for p in video_path.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES), key=_frame_sort_key)
        else:
            frames = self._extract_frames(video_path, workdir / "frames")
        chunks = []
        for n, group in enumerate(split_frame_folder(frames, chunk_len)):
            folder = workdir / f"chunk_{n:03d}"
            folder.mkdir(parents=True)
            for i, src in enumerate(group):  # SAM 3 expects <frame_index>.jpg in a folder
                shutil.copyfile(src, folder / f"{i}.jpg")
            chunks.append((str(folder), n * chunk_len))
        return chunks

    def _extract_frames(self, video_path: Path, out_dir: Path) -> list[Path]:
        import cv2

        out_dir.mkdir(parents=True)
        cap = cv2.VideoCapture(str(video_path))
        native = cap.get(cv2.CAP_PROP_FPS) or self.fps
        step = max(1, round(native / self.fps))
        frames, i = [], 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                dest = out_dir / f"{len(frames):06d}.jpg"
                cv2.imwrite(str(dest), frame)
                frames.append(dest)
            i += 1
        cap.release()
        return frames

    def track(self, video_path: Path, prompts: list[str]) -> list[TrackResult]:
        import torch

        if self._predictor is None:
            raise RuntimeError("call load() first")
        results: list[TrackResult] = []
        torch.cuda.reset_peak_memory_stats()
        with tempfile.TemporaryDirectory() as tmp:
            for chunk_no, (folder, frame_offset) in enumerate(self._chunks(Path(video_path), Path(tmp))):
                session = self._predictor.handle_request(dict(type="start_session", resource_path=folder))["session_id"]
                try:
                    for p_no, prompt in enumerate(prompts):
                        # A text prompt needs a fresh session state ("reset_session" in the example notebook).
                        self._predictor.handle_request(dict(type="reset_session", session_id=session))
                        self._predictor.handle_request(
                            dict(type="add_prompt", session_id=session, frame_index=0, text=prompt)
                        )
                        outputs = {
                            r["frame_index"]: r["outputs"]
                            for r in self._predictor.handle_stream_request(
                                dict(type="propagate_in_video", session_id=session)
                            )
                        }
                        # 1000 ids per (chunk, prompt) keeps keys unique without a cross-chunk linking step.
                        results += merge_frame_outputs(
                            outputs, frame_offset, key_offset=(chunk_no * len(prompts) + p_no) * 1000, label=prompt,
                            keep_masks=self.keep_masks,
                        )
                finally:
                    self._predictor.handle_request(dict(type="close_session", session_id=session))
        self.peak_vram_gb = torch.cuda.max_memory_allocated() / 1024**3
        return results
