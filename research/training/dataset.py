"""Sam3ImageDataset whose "epoch" is a fixed-size window over a seeded shuffle of all frames.

SAM 3's Trainer checkpoints once per epoch and resumes at the next one, so a full pass over tens of thousands of
frames (hours on one GPU) would lose everything on a crash. With windows, every epoch is short and sees new
frames, and resuming at epoch k selects exactly the same window it would have without the interruption.
"""

from __future__ import annotations

import torch
from sam3.train.data.sam3_image_dataset import Sam3ImageDataset


class WindowedSam3ImageDataset(Sam3ImageDataset):
    def __init__(self, *args, epoch_frames: int, shuffle_seed: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        generator = torch.Generator().manual_seed(shuffle_seed)
        self._order = self.ids[torch.randperm(len(self.ids), generator=generator)]
        self.epoch_frames = min(epoch_frames, len(self._order))
        self._select(0)

    def _select(self, epoch: int) -> None:
        start = (epoch * self.epoch_frames) % len(self._order)
        self.ids = self._order[(start + torch.arange(self.epoch_frames)) % len(self._order)]
        self.repeat_factors = torch.ones(len(self.ids), dtype=torch.float32)

    def set_epoch(self, epoch: int) -> None:
        self._select(int(epoch))
