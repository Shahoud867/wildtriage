"""BioCLIP 2 zero-shot species classifier (MIT, https://huggingface.co/imageomics/bioclip-2).

Reused as-is through OpenCLIP. Per-site linear heads (our code, wildtriage_ml.adaptation) can replace
the zero-shot text head once verified labels exist.
"""

from pathlib import Path

import numpy as np

from wildtriage_ml.adapters.base import BoxDetection, ClassScores, ModelInfo
from wildtriage_ml.pipeline.aggregate import expand_box

HF_ID = "hf-hub:imageomics/bioclip-2"


class BioCLIP2Classifier:
    info = ModelInfo("bioclip-2", "hf", "classify", "MIT", True, vram_gb=2.0)

    def __init__(self, prompt_template: str = "a photo of {label}.", box_expand: float = 0.5):
        self.prompt_template = prompt_template
        self.box_expand = box_expand  # StreamTrap enlarges MegaDetector boxes by 50%
        self._model = self._preprocess = self._tokenizer = None
        self._device = "cpu"
        self._text_cache: dict[tuple[str, ...], "object"] = {}

    def load(self, device: str) -> None:
        import open_clip

        from wildtriage_ml.device import pick_dtype

        self._device = device
        model, _, preprocess = open_clip.create_model_and_transforms(HF_ID)
        # bf16 on the lab PC (Ada), fp16 on the T1000 (Turing), fp32 on the Mac (MPS) and CPU.
        self._model = model.to(device=device, dtype=pick_dtype(device)).eval()
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(HF_ID)

    def unload(self) -> None:
        self._model = self._preprocess = self._tokenizer = None
        self._text_cache.clear()

    def _text_features(self, labels: list[str]):
        import torch

        key = tuple(labels)
        if key not in self._text_cache:
            tokens = self._tokenizer([self.prompt_template.format(label=lbl) for lbl in labels]).to(self._device)
            with torch.no_grad():
                feats = self._model.encode_text(tokens)
            self._text_cache[key] = feats / feats.norm(dim=-1, keepdim=True)
        return self._text_cache[key]

    def image_embedding(self, image_path: Path, box: BoxDetection | None) -> np.ndarray:
        """L2-normalised image embedding; cached per crop by the research pipeline."""
        import torch
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        if box is not None:
            x, y, w, h = expand_box((box.x, box.y, box.w, box.h), self.box_expand)
            W, H = img.size
            img = img.crop((int(x * W), int(y * H), int((x + w) * W), int((y + h) * H)))
        dtype = next(self._model.parameters()).dtype
        pixels = self._preprocess(img).unsqueeze(0).to(device=self._device, dtype=dtype)
        with torch.no_grad():
            feats = self._model.encode_image(pixels)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats[0].float().cpu().numpy()

    def classify(self, image_path: Path, box: BoxDetection | None, labels: list[str]) -> ClassScores:
        import torch

        emb = torch.from_numpy(self.image_embedding(image_path, box)).to(self._device)
        text = self._text_features(labels).float()
        with torch.no_grad():  # logit_scale is a Parameter, so the product would otherwise require grad
            scale = self._model.logit_scale.exp().float()
            logits = (scale * emb @ text.T).cpu().numpy()
        return ClassScores(labels=list(labels), logits=logits)
