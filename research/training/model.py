"""SAM 3 image model with selected modules frozen, for fine-tuning the detector on SA-FARI.

Stage A freezes both backbones (vision trunk 446 M, text encoder 353 M) and trains the neck, detector
transformer, scoring and mask heads (about 40 M parameters), which fits a 20 GB card with room for batching.
Frozen modules need no gradients, so their activations are not stored either.
"""

from __future__ import annotations

import functools
import logging

import torch

STAGE_A_FROZEN = ("backbone.vision_backbone.trunk", "backbone.language_backbone")
# Present in the model but not in the loss (we train with loss_fn_semantic_seg: null): DDP would otherwise
# report them as parameters that "did not receive grad" and break gradient accumulation.
UNUSED_HEADS = ("segmentation_head.semantic_seg_head",)


def _run_without_grad(module: torch.nn.Module) -> None:
    """Make a frozen module's forward run under no_grad.

    Besides saving memory, this is required for the vision trunk: its MLP uses SAM 3's fused inference kernel
    (sam3/perflib/fused.py), which raises if autograd is on. Unfreezing trunk blocks (a Stage B) therefore also
    needs that kernel replaced by the plain `act(fc1(x))` path in sam3/model/vitdet.py.
    """
    forward = module.forward

    @functools.wraps(forward)
    def no_grad_forward(*args, **kwargs):
        with torch.no_grad():
            return forward(*args, **kwargs)

    module.forward = no_grad_forward


def build_sam3_for_finetune(freeze=STAGE_A_FROZEN, **kwargs):
    from sam3.model_builder import build_sam3_image_model

    prefixes = tuple(freeze)
    model = build_sam3_image_model(**kwargs)
    trainable = frozen = 0
    for name, p in model.named_parameters():
        p.requires_grad = not name.startswith(prefixes + UNUSED_HEADS)
        if p.requires_grad:
            trainable += p.numel()
        else:
            frozen += p.numel()
    for prefix in prefixes:
        _run_without_grad(model.get_submodule(prefix))
    logging.info("SAM 3 fine-tune: %.1f M trainable, %.1f M frozen (%s)", trainable / 1e6, frozen / 1e6, prefixes)
    return model
