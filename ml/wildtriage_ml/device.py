"""Device and precision selection for the machines the team uses.

- Lab PC (RTX 4000 Ada) and the T1000 PC: CUDA.
- MacBook Pro (M1 Pro): Apple's MPS backend.
- Laptop, CI: CPU.

Order: an explicit choice (argument or the WT_DEVICE environment variable), else CUDA, else MPS, else CPU.
Works without PyTorch installed (stub models on the laptop and in CI): "auto" then means CPU.
"""

import os

DEVICES = ("auto", "cuda", "mps", "cpu")


def _torch():
    try:
        import torch
    except ImportError:
        return None
    return torch


def _mps_available(torch) -> bool:
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    return bool(mps is not None and mps.is_available())


def pick_device(preferred: str | None = None) -> str:
    """Resolve "auto" (or None) to cuda, mps or cpu. An explicit device that is not available is an error,
    so a misconfigured worker fails at start-up instead of silently running on the CPU."""
    choice = (preferred or "").strip().lower()
    if choice in ("", "auto"):
        choice = os.environ.get("WT_DEVICE", "auto").strip().lower() or "auto"
    kind = choice.split(":", 1)[0]  # "cuda:1" selects a GPU index
    if kind not in DEVICES or (":" in choice and kind != "cuda"):
        raise ValueError(f"unknown device {choice!r}; expected one of {DEVICES} (or cuda:<index>)")

    torch = _torch()
    if kind == "auto":
        if torch is None:
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        if _mps_available(torch):
            return "mps"
        return "cpu"
    if kind == "cpu":
        return "cpu"
    if torch is None:
        raise RuntimeError(f"device {choice!r} requested but PyTorch is not installed")
    if kind == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no CUDA GPU is available on this machine")
    if kind == "mps" and not _mps_available(torch):
        raise RuntimeError("MPS requested but Apple's MPS backend is not available (needs Apple silicon and macOS 14+)")
    return choice


def device_kind(device: str) -> str:
    return device.split(":", 1)[0]


def pick_dtype(device: str):
    """Precision per device.

    - CUDA: bf16 where supported (Ada, lab PC), else fp16 (Turing, T1000).
    - MPS: fp32. M1–M3 have no hardware bf16 (emulated and slower than fp32), and fp16 has had
      correctness bugs on some macOS versions (docs/AUDIT_2026-10-24.md §5.1).
    - CPU: fp32.
    """
    import torch

    if device_kind(device) == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def empty_cache(device: str) -> None:
    """Return cached allocator memory to the system after a model is unloaded."""
    torch = _torch()
    if torch is None:
        return
    kind = device_kind(device)
    if kind == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif kind == "mps":
        mps = getattr(torch, "mps", None)
        if mps is not None and hasattr(mps, "empty_cache"):
            mps.empty_cache()
