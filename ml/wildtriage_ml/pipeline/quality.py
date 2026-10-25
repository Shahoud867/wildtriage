"""Cheap image-quality features used by triage (never auto-accept dark, blurred or IR-ambiguous media)."""

import numpy as np


def _as_float(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    return img.astype(np.float32) / 255.0 if img.dtype == np.uint8 else img.astype(np.float32)


def to_gray(img: np.ndarray) -> np.ndarray:
    img = _as_float(img)
    if img.ndim == 2:
        return img
    return img[..., :3] @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def chroma(img: np.ndarray) -> float:
    """Mean colour spread; IR-flash night frames are (near-)grayscale, so this is ~0 for them."""
    img = _as_float(img)
    if img.ndim == 2 or img.shape[-1] == 1:
        return 0.0
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    return float((np.abs(r - g).mean() + np.abs(g - b).mean()) / 2)


def is_night(img: np.ndarray, chroma_threshold: float = 0.02) -> bool:
    return chroma(img) < chroma_threshold


def brightness(img: np.ndarray) -> float:
    return float(to_gray(img).mean())


def blur_score(img: np.ndarray) -> float:
    """Variance of the 4-neighbour Laplacian; low values mean blur (motion or focus)."""
    g = to_gray(img)
    if min(g.shape) < 3:
        return 0.0
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(lap.var())


def quality_features(img: np.ndarray) -> dict:
    return {
        "chroma": chroma(img),
        "is_night": is_night(img),
        "brightness": brightness(img),
        "blur": blur_score(img),
    }
