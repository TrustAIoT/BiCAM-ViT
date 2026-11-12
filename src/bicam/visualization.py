"""Small visualization helpers for signed attribution maps."""

from __future__ import annotations

import numpy as np


def _seismic(values: np.ndarray) -> np.ndarray:
    """Small NumPy implementation of the blue-white-red seismic colormap."""

    anchors = np.array(
        [
            [0.0, 0.0, 0.3],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    scaled = np.clip(values, 0.0, 1.0) * (len(anchors) - 1)
    lower = np.floor(scaled).astype(np.int64)
    upper = np.minimum(lower + 1, len(anchors) - 1)
    weight = (scaled - lower)[..., None]
    return anchors[lower] * (1.0 - weight) + anchors[upper] * weight


def paper_overlay(
    image: np.ndarray,
    attribution: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Recreate the signed seismic overlay used in the experiment notebook."""

    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("image must have shape [H, W, 3]")
    if attribution.shape != image.shape[:2]:
        raise ValueError("attribution and image spatial dimensions must match")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")

    base = image.astype(np.float32)
    if base.max() > 1.0:
        base = base / 255.0
    signed = attribution.astype(np.float32)
    signed = signed / max(float(np.abs(signed).max()), 1e-8)
    normalized = (signed + 1.0) / 2.0
    color = _seismic(normalized)
    return np.clip((1.0 - alpha) * base + alpha * color, 0.0, 1.0)


def signed_overlay(
    image: np.ndarray,
    attribution: np.ndarray,
    alpha: float = 0.55,
) -> np.ndarray:
    """Overlay supportive evidence in red and suppressive evidence in blue."""

    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("image must have shape [H, W, 3]")
    if attribution.shape != image.shape[:2]:
        raise ValueError("attribution and image spatial dimensions must match")

    base = image.astype(np.float32)
    if base.max() > 1.0:
        base = base / 255.0
    signed = attribution.astype(np.float32)
    max_abs = max(float(np.abs(signed).max()), 1e-8)
    signed = signed / max_abs

    color = np.zeros_like(base)
    color[..., 0] = np.clip(signed, 0.0, 1.0)
    color[..., 2] = np.clip(-signed, 0.0, 1.0)
    strength = np.abs(signed)[..., None] * alpha
    return np.clip(base * (1.0 - strength) + color * strength, 0.0, 1.0)
