"""Image loading/saving helpers that preserve the original alpha channel."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_rgb(path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Load an image as ``(H, W, 3)`` float RGB in ``[0, 1]``.

    Returns ``(rgb, alpha)`` where ``alpha`` is an ``(H, W)`` float array in
    ``[0, 1]`` if the source had transparency, otherwise ``None``.
    """
    img = Image.open(path)
    alpha = None
    if img.mode in ("RGBA", "LA", "PA") or (
        img.mode == "P" and "transparency" in img.info
    ):
        rgba = img.convert("RGBA")
        arr = np.asarray(rgba, dtype=np.float64) / 255.0
        return arr[..., :3], arr[..., 3]
    rgb = np.asarray(img.convert("RGB"), dtype=np.float64) / 255.0
    return rgb, alpha


def save_rgb(path: str | Path, rgb: np.ndarray,
             alpha: np.ndarray | None = None) -> None:
    """Save an ``(H, W, 3)`` float RGB array, re-attaching ``alpha`` if given."""
    data = np.clip(rgb, 0.0, 1.0)
    rgb8 = np.rint(data * 255.0).astype(np.uint8)
    if alpha is not None:
        a8 = np.rint(np.clip(alpha, 0.0, 1.0) * 255.0).astype(np.uint8)
        out = np.dstack([rgb8, a8])
        Image.fromarray(out, mode="RGBA").save(path)
    else:
        Image.fromarray(rgb8, mode="RGB").save(path)
