"""Image loading/saving helpers that preserve the original alpha channel."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image


def _rasterize_svg(path: str | Path, scale: float = 1.0) -> Image.Image:
    """Render an SVG file to a Pillow image via resvg.

    SVGs are resolution-independent, so the renderer uses the document's
    intrinsic ``width``/``height`` (or ``viewBox``) size. ``scale`` multiplies
    that: rather than upscaling a small raster, it re-renders at the larger size
    so the result stays crisp.
    """
    import resvg_py  # lazy: only needed when an SVG is actually loaded

    p = str(path)
    png = resvg_py.svg_to_bytes(svg_path=p)
    img = Image.open(io.BytesIO(png))
    if scale and scale != 1.0:
        width = max(1, round(img.width * scale))
        png = resvg_py.svg_to_bytes(svg_path=p, width=width)
        img = Image.open(io.BytesIO(png))
    return img


def load_rgb(path: str | Path,
             svg_scale: float = 1.0) -> tuple[np.ndarray, np.ndarray | None]:
    """Load an image as ``(H, W, 3)`` float RGB in ``[0, 1]``.

    Returns ``(rgb, alpha)`` where ``alpha`` is an ``(H, W)`` float array in
    ``[0, 1]`` if the source had transparency, otherwise ``None``. ``.svg``
    inputs are rasterized first (at ``svg_scale``x their intrinsic size); the
    renderer emits RGBA, so transparency flows through the same path as any
    other source.
    """
    if Path(path).suffix.lower() == ".svg":
        img = _rasterize_svg(path, svg_scale)
    else:
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
