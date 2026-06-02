"""Ordered-dither value sources.

The shader computes a per-pixel ``ditherValue`` in ``[0, 1]`` that is added to
the interpolation factor before flooring, producing ordered dithering between
the two nearest palette colours. Here we reproduce the three procedural sources
(nearest / sine / Bayer-matrix texture) over a full image grid.
"""

from __future__ import annotations

import numpy as np


def bayer_matrix(n: int) -> np.ndarray:
    """Return an ``n x n`` Bayer ordered-dither matrix normalised to ``[0, 1)``.

    ``n`` must be a power of two. Values are the classic recursive threshold map
    divided by ``n*n`` so they tile cleanly as a threshold pattern.
    """
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError("Bayer matrix size must be a power of two")
    m = np.zeros((1, 1), dtype=np.float64)
    size = 1
    while size < n:
        m = np.block([
            [4 * m + 0, 4 * m + 2],
            [4 * m + 3, 4 * m + 1],
        ])
        size *= 2
    return m / (n * n)


def halftone_field(height: int, width: int, *, cell: float = 8.0,
                   angle_deg: float = 45.0) -> np.ndarray:
    """Build an ``(height, width)`` halftone dot field in ``[0, 1]``.

    This is the procedural equivalent of the Godot project's "Screentone"
    dither (``screentonesdf.png``): a grid of round dots laid out on a lattice
    rotated by ``angle_deg``, where the value is the normalised distance from
    the nearest dot centre (``0`` at a centre, ``1`` at a cell corner). Used as
    an ordered-dither source, thresholding it grows/shrinks the dots with the
    blend factor -- the classic newspaper halftone look.

    ``cell`` is the dot spacing in pixels; ``angle_deg`` rotates the grid
    (``45`` matches the original screentone; ``0`` gives an axis-aligned grid).
    """
    cell = max(cell, 1.0)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    th = np.radians(angle_deg)
    cos, sin = np.cos(th), np.sin(th)
    # Rotate into lattice space and measure distance to the nearest cell centre.
    xr = (xx * cos - yy * sin) / cell
    yr = (xx * sin + yy * cos) / cell
    fx = xr - np.round(xr)
    fy = yr - np.round(yr)
    # Half the cell diagonal is sqrt(0.5); divide so corners reach exactly 1.
    dist = np.sqrt(fx * fx + fy * fy) / np.sqrt(0.5)
    return np.clip(dist, 0.0, 1.0)


def _tiled(values: np.ndarray, height: int, width: int, scale: float) -> np.ndarray:
    """Tile a small 2-D ``values`` grid across an ``(height, width)`` image.

    ``scale`` enlarges each cell of the source grid to ``scale`` pixels, matching
    the shader's ``ditherRes`` behaviour (``mod(uv, ditherRes) / ditherRes``).
    """
    th, tw = values.shape
    scale = max(scale, 1.0)
    ys = (np.floor(np.arange(height) / scale).astype(int)) % th
    xs = (np.floor(np.arange(width) / scale).astype(int)) % tw
    return values[np.ix_(ys, xs)]


def _resample(arr: np.ndarray, factor: float) -> np.ndarray:
    """Resize a 2-D array by ``factor`` with wrap-aware bilinear sampling.

    Wrapping keeps a seamless (tileable) texture seamless after scaling.
    ``factor`` may be greater or less than 1 (enlarge or shrink).
    """
    if factor == 1.0:
        return arr
    h, w = arr.shape
    nh = max(1, int(round(h * factor)))
    nw = max(1, int(round(w * factor)))
    # Map each output pixel centre back to a source coordinate.
    sy = (np.arange(nh) + 0.5) / factor - 0.5
    sx = (np.arange(nw) + 0.5) / factor - 0.5
    y0 = np.floor(sy).astype(int)
    x0 = np.floor(sx).astype(int)
    wy = (sy - y0)[:, None]
    wx = sx - x0
    y0m, y1m = y0 % h, (y0 + 1) % h
    x0m, x1m = x0 % w, (x0 + 1) % w
    top = arr[y0m][:, x0m] * (1 - wx) + arr[y0m][:, x1m] * wx
    bot = arr[y1m][:, x0m] * (1 - wx) + arr[y1m][:, x1m] * wx
    return top * (1 - wy) + bot * wy


def dither_field(
    kind: str,
    height: int,
    width: int,
    *,
    res: float = 2.0,
    matrix_size: int = 4,
    angle_deg: float = 45.0,
    scale: float = 1.0,
    texture: np.ndarray | None = None,
) -> np.ndarray:
    """Build an ``(height, width)`` array of dither values in ``[0, 1]``.

    ``kind`` is one of ``"nearest"``, ``"sine"``, ``"bayer"``, ``"halftone"`` or
    ``"texture"``. For ``"halftone"`` ``res`` is the dot spacing and
    ``angle_deg`` rotates the grid. For ``"texture"`` a single-channel
    ``texture`` array (any range, taken as its first channel) is resized by
    ``scale`` (e.g. ``10`` for 10x) and then tiled across the image.
    """
    if kind == "nearest":
        return np.full((height, width), 0.5)

    if kind == "halftone":
        return halftone_field(height, width, cell=res, angle_deg=angle_deg)

    if kind == "sine":
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
        uv_x = xx / (np.pi * 20.0)
        uv_y = yy / (np.pi * 20.0)
        return (np.sin(uv_x * res) + np.sin(uv_y * res) + 2.0) / 4.0

    if kind == "bayer":
        return _tiled(bayer_matrix(matrix_size), height, width, res)

    if kind == "texture":
        if texture is None:
            raise ValueError("dither kind 'texture' requires a texture array")
        tex = np.asarray(texture, dtype=np.float64)
        if tex.ndim == 3:
            tex = tex[..., 0]
        if tex.max() > 1.0:
            tex = tex / 255.0
        tex = _resample(tex, scale)
        return _tiled(tex, height, width, 1.0)

    raise ValueError(f"unknown dither kind: {kind!r}")
