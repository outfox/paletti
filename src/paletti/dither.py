"""Ordered-dither value sources.

The shader computes a per-pixel ``ditherValue`` in ``[0, 1]`` that is added to
the interpolation factor before flooring, producing ordered dithering between
the two nearest palette colours. Here we reproduce the three procedural sources
(nearest / sine / Bayer-matrix texture) over a full image grid.
"""

from __future__ import annotations

import numpy as np


def bayer_matrix(n: int) -> np.ndarray:
    """Return an ``n x n`` Bayer ordered-dither matrix normalized to ``[0, 1)``.

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
    rotated by ``angle_deg``, where the value is the normalized distance from
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


def _sample_wrap(tex: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """Bilinearly sample ``tex`` at row coords ``ys`` and column coords ``xs``.

    Coordinates wrap around the texture (so it tiles seamlessly). Works for 2-D
    textures and ``(H, W, C)`` textures alike, preserving the channel axis.
    """
    th, tw = tex.shape[:2]
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    wy = ys - y0
    wx = xs - x0
    y0m, y1m = y0 % th, (y0 + 1) % th
    x0m, x1m = x0 % tw, (x0 + 1) % tw
    if tex.ndim == 3:  # broadcast weights over the trailing channel axis
        wy, wx = wy[:, None, None], wx[None, :, None]
    else:
        wy, wx = wy[:, None], wx[None, :]
    top = tex[y0m][:, x0m] * (1 - wx) + tex[y0m][:, x1m] * wx
    bot = tex[y1m][:, x0m] * (1 - wx) + tex[y1m][:, x1m] * wx
    return top * (1 - wy) + bot * wy


def texture_field(height: int, width: int, texture: np.ndarray, *,
                  scale: float = 1.0) -> np.ndarray:
    """Tile a texture across ``(height, width)``, then zoom it by ``scale``.

    The texture is laid over the image at a 1:1 texel-to-pixel ratio, repeating
    to fill the frame; ``scale`` then zooms that tiled field about the origin --
    ``2`` makes the pattern twice as large, ``0.5`` half. The zoom is isotropic
    (the same factor on both axes), so the pattern keeps its own aspect ratio
    regardless of the image's -- a round dot stays round on any frame shape.
    ``scale == 1.0`` is an exact 1:1 mapping with no interpolation.

    Returns ``(H, W)`` for a 2-D texture or ``(H, W, C)`` preserving channels.
    Values are normalized to ``[0, 1]``.
    """
    tex = np.asarray(texture, dtype=np.float64)
    if tex.max() > 1.0:
        tex = tex / 255.0
    if scale <= 0.0:
        scale = 1.0
    ys = np.arange(height) / scale
    xs = np.arange(width) / scale
    return _sample_wrap(tex, ys, xs)


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
    ``texture`` array (any range, taken as its first channel) is zoomed by
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
            tex = tex[..., 0]  # single greyscale field for the standard mode
        return texture_field(height, width, tex, scale=scale)

    raise ValueError(f"unknown dither kind: {kind!r}")
