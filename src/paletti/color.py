"""Vectorised RGB<->HSV conversions, mirroring the GLSL helpers in the
``palette-shader-2`` Godot project (``rgb2hsv`` / ``hsv2rgb``).

All values are floats in the ``[0, 1]`` range. Arrays may have any leading
shape as long as the final axis is the colour channel (size 3).
"""

from __future__ import annotations

import numpy as np


def rgb2hsv(rgb: np.ndarray) -> np.ndarray:
    """Convert an ``(..., 3)`` RGB array to HSV. Hue is in ``[0, 1]``."""
    rgb = np.asarray(rgb, dtype=np.float64)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    maxc = np.max(rgb, axis=-1)
    minc = np.min(rgb, axis=-1)
    delta = maxc - minc

    # Hue
    hue = np.zeros_like(maxc)
    # Avoid division by zero where delta == 0 (grey pixels keep hue 0).
    safe = delta > 1e-12
    # Masks for which channel is the maximum.
    rmask = safe & (maxc == r)
    gmask = safe & (maxc == g) & ~rmask
    bmask = safe & (maxc == b) & ~rmask & ~gmask

    d = np.where(safe, delta, 1.0)  # placeholder denominator to avoid /0
    hue[rmask] = ((g - b)[rmask] / d[rmask]) % 6.0
    hue[gmask] = ((b - r)[gmask] / d[gmask]) + 2.0
    hue[bmask] = ((r - g)[bmask] / d[bmask]) + 4.0
    hue /= 6.0
    hue %= 1.0

    sat = np.where(maxc > 1e-12, delta / np.where(maxc > 1e-12, maxc, 1.0), 0.0)
    val = maxc

    return np.stack([hue, sat, val], axis=-1)


def hsv2rgb(hsv: np.ndarray) -> np.ndarray:
    """Convert an ``(..., 3)`` HSV array (hue in ``[0, 1]``) to RGB."""
    hsv = np.asarray(hsv, dtype=np.float64)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    h = (h % 1.0) * 6.0
    i = np.floor(h).astype(int)
    f = h - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    i = i % 6
    r = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5],
                  [v, q, p, p, t, v])
    g = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5],
                  [t, v, v, q, p, p])
    b = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5],
                  [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1)


def hsv_adjust(rgb: np.ndarray, adjust: tuple[float, float, float]) -> np.ndarray:
    """Shift hue and scale saturation/value, as the shader does before matching.

    ``adjust`` is ``(hue_shift, sat_mul, val_mul)``; the identity is
    ``(0.0, 1.0, 1.0)``.
    """
    dh, ms, mv = adjust
    if dh == 0.0 and ms == 1.0 and mv == 1.0:
        return rgb
    hsv = rgb2hsv(rgb)
    hsv[..., 0] = (hsv[..., 0] + dh) % 1.0
    hsv[..., 1] = np.clip(hsv[..., 1] * ms, 0.0, 1.0)
    hsv[..., 2] = np.clip(hsv[..., 2] * mv, 0.0, 1.0)
    return hsv2rgb(hsv)
