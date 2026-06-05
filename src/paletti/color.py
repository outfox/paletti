"""Vectorized RGB<->HSV conversions

All values are floats in the ``[0, 1]`` range. Arrays may have any leading
shape as long as the final axis is the colour channel (size 3).
"""

from __future__ import annotations

import numpy as np


def safe_divide(num: np.ndarray, denom: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Elementwise ``num / denom``, yielding ``0`` where ``|denom| <= eps``."""
    safe = np.abs(denom) > eps
    return np.where(safe, num / np.where(safe, denom, 1.0), 0.0)


def _rgb_to_hue(r: np.ndarray, g: np.ndarray, b: np.ndarray,
                maxc: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Hue in ``[0, 1]`` from RGB channels and their max/range (grey -> 0)."""
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
    return hue


def rgb2hsv(rgb: np.ndarray) -> np.ndarray:
    """Convert an ``(..., 3)`` RGB array to HSV. Hue is in ``[0, 1]``."""
    rgb = np.asarray(rgb, dtype=np.float64)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    maxc = np.max(rgb, axis=-1)
    minc = np.min(rgb, axis=-1)
    delta = maxc - minc

    hue = _rgb_to_hue(r, g, b, maxc, delta)
    sat = safe_divide(delta, maxc)
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


def rgb2hsl(rgb: np.ndarray) -> np.ndarray:
    """Convert an ``(..., 3)`` RGB array to HSL. Hue is in ``[0, 1]``.

    Shares hue with :func:`rgb2hsv`; lightness is the midpoint of the channel
    extremes and saturation is normalized against that lightness.
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    maxc = np.max(rgb, axis=-1)
    minc = np.min(rgb, axis=-1)
    delta = maxc - minc

    hue = _rgb_to_hue(r, g, b, maxc, delta)

    lig = (maxc + minc) / 2.0
    # S = delta / (1 - |2L - 1|), guarding the L in {0, 1} extremes.
    denominator = 1.0 - np.abs(2.0 * lig - 1.0)
    sat = safe_divide(delta, denominator)

    return np.stack([hue, sat, lig], axis=-1)


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    """Undo the sRGB transfer function on ``[0, 1]`` values."""
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def rgb2oklab(rgb: np.ndarray) -> np.ndarray:
    """Convert an ``(..., 3)`` sRGB array to OKLab (Bjorn Ottosson, 2020).

    Input is treated as gamma-encoded sRGB in ``[0, 1]`` and linearized before
    the conversion, so Euclidean distance in the result approximates perceptual
    colour difference. Returns stacked ``(L, a, b)``.
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    lin = _srgb_to_linear(rgb)
    r, g, b = lin[..., 0], lin[..., 1], lin[..., 2]

    lms_l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    lms_m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    lms_s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b

    l_ = np.cbrt(lms_l)
    m_ = np.cbrt(lms_m)
    s_ = np.cbrt(lms_s)

    big_l = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    bb = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_

    return np.stack([big_l, a, bb], axis=-1)


def rgb2luma(rgb: np.ndarray) -> np.ndarray:
    """Rec. 709 relative luminance of an ``(..., 3)`` RGB array (``[0, 1]``)."""
    rgb = np.asarray(rgb, dtype=np.float64)
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


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
