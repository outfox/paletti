"""Vectorized colour-space conversions.

Thin wrappers over `colour-science <https://www.colour-science.org/>`__ so the
project has one well-tested source of truth for the colour maths. All values are
floats in the ``[0, 1]`` range (hue included) unless noted; CIELAB ``L`` is in
``[0, 100]`` and Lab/LCh hues are in degrees, matching CSS conventions. Arrays may
have any leading shape as long as the final axis is the colour channel (size 3).
"""

from __future__ import annotations

import warnings

import numpy as np

# colour-science warns once at import when Matplotlib is absent; we never use its
# plotting, so silence that specific notice to keep CLI output clean.
warnings.filterwarnings("ignore", message=r'.*"Matplotlib".*')

import colour  # noqa: E402  (import after the warning filter is intentional)


def safe_divide(num: np.ndarray, denom: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Elementwise ``num / denom``, yielding ``0`` where ``|denom| <= eps``."""
    safe = np.abs(denom) > eps
    return np.where(safe, num / np.where(safe, denom, 1.0), 0.0)


def rgb2hsv(rgb: np.ndarray) -> np.ndarray:
    """Convert an ``(..., 3)`` sRGB array to HSV. Hue is in ``[0, 1]``."""
    return colour.RGB_to_HSV(np.asarray(rgb, dtype=np.float64))


def hsv2rgb(hsv: np.ndarray) -> np.ndarray:
    """Convert an ``(..., 3)`` HSV array (hue in ``[0, 1]``) to sRGB."""
    return colour.HSV_to_RGB(np.asarray(hsv, dtype=np.float64))


def rgb2hsl(rgb: np.ndarray) -> np.ndarray:
    """Convert an ``(..., 3)`` sRGB array to HSL. Hue is in ``[0, 1]``."""
    return colour.RGB_to_HSL(np.asarray(rgb, dtype=np.float64))


def hsl2rgb(hsl: np.ndarray) -> np.ndarray:
    """Convert an ``(..., 3)`` HSL array (hue in ``[0, 1]``) to sRGB."""
    return colour.HSL_to_RGB(np.asarray(hsl, dtype=np.float64))


def hwb2rgb(hwb: np.ndarray) -> np.ndarray:
    """Convert an ``(..., 3)`` HWB array (hue in ``[0, 1]``) to sRGB.

    HWB mixes a pure hue with whiteness ``w`` and blackness ``b``; when
    ``w + b >= 1`` the colour collapses to the grey ``w / (w + b)``. (colour-science
    has no HWB model, so this stays a direct construction.)
    """
    hwb = np.asarray(hwb, dtype=np.float64)
    h, w, b = hwb[..., 0], hwb[..., 1], hwb[..., 2]
    base = hsv2rgb(np.stack([h, np.ones_like(h), np.ones_like(h)], axis=-1))
    scaled = base * (1.0 - w - b)[..., None] + w[..., None]
    total = w + b
    grey = (w / np.where(total == 0.0, 1.0, total))[..., None]
    grey = np.broadcast_to(grey, scaled.shape)
    return np.clip(np.where((total >= 1.0)[..., None], grey, scaled), 0.0, 1.0)


def rgb2oklab(rgb: np.ndarray) -> np.ndarray:
    """Convert an ``(..., 3)`` sRGB array to OKLab (perceptual; ``L`` in ``[0, 1]``).

    Euclidean distance in the result approximates perceptual colour difference,
    which is what the default matching metric relies on.
    """
    return colour.XYZ_to_Oklab(colour.sRGB_to_XYZ(np.asarray(rgb, dtype=np.float64)))


def oklab2rgb(lab: np.ndarray) -> np.ndarray:
    """Convert an ``(..., 3)`` OKLab array back to sRGB, clipped to ``[0, 1]``.

    OKLab can name colours outside the sRGB gamut; out-of-gamut results are
    clipped (not gamut-mapped).
    """
    rgb = colour.XYZ_to_sRGB(colour.Oklab_to_XYZ(np.asarray(lab, dtype=np.float64)))
    return np.clip(rgb, 0.0, 1.0)


def lab2rgb(lab: np.ndarray) -> np.ndarray:
    """Convert an ``(..., 3)`` CIELAB array (D65, ``L`` in ``[0, 100]``) to sRGB.

    Clipped to the sRGB gamut. D65 (colour-science's default illuminant) is used
    rather than CSS's strict D50 for ``lab()``/``lch()``.
    """
    rgb = colour.XYZ_to_sRGB(colour.Lab_to_XYZ(np.asarray(lab, dtype=np.float64)))
    return np.clip(rgb, 0.0, 1.0)


def lch2lab(lch: np.ndarray) -> np.ndarray:
    """Polar -> Cartesian: an ``(..., 3)`` ``(L, C, H_deg)`` to ``(L, a, b)``.

    The polar transform is identical for CIELCh (-> :func:`lab2rgb`) and OKLCh
    (-> :func:`oklab2rgb`), so a single wrapper over ``LCHab_to_Lab`` serves both.
    ``H`` is in degrees.
    """
    return colour.LCHab_to_Lab(np.asarray(lch, dtype=np.float64))


def rgb2luma(rgb: np.ndarray) -> np.ndarray:
    """Rec. 709 luma of an ``(..., 3)`` RGB array (weighted sum on ``[0, 1]``).

    A cheap one-dimensional matching axis; deliberately the gamma-domain *luma*,
    not relative luminance, so it is kept as a direct weighted sum.
    """
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
