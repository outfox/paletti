"""The palette-application logic, ported from ``PaletteShader2.shader``.

For every pixel the shader finds the two nearest palette colours (``pointA`` the
closest, ``pointB`` the second closest) and then, depending on the render mode,
either snaps to one of them, smoothly blends between them, or ordered-dithers
between them. We reproduce that here as vectorised NumPy over the whole image.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import color, dither

# Render modes (see the shader's ``renderMode`` switch).
MODES = ("nearest", "second", "factor", "blend", "dither")

# Distance metrics for matching a pixel to a palette colour.
METRICS = ("rgb", "hsv")

DITHER_KINDS = ("nearest", "sine", "bayer", "halftone", "texture")


@dataclass
class Options:
    mode: str = "nearest"
    metric: str = "rgb"
    hsv_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
    hsv_adjust: tuple[float, float, float] = (0.0, 1.0, 1.0)
    dither_kind: str = "bayer"
    dither_res: float = 2.0
    bayer_size: int = 4
    halftone_angle: float = 45.0
    dither_scale: float = 1.0
    dither_texture: np.ndarray | None = None
    prefer_smallest: bool = False

    def validate(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if self.metric not in METRICS:
            raise ValueError(f"metric must be one of {METRICS}")
        if self.dither_kind not in DITHER_KINDS:
            raise ValueError(f"dither kind must be one of {DITHER_KINDS}")


def _hue_dist(h0: np.ndarray, h1: np.ndarray) -> np.ndarray:
    """Shortest distance between two hues on the ``[0, 1]`` circle."""
    d = np.abs(h0 - h1)
    return np.minimum(d, 1.0 - d)


def _two_nearest(pixels: np.ndarray, palette: np.ndarray,
                 opts: Options) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(idxA, idxB)`` palette indices of the two closest colours.

    ``pixels`` is ``(P, 3)`` RGB, ``palette`` is ``(N, 3)`` RGB. Distances are
    computed in chunks to keep the ``(P, N)`` working set bounded for big images.
    """
    n = palette.shape[0]
    p = pixels.shape[0]
    idx_a = np.empty(p, dtype=np.intp)
    idx_b = np.empty(p, dtype=np.intp)

    if opts.metric == "hsv":
        pal = color.rgb2hsv(palette)
        wt = np.asarray(opts.hsv_weights, dtype=np.float64)
    else:
        pal = palette

    chunk = max(1, (1 << 22) // max(n, 1))  # ~4M distance entries per chunk
    for start in range(0, p, chunk):
        block = pixels[start : start + chunk]
        if opts.metric == "hsv":
            hsv = color.rgb2hsv(block)
            dh = _hue_dist(hsv[:, None, 0], pal[None, :, 0]) * wt[0]
            ds = (hsv[:, None, 1] - pal[None, :, 1]) * wt[1]
            dv = (hsv[:, None, 2] - pal[None, :, 2]) * wt[2]
            dist = np.sqrt(dh * dh + ds * ds + dv * dv)
        else:
            diff = block[:, None, :] - pal[None, :, :]
            dist = np.sqrt(np.einsum("pnc,pnc->pn", diff, diff))

        if n == 1:
            idx_a[start : start + block.shape[0]] = 0
            idx_b[start : start + block.shape[0]] = 0
        else:
            part = np.argpartition(dist, 1, axis=1)[:, :2]
            d2 = np.take_along_axis(dist, part, axis=1)
            order = np.argsort(d2, axis=1)
            two = np.take_along_axis(part, order, axis=1)
            idx_a[start : start + block.shape[0]] = two[:, 0]
            idx_b[start : start + block.shape[0]] = two[:, 1]

    return idx_a, idx_b


def _abc_lerp(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Project ``c`` onto segment ``a->b``; return the clamped ``[0, 1]`` factor.

    Mirrors the shader's ``abcLerp``: ``clamp(dot(AC, AB) / |AB|^2, 0, 1)``.
    """
    ab = b - a
    ac = c - a
    denom = np.einsum("pc,pc->p", ab, ab)
    denom = np.where(denom < 1e-12, 1e-12, denom)
    t = np.einsum("pc,pc->p", ac, ab) / denom
    return np.clip(t, 0.0, 1.0)


def apply(image: np.ndarray, palette: np.ndarray, opts: Options) -> np.ndarray:
    """Apply ``palette`` to ``image``.

    ``image`` is ``(H, W, 3)`` float RGB in ``[0, 1]``; ``palette`` is ``(N, 3)``
    float RGB. Returns a new ``(H, W, 3)`` float array.
    """
    opts.validate()
    if palette.shape[0] == 0:
        raise ValueError("palette is empty")

    h, w = image.shape[:2]
    rgb = color.hsv_adjust(image, opts.hsv_adjust)
    pixels = rgb.reshape(-1, 3)

    idx_a, idx_b = _two_nearest(pixels, palette, opts)
    point_a = palette[idx_a]
    point_b = palette[idx_b]

    if opts.mode == "nearest":
        out = point_a
    elif opts.mode == "second":
        out = point_b
    elif opts.mode == "factor":
        t = _abc_lerp(point_a, point_b, pixels)
        out = np.repeat(t[:, None], 3, axis=1)
    elif opts.mode == "blend":
        t = _abc_lerp(point_a, point_b, pixels)[:, None]
        out = point_a + t * (point_b - point_a)
    elif opts.mode == "dither":
        a, b = point_a, point_b
        if opts.prefer_smallest:
            # Order so the darker (smaller magnitude) colour is A.
            la = np.einsum("pc,pc->p", a, a)
            lb = np.einsum("pc,pc->p", b, b)
            swap = la > lb
            a = np.where(swap[:, None], point_b, point_a)
            b = np.where(swap[:, None], point_a, point_b)
        t = _abc_lerp(a, b, pixels)
        field = dither.dither_field(
            opts.dither_kind, h, w,
            res=opts.dither_res, matrix_size=opts.bayer_size,
            angle_deg=opts.halftone_angle, scale=opts.dither_scale,
            texture=opts.dither_texture,
        ).reshape(-1)
        pick = np.floor(t + field)  # 0 -> A, 1 -> B
        out = np.where(pick[:, None] >= 0.5, b, a)
    else:  # pragma: no cover - guarded by validate()
        raise ValueError(opts.mode)

    return out.reshape(h, w, 3)
