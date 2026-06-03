"""The palette-application logic, ported from ``PaletteShader2.shader``.

For every pixel the shader finds the two nearest palette colours (``point A`` the
closest, ``point B`` the second closest) and then, depending on the render mode,
either snaps to one of them, smoothly blends between them, or ordered-dithers
between them. We reproduce that here as vectorized NumPy over the whole image.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import color, dither

# Render modes (see the shader's ``renderMode`` switch). ``dither-rgb`` is an
# addition: ordered dither applied to each RGB channel independently before
# snapping to the palette, rather than dithering along the two-nearest line.
MODES = ("nearest", "blend", "dither", "dither-rgb")

# Distance metrics for matching a pixel to a palette colour. "oklab" measures
# perceptual difference and is the default; "rgb" is plain Euclidean; "hsl"/
# "hsv" weight the cylindrical channels (hue handled on its circle); "hue" and
# "luma" match on that single axis alone.
METRICS = ("oklab", "rgb", "hsl", "hsv", "hue", "luma")

DITHER_KINDS = ("nearest", "sine", "bayer", "halftone", "texture")


@dataclass
class Options:
    mode: str = "nearest"
    metric: str = "oklab"
    pre_blur: float = 0.0
    hsv_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
    hsv_adjust: tuple[float, float, float] = (0.0, 1.0, 1.0)
    dither_kind: str = "bayer"
    dither_res: float = 2.0
    bayer_size: int = 4
    halftone_angle: float = 45.0
    dither_scale: float = 1.0
    dither_softness: float = 0.0
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


def _to_metric_space(rgb: np.ndarray, metric: str) -> np.ndarray:
    """Project an ``(..., 3)`` RGB array into the feature space of ``metric``.

    Cylindrical metrics keep their channel layout (hue stays in channel 0 so the
    distance step can treat it on its circle); ``hue`` and ``luma`` collapse to a
    single trailing axis.
    """
    if metric == "oklab":
        return color.rgb2oklab(rgb)
    if metric == "hsv":
        return color.rgb2hsv(rgb)
    if metric == "hsl":
        return color.rgb2hsl(rgb)
    if metric == "hue":
        return color.rgb2hsv(rgb)[..., 0]
    if metric == "luma":
        return color.rgb2luma(rgb)
    return np.asarray(rgb, dtype=np.float64)  # rgb


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

    metric = opts.metric
    pal = _to_metric_space(palette, metric)
    wt = np.asarray(opts.hsv_weights, dtype=np.float64)

    chunk = max(1, (1 << 22) // max(n, 1))  # ~4M distance entries per chunk
    for start in range(0, p, chunk):
        block = pixels[start : start + chunk]
        feat = _to_metric_space(block, metric)
        if metric in ("hsv", "hsl"):
            # Hue lives on a circle; saturation/value (or lightness) are linear.
            # The hsv metric additionally weights the three axes by --hsv-weights.
            w = wt if metric == "hsv" else np.ones(3)
            dh = _hue_dist(feat[:, None, 0], pal[None, :, 0]) * w[0]
            ds = (feat[:, None, 1] - pal[None, :, 1]) * w[1]
            dz = (feat[:, None, 2] - pal[None, :, 2]) * w[2]
            dist = np.sqrt(dh * dh + ds * ds + dz * dz)
        elif metric == "hue":
            dist = _hue_dist(feat[:, None], pal[None, :])
        elif metric == "luma":
            dist = np.abs(feat[:, None] - pal[None, :])
        else:  # rgb, oklab: plain Euclidean in the feature space
            diff = feat[:, None, :] - pal[None, :, :]
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
    denominator = np.einsum("pc,pc->p", ab, ab)
    denominator = np.where(denominator < 1e-12, 1e-12, denominator)
    t = np.einsum("pc,pc->p", ac, ab) / denominator
    return np.clip(t, 0.0, 1.0)


def _gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur of an ``(H, W, C)`` float image.

    Used as an optional pre-pass: smoothing the source before matching keeps the
    per-pixel choice of palette colours (and the blend factor) spatially
    coherent, which suppresses the sharp pixel-sized speckle that source noise
    otherwise produces near palette-colour boundaries.
    """
    if sigma <= 0.0:
        return image
    radius = max(1, int(round(sigma * 3.0)))
    xs = np.arange(-radius, radius + 1)
    kernel = np.exp(-(xs * xs) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()

    def conv(arr: np.ndarray, axis: int) -> np.ndarray:
        pad = [(0, 0)] * arr.ndim
        pad[axis] = (radius, radius)
        padded = np.pad(arr, pad, mode="edge")
        n = arr.shape[axis]
        out = np.zeros_like(arr)
        for i, weight in enumerate(kernel):
            sl = [slice(None)] * arr.ndim
            sl[axis] = slice(i, i + n)
            out += weight * padded[tuple(sl)]
        return out

    return conv(conv(image, 0), 1)


def _make_field(opts: Options, h: int, w: int) -> np.ndarray:
    """Build the flat ``(H*W, )`` dither field selected by ``opts``."""
    return dither.dither_field(
        opts.dither_kind, h, w,
        res=opts.dither_res, matrix_size=opts.bayer_size,
        angle_deg=opts.halftone_angle, scale=opts.dither_scale,
        texture=opts.dither_texture,
    ).reshape(-1)


def _make_field_rgb(opts: Options, h: int, w: int) -> np.ndarray:
    """Return an ``(H*W, 3)`` per-channel dither field for ``dither-rgb``.

    If the dither texture is genuinely RGB (its channels actually differ), each
    of its R/G/B channels drives the matching image channel. Otherwise, a single
    field is reused, rotated by 1/3 per channel to decorrelate the noise (that
    decorrelation is what dissolves banding in the hard 1-bit path).
    """
    if opts.dither_kind == "texture" and opts.dither_texture is not None:
        tex = np.asarray(opts.dither_texture, dtype=np.float64)
        if tex.ndim == 3 and tex.shape[-1] >= 3:
            rgb = tex[..., :3]
            if rgb.max() > 1.0:
                rgb = rgb / 255.0
            # Treat as colour only if some pixel has a real channel spread.
            if float(np.ptp(rgb, axis=-1).max()) > 1e-3:
                field = dither.texture_field(h, w, rgb, scale=opts.dither_scale)
                return field.reshape(-1, 3)

    field = _make_field(opts, h, w)
    phases = np.array([0.0, 1.0 / 3.0, 2.0 / 3.0])
    return (field[:, None] + phases[None, :]) % 1.0


def apply(image: np.ndarray, palette: np.ndarray, opts: Options) -> np.ndarray:
    """Apply ``palette`` to ``image``.

    ``image`` is ``(H, W, 3)`` float RGB in ``[0, 1]``; ``palette`` is ``(N, 3)``
    float RGB. Returns a new ``(H, W, 3)`` float array.
    """
    opts.validate()
    if palette.shape[0] == 0:
        raise ValueError("palette is empty")

    h, w = image.shape[:2]
    base = _gaussian_blur(image, opts.pre_blur)
    rgb = color.hsv_adjust(base, opts.hsv_adjust)
    pixels = rgb.reshape(-1, 3)

    if opts.mode == "dither-rgb":
        # Per-channel ordered dither between the two nearest palette colours, then
        # snap the (possibly channel-mixed) result back onto the palette. Each
        # channel is thresholded against its own dither field by per-channel
        # intensity (how far the channel sits from A toward B), so a distance-field
        # texture yields a crisp dot whose radius tracks intensity; a colour
        # texture drives each channel from its own channel, a grey field is rotated
        # 1/3 per channel to decorrelate (which dissolves banding for bayer/noise).
        idx_a, idx_b = _two_nearest(pixels, palette, opts)
        a, b = palette[idx_a], palette[idx_b]
        denominator = b - a
        safe = np.abs(denominator) > 1e-9
        t = np.clip(np.where(safe, (pixels - a) / np.where(safe, denominator, 1.0), 0.0),
                    0.0, 1.0)
        field_rgb = _make_field_rgb(opts, h, w)
        target = a + (t + field_rgb - 1.0 >= 0.0) * (b - a)
        snap, _ = _two_nearest(target, palette, opts)
        crisp = palette[snap].reshape(h, w, 3)

        # ``softness`` anti-aliases the dot edges. The per-channel snap is 1-bit
        # and on-palette, and A/B are nearest neighbours with no palette colour
        # between them, so the threshold itself can never hold the in-between
        # tones an anti-aliased edge needs -- softening it would only shift the
        # boundary (shrink the dots) -- instead we soften the *rendered* result,
        # which lets neighbouring palette colours blend across each boundary; the
        # blur radius grows with ``soft`` and 0 leaves the crisp dither untouched.
        soft = opts.dither_softness
        return crisp if soft <= 0.0 else _gaussian_blur(crisp, soft)

    idx_a, idx_b = _two_nearest(pixels, palette, opts)
    point_a = palette[idx_a]
    point_b = palette[idx_b]

    if opts.mode == "nearest":
        out = point_a
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
        field = _make_field(opts, h, w)
        # The A/B boundary sits where ``t + field == 1``; ``edge`` is the signed
        # distance to it (>0 picks B, <0 picks A).
        edge = t + field - 1.0
        soft = opts.dither_softness
        if soft <= 0.0:
            blend = (edge >= 0.0).astype(np.float64)  # hard 1-bit pick
        else:
            # Smoothstep a gradient band of total width ``soft`` across the edge.
            x = np.clip(edge / soft + 0.5, 0.0, 1.0)
            blend = x * x * (3.0 - 2.0 * x)
        out = a + blend[:, None] * (b - a)
    else:  # pragma: no cover - guarded by validate()
        raise ValueError(opts.mode)

    return out.reshape(h, w, 3)
