"""Loading palettes from images or JSON arrays.

A palette is represented as an ``(N, 3)`` float array in ``[0, 1]`` RGB.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageColor

from . import color

_HEX_RE = re.compile(r"^#?([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")

# CSS colour functions we parse ourselves (Pillow only handles the comma form of
# rgb/hsl/hsv and none of the Lab family). Matches 'name( ... )'.
_CSS_FUNC_RE = re.compile(r"^([a-zA-Z]+)\(\s*(.*?)\s*\)$", re.DOTALL)
_CSS_FUNCS = frozenset({
    "rgb", "rgba", "hsl", "hsla", "hsv", "hsb", "hwb",
    "lab", "lch", "oklab", "oklch",
})


def _css_angle(tok: str) -> float:
    """A CSS hue token to degrees, honouring deg/grad/rad/turn units."""
    t = tok.lower()
    for unit, mul in (("turn", 360.0), ("grad", 0.9),
                      ("rad", 180.0 / math.pi), ("deg", 1.0)):
        if t.endswith(unit):
            return float(t[: -len(unit)]) * mul
    return float(t)


def _css_num(tok: str, *, pct_scale: float = 1.0, div: float = 1.0) -> float:
    """A CSS numeric token to a float.

    ``x%`` becomes ``x / 100 * pct_scale``; a bare number is divided by ``div``
    (used to fold 0..255 rgb channels to 0..1). ``none`` reads as 0.
    """
    if tok.lower() == "none":
        return 0.0
    if tok.endswith("%"):
        return float(tok[:-1]) / 100.0 * pct_scale
    return float(tok) / div


def _eval_css_function(name: str, tokens: list[str]) -> tuple[float, float, float]:
    """Evaluate a parsed CSS colour function to an ``[0, 1]`` RGB tuple."""
    h = lambda: (_css_angle(tokens[0]) / 360.0) % 1.0  # noqa: E731
    if name in ("rgb", "rgba"):
        rgb = [_css_num(t, pct_scale=1.0, div=255.0) for t in tokens[:3]]
    elif name in ("hsl", "hsla"):
        rgb = color.hsl2rgb([h(), _css_num(tokens[1]), _css_num(tokens[2])])
    elif name in ("hsv", "hsb"):
        rgb = color.hsv2rgb([h(), _css_num(tokens[1]), _css_num(tokens[2])])
    elif name == "hwb":
        rgb = color.hwb2rgb([h(), _css_num(tokens[1]), _css_num(tokens[2])])
    elif name == "lab":
        rgb = color.lab2rgb([_css_num(tokens[0], pct_scale=100.0),
                             _css_num(tokens[1], pct_scale=125.0),
                             _css_num(tokens[2], pct_scale=125.0)])
    elif name == "lch":
        rgb = color.lab2rgb(color.lch2lab(
            [_css_num(tokens[0], pct_scale=100.0),
             _css_num(tokens[1], pct_scale=150.0), _css_angle(tokens[2])]))
    elif name == "oklab":
        rgb = color.oklab2rgb([_css_num(tokens[0], pct_scale=1.0),
                               _css_num(tokens[1], pct_scale=0.4),
                               _css_num(tokens[2], pct_scale=0.4)])
    else:  # oklch
        rgb = color.oklab2rgb(color.lch2lab(
            [_css_num(tokens[0], pct_scale=1.0),
             _css_num(tokens[1], pct_scale=0.4), _css_angle(tokens[2])]))
    arr = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0)
    return float(arr[0]), float(arr[1]), float(arr[2])


def _parse_css_function(spec: str):
    """Parse a ``name(...)`` CSS colour to RGB, or ``None`` if not a known one.

    Accepts both the legacy comma form and the modern space-separated CSS Color 4
    syntax (``rgb(255 0 0)``, ``hsl(120deg 100% 50%)``), with an optional
    ``/ alpha`` that we drop. Raises ``ValueError`` if a recognised function has
    a malformed body.
    """
    m = _CSS_FUNC_RE.match(spec)
    if m is None:
        return None
    name = m.group(1).lower()
    if name not in _CSS_FUNCS:
        return None
    body = m.group(2).split("/", 1)[0]  # discard the alpha component
    tokens = [t for t in re.split(r"[\s,]+", body.strip()) if t]
    try:
        if len(tokens) < 3:
            raise ValueError
        return _eval_css_function(name, tokens)
    except (ValueError, IndexError):
        raise ValueError(f"invalid colour: {spec!r}")


def _hex_to_rgb(value: str) -> tuple[float, float, float]:
    m = _HEX_RE.match(value.strip())
    if not m:
        raise ValueError(f"invalid hex colour: {value!r}")
    h = m.group(1)
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return r / 255.0, g / 255.0, b / 255.0


def parse_color(value: str) -> tuple[float, float, float]:
    """Parse a single colour token into ``[0, 1]`` RGB.

    Accepts, in order: the hex forms a JSON palette takes (``#1a1c2c``,
    ``1a1c2c``, ``#fff`` or bare ``fff``); a CSS colour function -- ``rgb()``,
    ``hsl()``, ``hsv()``/``hsb()``, ``hwb()``, ``lab()``, ``lch()``, ``oklab()``,
    ``oklch()`` -- in either the comma or the space-separated CSS Color 4 syntax;
    or any CSS/SVG colour name (``white``, ``rebeccapurple``). This is the shared
    scalar parser behind both JSON string entries and bare ``-p`` colour tokens,
    so the two stay equivalent.
    """
    s = value.strip()
    if _HEX_RE.match(s):
        return _hex_to_rgb(s)
    rgb = _parse_css_function(s)
    if rgb is not None:
        return rgb
    try:
        r, g, b = ImageColor.getrgb(s)[:3]
    except ValueError:
        raise ValueError(f"invalid colour: {value!r}")
    return r / 255.0, g / 255.0, b / 255.0


def from_colors(tokens) -> np.ndarray:
    """Build a palette from individual colour tokens (hex or names).

    The counterpart to :func:`from_json`/:func:`from_image` for colours listed
    one-by-one (each bare ``-p`` colour token): every token is parsed by
    :func:`parse_color`, yielding an ``(N, 3)`` ``[0, 1]`` array.
    """
    if not tokens:
        raise ValueError("no colours given")
    return np.array([parse_color(t) for t in tokens], dtype=np.float64)


def from_json(data, *, assume_range: str = "auto") -> np.ndarray:
    """Parse a palette from already-decoded JSON ``data``.

    Accepted shapes:
      * ``["#ff0000", "00ff00", "white", ...]``         -- hex or CSS names
      * ``[[255, 0, 0], [0, 128, 64], ...]``            -- 0..255 integers
      * ``[[1.0, 0.0, 0.0], ...]``                      -- 0..1 floats
      * ``{"colors": [...]}`` / ``{"palette": [...]}``  -- wrapped in an object

    ``assume_range`` controls numeric interpretation: ``"255"``, ``"unit"`` or
    ``"auto"`` (floats present or all values <= 1 -> unit, else 0..255).
    """
    if isinstance(data, dict):
        for key in ("colors", "palette", "swatches"):
            if key in data:
                data = data[key]
                break
        else:
            raise ValueError("JSON object has no 'colors'/'palette' key")

    if not isinstance(data, list) or not data:
        raise ValueError("palette JSON must be a non-empty array of colours")

    # All-string -> hex/name list.
    if all(isinstance(c, str) for c in data):
        return np.array([parse_color(c) for c in data], dtype=np.float64)

    rows = []
    for c in data:
        if isinstance(c, str):
            rows.append(parse_color(c))
            continue
        if not isinstance(c, (list, tuple)) or len(c) < 3:
            raise ValueError(f"colour must be hex or [r, g, b]: {c!r}")
        rows.append(tuple(float(x) for x in c[:3]))

    arr = np.array(rows, dtype=np.float64)

    # Hex entries were already normalized; only treat numeric rows for range.
    if assume_range == "unit":
        scale = False
    elif assume_range == "255":
        scale = True
    else:  # auto
        any_float = any(
            isinstance(x, float) and not float(x).is_integer()
            for c in data
            if isinstance(c, (list, tuple))
            for x in c[:3]
        )
        # Noninteger floats imply unit range; otherwise a max above 1 means the
        # values are 0..255 integers. All-0/1 integers are read as unit floats.
        scale = (not any_float) and (arr.max() > 1.0)

    if scale:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def from_json_file(path: str | Path, *, assume_range: str = "auto") -> np.ndarray:
    with open(path, "r", encoding="utf-8") as fh:
        return from_json(json.load(fh), assume_range=assume_range)


def from_image(path: str | Path, *, max_colors: int | None = None) -> np.ndarray:
    """Extract a palette from an image's distinct colours.

    Colours are returned most-frequent first. ``max_colors`` caps the result to
    the N most common colours, which is useful when importing from a palette
    strip that may contain antialiasing fringe pixels.
    """
    img = Image.open(path).convert("RGB")
    pixels = np.asarray(img, dtype=np.uint8).reshape(-1, 3)

    # Count occurrences of each unique colour, preserving frequency order.
    counts = Counter(map(tuple, pixels))
    ordered = [c for c, _ in counts.most_common()]
    if max_colors is not None:
        ordered = ordered[:max_colors]

    return np.array(ordered, dtype=np.float64) / 255.0


def half_brite(pal: np.ndarray) -> np.ndarray:
    """Append a half-brightness copy of every colour, Amiga "Extra Half-Brite".

    Each colour gains a twin at half its RGB value (halving the channels halves
    luminance while keeping hue and saturation), doubling an ``(N, 3)`` palette to
    ``(2N, 3)``. The dimmed copies follow the originals in order.
    """
    return np.vstack([pal, pal * 0.5])


def _is_inline_json(spec: str) -> bool:
    """True if ``spec`` is an inline JSON array/object rather than a path."""
    return spec.strip()[:1] in "[{"


def is_json_spec(spec: str) -> bool:
    """True if ``load`` would parse ``spec`` as JSON (inline or a ``.json`` file)."""
    return _is_inline_json(spec) or Path(spec).suffix.lower() == ".json"


def source_kind(spec: str) -> str:
    """Classify a single :func:`load` token as ``"json"``, ``"image"`` or ``"color"``.

    Mirrors :func:`load`'s dispatch without touching disk beyond an existence
    check, so callers can tell which options (``--max-colors`` for images,
    ``--palette-range`` for JSON) a given source will actually consume.
    """
    s = spec.strip()
    if _is_inline_json(s) or Path(s).suffix.lower() == ".json":
        return "json"
    if Path(s).exists():
        return "image"
    try:
        parse_color(s)
        return "color"
    except ValueError:
        return "image"


def load(spec: str, *, max_colors: int | None = None,
         assume_range: str = "auto") -> np.ndarray:
    """Load a palette from a single source token.

    Dispatches on the argument:
      * a string starting with ``[`` or ``{`` is parsed as inline JSON;
      * a ``.json`` file is parsed as JSON;
      * an existing file is read as an image palette;
      * otherwise a lone hex/name token (``000``, ``lavender``) is one colour;
      * a non-existent path falls through to the image reader so its error names
        the file.
    """
    s = spec.strip()
    if _is_inline_json(s):
        return from_json(json.loads(s), assume_range=assume_range)

    path = Path(s)
    if path.suffix.lower() == ".json":
        return from_json_file(path, assume_range=assume_range)
    if path.exists():
        return from_image(path, max_colors=max_colors)
    try:
        return from_colors([s])
    except ValueError:
        return from_image(path, max_colors=max_colors)
