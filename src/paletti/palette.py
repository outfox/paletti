"""Loading palettes from images or JSON arrays.

A palette is represented as an ``(N, 3)`` float array in ``[0, 1]`` RGB.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

_HEX_RE = re.compile(r"^#?([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")


def _hex_to_rgb(value: str) -> tuple[float, float, float]:
    m = _HEX_RE.match(value.strip())
    if not m:
        raise ValueError(f"invalid hex colour: {value!r}")
    h = m.group(1)
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return r / 255.0, g / 255.0, b / 255.0


def from_json(data, *, assume_range: str = "auto") -> np.ndarray:
    """Parse a palette from already-decoded JSON ``data``.

    Accepted shapes:
      * ``["#ff0000", "00ff00", ...]``                  -- hex strings
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

    # All-string -> hex list.
    if all(isinstance(c, str) for c in data):
        return np.array([_hex_to_rgb(c) for c in data], dtype=np.float64)

    rows = []
    for c in data:
        if isinstance(c, str):
            rows.append(_hex_to_rgb(c))
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

    arr = np.array(ordered, dtype=np.float64) / 255.0
    return arr


def load(spec: str, *, max_colors: int | None = None,
         assume_range: str = "auto") -> np.ndarray:
    """Load a palette from a path or an inline JSON string.

    Dispatches on the argument:
      * a string starting with ``[`` or ``{`` is parsed as inline JSON;
      * a ``.json`` file is parsed as JSON;
      * anything else is treated as an image path.
    """
    text = spec.strip()
    if text[:1] in "[{":
        return from_json(json.loads(text), assume_range=assume_range)

    path = Path(spec)
    if path.suffix.lower() == ".json":
        return from_json_file(path, assume_range=assume_range)
    return from_image(path, max_colors=max_colors)
