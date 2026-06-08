"""Recolour an SVG's vector colours onto a palette, keeping it vector.

The raster path rasterizes an SVG and runs the full pixel pipeline; this is the
other half -- when the output is itself an ``.svg``, every colour the document
*uses* (fills, strokes, gradient stops, inline and ``<style>`` CSS) is snapped to
its nearest palette colour via the very same matcher as the pixel path, and the
vectors are written back unchanged. Each distinct source colour maps to exactly
one palette colour, so this is a faithful vector recolour, not a rasterization.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import core
from . import palette as palette_mod

_SVG_NS = "http://www.w3.org/2000/svg"

# Presentation attributes (and CSS properties) whose value is a single colour.
_COLOR_PROPS = ("fill", "stroke", "stop-color", "flood-color",
                "lighting-color", "color")

# Keywords that are valid colour-slot values but not real colours to remap.
_SKIP = {"none", "currentcolor", "inherit", "initial", "unset", "transparent",
         "context-fill", "context-stroke"}

# A ``prop: value`` colour declaration inside a style="" attr or a <style> block.
_STYLE_DECL_RE = re.compile(
    r"(" + "|".join(_COLOR_PROPS) + r")(\s*:\s*)([^;}]+)", re.IGNORECASE)


def _local(tag: str) -> str:
    """Strip an ElementTree ``{namespace}local`` tag down to ``local``."""
    return tag.rsplit("}", 1)[-1]


def _is_real_color(value: str) -> bool:
    """True if ``value`` is a colour to remap (not ``none``/``url(...)``/etc.)."""
    v = value.strip()
    if not v or v.lower() in _SKIP:
        return False
    return not v.lower().startswith("url(")


def _rgb_to_hex(rgb: np.ndarray) -> str:
    r, g, b = (int(x) for x in np.rint(np.clip(rgb, 0.0, 1.0) * 255.0))
    return f"#{r:02x}{g:02x}{b:02x}"


def _build_mapping(tokens: set[str], palette: np.ndarray,
                   opts: core.Options) -> dict[str, str]:
    """Map each source colour token to its palette hex via ``core.apply``.

    Tokens are matched as a ``(1, K, 3)`` swatch strip, so the exact same
    metric/two-nearest logic the pixel path uses drives the recolour. Spatial
    pre-passes (blur/denoise) would bleed neighbouring swatches into each other,
    so they are neutralised here; the per-pixel ``hsv_adjust`` is kept.
    """
    parsed: dict[str, tuple[float, float, float]] = {}
    for tok in tokens:
        try:
            parsed[tok] = palette_mod.parse_color(tok)
        except ValueError:
            pass  # unrecognised token -> leave it untouched in the document
    if not parsed:
        return {}

    keys = list(parsed)
    swatches = np.array([parsed[k] for k in keys], dtype=np.float64)
    opts = replace(opts, pre_blur=0.0, denoise=0.0)
    mapped = core.apply(swatches.reshape(1, -1, 3), palette, opts).reshape(-1, 3)
    return {k: _rgb_to_hex(mapped[i]) for i, k in enumerate(keys)}


def _rewrite_style(text: str, mapping: dict[str, str]) -> str:
    """Rewrite colour declarations within a style="" / <style> string."""
    def repl(m: re.Match) -> str:
        value = m.group(3).strip()
        new = mapping.get(value)
        return f"{m.group(1)}{m.group(2)}{new}" if new else m.group(0)

    return _STYLE_DECL_RE.sub(repl, text)


def recolor(in_path: str | Path, out_path: str | Path,
            palette: np.ndarray, opts: core.Options) -> dict[str, str]:
    """Recolour ``in_path`` onto ``palette`` and write the SVG to ``out_path``.

    Returns the ``source token -> palette hex`` mapping that was applied.
    """
    # Emit a bare ``svg`` namespace (no ``ns0:`` prefixes) on write-out.
    ET.register_namespace("", _SVG_NS)
    try:
        tree = ET.parse(in_path)
    except ET.ParseError as exc:
        raise ValueError(f"invalid SVG: {exc}") from exc
    root = tree.getroot()

    # Pass 1: gather every distinct colour token the document uses.
    tokens: set[str] = set()
    for el in root.iter():
        for prop in _COLOR_PROPS:
            value = el.get(prop)
            if value is not None and _is_real_color(value):
                tokens.add(value.strip())
        for source in (el.get("style"), el.text if _local(el.tag) == "style" else None):
            if source:
                for m in _STYLE_DECL_RE.finditer(source):
                    if _is_real_color(m.group(3)):
                        tokens.add(m.group(3).strip())

    mapping = _build_mapping(tokens, palette, opts)
    if not mapping:
        Path(out_path).write_text(ET.tostring(root, encoding="unicode"),
                                  encoding="utf-8")
        return mapping

    # Pass 2: rewrite those same slots with the mapped palette hex.
    for el in root.iter():
        for prop in _COLOR_PROPS:
            value = el.get(prop)
            if value is not None and value.strip() in mapping:
                el.set(prop, mapping[value.strip()])
        style = el.get("style")
        if style:
            el.set("style", _rewrite_style(style, mapping))
        if _local(el.tag) == "style" and el.text:
            el.text = _rewrite_style(el.text, mapping)

    Path(out_path).write_text(ET.tostring(root, encoding="unicode"),
                              encoding="utf-8")
    return mapping
