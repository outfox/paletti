"""Command-line interface for paletti."""

from __future__ import annotations

import sys
from pathlib import Path

import click
import numpy as np

from . import core, imageio, palette as palette_mod
from .core import DITHER_KINDS, METRICS, MODES


def _parse_triplet(ctx, param, value):
    """Parse a ``a,b,c`` option into a float 3-tuple."""
    if value is None:
        return None
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 3:
        raise click.BadParameter("expected three comma-separated numbers")
    try:
        return tuple(float(p) for p in parts)
    except ValueError as exc:
        raise click.BadParameter(str(exc))


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("input_image", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output_image", type=click.Path(dir_okay=False, path_type=Path))
@click.option("-p", "--palette", "palette_spec", required=True, metavar="SPEC",
              help="Palette source: an image path, a .json file, or an inline "
                   "JSON array such as '[\"#1a1c2c\",\"#5d275d\"]'.")
@click.option("-m", "--mode", type=click.Choice(MODES), default="nearest",
              show_default=True,
              help="nearest: snap to closest colour. second: 2nd closest. "
                   "blend: smooth lerp between the two nearest. "
                   "dither: ordered dither between the two nearest. "
                   "dither-rgb: ordered dither each RGB channel independently, "
                   "then snap to the palette. "
                   "factor: visualise the blend factor as greyscale.")
@click.option("--metric", type=click.Choice(METRICS), default="rgb",
              show_default=True, help="Colour-distance metric for matching.")
@click.option("--smooth", type=float, default=0.0, show_default=True,
              metavar="SIGMA",
              help="Gaussian-blur the source by this sigma (in pixels) before "
                   "palettizing. Suppresses sharp pixel-sized artifacts from "
                   "source noise; try 0.5-2.")
@click.option("--max-colors", type=int, default=None,
              help="When importing a palette from an image, keep only the N "
                   "most frequent colours.")
@click.option("--palette-range", type=click.Choice(["auto", "unit", "255"]),
              default="auto", show_default=True,
              help="How to interpret numeric JSON palette values.")
@click.option("--dither", "dither_kind", type=click.Choice(DITHER_KINDS),
              default="bayer", show_default=True,
              help="Dither value source (used by --mode dither).")
@click.option("--dither-res", type=float, default=2.0, show_default=True,
              help="Dither cell size in pixels / pattern frequency. For "
                   "--dither halftone this is the dot spacing (try 6-12).")
@click.option("--bayer-size", type=int, default=4, show_default=True,
              help="Bayer matrix size (power of two) for --dither bayer.")
@click.option("--halftone-angle", type=float, default=45.0, show_default=True,
              metavar="DEG", help="Grid rotation for --dither halftone "
                                   "(45 = classic screentone, 0 = square grid).")
@click.option("--dither-texture", type=click.Path(exists=True, dir_okay=False),
              default=None, help="Image to tile as the dither pattern "
                                 "(for --dither texture).")
@click.option("--dither-scale", type=float, default=1.0, show_default=True,
              help="Zoom the tiled dither texture by this factor (e.g. 10 for "
                   "10x, 0.5 to shrink). At 1.0 the texture maps 1:1 to image "
                   "pixels. Applies to --dither texture.")
@click.option("--dither-softness", type=float, default=0.0, show_default=True,
              help="Soften the colour boundary in dither / dither-rgb modes. "
                   "0 = hard 1-bit edges; ~0.2-0.4 gives anti-aliased, smoothly "
                   "blended edges (e.g. cleaner halftone circles).")
@click.option("--dither-strength", type=float, default=1.0, show_default=True,
              help="Per-channel dither amplitude for --mode dither-rgb, scaled "
                   "by the palette spacing. ~1 is balanced; higher = grainier.")
@click.option("--prefer-smallest", is_flag=True,
              help="In dither mode, bias toward the darker of the two colours.")
@click.option("--hsv-weights", callback=_parse_triplet, default=None,
              metavar="H,S,V", help="Weights for the hsv metric (default 1,1,1).")
@click.option("--hsv-adjust", callback=_parse_triplet, default=None,
              metavar="H,S,V",
              help="Pre-shift hue (add) and scale sat/val (multiply) before "
                   "matching. Identity is 0,1,1.")
def main(input_image, output_image, palette_spec, mode, metric, smooth,
         max_colors, palette_range, dither_kind, dither_res, bayer_size,
         halftone_angle,
         dither_texture, dither_scale, dither_softness, dither_strength,
         prefer_smallest, hsv_weights, hsv_adjust):
    """Apply a colour PALETTE to an image.

    \b
    Examples:
      paletti in.png out.png -p palette.png
      paletti in.png out.png -p sweetie16.json -m dither --dither bayer
      paletti in.png out.png -p '[[26,28,44],[93,39,93]]' -m blend
    """
    try:
        pal = palette_mod.load(
            palette_spec, max_colors=max_colors, assume_range=palette_range
        )
    except (OSError, ValueError) as exc:
        raise click.ClickException(f"could not load palette: {exc}")

    if pal.shape[0] == 0:
        raise click.ClickException("palette has no colours")

    rgb, alpha = imageio.load_rgb(input_image)

    tex = None
    if dither_texture is not None:
        tex, _ = imageio.load_rgb(dither_texture)

    opts = core.Options(
        mode=mode,
        metric=metric,
        pre_blur=smooth,
        hsv_weights=hsv_weights or (1.0, 1.0, 1.0),
        hsv_adjust=hsv_adjust or (0.0, 1.0, 1.0),
        dither_kind=dither_kind,
        dither_res=dither_res,
        bayer_size=bayer_size,
        halftone_angle=halftone_angle,
        dither_scale=dither_scale,
        dither_softness=dither_softness,
        dither_strength=dither_strength,
        dither_texture=tex,
        prefer_smallest=prefer_smallest,
    )

    try:
        out = core.apply(rgb, pal, opts)
    except ValueError as exc:
        raise click.ClickException(str(exc))

    # The "factor" debug view has no meaningful alpha; keep it opaque.
    out_alpha = None if mode == "factor" else alpha
    imageio.save_rgb(output_image, out, out_alpha)

    click.echo(
        f"Applied {pal.shape[0]}-colour palette ({mode}/{metric}) "
        f"-> {output_image}"
    )


if __name__ == "__main__":
    main()
