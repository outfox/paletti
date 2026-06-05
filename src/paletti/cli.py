"""Command-line interface for paletti."""

from __future__ import annotations

from pathlib import Path

import click

from . import core, imageio, palette as palette_mod
from .core import DITHER_KINDS, METRICS


def _parse_triplet(_0, _1, value):
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


def _warn_unused(ctx, *, mode, dither_kind, metric, palette_spec):
    """Warn (on stderr) for each explicitly-set option the run will ignore.

    Options silently flowed into ``core.Options`` regardless of mode before; now
    anything the active configuration won't consume is flagged so the user knows
    their flag had no effect, rather than wondering why nothing changed.
    """
    def given(parameter_name: str) -> bool:
        return ctx.get_parameter_source(parameter_name) == click.core.ParameterSource.COMMANDLINE

    def warn(parameter_flag: str, why: str) -> None:
        click.echo(f"warning: {parameter_flag} ignored ({why})", err=True)

    dithering = mode in ("dither", "dither-rgb")

    if not dithering:
        no_dither = "no --dither"
        for name, flag in (("res", "--res"), ("bayer", "--bayer"),
                           ("angle", "--angle"), ("texture", "--texture"),
                           ("scale", "--scale"), ("softness", "--antialias"),
                           ("prefer_smallest", "--prefer-smallest"),
                           ("rgb", "--rgb")):
            if given(name):
                warn(flag, no_dither)
    else:
        if given("bayer") and dither_kind != "bayer":
            warn("--bayer", "only --dither bayer uses it")
        if given("angle") and dither_kind != "halftone":
            warn("--angle", "only --dither halftone uses it")
        if given("texture") and dither_kind != "texture":
            warn("--texture", "only --dither texture uses it")
        if given("scale") and dither_kind != "texture":
            warn("--scale", "only --dither texture uses it")
        if given("prefer_smallest") and mode != "dither":
            warn("--prefer-smallest", "only plain dither uses it, not --rgb")

    if given("hsv_weights") and metric != "hsv":
        warn("--hsv-weights", "only --metric hsv uses it")

    if palette_mod.is_json_spec(palette_spec):
        if given("max_colors"):
            warn("--max-colors", "only image palettes use it")
    else:
        if given("palette_range"):
            warn("--palette-range", "only JSON palettes use it")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("input_image", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output_image", required=False, default=None,
                type=click.Path(dir_okay=False, path_type=Path))
@click.option("-p", "--palette", "palette_spec", required=True, metavar="SPEC",
              help="Palette source: an image path, a .json file, or an inline "
                   "JSON array such as '[\"#1a1c2c\",\"#5d275d\"]'.")
@click.option("--blend", is_flag=True,
              help="Smoothly lerp between the two nearest palette colours "
                   "instead of snapping to the closest (the default).")
@click.option("--dither", "dither_kind", type=click.Choice(DITHER_KINDS),
              default=None,
              help="Ordered-dither between the two nearest colours using this "
                   "pattern (omit to just snap). Add --rgb to dither each RGB "
                   "channel independently.")
@click.option("--rgb", is_flag=True,
              help="With --dither, dither each RGB channel independently before "
                   "snapping (dissolves banding).")
@click.option("--metric", type=click.Choice(METRICS), default="oklab",
              show_default=True,
              help="Colour-distance metric for matching: oklab (perceptual), "
                   "rgb (Euclidean), hsl/hsv (cylindrical), hue or luma "
                   "(single-axis).")
@click.option("--blur", type=float, default=0.0, show_default=True,
              metavar="SIGMA",
              help="Gaussian-blur the source by this sigma (in pixels) before "
                   "palettizing. Suppresses sharp pixel-sized artifacts from "
                   "source noise; try 0.5-2.")
@click.option("--denoise", type=float, default=0.0, show_default=True,
              metavar="STRENGTH",
              help="Edge-preserving bilateral denoise of the source before "
                   "palettizing. Smooths flat regions while keeping colour edges "
                   "(unlike --blur); STRENGTH is the colour sigma in [0,1] units "
                   "(try 0.05-0.3). Requires scikit-image; slower than --blur.")
@click.option("--max-colors", type=int, default=None,
              help="When importing a palette from an image, keep only the N "
                   "most frequent colours.")
@click.option("--palette-range", type=click.Choice(["auto", "unit", "255"]),
              default="auto", show_default=True,
              help="How to interpret numeric JSON palette values.")
@click.option("-ehb", "--extra-half-brite", "extra_half_brite", is_flag=True,
              help="Double the palette by adding a half-brightness copy of every "
                   "colour (Amiga Extra-Half-Brite) before matching.")
@click.option("--res", type=float, default=2.0, show_default=True,
              help="Dither cell size in pixels / pattern frequency. For "
                   "--dither halftone this is the dot spacing (try 6-12).")
@click.option("--bayer", type=int, default=4, show_default=True,
              help="Bayer matrix size (power of two) for --dither bayer.")
@click.option("--angle", type=float, default=45.0, show_default=True,
              metavar="DEG", help="Grid rotation for --dither halftone "
                                   "(45 = classic screentone, 0 = square grid).")
@click.option("--texture", type=click.Path(exists=True, dir_okay=False),
              default=None, help="Image to tile as the dither pattern "
                                 "(for --dither texture).")
@click.option("--scale", type=float, default=1.0, show_default=True,
              help="Zoom the tiled dither texture by this factor (e.g. 10 for "
                   "10x, 0.5 to shrink). At 1.0 the texture maps 1:1 to image "
                   "pixels. The zoom is isotropic, so the pattern keeps its "
                   "aspect ratio on any frame shape. Applies to --dither texture.")
@click.option("--antialias", type=float, default=0.0, show_default=True,
              help="Anti-alias dithered edges (e.g. halftone/texture dots). 0 = "
                   "hard 1-bit edges. For --dither it is the smoothstep blend "
                   "width across the A/B boundary (~1 = across the whole dot). "
                   "For --rgb the crisp per-channel result is on-palette, so its "
                   "edges are instead softened in the render by a blur that grows "
                   "with the value (try ~0.5-1.5). Higher = softer.")
@click.option("--prefer-smallest", is_flag=True,
              help="When dithering, bias toward the darker of the two colours.")
@click.option("--hsv-weights", callback=_parse_triplet, default=None,
              metavar="H,S,V", help="Weights for the hsv metric (default 1,1,1).")
@click.option("--hsv-adjust", callback=_parse_triplet, default=None,
              metavar="H,S,V",
              help="Pre-shift hue (add) and scale sat/val (multiply) before "
                   "matching. Identity is 0,1,1.")
@click.pass_context
def main(ctx, input_image, output_image, palette_spec, blend, dither_kind, rgb,
         metric, blur, denoise, max_colors, palette_range, extra_half_brite,
         res, bayer, angle, texture, scale, antialias, prefer_smallest,
         hsv_weights, hsv_adjust):
    """Apply a colour PALETTE to an image.

    By default, each pixel snaps to its nearest palette colour. Use --blend for a
    smooth lerp between the two nearest, or --dither KIND to ordered-dither
    between them (add --rgb to dither each channel independently).

    If OUTPUT_IMAGE is omitted, the result is written next to the input as
    "paletti-<input-name>.png".

    \b
    Examples:
      paletti in.png -p palette.png                    # -> paletti-in.png
      paletti in.png out.png -p sweetie16.json --dither bayer --bayer 8
      paletti in.png out.png -p '[[26,28,44],[93,39,93]]' --blend
    """
    if blend and dither_kind:
        raise click.ClickException("--blend and --dither are mutually exclusive")
    if dither_kind:
        mode = "dither-rgb" if rgb else "dither"
    elif blend:
        mode = "blend"
    else:
        mode = "nearest"

    _warn_unused(ctx, mode=mode, dither_kind=dither_kind, metric=metric,
                 palette_spec=palette_spec)

    if output_image is None:
        output_image = input_image.with_name(f"paletti-{input_image.stem}.png")

    try:
        pal = palette_mod.load(
            palette_spec, max_colors=max_colors, assume_range=palette_range
        )
    except (OSError, ValueError) as exc:
        raise click.ClickException(f"could not load palette: {exc}")

    if pal.shape[0] == 0:
        raise click.ClickException("palette has no colours")

    if extra_half_brite:
        pal = palette_mod.half_brite(pal)

    image_rgb, alpha = imageio.load_rgb(input_image)

    tex = None
    if texture is not None:
        tex, _ = imageio.load_rgb(texture)

    # Defaults for the optional triplets live on core.Options; only override when
    # the user actually supplied a value.
    extra = {}
    if hsv_weights is not None:
        extra["hsv_weights"] = hsv_weights
    if hsv_adjust is not None:
        extra["hsv_adjust"] = hsv_adjust

    opts = core.Options(
        mode=mode,
        metric=metric,
        pre_blur=blur,
        denoise=denoise,
        dither_kind=dither_kind or "bayer",
        dither_res=res,
        bayer_size=bayer,
        halftone_angle=angle,
        dither_scale=scale,
        dither_softness=antialias,
        dither_texture=tex,
        prefer_smallest=prefer_smallest,
        **extra,
    )

    try:
        out = core.apply(image_rgb, pal, opts)
    except ValueError as exc:
        raise click.ClickException(str(exc))

    imageio.save_rgb(output_image, out, alpha)

    click.echo(
        f"Applied {pal.shape[0]}-colour palette ({mode}/{metric}) "
        f"-> {output_image}"
    )


if __name__ == "__main__":
    main()
