"""Command-line interface for paletti."""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np

from . import core, imageio, palette as palette_mod, svg as svg_mod
from .core import DITHER_KINDS, METRICS


class _EatAllOption(click.Option):
    """An option that greedily collects every following non-option token.

    Click options are otherwise fixed-arity, so this is what lets ``-p`` take a
    whole space-separated list -- ``-p 000 pal.json lospec.png`` -- instead of
    forcing one ``-p`` per source. Consumption stops at the next option-like token
    (one starting with ``-``), so later flags are untouched; it does, however,
    also swallow positional arguments that follow it, so ``-p`` is best placed
    after the input/output paths.
    """

    def add_to_parser(self, parser, ctx):
        super().add_to_parser(parser, ctx)
        for name in self.opts:
            parser_opt = parser._long_opt.get(name) or parser._short_opt.get(name)
            if parser_opt is None:
                continue
            consume_one = parser_opt.process

            def process(value, state, _consume_one=consume_one):
                _consume_one(value, state)
                while state.rargs:
                    nxt = state.rargs[0]
                    if nxt[:1] == "-" and nxt != "-":
                        break
                    _consume_one(state.rargs.pop(0), state)

            parser_opt.process = process
            break


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


def _warn_unused(ctx, *, mode, dither_kind, metric, palette_specs):
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

    # With mixed -p sources, --max-colors only bites if some source is an image
    # and --palette-range only if some source is JSON; flag each when no source
    # of its kind is present (e.g. a palette of bare colour tokens uses neither).
    # Only classify the sources when one of these options was actually given.
    if given("max_colors") or given("palette_range"):
        kinds = {palette_mod.source_kind(s) for s in palette_specs}
        if given("max_colors") and "image" not in kinds:
            warn("--max-colors", "only image palettes use it")
        if given("palette_range") and "json" not in kinds:
            warn("--palette-range", "only JSON palettes use it")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("input_image", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output_image", required=False, default=None,
                type=click.Path(dir_okay=False, path_type=Path))
@click.option("-p", "--palette", "palette_specs", cls=_EatAllOption,
              multiple=True, default=(), metavar="SPEC...",
              help="Palette source(s): an image path, a .json file, an inline "
                   "JSON array such as '[\"#1a1c2c\",\"#5d275d\"]', or a bare "
                   "hex/name colour ('000', 'lavender'). Repeatable and variadic "
                   "-- '-p 000 -p pal.json' or '-p 000 pal.json lospec.png "
                   "lavender'; every source is concatenated into one palette. "
                   "Place it after the image paths as it eats the values that "
                   "follow it.")
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
@click.option("--svg-scale", type=float, default=1.0, show_default=True,
              help="When the input is an SVG rasterized to a raster output, "
                   "render it this many times larger (e.g. 4 for 4x). Vector "
                   "SVGs have no inherent resolution, so bump this for a crisper "
                   "result. Ignored for non-SVG inputs and for .svg output.")
@click.pass_context
def main(ctx, input_image, output_image, palette_specs, blend,
         dither_kind, rgb, metric, blur, denoise, max_colors, palette_range,
         extra_half_brite, res, bayer, angle, texture, scale, antialias,
         prefer_smallest, hsv_weights, hsv_adjust, svg_scale):
    """Apply a colour PALETTE to an image.

    By default, each pixel snaps to its nearest palette colour. Use --blend for a
    smooth lerp between the two nearest, or --dither KIND to ordered-dither
    between them (add --rgb to dither each channel independently).

    If OUTPUT_IMAGE is omitted, the result is written next to the input as
    "paletti-<input-name>.png" (or ".svg" for an SVG input).

    SVG inputs are supported two ways, chosen by the output extension: a raster
    output rasterizes the SVG and runs the full pixel pipeline, while a .svg
    output snaps each colour the SVG uses to the nearest palette colour and
    keeps it vector (dither does not apply there).

    \b
    Examples:
      paletti in.png -p palette.png                    # -> paletti-in.png
      paletti in.png out.png -p sweetie16.json --dither bayer --bayer 8
      paletti in.png out.png -p '[[26,28,44],[93,39,93]]' --blend
      paletti in.png out.png -p 000 pal.json lospec.png lavender  # mixed sources
      paletti in.png out.png -p base.json FFFFFF 000000  # augment a palette
      paletti logo.svg out.png -p sweetie16.json --svg-scale 8  # rasterize
      paletti logo.svg out.svg -p sweetie16.json       # recolour, keep vector
    """
    if blend and dither_kind:
        raise click.ClickException("--blend and --dither are mutually exclusive")
    if dither_kind:
        mode = "dither-rgb" if rgb else "dither"
    elif blend:
        mode = "blend"
    else:
        mode = "nearest"

    if not palette_specs:
        raise click.UsageError(
            "no palette given: provide a source with -p/--palette"
        )

    _warn_unused(ctx, mode=mode, dither_kind=dither_kind, metric=metric,
                 palette_specs=palette_specs)

    # An SVG input round-trips to SVG by default; everything else defaults to
    # PNG. The output extension then picks the path: a .svg output recolours the
    # vectors in place, any raster extension rasterizes and runs the pixel
    # pipeline. (A non-SVG default keeps existing behaviour unchanged.)
    input_is_svg = input_image.suffix.lower() == ".svg"
    if output_image is None:
        suffix = ".svg" if input_is_svg else ".png"
        output_image = input_image.with_name(f"paletti-{input_image.stem}{suffix}")

    produce_svg = output_image.suffix.lower() == ".svg"
    if produce_svg and not input_is_svg:
        raise click.ClickException("SVG (vector) output requires an SVG input")
    if produce_svg and mode in ("dither", "dither-rgb"):
        click.echo("warning: dither has no meaning for vector SVG output; "
                   "snapping to nearest instead", err=True)
        mode = "nearest"
    if (ctx.get_parameter_source("svg_scale")
            == click.core.ParameterSource.COMMANDLINE
            and (produce_svg or not input_is_svg)):
        why = "vector .svg output" if produce_svg else "input is not an SVG"
        click.echo(f"warning: --svg-scale ignored ({why})", err=True)

    # Build the palette from every -p source, in order.
    parts = []
    for spec in palette_specs:
        try:
            parts.append(palette_mod.load(
                spec, max_colors=max_colors, assume_range=palette_range
            ))
        except (OSError, ValueError) as exc:
            raise click.ClickException(f"could not load palette {spec!r}: {exc}")
    pal = np.vstack(parts)

    if pal.shape[0] == 0:
        raise click.ClickException("palette has no colours")

    if extra_half_brite:
        pal = palette_mod.half_brite(pal)

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
        if produce_svg:
            # Snap the SVG's own colours onto the palette, keeping it vector.
            svg_mod.recolor(input_image, output_image, pal, opts)
        else:
            image_rgb, alpha = imageio.load_rgb(input_image, svg_scale=svg_scale)
            out = core.apply(image_rgb, pal, opts)
            imageio.save_rgb(output_image, out, alpha)
    except (OSError, ValueError) as exc:
        raise click.ClickException(str(exc))

    click.echo(
        f"Applied {pal.shape[0]}-colour palette ({mode}/{metric}) "
        f"-> {output_image}"
    )


if __name__ == "__main__":
    main()
