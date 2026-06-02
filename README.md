# paletti

Apply colour palettes to images from the command line.

`paletti` is a Python port of the palette / dithering shaders from the
`palette-shader-2` Godot project. For each pixel it finds the two nearest
palette colours and then snaps, blends, or ordered-dithers between them.

## Install / run

This is a [uv](https://docs.astral.sh/uv/) project:

```sh
uv sync
uv run paletti --help
```

## Usage

```sh
paletti INPUT OUTPUT -p PALETTE [options]
```

The palette (`-p` / `--palette`) can be:

- **an image** — its distinct colours become the palette
  (`-p palette.png`, optionally `--max-colors 16`);
- **a JSON file** — `-p sweetie16.json`;
- **an inline JSON array** — `-p '["#1a1c2c","#5d275d"]'`.

JSON palettes accept hex strings (`"#1a1c2c"` or `"1a1c2c"`), `0..255` integer
triples (`[26, 28, 44]`), or `0..1` float triples (`[0.1, 0.11, 0.17]`). The
numeric range is detected automatically; override with `--palette-range`.

### Modes (`-m` / `--mode`)

| mode      | result                                                          |
|-----------|----------------------------------------------------------------|
| `nearest` | snap each pixel to the closest palette colour (default)        |
| `second`  | use the second-closest colour                                  |
| `blend`   | smooth lerp between the two nearest colours                    |
| `dither`  | ordered dither between the two nearest colours (1-bit edges, or soften with `--dither-softness`) |
| `dither-rgb` | ordered dither each RGB channel independently, then snap to the palette (dissolves banding; great with `--dither bayer` or a blue-noise `--dither texture`). With an RGB `--dither texture` each colour channel drives the matching image channel; a greyscale texture is reused with a 1/3 phase shift per channel. |
| `factor`  | debug view of the blend factor as greyscale                    |

### Examples

```sh
# Quantise to a palette extracted from an image
paletti photo.png out.png -p lospec-palette.png

# Dither against a 16-colour palette using an 8x8 Bayer matrix
paletti photo.png out.png -p sweetie16.json -m dither --dither bayer --bayer-size 8

# Halftone / screentone dots (classic 45-degree grid, 8px dot spacing)
paletti photo.png out.png -p sweetie16.json -m dither --dither halftone --dither-res 8

# Tile an arbitrary dither texture, scaled up 10x
paletti photo.png out.png -p sweetie16.json -m dither \
    --dither texture --dither-texture screentone.png --dither-scale 10

# Smooth two-tone blending with an inline palette
paletti photo.png out.png -p '[[26,28,44],[244,244,244]]' -m blend

# Per-channel ordered dithering to dissolve banding (Bayer or blue-noise)
paletti photo.png out.png -p sweetie16.json -m dither-rgb --dither bayer --bayer-size 8
paletti photo.png out.png -p sweetie16.json -m dither-rgb --dither texture --dither-texture bluenoise.png

# Match in HSV space, weighting hue twice as heavily
paletti photo.png out.png -p sweetie16.json --metric hsv --hsv-weights 2,1,1
```

### Other options

- `--smooth SIGMA` — Gaussian-blur the source (sigma in pixels) before
  palettizing. Matching is per-pixel, so source noise / JPEG blocking / faint
  gradients near a palette-colour boundary flip the chosen colours and show up
  as sharp pixel-sized speckle. A small pre-blur (try `0.5`-`2`) makes the
  selection spatially coherent and cleans that up while leaving the dither
  pattern intact.
- `--metric {rgb,hsv}` — colour-distance metric used for matching.
- `--hsv-adjust H,S,V` — pre-shift hue (add) and scale saturation/value
  (multiply) before matching; identity is `0,1,1`.
- `--dither {nearest,sine,bayer,halftone,texture}`, `--dither-res`,
  `--bayer-size`, `--halftone-angle`, `--dither-texture` — control the dither
  pattern used by `--mode dither`. `halftone` reproduces the Godot project's
  "Screentone" pattern as procedural dots: `--dither-res` sets the dot spacing
  in pixels (try 6-12) and `--halftone-angle` rotates the grid (`45` = classic
  screentone, `0` = an axis-aligned square grid). `texture` tiles an arbitrary
  image (e.g. the original `screentonesdf.png`) via `--dither-texture`, and
  `--dither-scale` zooms that tiled texture (e.g. `10` for 10x, `0.5` to
  shrink). The texture is laid over the image at a 1:1 pixel ratio and repeated
  to fill it, so `--dither-scale 1.0` is an exact 1:1 mapping; other values
  zoom the tiled field about the origin via seamless bilinear sampling.
- `--dither-softness` — by default `dither` and `dither-rgb` pick one palette
  colour per pixel (hard 1-bit edges). A value like `0.2`-`0.4` smoothstep-blends
  a gradient band across the colour boundary, anti-aliasing the pattern (e.g.
  smooth halftone circles); `0` keeps the original sharp look.
- `--dither-strength` — per-channel dither amplitude for `--mode dither-rgb`,
  auto-scaled by the palette spacing (`~1` is balanced, higher is grainier).
- `--prefer-smallest` — in dither mode, bias toward the darker of the two
  colours.

Transparency in the source image is preserved.

## Project layout

```
src/paletti/
  cli.py        # click command-line interface
  core.py       # the shader port: two-nearest match + mode rendering
  color.py      # vectorised rgb<->hsv (ports rgb2hsv / hsv2rgb)
  dither.py     # ordered-dither value sources (nearest/sine/bayer/texture)
  palette.py    # load palettes from images or JSON
  imageio.py    # image load/save with alpha preservation
```
