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
| `dither`  | ordered dither between the two nearest colours                 |
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

# Match in HSV space, weighting hue twice as heavily
paletti photo.png out.png -p sweetie16.json --metric hsv --hsv-weights 2,1,1
```

### Other options

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
  `--dither-scale` resizes that texture before tiling (e.g. `10` for 10x, `0.5`
  to shrink) using seamless bilinear resampling.
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
