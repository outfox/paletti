"""End-to-end and unit tests for paletti."""

import json

import numpy as np
import pytest
from PIL import Image

from paletti import color, core, dither, imageio
from paletti import palette as palette_mod


# --- colour conversions ----------------------------------------------------

def test_rgb_hsv_roundtrip():
    rng = np.random.default_rng(0)
    rgb = rng.random((500, 3))
    back = color.hsv2rgb(color.rgb2hsv(rgb))
    assert np.abs(back - rgb).max() < 1e-9


def test_rgb2hsv_known_values():
    hsv = color.rgb2hsv(np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]))
    assert np.allclose(hsv[:, 0], [0.0, 1 / 3, 2 / 3])
    assert np.allclose(hsv[:, 1], 1.0)
    assert np.allclose(hsv[:, 2], 1.0)


# --- palette loading -------------------------------------------------------

def test_palette_hex():
    pal = palette_mod.from_json(["#ff0000", "00ff00"])
    assert np.allclose(pal, [[1, 0, 0], [0, 1, 0]])


def test_palette_255_ints():
    pal = palette_mod.from_json([[255, 0, 0], [0, 128, 0]])
    assert np.allclose(pal, [[1, 0, 0], [0, 128 / 255, 0]])


def test_palette_unit_floats():
    pal = palette_mod.from_json([[0.5, 0.0, 0.25]])
    assert np.allclose(pal, [[0.5, 0.0, 0.25]])


def test_palette_object_wrapper():
    pal = palette_mod.from_json({"colors": ["#000000", "#ffffff"]})
    assert np.allclose(pal, [[0, 0, 0], [1, 1, 1]])


def test_palette_bad_hex():
    with pytest.raises(ValueError):
        palette_mod.from_json(["nothex"])


def test_palette_from_image(tmp_path):
    colours = np.array([[10, 20, 30], [200, 100, 50]], dtype=np.uint8)
    # 3 of the first colour, 1 of the second -> frequency order is stable.
    px = np.array([colours[0], colours[0], colours[0], colours[1]], dtype=np.uint8)
    p = tmp_path / "pal.png"
    Image.fromarray(px[None, :, :], "RGB").save(p)
    pal = palette_mod.from_image(p)
    assert np.allclose(pal[0], colours[0] / 255.0)
    pal1 = palette_mod.from_image(p, max_colors=1)
    assert pal1.shape == (1, 3)


# --- dither ----------------------------------------------------------------

def test_bayer_matrix_properties():
    m = dither.bayer_matrix(4)
    assert m.shape == (4, 4)
    # 16 distinct threshold levels in [0, 1).
    assert len(np.unique(m)) == 16
    assert m.min() >= 0 and m.max() < 1


def test_dither_field_nearest_constant():
    f = dither.dither_field("nearest", 5, 7)
    assert f.shape == (5, 7) and np.all(f == 0.5)


def test_halftone_field_dot_polarity():
    # Axis-aligned grid: dot centres read 0, cell corners read 1.
    f = dither.halftone_field(32, 32, cell=16, angle_deg=0.0)
    assert f.shape == (32, 32)
    assert f.min() == pytest.approx(0.0)
    assert f.max() == pytest.approx(1.0)
    assert f[0, 0] == pytest.approx(0.0)          # dot centre
    assert f[8, 8] == pytest.approx(1.0)          # cell corner


def test_texture_field_1to1_tiling_at_scale_one():
    # At scale 1.0 the texture maps 1:1 onto the image (exact, no interpolation),
    # repeating to fill it. A 2x2 texture tiled into 4x4 repeats verbatim.
    src = np.array([[0.0, 1.0], [1.0, 0.0]])
    f = dither.texture_field(4, 4, src, scale=1.0)
    assert f.shape == (4, 4)
    expected = np.tile(src, (2, 2))
    assert np.array_equal(f, expected)


def test_texture_field_zoom_is_isotropic_on_nonsquare_image():
    # The zoom uses the same factor on both axes, so a square texture keeps its
    # aspect on a NON-square image -- a round feature never gets stretched to the
    # frame. Tile a 2x2 texture (bright texel at the origin) into a wide 4x12
    # image at scale 2: even output pixels land exactly on texels, and the bright
    # texel recurs with the SAME period (4 = 2 texels * scale 2) horizontally and
    # vertically, proving isotropy.
    src = np.array([[1.0, 0.0], [0.0, 0.0]])
    f = dither.texture_field(4, 12, src, scale=2.0)
    assert f.shape == (4, 12)
    assert list(np.where(np.isclose(f[:, 0], 1.0))[0]) == [0]          # period 4 down (image only 4 tall)
    assert list(np.where(np.isclose(f[0, :], 1.0))[0]) == [0, 4, 8]    # period 4 across -> same as vertical
    assert f[0, 0] == src[0, 0]  # origin is pinned


def test_texture_scale_enlarges_features():
    # A checkerboard texture: scaling up reduces the number of transitions
    # across a row (each cell becomes bigger).
    tex = np.indices((8, 8)).sum(axis=0) % 2
    f1 = dither.dither_field("texture", 64, 64, scale=1, texture=tex)
    f8 = dither.dither_field("texture", 64, 64, scale=8, texture=tex)

    def transitions(f):
        return int(np.sum(np.abs(np.diff((f[32] > 0.5).astype(int)))))

    assert transitions(f8) < transitions(f1)


def test_halftone_dots_grow_with_brightness():
    # Dither a dark->light ramp between black and white; dot coverage of the
    # lighter colour must increase monotonically across the ramp.
    h, w = 32, 256
    ramp = np.repeat(np.linspace(0, 1, w)[None, :, None], h, axis=0).repeat(3, axis=2)
    pal = np.array([[0, 0, 0], [1, 1, 1]], dtype=float)
    out = core.apply(ramp, pal, core.Options(
        mode="dither", dither_kind="halftone", dither_res=10))
    allowed = {(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)}
    assert set(map(tuple, np.unique(out.reshape(-1, 3), axis=0))) <= allowed
    white = out.sum(axis=2) > 1.5
    frac = [white[:, i * 32:(i + 1) * 32].mean() for i in range(8)]
    assert all(frac[i] <= frac[i + 1] + 1e-9 for i in range(7))


# --- core application ------------------------------------------------------

def _img():
    rng = np.random.default_rng(1)
    return rng.random((16, 24, 3))


def test_nearest_only_palette_colours():
    img = _img()
    pal = np.array([[0, 0, 0], [1, 1, 1], [1, 0, 0]], dtype=float)
    out = core.apply(img, pal, core.Options(mode="nearest"))
    flat = out.reshape(-1, 3)
    allowed = set(map(tuple, pal))
    assert set(map(tuple, np.unique(flat, axis=0))) <= allowed


def test_dither_only_palette_colours():
    img = _img()
    pal = np.array([[0, 0, 0], [1, 1, 1]], dtype=float)
    out = core.apply(img, pal, core.Options(mode="dither", dither_kind="bayer"))
    allowed = set(map(tuple, pal))
    assert set(map(tuple, np.unique(out.reshape(-1, 3), axis=0))) <= allowed


def test_gaussian_blur_reduces_noise_preserves_mean():
    rng = np.random.default_rng(3)
    img = np.clip(0.5 + 0.05 * rng.standard_normal((64, 64, 3)), 0, 1)
    blurred = core._gaussian_blur(img, 1.5)
    assert blurred.std() < img.std()
    # 'edge' padding means no border darkening -> mean stays put.
    assert blurred.mean() == pytest.approx(img.mean(), abs=1e-3)
    assert core._gaussian_blur(img, 0.0) is img  # no-op fast path


def test_pre_blur_reduces_speckle():
    # A noisy near-boundary flat field produces per-pixel pair flips; smoothing
    # the source should reduce the number of colour transitions.
    rng = np.random.default_rng(4)
    flat = np.clip(np.full((96, 96, 3), 0.5) + 0.03 * rng.standard_normal((96, 96, 3)), 0, 1)
    pal = np.array([[0.3, 0.3, 0.3], [0.7, 0.7, 0.7], [0.5, 0.5, 0.55]])

    def transitions(sigma):
        out = core.apply(flat, pal, core.Options(
            mode="dither", dither_kind="halftone", dither_res=12, pre_blur=sigma))
        return int(np.sum(np.abs(np.diff(out[..., 0], axis=1)) > 1e-6))

    assert transitions(1.5) < transitions(0.0)


def test_dither_softness_introduces_gradient():
    img = _img()
    pal = np.array([[0, 0, 0], [1, 1, 1]], dtype=float)
    sharp = core.apply(img, pal, core.Options(mode="dither", dither_softness=0.0))
    soft = core.apply(img, pal, core.Options(mode="dither", dither_softness=0.4))
    # Sharp stays 1-bit; soft introduces intermediate (in-between) colours.
    assert len(np.unique(sharp.reshape(-1, 3), axis=0)) <= len(pal)
    assert len(np.unique(soft.reshape(-1, 3), axis=0)) > len(pal)
    assert soft.min() >= 0.0 and soft.max() <= 1.0


def test_dither_rgb_palette_only_and_reduces_banding():
    pal = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                    [1, 1, 0], [1, 1, 1]], dtype=float)
    h, w = 48, 256
    xx = np.linspace(0, 1, w)
    img = np.repeat(np.stack([xx, xx * 0.5, 1 - xx], -1)[None], h, 0)

    near = core.apply(img, pal, core.Options(mode="nearest"))
    drgb = core.apply(img, pal, core.Options(mode="dither-rgb", dither_kind="bayer"))

    # Output is palette-only (each channel snaps after perturbation).
    allowed = set(map(tuple, pal))
    assert set(map(tuple, np.unique(drgb.reshape(-1, 3), axis=0))) <= allowed

    # Local average of the dithered result tracks the gradient far better than
    # the hard-quantised nearest result (i.e. it dissolves the banding).
    def avg_err(out):
        return float(np.mean(np.abs(core._gaussian_blur(out, 2.0) - img)))

    assert avg_err(drgb) < avg_err(near) * 0.5


def test_dither_rgb_uses_texture_channels_when_rgb():
    rng = np.random.default_rng(7)
    tex = rng.random((8, 8, 3))  # genuinely coloured: channels differ
    opts = core.Options(mode="dither-rgb", dither_kind="texture",
                        dither_texture=tex, dither_scale=1.0)
    field = core._make_field_rgb(opts, 8, 8)
    expected = dither.texture_field(8, 8, tex, scale=1.0).reshape(-1, 3)
    assert np.allclose(field, expected)
    assert not np.allclose(field[:, 0], field[:, 1])  # channels stay distinct


def test_dither_rgb_greyscale_texture_falls_back_to_rotation():
    rng = np.random.default_rng(8)
    gray3 = np.repeat(rng.random((8, 8))[..., None], 3, axis=2)
    opts = core.Options(mode="dither-rgb", dither_kind="texture",
                        dither_texture=gray3, dither_scale=1.0)
    field = core._make_field_rgb(opts, 8, 8)
    single = dither.dither_field("texture", 8, 8, texture=gray3[..., 0]).reshape(-1)
    rotated = (single[:, None] + np.array([0.0, 1 / 3, 2 / 3])[None]) % 1.0
    assert np.allclose(field, rotated)


def test_texture_field_preserves_channels():
    tex = np.dstack([np.zeros((4, 4)), np.ones((4, 4)), np.full((4, 4), 0.5)])
    f = dither.texture_field(8, 8, tex, scale=1.0)
    assert f.shape == (8, 8, 3)
    assert f[..., 0].max() == 0.0 and f[..., 1].min() == 1.0


def test_dither_rgb_softness_blends_edges():
    pal = np.array([[0, 0, 0], [1, 1, 1], [0.5, 0.5, 0.5]], dtype=float)
    img = np.repeat(np.linspace(0, 1, 256)[None, :, None], 48, 0).repeat(3, 2)
    hard = core.apply(img, pal, core.Options(
        mode="dither-rgb", dither_kind="halftone", dither_res=10, dither_softness=0.0))
    soft = core.apply(img, pal, core.Options(
        mode="dither-rgb", dither_kind="halftone", dither_res=10, dither_softness=0.4))
    # softness 0 stays palette-only; softness > 0 introduces blended edge colours.
    assert set(map(tuple, np.unique(hard.reshape(-1, 3), axis=0))) <= set(map(tuple, pal))
    assert len(np.unique(soft.reshape(-1, 3), axis=0)) > len(pal)
    assert soft.min() >= 0.0 and soft.max() <= 1.0


def test_dither_rgb_single_colour_palette():
    img = _img()
    out = core.apply(img, np.array([[0.4, 0.4, 0.4]]),
                     core.Options(mode="dither-rgb"))
    assert np.allclose(out.reshape(-1, 3), [0.4, 0.4, 0.4])


def test_mean_palette_gap():
    assert core._mean_palette_gap(np.array([[0.5, 0.5, 0.5]])) == 0.0
    pal = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
    assert core._mean_palette_gap(pal) == pytest.approx(1.0)


def test_blend_introduces_new_colours():
    img = _img()
    pal = np.array([[0, 0, 0], [1, 1, 1]], dtype=float)
    out = core.apply(img, pal, core.Options(mode="blend"))
    assert len(np.unique(out.reshape(-1, 3), axis=0)) > len(pal)


def test_single_colour_palette():
    img = _img()
    pal = np.array([[0.2, 0.4, 0.6]])
    out = core.apply(img, pal, core.Options(mode="dither"))
    assert np.allclose(out.reshape(-1, 3), pal)


def test_hsv_metric_runs():
    img = _img()
    pal = np.array([[0, 0, 0], [1, 1, 1], [1, 0, 0]], dtype=float)
    out = core.apply(img, pal, core.Options(metric="hsv", hsv_weights=(2, 1, 1)))
    assert out.shape == img.shape


def test_rgb2hsl_known_values():
    # Primaries share hue with HSV; at full saturation HSL lightness is 0.5.
    hsl = color.rgb2hsl(np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]))
    assert np.allclose(hsl[:, 0], [0.0, 1 / 3, 2 / 3])
    assert np.allclose(hsl[:, 1], 1.0)
    assert np.allclose(hsl[:, 2], 0.5)
    # Black, mid-grey, white: zero saturation, lightness tracks the grey level.
    grey = color.rgb2hsl(np.array([[0, 0, 0], [0.5, 0.5, 0.5], [1, 1, 1.0]]))
    assert np.allclose(grey[:, 1], 0.0)
    assert np.allclose(grey[:, 2], [0.0, 0.5, 1.0])


def test_rgb2oklab_reference_values():
    # Reference L,a,b from Ottosson's published sRGB examples.
    lab = color.rgb2oklab(np.array([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]]))
    assert np.allclose(lab[0], [1.0, 0.0, 0.0], atol=1e-3)  # white
    assert np.allclose(lab[1], [0.628, 0.225, 0.126], atol=1e-3)  # red


def test_rgb2luma_weights():
    luma = color.rgb2luma(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1.0]]))
    assert np.allclose(luma, [0.2126, 0.7152, 0.0722])


@pytest.mark.parametrize("metric", core.METRICS)
def test_every_metric_runs_and_snaps(metric):
    img = _img()
    pal = np.array([[0, 0, 0], [1, 1, 1], [1, 0, 0], [0, 0, 1]], dtype=float)
    out = core.apply(img, pal, core.Options(mode="nearest", metric=metric))
    assert out.shape == img.shape
    # nearest mode must only ever emit exact palette colours.
    flat = out.reshape(-1, 3)
    assert np.all(np.any(np.all(flat[:, None, :] == pal[None, :, :], axis=2), axis=1))


def test_default_metric_is_oklab():
    assert core.Options().metric == "oklab"


def test_invalid_metric_raises():
    with pytest.raises(ValueError):
        core.apply(_img(), np.array([[0, 0, 0.0]]), core.Options(metric="nope"))


def test_empty_palette_raises():
    with pytest.raises(ValueError):
        core.apply(_img(), np.empty((0, 3)), core.Options())


# --- image io --------------------------------------------------------------

def test_alpha_roundtrip(tmp_path):
    rng = np.random.default_rng(2)
    rgb = rng.random((8, 8, 3))
    alpha = np.zeros((8, 8))
    alpha[2:5, 2:5] = 1.0
    p = tmp_path / "a.png"
    imageio.save_rgb(p, rgb, alpha)
    rgb2, alpha2 = imageio.load_rgb(p)
    assert alpha2 is not None
    assert np.allclose(alpha2, alpha, atol=1 / 255)


def test_full_pipeline_via_load(tmp_path):
    img = (_img() * 255).astype(np.uint8)
    inp = tmp_path / "in.png"
    Image.fromarray(img, "RGB").save(inp)
    pal = palette_mod.load(json.dumps(["#000000", "#ffffff"]))
    rgb, alpha = imageio.load_rgb(inp)
    out = core.apply(rgb, pal, core.Options(mode="nearest"))
    assert out.shape == rgb.shape and alpha is None


# --- cli -------------------------------------------------------------------

def test_cli_default_output_name(tmp_path):
    from click.testing import CliRunner
    from paletti.cli import main

    inp = tmp_path / "photo.jpeg"
    Image.fromarray((_img() * 255).astype(np.uint8), "RGB").save(inp)

    result = CliRunner().invoke(main, [str(inp), "-p", '["#000000","#ffffff"]'])
    assert result.exit_code == 0, result.output
    # Written next to the input as paletti-<stem>.png, regardless of input ext.
    expected = tmp_path / "paletti-photo.png"
    assert expected.exists()
    assert str(expected) in result.output


def test_cli_explicit_output_name(tmp_path):
    from click.testing import CliRunner
    from paletti.cli import main

    inp = tmp_path / "in.png"
    Image.fromarray((_img() * 255).astype(np.uint8), "RGB").save(inp)
    out = tmp_path / "custom.png"

    result = CliRunner().invoke(main, [str(inp), str(out), "-p", '["#000000","#ffffff"]'])
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert not (tmp_path / "paletti-in.png").exists()


def _run_cli(tmp_path, *extra):
    from click.testing import CliRunner
    from paletti.cli import main

    inp = tmp_path / "in.png"
    Image.fromarray((_img() * 255).astype(np.uint8), "RGB").save(inp)
    out = tmp_path / "out.png"
    argv = [str(inp), str(out), "-p", '["#000000","#ffffff"]', *extra]
    return CliRunner().invoke(main, argv), out


def test_cli_modes_derived_from_flags(tmp_path):
    for extra in ([], ["--blend"], ["--dither", "bayer"],
                  ["--dither", "bayer", "--rgb"]):
        result, out = _run_cli(tmp_path, *extra)
        assert result.exit_code == 0, (extra, result.output)
        assert out.exists()
    # The reported mode reflects the derived selection.
    assert "(dither-rgb/" in _run_cli(tmp_path, "--dither", "bayer", "--rgb")[0].output
    assert "(blend/" in _run_cli(tmp_path, "--blend")[0].output


def test_cli_blend_and_dither_mutually_exclusive(tmp_path):
    result, _ = _run_cli(tmp_path, "--blend", "--dither", "bayer")
    assert result.exit_code != 0
    assert "mutually exclusive" in result.stderr


def test_cli_warns_on_unused_options(tmp_path):
    # --bayer without --dither has no effect -> warning on stderr.
    result, _ = _run_cli(tmp_path, "--bayer", "8")
    assert result.exit_code == 0, result.output
    assert "--bayer ignored" in result.stderr

    # --angle is only used by the halftone kind.
    result, _ = _run_cli(tmp_path, "--dither", "bayer", "--angle", "30")
    assert result.exit_code == 0, result.output
    assert "--angle ignored" in result.stderr

    # --hsv-weights is only used by the hsv metric.
    result, _ = _run_cli(tmp_path, "--hsv-weights", "2,1,1")
    assert result.exit_code == 0, result.output
    assert "--hsv-weights ignored" in result.stderr
