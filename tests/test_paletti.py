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


def test_resample_dimensions_and_range():
    src = np.array([[0.0, 1.0], [1.0, 0.0]])
    up = dither._resample(src, 8)
    assert up.shape == (16, 16)
    assert 0.0 <= up.min() and up.max() <= 1.0
    down = dither._resample(np.ones((40, 40)) * 0.5, 0.25)
    assert down.shape == (10, 10)
    assert dither._resample(src, 1.0) is src  # no-op fast path


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


def test_blend_introduces_new_colours():
    img = _img()
    pal = np.array([[0, 0, 0], [1, 1, 1]], dtype=float)
    out = core.apply(img, pal, core.Options(mode="blend"))
    assert len(np.unique(out.reshape(-1, 3), axis=0)) > len(pal)


def test_factor_is_greyscale():
    img = _img()
    pal = np.array([[0, 0, 0], [1, 1, 1]], dtype=float)
    out = core.apply(img, pal, core.Options(mode="factor")).reshape(-1, 3)
    assert np.allclose(out[:, 0], out[:, 1]) and np.allclose(out[:, 1], out[:, 2])


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
