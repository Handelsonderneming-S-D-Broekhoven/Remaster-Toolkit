# SPDX-FileCopyrightText: 2026 Handelsonderneming S.D. Broekhoven, trading as ITEagle Europe
# SPDX-License-Identifier: GPL-3.0-only
import numpy as np
import pytest

from conftest import periodic_texture, ramp_texture
from remaster_toolkit import qa, tiling
from remaster_toolkit.resample import downscale_area, resize, weight_matrix


def blur_zero_pad_x2(x):
    """Stand-in 'network': 5x5 box blur with ZERO padding (not circular), then
    nearest x2. Receptive field radius 2 (source pixels)."""
    p = np.pad(x, ((2, 2), (2, 2), (0, 0)))
    h, w = x.shape[:2]
    acc = sum(p[dy:dy + h, dx:dx + w] for dy in range(5) for dx in range(5)) / 25.0
    return np.repeat(np.repeat(acc, 2, axis=0), 2, axis=1).astype(np.float32)


def blur_circular_x2(x):
    p = np.pad(x, ((2, 2), (2, 2), (0, 0)), mode="wrap")
    h, w = x.shape[:2]
    acc = sum(p[dy:dy + h, dx:dx + w] for dy in range(5) for dx in range(5)) / 25.0
    return np.repeat(np.repeat(acc, 2, axis=0), 2, axis=1).astype(np.float32)


def test_wrap_padding_equals_circular_network():
    x = periodic_texture(32)
    truth = blur_circular_x2(x)
    got = tiling.apply_tiled(blur_zero_pad_x2, x, 2, pad_px=2, edge="wrap")
    np.testing.assert_allclose(got, truth, atol=1e-6)
    naive = blur_zero_pad_x2(x)
    assert abs(qa.seam_ratio(got) - qa.seam_ratio(truth)) < 1e-6
    assert qa.seam_ratio(naive) > 1.5 * qa.seam_ratio(got)


def test_tiled_with_overlap_matches_untiled():
    x = periodic_texture(48, seed=3)
    whole = tiling.apply_tiled(blur_zero_pad_x2, x, 2, pad_px=4, edge="wrap")
    tiled = tiling.apply_tiled(blur_zero_pad_x2, x, 2, tile=24, overlap=8, pad_px=4, edge="wrap")
    np.testing.assert_allclose(tiled, whole, atol=1e-5)


def test_insufficient_overlap_shows_tile_seams():
    x = periodic_texture(48, seed=3)
    whole = tiling.apply_tiled(blur_zero_pad_x2, x, 2, pad_px=4, edge="wrap")
    tiled = tiling.apply_tiled(blur_zero_pad_x2, x, 2, tile=24, overlap=0, pad_px=4, edge="wrap",
                               blend="none")
    assert np.abs(tiled - whole).max() > 1e-3


def test_seam_ratio_calibration():
    assert qa.seam_ratio(periodic_texture(64)) < 1.5
    assert qa.seam_ratio(ramp_texture(64)) > 10


def test_seam_fix_repairs_wrap_seam():
    x = ramp_texture(64)
    fixed = tiling.seam_fix(x, 12)
    assert qa.seam_ratio(fixed) < 1.5 < qa.seam_ratio(x)
    np.testing.assert_allclose(fixed[24:40, 24:40], x[24:40, 24:40], atol=1e-6)  # interior kept


@pytest.mark.parametrize("filt", ["bilinear", "bicubic", "lanczos3", "area", "nearest"])
@pytest.mark.parametrize("mode", ["wrap", "clamp", "reflect"])
def test_weights_normalised(filt, mode):
    for n_in, n_out in ((16, 64), (64, 16), (17, 23)):
        w = weight_matrix(n_in, n_out, filt, mode)
        np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-12)


def test_wrap_resample_keeps_texture_seamless():
    x = periodic_texture(32)
    up = resize(x, 128, 128, "lanczos3", "wrap", clip=False)
    assert qa.seam_ratio(up) < 1.5
    clamp = resize(x, 128, 128, "lanczos3", "clamp", clip=False)
    assert qa.seam_ratio(clamp) > qa.seam_ratio(up)


def test_area_downscale_inverts_nearest_upscale():
    x = periodic_texture(16)
    up = resize(x, 64, 64, "nearest", "clamp")
    np.testing.assert_allclose(downscale_area(up, 16, 16), x, atol=1e-6)


def test_resample_deterministic_bytes():
    x = periodic_texture(32, seed=9)
    a = resize(x, 77, 51, "lanczos3", "reflect")
    b = resize(x.copy(), 77, 51, "lanczos3", "reflect")
    assert a.tobytes() == b.tobytes()
