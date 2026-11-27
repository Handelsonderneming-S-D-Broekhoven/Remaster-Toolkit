# SPDX-FileCopyrightText: 2026 Handelsonderneming S.D. Broekhoven, trading as ITEagle Europe
# SPDX-License-Identifier: GPL-3.0-only
"""Automated QA metrics (signals, not proof). Own implementations, CPU, numpy.

Perceptual metrics (LPIPS, DISTS) live in :mod:`remaster_toolkit.perceptual`
and are optional because they need torch and pretrained backbones.
"""

from __future__ import annotations

import numpy as np

from . import alpha as alpha_mod
from .pbr import luminance, srgb_to_linear
from .resample import downscale_area, resize


def _gauss1d(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    x = np.arange(size) - (size - 1) / 2
    g = np.exp(-(x**2) / (2 * sigma**2))
    return g / g.sum()


def _filt(x: np.ndarray, k: np.ndarray, wrap: bool = False) -> np.ndarray:
    r = len(k) // 2
    mode = "wrap" if wrap else "reflect"
    p = np.pad(x, r, mode=mode)
    h, w = x.shape
    tmp = np.zeros((h + 2 * r, w), dtype=np.float64)
    for i, kv in enumerate(k):
        tmp += kv * p[:, i:i + w]
    out = np.zeros((h, w), dtype=np.float64)
    for i, kv in enumerate(k):
        out += kv * tmp[i:i + h, :]
    return out


def ssim(x: np.ndarray, y: np.ndarray) -> float:
    """Mean SSIM (Wang et al. 2004; gaussian 11/1.5, K1=.01, K2=.03, L=1)."""
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    k = _gauss1d()
    c1, c2 = 0.01**2, 0.03**2
    mx, my = _filt(x, k), _filt(y, k)
    sxx = _filt(x * x, k) - mx * mx
    syy = _filt(y * y, k) - my * my
    sxy = _filt(x * y, k) - mx * my
    s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))
    return float(s.mean())


def luma_srgb(rgb: np.ndarray) -> np.ndarray:
    """Rec.709 weights on gamma-encoded values (common SSIM practice)."""
    return rgb[:, :, :3].astype(np.float64) @ np.array([0.2126, 0.7152, 0.0722])


def psnr(x: np.ndarray, y: np.ndarray) -> float:
    mse = float(np.mean((x.astype(np.float64) - y.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)


def _lab(rgb: np.ndarray) -> np.ndarray:
    lin = srgb_to_linear(rgb[:, :, :3])
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = lin @ m.T / np.array([0.95047, 1.0, 1.08883])
    d = 6 / 29
    f = np.where(xyz > d**3, np.cbrt(xyz), xyz / (3 * d * d) + 4 / 29)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def delta_e2000(rgb1: np.ndarray, rgb2: np.ndarray) -> np.ndarray:
    """CIEDE2000 per pixel (sRGB D65 input)."""
    L1, a1, b1 = np.moveaxis(_lab(rgb1), -1, 0)
    L2, a2, b2 = np.moveaxis(_lab(rgb2), -1, 0)
    C1, C2 = np.hypot(a1, b1), np.hypot(a2, b2)
    Cb = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt(Cb**7 / (Cb**7 + 25.0**7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    dLp = L2 - L1
    dCp = C2p - C1p
    dh = h2p - h1p
    dh = np.where(C1p * C2p == 0, 0, np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh)))
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dh) / 2)
    Lbp = (L1 + L2) / 2
    Cbp = (C1p + C2p) / 2
    hs = h1p + h2p
    hbp = np.where(C1p * C2p == 0, hs,
                   np.where(np.abs(h1p - h2p) <= 180, hs / 2,
                            np.where(hs < 360, (hs + 360) / 2, (hs - 360) / 2)))
    T = (1 - 0.17 * np.cos(np.radians(hbp - 30)) + 0.24 * np.cos(np.radians(2 * hbp))
         + 0.32 * np.cos(np.radians(3 * hbp + 6)) - 0.20 * np.cos(np.radians(4 * hbp - 63)))
    dtheta = 30 * np.exp(-(((hbp - 275) / 25) ** 2))
    Rc = 2 * np.sqrt(Cbp**7 / (Cbp**7 + 25.0**7))
    Sl = 1 + 0.015 * (Lbp - 50) ** 2 / np.sqrt(20 + (Lbp - 50) ** 2)
    Sc = 1 + 0.045 * Cbp
    Sh = 1 + 0.015 * Cbp * T
    Rt = -np.sin(np.radians(2 * dtheta)) * Rc
    return np.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2
                   + Rt * (dCp / Sc) * (dHp / Sh))


def seam_ratio(img: np.ndarray, wrap_u: bool = True, wrap_v: bool = True) -> float | None:
    """Wrap-boundary discontinuity relative to interior neighbour differences.

    For each wrapped axis: mean |first - last| line, divided by the mean
    absolute difference between adjacent interior lines. ~1 means the seam
    is statistically like the interior; returns the max over wrapped axes.
    """
    x = img[:, :, :3].astype(np.float64)
    vals = []
    if wrap_u:
        seam = np.abs(x[:, 0] - x[:, -1]).mean()
        inner = np.abs(np.diff(x, axis=1)).mean()
        vals.append(seam / max(inner, 1e-9))
    if wrap_v:
        seam = np.abs(x[0] - x[-1]).mean()
        inner = np.abs(np.diff(x, axis=0)).mean()
        vals.append(seam / max(inner, 1e-9))
    return float(max(vals)) if vals else None


def hist_bhattacharyya(a: np.ndarray, b: np.ndarray, bins: int = 64) -> float:
    """Mean over RGB channels of the Bhattacharyya distance of histograms."""
    ds = []
    for c in range(3):
        ha, _ = np.histogram(a[:, :, c], bins=bins, range=(0, 1))
        hb, _ = np.histogram(b[:, :, c], bins=bins, range=(0, 1))
        pa = ha / max(ha.sum(), 1)
        pb = hb / max(hb.sum(), 1)
        bc = np.sum(np.sqrt(pa * pb))
        ds.append(float(np.sqrt(max(0.0, 1 - bc))))
    return float(np.mean(ds))


def laplacian_var(img: np.ndarray) -> float:
    y = luma_srgb(img)
    p = np.pad(y, 1, mode="reflect")
    lap = p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] - 4 * p[1:-1, 1:-1]
    return float(lap.var())


def clip_fraction(img: np.ndarray) -> float:
    x = img[:, :, :3]
    return float(((x <= 0.5 / 255) | (x >= 1 - 0.5 / 255)).mean())


def is_pot(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def compare(out: np.ndarray, src: np.ndarray, wrap_u: bool, wrap_v: bool,
            alpha_kind: str = "none", cutout: float = 0.5,
            reference: np.ndarray | None = None) -> dict:
    """Compute all CPU metrics of an output against its source.

    ``out`` is at output resolution, ``src`` at source resolution; fidelity
    metrics are computed after an exact area downscale of ``out`` to the
    source size (avoids rewarding blur at high resolution).
    """
    sh, sw = src.shape[:2]
    oh, ow = out.shape[:2]
    mode = ("wrap" if wrap_v else "clamp", "wrap" if wrap_u else "clamp")
    down = downscale_area(out, sw, sh, mode)
    m: dict = {}
    m["dimensions"] = f"{ow}x{oh}"
    m["pot"] = bool(is_pot(ow) and is_pot(oh))
    m["aspect-preserved"] = bool(ow * sh == oh * sw)
    # With alpha, colour under (source-)transparent texels is irrelevant (and
    # deliberately changed by edge dilation): compare premultiplied by the
    # source alpha, and restrict colour statistics to visible texels.
    if alpha_kind != "none" and src.shape[2] == 4:
        w = src[:, :, 3:4].astype(np.float64)
        d_rgb, s_rgb = down[:, :, :3] * w, src[:, :, :3] * w
        vis = src[:, :, 3] >= cutout
        if not vis.any():
            vis = np.ones(src.shape[:2], bool)
    else:
        d_rgb, s_rgb = down[:, :, :3], src[:, :, :3]
        vis = np.ones(src.shape[:2], bool)
    m["ssim-downscaled"] = ssim(luma_srgb(d_rgb), luma_srgb(s_rgb))
    m["psnr-downscaled"] = psnr(d_rgb, s_rgb)
    # Colour drift = CIEDE2000 of low-passed images (gaussian sigma 1.5 source px):
    # measures colour/tone shift, not the texture detail SR is meant to add.
    k = _gauss1d(9, 1.5)
    lp = lambda x: np.stack([_filt(x[:, :, c], k, wrap=wrap_u or wrap_v) for c in range(3)], axis=2)
    if alpha_kind != "none" and src.shape[2] == 4:
        # alpha-weighted low-pass (weights = source alpha) so hidden colour never mixes in
        wa = src[:, :, 3:4].astype(np.float64)
        wsum = np.maximum(lp(np.repeat(wa, 3, axis=2)), 1e-6)
        de = delta_e2000(lp(down[:, :, :3] * wa) / wsum, lp(src[:, :, :3] * wa) / wsum)[vis]
    else:
        de = delta_e2000(lp(down[:, :, :3]), lp(src[:, :, :3]))[vis]
    m["deltaE2000-mean"] = float(de.mean())
    m["deltaE2000-p95"] = float(np.percentile(de, 95))
    m["deltaE2000-perpixel-mean"] = float(delta_e2000(down, src)[vis].mean())
    m["hist-bhattacharyya"] = hist_bhattacharyya(down[vis][:, None, :], src[vis][:, None, :])
    m["luminance-shift"] = float(luminance(down)[vis].mean() - luminance(src)[vis].mean())
    m["clip-fraction-delta"] = clip_fraction(out) - clip_fraction(src)
    if wrap_u or wrap_v:
        if alpha_kind != "none" and src.shape[2] == 4 and out.shape[2] == 4:
            pm = lambda x: x[:, :, :3] * x[:, :, 3:4]  # premultiplied: ignore hidden colour
            m["seam-ratio"] = seam_ratio(pm(out), wrap_u, wrap_v)
            m["seam-ratio-source"] = seam_ratio(pm(src), wrap_u, wrap_v)
        else:
            m["seam-ratio"] = seam_ratio(out, wrap_u, wrap_v)
            m["seam-ratio-source"] = seam_ratio(src, wrap_u, wrap_v)
    if reference is not None and reference.shape[:2] == out.shape[:2]:
        lv_ref = laplacian_var(reference)
        m["hf-energy-ratio"] = laplacian_var(out) / max(lv_ref, 1e-12)
    if alpha_kind != "none" and src.shape[2] == 4:
        sa = src[:, :, 3:4]
        oa = out[:, :, 3:4] if out.shape[2] == 4 else np.ones((oh, ow, 1), np.float32)
        src_up = resize(sa, ow, oh, "nearest", mode)
        m["alpha-iou"] = alpha_mod.iou(src_up >= cutout, oa >= cutout)
        m["alpha-mae"] = float(np.abs(downscale_area(oa, sw, sh, mode) - sa).mean())
        m["alpha-coverage-source"] = alpha_mod.coverage(sa, cutout)
        m["alpha-coverage-output"] = alpha_mod.coverage(oa, cutout)
    return m
