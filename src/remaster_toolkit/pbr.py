# SPDX-FileCopyrightText: 2026 Handelsonderneming S.D. Broekhoven, trading as ITEagle Europe
# SPDX-License-Identifier: GPL-3.0-only
"""Classical (non-ML) PBR derivation baseline: luminance -> height -> normal.

These maps are *inferred, not measured*. Deterministic on CPU; wrap-aware.
Normal maps are tangent space, encoded ``n * 0.5 + 0.5``; convention is
OpenGL (+Y up) unless ``directx`` is requested (green flipped).
"""

from __future__ import annotations

import numpy as np

from .resample import resize

LUMA = np.array([0.2126, 0.7152, 0.0722])


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x.astype(np.float64), 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def luminance(rgb: np.ndarray, srgb: bool = True) -> np.ndarray:
    lin = srgb_to_linear(rgb[:, :, :3]) if srgb else rgb[:, :, :3].astype(np.float64)
    return lin @ LUMA


def _blur(x: np.ndarray, radius: int, wrap: bool) -> np.ndarray:
    """Separable box blur (radius r, applied 3x ~ gaussian)."""
    mode = "wrap" if wrap else "reflect"
    out = x.astype(np.float64)
    for _ in range(3):
        for axis in (0, 1):
            p = np.pad(out, [(radius, radius) if a == axis else (0, 0) for a in range(out.ndim)],
                       mode=mode)
            c = np.cumsum(p, axis=axis, dtype=np.float64)
            c = np.concatenate([np.zeros_like(np.take(c, [0], axis=axis)), c], axis=axis)
            n = out.shape[axis]
            hi = np.take(c, np.arange(2 * radius + 1, 2 * radius + 1 + n), axis=axis)
            lo = np.take(c, np.arange(0, n), axis=axis)
            out = (hi - lo) / (2 * radius + 1)
    return out


def height_from_albedo(rgb: np.ndarray, wrap: bool, highpass_radius: int = 0,
                       srgb: bool = True) -> np.ndarray:
    """Height in [0,1] from luminance; optional low-frequency removal."""
    h = luminance(rgb, srgb)
    if highpass_radius > 0:
        h = h - _blur(h, highpass_radius, wrap) + 0.5
    lo, hi = np.percentile(h, 0.5), np.percentile(h, 99.5)
    h = (h - lo) / max(hi - lo, 1e-6)
    return np.clip(h, 0, 1).astype(np.float32)[:, :, None]


def normal_from_height(height: np.ndarray, strength: float, wrap: bool,
                       convention: str = "opengl-y-up") -> np.ndarray:
    """Scharr gradient -> tangent-space normal, encoded to [0,1] RGB."""
    hgt = height[:, :, 0].astype(np.float64)
    p = np.pad(hgt, 1, mode="wrap" if wrap else "reflect")
    # Scharr kernels
    dx = (3 * (p[:-2, 2:] - p[:-2, :-2]) + 10 * (p[1:-1, 2:] - p[1:-1, :-2])
          + 3 * (p[2:, 2:] - p[2:, :-2])) / 32.0
    dy = (3 * (p[2:, :-2] - p[:-2, :-2]) + 10 * (p[2:, 1:-1] - p[:-2, 1:-1])
          + 3 * (p[2:, 2:] - p[:-2, 2:])) / 32.0
    nx = -dx * strength
    ny = dy * strength  # image rows go down; +Y up in OpenGL convention
    if convention == "directx-y-down":
        ny = -ny
    nz = np.ones_like(nx)
    n = np.stack([nx, ny, nz], axis=2)
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    return (n * 0.5 + 0.5).astype(np.float32)


def normalize_normals(enc: np.ndarray) -> np.ndarray:
    n = enc[:, :, :3].astype(np.float64) * 2 - 1
    n[:, :, 2] = np.maximum(n[:, :, 2], 0.0)
    ln = np.linalg.norm(n, axis=2, keepdims=True)
    n = np.where(ln > 1e-8, n / np.maximum(ln, 1e-8), np.array([0, 0, 1.0]))
    return (n * 0.5 + 0.5).astype(np.float32)


def normal_validity(enc: np.ndarray, quant_tol: float = 0.02) -> dict:
    """Fraction of texels with |n| within tolerance of 1 and z >= 0."""
    n = enc[:, :, :3].astype(np.float64) * 2 - 1
    ln = np.linalg.norm(n, axis=2)
    unit = np.abs(ln - 1) <= quant_tol + 2.0 / 255 * np.sqrt(3)
    zpos = n[:, :, 2] >= -1.0 / 255
    return {"unitFraction": float(unit.mean()), "zPositiveFraction": float(zpos.mean())}


def resample_normals(enc: np.ndarray, width: int, height: int, filt: str, mode: str) -> np.ndarray:
    """Linear-space normal resample followed by renormalisation."""
    return normalize_normals(resize(enc[:, :, :3], width, height, filt, mode))
