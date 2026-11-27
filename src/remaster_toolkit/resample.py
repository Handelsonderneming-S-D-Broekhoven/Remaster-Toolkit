# SPDX-FileCopyrightText: 2026 Handelsonderneming S.D. Broekhoven, trading as ITEagle Europe
# SPDX-License-Identifier: GPL-3.0-only
"""Deterministic, edge-mode-aware separable resampling (CPU, numpy).

Each axis is resampled with an explicit (out x in) weight matrix, so the
boundary rule is exact: ``wrap`` treats the image as periodic (tileable
textures stay seamless), ``clamp`` replicates edges, ``reflect`` mirrors.
Computed in float64, returned as float32.
"""

from __future__ import annotations

import math

import numpy as np

FILTERS = ("nearest", "bilinear", "bicubic", "lanczos3", "area")


def _kernel(name: str):
    if name == "bilinear":
        return 1.0, lambda x: np.maximum(0.0, 1.0 - np.abs(x))
    if name == "bicubic":  # Keys, a = -0.5 (Catmull-Rom)
        a = -0.5

        def k(x):
            x = np.abs(x)
            return np.where(x < 1, (a + 2) * x**3 - (a + 3) * x**2 + 1,
                            np.where(x < 2, a * x**3 - 5 * a * x**2 + 8 * a * x - 4 * a, 0.0))
        return 2.0, k
    if name == "lanczos3":
        def k(x):
            x = np.asarray(x, dtype=np.float64)
            out = np.where(np.abs(x) < 3, np.sinc(x) * np.sinc(x / 3.0), 0.0)
            return out
        return 3.0, k
    if name == "area":
        return 0.5, lambda x: np.where(np.abs(x) <= 0.5, 1.0, 0.0)
    raise ValueError(f"unknown filter {name}")


def _index(j: np.ndarray, n: int, mode: str) -> np.ndarray:
    if mode == "wrap":
        return np.mod(j, n)
    if mode == "clamp":
        return np.clip(j, 0, n - 1)
    if mode == "reflect":  # half-sample symmetric
        period = 2 * n
        j = np.mod(j, period)
        return np.where(j >= n, period - 1 - j, j)
    raise ValueError(f"unknown edge mode {mode}")


def weight_matrix(n_in: int, n_out: int, filt: str, mode: str) -> np.ndarray:
    s = n_out / n_in
    W = np.zeros((n_out, n_in), dtype=np.float64)
    if filt == "nearest":
        src = np.floor((np.arange(n_out) + 0.5) / s).astype(np.int64)
        W[np.arange(n_out), _index(src, n_in, mode)] = 1.0
        return W
    support, k = _kernel(filt)
    fs = max(1.0, 1.0 / s)  # widen kernel when downscaling
    if filt == "area" and s >= 1:
        # area upscaling == nearest; keep exact box semantics
        return weight_matrix(n_in, n_out, "nearest", mode)
    sup = support * fs
    for i in range(n_out):
        c = (i + 0.5) / s
        j0 = int(math.floor(c - sup))
        j1 = int(math.ceil(c + sup))
        j = np.arange(j0, j1 + 1)
        x = (j + 0.5 - c) / fs
        w = k(x).astype(np.float64)
        if filt == "area":
            # exact pixel-overlap weights
            lo = np.maximum(j, c - 0.5 * fs)
            hi = np.minimum(j + 1, c + 0.5 * fs)
            w = np.maximum(0.0, hi - lo)
        tot = w.sum()
        if tot == 0:
            continue
        np.add.at(W[i], _index(j, n_in, mode), w / tot)
    return W


def resize(img: np.ndarray, width: int, height: int, filt: str = "lanczos3",
           mode="clamp", clip: bool = True) -> np.ndarray:
    """Resize HxWxC float image to (height, width).

    ``mode`` is one edge mode or a (rows, cols) pair, e.g. ("clamp", "wrap")
    for a texture that tiles horizontally only.
    """
    my, mx = (mode, mode) if isinstance(mode, str) else mode
    if img.ndim == 2:
        img = img[:, :, None]
    h, w, _ = img.shape
    a = img.astype(np.float64)
    if height != h or filt == "nearest":
        a = np.einsum("oi,ijc->ojc", weight_matrix(h, height, filt, my), a, optimize=False)
    if width != w or filt == "nearest":
        a = np.einsum("pj,ojc->opc", weight_matrix(w, width, filt, mx), a, optimize=False)
    if clip:
        a = np.clip(a, 0.0, 1.0)
    return a.astype(np.float32)


def downscale_area(img: np.ndarray, width: int, height: int, mode="clamp") -> np.ndarray:
    return resize(img, width, height, "area", mode)
