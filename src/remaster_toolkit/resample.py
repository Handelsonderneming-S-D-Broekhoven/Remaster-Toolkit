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


def footprint_minmax(img: np.ndarray, width: int, height: int,
                     mode="clamp") -> tuple[np.ndarray, np.ndarray]:
    """Per output texel, the min and max of the source texels around its centre.

    The window is the 2x2 bilinear footprint when magnifying and the box of
    source texels covered by the output texel when minifying (separable, so
    min/max over the rectangle is exact). Used to clamp a ringing kernel.
    """
    my, mx = (mode, mode) if isinstance(mode, str) else mode
    if img.ndim == 2:
        img = img[:, :, None]
    a = img.astype(np.float64)

    def axis(arr, n_out, m, ax):
        n_in = arr[0].shape[ax]
        s = n_out / n_in
        c = (np.arange(n_out) + 0.5) / s - 0.5
        r = max(0.0, 0.5 / s - 0.5)  # extra half-width when minifying
        j0 = np.floor(c - r).astype(np.int64)
        j1 = np.ceil(c + r).astype(np.int64)
        j1 = np.maximum(j1, j0 + 1)
        width_max = int((j1 - j0).max()) + 1
        lo = hi = None
        for k in range(width_max):
            j = np.minimum(j0 + k, j1)
            t_lo = np.take(arr[0], _index(j, n_in, m), axis=ax)
            t_hi = np.take(arr[1], _index(j, n_in, m), axis=ax)
            lo = t_lo if lo is None else np.minimum(lo, t_lo)
            hi = t_hi if hi is None else np.maximum(hi, t_hi)
        return lo, hi

    lo, hi = axis((a, a), height, my, 0)
    lo, hi = axis((lo, hi), width, mx, 1)
    return lo, hi


def resize(img: np.ndarray, width: int, height: int, filt: str = "lanczos3",
           mode="clamp", clip: bool = True, anti_ringing: bool = False) -> np.ndarray:
    """Resize HxWxC float image to (height, width).

    ``mode`` is one edge mode or a (rows, cols) pair, e.g. ("clamp", "wrap")
    for a texture that tiles horizontally only. ``anti_ringing`` clamps every
    output texel to the min/max of its source footprint
    (:func:`footprint_minmax`), so a sharp kernel (Lanczos, bicubic) cannot
    create overshoot halos, dark outlines or lift a black background.
    """
    if anti_ringing:
        out = resize(img, width, height, filt, mode, clip=False)
        lo, hi = footprint_minmax(img, width, height, mode)
        out = np.clip(out, lo, hi)
        if clip:
            out = np.clip(out, 0.0, 1.0)
        return out.astype(np.float32)
    my, mx = (mode, mode) if isinstance(mode, str) else mode
    if img.ndim == 2:
        img = img[:, :, None]
    h, w, _ = img.shape
    a = img.astype(np.float64)
    c = a.shape[2]
    # float64 BLAS matmul (numpy-bundled); run-to-run stability is checked by
    # the double-build hash tests rather than assumed.
    if height != h or filt == "nearest":
        a = (weight_matrix(h, height, filt, my) @ a.reshape(h, -1)).reshape(height, -1, c)
    if width != w or filt == "nearest":
        wm = weight_matrix(w, width, filt, mx)
        a = (wm @ a.transpose(1, 0, 2).reshape(w, -1)).reshape(width, -1, c).transpose(1, 0, 2)
    if clip:
        a = np.clip(a, 0.0, 1.0)
    return a.astype(np.float32)


def downscale_area(img: np.ndarray, width: int, height: int, mode="clamp") -> np.ndarray:
    return resize(img, width, height, "area", mode)
