# SPDX-FileCopyrightText: 2026 Handelsonderneming S.D. Broekhoven, trading as ITEagle Europe
# SPDX-License-Identifier: GPL-3.0-only
"""Alpha handling: classification, colour edge dilation, re-thresholding, IoU.

Alpha never goes through a generative model. RGB under fully transparent
texels is filled from opaque neighbours before filtering so that black or
garbage colour does not bleed into edges after resampling or mipmapping.
"""

from __future__ import annotations

import numpy as np


def classify(a: np.ndarray, eps: float = 1.5 / 255) -> str:
    """Return one of ``none | binary-cutout | multi-level | smooth``."""
    a = a.reshape(-1)
    if a.size == 0 or float(a.min()) >= 1 - eps:
        return "none"
    near01 = (a <= eps) | (a >= 1 - eps)
    if near01.mean() >= 0.995:
        return "binary-cutout"
    levels = np.unique(np.round(a * 255).astype(np.int32))
    return "multi-level" if levels.size <= 16 else "smooth"


def _shift_sum(x: np.ndarray, wrap: bool) -> np.ndarray:
    """Sum over the 3x3 neighbourhood (excluding centre)."""
    if wrap:
        p = np.pad(x, ((1, 1), (1, 1), (0, 0)), mode="wrap")
    else:
        p = np.pad(x, ((1, 1), (1, 1), (0, 0)), mode="constant")
    h, w = x.shape[:2]
    s = np.zeros_like(x, dtype=np.float64)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            if dy == 1 and dx == 1:
                continue
            s += p[dy:dy + h, dx:dx + w]
    return s


def edge_dilate(rgba: np.ndarray, threshold: float = 0.5, max_iter: int = 64,
                wrap: bool = False) -> np.ndarray:
    """Fill RGB of texels with alpha < threshold from valid neighbours."""
    rgb = rgba[:, :, :3].astype(np.float64)
    valid = (rgba[:, :, 3:4] >= threshold).astype(np.float64)
    if valid.all() or not valid.any():
        return rgba.copy()
    rgb = rgb * valid
    known = valid.copy()
    for _ in range(max_iter):
        if known.all():
            break
        csum = _shift_sum(rgb * known, wrap)
        ksum = _shift_sum(known, wrap)
        newly = (known == 0) & (ksum > 0)
        fill = np.where(ksum > 0, csum / np.maximum(ksum, 1e-12), 0.0)
        rgb = np.where(newly, fill, rgb)
        known = np.where(newly, 1.0, known)
    if not known.all():
        mean = (rgb * known).sum(axis=(0, 1)) / max(known.sum(), 1.0)
        rgb = np.where(known > 0, rgb, mean)
    out = rgba.copy()
    out[:, :, :3] = rgb.astype(np.float32)
    return out


def rethreshold(a: np.ndarray, t: float) -> np.ndarray:
    return (a >= t).astype(np.float32)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def coverage(a: np.ndarray, t: float) -> float:
    return float((a >= t).mean())
