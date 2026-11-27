# SPDX-FileCopyrightText: 2026 Handelsonderneming S.D. Broekhoven, trading as ITEagle Europe
# SPDX-License-Identifier: GPL-3.0-only
"""Seamless (wrap) padding and overlapped tiled inference with feathering.

``apply_tiled`` runs an arbitrary image->image function ``fn`` (for example
an SR network) that scales by an integer factor. For tileable textures the
input is padded circularly by ``pad`` pixels before inference and the centre
is cropped afterwards; if ``pad`` covers the model's receptive field the
result equals what a fully circular network would produce, so the texture
stays seamless. Large inputs are split into overlapping tiles processed in a
fixed (row-major) order and blended with a separable feather window.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

EDGE_MODES = {"wrap": "wrap", "reflect": "symmetric", "replicate": "edge", "zero": "constant"}


def pad(img: np.ndarray, n: int, mode) -> np.ndarray:
    """Pad by ``n`` pixels; ``mode`` is one edge mode or a (rows, cols) pair."""
    if n <= 0:
        return img
    my, mx = (mode, mode) if isinstance(mode, str) else mode
    img = np.pad(img, ((n, n), (0, 0), (0, 0)), mode=EDGE_MODES[my])
    return np.pad(img, ((0, 0), (n, n), (0, 0)), mode=EDGE_MODES[mx])


def _starts(length: int, tile: int, overlap: int) -> list[int]:
    if tile <= 0 or length <= tile:
        return [0]
    step = tile - overlap
    if step <= 0:
        raise ValueError("overlap must be smaller than tile size")
    s = list(range(0, length - tile, step))
    s.append(length - tile)
    return sorted(set(s))


def _feather(n: int, ramp_lo: int, ramp_hi: int, blend: str) -> np.ndarray:
    """Window that is 0 on the outer quarter of each overlap, then ramps to 1.

    The zero margin discards tile-border pixels that a network computed from
    its own (non-image) padding; it is valid if overlap >= 4x the receptive
    field radius. Adjacent tiles' non-zero ramps still overlap, so the
    normalised blend never divides by zero.
    """
    w = np.ones(n, dtype=np.float64)
    for r, lo_side in ((ramp_lo, True), (ramp_hi, False)):
        if r <= 0:
            continue
        m = r // 4
        L = r - m
        t = (np.arange(L) + 0.5) / L
        if blend == "cosine-feather":
            t = 0.5 - 0.5 * np.cos(np.pi * t)
        elif blend == "gaussian":
            t = np.exp(-0.5 * ((1 - t) * 3) ** 2)
        seg = np.concatenate([np.zeros(m), t])
        if lo_side:
            w[:r] = np.minimum(w[:r], seg)
        else:
            w[n - r:] = np.minimum(w[n - r:], seg[::-1])
    return w


def apply_tiled(fn: Callable[[np.ndarray], np.ndarray], img: np.ndarray, scale: int,
                tile: int = 0, overlap: int = 0, pad_px: int = 0, edge="reflect",
                blend: str = "linear-feather") -> np.ndarray:
    """Run ``fn`` (HxWxC -> (sH)x(sW)xC) with padding and overlapped tiles."""
    h, w, c = img.shape
    p = pad(img, pad_px, edge)
    ph, pw = p.shape[:2]
    ys, xs = _starts(ph, tile, overlap), _starts(pw, tile, overlap)
    if len(ys) == 1 and len(xs) == 1:
        out = fn(p)
    else:
        th, tw = min(tile, ph), min(tile, pw)
        acc = np.zeros((ph * scale, pw * scale, c), dtype=np.float64)
        wsum = np.zeros((ph * scale, pw * scale, 1), dtype=np.float64)
        for y in ys:  # row-major, fixed order
            for x in xs:
                t = fn(p[y:y + th, x:x + tw])
                ry_lo = overlap * scale if y > 0 else 0
                ry_hi = overlap * scale if y + th < ph else 0
                rx_lo = overlap * scale if x > 0 else 0
                rx_hi = overlap * scale if x + tw < pw else 0
                if blend == "none":
                    wy = np.ones(th * scale)
                    wx = np.ones(tw * scale)
                else:
                    wy = _feather(th * scale, ry_lo, ry_hi, blend)
                    wx = _feather(tw * scale, rx_lo, rx_hi, blend)
                wt = (wy[:, None] * wx[None, :])[:, :, None]
                acc[y * scale:(y + th) * scale, x * scale:(x + tw) * scale] += t * wt
                wsum[y * scale:(y + th) * scale, x * scale:(x + tw) * scale] += wt
        out = (acc / np.maximum(wsum, 1e-12)).astype(np.float32)
    if pad_px:
        o = pad_px * scale
        out = out[o:o + h * scale, o:o + w * scale]
    return np.ascontiguousarray(out, dtype=np.float32)


def offset_half(img: np.ndarray) -> np.ndarray:
    """Roll a texture by half its size (moves the wrap seam to the centre)."""
    return np.roll(img, (img.shape[0] // 2, img.shape[1] // 2), axis=(0, 1))


def seam_fix(img: np.ndarray, band: int, axes: str = "uv") -> np.ndarray:
    """Force wrap continuity by offset cross-fading (classical seam repair).

    Per axis: ``out = (1 - m) * img + m * roll(img, half)`` where ``m`` is 1 at
    the two borders of that axis and falls to 0 over ``band`` pixels. At the
    borders the output equals the half-rolled image, whose content is
    continuous across the wrap boundary; the interior is untouched. The x pass
    runs first; the y pass preserves the x continuity. Used only after steps
    that are not inherently circular (e.g. diffusion without circular
    convolution); it introduces cross-fade ghosting inside the band.
    """
    if band <= 0:
        return img
    out = img.astype(np.float64)
    for axis in [a for a, k in ((1, "u"), (0, "v")) if k in axes]:
        n = out.shape[axis]
        b = max(1, min(band, n // 4))
        d = np.minimum(np.arange(n) + 0.5, n - np.arange(n) - 0.5)
        m = np.clip(1.0 - d / b, 0.0, 1.0)
        m = 0.5 - 0.5 * np.cos(np.pi * m)
        shape = [1, 1, 1]
        shape[axis] = n
        m = m.reshape(shape)
        rolled = np.roll(out, n // 2, axis=axis)
        out = (1 - m) * out + m * rolled
    return np.clip(out, 0, 1).astype(np.float32)
