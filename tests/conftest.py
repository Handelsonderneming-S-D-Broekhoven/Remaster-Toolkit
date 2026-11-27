# SPDX-FileCopyrightText: 2026 Handelsonderneming S.D. Broekhoven, trading as ITEagle Europe
# SPDX-License-Identifier: GPL-3.0-only
"""Synthetic, procedurally generated fixtures only (no game data, no weights)."""

from __future__ import annotations

import numpy as np
import pytest

from remaster_toolkit import imagebuf, ops
from remaster_toolkit.hashing import sha256_bytes
from remaster_toolkit.resample import resize


def periodic_texture(n: int = 32, seed: int = 1, channels: int = 3) -> np.ndarray:
    """Sum of sinusoids with integer frequencies: exactly tileable."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:n, 0:n] / n
    out = np.zeros((n, n, channels))
    for c in range(channels):
        for _ in range(4):
            fx, fy = rng.integers(1, 5, size=2)
            ph = rng.uniform(0, 2 * np.pi)
            out[:, :, c] += np.sin(2 * np.pi * (fx * x + fy * y) + ph)
    out = (out - out.min()) / (out.max() - out.min())
    return out.astype(np.float32)


def ramp_texture(n: int = 32) -> np.ndarray:
    """Non-tileable gradient (strong wrap seam)."""
    y, x = np.mgrid[0:n, 0:n] / (n - 1)
    return np.stack([x, y, 0.5 * (x + y)], axis=2).astype(np.float32)


def cutout_texture(n: int = 32, seed: int = 2) -> np.ndarray:
    """Binary-alpha disk with leaf-like noise colour; RGB garbage under alpha=0."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:n, 0:n]
    r = np.hypot(x - n / 2 + 0.5, y - n / 2 + 0.5)
    a = (r < n * 0.35).astype(np.float32)
    rgb = np.stack([0.2 + 0.1 * rng.random((n, n)), 0.5 + 0.3 * rng.random((n, n)),
                    0.1 * rng.random((n, n))], axis=2).astype(np.float32)
    rgb[a == 0] = 0.0  # black under transparency (bleeds if not dilated)
    return np.concatenate([rgb, a[:, :, None]], axis=2)


def source_png(img: np.ndarray) -> tuple[bytes, str]:
    data = imagebuf.png_encode(img)
    return data, sha256_bytes(data)


def _stub_sr(step, inputs, ctx):
    """CPU stand-in for an SR network: bicubic x scale with the step's edge mode."""
    img = inputs[0]
    s = int(step.get("scale", 4))
    mode = ops._edge(step)
    return resize(img, img.shape[1] * s, img.shape[0] * s, "bicubic", mode), {}


@pytest.fixture
def stub_sr(monkeypatch):
    monkeypatch.setitem(ops.OPS, "sr", (_stub_sr, ["tiling"]))
    return _stub_sr


FAKE_MODEL = {"role": "sr", "hash": "sha256:" + "0" * 64, "name": "stub-sr",
              "licenseId": "NOASSERTION"}
