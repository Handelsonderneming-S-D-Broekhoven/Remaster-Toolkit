# SPDX-FileCopyrightText: 2026 Handelsonderneming S.D. Broekhoven, trading as ITEagle Europe
# SPDX-License-Identifier: GPL-3.0-only
"""Image buffers.

In memory an image is a ``numpy.float32`` array of shape (H, W, C) with
values in [0, 1], straight (non-premultiplied) alpha, C in {1, 3, 4}.

Intermediate step outputs are stored losslessly in a small canonical raw
container ("RTKIMG1"): magic, 4-byte little-endian header length, JCS header
``{"dtype":"<f4","shape":[h,w,c]}``, then the C-order little-endian float32
payload. Its bytes are a pure function of the pixel values, so the SHA-256 of
an intermediate is also its decoded pixel hash.

Final outputs are PNG, written with fixed settings and no metadata chunks.
"""

from __future__ import annotations

import io
import json
import struct

import numpy as np

from .hashing import sha256_bytes
from .jcs import canonicalize

MAGIC = b"RTKIMG1\n"


def to_bytes(img: np.ndarray) -> bytes:
    a = np.ascontiguousarray(img, dtype="<f4")
    if a.ndim == 2:
        a = a[:, :, None]
    if a.ndim != 3:
        raise ValueError("image must be HxWxC")
    hdr = canonicalize({"dtype": "<f4", "shape": list(a.shape)})
    return MAGIC + struct.pack("<I", len(hdr)) + hdr + a.tobytes()


def from_bytes(data: bytes) -> np.ndarray:
    if not data.startswith(MAGIC):
        raise ValueError("not an RTKIMG1 buffer")
    (n,) = struct.unpack_from("<I", data, len(MAGIC))
    off = len(MAGIC) + 4
    hdr = json.loads(data[off:off + n])
    arr = np.frombuffer(data, dtype="<f4", offset=off + n).reshape(hdr["shape"])
    return arr.astype(np.float32, copy=True)


def quantize(img: np.ndarray, bits: int = 8) -> np.ndarray:
    m = (1 << bits) - 1
    q = np.floor(np.clip(img, 0.0, 1.0) * m + 0.5)
    return q.astype(np.uint8 if bits == 8 else np.uint16)


def decoded_pixel_hash(img: np.ndarray, bits: int = 8) -> str:
    """Hash of the quantized pixel buffer (independent of PNG encoder)."""
    q = np.ascontiguousarray(quantize(img, bits))
    hdr = canonicalize({"bits": bits, "shape": list(q.shape)})
    return sha256_bytes(hdr + q.astype(q.dtype.newbyteorder("<")).tobytes())


def png_decode(data: bytes) -> np.ndarray:
    """Decode PNG (or any Pillow-readable image) to float32 RGBA/RGB/L."""
    from PIL import Image

    with Image.open(io.BytesIO(data)) as im:
        im.load()
        mode = im.mode
        if mode in ("I;16", "I;16B", "I;16L", "I"):
            a = np.asarray(im, dtype=np.float64) / 65535.0
            return a[:, :, None].astype(np.float32)
        if mode == "P":
            im = im.convert("RGBA" if "transparency" in im.info else "RGB")
        elif mode not in ("L", "LA", "RGB", "RGBA"):
            im = im.convert("RGBA")
        a = np.asarray(im, dtype=np.float32) / 255.0
    if a.ndim == 2:
        a = a[:, :, None]
    if a.shape[2] == 2:  # LA -> RGBA
        a = np.concatenate([a[:, :, :1]] * 3 + [a[:, :, 1:2]], axis=2)
    return a.astype(np.float32)


def png_encode(img: np.ndarray, bits: int = 8) -> bytes:
    """Deterministic PNG: fixed zlib level, no optimisation, no metadata."""
    from PIL import Image

    q = quantize(img, bits)
    c = q.shape[2] if q.ndim == 3 else 1
    if bits == 16:
        if c != 1:
            raise ValueError("16-bit PNG output supports single channel only")
        im = Image.fromarray(np.ascontiguousarray(q[:, :, 0].astype("<u2")))
    else:
        if c not in (1, 3, 4):
            raise ValueError(f"unsupported channel count {c}")
        im = Image.fromarray(np.ascontiguousarray(q[:, :, 0] if c == 1 else q))
    buf = io.BytesIO()
    im.save(buf, format="PNG", compress_level=6, optimize=False)
    return buf.getvalue()


def ensure_rgba(img: np.ndarray) -> np.ndarray:
    c = img.shape[2]
    if c == 4:
        return img
    if c == 3:
        return np.concatenate([img, np.ones(img.shape[:2] + (1,), np.float32)], axis=2)
    if c == 1:
        return np.concatenate([img, img, img, np.ones_like(img)], axis=2)
    raise ValueError(f"unsupported channel count {c}")
