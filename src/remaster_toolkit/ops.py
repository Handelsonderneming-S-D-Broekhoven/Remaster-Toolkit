# SPDX-FileCopyrightText: 2026 Handelsonderneming S.D. Broekhoven, trading as ITEagle Europe
# SPDX-License-Identifier: GPL-3.0-only
"""Step implementations (``step.op``) and their implementation identity.

Each op is ``fn(step, inputs, ctx) -> (output, info)`` where ``inputs`` are
decoded images (float32 HxWxC) except for ``decode`` (raw source bytes), and
the output is an image, or PNG bytes for ``encode``. ``IMPL_MODULES`` lists
the modules whose source text forms the op's ``stepImpl`` hash in the cache
key, so editing an op's code invalidates exactly the steps that use it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import alpha as alpha_mod
from . import pbr, tiling
from .hashing import sha256_bytes
from .imagebuf import ensure_rgba, png_decode, png_encode
from .resample import FILTERS, resize

EDGE_TO_RESAMPLE = {"wrap": "wrap", "replicate": "clamp", "reflect": "reflect", "zero": "clamp"}


def _axes(step: dict, mapping: dict | None = None):
    """Edge mode(s) of a step: a single mode, or (rows, cols) when
    ``params.wrapAxes`` restricts wrapping to ``"u"`` (horizontal) or ``"v"``."""
    e = (step.get("tiling") or {}).get("edgePadding", "replicate")
    conv = (lambda x: mapping[x]) if mapping else (lambda x: x)
    ax = (step.get("params") or {}).get("wrapAxes", "uv")
    if e != "wrap" or ax == "uv":
        return conv(e)
    other = "replicate" if mapping else "reflect"
    return (conv("wrap"), conv(other)) if ax == "v" else (conv(other), conv("wrap"))


def _edge(step: dict):
    return _axes(step, EDGE_TO_RESAMPLE)


def _wraps(step: dict) -> bool:
    """True only if the step wraps on both axes (2-D periodic helpers)."""
    return _edge(step) == "wrap"


def _target(step: dict, h: int, w: int) -> tuple[int, int]:
    if "targetWidth" in step or "targetHeight" in step:
        tw = int(step.get("targetWidth", round(w * step.get("targetHeight") / h)))
        th = int(step.get("targetHeight", round(h * tw / w)))
        return tw, th
    s = float(step.get("scale", 1))
    return int(round(w * s)), int(round(h * s))


def op_decode(step, inputs, ctx):
    return ensure_rgba(png_decode(inputs[0])), {}


def op_alpha_split(step, inputs, ctx):
    img = inputs[0]
    ch = step.get("channels", "rgb")
    if ch == "rgb":
        return np.ascontiguousarray(img[:, :, :3]), {}
    if ch == "a":
        if img.shape[2] < 4:
            return np.ones(img.shape[:2] + (1,), np.float32), {}
        return np.ascontiguousarray(img[:, :, 3:4]), {}
    raise ValueError(f"alpha-split channels {ch}")


def op_alpha_merge(step, inputs, ctx):
    rgb, a = inputs[0][:, :, :3], inputs[1][:, :, :1]
    if a.shape[:2] != rgb.shape[:2]:
        a = resize(a, rgb.shape[1], rgb.shape[0], "lanczos3", _edge(step))
    p = step.get("params") or {}
    t = p.get("cutoutThreshold")
    if t is not None:
        a = alpha_mod.rethreshold(a, float(t))
    out = np.concatenate([rgb, a], axis=2).astype(np.float32)
    if p.get("redilate", t is not None):
        out = alpha_mod.edge_dilate(out, float(t if t is not None else 0.5),
                                    wrap=_wraps(step))
    return out, {}


def op_edge_dilate(step, inputs, ctx):
    p = step.get("params") or {}
    img = ensure_rgba(inputs[0])
    return alpha_mod.edge_dilate(img, float(p.get("threshold", 0.5)),
                                 int(p.get("maxIter", 64)), wrap=_wraps(step)), {}


def op_resample(step, inputs, ctx):
    img = inputs[0]
    h, w = img.shape[:2]
    tw, th = _target(step, h, w)
    filt = step.get("filter", "lanczos3")
    if filt not in FILTERS:
        raise ValueError(f"filter {filt} not implemented")
    if (step.get("params") or {}).get("normalMap"):
        return pbr.resample_normals(img, tw, th, filt, _edge(step)), {}
    return resize(img, tw, th, filt, _edge(step)), {}


def op_sr(step, inputs, ctx):
    from . import sr

    m = [x for x in step["models"] if x["role"] == "sr"][0]
    entry = ctx.model_entry(m["hash"])
    path = ctx.model_path(m["hash"])
    return sr.run(inputs[0], path, entry["format"], step, ctx.device), {}


def op_diffusion(step, inputs, ctx):
    from . import diffusion

    roles = {x["role"]: x["hash"] for x in step["models"]}
    sd = ctx.model_path(roles["checkpoint"])
    cn = ctx.model_path(roles["controlnet"])
    return diffusion.run(inputs[0], str(sd), str(cn), step, ctx.device)


def op_seam_fix(step, inputs, ctx):
    band = int((step.get("params") or {}).get("band", 16))
    img = inputs[0]
    axes = (step.get("params") or {}).get("wrapAxes", "uv")
    return tiling.seam_fix(img, band, axes), {}


def op_pbr(step, inputs, ctx):
    p = step.get("params") or {}
    wrap = _wraps(step)
    kind = p.get("map", "normal")
    img = inputs[0]
    if kind == "height":
        return pbr.height_from_albedo(img, wrap, int(p.get("highpassRadius", 0))), {"inferred": True}
    if kind == "normal":
        h = img if img.shape[2] == 1 else pbr.height_from_albedo(img, wrap,
                                                                 int(p.get("highpassRadius", 0)))
        return pbr.normal_from_height(h, float(p.get("strength", 2.0)), wrap,
                                      p.get("convention", "opengl-y-up")), {"inferred": True}
    raise ValueError(f"pbr-derive map {kind}")


def op_normalize_normals(step, inputs, ctx):
    return pbr.normalize_normals(inputs[0]), {}


def op_sharpen(step, inputs, ctx):
    p = step.get("params") or {}
    img = inputs[0]
    r, amt = int(p.get("radius", 1)), float(p.get("amount", 0.5))
    wrap = _wraps(step)
    out = np.empty_like(img)
    for c in range(img.shape[2]):
        b = pbr._blur(img[:, :, c], r, wrap)
        out[:, :, c] = np.clip(img[:, :, c] + amt * (img[:, :, c] - b), 0, 1)
    return out.astype(np.float32), {}


def op_encode(step, inputs, ctx):
    img = inputs[0]
    ch = step.get("channels", "all")
    if ch == "rgb":
        img = img[:, :, :3]
    elif ch == "rgba":
        img = ensure_rgba(img)
    elif ch in ("a", "luma"):
        img = img[:, :, -1:] if ch == "a" else img[:, :, :1]
    bits = int((step.get("params") or {}).get("bits", 8))
    return png_encode(img, bits), {}


OPS = {
    "decode": (op_decode, ["imagebuf"]),
    "alpha-split": (op_alpha_split, []),
    "alpha-merge": (op_alpha_merge, ["alpha", "resample"]),
    "edge-dilate": (op_edge_dilate, ["alpha"]),
    "resample": (op_resample, ["resample", "pbr"]),
    "downscale": (op_resample, ["resample", "pbr"]),
    "sr": (op_sr, ["sr", "tiling"]),
    "controlled-diffusion": (op_diffusion, ["diffusion", "resample"]),
    "seam-fix": (op_seam_fix, ["tiling"]),
    "pbr-derive": (op_pbr, ["pbr"]),
    "normalize-normals": (op_normalize_normals, ["pbr"]),
    "sharpen": (op_sharpen, ["pbr"]),
    "encode": (op_encode, ["imagebuf"]),
}
GPU_OPS = frozenset({"sr", "controlled-diffusion"})

_IMPL_CACHE: dict[str, str] = {}


def impl_hash(op: str) -> str:
    """SHA-256 over this module's source plus the op's dependency modules."""
    if op not in _IMPL_CACHE:
        here = Path(__file__).parent
        parts = [b"ops.py\0" + (here / "ops.py").read_bytes()]
        for mod in OPS[op][1]:
            parts.append(f"{mod}.py\0".encode() + (here / f"{mod}.py").read_bytes())
        _IMPL_CACHE[op] = sha256_bytes(b"\0\0".join(p.replace(b"\r\n", b"\n") for p in parts))
    return _IMPL_CACHE[op]
