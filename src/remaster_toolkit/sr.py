# SPDX-FileCopyrightText: 2026 Handelsonderneming S.D. Broekhoven, trading as ITEagle Europe
# SPDX-License-Identifier: GPL-3.0-only
"""Conventional super-resolution through spandrel (MIT) model loading.

Weights are loaded by content hash from the :class:`~.models.ModelStore`.
Pickle checkpoints are read with ``torch.load(weights_only=True)`` and handed
to spandrel as a state dict (spandrel's own file loader would also try
TorchScript, which we do not allow). Precision and determinism settings come
from the step; the wrap/tiling logic is generic (:mod:`.tiling`).
"""

from __future__ import annotations

import numpy as np

from .tiling import apply_tiled

_LOADED: dict = {}


def load_model(path, fmt: str, device: str):
    import torch
    from spandrel import ModelLoader

    key = (str(path), device)
    if key in _LOADED:
        return _LOADED[key]
    if fmt == "safetensors":
        from safetensors.torch import load_file
        sd = load_file(str(path))
    elif fmt == "pth-weights-only":
        sd = torch.load(str(path), map_location="cpu", weights_only=True)
    else:
        raise ValueError(f"unsupported SR model format {fmt}")
    desc = ModelLoader(device="cpu").load_from_state_dict(sd)
    desc.model.eval()
    desc.to(device)
    _LOADED[key] = desc
    return desc


def make_fn(desc, precision: str, device: str):
    import torch

    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]
    if precision != "fp32":
        if precision == "fp16" and not desc.supports_half:
            raise ValueError("model does not support fp16")
        if precision == "bf16" and not desc.supports_bfloat16:
            raise ValueError("model does not support bf16")
    desc.model.to(dtype)

    def fn(tile: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.ascontiguousarray(tile.transpose(2, 0, 1)))[None]
        x = x.to(device=device, dtype=dtype)
        with torch.inference_mode():
            y = desc(x).float().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
        return y.astype(np.float32)

    return fn


def run(img: np.ndarray, model_path, model_format: str, step: dict, device: str) -> np.ndarray:
    """Upscale ``img`` (HxWx{1,3}) per the ``sr`` step parameters."""
    prec = (step.get("precision") or {}).get("compute", "fp32")
    tiling = step.get("tiling") or {}
    desc = load_model(model_path, model_format, device)
    scale = int(desc.scale)
    want = step.get("scale")
    if want is not None and int(want) != scale:
        raise ValueError(f"step scale {want} != model scale {scale}")
    fn = make_fn(desc, prec, device)
    single = img.shape[2] == 1
    x = np.repeat(img, 3, axis=2) if single else img[:, :, :3]
    attn = (step.get("precision") or {}).get("attentionBackend", "default")
    ctx = _sdpa_ctx(attn)
    with ctx:
        y = apply_tiled(fn, x, scale, tile=int(tiling.get("tileSize", 0)),
                        overlap=int(tiling.get("overlap", 0)),
                        pad_px=int(tiling.get("wrapPadding", 0)),
                        edge=_edge_modes(step),
                        blend=tiling.get("blend", "linear-feather"))
    if single:
        y = y.mean(axis=2, keepdims=True).astype(np.float32)
    return y


def _edge_modes(step: dict):
    from .ops import _axes
    return _axes(step)


def _sdpa_ctx(backend: str):
    import contextlib

    if backend in ("default", None):
        return contextlib.nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel

    b = {"math": SDPBackend.MATH, "flash": SDPBackend.FLASH_ATTENTION,
         "efficient": SDPBackend.EFFICIENT_ATTENTION, "cudnn": SDPBackend.CUDNN_ATTENTION}[backend]
    return sdpa_kernel(b)
