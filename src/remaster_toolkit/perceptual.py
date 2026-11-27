# SPDX-FileCopyrightText: 2026 Handelsonderneming S.D. Broekhoven, trading as ITEagle Europe
# SPDX-License-Identifier: GPL-3.0-only
"""Optional perceptual metrics: LPIPS v0.1 (``lpips``, BSD-2-Clause) and
DISTS (``DISTS_pytorch``, MIT). Both need torch and ImageNet-pretrained
torchvision backbones (AlexNet / VGG16), downloaded into ``TORCH_HOME`` on
first use. ``pyiqa`` is deliberately not used (PolyForm Noncommercial).

If the packages are missing, functions return ``None`` and QA records the
metric as not available instead of failing.
"""

from __future__ import annotations

import numpy as np

_CACHE: dict = {}
MIN_SIZE = 64


def available() -> bool:
    try:
        import torch  # noqa: F401
        import lpips  # noqa: F401
        return True
    except Exception:
        return False


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _t(img: np.ndarray):
    import torch
    x = torch.from_numpy(np.ascontiguousarray(img[:, :, :3].transpose(2, 0, 1))).float()[None]
    return x.to(_device())


def lpips_alex(a: np.ndarray, b: np.ndarray) -> float | None:
    try:
        import lpips
        import torch
    except Exception:
        return None
    if "lpips" not in _CACHE:
        _CACHE["lpips"] = lpips.LPIPS(net="alex", version="0.1", verbose=False).to(_device()).eval()
    with torch.no_grad():
        return float(_CACHE["lpips"](_t(a) * 2 - 1, _t(b) * 2 - 1).item())


def dists(a: np.ndarray, b: np.ndarray) -> float | None:
    try:
        from DISTS_pytorch import DISTS
        import torch
    except Exception:
        return None
    if "dists" not in _CACHE:
        _CACHE["dists"] = DISTS().to(_device()).eval()
    with torch.no_grad():
        return float(_CACHE["dists"](_t(a), _t(b)).item())


def compare(out: np.ndarray, ref: np.ndarray) -> dict:
    """LPIPS/DISTS between two same-sized images (values or None)."""
    if out.shape[:2] != ref.shape[:2]:
        raise ValueError("perceptual compare needs equal sizes")
    if min(out.shape[:2]) < MIN_SIZE:  # backbones need a minimum spatial extent
        return {"lpips-v0.1-alex": None, "dists": None}
    return {"lpips-v0.1-alex": lpips_alex(out, ref), "dists": dists(out, ref)}
