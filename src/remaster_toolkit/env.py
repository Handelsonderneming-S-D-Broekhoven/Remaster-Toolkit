# SPDX-FileCopyrightText: 2026 Handelsonderneming S.D. Broekhoven, trading as ITEagle Europe
# SPDX-License-Identifier: GPL-3.0-only
"""Runtime identity and determinism settings (RD-004 sections 6, 8, 9)."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import zlib
from functools import lru_cache

from . import __version__
from .hashing import jcs_hash

# Must be set before CUDA/cuBLAS initialise (RD-004 D1 settings).
REFERENCE_ENV = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONHASHSEED": "0"}


def set_reference_env() -> None:
    for k, v in REFERENCE_ENV.items():
        os.environ.setdefault(k, v)


def apply_torch_determinism(deterministic: bool = True, allow_tf32: bool = False) -> dict:
    """Apply D1 settings; returns the settings actually in force."""
    import torch

    set_reference_env()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.use_deterministic_algorithms(deterministic, warn_only=False)
    return {
        "deterministicAlgorithms": deterministic,
        "warnOnly": False,
        "cudnnBenchmark": False,
        "cudnnDeterministic": deterministic,
        "allowTF32": allow_tf32,
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def _git_commit() -> tuple[str | None, bool | None]:
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        c = subprocess.run(["git", "-C", here, "rev-parse", "HEAD"], capture_output=True,
                           text=True, timeout=10)
        d = subprocess.run(["git", "-C", here, "status", "--porcelain", "--", "."],
                           capture_output=True, text=True, timeout=10)
        if c.returncode == 0:
            return c.stdout.strip(), bool(d.stdout.strip())
    except Exception:
        pass
    return None, None


@lru_cache(maxsize=1)
def executor_identity() -> dict:
    commit, dirty = _git_commit()
    ident = {"name": "remaster-toolkit", "version": __version__}
    if commit:
        ident["gitCommit"] = commit
        ident["dirty"] = bool(dirty)
    return ident


def _versions() -> dict:
    """Installed package versions (metadata only; does not import heavy modules)."""
    from importlib import metadata

    v = {"python": platform.python_version(), "zlib": zlib.ZLIB_VERSION}
    for dist in ("numpy", "Pillow", "torch", "torchvision", "spandrel", "diffusers",
                 "transformers", "accelerate", "safetensors", "lpips", "DISTS_pytorch"):
        try:
            v[dist.lower()] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            pass
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.cuda.is_available():
                v["cuda"] = str(torch.version.cuda)
                v["cudnn"] = str(torch.backends.cudnn.version())
        except Exception:
            pass
    return v


@lru_cache(maxsize=1)
def gpu_info() -> list[dict]:
    """GPUs as seen by an already-imported torch (empty for CPU-only runs)."""
    out = []
    torch = sys.modules.get("torch")
    try:
        if torch is not None and torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                out.append({"vendor": "NVIDIA" if torch.version.cuda else "unknown",
                            "name": p.name, "computeCapability": f"{p.major}.{p.minor}",
                            "smCount": int(p.multi_processor_count),
                            "vramMiB": int(p.total_memory // (1024 * 1024))})
    except Exception:
        pass
    if out:
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=driver_version",
                                "--format=csv,noheader"], capture_output=True, text=True,
                               timeout=20)
            for g, d in zip(out, r.stdout.strip().splitlines()):
                g["driverVersion"] = d.strip()
        except Exception:
            pass
    return out


def environment(settings: dict | None = None, lockfile_hash: str | None = None) -> dict:
    env = {
        "executor": executor_identity(),
        "referenceEnvironmentId": None,
        "os": {"system": platform.system(), "release": platform.release(),
               "version": platform.version(), "machine": platform.machine()},
        "cpu": platform.processor() or platform.machine(),
        "gpus": gpu_info(),
        "runtimes": _versions(),
        "settings": dict(settings or {}),
        "containerDigest": None,
        "workflowEngine": None,
    }
    if lockfile_hash:
        env["lockfileHash"] = lockfile_hash
    env["environmentHash"] = jcs_hash(env)
    return env


def cpu_runtime_class() -> str:
    """Equivalence class for pure numpy CPU steps (assumed stable; RD-004)."""
    import numpy

    return f"cpu-numpy{numpy.__version__}-py{sys.version_info.major}.{sys.version_info.minor}"


def gpu_runtime_class(precision: dict) -> str:
    """Unproven GPU steps key on the full environment fingerprint (RD-004 s6)."""
    v = _versions()
    g = gpu_info()
    fp = {"gpus": [{k: x.get(k) for k in ("name", "smCount", "computeCapability", "driverVersion")}
                   for x in g],
          "torch": v.get("torch"), "cuda": v.get("cuda"), "cudnn": v.get("cudnn"),
          "os": platform.system(), "precision": precision}
    return "fp:" + jcs_hash(fp)[7:23]
