# SPDX-FileCopyrightText: 2026 Handelsonderneming S.D. Broekhoven, trading as ITEagle Europe
# SPDX-License-Identifier: GPL-3.0-only
"""SHA-256 helpers. Hash strings are written as ``sha256:<64 lowercase hex>``."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from .jcs import canonicalize

PREFIX = "sha256:"


def sha256_bytes(data: bytes) -> str:
    return PREFIX + hashlib.sha256(data).hexdigest()


def sha256_file(path: str | os.PathLike[str], chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return PREFIX + h.hexdigest()


def jcs_hash(value: Any) -> str:
    """sha256 over the RFC 8785 canonical form of ``value``."""
    return sha256_bytes(canonicalize(value))


def hex_of(h: str) -> str:
    if not h.startswith(PREFIX) or len(h) != len(PREFIX) + 64:
        raise ValueError(f"not a sha256: hash string: {h!r}")
    return h[len(PREFIX):]


def sha256_tree(root: str | os.PathLike[str]) -> tuple[str, dict[str, str]]:
    """Identity of a directory bundle (e.g. a multi-file diffusion model).

    Returns ``(bundle_hash, manifest)`` where manifest maps forward-slash
    relative paths to file hashes and bundle_hash = jcs_hash(manifest).
    """
    root = Path(root)
    manifest: dict[str, str] = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and not p.name.endswith(".part"):
            manifest[p.relative_to(root).as_posix()] = sha256_file(p)
    return jcs_hash(manifest), manifest
