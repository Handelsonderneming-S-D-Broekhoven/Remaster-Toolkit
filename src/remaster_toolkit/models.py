# SPDX-FileCopyrightText: 2026 Handelsonderneming S.D. Broekhoven, trading as ITEagle Europe
# SPDX-License-Identifier: GPL-3.0-only
"""Content-addressed model store (RD-004 section 10).

Recipes reference models only by SHA-256 (``modelRef.hash``); names, URLs and
revisions are informational and never resolved as mutable aliases. The store
is a JSON registry ``{hash: entry}`` kept outside any repository (the user's
workspace). A *file* model's hash is the SHA-256 of its bytes; a *bundle*
(multi-file diffusers directory) hash is the JCS hash of its
``{relative/path: sha256}`` manifest. Every model is re-hashed once per
process before first use (``verified``).

Pickle checkpoints (``.pth``) are only accepted when their hash is in the
registry with ``format: pth-weights-only``; they are then loaded with
``torch.load(weights_only=True)`` semantics by the loader.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

from .hashing import sha256_file, sha256_tree
from .jcs import canonicalize


class ModelError(RuntimeError):
    pass


class ModelStore:
    def __init__(self, registry_path: str | os.PathLike[str]):
        self.path = Path(registry_path)
        self.entries: dict[str, dict] = {}
        if self.path.exists():
            self.entries = json.loads(self.path.read_text(encoding="utf-8"))
        self._verified: set[str] = set()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_bytes(canonicalize(self.entries))
        os.replace(tmp, self.path)

    def register(self, path: str | os.PathLike[str], name: str, fmt: str,
                 license_id: str = "NOASSERTION", source_uri: str | None = None,
                 source_revision: str | None = None, license_note: str | None = None,
                 expected_hash: str | None = None) -> str:
        p = Path(path).resolve()
        if p.is_dir():
            h, manifest = sha256_tree(p)
            kind = "bundle"
        else:
            h, manifest, kind = sha256_file(p), None, "file"
        if expected_hash and h != expected_hash:
            raise ModelError(f"hash mismatch for {name}: {h} != {expected_hash}")
        e = {"name": name, "kind": kind, "format": fmt, "path": str(p),
             "licenseId": license_id}
        for k, v in (("sourceUri", source_uri), ("sourceRevision", source_revision),
                     ("licenseNote", license_note), ("manifest", manifest)):
            if v is not None:
                e[k] = v
        self.entries[h] = e
        self._verified.add(h)
        return h

    def entry(self, h: str) -> dict:
        if h not in self.entries:
            raise ModelError(f"model {h} not in registry {self.path}")
        return self.entries[h]

    def resolve(self, h: str, verify: bool = True) -> Path:
        """Return the local path of model ``h`` after (once) re-verifying bytes."""
        e = self.entry(h)
        p = Path(e["path"])
        if verify and h not in self._verified:
            got = sha256_tree(p)[0] if e["kind"] == "bundle" else sha256_file(p)
            if got != h:
                raise ModelError(f"model bytes changed: {p} hashes to {got}, expected {h}")
            self._verified.add(h)
        return p

    def is_verified(self, h: str) -> bool:
        return h in self._verified


def fetch(url: str, dest: str | os.PathLike[str], expected_hash: str) -> Path:
    """Explicit fetch step: download ``url`` and verify the SHA-256 before use."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and sha256_file(dest) == expected_hash:
        return dest
    fd, tmp = tempfile.mkstemp(dir=dest.parent, suffix=".part")
    os.close(fd)
    try:
        with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 22)
        got = sha256_file(tmp)
        if got != expected_hash:
            raise ModelError(f"downloaded {url} hashes to {got}, expected {expected_hash}")
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return dest
