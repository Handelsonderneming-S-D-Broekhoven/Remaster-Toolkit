# SPDX-FileCopyrightText: 2026 Handelsonderneming S.D. Broekhoven, trading as ITEagle Europe
# SPDX-License-Identifier: GPL-3.0-only
"""Content-addressed step cache (RD-004 sections 6-7).

Layout under a workspace root::

    cas/sha256/ab/<hex>        blobs, named by the SHA-256 of their bytes
    index/<stepKey hex>.json   {"outputHash": ..., "meta": {...}}

Writes are atomic (temp file in the same directory, fsync, ``os.replace``),
so an interrupted build never leaves a partially written blob or index entry
visible; a rerun re-plans and skips every step whose index entry points to a
present, hash-verified blob (resume). Corrupt blobs are detected by re-hashing
and recomputed.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from .hashing import hex_of, sha256_bytes
from .jcs import canonicalize


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


class Cache:
    def __init__(self, root: str | os.PathLike[str]):
        self.root = Path(root)
        self.stats = {"hits": 0, "misses": 0, "corrupt": 0}

    def blob_path(self, h: str) -> Path:
        x = hex_of(h)
        return self.root / "cas" / "sha256" / x[:2] / x

    def put_blob(self, data: bytes) -> str:
        h = sha256_bytes(data)
        p = self.blob_path(h)
        if not p.exists():
            _atomic_write(p, data)
        return h

    def get_blob(self, h: str) -> bytes | None:
        p = self.blob_path(h)
        if not p.exists():
            return None
        data = p.read_bytes()
        if sha256_bytes(data) != h:
            self.stats["corrupt"] += 1
            return None
        return data

    def _index_path(self, step_key: str) -> Path:
        return self.root / "index" / (hex_of(step_key) + ".json")

    def lookup(self, step_key: str) -> tuple[str, bytes, dict] | None:
        p = self._index_path(step_key)
        if not p.exists():
            self.stats["misses"] += 1
            return None
        rec = json.loads(p.read_text(encoding="utf-8"))
        data = self.get_blob(rec["outputHash"])
        if data is None:
            self.stats["misses"] += 1
            return None
        self.stats["hits"] += 1
        return rec["outputHash"], data, rec.get("meta", {})

    def commit(self, step_key: str, data: bytes, meta: dict | None = None) -> str:
        h = self.put_blob(data)
        _atomic_write(self._index_path(step_key),
                      canonicalize({"outputHash": h, "meta": meta or {}}))
        return h
