# SPDX-FileCopyrightText: 2026 Handelsonderneming S.D. Broekhoven, trading as ITEagle Europe
# SPDX-License-Identifier: GPL-3.0-only
"""Resolved recipes: layered resolution, validation, canonical hash (RD-004 s4-5).

The toolkit is game-agnostic: the caller supplies the ordered layer list.
Merge rules ``deep-merge/arrays-replace/null-deletes@1``: objects merge key by
key from lowest to highest precedence, arrays replace, ``null`` deletes.
"""

from __future__ import annotations

import copy
import json
import os
import unicodedata
from pathlib import Path
from typing import Any, Iterable

from .hashing import jcs_hash

LAYERS = ("global-defaults", "material-profile", "generated-recipe", "curated-override")
MERGE_RULES = "deep-merge/arrays-replace/null-deletes@1"
HASH_EXCLUDED = ("precedence", "$comment")
RECIPE_SCHEMA = "asset-recipe.schema.json"
PROVENANCE_SCHEMA = "build-provenance.schema.json"


class RecipeError(ValueError):
    pass


def _merge(base: Any, over: Any, path: str, sources: dict, layer: str) -> Any:
    if isinstance(base, dict) and isinstance(over, dict):
        out = dict(base)
        for k, v in over.items():
            p = path + "/" + k.replace("~", "~0").replace("/", "~1")
            if v is None:
                out.pop(k, None)
                for sk in [s for s in sources if s == p or s.startswith(p + "/")]:
                    del sources[sk]
                continue
            out[k] = _merge(base.get(k), v, p, sources, layer)
        return out
    # leaf (scalar or array): replace
    for sk in [s for s in sources if s == path or s.startswith(path + "/")]:
        del sources[sk]
    sources[path or "/"] = layer
    return copy.deepcopy(over)


def resolve(layers: Iterable[tuple[str, str, dict]]) -> dict:
    """Resolve ``[(layer_name, ref, document), ...]`` lowest precedence first."""
    result: dict = {}
    sources: dict[str, str] = {}
    recorded = []
    last = -1
    for name, ref, doc in layers:
        if name not in LAYERS:
            raise RecipeError(f"unknown layer {name}")
        idx = LAYERS.index(name)
        if idx < last:
            raise RecipeError("layers must be ordered lowest precedence first")
        last = idx
        result = _merge(result, doc, "", sources, name)
        recorded.append({"layer": name, "ref": ref, "hash": jcs_hash(doc)})
    result["precedence"] = {"layers": recorded, "fieldSources": dict(sorted(sources.items())),
                            "mergeRules": MERGE_RULES}
    return normalize(result)


def normalize(recipe: dict) -> dict:
    """Apply NFC to prompt strings (recipe format 0.x rule, RD-004 s5)."""
    r = copy.deepcopy(recipe)
    for st in r.get("steps", []):
        d = st.get("diffusion")
        if d:
            for k in ("prompt", "negativePrompt"):
                if isinstance(d.get(k), str):
                    d[k] = unicodedata.normalize("NFC", d[k])
    return r


def resolved_hash(recipe: dict) -> str:
    return jcs_hash({k: v for k, v in recipe.items() if k not in HASH_EXCLUDED})


def schemas_dir(explicit: str | None = None) -> Path | None:
    for cand in (explicit, os.environ.get("SSTRE_SCHEMAS_DIR")):
        if cand and Path(cand).is_dir():
            return Path(cand)
    return None


def validate(doc: dict, schema_name: str, schemas: str | None = None) -> list[str]:
    """Validate against an umbrella schema. Returns error strings ([] = valid).

    Raises RecipeError if the schema or ``jsonschema`` is unavailable.
    """
    d = schemas_dir(schemas)
    if d is None or not (d / schema_name).exists():
        raise RecipeError(f"schema {schema_name} not found (set SSTRE_SCHEMAS_DIR)")
    try:
        import jsonschema
    except ImportError as e:  # pragma: no cover
        raise RecipeError("jsonschema not installed") from e
    schema = json.loads((d / schema_name).read_text(encoding="utf-8"))
    v = jsonschema.Draft202012Validator(schema)
    return [f"{'/'.join(map(str, e.absolute_path))}: {e.message}" for e in v.iter_errors(doc)]


def load(path: str | os.PathLike[str]) -> dict:
    return normalize(json.loads(Path(path).read_text(encoding="utf-8")))
