# SPDX-FileCopyrightText: 2026 Handelsonderneming S.D. Broekhoven, trading as ITEagle Europe
# SPDX-License-Identifier: GPL-3.0-only
import copy
import json
from pathlib import Path

import pytest

from remaster_toolkit import recipe as R

SCHEMAS = R.schemas_dir()
needs_schema = pytest.mark.skipif(SCHEMAS is None or not (SCHEMAS / R.RECIPE_SCHEMA).exists(),
                                  reason="SSTRE_SCHEMAS_DIR not set")

DEFAULTS = {
    "recipeFormatVersion": "0.1.0", "assetId": "synthetic:a", "textureClass": "unknown",
    "source": {"hash": "sha256:" + "1" * 64, "kind": "classic"},
    "steps": [{"stepId": "decode", "op": "decode"}, {"stepId": "encode", "op": "encode"}],
    "fallbackPolicy": {"chain": [{"id": "original", "mode": "original"}]},
    "compression": {"container": "png", "format": "RGBA8"},
}
PROFILE = {"textureClass": "architecture", "compression": {"format": "BC7_UNORM_SRGB"}}
OVERRIDE = {"compression": {"container": None}, "steps": [{"stepId": "decode", "op": "decode"}]}


def test_precedence_merge_rules():
    r = R.resolve([("global-defaults", "d.json", DEFAULTS), ("material-profile", "p.json", PROFILE),
                   ("curated-override", "o.json", OVERRIDE)])
    assert r["textureClass"] == "architecture"
    assert r["compression"] == {"format": "BC7_UNORM_SRGB"}  # null deleted container
    assert r["steps"] == [{"stepId": "decode", "op": "decode"}]  # arrays replace
    fs = r["precedence"]["fieldSources"]
    assert fs["/textureClass"] == "material-profile"
    assert fs["/steps"] == "curated-override"
    assert "/compression/container" not in fs
    assert [l["layer"] for l in r["precedence"]["layers"]] == [
        "global-defaults", "material-profile", "curated-override"]


def test_layer_order_enforced():
    with pytest.raises(R.RecipeError):
        R.resolve([("curated-override", "o", {}), ("global-defaults", "d", {})])


def test_resolved_hash_excludes_precedence_and_is_layout_independent():
    a = R.resolve([("global-defaults", "d", DEFAULTS), ("material-profile", "p", PROFILE)])
    merged = copy.deepcopy(DEFAULTS)
    merged.update(textureClass="architecture")
    merged["compression"]["format"] = "BC7_UNORM_SRGB"
    b = R.resolve([("curated-override", "single", merged)])
    assert a["precedence"] != b["precedence"]
    assert R.resolved_hash(a) == R.resolved_hash(b)


def test_prompt_nfc():
    doc = {"steps": [{"stepId": "x", "op": "controlled-diffusion",
                      "diffusion": {"prompt": "café"}}]}
    assert R.normalize(doc)["steps"][0]["diffusion"]["prompt"] == "café"


@needs_schema
def test_resolved_recipe_validates():
    r = R.resolve([("global-defaults", "d", DEFAULTS), ("material-profile", "p", PROFILE)])
    assert R.validate(r, R.RECIPE_SCHEMA) == []
    bad = dict(r, textureClass="not-a-class")
    assert R.validate(bad, R.RECIPE_SCHEMA)


@needs_schema
def test_schema_example_validates():
    schema = json.loads((SCHEMAS / R.RECIPE_SCHEMA).read_text(encoding="utf-8"))
    for ex in schema.get("examples", []):
        assert R.validate(ex, R.RECIPE_SCHEMA) == []
