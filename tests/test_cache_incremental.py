# SPDX-FileCopyrightText: 2026 Handelsonderneming S.D. Broekhoven, trading as ITEagle Europe
# SPDX-License-Identifier: GPL-3.0-only
"""RD-004 s7 / RD-008 s7.3: no-change, single-parameter change, formatting-only
change, byte-identical upstream output, corrupt blob, resume after kill."""

import copy
import json

import pytest

from conftest import periodic_texture, source_png
from remaster_toolkit.executor import Context, build_asset, lock_entry


def make_recipe(src_hash, amount=0.5):
    return {
        "recipeFormatVersion": "0.1.0", "assetId": "synthetic:tiles/periodic",
        "source": {"hash": src_hash, "kind": "classic", "width": 32, "height": 32,
                   "alpha": {"present": False, "kind": "none"},
                   "tileable": {"wrapU": True, "wrapV": True, "source": "curated"}},
        "textureClass": "architecture",
        "steps": [
            {"stepId": "decode", "op": "decode", "inputs": ["source"]},
            {"stepId": "rgb", "op": "alpha-split", "inputs": ["decode"], "channels": "rgb"},
            {"stepId": "up", "op": "resample", "inputs": ["rgb"], "scale": 2, "filter": "lanczos3",
             "tiling": {"edgePadding": "wrap"}},
            {"stepId": "sharp", "op": "sharpen", "inputs": ["up"], "tiling": {"edgePadding": "wrap"},
             "params": {"radius": 1, "amount": amount}},
            {"stepId": "encode", "op": "encode", "inputs": ["sharp"], "channels": "rgb"},
        ],
        "outputs": [{"role": "albedo", "fromStep": "encode"}],
        "qaGates": [{"metric": "dimensions", "comparator": "==", "threshold": "64x64",
                     "onFail": "fail-asset"}],
        "fallbackPolicy": {"chain": [{"id": "original", "mode": "original"}]},
        "precedence": {"layers": []},
    }


@pytest.fixture
def src():
    return source_png(periodic_texture(32))


def test_no_change_zero_recompute(tmp_path, src):
    data, h = src
    ctx = Context(tmp_path / "ws", device="cpu")
    r1 = build_asset(make_recipe(h), data, ctx)
    assert len(ctx.executed) == 5 and r1["provenance"]["status"] == "succeeded"
    ctx2 = Context(tmp_path / "ws", device="cpu")
    r2 = build_asset(make_recipe(h), data, ctx2)
    assert ctx2.executed == []
    assert r2["provenance"]["status"] == "skipped-cached"
    assert lock_entry(r1) == lock_entry(r2)


def test_single_parameter_change_recomputes_only_affected(tmp_path, src):
    data, h = src
    ws = tmp_path / "ws"
    build_asset(make_recipe(h), data, Context(ws, device="cpu"))
    ctx = Context(ws, device="cpu")
    r = build_asset(make_recipe(h, amount=0.8), data, ctx)
    recomputed = [s["stepId"] for s in r["provenance"]["steps"] if not s["cacheHit"]]
    assert recomputed == ["sharp", "encode"]


def test_formatting_only_change_zero_recompute(tmp_path, src):
    data, h = src
    ws = tmp_path / "ws"
    rec = make_recipe(h)
    build_asset(rec, data, Context(ws, device="cpu"))
    # reorder keys + roundtrip through pretty JSON: canonical form is identical
    text = json.dumps(rec, indent=3, sort_keys=False)
    rec2 = json.loads(text)
    rec2["steps"] = [dict(reversed(list(s.items()))) for s in rec2["steps"]]
    ctx = Context(ws, device="cpu")
    build_asset(rec2, data, ctx)
    assert ctx.executed == []


def test_byte_identical_upstream_output_keeps_downstream(tmp_path, src):
    """Changing a step's informational field recomputes that step, but its
    output bytes are unchanged, so downstream keys (chained on output hashes)
    still hit."""
    data, h = src
    ws = tmp_path / "ws"
    build_asset(make_recipe(h), data, Context(ws, device="cpu"))
    rec = make_recipe(h)
    rec["steps"][2]["implementation"] = {"name": "resample", "version": "relabelled"}
    ctx = Context(ws, device="cpu")
    r = build_asset(rec, data, ctx)
    recomputed = [s["stepId"] for s in r["provenance"]["steps"] if not s["cacheHit"]]
    assert recomputed == ["up"]


def test_corrupt_blob_detected_and_recomputed(tmp_path, src):
    data, h = src
    ws = tmp_path / "ws"
    r1 = build_asset(make_recipe(h), data, Context(ws, device="cpu"))
    sharp = [s for s in r1["provenance"]["steps"] if s["stepId"] == "sharp"][0]
    ctx = Context(ws, device="cpu")
    p = ctx.cache.blob_path(sharp["outputHashes"][0])
    p.write_bytes(b"garbage")
    r2 = build_asset(make_recipe(h), data, ctx)
    assert ctx.cache.stats["corrupt"] >= 1
    assert [s["stepId"] for s in r2["provenance"]["steps"] if not s["cacheHit"]] == ["sharp"]
    assert lock_entry(r1) == lock_entry(r2)


class Kill(BaseException):
    pass


@pytest.mark.parametrize("kill_at", [1, 3, 4])
def test_resume_after_interruption(tmp_path, src, kill_at):
    data, h = src
    ref = build_asset(make_recipe(h), data, Context(tmp_path / "ref", device="cpu"))
    ws = tmp_path / "ws"
    count = {"n": 0}

    def killer(step):
        count["n"] += 1
        if count["n"] == kill_at + 1:
            raise Kill()

    with pytest.raises(Kill):
        build_asset(make_recipe(h), data, Context(ws, device="cpu", before_step=killer))
    ctx = Context(ws, device="cpu")
    r = build_asset(make_recipe(h), data, ctx)
    assert len(ctx.executed) == 5 - kill_at
    assert lock_entry(r) == lock_entry(ref)


def test_step_key_changes_with_model_hash(tmp_path, src, stub_sr):
    from conftest import FAKE_MODEL
    data, h = src
    rec = make_recipe(h)
    rec["steps"][2] = {"stepId": "up", "op": "sr", "inputs": ["rgb"], "scale": 2,
                       "models": [FAKE_MODEL], "precision": {"compute": "fp32"},
                       "tiling": {"edgePadding": "wrap", "wrapPadding": 4}}
    rec["qaGates"] = []
    ws = tmp_path / "ws"
    build_asset(rec, data, Context(ws, device="cpu"))
    rec2 = copy.deepcopy(rec)
    rec2["steps"][2]["models"] = [dict(FAKE_MODEL, hash="sha256:" + "f" * 64)]
    ctx = Context(ws, device="cpu")
    r = build_asset(rec2, data, ctx)
    # stub output is identical, so only the model-using step is recomputed
    assert [s["stepId"] for s in r["provenance"]["steps"] if not s["cacheHit"]] == ["up"]
