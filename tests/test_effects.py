# SPDX-FileCopyrightText: 2026 Handelsonderneming S.D. Broekhoven, trading as ITEagle Europe
# SPDX-License-Identifier: GPL-3.0-only
"""Effect-texture classification, effect/UI recipes, effect QA and mips.
Synthetic, procedurally generated images only (no game data, no weights)."""

import numpy as np
import pytest

from conftest import FAKE_MODEL, source_png
from remaster_toolkit import effects, imagebuf, ops, workflows
from remaster_toolkit import recipe as R
from remaster_toolkit.executor import Context, build_asset
from remaster_toolkit.resample import resize

SCHEMAS = R.schemas_dir()
HAVE_SCHEMAS = SCHEMAS is not None and (SCHEMAS / R.PROVENANCE_SCHEMA).exists()


# ------------------------------------------------------------------ fixtures

def glow(n=32, alpha=False):
    """Additive glow: bright core, purple falloff, exact black outside a radius."""
    y, x = np.mgrid[0:n, 0:n]
    r = np.hypot(x - n / 2 + 0.5, y - n / 2 + 0.5) / (n * 0.3)
    v = np.clip(1 - r, 0, 1) ** 1.5
    core = (r < 0.15).astype(np.float64)
    rgb = np.stack([0.6 * v + core * 0.4, 0.4 * v + core * 0.6, v], axis=2)
    rgb = np.clip(rgb, 0, 1)
    img = rgb if not alpha else np.concatenate([rgb, v[:, :, None]], axis=2)
    return img.astype(np.float32)


def bolt(n=32):
    """Bolt-like strip: a 1-2 texel bright core on black (hard edges ring)."""
    img = np.zeros((n, n, 3), np.float32)
    img[:, n // 2 - 1:n // 2 + 1] = 1.0
    img[:, n // 2 - 3:n // 2 - 1] = (0.4, 0.2, 0.6)
    img[:, n // 2 + 1:n // 2 + 3] = (0.4, 0.2, 0.6)
    return img


def smoke(n=32, seed=3):
    """Straight-alpha puff with garbage colour under alpha = 0."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:n, 0:n]
    r = np.hypot(x - n / 2 + 0.5, y - n / 2 + 0.5) / (n * 0.4)
    a = np.clip(1 - r, 0, 1)
    rgb = 0.5 + 0.1 * rng.random((n, n, 3))
    rgb[a == 0] = rng.random(((a == 0).sum(), 3))  # garbage that must not bleed
    return np.concatenate([rgb, a[:, :, None]], axis=2).astype(np.float32)


def src_meta(img, h, kind="none", premult=False):
    return {"hash": h, "kind": "classic", "width": img.shape[1], "height": img.shape[0],
            "colorSpace": "srgb",
            "alpha": {"present": kind != "none", "kind": kind, "premultiplied": premult},
            "tileable": {"wrapU": False, "wrapV": False, "source": "curated"}}


def run(tmp_path, img, cls=None, kind="none", ui_kind=None, texture_class="translucent-effect",
        scale=2, ctx=None):
    data, h = source_png(img)
    body = workflows.build("synthetic:fx", src_meta(img, h, kind), texture_class, {},
                           scale=scale, effect_class=cls, ui_kind=ui_kind)
    rec = R.resolve([("material-profile", "synthetic/profile.json", body)])
    if HAVE_SCHEMAS:
        assert R.validate(rec, R.RECIPE_SCHEMA) == []
    res = build_asset(rec, data, ctx or Context(tmp_path, device="cpu"))
    if HAVE_SCHEMAS:
        assert R.validate(res["provenance"], R.PROVENANCE_SCHEMA) == []
    return res


def attempts(res):
    return {a["attemptId"]: (a["outcome"], a.get("reason")) for a in
            res["provenance"]["fallback"]["attempts"]}


# ------------------------------------------------------------------ classification

def test_classify_from_declared_blend_not_pixels():
    st = effects.alpha_stats(glow())
    assert effects.classify([{"consumer": "particle", "blend": "additive"}], st)["effectClass"] \
        == "additive"
    # the same pixels drawn opaque are a surface, not an effect
    r = effects.classify([{"consumer": "model", "blend": "opaque"}], st)
    assert r["effectClass"] is None and r["reason"] == "SURFACE_ONLY"
    r = effects.classify(None, st)
    assert r["effectClass"] is None and r["reason"] == "NO_USAGE_METADATA"


def test_classify_addalpha_without_alpha_is_additive():
    r = effects.classify([{"consumer": "particle", "blend": "additive-alpha"}],
                         effects.alpha_stats(bolt()))
    assert r["effectClass"] == "additive" and "ADDALPHA_WITHOUT_ALPHA" in r["warnings"]
    r = effects.classify([{"consumer": "particle", "blend": "additive-alpha"}],
                         effects.alpha_stats(glow(alpha=True)))
    assert r["effectClass"] == "additive-alpha"


def test_classify_lut_mixed_surface_and_exceptions():
    st = effects.alpha_stats(glow())
    assert effects.classify([{"consumer": "texel-lookup", "blend": "unknown"}],
                            st)["effectClass"] == "lut"
    r = effects.classify([{"consumer": "particle", "blend": "additive"},
                          {"consumer": "particle", "blend": "multiply"}], st)
    assert r["effectClass"] == "mixed" and r["reason"] == "CONFLICTING_EFFECT_BLENDS"
    r = effects.classify([{"consumer": "model", "blend": "additive"},
                          {"consumer": "model", "blend": "opaque"}], st)
    assert r["effectClass"] == "additive" and r["surface"] is True
    with pytest.raises(ValueError):
        effects.classify([], st, exception={"effectClass": "additive"})
    r = effects.classify([], st, exception={"effectClass": "lut", "reason": "palette read by code"})
    assert r["effectClass"] == "lut" and r["reason"] == "DECLARED_EXCEPTION"
    with pytest.raises(ValueError):
        effects.classify([{"consumer": "particle", "blend": "screen"}], st)


def test_classify_premultiplied_needs_consistent_pixels():
    s = smoke()
    st = effects.alpha_stats(s)  # straight colour, not premultiplied
    r = effects.classify([{"consumer": "particle", "blend": "premultiplied"}], st)
    assert r["effectClass"] == "mixed" and r["reason"] == "PREMULTIPLIED_INCONSISTENT"
    pm = s.copy()
    pm[:, :, :3] *= pm[:, :, 3:4]
    assert effects.classify([{"consumer": "particle", "blend": "premultiplied"}],
                            effects.alpha_stats(pm))["effectClass"] == "premultiplied"


# ------------------------------------------------------------------ resampling

def test_anti_ringing_removes_overshoot_and_keeps_black():
    b = bolt()
    plain = resize(b, 64, 64, "lanczos3", "clamp", clip=False)
    safe = resize(b, 64, 64, "lanczos3", "clamp", anti_ringing=True)
    assert plain.min() < -0.02 or plain.max() > 1.02  # Lanczos rings on a hard strip
    assert safe.min() >= 0.0 and safe.max() <= 1.0
    # columns that are black with black neighbours in the source stay exactly black
    assert float(np.abs(safe[:, :20]).max()) == 0.0
    m = effects.qa_metrics(safe, b, "additive")
    assert m["halo-fraction"] == 0.0 and m["neutral-lift"] == 0.0


def test_straight_alpha_resample_does_not_bleed_hidden_colour():
    s = smoke()
    out = effects.resample(s, 64, 64, "alpha-blend")
    m = effects.qa_metrics(out, s, "alpha-blend")
    assert m["halo-fraction"] <= 0.002 and m["neutral-lift"] == 0.0
    assert m["alpha-mae"] < 0.01
    # a naive straight-RGBA resample mixes the garbage colour into the visible edge
    naive = resize(s, 64, 64, "lanczos3", "clamp")
    assert effects.qa_metrics(naive, s, "alpha-blend")["halo-fraction"] > m["halo-fraction"]


def test_premultiplied_and_multiply_invariants():
    s = smoke()
    s[:, :, :3] *= s[:, :, 3:4]
    out = effects.resample(s, 64, 64, "premultiplied")
    assert (out[:, :, :3] <= out[:, :, 3:4] + 1e-6).all()
    mul = 1.0 - glow()  # white = neutral
    out = effects.resample(mul, 64, 64, "multiply")
    assert effects.qa_metrics(out, mul, "multiply")["neutral-lift"] == 0.0
    shade = glow()  # ZERO, INV_SRC_COLOR: black = neutral
    out = effects.resample(shade, 64, 64, "multiply-inverse")
    m = effects.qa_metrics(out, shade, "multiply-inverse")
    assert m["neutral-lift"] == 0.0 and m["halo-fraction"] == 0.0


def test_coverage_preserving_mips():
    n = 64
    y, x = np.mgrid[0:n, 0:n]
    # thin cutout detail with anti-aliased edges
    a = np.clip((np.sin(x / 1.3) * np.sin(y / 1.7) - 0.3) * 3 + 0.5, 0, 1).astype(np.float32)
    img = np.concatenate([np.full((n, n, 3), 0.5, np.float32), a[:, :, None]], axis=2)
    box = effects.mip_chain(img, "alpha-blend")
    kept = effects.mip_chain(img, "alpha-blend", cutout=0.5)
    assert effects.mip_coverage_deviation(box, 0.5) > 0.05
    assert effects.mip_coverage_deviation(kept[:4], 0.5) < 0.01  # levels >= 8x8 texels


# ------------------------------------------------------------------ recipes end to end

def test_effect_recipe_additive_end_to_end(tmp_path):
    res = run(tmp_path, bolt(), "additive")
    p = res["provenance"]
    assert p["status"] == "succeeded", p["fallback"]
    assert res["ladder"] == "L2"
    out = imagebuf.png_decode(res["files"]["albedo"])
    assert out.shape == (64, 64, 3)
    assert float(out[:, :20].max()) == 0.0
    names = {r["metric"]: r for r in p["qa"]["results"]}
    assert names["effect:halo-fraction"]["passed"] and names["effect:neutral-lift"]["passed"]
    assert p["steps"] and all(s["stepId"] != "sr" for s in p["steps"])


def test_effect_recipe_alpha_and_premultiplied(tmp_path):
    res = run(tmp_path / "a", smoke(), "alpha-blend", kind="smooth")
    assert res["provenance"]["status"] == "succeeded", res["provenance"]["fallback"]
    out = imagebuf.png_decode(res["files"]["albedo"])
    assert out.shape == (64, 64, 4)
    res = run(tmp_path / "b", glow(alpha=True), "additive-alpha", kind="smooth")
    assert res["provenance"]["status"] == "succeeded", res["provenance"]["fallback"]


@pytest.mark.parametrize("cls,reason", [("lut", "EFFECT_LUT_ORIGINAL"),
                                        ("mixed", "EFFECT_MIXED_ORIGINAL")])
def test_keep_original_classes(tmp_path, cls, reason):
    img = glow()
    res = run(tmp_path, img, cls)
    assert res["ladder"] == "L0"
    assert attempts(res)["primary"] == ("not-run", reason)
    assert res["files"]["albedo"] == source_png(img)[0]


def test_learned_sr_refused_for_effects(tmp_path, stub_sr):
    img = glow()
    data, h = source_png(img)
    body = workflows.build("synthetic:fx", src_meta(img, h), "character-skin", {"sr": FAKE_MODEL},
                           scale=2)
    body["materialProfile"] = "conservative/effect/additive"
    rec = R.resolve([("material-profile", "synthetic/profile.json", body)])
    res = build_asset(rec, data, Context(tmp_path, device="cpu"))
    assert attempts(res)["primary"] == ("not-run", "EFFECT_NO_LEARNED")


def test_effect_qa_failure_falls_back_to_original(tmp_path, monkeypatch):
    orig = ops.OPS["resample"]

    def lifting(step, inputs, ctx):  # a faulty implementation that lifts black
        out, info = orig[0](step, inputs, ctx)
        return np.clip(out + 0.05, 0, 1).astype(np.float32), info

    monkeypatch.setitem(ops.OPS, "resample", (lifting, orig[1]))
    img = bolt()
    res = run(tmp_path, img, "additive")
    outcome, reason = attempts(res)["primary"]
    assert outcome == "gate-failed" and "NEUTRAL_LIFT" in reason
    assert res["ladder"] == "L0" and res["files"]["albedo"] == source_png(img)[0]


def test_ui_text_recipe_and_glyph_atlas(tmp_path):
    n = 32
    a = np.zeros((n, n), np.float32)
    a[8:24, 10:12] = 1.0  # letter strokes
    a[8:10, 10:22] = 1.0
    img = np.concatenate([np.ones((n, n, 3), np.float32), a[:, :, None]], axis=2)
    res = run(tmp_path / "t", img, kind="binary-cutout", ui_kind="text", texture_class="ui-text")
    p = res["provenance"]
    assert p["status"] == "succeeded", p["fallback"]
    assert p["recipe"] and res["ladder"] == "L2"
    out = imagebuf.png_decode(res["files"]["albedo"])
    m = effects.qa_metrics(out, img, "alpha-blend")
    assert m["halo-fraction"] == 0.0 and m["neutral-lift"] == 0.0
    res = run(tmp_path / "g", img, kind="binary-cutout", ui_kind="glyph-atlas",
              texture_class="ui-text")
    assert attempts(res)["primary"] == ("not-run", "UI_GLYPH_ATLAS_ORIGINAL")
    assert res["ladder"] == "L0"


def _letters(n=32):
    a = np.zeros((n, n), np.float32)
    a[6:26, 6:9] = 1.0    # "L" upright
    a[23:26, 6:16] = 1.0  # "L" foot
    a[6:26, 19:22] = 1.0  # "I"
    return np.concatenate([np.ones((n, n, 3), np.float32), a[:, :, None]], axis=2)


@pytest.mark.parametrize("damage,reason", [("drop-stroke", "ALPHA_DRIFT"),
                                           ("outline", "HALO")])
def test_glyph_qa_rejects_altered_letters(tmp_path, monkeypatch, damage, reason):
    orig = ops.OPS["resample"]

    def altered(step, inputs, ctx):
        out, info = orig[0](step, inputs, ctx)
        out = out.copy()
        if damage == "drop-stroke":  # a letter loses its foot (wrong glyph)
            out[46:52, 12:32, 3] = 0.0
        else:  # an artificial dark outline around every glyph
            a = out[:, :, 3]
            grown = a.copy()
            for dy in (-2, -1, 0, 1, 2):
                for dx in (-2, -1, 0, 1, 2):
                    grown = np.maximum(grown, np.roll(np.roll(a, dy, 0), dx, 1))
            ring = (grown > 0.5) & (a < 0.5)
            out[ring, :3] = 0.0
            out[ring, 3] = 1.0
        return out.astype(np.float32), info

    monkeypatch.setitem(ops.OPS, "resample", (altered, orig[1]))
    img = _letters()
    res = run(tmp_path, img, kind="binary-cutout", ui_kind="text", texture_class="ui-text")
    outcome, why = attempts(res)["primary"]
    assert outcome == "gate-failed" and reason in why, why
    assert res["ladder"] == "L0" and res["files"]["albedo"] == source_png(img)[0]


def test_effect_classes_never_get_surface_recipe():
    img = glow(alpha=True)
    for cls in effects.CLASSES:
        body = workflows.build("synthetic:fx", src_meta(img, "sha256:" + "2" * 64, "smooth"),
                               "translucent-effect", {"sr": FAKE_MODEL}, scale=2,
                               effect_class=cls)
        ops_used = {s["op"] for s in body["steps"]}
        assert not ops_used & {"sr", "controlled-diffusion", "sharpen", "pbr-derive"}, cls
        assert [c["mode"] for c in body["fallbackPolicy"]["chain"]] == ["original"]


def test_legacy_ui_workflow_unchanged():
    img = np.zeros((8, 8, 4), np.float32)
    body = workflows.build("synthetic:ui", src_meta(img, "sha256:" + "1" * 64, "smooth"),
                           "ui-text", {}, scale=2)
    assert body["materialProfile"] == "generic/ui"
    assert "params" not in body["steps"][1]
