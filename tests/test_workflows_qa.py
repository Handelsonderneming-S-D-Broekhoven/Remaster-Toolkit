# SPDX-FileCopyrightText: 2026 Handelsonderneming S.D. Broekhoven, trading as ITEagle Europe
# SPDX-License-Identifier: GPL-3.0-only
"""Material workflows end to end with a CPU stand-in SR op, QA metrics,
fallback ladder with reason codes, class policy and provenance schema."""

import copy

import numpy as np
import pytest

from conftest import FAKE_MODEL, cutout_texture, periodic_texture, ramp_texture, source_png
from remaster_toolkit import alpha, imagebuf, pbr, qa, workflows
from remaster_toolkit import recipe as R
from remaster_toolkit.executor import Context, build_asset

SCHEMAS = R.schemas_dir()
HAVE_SCHEMAS = SCHEMAS is not None and (SCHEMAS / R.PROVENANCE_SCHEMA).exists()
MODELS = {"sr": FAKE_MODEL,
          "checkpoint": {"role": "checkpoint", "hash": "sha256:" + "a" * 64},
          "controlnet": {"role": "controlnet", "hash": "sha256:" + "b" * 64}}


def _src_meta(img, h, kind="none", wrap=False):
    return {"hash": h, "kind": "classic", "width": img.shape[1], "height": img.shape[0],
            "colorSpace": "srgb",
            "alpha": {"present": kind != "none", "kind": kind, "premultiplied": False,
                      **({"cutoutThreshold": 0.5} if kind == "binary-cutout" else {})},
            "tileable": {"wrapU": wrap, "wrapV": wrap, "source": "curated"}}


def _resolve(body):
    return R.resolve([("material-profile", "synthetic/profile.json", body)])


def _check_schema(prov):
    if HAVE_SCHEMAS:
        assert R.validate(prov, R.PROVENANCE_SCHEMA) == []


def test_tileable_workflow_seamless_with_pbr(tmp_path, stub_sr):
    img = periodic_texture(32)
    data, h = source_png(img)
    body = workflows.build("synthetic:tile", _src_meta(img, h, wrap=True), "architecture", MODELS,
                           scale=4)
    rec = _resolve(body)
    if HAVE_SCHEMAS:
        assert R.validate(rec, R.RECIPE_SCHEMA) == []
    res = build_asset(rec, data, Context(tmp_path, device="cpu"))
    prov = res["provenance"]
    assert prov["status"] == "succeeded", prov["fallback"]
    assert set(res["files"]) == {"albedo", "height", "normal"}
    out = imagebuf.png_decode(res["files"]["albedo"])
    assert out.shape == (128, 128, 3)
    assert qa.seam_ratio(out) < 1.5
    n = imagebuf.png_decode(res["files"]["normal"])
    assert pbr.normal_validity(n)["unitFraction"] > 0.99
    assert qa.seam_ratio(n) < 2.0
    assert [o["inferred"] for o in prov["outputs"] if o["role"] == "normal"] == [True]
    _check_schema(prov)


def test_cutout_workflow_preserves_alpha(tmp_path, stub_sr):
    img = cutout_texture(32)
    data, h = source_png(img)
    body = workflows.build("synthetic:leaf", _src_meta(img, h, "binary-cutout"),
                           "vegetation-cutout", MODELS, scale=4)
    res = build_asset(_resolve(body), data, Context(tmp_path, device="cpu"))
    prov = res["provenance"]
    assert prov["status"] == "succeeded", prov["fallback"]
    out = imagebuf.png_decode(res["files"]["albedo"])
    a = out[:, :, 3]
    assert set(np.unique(a)) <= {0.0, 1.0}  # still binary
    iou = [r for r in prov["qa"]["results"] if r["metric"] == "alpha-iou"][0]
    assert iou["value"] >= 0.9 and iou["passed"]
    # dilation: no black fringe just outside... RGB of transparent texels was filled
    edge = (a == 0) & (np.pad(a, 1)[1:-1, 2:] == 1)
    assert out[:, :, 1][edge].mean() > 0.3
    _check_schema(prov)


def test_ui_and_data_workflows_have_no_generative_steps():
    img = periodic_texture(16)
    for cls in ("ui-text", "normal-map", "mask-technical"):
        body = workflows.build("synthetic:x", _src_meta(img, "sha256:" + "1" * 64), cls, MODELS,
                               generative=True)
        assert all(s["op"] not in ("controlled-diffusion", "sr") for s in body["steps"])


def test_policy_blocks_diffusion_for_forbidden_class(tmp_path, stub_sr):
    img = periodic_texture(16)
    data, h = source_png(img)
    body = workflows.build("synthetic:ui", _src_meta(img, h, wrap=True), "architecture", MODELS,
                           generative=True)
    body["textureClass"] = "ui-text"  # mislabelled recipe tries to diffuse text
    res = build_asset(_resolve(body), data, Context(tmp_path, device="cpu"))
    att = res["provenance"]["fallback"]["attempts"]
    assert att[0] == {"attemptId": "primary", "outcome": "not-run", "reason": "DIFFUSION_FORBIDDEN"}
    assert res["provenance"]["fallback"]["selectedAttemptId"] == "sr-only"
    assert res["ladder"] == "L2"
    _check_schema(res["provenance"])


def test_fallback_ladder_reason_codes(tmp_path, stub_sr):
    img = ramp_texture(16)
    data, h = source_png(img)
    body = workflows.build("synthetic:ramp", _src_meta(img, h), "character-skin", MODELS)
    body["qaGates"].append({"metric": "ssim-downscaled", "comparator": ">=", "threshold": 1.5,
                            "reference": "source", "onFail": "fallback"})
    res = build_asset(_resolve(body), data, Context(tmp_path, device="cpu"))
    prov = res["provenance"]
    att = prov["fallback"]["attempts"]
    assert [a["outcome"] for a in att] == ["gate-failed", "gate-failed", "passed"]
    assert all("STRUCT_LOW" in a["reason"] for a in att[:2])
    assert prov["fallback"]["selectedMode"] == "original" and res["ladder"] == "L0"
    assert prov["status"] == "succeeded-with-fallback"
    assert res["files"]["albedo"] == data  # original bytes, unmodified
    _check_schema(prov)


def test_step_error_falls_back(tmp_path, monkeypatch):
    from remaster_toolkit import ops

    def boom(step, inputs, ctx):
        raise RuntimeError("synthetic failure")
    monkeypatch.setitem(ops.OPS, "sr", (boom, []))
    img = periodic_texture(16)
    data, h = source_png(img)
    body = workflows.build("synthetic:err", _src_meta(img, h), "weapon", MODELS)
    res = build_asset(_resolve(body), data, Context(tmp_path, device="cpu"))
    att = res["provenance"]["fallback"]["attempts"]
    assert att[0]["outcome"] == "error" and att[0]["reason"].startswith("STEP_ERROR at sr")
    assert res["provenance"]["fallback"]["selectedMode"] == "conventional-resample"
    assert res["ladder"] == "L1"


def test_ssim_psnr_deltae_basics():
    x = periodic_texture(32)
    assert qa.ssim(qa.luma_srgb(x), qa.luma_srgb(x)) == pytest.approx(1.0)
    assert qa.delta_e2000(x, x).max() == pytest.approx(0.0, abs=1e-9)
    y = np.clip(x + 0.1, 0, 1)
    assert qa.delta_e2000(x, y).mean() > 1
    assert qa.ssim(qa.luma_srgb(x), qa.luma_srgb(np.random.default_rng(0).random(x.shape))) < 0.3


def test_deltae2000_reference_pair():
    # Sharma et al. (2005) test data pair 1: Lab (50, 2.6772, -79.7751) vs (50, 0, -82.7485)
    # -> dE00 = 2.0425. Checked through the Lab path of the implementation.
    L1 = np.array([[[50.0, 2.6772, -79.7751]]])
    L2 = np.array([[[50.0, 0.0, -82.7485]]])
    import remaster_toolkit.qa as q
    orig = q._lab
    try:
        q._lab = lambda rgb: rgb
        assert q.delta_e2000(L1, L2)[0, 0] == pytest.approx(2.0425, abs=1e-4)
    finally:
        q._lab = orig


def test_alpha_classify_and_iou():
    c = cutout_texture(32)
    assert alpha.classify(c[:, :, 3]) == "binary-cutout"
    assert alpha.classify(np.ones((4, 4))) == "none"
    assert alpha.classify(np.linspace(0, 1, 1024)) == "smooth"
    m = c[:, :, 3] > 0.5
    assert alpha.iou(m, m) == 1.0
    assert alpha.iou(m, ~m) == 0.0


def test_classical_normal_flat_and_slope():
    flat = np.full((8, 8, 1), 0.5, np.float32)
    n = pbr.normal_from_height(flat, 2.0, wrap=True)
    np.testing.assert_allclose(n, np.broadcast_to([0.5, 0.5, 1.0], n.shape), atol=1e-6)
    ramp = np.tile(np.linspace(0, 1, 8, dtype=np.float32), (8, 1))[:, :, None]
    n = pbr.normal_from_height(ramp, 2.0, wrap=False)
    assert n[4, 4, 0] < 0.5  # surface rises to +x -> normal tilts to -x


def test_horizontal_wrap_only_and_supersampled_target(tmp_path, stub_sr):
    """Sky-like strip: wraps in U only; SR x4 then area-downscale to x2."""
    tile = periodic_texture(32)
    grad = np.linspace(0.2, 1.0, 64, dtype=np.float32)[:, None, None]
    img = np.concatenate([tile, tile], axis=0) * grad  # 64x32: periodic in U, not in V
    data, h = source_png(img)
    meta = _src_meta(img, h)
    meta["tileable"] = {"wrapU": True, "wrapV": False, "source": "curated"}
    body = workflows.build("synthetic:sky", meta, "sky", MODELS, scale=4, target_scale=2)
    assert any(s["op"] == "downscale" for s in body["steps"])
    res = build_asset(_resolve(body), data, Context(tmp_path, device="cpu"))
    assert res["provenance"]["status"] == "succeeded", res["provenance"]["fallback"]
    out = imagebuf.png_decode(res["files"]["albedo"])
    assert out.shape == (128, 64, 3)
    assert qa.seam_ratio(out, wrap_u=True, wrap_v=False) < 1.5
    assert qa.seam_ratio(out, wrap_u=False, wrap_v=True) > 2.0
