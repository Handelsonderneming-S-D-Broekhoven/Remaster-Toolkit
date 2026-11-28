# SPDX-FileCopyrightText: 2026 Handelsonderneming S.D. Broekhoven, trading as ITEagle Europe
# SPDX-License-Identifier: GPL-3.0-only
"""Generic material workflows: recipe templates per texture class.

These are game-agnostic *defaults* (the "global-defaults"/"material-profile"
layer content a caller may start from); any game-specific choice belongs in
the caller's own layers. All QA thresholds here are **uncalibrated
placeholders** (RD-008 s6.1) until a human-audited benchmark sets them.

Workflows
---------
``tileable``      architecture / terrain / sky: wrap-padded SR (per-axis wrap),
                  optional controlled diffusion with circular convolution,
                  classical height + normal derivation (inferred maps).
``cutout``        alpha vegetation / grates: edge dilation, SR on RGB,
                  conservative Lanczos on alpha, re-threshold at the source
                  cutout threshold (binary alpha) or keep soft alpha.
``skin``          character / weapon UV atlases: SR with edge reflection (no
                  wrap across UV islands), alpha via Lanczos.
``ui``            UI / text: conservative resampling only (never generative).
                  With ``ui_kind`` ("text" | "glyph-atlas"): alpha-aware,
                  ringing-free resampling (no halos/outlines around lettering),
                  QA, fallback to the original; glyph atlases keep the original.
``effect``        effect textures (``effect_class``, see :mod:`effects`): alpha-
                  aware, ringing-free conventional resampling, RGB and alpha
                  independent, no learned SR / sharpening / diffusion, effect QA
                  (neutral lift, halos, energy), fallback straight to the original.
``data``          normal / height / mask / technical maps: linear-space
                  resample (normals renormalised), never generative.
"""

from __future__ import annotations

RECIPE_FORMAT = "0.1.0"

CLASS_WORKFLOW = {
    "architecture": "tileable", "terrain": "tileable", "sky": "tileable",
    "vegetation-cutout": "cutout", "translucent-effect": "skin",
    "character-skin": "skin", "weapon": "skin", "unknown": "skin",
    "animated-frame": "skin",
    "ui-text": "ui", "normal-map": "data", "height-map": "data",
    "mask-technical": "data", "lightmap": "data",
}

DEFAULT_NEGATIVE = "blurry, lowres, jpeg artifacts, text, watermark, logo, frame, border"


def _gate(metric, comp, thr, ref="source", on_fail="fallback"):
    return {"metric": metric, "comparator": comp, "threshold": thr, "reference": ref,
            "onFail": on_fail}


def _sr_step(sid, inp, model, scale, edge, wrap_pad, precision, tile=0, overlap=32, axes=None):
    st = {"stepId": sid, "op": "sr", "inputs": [inp], "models": [model], "scale": scale,
          "channels": "rgb",
          "precision": {"compute": precision, "allowTF32": False,
                        "deterministicAlgorithms": True, "attentionBackend": "math"},
          "tiling": {"tileSize": tile, "overlap": overlap if tile else 0,
                     "blend": "linear-feather", "order": "row-major",
                     "wrapPadding": wrap_pad, "edgePadding": edge}}
    if axes:
        st["params"] = dict(axes)
    return st


def build(asset_id: str, source: dict, texture_class: str, models: dict, *,
          scale: int = 4, target_scale: float | None = None, generative: bool = False,
          seed: int = 0, prompt: str | None = None, denoise: float = 0.25,
          precision: str = "fp32", sr_role: str = "sr", tile: int = 0,
          target_profile: str | None = None, circular: bool = True,
          derive_pbr: bool | None = None, workflow: str | None = None,
          effect_class: str | None = None, ui_kind: str | None = None) -> dict:
    """Return an *unresolved* recipe body (no ``precedence``) for one asset.

    ``models`` maps names (``sr``, ``checkpoint``, ``controlnet``, ...) to schema
    ``modelRef`` objects. ``source`` is the recipe ``source`` object. ``scale``
    is the SR model factor; ``target_scale`` (default = scale) the final
    factor -- when smaller, the SR output is area-downscaled before any
    refinement (supersampled SR).
    """
    wf = workflow or ("effect" if effect_class else CLASS_WORKFLOW.get(texture_class, "skin"))
    profile = f"generic/{wf}"
    ts = float(target_scale if target_scale is not None else scale)
    til = source.get("tileable") or {}
    wu, wv = bool(til.get("wrapU")), bool(til.get("wrapV"))
    wrap2d = wu and wv
    axes = {} if wu == wv else {"wrapAxes": "u" if wu else "v"}
    alpha = source.get("alpha") or {"present": False, "kind": "none"}
    akind = alpha.get("kind", "none")
    cut = float(alpha.get("cutoutThreshold", 0.5))
    edge = "wrap" if (wu or wv) else "replicate"
    w, h = int(source["width"]), int(source["height"])
    tw, th = int(round(w * ts)), int(round(h * ts))
    steps: list[dict] = [{"stepId": "decode", "op": "decode", "inputs": ["source"]}]
    outputs: list[dict] = []
    gates = [_gate("dimensions", "==", f"{tw}x{th}", on_fail="fail-asset")]
    chain: list[dict] = []

    def p(extra=None):
        d = dict(axes)
        d.update(extra or {})
        return d

    def axes_params():
        return {"params": p()} if axes else {}

    if wf in ("tileable", "skin", "cutout"):
        last = "decode"
        if akind != "none":
            steps.append({"stepId": "dilate", "op": "edge-dilate", "inputs": [last],
                          "tiling": {"edgePadding": edge}, "params": p({"threshold": cut})})
            last = "dilate"
        steps.append({"stepId": "rgb", "op": "alpha-split", "inputs": [last], "channels": "rgb"})
        steps.append(_sr_step("sr", "rgb", models[sr_role], scale,
                              "wrap" if (wu or wv) else "reflect", 32 if (wu or wv) else 8,
                              precision, tile, axes=axes))
        color = "sr"
        if ts != scale:
            steps.append({"stepId": "down", "op": "downscale", "inputs": ["sr"],
                          "targetWidth": tw, "targetHeight": th, "filter": "area",
                          "tiling": {"edgePadding": edge}, **axes_params()})
            color = "down"
        if generative and wf == "tileable":
            d = {"seed": int(seed), "rngDevice": "cpu", "noiseDtype": "float32",
                 "sampler": "euler", "scheduler": "normal", "steps": 20,
                 "denoise": float(denoise), "cfg": 5.0,
                 "prompt": prompt or "seamless tileable texture, sharp details, high quality",
                 "negativePrompt": DEFAULT_NEGATIVE, "controlnetStrength": 0.9,
                 "controlStart": 0.0, "controlEnd": 1.0, "workingResolution": 1024}
            steps.append({"stepId": "refine", "op": "controlled-diffusion", "inputs": [color],
                          "optional": True,
                          "models": [models["checkpoint"], models["controlnet"]],
                          "diffusion": d,
                          "precision": {"compute": "fp16", "vae": "fp32", "allowTF32": False,
                                        "deterministicAlgorithms": True,
                                        "attentionBackend": "math"},
                          "tiling": {"tileSize": 0, "circularConv": bool(circular and (wu or wv)),
                                     "edgePadding": edge}, **axes_params()})
            color = "refine"
            if (wu or wv) and not circular:
                steps.append({"stepId": "seamfix", "op": "seam-fix", "inputs": ["refine"],
                              "optional": True, "params": p({"band": 32})})
                color = "seamfix"
            gates += [_gate("lpips-v0.1-alex", "<=", 0.35, ref="sr-only"),
                      _gate("hf-energy-ratio", "<=", 3.0, ref="sr-only")]
            chain.append({"id": "sr-only", "mode": "skip-optional-steps"})
        if akind != "none":
            steps.append({"stepId": "a", "op": "alpha-split", "inputs": [last], "channels": "a"})
            steps.append({"stepId": "a-up", "op": "resample", "inputs": ["a"],
                          "targetWidth": tw, "targetHeight": th, "filter": "lanczos3",
                          "channels": "a", "tiling": {"edgePadding": edge}, **axes_params()})
            mp = {"cutoutThreshold": cut} if akind == "binary-cutout" else {"redilate": False}
            steps.append({"stepId": "merge", "op": "alpha-merge", "inputs": [color, "a-up"],
                          "tiling": {"edgePadding": edge}, "params": p(mp)})
            color = "merge"
            if akind == "binary-cutout":
                gates.append(_gate("alpha-iou", ">=", 0.9))
            else:
                gates.append(_gate("alpha-mae", "<=", 0.05))
        steps.append({"stepId": "encode", "op": "encode", "inputs": [color],
                      "channels": "rgba" if akind != "none" else "rgb"})
        outputs.append({"role": "albedo", "fromStep": "encode", "colorSpace": "srgb",
                        "inferred": False})
        gates += [_gate("ssim-downscaled", ">=", 0.75), _gate("deltaE2000-mean", "<=", 3.0)]
        if wu or wv:
            gates.append(_gate("seam-ratio", "<=", 2.0))
        if derive_pbr if derive_pbr is not None else (wf == "tileable" and wrap2d):
            steps += [{"stepId": "height", "op": "pbr-derive", "inputs": [color],
                       "tiling": {"edgePadding": edge},
                       "params": p({"map": "height", "highpassRadius": max(4, tw // 32)})},
                      {"stepId": "normal", "op": "pbr-derive", "inputs": ["height"],
                       "tiling": {"edgePadding": edge},
                       "params": p({"map": "normal", "strength": 4.0,
                                    "convention": "opengl-y-up"})},
                      {"stepId": "encode-height", "op": "encode", "inputs": ["height"],
                       "channels": "luma", "params": {"bits": 16}},
                      {"stepId": "encode-normal", "op": "encode", "inputs": ["normal"],
                       "channels": "rgb"}]
            outputs += [{"role": "height", "fromStep": "encode-height", "colorSpace": "linear",
                         "inferred": True},
                        {"role": "normal", "fromStep": "encode-normal", "colorSpace": "linear",
                         "normalConvention": "opengl-y-up", "inferred": True}]
        chain += [{"id": "conventional", "mode": "conventional-resample"},
                  {"id": "original", "mode": "original"}]
    elif wf == "effect" or (wf == "ui" and ui_kind):
        if wf == "effect":
            from .effects import CLASSES
            if effect_class not in CLASSES:
                raise ValueError(f"effect workflow needs effect_class in {CLASSES}")
            sem = effect_class if effect_class not in ("lut", "mixed") else "additive"
            profile = f"generic/effect/{effect_class}"
            ssim_min = 0.9
        else:
            if ui_kind not in ("text", "glyph-atlas"):
                raise ValueError("ui_kind must be 'text' or 'glyph-atlas'")
            sem = "alpha-blend" if akind != "none" else "additive"
            profile = f"generic/ui/{ui_kind}"
            ssim_min = 0.9
        steps.append({"stepId": "up", "op": "resample", "inputs": ["decode"],
                      "targetWidth": tw, "targetHeight": th, "filter": "lanczos3",
                      "channels": "rgba" if akind != "none" else "rgb",
                      "tiling": {"edgePadding": edge},
                      "params": p({"effectSemantics": sem, "antiRinging": True})})
        steps.append({"stepId": "encode", "op": "encode", "inputs": ["up"],
                      "channels": "rgba" if akind != "none" else "rgb"})
        outputs.append({"role": "albedo", "fromStep": "encode", "colorSpace": "srgb",
                        "inferred": False})
        gates += [_gate("ssim-downscaled", ">=", ssim_min)]
        if akind != "none":
            gates.append(_gate("alpha-mae", "<=", 0.03))
            if wf == "ui":  # glyph shapes: changed/missing strokes fail the IoU
                gates.append(_gate("alpha-iou", ">=", 0.9))
        if wu or wv:
            gates.append(_gate("seam-ratio", "<=", 2.0))
        # no conventional-resample rung: the generic Lanczos fallback is not effect-safe
        chain += [{"id": "original", "mode": "original"}]
    elif wf == "ui":
        steps.append({"stepId": "up", "op": "resample", "inputs": ["decode"],
                      "targetWidth": tw, "targetHeight": th, "filter": "lanczos3",
                      "channels": "rgba", "tiling": {"edgePadding": "replicate"}})
        steps.append({"stepId": "encode", "op": "encode", "inputs": ["up"],
                      "channels": "rgba" if akind != "none" else "rgb"})
        outputs.append({"role": "albedo", "fromStep": "encode", "colorSpace": "srgb",
                        "inferred": False})
        gates += [_gate("ssim-downscaled", ">=", 0.85), _gate("deltaE2000-mean", "<=", 2.0)]
        if akind != "none":
            gates.append(_gate("alpha-mae", "<=", 0.05))
        chain += [{"id": "original", "mode": "original"}]
    else:  # data maps
        normal = texture_class == "normal-map"
        steps.append({"stepId": "up", "op": "resample", "inputs": ["decode"],
                      "targetWidth": tw, "targetHeight": th, "filter": "lanczos3",
                      "channels": "rgb", "tiling": {"edgePadding": edge},
                      "params": p({"normalMap": normal})})
        steps.append({"stepId": "encode", "op": "encode", "inputs": ["up"], "channels": "rgb"})
        outputs.append({"role": "normal" if normal else "albedo", "fromStep": "encode",
                        "colorSpace": "linear", "inferred": False,
                        **({"normalConvention": "opengl-y-up"} if normal else {})})
        gates += [_gate("ssim-downscaled", ">=", 0.85)]
        if wu or wv:
            gates.append(_gate("seam-ratio", "<=", 2.0))
        chain += [{"id": "original", "mode": "original"}]
    recipe = {
        "recipeFormatVersion": RECIPE_FORMAT,
        "assetId": asset_id,
        "source": source,
        "textureClass": texture_class,
        "materialProfile": profile,
        "steps": steps,
        "outputs": outputs,
        "qaGates": gates,
        "fallbackPolicy": {"chain": chain, "maxAttempts": len(chain)},
    }
    if target_profile:
        recipe["targetProfile"] = target_profile
    return recipe
