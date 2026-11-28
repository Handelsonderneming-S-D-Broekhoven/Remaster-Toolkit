# SPDX-FileCopyrightText: 2026 Handelsonderneming S.D. Broekhoven, trading as ITEagle Europe
# SPDX-License-Identifier: GPL-3.0-only
"""Recipe executor: cached steps, QA gates, fallback ladder, provenance.

One call to :func:`build_asset` runs a resolved recipe on one source image:

1. Steps are planned; each gets a content-addressed ``stepKey`` chained on the
   *output hashes* of its inputs (RD-004 s6). Cached outputs are reused, new
   ones are committed immediately (resume after interruption = rerun).
2. The class policy rejects generative steps for forbidden classes.
3. QA gates are evaluated on the albedo output; a failing gate with
   ``onFail: fallback`` (or a step error / OOM) descends the fallback ladder
   (primary -> skip optional steps -> conventional resample -> original),
   recording reason codes.
4. A build-provenance record (umbrella schema) is returned.
"""

from __future__ import annotations

import copy
import datetime as _dt
import time
import uuid
from pathlib import Path
from typing import Callable

import numpy as np

from . import __version__, effects, imagebuf, perceptual, policy, qa
from .cache import Cache
from .env import cpu_runtime_class, environment, executor_identity, gpu_runtime_class
from .hashing import jcs_hash, sha256_bytes
from .models import ModelStore
from .ops import GPU_OPS, OPS, impl_hash
from .recipe import resolved_hash

KEY_VERSION = 1
PROVENANCE_FORMAT = "0.1.0"
REASON = {
    "ssim-downscaled": "STRUCT_LOW", "psnr-downscaled": "STRUCT_LOW",
    "lpips-v0.1-alex": "STRUCT_LOW", "lpips-v0.1-vgg": "STRUCT_LOW", "dists": "STRUCT_LOW",
    "hf-energy-ratio": "STRUCT_LOW", "deltaE2000-mean": "COLOR_DRIFT",
    "deltaE2000-p95": "COLOR_DRIFT", "seam-ratio": "SEAM", "alpha-iou": "ALPHA_DRIFT",
    "alpha-mae": "ALPHA_DRIFT", "dimensions": "DIMS", "custom": "CUSTOM",
}


class StepFailure(RuntimeError):
    def __init__(self, step_id: str, reason: str, detail: str):
        super().__init__(f"{step_id}: {reason}: {detail}")
        self.step_id, self.reason, self.detail = step_id, reason, detail


class Context:
    """Per-process build context (cache, models, device, counters)."""

    def __init__(self, workspace: str | Path, models: ModelStore | None = None,
                 device: str | None = None, settings: dict | None = None,
                 perceptual_metrics: bool = False,
                 before_step: Callable[[dict], None] | None = None):
        self.workspace = Path(workspace)
        self.cache = Cache(self.workspace / "cache")
        self.models = models
        if device is None:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        self.device = device
        self.settings = dict(settings or {})
        self.perceptual = perceptual_metrics
        self.before_step = before_step
        self.executed: list[str] = []  # stepKeys actually computed (not cached)

    def model_entry(self, h: str) -> dict:
        if self.models is None:
            raise RuntimeError("no model store configured")
        return self.models.entry(h)

    def model_path(self, h: str) -> Path:
        if self.models is None:
            raise RuntimeError("no model store configured")
        return self.models.resolve(h)


# ---------------------------------------------------------------- planning

def _with_default_inputs(steps: list[dict]) -> list[dict]:
    out, prev, seen = [], "source", set()
    for s in steps:
        s = copy.deepcopy(s)
        if s["stepId"] in seen or s["stepId"] == "source":
            raise ValueError(f"duplicate stepId {s['stepId']}")
        s.setdefault("inputs", [prev])
        for i in s["inputs"]:
            if i != "source" and i not in seen:
                raise ValueError(f"step {s['stepId']} input {i} not defined earlier")
        if s["op"] not in OPS:
            raise ValueError(f"op {s['op']} not implemented")
        seen.add(s["stepId"])
        prev = s["stepId"]
        out.append(s)
    return out


def skip_optional(steps: list[dict]) -> list[dict]:
    steps = _with_default_inputs(steps)
    removed: dict[str, str] = {}
    out = []
    for s in steps:
        ins = [removed.get(i, i) for i in s["inputs"]]
        if s.get("optional"):
            removed[s["stepId"]] = ins[0]
            continue
        s = dict(s, inputs=ins)
        out.append(s)
    return out


def infer_size(steps: list[dict], w: int, h: int) -> dict[str, tuple[int, int]]:
    from .ops import _target

    sizes = {"source": (w, h)}
    for s in _with_default_inputs(steps):
        iw, ih = sizes[s["inputs"][0]]
        if s["op"] == "sr":
            f = int(s.get("scale", 4))
            sizes[s["stepId"]] = (iw * f, ih * f)
        elif s["op"] in ("resample", "downscale"):
            sizes[s["stepId"]] = _target(s, ih, iw)
        else:
            sizes[s["stepId"]] = (iw, ih)
    return sizes


def conventional_steps(recipe: dict, tw: int, th: int) -> list[dict]:
    src = recipe.get("source", {})
    til = src.get("tileable") or {}
    wu, wv = bool(til.get("wrapU")), bool(til.get("wrapV"))
    edge = {"edgePadding": "wrap" if (wu or wv) else "replicate"}
    axes = {} if (wu == wv) else {"wrapAxes": "u" if wu else "v"}
    al = src.get("alpha") or {}
    kind = al.get("kind", "none")
    st = [{"stepId": "fb-decode", "op": "decode", "inputs": ["source"]}]
    last = "fb-decode"
    if kind != "none":
        st.append({"stepId": "fb-dilate", "op": "edge-dilate", "inputs": [last], "tiling": edge,
                   "params": {"threshold": al.get("cutoutThreshold", 0.5), **axes}})
        last = "fb-dilate"
    st.append({"stepId": "fb-resample", "op": "resample", "inputs": [last], "filter": "lanczos3",
               "targetWidth": tw, "targetHeight": th, "channels": "rgba", "tiling": edge,
               **({"params": axes} if axes else {})})
    last = "fb-resample"
    if kind == "binary-cutout":
        st += [{"stepId": "fb-rgb", "op": "alpha-split", "inputs": [last], "channels": "rgb"},
               {"stepId": "fb-a", "op": "alpha-split", "inputs": [last], "channels": "a"},
               {"stepId": "fb-merge", "op": "alpha-merge", "inputs": ["fb-rgb", "fb-a"],
                "tiling": edge, "params": {"cutoutThreshold": al.get("cutoutThreshold", 0.5),
                                           **axes}}]
        last = "fb-merge"
    st.append({"stepId": "fb-encode", "op": "encode", "inputs": [last],
               "channels": "rgba" if kind != "none" else "rgb"})
    return st


# ---------------------------------------------------------------- execution

def step_key(step: dict, input_hashes: list[str], runtime: str, target: dict | None) -> str:
    doc = {
        "keyVersion": KEY_VERSION,
        "inputs": input_hashes,
        "step": step,
        "models": [m["hash"] for m in step.get("models", [])],
        "code": {"executor": f"remaster-toolkit {__version__}", "stepImpl": impl_hash(step["op"])},
        "runtime": runtime,
    }
    if target is not None:
        doc["target"] = target
    return jcs_hash(doc)


def _runtime_class(step: dict, ctx: Context) -> str:
    if step["op"] in GPU_OPS and ctx.device != "cpu":
        return gpu_runtime_class(step.get("precision") or {})
    if step["op"] in GPU_OPS:
        return "cpu-torch-" + cpu_runtime_class()
    return cpu_runtime_class()


def run_steps(steps: list[dict], source: bytes, source_hash: str, recipe: dict,
              ctx: Context) -> tuple[dict, list[dict]]:
    """Execute ``steps``; returns ({stepId: (hash, data)}, step records)."""
    steps = _with_default_inputs(steps)
    hashes: dict[str, str] = {"source": source_hash}
    blobs: dict[str, bytes] = {"source": source}
    images: dict[str, np.ndarray] = {}
    records = []
    target = None
    if recipe.get("targetProfile"):
        target = {"targetProfile": recipe["targetProfile"]}

    def img(sid: str) -> np.ndarray:
        if sid not in images:
            images[sid] = imagebuf.from_bytes(blobs[sid])
        return images[sid]

    for s in steps:
        ins = s["inputs"]
        runtime = _runtime_class(s, ctx)
        key = step_key(s, [hashes[i] for i in ins], runtime,
                       target if s["op"] == "encode" else None)
        rec = {"stepId": s["stepId"], "stepKey": key, "inputHashes": [hashes[i] for i in ins],
               "runtimeClass": runtime}
        hit = ctx.cache.lookup(key)
        models = []
        for m in s.get("models", []):
            e = ctx.models.entries.get(m["hash"], {}) if ctx.models else {}
            models.append({"role": m["role"], "hash": m["hash"],
                           "verified": bool(ctx.models and ctx.models.is_verified(m["hash"])),
                           "licenseId": m.get("licenseId") or e.get("licenseId", "NOASSERTION")})
        if hit is not None:
            h, data, meta = hit
            rec.update(cacheHit=True, outputHashes=[h])
            if "decodedPixelHash" in meta:
                rec["decodedPixelHash"] = meta["decodedPixelHash"]
        else:
            if ctx.before_step:
                ctx.before_step(s)
            fn = OPS[s["op"]][0]
            inputs = [blobs["source"]] if s["op"] == "decode" else [img(i) for i in ins]
            peak = None
            cuda = s["op"] in GPU_OPS and ctx.device.startswith("cuda")
            if cuda:
                import torch
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            try:
                out, info = fn(s, inputs, ctx)
            except Exception as e:  # noqa: BLE001 - per-asset isolation
                reason = "OOM" if type(e).__name__ == "OutOfMemoryError" else "STEP_ERROR"
                raise StepFailure(s["stepId"], reason, f"{type(e).__name__}: {e}") from e
            if cuda:
                torch.cuda.synchronize()
                peak = int(torch.cuda.max_memory_allocated())
                info = dict(info, peakVramReservedBytes=int(torch.cuda.max_memory_reserved()))
            wall = time.perf_counter() - t0
            if isinstance(out, (bytes, bytearray)):
                data = bytes(out)
                meta = {"decodedPixelHash": imagebuf.decoded_pixel_hash(imagebuf.png_decode(data))}
            else:
                data = imagebuf.to_bytes(out)
                images[s["stepId"]] = imagebuf.from_bytes(data)
                meta = {}
            meta.update(wallTimeSeconds=wall, **({"peakVramBytes": peak} if peak is not None else {}),
                        info=info)
            h = ctx.cache.commit(key, data, meta)
            ctx.executed.append(key)
            rec.update(cacheHit=False, outputHashes=[h], wallTimeSeconds=wall,
                       deterministicExceptions=[], error=None)
            if "decodedPixelHash" in meta:
                rec["decodedPixelHash"] = meta["decodedPixelHash"]
            if peak is not None:
                rec["peakVramBytes"] = peak
        if models:
            rec["models"] = models
        hashes[s["stepId"]] = h
        blobs[s["stepId"]] = data
        records.append(rec)
    return {k: (hashes[k], blobs[k]) for k in hashes}, records


# ---------------------------------------------------------------- QA

def _cmp(v, comp: str, thr) -> bool:
    if comp == "<=":
        return v <= thr
    if comp == ">=":
        return v >= thr
    return v == thr


def _effect_semantics(recipe: dict) -> str | None:
    """Semantics for the implicit effect QA: the effect class, or for alpha-aware UI
    recipes straight alpha (alpha present) / plain colour (no alpha)."""
    cls = policy.profile_segment(recipe, "effect")
    if cls is not None:
        return None if cls in policy.EFFECT_KEEP_ORIGINAL else cls
    if policy.profile_segment(recipe, "ui") is not None:
        kind = ((recipe.get("source") or {}).get("alpha") or {}).get("kind", "none")
        return "alpha-blend" if kind != "none" else "additive"
    return None


def evaluate_gates(recipe: dict, out_img: np.ndarray, src_img: np.ndarray,
                   sr_only: np.ndarray | None, ctx: Context, attempt: str) -> tuple[list, list, dict]:
    src = recipe.get("source", {})
    til = src.get("tileable") or {}
    al = src.get("alpha") or {}
    kind = al.get("kind", "none")
    cut = float(al.get("cutoutThreshold", 0.5))
    m = qa.compare(out_img, src_img, bool(til.get("wrapU")), bool(til.get("wrapV")),
                   kind, cut, reference=sr_only)
    gates = recipe.get("qaGates", [])
    need_p = ctx.perceptual or any(g["metric"] in ("lpips-v0.1-alex", "dists") for g in gates)
    if need_p:
        from .resample import downscale_area
        sh, sw = src_img.shape[:2]
        mode = ("wrap" if til.get("wrapV") else "clamp", "wrap" if til.get("wrapU") else "clamp")
        down = downscale_area(out_img, sw, sh, mode)
        def pm(x):  # premultiply by source/own alpha so hidden colour is ignored
            return x[:, :, :3] * x[:, :, 3:4] if x.shape[2] == 4 else x
        for k, v in perceptual.compare(pm(down), pm(src_img)).items():
            m[k] = v
        if sr_only is not None and sr_only.shape == out_img.shape:
            for k, v in perceptual.compare(pm(out_img), pm(sr_only)).items():
                m[k + "@sr-only"] = v
    import math
    # JSON/JCS cannot carry non-finite numbers (e.g. PSNR of identical images = inf):
    # record them as strings ("inf", "nan") so provenance stays canonicalisable.
    for k, v in list(m.items()):
        if isinstance(v, float) and not math.isfinite(v):
            m[k] = "inf" if v > 0 else ("-inf" if v < 0 else "nan")
    results, reasons = [], []
    # implicit hard checks (RD-008 s6.1)
    hard = []
    if not m["aspect-preserved"]:
        hard.append(("aspect", "DIMS"))
    if kind != "none" and out_img.shape[2] != 4:
        hard.append(("alpha-presence", "ALPHA_MISSING"))
    if recipe.get("textureClass") == "normal-map":
        from .pbr import normal_validity
        nv = normal_validity(out_img)
        m["normal-unit-fraction"] = nv["unitFraction"]
        if nv["unitFraction"] < 0.99 or nv["zPositiveFraction"] < 0.999:
            hard.append(("normal-validity", "NORMAL_INVALID"))
    for name, code in hard:
        results.append({"metric": name, "value": None, "passed": False, "attemptId": attempt})
        reasons.append((code, "fail-asset" if code != "ALPHA_MISSING" else "fallback"))
    # implicit effect / UI checks (effects.QA_LIMITS): the gate metric enum of the recipe
    # schema has no effect metrics yet, so they are evaluated here and recorded by name.
    sem = _effect_semantics(recipe)
    if sem is not None:
        mode = ("wrap" if til.get("wrapV") else "clamp", "wrap" if til.get("wrapU") else "clamp")
        is_ui = policy.profile_segment(recipe, "effect") is None
        em = effects.qa_metrics(out_img, src_img, sem, mode,
                                cut if kind == "binary-cutout" else None, coverage=not is_ui)
        for k, v in em.items():
            m["effect:" + k] = v
            lim = effects.QA_LIMITS.get(k)
            if lim is None:
                continue
            ok = v <= lim
            results.append({"metric": "effect:" + k, "value": v, "threshold": lim,
                            "comparator": "<=", "reference": "source", "passed": bool(ok),
                            "attemptId": attempt})
            if not ok:
                reasons.append((effects.REASONS[k], "fallback"))
    for g in gates:
        metric, ref = g["metric"], g.get("reference", "source")
        key = metric + "@sr-only" if (ref == "sr-only" and metric in ("lpips-v0.1-alex", "dists")) else metric
        v = m.get(key)
        thr = g["threshold"]
        if metric == "dimensions":
            ok = (m["pot"] if thr == "pot" else m["dimensions"] == thr)
            v = m["dimensions"]
        elif v is None or isinstance(v, str):
            ok = None  # metric unavailable or non-finite: recorded, non-blocking
        else:
            ok = bool(_cmp(float(v), g["comparator"], float(thr)))
        results.append({"metric": metric, "value": v, "threshold": thr,
                        "comparator": g["comparator"], "reference": ref,
                        "passed": bool(ok), "attemptId": attempt})
        if ok is False and g.get("onFail", "fallback") != "warn":
            reasons.append((REASON.get(metric, "CUSTOM"), g.get("onFail", "fallback")))
    # informational signals
    for k in sorted(m):
        if k in {g["metric"] for g in gates} or k == "dimensions":
            continue
        if k.startswith("effect:") and k[7:] in effects.QA_LIMITS:
            continue
        v = m[k]
        if isinstance(v, bool):
            v = str(v).lower()
        results.append({"metric": k, "value": v, "passed": True, "attemptId": attempt,
                        "reference": "sr-only" if k.endswith("@sr-only") else "source"})
    return results, reasons, m


# ---------------------------------------------------------------- build

def _albedo_step(recipe: dict, steps: list[dict]) -> str:
    ids = [s["stepId"] for s in steps]
    for o in recipe.get("outputs", []):
        if o["role"] == "albedo" and o["fromStep"] in ids:
            return o["fromStep"]
    return ids[-1]


def attempts_for(recipe: dict, src_w: int, src_h: int) -> list[tuple[str, str, list[dict] | None]]:
    """[(attemptId, mode, steps)] in ladder order; steps None for 'original'."""
    prim = recipe["steps"]
    out = [("primary", "primary", prim)]
    sizes = infer_size(prim, src_w, src_h)
    final = _albedo_step(recipe, _with_default_inputs(prim))
    tw, th = sizes[final]
    for e in recipe["fallbackPolicy"]["chain"]:
        mode = e["mode"]
        if mode == "skip-optional-steps":
            out.append((e["id"], mode, skip_optional(prim)))
        elif mode == "alternate-steps":
            out.append((e["id"], mode, e["steps"]))
        elif mode == "conventional-resample":
            out.append((e["id"], mode, conventional_steps(recipe, tw, th)))
        elif mode == "original":
            out.append((e["id"], mode, None))
    max_att = recipe["fallbackPolicy"].get("maxAttempts")
    if max_att:
        out = out[:1 + int(max_att)]
    return out


def build_asset(recipe: dict, source: bytes, ctx: Context) -> dict:
    """Run ``recipe`` on ``source`` bytes. Returns {provenance, outputs, metrics}."""
    started = _dt.datetime.now(_dt.timezone.utc)
    source_hash = sha256_bytes(source)
    rhash = resolved_hash(recipe)
    if recipe["source"]["hash"] != source_hash:
        raise ValueError(f"source bytes hash {source_hash} != recipe source.hash")
    src_img = imagebuf.ensure_rgba(imagebuf.png_decode(source))
    sh, sw = src_img.shape[:2]
    if (recipe.get("source", {}).get("alpha") or {}).get("kind", "none") == "none":
        src_img = src_img[:, :, :3]
    attempts = attempts_for(recipe, sw, sh)
    att_records, qa_results = [], []
    chosen = None
    status = "failed"
    for att_id, mode, steps in attempts:
        if steps is None:  # original
            chosen = (att_id, mode, {"albedo": (source_hash, source)}, [], src_img)
            att_records.append({"attemptId": att_id, "outcome": "passed",
                                "reason": "original asset, unmodified"})
            status = "succeeded-with-fallback"
            break
        code = policy.check(recipe, steps)
        if code:
            att_records.append({"attemptId": att_id, "outcome": "not-run", "reason": code})
            continue
        try:
            outs, recs = run_steps(steps, source, source_hash, recipe, ctx)
        except StepFailure as f:
            att_records.append({"attemptId": att_id, "outcome": "error",
                                "reason": f"{f.reason} at {f.step_id}: {f.detail}"[:500]})
            continue
        alb = _albedo_step(recipe, _with_default_inputs(steps))
        out_img = imagebuf.png_decode(outs[alb][1])
        sr_only = None
        if any(s.get("optional") for s in steps):
            try:
                so, _ = run_steps(skip_optional(steps), source, source_hash, recipe, ctx)
                sr_only = imagebuf.png_decode(so[_albedo_step(recipe, skip_optional(steps))][1])
            except StepFailure:
                sr_only = None
        res, reasons, _m = evaluate_gates(recipe, out_img, src_img, sr_only, ctx, att_id)
        qa_results += res
        if any(r[1] == "fail-asset" for r in reasons):
            att_records.append({"attemptId": att_id, "outcome": "gate-failed",
                                "reason": ",".join(sorted({r[0] for r in reasons})) + " (fail-asset)"})
            status = "failed"
            chosen = None
            break
        if reasons:
            att_records.append({"attemptId": att_id, "outcome": "gate-failed",
                                "reason": ",".join(sorted({r[0] for r in reasons}))})
            continue
        att_records.append({"attemptId": att_id, "outcome": "passed"})
        produced = {}
        ids = {s["stepId"] for s in _with_default_inputs(steps)}
        for o in recipe.get("outputs", []) or [{"role": "albedo", "fromStep": alb}]:
            sid = alb if o["role"] == "albedo" else o["fromStep"]
            if sid in ids:
                produced[o["role"]] = outs[sid]
        if not produced:
            produced["albedo"] = outs[alb]
        chosen = (att_id, mode, produced, recs, out_img)
        all_hit = all(r["cacheHit"] for r in recs)
        status = ("skipped-cached" if all_hit else "succeeded") if att_id == "primary" \
            else "succeeded-with-fallback"
        break
    # attempts that were never reached
    done = {a["attemptId"] for a in att_records}
    for att_id, _, _ in attempts:
        if att_id not in done:
            att_records.append({"attemptId": att_id, "outcome": "not-run"})

    step_records = chosen[3] if chosen else []
    asset_key = jcs_hash({"keyVersion": KEY_VERSION, "assetId": recipe["assetId"],
                          "sourceHash": source_hash, "resolvedRecipeHash": rhash,
                          "stepKeys": [r["stepKey"] for r in step_records],
                          "targetProfileHash": jcs_hash(recipe.get("targetProfile"))})
    outputs = []
    files = {}
    if chosen:
        inferred_roles = {o["role"]: o.get("inferred", False) for o in recipe.get("outputs", [])}
        for role, (h, data) in sorted(chosen[2].items()):
            im = imagebuf.png_decode(data)
            outputs.append({"role": role, "hash": h,
                            "decodedPixelHash": imagebuf.decoded_pixel_hash(im),
                            "mediaType": "image/png",
                            "format": {1: "R8", 3: "RGB8", 4: "RGBA8"}.get(im.shape[2], "?"),
                            "width": int(im.shape[1]), "height": int(im.shape[0]),
                            "mipLevels": 1, "sizeBytes": len(data),
                            "inferred": bool(inferred_roles.get(role, False))})
            files[role] = data
    layers = recipe.get("precedence", {}).get("layers", [])
    prov = {
        "provenanceFormatVersion": PROVENANCE_FORMAT,
        "assetId": recipe["assetId"],
        "status": status,
        "assetKey": asset_key,
        "input": {"sourceHash": source_hash, "sourceKind": recipe["source"]["kind"]},
        "recipe": {"resolvedRecipeHash": rhash,
                   "recipeFormatVersion": recipe["recipeFormatVersion"],
                   "layerHashes": [{k: l[k] for k in ("layer", "ref", "hash") if k in l}
                                   for l in layers]},
        "steps": step_records,
        "environment": environment(ctx.settings),
        "outputs": outputs,
        "qa": {"passed": bool(chosen) and chosen[0] == "primary", "results": qa_results,
               "toleranceClassClaimed": "unmeasured", "humanReview": None},
        "fallback": {"used": bool(chosen) and chosen[0] != "primary",
                     "selectedAttemptId": chosen[0] if chosen else "none",
                     "selectedMode": chosen[1] if chosen else "original",
                     "attempts": att_records},
        "run": {"invocationId": str(uuid.uuid4()), "startedOn": started.isoformat(),
                "finishedOn": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                "resumedFrom": None},
    }
    if layers:
        prov["recipe"]["precedenceSource"] = layers[-1]["layer"]
    if recipe.get("targetProfile"):
        prov["recipe"]["targetProfile"] = recipe["targetProfile"]
        prov["recipe"]["targetProfileHash"] = jcs_hash(recipe["targetProfile"])
    ladder = None
    if chosen:
        gen = bool(policy.generative_steps(recipe["steps"]))
        ladder = "L3" if (gen and chosen[0] == "primary") else policy.LADDER.get(chosen[1], "L2")
    return {"provenance": prov, "files": files, "ladder": ladder,
            "executor": executor_identity()}


def lock_entry(result: dict) -> dict:
    p = result["provenance"]
    volatile = {"wallTimeSeconds", "modelLoadSeconds", "peakVramBytes", "cacheHit",
                "deterministicExceptions", "error"}
    body = {k: v for k, v in p.items() if k not in ("run", "status")}
    body["steps"] = [{k: v for k, v in r.items() if k not in volatile} for r in p["steps"]]
    return {"assetId": p["assetId"], "assetKey": p["assetKey"],
            "outputHashes": [o["hash"] for o in p["outputs"]],
            "provenanceHash": jcs_hash(body)}
