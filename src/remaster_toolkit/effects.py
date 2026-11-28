# SPDX-FileCopyrightText: 2026 Handelsonderneming S.D. Broekhoven, trading as ITEagle Europe
# SPDX-License-Identifier: GPL-3.0-only
"""Effect-texture classification, alpha-aware effect resampling, mips and QA.

Game-agnostic. The caller supplies **usage metadata**: how the engine draws the
texture (``consumer``: particle, model, brush, billboard, beam, flare, ui,
``texel-lookup`` when code reads texels as data) and with which **blend
semantics** (:data:`BLENDS`). Filenames and colours are never used to decide a
class; pixel statistics only confirm or contradict the declared semantics.
Unavoidable exceptions are declarative entries with a mandatory reason.

Effect classes and their recipes (all non-generative, no learned SR, no
sharpening; QA failure falls back to the original):

======================  ===================================================
``additive``            RGB is light added to the frame; black = invisible.
                        Resample RGB with a ringing-free kernel; alpha (if
                        any) independently.
``additive-alpha``      RGB*A added. Straight alpha: dilate RGB under A=0,
                        resample premultiplied, alpha independently.
``alpha-blend``         straight alpha coverage; same path as additive-alpha.
``premultiplied``       RGB already multiplied by A; resample as is, keep
                        RGB <= A.
``multiply``            frame * RGB (DST_COLOR, ZERO); white = neutral.
``multiply-inverse``    frame * (1 - RGB) (ZERO, INV_SRC_COLOR, a darkening
                        "shade"); black = neutral.
``lut``                 code reads texels as data (gradients, palettes): kept
                        original (any resampling changes the lookup).
``mixed``               conflicting effect blends: kept original.
======================  ===================================================

A texture used both by an effect and by an opaque/alpha-tested surface (one
atlas) gets the effect recipe for the whole atlas (``surface: True``): the
effect constraints are the stricter ones.
"""

from __future__ import annotations

import numpy as np

from . import alpha as alpha_mod
from .resample import downscale_area, footprint_minmax, resize

BLENDS = ("opaque", "alpha-test", "alpha-blend", "premultiplied", "additive", "additive-alpha",
          "multiply", "multiply-inverse", "unknown")
EFFECT_BLENDS = frozenset({"alpha-blend", "premultiplied", "additive", "additive-alpha",
                           "multiply", "multiply-inverse"})
SURFACE_BLENDS = frozenset({"opaque", "alpha-test"})
CLASSES = ("additive", "additive-alpha", "alpha-blend", "premultiplied", "multiply",
           "multiply-inverse", "lut", "mixed")
KEEP_ORIGINAL = frozenset({"lut", "mixed"})

# Uncalibrated placeholders (RD-008 s6.1), conservative by construction: the recipe
# cannot overshoot (anti-ringing), so these mainly catch implementation errors.
# hidden-rgb-delta and mip-coverage-delta are the Preview3 P3-6 limits (4/255, 2 pp).
QA_LIMITS = {"neutral-lift": 0.002, "halo-fraction": 0.002, "energy-drift": 0.03,
             "alpha-mae": 0.03, "hidden-rgb-delta": 4.0 / 255, "mip-coverage-delta": 0.02}
REASONS = {"neutral-lift": "NEUTRAL_LIFT", "halo-fraction": "HALO",
           "energy-drift": "ENERGY_DRIFT", "alpha-mae": "ALPHA_DRIFT",
           "hidden-rgb-delta": "HIDDEN_RGB_DRIFT", "mip-coverage-delta": "MIP_COVERAGE"}
EPS = 1.0 / 255


# ------------------------------------------------------------------ statistics

def alpha_stats(img: np.ndarray) -> dict:
    """Pixel facts used to confirm declared semantics (never to choose them)."""
    rgb = img[:, :, :3].astype(np.float64)
    a = img[:, :, 3].astype(np.float64) if img.shape[2] == 4 else np.ones(img.shape[:2])
    kind = alpha_mod.classify(a)
    zero = a <= EPS
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    return {
        "alphaKind": kind,
        "alphaMin": float(a.min()),
        "premultipliedConsistent": float((rgb.max(axis=2) <= a + 2 * EPS).mean()),
        "rgbUnderZeroAlphaMax": float(rgb[zero].max()) if zero.any() else None,
        "rgbUnderZeroAlphaBlackFraction": (float((rgb[zero].max(axis=1) <= EPS).mean())
                                           if zero.any() else None),
        "blackFraction": float((rgb.max(axis=2) <= EPS).mean()),
        "borderMax": float(border.max()),
        "coverage@0.5": float((a >= 0.5).mean()),
    }


# ------------------------------------------------------------------ classification

def classify(usages: list[dict] | None, stats: dict | None = None,
             exception: dict | None = None) -> dict:
    """Return ``{effectClass, surface, reason, blends, warnings}``.

    ``effectClass`` is None when the texture is not an effect (or nothing is
    known); the caller then keeps its own (surface) class. ``exception`` is a
    declarative override ``{"effectClass": ..., "reason": "..."}``.
    """
    stats = stats or {}
    if exception is not None:
        if not exception.get("reason"):
            raise ValueError("an effect-class exception needs a documented reason")
        cls = exception.get("effectClass")
        if cls is not None and cls not in CLASSES:
            raise ValueError(f"unknown effect class {cls}")
        return {"effectClass": cls, "surface": bool(exception.get("surface", False)),
                "reason": "DECLARED_EXCEPTION", "blends": [], "warnings": [exception["reason"]]}
    vis = [u for u in (usages or []) if u.get("visible", True)]
    for u in vis:
        if u.get("blend", "unknown") not in BLENDS:
            raise ValueError(f"unknown blend {u.get('blend')}")
    blends = sorted({u.get("blend", "unknown") for u in vis})
    known = [b for b in blends if b != "unknown"]
    out = {"effectClass": None, "surface": bool(set(known) & SURFACE_BLENDS), "reason": "",
           "blends": blends, "warnings": []}
    if any(u.get("consumer") == "texel-lookup" for u in vis):
        out.update(effectClass="lut", reason="TEXEL_LOOKUP")
        return out
    if not known:
        out["reason"] = "NO_USAGE_METADATA"
        return out
    eff = set(known) & EFFECT_BLENDS
    if "unknown" in blends and eff:
        out["warnings"].append("SOME_USAGES_UNKNOWN")
    no_alpha = stats.get("alphaMin", 1.0) >= 1 - EPS
    if no_alpha and "additive-alpha" in eff:
        eff = (eff - {"additive-alpha"}) | {"additive"}  # A == 1: identical maths
        out["warnings"].append("ADDALPHA_WITHOUT_ALPHA")
    if not eff:
        out["reason"] = "SURFACE_ONLY"
        return out
    if len(eff) > 1:
        out.update(effectClass="mixed", reason="CONFLICTING_EFFECT_BLENDS")
        return out
    cls = next(iter(eff))
    if cls == "premultiplied" and stats.get("premultipliedConsistent", 1.0) < 0.99:
        out.update(effectClass="mixed", reason="PREMULTIPLIED_INCONSISTENT")
        return out
    out.update(effectClass=cls, reason="DECLARED_BLEND")
    return out


def material_profile(cls: str, base: str = "generic") -> str:
    """``<base>/effect/<cls>``; the ``effect/<cls>`` segment is what policy and QA read."""
    return f"{base}/effect/{cls}"


def class_of(recipe: dict) -> str | None:
    from .policy import profile_segment
    return profile_segment(recipe, "effect")


# ------------------------------------------------------------------ resampling

def _dilate(img: np.ndarray, wrap: bool) -> np.ndarray:
    return alpha_mod.edge_dilate(img, threshold=EPS, wrap=wrap)


def resample(img: np.ndarray, width: int, height: int, semantics: str, filt: str = "lanczos3",
             mode="clamp", anti_ringing: bool = True, hidden_rgb: str = "original") -> np.ndarray:
    """Alpha-aware effect resampling. RGB and alpha are filtered independently;
    straight alpha is filtered premultiplied so hidden colour never bleeds into
    visible texels. ``hidden_rgb`` sets the colour of fully transparent output
    texels: ``original`` (default) resamples the source colour there, so the
    hidden colour the engine's own filtering sees stays as in the original;
    ``dilate`` fills it from visible neighbours (softer edges, but a deliberate
    change of the original's edge colour)."""
    wrap = mode == "wrap"
    has_a = img.shape[2] == 4

    def rs(x):
        return resize(x, width, height, filt, mode, anti_ringing=anti_ringing)

    if not has_a:
        return rs(img[:, :, :3])
    rgb, a = img[:, :, :3], img[:, :, 3:4]
    if semantics in ("alpha-blend", "additive-alpha"):
        pm = rs(rgb * a)
        a2 = rs(a)
        safe = np.maximum(a2, 1e-6)
        hidden = a2 <= EPS
        rgb2 = np.where(hidden, 0.0, np.clip(pm / safe, 0.0, 1.0))
        if hidden_rgb == "dilate":
            return _dilate(np.concatenate([rgb2, a2], axis=2).astype(np.float32), wrap)
        if hidden_rgb != "original":
            raise ValueError(f"hidden_rgb {hidden_rgb}")
        # nearest: fully transparent texels keep exactly the original hidden colour
        rgb2 = np.where(hidden, resize(rgb, width, height, "nearest", mode), rgb2)
        return np.concatenate([rgb2, a2], axis=2).astype(np.float32)
    if semantics == "premultiplied":
        a2 = rs(a)
        rgb2 = np.minimum(rs(rgb), a2)
        return np.concatenate([rgb2, a2], axis=2).astype(np.float32)
    # additive / multiply(-inverse): colour is the signal; alpha (if any) separately
    return np.concatenate([rs(rgb), rs(a)], axis=2).astype(np.float32)


def visible(img: np.ndarray, semantics: str) -> np.ndarray:
    """Per-texel contribution of the texture to the frame relative to its
    neutral value (0 = no visible effect)."""
    rgb = img[:, :, :3].astype(np.float64)
    a = img[:, :, 3:4].astype(np.float64) if img.shape[2] == 4 else np.ones(rgb.shape[:2] + (1,))
    if semantics in ("additive", "multiply-inverse"):
        return rgb
    if semantics in ("additive-alpha", "premultiplied"):
        return rgb * a if semantics == "additive-alpha" else np.concatenate([rgb, a], axis=2)
    if semantics == "alpha-blend":
        return a
    if semantics == "multiply":
        return 1.0 - rgb
    raise ValueError(f"semantics {semantics}")


def signal(img: np.ndarray, semantics: str) -> np.ndarray:
    """Channels whose values the renderer actually uses: straight-alpha colour
    counts only premultiplied (colour under A = 0 is invisible)."""
    if img.shape[2] == 4 and semantics in ("alpha-blend", "additive-alpha"):
        return np.concatenate([img[:, :, :3] * img[:, :, 3:4], img[:, :, 3:4]], axis=2)
    return img


# ------------------------------------------------------------------ mips

def mip_chain(img: np.ndarray, semantics: str, cutout: float | None = None,
              levels: int | None = None) -> list[np.ndarray]:
    """Box mip chain that keeps effect semantics.

    Straight alpha is averaged premultiplied (no dark fringes). With
    ``cutout`` (alpha-tested use) each level's alpha is rescaled so that the
    fraction of texels passing the test equals level 0 (alpha-coverage
    preservation), so cutouts do not thin out or vanish in the distance.
    """
    out = [img.astype(np.float32)]
    target = alpha_mod.coverage(img[:, :, 3], cutout) if (cutout is not None
                                                          and img.shape[2] == 4) else None
    while out[-1].shape[0] > 1 or out[-1].shape[1] > 1:
        if levels is not None and len(out) >= levels:
            break
        p = out[-1]
        h, w = max(1, p.shape[0] // 2), max(1, p.shape[1] // 2)
        if p.shape[2] == 4 and semantics in ("alpha-blend", "additive-alpha"):
            pm = downscale_area(np.concatenate([p[:, :, :3] * p[:, :, 3:4], p[:, :, 3:4]], 2), w, h)
            a = pm[:, :, 3:4]
            rgb = np.where(a > EPS, np.clip(pm[:, :, :3] / np.maximum(a, 1e-6), 0, 1), 0.0)
            n = np.concatenate([rgb, a], axis=2).astype(np.float32)
        else:
            n = downscale_area(p, w, h)
        if target is not None:
            n[:, :, 3] = _match_coverage(n[:, :, 3], cutout, target)
        out.append(n)
    return out


def _match_coverage(a: np.ndarray, t: float, target: float) -> np.ndarray:
    lo, hi = 0.0, 8.0
    for _ in range(24):
        mid = (lo + hi) / 2
        if float((np.clip(a * mid, 0, 1) >= t).mean()) < target:
            lo = mid
        else:
            hi = mid
    s = hi if abs(float((np.clip(a * hi, 0, 1) >= t).mean()) - target) <= \
        abs(float((np.clip(a * lo, 0, 1) >= t).mean()) - target) else lo
    return np.clip(a * s, 0, 1).astype(np.float32)


def mip_coverage_deviation(chain: list[np.ndarray], cutout: float) -> float:
    if chain[0].shape[2] < 4:
        return 0.0
    c0 = alpha_mod.coverage(chain[0][:, :, 3], cutout)
    return float(max(abs(alpha_mod.coverage(m[:, :, 3], cutout) - c0) for m in chain))


# ------------------------------------------------------------------ QA

def qa_metrics(out: np.ndarray, src: np.ndarray, semantics: str, mode="clamp",
               cutout: float | None = None, coverage: bool = True) -> dict:
    """Effect QA signals (out at output size, src at source size).

    ``neutral-lift``: fraction of output texels inside the source's neutral
    (invisible) interior that became visible (> 2/255): a lifted black
    background or alpha makes the quad outline visible.
    ``halo-fraction``: fraction of output texels outside the min/max of their
    source footprint by > 2/255 (ringing halos, dark or bright outlines, SR
    sharpening artefacts).
    ``energy-drift``: relative change of the mean visible contribution after an
    exact area downscale back to the source size.
    """
    sh, sw = src.shape[:2]
    oh, ow = out.shape[:2]
    if out.shape[2] == 4 and src.shape[2] == 3:
        src = np.concatenate([src, np.ones((sh, sw, 1), np.float32)], axis=2)
    if src.shape[2] == 4 and out.shape[2] == 3:
        out = np.concatenate([out, np.ones((oh, ow, 1), np.float32)], axis=2)
    vs, vo = visible(src, semantics), visible(out, semantics)
    neutral = vs.max(axis=2) <= EPS
    # interior of the neutral region (3x3 erosion): edges of a glow may legitimately move
    p = np.pad(neutral, 1, mode="edge")
    inner = np.ones_like(neutral)
    for dy in range(3):
        for dx in range(3):
            inner &= p[dy:dy + sh, dx:dx + sw]
    m = resize(inner[:, :, None].astype(np.float32), ow, oh, "nearest", mode)[:, :, 0] > 0.5
    lift = (vo.max(axis=2) > 2 * EPS) & m
    ss, so = signal(src, semantics), signal(out, semantics)
    lo, hi = footprint_minmax(ss, ow, oh, mode)
    tol = 2 * EPS
    oob = ((so < lo - tol) | (so > hi + tol)).any(axis=2)
    vd = visible(downscale_area(out, sw, sh, mode), semantics)
    es, ed = float(vs.mean()), float(vd.mean())
    res = {
        "neutral-lift": float(lift.sum() / max(m.sum(), 1)),
        "neutral-interior-fraction": float(inner.mean()),
        "halo-fraction": float(oob.mean()),
        "energy-drift": abs(ed - es) / max(es, 1e-3),
    }
    if src.shape[2] == 4:
        res["alpha-mae"] = float(np.abs(downscale_area(out[:, :, 3:4], sw, sh, mode)
                                        - src[:, :, 3:4]).mean())
        t = 0.5 if cutout is None else cutout
        hc = hidden_rgb_and_coverage(out, src, semantics, t, mode)
        if not coverage:
            hc.pop("mip-coverage-delta")
        res.update(hc)
    return res


def hidden_rgb_and_coverage(out: np.ndarray, src: np.ndarray, semantics: str, threshold: float,
                            mode="clamp", min_texels: int = 16) -> dict:
    """``hidden-rgb-delta``: mean |RGB| difference (0..1) between output and source
    in texels that are fully transparent in both (source nearest-upsampled).
    ``mip-coverage-delta``: max over mip levels (box chain, levels with at least
    ``min_texels`` texels) of |coverage(out level) - coverage(src level)| at
    ``threshold``, comparing levels of equal size."""
    sh, sw = src.shape[:2]
    oh, ow = out.shape[:2]
    up = resize(src, ow, oh, "nearest", mode)
    hid = (up[:, :, 3] <= EPS) & (out[:, :, 3] <= EPS)
    d = float(np.abs(out[:, :, :3][hid] - up[:, :, :3][hid]).mean()) if hid.any() else 0.0
    cs = {m.shape[:2]: alpha_mod.coverage(m[:, :, 3], threshold) for m in mip_chain(src, semantics)}
    dev = 0.0
    for m in mip_chain(out, semantics):
        if m.shape[:2] in cs and m.shape[0] * m.shape[1] >= min_texels:
            dev = max(dev, abs(alpha_mod.coverage(m[:, :, 3], threshold) - cs[m.shape[:2]]))
    return {"hidden-rgb-delta": d, "mip-coverage-delta": dev}
