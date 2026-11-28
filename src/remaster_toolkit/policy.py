# SPDX-FileCopyrightText: 2026 Handelsonderneming S.D. Broekhoven, trading as ITEagle Europe
# SPDX-License-Identifier: GPL-3.0-only
"""Per-class processing policy (RD-003 section 5.4 / 6, RD-008 section 6.1).

Class names are the generic ``textureClass`` enum of the recipe schema.
"""

from __future__ import annotations

# Classes whose recipes must never contain a generative step.
DIFFUSION_FORBIDDEN = frozenset({
    "ui-text", "normal-map", "height-map", "mask-technical", "lightmap",
    "cubemap-face", "animated-frame",
})
# Classes where diffusion is allowed only by curated override.
DIFFUSION_CURATED_ONLY = frozenset({"character-skin", "weapon", "vegetation-cutout",
                                    "translucent-effect", "unknown"})
GENERATIVE_OPS = frozenset({"controlled-diffusion", "custom-workflow"})
LEARNED_SR_OPS = frozenset({"sr"})

# Conservative profile: classes whose primary (learned SR) attempt is not run, so the
# fallback ladder goes straight to its next rung (conventional-resample, L1). The reason
# code is recorded in the provenance attempts. ``sky``: sky/background panoramas are
# magnified strongly in-engine and general SR models merge their fine rock/cloud grain
# into flat patches with hard edges (a posterised look), which conflicts with the
# Classic art direction (ADR-0009).
CONSERVATIVE_PROFILE_PREFIX = "conservative/"
CONSERVATIVE_NO_SR = {"sky": "SKY_FLATTEN"}

# Fallback ladder levels (RD-008 section 6.2)
LADDER = {"primary-generative": "L3", "primary": "L2", "skip-optional-steps": "L2",
          "alternate-steps": "L2", "conventional-resample": "L1", "original": "L0"}


def generative_steps(steps: list[dict]) -> list[str]:
    return [s["stepId"] for s in steps if s.get("op") in GENERATIVE_OPS]


def check(recipe: dict, steps: list[dict]) -> str | None:
    """Return a reason code if ``steps`` violate the class policy, else None."""
    cls = recipe.get("textureClass", "unknown")
    if (str(recipe.get("materialProfile", "")).startswith(CONSERVATIVE_PROFILE_PREFIX)
            and cls in CONSERVATIVE_NO_SR
            and any(s.get("op") in LEARNED_SR_OPS for s in steps)):
        return CONSERVATIVE_NO_SR[cls]
    gen = generative_steps(steps)
    if not gen:
        return None
    alpha = (recipe.get("source", {}).get("alpha") or {}).get("kind")
    if cls in DIFFUSION_FORBIDDEN or alpha == "data":
        return "DIFFUSION_FORBIDDEN"
    if cls in DIFFUSION_CURATED_ONLY:
        layers = [l.get("layer") for l in recipe.get("precedence", {}).get("layers", [])]
        if "curated-override" not in layers:
            return "DIFFUSION_FORBIDDEN"
    return None
