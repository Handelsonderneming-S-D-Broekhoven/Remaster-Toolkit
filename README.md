# Remaster-Toolkit

A game-agnostic toolkit for texture processing, workflow execution, QA,
caching and material generation, developed by ITEagle as part of
**SSTRE — Serious Sam: The Raytraced Encounter** but intended to be reusable
for other remaster projects. SSTRE is an unofficial project, independent of
Croteam and Devolver Digital.

## Status

**Proof of concept (SSTRE sprint 2, EXP-013).** A minimal Python package
`remaster_toolkit` exists: recipe resolution and RFC 8785 canonical hashing,
a content-addressed step cache with resume and incremental invalidation,
spandrel-loaded super-resolution, opt-in SDXL + tile-ControlNet refinement,
seamless (wrap) tiling, alpha handling, classical normal/height derivation,
automated QA with a fallback ladder, and build-provenance records. Interfaces
are unstable (`0.1.0.dev0`). Model choices are not final; see SSTRE
`docs/experiments/EXP-013-ai-texture-poc.md` for measured results.

## Role

Planned responsibilities:

- execute resolved, locked processing recipes on standard intermediate images
  (PNG/TIFF/EXR): conventional super-resolution first, optional controlled
  diffusion where a recipe allows it;
- handle tiling, seams, alpha and masks; keep technical textures out of
  diffusion unless explicitly configured;
- generate PBR maps where supported, labelled as inferred, not measured;
- automated QA with fallback to conservative processing or the original;
  per-class policy rules can skip a rung with a recorded reason code (for
  example the conservative profile resamples `sky` conventionally,
  `SKY_FLATTEN`, because general SR models flatten its fine grain);
- content-addressed caching and provenance records;
- later, curation tooling.

### Non-goals

- **Anything specific to Serious Sam** — formats, paths, asset IDs, recipes.
  See [docs/boundaries.md](./docs/boundaries.md).
- Runtime use: release builds of a game must not need this toolkit, a live LLM
  or an external AI API at runtime.
- Distributing model weights. Models will be referenced by source and hash and
  downloaded by the user after verification.

## Architecture and dependencies

- Depends on: cross-repo schemas in the
  [SSTRE umbrella](https://github.com/Handelsonderneming-S-D-Broekhoven/SSTRE) `schemas/`
  (resolved recipes, provenance). No dependency on any other SSTRE repository.
- Invoked by: [SSTRE-AssetCompiler](https://github.com/Handelsonderneming-S-D-Broekhoven/SSTRE-AssetCompiler)
  as a separate process via a documented CLI.
- Boundaries, interfaces and pinning: SSTRE
  [`docs/architecture/cross-repo-contracts.md`](https://github.com/Handelsonderneming-S-D-Broekhoven/SSTRE/blob/main/docs/architecture/cross-repo-contracts.md).

## Prerequisites

| Item | Status |
|---|---|
| Language | Python >= 3.10 |
| Core dependencies (CPU, CI) | `numpy`, `Pillow` |
| Optional: SR (`[sr]`) | `torch`, `spandrel` (MIT), `safetensors` |
| Optional: diffusion (`[diffusion]`) | `diffusers`, `transformers`, `accelerate` (Apache-2.0) |
| Optional: perceptual metrics (`[metrics]`) | `lpips` (BSD-2-Clause), `DISTS_pytorch` (MIT); `pyiqa` is deliberately not used (non-commercial licence) |
| Models | Never bundled or fetched implicitly. Users download weights themselves and register them by SHA-256 (`remaster-toolkit models-register`); recipes reference models by hash only. |
| GPU | Optional; CI runs CPU-only synthetic tests |

## Development

```sh
python -m pip install -e ".[test]"        # core + tests (CPU only)
SSTRE_SCHEMAS_DIR=../SSTRE/schemas python -m pytest
python SSTRE/scripts/validate_repo.py --root Remaster-Toolkit
reuse lint
```

CLI (`remaster-toolkit <verb>`; exit codes 0 ok, 1 completed with fallbacks
or failures, 2 usage/environment error, 3 schema violation):

| Verb | Purpose |
|---|---|
| `resolve --layer NAME=PATH ...` | Merge recipe layers (lowest precedence first; `deep-merge/arrays-replace/null-deletes@1`) |
| `hash FILE [--recipe]` | RFC 8785 JCS hash / `resolvedRecipeHash` |
| `inspect IMAGE` | Size, alpha kind, power-of-two, wrap-seam ratio |
| `models-register` | Hash and register a local model file or directory bundle |
| `run --recipe R --source IMG --workspace W --output O` | Execute one resolved recipe with caching, QA, fallback and provenance |
| `compare A B` | Pixel-level comparison of two images |

Package layout: `src/remaster_toolkit/` (`jcs`, `recipe`, `cache`, `executor`,
`ops`, `sr`, `diffusion`, `tiling`, `resample`, `alpha`, `pbr`, `qa`,
`perceptual`, `workflows`, `policy`, `effects`, `models`, `env`, `cli`); tests use only
procedurally generated images and CPU stand-ins for GPU steps.

### Effect and UI textures (SSTRE M0.1, POC-85/86)

- `effects.classify(usages, stats, exception)` assigns an effect class from caller-supplied
  **usage metadata** (consumer and blend semantics: `additive`, `additive-alpha`,
  `alpha-blend`, `premultiplied`, `multiply`, `multiply-inverse`, or `texel-lookup`
  consumers). Pixel statistics only confirm the declaration; filenames and colours are
  never used. Exceptions are declarative entries with a mandatory reason.
- `workflows.build(..., effect_class=...)` gives the `effect` workflow: ringing-free
  (anti-ringing clamped Lanczos) resampling, RGB and alpha filtered independently,
  straight alpha filtered premultiplied, colour under fully transparent texels kept as
  in the original, no learned SR, diffusion or sharpening (refused by `policy` with
  `EFFECT_NO_LEARNED`), effect QA (neutral lift, halos, energy, alpha, hidden colour,
  alpha coverage per mip level) and a fallback straight to the original. `lut` and
  `mixed` classes keep the original.
- `ui_kind="text"` gives the same alpha-aware, ringing-free path for UI textures, with a
  glyph IoU gate; `ui_kind="glyph-atlas"` keeps font atlases original. Without
  `ui_kind` the legacy `ui` workflow is unchanged.
- `effects.mip_chain(..., cutout=t)` builds an alpha-coverage-preserving mip chain for
  packagers.

## Licensing

Copyright © 2026 Handelsonderneming S.D. Broekhoven, trading as ITEagle Europe, and contributors.

License: GPL-3.0-only — licensed under the GNU General Public License, version 3 only ([`LICENSE`](LICENSE); machine-readable in [`REUSE.toml`](REUSE.toml)). See SSTRE [ADR-0006](https://github.com/Handelsonderneming-S-D-Broekhoven/SSTRE/blob/main/docs/adr/0006-licensing-boundaries.md) for the licensing boundaries between repositories. Model and dependency licenses are recorded before any is adopted.


## Related repositories

- [SSTRE](https://github.com/Handelsonderneming-S-D-Broekhoven/SSTRE) — umbrella, documentation, release pins
- [SSTRE-Engine](https://github.com/Handelsonderneming-S-D-Broekhoven/SSTRE-Engine) — engine fork and runtime
- [SSTRE-AssetCompiler](https://github.com/Handelsonderneming-S-D-Broekhoven/SSTRE-AssetCompiler) — Serious Sam-specific extraction, mapping and packaging
- [SSTRE-Manifests](https://github.com/Handelsonderneming-S-D-Broekhoven/SSTRE-Manifests) — recipes, mappings, overrides, locks

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md). Keep changes game-agnostic.
