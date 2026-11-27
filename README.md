# Remaster-Toolkit

A game-agnostic toolkit for texture processing, workflow execution, QA,
caching and material generation, developed by ITEagle as part of
**SSTRE — Serious Sam: The Raytraced Encounter** but intended to be reusable
for other remaster projects. SSTRE is an unofficial project, independent of
Croteam and Devolver Digital.

## Status

**Phase 0 — bootstrap and research. No code exists yet.** Model, workflow and
framework choices are under research in the SSTRE umbrella
(`docs/research/RD-003-ai-pipeline.md`, `docs/research/RD-004-reproducibility.md`)
and are not decided.

## Role

Planned responsibilities:

- execute resolved, locked processing recipes on standard intermediate images
  (PNG/TIFF/EXR): conventional super-resolution first, optional controlled
  diffusion where a recipe allows it;
- handle tiling, seams, alpha and masks; keep technical textures out of
  diffusion unless explicitly configured;
- generate PBR maps where supported, labelled as inferred, not measured;
- automated QA with fallback to conservative processing or the original;
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
| Implementation language | **TBD** (Python is a candidate, not decided) |
| Inference stack (e.g. PyTorch, ComfyUI or a headless equivalent) | **TBD** — to be benchmarked |
| Models (e.g. Real-ESRGAN/SwinIR, SDXL with tiled conditioning) | **TBD** — availability, licenses and quality to be verified |
| GPU / VRAM requirements | **TBD** — to be measured; CI will not require a GPU |
| Reference environment for reproducibility | **TBD** — see RD-004 |

## Development

There is nothing to build or test yet. The planned `src/`, `tests/fixtures/`
and `workflows/` directories are intentionally absent until implementation
starts, so the repository does not contain placeholder code.

What works today:

```sh
git clone https://github.com/Handelsonderneming-S-D-Broekhoven/Remaster-Toolkit.git
git clone https://github.com/Handelsonderneming-S-D-Broekhoven/SSTRE.git
# Link and forbidden-file checks for this repository
# (requires Python 3.10+; see SSTRE CONTRIBUTING.md):
python SSTRE/scripts/validate_repo.py --root Remaster-Toolkit --skip-schemas
```

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
