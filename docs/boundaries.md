# Boundaries: Remaster-Toolkit is game-agnostic

Status: **Rule, adopted with the repository separation** (SSTRE ADR-001,
repository separation). The implementation itself does not exist yet.

## The rule

**Remaster-Toolkit contains no Serious Sam-specific code.** It must not import,
name, parse or special-case anything that only exists in Serious Sam or
Serious Engine.

Concretely, the following must not appear in this repository's code, tests or
default configuration:

- Serious Engine file formats or readers/writers for them (`.gro`, `.tex`,
  `.mdl`, `.wld`, ...). These live in
  [SSTRE-AssetCompiler](https://github.com/Handelsonderneming-S-D-Broekhoven/SSTRE-AssetCompiler).
- Game names, edition detection, in-game paths, asset IDs or hard-coded lists of
  game assets.
- Recipes, material profiles, mappings or overrides for a particular game.
  These live in
  [SSTRE-Manifests](https://github.com/Handelsonderneming-S-D-Broekhoven/SSTRE-Manifests).
- Engine runtime code or any dependency on
  [SSTRE-Engine](https://github.com/Handelsonderneming-S-D-Broekhoven/SSTRE-Engine).
- Imports from SSTRE-AssetCompiler, SSTRE-Manifests or the SSTRE umbrella.

Game knowledge reaches the toolkit only as **input data**: images in standard
intermediate formats (PNG/TIFF/EXR), metadata fields defined by the shared
schemas, and resolved recipes.

## What belongs here

- Generic image processing: conventional super-resolution, controlled
  diffusion passes, tiling/seam handling, alpha and mask handling, color-space
  conversion.
- Executing resolved recipes and pinned workflows deterministically where
  feasible, with caching keyed on input bytes, resolved recipe, workflow and
  model identity, runtime and target profile.
- Generic QA metrics and fallback decisions, reported as data.
- Provenance records (`build-provenance.schema.json` in the SSTRE umbrella).
- Generic layered-configuration resolution (proposed), where the caller
  supplies the ordered list of layers.
- Later: curation tooling that works on any game's data.

## How to check a change

Ask: *would this change make equal sense for a remaster of a different game?*
If not, it belongs in SSTRE-AssetCompiler or SSTRE-Manifests.

Test fixtures must be synthetic (generated procedurally or drawn for the
purpose) or explicitly licensed for redistribution. Never use game textures,
even cropped, as fixtures.

## Interface

The toolkit is invoked as a separate process through a documented CLI
following the conventions in SSTRE
[`docs/architecture/cross-repo-contracts.md`](https://github.com/Handelsonderneming-S-D-Broekhoven/SSTRE/blob/main/docs/architecture/cross-repo-contracts.md).
Keeping it a separate process also keeps it independent of the engine's
license (see SSTRE `docs/research/RD-007-licensing.md`).

## Enforcement (planned)

Once code exists, CI should fail on references to Serious Sam-specific
identifiers (a deny-list of format extensions and names in source files) in
addition to review. Not implemented yet, because there is no source to check.
