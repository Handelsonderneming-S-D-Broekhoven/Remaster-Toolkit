# Contributing to Remaster-Toolkit

This repository follows the shared SSTRE contribution guide:
**[SSTRE CONTRIBUTING.md](https://github.com/Handelsonderneming-S-D-Broekhoven/SSTRE/blob/main/CONTRIBUTING.md)**
(commit conventions, sign-off, branch naming, ADR process, the no-game-data rule
and the AI-generated-asset rule). Security reports: see the umbrella
[SECURITY.md](https://github.com/Handelsonderneming-S-D-Broekhoven/SSTRE/blob/main/SECURITY.md).

Repository-specific notes:

- This repository is **game-agnostic**. No Serious Sam-specific code, names, formats or paths; see [docs/boundaries.md](./docs/boundaries.md). A reviewer will ask: would this change make sense for a different game?
- Test fixtures must be synthetic or explicitly licensed for redistribution.
- Never commit game files, extracted assets, AI derivatives of game assets, model
  weights or secrets. The [`.gitignore`](./.gitignore) blocks the common cases; it is a
  safety net, not permission.

Licensing: pending owner confirmation — proposed GPL-3.0 (see SSTRE
`docs/research/RD-007-licensing.md`).
