# SPDX-FileCopyrightText: 2026 Handelsonderneming S.D. Broekhoven, trading as ITEagle Europe
# SPDX-License-Identifier: GPL-3.0-only
"""docs/boundaries.md enforcement: no game-specific identifiers in src/."""

import re
from pathlib import Path

DENY = re.compile(r"\.(gro|tex|mdl|wld|ani|bm|bs|ba|smc|fnt)\b|serious\s*(sam|engine)|croteam|"
                  r"\b(tfe|tse)\b|sstre[-_]assetcompiler|sstre[-_]manifests",
                  re.IGNORECASE)


def test_no_game_specific_identifiers_in_source():
    root = Path(__file__).resolve().parents[1] / "src"
    hits = []
    for p in root.rglob("*.py"):
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if DENY.search(line):
                hits.append(f"{p.relative_to(root)}:{n}: {line.strip()}")
    assert not hits, "\n".join(hits)
