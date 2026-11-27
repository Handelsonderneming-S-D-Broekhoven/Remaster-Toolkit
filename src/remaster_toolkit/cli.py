# SPDX-FileCopyrightText: 2026 Handelsonderneming S.D. Broekhoven, trading as ITEagle Europe
# SPDX-License-Identifier: GPL-3.0-only
"""Command line: ``remaster-toolkit <verb>``.

Verbs: ``version``, ``hash``, ``resolve``, ``inspect``, ``models-register``,
``run``, ``compare``. Exit codes (SSTRE cross-repo contracts): 0 success,
1 completed with recorded failures/fallbacks, 2 usage/environment error,
3 contract violation (schema validation failed). Logs go to stderr; machine
output to files or stdout (``--json``), canonical JSON (RFC 8785).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .jcs import canonicalize


def _out(doc, path: str | None) -> None:
    data = canonicalize(doc)
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(data + b"\n")
    else:
        sys.stdout.write(data.decode("utf-8") + "\n")


def cmd_version(a) -> int:
    from .env import executor_identity
    _out({**executor_identity(), "schemas": {"reads": ["asset-recipe@0.1.0"],
                                              "writes": ["build-provenance@0.1.0"]}}, None)
    return 0


def cmd_hash(a) -> int:
    from .hashing import jcs_hash
    from .recipe import resolved_hash
    doc = json.loads(Path(a.input).read_text(encoding="utf-8"))
    _out({"jcsHash": jcs_hash(doc),
          **({"resolvedRecipeHash": resolved_hash(doc)} if a.recipe else {})}, a.output)
    return 0


def cmd_resolve(a) -> int:
    from .recipe import PROVENANCE_SCHEMA, RECIPE_SCHEMA, RecipeError, resolve, validate  # noqa
    layers = []
    for spec in a.layer:
        name, _, path = spec.partition("=")
        layers.append((name, Path(path).as_posix() if a.keep_paths else Path(path).name,
                       json.loads(Path(path).read_text(encoding="utf-8"))))
    rec = resolve(layers)
    if a.schemas or a.validate:
        try:
            errs = validate(rec, RECIPE_SCHEMA, a.schemas)
        except RecipeError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        if errs:
            print("\n".join(errs), file=sys.stderr)
            return 3
    _out(rec, a.output)
    return 0


def cmd_inspect(a) -> int:
    import numpy as np
    from . import alpha, imagebuf, qa
    from .hashing import sha256_bytes
    data = Path(a.input).read_bytes()
    img = imagebuf.png_decode(data)
    h, w, c = img.shape
    kind = alpha.classify(img[:, :, 3]) if c == 4 else "none"
    seam = qa.seam_ratio(img, True, True)
    doc = {"hash": sha256_bytes(data), "width": w, "height": h, "channels": c,
           "alphaKind": kind, "pot": qa.is_pot(w) and qa.is_pot(h),
           "seamRatioSource": seam, "tileableGuess": bool(seam is not None and seam <= 1.5),
           "uniqueColors": int(len(np.unique(imagebuf.quantize(img).reshape(-1, c), axis=0)))}
    _out(doc, a.output)
    return 0


def cmd_models_register(a) -> int:
    from .models import ModelStore
    st = ModelStore(a.registry)
    h = st.register(a.path, a.name, a.format, a.license, a.source_uri, a.source_revision,
                    a.license_note, a.expect)
    st.save()
    _out({"hash": h, "name": a.name}, None)
    return 0


def cmd_run(a) -> int:
    from .env import apply_torch_determinism, set_reference_env
    from .executor import Context, build_asset, lock_entry
    from .models import ModelStore
    from .recipe import PROVENANCE_SCHEMA, RECIPE_SCHEMA, RecipeError, load, validate

    set_reference_env()
    settings = {"rngDevice": "cpu"}
    try:
        settings.update(apply_torch_determinism(not a.nondeterministic))
    except ImportError:
        settings["torch"] = "absent"
    rec = load(a.recipe)
    if a.schemas:
        try:
            errs = validate(rec, RECIPE_SCHEMA, a.schemas)
        except RecipeError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        if errs:
            print("\n".join(errs), file=sys.stderr)
            return 3
    ctx = Context(a.workspace, ModelStore(a.models) if a.models else None,
                  device=a.device, settings=settings, perceptual_metrics=a.perceptual)
    res = build_asset(rec, Path(a.source).read_bytes(), ctx)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    stem = a.name or rec["assetId"].replace("/", "_").replace(":", "_")
    for role, data in res["files"].items():
        (out / f"{stem}.{role}.png").write_bytes(data)
    _out(res["provenance"], str(out / f"{stem}.provenance.json"))
    _out(lock_entry(res), str(out / f"{stem}.lock.json"))
    if a.schemas:
        errs = validate(res["provenance"], PROVENANCE_SCHEMA, a.schemas)
        if errs:
            print("provenance schema errors:\n" + "\n".join(errs), file=sys.stderr)
            return 3
    st = res["provenance"]["status"]
    print(f"{rec['assetId']}: {st} ladder={res['ladder']} executed={len(ctx.executed)}",
          file=sys.stderr)
    return 0 if st in ("succeeded", "skipped-cached") else 1


def cmd_compare(a) -> int:
    from . import imagebuf, qa
    x = imagebuf.png_decode(Path(a.a).read_bytes())
    y = imagebuf.png_decode(Path(a.b).read_bytes())
    if x.shape != y.shape:
        print("error: shapes differ", file=sys.stderr)
        return 2
    d = (imagebuf.quantize(x).astype(int) - imagebuf.quantize(y).astype(int))
    doc = {"identicalPixels": bool((d == 0).all()), "maxAbsDiff8": int(abs(d).max()),
           "differingPixelFraction": float((abs(d).max(axis=2) > 0).mean()),
           "psnr": qa.psnr(x, y), "ssim": qa.ssim(qa.luma_srgb(x), qa.luma_srgb(y))}
    _out(doc, a.output)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="remaster-toolkit")
    p.add_argument("--version", action="version", version=f"remaster-toolkit {__version__}")
    sub = p.add_subparsers(dest="verb", required=True)
    sub.add_parser("version").set_defaults(fn=cmd_version)
    s = sub.add_parser("hash", help="JCS hash of a JSON document")
    s.add_argument("input")
    s.add_argument("--recipe", action="store_true", help="also print resolvedRecipeHash")
    s.add_argument("-o", "--output")
    s.set_defaults(fn=cmd_hash)
    s = sub.add_parser("resolve", help="resolve recipe layers (lowest first)")
    s.add_argument("--layer", action="append", required=True, metavar="NAME=PATH")
    s.add_argument("--schemas")
    s.add_argument("--validate", action="store_true")
    s.add_argument("--keep-paths", action="store_true")
    s.add_argument("-o", "--output")
    s.set_defaults(fn=cmd_resolve)
    s = sub.add_parser("inspect", help="image metadata (alpha kind, PoT, seam ratio)")
    s.add_argument("input")
    s.add_argument("-o", "--output")
    s.set_defaults(fn=cmd_inspect)
    s = sub.add_parser("models-register", help="hash and register a local model")
    s.add_argument("--registry", required=True)
    s.add_argument("--path", required=True)
    s.add_argument("--name", required=True)
    s.add_argument("--format", required=True,
                   choices=["safetensors", "pth-weights-only", "onnx", "gguf", "other"])
    s.add_argument("--license", default="NOASSERTION")
    s.add_argument("--license-note")
    s.add_argument("--source-uri")
    s.add_argument("--source-revision")
    s.add_argument("--expect", help="expected sha256:<hex>; mismatch is an error")
    s.set_defaults(fn=cmd_models_register)
    s = sub.add_parser("run", help="execute a resolved recipe on one source image")
    s.add_argument("--recipe", required=True)
    s.add_argument("--source", required=True)
    s.add_argument("--workspace", required=True)
    s.add_argument("--output", required=True)
    s.add_argument("--models")
    s.add_argument("--schemas")
    s.add_argument("--device")
    s.add_argument("--name")
    s.add_argument("--perceptual", action="store_true")
    s.add_argument("--nondeterministic", action="store_true")
    s.set_defaults(fn=cmd_run)
    s = sub.add_parser("compare", help="pixel comparison of two images")
    s.add_argument("a")
    s.add_argument("b")
    s.add_argument("-o", "--output")
    s.set_defaults(fn=cmd_compare)
    a = p.parse_args(argv)
    try:
        return a.fn(a)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
