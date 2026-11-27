# SPDX-FileCopyrightText: 2026 Handelsonderneming S.D. Broekhoven, trading as ITEagle Europe
# SPDX-License-Identifier: GPL-3.0-only
"""RFC 8785 JSON Canonicalization Scheme (JCS).

Own implementation (stdlib only) so that hashing has no third-party
dependency. Rules (RFC 8785 section 3.2):

* object members sorted by the UTF-16 code units of their names;
* no insignificant whitespace;
* strings: only ``"``, ``\\`` and control characters < U+0020 are escaped
  (short forms ``\\b \\t \\n \\f \\r``, otherwise ``\\u00xx`` lowercase hex);
  everything else is emitted literally and the result is encoded as UTF-8;
* numbers: ECMAScript ``Number.prototype.toString`` serialization of the
  IEEE-754 double value; NaN and Infinity are errors; integers outside
  +/-2**53 are rejected (they would not survive a double round trip).
"""

from __future__ import annotations

import math
from decimal import Decimal
from typing import Any

MAX_SAFE_INT = 2**53


class JCSError(ValueError):
    """Value cannot be canonicalized under RFC 8785."""


def _number(x: float) -> str:
    if math.isnan(x) or math.isinf(x):
        raise JCSError("NaN/Infinity are not allowed in JCS")
    if x == 0:
        return "0"  # also -0
    sign = "-" if x < 0 else ""
    # repr() yields the shortest round-trip digit string (same as ES6).
    tup = Decimal(repr(abs(x))).as_tuple()
    digits = "".join(str(d) for d in tup.digits).rstrip("0") or "0"
    # value = 0.<digits> * 10**n  with ES naming: s * 10**(n-k)
    stripped = len("".join(str(d) for d in tup.digits)) - len(digits)
    k = len(digits)
    n = k + tup.exponent + stripped
    if k <= n <= 21:
        out = digits + "0" * (n - k)
    elif 0 < n <= 21:
        out = digits[:n] + "." + digits[n:]
    elif -6 < n <= 0:
        out = "0." + "0" * (-n) + digits
    else:
        e = n - 1
        es = ("+" if e >= 0 else "-") + str(abs(e))
        out = (digits if k == 1 else digits[0] + "." + digits[1:]) + "e" + es
    return sign + out


_SHORT = {0x08: "\\b", 0x09: "\\t", 0x0A: "\\n", 0x0C: "\\f", 0x0D: "\\r", 0x22: '\\"', 0x5C: "\\\\"}


def _string(s: str) -> str:
    out = ['"']
    for ch in s:
        o = ord(ch)
        if 0xD800 <= o <= 0xDFFF:
            raise JCSError("lone surrogate in string")
        if o in _SHORT:
            out.append(_SHORT[o])
        elif o < 0x20:
            out.append("\\u%04x" % o)
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def _ser(v: Any, parts: list[str]) -> None:
    if v is None:
        parts.append("null")
    elif v is True:
        parts.append("true")
    elif v is False:
        parts.append("false")
    elif isinstance(v, int):
        if abs(v) > MAX_SAFE_INT:
            raise JCSError(f"integer {v} outside +/-2**53")
        parts.append(_number(float(v)))
    elif isinstance(v, float):
        parts.append(_number(float(v)))  # float(): normalises float subclasses
    elif isinstance(v, str):
        parts.append(_string(v))
    elif isinstance(v, (list, tuple)):
        parts.append("[")
        for i, item in enumerate(v):
            if i:
                parts.append(",")
            _ser(item, parts)
        parts.append("]")
    elif isinstance(v, dict):
        for key in v:
            if not isinstance(key, str):
                raise JCSError("object keys must be strings")
        parts.append("{")
        for i, key in enumerate(sorted(v, key=lambda k: k.encode("utf-16-be"))):
            if i:
                parts.append(",")
            parts.append(_string(key))
            parts.append(":")
            _ser(v[key], parts)
        parts.append("}")
    else:
        raise JCSError(f"type {type(v).__name__} is not JSON")


def canonicalize(value: Any) -> bytes:
    """Return the RFC 8785 canonical UTF-8 bytes of a JSON-compatible value."""
    parts: list[str] = []
    _ser(value, parts)
    return "".join(parts).encode("utf-8")
