# SPDX-FileCopyrightText: 2026 Handelsonderneming S.D. Broekhoven, trading as ITEagle Europe
# SPDX-License-Identifier: GPL-3.0-only
import json
import math

import pytest

from remaster_toolkit.hashing import jcs_hash
from remaster_toolkit.jcs import JCSError, canonicalize

# RFC 8785 Appendix B number serialization samples (IEEE-754 value -> JCS text)
NUMBERS = [
    (0.0, "0"), (-0.0, "0"), (5e-324, "5e-324"), (-5e-324, "-5e-324"),
    (1.7976931348623157e308, "1.7976931348623157e+308"),
    (-1.7976931348623157e308, "-1.7976931348623157e+308"),
    (9007199254740992.0, "9007199254740992"), (-9007199254740992.0, "-9007199254740992"),
    (295147905179352830000.0, "295147905179352830000"), (9.999999999999997e22, "9.999999999999997e+22"),
    (1e23, "1e+23"), (1e21, "1e+21"),
    (999999999999999700000.0, "999999999999999700000"),
    (999999999999999900000.0, "999999999999999900000"),
    (0.000001, "0.000001"), (1e-7, "1e-7"), (0.000009999999999999999, "0.000009999999999999999"),
    (4.5, "4.5"), (0.002, "0.002"), (333333333.3333333, "333333333.3333333"),
    (1.0, "1"), (100.0, "100"), (-1.5e-9, "-1.5e-9"),
]


@pytest.mark.parametrize("value,text", NUMBERS)
def test_numbers(value, text):
    assert canonicalize(value) == text.encode()


def test_rfc_sorting_example():
    names = {0x20AC: "Euro Sign", 0x0D: "Carriage Return", 0xFB33: "Hebrew Letter Dalet",
             0x31: "One", 0x1F600: "Emoji: Grinning Face", 0x80: "Control",
             0xF6: "Latin Small Letter O With Diaeresis"}
    obj = {chr(k): v for k, v in names.items()}
    keys = [ord(k) for k in json.loads(canonicalize(obj)).keys()]
    assert keys == [0x0D, 0x31, 0x80, 0xF6, 0x20AC, 0x1F600, 0xFB33]


def test_rfc_structure_example():
    # RFC 8785 section 3.2.4 sample (string built from code points)
    string = "".join(chr(c) for c in (0x20AC, 0x24, 0x0F, 0x0A, 0x41, 0x27, 0x42, 0x22, 0x5C,
                                      0x5C, 0x22, 0x2F))
    obj = {"numbers": [333333333.33333329, 1e30, 4.50, 2e-3, 0.000000000000000000000000001],
           "string": string, "literals": [None, True, False]}
    bs = chr(0x5C)
    q = chr(0x22)
    expect = ("{" + q + "literals" + q + ":[null,true,false]," + q + "numbers" + q
              + ":[333333333.3333333,1e+30,4.5,0.002,1e-27]," + q + "string" + q + ":" + q
              + chr(0x20AC) + "$" + bs + "u000f" + bs + "nA'B" + bs + q + bs + bs + bs + bs
              + bs + q + "/" + q + "}")
    assert canonicalize(obj) == expect.encode("utf-8")


def test_whitespace_and_order_invariance():
    a = json.loads('{"b": [1, 2, {"z": 1, "a": 2}], "a": "x"}')
    b = json.loads('{\n  "a":"x",\n "b":[1,2,{"a":2,"z":1}]}')
    assert canonicalize(a) == canonicalize(b)
    assert jcs_hash(a) == jcs_hash(b)


@pytest.mark.parametrize("bad", [math.nan, math.inf, 2**53 + 1, "\ud800", {1: 2}, object()])
def test_rejects(bad):
    with pytest.raises(JCSError):
        canonicalize(bad)
