#!/usr/bin/env python3
"""fp32 `fsub` in the front end: the float add with the subtrahend's negate source modifier (ledger
g17-float-source-modifiers: there is no fsub instruction). Structural, CPU only: on a retained source,
`fsub %a, %b` compiles to the same length as `fadd %a, %b` and differs from it (the modifier word), and a
constant on either side takes the same modifier. Half and bfloat subtraction still refuse, naming fsub.
The values are checked on hardware against Apple's compile of the same source (evidence/g17-front-fsub-dot-sqrt-v1)."""
import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import g17front  # noqa: E402
from agxforge.g17 import cc  # noqa: E402

TAG = "syn-se8dbb86316"
LINE = "%11 = fadd fast float %10, %9"


class FloatSubtract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import _evidence
        snapshot, = _evidence.require("results/g17-source-admission-v3/air-snapshot.json",
                                      invariant="fp32 fsub lowers as the add with a negate modifier")
        air = {r["tag"]: r["air_text"] for r in json.load(open(snapshot))["records"] if "air_text" in r}
        cls.base = air[TAG]
        assert cls.base.count(LINE) == 1

    def _code(self, line):
        return cc.compile_function(g17front.to_ir(self.base.replace(LINE, line), name="k")).code

    def test_register_subtrahend_is_the_add_with_a_modifier(self):
        add, sub = self._code(LINE), self._code("%11 = fsub fast float %10, %9")
        self.assertEqual(len(add), len(sub))
        self.assertNotEqual(add, sub)
        fn = g17front.to_ir(self.base.replace(LINE, "%11 = fsub fast float %10, %9"), name="k")
        kinds = [o.kind for blk in fn.blocks for o in blk.ops]
        self.assertIn("fneg", kinds)
        self.assertNotIn("fsub", kinds)

    def test_a_constant_on_either_side_rides_the_same_modifier(self):
        # AIR float literals reach the builder as constant registers, so `x - 1.0` is x + (-k) with the
        # modifier on k: the same length as the add, and NOT the bytes of `x + -1.0` (whose constant is
        # already negative). 0.0 included: the modifier flips its sign exactly, to -0.0.
        for line, twin in (("%11 = fsub fast float %9, 1.000000e+00", "%11 = fadd fast float %9, -1.000000e+00"),
                           ("%11 = fsub fast float %9, 0.000000e+00", "%11 = fadd fast float %9, -0.000000e+00"),
                           ("%11 = fsub fast float 2.250000e+00, %9", "%11 = fadd fast float 2.250000e+00, %9")):
            with self.subTest(line=line):
                self.assertEqual(len(self._code(line)), len(self._code(twin)))
                self.assertNotEqual(self._code(line), self._code(twin))

    def test_narrow_subtraction_still_refuses(self):
        for ty in ("half", "bfloat"):
            with self.subTest(ty=ty), self.assertRaises(g17front.Unsupported) as caught:
                g17front.to_ir(self.base.replace(LINE, "%%11 = fsub fast %s %%10, %%9" % ty), name="k")
            self.assertIn("fsub", str(caught.exception))

class DotAndSqrt(unittest.TestCase):
    """air.dot.vNf32 and air.fast_sqrt.f32 in Apple's instruction order (MM 25.180): fmul of lane 0 then an fma per
    lane, left to right; x * rsqrt2(x). The capability-off arm refuses both again."""

    SRC = """#include <metal_stdlib>
using namespace metal;
kernel void k(device float4 *b0 [[buffer(0)]], device float *b1 [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    b1[256 + i] = distance(b0[i], b0[i + 64]);
}
"""

    @classmethod
    def setUpClass(cls):
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".metal", delete=False) as fh:
            fh.write(cls.SRC)
        try:
            cls.air = g17front.air_of(fh.name)
        except Exception as e:  # noqa: BLE001 - no Metal toolchain
            raise unittest.SkipTest("xcrun metal: %s" % e)
        finally:
            os.unlink(fh.name)

    def test_the_chain_is_apples(self):
        fn = g17front.to_ir(self.air, name="k")
        ops = [o for blk in fn.blocks for o in blk.ops]
        kinds = [o.kind for o in ops]
        i = kinds.index("fmul")
        self.assertEqual(kinds[i:i + 4], ["fmul", "fma", "fma", "fma"])
        for k in range(1, 4):                       # each fma accumulates into the previous link
            self.assertIs(ops[i + k].args[2], ops[i + k - 1].dest)
        r = kinds.index("rsqrt2")
        self.assertEqual(kinds[r + 1], "fmul")
        self.assertIs(ops[r + 1].args[1], ops[r].dest)
        self.assertIs(ops[r + 1].args[0], ops[r].args[0])
        cc.compile_function(fn)

    def test_the_off_arm_refuses(self):
        saved = g17front._NO_DOT_SQRT
        g17front._NO_DOT_SQRT = True
        try:
            with self.assertRaises(g17front.Unsupported) as caught:
                g17front.to_ir(self.air, name="k")
            self.assertIn("air.dot", str(caught.exception))
        finally:
            g17front._NO_DOT_SQRT = saved


if __name__ == "__main__":
    unittest.main()
