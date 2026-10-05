#!/usr/bin/env python3
"""int8 through gemm_generic (Set A item 4): the runtime admits it only there, the harness's four
int32 models are told apart by the authored inputs, and the saturating program carries the bit."""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import g17tensorcommonruntime as R  # noqa: E402
from agxforge.g17 import runtime  # noqa: E402


def spec(**kw):
    base = dict(M=32, N=32, K=64, lda=64, ldb=32, ldc=32, a_type="int8", b_type="int8", c_type="int",
                simdgroups=1, grid=(32, 1, 1), threadgroup=(32, 1, 1), composition="gemm_generic")
    base.update(kw)
    return runtime.TensorSpec(**base)


class Admission(unittest.TestCase):
    def test_generic_admits_int8_accumulate_and_saturate(self):
        self.assertTrue(spec(accumulate=True, saturate=True).saturate)
        self.assertIsNone(spec().accumulate)

    def test_splits_are_the_admitted_int8_split_class(self):
        # MM P13 (section 25.127): the simdgroup and grid splits, receipted on hardware
        self.assertEqual(spec(M=64, simdgroups=2, threadgroup=(64, 1, 1), grid=(64, 1, 1)).simdgroups, 2)
        self.assertEqual(spec(M=128, grid=(128, 1, 1)).grid[0], 128)

    def test_refusals(self):
        # each refusal names its own rule, so a malformed base cannot pass them all
        for bad, why in ((dict(c_type="float"), "int32"), (dict(b_type="half"), "paired"),
                         (dict(saturate=True), "verified as an accumulate"),
                         # MM P13: an epilogue is refused by its class name (splits are admitted, below)
                         (dict(epilogue=("relu",)), "class int8_epilogue is not admitted"),
                         (dict(a_type="half", b_type="half", c_type="float", accumulate=True), "int8 only"),
                         (dict(composition="gemm_grid"), "only gemm_generic carries int8")):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, why):
                spec(**bad)


class Models(unittest.TestCase):
    def test_the_seeded_inputs_separate_every_pair_of_models(self):
        rng = np.random.default_rng(1729)
        a = np.frombuffer(R._generic_draw(rng, 32 * 64, "int8"), np.int8)
        b = np.frombuffer(R._generic_draw(rng, 64 * 32, "int8"), np.int8)
        m = R.generic_int_models(a, b, R.generic_int_seed(rng, 32, 32), 32, 32, 64)
        names = sorted(m)
        for i, x in enumerate(names):
            for y in names[i + 1:]:
                with self.subTest(pair=(x, y)):
                    self.assertGreaterEqual(int((m[x] != m[y]).sum()), 10)

    def test_one_issue_clip_is_the_section_136_law(self):
        # K = 16 is one MMA: clip_issue must equal clip(C + exact sum), and the refuted per-product
        # clip must differ somewhere on a cancelling tile that crosses the rail
        a = np.array([[127] * 8 + [-128] * 8], np.int8).repeat(16, 0).ravel()
        b = np.full((16, 16), 127, np.int8).ravel()
        c = np.full((16, 16), R.INT32_MAX - 10, "<i4")
        m = R.generic_int_models(a, b, c, 16, 16, 16)
        exact = c.astype(np.int64) + (127 * 127 * 8 - 128 * 127 * 8)
        self.assertTrue(np.array_equal(m["clip_issue"], np.clip(exact, R.INT32_MIN, R.INT32_MAX)))
        self.assertFalse(np.array_equal(m["clip_issue"], m["clip_product"]))


class Program(unittest.TestCase):
    def test_saturate_program_carries_the_bit_on_every_issue(self):
        from agxforge.g17 import model
        s = R.generic_spec(dict(M=32, N=32, K=64, a="int8", b="int8", accumulate=True, saturate=True))
        mm = [i for i in model.decode(R.build_generic_program(s).code, 0)
              if i.opcode and i.opcode.id in (10384, 10385)]
        self.assertEqual([(i.opcode.id, i.values[2][1]) for i in mm], [(10384, 41)] * 16)


if __name__ == "__main__":
    unittest.main()
