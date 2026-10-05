#!/usr/bin/env python3
"""The fp16 FFN intermediate of the prefill (MM 25.183): the K-loop 'half' epilogue is admitted for exactly the three
receipted w1 launches and refused elsewhere; the fp16-input SwiGLU rows kernel is bit-exact on g17emu against its
order; the fast SwiGLU's enclosure measure counts ulps across zero and its wrong-base control is far outside. CPU only."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import numpy as np  # noqa: E402


class HalfEpilogueAdmission(unittest.TestCase):
    def test_the_receipted_launches_build(self):
        import g17qmm as QMM
        import g17tensorcommonruntime as R
        for M in (128, 256, 512):
            with self.subTest(M=M):
                R.build_generic_program(R.generic_spec(dict(QMM.role_spec("w1", M), epilogue=["half"])))

    def test_everything_else_refuses(self):
        import g17qmm as QMM
        import g17tensorcommonruntime as R
        for role, M in (("qkv", 256), ("w2", 256), ("wo", 128)):
            with self.subTest(role=role, M=M), self.assertRaises(ValueError) as caught:
                R.build_generic_program(R.generic_spec(dict(QMM.role_spec(role, M), epilogue=["half"])))
            self.assertIn("narrow_out_half", str(caught.exception))
        # the pair at an unreceipted shape names the combination's rule
        with self.assertRaises(ValueError) as caught:
            R.build_generic_program(R.generic_spec(dict(QMM.role_spec("w1", 256), K=1024, epilogue=["half"])))
        self.assertIn("together only as", str(caught.exception))


class SwigluRows(unittest.TestCase):
    def _run(self, lay, g, u):
        import g17deliver as DL
        import g17emu as EMU
        import g17decodeops as O
        import g17rows as W
        prog = W.build_rows(lay)
        a, b = bytearray(lay["a_bytes"]), bytearray(lay["b_bytes"])
        O._place(a, 0, g); O._place(b, 0, u)
        with tempfile.TemporaryDirectory() as t:
            d = DL.author(Path(t) / "sw", prog, bytes(a), bytes(b), DL.SENT * max(lay["c_bytes"], 4 * len(a)), lay)
            out, _m = EMU.run_bundle(d, 32 * lay["groups"], 32, 1, tier="wp")
        return np.frombuffer(out, "<u2", lay["M"] * lay["N"], 0)

    def test_in16_is_bit_exact(self):
        import g17rows as W
        lay = W.rows_layout("swiglu", 4, 1024, in16=True)
        rng = np.random.default_rng(5)
        g = (rng.standard_normal(4096) * 3).astype(np.float16)
        u = (rng.standard_normal(4096) * 2).astype(np.float16)
        want = W.rows_reference(lay, (g, u))["act"].view(np.uint16)
        self.assertEqual(int((self._run(lay, g, u) != want).sum()), 0)

    def test_fast_check_counts_ulps(self):
        import g17rows as W
        g = np.array([0.5, -2.0, 3.0], np.float32); u = np.array([1.0, 1.5, -0.25], np.float32)
        exact = (g.astype(np.float64) / (1 + np.exp(-g.astype(np.float64))) * u).astype(np.float16)
        self.assertEqual(W.fast_check(exact, g, u), 0)
        one_off = exact.view(np.uint16).copy(); one_off[0] += 1
        self.assertEqual(W.fast_check(one_off.view(np.float16), g, u), 1)
        self.assertGreater(W.fast_check(exact, g, u, base="2"), 10)
        # across zero: +0 and -0 are one apart in the monotone order, not 32768
        z = np.array([0.0], np.float16); nz = np.array([-0.0], np.float16)
        gz = np.array([0.0], np.float32); uz = np.array([1.0], np.float32)
        self.assertLessEqual(W.fast_check(nz, gz, uz), 1)
        self.assertEqual(W.fast_check(z, gz, uz), 0)


if __name__ == "__main__":
    unittest.main()
