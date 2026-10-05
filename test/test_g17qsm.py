#!/usr/bin/env python3
"""The batched-decode q4 projection on the tensor units (tools/g17qsm.py, MM 25.166): builds, bit-exact on g17emu against
its stated order at small shapes (plain, split-K, and the half-A form), with the control that a dequant without the fp16
rounding differs; refusals by name. CPU only."""
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import numpy as np  # noqa: E402


class BatchedProjection(unittest.TestCase):
    def _run(self, N, K, ks, sk, ahalf, xrows=False, mb=16):
        import g17qsm as Q
        lay = Q.layout(N, K, 1 if mb == 32 else 2, ks, sk, ahalf, xrows, mb=mb)
        prog = Q.build(lay)
        x, packed, s16, b16, q = Q.case(lay, batch=8)
        want = Q.reference(lay, x, q, s16, b16)
        with tempfile.TemporaryDirectory() as t:
            got = Q.emulate(lay, prog, x, packed, s16, b16, t)
        return lay, got, want, (x, q, s16, b16)

    def test_bit_exact_against_the_stated_order(self):
        for N, K, ks, sk, ahalf, xrows in ((64, 256, 4, 1, False, False), (128, 1024, 4, 4, False, False),
                                           (64, 256, 2, 1, True, False), (64, 256, 4, 1, False, True),
                                           (128, 1024, 4, 4, False, True)):
            with self.subTest(N=N, K=K, ks=ks, sk=sk, ahalf=ahalf, xrows=xrows):
                lay, got, want, _ = self._run(N, K, ks, sk, ahalf, xrows)
                self.assertEqual(int((got.view("<u4") != want.view("<u4")).sum()), 0)
                pad = got[8:] if sk == 1 else got[:, 8:]
                self.assertEqual(int((pad != 0).sum()), 0)        # rows past the batch stay zero

    def test_the_32_row_form(self):
        """mb 32 (MM 25.178): one n-tile a threadgroup, two batch-half bodies over one W (the first keeps its A
        registers for the second, tlower's a_keep); bit-exact with 20 of 32 rows live, unsplit and split-K."""
        for N, K, sk in ((64, 256, 1), (128, 1024, 4)):
            with self.subTest(N=N, K=K, sk=sk):
                import g17qsm as Q
                lay = Q.layout(N, K, 1, 4, sk, xrows=True, mb=32)
                prog = Q.build(lay)
                x, packed, s16, b16, q = Q.case(lay, batch=20)
                want = Q.reference(lay, x, q, s16, b16)
                with tempfile.TemporaryDirectory() as t:
                    got = Q.emulate(lay, prog, x, packed, s16, b16, t)
                self.assertEqual(int((got.view("<u4") != want.view("<u4")).sum()), 0)

    def test_256_head_slices(self):
        """MM 25.185: split-K with 256 n groups (the head grid's slices past the old 128), bit-exact on g17emu at
        N 8192 (w1's width), K 512, sk 2; 512 slices refuse."""
        import g17qsm as Q
        lay = Q.layout(8192, 512, 2, 4, 2, xrows=True)
        self.assertEqual((lay["G"], lay["groups"]), (256, 512))
        prog = Q.build(lay)
        x, packed, s16, b16, q = Q.case(lay, batch=8)
        want = Q.reference(lay, x, q, s16, b16)
        with tempfile.TemporaryDirectory() as t:
            got = Q.emulate(lay, prog, x, packed, s16, b16, t)
        self.assertEqual(int((got.view("<u4") != want.view("<u4")).sum()), 0)
        with self.assertRaises(ValueError):
            Q.layout(16384, 512, 2, 4, 2, xrows=True)

    def test_the_fp16_rounding_is_observed(self):
        """The control: an A dequantized to fp32 without the fp16 rounding (the MMA then truncates it) differs."""
        import g17qmm as QM
        import g17prefillmma as MM
        lay, got, want, (x, q, s16, b16) = self._run(64, 256, 4, 1, False)
        s = np.repeat(QM.Q._from_bf16(s16).astype(np.float32), 64, 1)
        b = np.repeat(QM.Q._from_bf16(b16).astype(np.float32), 64, 1)
        W32 = ((q.astype(np.float32) * s).astype(np.float32) + b).astype(np.float32)
        Y = np.zeros((64, 16), np.float32)
        Xt = np.asarray(x, np.float16).astype(np.float32).T
        for tr in range(lay["trips"]):
            Y = MM.gemm_mma_v(W32[:, 64 * tr:64 * tr + 64], Xt[64 * tr:64 * tr + 64], Y, truncate_a=True)
        self.assertGreater(int((Y.T.view("<u4") != got.view("<u4")).sum()), 0)

    def test_refusals(self):
        import g17qsm as Q
        for kw in (dict(nt=4), dict(ks=8), dict(sk=3), dict(K=16384 * 4, ks=1)):
            args = dict(N=2048, K=2048, nt=2, ks=4, sk=1)
            args.update(kw)
            with self.subTest(**kw), self.assertRaises(ValueError):
                Q.layout(args["N"], args["K"], args["nt"], args["ks"], args["sk"])


if __name__ == "__main__":
    unittest.main()
