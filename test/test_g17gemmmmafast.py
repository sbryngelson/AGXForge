"""The vectorised MMA reference equals the per-issue one bit for bit (MM 25.144.1).

`_gemm_mma` models one 16-wide MMA issue per (m, n, K-slice) in Python; at prefill shapes that is tens of millions of
issues. `_gemm_mma_fast` does the same fp32 operations vectorised. Values span a wide exponent range so every
rounding step matters; a changed pairing or accumulation order fails here."""
import os, sys, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))
import numpy as np


def _slow(R, a, b, c, M, N, K, **kw):
    out = np.empty((M, N), dtype="<f4")
    for m in range(M):
        for n in range(N):
            acc = None
            for s in range(0, K, 16):
                acc = R._mma16(a[m, s:s + 16], b[s:s + 16, n], acc, **kw)
            if c is not None:
                acc = R._rne32(acc + R._rne32(c[m, n]))
            out[m, n] = acc
    return out


class FastEqualsSlow(unittest.TestCase):
    def test_bit_equal(self):
        import g17tensorcommonruntime as R
        rng = np.random.default_rng(7)
        for M, N, K, withc, tr in ((16, 32, 64, False, False), (24, 40, 48, True, False), (16, 16, 32, False, True)):
            a = (rng.standard_normal((M, K)) * np.exp2(rng.integers(-8, 8, (M, K)))).astype(np.float16)
            b = (rng.standard_normal((K, N)) * np.exp2(rng.integers(-8, 8, (K, N)))).astype(np.float16)
            c = rng.standard_normal((M, N)).astype(np.float32) if withc else None
            if tr:
                a = (rng.standard_normal((M, K)) * 3).astype(np.float32)
            kw = dict(truncate_a=tr)
            want = _slow(R, a, b, c, M, N, K, **kw)
            got = R._gemm_mma_fast(a, b, c, M, N, K, **kw)
            self.assertTrue(np.array_equal(want.view(np.uint32), got.view(np.uint32)), (M, N, K, withc, tr))

    def test_a_different_order_is_caught(self):
        # the control: summing each issue's 16 products left to right gives different bits
        import g17tensorcommonruntime as R
        rng = np.random.default_rng(3)
        a = (rng.standard_normal((16, 64)) * np.exp2(rng.integers(-8, 8, (16, 64)))).astype(np.float16)
        b = (rng.standard_normal((64, 16)) * np.exp2(rng.integers(-8, 8, (64, 16)))).astype(np.float16)
        seq = np.zeros((16, 16), np.float32)
        for k in range(64):
            seq = seq + a[:, k:k + 1].astype(np.float32) * b[k:k + 1, :].astype(np.float32)
        got = R._gemm_mma_fast(a, b, None, 16, 16, 64)
        self.assertFalse(np.array_equal(seq.view(np.uint32), got.view(np.uint32)))


if __name__ == "__main__":
    unittest.main()
