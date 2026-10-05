"""g17qmv._fma32v, the vectorised fp32 fma every exact reference uses, against _fma32's exact rationals.

_fma32v computes p = a b in f64 (exact), rounds p + c to ODD in f64, then to fp32, which is one correct
rounding of the exact a b + c. It replaced a per-element fall back to exact rationals that made
test_g17attnlongctx the gate's slowest module. The adversarial set is built to double-round: a 13-bit
odd integer times a 12-bit odd integer is a 25-bit odd product, an exact fp32 midpoint, and a tiny c
decides the direction. The plain f64-then-fp32 rounding gets about half of them wrong, and the test
asserts it does, so the set cannot pass by being too easy."""
import os, sys, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))
import numpy as np

F32 = np.float32


def _midpoint_cases(n=4000, seed=3):
    rng = np.random.default_rng(seed)
    a = (rng.integers(1 << 12, 1 << 13, n) | 1).astype(F32)
    b = (rng.integers(1 << 11, 1 << 12, n) | 1).astype(F32)
    keep = a.astype(np.float64) * b.astype(np.float64) >= 2.0 ** 24
    a, b = a[keep], b[keep]
    c = (rng.choice([1, -1], a.size) * 2.0 ** -30).astype(F32)
    return a, b, c


class Fma32v(unittest.TestCase):
    def _exact(self, a, b, c):
        import g17qmv as Q
        return np.array([Q._fma32(a[i], b[i], c[i]) for i in range(a.size)], F32)

    def test_it_is_one_correct_rounding_on_midpoints(self):
        import g17qmv as Q
        a, b, c = _midpoint_cases()
        ref = self._exact(a, b, c)
        self.assertTrue((Q._fma32v(a, b, c).view(np.uint32) == ref.view(np.uint32)).all())
        naive = (a.astype(np.float64) * b.astype(np.float64) + c.astype(np.float64)).astype(F32)
        self.assertGreater(int((naive.view(np.uint32) != ref.view(np.uint32)).sum()), a.size // 4,
                           "the midpoint set no longer separates double rounding from one rounding")

    def test_it_matches_on_wide_exponents_and_subnormals(self):
        import g17qmv as Q
        rng = np.random.default_rng(0)
        for sa, sc in ((1, 1), (1e-3, 1e8), (1e20, 1e-30), (1e-20, 1e-39)):
            a = (rng.standard_normal(2000) * sa).astype(F32)
            b = rng.standard_normal(2000).astype(F32)
            c = (rng.standard_normal(2000) * sc).astype(F32)
            got, ref = Q._fma32v(a, b, c), self._exact(a, b, c)
            same = (got.view(np.uint32) == ref.view(np.uint32)) | (np.isnan(got) & np.isnan(ref))
            self.assertTrue(same.all(), (sa, sc))


if __name__ == "__main__":
    unittest.main()
