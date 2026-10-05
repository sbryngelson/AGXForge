"""op3850, the hardware rsqrt seed, as an exact function (MM 25.141.16), and the wide norm that uses it.

Measured: one dispatch per band of 2^24 inputs (x in [1,4), [2^-16, 2^-14), [2^10, 2^12)), sentinel over every
output. The seed obeys seed(x 4^k) == seed(x) 2^-k on every input at k = -8 and +5, and it equals the correctly
rounded rsqrt except at 725,821 (parity, mantissa) keys, stored in isa/g17-rsqrt-seed.npz. g17decodestep.rsqrt_seed
reproduced all 3 x 2^24 hardware values. Compile-only here:

- the table is well formed (sorted unique keys, every stored seed exactly one ulp from rsqrt_rn); a corrupted or
  truncated table fails;
- six hardware values pinned literally (four at exception keys, where rsqrt_rn is WRONG for the hardware) and the
  scaling law, applied to the model;
- the rs_seed norm carries no corrected-rsqrt tail and releases no register its loop carries."""
import os, sys, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))
import numpy as np

HARDWARE = [(0x3fc4da34, 0x3f4e6e4e), (0x400033f6, 0x3f34e041), (0x405cafb3, 0x3f09dc7c), (0x401b776f, 0x3f244098),
            (0x3f800000, 0x3f800000), (0x40000005, 0x3f3504f0)]


class RsqrtSeed(unittest.TestCase):
    def test_the_table_is_well_formed(self):
        import g17decodestep as D
        z = np.load(os.path.join(ROOT, "isa", "g17-rsqrt-seed.npz"))
        idx, seed = z["index"].astype(np.int64), z["seed"].astype(np.int64)
        self.assertEqual(len(idx), 725821)
        self.assertTrue(np.all(np.diff(idx) > 0))
        x = ((((idx >> 23) & 1) + 127) << 23 | (idx & 0x7FFFFF)).astype(np.uint32).view(np.float32)
        rn = D.rsqrt(x).view(np.uint32).astype(np.int64)
        self.assertTrue(np.all(np.abs(seed - rn) == 1))

    def test_the_model_gives_the_hardware_values(self):
        import g17decodestep as D
        xs = np.array([x for x, _ in HARDWARE], np.uint32).view(np.float32)
        want = np.array([y for _, y in HARDWARE], np.uint32)
        self.assertEqual(D.rsqrt_seed(xs).view(np.uint32).tolist(), want.tolist())
        # the first four are exception keys: the correctly rounded value is NOT the hardware's
        self.assertTrue(np.all(D.rsqrt(xs[:4]).view(np.uint32) != want[:4]))
        for k in (-8, -3, 5, 9):
            scaled = (xs.view(np.uint32).astype(np.int64) + (2 * k << 23)).astype(np.uint32).view(np.float32)
            self.assertEqual(D.rsqrt_seed(scaled).view(np.uint32).tolist(),
                             (want.astype(np.int64) - (k << 23)).astype(np.uint32).tolist(), k)

    def test_the_seed_norm_has_no_corrected_tail(self):
        import g17decodeops as O
        from test_g17phientry import released_before_loop
        lay = O.rmsnorm_loop_layout(2048, "half", groups=32, unroll=16, hoist=True)
        slow = O.build_rmsnorm_wide(dict(lay, rs_once=True, out32=True), 1e-5).code
        fast = O.build_rmsnorm_wide(dict(lay, rs_seed=True, out32=True), 1e-5).code
        self.assertLess(len(fast), len(slow) // 2)
        self.assertEqual(released_before_loop(fast), [])


if __name__ == "__main__":
    unittest.main()
