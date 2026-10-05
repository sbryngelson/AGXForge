"""build_qmv2's split-K form (lay["ksplit"], MM 25.141.15): S simdgroups take one threadgroup's rows over S contiguous
K slices and meet in threadgroup memory. Measured bit-exact on hardware at q4 2048 x 8192 (rows 1 and 2, S = 2, 4, 8)
and 7-9 us per layer faster than the unsplit w2 in the stamped chain. Compile-only:

- the program releases no register its loop carries before the loop (the phi-entry hazard, test_g17phientry);
- it stays inside the cooperative three-binding class: system registers 156 and 164 only, a declared threadgroup
  of 32 S threads, and it refuses to build outside that class;
- qmv2_ksplit_reference slices K, the scales and the biases consistently: it agrees with the float64 dot of the
  dequantized matrix to fp32 accuracy, which a slice taking the wrong scale group misses by orders of magnitude."""
import os, sys, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))
import numpy as np
from test_g17phientry import released_before_loop

BASE = dict(interleave=True, lean=True, coalesced=True, a16=True, xvec=True, wpt=2, vload=True, hoist_consts=True)


def _w2(rows, sgs, **extra):
    import g17qmv as Q
    lay = dict(Q.case(2048, 8192, 4, rows, nocarrier=True)[0], **BASE, sgs=sgs, ksplit=True, coop=True)
    lay.update(extra)
    return Q, Q.with_residual(lay, "add32_to16")


class KSplit(unittest.TestCase):
    def test_the_delivered_form_keeps_its_loop_registers(self):
        for rows, sgs in ((1, 4), (2, 2)):
            Q, lay = _w2(rows, sgs)
            self.assertEqual(released_before_loop(Q.build_qmv2(lay).code), [], (rows, sgs))

    def test_it_stays_in_the_cooperative_class(self):
        Q, lay = _w2(1, 4)
        p = Q.build_qmv2(lay)
        self.assertEqual(tuple(p.abi()["system_registers"]), (156, 164))
        from agxforge.g17 import cooperativemetadata as CM
        self.assertIn((128, 1, 1), CM.MEASURED_SIZES)
        with self.assertRaises(ValueError):
            Q.build_qmv2(dict(lay, coop=False))
        with self.assertRaises(ValueError):
            Q.build_qmv2(dict(lay, sgs=3))

    def test_the_reference_slices_scales_with_k(self):
        import g17qmv as Q
        lay, x, packed, s16, b16, q = Q.case(64, 8192, 4, 1, nocarrier=True)
        s32, b32 = Q._from_bf16(s16).astype(np.float64), Q._from_bf16(b16).astype(np.float64)
        g = lay["group"]
        deq = q.astype(np.float64) * np.repeat(s32, g, axis=1) + np.repeat(b32, g, axis=1)
        exact = deq @ np.asarray(x, np.float64)
        scale = np.abs(deq) @ np.abs(np.asarray(x, np.float64))
        for sgs in (2, 4, 8):
            got = Q.qmv2_ksplit_reference(dict(lay, **BASE, sgs=sgs), x, q, s16, b16).astype(np.float64)
            self.assertLess(float(np.max(np.abs(got - exact) / scale)), 1e-5, sgs)
        # the same check refuses a reference whose scales are misaligned by one group
        bad = Q.qmv2_ksplit_reference(dict(lay, **BASE, sgs=4), x, q, np.roll(s16, 1, axis=1), b16).astype(np.float64)
        self.assertGreater(float(np.max(np.abs(bad - exact) / scale)), 1e-3)


if __name__ == "__main__":
    unittest.main()
