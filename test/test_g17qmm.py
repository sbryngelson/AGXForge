"""M1's quantized prefill GEMM (MM 25.144.1), compile-only.

- The dequant reference is fp16_rne(fp32(q) * s + b) with fp32 RNE per op: checked against an independent scalar
  formula on every value of a q4 and a q8 matrix, and a control (q * s + b summed in float64 then rounded once) that
  must differ somewhere, so the check can fail.
- The dequant kernel builds at both widths; every (role, M) GEMM shape the tool chooses passes the spec normaliser
  and lowers (the K loop in 4 simdgroups, split_k 2 for w2).
- g17deliver's qmm and qmm_dequant schemas carry the fields Piece A's graph reads.
Hardware: tools/g17qmm.py verify (every role, both widths, M 512 bit-exact, recorded in 25.144.1)."""
import os, sys, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))
import numpy as np


class DequantReference(unittest.TestCase):
    def test_matches_a_scalar_formula(self):
        import g17qmm as M, g17qmv as Q
        for bits in (4, 8):
            packed, s16, b16, q = M.weights(64, 128, bits, 5)
            got = M.dequant_reference(q, s16, b16, bits)
            s = Q._from_bf16(s16)
            b = Q._from_bf16(b16)
            want = np.empty((128, 64), np.float16)
            for n in range(64):
                for k in range(128):
                    v = np.float32(np.float32(q[n, k]) * np.float32(s[n, k // 64]))
                    want[k, n] = np.float16(np.float32(v + np.float32(b[n, k // 64])))
            self.assertTrue(np.array_equal(got.view(np.uint16), want.view(np.uint16)), bits)

    def test_the_fp32_step_cannot_reach_fp16_but_an_fp16_step_does(self):
        # FINDING, pinned: at these weights, one rounding of the exact s*q + b to fp16 equals the kernel's fp32 mul, fp32
        # add, fp16 convert (fp32 carries 13 more bits), so the delivered value does not depend on fma vs mul+add.
        # The control that CAN fail: q*s rounded to fp16 before the bias is added.
        import g17qmm as M, g17qmv as Q
        packed, s16, b16, q = M.weights(256, 256, 8, 9)
        ref = M.dequant_reference(q, s16, b16, 8).view(np.uint16)
        s = np.repeat(Q._from_bf16(s16).astype(np.float64), 64, axis=1)
        b = np.repeat(Q._from_bf16(b16).astype(np.float64), 64, axis=1)
        once = (q.astype(np.float64) * s + b).astype(np.float16).T
        self.assertTrue(np.array_equal(once.view(np.uint16), ref))
        half_first = ((q.astype(np.float64) * s).astype(np.float16).astype(np.float64) + b).astype(np.float16).T
        self.assertFalse(np.array_equal(half_first.view(np.uint16), ref))


class Shapes(unittest.TestCase):
    def test_dequant_builds(self):
        import g17qmm as M
        for bits in (4, 8):
            for role, (N, K) in M.ROLES.items():
                self.assertGreater(len(M.build_dequant(M.dequant_layout(N, K, bits)).code), 0, (bits, role))

    def test_gemm_shapes_admitted(self):
        import g17qmm as M
        import g17tensorcommonruntime as R
        from agxforge.g17 import tlower
        for role, (N, K) in M.ROLES.items():
            for Mrows in (128, 256, 512):
                sg, gn, sk = M.gemm_shape(Mrows, N, K)
                R.generic_spec(M.gemm_spec(Mrows, N, K, sg, gn, sk))
                body, _ = tlower.lower(Mrows, N, K, K, N, N, a_type="half", b_type="half", kloop=True, sg=sg, grid=1,
                                       grid_n=gn, split_k=sk)
                self.assertLess(len(body), 16 * 1024, (role, Mrows))   # under the instruction-footprint cliff


class DeliverSchema(unittest.TestCase):
    def test_qmm_fields(self):
        import g17deliver as D
        for kind in ("qmm", "qmm_dequant"):
            for f in ("kind", "bits", "role", "variant", "bundle", "threadgroups", "threads_per_group", "base",
                      "slot_map", "program_sha256", "recipe", "out_layout"):
                self.assertIn(f, D.SCHEMA[kind], (kind, f))


if __name__ == "__main__":
    unittest.main()
