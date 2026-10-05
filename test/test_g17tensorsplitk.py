#!/usr/bin/env python3
"""Split-K in tlower: G threadgroups partition the K contraction, threadgroup t computing the PARTIAL
over K-slice [t*K/G, (t+1)*K/G) alone and writing it to slot t of a (G*M)xN buffer. The partials are
reduced outside the body by an ascending-t fp32 fold (Piece A's gemm_reference(split_k=G)). Compile only;
the hardware receipt for the partials and the fold is the runtime test."""
import hashlib
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))

from agxforge.g17 import model, tlower

NAMES = model.registers()


def srs(body):
    return [NAMES.get(v) for i in model.decode(body, 0) if i.opcode and i.opcode.id in (14059, 14060)
            for k, v in i.values if k == "reg" and str(NAMES.get(v, "")).startswith("SR")]


class SplitK(unittest.TestCase):
    def test_split_k_1_is_byte_identical(self):
        # split_k=1 is strictly additive: the body is exactly the unsplit one.
        for kw in (dict(kloop=True), dict()):
            base = tlower.lower(16, 16, 128, 128, 16, 16, a_type="half", b_type="half", **kw)[0]
            one = tlower.lower(16, 16, 128, 128, 16, 16, a_type="half", b_type="half", split_k=1, **kw)[0]
            self.assertEqual(base, one)

    def test_only_a_split_body_reads_the_threadgroup_index(self):
        self.assertEqual(srs(tlower.lower(16, 16, 128, 128, 16, 16, a_type="half", b_type="half", kloop=True)[0]),
                         ["SR_SIMD_ELEM"])
        self.assertEqual(srs(tlower.lower(16, 16, 128, 128, 16, 16, a_type="half", b_type="half", kloop=True, split_k=4)[0]),
                         ["SR_SIMD_ELEM", "SR_TG_X"])

    def test_each_threadgroup_runs_one_K_slice(self):
        # G threadgroups each run K/G of the K loop trips, so the MMA count per body is 1/G of the whole.
        whole = tlower.lower(16, 16, 128, 128, 16, 16, a_type="half", b_type="half", kloop=True)[1]
        split = tlower.lower(16, 16, 128, 128, 16, 16, a_type="half", b_type="half", kloop=True, split_k=4)[1]
        self.assertEqual(split["split_k"], 4)
        self.assertEqual(split["k_per_threadgroup"], 32)
        self.assertEqual(whole["k_per_threadgroup"], 128)

    def test_the_body_decodes(self):
        body = tlower.lower(16, 16, 128, 128, 16, 16, a_type="half", b_type="half", kloop=True, split_k=4)[0]
        self.assertGreater(len(list(model.decode(body, 0))), 0)

    def test_refusals(self):
        # K must split into whole 16-wide issues; split-K never accumulates in the partial body; one axis.
        cases = [
            dict(split_k=3),                       # 128 % (16*3) != 0
            dict(split_k=2, accumulate=True),      # C is added in the fold, not the body
            dict(split_k=257),                     # index mask is eight bits
        ]
        for kw in cases:
            with self.subTest(**kw), self.assertRaises(ValueError):
                tlower.lower(16, 16, 128, 128, 16, 16, a_type="half", b_type="half", kloop=True, **kw)

    def test_combines_with_the_grid_n_column_grid(self):
        # split_k composes with grid_n: the id decomposes as kg*grid_n + col, so a wide projection
        # column-splits AND K-splits in one launch. One shared SR_TG_X read.
        body, plan = tlower.lower(64, 64, 128, 128, 64, 64, a_type="half", b_type="half", kloop=True,
                                  grid_n=2, split_k=2)
        self.assertEqual((plan["grid_n"], plan["split_k"]), (2, 2))
        self.assertEqual(srs(body), ["SR_SIMD_ELEM", "SR_TG_X"])

    def test_still_refuses_the_untested_axes(self):
        # a non-power-of-two column grid (the id split is a shift)
        with self.assertRaises(ValueError):
            tlower.lower(64, 96, 128, 128, 96, 96, a_type="half", b_type="half", kloop=True, split_k=2, grid_n=3)

    def test_combines_with_a_simdgroup_split(self):
        # MM 25.144.1: split_k with 2 or 4 simdgroups, bit-exact on hardware (w2 M 512 at 4 x split_k 2, wo M 128 at
        # 2 x 2). The simdgroup offsets its rows first, then the K group its slice and slot; both SRs are read.
        body, plan = tlower.lower(64, 64, 128, 128, 64, 64, a_type="half", b_type="half", kloop=True, split_k=2, sg=2)
        self.assertEqual(plan["split_k"], 2)
        self.assertEqual(srs(body), ["SR_SIMD_ELEM", "SR_SIMD_GRP", "SR_TG_X"])

    def test_the_spec_refuses_split_k_at_eight_simdgroups_by_name(self):
        # M6's fuzzer: sg 8 x split_k 2 compiled in tlower, then the manifest (TensorSpec) refused it. generic_spec
        # now refuses it by name before any build; the control is the same spec at 4 simdgroups.
        import g17tensorcommonruntime as GR
        spec = dict(M=128, N=64, K=1024, kloop=True, threadgroups=1, grid_n=2, split_k=2)
        GR.generic_spec(dict(spec, simdgroups=4))
        with self.assertRaisesRegex(ValueError, "refused: gemm_generic: split_k combines with 1, 2 or 4"):
            GR.generic_spec(dict(spec, simdgroups=8))

    def test_the_head_grid_and_the_key_block_offset_refuse_split_k_and_grid_n(self):
        # decodefull merge: head_index (the head grid, MM 25.135) and b_index (the register-held key-block
        # B offset, 25.114.3) each take the threadgroup id or the B index for themselves, so neither is
        # combined with the column or K grid. The controls: each grid lowers without them, and each of
        # them lowers without a grid, so the refusal is the combination's.
        for kw in (dict(grid_n=2), dict(split_k=2)):
            tlower.lower(32, 32, 128, 128, 32, 32, a_type="half", b_type="half", reserved=(120,), **kw)
        tlower.lower(32, 32, 128, 128, 32, 32, a_type="half", b_type="half", head_index=(4096, 2048, 512))
        tlower.lower(32, 32, 128, 128, 32, 32, a_type="half", b_type="half", reserved=(120,), b_index=(120, 256))
        for kw in (dict(grid_n=2), dict(split_k=2)):
            with self.subTest(head_index=True, **kw), self.assertRaises(ValueError):
                tlower.lower(32, 32, 128, 128, 32, 32, a_type="half", b_type="half", head_index=(4096, 2048, 512), **kw)
            with self.subTest(b_index=True, **kw), self.assertRaises(ValueError):
                tlower.lower(32, 32, 128, 128, 32, 32, a_type="half", b_type="half", reserved=(120,), b_index=(120, 256), **kw)


if __name__ == "__main__":
    unittest.main()
