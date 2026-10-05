#!/usr/bin/env python3
"""Fusion (Set A goal item 6): the exp2 register epilogue, and the witnessed-stream route declining
attributes it cannot express instead of dropping them."""
import hashlib
import os
import struct
import sys
import unittest
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import g17tensorcommonruntime as R  # noqa: E402
from agxforge.g17 import epienc, model, tlower  # noqa: E402


def ops(spec):
    return Counter(i.opcode.id for i in model.decode(R.build_generic_program(R.generic_spec(spec)).code, 0)
                   if i.opcode)


class Exp2Epilogue(unittest.TestCase):
    def test_exp2_is_ccs_own_instruction(self):
        # cc's exp2 of a loaded value (R6 <- R5, source released) and of an ALU result, source kept
        self.assertEqual(epienc.exp2(6, 5).hex(), "2702042a2a20a51a3000")
        self.assertEqual(epienc.exp2(5, 6, keep_src=True).hex(), "3780043a2a20a5121000")

    def test_one_exp2_per_accumulator_slot(self):
        body, _ = tlower.lower(32, 32, 64, 64, 32, 32, a_type="half", b_type="half", epilogue=(("exp2",),))
        self.assertEqual(sum(1 for i in model.decode(body, 0) if i.opcode and i.opcode.id == 1272), 32)

    def test_unknown_steps_still_refuse(self):
        with self.assertRaises(ValueError):
            tlower.lower(32, 32, 64, 64, 32, 32, epilogue=(("tanh",),))


class TheWitnessedStreamDeclinesWhatItCannotExpress(unittest.TestCase):
    """The 32x32xK half stream used to take the shape and drop an epilogue, a grid or simdgroup
    split, split-fp32 or saturation silently: an exp2 epilogue compiled to a bare GEMM."""

    def test_an_epilogue_on_the_witnessed_shape_is_emitted(self):
        c = ops(dict(M=32, N=32, K=64, epilogue=["scale:0x3fb8aa3b", "exp2"]))
        self.assertEqual((c[3290], c[1272]), (32, 32))

    def test_a_plain_gemm_keeps_the_stream_bytes(self):
        # main's program for the plain witnessed shape, unchanged
        code = R.build_generic_program(R.generic_spec(dict(M=32, N=32, K=64))).code
        self.assertEqual(hashlib.sha256(code).hexdigest()[:16], "14a748b82aeed714")

    def test_the_epilogue_program_is_not_the_plain_one(self):
        a = R.build_generic_program(R.generic_spec(dict(M=32, N=32, K=64))).code
        b = R.build_generic_program(R.generic_spec(dict(M=32, N=32, K=64, epilogue=["exp2"]))).code
        self.assertNotEqual(a, b)


class TheExp2Reference(unittest.TestCase):
    def test_the_bound_is_one_ulp_only_for_exp2_specs(self):
        self.assertEqual(R.generic_ulp_bound(dict(epilogue=["exp2"])), 1)
        self.assertEqual(R.generic_ulp_bound(dict(epilogue=["relu"])), 0)
        self.assertEqual(R.generic_ulp_bound(dict(epilogue=[])), 0)

    def test_ulp_distance(self):
        import numpy as np
        one = np.float32(1.0)
        nxt = np.frombuffer(struct.pack("<I", 0x3F800001), "<f4")[0]
        self.assertEqual(int(R._ulp_distance(np.array([one]), np.array([nxt]))[0]), 1)
        self.assertEqual(int(R._ulp_distance(np.array([np.float32(-0.0)]), np.array([np.float32(0.0)]))[0]), 0)




class IndependentGemmsOverSharedBuffers(unittest.TestCase):
    def test_two_and_four_bodies_compile_with_their_own_offsets(self):
        self.assertEqual(ops(dict(M=64, N=32, K=64, independent=2))[17257], 16)
        self.assertEqual(ops(dict(M=128, N=32, K=64, independent=4))[17257], 32)

    def test_the_wrong_a_control_is_a_different_program(self):
        a = R.build_generic_program(R.generic_spec(dict(M=64, N=32, K=64, independent=2))).code
        b = R.build_generic_program(R.generic_spec(dict(M=64, N=32, K=64, independent=2,
                                                        independent_wrong_a=True))).code
        self.assertNotEqual(a, b)

    def test_overlapping_c_regions_are_not_admitted(self):
        from agxforge.g17 import cc, ir
        a = ir.Buffer("A", 1, elem=ir.F16); b = ir.Buffer("B", 2, elem=ir.F16); c = ir.Buffer("C", 3, elem=ir.F32)
        fn = ir.Function("k", [a, b, c]); bl = ir.Builder(fn, fn.block("entry"))
        bl.tensor_matmul(a, b, c, M=32, N=32, K=64)
        bl.tensor_matmul(a, b, c, M=32, N=32, K=64, offsetC=16 * 32 * 4)     # half overlaps the first
        ops_ = [o for blk in fn.blocks for o in blk.ops if o.kind == "tensor_matmul"]
        self.assertFalse(cc._independent_tensor_group(fn, ops_))


class ScalarWorkBetweenChainBodies(unittest.TestCase):
    def test_the_scalar_step_sits_between_the_two_bodies(self):
        code = R.build_generic_program(R.generic_spec(dict(M=16, N=32, K=32, stages=[[16, 32, "float"]],
                                                           between="add1"))).code
        seq = [i.opcode.id for i in model.decode(code, 0) if i.opcode]
        first_float = min(j for j, o in enumerate(seq) if o in (5100, 5101))
        last_half = max(j for j, o in enumerate(seq[:first_float]) if o in (5106, 5107))
        self.assertIn(998, seq[last_half:first_float])                    # the fadd of the scalar step
        self.assertIn(17229, seq[last_half:first_float])                  # its store to C

    def test_one_dispatch_attention_compiles_two_softmax_rows(self):
        c = ops(dict(M=32, N=16, K=64, stages=[[16, 16, "float"]], between="softmax:2"))
        self.assertEqual((c[1272], c[3658]), (16, 2))

    def test_the_remaining_row_limit_is_the_emitters_measured_tile(self):
        # This pinned a four-row REFUSAL, blamed on the composition route's register reservation. The
        # cause was read_sr pre-colouring holding all twelve narrow registers; the allocator's compact
        # retry admits 4, 8 and 16 rows (test_g17tensorattention, machine model 25.110). What
        # still refuses is the row softmax's measured domain: rows 0..15 of the first tile.
        R.build_generic_program(R.generic_spec(dict(M=32, N=16, K=64, stages=[[16, 16, "float"]],
                                                    between="softmax:4")))
        with self.assertRaises(Exception):
            R.build_generic_program(R.generic_spec(dict(M=32, N=16, K=64, stages=[[16, 16, "float"]],
                                                        between="softmax:32")))


class NoRowButRowZeroBroadcastsFromLaneZero(unittest.TestCase):
    """The row butterfly (lane bits 0 and 3) leaves each row's result in the four lanes that own it.
    simd_broadcast_first copies lane 0, which owns rows 0 and 8 only, so a broadcast for any other
    row spreads the inactive identity: one-dispatch attention's row 1 came back NaN on hardware."""

    @staticmethod
    def _broadcasts(rows):
        from agxforge.g17 import ir
        a = ir.Buffer("A", 1, elem=ir.F16); b = ir.Buffer("B", 2, elem=ir.F16); c = ir.Buffer("C", 3, elem=ir.F32)
        fn = ir.Function("k", [a, b, c]); bl = ir.Builder(fn, fn.block("entry"))
        for r in rows:
            bl.tensor_row_softmax(c, row=r, M=32, N=16, K=64)
        return sum(1 for blk in fn.blocks for o in blk.ops if o.kind == "simd_broadcast")

    def test_row_zero_keeps_its_two_measured_broadcasts(self):
        self.assertEqual(self._broadcasts([0]), 2)

    def test_no_other_row_broadcasts(self):
        for r in (1, 2, 5, 7, 9, 15):
            with self.subTest(row=r):
                self.assertEqual(self._broadcasts([r]), 0)

    def test_the_released_row_zero_program_is_unchanged(self):
        self.assertEqual(hashlib.sha256(R.build_reduction_program().code).hexdigest()[:16], "c801921785317375")


if __name__ == "__main__":
    unittest.main()
