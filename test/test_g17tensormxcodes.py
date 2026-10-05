#!/usr/bin/env python3
"""Production row P5 (machine model 25.128): E8M0 scale codes decoded in the kernel, the code-0
refusal and code-255 NaN, the fp8 INPUT nonfinite policy, and fp8 and MX inside the K loop. Compile
only; every pinned program ran on hardware as preregistered (the g17-tensor-p5-v1 evidence set, packed into
evidence/g17-seta.zip)."""
import hashlib
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import g17tensorcommonruntime as R  # noqa: E402
from agxforge.g17 import model, tlower  # noqa: E402

MX = dict(M=32, N=32, K=64, a="fp8e4m3", b="fp8e5m2", epilogue=["mx32e8m0"])
KL = dict(M=32, N=32, K=512, a="fp8e4m3", b="fp8e5m2", kloop=True)
# spec -> the first 16 hex of the sha256 of the program that passed 3/3 on hardware (25.128.2)
DISPATCHED = (
    (MX, "db4aa54f9b5fb812"),
    (dict(M=32, N=32, K=96, a="bfloat", b="bfloat", epilogue=["mx32e8m0"]), "ab3da29b757db122"),
    (dict(M=64, N=32, K=128, a="fp8e5m2", b="fp8e4m3", epilogue=["mx32e8m0"], threadgroups=2), "32dfe9f23e4e91f3"),
    (dict(M=32, N=32, K=64, a="fp8e4m3", b="fp8e5m2"), "2e83bd6f0be3a23d"),     # the fp8_nonfinite program
    (KL, "826d2a9438155282"),
    (dict(KL, epilogue=["mx32e8m0"]), "024dfc60459e61a5"),
)


def opcodes(code):
    return [i.opcode.id for i in model.decode(code, 0) if i.opcode]


class Decode(unittest.TestCase):
    def test_codes_1_to_254_decode_exactly(self):
        codes = np.arange(1, 255, dtype=np.uint8)
        got = R.e8m0_decode_bits(codes).astype("<u4").view("<f4")
        want = np.ldexp(np.float32(1.0), codes.astype(np.int32) - 127).astype("<f4")
        self.assertTrue(np.array_equal(got.view("<u4"), want.view("<u4")))

    def test_code_255_is_the_quiet_nan_and_not_infinity(self):
        self.assertEqual(int(R.e8m0_decode_bits([255])[0]), 0x7FC00000)
        # the control: the plain e << 23 would be +infinity
        self.assertEqual(255 << 23, 0x7F800000)

    def test_code_0_decodes_to_zero_which_is_why_it_is_refused(self):
        self.assertEqual(int(R.e8m0_decode_bits([0])[0]), 0)
        self.assertNotEqual(np.float32(0.0), np.float32(2.0 ** -127))

    def test_the_host_refuses_code_0_by_name_and_admits_the_rest(self):
        R.check_e8m0_codes(np.arange(1, 256, dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "scale code.s. 0: E8M0 scale code 0 means 2\\^-127"):
            R.check_e8m0_codes(np.array([5, 0, 9], np.uint8))
        with self.assertRaises(ValueError):
            R.e8m0_factors(np.array([0], np.uint8), "ocp")

    def test_the_code_form_reference_is_the_table_form_reference(self):
        rng = np.random.default_rng(11)
        a = rng.uniform(-2, 2, (16, 64)).astype(np.float16).astype(np.float32)
        b = rng.uniform(-2, 2, (64, 16)).astype(np.float16).astype(np.float32)
        ca, fa = R.mx_scale_table(rng, 2 * 16)
        cb, fb = R.mx_scale_table(rng, 2 * 16)
        by_table = R.mx_post_reference(a, b, fa.reshape(2, 16), fb.reshape(2, 16))
        by_codes = R.mx_post_reference(a, b, R.e8m0_factors(ca).reshape(2, 16), R.e8m0_factors(cb).reshape(2, 16))
        self.assertTrue(np.array_equal(by_table.view("<u4"), by_codes.view("<u4")))

    def test_a_nan_code_makes_its_row_and_column_nan(self):
        a = np.ones((16, 32), np.float32); b = np.ones((32, 16), np.float32)
        sa = np.full((1, 16), 127, np.uint8); sb = np.full((1, 16), 127, np.uint8)
        sa[0, 3] = 255; sb[0, 7] = 255
        out = R.mx_post_reference(a, b, R.e8m0_factors(sa), R.e8m0_factors(sb))
        self.assertEqual(int(np.isnan(out).sum()), 16 + 16 - 1)
        inf = R.mx_post_reference(a, b, R.e8m0_factors(sa, "inf"), R.e8m0_factors(sb, "inf"), check=False)
        self.assertEqual(int(np.isnan(inf).sum()), 0)                 # the rival reading differs


class Emission(unittest.TestCase):
    def test_dispatched_programs_keep_their_hashes(self):
        for spec, want in DISPATCHED:
            with self.subTest(spec=spec):
                self.assertEqual(hashlib.sha256(R.build_generic_program(spec).code).hexdigest()[:16], want)

    def test_each_tile_block_loads_three_code_words_and_decodes_six_codes(self):
        ops = opcodes(tlower.lower(32, 32, 64, 64, 32, 32, a_type="fp8e4m3", b_type="fp8e5m2",
                                   epilogue=(("mx", 0, 2048, 1, 2048, "e8m0"),))[0])
        tile_blocks = 4 * 2
        self.assertEqual(ops.count(17014), 2 * tile_blocks)               # SA row byte, register shift
        # the table form reads fp32 words (op12709) and decodes nothing
        table = opcodes(tlower.lower(32, 32, 64, 64, 32, 32, a_type="fp8e4m3", b_type="fp8e5m2",
                                     epilogue=(("mx", 0, 2048, 1, 2048),))[0])
        self.assertEqual(table.count(17014), 0)
        self.assertEqual(ops.count(12656) - table.count(12656), 3 * tile_blocks)    # SB word + two SA words
        # six codes per tile-block, each decoded by e+1, >>8, <<22, <<23 and an add
        self.assertEqual(ops.count(10282) - table.count(10282), 6 * tile_blocks)
        # plus the prologue's two: saq = (rowg >> 2) << 2 and ssh = (rowg & 3) << 3
        self.assertEqual(ops.count(14391) - table.count(14391), 2 * 6 * tile_blocks + 2)

    def test_fp8_in_the_k_loop_unpacks_inside_the_body(self):
        code = tlower.lower(32, 32, 512, 512, 32, 32, a_type="fp8e4m3", b_type="fp8e5m2", kloop=True)[0]
        ins = [i for i in model.decode(code, 0) if i.opcode]
        edge = next(i for i in ins if i.opcode.id == 458)
        from agxforge.g17 import asm
        top = edge.offset + asm.decode_branch10(bytes(code[edge.offset:edge.offset + 10]))
        body = [i.opcode.id for i in ins if top <= i.offset < edge.offset]
        self.assertEqual(body.count(17642), 4 * (2 + 2))  # four per fragment: two A row tiles, two B column tiles
        cmp = next(i for i in ins if i.opcode.id == 10369)
        self.assertEqual([v for _k, v in cmp.values][5], 31)

    def test_a_microscaled_loop_runs_one_block_per_trip(self):
        code = tlower.lower(32, 32, 512, 512, 32, 32, a_type="fp8e4m3", b_type="fp8e5m2", kloop=True,
                            epilogue=(("mx", 0, 32 * 512, 1, 512 * 32, "e8m0"),))[0]
        trips = [[v for _k, v in i.values][5] for i in model.decode(code, 0) if i.opcode and i.opcode.id == 10369]
        self.assertEqual(trips, [15, 15])                 # one loop per accumulator group, K/32 - 1 trips

    def test_refusals_by_name(self):
        for kw, why in ((dict(kloop=True, epilogue=(("mx", 0, 2048, 1, 2048),)), "e8m0 form"),
                        (dict(epilogue=(("mx", 0, 2050, 1, 2048, "e8m0"),)), "multiples of 4"),
                        (dict(epilogue=(("mx", 0, 2048, 1, 2048, "e8m1"),)), "sb_off.,? e8m0"),
                        (dict(epilogue=(("fp8", "e4m3fn"),), keep=True), "cannot hand fp32 registers on")):
            with self.subTest(kw=kw), self.assertRaisesRegex(ValueError, why):
                tlower.lower(32, 32, 64, 64, 32, 32, a_type="fp8e4m3", b_type="fp8e5m2", **kw)
        with self.assertRaisesRegex(ValueError, "2..256 blocks"):
            tlower.lower(32, 32, 32, 32, 32, 32, a_type="fp8e4m3", b_type="fp8e5m2", kloop=True,
                         epilogue=(("mx", 0, 1024, 1, 1024, "e8m0"),))
        with self.assertRaisesRegex(ValueError, "mx32e8m0"):
            R.generic_spec(dict(KL, epilogue=["mx32"]))
        with self.assertRaises(ValueError):
            R.generic_spec(dict(M=32, N=32, K=64, epilogue=["mx32"], mx_inject="code255"))

    def test_the_runtime_admits_the_code_word(self):
        from agxforge.g17 import runtime
        base = dict(M=32, N=32, K=64, lda=64, ldb=32, ldc=32, a_type="fp8e4m3", b_type="fp8e5m2", c_type="float",
                    simdgroups=1, grid=(32, 1, 1), threadgroup=(32, 1, 1), composition="gemm_generic")
        runtime.TensorSpec(**base, epilogue=("mx32e8m0",))
        with self.assertRaises(ValueError):
            runtime.TensorSpec(**base, epilogue=("mx32e8m0", "mx32"))


class NonfiniteInputs(unittest.TestCase):
    def test_every_nonfinite_code_is_named_and_decodes_nonfinite(self):
        for fmt, codes in R.FP8_NONFINITE.items():
            vals = R._fp8_values(np.array(codes, np.uint8), fmt)
            self.assertFalse(np.isfinite(vals).any(), fmt)
            finite = [c for c in range(256) if c not in codes]
            self.assertTrue(np.isfinite(R._fp8_values(np.array(finite, np.uint8), fmt)).all(), fmt)
        self.assertEqual(int(np.isinf(R._fp8_values(np.array(R.FP8_NONFINITE["e4m3"], np.uint8), "e4m3")).sum()), 0)

    def test_the_policy_is_ieee_and_both_rivals_differ(self):
        rng = np.random.default_rng(5)
        a = R._fp8_values(R._fp8_codes(rng, 32 * 32, "e4m3"), "e4m3").reshape(32, 32)
        b = R._fp8_values(R._fp8_codes(rng, 32 * 32, "e5m2"), "e5m2").reshape(32, 32)
        a[2, 5] = np.nan; b[3, 4] = np.inf; b[8, 7] = -np.inf; a[4, :] = 1
        b[13, 10] = np.inf; b[14, 10] = -np.inf; b[20, 11] = np.inf; a[6, 20] = 0
        s = dict(a="fp8e4m3", b="fp8e5m2", threadgroups=2)
        ocp = R._nonfinite_reference(a, b, dict(s, nonfinite_model="ocp"))
        self.assertTrue(np.isnan(ocp[2]).all())                      # a NaN code poisons its row
        self.assertTrue(np.isnan(ocp[4, 10]))                        # +inf and -inf meet on row 4 of column 10
        self.assertTrue(np.isnan(ocp[6, 11]))                        # infinity times zero
        self.assertEqual(ocp[4, 4], np.inf)                          # row 4 is all ones: +inf reaches (4, 4)
        self.assertEqual(ocp[4, 7], -np.inf)
        for rival in ("saturate", "zero"):
            rv = R._nonfinite_reference(a, b, dict(s, nonfinite_model=rival))
            self.assertTrue(np.isfinite(rv).all(), rival)

    def test_nan_class_comparison_only_where_nan_is_planted(self):
        self.assertTrue(R.generic_nan_class(dict(fp8_nonfinite=True)))
        self.assertTrue(R.generic_nan_class(dict(mx_inject="code255")))
        self.assertFalse(R.generic_nan_class(dict(mx_inject="code0")))
        self.assertFalse(R.generic_nan_class(dict()))


class CutPoints(unittest.TestCase):
    """tensorcuts enumerates the hand-offs the lowering admits; it never ranks them (selection is P11's)."""

    def test_an_fp8_stage_into_a_bf16_stage_can_only_cut_through_fp8_bytes(self):
        from agxforge.g17 import tensorcuts
        r = tensorcuts.cut_points([dict(M=32, N=64, K=64, a="fp8e4m3", b="fp8e4m3"),
                                   dict(M=32, N=32, K=64, a=None, b="bfloat")])
        self.assertEqual(sorted((o["kind"], o["consumer_a"], o["dispatches"], o["producer_last_epilogue"])
                                for o in r["options"]),
                         [("cut_fp8", "fp8e4m3", 2, "fp8e4m3"), ("cut_fp8", "fp8e5m2", 2, "fp8e5m2")])
        kinds = {k for _b, k, _why in r["refused"]}
        self.assertTrue({"register", "memory", "cut_fp32"} <= kinds)

    def test_a_half_chain_keeps_its_one_dispatch_forms(self):
        from agxforge.g17 import tensorcuts
        r = tensorcuts.cut_points([dict(M=16, N=32, K=64, a="half", b="half"),
                                   dict(M=16, N=32, K=32, a="float", b="half")])
        kinds = [o["kind"] for o in r["options"]]
        self.assertEqual(kinds, ["register", "memory", "cut_fp32"])
        self.assertIn("chain_register", r["options"][0]["evidence"][0])   # the FEED_TABLE receipt

    def test_the_options_are_unranked_and_the_chain_is_checked(self):
        from agxforge.g17 import tensorcuts
        for o in tensorcuts.cut_points([dict(M=16, N=32, K=64, a="half", b="half"),
                                        dict(M=16, N=32, K=32, a=None, b="half")])["options"]:
            self.assertFalse({"cost", "time", "score", "rank"} & set(o))
        with self.assertRaisesRegex(ValueError, "stage 1's A is stage 0's output"):
            tensorcuts.cut_points([dict(M=16, N=32, K=64, a="half", b="half"),
                                   dict(M=16, N=32, K=48, a="float", b="half")])

    def test_every_p11_kind_maps_onto_emittable_hand_offs(self):
        from agxforge.g17 import tensorcuts
        emit = {"register", "memory", "cut_fp32", "cut_fp8"}
        self.assertEqual(set(tensorcuts.P11_KINDS), {"chain.fused", "chain.cut.quantize", "chain.cut.pre_quantize",
                                                     "chain.cut.all", "chain.cut.mixed_format"})
        for kind, (built, via, why) in tensorcuts.P11_KINDS.items():
            self.assertTrue(set(via) <= emit, kind)
            self.assertTrue(why, kind)
        fp8 = [dict(M=32, N=64, K=64, a="fp8e4m3", b="fp8e4m3"), dict(M=32, N=32, K=64, a=None, b="fp8e4m3")]
        got = {k: v[0] for k, v in tensorcuts.p11_kinds_for(fp8).items()}
        self.assertEqual(got, {"chain.fused": False, "chain.cut.quantize": True, "chain.cut.pre_quantize": False,
                               "chain.cut.all": True, "chain.cut.mixed_format": False})
        half = [dict(M=16, N=32, K=64, a="half", b="half"), dict(M=16, N=32, K=32, a="float", b="half")]
        self.assertTrue(tensorcuts.p11_kinds_for(half)["chain.fused"][0])
        self.assertFalse(tensorcuts.p11_kinds_for(half)["chain.cut.quantize"][0])


if __name__ == "__main__":
    unittest.main()
