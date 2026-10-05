#!/usr/bin/env python3
"""Register-feed modes B, At and Bt (recon section 132 part 3) and the register-resident column
reduction (goal items 7 and 8). Default lowerings are byte-identical to before; each new form is
pinned by its decoded shape and its reference by the recon's own transform table."""
import hashlib
import json
import itertools
import os
import sys
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import g17tensorcommonruntime as R  # noqa: E402
from agxforge.g17 import cc, ir, model, tensorreduce as TR, tlower  # noqa: E402

MMA = (5098, 5099, 5100, 5101, 5104, 5105, 5106, 5107)


def chain(mode, conv=False):
    a = ir.Buffer("A", 1, elem=ir.F16); b = ir.Buffer("B", 2, elem=ir.F16); c = ir.Buffer("C", 3, elem=ir.F32)
    fn = ir.Function("tensor_gemm_generic_runtime_demo", [a, b, c]); bl = ir.Builder(fn, fn.block("entry"))
    bl.tensor_matmul(a, b, c, M=32, N=32, K=64)
    ft = "half" if conv else "float"
    if mode in ("A", "At"):
        bl.tensor_matmul(c, b, c, M=32, N=32, K=32, a_dtype=ft, b_dtype="half",
                         a_converted_from="float" if conv else None, feed=mode)
    else:
        bl.tensor_matmul(b, c, c, M=32, N=32, K=32, a_dtype="half", b_dtype=ft,
                         b_converted_from="float" if conv else None, feed=mode)
    bl.ret()
    return [i for i in model.decode(cc.compile_function(fn).code, 0) if i.opcode]


class Defaults(unittest.TestCase):
    def test_default_and_mode_a_lowerings_are_byte_identical_to_before(self):
        # the same 110 lowerings hashed on main before this change (4dcdaebc96c06bb5)
        h = hashlib.sha256()
        for M, N, K in itertools.product((16, 32, 48, 64), (16, 32, 64), (16, 32, 64)):
            for a in ("half", "float", "int8"):
                try:
                    body = tlower.lower(M, N, K, K, N, N, a_type=a, b_type=a if a != "float" else "half")[0]
                except Exception as e:
                    body = repr(e).encode()
                h.update(body)
        for conv in (None, "half"):
            a_regs = {(mi, k): 40 + 8 * (2 * mi + k) for mi in range(2) for k in range(2)}
            body = tlower.lower(32, 32, 32, 32, 32, 32, a_type="half" if conv else "float", b_type="half",
                                a_regs=a_regs, a_convert=conv, reserved=tuple(range(40, 72)))[0]
            h.update(body)
        self.assertEqual(h.hexdigest()[:16], "4dcdaebc96c06bb5")


class FeedModes(unittest.TestCase):
    def test_every_mode_compiles_register_fed(self):
        for mode in ("A", "B", "At", "Bt"):
            for conv in (False, True):
                with self.subTest(mode=mode, half=conv):
                    ins = chain(mode, conv)
                    c = Counter(i.opcode.id for i in ins)
                    self.assertEqual(sum(c[o] for o in MMA), 24)          # 16 producer + 8 consumer
                    self.assertEqual(c[1016], 32 if conv else 0)           # one narrowing per fed element
                    self.assertEqual(c[17257], 8)                         # D1 elided, D2 stored

    def test_the_transposed_modes_differ_only_in_the_consumers_transpose_field(self):
        for base, tr, fields in (("A", "At", {3, 5}), ("B", "Bt", {6, 8})):
            with self.subTest(mode=tr):
                x, y = chain(base), chain(tr)
                moved = set()
                for i, j in zip(x, y):
                    if i.raw != j.raw:
                        self.assertIn(i.opcode.id, MMA)
                        moved |= {k for k, (p, q) in enumerate(zip(i.values, j.values)) if p != q}
                self.assertEqual(moved, fields)

    def test_the_reference_operands_reproduce_section_132s_transform_table(self):
        rng = np.random.default_rng(3); M = 32
        H = rng.standard_normal((M, M)); X = rng.standard_normal((M, M))
        perm = np.array([16 * (i // 16) + R._rotl1(i % 16) for i in range(M)])
        rowr = np.array([16 * (i // 16) + R._rotr1(i % 16) for i in range(M)])
        self.assertTrue(np.allclose(X[:, perm] @ R.feed_operand(H, "B"), X @ H))
        self.assertTrue(np.allclose(R.feed_operand(H, "At") @ X[perm, :], (H.T @ X)[rowr, :]))
        self.assertTrue(np.allclose(X @ R.feed_operand(H, "Bt"), (X @ H.T)[:, perm]))

    def test_the_hardware_read_the_logical_operand_not_section_132s(self):
        # results/g17-tensor-feedmodes-v1: in B, At and Bt, fp32 and half, the program matched the
        # identity reference bit for bit and failed section 132's relabeled one. So the default
        # reference is identity, and the relabeled model is the named rival.
        self.assertEqual(R.generic_spec(dict(M=32, N=32, K=64))["feed_model"], "identity")
        receipt = Path(__file__).resolve().parents[1] / "results/g17-tensor-feedmodes-v1/dispatch-receipt.json"
        if not receipt.exists():
            self.skipTest("evidence not extracted (make evidence)")
        r = json.loads(receipt.read_text())
        for mode in ("B", "At", "Bt"):
            with self.subTest(mode=mode):
                self.assertEqual(r["neg_%s_identity" % mode]["status"], "passed")
                self.assertEqual(r["neg_%s_claims_A" % mode]["status"], "FAILED")
                for t in ("float", "half"):
                    self.assertEqual(r["feed_%s_%s" % (mode, t)]["status"], "FAILED")

    def test_refusals(self):
        D = {(i, j): 40 + 8 * (2 * i + j) for i in range(2) for j in range(2)}
        with self.assertRaises(ValueError):
            tlower.lower(32, 32, 32, 32, 32, 32, a_type="float", b_type="float", a_regs=D, b_regs=D,
                         reserved=tuple(range(40, 72)))
        with self.assertRaises(ValueError):
            tlower.lower(32, 32, 32, 32, 32, 32, a_type="half", b_type="float", b_regs=D, accumulate=True,
                         reserved=tuple(range(40, 72)))
        with self.assertRaises(ValueError):
            R.generic_spec(dict(M=32, N=32, K=64, stages=[[32, 32, "half", "B"], [32, 32, "half"]]))


class ColumnReduction(unittest.TestCase):
    def test_the_reduction_is_folded_in_registers_and_stores_row_zero_only(self):
        for op in ("sum", "max"):
            body = tlower.lower(32, 32, 64, 64, 32, 32, reduce=("col", op))[0]
            c = Counter(i.opcode.id for i in model.decode(body, 0) if i.opcode)
            self.assertEqual(c[14169], 24)                        # 12 shuffles per column tile
            self.assertEqual(c[17257] + c[17258], 2)              # one row-0 store per column tile

    def test_the_witness_stream_declines_a_gemm_it_cannot_carry(self):
        def compiled(**kw):
            a = ir.Buffer("A", 1, elem=ir.F16); b = ir.Buffer("B", 2, elem=ir.F16); c = ir.Buffer("C", 3, elem=ir.F32)
            fn = ir.Function("tensor_gemm_generic_runtime_demo", [a, b, c]); bl = ir.Builder(fn, fn.block("entry"))
            bl.tensor_matmul(a, b, c, M=32, N=32, K=64, strideA=128, strideB=64, **kw); bl.ret()
            return Counter(i.opcode.id for i in model.decode(cc.compile_function(fn).code, 0) if i.opcode)
        self.assertEqual(compiled()[14169], 0)
        self.assertEqual(compiled(reduce=("col", "sum"))[14169], 24)

    def test_the_lane_model_equals_the_measured_column_reduction_on_one_tile(self):
        rng = np.random.default_rng(5)
        d = rng.standard_normal((16, 16)).astype(np.float32)
        got = R.column_reduction(d, "sum")
        for j in range(4):
            lanes = [[float(d[4 * ((L >> 4) & 1) + ((L >> 1) & 3), 8 * ((L >> 3) & 1) + 4 * (L & 1) + j]),
                      float(d[4 * ((L >> 4) & 1) + ((L >> 1) & 3) + 8, 8 * ((L >> 3) & 1) + 4 * (L & 1) + j])]
                     for L in range(32)]
            want = TR.reduce_columns(lanes, operation="sum")
            for L in range(32):
                self.assertEqual(np.float32(want[L]).view(np.uint32),
                                 got[8 * ((L >> 3) & 1) + 4 * (L & 1) + j].view(np.uint32))


if __name__ == "__main__":
    unittest.main()
