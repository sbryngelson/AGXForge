#!/usr/bin/env python3
"""The split-K fold kernel (tensorreduce.emit_split_k_fold): fold G stacked M x N partials into one
M x N tile by an ascending-t fp32 left fold - the reduce half of gemm_reference(split_k=G). Compile
only here; the bit-exact hardware receipt against the fold reference is the runtime test."""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agxforge.g17 import cc, ir, model, tensorreduce


def fold_program(G, M, N, M_live=None):
    partials = ir.Buffer("partials", 1, elem=ir.F32)
    out = ir.Buffer("out", 3, elem=ir.F32)
    fn = ir.Function("tensor_split_k_fold_demo", [partials, out])
    b = ir.Builder(fn, fn.block("entry"))
    tensorreduce.emit_split_k_fold(b, partials, out, G=G, M=M, N=N, M_live=M_live)
    b.ret()
    ir.verify(fn)
    return cc.compile_function(fn)


class SplitKFold(unittest.TestCase):
    def test_it_compiles_and_decodes(self):
        for G, M, N in ((8, 16, 32), (4, 16, 64), (2, 16, 32)):
            with self.subTest(G=G, M=M, N=N):
                prog = fold_program(G, M, N)
                self.assertGreater(len(list(model.decode(prog.code, 0))), 0)

    def test_it_reads_the_threadgroup_and_lane(self):
        prog = fold_program(8, 16, 32)
        names = model.registers()
        srs = [names.get(v) for i in model.decode(prog.code, 0) if i.opcode and i.opcode.id in (14059, 14060)
               for k, v in i.values if k == "reg" and str(names.get(v, "")).startswith("SR")]
        self.assertIn("SR_TG_X", srs)               # the column grid's threadgroup id
        self.assertEqual(tuple(prog.abi()["system_registers"]), (130, 156))

    def test_row_limited_fold_does_less_work(self):
        # a decode step pads its one token to 16 rows; M_live=1 folds only row 0, ~M times less work
        full = fold_program(8, 16, 128)
        live1 = fold_program(8, 16, 128, M_live=1)
        nf = len(list(model.decode(full.code, 0)))
        nl = len(list(model.decode(live1.code, 0)))
        self.assertLess(nl * 8, nf)                 # far fewer instructions (about 1/M the row work)
        self.assertEqual(tuple(live1.abi()["system_registers"]), (130, 156))   # same SR set

    def test_refusals(self):
        for kw in (dict(G=1, M=16, N=32), dict(G=4, M=16, N=48), dict(G=0, M=16, N=32),
                   dict(G=4, M=16, N=32, M_live=0), dict(G=4, M=16, N=32, M_live=17)):
            with self.subTest(**kw), self.assertRaises(tensorreduce.UnsupportedReduction):
                p = ir.Buffer("p", 1, elem=ir.F32); o = ir.Buffer("o", 3, elem=ir.F32)
                f = ir.Function("x", [p, o])
                b = ir.Builder(f, f.block("entry"))
                tensorreduce.emit_split_k_fold(b, p, o, **kw)


if __name__ == "__main__":
    unittest.main()
