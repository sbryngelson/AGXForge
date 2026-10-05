#!/usr/bin/env python3
"""Register-direct GEMM -> GEMM with a HALF consumer (Set A item 1): D is narrowed with op1016 in
registers, exactly Apple's MMA / eight op1016 / MMA recipe (compile only)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agxforge.g17 import cc, epienc, ir, model

NAMES = model.registers()


def chain(convert=True, adjacent=True):
    a = ir.Buffer("A", 1, elem=ir.F16); b = ir.Buffer("B", 2, elem=ir.F16); c = ir.Buffer("C", 3, elem=ir.F32)
    fn = ir.Function("halffeed", [a, b, c]); bl = ir.Builder(fn, fn.block("entry"))
    bl.tensor_matmul(a, b, c, M=32, N=32, K=64)
    if not adjacent:
        t = bl.builtin("threadgroup_position_in_grid", name="t")
        bl.store_at(c, t, bl.load(c, t, type=ir.F32, name="v"))
    bl.tensor_matmul(c, b, c, M=32, N=32, K=32, a_dtype="half", b_dtype="half",
                     a_converted_from="float" if convert else None)
    bl.ret(); ir.verify(fn)
    return fn


class HalfFeed(unittest.TestCase):
    def test_apples_conversion_bytes(self):
        # the two conversions decoded from Apple's own chain (seta-chainA-1x1-air3): R12L <- R0, R12H <- R1
        self.assertEqual(epienc.cvt_f32_to_f16("R12L", 0).hex(), "a90004082230a01284064200")
        self.assertEqual(epienc.cvt_f32_to_f16("R12H", 1).hex(), "a90204092230a01284064200")

    def test_eight_narrowings_per_fed_tile_and_the_half_mma_reads_them(self):
        ins = [i for i in model.decode(cc.compile_function(chain()).code, 0) if i.opcode]
        conv = [i for i in ins if i.opcode.id == 1016]
        self.assertEqual(len(conv), 4 * 8)                   # four fed D tiles (2x2), eight elements each
        narrowed = {NAMES.get(i.values[0][1]) for i in conv}
        second = [i for i in ins if i.opcode.id in (5106, 5107)][16:]
        a_tuples = {NAMES.get(i.values[3][1]) for i in second}
        for tup in a_tuples:
            regs = tup.split("_")
            self.assertTrue(all(r + h in narrowed for r in regs for h in "LH"), tup)

    def test_a_declared_conversion_without_a_register_feed_refuses(self):
        with self.assertRaises(cc.Unsupported):
            cc.compile_function(chain(adjacent=False))


if __name__ == "__main__":
    unittest.main()
