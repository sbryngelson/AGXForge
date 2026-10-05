#!/usr/bin/env python3
"""The cross-simdgroup combine (Set A item 7): one threadgroup barrier between the stores and the
loads, the simdgroup index read from SR_SIMD_GRP, and an ascending fold (compile only)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agxforge.g17 import cc, ir, model, tensorreduce as TR


def build(simdgroups, operation):
    f = ir.Function("combine_probe", [ir.Buffer("X", 1, elem=ir.F32), ir.Buffer("Y", 2, elem=ir.F32)])
    f.declare_threadgroup(words=4, size=(32 * simdgroups, 1, 1))
    b = ir.Builder(f, f.block("entry"))
    t = b.builtin("thread_position_in_grid", name="t")
    v = b.load(f.buffers[0], t, type=ir.F32, name="v")
    b.store_at(f.buffers[1], t, TR.emit_simdgroup_combine(b, v, simdgroups=simdgroups, operation=operation))
    b.ret(); ir.verify(f)
    return f


class Combine(unittest.TestCase):
    def test_one_barrier_and_the_simdgroup_index(self):
        names = model.registers()
        for sgs, op in ((2, "sum"), (4, "max")):
            p = cc.compile_function(build(sgs, op))
            ins = [i for i in model.decode(p.code, 0) if i.opcode]
            self.assertEqual(sum(i.opcode.id == 447 for i in ins), 1)
            srs = {names.get(v) for i in ins if i.opcode.id in (14059, 14060) for k, v in i.values if k == "reg"}
            self.assertIn("SR_SIMD_GRP", srs)
            self.assertIn(133, tuple(p.abi()["system_registers"]))

    def test_the_fold_is_ascending_and_fixed(self):
        f = build(4, "sum")
        loads = [o for blk in f.blocks for o in blk.ops if o.kind == "load_tg"]
        self.assertEqual([o.dest.name for o in loads], ["combine_part%d" % s for s in range(4)])

    def test_unmeasured_widths_refuse(self):
        f = ir.Function("x", [ir.Buffer("X", 1, elem=ir.F32)]); b = ir.Builder(f, f.block("entry"))
        with self.assertRaises(TR.UnsupportedReduction):
            TR.emit_simdgroup_combine(b, b.const(0), simdgroups=3)


if __name__ == "__main__":
    unittest.main()
