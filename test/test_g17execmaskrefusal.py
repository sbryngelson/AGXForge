#!/usr/bin/env python3
"""A bare exec_mask emits op582 with no compare in front, so its mask is empty whatever the
condition (Set C round 4: the guarded store was dropped even under an always-true condition).
cc refuses it; the branch-region form emits the measured cmp -> op582 -> region -> op577."""
import os
import struct
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from agxforge.g17 import cc, ir, model  # noqa: E402


def _fn():
    o = ir.Buffer("out", 0, elem=ir.I32)
    f = ir.Function("guarded", [o])
    return f, o, ir.Builder(f, f.block("entry"))


class ExecMask(unittest.TestCase):
    def test_a_bare_exec_mask_refuses(self):
        f, o, b = _fn()
        t = b.builtin("thread_position_in_grid", name="t")
        b.exec_mask(b.icmp(t, b.const(0, name="z"), rel="eq", name="p"))
        b.store_at(o, t, t)
        b.exec_restore()
        b.ret()
        with self.assertRaisesRegex(cc.Unsupported, "no compare in front"):
            cc.compile_function(f)

    def test_the_branch_region_emits_cmp_push_region_pop(self):
        f, o, b = _fn()
        t = b.builtin("thread_position_in_grid", name="t")
        region, join = f.block("tail"), f.block("join")
        b.br_cond(b.cmp(t, 1, "lt", name="p"), region, join)
        r = ir.Builder(f, region)
        r.store_at(o, t, t)
        r.br(join)
        ir.Builder(f, join).ret()
        ops = [i.opcode.id for i in model.decode(cc.compile_function(f).code, 0) if i.opcode]
        self.assertIn(582, ops)
        self.assertIn(577, ops)
        push, pop = ops.index(582), ops.index(577)
        self.assertIn(10369, ops[:push])                        # the flag's writer precedes the push
        self.assertTrue(any(o in ops[push:pop] for o in (17229, 17244, 17235)))   # the store is inside

    def test_the_generic_split_guard_uses_the_branch_form(self):
        import g17tensorcommonruntime as R
        s = R.generic_spec(dict(M=64, N=32, K=64, simdgroups=2))
        ops = [i.opcode.id for i in model.decode(R.build_generic_program(s).code, 0) if i.opcode]
        self.assertNotIn(11372, ops[ops.index(582) - 3:ops.index(582)])
        self.assertIn(10369, ops[:ops.index(582)])


if __name__ == "__main__":
    unittest.main()
