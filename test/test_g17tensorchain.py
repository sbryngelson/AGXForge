#!/usr/bin/env python3
"""Adjacent GEMM chains of any whole-tile shapes compile through the composition route (item 2)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agxforge.g17 import cc, ir, model


def chain(shapes, scalar_between=False):
    a = ir.Buffer("A", 1, elem=ir.F16); b = ir.Buffer("B", 2, elem=ir.F16); c = ir.Buffer("C", 3, elem=ir.F32)
    fn = ir.Function("chain_probe", [a, b, c]); bl = ir.Builder(fn, fn.block("entry"))
    for n, (M, N, K) in enumerate(shapes):
        if n == 0:
            bl.tensor_matmul(a, b, c, M=M, N=N, K=K)
        else:
            bl.tensor_matmul(c, b, c, M=M, N=N, K=K, a_dtype="float", b_dtype="half")
        if scalar_between and n == 0:
            t = bl.builtin("threadgroup_position_in_grid", name="t")
            bl.store_at(c, t, bl.load(c, t, type=ir.F32, name="v"))
    bl.ret(); ir.verify(fn)
    return fn


def mmas(fn):
    return sum(1 for i in model.decode(cc.compile_function(fn).code, 0) if i.opcode and 5098 <= i.opcode.id <= 5107)


class AdjacentChains(unittest.TestCase):
    def test_mixed_shapes_and_depths_compile(self):
        fn = chain([(32, 64, 64), (32, 16, 64)])
        self.assertTrue(cc._adjacent_tensor_chain(fn, [o for b in fn.blocks for o in b.ops if o.kind == "tensor_matmul"]))
        self.assertEqual(mmas(chain([(32, 64, 64), (32, 16, 64)])), 2 * 4 * 4 + 2 * 1 * 4)
        self.assertEqual(mmas(chain([(16, 32, 32), (16, 48, 32), (16, 16, 48)])), 2 * 2 + 3 * 2 + 1 * 3)
        self.assertEqual(mmas(chain([(16, 16, 16)] * 5)), 5)

    def test_what_is_not_a_chain_keeps_its_old_answer(self):
        fn = chain([(32, 64, 64), (32, 16, 64)], scalar_between=True)
        ops = [o for b in fn.blocks for o in b.ops if o.kind == "tensor_matmul"]
        self.assertFalse(cc._adjacent_tensor_chain(fn, ops))
        fn = chain([(32, 32, 64), (32, 32, 48)])          # K2 != N1
        ops = [o for b in fn.blocks for o in b.ops if o.kind == "tensor_matmul"]
        self.assertFalse(cc._adjacent_tensor_chain(fn, ops))
        with self.assertRaises(Exception):
            cc.compile_function(chain([(17, 16, 16), (17, 16, 16)]))


if __name__ == "__main__":
    unittest.main()
