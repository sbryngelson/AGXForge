"""op3290/4 reading two loaded sources gets each through a waiting add first.

Measured (results/g17-formreceipt-v1): straight from two loads the 4-byte multiply returned 0 on all
32 lanes in two workers; with each source copied through an add of 0 carrying the wait it was
bit-exact on 192 words in two workers.
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import g17cc
import g17ir as ir


def forms(fn):
    p = g17cc.compile_function(fn())
    return [(m.form, m.fields.get("load_wait")) for _a, _b, m in p.layout]


def loaded():
    f = ir.Function("k", [ir.Buffer("S", 1), ir.Buffer("O", 2)])
    b = ir.Builder(f, f.block("entry"))
    t = b.builtin("thread_position_in_grid", name="t")
    x = b.load(f.buffers[0], t, name="x")
    y = b.load(f.buffers[0], t, offset=32, name="y")
    b.store_at(f.buffers[1], t, b._def("fmul", [x, y], type=ir.F32, name="v", length=4))
    b.ret()
    return f


def computed():
    f = ir.Function("k", [ir.Buffer("S", 1), ir.Buffer("O", 2)])
    b = ir.Builder(f, f.block("entry"))
    t = b.builtin("thread_position_in_grid", name="t")
    x = b.add(b.load(f.buffers[0], t, name="x"), ir.Imm(0), name="xa")
    y = b.add(b.load(f.buffers[0], t, offset=32, name="y"), ir.Imm(0), name="ya")
    b.store_at(f.buffers[1], t, b._def("fmul", [x, y], type=ir.F32, name="v", length=4))
    b.ret()
    return f


class TheFourByteMultiplyWaitsThroughAnAdd(unittest.TestCase):

    def test_each_loaded_source_is_copied_through_a_waiting_add(self):
        f = forms(loaded)
        i = [k for k, (form, _w) in enumerate(f) if form == "alu.fmul.4"][0]
        before = f[:i]
        waits = [w for form, w in before if form == "alu.12"]
        self.assertEqual(waits, [1, 1], f)

    def test_a_source_that_is_not_a_load_is_not_copied_again(self):
        """The control: operands already computed by an add get no second copy."""
        f = forms(computed)
        self.assertEqual(sum(1 for form, _w in f if form == "alu.12"), 2, f)


if __name__ == "__main__":
    unittest.main()
