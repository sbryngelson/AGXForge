"""A loop inside a guarded region must keep the region's lanes masked off until the region closes.

The latch is `pop, cmp, push, back edge` and the exit one more pop. Before the fix the loop entry
pushed nothing, so the first trip's pop removed the REGION's level: the lanes the guard had switched
off came back on and ran everything after the loop inside the region. A masked-off qmv arm stored
threadgroup 0's rows from registers nobody wrote (MM 25.139.9). divgemv, the one recorded kernel of
this shape, passed anyway because nothing but the latch compare follows its loop inside the region.

The check simulates the exec-mask stack over the emitted instructions (op582 pushes one level, op577
pops one) and requires the region's level to hold from its push to its restore: no instruction in
between runs below depth 1. On the unfixed compiler this program reaches depth 0 at the store after
the loop. Top-level loops are the control: their bytes are unchanged by the fix.

Compile-only."""
import os, sys, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))
import g17ir as ir
from agxforge.g17 import cc, model

LIM = 8


def _region_loop(store_after=True):
    """if (t < 8) { x = t; do { acc += B[t*8 + x]; x += 1 } while (x < 8); C[t + 32] = acc }; C[t] = t"""
    f = ir.Function("loopinregion", [ir.Buffer("B", 1), ir.Buffer("C", 2)])
    pre = f.block("pre"); body = f.block("body"); tail = f.block("tail"); ex = f.block("exit")
    b = ir.Builder(f, pre)
    t = b.builtin("thread_position_in_grid", name="t")
    row = b.mul(t, ir.Imm(LIM), name="row")
    x0 = b.add(t, ir.Imm(0), name="x0")
    a0 = b.const(0, name="a0")
    b.br_cond(b.cmp(x0, LIM, "lt", name="g"), body, ex)
    b.at(body)
    x = b.phi(x0, name="x"); acc = b.phi(a0, name="acc")
    v = b.load(f.buffers[0], b.add(row, x, name="ix"), name="v")
    accn = b.add(acc, v, name="accn")
    xn = b.add(x, ir.Imm(1), name="xn")
    ir.Builder.phi_latch(x, xn); ir.Builder.phi_latch(acc, accn)
    b.br_cond(b.cmp(xn, LIM, "lt", name="p"), body, tail)
    b.at(tail)
    if store_after:
        b.store_at(f.buffers[1], b.add(t, ir.Imm(32), name="hi"), accn)
    b.br(ex)
    b.at(ex); b.store_at(f.buffers[1], t, t); b.ret()
    return f


def _depths(code):
    """[(opcode name, mask depth BEFORE it)] in emission order."""
    out, d = [], 0
    for i in model.decode(code, 0):
        name = str(i).split()[2]
        out.append((name, d))
        if name == "op582":
            d += 1
        elif name == "op577":
            d -= 1
    return out


class LoopInRegion(unittest.TestCase):
    def test_the_region_level_holds_across_the_loop(self):
        seq = _depths(cc.compile_function(_region_loop()).code)
        names = [n for n, _ in seq]
        first, last = names.index("op582"), len(names) - 1 - names[::-1].index("op577")
        inside = [(k, n, d) for k, (n, d) in enumerate(seq) if first < k <= last]
        self.assertTrue(inside)
        low = [(k, n, d) for k, n, d in inside if d < 1]
        self.assertEqual(low, [], "instructions inside the guarded region run below its mask level")

    def test_the_loop_pushes_on_entry(self):
        """The entry push is the fix: region push, then the loop's own push, before the loop body."""
        names = [n for n, _ in _depths(cc.compile_function(_region_loop()).code)]
        pushes = [k for k, n in enumerate(names) if n == "op582"]
        back = names.index("op458")
        self.assertGreaterEqual(len([k for k in pushes if k < back]), 3,
                                "region push, loop-entry push and latch push precede the back edge")


if __name__ == "__main__":
    unittest.main()
