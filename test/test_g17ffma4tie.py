"""op2190/4 outside an accumulator loop: the tie, the accumulator copy, the load wait, the receipt.

op2190/4 encodes d = a*b + d: the accumulator IS the destination. cc refused it everywhere but a
phi-coalesced loop until the allocator could be asked for one register (TIED_FORMS). Executing
it then showed a second defect nobody had met, because /4 had never run: fed straight from loads
it returned 0, so loaded sources now pass through the measured waiting copy.
"""
import json, os, sys, unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))
import g17cc, g17ir as ir, g17packedcheck as D
from agxforge.g17 import cc

END = bytes.fromhex("0e000000")


def build(acc_read_later=False, from_loads=True):
    f = ir.Function("t", [ir.Buffer("A", 1), ir.Buffer("C", 2)])
    b = ir.Builder(f, f.block("entry"))
    t = b.builtin("thread_position_in_grid", name="t")
    base = b.mul(t, ir.Imm(4), name="base")
    v = [b.load(f.buffers[0], base, offset=k, name="l%d" % k) for k in range(3)]
    if not from_loads:
        v = [b.add(x, ir.Imm(0), name="c%d" % k) for k, x in enumerate(v)]
    r = b._def("fma", v, type=ir.F32, name="r", length=4)
    if acc_read_later:
        r = b._def("fadd", [r, v[2]], type=ir.F32, name="s")
    b.store_at(f.buffers[1], t, r)
    b.ret()
    return g17cc.compile_function(f)


def forms(p):
    return [(m.form, bytes(raw)) for off, raw, m in p.layout]


def fma(p):
    raw = [r for f, r in forms(p) if f == "alu.ffma.4"]
    assert len(raw) == 1
    return D.decode(raw[0] + END)[0][3]


class TheTie(unittest.TestCase):

    def test_straight_line_compiles_with_the_accumulator_as_destination(self):
        toks = fma(build(from_loads=False))
        self.assertEqual(toks[0], toks[6])

    def test_without_the_tie_the_post_allocation_check_still_refuses(self):
        saved = cc.TIED_FORMS
        try:
            cc.TIED_FORMS = ()
            with self.assertRaisesRegex(Exception, "two-address"):
                build(from_loads=False)
        finally:
            cc.TIED_FORMS = saved

    def test_an_accumulator_read_again_is_copied_not_overwritten(self):
        p = build(acc_read_later=True, from_loads=False)
        moves = [D.decode(r + END)[0][3] for f, r in forms(p) if f == "mov.4"]
        toks = fma(p)
        self.assertTrue(any(m[0] == toks[6] and m[2] != toks[0] for m in moves),
                        "no copy into the tied register from a different one")


class TheLoadWait(unittest.TestCase):

    def test_loaded_sources_pass_through_a_waiting_copy(self):
        p = build(from_loads=True)
        seq = [f for f, r in forms(p)]
        i = seq.index("alu.ffma.4")
        self.assertGreaterEqual(seq[:i].count("alu.12"), 3)


class TheReceipt(unittest.TestCase):

    def test_both_widths_round_once_on_hardware(self):
        d = json.load(open(os.path.join(ROOT, "isa", "g17-execution-ffma4-tie-results.json")))
        for k in ("four_byte", "default_width"):
            with self.subTest(width=k):
                r = d[k]
                self.assertTrue(r["all_fused"])
                self.assertEqual(r["rounding_witness_lanes_unfused"], 0)
                witnesses = [x for x in r["lanes"] if x["want_fused"] != x["want_unfused"]]
                self.assertEqual(len(witnesses), 16)


if __name__ == "__main__":
    unittest.main()
