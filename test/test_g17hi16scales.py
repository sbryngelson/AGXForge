"""bf16 scales and biases loaded into HIGH HALVES (build_qmv2 lay["hi16_scales"], ir.load_hi16; MM 25.141.3 item 3).

Apple's compile of MLX's qmv loads each bf16 scale with op12646 and byte4[2] (hi16) set, whose destination decodes
as a HIGH-HALF register (R55H): the load writes the high 16 bits and leaves the low 16. A register whose low half
is zero then holds the fp32 of the bf16 with no shift. Our form was a half load into a low half plus op14391
(shl 16), one per scale and bias per trip.

With the flag, each row's scale and bias is a loop phi: its register is zeroed whole once before the loop, every
trip's load is TIED to it (the destination is the phi's register), and nothing else in the loop writes it. Checked
on the emitted bytes, compile-only:

- the loop has no op14391 left, and it differs from the unflagged loop by exactly those shifts (no copy added);
- every op12646 in the loop carries hi16 and decodes to an RnH destination;
- no other loop instruction writes either half of that register, the loop's readers keep it (it is carried), and
  the last write before the loop is a whole-register immediate zero;
- the loop carries no register released before it (test_g17phientry's check);
- the flag defaults off, so delivered bytes are unchanged."""
import collections, os, sys, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))
from test_g17phientry import released_before_loop

BASE = dict(interleave=True, lean=True, coalesced=True, a16=True, xvec=True, wpt=2, vload=True, hoist_consts=True)


def _forms():
    import g17qmv as Q
    return {
        "q4 w2 split-K r1 S4": lambda h: Q.with_residual(dict(Q.case(2048, 8192, 4, 1, nocarrier=True)[0], **BASE, sgs=4,
                                                               ksplit=True, coop=True, hi16_scales=h), "add32_to16"),
        "q4 qkv r4": lambda h: dict(Q.case(4096, 2048, 4, 4, nocarrier=True)[0], **BASE, hi16_scales=h),
        "q4 ffn split-K r1 S2": lambda h: dict(Q.qmv_swiglu_layout(8192, 2048, 4, 1), **BASE, act32=True, sgs=2,
                                               ksplit=True, coop=True, hi16_scales=h),
    }


def _loop(code):
    from agxforge.g17 import model, tensorview as TV
    v = TV.view(code)
    loops = TV.loops(v)
    assert len(loops) == 1, loops
    first, back = loops[0]
    return v, list(model.decode(code, 0)), first, back


def _op(ins):
    return str(ins.opcode).split("(")[0]


class HighHalfScales(unittest.TestCase):
    def test_the_shifts_leave_and_nothing_replaces_them(self):
        import g17qmv as Q
        for name, mk in _forms().items():
            counts = []
            for flag in (False, True):
                _, ins, first, back = _loop(Q.build_qmv2(mk(flag)).code)
                counts.append(collections.Counter(_op(i) for i in ins[first:back + 1]))
            before, after = counts
            shifts = before["op14391"]
            self.assertGreater(shifts, 0, name)
            self.assertEqual(after["op14391"], 0, name)
            expect = before.copy()
            del expect["op14391"]
            self.assertEqual(+after, +expect, "%s: the loop must differ only by the removed shifts" % name)

    def test_each_load_writes_a_high_half_that_only_it_writes(self):
        import g17qmv as Q
        from agxforge.g17 import model
        names = model.registers()
        for name, mk in _forms().items():
            v, ins, first, back = _loop(Q.build_qmv2(mk(True)).code)
            loads = [n for n in range(first, back + 1) if _op(ins[n]) == "op12646"]
            self.assertTrue(loads, name)
            for n in loads:
                raw = bytes(ins[n].raw)
                self.assertEqual((raw[4] >> 2) & 1, 1, "%s: op12646 at %d without hi16" % (name, n))
                dest = names.get(ins[n].values[0][1])
                self.assertTrue(dest.startswith("R") and dest.endswith("H"), "%s: destination %s" % (name, dest))
                reg = int(dest[1:-1])
                halves = {2 * reg, 2 * reg + 1}
                others = [m for m in range(first, back + 1) if m != n and halves & set(v[m].defs)]
                self.assertEqual(others, [], "%s: %s is also written in the loop" % (name, dest))
                pre = [m for m in range(first) if halves & set(v[m].defs)]
                self.assertTrue(pre, "%s: %s has no value on loop entry" % (name, dest))
                last = pre[-1]
                self.assertEqual(set(v[last].defs) & halves, halves, "%s: entry write is not whole" % name)
                self.assertEqual(ins[last].values[-1], ("imm", 0), "%s: entry write is not zero" % name)

    def test_the_carried_registers_survive_the_entry(self):
        import g17qmv as Q
        for name, mk in _forms().items():
            self.assertEqual(released_before_loop(Q.build_qmv2(mk(True)).code), [], name)

    def test_the_flag_defaults_off(self):
        import g17qmv as Q
        for name, mk in _forms().items():
            lay = mk(False)
            lay.pop("hi16_scales")
            self.assertEqual(Q.build_qmv2(lay).code, Q.build_qmv2(mk(False)).code, name)

    def test_the_ir_refuses_a_non_buffer(self):
        from agxforge.g17 import ir
        c = ir.Buffer("C", 3, elem=ir.F32)
        fn = ir.Function("k", [c])
        b = ir.Builder(fn, fn.block("entry"))
        t = b.builtin("thread_position_in_grid", name="t")
        with self.assertRaises(ir.IRError):
            b.load_hi16(t, t, t)


if __name__ == "__main__":
    unittest.main()
