#!/usr/bin/env python3
"""The binary16 fused multiply-add, op798 (MM 25.196): cc emits Apple's bytes for Apple's operands and names either half
of a 32-bit word as a source; g17emu rounds once, exactly (against rational arithmetic), where the multiply-then-add
control differs; the half-A dequant of g17qsm is bit-exact on g17emu; and the forms refuse what they cannot express.
CPU only."""
import os
import sys
import tempfile
import unittest
from fractions import Fraction

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import numpy as np  # noqa: E402


def _kernel(packed=False):
    from agxforge.g17 import cc, ir
    A = ir.Buffer("A", 1, elem=ir.F32); B = ir.Buffer("B", 2, elem=ir.F32); C = ir.Buffer("C", 3, elem=ir.F32)
    fn = ir.Function("tensor_gemm_generic_runtime_demo", [A, B, C]); b = ir.Builder(fn, fn.block("entry"))
    t = b.builtin("thread_position_in_grid", name="t")
    x = b.load(A, t, name="x"); y = b.load(B, t, name="y")
    hx = b.f32_to_f16_rte(x, name="hx")
    if packed:
        w = b.add(y, ir.Imm(0), name="w")
        r = b.fma16(hx, w, w, halves=("lo", "lo", "hi"), name="r")
    else:
        z = b.load(A, b.add(t, b.const(64, name="nn"), name="tz"), name="z")
        r = b.fma16(hx, b.f32_to_f16_rte(y, name="hy"), b.f32_to_f16_rte(z, name="hz"), name="r")
    b.store_at(C, t, b.f16_to_f32(r, name="rw"))
    b.ret()
    return cc.compile_function(fn)


class Encoding(unittest.TestCase):
    def test_apples_bytes(self):
        import g17as
        from agxforge.g17 import cc
        line = cc._as_line("ffma.f16.l12", 798, 12, {0: "r426", 2: "r427", 4: "r283", 6: "r426"},
                           pinned={1: 2147483648, 3: 16, 5: 16, 7: 16})
        self.assertEqual(g17as.assemble(line).text.hex(), "3800060a2320a40245018000")

    def test_emitted_forms(self):
        import g17emu as E
        plain = [i for i in E.decode(_kernel().code) if i[2] == 798]
        self.assertEqual(len(plain), 1)
        self.assertTrue(all(425 <= int(t[4:]) < 553 for t in plain[0][3][0:8:2]))      # every operand a low half
        (op,) = [i for i in E.decode(_kernel(packed=True).code) if i[2] == 798]
        b_, c_ = int(op[3][4][4:]), int(op[3][6][4:])
        self.assertEqual((b_ - 425, c_ - 281), (b_ - 425, b_ - 425))                   # one word, low and high half

    def test_refusals(self):
        from agxforge.g17 import cc, ir
        fn = ir.Function("f", [ir.Buffer("A", 1, elem=ir.F32)]); b = ir.Builder(fn, fn.block("entry"))
        h = b.f32_to_f16_rte(b.const(0x3F800000, name="one"), name="h")
        with self.assertRaises(ir.IRError):
            b.fma16(h, h, h, halves=("hi", "lo", "lo"))                 # an I16 value is a low half
        with self.assertRaises(ir.IRError):
            b.fma16(h, h, ir.Imm(1))
        A = ir.Buffer("A", 1, elem=ir.F32); C = ir.Buffer("C", 3, elem=ir.F32)
        fn = ir.Function("tensor_gemm_generic_runtime_demo", [A, C]); b = ir.Builder(fn, fn.block("entry"))
        t = b.builtin("thread_position_in_grid", name="t")
        w = b.load(A, t, name="w")
        b.store_at(C, t, b.f16_to_f32(b.fma16(w, w, w, halves=("lo", "lo", "hi")), name="o"))
        b.ret()
        # a loaded source is copied through a waited alu.12 first (op798's own wait field is not located)
        ops = [i for i in __import__("g17emu").decode(cc.compile_function(fn).code)]
        self.assertEqual([o[2] for o in ops].count(10279), 1)


class Semantics(unittest.TestCase):
    def test_one_exact_rounding(self):
        import g17emu as E
        rng = np.random.default_rng(3)
        h = lambda n: (rng.standard_normal(n) * 2.0 ** rng.integers(-10, 8, n)).astype(np.float16).astype(np.float64)
        a, b, c = h(3000), h(3000), h(3000)
        got = E.fma16_exact(a, b, c)
        for i in range(0, 3000, 3):
            ex = Fraction(a[i]) * Fraction(b[i]) + Fraction(c[i])
            f = np.float16(float(ex))
            cands = [np.nextafter(f, np.float16(-np.inf)), f, np.nextafter(f, np.float16(np.inf))]
            best = min(cands, key=lambda v: (abs(Fraction(float(v)) - ex), int(np.array(v).view(np.uint16)) & 1))
            if float(best) == 0.0:
                continue
            self.assertEqual(int(np.array(best).view(np.uint16)), int(got[i]))
        two = ((a.astype(np.float32) * b).astype(np.float16).astype(np.float32) + c).astype(np.float16).view(np.uint16)
        self.assertGreater(int((two != got).sum()), 100)                  # the control can fail

    def test_emulated_kernel(self):
        import g17deliver as DL
        import g17emu as E
        rng = np.random.default_rng(4)
        a, b, c = ((rng.standard_normal(64) * 3).astype(np.float16) for _ in range(3))
        for packed in (False, True):
            with self.subTest(packed=packed), tempfile.TemporaryDirectory() as t:
                A = np.concatenate([a, c]).astype(np.float32).tobytes()
                Bv = ((b.view(np.uint16).astype(np.uint32) | (c.view(np.uint16).astype(np.uint32) << 16)).astype("<u4")
                      .tobytes() if packed else b.astype(np.float32).tobytes())
                d = DL.author(os.path.join(t, "k"), _kernel(packed), A, Bv, b"\xa5" * 256, dict(n=64))
                out = E.run_bundle(d, 64, 32, 1, tier="wp")
                out = out[0] if isinstance(out, tuple) else out
                got = np.frombuffer(out, "<f4", 64).astype(np.float16).view(np.uint16)
                want = E.fma16_exact(a.astype(np.float64), b.astype(np.float64), c.astype(np.float64))
                self.assertEqual(int((got != want).sum()), 0)


class HalfDequant(unittest.TestCase):
    def test_qsm_h16_bit_exact(self):
        import g17emu as E
        import g17qsm as Q
        lay = Q.layout(64, 512, 2, 4, 1, xrows=True, h16=True)
        prog = Q.build(lay)
        ops = [i[2] for i in E.decode(prog.code)]
        # no fp32 dequant: no convert or fp32 fma; op1016 only for the four row halves' scale and bias
        self.assertEqual((ops.count(2190), ops.count(11179), ops.count(1016), ops.count(798)), (0, 0, 8, 128))
        x, packed, s16, b16, q = Q.case(lay)
        want = Q.reference(lay, x, q, s16, b16)
        a, bb, c = Q.io(lay, x, packed, s16, b16)
        with tempfile.TemporaryDirectory() as t:
            d = Q.author(os.path.join(t, "q"), prog, a, bb, c, lay)
            out = E.run_bundle(d, lay["groups"] * 32, 32, 1, tier="wp")
        out = out[0] if isinstance(out, tuple) else out
        got = np.frombuffer(out, "<f4", want.size).reshape(want.shape)
        self.assertEqual(int((got.view(np.uint32) != want.view(np.uint32)).sum()), 0)
        base = Q.reference(Q.layout(64, 512, 2, 4, 1, xrows=True), x, q, s16, b16)
        self.assertEqual(int((base.view(np.uint32) != want.view(np.uint32)).sum()), 0)

    def test_a_half_accumulator_is_read_only_as_half(self):
        """The h16 program with its half accumulator read as an fp32 A (two fp32 words' halves as eight halves)."""
        import unittest.mock as mock
        from agxforge.g17 import cc, ir
        import g17qsm as Q
        real = ir.Builder.tensor_matmul

        def as_float(self, *args, **kw):
            if kw.get("a_acc"):
                kw["a_dtype"] = "float"
            return real(self, *args, **kw)
        with mock.patch.object(ir.Builder, "tensor_matmul", as_float), self.assertRaises(cc.Unsupported) as caught:
            Q.build(Q.layout(64, 512, 2, 4, 1, xrows=True, h16=True))
        self.assertIn("half accumulator", str(caught.exception))

if __name__ == "__main__":
    unittest.main()
