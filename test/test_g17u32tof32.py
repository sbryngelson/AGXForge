"""Unsigned 32-bit integer to FP32: op11179, from a witnessed template, writing two registers only.

WHAT IS MEASURED AND WHAT IS NOT. The form is Apple's and heavily witnessed - 2,641 instances
across 126 of the 6,594 corpus programs - and a bit-role sweep of all eighty bits through Apple's
decoder puts operand 0 at exactly ALU_DEST and operand 4 at a source map that is NEITHER of the two
the float unaries use. Those two fields are what this compiler writes.

Operands 1, 2, 3 and 5 are NOT written. Operand 1's fifteen bits are locatable by the same sweep,
but what the field MEANS is unmeasured, and the value 32 appearing there is not evidence that it is
the KEEP the lifetime family writes elsewhere. Operand 5 is two located bits with equally unmeasured
semantics. Neither is written by this compiler: the cases below check that nothing it does disturbs
them - including the hazard write, which would clear byte4[3] that the witness has SET.

THE EMITTED PROGRAM DOES NOT CARRY THE WITNESS'S OPERAND 5, AND THAT IS NOT A LEAK. Every form this
compiler emits takes its unwritten bits from isa/g17-form-constants.toml, whose value is, per role
group, the MODAL JOINT PATTERN over that form's instances in the build cache - not from whichever
single instruction was used to locate the fields. For op11179/10 the registry and the witness differ
in exactly two unwritten bits, b1[7] and b8[5], which are exactly operand 5: the registry carries
the modal 16 (5,667 of 7,903 cache instances) where the witness carries 32. Both values are shipped.
The difference is bounded and named by test_the_registry_and_the_witness_differ_only_in_operand_5,
and it is NOT read here as a lifetime: if operand 5 turns out to be the 32-keep/16-release operand
the twelve-byte ALU carries, the release is safe only because this lowering REFUSES a source with a
later reader - which is a refusal, not a measurement of this field.

TWO POPULATIONS, BOTH NAMED. 2,641 instances across 126 of the 6,594 SHIPPED programs
(isa/g17-corpus-programs.jsonl); 7,903 instances in the BUILD CACHE, which is what the registry was
built over and the only population its counts describe.

NOTHING HERE IS EXECUTED. The expected values are what the source says the conversion computes,
computed in plain Python; whether the hardware agrees is root's to establish.
"""
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, ROOT)

import g17front
import g17packedcheck as D
from agxforge.g17 import asm, cc, ir

OPCODE = asm.CVT_I2F_OPCODE
WITNESS_BYTES = bytes.fromhex("378000022a80af120800")
PINNED = ("imm:32", "imm:4", "imm:0", "imm:32")          # operands 1, 2, 3, 5 as WITNESSED
# ... and as the form-constant REGISTRY carries them, which is what an emitted program gets. The
# only disagreement is operand 5, and it is asserted as a difference below rather than glossed.
REGISTRY_PINNED = ("imm:32", "imm:4", "imm:0", "imm:16")

SOURCE = """#include <metal_stdlib>
using namespace metal;
kernel void k(device uint *u [[buffer(0)]], device float *f [[buffer(1)]],
              uint3 t [[thread_position_in_grid]]) {
  f[0] = (float)u[t.x];
}
"""


def apple_decode(b):
    """Apple's own decoder on one instruction, so the check is not this compiler's opinion."""
    import g17fields as FL
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as fh:
        fh.write(b)
        path = fh.name
    out = subprocess.run([FL.DIS, path, "0", str(len(b)), "--pc", "0", "--expr"],
                         capture_output=True, text=True).stdout
    for line in out.splitlines():
        f = line.split()
        if len(f) > 2 and f[2] == str(OPCODE):
            return f[3:9]
    return None


def compile_source(src=SOURCE):
    with tempfile.NamedTemporaryFile("w", suffix=".metal", delete=False) as fh:
        fh.write(src)
        path = fh.name
    return cc.compile_function(g17front.from_metal(path))


def reference(value):
    """INDEPENDENT ARITHMETIC: the unsigned interpretation, in plain Python."""
    import struct
    assert 0 <= value <= 0xFFFFFFFF
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


class TheFormIsTheWitnessedOne(unittest.TestCase):
    def test_the_template_is_the_recorded_witness(self):
        self.assertEqual(asm.CVT_I2F_TEMPLATE, WITNESS_BYTES)
        host, offset, note = asm.CVT_I2F_WITNESS
        self.assertIn("ds_setup_indirect_update_mapping", host)
        self.assertEqual(offset, 0x152)
        self.assertIn("unmeasured", note)

    def test_apple_reads_the_template_as_this_opcode(self):
        got = apple_decode(asm.CVT_I2F_TEMPLATE)
        self.assertIsNotNone(got, "Apple's decoder does not read the template as op11179")
        self.assertEqual((got[1], got[2], got[3], got[5]), PINNED)

    def test_no_keep_bit_is_registered_for_it(self):
        """So no lifetime request can reach bits whose meaning is unmeasured."""
        self.assertIsNone(asm.UNARY_KEEP[OPCODE])
        self.assertEqual(asm.UNARY_FORM[OPCODE][0], "cvt.i2f")
        self.assertEqual(asm.UNARY_FORM[OPCODE][1], 10)

    def test_the_source_map_is_not_the_float_unary_one(self):
        """Assuming the family would have written the wrong operand."""
        self.assertNotEqual(asm.UNARY_FORM[OPCODE][2], asm._UNARY_SRC10)
        self.assertEqual(len(asm.UNARY_FORM[OPCODE][2]), 8)


class WritingRegistersDisturbsNothingElse(unittest.TestCase):
    """The constraint that bounds this work: the unmeasured operands must come back unchanged."""

    PAIRS = ((0, 0), (1, 2), (7, 33), (31, 64), (63, 100), (110, 105), (127, 127))

    def test_both_registers_round_trip_through_apples_decoder(self):
        for dest, src in self.PAIRS:
            with self.subTest(dest=dest, src=src):
                got = apple_decode(asm.encode_unary(OPCODE, dest, src, asm.CVT_I2F_TEMPLATE))
                self.assertIsNotNone(got)
                self.assertEqual(int(got[0].split(":")[1]) - 105, dest)
                self.assertEqual(int(got[4].split(":")[1]) - 105, src)

    def test_the_unmeasured_operands_are_never_touched(self):
        for dest, src in self.PAIRS:
            with self.subTest(dest=dest, src=src):
                got = apple_decode(asm.encode_unary(OPCODE, dest, src, asm.CVT_I2F_TEMPLATE))
                self.assertEqual((got[1], got[2], got[3], got[5]), PINNED)

    def test_a_hazard_request_would_touch_them_which_is_why_none_is_passed(self):
        """The control for the choice: asking for hazard=0 clears byte4[3], set in the witness."""
        plain = asm.encode_unary(OPCODE, 3, 2, asm.CVT_I2F_TEMPLATE)
        hazarded = asm.encode_unary(OPCODE, 3, 2, asm.CVT_I2F_TEMPLATE, hazard=0)
        self.assertNotEqual(plain, hazarded)
        self.assertEqual(plain[4], asm.CVT_I2F_TEMPLATE[4])
        self.assertNotEqual(hazarded[4], asm.CVT_I2F_TEMPLATE[4])

    def test_the_emitted_program_carries_the_registry_operands(self):
        """What a compiled program actually carries: the registry's constant, operand 5 included."""
        code = bytes(compile_source().code)
        rows = [r for r in D.decode(code) if r[2] == OPCODE]
        self.assertEqual(len(rows), 1, "expected exactly one conversion")
        toks = rows[0][3]
        self.assertEqual((toks[1], toks[2], toks[3], toks[5]), REGISTRY_PINNED)

    def test_the_registry_and_the_witness_differ_only_in_operand_5(self):
        """The bound on the disagreement, stated as bits rather than left to the operand printer.

        If a regenerated registry ever moved operand 1, the hazard bit, or anything else this
        lowering does not write, this goes red - which is the whole reason the difference is
        measured here instead of being absorbed by relaxing the expectation above.
        """
        from agxforge.g17 import const as g17const
        e = g17const.load()[("unary", OPCODE, 10)]
        value, written = e["value"], e["written"]
        self.assertEqual(len(value), 10)
        unwritten = [(i, b) for i in range(10) for b in range(8)
                     if ((WITNESS_BYTES[i] ^ value[i]) >> b) & 1 and not ((written[i] >> b) & 1)]
        self.assertEqual(unwritten, [(1, 7), (8, 5)])
        self.assertIn("operand 5: b1[7] b8[5]", " ".join(e["roles"]))
        self.assertEqual(value[4], WITNESS_BYTES[4], "the hazard byte must survive")

    def test_the_registry_constant_decodes_to_the_registry_operands(self):
        """Apple's decoder on the registry value, so the expectation above is not self-referential."""
        from agxforge.g17 import const as g17const
        value = g17const.load()[("unary", OPCODE, 10)]["value"]
        got = apple_decode(asm.encode_unary(OPCODE, 3, 2, value))
        self.assertIsNotNone(got)
        self.assertEqual((got[1], got[2], got[3], got[5]), REGISTRY_PINNED)


class SourceToIRToCode(unittest.TestCase):
    def test_the_metal_source_compiles_end_to_end(self):
        prog = compile_source()
        self.assertGreater(len(prog.code), 0)
        opcodes = {r[2] for r in D.decode(bytes(prog.code))}
        self.assertIn(OPCODE, opcodes)

    def test_the_front_end_reads_the_air_call(self):
        self.assertIn("air.convert.f.f32.u.i32", g17front.CALLS)
        self.assertEqual(g17front.CALLS["air.convert.f.f32.u.i32"], ("u32_to_f32", 1))

    def test_the_other_conversions_are_still_refused_by_name(self):
        """The inverse and the half-result directions are different instructions."""
        for call in ("air.convert.u.i32.f.f32", "air.convert.f.f16.u.i32",
                     "air.convert.s.i32.f.f16", "air.convert.u.i64.f.f32"):
            with self.subTest(call):
                self.assertNotIn(call, g17front.CALLS)


class TheUnsupportedDomainsRefuse(unittest.TestCase):
    def build(self, *, from_load=True, read_again=False):
        fn = ir.Function("k", buffers=[ir.Buffer("u", 0, "i32"), ir.Buffer("f", 1, "i32")])
        b = ir.Builder(fn, fn.block("entry"))
        u, f = fn.buffers
        t = b.builtin("thread_position_in_grid")
        x = b.load(u, t)
        if not from_load:
            x = b.add(x, ir.Imm(3))
        b.store(f, ir.Imm(0), b.u32_to_f32(x))
        if read_again:
            b.store(f, ir.Imm(1), x)
        b.ret()
        return fn

    def compile(self, fn):
        return cc.emit(cc.Alloc(regs=range(0, 40)).run(cc.select(fn)))[0]

    def test_an_immediate_operand_refuses(self):
        fn = ir.Function("k", buffers=[ir.Buffer("f", 1, "i32")])
        b = ir.Builder(fn, fn.block("entry"))
        with self.assertRaises((cc.Unsupported, ir.IRError)):
            b.u32_to_f32(ir.Imm(7))

    def test_with_isolation_off_a_load_derived_source_refuses(self):
        held = cc._NO_BITWISE_ISOLATION
        cc._NO_BITWISE_ISOLATION = True
        try:
            with self.assertRaisesRegex(cc.Unsupported, "whether op11179 waits is unmeasured"):
                self.compile(self.build(from_load=True))
        finally:
            cc._NO_BITWISE_ISOLATION = held

    def test_with_isolation_off_a_reread_source_refuses(self):
        held = cc._NO_BITWISE_ISOLATION
        cc._NO_BITWISE_ISOLATION = True
        try:
            with self.assertRaisesRegex(cc.Unsupported, "releases its source is unmeasured"):
                self.compile(self.build(from_load=False, read_again=True))
        finally:
            cc._NO_BITWISE_ISOLATION = held

    def test_and_with_isolation_on_both_compile(self):
        """PAIRED: the refusals above are about unmeasured bits, not about the programs."""
        for kw in (dict(from_load=True), dict(from_load=False, read_again=True),
                   dict(from_load=True, read_again=True)):
            with self.subTest(**kw):
                self.assertGreater(len(self.compile(self.build(**kw))), 0)


class TheIndependentReference(unittest.TestCase):
    """Unsigned interpretation and rounding boundaries, as the contract for root's run."""

    def test_the_high_bit_is_unsigned_not_negative(self):
        self.assertEqual(reference(0x80000000), 2147483648.0)
        self.assertEqual(reference(0xFFFFFFFF), 4294967296.0)
        self.assertNotEqual(reference(0x80000000), -2147483648.0)

    def test_the_rounding_boundary_is_stated_not_assumed(self):
        """2^24+1 has no FP32 representation; the reference says what the rounding gives."""
        self.assertEqual(reference(1 << 24), 16777216.0)
        self.assertEqual(reference((1 << 24) + 1), 16777216.0)
        self.assertEqual(reference((1 << 24) + 2), 16777218.0)
        self.assertEqual(reference((1 << 24) + 3), 16777220.0)

    def test_small_values_are_exact(self):
        for v in (0, 1, 2, 7, 255, 65535, (1 << 23)):
            with self.subTest(v=v):
                self.assertEqual(reference(v), float(v))


if __name__ == "__main__":
    unittest.main()
