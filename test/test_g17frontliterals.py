"""Ordinary FP32 literals in AIR: the three forms Apple emits, and the ones this front end refuses.

The file is named to fall inside root's acceptance glob (`test_g17front*.py`) and keeps its subject
separate from the buffer-type and builtin front-end suites.

WHAT THE POPULATION ACTUALLY CONTAINS, measured over root's retained AIR for all sixteen literal
candidates (results/g17-source-admission-v2/construct-ranking.json): 195 exponent tokens, 170 LLVM
hex tokens, 16 plain decimals. So exponent notation is the common case, not the exotic one.

THE HEX FORM IS A DOUBLE BIT PATTERN. LLVM prints a `float` constant in hex as the sixty-four-bit
double whose value the float equals exactly, so `0x3FF003AFC0000000` is 1.0009000301361084 - reading
those bits as a binary32 would be the numeric-conversion-versus-bit-reinterpretation confusion this
campaign was warned about, and the case below pins the difference rather than trusting a comment.

THE NEGATIVE CONTROL IS THE TRUNCATED TOKEN. The call parser's character class `[\\w.-]` admitted the
minus of a negative exponent and not the plus of a positive one, so `5.000000e-01` arrived whole and
`1.000000e+00` arrived as `1.000000e`. That is the worst kind of truncation: at exponent +00 the
dropped text changes nothing, so a lenient parser reads 1.0 and is RIGHT BY ACCIDENT - and then reads
`1.000000e+03` as 1.0, which syn-s47f3984d60 actually contains. `1.000000e` must refuse.

Nothing here dispatches.
"""
import os
import re
import struct
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, ROOT)

import g17cc
import g17front
from agxforge.g17 import ir

bits_of = lambda f: struct.unpack("<I", struct.pack("<f", f))[0]
as_f32 = lambda b: struct.unpack("<f", struct.pack("<I", b))[0]
HEAD = "#include <metal_stdlib>\nusing namespace metal;\n"


def compile_source(text):
    with tempfile.NamedTemporaryFile("w", suffix=".metal", delete=False) as fh:
        fh.write(text)
        path = fh.name
    try:
        return g17front.from_metal(path)
    finally:
        os.unlink(path)


class TheThreeFormsApplesAIRUses(unittest.TestCase):
    def test_a_plain_decimal(self):
        self.assertEqual(g17front.f32_literal_bits("1.25"), bits_of(1.25))

    def test_a_positive_exponent(self):
        self.assertEqual(g17front.f32_literal_bits("1.000000e+03"), bits_of(1000.0))

    def test_a_negative_exponent(self):
        self.assertEqual(g17front.f32_literal_bits("5.000000e-01"), bits_of(0.5))

    def test_an_llvm_hex_constant_is_the_DOUBLE_bit_pattern(self):
        got = g17front.f32_literal_bits("0x3FF003AFC0000000")
        self.assertEqual(got, bits_of(1.0009000301361084))
        self.assertEqual(got, 0x3F801D7E)

    def test_reading_the_hex_as_binary32_bits_would_give_something_else(self):
        """The confusion, pinned: the low 32 bits of that double are not the float's bits."""
        low32 = 0x3FF003AFC0000000 & 0xFFFFFFFF
        high32 = 0x3FF003AFC0000000 >> 32
        correct = g17front.f32_literal_bits("0x3FF003AFC0000000")
        self.assertNotEqual(correct, low32)
        self.assertNotEqual(correct, high32)
        self.assertEqual(as_f32(correct), 1.0009000301361084)

    def test_an_integer_is_still_an_integer(self):
        """Integer parsing is untouched: it must not become a float constant."""
        self.assertIsNone(g17front.f32_literal_bits("7"))
        self.assertIsNone(g17front.f32_literal_bits("-7"))

    def test_a_name_is_not_a_literal(self):
        self.assertIsNone(g17front.f32_literal_bits("%12"))


class SignedZeroSurvives(unittest.TestCase):
    """The sign of a zero is observable in a stored word, so it is not collapsed."""

    def test_negative_zero_keeps_its_sign_bit(self):
        self.assertEqual(g17front.f32_literal_bits("-0.0"), 0x80000000)

    def test_positive_zero_is_distinct_from_it(self):
        self.assertEqual(g17front.f32_literal_bits("0.0"), 0x00000000)
        self.assertNotEqual(g17front.f32_literal_bits("-0.0"),
                            g17front.f32_literal_bits("0.0"))

    def test_the_hex_spelling_of_negative_zero_agrees(self):
        self.assertEqual(g17front.f32_literal_bits("0x8000000000000000"), 0x80000000)


class TheParserNegativeControl(unittest.TestCase):
    """Root's required negative control, and the asymmetry that made it necessary."""

    def test_the_truncated_exponent_token_IS_NOT_A_LITERAL(self):
        """It does not parse, and the refusal lands in the caller rather than the parser.

        The first version of this case asserted `f32_literal_bits` RAISES, and it does not - it
        returns None, meaning "this token is not a float literal", and `operand()` then refuses it
        by name. Both are refusals; the one that matters is that `1.000000e` NEVER becomes 1.0, so
        that is what is asserted, at the parser and at the front end.
        """
        self.assertIsNone(g17front.f32_literal_bits("1.000000e"))
        self.assertNotEqual(g17front.f32_literal_bits("1.000000e+00"), None)

    def test_real_AIR_refuses_by_the_truncated_NAME_with_the_capability_off(self):
        """On Apple's own AIR, not a hand-written stub - the first stub was not even a valid kernel.

        `fma(x, 1000.0f, x)` puts `float 1.000000e+03` in a call argument. With the capability off
        the old character class cuts it at the `+`, and the refusal names `1.000000e` - which is
        verbatim the first refusal root's retained record carries for sl32-u148. Switching the
        capability off is how that historical refusal stays reproducible.
        """
        text = (HEAD + "kernel void k(device float *b [[buffer(0)]],\n"
                       "              uint3 t [[thread_position_in_grid]])"
                       " { b[t.x] = fma(b[t.x], 1000.0f, b[t.x]); }\n")
        g17front._NO_FP32_LITERALS = True
        try:
            with self.assertRaises(g17front.Unsupported) as cm:
                compile_source(text)
        finally:
            g17front._NO_FP32_LITERALS = False
        self.assertIn("1.000000e", str(cm.exception))
        self.assertNotIn("1.000000e+03", str(cm.exception), "the token arrived truncated")
        # and with it on, the same source reads and carries the RIGHT value
        fn = compile_source(text)
        vals = {a.v for blk in fn.blocks for o in blk.ops if o.kind == "const"
                for a in o.args if isinstance(a, ir.Imm)}
        self.assertIn(bits_of(1000.0), vals)
        self.assertNotIn(bits_of(1.0), vals, "1000.0 must not have become 1.0")

    def test_the_old_character_class_truncated_a_positive_exponent_only(self):
        """Reproduced directly, because this is why the control exists."""
        old, new = r"(?:float)\s+(%?[\w.-]+)", r"(?:float)\s+(%?[\w.+-]+)"
        line = "call float @air.fma.f32(float %5, float 1.000000e+03, float %7)"
        self.assertEqual(re.findall(old, line)[1], "1.000000e")
        self.assertEqual(re.findall(new, line)[1], "1.000000e+03")
        negative = "call float @f(float 5.000000e-01)"
        self.assertEqual(re.findall(old, negative)[0], "5.000000e-01")

    def test_a_lenient_parser_would_have_been_right_by_ACCIDENT_at_e00(self):
        """1.0 either way - which is how a 1000x error hides behind a passing case."""
        self.assertEqual(float("1.000000e+00"), 1.0)
        self.assertEqual(g17front.f32_literal_bits("1.000000e+00"), bits_of(1.0))
        self.assertNotEqual(g17front.f32_literal_bits("1.000000e+03"), bits_of(1.0))

    def test_malformed_literals_refuse_or_are_not_literals(self):
        for tok in ("1.0.0", "e+00", "1.0e", "0x123", "--1.0", "1.0f"):
            with self.subTest(tok=tok):
                try:
                    got = g17front.f32_literal_bits(tok)
                except g17front.Unsupported:
                    continue
                self.assertIsNone(got, "%r must not parse as a float" % tok)


class TheWidthsAndValuesItRefuses(unittest.TestCase):
    def test_llvms_other_floating_point_widths_refuse_by_name(self):
        for tok, name in (("0xH3C00", "half"), ("0xK401E000000000000000", "x86_fp80"),
                          ("0xL00000000000000003FFF000000000000", "fp128"),
                          ("0xM400C000000000000", "ppc_fp128")):
            with self.subTest(tok=tok):
                with self.assertRaises(g17front.Unsupported) as cm:
                    g17front.f32_literal_bits(tok)
                self.assertIn(name, str(cm.exception))

    def test_a_non_finite_hex_constant_refuses(self):
        for tok in ("0x7FF0000000000000", "0xFFF0000000000000", "0x7FF8000000000000"):
            with self.subTest(tok=tok):
                with self.assertRaises(g17front.Unsupported) as cm:
                    g17front.f32_literal_bits(tok)
                self.assertIn("not finite", str(cm.exception))

    def test_a_hex_constant_that_is_not_exactly_a_float_refuses(self):
        """LLVM uses the hex form when the decimal is inexact, so a double that is NOT a float
        exactly is not an f32 constant - reinterpreting it would be a guess."""
        inexact = struct.unpack("<Q", struct.pack("<d", 0.1))[0]
        self.assertNotEqual(struct.unpack("<f", struct.pack("<f", 0.1))[0], 0.1)
        with self.assertRaises(g17front.Unsupported) as cm:
            g17front.f32_literal_bits("0x%016X" % inexact)
        self.assertIn("as binary32", str(cm.exception))

    def test_every_hex_constant_in_the_RETAINED_population_is_exactly_a_float(self):
        """The guard accepts ground truth: 150 of 150, so it is a boundary and not a wall."""
        import json
        path = os.path.join(ROOT, "results", "g17-source-admission-v2", "construct-ranking.json")
        if not os.path.isfile(path):
            self.skipTest("root's retained ranking is not extracted in this checkout")
        r = json.load(open(path))
        hexes = {h for c in r["candidates"] for h in re.findall(r"0x[0-9A-F]{16}", c["air_text"])}
        self.assertGreaterEqual(len(hexes), 150)
        for h in sorted(hexes):
            self.assertIsInstance(g17front.f32_literal_bits(h), int)


class OutOfRangeLiteralsRefuseRatherThanCRASH(unittest.TestCase):
    """Root's review finding: `1.0e+40` and `0x7FEFFFFFFFFFFFFF` raised OverflowError out of
    struct.pack - a crash, not a refusal. Both forms are controlled here, and the same check turned
    up a third case root did not name: a nonzero literal narrowing SILENTLY to zero."""

    OVERFLOW_DEC = ("1.0e+40", "-1.0e+40", "3.5e38", "1e308")
    # 0x47EFFFFFE0000000 is NOT here on purpose: it is exactly FLT_MAX, and my first draft listed
    # it as an overflow. The code was right and the case was wrong. 0x47F0000000000000 is 2**128,
    # the first power of two past the range, and 0x47EFFFFFE0000001 is FLT_MAX by one double ulp.
    OVERFLOW_HEX = ("0x7FEFFFFFFFFFFFFF", "0xFFEFFFFFFFFFFFFF", "0x47F0000000000000",
                    "0x47EFFFFFE0000001")

    def test_a_decimal_beyond_binary32_refuses_by_name(self):
        for tok in self.OVERFLOW_DEC:
            with self.subTest(tok=tok):
                with self.assertRaises(g17front.Unsupported) as cm:
                    g17front.f32_literal_bits(tok)
                self.assertIn("outside binary32", str(cm.exception))

    def test_a_hex_double_beyond_binary32_refuses_by_name(self):
        for tok in self.OVERFLOW_HEX:
            with self.subTest(tok=tok):
                with self.assertRaises(g17front.Unsupported) as cm:
                    g17front.f32_literal_bits(tok)
                self.assertIn("outside binary32", str(cm.exception))

    def test_neither_form_raises_OverflowError_any_more(self):
        """The control for the finding as root stated it: the exception TYPE changed."""
        for tok in self.OVERFLOW_DEC + self.OVERFLOW_HEX:
            with self.subTest(tok=tok):
                try:
                    g17front.f32_literal_bits(tok)
                except g17front.Unsupported:
                    pass
                except OverflowError as ex:
                    self.fail("%r still raises OverflowError: %s" % (tok, ex))

    def test_a_nonzero_literal_that_narrows_to_zero_refuses(self):
        """Not named in the review, and the quieter half: 1.0e-60 became 0x00000000."""
        for tok in ("1.0e-60", "-1.0e-60", "1e-300"):
            with self.subTest(tok=tok):
                with self.assertRaises(g17front.Unsupported) as cm:
                    g17front.f32_literal_bits(tok)
                self.assertIn("narrows to exactly zero", str(cm.exception))

    def test_a_subnormal_refuses_because_denormals_are_opcode_dependent_here(self):
        with self.assertRaises(g17front.Unsupported) as cm:
            g17front.f32_literal_bits("1.0e-44")
        self.assertIn("SUBNORMAL", str(cm.exception))

    def test_the_boundary_values_themselves_still_convert(self):
        """A guard must first accept ground truth: the largest finite and smallest normal do."""
        largest = struct.unpack("<f", struct.pack("<I", 0x7F7FFFFF))[0]
        smallest = struct.unpack("<f", struct.pack("<I", 0x00800000))[0]
        self.assertEqual(g17front.f32_literal_bits(repr(largest)), 0x7F7FFFFF)
        self.assertEqual(g17front.f32_literal_bits(repr(smallest)), 0x00800000)

    def test_and_zero_is_not_mistaken_for_an_underflow(self):
        self.assertEqual(g17front.f32_literal_bits("0.0"), 0)
        self.assertEqual(g17front.f32_literal_bits("-0.0"), 0x80000000)
        self.assertEqual(g17front.f32_literal_bits("0.000000e+00"), 0)


class StructurallyDifferentSourcePositives(unittest.TestCase):
    """Root's required positives: the literal in a call argument, in a store, and as an exponent."""

    def one(self, body, decl="device float *b [[buffer(0)]]"):
        return compile_source(HEAD + "kernel void k(%s,\n              uint3 t "
                              "[[thread_position_in_grid]]) {\n  %s\n}\n" % (decl, body))

    def test_a_literal_as_an_fma_call_argument(self):
        fn = self.one("b[t.x] = fma(b[t.x], 1.25f, b[t.x]);")
        self.assertTrue(any(o.kind == "const" for blk in fn.blocks for o in blk.ops))

    def test_a_literal_stored_directly(self):
        fn = self.one("b[8] = 1000.0f;")
        prog = g17cc.compile_function(fn)
        self.assertGreater(len(bytes(prog.code)), 0)

    def test_a_literal_with_an_exponent_compiles_end_to_end(self):
        prog = g17cc.compile_function(self.one("b[8] = 1.0e+03f;"))
        self.assertGreater(len(bytes(prog.code)), 0)

    def test_the_stored_word_is_the_LITERALS_bit_pattern_and_not_its_integer(self):
        """The denormal incident, guarded: 1000.0 must not become the immediate 1000."""
        fn = self.one("b[8] = 1000.0f;")
        consts = [o for blk in fn.blocks for o in blk.ops if o.kind == "const"]
        self.assertTrue(consts)
        vals = {a.v for o in consts for a in o.args if isinstance(a, ir.Imm)}
        self.assertIn(bits_of(1000.0), vals)
        self.assertNotIn(1000, vals)

    def test_one_literal_used_twice_materialises_once(self):
        fn = self.one("b[t.x] = fma(b[t.x], 2.5f, 2.5f);")
        consts = [o for blk in fn.blocks for o in blk.ops if o.kind == "const"]
        wanted = [o for o in consts if any(isinstance(a, ir.Imm) and a.v == bits_of(2.5)
                                           for a in o.args)]
        self.assertEqual(len(wanted), 1, "one distinct literal, one const op")


class ThePairedFMAWaitingControl(unittest.TestCase):
    """Root's requested pair: the normal program versus one requesting the ADMITTED wait everywhere.

    The retained sl32-u148 image carries 10737418240 - byte0[3] SET - on its first FMA, whose
    operand comes from a load, and 8589934592 on the 147 that chain from the previous FMA. When this
    class was written the arithmetic checker admitted only the first value, because the only
    retained receipt carrying the second was a FAILED one whose cause is confounded with a wrong
    immediate; a failing normal program alone would not have proved bit 31 caused it, and the pair
    is what would.

    THE PAIR THEN RAN (results/g17-source-literals-runtime-v2) and both values are now admitted,
    each with its own evidence in `g17normcheck.LEAD_MODIFIER_EVIDENCE`. That does not retire this
    control: the option below still requests a value that has executed receipts rather than an
    unmodelled state, and the never-executed third value must still refuse.

    `g17cc._FMA_ALWAYS_LOAD_WAIT` is the one degree of freedom. Default OFF. ON requests a state the
    checker ALREADY admits, which is the safe direction to author: it can only make an instruction
    wait longer than it needs to.
    """

    CHAIN = (HEAD + "kernel void k(device float *in [[buffer(1)]], device float *out [[buffer(2)]],\n"
                    "              uint3 t [[thread_position_in_grid]]) {\n"
                    "  float a = in[t.x];\n"
                    "  a = fma(a, 1.25f, 0.5f);\n"
                    "  a = fma(a, 1.5f, 0.25f);\n"
                    "  a = fma(a, 1.75f, 0.125f);\n"
                    "  out[t.x] = a;\n}\n")
    # The two values by WHAT THEY ENCODE, not by whether the checker happened to admit them when
    # this class was written: WAITED is byte0[3] set, CHAINED is clear. Both are admitted now.
    WAITED, CHAINED, NEVER_EXECUTED, FMA = 10737418240, 8589934592, 12884901888, 2190

    def both(self, text=None):
        """Through the KEYWORD, which is the only supported way to ask.

        This helper used to assign `g17cc._FMA_ALWAYS_LOAD_WAIT` directly, and the moment
        compile_function took the option as a keyword that stopped working - the keyword's default
        overwrites the global at entry, so the "forced" arm silently compiled as the normal one and
        two structural cases went green on two identical programs. That is the caller pattern root
        asked to make impossible, and it turns out the tests were one of the callers.
        """
        import g17cc
        return [g17cc.compile_function(compile_source(text or self.CHAIN),
                                       fma_always_load_wait=force)
                for force in (False, True)]

    def test_the_option_defaults_off(self):
        import g17cc
        self.assertFalse(g17cc._FMA_ALWAYS_LOAD_WAIT)

    def test_the_control_requests_a_value_WITH_EXECUTED_RECEIPTS(self):
        """Not a new state: the value every FMA of the executed control arm carries.

        This assertion used to be `LEAD_MODIFIERS[FMA] == {ADMITTED}` and it went red the day the
        paired run admitted the other value - a guard firing because the thing it guarded was
        lifted, not because anything broke. What it protects is unchanged: the option may only ask
        for a value some program carried and whose output was checked, never an unmodelled one.
        """
        import g17normcheck as N
        # 32, the four- and six-byte forms' destination lifetime, joined 2026-09-23 on its own
        # executed records (isa/g17-execution-ffma4-results.json, -ffma6-results.json) - the same
        # shape of event this docstring describes, and the protection is the same: every admitted
        # value names the execution that admitted it
        self.assertEqual(N.LEAD_MODIFIERS[N.FMA], {self.WAITED, self.CHAINED, 32})
        for value in (self.WAITED, self.CHAINED, 32):
            self.assertIn("campaign" if value == self.CHAINED else "execution_evidence",
                          N.LEAD_MODIFIER_EVIDENCE[(N.FMA, value)])
        self.assertNotIn(self.NEVER_EXECUTED, N.LEAD_MODIFIERS[N.FMA])

    def test_the_two_programs_are_structurally_IDENTICAL(self):
        import g17packedcheck as D
        a, b = (bytes(p.code) for p in self.both())
        self.assertEqual(len(a), len(b))
        ra, rb = D.decode(a), D.decode(b)
        self.assertEqual([r[0] for r in ra], [r[0] for r in rb], "instruction boundaries")
        self.assertEqual([r[1] for r in ra], [r[1] for r in rb], "instruction lengths")
        self.assertEqual([r[2] for r in ra], [r[2] for r in rb], "opcodes")

    def test_only_chained_FMA_lead_modifiers_differ(self):
        import g17packedcheck as D
        a, b = (bytes(p.code) for p in self.both())
        ra, rb = D.decode(a), D.decode(b)
        differ = [(x[2], x[3], y[3]) for x, y in zip(ra, rb) if x[3] != y[3]]
        self.assertTrue(differ, "the control must change something")
        for opc, ta, tb in differ:
            self.assertEqual(opc, self.FMA)
            self.assertEqual([t for t in ta if not t.startswith("imm:")],
                             [t for t in tb if not t.startswith("imm:")],
                             "registers and lifetimes must not move")
            ia = [t for t in ta if t.startswith("imm:")]
            ib = [t for t in tb if t.startswith("imm:")]
            self.assertEqual(ia[0], "imm:%d" % self.CHAINED)
            self.assertEqual(ib[0], "imm:%d" % self.WAITED)
            self.assertEqual(ia[1:], ib[1:], "no other immediate or constant may move")

    def test_the_byte_difference_is_byte0_bit_3_AND_NOTHING_ELSE(self):
        """Field ownership through the existing encoder and decoder, not by assertion."""
        import g17packedcheck as D
        a, b = (bytes(p.code) for p in self.both())
        differing = [i for i in range(len(a)) if a[i] != b[i]]
        self.assertTrue(differing)
        starts = {r[0] for r in D.decode(a) if r[2] == self.FMA}
        for i in differing:
            self.assertEqual(a[i] ^ b[i], 0x08, "only byte0 bit 3 may change")
            self.assertIn(i, starts, "every differing byte must be an op%d byte0" % self.FMA)

    # A NO-FMA PROGRAM THAT ACTUALLY CONTAINS THE CONSTRUCT IT BOUNDS. The first version of this
    # case stored a constant and nothing else, so it held no wait-capable operation at all and could
    # not have caught the defect root found: `opc in AUTH_LOAD_WAIT` also matches fadd, fmul, madd,
    # fsat and five bitwise opcodes on the same selection path, so the option was changing
    # instructions the assignment never mentioned. This program CHAINS fmul and fadd off a load -
    # the first consumes the load and carries the wait naturally, the later ones clear it and are
    # exactly what a broad predicate would flip.
    NO_FMA_CHAIN = (HEAD + "kernel void k(device float *in [[buffer(1)]], device float *out [[buffer(2)]],\n"
                           "              uint3 t [[thread_position_in_grid]]) {\n"
                           "  float a = in[t.x];\n"
                           "  a = a * 1.25f;\n"
                           "  a = a + 0.5f;\n"
                           "  a = a * 1.5f;\n"
                           "  a = a + 0.25f;\n"
                           "  out[t.x] = a;\n}\n")

    def test_the_no_fma_program_really_contains_wait_capable_operations(self):
        """The bound is worthless unless the program holds the thing being bounded."""
        import g17cc
        import g17packedcheck as D
        prog = g17cc.compile_function(compile_source(self.NO_FMA_CHAIN))
        opcodes = {r[2] for r in D.decode(bytes(prog.code))}
        self.assertNotIn(self.FMA, opcodes, "this program must have no FMA")
        self.assertTrue(opcodes & set(g17cc.AUTH_LOAD_WAIT),
                        "it must contain a wait-capable operation: %s" % sorted(opcodes))

    def test_it_is_byte_identical_under_the_REPAIRED_option(self):
        a, b = (bytes(p.code) for p in self.both(self.NO_FMA_CHAIN))
        self.assertEqual(a, b, "the option must touch op2190 and nothing else")

    def test_and_the_OLD_BROAD_PREDICATE_would_have_changed_it(self):
        """The discriminating half: without this the repair is unfalsifiable.

        Re-creates the predicate as it was - `opc in AUTH_LOAD_WAIT` - by asking for the wait on the
        very opcodes the chain contains, and shows the program's bytes move. So the case above is
        not passing because the program is inert.
        """
        import g17cc
        import g17packedcheck as D
        normal = bytes(g17cc.compile_function(compile_source(self.NO_FMA_CHAIN)).code)
        present = {r[2] for r in D.decode(normal)} & set(g17cc.AUTH_LOAD_WAIT)
        self.assertTrue(present)
        saved = g17cc.FMA_OPCODE
        moved = set()
        for opc in sorted(present):
            g17cc.FMA_OPCODE = opc            # the broad predicate, one opcode at a time
            try:
                broad = bytes(g17cc.compile_function(compile_source(self.NO_FMA_CHAIN),
                                                     fma_always_load_wait=True).code)
            finally:
                g17cc.FMA_OPCODE = saved
            if broad != normal:
                moved.add(opc)
        self.assertTrue(moved, "no wait-capable opcode in this chain moved, so the case is inert: "
                               "%s" % sorted(present))
        self.assertNotIn(self.FMA, moved)

    def test_the_normal_arm_reproduces_ROOTS_RETAINED_IMAGE(self):
        """The pair is only a control if arm A is the program root actually retained."""
        import hashlib
        import json as _json
        base = os.path.join(ROOT, "results", "g17-source-admission-v2", "baseline.json")
        image = os.path.join(ROOT, "results", "g17-source-literals-runtime-v1", "programs",
                             "source_literals", "program.bin")
        if not (os.path.isfile(base) and os.path.isfile(image)):
            self.skipTest("root's baseline and retained literal image are not extracted here")
        text = _json.load(open(base))["source_texts"]["sl32-u148"]
        a, b = self.both(text)
        self.assertEqual(hashlib.sha256(bytes(a.code)).hexdigest(),
                         hashlib.sha256(open(image, "rb").read()).hexdigest())
        self.assertNotEqual(bytes(a.code), bytes(b.code))

    def test_the_ABI_is_the_same_for_both_arms(self):
        import hashlib
        import json as _json
        a, b = self.both()
        canon = lambda p: hashlib.sha256(_json.dumps(p.abi(), sort_keys=True,
                                                     default=str).encode()).hexdigest()
        self.assertEqual(canon(a), canon(b))


class TheCapabilityIsSwITCHABLE(unittest.TestCase):
    """_NO_FP32_LITERALS restores the pre-change reading, which is what keeps the byte-identity of
    the 52 previously-compiled sources measurable instead of asserted."""

    def test_with_it_off_the_literal_refuses_again(self):
        g17front._NO_FP32_LITERALS = True
        try:
            self.assertIsNone(g17front.f32_literal_bits("1.25"))
            with self.assertRaises(g17front.Unsupported):
                compile_source(HEAD + "kernel void k(device float *b [[buffer(0)]],\n"
                                      "              uint3 t [[thread_position_in_grid]])"
                                      " { b[8] = 1000.0f; }\n")
        finally:
            g17front._NO_FP32_LITERALS = False

    def test_and_with_it_on_the_same_source_reads(self):
        fn = compile_source(HEAD + "kernel void k(device float *b [[buffer(0)]],\n"
                                   "              uint3 t [[thread_position_in_grid]])"
                                   " { b[8] = 1000.0f; }\n")
        self.assertTrue(any(o.kind == "const" for blk in fn.blocks for o in blk.ops))


if __name__ == "__main__":
    unittest.main(verbosity=1)
