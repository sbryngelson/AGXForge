"""The shared interpreter's conversion arm: a CANDIDATE, kept out of the sets it would corrupt.

Root asked for "the smallest interpreter extension needed for the delivered form... keep
interpretation a candidate until hardware measures it". Three things make that more than a promise:

  * op11179 is NOT in SEMANTICS. Three sets are derived from it - RSQRT_OPCODES, ARITH_FP, and the
    Machine arm that reads its source AS A FLOAT - so a member added there would be called a
    reciprocal square root, enrolled in float arithmetic, and handed a uint32 bit pattern to read as
    a float. Its own table and its own raw-bits arm instead.
  * the confidence is "candidate", which no gate admits: g17layernormimagecheck takes only
    ("executed", "bounded") and g17attentioninterpret raises on a verdict it does not list.
  * compare() treats a candidate BIT-EXACTLY rather than with a tolerance, because at 2^24+3 the
    difference between rounding and truncation is 1.2e-7 relative - inside the 1e-6 the isolation
    path would have allowed. A candidate model a comparison cannot refute is not a model.

Nothing here runs on hardware.
"""
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, ROOT)

import g17normcheck as N

OPCODE = 11179
_bits = lambda f: struct.unpack("<I", struct.pack("<f", f))[0]
_r32 = lambda x: struct.unpack("<f", struct.pack("<f", x))[0]


class TheConversionIsItsOwnClass(unittest.TestCase):
    def test_it_is_not_in_the_transcendental_table(self):
        self.assertNotIn(OPCODE, N.SEMANTICS)
        self.assertNotIn(OPCODE, N.RSQRT_OPCODES)

    def test_it_is_not_in_the_float_arithmetic_set(self):
        """Being in ARITH_FP would demand EXECUTION evidence and read its source as a float."""
        self.assertNotIn(OPCODE, N.ARITH_FP)

    def test_it_is_in_the_conversion_table_at_one_length(self):
        self.assertIn(OPCODE, N.CONVERSIONS)
        self.assertEqual(N.CONVERSION_LENGTHS, (10,))
        self.assertEqual(N.CONVERSIONS[OPCODE][0], "cvt.u32.f32")

    def test_an_unmodelled_conversion_refuses_by_name(self):
        with self.assertRaises(N.Unexecuted):
            N.interpret_conversion(11185, 0)


class TheModelAndItsConfidence(unittest.TestCase):
    CASES = (0, 1, 2 ** 24 - 1, 2 ** 24, 2 ** 24 + 1, 2 ** 24 + 3,
             2 ** 31 - 1, 2 ** 31, 2 ** 32 - 1)

    def test_it_converts_unsigned_and_never_signed(self):
        for u in self.CASES:
            with self.subTest(u=u):
                value, _c, _w = N.interpret_conversion(OPCODE, u)
                self.assertEqual(_bits(value), _bits(_r32(float(u))))
                if u >= 2 ** 31:
                    self.assertNotEqual(_bits(value), _bits(_r32(float(u - 2 ** 32))))

    def test_the_confidence_is_candidate_and_says_what_it_rests_on(self):
        _v, conf, why = N.interpret_conversion(OPCODE, 7)
        self.assertEqual(conf, "candidate")
        self.assertIn("CANDIDATE", why)
        self.assertIn("none of it execution", why)

    def test_candidate_ranks_below_isolation_and_above_nothing(self):
        self.assertIn("candidate", N.CONFIDENCE_ORDER)
        self.assertEqual(N.weakest("executed", "candidate"), "candidate")
        self.assertEqual(N.weakest("isolation", "candidate"), "candidate")
        self.assertEqual(N.weakest("candidate", "unverified"), "unverified")

    def test_a_gate_that_admits_only_measured_evidence_refuses_a_candidate(self):
        self.assertNotIn("candidate", ("executed", "bounded"))
        src = open(os.path.join(ROOT, "tools", "g17layernormimagecheck.py")).read()
        self.assertIn('confidence not in ("executed", "bounded")', src)


class TheComparisonCanRefuteTheModel(unittest.TestCase):
    def test_a_candidate_is_compared_bit_exactly(self):
        same, claim = N.compare(1.0, 1.0, "candidate")
        self.assertTrue(same)
        self.assertIn("bit-exact", claim)
        self.assertIn("CANDIDATE", claim)

    def test_truncation_is_refuted_at_two_to_the_24_plus_3(self):
        """The case the isolation path's 1e-6 tolerance would have accepted."""
        rounded = _r32(float(2 ** 24 + 3))
        truncated = _r32(float(2 ** 24 + 2))
        self.assertLess(abs(rounded - truncated) / rounded, 1e-6)      # inside the old tolerance
        self.assertTrue(N.compare(truncated, rounded, "isolation")[0])  # ... and it would pass
        self.assertFalse(N.compare(truncated, rounded, "candidate")[0])  # ... but not here

    def test_a_signed_conversion_is_refuted_at_uint32_max(self):
        want, _c, _w = N.interpret_conversion(OPCODE, 2 ** 32 - 1)
        self.assertFalse(N.compare(-1.0, want, "candidate")[0])


class TheAddCanHideAWrongConversion(unittest.TestCase):
    """Root's review of 156d4e68, reproduced: nine receipts, five identifying inputs.

    The stored word is the conversion followed by an op998 ADD, so a case whose addend is nonzero
    does not identify the conversion's intermediate bits. Root's counterexample is exact and is the
    first case below. Input 0 fails for a different reason - the sign survives nowhere. This class
    is the failing control root asked for, plus the positive half: on the five that DO identify, a
    one-ulp wrong intermediate is visible in the word.
    """

    CASES = [(0, 0.0), (1, 0.5), (2 ** 24 - 1, -1.25), (2 ** 24, 1.25), (2 ** 24 + 1, 0.0),
             (2 ** 24 + 3, 0.0), (2 ** 31 - 1, 0.0), (2 ** 31, 0.0), (2 ** 32 - 1, 0.0)]

    @staticmethod
    def _asf(b):
        return struct.unpack("<f", struct.pack("<I", b & 0xFFFFFFFF))[0]

    def step(self, f, d):
        """the true float32 neighbour, by bit - binary64 nextafter then rounding gives f back."""
        return self._asf(_bits(f) + d)

    def test_roots_counterexample_reproduces_exactly(self):
        one = _r32(1.0)
        predecessor = self.step(one, -1)
        self.assertAlmostEqual(predecessor, 0.99999994039535522, places=16)
        self.assertEqual(_bits(predecessor), 0x3F7FFFFF)
        self.assertEqual(_bits(_r32(one + _r32(0.5))), _bits(_r32(predecessor + _r32(0.5))))
        self.assertEqual(_bits(_r32(one + _r32(0.5))), 0x3FC00000)          # both give 1.5

    def test_input_one_is_therefore_NOT_in_the_measured_set(self):
        self.assertNotIn(1, N.CONVERSION_MEASURED[(OPCODE, 16)])
        self.assertIn(1, N.CONVERSION_WORKLOAD)

    def test_input_zero_loses_the_sign_and_is_also_out(self):
        self.assertEqual(_bits(_r32(-0.0 + 0.0)), _bits(_r32(0.0 + 0.0)))
        self.assertNotIn(0, N.CONVERSION_MEASURED[(OPCODE, 16)])
        self.assertIn(0, N.CONVERSION_WORKLOAD)

    def test_the_measured_set_is_exactly_the_positive_zero_addend_inputs(self):
        want = frozenset(u for u, f in self.CASES if f == 0.0 and u > 0)
        self.assertEqual(N.CONVERSION_MEASURED[(OPCODE, 16)], want)
        self.assertEqual(len(want), 5)

    def test_on_the_five_a_one_ulp_wrong_intermediate_IS_visible(self):
        """The positive half: these five are not degenerate, which is why they survive."""
        for u in sorted(N.CONVERSION_MEASURED[(OPCODE, 16)]):
            with self.subTest(u=u):
                exact = _r32(float(u))
                word = _bits(_r32(exact + 0.0))
                for d in (-1, 1):
                    self.assertNotEqual(_bits(_r32(self.step(exact, d) + 0.0)), word)

    def test_the_two_large_nonzero_addend_cases_are_excluded_CONSERVATIVELY(self):
        """Stated so nobody restores them from a neighbour scan: +-1 and +-2 ulp ARE visible in
        their words, and they are still excluded, because identifying the conversion through a
        nonzero addend needs an argument about the add and not a scan of neighbours."""
        for u, f in ((2 ** 24 - 1, -1.25), (2 ** 24, 1.25)):
            with self.subTest(u=u):
                exact = _r32(float(u))
                word = _bits(_r32(exact + _r32(f)))
                for d in (-2, -1, 1, 2):
                    self.assertNotEqual(_bits(_r32(self.step(exact, d) + _r32(f))), word)
                self.assertNotIn(u, N.CONVERSION_MEASURED[(OPCODE, 16)])

    def test_the_workload_is_still_nine_and_named_separately(self):
        self.assertEqual(N.CONVERSION_WORKLOAD, frozenset(u for u, _f in self.CASES))
        self.assertEqual(len(N.CONVERSION_WORKLOAD), 9)


class TheThreadgroupRegister(unittest.TestCase):
    """SR156 = threadgroup_position_in_grid.x, and the register number is MEASURED."""

    def test_the_decoder_places_sr156_at_reg54(self):
        import subprocess
        import tempfile
        import g17fields as FL
        from agxforge.g17 import asm, const as g17const
        tmpl = g17const.load()[("read_sr.4", None, 4)]["value"]
        seen = {}
        for sr in (156, 160, 161):
            b = asm.encode_sr(dest=4, sr=sr, seq=0, template=tmpl)
            with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as fh:
                fh.write(b)
                path = fh.name
            out = subprocess.run([FL.DIS, path, "0", str(len(b)), "--pc", "0", "--expr"],
                                 capture_output=True, text=True).stdout
            os.unlink(path)
            seen[sr] = [t for t in out.split() if t.startswith("reg:")][-1]
        self.assertEqual(seen[156], "reg:54")
        self.assertEqual(seen[160], "reg:61")       # the one this checker already modelled
        self.assertEqual(seen[161], "reg:62")

    def test_a_multi_thread_launch_without_a_group_is_refused(self):
        """The threadgroup position is NOT the thread index, and guessing is what this refuses."""
        m = N.Machine([(0, 0, False), (7, 2, True)], {0: [0.0], 1: [0.0]}, 3,
                      frozenset(), frozenset())
        toks = ["reg:109", "imm:1048576", "reg:54", "imm:0"]
        with self.assertRaises(N.Unexecuted) as cm:
            m.step(0, 4, N.READ_SR, toks)
        self.assertIn("which threadgroup", str(cm.exception))

    def test_one_thread_reads_threadgroup_zero(self):
        m = N.Machine([(0, 0, False), (7, 2, True)], {0: [0.0], 1: [0.0]}, 0,
                      frozenset(), frozenset())
        m.step(0, 4, N.READ_SR, ["reg:109", "imm:1048576", "reg:54", "imm:0"])
        self.assertEqual(m.rd("reg:109"), 0)

    def test_an_explicit_group_is_read_instead_of_the_thread_index(self):
        m = N.Machine([(0, 0, False), (7, 2, True)], {0: [0.0], 1: [0.0]}, 2,
                      frozenset(), frozenset())
        m.group = 5
        m.step(0, 4, N.READ_SR, ["reg:109", "imm:1048576", "reg:54", "imm:0"])
        self.assertEqual(m.rd("reg:109"), 5)


class TheArmRefusesWhatItHasNotMeasured(unittest.TestCase):
    def machine(self):
        return N.Machine([(0, 0, False), (7, 2, True)], {0: [0.0], 1: [0.0]}, 0,
                         frozenset(), frozenset())

    TOKS = ["reg:111", "imm:32", "imm:4", "imm:0", "reg:110", "imm:16"]

    def test_it_reads_the_source_as_BITS_and_not_as_a_float(self):
        m = self.machine()
        m.wr("reg:110", 7)                       # the integer 7, not the float bits of 7.0
        m.step(0, 10, 11179, self.TOKS)
        self.assertEqual(m.rd("reg:111"), _bits(7.0))

    def test_a_wrong_length_is_refused(self):
        with self.assertRaises(N.Unexecuted) as cm:
            self.machine().step(0, 12, 11179, self.TOKS)
        self.assertIn("is 12 bytes", str(cm.exception))

    def test_a_layout_that_is_not_apples_is_refused(self):
        for toks in (["reg:111", "imm:32", "reg:110", "imm:16"],
                     ["imm:32", "imm:4", "imm:0", "imm:0", "reg:110", "imm:16"],
                     ["reg:111", "imm:32", "imm:4", "imm:0", "imm:9", "imm:16"]):
            with self.subTest(toks=toks):
                with self.assertRaises(N.Unexecuted):
                    self.machine().step(0, 10, 11179, toks)

    def test_an_operand_5_apple_never_ships_is_refused(self):
        toks = list(self.TOKS)
        toks[-1] = "imm:48"
        with self.assertRaises(N.Unexecuted) as cm:
            self.machine().step(0, 10, 11179, toks)
        self.assertIn("operand 5 = 48", str(cm.exception))

    def test_the_three_shipped_values_are_admitted(self):
        for v in (0, 16, 32):
            with self.subTest(v=v):
                toks = list(self.TOKS)
                toks[-1] = "imm:%d" % v
                m = self.machine()
                m.wr("reg:110", 3)
                m.step(0, 10, 11179, toks)
                self.assertEqual(m.rd("reg:111"), _bits(3.0))


if __name__ == "__main__":
    unittest.main(verbosity=1)
