#!/usr/bin/env python3
"""The saturating int8 MMA encoding against Apple's own bytes (compile only; recon section 136 part 6).

Both witnesses are Apple's compilation of one hand-written AIR kernel,
`widening_multiply_accumulate` and `widening_multiply_accumulate_saturate` with .s.s.s.s int8
operands, identical except for the spelling. Decoded, they differ only in op10384's operand 2
(9 and 41). The bytes are pinned here, so the test needs no toolchain.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agxforge.g17 import mmaenc

APPLE_PLAIN = bytes.fromhex("2f00250a220aa1024004")
APPLE_SATURATE = bytes.fromhex("2f00250a220aa1025004")


class SaturatingInt8(unittest.TestCase):
    def test_both_spellings_are_reproduced_byte_for_byte(self):
        # D = C = R0..R7, A = R10_R11, B = R8_R9, both released, the wait bit set: Apple's operands
        common = dict(a_type="int8", b_type="int8", wait=True, a_last=True, b_last=True)
        self.assertEqual(mmaenc.mma(0, 10, 8, 0, **common), (10384, APPLE_PLAIN))
        self.assertEqual(mmaenc.mma(0, 10, 8, 0, saturate=True, **common), (10384, APPLE_SATURATE))

    def test_the_witnesses_differ_in_one_bit(self):
        diff = [(i, a ^ b) for i, (a, b) in enumerate(zip(APPLE_PLAIN, APPLE_SATURATE)) if a != b]
        self.assertEqual(diff, [(8, 0x10)])       # byte 8 bit 4, as section 136 names it

    def test_forms_it_is_not_measured_for_refuse(self):
        with self.assertRaises(ValueError):
            mmaenc.mma(0, 10, 8, None, "int8", "int8", saturate=True)       # no-C form
        with self.assertRaises(ValueError):
            mmaenc.mma(0, 8, 12, 0, "half", "half", saturate=True)          # a float form


if __name__ == "__main__":
    unittest.main()
