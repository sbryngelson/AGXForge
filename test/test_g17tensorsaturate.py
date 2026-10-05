#!/usr/bin/env python3
"""Saturating int8 GEMM in tlower (Set A item 4): which issues carry the bit, and what refuses."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agxforge.g17 import model, tlower


def int8_mmas(**kw):
    body = tlower.lower(32, 32, 64, 64, 32, 32, a_type="int8", b_type="int8", **kw)[0]
    return [i for i in model.decode(body, 0) if i.opcode and i.opcode.id in (10384, 10385)]


class Saturate(unittest.TestCase):
    def test_every_issue_with_c_saturates_and_the_no_c_issue_is_unchanged(self):
        mm = int8_mmas(saturate=True)
        self.assertEqual(sum(i.opcode.id == 10385 for i in mm), 4)
        self.assertTrue(all(i.values[2][1] == 41 for i in mm if i.opcode.id == 10384))
        self.assertEqual(sum(i.opcode.id == 10384 for i in mm), 12)

    def test_the_default_wraps(self):
        self.assertTrue(all(i.values[2][1] == 9 for i in int8_mmas() if i.opcode.id == 10384))

    def test_a_saturating_accumulate_seeds_d_with_c_and_saturates_every_issue(self):
        body = tlower.lower(32, 32, 64, 64, 32, 32, a_type="int8", b_type="int8", saturate=True, accumulate=True)[0]
        ins = [i for i in model.decode(body, 0) if i.opcode]
        mm = [i for i in ins if i.opcode.id in (10384, 10385)]
        self.assertEqual([i.opcode.id for i in mm], [10384] * 16)             # no no-C issue: D starts at C
        self.assertTrue(all(i.values[2][1] == 41 for i in mm))
        self.assertTrue((mm[0].values[1][1] >> 30) & 1)                        # waits for the seed (slot 6)
        self.assertFalse(any(i.opcode.id == 10282 for i in ins[ins.index(mm[-1]):]))   # no wrapping iadd

    def test_unimplemented_forms_refuse(self):
        with self.assertRaises(ValueError):
            tlower.lower(32, 32, 64, 64, 32, 32, a_type="half", b_type="half", saturate=True)
        with self.assertRaises(ValueError):
            tlower.lower(17, 32, 64, 64, 32, 32, a_type="int8", b_type="int8", saturate=True, accumulate=True)


if __name__ == "__main__":
    unittest.main()
