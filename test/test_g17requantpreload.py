#!/usr/bin/env python3
"""Fast structural checks for the retained requantization constant program.

These tests do not dispatch.  They ensure that the measured prologue is reproducible and that the
compiler route cannot widen its binding or system-register domain by accident.
"""

import hashlib
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agxforge.g17 import requantpreload


class RequantPreload(unittest.TestCase):
    def test_measured_prologue_is_exactly_64_bytes(self):
        got = requantpreload.build()
        self.assertEqual(len(got), 64)
        self.assertEqual(got[:38], requantpreload.MEASURED_HEAD)
        self.assertEqual(got[38:], requantpreload.FILLER * 13)
        self.assertEqual(hashlib.sha256(got).hexdigest(), requantpreload.PROLOGUE_SHA256)

    def test_build_does_not_accept_unmeasured_entry(self):
        for entry in (None, 38, 62, 66, 128):
            with self.subTest(entry=entry):
                with self.assertRaisesRegex(ValueError, "64-byte entry"):
                    requantpreload.build(entry=entry)

    def test_measured_binding_and_system_register_domain(self):
        requantpreload.validate_request(
            ((0, 0, False), (1, 2, False), (2, 4, True)), (160,))
        with self.assertRaisesRegex(ValueError, "bindings"):
            requantpreload.validate_request(
                ((0, 0, False), (1, 2, False), (2, 4, False)), (160,))
        with self.assertRaisesRegex(ValueError, "system registers"):
            requantpreload.validate_request(
                ((0, 0, False), (1, 2, False), (2, 4, True)), (130,))


if __name__ == "__main__":
    unittest.main()
