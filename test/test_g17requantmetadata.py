#!/usr/bin/env python3
"""Static checks for the measured scalar requantization metadata handoff.

These tests prove structural serialization.  Public authoring now admits this exact class, while
the compiler body and the two-dispatch runtime contract remain separate acceptance boundaries.
"""
import hashlib
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, ROOT)

from agxforge.g17 import requantmetadata as R
from agxforge.g17 import scanlink, verify


class RequantMetadata(unittest.TestCase):
    def test_measured_class_serializes_byte_identically_without_a_donor_blob(self):
        section = R.build()
        self.assertEqual(len(section), 476)
        self.assertEqual(hashlib.sha256(section).hexdigest(), R.WITNESS_METADATA_SHA256)
        self.assertEqual(verify.verify_metadata(section), [])
        self.assertEqual(scanlink.binding_records(section), list(R.WITNESS_RECORDS))

    def test_register_count_is_the_only_program_override(self):
        self.assertNotEqual(R.build(3), R.build(2))
        self.assertEqual(len(R.build(3)), 476)
        self.assertEqual(scanlink.binding_records(R.build(3)), list(R.WITNESS_RECORDS))

    def test_only_the_measured_three_record_sr160_class_is_accepted(self):
        R.validate_request(R.WITNESS_RECORDS, (160,))
        with self.assertRaisesRegex(ValueError, "measured only for bindings"):
            R.validate_request(((0, 0, False), (1, 2, True)), (160,))
        with self.assertRaisesRegex(ValueError, "system registers"):
            R.validate_request(R.WITNESS_RECORDS, (130, 156))


if __name__ == "__main__":
    unittest.main()
