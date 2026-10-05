#!/usr/bin/env python3
"""The op592 publish field is an index 0..8, not a two-bit field plus a flag at bit 2.

Section 25.27 read the second immediate as a two-bit low part plus independent flags, and left a
residue: what flag bit 2 selects, and why it never combines with the low field. Both halves were
artifacts of a 1,010-instance frame that contained indices 0 through 4 only. Section 25.55.

The discriminator is what a wider population does at 9..15: four independent flags would populate
those as combinations; an index stops. It stops at 8.
"""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CENSUS = ROOT / "results/g17-publish-index-v1/index-census.json"


class ThePublishFieldIsAnIndex(unittest.TestCase):
    def setUp(self):
        if not CENSUS.is_file():
            self.skipTest("run g17evidence.py extract --archive evidence/g17-vendor-shader-corpus.zip")
        self.doc = json.loads(CENSUS.read_text())

    def test_the_index_is_contiguous_over_zero_to_eight(self):
        index = {int(k): v for k, v in self.doc["vendor"]["index"].items()}
        for value in range(9):
            self.assertGreater(index[value], 0, "index %d must be witnessed" % value)

    def test_nothing_above_eight_is_ever_emitted(self):
        # THE DISCRIMINATOR. A 4-bit flag field would reach 9..15 by combination; an index does not.
        index = {int(k): v for k, v in self.doc["vendor"]["index"].items()}
        self.assertEqual([index[v] for v in range(9, 16)], [0] * 7)

    def test_bit_two_combines_with_the_low_field(self):
        # The residue section 25.27 left: "bit 2 never combines with the low field".
        index = {int(k): v for k, v in self.doc["vendor"]["index"].items()}
        for value in (5, 6, 7):          # bit2 with bit0, with bit1, with both
            self.assertGreater(index[value], 1000,
                               "index %d is bit 2 combined with the low field" % value)

    def test_index_five_is_not_untouched(self):
        index = {int(k): v for k, v in self.doc["vendor"]["index"].items()}
        self.assertGreater(index[5], 0, "'index 5 is the one no vendor object touches' was frame-bound")

    def test_the_old_frame_explains_the_old_reading(self):
        # Why 25.27 could not have concluded otherwise: its population stopped at 4.
        old = {int(k): v for k, v in self.doc["old_corpus"]["index"].items()}
        self.assertTrue(all(old[v] > 0 for v in range(5)))
        self.assertEqual([old[v] for v in range(5, 16)], [0] * 11)

    def test_the_old_counts_reconcile_with_the_published_population(self):
        old = {int(k): v for k, v in self.doc["old_corpus"]["index"].items()}
        self.assertEqual(self.doc["old_corpus"]["instances"], 4694)
        self.assertEqual(sum(v for k, v in old.items() if k), 1010)


if __name__ == "__main__":
    unittest.main()
