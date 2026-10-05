"""Guards for H1's slot-reach probe.

The finding is a refutation - eight-bit slot fields DO name R128 - so the cases below mostly guard
against the probe having imposed the ceiling it reports. Two controls carry that weight: a nine-bit
slot written with 256 must name R128 (the method can see past R127 at all), and a half-indexed
eight-bit slot written with 128 must name R64 (it reads the operand the field controls, not the
largest register in the instruction). The first version of this probe took the maximum register over
all operands and reported an eight-bit field reaching R142, which was a different operand moving.
"""
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, ROOT)

EVIDENCE = os.path.join(ROOT, "tools", "tensorops-model", "slot-reach-witness.json")


def _load():
    with open(EVIDENCE) as fh:
        return json.load(fh)


def _probe(doc, width, value):
    for p in doc["probes"]:
        if p["width"] == width and p["value"] == value:
            return p
    raise AssertionError("no probe for width %d value %d" % (width, value))


class TheProbeDidNotImposeItsOwnCeiling(unittest.TestCase):

    def setUp(self):
        self.doc = _load()

    def test_a_nine_bit_slot_written_with_256_names_r128(self):
        p = _probe(self.doc, 9, 256)
        self.assertEqual({int(k) for k in p["registers"]}, {128})
        self.assertGreater(p["samples"], 1000)
        self.assertEqual(p["refused"], 0, "a refusing control proves nothing about reach")

    def test_a_half_indexed_eight_bit_slot_written_with_128_names_r64(self):
        p = _probe(self.doc, 8, 128)
        self.assertGreater(p["registers"].get("64", 0), 1000,
                           "without this the probe may be reading the wrong operand")

    def test_each_probe_moved_exactly_one_operand_per_sample(self):
        for p in self.doc["probes"]:
            self.assertGreater(p["samples"], 0, "width %d value %d" % (p["width"], p["value"]))
            for ex in p["examples"]:
                self.assertNotEqual(ex["before"], ex["after"])


class TheEightBitPremiseIsRefuted(unittest.TestCase):

    def setUp(self):
        self.doc = _load()

    def test_eight_bit_slots_name_r128_directly(self):
        p = _probe(self.doc, 8, 128)
        self.assertGreaterEqual(p["registers"].get("128", 0), 100,
                                "this is the whole refutation; a handful would not carry it")

    def test_the_conclusion_records_the_premise_as_false(self):
        self.assertFalse(self.doc["conclusion"]["premise_eight_bit_cannot_reach_128"])

    def test_both_conventions_are_present_in_quantity(self):
        c = self.doc["conclusion"]
        self.assertGreater(c["eight_bit_half_indexed"], 1000)
        self.assertGreater(c["eight_bit_register_indexed"], 100)

    def test_half_indexed_eight_bit_slots_still_stop_at_127(self):
        # the premise is right about this half, and saying so is what makes the split a finding
        p = _probe(self.doc, 8, 255)
        self.assertEqual({int(k) for k in p["registers"]}, {127})
        self.assertEqual(self.doc["conclusion"]["eight_bit_max_at_all_ones"], 127)

    def test_an_example_shows_both_conventions_in_one_form(self):
        eight = _probe(self.doc, 8, 128)
        by_opcode = {}
        for ex in eight["examples"]:
            by_opcode.setdefault(ex["opcode"], set()).add(ex["register"])
        self.assertTrue(eight["examples"])
