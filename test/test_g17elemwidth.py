"""Guards for R7's element-width witness.

The finding is "the code equals the width", which a harness that read the width back out of its own
input would produce for free. So the cases below pin the two things that make it evidence: that the
population contains widths the vendor corpus does not have (1, 8 and 16 - without them width and
direction agree everywhere and the result is vacuous), and that the rival reading is asserted to
FAIL. If the direction check ever stops failing, the discriminating shapes have left the sweep.
"""
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, ROOT)

EVIDENCE = os.path.join(ROOT, "tools", "tensorops-model", "element-width-witness.json")


def _load():
    with open(EVIDENCE) as fh:
        return json.load(fh)


class ThePopulationBreaksTheConfound(unittest.TestCase):

    def setUp(self):
        self.doc = _load()

    def test_widths_outside_the_vendor_corpus_are_present(self):
        widths = set(self.doc["conclusion"]["widths_witnessed"])
        self.assertTrue({1, 8, 16} <= widths,
                        "without 1, 8 and 16 the width and direction readings agree everywhere")

    def test_an_int8_tensor_case_was_actually_built(self):
        int8 = [r for r in self.doc["tensor"] if r["operand_bytes"] == 1]
        self.assertTrue(int8)
        for row in int8:
            self.assertEqual(row["status"], "ok")
            self.assertTrue(row["load_codes"], "%s produced no tensor load" % row["tag"])

    def test_the_direction_reading_is_asserted_to_fail(self):
        # direction predicts 2 on every load; a load carrying 1 or 8 or 16 refutes it
        loads = [int(k) for r in self.doc["access"] for codes in r["codes"].values() for k in codes]
        tensor = [int(k) for r in self.doc["tensor"] for codes in r["load_codes"].values() for k in codes]
        self.assertTrue(set(loads + tensor) - {2},
                        "if every load carried 2 the direction reading would still stand")


class TheCodeEqualsTheWidth(unittest.TestCase):

    def setUp(self):
        self.doc = _load()

    def test_every_tensor_load_carries_its_operand_width(self):
        self.assertTrue(self.doc["tensor"])
        for row in self.doc["tensor"]:
            if row["status"] != "ok":
                continue
            for op, codes in row["load_codes"].items():
                self.assertEqual({int(k) for k in codes}, {row["operand_bytes"]},
                                 "%s op%s" % (row["tag"], op))

    def test_every_plain_access_carries_its_access_width(self):
        self.assertTrue(self.doc["access"])
        for row in self.doc["access"]:
            if row["status"] != "ok":
                continue
            for op, codes in row["codes"].items():
                self.assertEqual({int(k) for k in codes}, {row["access_bytes"]},
                                 "%s op%s" % (row["tag"], op))

    def test_the_tool_recorded_both_rules_as_holding(self):
        self.assertTrue(self.doc["conclusion"]["tensor_rule_holds"])
        self.assertTrue(self.doc["conclusion"]["access_rule_holds"])


class TheWidthIsCarriedByTheOpcode(unittest.TestCase):

    def setUp(self):
        self.doc = _load()

    def test_each_operand_width_uses_its_own_load_opcodes(self):
        by_width = {}
        for row in self.doc["tensor"]:
            if row["status"] == "ok":
                by_width.setdefault(row["operand_bytes"], set()).update(int(k) for k in row["load_codes"])
        self.assertGreaterEqual(len(by_width), 3, "need at least three widths to show the split")
        seen = list(by_width.values())
        for i, a in enumerate(seen):
            for b in seen[i + 1:]:
                self.assertFalse(a & b, "two widths share a load opcode: %s %s" % (a, b))

    def test_value_16_is_witnessed_on_a_sixteen_byte_access(self):
        sixteen = [r for r in self.doc["access"] if r["access_bytes"] == 16 and r["status"] == "ok"]
        self.assertGreaterEqual(len(sixteen), 2, "one type could be a quirk of that type")
        for row in sixteen:
            self.assertTrue(row["codes"], "%s produced no memory opcode" % row["tag"])
            for codes in row["codes"].values():
                self.assertEqual({int(k) for k in codes}, {16})

    def test_the_store_opcodes_do_not_share_one_code(self):
        # the direction reading needed every store to be marked alike
        stores = {}
        for row in self.doc["access"]:
            for op, codes in row["codes"].items():
                if int(op) in (17229, 17256):
                    stores.setdefault(int(op), set()).update(int(k) for k in codes)
        self.assertGreaterEqual(len(stores), 2)
        self.assertGreater(len({frozenset(v) for v in stores.values()}), 1,
                           "two store opcodes carrying different codes is what kills 'direction'")
