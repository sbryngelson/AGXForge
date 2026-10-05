import copy
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import g17rebuilddrift as D

# A fake disassembly keyed by hex, so each class is exercised without building anything.
FAKE = {"aa": ["4", "998", "reg:1", "imm:0", "reg:2"], "ab": ["4", "998", "reg:3", "imm:0", "reg:2"],
        "ac": ["4", "998", "reg:1", "imm:16", "reg:2"], "ad": ["6", "998", "reg:1", "imm:0", "reg:2"],
        "ae": ["4", "998", "reg:1", "reg:5", "reg:2"], "af": ["4", "999", "reg:1", "imm:0", "reg:2"]}


def fake(hexes):
    return [FAKE[h] for h in hexes]


class RebuildDriftTests(unittest.TestCase):
    def test_each_class_is_named(self):
        c = lambda new: D.classify_pair(["aa"], new, dis=fake)
        self.assertEqual(c(["aa"]), "identical")
        self.assertEqual(c(["ab"]), "reg")
        self.assertEqual(c(["ac"]), "imm")
        self.assertEqual(c(["ad"]), "opcode or length")
        self.assertEqual(c(["ae"]), "operand kinds")
        self.assertEqual(c(["af"]), "opcode or length")
        self.assertEqual(c(["aa", "aa"]), "instance count differs")

    def test_the_written_report_checks(self):
        doc = D.check()
        self.assertEqual(doc["summary"]["records_whose_rebuild_is_a_different_instruction"], [])

    def _write(self, doc):
        path = os.path.join(os.environ.get("TMPDIR", "/tmp"), "rebuilddrift-%d.json" % os.getpid())
        with open(path, "w") as fh:
            json.dump(doc, fh)
        self.addCleanup(os.remove, path)
        return path

    def test_a_semantic_drift_refuses(self):
        doc = json.load(open(D.DEST))
        doc["rows"][0]["cls"] = "opcode or length"
        doc["summary"] = D.summarise(doc["rows"])
        with self.assertRaisesRegex(D.Refused, "DIFFERENT instruction"):
            D.check(self._write(doc))

    def test_a_summary_that_disagrees_with_its_rows_refuses(self):
        doc = json.load(open(D.DEST))
        doc["summary"]["classes"]["identical"] += 1
        with self.assertRaisesRegex(D.Refused, "not what its rows give"):
            D.check(self._write(doc))


if __name__ == "__main__":
    unittest.main()
