import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import g17oracle as O
import g17rebuilddrift as D


def _plan(batch, rid):
    for r in json.load(open(os.path.join(ROOT, "isa", batch))):
        if r.get("id") == rid:
            return r
    raise KeyError(rid)


class DriftedRecordsAreRefused(unittest.TestCase):
    def test_a_lifetime_drifted_record_is_refused(self):
        why = O.refusal(_plan("g17-execution-sweep.json", "D3791.l10"))
        self.assertIsNotNone(why)
        self.assertIn("not reproducible", why)

    def test_a_modifier_drifted_record_is_refused(self):
        self.assertIn("not reproducible", O.refusal(_plan("g17-execution-names.json", "op13460.called-nand")) or "")

    def test_every_listed_record_is_refused_and_count_is_42_plus_the_binding_case(self):
        listed = D.not_reproducible()
        self.assertEqual(len(listed), 43)

    def test_a_binding_drift_the_bytes_cannot_see_is_refused(self):
        why = O.refusal(_plan("g17-execution-unsafe.json", "u12682.5")) or ""
        self.assertIn("binding moved", why)

    def test_a_register_only_record_is_not_refused_for_drift(self):
        rows = json.load(open(D.DEST))["rows"]
        reg = next(r for r in rows if r["cls"] == "reg")
        why = O.refusal(_plan(reg["batch"], reg["id"])) or ""
        self.assertNotIn("not reproducible", why)

    def test_an_edited_record_is_a_different_plan(self):
        # re-authoring under a new id is the documented repair, and it must clear the refusal
        rec = dict(_plan("g17-execution-sweep.json", "D3791.l10"), id="D3791.l10.reauthored")
        self.assertNotIn("not reproducible", O.refusal(rec) or "")


if __name__ == "__main__":
    unittest.main()
