"""32 on the four-byte fmul/fadd is admitted from isolated execution, and says so."""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import g17normcheck as N


class TheFourByteLeadModifierIsAdmittedOnEvidence(unittest.TestCase):

    def test_32_is_admitted_for_fmul_and_fadd(self):
        self.assertIn(32, N.LEAD_MODIFIERS[N.FMUL])
        self.assertIn(32, N.LEAD_MODIFIERS[N.FADD])

    def test_each_admission_names_its_executed_record_and_its_scope(self):
        for op, rid in ((N.FMUL, "op3290.at4"), (N.FADD, "op998.at4")):
            with self.subTest(op=op):
                e = N.LEAD_MODIFIER_EVIDENCE[(op, 32)]
                self.assertIn(rid, e["execution_evidence"])
                self.assertIn("4 bytes only", e["scope"])

    def test_the_cited_record_exists_and_carries_32(self):
        """A citation to a record that is not there, or that does not carry the value, is a claim."""
        import json
        recs = {r["id"]: r for r in json.load(open(os.path.join(
            ROOT, "isa", "g17-execution-shortforms2-results.json")))}
        for rid in ("op3290.at4", "op998.at4"):
            with self.subTest(rid=rid):
                self.assertEqual(recs[rid]["status"], "ok")
                self.assertEqual(recs[rid]["length"], 4)


if __name__ == "__main__":
    unittest.main()
