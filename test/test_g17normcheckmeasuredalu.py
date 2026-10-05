"""g17normcheck interprets the MEASURED_ALU opcodes at exactly what was measured, and nothing else."""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import g17normcheck as N
import g17fitfromexecution as FF


class EveryEntryIsADeterminedFunction(unittest.TestCase):

    def test_each_named_function_exists_in_the_fit_library(self):
        for op, (name, nsrc, _ln, _leads) in N.MEASURED_ALU.items():
            with self.subTest(op=op):
                self.assertIn(name, FF.INT2 if nsrc == 2 else FF.FLOAT1)

    def test_each_entry_is_promotable_in_the_fits(self):
        """The admission must rest on a promotion, not a guess: the fits artifact promotes each form."""
        import json
        prom = json.load(open(os.path.join(ROOT, "isa", "g17-execution-fits.json")))["forms_promotable"]
        for op, (name, _nsrc, ln, _leads) in N.MEASURED_ALU.items():
            with self.subTest(op=op):
                if op in N.MEASURED_ALU_EVIDENCE:
                    # admitted on a named record instead - that record must exist and be ok
                    path, rid = N.MEASURED_ALU_EVIDENCE[op].split()
                    recs = {r["id"]: r for r in json.load(open(os.path.join(ROOT, path)))}
                    self.assertEqual(recs[rid]["status"], "ok")
                    continue
                self.assertEqual(prom.get("%d/%d" % (op, ln), {}).get("function"), name)

    def test_op3818_keeps_its_own_scoped_arm(self):
        self.assertNotIn(3818, N.MEASURED_ALU)


if __name__ == "__main__":
    unittest.main()
