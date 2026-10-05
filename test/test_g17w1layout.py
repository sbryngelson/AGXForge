"""Guards for W1's layout precondition.

What this evidence licenses is a decision about a dispatch that previously rebooted the machine, so
the cases below are stricter than the finding needs. The precondition is only meaningful if the
guard behind it can fail, so `assert_fits` refusing one byte less is asserted as hard as the fit
itself. And the evidence must keep recording that nothing was dispatched: if it is ever regenerated
by something that runs kernels, this file should go red rather than quietly bless it.
"""
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, ROOT)

EVIDENCE = os.path.join(ROOT, "tools", "tensorops-model", "w1-layout-precondition.json")


def _load():
    with open(EVIDENCE) as fh:
        return json.load(fh)


class NothingWasRun(unittest.TestCase):

    def test_the_evidence_records_no_dispatch_and_no_build(self):
        d = _load()
        self.assertFalse(d["dispatched"], "rule 33's configuration must not be dispatched to produce this")
        self.assertFalse(d["built_kernels"])

    def test_it_is_the_configuration_rule_33_names(self):
        c = _load()["configuration"]
        self.assertEqual(c["fmt"], "bf16")
        self.assertEqual((c["d"], c["Hh"], c["nsg"], c["mt"]), (2048, 8192, 32, 2))
        self.assertEqual(c["share"], "dev")


class TheGuardCanFail(unittest.TestCase):

    def setUp(self):
        self.doc = _load()

    def test_assert_fits_refuses_an_allocation_one_byte_smaller(self):
        self.assertTrue(self.doc["assert_fits_refuses_one_byte_less"],
                        "a guard that accepts anything cannot establish the precondition")

    def test_assert_fits_accepts_the_allocation_the_driver_makes(self):
        self.assertTrue(self.doc["assert_fits_accepts_correct_allocation"])


class ThePrecondition(unittest.TestCase):

    def setUp(self):
        self.doc = _load()

    def test_no_kernel_overflows_the_allocation(self):
        self.assertLessEqual(self.doc["overflow"], 0)
        for k in self.doc["kernels"]:
            self.assertLessEqual(k["out_bytes"], self.doc["allocation"],
                                 "%s %s" % (k["variant"], k["part"]))

    def test_the_allocation_is_the_largest_layout_not_the_cut_one(self):
        # the defect was sizing from a sibling's layout; the fix is sizing from the maximum
        self.assertEqual(self.doc["allocation"], self.doc["fused_out_bytes"])
        self.assertGreater(self.doc["fused_out_bytes"], self.doc["cut_out_bytes"])

    def test_the_omitted_region_is_recorded_and_independent_of_sets(self):
        self.assertEqual(self.doc["omitted_region_bytes"],
                         self.doc["fused_out_bytes"] - self.doc["cut_out_bytes"])
        self.assertGreater(self.doc["omitted_region_bytes"], 0)
        self.assertTrue(self.doc["omitted_region_independent_of_sets"])
        gaps = set(self.doc["omitted_region_by_sets"].values())
        self.assertEqual(len(gaps), 1)

    def test_the_precondition_is_recorded_as_met(self):
        self.assertTrue(self.doc["precondition_met"])

    def test_the_fused_variant_is_present(self):
        # the defect is specifically that the fused device-shared kernel has a region the others lack
        variants = {k["variant"] for k in self.doc["kernels"]}
        self.assertIn("fused", variants)
        self.assertTrue(variants - {"fused"}, "need the cut schedules too, or there is no comparison")
