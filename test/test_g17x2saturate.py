"""Guards for X2's dispatched answer.

The finding rests on a patched instruction having executed, and the first version of this experiment
reported a real patch as inert because Metal served a cached pipeline. So the control that proves the
patch reached execution is asserted as hard as the result, and if it ever stops discriminating this
file should go red before anything is read from the saturation arm.

The in-range arm is the other load-bearing control: it shows the bit changes only overflow behaviour
rather than perturbing the arithmetic, which is what makes "the patched run differs" mean saturation
instead of damage.
"""
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, ROOT)

EVIDENCE = os.path.join(ROOT, "tools", "tensorops-model", "x2-saturation-under-transpose.json")


def _load():
    with open(EVIDENCE) as fh:
        return json.load(fh)


def _role(doc, role):
    for r in doc["runs"]:
        if r["role"] == role:
            return r
    raise AssertionError("no run with role %s" % role)


class TheReferencesActuallyDiffer(unittest.TestCase):

    def setUp(self):
        self.doc = _load()

    def test_the_input_overflows_int32(self):
        self.assertGreater(self.doc["exact_sum"], 2 ** 31 - 1,
                           "an in-range sum would make the two references agree and the test unfailable")

    def test_the_two_references_are_far_apart(self):
        self.assertNotEqual(self.doc["wrap_reference"], self.doc["saturating_reference"])
        self.assertGreater(abs(self.doc["saturating_reference"] - self.doc["wrap_reference"]), 2 ** 31)

    def test_the_wrap_reference_is_the_exact_sum_modulo_2_32(self):
        self.assertEqual(self.doc["wrap_reference"], self.doc["exact_sum"] - 2 ** 32)
        self.assertEqual(self.doc["saturating_reference"], 2 ** 31 - 1)


class ThePatchReachedExecution(unittest.TestCase):
    """Without this the saturation arm proves nothing about the bit."""

    def setUp(self):
        self.doc = _load()

    def test_the_control_pair_differs(self):
        a = _role(self.doc, "patch_reaches_execution_unpatched")
        b = _role(self.doc, "patch_reaches_execution_patched")
        self.assertEqual(a["status"], "ok")
        self.assertEqual(b["status"], "ok")
        self.assertNotEqual((a["first"], a["distinct"]), (b["first"], b["distinct"]),
                            "a patch that changes nothing means the dispatch ran stale code")

    def test_the_conclusion_records_it(self):
        self.assertTrue(self.doc["conclusion"]["patch_reaches_execution"])

    def test_the_control_actually_patched_instructions(self):
        b = _role(self.doc, "patch_reaches_execution_patched")
        self.assertGreater(b["patched"], 0)
        a = _role(self.doc, "patch_reaches_execution_unpatched")
        self.assertEqual(a["patched"], 0)
        self.assertNotEqual(a["byte8"], b["byte8"], "the bytes on disk must differ")


class TheResult(unittest.TestCase):

    def setUp(self):
        self.doc = _load()

    def test_transposed_wraps_without_the_bit(self):
        r = _role(self.doc, "transposed_unpatched")
        self.assertEqual(r["status"], "ok")
        self.assertTrue(r["all_wrap"])
        self.assertFalse(r["all_sat"])
        self.assertEqual(r["distinct"], 1)

    def test_transposed_saturates_with_the_bit(self):
        r = _role(self.doc, "transposed_saturating")
        self.assertEqual(r["status"], "ok")
        self.assertTrue(r["all_sat"])
        self.assertFalse(r["all_wrap"])
        self.assertEqual(r["distinct"], 1)
        self.assertGreater(r["patched"], 0)

    def test_every_output_agrees_not_just_a_sample(self):
        for role in ("transposed_unpatched", "transposed_saturating"):
            self.assertGreaterEqual(_role(self.doc, role)["outputs"], 1024)


class TheBoundingArms(unittest.TestCase):

    def setUp(self):
        self.doc = _load()

    def test_the_bit_saturates_without_transpose_too(self):
        self.assertTrue(_role(self.doc, "untransposed_unpatched")["all_wrap"])
        self.assertTrue(_role(self.doc, "untransposed_saturating")["all_sat"])

    def test_an_in_range_sum_is_unaffected_by_the_bit(self):
        a = _role(self.doc, "in_range_unpatched")
        b = _role(self.doc, "in_range_saturating")
        self.assertEqual(a["digest"], b["digest"],
                         "if the bit moved an in-range result it is not a saturation selector")
        self.assertGreater(b["patched"], 0, "the in-range arm must actually be patched")

    def test_the_overall_conclusion_needs_all_four_arms(self):
        c = self.doc["conclusion"]
        self.assertTrue(c["saturation_applies_under_transpose"])
        for k in ("patch_reaches_execution", "transposed_wraps_unpatched",
                  "transposed_saturates_patched", "in_range_unaffected"):
            self.assertTrue(c[k], k)
