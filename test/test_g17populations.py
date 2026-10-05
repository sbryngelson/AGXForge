"""Guards for the published coverage populations.

T6 and T7 were both blocked because `isa/g17-coverage.json` published its populations as counts with
no lists. These pin that the lists exist, that each agrees with the count it belongs to, and - the
one that matters - that the never-dispatched tail is a SET DIFFERENCE and not a subtraction, because
the subtraction is what made section 25.65 publish 221 for a 388-form tail.
"""
import json, os, unittest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
COVERAGE = os.path.join(ROOT, "isa", "g17-coverage.json")
NEEDED = ("d2_encoded", "dispatched", "executed_unchecked",
          "executed_with_no_semantic_column", "executed_with_no_LOCAL_semantic_column",
          "encoded_never_dispatched")


def _forms():
    with open(COVERAGE, encoding="utf-8") as fh:
        return json.load(fh)["forms"]


class EveryCountHasItsPopulation(unittest.TestCase):

    def test_each_population_is_published(self):
        pops = _forms()["populations"]
        for k in NEEDED:
            with self.subTest(population=k):
                self.assertIn(k, pops)
                self.assertTrue(pops[k], "%s is published but empty" % k)

    def test_each_population_agrees_with_its_count(self):
        """A list that drifts from its own count is worse than no list: it reads as auditable."""
        agree = _forms()["populations_agree_with_totals"]
        for k, ok in sorted(agree.items()):
            with self.subTest(population=k):
                self.assertTrue(ok, "%s's list disagrees with its total" % k)

    def test_form_ids_are_opcode_slash_width(self):
        """The first version published bare widths - ['len2', 'len4'] - with the opcode dropped."""
        for k, forms in _forms()["populations"].items():
            with self.subTest(population=k):
                for f in forms[:20]:
                    self.assertRegex(f, r"^\d+/\d+$", "%r is not opcode/width" % f)


class TheTailIsASetDifferenceNotASubtraction(unittest.TestCase):
    """d2_encoded - dispatched assumes a subset relation that does not hold."""

    def test_the_tail_matches_the_set_difference(self):
        f = _forms()
        pops = f["populations"]
        expected = set(pops["d2_encoded"]) - set(pops["dispatched"])
        self.assertEqual(len(pops["encoded_never_dispatched"]), len(expected))
        self.assertEqual(set(pops["encoded_never_dispatched"]), expected)

    def test_the_naive_subtraction_would_be_wrong(self):
        """Pinned so nobody restores the arithmetic: dispatched is NOT inside d2_encoded."""
        f = _forms()
        t, pops = f["totals"], f["populations"]
        naive = t["d2_encoded"] - t["dispatched"]
        real = len(pops["encoded_never_dispatched"])
        self.assertNotEqual(naive, real,
                            "the subtraction now agrees with the set difference; if dispatched has "
                            "become a subset of d2_encoded, section 25.74's argument needs revisiting")
        self.assertTrue(set(pops["dispatched"]) - set(pops["d2_encoded"]),
                        "no dispatched form is outside d2_encoded, so the subtraction would be valid")


class TheIntersectionThatT6NeededIsPossible(unittest.TestCase):

    def test_the_checked_forms_can_be_intersected_with_the_unchecked_population(self):
        checked = set(json.load(open(
            os.path.join(ROOT, "tools", "tensorops-model", "t6-checked-forms.json"),
            encoding="utf-8"))["forms"])
        unchecked = set(_forms()["populations"]["executed_unchecked"])
        overlap = checked & unchecked
        self.assertGreater(len(overlap), 50,
                           "the intersection collapsed; section 25.74 quotes 75")


if __name__ == "__main__":
    unittest.main()
