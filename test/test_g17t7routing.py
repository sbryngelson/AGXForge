"""Guards for the T7 routing result and the enumerated-claim checks.

Two claims in `isa/g17-coverage.json` are refuted here, so these pin that the refutations rest on
witnesses actually found, and that the tool still reports the tail it cannot enumerate rather than
quietly omitting it.
"""
import json, os, unittest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RECORD = os.path.join(ROOT, "tools", "tensorops-model", "t7-routing.json")
ABSENT = "named_unknowns.tier1_enumerable.forms_absent_from_apples_corpus_entirely.forms"
NOCORPUS = "named_unknowns.tier1_enumerable.forms_whose_undetermined_bits_no_corpus_can_settle.forms"


def _r():
    with open(RECORD, encoding="utf-8") as fh:
        return json.load(fh)


class TheTailIsAPublishedPopulation(unittest.TestCase):
    """This class replaces `TheTailIsDefinedArithmetically`, whose premise was refuted.

    Both of its tests went red the day the situation IMPROVED, and one of them named its own
    repair in its failure message: "a large list is published now - the tail may be enumerable;
    re-read 25.65".  A guard that fires because its subject got better is a guard written around a
    temporary limitation, so the limitation is what gets deleted, not the guard.  What is pinned
    now is the CURRENT model and, specifically, that the two ways of counting the tail disagree.
    """

    def test_the_tail_comes_from_the_enumerated_population(self):
        r = _r()
        self.assertGreater(r["never_dispatched"], 0)
        self.assertIn("enumerated", r["never_dispatched_from"])
        self.assertEqual(r["never_dispatched"],
                         r["published_lists"]["forms.populations.encoded_never_dispatched"],
                         "the tail must BE the published list's length, not a number beside it")

    def test_the_subtraction_is_kept_and_disagrees(self):
        """The whole finding. If these ever coincide, the disagreement needs re-deriving."""
        r = _r()
        self.assertEqual(r["never_dispatched_by_subtraction"],
                         r["totals"]["d2_encoded"] - r["totals"]["dispatched"])
        self.assertNotEqual(r["never_dispatched"], r["never_dispatched_by_subtraction"],
                            "d2_encoded - dispatched assumes dispatched forms are a subset of "
                            "encoded ones; it is not, and the tail is larger than the "
                            "subtraction. Equality here means one of the populations moved")
        self.assertGreater(r["never_dispatched"], r["never_dispatched_by_subtraction"],
                           "the subtraction understates the tail, which is the direction that "
                           "made 221 publishable in the first place")

    def test_the_tail_is_enumerable_now(self):
        """The inverse of the old test: a list for the tail must be published, and be large."""
        sizes = _r()["published_lists"]
        self.assertTrue(sizes, "no lists at all; the scan is broken")
        self.assertGreaterEqual(sizes["forms.populations.encoded_never_dispatched"], 100,
                                "T6 and T7 were one repair: publish the population. If this "
                                "drops below 100 the artifact has stopped publishing it")


class TheRefutationsRestOnWitnesses(unittest.TestCase):

    def test_the_absent_form_is_witnessed(self):
        c = _r()["enumerated_claim_checks"][ABSENT]
        self.assertEqual(c["witnessed"], c["total"])
        self.assertTrue(c["examples"])
        self.assertTrue(all(v > 0 for v in c["examples"].values()))

    def test_most_of_the_unsettleable_forms_are_witnessed(self):
        c = _r()["enumerated_claim_checks"][NOCORPUS]
        self.assertGreater(c["witnessed"], 0.8 * c["total"])

    def test_not_every_list_is_fully_witnessed(self):
        """A control: if EVERY form matched, suspect the matcher rather than the claims."""
        checks = _r()["enumerated_claim_checks"]
        self.assertTrue(any(c["witnessed"] < c["total"] for c in checks.values()),
                        "every listed form is witnessed - that is too clean; suspect the lookup")


if __name__ == "__main__":
    unittest.main()
