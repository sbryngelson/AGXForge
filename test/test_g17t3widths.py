"""Guards for the T3 width re-ask.

The headline is a NEGATIVE - three blockers survive a larger population - and a negative from a
search is only as good as the search's ability to find something. So these mostly pin that the
instrument does find rules when they exist, and that it excludes the bits which would let it find
fake ones.
"""
import json, os, sys, unittest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
RECORD = os.path.join(ROOT, "tools", "tensorops-model", "t3-width-admission.json")


def _r():
    with open(RECORD, encoding="utf-8") as fh:
        return json.load(fh)


class TheSearchCanFindRules(unittest.TestCase):
    """Without this, 'the blocker stands' is indistinguishable from a broken search."""

    def test_exact_rules_were_found_for_the_predicted_widths(self):
        b = _r()["blockers"]
        found = [(k, w) for k, v in b.items() if isinstance(v.get("exact_rules"), dict)
                 for w in v["exact_rules"]]
        self.assertTrue(found, "no exact rule anywhere; the separator cannot find rules at all")

    def test_op11842_width_two_is_a_single_bit(self):
        r = _r()["blockers"]["11842"]["exact_rules"]["2"]
        self.assertEqual(len(r["bits"]), 1)
        self.assertGreater(r["instances"], 1000)


class ConstantBitsAreExcluded(unittest.TestCase):
    """A constant bit 'isolates' every width and would be a fact about the opcode."""

    def test_constant_bits_are_recorded_and_not_used(self):
        for k, v in _r()["blockers"].items():
            if "constant_bits" not in v:
                continue
            with self.subTest(opcode=k):
                const = set(v["constant_bits"])
                self.assertTrue(const, "no constant bits found for op%s - suspect the scan" % k)
                for w, rule in v["exact_rules"].items():
                    self.assertFalse(const & set(rule["bits"]),
                                     "width %s rule uses a constant bit" % w)


class TheBlockersStand(unittest.TestCase):

    def test_the_retried_blockers_still_stand(self):
        for k, v in _r()["blockers"].items():
            if v.get("status") in ("closed", "stands"):
                with self.subTest(opcode=k):
                    self.assertEqual(v["status"], "stands",
                                     "op%s now closes - update section 25.62" % k)

    def test_op11842_was_retried_at_a_much_larger_population(self):
        v = _r()["blockers"]["11842"]
        self.assertGreater(v["vendor_instances"], 10 * v["old_instances"])


class TheWitnessCountsMoved(unittest.TestCase):

    def test_most_of_the_unwitnessed_forms_now_have_a_witness(self):
        r = _r()
        self.assertEqual(r["unwitnessed_before"], 173)
        self.assertGreater(len(r["newly_witnessed"]), 50)
        self.assertEqual(len(r["newly_witnessed"]) + len(r["still_unwitnessed"]), 173)

    def test_the_named_contradictions_are_all_witnessed(self):
        for form, v in _r()["named_contradictions"].items():
            with self.subTest(form=form):
                self.assertGreater(v["instances"], 0)


if __name__ == "__main__":
    unittest.main()
