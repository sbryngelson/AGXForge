"""Guards for the T5 framing result.

The headline refutes a standing claim, so the tests care most about the ways a framing comparison
can agree for the wrong reason: an empty comparison, a decoder compared against itself, or a sample
that never reaches the classes in dispute.
"""
import json, os, sys, unittest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RECORD = os.path.join(ROOT, "tools", "tensorops-model", "t5-framing-agreement.json")
DISPUTED = ("0x2", "0x5", "0x8", "0x9", "0xa", "0xe")


def _r():
    with open(RECORD, encoding="utf-8") as fh:
        return json.load(fh)


class TheComparisonActuallyHappened(unittest.TestCase):
    """Agreement counts nothing if Apple's decoder produced nothing to compare against."""

    def test_apple_emitted_something_for_every_sampled_body(self):
        r = _r()
        self.assertEqual(r["apple_empty"], 0,
                         "%d bodies had no Apple decode; those agreements are vacuous"
                         % r["apple_empty"])

    def test_a_real_sample_was_compared(self):
        self.assertGreaterEqual(_r()["sampled"], 100)

    def test_every_body_was_framed_for_the_self_consistency_arm(self):
        r = _r()
        self.assertEqual(r["framed_exactly"], r["bodies"])
        self.assertEqual(r["not_consumed"], 0)
        self.assertEqual(r["raised"], 0)


class TheDisputedClassesWereReached(unittest.TestCase):
    """A sample that never contains class 0x5 cannot refute a claim about class 0x5."""

    def test_each_disputed_class_appears_in_the_agreeing_bodies(self):
        d = {k.lower(): v for k, v in _r()["disputed_class_instructions"].items()}
        for c in DISPUTED:
            with self.subTest(cls=c):
                self.assertGreater(d.get(c, 0), 0,
                                   "class %s never appeared, so it is untested here" % c)

    def test_the_coverage_is_substantial(self):
        self.assertGreater(_r()["disputed_total"], 10000)


class TheResultIsAgreement(unittest.TestCase):

    def test_no_body_disagreed(self):
        r = _r()
        self.assertEqual(r["disagree"], 0, "disagreements found: %r" % r["examples"])
        self.assertEqual(r["agree"], r["sampled"])

    def test_the_verdict_records_the_scope(self):
        """Framing, not operand meaning - the scope is part of the claim."""
        self.assertIn("disputed classes", _r()["verdict"])


if __name__ == "__main__":
    unittest.main()
