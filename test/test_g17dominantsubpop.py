"""Guards for the dominant-sub-population witnesses (op2190/8, op3290/6).

Both forms were refused because every source control reached only a minority of the form. These
pin that the refusal's premise is now false, and pin the thing that made it answerable: operand 1
is a one-hot field with a zero case, not a boolean "bit 24".

Read from the retained record rather than re-compiling: the witnesses come from the Metal compiler,
which is slow and environment-dependent, and a test that silently skips when it is unavailable
would report the same green as one that checked.
"""
import json, os, sys, unittest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

RECORD = os.path.join(ROOT, "tools", "tensorops-model", "dominant-subpopulation.json")


def _record():
    with open(RECORD, encoding="utf-8") as fh:
        return json.load(fh)


class EveryControlReachesItsTarget(unittest.TestCase):

    def test_each_control_reached_the_sub_population_it_names(self):
        for tag, d in sorted(_record().items()):
            with self.subTest(control=tag):
                self.assertTrue(d["reached"],
                                "%s no longer reaches operand1=%d on op%d/%d; the refusal it "
                                "retired would stand again" % (tag, d["target_operand1"],
                                                               d["opcode"], d["length"]))

    def test_the_two_refused_forms_are_both_covered(self):
        covered = {(d["opcode"], d["length"]) for d in _record().values() if d["reached"]}
        self.assertIn((2190, 8), covered)
        self.assertIn((3290, 6), covered)

    def test_the_witnesses_carry_no_high_code(self):
        """The whole point: operand 1 with no high bit, which is Apple's dominant value."""
        for tag, d in sorted(_record().items()):
            for run in d["runs"]:
                if run.get("status") != "ok":
                    continue
                for v in run["operand1_values"]:
                    with self.subTest(control=tag, operand1=v):
                        self.assertEqual(v >> 24, 0, "a high code leaked into the witness set")


class TheControlsAreNotVacuous(unittest.TestCase):
    """A run that compiled nothing would report no failures and no witnesses."""

    def test_every_control_compiled_at_least_one_size(self):
        for tag, d in sorted(_record().items()):
            with self.subTest(control=tag):
                self.assertTrue([r for r in d["runs"] if r.get("status") == "ok"],
                                "%s never compiled; its 'reached' verdict would be empty" % tag)

    def test_at_least_one_witness_carries_a_register_above_r63(self):
        """Register pressure is the mechanism claimed; if none is high, the explanation is wrong."""
        high = [e for d in _record().values() for r in d["runs"] if r.get("status") == "ok"
                for e in r["examples"] if e["above_r63"]]
        self.assertTrue(high, "no witness ran above R63, so pressure is not what reached these")

    def test_the_uniform_control_supplies_the_expr_operand(self):
        """90.9% of op3290/6's high=none population carries an expr operand."""
        d = _record()["mul-uniform"]
        exprs = [e for r in d["runs"] if r.get("status") == "ok"
                 for e in r["examples"] if e["has_expr"]]
        self.assertTrue(exprs, "the uniform multiplicand must produce the expr operand")


class OperandOneIsAFieldNotABoolean(unittest.TestCase):
    """The reframing that made the question answerable, pinned so it is not lost again."""

    def test_the_target_values_are_the_zero_code_not_a_cleared_bit(self):
        for tag, d in sorted(_record().items()):
            with self.subTest(control=tag):
                self.assertIn(d["target_operand1"], (0, 32),
                              "the targets are the low flag with NO high code; a value with a high "
                              "code would be the minority case the refusal already had")


if __name__ == "__main__":
    unittest.main()
