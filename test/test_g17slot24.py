"""The atomic LD_MD declaration (t136 slot 24): the retained dispatch says it is tolerated when absent."""
import json, os, unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REC = json.load(open(os.path.join(ROOT, "isa", "g17-slot24-receipt.json")))


class Slot24(unittest.TestCase):
    def test_the_arms_differ_in_exactly_the_four_declaration_bytes(self):
        self.assertEqual(REC["declaration_bytes"], [40, 48, 102, 154])
        self.assertEqual(REC["object_bytes_differing"], 4)

    def test_the_control_is_exact_every_run(self):
        runs = REC["arms"]["control"]["runs"]
        self.assertEqual(len(runs), 3)
        for r in runs:
            self.assertEqual(r["counter"], REC["fill"] + REC["lanes"])
            self.assertEqual(sorted(r["olds"]), [REC["fill"] + i for i in range(REC["lanes"])])

    def test_the_removed_arm_is_exact_too_so_the_declaration_is_tolerated(self):
        self.assertTrue(all(r["exact"] for r in REC["arms"]["removed"]["runs"]))
        self.assertEqual(REC["verdict"], "tolerated")

    def test_the_pass_condition_can_fail(self):
        # a lost update or a non-atomic add leaves a duplicate old value and a short counter
        olds = [REC["fill"] + i for i in range(REC["lanes"] - 1)] + [REC["fill"]]
        self.assertNotEqual(sorted(olds), [REC["fill"] + i for i in range(REC["lanes"])])


if __name__ == "__main__":
    unittest.main()
