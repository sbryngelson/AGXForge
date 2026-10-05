"""tools/g17adoptcheck.py's pairs, checked offline. Nothing here builds a program or dispatches.

`arms()` is pure over the committed plan; `drift_forms`, `pair_diff` and `score` are pure over their
arguments. These pin what makes the batch worth running: the population is the forms whose EMITTED
bytes adoption changes, every pair's arms differ only at those bits, the positive control can fail
the batch, and a pair that disagrees is reported as a miscompile rather than absorbed.
"""
import json, os, sys, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import g17adoptcheck as T


class ThePopulationIsTheEmittedByteChange(unittest.TestCase):

    def test_emitted_follows_the_encoder(self):
        self.assertEqual(T.emitted(["constant", 1]), 1)
        self.assertEqual(T.emitted(["constant", 0]), 0)
        self.assertEqual(T.emitted(["varies", None]), 0)
        self.assertEqual(T.emitted(["undetermined", None]), 0)

    def test_a_verdict_change_that_emits_the_same_bit_is_not_in_it(self):
        drift = [[1, 4, "0.0", ["undetermined", None], ["constant", 0]],   # 0 -> 0: no change
                 [1, 4, "0.1", ["constant", 0], ["varies", None]],        # 0 -> 0: no change
                 [2, 6, "1.2", ["undetermined", None], ["constant", 1]],  # 0 -> 1
                 [3, 8, "2.2", ["constant", 1], ["varies", None]]]        # 1 -> 0
        self.assertEqual(T.drift_forms(drift),
                         {(2, 6): [("1.2", 0, 1)], (3, 8): [("2.2", 1, 0)]})

    def test_the_committed_plan_is_31_forms_of_108(self):
        """The plan is FROZEN at the batch it describes (43662bff). Since then the 13 cleared
        forms were adopted, so the live drift is the plan minus exactly those - no more, no less."""
        plan = T.load_plan()
        self.assertEqual(plan["drift_forms"], 108)
        self.assertEqual(plan["drift_bits"], 672)
        self.assertEqual(len(plan["forms"]), 31)
        self.assertEqual(sum(len(f["bits"]) for f in plan["forms"]), 36)
        planned = {(f["op"], f["length"]) for f in plan["forms"]}
        self.assertEqual(planned - set(T.drift_forms()), set(T.ADOPTED))
        self.assertEqual(set(T.drift_forms()) - planned, set())

    def test_adopted_is_exactly_the_cleared_pairs(self):
        with open(T.OUT) as fh:
            rows = json.load(fh)["rows"]
        cleared = {(r["op"], r["length"]) for r in rows
                   if not r["control"] and r["verdict"] == "as-predicted"}
        self.assertEqual(cleared, set(T.ADOPTED))
        self.assertEqual(len(T.ADOPTED), 13)
        self.assertNotIn((11482, 14), T.ADOPTED)

    def test_every_unadopted_form_carries_its_reason(self):
        planned = {(f["op"], f["length"]) for f in T.load_plan()["forms"]}
        self.assertEqual(planned - set(T.ADOPTED), set(T.NOT_ADOPTED))
        self.assertIn("LIVE ON HARDWARE", T.NOT_ADOPTED[(11482, 14)])


class TheArms(unittest.TestCase):
    def setUp(self):
        self.arms = T.arms()
        self.by_id = {a["id"]: a for a in self.arms}

    def test_controls_are_present(self):
        for c in ("ctl.op10279.mandatory", "ctl.null.op10279", "ctl.positive.op612"):
            self.assertIn(c, self.by_id)
        self.assertTrue(any(i.startswith("ctl.knowngood.") for i in self.by_id))
        self.assertEqual(self.by_id["ctl.positive.op612"]["predict"], "differ")

    def test_the_positive_control_predictions_differ(self):
        a, b = self.by_id["ctl.positive.op612"]["records"]
        self.assertEqual(a["expect"], [7, 0, 15, 8])     # w612.lo16, already measured
        self.assertEqual(b["expect"], [0, 0, 0, 0])
        self.assertNotEqual(a["expect"], b["expect"])

    def test_every_adoption_pair_predicts_agreement_and_differs_only_at_its_bits(self):
        pairs = [a for a in self.arms if a["id"].startswith("adopt.")]
        self.assertEqual(len(pairs), 31)
        for a in pairs:
            with self.subTest(pair=a["id"]):
                self.assertEqual(a["predict"], "agree")
                if a.get("refused"):
                    continue
                for route in a["routes"]:
                    ra, rb = route["records"]
                    x = bytes(p ^ q for p, q in zip(bytes.fromhex(ra["bytes"]),
                                                    bytes.fromhex(rb["bytes"])))
                    self.assertEqual(x, T.adopted_mask(a["length"], a["adopted_bits"]))
                    self.assertEqual(ra["cases"], rb["cases"])
                    self.assertEqual(len(bytes.fromhex(ra["bytes"])), a["length"])

    def test_memory_forms_are_refused_not_dispatched(self):
        refused = sorted(a["op"] for a in self.arms if a.get("refused"))
        self.assertEqual(refused, [445, 9328, 10022, 10090, 11769])
        for a in self.arms:
            if a.get("refused"):
                self.assertIn("safety gate", a["refused"])

    def test_inputs_are_distinct_per_source_and_never_the_sentinel(self):
        for n in range(1, 6):
            for case in T.cases(n):
                self.assertEqual(len(set(case)), n)
                self.assertNotIn(0xDEADBEEF, case)


class PairDiffCanFail(unittest.TestCase):
    """The offline pair check must refuse each way a pair can be dirty."""
    A = bytes.fromhex("aa" * 4 + "2700201a" + "bb" * 4)
    ENC = "2700201a"

    def _b(self, code):
        return code

    def test_a_clean_pair_passes(self):
        b = bytearray(self.A); b[4 + 3] ^= 0x02                       # bit 3.1 of the instruction
        self.assertIsNone(T.pair_diff(self.A, bytes(b), [self.ENC], ["2700203a"[:6] + "18"],
                                      4, ["3.1"], [4]))

    def test_a_difference_outside_the_instruction_is_refused(self):
        b = bytearray(self.A); b[4 + 3] ^= 0x02; b[0] ^= 1
        self.assertIn("outside the adopted bits",
                      T.pair_diff(self.A, bytes(b), [self.ENC], ["27002018"], 4, ["3.1"], [4]))

    def test_a_wrong_instruction_mask_is_refused(self):
        self.assertIn("not the adopted mask",
                      T.pair_diff(self.A, self.A, [self.ENC], ["2700201b"], 4, ["3.1"], [4]))

    def test_different_lengths_are_refused(self):
        self.assertIsNotNone(T.pair_diff(self.A, self.A + b"\x00", [self.ENC], [self.ENC],
                                         4, ["3.1"], [4]))


class ScoreNamesTheMiscompile(unittest.TestCase):
    PAIR = dict(id="adopt.x", kind="pair", control=False, predict="agree",
                records=[dict(id="adopt.x.A"), dict(id="adopt.x.B")])

    def _res(self, a, b):
        return {"adopt.x.A": dict(status="ok", values=a), "adopt.x.B": dict(status="ok", values=b)}

    def test_agreement_with_a_varying_output_clears(self):
        self.assertEqual(T.score(self.PAIR, self._res([1, 2], [1, 2]))["verdict"], "as-predicted")

    def test_disagreement_is_a_miscompile(self):
        self.assertEqual(T.score(self.PAIR, self._res([1, 2], [1, 3]))["verdict"],
                         "ADOPTION-WOULD-MISCOMPILE")

    def test_a_constant_output_does_not_clear(self):
        self.assertEqual(T.score(self.PAIR, self._res([5, 5], [5, 5]))["verdict"],
                         "agree-but-uninformative")

    def test_a_positive_control_that_agrees_voids_the_batch(self):
        ctl = dict(self.PAIR, control=True, predict="differ")
        self.assertEqual(T.score(ctl, self._res([7, 0], [7, 0]))["verdict"], "CONTROL-FAILED")

    def test_a_failed_run_is_no_result(self):
        r = {"adopt.x.A": dict(status="fault"), "adopt.x.B": dict(status="ok", values=[1])}
        self.assertEqual(T.score(self.PAIR, r)["verdict"], "no-result")


if __name__ == "__main__":
    unittest.main()
