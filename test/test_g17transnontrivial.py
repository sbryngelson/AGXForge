import json
import os
import struct
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import g17fitfromexecution as X
import g17normcheck as N


def _entries():
    d = json.load(open(os.path.join(ROOT, "isa", "g17-execution-fits.json")))
    ta = d["transcendental_accuracy"]
    return ta if isinstance(ta, list) else list(ta.values())


class NonTrivialDenominator(unittest.TestCase):
    """exp2 read 39 of 46 exact; 38 of the 46 could not tell rounding apart."""

    def test_exp2_f32_agrees_with_the_independent_count(self):
        e = [x for x in _entries() if x["name"] == "exp2" and x["destination_width"] == 32][0]
        # THE SAME TRIVIALITY RULE ON BOTH SIDES. normcheck drops integer inputs; the fits drop any
        # input whose true value is within TRIVIAL_MARGIN_ULP of a representable float. With the eight
        # marginpairs inputs alone the two rules chose the same cases; the dense sweep (transsweep,
        # 2026-09-23) has eight fractional inputs that land within 1/64 ulp, so the rules now differ
        # by exactly those eight and the comparison must apply the fits' rule to normcheck's rows.
        f = lambda b: struct.unpack("<f", struct.pack("<I", b))[0]
        rows = [r for r in N.isolated_rounding(1272)
                if not abs(2.0 ** r[2] - f(r[4])) < abs(f(r[4] + 1) - f(r[4])) * X.TRIVIAL_MARGIN_ULP]
        self.assertEqual((e["interior_nontrivial_exact"], e["interior_nontrivial_cases"]),
                         (sum(1 for r in rows if r[5] == 0), len(rows)))
        self.assertEqual((e["interior_nontrivial_exact"], e["interior_nontrivial_cases"]), (164, 256))

    def test_integer_inputs_are_trivial_and_fractions_are_not(self):
        f = lambda x: struct.unpack("<I", struct.pack("<f", x))[0]
        plans = {("s", "r"): {"cases": [[f(1.0)], [f(2.0)], [f(0.5)], [f(1.5)]]}}
        values = {("s", "r"): [f(2.0), f(4.0), f(2 ** 0.5), f(2 ** 1.5)]}
        pack = dict(X.DEST_READINGS_32)["f32"]
        unpack = dict(X.SOURCE_READINGS_32)["f32"]
        got = X._score_transcendental([("s", "r")], "exp2", unpack, pack, 32, plans, values, "f32")
        self.assertEqual(got["interior_cases"], 4)
        self.assertEqual(got["interior_nontrivial_cases"], 2)

    def test_the_verdict_names_the_nontrivial_pair(self):
        e = dict(interior_cases=46, interior_exact=39, interior_max_ulp=1, interior_nontrivial_cases=8,
                 interior_nontrivial_exact=1, source_reading="f32", destination_reading="f32",
                 input_denormal_flush_determined=True)
        self.assertIn("1 of the 8 cases that can tell rounding apart", X._transcendental_verdict(e))


if __name__ == "__main__":
    unittest.main()
