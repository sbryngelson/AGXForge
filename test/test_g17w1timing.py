"""Guards for W1's simdgroup timing answer.

A timing claim is only worth the margin it clears, so the run-to-run spread is asserted to be small
beside the effect rather than left for a reader to eyeball. The comparison must also stay a
comparison: if the sweep ever collapses to one simdgroup count, or stops repeating each point, the
answer is unsupported and this file should say so before the row does.
"""
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, ROOT)

EVIDENCE = os.path.join(ROOT, "tools", "tensorops-model", "w1-simdgroup-timing.json")
SCHEDULES = ("fused", "two", "three")


def _load():
    with open(EVIDENCE) as fh:
        return json.load(fh)


def _by_nsg(doc):
    return {p["nsg"]: p for p in doc["points"]}


class TheSweepIsAComparison(unittest.TestCase):

    def setUp(self):
        self.doc = _load()
        self.by = _by_nsg(self.doc)

    def test_more_than_one_simdgroup_count_was_measured(self):
        self.assertGreaterEqual(len(self.by), 3, "one point cannot answer a comparative question")
        self.assertTrue({16, 32} <= set(self.by))

    def test_every_point_was_repeated(self):
        self.assertGreaterEqual(self.doc["repeats"], 2)
        for nsg, p in self.by.items():
            self.assertGreaterEqual(len(p["runs"]), 2, "nsg=%d" % nsg)

    def test_only_the_simdgroup_count_varied(self):
        fixed = self.doc["fixed"]
        for key in ("fmt", "d", "Hh", "ntg", "sets", "mt", "epi", "share"):
            self.assertIn(key, fixed)
        self.assertNotIn("nsg", fixed, "the swept variable must not also be pinned")


class TheEffectClearsTheNoise(unittest.TestCase):

    def setUp(self):
        self.doc = _load()
        self.by = _by_nsg(self.doc)

    def test_the_spread_is_recorded_for_every_point(self):
        for nsg, p in self.by.items():
            for s in SCHEDULES:
                self.assertIn(s, p["spread_ms"], "nsg=%d %s" % (nsg, s))

    def test_the_difference_is_many_times_the_spread(self):
        for s in SCHEDULES:
            a, b = self.by[16], self.by[32]
            effect = abs(b["best_ms"][s] - a["best_ms"][s])
            noise = max(a["spread_ms"][s], b["spread_ms"][s])
            self.assertGreater(effect, 5 * noise,
                               "%s: effect %.3f against spread %.3f" % (s, effect, noise))

    def test_the_conclusion_records_the_margin_check(self):
        self.assertTrue(self.doc["conclusion"]["effect_exceeds_spread"])


class TheAnswer(unittest.TestCase):

    def setUp(self):
        self.doc = _load()
        self.by = _by_nsg(self.doc)

    def test_thirty_two_is_not_faster(self):
        self.assertFalse(self.doc["conclusion"]["thirty_two_is_faster"])

    def test_thirty_two_is_slower_on_every_schedule(self):
        for s in SCHEDULES:
            self.assertGreater(self.by[32]["best_ms"][s], self.by[16]["best_ms"][s], s)
            self.assertGreater(self.doc["conclusion"]["thirty_two_ratio_to_sixteen"][s], 1.0, s)

    def test_sixteen_is_fastest_on_the_best_schedule(self):
        self.assertEqual(self.doc["conclusion"]["fastest_simdgroup_count"], 16)
        self.assertEqual(min(self.by, key=lambda n: self.by[n]["best_ms"]["two"]), 16)

    def test_the_sets_caveat_is_recorded(self):
        # the configuration rule 33 names used SETS=3 and is no longer runnable as written
        self.assertEqual(self.doc["fixed"]["sets"], 4)
        self.assertIn("power of two", self.doc["note"])
