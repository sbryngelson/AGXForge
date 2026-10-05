"""What op16842 (simd sum) and op16841 (simd prefix sum) do ACROSS lanes, read on every lane.

The predictions in isa/g17-execution-lanelaws.json were committed before dispatch (e87afaec); every
candidate ordering predicts a different word per lane, so a match names exactly one law.
"""
import json, os, sys, unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import g17oracle

PLAN = os.path.join(ROOT, "isa", "g17-execution-lanelaws.json")
RESULTS = os.path.join(ROOT, "isa", "g17-execution-lanelaws-results.json")


def load(p):
    with open(p) as f:
        return {r["id"]: r for r in json.load(f)}


class LaneLawTests(unittest.TestCase):
    def setUp(self):
        self.plan, self.res = load(PLAN), load(RESULTS)

    def matching(self, rid):
        return [k for k, v in self.plan[rid]["predictions"].items() if v == self.res[rid]["values"]]

    def test_every_prediction_set_discriminates(self):
        # A set where two laws share a prediction cannot name one; checked, not assumed.
        for rid, rec in self.plan.items():
            if "predictions" in rec:
                vals = [tuple(v) for v in rec["predictions"].values()]
                self.assertEqual(len(vals), len(set(vals)), rid)

    def test_the_control_reproduced(self):
        self.assertTrue(all(self.res["CONTROL.op10279"]["match"]))

    def test_simd_sum_is_the_adjacent_pair_tree(self):
        for rid in ("lane.sum.f32.d0", "lane.sum.f32.d1"):
            self.assertEqual(self.matching(rid), ["adjacent"], rid)

    def test_prefix_sum_is_an_exclusive_sklansky_scan_from_negative_zero(self):
        for rid in ("lane.prefix.f32.d0", "lane.prefix.f32.d1"):
            self.assertEqual(self.matching(rid), ["sklansky"], rid)
            self.assertEqual(self.res[rid]["values"][0], 0x80000000, rid)

    def test_fmax_orders_floats_not_bits(self):
        self.assertEqual(self.matching("lane.fmax.f32.neg"), ["float_max"])

    def test_every_lane_was_read(self):
        for rid, rec in self.plan.items():
            if rec.get("per_lane"):
                self.assertEqual(len(self.res[rid]["values"]), 32, rid)
                self.assertIsNone(g17oracle.refusal(rec), rid)


class PerLaneRefusalTests(unittest.TestCase):
    def test_per_lane_refuses_a_mismatched_read_set(self):
        rec = dict(next(r for r in load(PLAN).values() if r.get("per_lane")))
        rec["read_slots"] = rec["read_slots"][:-1]
        self.assertIsNotNone(g17oracle.refusal(rec))

    def test_per_lane_refuses_fewer_than_32_threads(self):
        rec = dict(next(r for r in load(PLAN).values() if r.get("per_lane")), threads=16)
        self.assertIsNotNone(g17oracle.refusal(rec))


class ShuffleTests(unittest.TestCase):
    """isa/g17-execution-perlane-results.json: input on lane L is 4096 + L."""

    def setUp(self):
        self.res = load(os.path.join(ROOT, "isa", "g17-execution-perlane-results.json"))

    def test_shuffle_xor_reads_lane_l_xor_mask(self):
        for mask in (1, 5):
            self.assertEqual(self.res["lane.shufxor.m%d" % mask]["values"],
                             [4096 + (L ^ mask) for L in range(32)])

    def test_the_uninitialised_shuffle_index_reads_lane_zero(self):
        # The index operand is not set by the record, so this is lane 0 of an inherited 0, not a law.
        self.assertEqual(set(self.res["lane.shuffle.default"]["values"]), {4096})


if __name__ == "__main__":
    unittest.main()
