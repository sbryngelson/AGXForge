import json
import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ISA = os.path.join(ROOT, "isa")


class DriftedRecordsReproduceTheirValues(unittest.TestCase):
    """isa/g17-execution-redispatch (2026-09-23): 20 drifted records re-run on the GPU - 14 that differ
    only in registers, 6 whose destination modifier or source lifetime changed - and every one
    returned its retained values word for word. The refusal in g17oracle stays: the bytes differ,
    and a sample is not the population; what this establishes is that drift in the sample did not
    change what was measured."""

    def test_every_redispatched_record_matches_its_retained_values(self):
        res = {r["id"]: r for r in json.load(open(os.path.join(ISA, "g17-execution-redispatch-results.json")))}
        plan = json.load(open(os.path.join(ISA, "g17-execution-redispatch.json")))
        checked = 0
        for p in plan[1:]:
            _, batch, orig = p["id"].split(".", 2)
            old = {r["id"]: r for r in json.load(open(os.path.join(ISA, "g17-execution-%s-results.json" % batch)))}[orig]
            new = res[p["id"]]
            self.assertEqual(new.get("status"), "ok", p["id"])
            self.assertEqual(new.get("values"), old.get("values"), p["id"])
            checked += 1
        self.assertEqual(checked, 20)

    def test_the_control_reproduced(self):
        res = {r["id"]: r for r in json.load(open(os.path.join(ISA, "g17-execution-redispatch-results.json")))}
        plan = {p["id"]: p for p in json.load(open(os.path.join(ISA, "g17-execution-redispatch.json")))}
        self.assertEqual(res["CONTROL.op10279"]["values"], plan["CONTROL.op10279"]["expect"])


if __name__ == "__main__":
    unittest.main()
