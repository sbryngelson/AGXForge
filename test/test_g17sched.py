"""The scalar timing model and the first scheduler: what may move, what may not, and the model's classes."""
import os, sys, unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from agxforge.g17 import ir, sched, schedmodel, cc


def two_chains(with_store_between=False):
    f = ir.Function("k", [ir.Buffer("S", 1), ir.Buffer("O", 2)])
    b = ir.Builder(f, f.block("e"))
    t = b.builtin("thread_position_in_grid", name="t")
    v = b.add(b.load(f.buffers[0], t, name="l"), ir.Imm(0), name="v")
    a, c = v, b.add(v, ir.Imm(1), name="c0")
    for i in range(4):
        a = b.add(a, ir.Imm(3), name="a%d" % i)
    if with_store_between:
        b.store_at(f.buffers[1], t, a)
    for i in range(4):
        c = b.add(c, ir.Imm(5), name="c%d" % (i + 1))
    b.store_at(f.buffers[1], t, b.xor(a, c, name="r"))
    b.ret()
    return f


def names(f):
    return [o.dest.name if o.dest is not None else o.kind for o in f.blocks[0].ops]


class TheScheduler(unittest.TestCase):

    def test_two_chains_written_in_sequence_interleave(self):
        f = two_chains()
        self.assertGreater(sched.schedule(f), 0)
        n = names(f)
        self.assertEqual(n[n.index("a0"):n.index("r")], ["a0", "c1", "a1", "c2", "a2", "c3", "a3", "c4"])

    def test_it_only_reorders(self):
        f, g = two_chains(), two_chains()
        sched.schedule(g)
        self.assertEqual(sorted(map(str, names(f))), sorted(map(str, names(g))))
        cc.compile_function(g)

    def test_fences_keep_their_order_and_nothing_crosses_a_store_it_feeds(self):
        f = two_chains(with_store_between=True)
        sched.schedule(f)
        n = names(f)
        stores = [i for i, x in enumerate(n) if x == "store_at"]
        self.assertEqual(len(stores), 2)
        self.assertLess(n.index("a3"), stores[0])        # the first store's value is ready before it
        self.assertLess(n.index("l"), stores[0])         # the load stays before both stores

    def test_the_terminator_and_phis_stay_put(self):
        f = two_chains()
        sched.schedule(f)
        self.assertEqual(f.blocks[0].ops[-1].kind, "ret")


class TheModel(unittest.TestCase):

    def test_measured_classes(self):
        self.assertEqual(schedmodel.cost(10282), (2, 1))       # iadd
        self.assertEqual(schedmodel.cost(10825), (5, 2))       # imul

    def test_the_preregistered_predictions_are_by_class(self):
        for op, (_n, cls, want) in schedmodel.PREDICTIONS.items():
            self.assertEqual(schedmodel.PREDICTED_BY_CLASS[cls], want)

    def test_the_recorded_predictions_held(self):
        import json
        rows = json.load(open(os.path.join(ROOT, "isa", "g17-latency.json")))["loop"]["rows"]
        preds = [r["prediction"] for r in rows.values() if "prediction" in r]
        self.assertEqual(len(preds), len(schedmodel.PREDICTIONS))
        self.assertTrue(all(p["holds"] for p in preds), preds)

    def test_the_scheduler_measured_its_speedup_on_identical_outputs(self):
        import json
        rows = json.load(open(os.path.join(ROOT, "isa", "g17-latency.json")))["scheduler"]["rows"]
        for op, r in rows.items():
            self.assertTrue(r["identical_outputs"], op)
            self.assertGreater(r["speedup"], 1.8, op)


if __name__ == "__main__":
    unittest.main()
