"""P1, finite graph admission (machine model 25.124.4): the planner derives regions, lifetimes, cuts and
dispatch boundaries and emits only what a measured cc route admits. Compile-only; nothing dispatches.

Controls that can fail: the canonical reproductions pin two retained program hashes; every refusal is
provoked; the plan guard is fed hand-altered plans; the forced-fused and forced-split paths are driven
by patching the module graphplan imports; the offset control is shown to catch cc's single-body path.
"""
import os
import random
import sys
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import g17tensorcommonruntime as CR  # noqa: E402
from agxforge.g17 import cc, graphplan as G, ir, tensorsched  # noqa: E402

T, MM = G.Tensor, G.MatMul
DOC = open(os.path.join(ROOT, "docs", "g17-tensorops-machine-model.md"), encoding="utf-8").read()


def indep(k, rows=32, N=32, K=64):
    ts = [T("W", K, N, "half", "weight")] + [T(f"X{i}", rows, K, "half", "input") for i in range(k)] + \
         [T(f"Y{i}", rows, N, "float", "value") for i in range(k)]
    return G.Graph(tuple(ts), tuple(MM(f"m{i}", f"X{i}", "W", f"Y{i}") for i in range(k)))


def chain(shared_weight=False):
    ts = [T("X", 16, 32, "half", "input"), T("W1", 32, 32, "half", "weight"),
          T("Y1", 16, 32, "float", "value"), T("Y2", 16, 32, "float", "value")]
    if not shared_weight:
        ts.insert(2, T("W2", 32, 32, "half", "weight"))
    return G.Graph(tuple(ts), (MM("g1", "X", "W1", "Y1"), MM("g2", "Y1", "W1" if shared_weight else "W2", "Y2")))


class CanonicalReproduction(unittest.TestCase):
    """A FEW canonical retained classes only (Set C's review note): a broad sha pin reds on unrelated codegen."""

    def test_independent_groups_reproduce_their_retained_programs(self):
        for k, spec, sha16 in ((2, dict(M=64, N=32, K=64, independent=2), "8200484b82187c26"),
                               (4, dict(M=128, N=32, K=64, independent=4), "c91c464637cfdbd4")):
            self.assertIn(sha16, DOC)                            # the 25.102.2 bundle table
            p = G.plan(indep(k), name=CR.GENERIC_NAME, tail=True)
            self.assertEqual(len(p.dispatches), 1)
            d = p.dispatches[0]
            self.assertEqual((d.route, d.code_sha256[:16]), ("independent_group", sha16))
            ref = CR.build_generic_program(CR.generic_spec(spec))
            self.assertEqual(G.runtime_status(d, ref), "runtime-admitted: gemm_generic (byte-identical)")
            self.assertIn("MEASURED", d.metadata_class)

    def test_without_the_worker_tail_it_is_runtime_unadmitted(self):
        d = G.plan(indep(2), name=CR.GENERIC_NAME).dispatches[0]
        ref = CR.build_generic_program(CR.generic_spec(dict(M=64, N=32, K=64, independent=2)))
        self.assertEqual(d.runtime, "compile-admitted, runtime-unadmitted: P3/P8")
        self.assertEqual(G.runtime_status(d, ref), "compile-admitted, runtime-unadmitted: P3/P8")


class CutsAndFeeds(unittest.TestCase):
    def test_a_novel_chain_is_a_memory_bridge_because_choose_cuts_falls_back(self):
        p = G.plan(chain())
        self.assertEqual([d.route for d in p.dispatches], ["memory_stream"])
        self.assertEqual(p.feeds, {("g1", "g2"): "memory_bridge"})
        d = p.cut_decisions[("g1", "g2")]
        self.assertTrue(d.fallback)
        self.assertEqual(d.schedule.kind, "chain.cut.quantize")
        self.assertEqual(G.check(p), [])

    def test_a_fused_decision_emits_the_released_in_place_chain(self):
        fused = tensorsched.Decision(None, tensorsched._schedule("chain.fused", {}, "test", 0, 1, 1), 1.0,
                                     "estimate", False, "test", (), ())
        with mock.patch.object(G.tensorsched, "choose_cuts", return_value=fused):
            p = G.plan(chain(shared_weight=True), name=CR.GENERIC_NAME, tail=True)
        self.assertEqual([d.route for d in p.dispatches], ["adjacent_chain"])
        self.assertEqual(p.feeds, {("g1", "g2"): "register:A"})
        ref = CR.build_generic_program(CR.generic_spec(dict(M=16, N=32, K=32, stages=[[32, 32, "float"]])))
        self.assertEqual(G.runtime_status(p.dispatches[0], ref), "runtime-admitted: gemm_generic (byte-identical)")
        self.assertEqual(G.check(p), [])


class Refusals(unittest.TestCase):
    def refused(self, graph):
        with self.assertRaises(G.PlanRefused) as cm:
            G.plan(graph)
        return [c[0] for c in cm.exception.checks]

    def test_each_named_refusal(self):
        X, W, Y = T("X", 16, 32, "half", "input"), T("W", 32, 32, "half", "weight"), T("Y", 16, 32, "float", "value")
        self.assertIn("row_stage", self.refused(G.Graph((X, W, Y), (MM("m", "X", "W", "Y"),
                                                                     G.RowStage("s", "softmax", "Y")))))
        self.assertIn("dtype", self.refused(G.Graph((T("X", 16, 32, "float", "input"), W, Y), (MM("m", "X", "W", "Y"),))))
        self.assertIn("feed", self.refused(G.Graph((X, T("V", 32, 32, "float", "value"), Y), (MM("m", "X", "V", "Y"),))))
        self.assertIn("tiles", self.refused(G.Graph((T("X", 16, 24, "half", "input"), T("W", 24, 32, "half", "weight"), Y),
                                                    (MM("m", "X", "W", "Y"),))))
        self.assertIn("extent", self.refused(G.Graph((T("X", 16, 512, "half", "input"), T("W", 512, 32, "half", "weight"), Y),
                                                     (MM("m", "X", "W", "Y"),))))
        self.assertIn("accumulate", self.refused(G.Graph((X, W, Y), (MM("m", "X", "W", "Y", accumulate=True),))))

    def test_b_offsets_past_the_measured_domain_refuse(self):
        ts = [T("X", 16, 128, "half", "input")] + [T(f"W{i}", 128, 128, "half", "weight") for i in range(3)] + \
             [T(f"Y{i}", 16, 128, "float", "value") for i in range(3)]
        g = G.Graph(tuple(ts), tuple(MM(f"m{i}", "X", f"W{i}", f"Y{i}") for i in range(3)))
        self.assertEqual(self.refused(g), ["offsetB"])           # W2 at 65,536 > 47,104

    def test_every_failure_is_listed_not_the_first(self):
        g = G.Graph((T("X", 16, 24, "float", "input"), T("W", 24, 32, "half", "weight"), T("Y", 16, 32, "float", "value")),
                    (MM("m", "X", "W", "Y", accumulate=True), G.RowStage("s", "gelu", "Y")))
        self.assertGreaterEqual(len(set(self.refused(g))), 4)

    def test_a_single_body_with_an_offset_is_refused(self):
        # cc's single-body path drops offsets (measured: identical bytes); tensor_route says so
        g = G.Graph((T("Y", 16, 32, "float", "value"), T("W", 32, 32, "half", "weight"), T("Z", 16, 32, "float", "value")),
                    (MM("m", "Y", "W", "Z"),))
        self.assertEqual(self.refused(g), ["admission"])


class TheOffsetControl(unittest.TestCase):
    def test_it_catches_a_route_that_drops_offsets(self):
        # cc's single-body path used to drop offsets (same bytes); cc now refuses it by name, so the
        # planner's control is exercised on a stand-in that drops them: a program compiled with the
        # offsets zeroed is indistinguishable from the zeroed variant
        tensors = {t.name: t for t in (T("X", 16, 32, "half", "input"), T("W", 32, 32, "half", "weight"),
                                       T("Z", 16, 32, "float", "value"))}
        node = MM("m", "X", "W", "Z")
        regions = {"X": G.Region(1, 0, 1024, "half"), "W": G.Region(2, 0, 2048, "half"),
                   "Z": G.Region(3, 2048, 2048, "float")}
        zero = {k: G.Region(r.buffer, 0, r.bytes, r.dtype) for k, r in regions.items()}
        dropped = cc.compile_function(G._build([node], tensors, zero, "k"))
        self.assertFalse(G.offsets_reach_code([node], tensors, regions, "k", False, dropped))
        with self.assertRaises(cc.Unsupported):
            cc.compile_function(G._build([node], tensors, regions, "k"))
        self.assertEqual(G.plan(indep(2)).dispatches[0].route, "independent_group")


class TensorRoute(unittest.TestCase):
    def fn(self, bodies):
        a = ir.Buffer("A", 1, elem=ir.F16); b = ir.Buffer("B", 2, elem=ir.F16); c = ir.Buffer("C", 3, elem=ir.F32)
        fn = ir.Function("k", [a, b, c]); bl = ir.Builder(fn, fn.block("entry"))
        for kw in bodies:
            bl.tensor_matmul(a, b, c, M=32, N=32, K=64, **kw)
        bl.ret()
        return fn

    def test_names_and_refusals(self):
        self.assertEqual(cc.tensor_route(self.fn([{}])), "single_body")
        self.assertEqual(cc.tensor_route(self.fn([{}, dict(offsetA=4096, offsetC=4096)])), "independent_group")
        self.assertIsInstance(cc.tensor_route(self.fn([])), cc.TensorRouteRefusal)
        # an odd B offset: every route refuses it
        self.assertIsInstance(cc.tensor_route(self.fn([{}, dict(offsetB=3)])), cc.TensorRouteRefusal)
        # OVERLAPPING C regions: the memory stream admitted them until cc's overlap refusal (MM 25.124.4)
        self.assertIsInstance(cc.tensor_route(self.fn([{}, dict(offsetC=2048)])), cc.TensorRouteRefusal)
        self.assertIsInstance(cc.tensor_route(self.fn([dict(offsetC=64)])), cc.TensorRouteRefusal)

    def test_it_is_read_only(self):
        fn = self.fn([{}, dict(offsetA=4096, offsetC=4096)])
        before = cc.compile_function(self.fn([{}, dict(offsetA=4096, offsetC=4096)])).code
        cc.tensor_route(fn)
        self.assertEqual(cc.compile_function(fn).code, before)


class ThePlanGuard(unittest.TestCase):
    def test_a_good_plan_passes_and_altered_plans_fail(self):
        p = G.plan(chain())
        self.assertEqual(G.check(p), [])
        bad = G.Plan(p.dispatches, dict(p.regions, Y2=G.Region(3, 1024, 2048, "float")), p.lifetimes, p.feeds,
                     p.cut_decisions, p.boundaries)
        self.assertIn("disjoint", [c[0] for c in G.check(bad)])
        bad = G.Plan(p.dispatches, dict(p.regions, W2=G.Region(2, 47106, 2048, "half")), p.lifetimes, p.feeds,
                     p.cut_decisions, p.boundaries)
        self.assertIn("offsetB", [c[0] for c in G.check(bad)])
        bad = G.Plan(p.dispatches, dict(p.regions, Y2=G.Region(3, 4098, 2048, "float")), p.lifetimes, p.feeds,
                     p.cut_decisions, p.boundaries)
        self.assertIn("offsetC", [c[0] for c in G.check(bad)])

    def test_a_register_feed_across_a_dispatch_boundary_fails(self):
        p = G.plan(chain())
        split = [G.Dispatch(("g1",), "single_body", "", G.BINDINGS, "", ""),
                 G.Dispatch(("g2",), "single_body", "", G.BINDINGS, "", "")]
        bad = G.Plan(split, p.regions, p.lifetimes, {("g1", "g2"): "register:A"}, p.cut_decisions, p.boundaries)
        self.assertIn("feed", [c[0] for c in G.check(bad)])


class DispatchBoundaries(unittest.TestCase):
    def test_a_route_that_stops_admitting_opens_a_new_dispatch(self):
        real = cc.tensor_route

        def at_most_two(fn):
            n = sum(1 for b in fn.blocks for o in b.ops if o.kind == "tensor_matmul")
            return cc.TensorRouteRefusal("test: at most two bodies") if n > 2 else real(fn)
        with mock.patch.object(G.cc, "tensor_route", side_effect=at_most_two):
            p = G.plan(indep(4))
        self.assertEqual([d.nodes for d in p.dispatches], [("m0", "m1"), ("m2", "m3")])
        self.assertEqual(len(p.boundaries), 1)
        self.assertIn("at most two", p.boundaries[0][1])
        self.assertEqual(G.check(p), [])
        self.assertEqual(p.lifetimes["W"], ((0, 0), (1, 3)))


class RandomGraphs(unittest.TestCase):
    def test_every_graph_plans_cleanly_or_refuses_by_name(self):
        rng = random.Random(20260924)
        planned = refused = 0
        for trial in range(10):
            ts, ns = [T("W0", 32, 32, "half", "weight"), T("W1", 32, 32, "half", "weight")], []
            values = []
            for i in range(rng.randint(1, 4)):
                if values and rng.random() < 0.5:
                    a = rng.choice(values)
                else:
                    a = f"X{i}"
                    ts.append(T(a, 16, 32, "half", "input"))
                out = f"Y{i}"
                ts.append(T(out, 16, 32, "float", "value"))
                ns.append(MM(f"m{i}", a, rng.choice(["W0", "W1"]), out))
                values.append(out)
            try:
                p = G.plan(G.Graph(tuple(ts), tuple(ns)))
            except G.PlanRefused as e:
                self.assertTrue(e.checks and all(len(c) == 3 for c in e.checks))
                refused += 1
                continue
            planned += 1
            self.assertEqual(G.check(p), [])
            for d in p.dispatches:
                self.assertTrue(d.route in cc.TENSOR_RULE_ROUTES or d.route == "single_body", d.route)
                self.assertEqual(d.bindings, G.BINDINGS)
                self.assertTrue(d.runtime.startswith("compile-admitted"))
        self.assertGreater(planned, 0)


if __name__ == "__main__":
    unittest.main()
