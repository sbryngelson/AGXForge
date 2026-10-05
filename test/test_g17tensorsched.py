"""P11, the native tensor scheduler: the section-150 cost model as code, its domain, and its choices,
checked against HELD-OUT hardware timings that the recon record retains as tables.

The recon logs themselves (mxattn_time_b*.log, mxlayer_ffn_time1.log, mxcost_model_validation.json) are
not in this repository. Every measured number below is therefore transcribed from the table of the
section that measured it, and ``TheTranscription`` checks that each one appears verbatim in that
document, so a mistyped row fails here rather than silently moving a median.

Controls that can fail: a causal workload must come back as a LOWER BOUND and refuse a ratio; an
out-of-domain workload must fall back with no number; the memory-step check must fire only where the
unmeasured bandwidth decides the time.
"""
import hashlib
import json
import os
import statistics
import sys
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "test"))
from agxforge.g17 import tensorsched as T  # noqa: E402

RECON = open(os.path.join(ROOT, "docs", "g17-tensorops-accelerator-recon.md"), encoding="utf-8").read()
A = T.Attention


def F(*a, **k):
    """An FFN priced by the RECON family's constants (sections 146 and 149): the published tables."""
    return T.FFN(*a, kernel="recon146", **k)


def FL(*a, **k):
    """An FFN priced by the retained layer harness's constants (MM 25.126)."""
    return T.FFN(*a, kernel="layerbench", **k)

# recon 148 fused rows NOT used to fit the attention constants (the fit used the unsplit decode row
# 2.739, the 512-task row 2.840 and the two bf16 rows 1.280 and 4.988). (workload, measured ms, the
# text that carries the number in recon 148)
ATTN_HELD_OUT = [
    (A(16, 128, 8192, queries=256, sets=2), 1.455, "1.455 (11.8,  42)"),
    (A(16, 128, 2048, queries=256), 0.330, "0.330 (13.0,  32)"),
    (A(16, 128, 8192, queries=256), 1.495, "1.495 (11.5,  36)"),
    (A(8, 128, 32768, queries=512, sets=2), 5.844, "5.844 (11.8,  42)"),
    (A(16, 128, 8192, queries=256, sets=2, dtype="int8"), 1.409, "1.409 (12.2,  27)"),
    (A(16, 128, 8192, queries=256, sets=2), 1.474, "1.474 (11.7,  42)"),        # tb1 tb2
    (A(16, 128, 8192, queries=256, sets=2), 1.425, "gives 1.425 ms"),           # cross_head
    (A(32, 128, 4096, queries=16), 0.682, "0.682 ms at 4,096 keys"),
    (A(64, 128, 16384, queries=16), 2.741, "2.739 to 2.741 ms"),
    (A(128, 128, 2048, queries=16), 0.410, "e4m3 0.410 ms"),
    (A(128, 128, 2048, queries=16, dtype="int8"), 0.398, "int8 0.398"),
    (A(128, 128, 2048, queries=16, dtype="bf16"), 0.490, "bf16 0.490"),
]

# recon 146 part 3, sets = 1, two-kernel column, the rows NOT at 20 threadgroups (the wave constants are
# the 20-threadgroup rows); the dev+ rows are withdrawn and excluded
FFN_TP_HELD_OUT = [
    (F(160, dtype="fp8"), 1.757), (F(640, dtype="fp8"), 3.621), (F(1280, dtype="fp8"), 8.067),
    (F(160, dtype="int8"), 1.621), (F(640, dtype="int8"), 3.181), (F(1280, dtype="int8"), 7.138),
    (F(160, dtype="bf16"), 1.841), (F(640, dtype="bf16"), 4.251), (F(1280, dtype="bf16"), 8.332),
]

# recon 146 part 3, sets = 1: (dtype, tokens, fused, two, three) for every row that is not withdrawn
FFN_CUT_ROWS = [
    ("fp8", 160, 1.780, 1.757, 1.806), ("int8", 160, 1.626, 1.621, 1.651), ("bf16", 160, 2.868, 1.841, 1.877),
    ("fp8", 320, 1.879, 1.839, 1.905), ("int8", 320, 1.703, 1.722, 1.777), ("bf16", 320, 3.056, 2.057, 2.180),
    ("fp8", 640, 3.837, 3.621, 3.818), ("int8", 640, 3.393, 3.181, 3.356), ("bf16", 640, 6.492, 4.251, 4.501),
    ("fp8", 1280, 8.842, 8.067, 8.488), ("int8", 1280, 7.713, 7.138, 7.497), ("bf16", 1280, 12.418, 8.332, 8.839),
]

# recon 149 part 3: weight-stationary total against the token-parallel two-kernel time, same tokens
FFN_WS_VS_TP = [(F(16, dtype="fp8"), 0.442, 1.719), (F(32, dtype="fp8"), 0.841, 3.149),
                (F(16, dtype="bf16"), 0.388, 1.714)]

# recon 149 part 4: weight-stationary stage 1, e4m3, 16 threadgroups, 16/8/4/2 chunks per simdgroup
WS_STAGE1 = [(16, 1, 0.949), (16, 2, 0.479), (16, 4, 0.251), (16, 8, 0.158)]

# recon 148 part 3: simdgroups per threadgroup at 256 tasks
ATTN_NSG = {1: 1.459, 2: 1.396, 4: 1.449, 8: 1.455, 16: 1.544}

# recon 148 part 5: the decode case (32 tasks, 16,384 keys) split 2/4/8/16 ways, no merge
ATTN_SPLIT = {1: 2.739, 2: 1.409, 4: 0.788, 8: 0.613, 16: 0.649}

# recon 155: causal + diagonal skip at 1, 8, 32 heads of 32 query tiles, S = 512 (ns), and the unmasked
CAUSAL = [(1, 52188, 49317), (8, 76206, 63102), (32, 239400, 174350)]


def err(pred, meas):
    return (pred - meas) / meas * 100


class TheTranscription(unittest.TestCase):
    def test_every_held_out_number_is_in_the_recon_record(self):
        for _, _, text in ATTN_HELD_OUT:
            self.assertIn(text, RECON)
        for w, m in FFN_TP_HELD_OUT:
            self.assertIn(f"{m:.3f} (", RECON)
        for row in FFN_CUT_ROWS:
            for v in row[2:]:
                self.assertIn(f"{v:.3f} (", RECON)
        for _, ws, tp in FFN_WS_VS_TP:
            self.assertIn(f"{ws:.3f} ms (", RECON)
            self.assertIn(f"{tp:.3f}", RECON)
        self.assertIn("0.949, 0.479, 0.251 and 0.158 ms for 16, 8, 4 and 2 chunks", RECON)
        self.assertIn("1.459, 1.396, 1.449, 1.455 and 1.544 ms", RECON)
        self.assertIn("1.409, 0.788, 0.613 and 0.649 ms", RECON)
        for _, u, c in CAUSAL:
            self.assertIn(f"{u:,} ns | {c:,} ns", RECON)


class TheCostFunction(unittest.TestCase):
    def test_latency_is_the_maximum_over_simdgroups(self):
        p = T.predict([1, 1, 16], 1000.0, 0.0)
        self.assertAlmostEqual(p.t_latency_ms, 0.016)
        self.assertEqual(p.bound, "lower_bound")
        self.assertEqual(T.predict([4, 4, 4], 1000.0, 0.0).bound, "estimate")

    def test_it_is_the_max_of_three_terms(self):
        p = T.predict([10] * 40, 1000.0, 1000.0, unique_bytes=280_000_000, footprint_bytes=280_000_000)
        self.assertAlmostEqual(p.t_latency_ms, 0.010)
        self.assertAlmostEqual(p.t_throughput_ms, 0.020)
        self.assertAlmostEqual(p.t_memory_ms, 1.0)
        self.assertEqual((p.ms, p.dominant), (1.0, "memory"))

    def test_every_constant_names_a_section_and_a_receipt(self):
        for k, c in T.C.items():
            self.assertTrue(c.section and c.receipt, k)
        self.assertEqual((T.C["mma_dep_1sg_bf16"].value, T.C["mma_dep_64sg_bf16"].value), (62.6, 5.8))
        self.assertEqual((T.C["mma_dep_1sg_fp8"].value, T.C["mma_dep_64sg_fp8"].value), (68.2, 7.0))
        self.assertEqual((T.C["instr_1sg"].value, T.C["instr_64sg"].value), (1.6, 0.34))

    def test_the_chain_fit_refuses_an_occupancy_it_was_not_fitted_at(self):
        self.assertAlmostEqual(T.chain_ns(32, 100, "bf16", 1), 32 * 62.6 + 100 * 1.6)
        with self.assertRaises(ValueError):
            T.chain_ns(32, 100, "bf16", 12)


class HeldOutErrors(unittest.TestCase):
    """Uniform workloads, rows the constants were not fitted to."""

    def test_attention_median_is_within_the_stated_median(self):
        errs = []
        for w, m, _ in ATTN_HELD_OUT:
            d = T.choose(w)
            self.assertFalse(d.fallback, d.reasons)
            self.assertEqual(d.bound, "estimate")
            errs.append(abs(err(d.predicted_ms, m)))
        self.assertLessEqual(statistics.median(errs), T.C["median_attn"].value)
        # recon 150 part 5's worst held-out row is -16.5 percent (128 heads of 2,048 keys, fp8)
        self.assertAlmostEqual(max(errs), 16.5, delta=0.1)

    def test_weight_stationary_stage1_median_is_within_the_stated_median(self):
        errs = [abs(err(T.ws_stage1_ms(hg, nsg).ms, m)) for hg, nsg, m in WS_STAGE1]
        self.assertLessEqual(statistics.median(errs), T.C["median_ws_stage1"].value)

    def test_token_parallel_rows_stay_inside_the_between_process_spread(self):
        # The stated 3.3 percent median is over 16 rows, 7 from a fresh-session log that is not retained.
        # The 9 retained rows give a larger median (MM 25.124); the bound that DOES hold on them is recon
        # 146's between-process spread of 9 to 13 percent.
        errs = [abs(err(T.choose(w).predicted_ms, m)) for w, m in FFN_TP_HELD_OUT]
        self.assertLessEqual(max(errs), 13.0)
        self.assertGreater(statistics.median(errs), T.C["median_ffn_tp"].value)


class TheChooserAgainstTheMeasurement(unittest.TestCase):
    def test_weight_stationary_where_it_was_measured_faster(self):
        for w, ws, tp in FFN_WS_VS_TP:
            d = T.choose(w)
            self.assertEqual(d.schedule.kind, "ffn.weight_stationary")
            self.assertLess(ws, tp)
            self.assertEqual(d.basis, "measured")
            self.assertAlmostEqual(d.predicted_ms, ws)

    def test_token_parallel_two_kernel_at_layer_scale(self):
        for dt, tokens, fused, two, three in FFN_CUT_ROWS:
            d = T.choose(F(tokens, dtype=dt))
            self.assertEqual(d.schedule.kind, "ffn.token_parallel.two_kernel")
            # the chosen cut is the fastest, or within recon 146's 1.5 percent in-process reproducibility
            self.assertLessEqual(two, min(fused, two, three) * 1.015, (dt, tokens))

    def test_an_undispatched_point_is_selected_conservatively(self):
        # ledger row M5: 64 fp8 tokens is predicted 1.682 ms weight-stationary against the 1.84 ms wave,
        # a margin inside recon 146's 13 percent between-process spread, so the known-safe schedule wins
        d = T.choose(F(64, dtype="fp8"))
        self.assertEqual(d.schedule.kind, "ffn.token_parallel.two_kernel")
        self.assertIn("conservative", d.basis)
        ws = next(p for s, p, _ in d.candidates if s.kind == "ffn.weight_stationary")
        self.assertAlmostEqual(ws.ms, 2 * 0.841)
        self.assertEqual(T.choose(F(80, dtype="bf16")).schedule.kind, "ffn.token_parallel.two_kernel")
        # 48 tokens: one block of 32 and one of 16, 1.283 ms, a margin far outside the spread
        d = T.choose(F(48, dtype="fp8"))
        self.assertEqual(d.schedule.kind, "ffn.weight_stationary")
        self.assertIn("prediction", d.basis)
        self.assertAlmostEqual(d.predicted_ms, 0.841 + 0.442)
        self.assertEqual(T.choose(F(128, dtype="fp8")).schedule.kind, "ffn.token_parallel.two_kernel")

    def test_int8_never_gets_an_unmeasured_weight_stationary_schedule(self):
        d = T.choose(F(16, dtype="int8"))
        self.assertEqual(d.schedule.kind, "ffn.token_parallel.two_kernel")

    def test_simdgroups_per_threadgroup_avoids_the_idle_core_row(self):
        d = T.choose(A(16, 128, 8192, queries=256, sets=2))
        nsg = d.schedule.param("nsg")
        self.assertNotEqual(nsg, 16)                  # 16 threadgroups leave 4 cores idle: 6 percent slower
        self.assertLessEqual(ATTN_NSG[nsg], min(ATTN_NSG.values()) * 1.05)

    def test_no_kv_split_unless_a_value_change_is_allowed(self):
        d = T.choose(A(32, 128, 16384, queries=16))
        self.assertEqual(d.schedule.param("split"), 1)
        self.assertAlmostEqual(d.predicted_ms, ATTN_SPLIT[1], delta=0.01)

    def test_the_split_it_picks_is_the_measured_fastest_and_its_times_are_lower_bounds(self):
        d = T.choose(A(32, 128, 16384, queries=16, allow_value_change=True))
        s = d.schedule.param("split")
        self.assertEqual(s, min(ATTN_SPLIT, key=ATTN_SPLIT.get))
        self.assertEqual(d.bound, "lower_bound")
        for sch, p, excl in d.candidates:
            if p is not None and sch.kind == "attention.fused.kv_split":
                self.assertLessEqual(p.ms, ATTN_SPLIT[sch.param("split")])

    def test_mt2_is_never_chosen(self):
        for w, _, _ in ATTN_HELD_OUT:
            self.assertEqual(T.choose(w).schedule.param("mt"), 1)


class NonUniformIsALowerBound(unittest.TestCase):
    def causal(self, heads):
        return A(heads, 128, 512, queries=512, causal=True, kernel="recon155")

    def test_causal_attention_gets_the_lower_bound_treatment(self):
        for heads, unmasked, skip in CAUSAL:
            d = T.choose(self.causal(heads))
            self.assertFalse(d.fallback, d.reasons)
            self.assertEqual(d.schedule.kind, "attention.fused.causal_skip")
            self.assertEqual(d.bound, "lower_bound")
            self.assertLessEqual(d.predicted_ms * 1e6, skip)       # it IS a lower bound on the measurement
            self.assertLess(skip, unmasked)                        # and the skip was measured faster

    def test_it_refuses_to_form_a_ratio(self):
        d = T.choose(self.causal(32))
        p = next(p for s, p, _ in d.candidates if s.kind == "attention.fused.causal_skip")
        q = next(p for s, p, _ in d.candidates if s.kind == "attention.fused.causal_full_loop")
        self.assertEqual(q.bound, "estimate")
        with self.assertRaises(T.RatioRefused):
            T.speedup(q, p)

    def test_the_uniform_arm_of_the_same_kernels_is_an_estimate_within_the_refit_error(self):
        # recon 155.1: the best max() fit to these three uniform points leaves a worst error of 12.1 percent
        for heads, unmasked, _ in CAUSAL:
            d = T.choose(A(heads, 128, 512, queries=512, kernel="recon155"))
            self.assertEqual(d.bound, "estimate")
            self.assertLessEqual(abs(err(d.predicted_ms * 1e6, unmasked)), 12.2)

    def speedups(self, schedule_term):
        L, Cs = T.C["attn155_L"].value, T.C["attn155_C"].value
        out = []
        for heads, unmasked, skip in CAUSAL:
            u = T.predict([16] * 32 * heads, L, Cs)
            c = T.predict(T.causal_steps(512, 512) * heads, L, Cs, schedule_term=schedule_term)
            out.append((u.ms / c.ms, unmasked / skip, c.ms * 1e6, skip))
        return out

    def test_without_the_scheduling_term_it_reproduces_155_1(self):
        # the control: the old form's optimism, 20.9 and 37.1 percent at 256 and 1,024 tasks
        old = self.speedups(False)
        self.assertAlmostEqual((old[1][0] / old[1][1] - 1) * 100, 20.9, delta=0.2)
        self.assertAlmostEqual((old[2][0] / old[2][1] - 1) * 100, 37.1, delta=0.2)

    def test_the_scheduling_term_tightens_the_bound_and_stays_under_the_measurement(self):
        old, new = self.speedups(False), self.speedups(True)
        # speedup optimism falls to 7.5 and 26.5 percent; causal time error -18.3 and -11.6 (MM 25.124.1)
        self.assertAlmostEqual((new[1][0] / new[1][1] - 1) * 100, 7.5, delta=0.2)
        self.assertAlmostEqual((new[2][0] / new[2][1] - 1) * 100, 26.5, delta=0.2)
        for (_, _, t_old, meas), (_, _, t_new, _) in zip(old, new):
            self.assertGreaterEqual(t_new, t_old - 1e-6)       # never looser than the bound
            self.assertLessEqual(t_new, meas)                  # still under every measured time

    def test_the_term_is_the_section_150_form_on_uniform_work(self):
        L, Cs = 2868.0, 327.0
        for n in (20, 40, 260, 1040):   # a whole number of tasks per core
            self.assertAlmostEqual(T.list_makespan_ns([16] * n, L, Cs),
                                   max(16 * L, n / 20 * 16 * Cs), delta=1e-6 * 16 * L)

    def test_dispatch_order_is_a_parameter_and_longest_first_is_not_free(self):
        L, Cs = T.C["attn155_L"].value, T.C["attn155_C"].value
        steps = T.causal_steps(512, 512) * 32
        self.assertNotEqual(T.list_makespan_ns(steps, L, Cs, order="lpt"),
                            T.list_makespan_ns(steps, L, Cs, order="index"))
        with self.assertRaises(ValueError):
            T.list_makespan_ns(steps, L, Cs, order="random")

    def test_latency_uses_the_longest_tile(self):
        # one head: the longest tile walks all 16 blocks, so the lower bound is the full-loop latency
        d = T.choose(self.causal(1))
        self.assertAlmostEqual(d.predicted_ms, 16 * T.C["attn155_L"].value / 1e6)


LB = "results/g17-layer-v1/measure"
LB_FILES = ("ffn-e4m3-m5", "ffn-bf16-m5", "attn-m4", "attn-decode64", "attn-decode96", "attn-decode144",
            "attn-decode192", "attn-decode256")
FIXTURE = json.load(open(os.path.join(ROOT, "isa", "g17-tensorsched-layerbench.json"), encoding="utf-8"))


def clean_medians(res, tol=1.2):
    """tools/g17layerbench.py clean_medians: the median per label over the records whose every reference
    dispatch stayed within tol x the run's median reference time."""
    refs = [k["ref_ms"] for r in res["records"] for k in r["kernels"]]
    lim = tol * statistics.median(refs)
    out = {}
    for r in res["records"]:
        if all(k["ref_ms"] <= lim for k in r["kernels"]):
            out.setdefault(r["label"], []).append(r["ms"])
    return {k: statistics.median(v) for k, v in out.items()}


M6_FILES = ("ffn-m6", "ffn-m6-repeat")


def derive_m6(directory):
    """Piece A's one-process M6 timing (MM 25.126, receipts f1e93ac96): per schedule, the process median
    and its kernels' medians (A, B, C per block), as the run's summary records them."""
    out = {"sha256": {}, "median_ms": {}, "kernel_median_ms": {}}
    for name in M6_FILES:
        path = os.path.join(directory, name + ".json")
        out["sha256"][name] = hashlib.sha256(open(path, "rb").read()).hexdigest()
        summ = json.load(open(path, encoding="utf-8"))["summary"]
        out["median_ms"][name] = {k: round(v["median_ms"], 6) for k, v in summ.items()}
        out["kernel_median_ms"][name] = {k: [round(x, 6) for x in v["kernel_median_ms"]] for k, v in summ.items()
                                         if "_ws" in k}
    return out


def derive_layerbench(directory):
    """The fixture, derived from Piece A's raw receipts (MM 25.126)."""
    out = {"sha256": {}, "clean_median_ms": {}}
    for name in LB_FILES:
        path = os.path.join(directory, name + ".json")
        out["sha256"][name] = hashlib.sha256(open(path, "rb").read()).hexdigest()
        res = json.load(open(path, encoding="utf-8"))
        out["clean_median_ms"][name] = {k: round(v, 6) for k, v in clean_medians(res).items()}
        if name.startswith("ffn-"):
            # the 16-token WS schedule's kernels A, B, C, each a clean median: the per-stage times of a
            # format mix (MM 25.124.3)
            out.setdefault("ws16_kernel_clean_median_ms", {})[name] = ws16_kernels(res)
    return out


def ws16_kernels(res, tol=1.2):
    refs = [k["ref_ms"] for r in res["records"] for k in r["kernels"]]
    lim = tol * statistics.median(refs)
    per = {}
    for r in res["records"]:
        if r["label"] == "ws16" and all(k["ref_ms"] <= lim for k in r["kernels"]):
            for stage, k in zip("ABC", r["kernels"]):
                per.setdefault(stage, []).append(k["ms"])
    return {k: round(statistics.median(v), 6) for k, v in per.items()}


def lb(name, label):
    return FIXTURE["clean_median_ms"][name][label]


# the harness M5 points: (dtype, file, tokens) -> every schedule measured at that token count, same process
M5_POINTS = [("fp8", "ffn-e4m3-m5", t) for t in (16, 32, 48, 64, 96, 128)] + \
            [("bf16", "ffn-bf16-m5", t) for t in (16, 48, 64, 96, 128)]
KIND = {"ws": "ffn.weight_stationary", "tp": "ffn.token_parallel.two_kernel",
        "1pass": "ffn.weight_stationary.one_pass"}


def measured_at(name, tokens):
    """{schedule kind: measured ms} at one token count. TP at 16 tokens is one threadgroup (tp16_1tg);
    TP at 32 fp8 tokens was timed only as one threadgroup at mt 2 (tp32_1tg_mt2)."""
    out = {}
    for label, ms in FIXTURE["clean_median_ms"][name].items():
        head = label.split("_")[0]
        if head[2:] != str(tokens):
            continue
        if label.endswith("_1pass"):
            out[KIND["1pass"]] = ms
        elif label == "tp32_1tg_mt2":
            out["ffn.token_parallel.mt2_one_threadgroup"] = ms
        else:
            out[KIND[head[:2]]] = ms
    return out


class TheLayerbenchReceipts(unittest.TestCase):
    def test_the_fixture_is_the_receipts(self):
        d = os.path.join(ROOT, LB)
        paths = [os.path.join(LB, n + ".json") for n in LB_FILES]
        from _evidence import require
        require(*paths, invariant="the committed layerbench fixture equals Piece A's raw M4/M5 receipts")
        derived = derive_layerbench(d)
        self.assertEqual(derived["sha256"], FIXTURE["sha256"])
        self.assertEqual(derived["clean_median_ms"], FIXTURE["clean_median_ms"])
        self.assertEqual(derived["ws16_kernel_clean_median_ms"], FIXTURE["ws16_kernel_clean_median_ms"])

    def test_every_constant_is_its_receipt(self):
        e, b = "ffn-e4m3-m5", "ffn-bf16-m5"
        self.assertEqual(T.C["lb_ws16_fp8"].value, lb(e, "ws16"))
        self.assertEqual(T.C["lb_ws32_fp8"].value, lb(e, "ws32"))
        self.assertEqual(T.C["lb_onepass48_fp8"].value, lb(e, "ws48_1pass"))
        self.assertEqual(T.C["lb_onepass64_fp8"].value, lb(e, "ws64_1pass"))
        self.assertEqual(T.C["lb_ws16_bf16"].value, lb(b, "ws16"))
        for key, name in (("lb_tp_fp8", e), ("lb_tp_bf16", b)):
            flat = [lb(name, x) for x in ("tp16_1tg", "tp48", "tp64", "tp96", "tp128")]
            self.assertEqual(T.C[key].value, statistics.median(flat))
            self.assertLess(max(flat) / min(flat), 1.10)          # flat below one wave (MM 25.126)


class TheChooserOnTheRetainedHarness(unittest.TestCase):
    def test_it_picks_the_measured_fastest_schedule_at_every_harness_point(self):
        picks = []
        for dt, name, t in M5_POINTS:
            d = T.choose(FL(t, dtype=dt))
            self.assertFalse(d.fallback, (dt, t, d.reasons))
            m = measured_at(name, t)
            chosen = m.get(d.schedule.kind)
            if chosen is None and d.schedule.kind == KIND["tp"] and t == 32:
                # TP at 32 fp8 tokens (2 threadgroups) was not timed; it is flat from 1 to 8 (>= 1.217 ms)
                chosen = min(lb(name, x) for x in ("tp16_1tg", "tp48", "tp64", "tp96", "tp128"))
            self.assertIsNotNone(chosen, (dt, t, d.schedule.kind, m))
            self.assertEqual(chosen, min(m.values()), (dt, t, d.schedule.kind, m))
            picks.append((dt, t, d.schedule.kind))
        # the crossovers the harness measured (MM 25.126): fp8 between 32 and 48 for blocked WS, with the
        # one-pass WS still winning at 48; bf16 between 48 and 64
        kinds = {(dt, t): k for dt, t, k in picks}
        self.assertEqual(kinds[("fp8", 32)], KIND["ws"])
        self.assertEqual(kinds[("fp8", 48)], KIND["1pass"])
        self.assertEqual(kinds[("fp8", 64)], KIND["tp"])
        self.assertEqual(kinds[("bf16", 48)], KIND["ws"])
        self.assertEqual(kinds[("bf16", 64)], KIND["tp"])

    def test_blocked_ws_alone_crosses_between_32_and_48_fp8(self):
        # without the one-pass arm the model places the fp8 crossover where the harness did
        for t, winner in ((32, "ws"), (48, "tp"), (64, "tp")):
            d = T.choose(FL(t, dtype="fp8"))
            live = {s.kind: p.ms for s, p, e in d.candidates if p is not None and e is None
                    and s.kind != KIND["1pass"]}
            self.assertEqual(min(live, key=live.get), KIND[winner], (t, live))

    def test_the_additive_ws_form_holds_in_process(self):
        # 25.126: the sum of same-process blocks is within 1 to 4 percent at 48 to 128 tokens
        for dt, name, t in M5_POINTS:
            if t < 48:
                continue
            p = next(p for s, p, _ in T.choose(FL(t, dtype=dt)).candidates if s.kind == KIND["ws"])
            self.assertLess(abs(p.ms / lb(name, "ws%d" % t) - 1), 0.05, (dt, t))

    def test_beyond_the_sweep_it_falls_back(self):
        d = T.choose(FL(160))
        self.assertTrue(d.fallback)
        self.assertTrue(any("tokens" in r for r in d.reasons))
        self.assertTrue(T.choose(FL(16, dtype="int8")).fallback)


class TheReconConstantsFailOnTheHarness(unittest.TestCase):
    """The control: the recon family's constants name the wrong winner where the harness measured."""

    def raw_pick(self, t, dt):
        d = T.choose(F(t, dtype=dt))
        live = {s.kind: p.ms for s, p, e in d.candidates if p is not None and e is None}
        return min(live, key=live.get), d.schedule.kind

    def test_the_raw_recon_model_is_wrong_at_fp8_48_and_64_and_bf16_64(self):
        for dt, name, t in (("fp8", "ffn-e4m3-m5", 48), ("fp8", "ffn-e4m3-m5", 64), ("bf16", "ffn-bf16-m5", 64)):
            raw, _ = self.raw_pick(t, dt)
            self.assertEqual(raw, KIND["ws"])
            self.assertGreater(lb(name, "ws%d" % t), lb(name, "tp%d" % t))       # measured: TP was faster

    def test_its_13_percent_margin_rescues_only_fp8_64(self):
        self.assertEqual(self.raw_pick(64, "fp8")[1], KIND["tp"])
        self.assertEqual(self.raw_pick(48, "fp8")[1], KIND["ws"])
        self.assertEqual(self.raw_pick(64, "bf16")[1], KIND["ws"])


# (head width, file, label) -> same-process ratio to width 128 (MM 25.126)
HW = [(dh, "attn-m4", f"att{dh}_16x16_k{k}_s{s}", f"att128_16x16_k{k}_s{s}")
      for dh in (64, 96, 144, 192, 256) for k, s in ((2048, 1), (8192, 2))] + \
     [(dh, f"attn-decode{dh}", f"att{dh}_32x1_k16384_s1", "att128_32x1_k16384_s1") for dh in (64, 96, 144, 192, 256)]


class HeadWidthScaling(unittest.TestCase):
    def ratios(self):
        for dh, name, lab, base in HW:
            yield dh, lb(name, lab) / lb(name, base)

    def test_proportional_from_96_to_256_and_the_144_remainder_rule(self):
        for dh, r in self.ratios():
            f = T.head_width_factor(dh)
            err = f / r - 1
            if dh == 64:
                self.assertLess(abs(err), 0.15, (dh, r))
            elif dh == 144:
                self.assertEqual(f, 1.25)
                self.assertTrue(-0.06 < err < 0.0, (dh, r, err))
            else:
                self.assertTrue(-0.07 < err < 0.03, (dh, r, err))

    def test_the_affine_extrapolation_fails_above_128(self):
        # the control: section 150 part 6's per-MMA constants added per extra MMA (dh / 4 per key block)
        L, Cs = T.C["attn_L_fp8"].value, T.C["attn_C_fp8"].value
        for dh, r in self.ratios():
            if dh < 192:
                continue
            extra = dh / 4 - 32
            affine = (L + extra * T.C["mma_dep_1sg_fp8"].value) / L          # latency-bound ratio
            affine_thr = (Cs + extra * T.C["mma_dep_64sg_fp8"].value) / Cs  # throughput-bound ratio
            self.assertLess(max(affine, affine_thr) / r - 1, -0.15, (dh, r))
            self.assertLess(abs(T.head_width_factor(dh) / r - 1), 0.07)

    def test_the_chooser_uses_it(self):
        d128 = T.choose(A(16, 128, 8192, queries=256, sets=2))
        d256 = T.choose(A(16, 256, 8192, queries=256, sets=2))
        self.assertFalse(d256.fallback, d256.reasons)
        self.assertAlmostEqual(d256.predicted_ms / d128.predicted_ms, 2.0, delta=0.01)


class OutsideTheDomainFallsBack(unittest.TestCase):
    def assertFallback(self, w, needle):
        d = T.choose(w)
        self.assertTrue(d.fallback)
        self.assertIsNone(d.predicted_ms)
        self.assertTrue(any(needle in r for r in d.reasons), d.reasons)
        return d

    def test_an_unmeasured_ffn_shape(self):
        d = self.assertFallback(F(16, d=4096), "shape")
        self.assertEqual(d.schedule.kind, "ffn.token_parallel.two_kernel")

    def test_an_unmeasured_dtype(self):
        self.assertFallback(F(16, dtype="fp16"), "dtype")
        self.assertFallback(A(16, 128, 8192, queries=256, dtype="fp16"), "constants")

    def test_an_unmeasured_head_width(self):
        d = self.assertFallback(A(16, 80, 8192, queries=256), "head_dim")
        self.assertFallback(A(16, 176, 8192, queries=256), "head_dim")   # a 16-wide remainder other than 144
        self.assertEqual(d.schedule.kind, "attention.fused")

    def test_a_register_count_outside_the_residency_measurements(self):
        self.assertFallback(A(16, 128, 8192, queries=256, registers=140), "registers")
        self.assertFalse(T.choose(A(16, 128, 8192, queries=256, registers=100)).fallback)

    def test_a_kernel_the_constants_were_not_fitted_to(self):
        self.assertFallback(A(16, 128, 8192, queries=256, kernel="tlower"), "constants")

    def test_more_waves_than_were_measured(self):
        self.assertFallback(F(2000), "tokens")

    def test_the_memory_step_check_fires_only_where_bandwidth_decides(self):
        # 200 heads x 512 keys: 31 MiB of KV, in the unmeasured 20..40 MiB step, and memory-bound
        self.assertFallback(A(200, 128, 512, queries=16), "memory")
        # 96 heads x 1,024 keys: 30 MiB, also in the step, but latency-bound either way
        self.assertFalse(T.choose(A(96, 128, 1024, queries=16)).fallback)


class WhatTheRuntimeCanRealize(unittest.TestCase):
    def test_limits_are_read_from_the_runtime(self):
        from typing import get_args
        from agxforge.g17.runtime import TensorSpec
        f = TensorSpec.model_fields
        lim = T.runtime_limits()
        self.assertEqual(lim["N"], next(m.le for m in f["N"].metadata if hasattr(m, "le")))
        self.assertEqual(lim["simdgroups"], get_args(f["simdgroups"].annotation))
        self.assertIn("attention", lim["compositions"])      # P7's class (MM 25.129)

    def test_no_layer_schedule_is_realizable_yet(self):
        """A layer-scale FFN is still wider than gemm_generic's N; a layer-scale attention is outside P7's
        class shape. Naming the attention family does not make a head-128, 8,192-key workload realizable
        (it did, silently, the day P7's class landed)."""
        for w in (F(16), F(640), A(16, 128, 8192, queries=256), A(32, 128, 16384, queries=16)):
            d = T.choose(w)
            self.assertFalse(d.schedule.runtime_realizable)
            self.assertTrue(d.schedule.runtime_reason)
        why = T.choose(A(16, 128, 8192, queries=256)).schedule.runtime_reason
        for gap in ("8192 keys", "256 query rows"):
            self.assertIn(gap, why)
        # head 64 at 16 heads is outside every shape (the head grid is phase grid's, head 128)
        self.assertIn("16 heads", T.choose(A(16, 64, 128, queries=1)).schedule.runtime_reason)
        self.assertIn("12 heads", T.choose(A(12, 128, 128, queries=1)).schedule.runtime_reason)

    def test_the_attention_class_shape_is_realizable(self):
        """The control: a workload inside P7's class (one head of 64, 8 blocks of 16 keys, 16 rows, one
        simdgroup) is realizable, so the refusal above is about shape, not about the family."""
        d = T.choose(A(1, 64, 128, queries=16, dtype="bf16"))
        self.assertTrue(d.schedule.runtime_realizable, d.schedule.runtime_reason)
        # the counted key-block loop's 64 blocks of 16 keys (MM 25.114.5): 1,024 keys are inside, 1,040 are not
        self.assertTrue(T.choose(A(1, 64, 1024, queries=16, dtype="bf16")).schedule.runtime_realizable)
        self.assertFalse(T.choose(A(1, 64, 1040, queries=16, dtype="bf16")).schedule.runtime_realizable)


class TheNTiledGrid(unittest.TestCase):
    """MM 25.134: an output wider than one threadgroup's 256 columns is realizable through the N-tiled grid
    when a power-of-two grid_n splits it into whole 16-column tiles of at most 256."""

    def test_the_decode_widths_split(self):
        self.assertEqual(1, T.grid_split(256))
        self.assertEqual(8, T.grid_split(2048))
        self.assertEqual(32, T.grid_split(8192))

    def test_a_width_no_power_of_two_splits_is_refused(self):
        # 6144 = 3 x 2048: 6144 / 16 is 384 tiles, which no power of two brings under 256 columns in whole
        # tiles (Set C: split Q/K/V as 3 x 2048 instead)
        self.assertIsNone(T.grid_split(6144))
        self.assertIsNone(T.grid_split(24))

    def test_the_verdict_names_the_width_only_when_no_split_exists(self):
        ok, why = T._runtime_verdict(2048, 1, 1, "ffn")
        self.assertNotIn("output width", why)
        ok, why = T._runtime_verdict(6144, 1, 1, "ffn")
        self.assertIn("output width of 6144", why)

    def test_the_decode_milestone_attention_is_realizable(self):
        """Phase grid (MM 25.135): 16 heads of 128, one decode row, 272 keys in one dispatch is inside; 288
        keys, or 16 rows at 128 keys (past the image contract), are not."""
        self.assertTrue(T.choose(A(16, 128, 272, queries=1, dtype="bf16")).schedule.runtime_realizable)
        self.assertTrue(T.choose(A(1, 128, 64, queries=16, dtype="bf16")).schedule.runtime_realizable)
        self.assertFalse(T.choose(A(16, 128, 288, queries=1, dtype="bf16")).schedule.runtime_realizable)
        self.assertFalse(T.choose(A(1, 128, 128, queries=16, dtype="bf16")).schedule.runtime_realizable)


class ItDoesNotTouchTlower(unittest.TestCase):
    def test_no_tlower_import(self):
        src = open(T.__file__, encoding="utf-8").read()
        self.assertNotIn("import tlower", src)
        self.assertNotIn("from agxforge.g17 import tlower", src)
        self.assertNotIn("from agxforge.g17.tlower", src)



# ---------------------------------------------------------------------------------------------- 25.124.3
S, Ch = T.Stage, T.Chain

# recon 142 part 2, e4m3fn FFN with silu: (shape, fused spill stores, ratios at 64 SG (mem, p12_p3, p1_p23,
# p1_p2_p3), ratios at 1 SG (same)); the CUT_RATIO table must be these rows' ranges
R142 = [
    ("(1, 2, 2, 64, 8)", 0, (1.06, 1.10, 1.08, 1.20), (1.08, 1.05, 1.05, 1.11)),
    ("(1, 2, 4, 64, 8)", 0, (1.08, 1.09, 1.06, 1.16), (1.08, 1.02, 1.01, 1.07)),
    ("(1, 4, 4, 64, 4)", 22, (1.07, 0.98, 0.96, 1.06), (1.09, 0.90, 0.88, 0.93)),
    ("(2, 2, 2, 256, 4)", 15, (0.98, 1.00, 0.95, 1.01), (0.89, 0.92, 0.92, 0.93)),
    ("(1, 2, 8, 128, 8)", 23, (1.02, 0.95, 0.96, 0.99), (1.02, 0.83, 0.87, 0.85)),
    ("(2, 2, 4, 128, 4)", 76, (1.02, 1.00, 0.98, 1.04), (1.04, 0.95, 0.99, 0.98)),
    ("(1, 2, 2, 1024, 4)", 0, (1.00, 1.05, 1.06, 1.10), (1.01, 0.97, 0.93, 0.94)),
    ("(1, 2, 2, 4096, 2)", 0, (1.03, 1.06, 1.07, 1.08), (1.03, 0.99, 0.94, 0.94)),
]
COL = {"quantize": 1, "pre_quantize": 2, "all": 3}


def m6_stages():
    # recon 149 part 3 and row M6: kernel A, B, C in e4m3 and bf16 (16 tokens)
    return (S("A", "fp8", measured_ms={"fp8": 0.148, "bf16": 0.234}),
            S("B", "fp8", measured_ms={"fp8": 0.287, "bf16": 0.147}),
            S("C", "fp8", measured_ms={"fp8": 0.007, "bf16": 0.007}))


class CutPoints(unittest.TestCase):
    def test_the_cut_ratio_table_is_recon_142s_rows(self):
        for row in R142:
            self.assertIn(row[0], RECON)
        cases = {("spill_free", "many"): [r[2] for r in R142 if r[1] == 0 and r[0] not in
                                           ("(1, 2, 2, 1024, 4)", "(1, 2, 2, 4096, 2)")] +
                 [r[2] for r in R142 if r[0] in ("(1, 2, 2, 1024, 4)", "(1, 2, 2, 4096, 2)")],
                 ("spills", "many"): [r[2] for r in R142 if r[1] >= 15],
                 ("spills", "one"): [r[3] for r in R142 if r[1] >= 15],
                 ("spill_free_long_k", "one"): [r[3] for r in R142 if r[0] in ("(1, 2, 2, 1024, 4)", "(1, 2, 2, 4096, 2)")]}
        for regime, rows in cases.items():
            for k, col in COL.items():
                vals = [r[col] for r in rows]
                self.assertEqual(T.CUT_RATIO[regime][k], (min(vals), max(vals)), (regime, k))

    def chain(self, spills, sg=64, kb=2, fmt="fp8"):
        return Ch((S("stage1", fmt, 96, 400), S("stage2", fmt, 32, 240)), fused_spill_stores=spills,
                  sg_per_core=sg, k_blocks=kb)

    def test_the_measured_rules(self):
        self.assertEqual(T.choose_cuts(self.chain(0)).schedule.kind, "chain.fused")
        self.assertTrue(T.choose_cuts(self.chain(22)).schedule.kind.startswith("chain.cut"))
        self.assertTrue(T.choose_cuts(self.chain(22, sg=1)).schedule.kind.startswith("chain.cut"))
        self.assertTrue(T.choose_cuts(self.chain(0, sg=1, kb=1024)).schedule.kind.startswith("chain.cut"))

    def test_a_cut_is_priced_as_fused_time_times_its_measured_ratio(self):
        d = T.choose_cuts(self.chain(22, sg=1))
        fused = (T.chain_ns(96, 400, "fp8", 1) + T.chain_ns(32, 240, "fp8", 1)) / 1e6
        self.assertAlmostEqual(d.predicted_ms, fused * 0.95)          # the quantize cut's upper end
        self.assertEqual(d.schedule.kind, "chain.cut.quantize")

    def test_the_rule_would_be_wrong_with_the_ratios_flipped(self):
        # the control: the decision follows the measured ratios, so swapping the spill regimes flips it
        saved = dict(T.CUT_RATIO)
        try:
            T.CUT_RATIO[("spill_free", "many")] = saved[("spills", "many")]
            self.assertNotEqual(T.choose_cuts(self.chain(0)).schedule.kind, "chain.fused")
        finally:
            T.CUT_RATIO.clear()
            T.CUT_RATIO.update(saved)

    def test_three_gemm_chain_only_where_measured(self):
        three = (S("a", "bf16", 32, 200), S("b", "bf16", 32, 200), S("c", "bf16", 32, 200))
        self.assertEqual(T.choose_cuts(Ch(three, fused_spill_stores=0, k_blocks=2)).schedule.kind, "chain.fused")
        self.assertTrue(T.choose_cuts(Ch(three, fused_spill_stores=0, k_blocks=32)).fallback)

    def test_out_of_domain_falls_back_to_the_quantize_cut(self):
        for c, needle in ((self.chain(None), "spills"), (self.chain(5), "regime"), (self.chain(0, sg=12), "regime"),
                          (self.chain(0, sg=1, kb=2), "regime"), (self.chain(0, fmt="fp4"), "dtype"),
                          (Ch((S("x", "fp8"),), fused_spill_stores=0), "stages"),
                          (Ch((S("a", "fp8", 32, 100), S("b", "bf16", 32, 100)), fused_spill_stores=0), "fused")):
            d = T.choose_cuts(c)
            self.assertTrue(d.fallback, needle)
            self.assertIsNone(d.predicted_ms)
            self.assertEqual(d.schedule.kind, "chain.cut.quantize")
            self.assertTrue(any(r.startswith(needle) for r in d.reasons), (needle, d.reasons))


def m6_run(name, block):
    """(stages, measured totals) for one M6 process at one block size: per format, the kernels A, B, C of
    its ws{block} schedule, all timed in that one process."""
    k = FIXTURE["m6"]["kernel_median_ms"][name]
    med = FIXTURE["m6"]["median_ms"][name]
    per = {"fp8": k[f"e4m3_ws{block}"], "bf16": k[f"bf16_ws{block}"]}
    stages = tuple(S(n, "fp8", measured_ms={f: per[f][i] for f in per}) for i, n in enumerate("ABC"))
    totals = {"fp8": med[f"e4m3_ws{block}"], "bf16": med[f"bf16_ws{block}"], "mix": med[f"mix_ws{block}"]}
    return stages, totals


class FormatMix(unittest.TestCase):
    """25.124.3 as corrected by Piece A's one-process M6 timing (MM 25.126)."""

    def test_the_m6_fixture_is_the_receipts(self):
        from _evidence import require
        require(*[os.path.join(LB, n + ".json") for n in M6_FILES],
                invariant="the committed M6 stage times equal Piece A's one-process receipts")
        self.assertEqual(derive_m6(os.path.join(ROOT, LB)), FIXTURE["m6"])

    def test_one_process_at_the_block_it_runs_picks_a_measured_fastest_schedule(self):
        kind = {"fp8": "fp8", "bf16": "bf16"}
        for name in M6_FILES:
            for block in (16, 32):
                stages, totals = m6_run(name, block)
                # allow_unbuilt: the harness built the mix (25.126), the compiler cannot lower it (P5)
                d = T.choose_cuts(Ch(stages, process={"fp8": name, "bf16": name}, block_tokens=block,
                                     mix_built=True, allow_unbuilt=True))
                self.assertFalse(d.fallback, d.reasons)
                pick = "mix" if d.schedule.kind == "chain.cut.mixed_format" else kind[d.schedule.param("format")]
                # the pick is the measured fastest, or tied with it inside the 4 percent in-process band
                self.assertLessEqual(totals[pick], min(totals.values()) * (1 + T.FORMAT_TIE), (name, block, pick, totals))
                if block == 32:
                    self.assertEqual(pick, "mix", (name, totals))           # e4m3 stage B 0.81 vs bf16 0.49
                    self.assertGreater(totals["fp8"] / totals["mix"], 1.35)

    def test_cross_process_or_blockless_stage_times_are_refused(self):
        k = FIXTURE["ws16_kernel_clean_median_ms"]
        e, b = k["ffn-e4m3-m5"], k["ffn-bf16-m5"]
        stages = tuple(S(n, "fp8", measured_ms={"fp8": e[n], "bf16": b[n]}) for n in "ABC")
        for kw, needle in (({"process": {"fp8": "ffn-e4m3-m5", "bf16": "ffn-bf16-m5"}, "block_tokens": 16}, "process"),
                           ({"block_tokens": 16}, "process"),
                           ({"process": {"fp8": "p", "bf16": "p"}}, "block")):
            d = T.choose_cuts(Ch(stages, mix_built=True, **kw))
            self.assertTrue(d.fallback)
            self.assertIsNone(d.predicted_ms)
            self.assertTrue(any(r.startswith(needle) for r in d.reasons), (needle, d.reasons))

    def test_the_cross_process_inputs_would_have_picked_wrong(self):
        # THE CONTROL: the M5 kernels came from two processes and made e4m3 stage B look faster (0.176 vs
        # 0.201). Taking each stage's fastest format from them -- the rule 25.124.3 first applied -- says
        # pure e4m3; at 32 tokens, in ONE process, pure e4m3 is 1.39x to 1.40x the mix.
        k = FIXTURE["ws16_kernel_clean_median_ms"]
        e, b = k["ffn-e4m3-m5"], k["ffn-bf16-m5"]
        old_pick = {n: ("fp8" if e[n] <= b[n] else "bf16") for n in "AB"}
        self.assertEqual(old_pick, {"A": "fp8", "B": "fp8"})
        for name in M6_FILES:
            _, totals = m6_run(name, 32)
            self.assertGreater(totals["fp8"] / totals["mix"], 1.35)
            # and in one process stage B at 16 tokens is a tie across formats (0.184 to 0.200)
            k16 = FIXTURE["m6"]["kernel_median_ms"][name]
            bs = [k16[f"{f}_ws16"][1] for f in ("e4m3", "bf16", "mix")]
            self.assertLess(max(bs) / min(bs), 1.09)

    def test_the_recon_m6_prediction_is_refused_now(self):
        # recon 149's stage times carry no process: the 0.302 ms mix is no longer selected from them
        d = T.choose_cuts(Ch(m6_stages(), allow_unbuilt=True, block_tokens=16))
        self.assertTrue(d.fallback)
        self.assertTrue(any(r.startswith("process") for r in d.reasons))
        # the arithmetic still stands when one process is ASSERTED by a caller
        d = T.choose_cuts(Ch(m6_stages(), allow_unbuilt=True, block_tokens=16, process={"fp8": "x", "bf16": "x"}))
        self.assertAlmostEqual(d.predicted_ms, 0.302)
        self.assertIn("not built", d.basis)

    def test_an_unbuilt_mix_needs_the_opt_in(self):
        stages, _ = m6_run("ffn-m6", 32)
        d = T.choose_cuts(Ch(stages, process={"fp8": "ffn-m6", "bf16": "ffn-m6"}, block_tokens=32))
        self.assertEqual(d.schedule.kind, "chain.cut.all")
        self.assertIn("not built", d.basis)


class OnlyWhatTheCompilerCanLower(unittest.TestCase):
    """choose_cuts filters its decided kind through P5's tensorcuts.p11_kinds_for (MM 25.128.3)."""

    def spilling(self, **kw):
        # 15+ spills at 64 SG: the measured ratios decide chain.cut.pre_quantize (0.95..0.98), which P5
        # reports unbuilt (tlower has no fp32-to-fp8 operand conversion)
        return Ch((S("s1", "fp8", 96, 400), S("s2", "fp8", 32, 240)), fused_spill_stores=22, **kw)

    def test_an_unbuilt_kind_is_dropped_and_named(self):
        d = T.choose_cuts(self.spilling())
        self.assertTrue(d.fallback)
        self.assertIsNone(d.predicted_ms)
        self.assertEqual(d.schedule.kind, "chain.cut.quantize")
        self.assertEqual(d.schedule.param("dropped"), "chain.cut.pre_quantize")
        named = [r for r in d.reasons if r.startswith("unlowerable ")]
        self.assertTrue(any("chain.cut.pre_quantize" in r for r in named))
        self.assertTrue(any("chain.cut.mixed_format" in r for r in named))   # every unbuilt kind is named

    def test_allow_unbuilt_keeps_it(self):
        d = T.choose_cuts(self.spilling(allow_unbuilt=True))
        self.assertEqual(d.schedule.kind, "chain.cut.pre_quantize")
        self.assertFalse(d.fallback)

    def test_the_control_bypassing_the_filter_returns_the_unbuilt_kind(self):
        from agxforge.g17 import tensorcuts
        everything = {k: (True, "test") for k in tensorcuts.P11_KINDS}
        with mock.patch.object(tensorcuts, "p11_kinds_for", return_value=everything), \
                mock.patch.dict(tensorcuts.P11_KINDS, {k: (True, v[1], v[2]) for k, v in tensorcuts.P11_KINDS.items()}):
            d = T.choose_cuts(self.spilling())
        self.assertEqual(d.schedule.kind, "chain.cut.pre_quantize")

    def test_a_built_mix_the_compiler_cannot_lower_is_dropped(self):
        stages, _ = m6_run("ffn-m6", 32)
        d = T.choose_cuts(Ch(stages, process={"fp8": "ffn-m6", "bf16": "ffn-m6"}, block_tokens=32, mix_built=True))
        self.assertTrue(d.fallback)
        self.assertEqual(d.schedule.param("dropped"), "chain.cut.mixed_format")

    def test_a_chains_own_boundaries_are_checked(self):
        # a half x half producer into a float consumer: no fp8 boundary, so the quantize cut is not lowerable
        # for THIS chain, and the replacement is the next lowerable kind
        lowering = (dict(M=32, N=32, K=64, a="half", b="half"), dict(M=32, N=32, K=32, a="float", b="half"))
        from agxforge.g17 import tensorcuts
        kinds = tensorcuts.p11_kinds_for([dict(x) for x in lowering])
        self.assertFalse(kinds["chain.cut.quantize"][0])
        d = T.choose_cuts(self.spilling(lowering=lowering))
        self.assertTrue(d.fallback)
        self.assertNotIn(d.schedule.kind, ("chain.cut.quantize", "chain.cut.pre_quantize"))
        self.assertTrue(kinds[d.schedule.kind][0])


class FusedOrCutAttention(unittest.TestCase):
    W = A(16, 128, 8192, queries=256, sets=2)

    def test_the_ratios_are_recon_148s(self):
        self.assertIn("P cut 1.03 to 1.30 times the fused time (median 1.13), S cut 1.07 to 1.42 (1.28), "
                      "both 1.14 to 2.03 (1.65)", RECON)
        self.assertEqual(T.ATTN_CUT_RATIO, {"p_cut": (1.03, 1.13, 1.30), "s_cut": (1.07, 1.28, 1.42),
                                            "both_cuts": (1.14, 1.65, 2.03)})

    def test_fused_when_exposed_and_the_cheapest_cut_otherwise(self):
        self.assertEqual(T.choose_attention_cut(self.W).schedule.kind, "attention.fused")
        self.assertEqual(T.choose_attention_cut(self.W, ("s_cut", "p_cut", "both_cuts")).schedule.kind,
                         "attention.p_cut")
        self.assertEqual(T.choose_attention_cut(self.W, ("both_cuts", "s_cut")).schedule.kind, "attention.s_cut")

    def test_against_the_measured_rows(self):
        # recon 148 part 2: fused 1.455, P cut 1.635, S cut 1.856, both 2.523 ms (e4m3 16 x 16, 8,192 keys)
        measured = {"fused": 1.455, "p_cut": 1.635, "s_cut": 1.856, "both_cuts": 2.523}
        for exposed in (("fused", "p_cut", "s_cut", "both_cuts"), ("p_cut", "s_cut"), ("s_cut", "both_cuts")):
            d = T.choose_attention_cut(self.W, exposed)
            pick = d.schedule.kind.split(".")[1]
            self.assertEqual(measured[pick], min(measured[e] for e in exposed))

    def test_causal_is_a_lower_bound(self):
        d = T.choose_attention_cut(A(32, 128, 512, queries=512, causal=True, kernel="recon155"))
        self.assertEqual(d.bound, "lower_bound")
        self.assertTrue(any("lower bound" in r for r in d.reasons))

    def test_out_of_domain_and_bad_input(self):
        d = T.choose_attention_cut(A(16, 80, 8192, queries=256), ("both_cuts", "p_cut"))
        self.assertTrue(d.fallback)
        self.assertIsNone(d.predicted_ms)
        self.assertEqual(d.schedule.kind, "attention.p_cut")
        with self.assertRaises(ValueError):
            T.choose_attention_cut(self.W, ("mem",))


class ProjCut(unittest.TestCase):
    """P7's attention.proj_cut (MM 25.129): exposed, bit-identical to fused, and UNPRICED."""
    W = A(16, 128, 8192, queries=256, sets=2)

    def test_the_kind_string_matches_the_runtime_schedule(self):
        self.assertIn("proj_cut", T.ATTN_SCHEDULES)
        d = T.choose_attention_cut(self.W, ("proj_cut",))
        self.assertEqual(d.schedule.kind, "attention.proj_cut")     # runtime.ATTENTION_SCHEDULES' key

    def test_fused_is_preferred_and_no_ratio_is_guessed(self):
        d = T.choose_attention_cut(self.W, ("fused", "proj_cut"))
        self.assertEqual(d.schedule.kind, "attention.fused")
        self.assertTrue(any("proj_cut is unpriced" in r for r in d.reasons))
        self.assertNotIn("proj_cut", T.ATTN_CUT_RATIO)

    def test_alone_it_is_a_fallback_with_no_number(self):
        d = T.choose_attention_cut(self.W, ("proj_cut",))
        self.assertTrue(d.fallback)
        self.assertIsNone(d.predicted_ms)

    def test_a_priced_cut_beats_an_unpriced_one(self):
        self.assertEqual(T.choose_attention_cut(self.W, ("proj_cut", "p_cut")).schedule.kind, "attention.p_cut")


class KVSplit(unittest.TestCase):
    def test_never_split_without_the_opt_in(self):
        for w in (A(1, 128, 512, queries=16), A(32, 128, 16384, queries=16)):
            self.assertEqual(T.choose_kv_split(w).schedule.param("split"), 1)

    def test_the_measured_merged_split_is_4_ways(self):
        self.assertIn("| 4 | 17,594 | **2.52** | 31,663 | **3.32** |", open(
            os.path.join(ROOT, "docs", "g17-tensorops-accelerator-recon.md"), encoding="utf-8").read())
        for seq, x in ((512, 2.52), (1024, 3.32)):
            d = T.choose_kv_split(A(1, 128, seq, queries=16, allow_value_change=True))
            self.assertEqual(d.schedule.param("split"), 4)
            self.assertEqual(d.bound, "measured_ratio")
            self.assertIn(f"{x:.2f}", d.basis)

    def test_elsewhere_a_split_is_a_lower_bound_and_capped_at_8(self):
        d = T.choose_kv_split(A(32, 128, 16384, queries=16, allow_value_change=True))
        self.assertLessEqual(d.schedule.param("split"), 8)
        self.assertEqual(d.bound, "lower_bound")

    def test_the_value_change_is_stated(self):
        self.assertIn("0.4 to 4 percent", T.choose_kv_split.__doc__)
        d = T.choose(A(32, 128, 16384, queries=16))
        self.assertTrue(any("not value-preserving" in r for r in d.reasons))

class DecodeStepPlan(unittest.TestCase):
    """Bet 3's first milestone as a plan: each stage's floor is its weight bytes over DRAM bandwidth,
    the priced stages are exactly the scheduler's own answers, and the unpriced ones are named."""

    def test_the_floor_is_the_bytes_over_dram(self):
        p = T.decode_step(T.DecodeStep())
        weights = 4 * 2048 * 2048 * 2 + 3 * 2048 * 8192 * 2 + 2 * 2048 * 2
        kv = 2 * 256 * 2048 * 2
        self.assertEqual(weights + kv, sum(s.bytes for s in p.stages))
        self.assertAlmostEqual((weights + kv) / (T.C["bw_dram"].value * 1e9) * 1e3, p.floor_ms)
        # fp8 halves every weight byte but not the bf16 KV cache
        q = T.decode_step(T.DecodeStep(dtype="fp8"))
        self.assertEqual(weights // 2 + kv, sum(s.bytes for s in q.stages))

    def test_priced_stages_are_the_schedulers_answers(self):
        p = {s.name: s for s in T.decode_step(T.DecodeStep()).stages}
        self.assertEqual(T.choose(T.FFN(tokens=1, dtype="bf16")).predicted_ms, p["ffn"].predicted_ms)
        self.assertEqual(T.choose(A(heads=16, head_dim=128, seq=256, queries=1, dtype="bf16")).predicted_ms,
                         p["attention"].predicted_ms)

    def test_unpriced_stages_are_named_not_timed(self):
        p = T.decode_step(T.DecodeStep())
        self.assertEqual(["attn_norm", "qkv_proj", "o_proj", "ffn_norm"], p.unpriced)
        for s in p.stages:
            if s.name in p.unpriced:
                self.assertIsNone(s.predicted_ms)
                self.assertGreater(s.floor_ms, 0)

    def test_refusals(self):
        with self.assertRaises(ValueError):
            T.decode_step(T.DecodeStep(dtype="fp4"))
        with self.assertRaises(ValueError):
            T.decode_step(T.DecodeStep(heads=3))

if __name__ == "__main__":
    unittest.main()
