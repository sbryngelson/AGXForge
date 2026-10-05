#!/usr/bin/env python3
"""Production row P7, the fused attention path (docs/g17-tensorops-machine-model.md 25.129): the named
attention class's admission and refusals, the half-out projection store, the causal mask's compile-time
elision, the memory-stream route's two new forms, and the preregistered program hashes. Offline:
nothing here dispatches."""
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import g17tensorcommonruntime as T  # noqa: E402
from agxforge.g17 import cc, ir, model, runtime, tensorreduce as TR, tlower  # noqa: E402

# The preregistered programs (25.129.1), committed before any dispatch. Only the two cheapest are
# recompiled here; tools/g17tensorattnfused.py predict recomputes all nine.
PREREGISTERED = {
    "cut_project": ({"new_blocks": 3, "causal": True, "q0": 16, "phase": "project"},
                    "fc76b5ce0cd92f611d4c4cec7410618a9cb104835ad6316919bd37a86e15397f"),
    "fused_n2": ({"new_blocks": 2}, "918cdfd080c9f2686a40c019410a292c1ca0ec51d07374db34c5eb21ce686b13"),
}


class TheAttentionClassAdmits(unittest.TestCase):
    def refused(self, code, **kw):
        with self.assertRaises(runtime.AttentionRefused) as ctx:
            runtime.attention_spec(kw)
        self.assertEqual(ctx.exception.code, code)
        self.assertTrue(str(ctx.exception).startswith("refused: " + code))

    def test_every_refusal_is_named(self):
        self.refused("attention_rows", rows=17)
        self.refused("attention_rows", rows=0)
        self.refused("attention_blocks", new_blocks=runtime.ATTENTION_MAX_BLOCKS + 1)
        self.refused("attention_blocks", cache_blocks=5, new_blocks=4)
        self.refused("attention_cache", new_blocks=0)
        self.refused("attention_shape", head=128)
        self.refused("attention_shape", value=64)
        self.refused("attention_shape", heads=2)
        self.refused("attention_launch", simdgroups=2)
        self.refused("attention_mask", mask="sliding_window")
        self.refused("attention_mask", causal=True, q0=-1)
        self.refused("attention_mask", q0=4)                       # q0 without the causal mask
        self.refused("attention_runtime_offset", runtime_offset=True)
        self.refused("attention_kv_split", kv_split=2)
        self.refused("attention_schedule", schedule="attention.fused")
        self.refused("attention_schedule", phase="both")
        self.refused("attention_cache", new_blocks=0, cache_blocks=1, phase="project")

    def test_the_admitted_request_is_normalised(self):
        a = runtime.attention_spec({"cache_blocks": 2, "new_blocks": 1, "causal": True})
        self.assertEqual((a["blocks"], a["q0"], a["visible"]), (3, 32, [0, 1, 2]))
        t = runtime.attention_spec({"new_blocks": 3, "causal": True, "q0": 16, "truncate": True})
        self.assertEqual(t["visible"], [0, 1])                     # block 2's keys 32.. follow every row's
        f = runtime.attention_spec({"new_blocks": 3, "causal": True, "q0": 16})
        self.assertEqual(f["visible"], [0, 1, 2])

    def test_both_schedules_are_exposed_by_name_and_nothing_chooses(self):
        sch = runtime.attention_schedules({"new_blocks": 3, "causal": True, "q0": 16})
        self.assertEqual(sorted(sch), ["attention.fused", "attention.proj_cut"])
        # P11's vocabulary (tensorsched.choose_attention_cut): each exposed kind names one of our schedules
        self.assertEqual(runtime.ATTENTION_P11_EXPOSED, ("fused", "proj_cut"))
        self.assertEqual({"attention." + k for k in runtime.ATTENTION_P11_EXPOSED}, set(runtime.ATTENTION_SCHEDULES))
        self.assertEqual([p["phase"] for p in sch["attention.fused"]], ["fused"])
        self.assertEqual([p["phase"] for p in sch["attention.proj_cut"]], ["project", "attend"])
        # a cache-only workload has nothing to project: its cut schedule is the attend phase alone
        self.assertEqual([p["phase"] for p in runtime.attention_schedules(
            {"cache_blocks": 2, "new_blocks": 0})["attention.proj_cut"]], ["attend"])

    def test_the_transport_is_a_tensor_spec_composition(self):
        lay = runtime.attention_layout(runtime.attention_spec({"new_blocks": 2}))
        kw = dict(M=lay["M"], N=80, K=64, lda=64, ldb=80, ldc=80, a_type="half", b_type="half", c_type="float",
                  simdgroups=1, grid=(32, 1, 1), threadgroup=(32, 1, 1), composition="attention")
        runtime.TensorSpec(**kw)
        for bad in (dict(simdgroups=2, threadgroup=(64, 1, 1), grid=(64, 1, 1)), dict(N=64, ldb=64, ldc=64),
                    dict(epilogue=("relu",))):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                runtime.TensorSpec(**dict(kw, **bad))

    def test_the_layout_fits_the_transport(self):
        for blocks in range(1, runtime.ATTENTION_MAX_BLOCKS + 1):
            a = runtime.attention_spec({"new_blocks": blocks})
            lay = runtime.attention_layout(a)
            self.assertLessEqual(lay["c_bytes"], lay["M"] * lay["N"] * 4)
            self.assertLessEqual(runtime.ATTENTION_A["X"] + 16 * blocks * 64 * 2, lay["M"] * lay["K"] * 2)
            self.assertLessEqual(lay["VC"] + blocks * 512, cc.TENSOR_STREAM_OFFSET_MAX)   # every B offset


class TheHalfOutStore(unittest.TestCase):
    def test_refusals(self):
        for kw in (dict(epilogue=(("half_kv",), ("relu",))), dict(epilogue=(("half_kv",),), keep=True),
                   dict(epilogue=(("half_kv",),), accumulate=True), dict(epilogue=(("half_kv",),), grid=2),
                   dict(epilogue=(("half_kv",),), offsets=(0, 0, 2))):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                tlower.lower(32 if kw.get("grid") else 16, 64, 64, 64, 64, 64, **kw)
        with self.assertRaises(ValueError):
            tlower.lower(16, 63, 64, 64, 63, 63, epilogue=(("half_kv",),))       # an odd row of halves

    def test_each_tile_narrows_eight_values_and_stores_four_words_where_the_layout_says(self):
        c_off = 8192
        body, plan = tlower.lower(16, 64, 64, 64, 64, 64, offsets=(0, 0, c_off), epilogue=(("half_kv",),))
        ops = [i.opcode.id for i in model.decode(body, 0) if i.opcode]
        self.assertEqual((ops.count(1016), ops.count(17202)), (8 * 4, 4 * 4))
        self.assertNotIn(17257, ops)                                # no fp32 store remains
        words = sorted(int(o["what"].split()[-1]) for o in plan["ops"]           # the movimm constants
                       if o["op"] == 11842 and o["what"].startswith(("half base words", "half second word")))
        # row m, column col of a keys x head half matrix is word (C_OFF + 2 (m 64 + col)) / 4; the lane
        # term idxC / 2 is added at run time, so the constants are the tile and row-half starts
        want = sorted(w for ni in range(4) for p in range(2)
                      for w in ((c_off + 2 * (8 * p * 64 + 16 * ni)) // 4,
                                (c_off + 2 * (8 * p * 64 + 16 * ni)) // 4 + 1))
        self.assertEqual(words, want)


class OneBufferTwoOffsets(unittest.TestCase):
    def test_a_and_b_in_one_binding_keep_their_own_offsets(self):
        # the first fused_n2 run (25.129.2): tlower keyed the operand offset by BINDING, so with P and V
        # both in buffer 3 the A loads took B's offset. Moving only A's offset must move the bytes.
        one = tlower.lower(32, 16, 16, 16, 16, 16, a_type="float", binds=(2, 2, 2), offsets=(0, 4096, 2048))[0]
        two = tlower.lower(32, 16, 16, 16, 16, 16, a_type="float", binds=(2, 2, 2), offsets=(256, 4096, 2048))[0]
        self.assertNotEqual(one, two)
        # and distinct bindings are unchanged by the fix: the same offsets, the same bytes as before
        self.assertEqual(tlower.lower(32, 16, 16, 16, 16, 16, a_type="float", binds=(2, 1, 2), offsets=(0, 4096, 2048))[0],
                         tlower.lower(32, 16, 16, 16, 16, 16, a_type="float", binds=(2, 1, 2), offsets=(0, 4096, 2048))[0])


class TheCausalMask(unittest.TestCase):
    def emitted(self, threshold):
        a = ir.Buffer("A", 1, elem=ir.F32)
        fn = ir.Function("m", [a]); bl = ir.Builder(fn, fn.block("entry"))
        lane = bl.builtin("thread_index_in_simdgroup", name="lane")
        vals = [bl.load(a, bl.const(i), type=ir.F32, name="v%d" % i) for i in range(4)]
        before = len(fn.blocks[0].ops)
        out = TR.apply_causal_mask(bl, lane, vals, threshold)
        kinds = [o.kind for o in fn.blocks[0].ops[before:]]
        return vals, out, kinds

    def test_what_the_compiler_can_decide_it_decides(self):
        vals, out, kinds = self.emitted(15)
        self.assertEqual((out, kinds), (vals, []))                   # nothing masked: no instruction
        vals, out, kinds = self.emitted(None)
        self.assertEqual((out, kinds), (vals, []))
        vals, out, kinds = self.emitted(-1)
        self.assertEqual(kinds.count("csel"), 0)                    # everything masked: one constant
        self.assertTrue(all(o is out[0] for o in out))
        vals, out, kinds = self.emitted(12)
        self.assertEqual(kinds.count("csel"), 3)                    # element 0 (bound 12) never masked
        self.assertIs(out[0], vals[0])

    def test_the_threshold_is_the_diagonal(self):
        self.assertEqual(TR.causal_threshold(16, 3, 16), 3)          # row 3 at position 19 sees keys 16..19
        self.assertEqual(TR.causal_threshold(16, 15, 32), -1)       # and none of block 2
        self.assertEqual(TR.CAUSAL_MASK_VALUE, float(np.float32(-1.0e30)))

    def test_the_reference_masks_the_same_columns(self):
        q0, row, key0 = 16, 5, 16
        t = TR.causal_threshold(q0, row, key0)
        cols = [c for c in range(16) if c > t]
        self.assertEqual(cols, list(range(6, 16)))
        self.assertTrue(all(key0 + c > q0 + row for c in cols))


class TheStreamRouteAdmitsOnlyThePathForms(unittest.TestCase):
    def group(self, *bodies):
        a = ir.Buffer("A", 1, elem=ir.F16); b = ir.Buffer("B", 2, elem=ir.F16); c = ir.Buffer("C", 3, elem=ir.F32)
        bufs = {"a": a, "b": b, "c": c}
        fn = ir.Function("f", [a, b, c]); bl = ir.Builder(fn, fn.block("entry"))
        for args, kw in bodies:
            bl.tensor_matmul(*[bufs[x] for x in args], **kw)
        return cc._memory_stream_group(fn, [o for blk in fn.blocks for o in blk.ops if o.kind == "tensor_matmul"])

    PROJ = (("a", "b", "c"), dict(M=16, N=64, K=64, offsetA=4096, offsetC=8192, epilogue=(("half_kv",),)))
    QK = (("a", "c", "c"), dict(M=32, N=16, K=64, transB=True, offsetB=8192))
    PV = (("c", "c", "c"), dict(M=32, N=16, K=16, a_dtype="float", offsetB=12288, offsetC=2048))

    def test_the_path_is_admitted(self):
        self.assertTrue(self.group(self.PROJ, self.QK, self.PV))

    def test_everything_next_to_it_stays_refused(self):
        bad_epi = (self.PROJ[0], dict(self.PROJ[1], epilogue=(("relu",),)))
        bad_trans = (("a", "b", "c"), dict(M=32, N=16, K=64, transB=True))       # transB off the cache
        acc_half = (self.PROJ[0], dict(self.PROJ[1], accumulate=True))
        consumer_half = (self.QK[0], dict(self.QK[1], epilogue=(("half_kv",),)))
        for body in (bad_epi, bad_trans, acc_half, consumer_half):
            with self.subTest(body=body):
                self.assertFalse(self.group(body, self.QK, self.PV))


class ThePreregisteredPrograms(unittest.TestCase):
    def test_hashes(self):
        for arm, (request, digest) in PREREGISTERED.items():
            with self.subTest(arm=arm):
                code = T.build_generic_program(T.generic_spec({"attention": request})).code
                self.assertEqual(hashlib.sha256(code).hexdigest(), digest)

    def test_the_reference_and_its_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = Path(tmp) / "b"
            T.author_generic(b, {"attention": {"new_blocks": 1}})
            s = T.generic_spec(json.loads((b / "generic.json").read_text()))
            ref = T.attention_reference(b, s)
            bound = T.attention_bound(b, s)
            lay = runtime.attention_layout(s["attention"])
            u = ref.view("<u4").ravel()
            # the new block's cache words are the projection rounded once to half, not the sentinel
            kc, vc, kf, vf = T.attention_cache(b, s)
            self.assertTrue(np.array_equal(kc[:16], kf[0].astype(np.float16)))
            self.assertNotIn(int(np.float32(T.ATTENTION_SENTINEL).view("<u4")), set(u[lay["KC"] // 4:lay["c_bytes"] // 4]))
            self.assertEqual(float(bound.ravel()[lay["KC"] // 4:].max()), 0.0)   # the cache compares bit for bit
            for claim in ("kv_fp32", "cache_transposed"):
                with self.subTest(claim=claim):
                    ctl = T.attention_reference(b, s, claim)
                    with np.errstate(invalid="ignore"):
                        outside = (ctl.view("<u4") != ref.view("<u4")) & (np.abs(ctl.astype(np.float64) - ref) > bound)
                    self.assertGreater(int(np.count_nonzero(outside)), 0)


if __name__ == "__main__":
    unittest.main()
