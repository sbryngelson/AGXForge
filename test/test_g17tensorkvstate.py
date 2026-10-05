#!/usr/bin/env python3
"""Production row P9, the stateful attention runtime (docs/g17-tensorops-machine-model.md 25.131): the
KVCache ownership contract, phase step's admission and named refusals, the one-program-for-every-length
property, the runtime mask, the append at a runtime row, the merge, and the preregistered program
hashes. Offline: nothing here dispatches."""
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
from agxforge.g17 import cc, ir, runtime, tensorreduce as TR  # noqa: E402

STEP = {"phase": "step", "capacity_blocks": 2}
# The preregistered programs (25.131.1), committed before any dispatch. Only the cheapest are recompiled.
PREREGISTERED_CAP8 = "23a88fefcdc1875c"


def program(request, **top):
    return T.build_generic_program(T.generic_spec(dict(top, attention=dict(request)))).code


class TheStepAdmits(unittest.TestCase):
    def refused(self, code, **kw):
        with self.assertRaises(runtime.AttentionRefused) as ctx:
            runtime.attention_spec(dict(kw))
        self.assertEqual(ctx.exception.code, code)
        self.assertTrue(str(ctx.exception).startswith("refused: " + code))

    def test_every_refusal_is_named(self):
        self.refused("kv_capacity", phase="step")
        self.refused("kv_capacity", phase="step", capacity_blocks=0)
        self.refused("kv_capacity", phase="step", capacity_blocks=runtime.ATTENTION_MAX_BLOCKS + 1)
        self.refused("kv_step", phase="step", capacity_blocks=2, cache_blocks=1)      # the length is runtime
        self.refused("kv_step", phase="step", capacity_blocks=2, q0=16)
        self.refused("kv_step", phase="step", capacity_blocks=2, new_blocks=2)
        self.refused("kv_step", phase="step", capacity_blocks=2, runtime_offset=True)
        self.refused("kv_step", phase="step", capacity_blocks=2, causal=False)
        self.refused("attention_rows", phase="step", capacity_blocks=2, rows=17)
        self.refused("attention_launch", phase="step", capacity_blocks=2, simdgroups=2)
        self.refused("attention_kv_split", phase="step", capacity_blocks=2, kv_split=2)   # no opt-in
        self.refused("attention_kv_split", phase="step", capacity_blocks=4, kv_split=4, allow_value_change=True)
        self.refused("attention_kv_split", phase="step", capacity_blocks=1, kv_split=2, allow_value_change=True)
        # P7's phases still refuse a runtime offset and a split, and now name where they live
        with self.assertRaises(runtime.AttentionRefused) as ctx:
            runtime.attention_spec({"runtime_offset": True})
        self.assertIn("phase step", str(ctx.exception))
        with self.assertRaises(runtime.AttentionRefused) as ctx:
            runtime.attention_spec({"kv_split": 2})
        self.assertIn("allow_value_change", str(ctx.exception))

    def test_the_admitted_step(self):
        a = runtime.attention_spec(dict(STEP))
        self.assertEqual((a["q0"], a["visible"], a["ranges"], a["kv_split"]), (None, [0, 1], [[0, 1]], 1))
        s = runtime.attention_spec({"phase": "step", "capacity_blocks": 5, "kv_split": 2, "allow_value_change": True})
        self.assertEqual(s["ranges"], [[0, 1, 2], [3, 4]])


class TheKVCacheOwnsTheState(unittest.TestCase):
    def test_the_region_is_disjoint_from_the_scratch_and_inside_p10s_domains(self):
        for cap in range(1, runtime.ATTENTION_MAX_BLOCKS + 1):
            for split in (1, 2) if cap > 1 else (1,):
                kv = runtime.KVCache(cap, kv_split=split, allow_value_change=split == 2)
                lay = kv.layout
                reg = kv.region
                self.assertEqual(reg["k"][0], runtime.ATTENTION_C["KC"])            # after S, O, M, L
                self.assertEqual((reg["k"][1], reg["v"][1]), (lay["VC"], lay["SK"]))
                self.assertEqual(reg["k"][1] - reg["k"][0], cap * 16 * 128)
                self.assertLessEqual(lay["c_bytes"], lay["M"] * lay["N"] * 4)
                self.assertLessEqual(lay["LEN"] + 4, lay["M"] * lay["K"] * 2)       # the uniform fits buffer 1
                # every tensor B offset the step reads (K and V cache blocks) is in P10's measured even domain,
                # every C offset fp32-aligned
                for j in range(cap):
                    for off in (lay["KC"] + 2048 * j, lay["VC"] + 512 * j):
                        self.assertTrue(cc._measured_tensor_stream_offset(off), off)
                for off in (lay["SK"], lay["SV"]) + tuple(lay[k] for k in ("O1",) if k in lay):
                    self.assertEqual(off % 4, 0)
                    self.assertTrue(cc._measured_tensor_stream_offset(off) or off > cc.TENSOR_STREAM_OFFSET_MAX)

    def test_the_length_is_host_owned_and_bounded(self):
        kv = runtime.KVCache(2)
        self.assertEqual(kv.length_word(), (0).to_bytes(4, "little"))
        self.assertEqual(kv.advance(), 16)
        self.assertEqual(kv.length_word(), (16).to_bytes(4, "little"))
        self.assertEqual(kv.advance(), 32)                   # full: a state
        for step in (kv.length_word, kv.advance):             # the step after it is refused
            with self.assertRaises(runtime.AttentionRefused) as ctx:
                step()
            self.assertEqual(ctx.exception.code, "kv_length")
        with self.assertRaises(runtime.AttentionRefused):
            runtime.KVCache(2, length=33)
        with self.assertRaises(runtime.AttentionRefused):
            runtime.KVCache(2, length=17).length_word()
        runtime.KVCache(2, length=13).length_word()           # any whole number of tokens
        with self.assertRaises(runtime.AttentionRefused):     # the bundle author refuses the same step
            T.generic_spec({"attention": dict(STEP), "kv_length": 17})

    def test_zero_fill_and_truncate_keep_masked_rows_finite(self):
        kv = runtime.KVCache(3, length=40 - 8)
        img = bytearray(b"\xff" * kv.layout["c_bytes"])
        kv.zero_fill(img)
        lay = kv.layout
        self.assertEqual(bytes(img[lay["KC"]:lay["SK"]]), bytes(lay["SK"] - lay["KC"]))
        self.assertEqual(img[lay["KC"] - 1], 0xFF)           # nothing outside the cache touched
        ranges = kv.truncate(8)
        self.assertEqual(kv.length, 8)
        self.assertEqual(ranges, [(lay["KC"] + 8 * 128, lay["KC"] + 32 * 128), (lay["VC"] + 8 * 32, lay["VC"] + 32 * 32)])
        with self.assertRaises(runtime.AttentionRefused):
            kv.truncate(9)

    def test_the_sync_contract_states_its_parts(self):
        c = runtime.KV_SYNC_CONTRACT
        self.assertIs(c["completion_wait_between_steps"], False)
        for key in ("queue", "order", "wait_needed_for", "length", "not_admitted"):
            self.assertTrue(c[key])


class OneProgramForEveryLength(unittest.TestCase):
    def test_the_length_is_an_input_not_a_program_fact(self):
        codes = {program(STEP, kv_length=n) for n in (0, 5, 16)}
        self.assertEqual(len(codes), 1)
        # and it is READ: the program contains the word load of buffer 1 at the uniform's word index
        # (removing the step's runtime pieces - the P7 fused program of the same size - is different bytes)
        self.assertNotEqual(program(STEP), program({"new_blocks": 1, "cache_blocks": 1, "causal": True}))

    def test_the_preregistered_capacity_8_program(self):
        code = program({"phase": "step", "capacity_blocks": 8})
        self.assertEqual(hashlib.sha256(code).hexdigest()[:16], PREREGISTERED_CAP8)

    def test_p7s_phases_keep_their_bytes(self):
        code = program({"new_blocks": 2})
        self.assertEqual(hashlib.sha256(code).hexdigest(),
                         "918cdfd080c9f2686a40c019410a292c1ca0ec51d07374db34c5eb21ce686b13")


class TheRuntimeMask(unittest.TestCase):
    def emit(self, row, key0):
        a = ir.Buffer("A", 1, elem=ir.F32)
        fn = ir.Function("m", [a]); bl = ir.Builder(fn, fn.block("entry"))
        lane = bl.builtin("thread_index_in_simdgroup", name="lane")
        length = bl.load(a, bl.const(0), type=ir.I32, name="len")
        vals = [bl.load(a, bl.const(i + 1), type=ir.F32, name="v%d" % i) for i in range(4)]
        before = len(fn.blocks[0].ops)
        out = TR.apply_causal_mask(bl, lane, vals, TR.RuntimeThreshold(length, row, key0, runtime.KV_MASK_BIAS))
        return fn.blocks[0].ops[before:], out

    def test_every_element_is_one_select_with_a_non_negative_compare(self):
        ops, out = self.emit(row=0, key0=112)
        self.assertEqual([o.kind for o in ops].count("csel"), 4)
        consts = [o.attrs.get("value", o.attrs.get("v")) for o in ops if o.kind == "const"]
        # col + 128 against length + (128 + row - key0 - i): the smallest constant is 128 + 0 - 112 - 3
        self.assertGreaterEqual(runtime.KV_MASK_BIAS + 0 - 112 - 3, 0)
        self.assertEqual(len(out), 4)

    def test_the_bias_must_cover_the_key(self):
        with self.assertRaises(ValueError):
            TR.RuntimeThreshold(None, 0, 128, runtime.KV_MASK_BIAS)

    def test_the_runtime_rule_is_the_compile_time_rule(self):
        # masked iff col > length + row - key0, which is causal_threshold(length, row, key0)
        for length in (0, 16, 37):
            for row in (0, 7, 15):
                for key0 in (0, 16, 48):
                    t = TR.causal_threshold(length, row, key0)
                    for col in range(16):
                        biased = col + runtime.KV_MASK_BIAS > length + runtime.KV_MASK_BIAS + row - key0
                        self.assertEqual(biased, col > t)


class TheReferenceAndItsControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.b = Path(cls.tmp.name) / "b"
        T.author_generic(cls.b, {"attention": dict(STEP), "kv_length": 11})
        cls.s = T.generic_spec(json.loads((cls.b / "generic.json").read_text()))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def outside(self, claim):
        ref = T.attention_reference(self.b, self.s)
        bound = T.attention_bound(self.b, self.s)
        ctl = T.attention_reference(self.b, self.s, claim)
        with np.errstate(invalid="ignore"):
            return int(np.count_nonzero((ctl.view("<u4") != ref.view("<u4")) & (np.abs(ctl.astype(np.float64) - ref) > bound)))

    def test_the_uniform_is_in_buffer_1_and_the_block_lands_at_the_runtime_row(self):
        self.assertEqual(T.kv_length_of(self.b), 11)
        kc, vc, k16, v16, q0 = T.kv_step_cache(self.b, self.s)
        self.assertEqual(q0, 11)
        self.assertTrue(np.array_equal(kc[11:27], k16))
        self.assertTrue(np.array_equal(vc[11:27], v16))
        self.assertFalse(kc[27:].any())                      # zero-filled past the new block
        bound = T.attention_bound(self.b, self.s).ravel()
        lay = runtime.attention_layout(self.s["attention"])
        self.assertEqual(float(bound[lay["KC"] // 4:lay["c_bytes"] // 4].max()), 0.0)   # cache and staging exact

    def test_the_controls_fail(self):
        for claim in ("length_zero", "unmasked"):
            with self.subTest(claim=claim):
                self.assertGreater(self.outside(claim), 0)

    def test_scratch_does_not_reach_the_next_step(self):
        # the sequence's premise (25.131.1 prediction 3): a step reads only the cache from buffer 3, so
        # perturbing S, O, M, L before it leaves every word it writes unchanged
        c = np.frombuffer((self.b / "c.f32").read_bytes(), dtype="<f4").copy()
        c[:runtime.ATTENTION_C["KC"] // 4] = np.float32(7.25)
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "c.f32"
            f.write_bytes(c.tobytes())
            p = Path(tmp) / "p"
            T.author_generic(p, {"attention": dict(STEP), "kv_length": 11, "c_from": str(f)})
            sp = T.generic_spec(json.loads((p / "generic.json").read_text()))
            touched = sorted(T._attention_touched(sp))
            a = T.attention_reference(self.b, self.s).ravel().view("<u4")
            b = T.attention_reference(p, sp).ravel().view("<u4")
            self.assertTrue(np.array_equal(a[touched], b[touched]))


class TheMerge(unittest.TestCase):
    def test_a_masked_second_range_is_exactly_the_sequential_chain(self):
        # keys past length + 15 are masked for every row; when the second range holds only those, its
        # merge weight is exp2(-1e30 - m) = 0 and the split equals the chain (25.131.2: bitwise on hardware)
        with tempfile.TemporaryDirectory() as tmp:
            sp = Path(tmp) / "split"
            T.author_generic(sp, {"attention": {"phase": "step", "capacity_blocks": 2, "kv_split": 2,
                                                "allow_value_change": True}, "kv_length": 0})
            s = T.generic_spec(json.loads((sp / "generic.json").read_text()))
            o = slice(512, 768)
            split = T.attention_reference(sp, s).ravel().view("<u4")[o]
            chain = T.attention_reference(sp, s, "unsplit").ravel().view("<u4")[o]
            self.assertTrue(np.array_equal(split, chain))
            ref = T.attention_reference(sp, s)
            bound = T.attention_bound(sp, s)
            ctl = T.attention_reference(sp, s, "no_merge_scale")
            with np.errstate(invalid="ignore"):
                self.assertGreater(int(np.count_nonzero((ctl.view("<u4") != ref.view("<u4")) &
                                                        (np.abs(ctl.astype(np.float64) - ref) > bound))), 0)


if __name__ == "__main__":
    unittest.main()
