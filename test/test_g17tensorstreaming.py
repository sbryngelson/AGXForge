#!/usr/bin/env python3
"""Online-softmax attention over two key blocks (Set A goal item 6, docs/g17-tensorops-machine-model.md 25.114):
the memory-stream composition route, the streaming row stages, and the reference with its derived
enclosure. Offline: nothing here dispatches."""
import collections
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
import g17tensorcommonruntime as R  # noqa: E402
from agxforge.g17 import cc, ir, model, tensorlife  # noqa: E402

# Programs that ran on hardware and must keep their bytes: one-dispatch attention at 2 and 16 rows
# (machine model 25.110), whose row softmax shares the row addressing this work extended
# with a base offset, and the GELU register epilogue at M32 (machine model 25.109).
DISPATCHED = [
    ({"M": 32, "N": 16, "K": 64, "stages": [[16, 16, "float"]], "between": "softmax:2"}, "51c90d46e4cffc86"),
    ({"M": 32, "N": 16, "K": 64, "stages": [[16, 16, "float"]], "between": "softmax:16"}, "33525e6aeeb0c65a"),
    ({"M": 32, "N": 32, "K": 64, "epilogue": ["gelu"]}, "e0d64a81da846316"),
]
# v2 programs (docs/g17-tensorops-machine-model.md 25.114.2, v2 preregistration). v1's two-row program (a1cd96de3ae87697) passed on hardware
# by timing, carrying the same eighteen dead loads that broke rows 2 and 9 of the 16-row one.
STREAM = {2: "a31f5327225144c5", 4: "61fef36888018333", 8: "ed98e2b7893093b6", 16: "8f1d7e05d32f293a"}
STREAM2 = STREAM[2]


def stream_bundle(tmp, **kw):
    b = Path(tmp) / "b"
    R.author_generic(b, dict(M=32, N=80, K=64, stream=2, **kw))
    return b, R.generic_spec(json.loads((b / "generic.json").read_text()))


class DispatchedProgramsKeepTheirBytes(unittest.TestCase):
    def test_every_dispatched_program_is_byte_identical(self):
        for spec, digest in DISPATCHED:
            with self.subTest(spec=spec):
                code = R.build_generic_program(spec).code
                self.assertEqual(hashlib.sha256(code).hexdigest()[:16], digest)


class TheStreamRoute(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.program = R.build_generic_program(dict(M=32, N=80, K=64, stream=2))
        cls.ops = collections.Counter(i.opcode.id for i in model.decode(cls.program.code, 0) if i.opcode)

    def test_the_two_row_program_is_the_preregistered_one(self):
        self.assertEqual(hashlib.sha256(self.program.code).hexdigest()[:16], STREAM2)

    def test_four_bodies_two_score_gemms_and_two_value_gemms(self):
        # two 32x16x64 half bodies (2 tiles x 4 issues each: 12 op5106 + 4 no-C op5107) and two
        # 32x16x16 float x half bodies (2 tiles, 1 issue each: 4 no-C op5101)
        self.assertEqual((self.ops[5106], self.ops[5107], self.ops[5101]), (12, 4, 4))
        # per row: 4 exp2 for P1, 1 for alpha, 4 for P2; one recip for 1/l
        self.assertEqual((self.ops[1272], self.ops[3658]), (18, 2))

    def test_the_bytes_are_clean_and_the_metadata_set_is_measured(self):
        self.assertEqual(tensorlife.released_reads(bytes(self.program.code)), [])
        self.assertEqual(tuple(self.program.abi()["system_registers"]), (130, 156))

    def test_the_route_is_taken_only_where_every_older_route_refuses(self):
        a = ir.Buffer("A", 1, elem=ir.F16); b = ir.Buffer("B", 2, elem=ir.F16); c = ir.Buffer("C", 3, elem=ir.F32)
        fn = ir.Function("f", [a, b, c]); bl = ir.Builder(fn, fn.block("entry"))
        bl.tensor_matmul(a, b, c, M=32, N=32, K=64)
        bl.tensor_matmul(c, b, c, M=32, N=32, K=32, a_dtype="float")
        ops = [o for blk in fn.blocks for o in blk.ops if o.kind == "tensor_matmul"]
        self.assertFalse(cc._memory_stream_group(fn, ops))          # the released chain keeps its route
        fn2 = ir.Function("g", [a, b, c]); bl2 = ir.Builder(fn2, fn2.block("entry"))
        bl2.tensor_matmul(a, b, c, M=32, N=16, K=64, offsetC=0)
        bl2.tensor_matmul(a, b, c, M=32, N=16, K=64, offsetB=2048, offsetC=2048)
        ops2 = [o for blk in fn2.blocks for o in blk.ops if o.kind == "tensor_matmul"]
        self.assertTrue(cc._memory_stream_group(fn2, ops2))          # a later body reads buffer 1
        ops2[1].attrs["transA"] = True
        self.assertFalse(cc._memory_stream_group(fn2, ops2))         # and nothing it does not state

    def test_rows_past_the_measured_range_refuse(self):
        for kw in (dict(stream=17), dict(stream=2, N=64), dict(stream=2, simdgroups=2)):
            spec = dict(dict(M=32, N=80, K=64), **kw)
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                R.generic_spec(spec)


class TheReference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.bundle, cls.s = stream_bundle(cls.tmp.name)
        cls.e = {k: v // 4 for k, v in R.STREAM_C.items()}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def O(self, flat):
        return np.asarray(flat, dtype=np.float64).ravel()[self.e["O"]: self.e["O"] + 512].reshape(32, 16)

    def test_the_reference_lies_in_its_enclosure_and_only_transcendental_elements_have_a_bound(self):
        bound = R.stream_bound(self.bundle, self.s)
        self.assertEqual(self.O(bound)[2:].max(), 0.0)            # rows past the softmaxed ones: bit exact
        self.assertGreater(self.O(bound)[:2].min(), 0.0)

    def test_the_one_shot_attention_is_inside_the_online_enclosure_and_outside_the_controls(self):
        # exact base-2 softmax attention over all 32 keys, in float64: the equivalence online softmax
        # exists to keep, checked against an enclosure that contains exact arithmetic
        exact = R.stream_one_shot(self.bundle, self.s)
        for model_name, inside in (("online", True), ("no_alpha", False), ("first_block", False)):
            lo, hi = R.stream_enclosure(self.bundle, self.s, mode="exact", model=model_name)
            with self.subTest(model=model_name):
                self.assertEqual(bool(np.all((exact >= self.O(lo)[:2]) & (exact <= self.O(hi)[:2]))), inside)

    def test_the_no_alpha_control_differs_exactly_where_the_running_max_moved(self):
        ref = R.stream_reference(self.bundle, self.s).astype(np.float64).ravel()
        bound = R.stream_bound(self.bundle, self.s).ravel()
        ctl = np.asarray(R._stream_trace(self.bundle, dict(self.s, stream_model="no_alpha"), R._StreamPoint()),
                         dtype=np.float64)
        q, k1, k2, *_ = R._stream_operands(self.bundle)
        moved = [r for r in range(2) if (q @ k2).max(axis=1)[r] > (q @ k1).max(axis=1)[r]]
        beyond = np.abs(ctl - ref) > bound
        rows = sorted({int(i - self.e["O"]) // 16 for i in np.nonzero(beyond)[0]
                       if self.e["O"] <= i < self.e["O"] + 512})
        self.assertTrue(moved)
        self.assertEqual(rows, moved)

    def test_a_subnormal_is_flushed_in_both_arithmetics(self):
        tiny = 2.0 ** -130
        self.assertEqual(R._StreamPoint().fmul(tiny, 1.0), 0.0)
        self.assertEqual(R._StreamInterval().fmul((tiny, tiny), (1.0, 1.0)), (0.0, 0.0))
        self.assertEqual(R._StreamPoint().fmul(2.0 ** -120, 1.0), 2.0 ** -120)   # the control


class NoLoadIsLeftUnread(unittest.TestCase):
    """The v1 defect: a four-word row load with one word used; the dead loads' registers were
    reused while the loads were in flight. cc now refuses that, and the stats read one word."""

    def test_every_row_count_compiles_to_its_preregistered_program(self):
        for rows, digest in STREAM.items():
            with self.subTest(rows=rows):
                code = R.build_generic_program(dict(M=32, N=80, K=64, stream=rows)).code
                self.assertEqual(hashlib.sha256(code).hexdigest()[:16], digest)

    def test_the_check_refuses_the_v1_four_word_stat_read(self):
        from agxforge.g17 import tensorreduce as TR
        fixed = TR._row_stat

        def four_words(builder, buffer, *, row, base):
            _l, _a, _i, values = TR._load_values(builder, buffer, row=row, stride=16, M=32, base=base)
            return values[0]
        TR._row_stat = four_words
        saved = cc.DEAD_LOAD_KINDS
        try:
            # with dead-load elimination off, the emission check refuses the v1 program...
            cc.DEAD_LOAD_KINDS = ()
            with self.assertRaises(cc.Unsupported):
                R.build_generic_program(dict(M=32, N=80, K=64, stream=2))
            # ...and with it on (the default) the dead loads never reach emission
            cc.DEAD_LOAD_KINDS = saved
            R.build_generic_program(dict(M=32, N=80, K=64, stream=2))
        finally:
            TR._row_stat = fixed
            cc.DEAD_LOAD_KINDS = saved

    def test_the_check_fires_on_an_alu_write_not_only_a_second_load(self):
        L = [(0, b"", cc.MInst("load.14", 14, dict(_defs=[5], _uses=[2]))),
             (14, b"", cc.MInst("alu.12", 12, dict(_defs=[5], _uses=[3])))]
        self.assertEqual(cc.unread_load_destination_reuse(L), [(0, 14, 5)])
        L[1] = (14, b"", cc.MInst("alu.12", 12, dict(_defs=[6], _uses=[5])))   # the control: read first
        self.assertEqual(cc.unread_load_destination_reuse(L), [])

    def test_an_unread_load_is_dropped_before_selection(self):
        a = ir.Buffer("A", 1, elem=ir.F32)
        fn = ir.Function("d", [a]); bl = ir.Builder(fn, fn.block("entry"))
        bl.load(a, bl.const(0), type=ir.F32, name="dead")
        live = bl.load(a, bl.const(1), type=ir.F32, name="live")
        bl.store_at(a, bl.const(2), live)
        bl.ret()
        self.assertEqual(cc._drop_unread_loads(fn), 1)
        self.assertEqual([o.kind for blk in fn.blocks for o in blk.ops].count("load"), 1)
        self.assertEqual(cc._drop_unread_loads(fn), 0)              # the live one stays

    def test_a_loaded_recip_source_goes_through_the_waiting_copy(self):
        a = ir.Buffer("A", 1, elem=ir.F32)
        fn = ir.Function("r", [a]); bl = ir.Builder(fn, fn.block("entry"))
        x = bl.load(a, bl.const(0), type=ir.F32, name="x")
        bl.store_at(a, bl.const(1), bl.recip(x, name="inv"))
        bl.ret()
        forms = [m.form for _o, _b, m in cc.compile_function(fn).layout]
        i = forms.index("float.unary")
        self.assertEqual(forms[i - 1], "alu.12")


if __name__ == "__main__":
    unittest.main()
