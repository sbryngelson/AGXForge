#!/usr/bin/env python3
"""MM P13, operand and tile generality: gemm_generic's admitted-class table (agxforge.g17.runtime.
GENERIC_CLASSES, docs/g17-tensorops-machine-model.md section 25.127).

Every widening class is either admitted with its hardware receipt or refused BY ITS NAME, at spec time
(generic_spec), at contract time (TensorSpec) and, for the numeric bounds, in the native worker's
--describe-layout. An unknown spec key is refused instead of dropped. Compile and describe only: nothing
here dispatches."""
import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "tools")]

import numpy as np

from agxforge.g17 import runtime, tensorlife, tensorview, tlower
import g17tensorcommonruntime as R

# one request per class of the table, as a generic spec
CLASS_REQUESTS = {
    "wide_n": dict(M=32, N=256, K=64),
    "n_tiled_grid": dict(M=16, N=512, K=64, grid_n=2),
    "split_k_grid": dict(M=16, N=128, K=256, split_k=2),
    "transposed": dict(M=32, N=32, K=64, transA=True),
    "simdgroups_8": dict(M=128, N=32, K=64, simdgroups=8),
    "int8_split": dict(M=128, N=32, K=64, a="int8", b="int8", threadgroups=4),
    "narrow_out_half": dict(M=32, N=32, K=64, threadgroups=2, epilogue=["half"]),
    "simdgroups_16": dict(M=256, N=32, K=64, simdgroups=16),
    "sub_byte_int4": dict(M=32, N=32, K=64, a="int4", b="int4"),
    "sub_byte_fp4": dict(M=32, N=32, K=64, a="fp4e2m1", b="bfloat"),
    "sub_byte_fp6": dict(M=32, N=32, K=64, a="fp6e3m2", b="bfloat"),
    "mixed_accumulator": dict(M=32, N=32, K=64, c="half"),
    "narrow_out_bfloat": dict(M=32, N=32, K=64, threadgroups=2, epilogue=["bfloat"]),
    "int8_epilogue": dict(M=32, N=32, K=64, a="int8", b="int8", epilogue=["relu"]),
    "int8_chain": dict(M=32, N=32, K=64, a="int8", b="int8", stages=[[32, 32, "float"]]),
}


def refusal(fn, *args, **kw):
    try:
        fn(*args, **kw)
    except ValueError as error:
        return str(error)
    return None


def tensor_of(**kw):
    base = dict(M=32, N=32, K=64, lda=64, ldb=32, ldc=32, a_type="half", b_type="half", c_type="float",
                simdgroups=1, grid=(32, 1, 1), threadgroup=(32, 1, 1), composition="gemm_generic")
    base.update(kw)
    return base


class Table(unittest.TestCase):
    def test_every_class_is_admitted_with_a_receipt_or_refused_with_a_reason(self):
        for name, entry in runtime.GENERIC_CLASSES.items():
            with self.subTest(name):
                self.assertIn(entry["status"], ("admitted", "refused"))
                if entry["status"] == "admitted":
                    self.assertTrue(entry["receipt"] and entry["rule"])
                else:
                    self.assertTrue(entry["reason"])

    def test_the_requests_cover_the_table_and_classify_as_named(self):
        self.assertEqual(set(CLASS_REQUESTS), set(runtime.GENERIC_CLASSES))
        for name, spec in CLASS_REQUESTS.items():
            s = dict(spec)
            view = dict(a=s.get("a", "half"), b=s.get("b", "half"), c=s.get("c"), M=s["M"], N=s["N"], K=s["K"],
                        simdgroups=s.get("simdgroups", 1), groups=s.get("threadgroups", 1),
                        grid_n=s.get("grid_n", 1), split_k=s.get("split_k", 1),
                        transA=s.get("transA", False), transB=s.get("transB", False),
                        epilogue=s.get("epilogue", []), stages=bool(s.get("stages")), kloop=False, extras=[])
            with self.subTest(name):
                self.assertIn(name, runtime.generic_classes(view))

    def test_the_base_class_belongs_to_no_widening_class(self):
        view = dict(a="half", b="half", c=None, M=64, N=128, K=64, simdgroups=4, groups=2, transA=False,
                    transB=False, epilogue=["relu"], stages=False, kloop=False, extras=[])
        self.assertEqual(runtime.generic_classes(view), [])
        self.assertIsNone(runtime.generic_class_refusal(view))

    def test_n_tiled_grid_needs_power_of_two_tiles_per_threadgroup(self):
        # the column offset is a shift, so N/(16*grid_n) must be a power of two, not just grid_n. N=6144
        # grid_n=32 gives 12 tiles/tg -> outside the class (and the launch validator refuses it at spec
        # time with the 3 x 2048 hint) rather than admitting a spec tlower then refuses at build.
        def view(N, gn):
            return dict(a="half", b="half", c=None, M=16, N=N, K=64, grid_n=gn, groups=gn, simdgroups=1,
                        transA=False, transB=False, epilogue=[], stages=False, kloop=False, extras=[])
        self.assertFalse(runtime._p13_fits("n_tiled_grid", view(6144, 32)))     # 12 tiles/tg
        self.assertTrue(runtime._p13_fits("n_tiled_grid", view(2048, 16)))      # 8 tiles/tg
        self.assertTrue(runtime._p13_fits("n_tiled_grid", view(8192, 64)))      # 8 tiles/tg

        def launch(N, gn):
            return runtime.TensorSpec(M=16, N=N, K=64, lda=64, ldb=N, ldc=N, a_type="half", b_type="half",
                                      c_type="float", simdgroups=1, grid=(32 * gn, 1, 1),
                                      threadgroup=(32, 1, 1), grid_n=gn, composition="gemm_generic")
        with self.assertRaises(Exception) as cm:
            launch(6144, 32)
        self.assertIn("tile columns per threadgroup", str(cm.exception))
        launch(2048, 16)                                                        # a valid width does not raise

    def test_split_k_lifts_the_k_cap_to_per_threadgroup(self):
        # the K bound is per-threadgroup (K/split_k <= 4096, the kloop's 256 trips), so split_k lets the
        # total K exceed 4096. The FFN down projection K=8192 fits as ONE dispatch at split_k>=2.
        def launch(K, gk, gn=16):
            return runtime.TensorSpec(M=16, N=2048, K=K, lda=K, ldb=2048, ldc=2048, a_type="half",
                                      b_type="half", c_type="float", simdgroups=1,
                                      grid=(32 * gn * gk, 1, 1), threadgroup=(32, 1, 1),
                                      grid_n=gn, split_k=gk, composition="gemm_generic")
        launch(8192, 4)                                                         # per-tg K 2048: one dispatch
        launch(8192, 2)                                                         # per-tg K 4096
        # MM 25.144.1: the loop is bounded by TRIPS (<= 255), so 2+ slices per trip admit per-tg K 8192 (tlower
        # refuses a single-slice 8192 loop by its trip count); 16384 stays refused here
        launch(8192, 1)                                                         # per-tg K 8192: the boundary
        with self.assertRaises(Exception) as cm:
            launch(16384, 1)                                                    # per-tg K 16384 > 8192
        self.assertIn("per-threadgroup K", str(cm.exception))


class SpecTime(unittest.TestCase):
    """generic_spec: each refused class by its name, each admitted class builds."""

    def test_each_refused_class_refuses_by_name(self):
        for name, spec in CLASS_REQUESTS.items():
            if runtime.GENERIC_CLASSES[name]["status"] == "admitted":
                continue
            with self.subTest(name):
                why = refusal(R.generic_spec, spec)
                self.assertIsNotNone(why)
                self.assertIn("gemm_generic class %s is not admitted (MM P13" % name, why)

    def test_each_admitted_class_builds_and_passes_the_static_checks(self):
        for name, spec in CLASS_REQUESTS.items():
            if runtime.GENERIC_CLASSES[name]["status"] != "admitted":
                continue
            with self.subTest(name):
                s = R.generic_spec(spec)
                p = R.build_generic_program(s)
                self.assertEqual(tensorlife.released_reads(p.code), [])
                self.assertEqual(tensorlife.aliased_store_releases(p.code), [])
                self.assertEqual(tensorview.hazards(tensorview.view(p.code)), [])
                runtime.ImageContract.read(R.manifest_for(p, generic=s).model_dump(mode="json"))

    def test_outside_an_admitted_rule_refuses_by_the_class_name(self):
        outside = {
            "wide_n": [dict(M=32, N=272, K=64), dict(M=16, N=160, K=32, a="float", b="float"),
                       dict(M=32, N=256, K=64, epilogue=["relu"]), dict(M=32, N=256, K=512, kloop=True)],
            "transposed": [dict(M=32, N=32, K=64, transB=True, a="float", b="float"),
                           dict(M=32, N=32, K=64, transA=True, epilogue=["relu"])],
            "simdgroups_8": [dict(M=128, N=32, K=64, simdgroups=8, epilogue=["relu"])],
            # the K loop is inside the rule since MM 25.145.4; an extra feature (here a column reduction) is not
            "int8_split": [dict(M=128, N=32, K=64, a="int8", b="int8", threadgroups=4, reduce="colsum")],
            "narrow_out_half": [dict(M=32, N=32, K=64, epilogue=["half"]),                   # one threadgroup
                                dict(M=32, N=32, K=64, threadgroups=2, epilogue=["half", "relu"]),
                                dict(M=32, N=32, K=64, threadgroups=2, epilogue=["exp2", "half"])],
        }
        for name, specs in outside.items():
            for spec in specs:
                with self.subTest(name=name, spec=str(spec)):
                    why = refusal(R.generic_spec, spec)
                    self.assertIsNotNone(why)
                    self.assertIn("gemm_generic class %s is admitted only as" % name, why)

    def test_int8_admits_the_k_loop_with_a_grid(self):
        """MM 25.145.4: int8 x int8 -> int32 with the K loop under grid_n and 4 simdgroups, receipted bit-exact
        (isa/g17-int8-kloop-grid-results.json); K beyond tlower's 256 slices stays refused by the lowering."""
        s = R.generic_spec(dict(M=256, N=2048, K=2048, a="int8", b="int8", kloop=True, grid_n=64, simdgroups=4))
        self.assertTrue(s["kloop"])
        p = R.build_generic_program(s)
        self.assertEqual(tensorview.hazards(tensorview.view(p.code)), [])
        import json, os
        doc = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                          "isa", "g17-int8-kloop-grid-results.json")))
        arms = {a["name"]: a for a in doc["arms"]}
        self.assertTrue(set(runtime.GENERIC_CLASSES["int8_split"]["receipt"]) - {"int8_grid4", "int8_sg2_saturate"}
                        <= set(arms))
        self.assertTrue(all(a["gpu_vs_ref"] == 0 and a["gpu_vs_emu"] == 0 and a["gpu_status"] == "passed"
                            for a in arms.values()))

    def test_two_widening_classes_refuse_as_a_combination(self):
        why = refusal(R.generic_spec, dict(M=128, N=256, K=64, simdgroups=8))
        self.assertIn("wide_n + simdgroups_8 are admitted one at a time", why)

    def test_an_unknown_key_refuses_instead_of_building_something_else(self):
        for key, value in (("transpose_a", True), ("nsg", 8), ("tile", [32, 32, 32]), ("accumulator", "half")):
            with self.subTest(key):
                why = refusal(R.generic_spec, dict(M=32, N=32, K=64, **{key: value}))
                self.assertIn("gemm_generic spec key(s) %s are not in the admitted class table" % key, why)

    def test_negative_control_every_normalised_key_is_accepted(self):
        # the refusal must not fire on ground truth: a normalised spec (what generic.json holds) re-reads
        s = R.generic_spec(dict(M=64, N=32, K=64, simdgroups=2))
        self.assertEqual(R.generic_spec(json.loads(json.dumps(s))), s)
        self.assertTrue(set(s) <= R.GENERIC_SPEC_KEYS)

    def test_a_c_key_must_match_the_operands(self):
        self.assertEqual(R.generic_spec(dict(M=32, N=32, K=64, c="float"))["M"], 32)
        self.assertIn("accumulator is not a gemm_generic class", refusal(R.generic_spec, dict(M=32, N=32, K=64, c="int")))


class ContractTime(unittest.TestCase):
    """runtime.TensorSpec: the same table, before pydantic's Literals."""

    def test_refused_classes_refuse_by_name(self):
        for name, kw in (("sub_byte_int4", dict(a_type="int4", b_type="int4")),
                         ("sub_byte_fp4", dict(a_type="fp4e2m1")),
                         ("sub_byte_fp6", dict(b_type="fp6e2m3")),
                         ("simdgroups_16", dict(M=256, simdgroups=16, threadgroup=(512, 1, 1), grid=(512, 1, 1))),
                         ("mixed_accumulator", dict(c_type="half")),
                         ("narrow_out_bfloat", dict(epilogue=("bfloat",), grid=(64, 1, 1))),
                         ("int8_epilogue", dict(a_type="int8", b_type="int8", c_type="int", epilogue=("relu",)))):
            with self.subTest(name):
                why = refusal(runtime.TensorSpec, **tensor_of(**kw))
                self.assertIn("gemm_generic class %s is not admitted (MM P13" % name, why)

    def test_admitted_classes_validate(self):
        for kw in (dict(N=256, ldb=256, ldc=256), dict(transA=True), dict(transB=True),
                   dict(M=128, simdgroups=8, threadgroup=(256, 1, 1), grid=(256, 1, 1)),
                   dict(M=128, a_type="int8", b_type="int8", c_type="int", grid=(128, 1, 1)),
                   dict(M=64, a_type="int8", b_type="int8", c_type="int", simdgroups=2, threadgroup=(64, 1, 1),
                        grid=(64, 1, 1), accumulate=True, saturate=True),
                   dict(epilogue=("half",), grid=(64, 1, 1))):
            with self.subTest(**{k: str(v) for k, v in kw.items()}):
                runtime.TensorSpec(**tensor_of(**kw))

    def test_other_classes_keep_n_128_and_no_transpose(self):
        grid_class = dict(M=128, N=32, K=64, lda=64, ldb=32, ldc=32, a_type="half", b_type="half",
                          c_type="float", simdgroups=1, grid=(128, 1, 1), threadgroup=(32, 1, 1),
                          composition="gemm_grid")
        for kw in (dict(N=256, ldb=256, ldc=256), dict(transA=True), dict(simdgroups=8)):
            with self.subTest(**{k: str(v) for k, v in kw.items()}), self.assertRaises(ValueError):
                runtime.TensorSpec(**dict(grid_class, **kw))


class Lowering(unittest.TestCase):
    def test_the_and16_mask_is_unchanged_through_4_simdgroups_and_widens_at_8(self):
        # a mask of 3 at 8 simdgroups folds simdgroups 4..7 onto 0..3's rows (the sg_fold4 control)
        def mask(sg):
            from agxforge.g17 import model
            names = model.registers()
            body = tlower.lower(16 * sg, 32, 64, 64, 32, 32, sg=sg)[0]
            after_sr = False                             # the and16 that follows the SR_SIMD_GRP read
            for ins in model.decode(body, 0):
                if not ins.opcode:
                    continue
                if ins.opcode.id in (14059, 14060):
                    after_sr = any(k == "reg" and names.get(v) == "SR_SIMD_GRP" for k, v in ins.values)
                elif ins.opcode.id == 426 and after_sr:
                    return ins.values[-1][1]
        self.assertEqual([mask(sg) for sg in (2, 4, 8)], [3, 3, 7])

    def test_the_half_narrowing_is_four_conversions_and_two_word_stores_per_row_pair(self):
        body, plan = tlower.lower(32, 32, 64, 64, 32, 32, grid=2, epilogue=(("half",),))
        ops = [o["op"] for o in plan["ops"]]
        tiles = 1 * 2                                   # 16 rows per threadgroup x two column tiles
        self.assertEqual(ops.count(1016), 8 * tiles)
        self.assertEqual(ops.count(17202), 4 * tiles)
        self.assertNotIn(17257, ops)                    # no fp32 tensor store

    def test_the_half_reference_rounds_to_nearest_even_and_its_control_does_not(self):
        d = np.array([[1.0 + 2.0 ** -11, 1.0 + 3 * 2.0 ** -11, 70000.0, -1.0 - 2.0 ** -11]], dtype=np.float32)
        rne = R._half_out_buffer(d).view(np.uint16).reshape(-1)[:4]
        rz = R._half_out_buffer(d, rz=True).view(np.uint16).reshape(-1)[:4]
        self.assertEqual([hex(v) for v in rne], ["0x3c00", "0x3c02", "0x7c00", "0xbc00"])
        self.assertEqual([hex(v) for v in rz], ["0x3c00", "0x3c01", "0x7bff", "0xbc00"])


class WorkerBounds(unittest.TestCase):
    """The native worker's numeric bounds, in --describe-layout (no device, no pipeline, no dispatch)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="g17-p13-worker-")
        cls.worker = Path(cls.tmp.name) / "g17commonworker"
        R.build_worker(cls.worker)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def describe(self, manifest):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "manifest.json").write_text(json.dumps(manifest))
            return subprocess.run([str(self.worker), d, "--describe-layout"], capture_output=True, text=True,
                                  timeout=10)

    def manifest(self, spec):
        s = R.generic_spec(spec)
        return R.manifest_for(R.build_generic_program(s), generic=s).model_dump(mode="json")

    def test_admitted_classes_describe(self):
        for name, spec in CLASS_REQUESTS.items():
            if runtime.GENERIC_CLASSES[name]["status"] != "admitted":
                continue
            with self.subTest(name):
                result = self.describe(self.manifest(spec))
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIs(json.loads(result.stdout)["gpu_dispatched"], False)

    def test_past_the_bounds_the_worker_refuses(self):
        good = self.manifest(dict(M=128, N=32, K=64, simdgroups=8))
        for key, change in (("simdgroups 16", lambda t: t.update(simdgroups=16, threadgroup=[512, 1, 1])),
                            ("N 272", lambda t: t.update(N=272, ldb=272, ldc=272)),
                            ("transA false", lambda t: t.update(transA=False))):
            bad = copy.deepcopy(good)
            change(bad["tensor"])
            with self.subTest(key):
                result = self.describe(bad)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("tensor_spec", result.stderr)


if __name__ == "__main__":
    unittest.main()
