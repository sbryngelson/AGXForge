#!/usr/bin/env python3
"""MM P12, closed by its second branch: production refuses cooperative threadgroup-memory SHARING by
name, at spec, contract, compile and worker-contract time, while the admitted 1/2/4-simdgroup tile
split (gemm_generic) still builds. Compile and describe only: nothing here dispatches."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

from pydantic import ValidationError

from agxforge.g17 import cc, ir, runtime
import g17tensorcommonruntime as R

PREFIX = "refused: cooperative threadgroup-memory sharing is not admitted in production (MM P12"
# spellings a caller might use; each must refuse, never be dropped and built as the tile split
SHARING_KEYS = ("sharing", "shared", "shared_a", "shared_b", "share_operands", "cooperative",
                "cooperative_sharing", "threadgroup_memory", "Threadgroup_Memory_Bytes", "tg_mem")


def refusal_text(error):
    """The refusal's own text: a pydantic ValidationError wraps the validator's ValueError."""
    if isinstance(error, ValidationError):
        texts = [str(e.get("ctx", {}).get("error", "")) for e in error.errors()]
        return next((t for t in texts if t.startswith("refused:")), "; ".join(texts))
    return str(error)


def generic_tensor(**kw):
    base = dict(M=64, N=32, K=64, lda=64, ldb=32, ldc=32, a_type="half", b_type="half", c_type="float",
                simdgroups=2, grid=(64, 1, 1), threadgroup=(64, 1, 1), composition="gemm_generic")
    base.update(kw)
    return base


def tile_split_manifest(simdgroups=2):
    s = R.generic_spec(dict(M=64, N=32, K=64, simdgroups=simdgroups))
    p = R.build_generic_program(s)
    return p, R.manifest_for(p, generic=s).model_dump(mode="json")


class SpecRefusals(unittest.TestCase):
    """generic_spec read known keys with .get and dropped the rest: a sharing key built silently."""

    def test_every_sharing_key_refuses_by_name(self):
        for key in SHARING_KEYS:
            for value in (True, False, 0, "a"):          # the KEY is the request, whatever its value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError) as caught:
                    R.generic_spec(dict(M=64, N=32, K=64, simdgroups=2, **{key: value}))
                self.assertTrue(str(caught.exception).startswith(PREFIX), str(caught.exception))
                self.assertIn(key, str(caught.exception))

    def test_no_admitted_spec_key_reads_as_a_sharing_request(self):
        # the negative control: the refusal must accept ground truth - every key the normaliser
        # knows, and every raw key a real bundle carries
        s = R.generic_spec(dict(M=64, N=32, K=64, simdgroups=4))
        self.assertEqual(runtime.sharing_request(s), [])
        self.assertEqual(runtime.sharing_request(generic_tensor()), [])

    def test_author_refuses_before_writing_a_bundle(self):
        with tempfile.TemporaryDirectory() as td:
            bundle = Path(td) / "b"
            with self.assertRaises(ValueError) as caught:
                R.author_generic(bundle, dict(M=64, N=32, K=64, simdgroups=2, sharing=True))
            self.assertTrue(str(caught.exception).startswith(PREFIX))
            self.assertFalse(bundle.exists())

    def test_cli_prints_the_named_refusal_once(self):
        with tempfile.TemporaryDirectory() as td:
            env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
            result = subprocess.run(
                [sys.executable, str(ROOT / "tools/g17tensorcommonruntime.py"), str(Path(td) / "b"),
                 "--prepare", "--composition", "generic",
                 "--generic", json.dumps(dict(M=64, N=32, K=64, simdgroups=2, cooperative=True))],
                cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertTrue(result.stderr.startswith(PREFIX), result.stderr)
            self.assertFalse((Path(td) / "b").exists())


class ContractRefusals(unittest.TestCase):
    def test_tensor_spec_sharing_keys_refuse_by_name(self):
        for key in SHARING_KEYS:
            with self.subTest(key=key), self.assertRaises(ValueError) as caught:
                runtime.TensorSpec(**generic_tensor(**{key: True}))
            self.assertTrue(refusal_text(caught.exception).startswith(PREFIX), refusal_text(caught.exception))

    def test_manifest_variants_refuse_by_name(self):
        _p, value = tile_split_manifest()
        runtime.ImageContract.read(value)                       # the admitted form, unchanged
        variants = {
            "tensor sharing key": lambda v: v["tensor"].update(sharing=True),
            "abi uses threadgroup": lambda v: v["abi"].update(uses_threadgroup=True),
            "abi threadgroup block": lambda v: v["abi"].update(threadgroup=dict(
                required_size=[64, 1, 1], static_memory_bytes=4096, static_memory_alignment=4,
                dynamic_memory=[])),
        }
        for name, change in variants.items():
            bad = copy.deepcopy(value)
            change(bad)
            with self.subTest(name), self.assertRaises(ValueError) as caught:
                runtime.ImageContract.read(bad)
            self.assertTrue(refusal_text(caught.exception).startswith(PREFIX), refusal_text(caught.exception))


class CompileRefusal(unittest.TestCase):
    """cc: a program that executes a tensor form and uses or declares threadgroup memory."""

    def tensor_fn(self, declare=False, touch=False):
        a, b, c = ir.Buffer("A", 1, elem=ir.F16), ir.Buffer("B", 2, elem=ir.F16), ir.Buffer("C", 3, elem=ir.F32)
        fn = ir.Function("tensor_sharing_probe", [a, b, c])
        bb = ir.Builder(fn, fn.block("entry"))
        bb.tensor_matmul(a, b, c, M=16, N=16, K=16)
        if touch:
            tid = bb.builtin("thread_position_in_grid", name="tid")
            bb.store_tg(tid, tid)                  # a threadgroup-memory store: TG_STORE_OPCODE in the layout
        bb.ret()
        if declare:
            fn.declare_threadgroup(32)
        return fn

    def test_declared_or_used_threadgroup_memory_refuses_by_name(self):
        for declare, touch in ((True, False), (True, True), (False, True)):
            with self.subTest(declare=declare, touch=touch):
                with self.assertRaises(cc.CooperativeSharingRefused) as caught:
                    cc.compile_function(self.tensor_fn(declare, touch))
                self.assertIsInstance(caught.exception, ValueError)
                self.assertTrue(str(caught.exception).startswith(PREFIX), str(caught.exception))

    def test_the_same_tensor_program_without_threadgroup_memory_compiles(self):
        p = cc.compile_function(self.tensor_fn())
        self.assertFalse(p.abi()["uses_threadgroup"])

    def test_the_admitted_tile_split_builds_and_validates(self):
        for sg in (1, 2, 4):
            with self.subTest(simdgroups=sg):
                p, value = tile_split_manifest(sg)
                self.assertFalse(p.abi()["uses_threadgroup"])
                self.assertNotIn("threadgroup", p.abi())
                contract = runtime.ImageContract.read(value)
                self.assertEqual(contract.tensor.simdgroups, sg)
                self.assertEqual(contract.tensor.threadgroup[0], 32 * sg)


class WorkerRefusal(unittest.TestCase):
    """The native worker's contract, in --describe-layout (no device, no pipeline, no dispatch)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="g17-sharing-worker-")
        cls.worker = Path(cls.tmp.name) / "g17commonworker"
        R.build_worker(cls.worker)
        _p, cls.value = tile_split_manifest()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def describe(self, manifest):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "manifest.json").write_text(json.dumps(manifest))
            return subprocess.run([str(self.worker), d, "--describe-layout"],
                                  capture_output=True, text=True, timeout=10)

    def test_the_admitted_tile_split_describes(self):
        result = self.describe(self.value)
        self.assertEqual(result.returncode, 0, result.stderr)
        layout = json.loads(result.stdout)
        self.assertIs(layout["gpu_dispatched"], False)
        self.assertEqual(layout["tensor"]["simdgroups"], 2)

    def test_every_sharing_request_refuses_with_the_named_phase(self):
        variants = {("tensor", key): (lambda v, k=key: v["tensor"].update({k: True})) for key in SHARING_KEYS}
        variants[("abi", "uses_threadgroup")] = lambda v: v["abi"].update(uses_threadgroup=True)
        variants[("abi", "threadgroup")] = lambda v: v["abi"].update(threadgroup={"static_memory_bytes": 4096})
        variants[("manifest", "sharing")] = lambda v: v.update(sharing=True)
        for name, change in variants.items():
            bad = copy.deepcopy(self.value)
            change(bad)
            with self.subTest(name):
                result = self.describe(bad)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("phase=tensor_cooperative_sharing", result.stderr)
                self.assertNotIn("create_pipeline", result.stderr)


if __name__ == "__main__":
    unittest.main()
