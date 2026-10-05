"""cc refuses what it cannot apply (MM 25.124.4): a nonzero offsetA/B/C on the single-body path, which
emitted the offset-free program (d87f982e3641 at offsets 0, 1,024 and 2,048), and overlapping fp32 C
regions on the memory-stream route. Compile-only.

This is the offset-consumed control at cc's layer. It mirrors Set C's tlower guard, which exercises
tlower.lower's base-register fold directly with single-body offsets, below cc's admission; that path is
untouched here.
"""
import hashlib
import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "test"))

from agxforge.g17 import cc, ir  # noqa: E402

STREAM = "results/g17-tensor-stream-v2"
STREAM_ARMS = ("stream2_M32N16K64x2", "stream4_M32N16K64x2", "stream8_M32N16K64x2", "stream16_M32N16K64x2")


def single(**kw):
    a = ir.Buffer("A", 1, elem=ir.F16); b = ir.Buffer("B", 2, elem=ir.F16); c = ir.Buffer("C", 3, elem=ir.F32)
    fn = ir.Function("k", [a, b, c]); bl = ir.Builder(fn, fn.block("entry"))
    bl.tensor_matmul(a, b, c, M=16, N=32, K=32, **kw)
    bl.ret(); ir.verify(fn)
    return fn


def pair(second):
    a = ir.Buffer("A", 1, elem=ir.F16); b = ir.Buffer("B", 2, elem=ir.F16); c = ir.Buffer("C", 3, elem=ir.F32)
    fn = ir.Function("k", [a, b, c]); bl = ir.Builder(fn, fn.block("entry"))
    bl.tensor_matmul(a, b, c, M=32, N=16, K=64, offsetC=1024)
    bl.tensor_matmul(c, b, c, M=32, N=16, K=16, a_dtype="float", b_dtype="half", offsetA=1024, **second)
    bl.ret(); ir.verify(fn)
    return fn


class SingleBodyOffsets(unittest.TestCase):
    def test_offset_zero_still_compiles_to_the_same_bytes(self):
        self.assertEqual(hashlib.sha256(cc.compile_function(single()).code).hexdigest()[:12], "d87f982e3641")

    def test_each_nonzero_offset_refuses_by_name(self):
        for key, value in (("offsetA", 1024), ("offsetB", 2048), ("offsetC", 2048)):
            with self.assertRaises(cc.Unsupported) as cm:
                cc.compile_function(single(**{key: value}))
            self.assertIn(f"{key}={value}", str(cm.exception))
            self.assertIn("does not apply offsets", str(cm.exception))

    def test_the_refusal_fires_only_on_a_nonzero_offset(self):
        # the control: an explicit zero is not an offset
        for key in ("offsetA", "offsetB", "offsetC"):
            self.assertEqual(hashlib.sha256(cc.compile_function(single(**{key: 0})).code).hexdigest()[:12],
                             "d87f982e3641")

    def test_a_route_that_applies_offsets_is_left_alone(self):
        # the independent group applies them: its retained program still compiles (25.102.2)
        import g17tensorcommonruntime as CR
        code = CR.build_generic_program(CR.generic_spec(dict(M=64, N=32, K=64, independent=2))).code
        self.assertEqual(hashlib.sha256(code).hexdigest()[:16], "8200484b82187c26")


class MemoryStreamOverlap(unittest.TestCase):
    def test_an_overlapping_pair_refuses_by_name(self):
        # body 2 writes [1024, 3072) over body 1's [1024, 3072) without accumulating; and a partial overlap
        for second in (dict(offsetC=1024), dict(offsetC=2048)):
            with self.assertRaises(cc.Unsupported) as cm:
                cc.compile_function(pair(second))
            self.assertIn("overlapping C regions", str(cm.exception))
            self.assertIsInstance(cc.tensor_route(pair(second)), cc.TensorRouteRefusal)

    def test_disjoint_and_exact_accumulate_are_admitted(self):
        self.assertEqual(cc.tensor_route(pair(dict(offsetC=4096))), "memory_stream")
        cc.compile_function(pair(dict(offsetC=4096)))
        # the measured sharing: an accumulate into exactly an earlier body's region (online-softmax O)
        a = ir.Buffer("A", 1, elem=ir.F16); b = ir.Buffer("B", 2, elem=ir.F16); c = ir.Buffer("C", 3, elem=ir.F32)
        fn = ir.Function("k", [a, b, c]); bl = ir.Builder(fn, fn.block("entry"))
        bl.tensor_matmul(a, b, c, M=32, N=16, K=64, offsetC=1024)
        bl.tensor_matmul(a, b, c, M=32, N=16, K=64, offsetB=2048, offsetC=1024, accumulate=True)
        bl.ret(); ir.verify(fn)
        self.assertEqual(cc.tensor_route(fn), "memory_stream")

    def test_the_retained_stream_programs_compile_byte_identical(self):
        from _evidence import require
        require(*[os.path.join(STREAM, arm, "program.bin") for arm in STREAM_ARMS],
                invariant="the retained online-softmax stream programs are unchanged by the overlap refusal")
        import g17tensorcommonruntime as CR
        for arm in STREAM_ARMS:
            spec = json.load(open(os.path.join(ROOT, STREAM, arm, "generic.json")))
            retained = open(os.path.join(ROOT, STREAM, arm, "program.bin"), "rb").read()
            self.assertEqual(CR.build_generic_program(spec).code, retained, arm)


if __name__ == "__main__":
    unittest.main()
