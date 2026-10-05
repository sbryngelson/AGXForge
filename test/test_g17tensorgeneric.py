#!/usr/bin/env python3
"""gemm_generic, the rule-based tensor runtime class (Set A with Set C): its rules, and that the
narrower classes keep their own domain (compile and validation only, no dispatch)."""
import copy
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "tools")]

from agxforge.g17 import runtime
import g17tensorcommonruntime as R


def spec_of(**kw):
    base = dict(M=48, N=32, K=64, lda=64, ldb=32, ldc=32, a_type="half", b_type="half", c_type="float",
                simdgroups=1, grid=(32, 1, 1), threadgroup=(32, 1, 1), composition="gemm_generic")
    base.update(kw)
    return base


class Rules(unittest.TestCase):
    def test_admitted(self):
        for kw in (dict(), dict(M=64, grid=(64, 1, 1)), dict(M=64, simdgroups=2, threadgroup=(64, 1, 1), grid=(64, 1, 1)),
                   dict(a_type="float", b_type="float"), dict(a_type="fp8e4m3", b_type="fp8e5m2"),
                   dict(a_type="bfloat", b_type="bfloat", epilogue=("scale:0x3f400000", "relu")),
                   dict(epilogue=("gelu",))):                                    # admitted since item 6
            with self.subTest(**{k: str(v) for k, v in kw.items()}):
                runtime.TensorSpec(**spec_of(**kw))

    def test_refused(self):
        for kw in (dict(M=40), dict(lda=65), dict(a_type="fp8e4m3", b_type="half"),
                   # combined splits now lower (item 3), but not with rows that are not whole tiles
                   dict(M=96, simdgroups=2, threadgroup=(64, 1, 1), grid=(128, 1, 1)),
                   dict(threadgroup=(64, 1, 1)), dict(grid=(96, 1, 1)), dict(epilogue=("tanh",)),
                   dict(K2=32)):
            with self.subTest(**{k: str(v) for k, v in kw.items()}):
                with self.assertRaises(ValueError):
                    runtime.TensorSpec(**spec_of(**kw))

    def test_other_classes_keep_their_domain(self):
        grid_class = dict(M=128, N=32, K=64, lda=64, ldb=32, ldc=32, a_type="half", b_type="half",
                          c_type="float", simdgroups=1, grid=(128, 1, 1), threadgroup=(32, 1, 1),
                          composition="gemm_grid")
        runtime.TensorSpec(**grid_class)
        for kw in (dict(simdgroups=2), dict(threadgroup=(64, 1, 1)), dict(epilogue=("relu",)), dict(grid=(512, 1, 1))):
            with self.subTest(**{k: str(v) for k, v in kw.items()}):
                with self.assertRaises(ValueError):
                    runtime.TensorSpec(**dict(grid_class, **kw))


class CombinedSplits(unittest.TestCase):
    def test_grid_and_simdgroup_splits_validate_together(self):
        # performance item 3: 8+ simdgroups per core needs both splits; up to 256 threadgroups
        for kw in (dict(M=64, simdgroups=2, threadgroup=(64, 1, 1), grid=(128, 1, 1)),
                   dict(M=4096, simdgroups=4, threadgroup=(128, 1, 1), grid=(8192, 1, 1))):
            with self.subTest(**{k: str(v) for k, v in kw.items()}):
                runtime.TensorSpec(**spec_of(**kw))
        for kw in (dict(M=8192, simdgroups=1, threadgroup=(32, 1, 1), grid=(32, 1, 1)),     # 512 tile rows
                   dict(M=4096, simdgroups=2, threadgroup=(64, 1, 1), grid=(64, 1, 1))):    # 128 per simdgroup
            with self.subTest(**{k: str(v) for k, v in kw.items()}), self.assertRaisesRegex(ValueError, "per simdgroup"):
                runtime.TensorSpec(**spec_of(**kw))
        with self.assertRaises(ValueError):          # 3 threadgroups is not a power of two
            runtime.TensorSpec(**spec_of(M=96, simdgroups=1, threadgroup=(32, 1, 1), grid=(96, 1, 1)))


class Programs(unittest.TestCase):
    def test_the_split_body_declares_the_split_set_and_validates(self):
        s = R.generic_spec(dict(M=64, N=32, K=64, simdgroups=2))
        p = R.build_generic_program(s)
        self.assertEqual(tuple(p.abi()["system_registers"]), (130, 133, 156))
        runtime.ImageContract.read(R.manifest_for(p, generic=s).model_dump(mode="json"))

    def test_a_grid_body_has_no_scalar_epilogue(self):
        s = R.generic_spec(dict(M=64, N=32, K=64, threadgroups=2))
        p = R.build_generic_program(s)
        self.assertEqual(tuple(p.abi()["system_registers"]), (130, 156))


if __name__ == "__main__":
    unittest.main()
