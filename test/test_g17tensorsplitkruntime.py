#!/usr/bin/env python3
"""Split-K through the generic runtime, GPU-free: generic_spec keeps split_k, build_generic_program emits
the partial body (no SR156 tail), manifest_for launches split_k threadgroups and sizes C for the stacked
(split_k*M) x N partials, and generic_reference's partials fold ascending-t fp32 to exactly Piece A's
gemm_reference(split_k=G). Covers split_k alone and combined with the grid_n column grid. The hardware
receipt is in the machine model (25.134); this pins the plumbing and the fold contract off-GPU."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import numpy as np

from agxforge.g17 import runtime
import g17tensorcommonruntime as R
import g17decodestep as D


class SplitKRuntime(unittest.TestCase):
    def _prepare(self, tmp, gen):
        bundle = Path(tmp) / "b"
        R.author_generic(bundle, gen)
        return bundle

    def test_generic_spec_keeps_split_k_and_admits_the_class(self):
        s = R.generic_spec(dict(M=16, N=128, K=2048, kloop=True, split_k=8))
        self.assertEqual(s["split_k"], 8)
        view = runtime._generic_view(dict(a_type="half", b_type="half", c_type="float", M=16, N=128, K=2048,
                                          simdgroups=1, grid=[256, 1, 1], threadgroup=[32, 1, 1], split_k=8))
        self.assertIn("split_k_grid", runtime.generic_classes(view))
        self.assertIsNone(runtime.generic_class_refusal(view))

    def test_manifest_stacks_the_partials_and_launches_split_k_threadgroups(self):
        s = R.generic_spec(dict(M=16, N=128, K=2048, kloop=True, split_k=8))
        prog = R.build_generic_program(s)
        man = R.manifest_for(prog, generic=s)
        self.assertEqual(man.tensor.split_k, 8)
        self.assertEqual(man.tensor.grid[0], 32 * 8)               # 8 threadgroups
        self.assertEqual((man.shape.rows, man.shape.columns), (8 * 16, 128))   # (split_k*M) x N

    def test_partials_fold_to_piece_a_reference(self):
        for gen in (dict(M=16, N=128, K=2048, kloop=True, split_k=8),
                    dict(M=16, N=2048, K=2048, kloop=True, grid_n=16, split_k=2)):
            with self.subTest(**gen), tempfile.TemporaryDirectory() as tmp:
                s = R.generic_spec(gen)
                bundle = self._prepare(tmp, gen)
                ref = R.generic_reference(bundle, s)
                G, M, N, K = s["split_k"], s["M"], s["N"], s["K"]
                self.assertEqual(ref.shape, (G * M, N))
                a = R._generic_values((bundle / "a.f16").read_bytes(), "half")[:M * K].reshape(M, K)
                b = R._generic_values((bundle / "b.f16").read_bytes(), "half")[:K * N].reshape(K, N)
                acc = ref[:M].astype(np.float32)
                for t in range(1, G):
                    acc = (acc + ref[t * M:(t + 1) * M]).astype(np.float32)
                self.assertTrue(np.array_equal(acc, D.gemm_reference(a, b, split_k=G)))
                # split-K is a different value than the single chain, and the tolerance states the change
                self.assertGreater(D.split_k_tolerance(a, b, split_k=G), 0.0)

    def test_split_k_1_is_the_plain_program(self):
        # split_k defaults to 1 and the manifest then carries no extra threadgroups or stacked rows
        s = R.generic_spec(dict(M=16, N=128, K=256, kloop=True))
        self.assertEqual(s["split_k"], 1)
        man = R.manifest_for(R.build_generic_program(s), generic=s)
        self.assertEqual((man.shape.rows, man.shape.columns), (16, 128))


if __name__ == "__main__":
    unittest.main()
