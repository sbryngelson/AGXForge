#!/usr/bin/env python3
"""The multi-vector q4 projection (MM 25.208): g17qmvw's kl8 and r4 forms run bit-exact against their stated order in
g17emu at nb 1 and 4, and g17prefillgraph's qmvw route resolves the qmvw projections, fp32-out norms and SwiGLU rows,
the split-1 psum passes and the fp32-out attention. CPU only."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))


class Qmvw(unittest.TestCase):
    def test_both_forms_bit_exact_in_the_emulator(self):
        import g17emu as E
        import g17qmvw as W
        for form in ("kl8", "r4"):
            for nb in (1, 4):
                with self.subTest(form=form, nb=nb):
                    lay = W.layout(64, 1024, nb, form=form)
                    prog = W.build(lay)
                    packed, s16, b16, q, x = W.case(lay)
                    want = W.reference(lay, q, s16, b16, x)
                    a, bb, c = W.io(lay, packed, s16, b16, x)
                    with tempfile.TemporaryDirectory() as t:
                        d = W.author(Path(t) / "w", prog, a, bb, c, lay)
                        out = E.run_bundle(d, 64 * lay["groups"], 64, 1, tier="wp")
                    out = out[0] if isinstance(out, tuple) else out
                    got = np.frombuffer(out, "<u4", nb * lay["N"])
                    self.assertEqual(int((got != want.view(np.uint32).reshape(-1)).sum()), 0)

    def test_the_prefill_route_asks_for_qmvw(self):
        import g17prefillgraph as PG
        asked = []

        def pick(kind, **match):
            asked.append((kind, match.get("role"), dict(match.get("variant") or {})))
            e = dict(kind=kind, **match, S=0, B=0, recipe=dict(layout=dict(a_bytes=256)))
            return e
        k = PG.kernels(pick, 4, 2048, 4, gemm_M=4, qmvw=4, attn_opts=["hw_exp2", "kvvec"])
        kinds = {a[0] for a in asked}
        self.assertIn("qmvw", kinds)
        self.assertNotIn("qsm", kinds)
        self.assertNotIn("qmm", kinds)
        self.assertEqual(k["qmvw_w1"]["role"], "w1_block")
        self.assertEqual(k["ps_w2"]["variant"], {"mode": "fold16", "rows": 4, "sk": 1})
        self.assertEqual(k["attn"]["variant"], {"scalar": True, "hw_exp2": True, "kvvec": True})
        self.assertTrue(all(a[2].get("out32") for a in asked if a[0] == "norm"))
        with self.assertRaises(ValueError):
            PG.kernels(pick, 8, 2048, 4, gemm_M=4, qmvw=4)


if __name__ == "__main__":
    unittest.main()
