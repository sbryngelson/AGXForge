#!/usr/bin/env python3
"""The short-prompt q4 prefill route (MM 25.202): with cfg "qsm", g17prefillgraph.kernels resolves the qsm projections
(h16, xrows, the standalone w1 block) and their psum passes instead of the W16 GEMMs and the row kernels, only at 16-row
slices of q4 weights. CPU only (a fake index)."""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))


def _pick(asked):
    def pick(kind, **match):
        asked.append((kind, match.get("role"), tuple(sorted((match.get("variant") or {}).items()))))
        return dict(kind=kind, **match)
    return pick


class QsmRoute(unittest.TestCase):
    def test_the_route_asks_for_qsm_and_psums(self):
        import g17prefillgraph as PG
        asked = []
        k = PG.kernels(_pick(asked), 4, 2048, 128, route="mma", rego=True, gemm_M=16,
                       qsm=dict(qkv=4, wo=8, w1=2, w2=8))
        kinds = {a[0] for a in asked}
        self.assertIn("qsm", kinds)
        self.assertIn("psum", kinds)
        self.assertNotIn("qmm", kinds)
        self.assertNotIn("residual_rows", kinds)
        self.assertEqual(k["qsm_w1"]["role"], "w1_block")
        self.assertEqual(k["ps_w1"]["variant"], {"mode": "swiglu", "rows": 16, "sk": 2})
        self.assertTrue(all(dict(a[2]).get("h16") for a in asked if a[0] == "qsm"))

    def test_only_16_row_q4_slices(self):
        import g17prefillgraph as PG
        for bits, G in ((8, 16), (4, 32)):
            with self.subTest(bits=bits, G=G), self.assertRaises(ValueError):
                PG.kernels(_pick([]), bits, 2048, 128, route="mma", rego=True, gemm_M=G, qsm=dict(qkv=4, wo=8, w1=2, w2=8))

    def test_the_default_route_is_unchanged(self):
        import g17prefillgraph as PG
        asked = []
        PG.kernels(_pick(asked), 4, 2048, 128, route="mma", rego=True, gemm_M=128)
        kinds = {a[0] for a in asked}
        self.assertIn("qmm", kinds)
        self.assertNotIn("qsm", kinds)


class SpecDriver(unittest.TestCase):
    """MM 25.203: the verify step is the prefill section at M 16 on the scalar attention (runtime p0) and the qsm
    route; tools/g17specgen.m is its driver, built by `make native-tools`."""

    def test_the_spec_cfg(self):
        import g17q4graph as G
        saved = dict(G.CONFIG)
        try:
            G.CONFIG.clear(); G.CONFIG.update(spec={})
            self.assertEqual(G._spec_cfg(), dict(M=16, attn="scalar", gemm_M=16, qsm=dict(qkv=4, wo=8, w1=2, w2=8)))
        finally:
            G.CONFIG.clear(); G.CONFIG.update(saved)

    def test_the_scalar_route_resolves_with_qsm(self):
        import g17prefillgraph as PG
        asked = []
        k = PG.kernels(_pick(asked), 4, 2048, 16, route="scalar", gemm_M=16, qsm=dict(qkv=4, wo=8, w1=2, w2=8))
        self.assertEqual((k["append"]["kind"], k["attn"]["kind"]), ("prefill_attn", "prefill_attn"))
        self.assertIn("qsm_qkv", k)

    def test_the_driver_source_states_its_protocol(self):
        src = open(os.path.join(ROOT, "tools", "g17specgen.m")).read()
        self.assertIn("g17_gpu_cb", src)               # the machine GPU lock
        self.assertIn("--plain", src)                  # the same driver's baseline
        self.assertIn("tools/g17specgen:", open(os.path.join(ROOT, "Makefile")).read())


class CheckStep(unittest.TestCase):
    """MM 25.205: the verify step's 16-row argmax pass and its attention with decode's kvvec loads, bit-exact on
    g17emu; the default attention's bytes are the delivered ones."""

    def _run(self, job):
        import g17emu as E
        out = E.run_bundle(job["dir"], job["threads"], job["group"], job["base"], tier="wp")
        return job["check"](out[0] if isinstance(out, tuple) else out)

    def test_argmax_rows_pairs(self):
        import tempfile
        from pathlib import Path
        import g17deliver as DL
        with tempfile.TemporaryDirectory() as t:
            self.assertEqual(self._run(DL.build_argmax_rows(Path(t))[0]), 0)

    def test_kvvec_attention_and_the_default_bytes(self):
        import hashlib
        import tempfile
        from pathlib import Path
        import g17deliver as DL
        import g17prefillattn as P
        with tempfile.TemporaryDirectory() as t:
            self.assertEqual(self._run(DL.build_prefill(Path(t), 272, out16=True, kvvec=True)[0]), 0)
        lay = P.prefill_layout(2048, 2048, out16=True)
        self.assertEqual(hashlib.sha256(P.build_prefill_attn(lay).code).hexdigest()[:12], "6626c4a53278")


if __name__ == "__main__":
    unittest.main()
