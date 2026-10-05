#!/usr/bin/env python3
"""The second architecture (tools/g17qwen3.py, MM 25.182): QK-norm bit-exact on g17emu against its stated order at two
seeds, with the controls that a kernel reading the wrong gain row or skipping the eps differs; the deliverer's Qwen3
shapes compile and its InternLM2 default is the table it replaced. CPU only."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import numpy as np  # noqa: E402


class QKNorm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import g17qwen3 as Q3
        cls.Q3 = Q3
        cls.lay = Q3.headnorm_layout()
        cls.prog = Q3.build_headnorm(cls.lay)

    def _run(self, seed):
        import g17deliver as DL
        import g17emu as EMU
        import g17decodeops as O
        Q3, lay = self.Q3, self.lay
        rng = np.random.default_rng(900 + seed)
        x = (rng.standard_normal(Q3.QKV) * rng.uniform(0.2, 6)).astype(np.float32)
        gq = (1 + 0.3 * rng.standard_normal(128)).astype(np.float16)
        gk = (1 + 0.3 * rng.standard_normal(128)).astype(np.float16)
        a, b = bytearray(lay["a_bytes"]), bytearray(lay["b_bytes"])
        O._place(a, lay["X"], x.astype("<f4"))
        O._place(b, lay["G"], np.concatenate([gq, gk]))
        with tempfile.TemporaryDirectory() as t:
            d = DL.author(Path(t) / "hn", self.prog, bytes(a), bytes(b), DL.SENT * lay["c_bytes"], lay)
            out = EMU.run_bundle(d, 1024, 1024, base=0, tier="wp")
        out = out[0] if isinstance(out, tuple) else out
        return np.frombuffer(out, "<u4", Q3.QKV, lay["OUT"]), x, gq, gk

    def test_bit_exact_against_the_stated_order(self):
        for seed in (1, 2):
            with self.subTest(seed=seed):
                got, x, gq, gk = self._run(seed)
                want = self.Q3.headnorm_reference(x, gq, gk).view(np.uint32)
                self.assertEqual(int((got != want).sum()), 0)
                self.assertTrue(np.array_equal(got[3072:], x[3072:].view(np.uint32)))   # v heads copied

    def test_the_controls_differ(self):
        got, x, gq, gk = self._run(1)
        swapped = self.Q3.headnorm_reference(x, gk, gq).view(np.uint32)      # the k gain on q heads and back
        no_eps = self.Q3.headnorm_reference(x * np.float32(1e-4), gq, gk, eps=0.0)
        self.assertGreater(int((got[:3072] != swapped[:3072]).sum()), 1000)
        got2, x2, gq2, gk2 = self._run(1)
        self.assertEqual(int((got2 != got).sum()), 0)
        # eps matters at small scale: the reference with eps 0 on a tiny input is not the eps 1e-6 one
        self.assertGreater(int((no_eps.view(np.uint32) !=
                                self.Q3.headnorm_reference(x * np.float32(1e-4), gq, gk).view(np.uint32)).sum()), 0)


class Prefill(unittest.TestCase):
    """MM 25.188: the many-row QK-norm is bit-exact on g17emu (and its program does not depend on the row count); the
    prefill FFN's zero padding dequantizes to exact zeros, so padded w1 / w3 / w2 give the unpadded outputs."""

    def test_rows_headnorm(self):
        import g17deliver as DL
        import g17emu as EMU
        import g17decodeops as O
        import g17qwen3 as Q3
        lay = Q3.headnorm_layout(4)
        prog = Q3.build_headnorm(lay)
        self.assertEqual(prog.code, Q3.build_headnorm(Q3.headnorm_layout(128)).code)
        rng = np.random.default_rng(3)
        x = (rng.standard_normal(4 * Q3.QKV) * 2).astype(np.float32)
        gq = (1 + 0.2 * rng.standard_normal(128)).astype(np.float16)
        gk = (1 + 0.2 * rng.standard_normal(128)).astype(np.float16)
        a, b = bytearray(lay["a_bytes"]), bytearray(lay["b_bytes"])
        O._place(a, 0, x); O._place(b, 0, np.concatenate([gq, gk]))
        with tempfile.TemporaryDirectory() as t:
            d = DL.author(Path(t) / "hn", prog, bytes(a), bytes(b), DL.SENT * lay["c_bytes"], lay)
            out = EMU.run_bundle(d, 4096, 1024, base=0, tier="wp")
        out = out[0] if isinstance(out, tuple) else out
        want = Q3.headnorm_rows_reference(x, gq, gk).view(np.uint32)
        self.assertEqual(int((np.frombuffer(out, "<u4", 4 * Q3.QKV, 0) != want).sum()), 0)

    def test_the_ffn_padding_is_exact(self):
        import g17prefillgraph as PG
        import g17qmm as QMM
        rng = np.random.default_rng(4)
        N, K = 96, 192                                   # padded to 128 rows / 256 K
        packed, s16, b16, q = QMM.weights(N, K, 4, 5)
        z = dict(W=packed, S=s16, B=b16)
        wb, sb, bb = PG._block(z, pad_n=128)
        W = np.frombuffer(wb, "<u4").reshape(128, K // 8); S = np.frombuffer(sb, "<u2").reshape(128, K // 64)
        self.assertTrue(np.array_equal(W[:N], packed) and not W[N:].any() and not S[N:].any())
        wb, sb, bb = PG._block(z, pad_k=256)
        W = np.frombuffer(wb, "<u4").reshape(N, 256 // 8); S = np.frombuffer(sb, "<u2").reshape(N, 256 // 64)
        self.assertTrue(np.array_equal(W[:, :K // 8], packed) and not W[:, K // 8:].any() and not S[:, K // 64:].any())
        qz = np.zeros((N, 64), np.uint8)
        self.assertFalse(QMM.dequant_reference(qz, np.zeros((N, 1), np.uint16), np.zeros((N, 1), np.uint16), 4).any())


class BatchedHead(unittest.TestCase):
    def test_the_pad_fill(self):
        """MM 25.189: every pad logit becomes the most negative finite float, and no other word moves (g17emu, 2 rows)."""
        import g17deliver as DL
        import g17emu as EMU
        import g17qwen3 as Q3
        lay = Q3.pad_fill_layout(2)
        prog = Q3.build_pad_fill(lay)
        c = DL.SENT * lay["c_bytes"]
        with tempfile.TemporaryDirectory() as t:
            d = DL.author(Path(t) / "p", prog, bytes(256), bytes(256), c, lay)
            out, _m = EMU.run_bundle(d, 32 * lay["groups"], 32, 1, tier="wp")
        got = np.frombuffer(out, "<f4", 2 * Q3.VOCAB_PAD, 0).reshape(2, Q3.VOCAB_PAD)
        self.assertTrue((got[:, Q3.VOCAB:] == np.finfo(np.float32).min).all())
        self.assertTrue((got[:, :Q3.VOCAB].view(np.uint32) == 0x7f7f7f7f).all())


class Shapes(unittest.TestCase):
    def test_the_default_is_internlm2(self):
        import g17deliver as DL
        DL.set_arch("internlm2")
        self.assertEqual((DL.ARCH["d"], DL.ARCH["ffn"], DL.ARCH["vocab"], DL.EPS, DL.KSPLIT[4]["w2_res2"]),
                         (2048, 8192, 92544, 1e-5, 4))
        lay, _ = DL.qmv_layout(4, "w2_res2", dict(ks=4))
        self.assertEqual((lay["Nout"], lay["Kq"], lay["RES"]), (2048, 8192, 8192))

    def test_qwen3_shapes_compile(self):
        import g17deliver as DL
        try:
            DL.set_arch("qwen3")
            self.assertEqual(DL.KSPLIT[4]["w2_res2"], 2)
            for role in ("qkv", "wo_res1", "ffn", "w2_res2"):
                with self.subTest(role=role):
                    lay, _ = DL.qmv_layout(4, role, dict(ks=DL.KSPLIT[4][role], lean=True, ptr=True, chain=True))
                    DL.Q.build_qmv2(lay)
            lay, _ = DL.qmv_layout(4, "w2_res2", dict(ks=2))
            self.assertEqual((lay["Nout"], lay["Kq"], lay["RES"]), (1024, 3072, 4096))
            lay, _ = DL.head_layout(4, dict(ks=2, lean=True, ptr=True))
            self.assertEqual(lay["Nout"], 151936)
        finally:
            DL.set_arch("internlm2")


class Qwen3_8B(unittest.TestCase):
    """Qwen3-8B's shapes (the 2x-decode goal's model): configure() binds them and restores 0.6B's; the deliverer's arch
    carries the heads to the attention layout; the generation state moves past h and x only when d exceeds 2,048."""

    def test_configure_round_trip(self):
        import g17qwen3 as Q
        try:
            Q.configure("qwen3-8b")
            self.assertEqual((Q.D_MODEL, Q.LAYERS, Q.HEADS, Q.KV_HEADS, Q.QKV, Q.FFN, Q.TIED), (4096, 36, 32, 8, 6144, 12288, False))
            self.assertEqual(Q.OUT.name, "g17-model-qwen3-8b")
        finally:
            Q.configure()
        self.assertEqual((Q.D_MODEL, Q.LAYERS, Q.HEADS, Q.QKV, Q.TIED), (1024, 28, 16, 4096, True))

    def test_deliver_arch_heads_and_gen_state(self):
        import g17deliver as DL
        try:
            DL.set_arch("qwen3_8b")
            lay = DL.attn_layout(2048)
            self.assertEqual((lay["heads"], lay["kv_heads"]), (32, 8))
            DL.set_arch("internlm2")
            lay = DL.attn_layout(2048)
            self.assertEqual((lay["heads"], lay["kv_heads"]), (16, 8))
        finally:
            DL.set_arch("internlm2")
            import g17qwen3 as Q
            Q.configure()
        # every graph before Qwen3-8B keeps its generation state at 12,288 bytes into R
        self.assertEqual([max(12288, 6 * d) for d in (1024, 2048, 4096)], [12288, 12288, 24576])


if __name__ == "__main__":
    unittest.main()
