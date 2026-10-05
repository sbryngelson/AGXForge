#!/usr/bin/env python3
"""The matched study's twins (MM 25.211): each Metal twin in tools/twins/ compiles with Apple's front end and backend
flags the study uses, every twinned form maps to its twin's macros, and a form the twins do not reproduce is refused by
name rather than twinned approximately. CPU only (compile, no dispatch)."""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

QMV = dict(Kq=2048, Nout=4096, sgs=2, S=4194304, B=4456448, wpt=2, rows=1, coalesced=True, ksplit=True, group=64, chains=True)
ATTN = dict(wide=True, bfly_merge=True, kvvec=True, attn32=True, slices=32, heads=16, kv_heads=8, cap=2048, KOFF=135424,
            VOFF=4329728, OUT_AT=8524032, COST=8448, SINT=532736)


def _xcrun():
    return shutil.which("xcrun") and subprocess.run(["xcrun", "-sdk", "macosx", "-f", "metal"], capture_output=True).returncode == 0


class Twins(unittest.TestCase):
    def test_forms_map_to_their_twins(self):
        import g17twin as T
        self.assertEqual(T.twin_spec("qkv", QMV)[2]["EPI"], 0)
        self.assertEqual(T.twin_spec("wo_res1", dict(QMV, res="add16", RES=8192))[2]["EPI"], 1)
        self.assertEqual(T.twin_spec("w2_res2", dict(QMV, res="add32_to16", RES=8192))[2]["EPI"], 2)
        sw = T.twin_spec("ffn", dict(QMV, swiglu=True, ffn=8192, Nout=16384))[2]
        self.assertEqual((sw["EPI"], sw["FFN"]), (3, 8192))
        self.assertEqual(T.twin_spec("lm", dict(QMV, chains=False))[2]["CHAINS"], 0)
        self.assertEqual(T.twin_spec("attn", ATTN)[1], "attn_twin")
        self.assertEqual(T.kind_of("L7.attn_fused"), "attn")
        self.assertEqual(T.kind_of("head.gen_step"), "gen")

    def test_untwinned_forms_are_refused(self):
        import g17twin as T
        for kind, lay in (("qkv", dict(QMV, wpt=4)), ("qkv", dict(QMV, rows=4)), ("attn", dict(ATTN, qknorm=True)),
                          ("attn", dict(ATTN, keyblock=2)), ("norm", dict(d=4096, out32=True, rs_seed=True, in_dtype="half"))):
            with self.subTest(kind=kind, lay=sorted(set(lay.items()) ^ set((QMV if kind == "qkv" else ATTN).items()))):
                with self.assertRaises(ValueError):
                    T.twin_spec(kind, lay)

    @unittest.skipUnless(_xcrun(), "Apple's Metal toolchain")
    def test_every_twin_compiles(self):
        import g17twin as T
        cases = [T.twin_spec(k, l) for k, l in (("qkv", QMV), ("wo_res1", dict(QMV, res="add16", RES=8192)),
                                                ("ffn", dict(QMV, swiglu=True, ffn=8192, Nout=16384)), ("attn", ATTN),
                                                ("norm", dict(d=2048, out32=True, rs_seed=True, in_dtype="float", X=16384,
                                                              OUT=32768, G=512)))]
        cases.append(("misc_twin.metal", "gen_twin", dict(A_C=384, A_PL=12, A_G=241, A_CAP=2048, A_D=2048, A_RX16=8192,
                                                           A_GEN=12288, A_LOG=12292)))
        with tempfile.TemporaryDirectory() as t:
            for i, (src, fn, defs) in enumerate(cases):
                with self.subTest(src=src, fn=fn):
                    lib = T.metallib(src, Path(t) / str(i), defs)
                    self.assertGreater(lib.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
