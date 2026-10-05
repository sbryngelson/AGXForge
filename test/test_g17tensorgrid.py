#!/usr/bin/env python3
"""The grid split of a tensor body: G threadgroups each compute an M/G row block (compile only)."""
import hashlib
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agxforge.g17 import cc, ir, model, tlower

NAMES = model.registers()


def srs(body):
    return [NAMES.get(v) for i in model.decode(body, 0) if i.opcode and i.opcode.id in (14059, 14060)
            for k, v in i.values if k == "reg" and str(NAMES.get(v, "")).startswith("SR")]


class GridSplit(unittest.TestCase):
    def test_only_a_split_body_reads_the_threadgroup_index(self):
        self.assertEqual(srs(tlower.lower(128, 32, 64, 64, 32, 32)[0]), ["SR_SIMD_ELEM"])
        self.assertEqual(srs(tlower.lower(128, 32, 64, 64, 32, 32, grid=4)[0]), ["SR_SIMD_ELEM", "SR_TG_X"])

    def test_each_threadgroup_runs_one_block_of_tiles(self):
        whole = tlower.lower(128, 32, 64, 64, 32, 32)[1]
        split = tlower.lower(128, 32, 64, 64, 32, 32, grid=4)[1]
        self.assertEqual(split["rows_per_threadgroup"], 32)
        mma = lambda plan: sum(1 for o in plan["ops"] if 5098 <= o["op"] <= 5107)
        self.assertEqual(mma(whole), 4 * mma(split))

    def test_the_split_stays_in_the_measured_metadata_class(self):
        a = ir.Buffer("A", 1, elem=ir.F16); b = ir.Buffer("B", 2, elem=ir.F16); c = ir.Buffer("C", 3, elem=ir.F32)
        fn = ir.Function("grid_probe", [a, b, c]); bl = ir.Builder(fn, fn.block("entry"))
        bl.tensor_matmul(a, b, c, M=128, N=32, K=64, threadgroups=4)
        bl.ret(); ir.verify(fn)
        self.assertEqual(tuple(cc.compile_function(fn).abi()["system_registers"]), (130, 156))

    def test_unmeasured_splits_refuse(self):
        # (M=128, grid=4, sg=2) now lowers (item 3, test below); a combined split whose simdgroup
        # rows are not whole tiles still refuses
        for kwargs in (dict(M=48, grid=2), dict(M=96, grid=2), dict(M=96, grid=2, sg=2), dict(M=4096, grid=257)):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                tlower.lower(kwargs["M"], 32, 64, 64, 32, 32, grid=kwargs["grid"], sg=kwargs.get("sg", 1))


class CombinedSplit(unittest.TestCase):
    def test_grid_and_simdgroup_splits_compose(self):
        body, plan = tlower.lower(128, 32, 64, 64, 32, 32, grid=4, sg=2)
        self.assertEqual((plan["grid"], plan["simdgroups"]), (4, 2))
        # the grid-only body is unchanged by the composition (the threadgroup shift is 16*MT*sg)
        solo = tlower.lower(128, 32, 64, 64, 32, 32, grid=4)[0]
        self.assertEqual(hashlib.sha256(solo).hexdigest()[:16], GRID4_SOLO)


    def test_the_simdgroup_only_body_is_unchanged_by_the_composition(self):
        solo = tlower.lower(64, 32, 64, 64, 32, 32, sg=2)[0]
        self.assertEqual(hashlib.sha256(solo).hexdigest()[:16], SG2_SOLO)


class RowAndColumnGrid(unittest.TestCase):
    """MM 25.157: the row grid combines with the column grid, tg = col*grid + row (the rows in the low bits)."""

    def _emulate(self, tg, sg, gn, launched=None):
        import json, tempfile
        from pathlib import Path
        import numpy as np
        sys.path[:0] = [os.path.join(ROOT, "tools")]
        import g17emu as EMU
        import g17qmm as Q
        import g17tensorcommonruntime as R
        M, N, K = 128, 64, 64
        with tempfile.TemporaryDirectory() as t:
            g = Path(t) / "b"
            R.author_generic(g, Q.gemm_spec(M, N, K, sg, gn, 1, 1, False, tg))
            rng = np.random.default_rng(1)
            (g / "a.f16").write_bytes(np.asarray(rng.standard_normal((M, K)), np.float16).tobytes())
            (g / "b.f16").write_bytes(np.asarray(rng.standard_normal((K, N)), np.float16).tobytes())
            groups = tg * gn if launched is None else launched
            out, _ = EMU.run_bundle(g, groups * 32 * sg, 32 * sg, 1, tier="wp")
            exp = np.asarray(R.generic_reference(g, R.generic_spec(json.loads((g / "generic.json").read_text()))), "<f4")
            got = np.frombuffer(out, "<f4")[:exp.size].reshape(exp.shape)
            return int((got.view("<u4") != exp.view("<u4")).sum())

    def test_every_row_and_column_block_is_written_bit_exact(self):
        for tg, sg, gn in ((2, 2, 2), (4, 1, 2)):
            with self.subTest(tg=tg, sg=sg, gn=gn):
                self.assertEqual(self._emulate(tg, sg, gn), 0)

    def test_half_the_threadgroups_leave_blocks_unwritten(self):
        # the control: the check above can fail - launching half the grid leaves half of C at the sentinel
        self.assertGreater(self._emulate(2, 2, 2, launched=2), 0)

    def test_the_column_only_body_is_unchanged(self):
        self.assertEqual(hashlib.sha256(tlower.lower(128, 64, 64, 64, 64, 64, grid_n=2)[0]).hexdigest()[:16],
                         GN2_SOLO)


GN2_SOLO = "d3584a997535d02c"   # measured on main before the row and column grids combined

GRID4_SOLO = "7157eefc3cc7c692"   # measured on main before the composition change

SG2_SOLO = "7a7663a018feae36"   # measured on main before the composition change


if __name__ == "__main__":
    unittest.main()

