#!/usr/bin/env python3
"""The decode step's GPU stage programs (tools/g17decodeops.py, MM 25.136): RMSNorm, RoPE with the 1-row
KV append, SwiGLU and the transcendental probe. CPU only, nothing dispatches: the programs compile, their
bytes are the hashes the hardware receipts ran (the g17-decodeops-v1 receipts in evidence/g17-seta.zip; this suite reads none of them), they pass the fuzzer's
pre-dispatch checks, the host models of the exact midpoint test and exp2_soft hold, and the pipeline's
dry run puts the four stages on the programs."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "tools")]

import numpy as np

import g17decodeops as O
import g17decodestep as D

# the programs the MM 25.136 receipts dispatched (first 16 hex of sha256 of program.bin). A change here
# is a new program: it needs its own hardware receipt, not a re-pin
PINNED = {"probe": "cb58771e5ad90e20", "rmsnorm_half": "39622e76ec0dcb2e", "rmsnorm_float": "bd7eae13717c94d4",
          "rope_append": "8c5cabdb685fa2ff", "swiglu": "62a4c754ee34f51e"}
# each control's preregistered count (halves of buffer 3 in which the wrong input's image differs)
CONTROLS = {"attn_norm": {"x_rolled": 2047}, "ffn_norm": {"g1_for_g2": 2040},
            "rope_append": {"stale_length": 8192, "cos_sin_swapped": 4094},
            "rope_append_len37": {"stale_length": 8191}, "ffn_swiglu": {"gate_up_swapped": 8187}}


def u32(x):
    return np.asarray(x, np.float32).view(np.uint32).astype(np.int64)


class HostModels(unittest.TestCase):
    def test_midpoint_test_rounds_once_from_either_neighbour(self):
        rng = np.random.default_rng(136)
        x = np.exp(rng.uniform(np.log(1e-30), np.log(1e30), 100000)).astype(np.float32)
        p2 = np.float32(2.0) ** np.arange(-60, 60, dtype=np.float32)
        x = np.concatenate([x, p2, np.nextafter(p2, np.float32(0)), np.nextafter(p2, np.float32(np.inf))])
        g = np.exp(rng.uniform(0.0, np.log(2.0 ** 125), 100000)).astype(np.float32)
        for fn, below, v in ((D.rsqrt, D.mid_below_rsqrt, x), (D.recip, D.mid_below_recip, g)):
            rn = fn(v)
            for d in (-1, 0, 1):
                seed = (u32(rn) + d).astype(np.uint32).view(np.float32)
                self.assertTrue(np.array_equal(u32(D.round_once(below, v, seed)), u32(rn)), (fn.__name__, d))
            # and the corrected value is within half an ulp of the exact one (it IS the rounding)
            exact = (1.0 / np.sqrt(v.astype(np.float64))) if fn is D.rsqrt else (1.0 / v.astype(np.float64))
            ulp = np.spacing(rn).astype(np.float64)
            self.assertTrue(np.all(np.abs(rn.astype(np.float64) - exact) <= ulp / 2))

    def test_a_seed_two_ulps_off_is_not_repaired(self):
        # the precondition is a seed within one ulp (the probe measured it); two off is outside it
        x = np.float32([1.0325263, 7.5, 0.01])
        rn = D.rsqrt(x)
        seed = (u32(rn) + 2).astype(np.uint32).view(np.float32)
        self.assertFalse(np.array_equal(u32(D.round_once(D.mid_below_rsqrt, x, seed)), u32(rn)))

    def test_exp2_soft_is_within_one_ulp(self):
        t = np.random.default_rng(3).uniform(-125, 125, 200000).astype(np.float32)
        d = u32(D.exp2_soft(t)) - u32(D.exp2(t))
        self.assertLessEqual(int(np.max(np.abs(d))), 1)
        self.assertGreater(float(np.mean(d == 0)), 0.85)
        self.assertTrue(np.all(np.isfinite(D.exp2_soft(np.float32([-1e30, 1e30, 200, -200])))))

    def test_milestone_swiglu_matches_the_rounded_once_exp2(self):
        # the reference change moved no SwiGLU output half at the milestone (25.132's figures stand)
        _inp, env = O.milestone_inputs()
        import g17tensorcommonruntime as TCR
        t = D.fmul(env["gate32"], TCR.GELU_NEG_INV_LN2)
        old = D.fmul(env["gate32"], D.recip(D.fadd(D.exp2(t), np.float32(1.0))))
        self.assertTrue(np.array_equal(D.narrow(D.fmul(old, env["up32"])), env["act"]))

    def test_the_length_word_is_p9s(self):
        from agxforge.g17 import runtime
        self.assertEqual(O.KV_LENGTH_BYTE, runtime.KV_LENGTH_BYTE)
        self.assertEqual(O.rope_layout(16, 128, 272)["KC"], runtime.ATTENTION_C["KC"])


class Programs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.arms = O.arms()
        cls.programs = {a["program"]: a["build"]() for a in cls.arms if a["build"] is not None}

    def test_pinned_hashes(self):
        import g17tensorcommonruntime as TCR
        self.assertEqual({k: TCR.sha(p.code)[:16] for k, p in self.programs.items()}, PINNED)

    def test_pre_dispatch_checks_and_transport(self):
        from agxforge.g17 import tensorlife, tensorview as TV
        for arm in self.arms:
            lay = arm["layout"]
            self.assertLessEqual(lay["K"], 256, arm["name"])          # gemm_generic's class table
            self.assertLessEqual(lay["M"] // lay["groups"], 1024)
            self.assertEqual(lay["M"] % (16 * lay["groups"]), 0)
            a, b, c, _want = arm["io"]
            self.assertEqual((len(a), len(b), len(c)), (lay["a_bytes"], lay["b_bytes"], lay["c_bytes"]))
        for key, p in self.programs.items():
            code = bytes(p.code)
            view = TV.view(code)
            self.assertEqual(TV.hazards(view), [], key)
            self.assertEqual(tensorlife.aliased_store_releases(code), [], key)
            self.assertFalse(TV.loops(view), key)
            self.assertFalse(any(i.opcode == 458 for i in view), key)   # no back edge
            abi = p.abi_plain(p.abi())
            self.assertEqual(abi["system_registers"], [130, 156], key)
            self.assertEqual(abi["abi_version"], 5, key)

    def test_controls_differ_by_the_preregistered_counts(self):
        got = {a["name"]: {k: O.halves_differing(a["io"][3], v) for k, v in a["controls"].items()}
               for a in self.arms if a["controls"]}
        self.assertEqual(got, CONTROLS)

    def test_refusals_are_named(self):
        with self.assertRaises(ValueError):
            O.rope_layout(2, 64, 32)          # TINY's two heads: not four threadgroups' worth
        with self.assertRaises(ValueError):
            O.swiglu_layout(100)
        with self.assertRaises(ValueError):
            O.rmsnorm_layout(100, "half")
        with self.assertRaises(ValueError):
            O.swiglu_layout(8192, 512)         # 16 elements per group: less than one per lane

    def test_swiglu_groups_admit_the_transport(self):
        # MM 25.132.4: 8 (the pinned default) to 256 groups (one element per lane) all have a transport
        self.assertEqual(O.swiglu_layout(8192), O.swiglu_layout(8192, 8))
        for g in (8, 32, 64, 128, 256):
            lay = O.swiglu_layout(8192, g)
            self.assertLessEqual(lay["K"], 256, g)
            self.assertEqual(lay["M"] % (16 * g), 0, g)
            self.assertLessEqual(lay["M"] // g, 1024, g)
            self.assertGreaterEqual(lay["b_bytes"], 2 * O.CARRIER * O.CARRIER, g)


class PipelineDryRun(unittest.TestCase):
    def test_the_four_stages_run_on_the_programs(self):
        # the smallest layer every program admits (four heads of 64); TODAY is the same path, 4x slower
        import g17decodestep_gpu as G
        spec = D.LayerSpec(d_model=256, n_heads=4, head_dim=64, ffn_dim=256, kv_len=20, k_chunk=64)
        inputs = D.make_inputs(spec, 20260924)
        with tempfile.TemporaryDirectory(prefix="g17-decodeops-dry-") as td:
            ops = G.DecodeOps(Path(td), dry_run=True, log=lambda m: None)
            pipe = G.Pipeline(spec, G.DryRunDispatcher(Path(td)), log=lambda m: None, ops=ops)
            env = pipe.run(inputs)
        # the four stage programs, and the split-K fold after every split-K projection block (MM 25.132.2)
        self.assertEqual([c[0] for c in ops.checks if not c[0].endswith("split-K fold")],
                         ["attn_norm", "rope_append", "ffn_norm", "ffn_swiglu"])
        folds = [c[0] for c in ops.checks if c[0].endswith("split-K fold")]
        self.assertEqual(len(folds), sum(r["blocks"] for r in pipe.routes.values() if r["split_k"] > 1))
        self.assertTrue(all(r["gpu_folds"] == r["blocks"] for r in pipe.routes.values() if r["split_k"] > 1))
        self.assertTrue(all(c[3] for c in ops.checks))
        ref = D.reference(spec, inputs)["env"]
        for k in ("h1", "q16", "k_all", "v_all", "h2", "act", "out"):
            self.assertTrue(np.array_equal(u32(env[k]), u32(ref[k])), k)

    def test_a_refused_shape_falls_back_to_the_stub(self):
        import g17decodestep_gpu as G
        spec = D.TINY
        with tempfile.TemporaryDirectory(prefix="g17-decodeops-dry-") as td:
            ops = G.DecodeOps(Path(td), dry_run=True, log=lambda m: None)
            pipe = G.Pipeline(spec, G.DryRunDispatcher(Path(td)), log=lambda m: None, ops=ops)
            env = pipe.run(D.make_inputs(spec, 5))
        self.assertNotIn("rope_append", [c[0] for c in ops.checks])
        ref = D.reference(spec, D.make_inputs(spec, 5))["env"]
        self.assertTrue(np.array_equal(u32(env["out"]), u32(ref["out"])))


class Authoring(unittest.TestCase):
    def test_a_program_authors_a_bundle_through_the_generic_manifest(self):
        """Compile only, no device. The dry run returns references and never authors, so a manifest
        field the generic transport reads and generic_view lacks (grid_n and split_k, added by Set C's
        grid, 25.134) passed every dry-run test and raised KeyError on the first real dispatch."""
        spec = D.TINY
        I = D.make_inputs(spec, 20260924)
        key, lay, build, (a, b, c, _want) = O.rmsnorm_stage(spec, I["x"], I["g1"], "half")
        with tempfile.TemporaryDirectory(prefix="g17-decodeops-author-") as td:
            O.author(Path(td) / key, lay, build(), a, b, c, extra={"program": key})
            names = {p.name for p in (Path(td) / key).iterdir()}
        self.assertTrue({"manifest.json", "program.bin", "a.f16", "b.f16", "c.f32"} <= names, names)


class SplitKFold(unittest.TestCase):
    def test_the_fold_program_is_main_s_kernel_after_the_carrier(self):
        """Compile only. Set C's emit_split_k_fold (#197) wrapped as a decode op: the SR set stays the
        v5 tensor ABI's (130, 156), the fold's regions start past the carrier's tiles, and the hash is the
        one MM 25.132.1 preregistered before the fold was dispatched."""
        import hashlib
        lay = O.fold_layout(8, 16, 2048)
        self.assertGreaterEqual(lay["P"], 2 * 16 * 16 * lay["groups"])         # the carrier's A tile
        self.assertGreaterEqual(lay["OUT"], 4 * 16 * 16 * lay["groups"])       # the carrier's D tile
        prog = O.build_split_k_fold(lay)
        self.assertEqual(tuple(prog.abi()["system_registers"]), (130, 156))
        self.assertEqual(hashlib.sha256(prog.code).hexdigest()[:16], "f513bce322706942")

    def test_the_host_fold_is_the_pipeline_s_and_the_reverse_control_differs(self):
        import g17decodestep_gpu as G
        rng = np.random.default_rng(3)
        p = rng.standard_normal((8 * 16, 64)).astype(np.float32)
        self.assertTrue(np.array_equal(O.host_fold(p, 8, 16).view(np.uint32), G.fold_split_k(p, 8, 16).view(np.uint32)))
        self.assertFalse(np.array_equal(O.host_fold(p, 8, 16).view(np.uint32),
                                        O.host_fold(p, 8, 16, reverse=True).view(np.uint32)))

    def test_the_row_limited_fold_programs_are_the_preregistered_ones(self):
        """Set C's M_live (linker/g17-setc-foldrow): decode folds row 0 only. Same SR set; hashes as MM
        25.132.5 preregistered; rows past M_live keep buffer 3's input (the expected buffer says so)."""
        import hashlib
        for G, h in ((8, "bf4775f32777537c"), (4, "72f7faff1a95eb14")):
            lay = O.fold_layout(G, 16, 2048, 1)
            prog = O.build_split_k_fold(lay)
            self.assertEqual(tuple(prog.abi()["system_registers"]), (130, 156))
            self.assertEqual(hashlib.sha256(prog.code).hexdigest()[:16], h)
        rng = np.random.default_rng(5)
        p = rng.standard_normal((4 * 16, 64)).astype(np.float32)
        lay = O.fold_layout(4, 16, 64, 1)
        _a, _b, c, want = O.fold_io(lay, p)
        out = np.frombuffer(want, "<f4", 16 * 64, lay["OUT"]).reshape(16, 64)
        self.assertTrue(np.array_equal(out[0].view(np.uint32), O.host_fold(p, 4, 16)[0].view(np.uint32)))
        self.assertFalse(out[1:].any())


if __name__ == "__main__":
    unittest.main()
