#!/usr/bin/env python3
"""The decode-step reference, plan and host pipeline (tools/g17decodestep.py, MM 25.132). CPU only: nothing
here dispatches (the pipeline test uses the dry-run dispatcher, which returns the repository's own
reference for each bundle's files)."""
import dataclasses
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "tools")]

import numpy as np

import g17decodestep as D
import g17tensorcommonruntime as TCR


def bits(x):
    return np.asarray(x, np.float32).view(np.uint32)


class ReferenceShapes(unittest.TestCase):
    def test_tiny_and_milestone_shapes(self):
        for spec in (D.TINY, D.MILESTONE):
            env = D.reference(spec, D.make_inputs(spec, 3))["env"]
            d, H, hd, f, n = spec.d_model, spec.n_heads, spec.head_dim, spec.ffn_dim, spec.kv_len
            want = dict(h1=(d,), qkv32=(3 * d,), q16=(H, hd), k_new=(H, hd), v_new=(H, hd),
                        k_all=(H, n + 1, hd), v_all=(H, n + 1, hd), o32=(H, hd), attn=(d,), h=(d,), h2=(d,),
                        gate32=(f,), up32=(f,), act=(f,), out32=(d,), out=(d,))
            for k, shape in want.items():
                self.assertEqual(env[k].shape, shape, (spec, k))
                self.assertEqual(env[k].dtype, np.float32, k)
                self.assertTrue(np.all(np.isfinite(env[k])), k)
            # every 16-bit output holds half values exactly
            for k in ("h1", "q16", "attn", "h2", "act", "out"):
                self.assertTrue(np.array_equal(env[k], env[k].astype(np.float16).astype(np.float32)), k)
            # the new token is appended at row kv_len
            self.assertTrue(np.array_equal(env["k_all"][:, n], env["k_new"]))

    def test_spec_refusals(self):
        with self.assertRaises(ValueError):
            D.LayerSpec(d_model=2048, n_heads=16, head_dim=64)
        with self.assertRaises(ValueError):
            D.LayerSpec(k_chunk=100)
        with self.assertRaises(ValueError):
            D.LayerSpec(storage="float")


class Determinism(unittest.TestCase):
    def test_same_seed_same_bits_other_seed_differs(self):
        a = D.reference(D.TINY, D.make_inputs(D.TINY, 11))["env"]
        b = D.reference(D.TINY, D.make_inputs(D.TINY, 11))["env"]
        c = D.reference(D.TINY, D.make_inputs(D.TINY, 12))["env"]
        for k in ("h1", "qkv32", "o32", "out32", "out"):
            self.assertTrue(np.array_equal(bits(a[k]), bits(b[k])), k)
            self.assertFalse(np.array_equal(bits(a[k]), bits(c[k])), k)


class Chaining(unittest.TestCase):
    def test_each_stage_reproduces_from_its_recorded_inputs(self):
        spec = D.TINY
        inputs = D.make_inputs(spec, 5)
        trace = D.reference(spec, inputs)
        self.assertEqual(tuple(k for k in trace if k != "env"), D.STAGE_NAMES)
        produced = set(inputs)
        for name, _fn, ins, outs in D.STAGES:
            self.assertTrue(set(ins) <= produced, "%s reads an array nothing produced" % name)
            got = D.run_stage(spec, name, trace[name]["in"])
            for k in outs:
                self.assertTrue(np.array_equal(bits(got[k]), bits(trace[name]["out"][k])), (name, k))
            produced |= set(outs)
        # a stage fed a perturbed input changes (the chain is not reading a stale array)
        env = dict(trace["env"]); env["h2"] = env["h2"].copy(); env["h2"][0] += 1
        self.assertFalse(np.array_equal(bits(D.run_stage(spec, "ffn_gate_up", env)["gate32"]),
                                        bits(trace["ffn_gate_up"]["out"]["gate32"])))

    def test_reference_near_ideal_but_not_equal(self):
        spec = D.TINY
        inputs = D.make_inputs(spec, 5)
        env, ide = D.reference(spec, inputs)["env"], D.ideal(spec, inputs)
        for k in ("h1", "qkv32", "o32", "h", "out32"):
            err = D.error_report(env[k], ide[k])
            self.assertLess(err["max_rel_to_max"], 2e-3, k)
            self.assertGreater(err["max_abs"], 0.0, k)     # the reference is not the float64 answer


class Numerics(unittest.TestCase):
    def test_gemm_equals_gemm_mma(self):
        rng = np.random.default_rng(7)
        for M, N, K, trunc in ((16, 32, 64, False), (1, 48, 128, False), (3, 16, 32, True)):
            a = rng.standard_normal((M, K)).astype(np.float32 if trunc else np.float16).astype(np.float32)
            b = rng.standard_normal((K, N)).astype(np.float16).astype(np.float32)
            c = rng.standard_normal((M, N)).astype(np.float32)
            for cc in (None, c):
                got = D.gemm(a, b, cc, truncate_a=trunc)
                want = TCR._gemm_mma(a, b, cc, M, N, K, truncate_a=trunc)
                self.assertTrue(np.array_equal(bits(got), bits(want)), (M, N, K, trunc, cc is None))

    def test_gemm_is_not_exact_rounded_once(self):
        # the control: the section-136 order differs from the exact sum rounded once on this case
        rng = np.random.default_rng(8)
        a = rng.standard_normal((16, 128)).astype(np.float16).astype(np.float32)
        b = rng.standard_normal((128, 32)).astype(np.float16).astype(np.float32)
        exact = (a.astype(np.float64) @ b.astype(np.float64)).astype(np.float32)
        self.assertFalse(np.array_equal(bits(D.gemm(a, b)), bits(exact)))

    def test_k_chunk_is_partials_summed_in_order(self):
        rng = np.random.default_rng(9)
        x = rng.standard_normal(256).astype(np.float16).astype(np.float32)
        w = rng.standard_normal((256, 32)).astype(np.float16).astype(np.float32)
        c = rng.standard_normal(32).astype(np.float32)
        parts = [D.gemm(x[None, s:s + 64], w[s:s + 64])[0] for s in range(0, 256, 64)]
        acc = parts[0]
        for p in parts[1:]:
            acc = (acc + p).astype(np.float32)
        acc = (acc + c).astype(np.float32)
        self.assertTrue(np.array_equal(bits(D.gemv(x, w, c, k_chunk=64)), bits(acc)))
        self.assertTrue(np.array_equal(bits(D.gemv(x, w, c)), bits(D.gemm(x[None], w, c[None])[0])))

    def test_attention_value_slices_are_exact(self):
        spec = D.TINY
        env = D.reference(spec, D.make_inputs(spec, 4))["env"]
        full = D.attention_head(env["q16"][0], env["k_all"][0], env["v_all"][0], spec.kv_len)
        for v0 in range(0, spec.head_dim, 16):
            part = D.attention_head(env["q16"][0], env["k_all"][0], env["v_all"][0][:, v0:v0 + 16], spec.kv_len)
            self.assertTrue(np.array_equal(bits(part), bits(full[v0:v0 + 16])))

    def test_silu_is_the_gelu_step_without_its_alpha(self):
        x = np.linspace(-8, 8, 257, dtype=np.float32)
        # gelu_model(x / 1.702) is not exactly silu (the division rounds), so compare the formula instead
        want = x / (1.0 + np.exp(-x.astype(np.float64)))
        self.assertLess(np.max(np.abs(D.silu(x) - want)), 1e-5)


class SplitK(unittest.TestCase):
    """gemm_reference's split-K contract (Set C's kernel implements it)."""

    def operands(self, M=4, N=24, K=128, seed=21):
        rng = np.random.default_rng(seed)
        A = rng.standard_normal((M, K)).astype(np.float16).astype(np.float32)
        B = rng.standard_normal((K, N)).astype(np.float16).astype(np.float32)
        C = (rng.standard_normal((M, N)) * 4).astype(np.float32)
        return A, B, C

    def test_g1_is_the_single_chain_reference(self):
        A, B, C = self.operands()
        for cc in (None, C):
            got = D.gemm_reference(A, B, cc, split_k=1)
            self.assertTrue(np.array_equal(bits(got), bits(D.gemm(A, B, cc))))
            self.assertTrue(np.array_equal(bits(got), bits(TCR._gemm_mma(A, B, cc, 4, 24, 128))))

    def test_g2_g4_are_a_left_fold_of_slice_chains_then_c(self):
        A, B, C = self.operands()
        for G in (2, 4):
            ks = 128 // G
            parts = [TCR._gemm_mma(A[:, t * ks:(t + 1) * ks], B[t * ks:(t + 1) * ks], None, 4, 24, ks)
                     for t in range(G)]
            acc = parts[0]
            for p in parts[1:]:                              # ((p0 + p1) + p2) + p3, each RNE fp32
                acc = (acc + p).astype(np.float32)
            self.assertTrue(np.array_equal(bits(D.gemm_reference(A, B, None, split_k=G)), bits(acc)), G)
            out = (acc + C).astype(np.float32)               # C LAST
            self.assertTrue(np.array_equal(bits(D.gemm_reference(A, B, C, split_k=G)), bits(out)), G)
            if G == 4:
                # the controls: a right fold and a C-first fold are different values on these operands
                right = (parts[0] + (parts[1] + (parts[2] + parts[3]).astype(np.float32)).astype(np.float32)).astype(np.float32)
                cfirst = C.copy()
                for p in parts:
                    cfirst = (cfirst + p).astype(np.float32)
                self.assertFalse(np.array_equal(bits(acc), bits(right)))
                self.assertFalse(np.array_equal(bits(out), bits(cfirst)))

    def test_refusal_and_tolerance(self):
        A, B, C = self.operands()
        for G in (0, 3, 16):                                 # 128 % (16 * G) != 0, or G < 1
            with self.assertRaises(ValueError):
                D.gemm_reference(A, B, C, split_k=G)
        tol = D.split_k_tolerance(A, B, C, split_k=4)
        diff = np.abs(D.gemm_reference(A, B, C, 4).astype(np.float64) - D.gemm_reference(A, B, C, 1))
        self.assertEqual(tol, float(diff.max()))
        self.assertGreater(tol, 0.0)                       # a value change, not bit-identity
        self.assertLess(tol, 1e-4 * float(np.abs(D.gemm(A, B, C)).max()) + 1e-5)
        self.assertEqual(D.split_k_tolerance(A, B, C, split_k=1), 0.0)

    def test_projection_plan_routes_through_the_n_tiled_grid(self):
        # MM 25.132/25.134: the 2048-wide projections take split-K (G 8, K-chain-bound on 16 threadgroups),
        # the FFN gate/up does not (64 threadgroups reach its floor); every route is one launch per block
        want = {"qkv_proj": (16, 8, 3, 1), "o_proj": (16, 8, 1, 1), "ffn_gate_up": (64, 1, 2, 1),
                "ffn_down": (16, 4, 1, 1)}                 # MM 25.132.5: one launch at split_k 4
        for op in D.plan(D.MILESTONE):
            if op["kind"] == "gemm":
                sk = op["split_k"]
                self.assertEqual((sk["grid_n"], sk["G"], sk["blocks"], sk["launches"]), want[op["op"]], op["op"])
                self.assertEqual(sk["G"] * sk["slice_k"], op["shape"]["K"])
                self.assertTrue(op["covered_by"].startswith("N-tiled grid (25.134)"))
                self.assertEqual(sk["G"] > 1, any("split-K fold" in m for m in op["missing"]), op["op"])

    def test_projection_route_limits(self):
        self.assertEqual(D.projection_route(2048, 2048), (16, 8, 1))
        self.assertEqual(D.projection_route(8192, 2048), (64, 1, 1))
        self.assertEqual(D.projection_route(2048, 8192), (16, 4, 1))    # one dispatch, split_k 4 (Set C's one-shot)
        self.assertEqual(D.projection_route(512, 1024, k_chunk=256), (4, 4, 1))   # k_chunk fixes G
        self.assertEqual(D.projection_route(128, 128, k_chunk=64)[0], 2)           # never one threadgroup
        with self.assertRaises(ValueError):
            D.projection_route(6144, 2048)                                         # grid_n 48 is not a power of two: 3 blocks

    def test_host_fold_is_gemm_references_fold(self):
        import g17decodestep_gpu as G
        rng = np.random.default_rng(3)
        A = rng.standard_normal((16, 256)).astype(np.float16).astype(np.float32)
        B = rng.standard_normal((256, 32)).astype(np.float16).astype(np.float32)
        C = rng.standard_normal((16, 32)).astype(np.float32)
        for g in (1, 2, 4, 8):
            ks = 256 // g
            parts = np.concatenate([D.gemm(A[:, t * ks:(t + 1) * ks], B[t * ks:(t + 1) * ks]) for t in range(g)])
            self.assertTrue(np.array_equal(bits(G.fold_split_k(parts, g, 16, C)),
                                           bits(D.gemm_reference(A, B, C, split_k=g))), g)
        # the control: a descending fold is a different value on these operands
        parts = np.concatenate([D.gemm(A[:, t * 32:(t + 1) * 32], B[t * 32:(t + 1) * 32]) for t in range(8)])
        rev = G.fold_split_k(parts.reshape(8, 16, 32)[::-1].reshape(128, 32), 8, 16, C)
        self.assertFalse(np.array_equal(bits(rev), bits(D.gemm_reference(A, B, C, split_k=8))))


class Plan(unittest.TestCase):
    def test_milestone_plan(self):
        ops = D.plan(D.MILESTONE)
        self.assertEqual(len(ops), 8)
        self.assertEqual([o["op"] for o in ops], ["attn_norm", "qkv_proj", "rope_append", "attention", "o_proj",
                                                  "ffn_norm", "ffn_gate_up", "ffn_down"])
        stages = [s for o in ops for s in o["stages"]]
        self.assertEqual(tuple(stages), D.STAGE_NAMES)
        shape = {o["op"]: o["shape"] for o in ops}
        self.assertEqual(shape["qkv_proj"], dict(M=16, N=6144, K=2048))
        self.assertEqual(shape["o_proj"], dict(M=16, N=2048, K=2048))
        self.assertEqual(shape["ffn_gate_up"], dict(M=16, N=16384, K=2048))
        self.assertEqual(shape["ffn_down"], dict(M=16, N=2048, K=8192))
        self.assertEqual(shape["attention"]["keys"], 257)
        self.assertEqual(shape["attention"]["key_blocks"], 17)
        for o in ops:
            self.assertIn(o["status"], (D.AVAILABLE, D.NEEDS_CLASS, D.MISSING))
            self.assertIn(o["p11_stage"], (None, "attn_norm", "qkv_proj", "attention", "o_proj", "ffn_norm", "ffn"))
            # every op that is not available names what it lacks (an available op may name a cost)
            if o["status"] != D.AVAILABLE:
                self.assertTrue(o["missing"], o["op"])
        # MM 25.136: the norms and the RoPE append run as decodeops programs at the milestone shape;
        # item 1 (MM 25.135): the attention runs at the milestone shape in phase grid, one dispatch
        self.assertEqual({o["op"] for o in ops if o["status"] == D.AVAILABLE},
                         {"attn_norm", "rope_append", "ffn_norm", "attention"})
        self.assertFalse([o for o in ops if o["status"] == D.MISSING])
        for o in ops:
            if o["op"] in ("attn_norm", "rope_append", "ffn_norm"):
                self.assertFalse(o["missing"], o["op"])
        att = [o for o in ops if o["op"] == "attention"][0]
        self.assertTrue(att["today"].startswith("gpu: 1 x attention phase grid (16 heads"))
        # the floor: 134.2 MB of weights, 2.1 MB of KV, 0.506 ms at 270 GB/s
        tot = D.step_floor(D.MILESTONE)
        self.assertEqual(tot["weights"], 2 * (3 * 2048 * 2048 + 2048 * 2048 + 3 * 2048 * 8192 + 2 * 2048) + 4 * 128)
        # the attention reads K and V for 257 keys; rope_append writes the new token's K and V row
        self.assertEqual(tot["kv"], (2 * 257 + 2) * 16 * 128 * 2)
        self.assertAlmostEqual(tot["floor_ms"], 0.5055, places=3)
        self.assertAlmostEqual(sum(o["floor_us"] for o in ops) / 1e3, tot["floor_ms"], places=9)

    def test_buffer_regions_do_not_overlap(self):
        for buf, table in D.buffers(D.MILESTONE).items():
            spans = sorted((off, off + n) for off, n, _dt, _shape in table.values())
            for (a0, a1), (b0, _b1) in zip(spans, spans[1:]):
                self.assertLessEqual(a1, b0, buf)

    def test_today_attention_is_available(self):
        att = [o for o in D.plan(D.TODAY) if o["op"] == "attention"][0]
        self.assertEqual(att["status"], D.AVAILABLE)
        self.assertTrue(att["today"].startswith("gpu: 32 x attention"))


class Isolation(unittest.TestCase):
    def test_reference_imports_no_gpu_code(self):
        code = ("import sys; sys.path[:0] = [%r, %r]; import g17decodestep as D; "
                "D.reference(D.TINY, D.make_inputs(D.TINY)); "
                "bad = [m for m in sys.modules if m.split('.')[0] in ('torch', 'mlx', 'g17packeddispatch', "
                "'g17commonstage') or m in ('agxforge.g17.runtime', 'agxforge.g17.cc', 'agxforge.g17.tlower')]; "
                "print(bad); raise SystemExit(1 if bad else 0)") % (ROOT, os.path.join(ROOT, "tools"))
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


class DryRunPipeline(unittest.TestCase):
    def test_tiny_pipeline_matches_the_repository_references(self):
        import g17decodestep_gpu as G
        spec = D.TINY
        inputs = D.make_inputs(spec, 20260924)
        with tempfile.TemporaryDirectory(prefix="g17-decode-dry-") as td:
            pipe = G.Pipeline(spec, G.DryRunDispatcher(Path(td)), log=lambda m: None)
            env = pipe.run(inputs)
        self.assertTrue(all(ok for *_x, ok in pipe.checks), pipe.checks)
        self.assertEqual(pipe.dx.count, 15)                # 7 projection launches + 8 attention
        self.assertEqual({k: (r["grid_n"], r["split_k"], r["launches"]) for k, r in pipe.routes.items()},
                         {"qkv_proj": (2, 2, 3), "o_proj": (2, 2, 1), "ffn_gate_up": (2, 2, 2), "ffn_down": (2, 4, 1)})
        ref = D.reference(spec, inputs)["env"]
        for k in ("qkv32", "o32", "h", "out32", "out"):
            self.assertTrue(np.array_equal(bits(env[k]), bits(ref[k])), k)

    def test_pipeline_refuses_what_today_cannot_run(self):
        import g17decodestep_gpu as G
        dx = G.DryRunDispatcher.__new__(G.DryRunDispatcher)
        # bfloat storage is refused (no bfloat narrowing in tlower). The milestone builds: its attention is
        # phase grid (MM 25.135), and the projections need no k_chunk (the K loop replaced the host K tiling)
        with self.assertRaises(ValueError):
            G.Pipeline(dataclasses.replace(D.TODAY, storage="bfloat"), dx)
        G.Pipeline(D.MILESTONE, dx)
        G.Pipeline(dataclasses.replace(D.TODAY, k_chunk=0), dx)


class KRoute(unittest.TestCase):
    def test_k_route_gives_each_projection_its_route_split_k(self):
        """The whole-step reference the GPU pipeline is checked against at the milestone (MM 25.132.1):
        each projection at projection_route's G (8, 8, 1, 4), not the single chain and not one k_chunk."""
        spec = dataclasses.replace(D.MILESTONE, k_route=True)
        d, f = spec.d_model, spec.ffn_dim
        self.assertEqual([D.proj_chunk(spec, n, k) for n, k in ((d, d), (d, d), (f, d), (d, f))], [256, 256, 0, 2048])
        self.assertEqual(D.proj_chunk(D.MILESTONE, d, d), 0)          # no flag: the single chain
        small = D.LayerSpec(d_model=512, n_heads=4, head_dim=128, ffn_dim=512, kv_len=20)
        routed = dataclasses.replace(small, k_route=True)
        I = D.make_inputs(small, 7)
        h1 = D.stage_attn_norm(small, I["x"], I["g1"])["h1"]
        got = D.stage_qkv_proj(routed, h1, I["wqkv"])["qkv32"]
        G = D.projection_route(small.d_model, small.d_model)[1]
        self.assertEqual(G, 2)
        self.assertTrue(np.array_equal(bits(got), bits(D.gemv(h1, I["wqkv"], k_chunk=small.d_model // G))))
        # the control: at G 2 the value differs from the single chain somewhere
        self.assertFalse(np.array_equal(bits(got), bits(D.stage_qkv_proj(small, h1, I["wqkv"])["qkv32"])))


if __name__ == "__main__":
    unittest.main()
