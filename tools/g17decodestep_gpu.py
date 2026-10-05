#!/usr/bin/env python3
"""The decode step END TO END on the GPU with the classes that exist today, orchestrated from the host
(MM 25.132). A stopgap: each missing class is a host stub that a real class replaces piece by piece.

    attn_norm     GPU: g17decodeops' RMSNorm (MM 25.136), bit-exact; the host stub if the shape is refused
    qkv_proj      GPU: Set C's N-tiled grid (MM 25.134), one launch per 2048-wide block (Q, K, V): grid_n
                  threadgroups each own N/grid_n columns, the K loop over the whole K, and split_k
                  threadgroups partitioning K where the column grid alone leaves the GPU's cores idle. The
                  (split_k*M) x N partials are folded on the host: HOST STUB pending Set C's reduce kernel,
                  exactly gemm_reference(split_k=G) (fp32 left fold ascending t, C last)
    rope_append   GPU: g17decodeops' RoPE and 1-row append at the runtime length into a P9-layout cache
                  (MM 25.136); the attention reads K and V back from that cache. Host stub if refused
    attention     GPU. At head 128 (the milestone): the attention class's phase grid (MM 25.135), every
                  head in ONE dispatch (16 heads as 16 threadgroups), value 128, one decode row, up to 17
                  key blocks on the counted key-block loop (MM 25.135.4), falling back to straight-line
                  register key offsets where the loop is refused (a longer cache: a chain of such
                  dispatches, each resuming the state the previous left in buffer 3). At head 64: phase attend, one dispatch
                  per head and 16-wide V slice (8 key blocks). Q and the K/V cache written by the host;
                  causal q0 = kv_len. When the class refuses the shape, the HOST STUB (g17decodestep's
                  attention stage) runs instead, and the report says so
    o_proj        GPU as qkv_proj; the residual is the fold's C, added last on the host
    ffn_norm      GPU: RMSNorm over the fp32 row (MM 25.136); host stub if refused
    ffn_gate_up   GPU as qkv_proj, gate and up one launch each
    ffn_swiglu    GPU: g17decodeops' SwiGLU body (MM 25.136); host stub if refused
    ffn_down      GPU as qkv_proj; the residual (the fold's C) and the narrowing on the host

Every GEMM dispatch is compared bit for bit twice: its (split_k*M) x N readback against the single-chain
partial over each K slice, and the host fold of those partials against g17decodestep.gemm_reference(split_k=G)
on the dispatch's actual inputs; every attention dispatch by the class's own rule (bitwise, or inside
attention_bound: O lies downstream of exp2, which is within one ulp, not exact); the stages chain the GPU's outputs, and the end is compared with the
reference of the whole step. The projections take the K loop; the attention takes the counted key-block loop
at head 128 where the class admits it (MM 25.135.4; --attention-straight, or a refusal, keeps the straight-line
phase grid program); one program per shape, authored once and reused with new input files; one GPU dispatch per worker
process. At the milestone the whole step is compared with g17decodestep.reference at k_route (each projection
at its route's split_k), MM 25.132. `--projections` runs the four projection stages alone at any spec (the
milestone by default), each against gemm_reference at its route's split_k.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

import g17decodestep as D                 # noqa: E402
import g17tensorcommonruntime as TCR      # noqa: E402

F32 = np.float32
GEMM_M = 16                               # the decode row padded to one tile; a grid split has no C[0,0] += 1 tail
# the projection route (grid_n, split_k, launches) is the plan's (g17decodestep.projection_route)
projection_route = D.projection_route
MAX_LAUNCH_K = D.MAX_LAUNCH_K


KLOOP_UNROLL = 2                    # the projections' K-loop body (1 or 2); main's #198
SWIGLU_GROUPS = 256                 # the SwiGLU body's threadgroups (one element per lane at 8192; 25.132.4)
FOLD_LIVE_ROWS = 1                  # the GPU fold folds row 0 only (Set C's M_live): the token; rows 1..15 are padding


def projection_spec(N, K, grid_n, G, unroll=1):
    """The gemm_generic request for one projection block (25.134's n_tiled_grid / split_k_grid classes)."""
    # threadgroups counts the ROW groups (1: one 16-row tile); the launch is threadgroups x grid_n x split_k
    s = dict(M=GEMM_M, N=N, K=K, a="half", b="half", threadgroups=1, grid_n=grid_n)
    if G > 1:
        s["split_k"] = G
    if K > 256:
        s["kloop"] = True
    if unroll != 1:
        s["kloop_unroll"] = unroll
    return s


def fold_split_k(partials, G, M, C=None):
    """HOST STUB pending Set C's split-K reduce kernel (25.134: "the G partials are reduced OFF-GPU"). The
    (G*M) x N readback, block t the partial over K slice t, folded EXACTLY as
    g17decodestep.gemm_reference(split_k=G): an fp32 left fold in ascending t, each add RNE, C added last."""
    p = np.asarray(partials, F32).reshape(G, M, -1)
    acc = p[0].copy()
    with np.errstate(over="ignore", invalid="ignore"):
        for t in range(1, G):
            acc = (acc + p[t]).astype(F32)
        if C is not None:
            acc = (acc + np.asarray(C, F32).reshape(acc.shape)).astype(F32)
    return acc


class Dispatcher:
    """Author a program per shape once, then dispatch it with new input files, one worker process per
    dispatch under the machine-wide GPU lock, with the vendor GPU event counters checked around it."""

    def __init__(self, workdir: Path):
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.worker = self.workdir / "common-worker"
        if not self.worker.exists():
            TCR.build_worker(self.worker)
        self.bundles = {}
        self.count = 0
        self.gpu_seconds = 0.0

    def bundle(self, key, spec):
        if key in self.bundles:
            return self.bundles[key]
        path = self.workdir / key
        if not path.exists():
            TCR.author_generic(path, spec)
        from agxforge.g17 import runtime
        manifest = runtime.ImageContract.read(json.loads((path / "manifest.json").read_text()))
        sizes = {n: (path / n).stat().st_size for n in ("a.f16", "b.f16", "c.f32")}
        self._load(path)
        self.bundles[key] = (path, manifest, sizes)
        return self.bundles[key]

    def _load(self, path):
        import g17commonstage, g17packeddispatch
        with g17commonstage.lock_gpu():
            before = g17packeddispatch.gpu_events()
            load = subprocess.run([str(self.worker), str(path), "--tensor-load-approved"], capture_output=True,
                                  timeout=30)
            if load.returncode or json.loads(load.stdout) != {"status": 0, "load_only": True, "gpu_dispatched": False}:
                raise RuntimeError("tensor load-only failed: " + load.stderr.decode(errors="replace")[-2000:])
            if before != g17packeddispatch.gpu_events():
                raise RuntimeError("GPU diagnostics changed during tensor load-only")

    def dispatch(self, key, a, b, c):
        """Write a.f16 / b.f16 / c.f32 (sizes must match the authored bundle), dispatch once, return C."""
        import g17commonstage, g17packeddispatch
        path, manifest, sizes = self.bundles[key]
        for name, data in (("a.f16", a), ("b.f16", b), ("c.f32", c)):
            raw = np.ascontiguousarray(data).tobytes()
            if len(raw) != sizes[name]:
                raise ValueError("%s: %s is %d bytes, the program's is %d" % (key, name, len(raw), sizes[name]))
            (path / name).write_bytes(raw)
        with g17commonstage.lock_gpu():
            before = g17packeddispatch.gpu_events()
            run = subprocess.run([str(self.worker), str(path), "tensor-inputs", "1", "--tensor-dispatch-approved"],
                                 capture_output=True, timeout=30)
            if run.returncode:
                raise RuntimeError("tensor dispatch failed: " + run.stderr.decode(errors="replace")[-2000:])
            if before != g17packeddispatch.gpu_events():
                raise RuntimeError("GPU diagnostics changed during tensor dispatch")
        frames = TCR._frames(run.stdout)
        if len(frames) != 2 or frames[0][0].get("sequence") != 0:
            raise ValueError("tensor worker handshake or frame count differs from the contract")
        header, payload = frames[1]
        if (header.get("sequence"), header.get("status"), header.get("gpu_dispatched"),
                header.get("boundary_guard"), header.get("readonly_inputs")) != (1, 0, True, True, True):
            raise ValueError("tensor worker response is not fully checked")
        self.count += 1
        self.gpu_seconds += float(header.get("gpu_seconds") or 0.0)
        return np.frombuffer(payload, dtype="<f4").copy().reshape(int(manifest.shape.rows), int(manifest.shape.columns))


class DryRunDispatcher(Dispatcher):
    """No GPU: each "dispatch" returns the REPOSITORY's own reference for the bundle's files
    (g17tensorcommonruntime.generic_reference / attention_reference), so a dry run checks the
    orchestration and cross-checks g17decodestep against those references on every dispatch."""

    def __init__(self, workdir: Path):
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.bundles, self.count, self.gpu_seconds = {}, 0, 0.0

    def _load(self, path):
        pass

    def dispatch(self, key, a, b, c):
        path, manifest, sizes = self.bundles[key]
        for name, data in (("a.f16", a), ("b.f16", b), ("c.f32", c)):
            raw = np.ascontiguousarray(data).tobytes()
            if len(raw) != sizes[name]:
                raise ValueError("%s: %s is %d bytes, the program's is %d" % (key, name, len(raw), sizes[name]))
            (path / name).write_bytes(raw)
        s = TCR.generic_spec(json.loads((path / "generic.json").read_text()))
        self.count += 1
        out = TCR.attention_reference(path, s) if s.get("attention") else TCR.generic_reference(path, s)
        return np.asarray(out, "<f4").reshape(int(manifest.shape.rows), int(manifest.shape.columns))


class DecodeOps:
    """The MM 25.136 stage programs (tools/g17decodeops.py): one program per stage and shape, authored
    once, dispatched with the stage's actual inputs, and checked word for word against the reference's
    expected buffer 3. `dry_run` returns the expected buffer instead of dispatching. A stage whose shape
    the programs refuse returns None, and the pipeline runs the host stub."""

    def __init__(self, workdir: Path, dry_run=False, log=print):
        import g17decodeops as O
        self.O, self.workdir, self.dry_run, self.log = O, Path(workdir) / "decodeops", dry_run, log
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.worker = self.workdir / "common-worker"
        if not dry_run and not self.worker.exists():
            TCR.build_worker(self.worker)
        self.authored, self.count = set(), 0
        self.checks = []                   # (stage, what, dispatches, passed)

    def _run(self, stage, made):
        key, lay, build, (a, b, c, want) = made
        path = self.workdir / key
        if self.dry_run:
            got = want
        else:
            if key not in self.authored and not path.exists():
                self.O.author(path, lay, build(), a, b, c, extra={"program": key})
            self.authored.add(key)
            got = self.O.dispatch(path, self.worker, queries=1, inputs=(a, b, c))[0]
        self.count += 1
        ok = got == want
        if not ok:
            self.log("MISMATCH %s: %d words" % (stage, self.O.compare_words(got, want)))
        self.checks.append((stage, "decodeops program vs reference (every word of buffer 3)", 1, ok))
        return lay, got

    def _try(self, stage, fn, *args):
        try:
            made = fn(*args)
        except ValueError as e:
            self.log("%s: host stub (%s)" % (stage, e))
            return None
        return self._run(stage, made)

    def rmsnorm(self, stage, spec, v, g, in_dtype):
        r = self._try(stage, self.O.rmsnorm_stage, spec, v, g, in_dtype)
        return None if r is None else self.O.rmsnorm_out(*r)

    def rope_append(self, spec, qkv32, rope_cos, rope_sin, k_cache, v_cache, cache="p9"):
        r = self._try("rope_append", self.O.rope_stage, spec, qkv32, rope_cos, rope_sin, k_cache, v_cache, cache)
        return None if r is None else self.O.rope_out(r[0], r[1], spec.kv_len)

    def fold(self, stage, partials, G, M, M_live=None):
        """Set C's split-K fold on the GPU (the partials of one projection block, (G*M) x N), checked word
        for word against fold_split_k without C; only rows [0, M_live) are folded (decode: row 0, the
        token). None (the host fold) if the layout refuses."""
        r = self._try(stage + " split-K fold", self.O.fold_stage, partials, G, M, M_live)
        return None if r is None else self.O.fold_out(*r)

    def swiglu(self, spec, gate32, up32):
        # SWIGLU_GROUPS 256 (Piece B, MM 25.132.4: 40.8 us at 8 groups, 4.4 us at 256, bit-exact); a width
        # it does not divide falls back to the pinned 8-group program
        groups = SWIGLU_GROUPS if spec.ffn_dim % (32 * SWIGLU_GROUPS) == 0 else 8
        r = self._try("ffn_swiglu", self.O.swiglu_stage, spec, gate32, up32, groups)
        return None if r is None else self.O.swiglu_out(*r)


def _bits_equal(x, y):
    return bool(np.array_equal(np.asarray(x, F32).view(np.uint32), np.asarray(y, F32).view(np.uint32)))


class Pipeline:
    def __init__(self, spec: D.LayerSpec, dispatcher: Dispatcher, log=print, ops=None, attention_loop=True,
                 kv_split=None, merge_tiles=True, runtime_length=False, value_change_report=True):
        if spec.storage != "half":
            raise ValueError("the GPU pipeline is half storage (no bfloat narrowing in tlower)")
        self.spec, self.dx, self.log = spec, dispatcher, log
        self.ops = ops                     # DecodeOps, or None for the host stubs
        # phase grid on the counted key-block loop (MM 25.135.4) where the class admits it; False (or a
        # refusal) keeps the straight-line phase grid program (MM 25.135)
        self.attention_loop = attention_loop
        # THE KV SPLIT (MM 25.114.6 split, 25.135.5 merge): S threadgroups per head on the loop, then the merge
        # dispatch. It CHANGES THE VALUES (a different association), so it runs only when asked (kv_split=S)
        self.kv_split = kv_split
        self.merge_tiles = merge_tiles     # the split's merge per (head, O tile) (MM 25.132.7); False: per head
        self.runtime_length = runtime_length   # the split reads the query position at run time (MM 25.138.1)
        # the split's value change against the unsplit attention_head: a report (scalar host arithmetic per head,
        # seconds a layer), not a check; the model runner turns it off
        self.value_change_report = value_change_report
        self.attn_programs = []            # per grid dispatch: "loop" or the straight-line key_offsets
        self.routes = {}                   # stage -> [(block N, K, grid_n, split_k)]
        self.checks = []                   # (stage, what, dispatches, passed)
        self.attn_words = self.attn_bitwise = 0
        self.attn_max_abs = self.attn_max_bound_frac = 0.0
        self.attn_route = None                  # "grid", "attend" or "host stub" (and why)
        self.attn_value_change = None           # the KV split's O against the unsplit attention_head (row-max fraction)

    # ---- GEMM: Set C's N-tiled grid (+ split-K) per column block; each dispatch checked on its own
    def gemv(self, stage, x, w, c=None, blocks=1):
        return project(self, stage, x, w, c, blocks)

    # ---- attention: the class's phase grid (head 128), phase attend (head 64), or the host stub
    def attention(self, q16, k_all, v_all):
        from agxforge.g17 import runtime as RT
        spec = self.spec
        if spec.head_dim == 128:
            try:
                return self.attention_grid(q16, k_all, v_all)
            except RT.AttentionRefused as why:
                return self._attention_host(why)
        if spec.head_dim != 64 or spec.key_blocks > RT.ATTENTION_MAX_BLOCKS:
            return self._attention_host("head %d, %d key blocks: neither phase grid (head 128) nor phase attend "
                                        "(head 64, %d blocks)" % (spec.head_dim, spec.key_blocks, RT.ATTENTION_MAX_BLOCKS))
        self.attn_route = "attend"
        return self.attention_attend(q16, k_all, v_all)

    def _attention_host(self, why):
        """THE HOST STUB, when the class refuses: g17decodestep's attention stage on the host."""
        self.attn_route = "host stub (%s)" % (why,)
        self.log("attention: HOST STUB - the class refused: %s" % (why,))
        self.checks.append(("attention", "host stub (the class refused the shape)", 0, True))
        return None

    def attention_grid(self, q16, k_all, v_all):
        """Phase grid (MM 25.135): all heads in one dispatch per chunk of at most ATTENTION_GRID_CAPACITY key
        blocks; a chunk after the first resumes the previous chunk's GPU output. Each dispatch is checked
        against the class's reference and bound on its own files; the final O against g17decodestep."""
        from agxforge.g17 import runtime as RT
        spec = self.spec
        H, nb = spec.n_heads, spec.key_blocks
        cap = RT.ATTENTION_GRID_CAPACITY
        chunks = [(j0, min(cap, nb - j0)) for j0 in range(0, nb, cap)]
        reqs = [dict(phase="grid", heads=H, rows=1, blocks=n, q0=spec.kv_len, first_block=j0,
                     key_offsets="register" if n > RT.ATTENTION_MAX_BLOCKS else "immediate",
                     resume=j0 > 0, normalize=(j0, n) == chunks[-1]) for j0, n in chunks]
        for r in reqs:
            RT.attention_spec(r)                                             # refuses by name, before any work
        if self.attention_loop:
            # THE KEY-BLOCK LOOP (MM 25.135.4): a chunk that starts its own softmax runs as the loop program
            # when the class admits it; a refusal (a resumed chunk, or a mask on every block) keeps the
            # straight-line program already admitted above - the fallback
            for i, r in enumerate(reqs):
                try:
                    RT.attention_spec(dict(r, key_offsets="loop"))
                except RT.AttentionRefused as why:
                    self.log("attention: chunk %d stays straight-line: %s" % (i, why))
                    continue
                reqs[i] = dict(r, key_offsets="loop")
        self.attn_programs = [r["key_offsets"] for r in reqs]
        kp = np.zeros((H, nb * D.KEY_BLOCK, spec.head_dim), np.float16); kp[:, :spec.n_keys] = k_all
        vp = np.zeros((H, nb * D.KEY_BLOCK, spec.head_dim), np.float16); vp[:, :spec.n_keys] = v_all
        q = np.asarray(q16, np.float16).reshape(H, 1, spec.head_dim)
        if self.kv_split:
            # with a runtime length the split program covers the capacity, whatever this length's own route is
            if not self.runtime_length and (len(reqs) != 1 or reqs[0]["key_offsets"] != "loop"):
                raise RT.AttentionRefused("attention_kv_split", "the split runs on one key-block loop chunk")
            if self.runtime_length:
                # THE RUNTIME LENGTH (MM 25.138.1): one program over the layout's whole capacity serves every
                # length; keys past the query position are masked by the length word, the cache past it is zero
                cap = RT.ATTENTION_GRID_CAPACITY
                kp = np.zeros((H, cap * D.KEY_BLOCK, spec.head_dim), np.float16); kp[:, :spec.n_keys] = k_all
                vp = np.zeros((H, cap * D.KEY_BLOCK, spec.head_dim), np.float16); vp[:, :spec.n_keys] = v_all
            return self.attention_split(q16, k_all, v_all, q, kp, vp)
        prev, n, exact = None, 0, True
        for (j0, nblk), req in zip(chunks, reqs):
            key = "grid_h%d_b%d_f%d_q%d%s%s" % (H, nblk, j0, spec.kv_len, "_n" if req["normalize"] else "",
                                                "_loop" if req["key_offsets"] == "loop" else "")
            aspec = {"attention": req}
            if req["resume"]:
                seed_c = self.dx.workdir / (key + ".author-c.f32")          # authoring needs a buffer 3; the
                seed_c.write_bytes(np.full(RT.attention_layout(RT.attention_spec(req))["M"] * 256,  # dispatch
                                           TCR.ATTENTION_SENTINEL, "<f4").tobytes())               # writes its own
                aspec["c_from"] = str(seed_c)
            path, _m, _s = self.dx.bundle(key, aspec)
            s = TCR.generic_spec(json.loads((path / "generic.json").read_text()))
            sl = slice(16 * j0, 16 * (j0 + nblk))
            a, b, c = TCR._grid_buffers(s["attention"], q, kp[:, sl], vp[:, sl], prev)
            got = self.dx.dispatch(key, a, b, c).ravel()
            ref = TCR.attention_reference(path, s).ravel()
            bound = TCR.attention_bound(path, s).ravel()
            same = got.view(np.uint32) == ref.view(np.uint32)
            ok = bool(np.all(same | (np.abs(got.astype(np.float64) - ref.astype(np.float64)) <= bound)))
            exact &= ok
            n += 1
            if not ok:
                self.log("OUT OF BOUND attention grid chunk %d: %d words" % (j0, int(np.sum(~same))))
            prev = got
        at = s["attention"]
        o32 = np.stack([TCR.grid_o_rows(prev, at, h)[0] for h in range(H)]).astype(F32)
        # the end: every head's O against g17decodestep's attention_head, bitwise or inside the enclosure of
        # the whole key range (one trace from scratch: the chain's arithmetic)
        iv = TCR._StreamInterval("hardware")
        for h in range(H):
            want = D.attention_head(q16[h], k_all[h], v_all[h], spec.kv_len)
            qp = np.zeros((32, spec.head_dim), F32); qp[0] = q16[h]
            tr = TCR.grid_head_trace(iv, qp, kp[h].astype(F32), vp[h].astype(F32), rows=1, causal=True,
                                     q0=spec.kv_len, first_block=0)
            lo = np.array([x[0] for x in tr["O"]["att"][0]]); hi = np.array([x[1] for x in tr["O"]["att"][0]])
            bound = np.maximum(want - lo, hi - want)
            got = o32[h]
            same = got.view(np.uint32) == want.view(np.uint32)
            dist = np.abs(got.astype(np.float64) - want.astype(np.float64))
            ok = bool(np.all(same | (dist <= bound)))
            exact &= ok
            self.attn_words += got.size
            self.attn_bitwise += int(same.sum())
            self.attn_max_abs = max(self.attn_max_abs, float(dist.max()))
            self.attn_max_bound_frac = max(self.attn_max_bound_frac,
                                           float(np.max(np.where(bound > 0, dist / np.where(bound > 0, bound, 1), 0))))
        self.attn_route = "grid (%d dispatch%s: %s)" % (n, "" if n == 1 else "es", ", ".join(
            "key-block loop" if k == "loop" else "straight-line %s" % k for k in self.attn_programs))
        self.checks.append(("attention", "phase grid dispatches and the final O vs g17decodestep: bitwise or "
                            "within the class's bound", n, exact))
        return o32

    def _check_dispatch(self, path, s, got):
        """The class's pass rule on one attention dispatch's own files: bitwise or inside attention_bound."""
        ref = np.asarray(TCR.attention_reference(path, s), "<f4").ravel()
        bound = np.asarray(TCR.attention_bound(path, s)).ravel()
        same = got.view(np.uint32) == ref.view(np.uint32)
        with np.errstate(invalid="ignore"):
            inside = same | (np.abs(got.astype(np.float64) - ref.astype(np.float64)) <= bound)
        return bool(np.all(inside)), int(np.sum(~same)), int(np.sum(~inside))

    def attention_split(self, q16, k_all, v_all, q, kp, vp):
        """THE KV SPLIT then THE MERGE, two dispatches sharing buffer 3: phase grid kv_split S on the counted
        loop (heads x S threadgroups, each slice's un-normalised partial in its slot), then phase grid_merge (one
        threadgroup per head) over the split's OUTPUT, which normalises into each head's grid O. Each dispatch is
        checked by the class's rule on its own files; the end against g17decodestep.attention_head_split (the
        split-aware reference) inside its bound, and its value change against attention_head reported."""
        from agxforge.g17 import runtime as RT
        spec, H, S = self.spec, self.spec.n_heads, self.kv_split
        sreq = dict(phase="grid", heads=H, rows=1, blocks=spec.key_blocks, q0=spec.kv_len, key_offsets="loop",
                    kv_split=S)
        if self.runtime_length:
            sreq = dict(sreq, blocks=RT.ATTENTION_GRID_CAPACITY, runtime_q0=True)
        mreq = dict(phase="grid_merge", heads=H, rows=1, kv_split=S, allow_value_change=True)
        RT.attention_spec(sreq), RT.attention_spec(mreq)                    # refuse by name, before any work
        if self.merge_tiles:
            # THE TILED MERGE (MM 25.132.7): one threadgroup per (head, O tile), the per-head merge's words;
            # a refusal keeps the per-head merge admitted above
            try:
                RT.attention_spec(dict(mreq, tile_groups=8))
                mreq = dict(mreq, tile_groups=8)
            except RT.AttentionRefused as why:
                self.log("attention merge stays per head: %s" % why)
        skey = ("grid_h%d_b%d_split%d_rt" % (H, RT.ATTENTION_GRID_CAPACITY, S) if self.runtime_length else
                "grid_h%d_b%d_q%d_split%d" % (H, spec.key_blocks, spec.kv_len, S))
        spath, _m, _s = self.dx.bundle(skey, {"attention": sreq})
        ss = TCR.generic_spec(json.loads((spath / "generic.json").read_text()))
        a, b, c = TCR._grid_buffers(ss["attention"], q, kp, vp)
        if self.runtime_length:
            c.view("<u4")[RT.ATTENTION_GRID_LENGTH_BYTE // 4] = spec.kv_len          # the query's position
        got = self.dx.dispatch(skey, a, b, c).ravel()
        ok_s, diff_s, out_s = self._check_dispatch(spath, ss, got)
        if not ok_s:
            self.log("OUT OF BOUND attention split: %d words outside" % out_s)
        # the merge's authoring needs partials to seed its bundle; the dispatch reads the split's buffer 3
        mkey = "grid_merge_h%d_s%d%s" % (H, S, "_t8" if mreq.get("tile_groups") else "")
        seed = self.dx.workdir / (mkey + ".author-inputs.npz")
        if not seed.exists():
            np.savez(seed, q=np.zeros((H, 1, 128), np.float16), k0=np.zeros((H, 16, 128), np.float16),
                     O=np.zeros((H, S, 1, 128), F32), M=np.zeros((H, S, 1), F32), L=np.ones((H, S, 1), F32))
        mpath, _m, msizes = self.dx.bundle(mkey, {"attention": mreq, "grid_inputs": str(seed)})
        ms = TCR.generic_spec(json.loads((mpath / "generic.json").read_text()))
        mb = np.zeros(msizes["b.f16"] // 2, np.float16)
        gotm = self.dx.dispatch(mkey, a, mb, got).ravel()
        ok_m, diff_m, out_m = self._check_dispatch(mpath, ms, gotm)
        if not ok_m:
            self.log("OUT OF BOUND attention merge: %d words outside" % out_m)
        at = ms["attention"]
        o32 = np.stack([TCR.grid_merge_rows(gotm, at, h)[0][0] for h in range(H)]).astype(F32)
        vc = 0.0
        for h in (range(H) if self.value_change_report else ()):
            want = D.attention_head(q16[h], k_all[h], v_all[h], spec.kv_len)
            dist = np.abs(o32[h].astype(np.float64) - want.astype(np.float64))
            vc = max(vc, float(dist.max() / max(np.abs(want).max(), 1e-30)))
            self.attn_words += o32[h].size
            self.attn_bitwise += int(np.sum(o32[h].view(np.uint32) == want.view(np.uint32)))
            self.attn_max_abs = max(self.attn_max_abs, float(dist.max()))
        self.attn_value_change = vc if self.value_change_report else None
        self.attn_programs = ["split%d" % S, "merge_t8" if mreq.get("tile_groups") else "merge"]
        self.attn_route = "grid KV split S=%d (2 dispatches: split on the key-block loop, merge)" % S
        self.checks.append(("attention", "KV split S=%d and merge dispatches, each by the class's rule on its own "
                            "files (split %d words not bitwise, merge %d)" % (S, diff_s, diff_m), 2, ok_s and ok_m))
        self.log("attention split S=%d: split %s, merge %s, value change vs unsplit %.2e of the row max"
                 % (S, ok_s, ok_m, vc))
        return o32

    # ---- phase attend: one dispatch per head and 16-wide V slice (head 64)
    def attention_attend(self, q16, k_all, v_all):
        from agxforge.g17 import runtime as RT
        spec = self.spec
        nb = spec.key_blocks
        at = dict(phase="attend", cache_blocks=nb, new_blocks=0, rows=1, causal=True, q0=spec.kv_len)
        key = "attend_b%d_q%d" % (nb, spec.kv_len)
        path, _m, _s = self.dx.bundle(key, {"attention": at})
        s = TCR.generic_spec(json.loads((path / "generic.json").read_text()))
        lay = RT.attention_layout(s["attention"])
        cap = nb * D.KEY_BLOCK
        o32 = np.zeros((spec.n_heads, spec.head_dim), F32)
        exact, n, cross = True, 0, None
        o_word = RT.ATTENTION_C["O"] // 4
        for h in range(spec.n_heads):
            a = np.zeros(lay["M"] * lay["K"], np.float16)
            a[:spec.head_dim] = q16[h]                                        # Q row 0; rows 1..31 zero
            b = np.zeros(lay["K"] * lay["N"], np.float16)                     # Wk | Wv: unread by attend
            kp = np.zeros((cap, spec.head_dim), np.float16); kp[:spec.n_keys] = k_all[h]
            for v0 in range(0, spec.head_dim, RT.ATTENTION_VALUE):
                vp = np.zeros((cap, RT.ATTENTION_VALUE), np.float16)
                vp[:spec.n_keys] = v_all[h][:, v0:v0 + RT.ATTENTION_VALUE]
                c = np.full(lay["M"] * lay["N"], TCR.ATTENTION_SENTINEL, "<f4")
                raw = c.view(np.uint8)
                raw[lay["KC"]:lay["KC"] + kp.nbytes] = np.frombuffer(kp.tobytes(), np.uint8)
                raw[lay["VC"]:lay["VC"] + vp.nbytes] = np.frombuffer(vp.tobytes(), np.uint8)
                got = self.dx.dispatch(key, a, b, c).ravel()[o_word:o_word + RT.ATTENTION_VALUE]
                want = D.attention_head(q16[h], k_all[h], v_all[h][:, v0:v0 + RT.ATTENTION_VALUE], spec.kv_len)
                # THE CLASS'S PASS RULE (MM 25.129, 25.131): bitwise, or within attention_bound - the
                # enclosure over exp2 (measured within one ulp) and recip (assumed within one ulp), zero
                # wherever no transcendental reaches. O is downstream of exp2, so it is bounded, not exact.
                bound = TCR.attention_bound(path, s).ravel()[o_word:o_word + RT.ATTENTION_VALUE]
                same = got.view(np.uint32) == want.view(np.uint32)
                dist = np.abs(got.astype(np.float64) - want.astype(np.float64))
                ok = bool(np.all(same | (dist <= bound)))
                self.attn_words += got.size
                self.attn_bitwise += int(same.sum())
                self.attn_max_abs = max(self.attn_max_abs, float(dist.max()))
                self.attn_max_bound_frac = max(self.attn_max_bound_frac,
                                               float(np.max(np.where(bound > 0, dist / np.where(bound > 0, bound, 1), 0))))
                if cross is None:
                    # the repository's own attention reference on this bundle's files agrees with ours
                    rep = TCR.attention_reference(path, s).ravel()[o_word:o_word + RT.ATTENTION_VALUE]
                    cross = _bits_equal(rep, want)
                exact &= ok
                n += 1
                if not ok:
                    self.log("OUT OF BOUND attention head %d v0 %d: %d elements" % (h, v0, int(np.sum(~same & (dist > bound)))))
                o32[h, v0:v0 + RT.ATTENTION_VALUE] = got
        self.checks.append(("attention", "head x V-slice dispatches: bitwise or within attention_bound", n, exact))
        self.checks.append(("attention", "g17tensorcommonruntime.attention_reference == g17decodestep", 1, bool(cross)))
        return o32

    def run(self, inputs):
        spec, I = self.spec, inputs
        env = dict(I)
        t0 = time.time()
        h1 = self.ops.rmsnorm("attn_norm", spec, I["x"], I["g1"], "half") if self.ops else None
        env.update(D.stage_attn_norm(spec, I["x"], I["g1"]) if h1 is None else dict(h1=h1))   # stub fallback
        env["qkv32"] = self.gemv("qkv_proj", env["h1"], I["wqkv"], blocks=spec.qkv_width // spec.d_model
                                 if spec.qkv_width % spec.d_model == 0 else 1)
        self.log("qkv_proj done (%d dispatches, %.0f s)" % (self.dx.count, time.time() - t0))
        # with a runtime length, the append writes the attention grid's own cache layout (MM 25.138.2)
        ra = self.ops.rope_append(spec, env["qkv32"], I["rope_cos"], I["rope_sin"], I["k_cache"], I["v_cache"],
                                  "grid" if self.runtime_length else "p9") if self.ops else None
        env.update(D.stage_rope_append(spec, env["qkv32"], I["rope_cos"], I["rope_sin"],
                                       I["k_cache"], I["v_cache"]) if ra is None else ra)  # stub fallback
        o32 = self.attention(env["q16"], env["k_all"], env["v_all"])
        if o32 is None:                                                                  # HOST STUB (refused)
            o32 = D.stage_attention(spec, env["q16"], env["k_all"], env["v_all"])["o32"]
        env["o32"] = o32
        env["attn"] = D.narrow(env["o32"]).reshape(-1)                                   # host narrowing
        self.log("attention done (%d dispatches, %.0f s)" % (self.dx.count, time.time() - t0))
        env["h"] = self.gemv("o_proj", env["attn"], I["wo"], I["x"])
        h2 = self.ops.rmsnorm("ffn_norm", spec, env["h"], I["g2"], "float") if self.ops else None
        env.update(D.stage_ffn_norm(spec, env["h"], I["g2"]) if h2 is None else dict(h2=h2))    # stub fallback
        g = self.gemv("ffn_gate_up", env["h2"], np.concatenate([I["wgate"], I["wup"]], axis=1), blocks=2)
        env["gate32"], env["up32"] = g[:spec.ffn_dim], g[spec.ffn_dim:]
        act = self.ops.swiglu(spec, env["gate32"], env["up32"]) if self.ops else None
        env.update(D.stage_ffn_swiglu(spec, env["gate32"], env["up32"]) if act is None else dict(act=act))
        self.log("ffn_gate_up done (%d dispatches, %.0f s)" % (self.dx.count, time.time() - t0))
        env["out32"] = self.gemv("ffn_down", env["act"], I["wdown"], env["h"])
        env["out"] = D.narrow(env["out32"])
        self.log("ffn_down done (%d dispatches, %.0f s)" % (self.dx.count, time.time() - t0))
        return env


def project(owner, stage, x, w, c=None, blocks=1):
    """One projection, x (K,) @ w (K x N) (+ c), as `blocks` equal column blocks, each ONE launch of Set
    C's N-tiled grid (+ split-K) (MM 25.134) at projection_route's (grid_n, split_k), with the split-K
    partials folded on the host (fold_split_k, the stub for Set C's reduce kernel). `owner` carries
    .spec, .dx (a Dispatcher), .log, .checks and .routes. Each dispatch is checked bit for bit: its
    partials against the single-chain gemm over each K slice, and the fold against
    g17decodestep.gemm_reference(split_k=G) - the residual c is the fold's C, added last."""
    K, N = w.shape
    if N % blocks:
        raise ValueError("%s: N %d does not split into %d blocks" % (stage, N, blocks))
    nb = N // blocks
    grid_n, G, L = projection_route(nb, K, owner.spec.k_chunk)
    kl, gl = K // L, G // L                                                   # per launch: K range, split_k
    # KLOOP_UNROLL 2 (MM 25.124.6, #198): a two-slice K-loop body. tlower admits it with every tile in one
    # register group (4 tiles), so the column grid doubles to 64 columns per threadgroup. Neither changes a
    # value: the columns are independent, and G (the K slices, the fold) is the route's.
    unroll = KLOOP_UNROLL if (kl > 256 and KLOOP_UNROLL > 1
                              and 2 * grid_n * gl <= D.MAX_THREADGROUPS and nb % (2 * grid_n * 16) == 0) else 1
    grid_n = grid_n * unroll
    key = "gemm_n%d_k%d_g%d_s%d%s" % (nb, kl, grid_n, gl, "_u2" if unroll == 2 else "")
    owner.dx.bundle(key, projection_spec(nb, kl, grid_n, gl, unroll))
    ks = K // G
    a = np.zeros((GEMM_M, K), np.float16)
    a[0] = np.asarray(x, np.float16).reshape(-1)
    a32 = a.astype(F32)
    out = np.zeros(N, F32)
    parts_ok = fold_ok = True
    gpu_folds = 0
    for j in range(blocks):
        n0 = j * nb
        b = np.ascontiguousarray(np.asarray(w[:, n0:n0 + nb], np.float16))
        b32 = b.astype(F32)
        got = np.concatenate([owner.dx.dispatch(key, np.ascontiguousarray(a[:, l * kl:(l + 1) * kl]),
                                                np.ascontiguousarray(b[l * kl:(l + 1) * kl]),
                                                np.zeros((GEMM_M * gl, nb), "<f4")) for l in range(L)])
        want = np.concatenate([D.gemm(a32[:, t * ks:(t + 1) * ks], b32[t * ks:(t + 1) * ks]) for t in range(G)])
        ok = _bits_equal(got, want)
        if not ok:
            owner.log("MISMATCH %s block %d partials: %d elements" % (stage, j, int(np.sum(
                got.view(np.uint32) != want.view(np.uint32)))))
        cj = None if c is None else np.asarray(c, F32).reshape(-1)[n0:n0 + nb]
        cpad = None
        if cj is not None:
            cpad = np.zeros((GEMM_M, nb), F32)
            cpad[0] = cj
        ops = getattr(owner, "ops", None)
        folded = ops.fold(stage, got, G, GEMM_M, FOLD_LIVE_ROWS) if (ops is not None and G > 1) else None
        if folded is None:
            row = fold_split_k(got, G, GEMM_M, cpad)[0]                       # HOST STUB: fold, C last
        else:                                                                 # GPU fold (Set C's kernel), then
            row = folded[0] if cj is None else (folded[0] + cj).astype(F32)   # the residual C on the host, last
            gpu_folds += 1
        ref = D.gemm_reference(a32[:1], b32, None if cj is None else cj[None], split_k=G)[0]
        fok = _bits_equal(row, ref)
        if not fok:
            owner.log("MISMATCH %s block %d fold vs gemm_reference(split_k=%d): %d elements" % (
                stage, j, G, int(np.sum(row.view(np.uint32) != ref.view(np.uint32)))))
        parts_ok &= ok
        fold_ok &= fok
        out[n0:n0 + nb] = row
    owner.routes[stage] = dict(blocks=blocks, N=nb, K=K, grid_n=grid_n, split_k=G, kloop_unroll=unroll, k_launches=L,
                               split_k_per_launch=gl, threadgroups_per_launch=grid_n * gl, launches=blocks * L,
                               key=key, host_fold=G > 1 and gpu_folds < blocks, gpu_folds=gpu_folds,
                               host_residual=c is not None)
    owner.checks.append((stage, "gemm partials vs single-chain slices (grid_n %d, split_k %d, %d block(s) x "
                         "%d launch(es), N %d)" % (grid_n, G, blocks, L, nb), blocks * L, parts_ok))
    owner.checks.append((stage, "gemm %s fold vs gemm_reference(split_k=%d)" % ("GPU" if gpu_folds else "host", G),
                         blocks * L, fold_ok))
    return out


class Projections:
    """The four projection stages alone, at any spec (the whole milestone step runs through Pipeline), fed
    the reference's own stage inputs; each output compared with gemm_reference at its route's split_k."""

    def __init__(self, spec: D.LayerSpec, dispatcher: Dispatcher, log=print):
        if spec.storage != "half":
            raise ValueError("the GPU projections are half storage")
        self.spec, self.dx, self.log = spec, dispatcher, log
        self.checks, self.routes = [], {}

    def run(self, inputs):
        spec, I = self.spec, inputs
        env = D.reference(spec, inputs)["env"]                    # the stage inputs (h1, attn, x, h2, act, h)
        t0, got = time.time(), {}
        got["qkv32"] = project(self, "qkv_proj", env["h1"], I["wqkv"], blocks=3)
        got["h"] = project(self, "o_proj", env["attn"], I["wo"], I["x"])
        g = project(self, "ffn_gate_up", env["h2"], np.concatenate([I["wgate"], I["wup"]], axis=1), blocks=2)
        got["gate32"], got["up32"] = g[:spec.ffn_dim], g[spec.ffn_dim:]
        got["out32"] = project(self, "ffn_down", env["act"], I["wdown"], env["h"])
        self.log("projections done (%d dispatches, %.0f s)" % (self.dx.count, time.time() - t0))
        return got, env


STAGE_PLACEMENT = {"attn_norm": "gpu (decodeops)", "qkv_proj": "gpu N-tiled grid + split-K (+ host fold)",
                   "rope_append": "gpu (decodeops)", "attention": "gpu",
                   "o_proj": "gpu N-tiled grid + split-K (+ host fold, residual as the fold's C)",
                   "ffn_norm": "gpu (decodeops)", "ffn_gate_up": "gpu N-tiled grid", "ffn_swiglu": "gpu (decodeops)",
                   "ffn_down": "gpu N-tiled grid + split-K (+ host fold, residual as the fold's C, narrowing)"}


def attention_stage_main(spec, inputs, dx, args):
    """The attention stage alone at `spec` (the milestone by default): Q, K and V from the reference's own
    upstream stages, the class on the GPU, then o32 and its half narrowing against the reference."""
    ref = D.reference(spec, inputs)["env"]
    pipe = Pipeline(D.LayerSpec(**{**spec.as_dict(), "k_chunk": spec.k_chunk or 256}), dx,
                    attention_loop=not args.attention_straight, kv_split=args.kv_split,
                    merge_tiles=not args.merge_per_head, log=lambda m: print(m, file=sys.stderr, flush=True))
    t0 = time.time()
    o32 = pipe.attention(ref["q16"], ref["k_all"], ref["v_all"])
    host = o32 is None
    if host:
        o32 = D.stage_attention(spec, ref["q16"], ref["k_all"], ref["v_all"])["o32"]
    attn = D.narrow(o32).reshape(-1)
    rep = {"spec": spec.as_dict(), "seed": args.seed, "dry_run": bool(args.dry_run), "stage": "attention",
           "route": pipe.attn_route, "dispatches": dx.count, "gpu_seconds_sum": dx.gpu_seconds,
           "wall_s": time.time() - t0,
           "checks": [dict(stage=s, what=w, dispatches=n, bit_exact=e) for s, w, n, e in pipe.checks],
           "all_dispatches_pass": all(e for *_x, e in pipe.checks),
           "attention_words": {"total": pipe.attn_words, "bitwise": pipe.attn_bitwise,
                               "max_abs": pipe.attn_max_abs, "max_fraction_of_bound": pipe.attn_max_bound_frac,
                               "kv_split_value_change_row_max_fraction": pipe.attn_value_change},
           "o32": {"bit_exact_vs_reference": _bits_equal(o32, ref["o32"]),
                   "elements_differing": int(np.sum(np.asarray(o32, F32).view(np.uint32) != np.asarray(ref["o32"], F32).view(np.uint32))),
                   "vs_reference": D.error_report(o32, ref["o32"])},
           "attn_half": {"bit_exact_vs_reference": _bits_equal(attn, ref["attn"]),
                         "elements_differing": int(np.sum(np.asarray(attn, F32).view(np.uint32) != np.asarray(ref["attn"], F32).view(np.uint32)))}}
    text = json.dumps(rep, indent=1)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)
    return 0 if rep["all_dispatches_pass"] else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("workdir", type=Path)
    ap.add_argument("--spec", choices=sorted(D.SPECS), default=None,
                    help="default: today (the whole step), milestone with --projections or --stage attention")
    ap.add_argument("--stage", choices=("all", "attention"), default="all",
                    help="attention: the attention stage alone, its Q, K and V from g17decodestep's reference "
                         "(the milestone's item 1, MM 25.135); every other stage stays on the host")
    ap.add_argument("--seed", type=int, default=20260924)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--host-stubs", action="store_true",
                    help="run attn_norm, rope_append, ffn_norm and ffn_swiglu as the host stubs (the fallback)")
    ap.add_argument("--attention-straight", action="store_true",
                    help="run phase grid's straight-line program, not the counted key-block loop (MM 25.135.4)")
    ap.add_argument("--kv-split", type=int, choices=(2, 4, 8), default=None,
                    help="attention as the KV split (S threadgroups per head) then the merge; CHANGES THE VALUES "
                         "(MM 25.114.6, 25.135.5)")
    ap.add_argument("--merge-per-head", action="store_true",
                    help="with --kv-split: the per-head merge (MM 25.135.5), not the tiled one (MM 25.132.7)")
    ap.add_argument("--dry-run", action="store_true",
                    help="no GPU: every dispatch returns the repository's reference for its files")
    ap.add_argument("--projections", action="store_true",
                    help="only the four projection stages, fed the reference's stage inputs (any spec)")
    args = ap.parse_args(argv)
    spec = D.SPECS[args.spec or ("milestone" if (args.projections or args.stage == "attention") else "today")]
    inputs = D.make_inputs(spec, args.seed)
    dx = (DryRunDispatcher if args.dry_run else Dispatcher)(args.workdir)
    log = lambda m: print(m, file=sys.stderr, flush=True)
    if args.stage == "attention":
        return attention_stage_main(spec, inputs, dx, args)
    if args.projections:
        return _projections_main(args, spec, inputs, dx)
    ops = None if args.host_stubs else DecodeOps(args.workdir, dry_run=args.dry_run, log=log)
    pipe = Pipeline(spec, dx, log=log, ops=ops, attention_loop=not args.attention_straight, kv_split=args.kv_split,
                    merge_tiles=not args.merge_per_head)
    t0 = time.time()
    env = pipe.run(inputs)
    wall = time.time() - t0
    # the whole step against the reference at the pipeline's own split-K routes (k_route: G per projection
    # as projection_route gives it; with a k_chunk that is k_chunk's G, as before); the single chain beside
    ref = D.reference(D.LayerSpec(**{**spec.as_dict(), "k_route": True}), inputs)["env"]
    single = D.reference(D.LayerSpec(**{**spec.as_dict(), "k_chunk": 0, "k_route": False}), inputs)["env"]
    ide = D.ideal(spec, inputs)
    stages = {k: {"bit_exact_vs_reference": _bits_equal(env[k], ref[k]),
                  "elements_differing": int(np.sum(np.asarray(env[k], F32).view(np.uint32) != np.asarray(ref[k], F32).view(np.uint32))),
                  "vs_reference": D.error_report(env[k], ref[k]),
                  "vs_single_chain_reference": D.error_report(env[k], single[k]),
                  "vs_ideal": D.error_report(env[k], ide[k])}
              for k in ("h1", "qkv32", "q16", "o32", "attn", "h", "h2", "gate32", "up32", "act", "out32", "out")}
    placement = dict(STAGE_PLACEMENT)
    for st in ("attn_norm", "rope_append", "ffn_norm", "ffn_swiglu"):
        if ops is None or not any(c[0] == st for c in ops.checks):
            placement[st] = "host stub"
    for st, r in pipe.routes.items():                    # the split-K fold ran on the GPU (Set C's kernel)
        if r.get("gpu_folds"):
            placement[st] = placement[st].replace("host fold", "GPU fold")
    rep = {"spec": spec.as_dict(), "seed": args.seed, "dry_run": bool(args.dry_run), "placement": placement,
           "reference": "g17decodestep.reference at k_route (each projection at its route's split_k)",
           "projection_routes": pipe.routes, "attention_route": pipe.attn_route,
           "dispatches": dx.count + (ops.count if ops else 0), "gpu_seconds_sum": dx.gpu_seconds, "wall_s": wall,
           "checks": [dict(stage=s, what=w, dispatches=n, bit_exact=e) for s, w, n, e in pipe.checks + (ops.checks if ops else [])],
           "all_dispatches_pass": all(e for *_x, e in pipe.checks + (ops.checks if ops else [])),
           "gemm_dispatches_bit_exact": all(e for st, w, _n, e in pipe.checks if w.startswith("gemm")),
           "attention_words": {"total": pipe.attn_words, "bitwise": pipe.attn_bitwise,
                               "max_abs": pipe.attn_max_abs, "max_fraction_of_bound": pipe.attn_max_bound_frac,
                               "kv_split_value_change_row_max_fraction": pipe.attn_value_change},
           "end_to_end_bit_exact": all(v["bit_exact_vs_reference"] for v in stages.values()),
           # o32 lies downstream of exp2 and is checked inside attention_bound, not bitwise; its half narrowing
           # (attn) and every stage after it are bitwise
           "stages_not_bit_exact": [k for k, v in stages.items() if not v["bit_exact_vs_reference"]],
           "stages": stages}
    text = json.dumps(rep, indent=1)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)
    return 0 if rep["all_dispatches_pass"] else 1


def _projections_main(args, spec, inputs, dx):
    proj = Projections(spec, dx, log=lambda m: print(m, file=sys.stderr, flush=True))
    t0 = time.time()
    got, env = proj.run(inputs)
    wall = time.time() - t0
    stages = {}
    for k in ("qkv32", "h", "gate32", "up32", "out32"):
        stages[k] = {"vs_single_chain_reference": D.error_report(got[k], env[k]),
                     "elements_differing_from_single_chain": int(np.sum(
                         np.asarray(got[k], F32).view(np.uint32) != np.asarray(env[k], F32).view(np.uint32)))}
    rep = {"spec": spec.as_dict(), "seed": args.seed, "dry_run": bool(args.dry_run), "mode": "projections",
           "routes": proj.routes, "dispatches": dx.count, "gpu_seconds_sum": dx.gpu_seconds, "wall_s": wall,
           "checks": [dict(stage=s, what=w, dispatches=n, bit_exact=e) for s, w, n, e in proj.checks],
           "all_projections_bit_exact": all(e for *_x, e in proj.checks),
           "host_side": ["the split-K fold (fold_split_k, pending Set C's reduce kernel)",
                         "the residual add (the fold's C, added last)", "the 1-row A padding to 16 rows"],
           "split_k_value_change_vs_single_chain": stages}
    text = json.dumps(rep, indent=1)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)
    return 0 if rep["all_projections_bit_exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
