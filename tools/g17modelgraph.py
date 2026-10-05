#!/usr/bin/env python3
"""THE CHAINED TOKEN (docs/g17-tensorops-machine-model.md 25.138.3): InternLM2.5-1.8B's whole decode step as ONE
ordered list of dispatches whose buffers are bound at offsets in shared arenas, so every dispatch reads its
inputs where its producer wrote them and nothing is copied. graph.json is what Set C's executor
(tools/g17decodegen.m) runs; the simulator here runs the same graph on the CPU first.

    python3 tools/g17modelgraph.py build          author every bundle, solve the offsets, write graph.json + arenas
    python3 tools/g17modelgraph.py simulate       run one token of the graph on the CPU (every dispatch's
                                                  repository reference on its arena bytes, carrier zones
                                                  poisoned with NaN) and compare the logits with the model's
                                                  dry-run chain (the same programs' references, host glue)

THE OFFSETS. Every program binds three buffers and indexes from binding + 0: its carrier tile at c + 0, its inputs
and outputs at its layout's offsets. A dataflow edge "producer writes region R of its buffer 3 at OUT, consumer
reads it from its buffer 1 at IN" is the constraint base(producer, c) + OUT = base(consumer, a) + IN. The
constraints form a forest over the (dispatch, slot) bases; a union-find with offsets solves it, and each tree is
placed in the activation arena. Weights are read-only, one arena per layer plus the head's. The structural check
lists every write range (carrier zones included) that lands on a protected range (the KV caches, the host-written
words, a value still to be read); the simulator is the proof.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import g17decodestep as D  # noqa: E402
import g17decodeops as O  # noqa: E402
import g17realmodel as M  # noqa: E402

OUT = M.OUT / "graph"
BUNDLES = OUT / "bundles"
F32 = np.float32
ACT = "act"
ALIGN = 256
GAP = 1 << 20                                  # between placed trees


def _al(v, a=ALIGN):
    return -(-v // a) * a


class Op:
    """One dispatch: its bundle key, layout and kind; slots 1 (a), 2 (b), 3 (c) and their byte extents."""

    def __init__(self, name, key, kind, lay, extents, **meta):
        self.name, self.key, self.kind, self.lay, self.ext, self.meta = name, key, kind, lay, extents, meta

    def __repr__(self):
        return "Op(%s)" % self.name


# ---------------------------------------------------------------------------------------------------
# the ops of one layer, their bundles, and the dataflow constraints

def _gemm_op(name, N, K, w_name):
    import g17decodestep_gpu as G
    grid_n, Gs, L = D.projection_route(N, K)
    kl, gl = K // L, Gs // L
    if L != 1:
        raise ValueError("%s: one launch per block" % name)
    unroll = G.KLOOP_UNROLL if (kl > 256 and G.KLOOP_UNROLL > 1 and 2 * grid_n * gl <= D.MAX_THREADGROUPS
                                and N % (2 * grid_n * 16) == 0) else 1
    grid_n *= unroll
    key = "gemm_n%d_k%d_g%d_s%d%s" % (N, kl, grid_n, gl, "_u2" if unroll == 2 else "")
    spec = G.projection_spec(N, kl, grid_n, gl, unroll)
    ext = {1: G.GEMM_M * K * 2, 2: K * N * 2, 3: G.GEMM_M * gl * N * 4}
    return Op(name, key, "gemm", dict(N=N, K=K, G=gl, spec=spec), ext, weights=w_name)


def _deco_op(name, key, kind, lay, **meta):
    return Op(name, key, kind, lay, {1: lay["a_bytes"], 2: lay["b_bytes"], 3: lay["c_bytes"]}, **meta)


def layer_ops(l, spec):
    """The ordered dispatches of layer l (kv before q and gate before up: see the module doc's order rule)."""
    d, f = spec.d_model, spec.ffn_dim
    n1 = _deco_op("L%d.attn_norm" % l, "rmsnorm_half_d%d_g32_u16_h" % d, "norm",
                  O.rmsnorm_loop_layout(d, "half", groups=32, unroll=16, hoist=True), g="L%d_g1" % l)
    kg = _gemm_op("L%d.kv_gemm" % l, d, d, "L%d_wkv" % l)
    kf = _deco_op("L%d.kv_fold" % l, "fold_g%d_m16_n%d_r1" % (kg.lay["G"], d), "fold", O.fold_layout(kg.lay["G"], 16, d, 1))
    qg = _gemm_op("L%d.q_gemm" % l, d, d, "L%d_wq" % l)
    qf = _deco_op("L%d.q_fold" % l, kf.key, "fold", kf.lay)
    cap = 272
    rl = O.rope_layout(spec.n_heads, spec.head_dim, cap, groups=16, cache="grid", kv_heads=spec.kv_heads, chained=True)
    rp = _deco_op("L%d.rope" % l, "rope_h%d_d%d_c%d_chained_kv%d" % (spec.n_heads, spec.head_dim, cap, spec.kv_heads),
                  "rope", rl)
    import agxforge.g17.runtime as R
    sreq = dict(phase="grid", heads=16, rows=1, blocks=R.ATTENTION_GRID_CAPACITY, q0=256, key_offsets="loop",
                kv_split=8, runtime_q0=True)
    at = R.attention_spec(sreq)
    slay = R.attention_layout(at)
    sp = Op("L%d.attn_split" % l, "grid_h16_b17_split8_rt", "split", dict(req=sreq, lay=slay),
            {1: 16 * 8192, 2: 16 * R.ATTENTION_GRID_STRIDE["B"], 3: slay["M"] * slay["N"] * 4})
    mreq = dict(phase="grid_merge", heads=16, rows=1, kv_split=8, allow_value_change=True, tile_groups=8)
    mlay = R.attention_layout(R.attention_spec(mreq))
    mg = Op("L%d.attn_merge" % l, "grid_merge_h16_s8_t8", "merge", dict(req=mreq, lay=mlay),
            {1: 16 * 8192, 2: mlay["K"] * mlay["N"] * 2, 3: mlay["M"] * mlay["N"] * 4})
    ga = _deco_op("L%d.attn_gather" % l, "attn_gather_h16_v128", "gather", O.attn_gather_layout(16, 128))
    og = _gemm_op("L%d.o_gemm" % l, d, d, "L%d_wo" % l)
    of = _deco_op("L%d.o_fold" % l, kf.key, "fold", kf.lay)
    r1 = _deco_op("L%d.residual1" % l, "residual_n%d_rh_o32" % d, "residual", O.residual_layout(d, "half", True, False))
    n2 = _deco_op("L%d.ffn_norm" % l, "rmsnorm_float_d%d_g32_u8_h" % d, "norm",
                  O.rmsnorm_loop_layout(d, "float", groups=32, unroll=8, hoist=True), g="L%d_g2" % l)
    gt = _gemm_op("L%d.gate_gemm" % l, f, d, "L%d_wgate" % l)
    up = _gemm_op("L%d.up_gemm" % l, f, d, "L%d_wup" % l)
    sw = _deco_op("L%d.swiglu" % l, "swiglu_f%d_g256" % f, "swiglu", O.swiglu_layout(f, 256))
    dg = _gemm_op("L%d.down_gemm" % l, d, f, "L%d_wdown" % l)
    df = _deco_op("L%d.down_fold" % l, "fold_g%d_m16_n%d_r1" % (dg.lay["G"], d), "fold", O.fold_layout(dg.lay["G"], 16, d, 1))
    r2 = _deco_op("L%d.residual2" % l, "residual_n%d_rf_oh" % d, "residual", O.residual_layout(d, "float", False, True))
    ops = [n1, kg, kf, qg, qf, rp, sp, mg, ga, og, of, r1, n2, gt, up, sw, dg, df, r2]
    x = (n1, 1, n1.lay["X"])
    h32 = (r1, 3, r1.lay["OUT32"])
    edges = [
        (x, (r1, 1, r1.lay["R"])),
        ((n1, 3, n1.lay["OUT"]), (kg, 1, 0)), ((n1, 3, n1.lay["OUT"]), (qg, 1, 0)),
        ((kg, 3, 0), (kf, 1, kf.lay["P"])), ((qg, 3, 0), (qf, 1, qf.lay["P"])),
        ((qf, 3, qf.lay["OUT"]), (rp, 1, rl["QKV"])), ((kf, 3, kf.lay["OUT"]), (rp, 1, rl["QKV"] + 4 * d)),
        ((rp, 3, rl["QA"]), (sp, 1, 0)), ((rp, 3, rl["VC"]), (sp, 2, 0)), ((rp, 3, rl["KC"]), (sp, 3, 0)),
        ((sp, 1, 0), (mg, 1, 0)), ((sp, 3, 0), (mg, 3, 0)), ((sp, 3, 0), (ga, 1, 0)),
        ((ga, 3, ga.lay["OUTH"]), (og, 1, 0)), ((og, 3, 0), (of, 1, of.lay["P"])),
        ((of, 3, of.lay["OUT"]), (r1, 1, r1.lay["F"])),
        (h32, (n2, 1, n2.lay["X"])), (h32, (r2, 1, r2.lay["R"])),
        ((n2, 3, n2.lay["OUT"]), (gt, 1, 0)), ((n2, 3, n2.lay["OUT"]), (up, 1, 0)),
        ((gt, 3, 0), (sw, 1, sw.lay["GATE"])), ((up, 3, 0), (sw, 1, sw.lay["UP"])),
        ((sw, 3, sw.lay["OUT"]), (dg, 1, 0)), ((dg, 3, 0), (df, 1, df.lay["P"])),
        ((df, 3, df.lay["OUT"]), (r2, 1, r2.lay["F"])),
    ]
    out = (r2, 3, r2.lay["OUTH"])
    return ops, edges, x, out


def head_ops(spec):
    d = spec.d_model
    fn = _deco_op("head.final_norm", "rmsnorm_half_d%d_g32_u16_h" % d, "norm",
                  O.rmsnorm_loop_layout(d, "half", groups=32, unroll=16, hoist=True), g="norm")
    lms = [_gemm_op("head.lm%d" % k, M.LM_BLOCK, d, "lm_head_%d" % k) for k in range(M.VOCAB_PADDED // M.LM_BLOCK)]
    edges = [((fn, 3, fn.lay["OUT"]), (lm, 1, 0)) for lm in lms]
    # the logits: block k's C row 0 at LOGITS + 32,768 k (its padding rows spill onto later blocks, which are
    # written after it; each block's own row 0 is written last)
    for k in range(1, len(lms)):
        edges.append(((lms[0], 3, 4 * M.LM_BLOCK * k), (lms[k], 3, 0)))
    return [fn] + lms, edges, (fn, 1, fn.lay["X"])


# ---------------------------------------------------------------------------------------------------
# the solver

class Solver:
    def __init__(self):
        self.parent, self.off = {}, {}

    def find(self, v):
        if v not in self.parent:
            self.parent[v], self.off[v] = v, 0
            return v, 0
        acc = 0
        while self.parent[v] != v:
            acc += self.off[v]
            v = self.parent[v]
        return v, acc

    def same(self, u, du, v, dv):
        """base(u) + du == base(v) + dv."""
        ru, ou = self.find(u)
        rv, ov = self.find(v)
        if ru == rv:
            if ou + du != ov + dv:
                raise ValueError("inconsistent constraint %s+%d = %s+%d" % (u, du, v, dv))
            return
        # base(u) = base(ru) + ou; base(ru) = base(rv) + (ov + dv - ou - du)
        self.parent[ru], self.off[ru] = rv, ov + dv - ou - du


def build(tokens_capacity=272):
    spec = M.layer_spec(0)
    ops, edges = [], []
    prev_out = None
    xs = []
    for l in range(M.LAYERS):
        lo, le, x, out = layer_ops(l, spec)
        ops += lo
        edges += le
        if prev_out is not None:
            edges.append((prev_out, x))
        xs.append(x)
        prev_out = out
    ho, he, hx = head_ops(spec)
    ops += ho
    edges += he
    edges.append((prev_out, hx))
    S = Solver()
    act_slots = set()
    for (p, ps, po), (c, cs, co) in edges:
        S.same((p.name, ps), po, (c.name, cs), co)
        act_slots |= {(p.name, ps), (c.name, cs)}
    # every a and c slot lives in the activation arena; b slots are weights (their own arenas) or the zero arena
    byname = {o.name: o for o in ops}
    for o in ops:
        for slot in (1, 3):
            S.find((o.name, slot))
            act_slots.add((o.name, slot))
    trees = {}
    for v in act_slots:
        r, off = S.find(v)
        trees.setdefault(r, []).append((v, off))
    base = {}
    cursor = GAP
    for r, members in sorted(trees.items(), key=lambda kv: str(kv[0])):
        lo = min(off for v, off in members)
        hi = max(off + byname[v[0]].ext[v[1]] for v, off in members)
        start = _al(cursor - lo)
        for v, off in members:
            base[v] = start + off
        cursor = start + hi + GAP
    act_bytes = _al(cursor)
    return spec, ops, edges, base, act_bytes, xs


# ---------------------------------------------------------------------------------------------------
# authoring, weights and graph.json

def _author(op, spec):
    import g17tensorcommonruntime as TCR
    path = BUNDLES / op.key
    if (path / "manifest.json").exists():
        return path
    if op.kind == "gemm":
        TCR.author_generic(path, op.lay["spec"])
    elif op.kind in ("split", "merge"):
        spec_ = {"attention": op.lay["req"]}
        if op.kind == "merge":
            seed = BUNDLES / "merge-author-inputs.npz"
            np.savez(seed, q=np.zeros((16, 1, 128), np.float16), k0=np.zeros((16, 16, 128), np.float16),
                     O=np.zeros((16, 8, 1, 128), F32), M=np.zeros((16, 8, 1), F32), L=np.ones((16, 8, 1), F32))
            spec_["grid_inputs"] = str(seed)
        TCR.author_generic(path, spec_)
    else:
        lay = op.lay
        builders = {"norm": O.build_rmsnorm_loop, "fold": O.build_split_k_fold, "swiglu": O.build_swiglu,
                    "residual": O.build_residual, "gather": O.build_attn_gather}
        if op.kind == "rope":
            prog = O.build_rope_append(lay, D.q_scale(spec))
        elif op.kind == "norm":
            prog = O.build_rmsnorm_loop(lay, spec.norm_eps)
        else:
            prog = builders[op.kind](lay)
        a, b, c = O._buffers(lay)
        O._with_carrier(lay, (a, b, c))
        O.author(path, lay, prog, bytes(a), bytes(b), bytes(c), extra={"program": op.key})
    return path


def weights_for(op, l_weights):
    """The bytes of op's buffer 2: a GEMM's weight block (K x N, fp16), a norm's carrier tile + gain at G."""
    if op.kind == "gemm":
        return np.ascontiguousarray(l_weights(op.meta["weights"]).astype(np.float16)).tobytes()
    if op.kind == "norm":
        b = bytearray(op.lay["b_bytes"])
        O._place(b, 0, O.carrier_tiles(op.lay)[1])
        O._place(b, op.lay["G"], np.asarray(l_weights(op.meta["g"]), np.float16))
        return bytes(b)
    return None


def _weight(name):
    """Named weight blocks: L{l}_wq / _wkv are the two N-2048 blocks of wqkv, lm_head_k the k-th 8,192 block."""
    if name.startswith("lm_head_"):
        k = int(name.split("_")[-1])
        return np.load(M.WEIGHTS / "lm_head.npy", mmap_mode="r")[:, k * M.LM_BLOCK:(k + 1) * M.LM_BLOCK]
    if name.endswith("_wq") or name.endswith("_wkv"):
        l = name.split("_")[0]
        w = np.load(M.WEIGHTS / (l + "_wqkv.npy"), mmap_mode="r")
        return w[:, :2048] if name.endswith("_wq") else w[:, 2048:4096]
    return np.load(M.WEIGHTS / (name + ".npy"), mmap_mode="r")


def write_graph():
    spec, ops, edges, base, act_bytes, xs = build()
    BUNDLES.mkdir(parents=True, exist_ok=True)
    arenas = {ACT: act_bytes, "zero": 4 << 20}
    files, binds = {}, []
    warena = {}
    for o in ops:
        _author(o, spec)
    for o in ops:
        arena_name = "W" + o.name.split(".")[0]
        wb = weights_for(o, _weight)
        b2 = None
        if wb is not None:
            wa = warena.setdefault(arena_name, bytearray())
            b2 = (arena_name, len(wa))
            wa += wb
            wa += bytes(_al(len(wa)) - len(wa))
        m = json.loads((BUNDLES / o.key / "manifest.json").read_text())
        t = m["tensor"]
        binds.append(dict(name=o.name, bundle=str(BUNDLES / o.key), threads=t["grid"][0], group=t["threadgroup"][0],
                          binds={"1": dict(arena=ACT, offset=base[(o.name, 1)], bytes=o.ext[1]),
                                 # buffer 2: weights, OR an activation region a dataflow edge placed (the split's V
                                 # cache is the RoPE's VC region), else the shared zeros
                                 "2": (dict(arena=b2[0], offset=b2[1], bytes=o.ext[2]) if b2 else
                                       dict(arena=ACT, offset=base[(o.name, 2)], bytes=o.ext[2])
                                       if (o.name, 2) in base else dict(arena="zero", offset=0, bytes=o.ext[2])),
                                 "3": dict(arena=ACT, offset=base[(o.name, 3)], bytes=o.ext[3])}))
    OUT.mkdir(parents=True, exist_ok=True)
    for name, wa in warena.items():
        f = OUT / ("%s.bin" % name)
        f.write_bytes(bytes(wa))
        arenas[name] = len(wa)
        files[name] = str(f)
    byname = {o.name: o for o in ops}
    per_token, zero = [], []
    import agxforge.g17.runtime as R
    for l in range(M.LAYERS):
        rp, sp = byname["L%d.rope" % l], byname["L%d.attn_split" % l]
        ra = base[(rp.name, 1)]
        per_token += [dict(arena=ACT, offset=ra + rp.lay["LEN"], bytes=4, source="rope_len_word"),
                      dict(arena=ACT, offset=ra + rp.lay["COS"], bytes=4 * 64, source="rope_cos_row"),
                      dict(arena=ACT, offset=ra + rp.lay["SIN"], bytes=4 * 64, source="rope_sin_row"),
                      dict(arena=ACT, offset=base[(sp.name, 3)] + R.ATTENTION_GRID_LENGTH_BYTE, bytes=4,
                           source="split_len_word")]
    x0 = xs[0]
    per_token.insert(0, dict(arena=ACT, offset=base[(x0[0].name, x0[1])] + x0[2], bytes=2 * spec.d_model,
                             source="embedding_row"))
    zero.append(dict(arena=ACT, offset=0, bytes=act_bytes))        # the whole activation arena starts zero
    lm0 = byname["head.lm0"]
    rope_tab = OUT / "rope_tables.npz"
    pos = np.arange(272)[:, None] * (M.ROPE_THETA ** (-np.arange(0, 128, 2, dtype=np.float64) / 128))[None, :]
    np.savez(rope_tab, cos=np.cos(pos).astype(F32), sin=np.sin(pos).astype(F32))
    (OUT / "rope_cos.f32").write_bytes(np.cos(pos).astype("<f4").tobytes())       # raw [272 x 64] for the executor
    (OUT / "rope_sin.f32").write_bytes(np.sin(pos).astype("<f4").tobytes())
    g = dict(model=M.MODEL_ID, arenas=arenas, arena_init=[dict(arena=k, offset=0, file=v) for k, v in files.items()],
             zero_init=zero, dispatches=binds, per_token_writes=per_token,
             tables=dict(embedding=str(M.WEIGHTS / "embed.npy"), rope=str(rope_tab)),
             rope_cos_file=str(OUT / "rope_cos.f32"), rope_sin_file=str(OUT / "rope_sin.f32"),
             prompt_ids=[1, 918, 11498, 2327, 1197, 48304, 416],
             logits_readback=dict(arena=ACT, offset=base[(lm0.name, 3)], bytes=4 * M.VOCAB_PADDED),
             logits_dtype="f32", vocab=M.VOCAB, capacity=272,
             notes="embedding_row: fp16 row of the embedding table at the token (prompt tokens first, then the "
                   "argmax of the previous logits over the first `vocab`); rope_*_row: row kv_len of the rope table; "
                   "len words: kv_len (the new token's position). One command buffer per token.")
    (OUT / "graph.json").write_text(json.dumps(g, indent=1) + "\n")
    return g, spec, ops, base


def overlaps(spec, ops, base):
    """Write ranges landing on the protected ranges: the KV caches and the host-written words (always live)."""
    import agxforge.g17.runtime as R
    byname = {o.name: o for o in ops}
    protected = []
    for l in range(M.LAYERS):
        rp = byname["L%d.rope" % l]
        c = base[(rp.name, 3)]
        for h in range(16):
            protected.append(("L%d K cache h%d" % (l, h), c + rp.lay["KC"] + h * 131072, 272 * 256))
        protected.append(("L%d V cache" % l, c + rp.lay["VC"], 16 * 131072))
        a = base[(rp.name, 1)]
        protected.append(("L%d rope words" % l, a + rp.lay["COS"], rp.lay["LEN"] + 4 - rp.lay["COS"]))
        protected.append(("L%d split len" % l, base[("L%d.attn_split" % l, 3)] + R.ATTENTION_GRID_LENGTH_BYTE, 4))
    bad = []
    for o in ops:
        c = base[(o.name, 3)]
        writes = []
        if o.kind in ("norm", "fold", "swiglu", "residual", "gather", "rope"):
            writes.append(("carrier", c, O._carrier_bytes(o.lay["groups"])))
        if o.kind == "gemm":
            writes.append(("C", c, o.ext[3]))
        for tag, s, n in writes:
            for pname, ps, pn in protected:
                if s < ps + pn and ps < s + n:
                    bad.append("%s %s [%d, %d) hits %s" % (o.name, tag, s, s + n, pname))
    return bad


# ---------------------------------------------------------------------------------------------------
# the CPU simulator

NAN = np.frombuffer(np.array([0x7FC00000], "<u4").tobytes(), "<f4")[0]


class Sim:
    """The graph's arenas as bytes, and each dispatch's effect computed by the reference its GPU check uses."""

    def __init__(self, g):
        self.g = g
        self.arena = {ACT: np.zeros(g["arenas"][ACT], np.uint8), "zero": np.zeros(g["arenas"]["zero"], np.uint8)}
        for init in g["arena_init"]:
            self.arena[init["arena"]] = np.memmap(init["file"], np.uint8, "r")
        self.embed = np.load(g["tables"]["embedding"], mmap_mode="r")
        z = np.load(g["tables"]["rope"])
        self.cos, self.sin = z["cos"], z["sin"]
        self.tmp = Path(tempfile.mkdtemp(prefix="g17graph-"))

    def v(self, arena, off, n, dt):
        return np.frombuffer(self.arena[arena][off:off + n * np.dtype(dt).itemsize].tobytes(), dt, n)

    def w(self, off, arr):
        raw = np.ascontiguousarray(arr).tobytes()
        self.arena[ACT][off:off + len(raw)] = np.frombuffer(raw, np.uint8)

    def poison(self, off, n):
        self.w(off, np.full(n // 4, NAN, "<f4"))

    def token(self, tok, pos, ops):
        g = self.g
        for pw in g["per_token_writes"]:
            src, off = pw["source"], pw["offset"]
            if src == "embedding_row":
                self.w(off, np.asarray(self.embed[int(tok)], np.float16))
            elif src in ("rope_len_word", "split_len_word"):
                self.w(off, np.array([pos], "<u4"))
            elif src == "rope_cos_row":
                self.w(off, self.cos[pos])
            elif src == "rope_sin_row":
                self.w(off, self.sin[pos])
        for d, op in zip(g["dispatches"], ops):
            self.run(d, op, pos)
        lr = g["logits_readback"]
        return self.v(lr["arena"], lr["offset"], g["vocab"], "<f4").copy()

    def run(self, d, op, pos):
        b1, b2, b3 = (d["binds"][k] for k in ("1", "2", "3"))
        a, c = b1["offset"], b3["offset"]
        lay, spec = op.lay, M.layer_spec(pos)
        if op.kind == "gemm":
            N, K, G = lay["N"], lay["K"], lay["G"]
            A = self.v(ACT, a, 16 * K, np.float16).reshape(16, K).astype(F32)
            B = self.v(b2["arena"], b2["offset"], K * N, np.float16).reshape(K, N).astype(F32)
            ks = K // G
            C = np.concatenate([D.gemm(A[:, t * ks:(t + 1) * ks], B[t * ks:(t + 1) * ks]) for t in range(G)])
            self.w(c, C.astype("<f4"))
            return
        if op.kind in ("split", "merge"):
            import g17tensorcommonruntime as TCR
            bdir = Path(d["bundle"])
            t = self.tmp / op.kind
            t.mkdir(exist_ok=True)
            for f in ("generic.json", "manifest.json"):
                shutil.copy(bdir / f, t / f)
            sizes = {n: (bdir / n).stat().st_size for n in ("a.f16", "b.f16", "c.f32")}
            av = np.zeros(sizes["a.f16"], np.uint8)
            av[:16 * 8192] = self.arena[ACT][a:a + 16 * 8192]
            (t / "a.f16").write_bytes(av.tobytes())
            (t / "b.f16").write_bytes(self.arena[b2["arena"]][b2["offset"]:b2["offset"] + sizes["b.f16"]].tobytes())
            (t / "c.f32").write_bytes(self.arena[ACT][c:c + sizes["c.f32"]].tobytes())
            s = TCR.generic_spec(json.loads((t / "generic.json").read_text()))
            self.w(c, np.asarray(TCR.attention_reference(t, s), "<f4"))
            return
        self.poison(c, O._carrier_bytes(lay["groups"]))
        if op.kind == "fold":
            G, N = lay["G"], lay["NF"]
            P = self.v(ACT, a + lay["P"], G * 16 * N, "<f4").reshape(G * 16, N)
            self.w(c + lay["OUT"], O.host_fold(P, G, 16)[:1].astype("<f4"))
        elif op.kind == "norm":
            n = spec.d_model
            x = self.v(ACT, a + lay["X"], n, np.float16 if lay["in_dtype"] == "half" else "<f4").astype(F32)
            gg = self.v(b2["arena"], b2["offset"] + lay["G"], n, np.float16).astype(F32)
            self.w(c + lay["OUT"], D.rmsnorm(x, gg, spec).astype(np.float16))
        elif op.kind == "residual":
            n = lay["n"]
            f = self.v(ACT, a + lay["F"], n, "<f4")
            r = self.v(ACT, a + lay["R"], n, np.float16 if lay["r_dtype"] == "half" else "<f4").astype(F32)
            y = (f + r).astype(F32)
            if lay["out32"]:
                self.w(c + lay["OUT32"], y)
            if lay["outh"]:
                self.w(c + lay["OUTH"], y.astype(np.float16))
        elif op.kind == "gather":
            o = np.zeros((16, 128), F32)
            for h in range(16):
                for t in range(8):
                    o[h, 16 * t:16 * t + 16] = self.v(ACT, a + h * lay["HSTRIDE"] + lay["O0"] + 2048 * t, 16, "<f4")
            self.w(c + lay["OUTH"], o.reshape(-1).astype(np.float16))
        elif op.kind == "swiglu":
            f = lay["ffn"]
            gate, upv = self.v(ACT, a + lay["GATE"], f, "<f4"), self.v(ACT, a + lay["UP"], f, "<f4")
            self.w(c + lay["OUT"], D.stage_ffn_swiglu(spec, gate, upv)["act"].astype(np.float16))
        elif op.kind == "rope":
            L = int(self.v(ACT, a + lay["LEN"], 1, "<u4")[0])
            sp = M.layer_spec(L)
            qkv = self.v(ACT, a + lay["QKV"], sp.qkv_width, "<f4")
            cs, sn = self.v(ACT, a + lay["COS"], 64, "<f4"), self.v(ACT, a + lay["SIN"], 64, "<f4")
            ki = self.v(ACT, c + lay["KC"], 16 * O.GRID_HEAD_ELEMS, np.float16).reshape(16, -1)
            vi = self.v(ACT, c + lay["VC"], 16 * O.GRID_HEAD_ELEMS, np.float16).reshape(16, -1)
            k, v = O.grid_cache_arrays(ki, vi, max(L, 1))
            ref = D.stage_rope_append(sp, qkv, cs, sn, k[:, :L], v[:, :L])
            for h in range(16):
                self.w(c + lay["QA"] + 8192 * h, ref["q16"][h].astype(np.float16))
                self.w(c + lay["KC"] + h * 131072 + L * 256, ref["k_new"][h].astype(np.float16))
                for t in range(8):
                    self.w(c + lay["VC"] + h * 131072 + (L // 16) * 4096 + t * 512 + (L % 16) * 32,
                           ref["v_new"][h][16 * t:16 * t + 16].astype(np.float16))
        else:
            raise ValueError(op.kind)


def simulate(prompt, out):
    g = json.loads((OUT / "graph.json").read_text())
    spec, ops, edges, base, act_bytes, xs = build()
    sim = Sim(g)
    ref = M.Model("dry", OUT / "dry-work", kv_split=8, log=lambda s: None)
    rows = []
    for pos, tok in enumerate(prompt):
        t0 = time.time()
        got = sim.token(tok, pos, ops)
        t1 = time.time()
        want = ref.step(tok)
        print("pos %d: graph %.0f s, dry chain %.0f s" % (pos, t1 - t0, time.time() - t1), flush=True)
        same = got.view("<u4") == np.asarray(want, "<f4").view("<u4")
        rows.append(dict(pos=pos, token=int(tok), bitwise=int(same.sum()), of=int(same.size),
                         max_abs=float(np.nanmax(np.abs(got - want))), nan=int(np.isnan(got).sum()),
                         argmax=[int(np.nanargmax(got)), int(np.argmax(want))]))
        print(rows[-1], flush=True)
    rep = dict(prompt=list(map(int, prompt)), rows=rows, all_bitwise=all(r["bitwise"] == r["of"] for r in rows))
    Path(out).write_text(json.dumps(rep, indent=1) + "\n")
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("build", "check", "simulate", "simonly"))
    ap.add_argument("--tokens", type=int, default=0)
    ap.add_argument("--prompt-ids", default="1,918,11498,2327,1197,48304,416")
    a = ap.parse_args(argv)
    if a.cmd == "simonly":
        g = json.loads((OUT / "graph.json").read_text())
        spec, ops, edges, base, act_bytes, xs = build()
        sim = Sim(g)
        prompt = [int(t) for t in a.prompt_ids.split(",")]
        toks, logits = [], []
        nxt = None
        for pos in range(len(prompt) + a.tokens):
            tok = prompt[pos] if pos < len(prompt) else nxt
            t0 = time.time()
            lg = sim.token(tok, pos, ops)
            nxt = int(np.argmax(lg))
            logits.append(lg)
            if pos >= len(prompt) - 1:
                toks.append(nxt)
            print("pos %d tok %d -> argmax %d (%.0f s)" % (pos, tok, nxt, time.time() - t0), flush=True)
        np.savez(OUT / "simonly.npz", tokens=np.array(toks), logits=np.stack(logits))
        print(json.dumps(dict(out_ids=toks)))
        return 0
    if a.cmd == "simulate":
        rep = simulate([int(t) for t in a.prompt_ids.split(",")], OUT / "simulate.json")
        return 0 if rep["all_bitwise"] else 1
    if a.cmd == "check":
        spec, ops, edges, base, act_bytes, xs = build()
        bad = overlaps(spec, ops, base)
        print("act arena %.1f MB, %d dispatches, %d protected-range hits" % (act_bytes / 1e6, len(ops), len(bad)))
        for b in bad[:20]:
            print("  ", b)
        return 1 if bad else 0
    g, spec, ops, base = write_graph()
    bad = overlaps(spec, ops, base)
    print("graph.json: %d dispatches, arenas %s, %d protected-range hits" % (
        len(g["dispatches"]), {k: "%.1f MB" % (v / 1e6) for k, v in g["arenas"].items()}, len(bad)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
