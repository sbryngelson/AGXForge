#!/usr/bin/env python3
"""Device-side greedy decoding (MM 25.140.2): the next token chosen on the GPU, so a generation needs no host round
trip between tokens.

argmax over V logits in two dispatches, the first maximum winning a tie (numpy's and mlx's argmax):
  (indices ride as exact floats: the lane shuffle is measured for fp32 only)
  pass 1  G threadgroups of 32 lanes; threadgroup t scans rows [t C, (t + 1) C), lane l the rows t C + l + 32 i
          ascending (a later equal value never displaces an earlier one), then the xor butterflies combine lanes
          with (larger value, or equal value and smaller index); writes (value, index) at [PAIRS + 2 t]
  pass 2  one threadgroup: lane l combines pairs l, l + 32, ... ascending, the same butterflies; writes the
          token at [TOK]
Bindings: pass 1  b1 logits fp32 [V] at 0 | b3 pairs;  pass 2  b1 pairs | b3 the token word (uint32).

    python3 tools/g17gen.py check
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import g17decodeops as O  # noqa: E402

F32 = np.float32
MASKS = (1, 8, 2, 4, 16)


def argmax_layout(V=92544, per_lane=12):
    C = 32 * per_lane
    if V % C:
        raise ValueError("argmax: V must be a multiple of 32 x per_lane")
    G = V // C
    if G > 32 * 8:
        raise ValueError("argmax: pass 2 combines at most 256 pairs")
    return dict(op="argmax", V=V, per_lane=per_lane, C=C, G=G, PAIRS=0, TOK=0, pairs_bytes=8 * G)


def _combine(b, ir, v, i, v2, i2, tag):
    """(v, i) vs (v2, i2): the larger value, or on equal values the smaller index. Indices are carried as EXACT
    floats (below 2^24): the cross-lane shuffle is measured for fp32 only, and fmin orders them exactly."""
    I = ir.I32
    m = b.fmax(v, v2, type=I, name=tag + "_m")
    ea = b.icmp(v, m, "eq", name=tag + "_ea")
    eb = b.icmp(v2, m, "eq", name=tag + "_eb")
    lo = b.fmin(i, i2, type=I, name=tag + "_lo")
    both = b.icmp(b.add(ea, eb, name=tag + "_s"), O._c(b, 2, tag + "_two"), "eq", name=tag + "_both")
    pick = b.csel(ea, O._c(b, 0, tag + "_z2"), i, i2, rel="gt", name=tag + "_pk")
    idx = b.csel(both, O._c(b, 0, tag + "_z3"), lo, pick, rel="gt", name=tag + "_ix")
    return m, idx


def _f32(b, ir, x, tag):
    """An F32-typed copy (x + 0.0): what the shuffle accepts."""
    return b.fadd(x, O._cf(b, F32(0.0), tag + "_zf"), type=ir.F32, name=tag + "_f")


def _lane_reduce(b, ir, v, i, tag):
    for k, mask in enumerate(MASKS):
        vs = b.simd_shuffle_xor(_f32(b, ir, v, "%s_v%d" % (tag, k)), mask, name="%s_vs%d" % (tag, k))
        is_ = b.simd_shuffle_xor(_f32(b, ir, i, "%s_i%d" % (tag, k)), mask, name="%s_is%d" % (tag, k))
        v, i = _combine(b, ir, v, i, vs, is_, "%s_c%d" % (tag, k))
    return v, i


def _to_int(b, ir, fi, tag):
    """An exact float index (< 2^23) to its integer: bits(fi + 2^23) - bits(2^23)."""
    s_ = b.fadd(fi, O._cf(b, F32(8388608.0), tag + "_magic"), type=ir.I32, name=tag + "_s")
    return b.sub(s_, O._c(b, 0x4B000000, tag + "_mb"), name=tag + "_int")


def build_pass1(lay):
    from agxforge.g17 import cc, ir
    fn, b, a, bb, c = O._function()
    I = ir.I32
    lane = b.builtin("thread_index_in_simdgroup", name="lane")
    t = b.builtin("threadgroup_position_in_grid", name="t")
    base = b.add(b.mul(t, O._c(b, lay["C"], "C"), name="tC"), lane, name="base")
    v, i = None, None
    if lay.get("batched"):
        # BATCHED (MM 25.144.7): every load issued before any is consumed; the combines keep their order
        idxs = [b.add(base, O._c(b, 32 * k, "o%d" % k), name="ix%d" % k) if k else base for k in range(lay["per_lane"])]
        xs = [b.load(a, idx, type=I, name="x%d" % k) for k, idx in enumerate(idxs)]
        for k, (idx, x) in enumerate(zip(idxs, xs)):
            xv = b.fadd(x, O._cf(b, F32(0.0), "z%d" % k), type=I, name="xv%d" % k)
            fi = b.u32_to_f32(idx, name="fi%d" % k)
            v, i = (xv, fi) if v is None else _combine(b, ir, v, i, xv, fi, "s%d" % k)
    for k in (range(lay["per_lane"]) if not lay.get("batched") else ()):
        idx = b.add(base, O._c(b, 32 * k, "o%d" % k), name="ix%d" % k) if k else base
        x = b.load(a, idx, type=I, name="x%d" % k)
        xv = b.fadd(x, O._cf(b, F32(0.0), "z%d" % k), type=I, name="xv%d" % k)   # through an ALU (load-use)
        fi = b.u32_to_f32(idx, name="fi%d" % k)
        if v is None:
            v, i = xv, fi
        else:
            v, i = _combine(b, ir, v, i, xv, fi, "s%d" % k)
    v, i = _lane_reduce(b, ir, v, i, "r")
    pb = b.add(b.shl(t, O._c(b, 1, "one"), name="t2"), O._c(b, lay["PAIRS"] // 4, "pb"), name="pw")
    b.store_at(c, pb, v)
    b.store_at(c, b.add(pb, O._c(b, 1, "one1"), name="pw1"), i)
    b.ret()
    ir.verify(fn)
    return cc.compile_function(fn)


def build_pass2(lay):
    from agxforge.g17 import cc, ir
    fn, b, a, bb, c = O._function()
    I = ir.I32
    lane = b.builtin("thread_index_in_simdgroup", name="lane")
    G = lay["G"]
    v, i = None, None
    for k in range(-(-G // 32)):
        # pair l + 32 k; lanes past G repeat pair 0, which cannot change the result (an equal value, a larger-or-
        # equal index is never preferred over itself)
        p = b.add(lane, O._c(b, 32 * k, "p%d" % k), name="pi%d" % k) if k else lane
        if 32 * (k + 1) > G:
            p = b.csel(p, O._c(b, G - 1, "gm1_%d" % k), O._c(b, 0, "z0_%d" % k), p, rel="gt", name="pc%d" % k)
        pw = b.add(b.shl(p, O._c(b, 1, "sh%d" % k), name="p2_%d" % k), O._c(b, lay["PAIRS"] // 4, "pb%d" % k), name="pw%d" % k)
        pv = b.fadd(b.load(a, pw, type=I, name="pv_%d" % k), O._cf(b, F32(0.0), "zz%d" % k), type=I, name="pvv%d" % k)
        pi = b.fadd(b.load(a, b.add(pw, O._c(b, 1, "o1_%d" % k), name="pwi%d" % k), type=I, name="pix_%d" % k),
                    O._cf(b, F32(0.0), "zi%d" % k), type=I, name="pii%d" % k)
        if v is None:
            v, i = pv, pi
        else:
            v, i = _combine(b, ir, v, i, pv, pi, "q%d" % k)
    v, i = _lane_reduce(b, ir, v, i, "r")
    # the token word at TOK + threadgroup (0 for this one-threadgroup launch): a program that reads no threadgroup
    # position authors as an unwitnessed class
    tg = b.builtin("threadgroup_position_in_grid", name="tg")
    b.store_at(c, b.add(tg, O._c(b, lay["TOK"] // 4, "tokw"), name="tokwi"), _to_int(b, ir, i, "tok"))
    b.ret()
    ir.verify(fn)
    return cc.compile_function(fn)


def gen_layout(V=92544, per_lane=12, d=2048, cap=272, R_X16=8192, GEN=12288):
    """Pass 2 as the generation step. Binding 3 is the layer region R: x fp16 [d] at R_X16 (the next layer 0
    input), and the generation state at GEN: the q0 word, then the token log int32 [cap] at GEN + 4 (the prompt
    pre-filled, 0xFFFFFFFF where the model chooses). Binding 2 is the embedding table fp16 [V][d]."""
    lay = argmax_layout(V, per_lane)
    return dict(lay, d=d, cap=cap, R_X16=R_X16, GEN=GEN, LOG=GEN + 4, region_bytes=GEN + 4 + 4 * cap)


def build_gen(lay):
    """Pass 2 plus the step: tok = log[q0 + 1] if the prompt fixed it, else the argmax; log[q0 + 1] = tok;
    x = embedding[tok]; q0 = q0 + 1 (clamped to cap - 1)."""
    from agxforge.g17 import cc, ir
    fn, b, a, bb, c = O._function()
    I = ir.I32
    lane = b.builtin("thread_index_in_simdgroup", name="lane")
    G = lay["G"]
    batched = lay.get("batched")
    v, i = None, None
    raw = []
    for k in range(-(-G // 32)):
        p = b.add(lane, O._c(b, 32 * k, "p%d" % k), name="pi%d" % k) if k else lane
        if 32 * (k + 1) > G:
            p = b.csel(p, O._c(b, G - 1, "gm1_%d" % k), O._c(b, 0, "z0_%d" % k), p, rel="gt", name="pc%d" % k)
        pw = b.add(b.shl(p, O._c(b, 1, "sh%d" % k), name="p2_%d" % k), O._c(b, lay["PAIRS"] // 4, "pb%d" % k), name="pw%d" % k)
        if batched:
            raw.append((b.load(a, pw, type=I, name="pv_%d" % k),
                        b.load(a, b.add(pw, O._c(b, 1, "o1_%d" % k), name="pwi%d" % k), type=I, name="pix_%d" % k)))
            continue
        pv = b.fadd(b.load(a, pw, type=I, name="pv_%d" % k), O._cf(b, F32(0.0), "zz%d" % k), type=I, name="pvv%d" % k)
        pi = b.fadd(b.load(a, b.add(pw, O._c(b, 1, "o1_%d" % k), name="pwi%d" % k), type=I, name="pix_%d" % k),
                    O._cf(b, F32(0.0), "zi%d" % k), type=I, name="pii%d" % k)
        if v is None:
            v, i = pv, pi
        else:
            v, i = _combine(b, ir, v, i, pv, pi, "q%d" % k)
    if batched:
        # BATCHED (MM 25.144.7): the pair loads all issued first; the combines keep their order
        for k, (lv, li_) in enumerate(raw):
            pv = b.fadd(lv, O._cf(b, F32(0.0), "zz%d" % k), type=I, name="pvv%d" % k)
            pi = b.fadd(li_, O._cf(b, F32(0.0), "zi%d" % k), type=I, name="pii%d" % k)
            v, i = (pv, pi) if v is None else _combine(b, ir, v, i, pv, pi, "q%d" % k)
    v, i = _lane_reduce(b, ir, v, i, "r")
    targ = _to_int(b, ir, i, "tok")
    tg = b.builtin("threadgroup_position_in_grid", name="tg")          # 0; read so the program authors
    q0 = b.load(c, b.add(tg, O._c(b, lay["GEN"] // 4, "q0w"), name="q0i"), type=I, name="q0")
    capm1 = O._c(b, lay["cap"] - 1, "capm1")
    nxt = b.add(q0, O._c(b, 1, "one"), name="nxt")
    nxt = b.csel(nxt, capm1, capm1, nxt, rel="gt", name="nxtc")
    li = b.add(nxt, O._c(b, lay["LOG"] // 4, "logb"), name="li")
    forced = b.add(b.load(c, li, type=I, name="forced_ld"), O._c(b, 0, "fz"), name="forced")
    free = b.icmp(forced, O._c(b, 0xFFFFFFFF, "unset"), "eq", name="free")
    tok = b.csel(free, O._c(b, 0, "z_free"), targ, forced, rel="gt", name="tokc")
    b.store_at(c, li, tok)
    b.store_at(c, b.add(tg, O._c(b, lay["GEN"] // 4, "q0w2"), name="q0i2"), nxt)
    if batched:
        # the embedding row copied as 32-bit WORDS (two fp16 each, the same bits), in batches of 16 loads issued
        # before their stores: one half load and store at a time paid a memory latency 64 times (MM 25.144.7)
        if lay["d"] % 64 or lay["R_X16"] % 4:
            raise ValueError("gen batched: d a multiple of 64 and x word-aligned")
        wrow = b.mul(tok, O._c(b, lay["d"] // 2, "dw"), name="wrow")
        ew = b.add(wrow, lane, name="ew")
        xw = b.add(lane, O._c(b, lay["R_X16"] // 4, "xw"), name="xw0")
        nw = lay["d"] // 64
        for k0 in range(0, nw, 16):
            ks = range(k0, min(nw, k0 + 16))
            ws = [b.load(bb, b.add(ew, O._c(b, 32 * k, "ewo%d" % k), name="ewi%d" % k) if k else ew, type=I,
                         name="ewd%d" % k) for k in ks]
            for k, w in zip(ks, ws):
                b.store_at(c, b.add(xw, O._c(b, 32 * k, "xwo%d" % k), name="xwi%d" % k) if k else xw,
                           b.add(w, O._c(b, 0, "ewz%d" % k), name="ewv%d" % k))
        b.ret()
        ir.verify(fn)
        return cc.compile_function(fn)
    row = b.mul(tok, O._c(b, lay["d"], "d"), name="row")
    eb = b.add(row, lane, name="eb")
    xb = b.add(lane, O._c(b, lay["R_X16"] // 2, "xb"), name="xb0")
    for k in range(lay["d"] // 32):
        h = b.load(bb, b.add(eb, O._c(b, 32 * k, "eo%d" % k), name="ei%d" % k) if k else eb, width="half", name="eh%d" % k)
        hv = b.f32_to_f16_rte(b.f16_to_f32(h, name="ew%d" % k), name="en%d" % k)      # exact: through an ALU
        b.store_at(c, b.add(xb, O._c(b, 32 * k, "xo%d" % k), name="xi%d" % k) if k else xb, hv, width="half")
    b.ret()
    ir.verify(fn)
    return cc.compile_function(fn)


def _align(v, a=256):
    return -(-v // a) * a


def gen_batch_layout(nb, V=92544, per_lane=12, d=2048, cap=272):
    """BATCHED GENERATION (MM 25.144.3), vector-major like the multi-vector qmv. Pass 1 is the SAME program launched
    over nb contiguous logit rows (nb G threadgroups): its indices are global, so row b's pairs land at
    [PAIRS + 8 G b, PAIRS + 8 G (b + 1)) and ties still resolve to the row's first maximum. The step runs nb
    threadgroups; sequence b's state lives in its own block of SS bytes at GEN + SS b: the q0 word, then the token log
    int32 [cap]. Its next-layer x (fp16 [d]) is at R_X16 + 2 d b, R_X16 = the batched w2 residual's RES (after nb fp32
    h rows), so the step writes where the batched layer reads."""
    lay = argmax_layout(V, per_lane)
    if nb * V >= (1 << 23):
        raise ValueError("gen batch: global logit indices must stay exact floats (below 2^23)")
    R_X16 = _align(4 * d * nb)
    GEN = _align(R_X16 + 2 * d * nb)
    SS = _align(4 + 4 * cap)
    return dict(lay, batch=nb, d=d, cap=cap, R_X16=R_X16, GEN=GEN, SS=SS, LOG=GEN + 4, pairs_bytes=8 * lay["G"] * nb,
                region_bytes=GEN + SS * nb)


def build_gen_batch(lay):
    """build_gen for nb sequences: threadgroup b reduces row b's pairs (the same lane order and butterflies), takes the
    token as the global index minus b V, and runs sequence b's step on its own state block and x row."""
    from agxforge.g17 import cc, ir
    fn, b, a, bb, c = O._function()
    I = ir.I32
    lane = b.builtin("thread_index_in_simdgroup", name="lane")
    tg = b.builtin("threadgroup_position_in_grid", name="tg")
    G, V, d, SS = lay["G"], lay["V"], lay["d"], lay["SS"]
    pbase = b.add(b.mul(tg, O._c(b, 2 * G, "g2"), name="pbo"), O._c(b, lay["PAIRS"] // 4, "pb"), name="pbase")
    v, i = None, None
    for k in range(-(-G // 32)):
        p = b.add(lane, O._c(b, 32 * k, "p%d" % k), name="pi%d" % k) if k else lane
        if 32 * (k + 1) > G:
            p = b.csel(p, O._c(b, G - 1, "gm1_%d" % k), O._c(b, 0, "z0_%d" % k), p, rel="gt", name="pc%d" % k)
        pw = b.add(b.shl(p, O._c(b, 1, "sh%d" % k), name="p2_%d" % k), pbase, name="pw%d" % k)
        pv = b.fadd(b.load(a, pw, type=I, name="pv_%d" % k), O._cf(b, F32(0.0), "zz%d" % k), type=I, name="pvv%d" % k)
        pi = b.fadd(b.load(a, b.add(pw, O._c(b, 1, "o1_%d" % k), name="pwi%d" % k), type=I, name="pix_%d" % k),
                    O._cf(b, F32(0.0), "zi%d" % k), type=I, name="pii%d" % k)
        if v is None:
            v, i = pv, pi
        else:
            v, i = _combine(b, ir, v, i, pv, pi, "q%d" % k)
    v, i = _lane_reduce(b, ir, v, i, "r")
    targ = b.sub(_to_int(b, ir, i, "tok"), b.mul(tg, O._c(b, V, "Vv"), name="tgV"), name="targ")
    sw = b.add(b.mul(tg, O._c(b, SS // 4, "ssw"), name="sso"), O._c(b, lay["GEN"] // 4, "q0w"), name="q0i")
    q0 = b.load(c, sw, type=I, name="q0")
    capm1 = O._c(b, lay["cap"] - 1, "capm1")
    nxt = b.add(q0, O._c(b, 1, "one"), name="nxt")
    nxt = b.csel(nxt, capm1, capm1, nxt, rel="gt", name="nxtc")
    li = b.add(b.add(nxt, sw, name="nq"), O._c(b, 1, "logo"), name="li")
    forced = b.add(b.load(c, li, type=I, name="forced_ld"), O._c(b, 0, "fz"), name="forced")
    free = b.icmp(forced, O._c(b, 0xFFFFFFFF, "unset"), "eq", name="free")
    tok = b.csel(free, O._c(b, 0, "z_free"), targ, forced, rel="gt", name="tokc")
    b.store_at(c, li, tok)
    b.store_at(c, sw, nxt)
    row = b.mul(tok, O._c(b, d, "d"), name="row")
    eb = b.add(row, lane, name="eb")
    xb = b.add(b.add(lane, b.mul(tg, O._c(b, d, "xrow"), name="xro"), name="xl"), O._c(b, lay["R_X16"] // 2, "xb"), name="xb0")
    for k in range(d // 32):
        h = b.load(bb, b.add(eb, O._c(b, 32 * k, "eo%d" % k), name="ei%d" % k) if k else eb, width="half", name="eh%d" % k)
        hv = b.f32_to_f16_rte(b.f16_to_f32(h, name="ew%d" % k), name="en%d" % k)
        b.store_at(c, b.add(xb, O._c(b, 32 * k, "xo%d" % k), name="xi%d" % k) if k else xb, hv, width="half")
    b.ret()
    ir.verify(fn)
    return cc.compile_function(fn)


def argmax_reference(logits):
    return int(np.argmax(np.asarray(logits, F32)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("check",))
    a = ap.parse_args(argv)
    lay = argmax_layout()
    print("pass1 %d bytes, pass2 %d bytes, G %d" % (len(build_pass1(lay).code), len(build_pass2(lay).code), lay["G"]))


if __name__ == "__main__":
    main()
