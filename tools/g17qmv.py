#!/usr/bin/env python3
"""Quantized matrix-vector product (qmv) for one-token decode, on MLX's affine weight format.

At decode the activation is ONE row. The tensor MMA path pads it to 16, so 15 rows are wasted, and each
threadgroup's K chain bounds the time (MM 25.124.6). A decode projection is a memory-bound GEMV, and with
4- or 8-bit weights the dequantize-and-accumulate ALU work per weight stays under the memory floor. So this
is a scalar-ALU kernel, as MLX's own qmv is:

    y[n] = sum_k x[k] * (scale[n, g] * q[n, k] + bias[n, g]),  g = k // group

WEIGHT FORMAT (mlx_lm.convert -q, affine): weight uint32 [N, K * bits / 32], element k of row n in bits
[bits * (k % (32 / bits)), ...) of word k // (32 / bits), low bits first; scales and biases bf16 [N, K / group].

MAPPING: G = N / R threadgroups of one simdgroup; threadgroup t computes rows [R t, R t + R). Lane l owns
the contiguous chunk k in [l K / 32, (l + 1) K / 32). A counted loop visits that chunk one packed word per
trip (32 / bits elements):
    - the trip's x values are loaded once (fp32) and summed in order into sx;
    - per row: t = q0 x0; t = t + q1 x1; ... (fmul and fadd, each rounded: no fma, so the host reference
      reproduces every rounding); then acc = (acc + s t) + b sx, with the group's scale and bias.
Then acc is summed across the 32 lanes by the measured butterfly (masks 1, 8, 2, 4, 16) and stored.
The order is defined here; qmv_reference replays it exactly in fp32.

TRANSPORT: the common worker admits tensor programs, so each threadgroup first runs decodeops' carrier (one
16 x 16 x 16 MMA into its own region), checked bit for bit as a positive control.

    python3 tools/g17qmv.py check [--bits 4] [--N 2048] [--K 2048] [--rows 4]   build, dispatch, compare
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import g17decodeops as O  # noqa: E402

F32 = np.float32
MASKS = (1, 8, 2, 4, 16)          # tensorreduce.ROW_BUTTERFLY_MASKS + COLUMN_BUTTERFLY_MASKS


def _align(v, a=256):
    return -(-v // a) * a


def big_transport(G, a_bytes, b_bytes, c_bytes):
    """Buffers for a program whose weights exceed decodeops' K <= 256 transport: the manifest describes a
    K-loop GEMM (admitted at K <= 4096, N <= 16384) with row groups x column groups = G threadgroups of 32
    lanes, so the worker launches exactly the kernel's grid. Only the buffer sizes and the launch matter;
    the program is not a GEMM."""
    for gn in (16, 8, 4, 2, 1):
        if G % gn == 0:
            break
    tg = G // gn
    Nv = 256 * gn if gn > 1 else 256
    unit = 16 * tg
    M = max(unit, -(-(-(-c_bytes // (4 * Nv))) // unit) * unit)
    if M // tg > 1024:
        raise ValueError("big transport: C needs %d rows per row group" % (M // tg))
    K = max(16, -(-max(-(-a_bytes // (2 * M)), -(-b_bytes // (2 * Nv))) // 16) * 16)
    if K > 4096:
        raise ValueError("big transport: B needs K %d > 4096" % K)
    view = dict(M=M, N=Nv, K=K, a="half", b="half", simdgroups=1, threadgroups=tg, grid_n=gn, split_k=1,
                kloop=True, epilogue=[], stages=[], accumulate=False, saturate=False)
    return dict(M=M, N=Nv, K=K, a_bytes=M * K * 2, b_bytes=K * Nv * 2, c_bytes=M * Nv * 4, view=view)


def qmv_layout(N, K, bits=4, group=64, rows=4, hoist=False, premask=False, fma=False, nocarrier=False):
    if bits not in (4, 8):
        raise ValueError("qmv: 4- or 8-bit weights")
    per_word = 32 // bits
    if K % (32 * per_word) or (K // 32) % per_word or group % per_word or K % group:
        raise ValueError("qmv: K must give each lane whole packed words and whole groups")
    if N % rows:
        raise ValueError("qmv: N must be a multiple of rows")
    G = N // rows
    # THE CARRIER IS BUILT FOR AT MOST 256 THREADGROUPS (its tile index is an 8-bit mask); a larger launch
    # repeats tiles t mod 256 with identical values. The carrier region is checked word for word.
    CG = min(G, 256)
    car = O._carrier_bytes(CG)
    X = car                                   # buffer 1: x, fp32 [K]
    W = _align(2 * O.CARRIER * O.CARRIER)     # buffer 2: after the carrier's B tile
    if nocarrier:
        # a directly bound pipeline (decodegen) has no carrier: every stream starts at offset 0 of its buffer
        X = W = car = 0
    words = K // per_word
    S = _align(W + 4 * N * words)             # scales bf16 [N, K/group]
    B = _align(S + 2 * N * (K // group))      # biases bf16 [N, K/group]
    OUT = car                                 # buffer 3: y, fp32 [N]
    # the manifest's view (the common worker's validator) admits at most 256 threadgroups; a larger launch is
    # given by the harness plan (projwarm / decodewarm), and correctness is checked from its dumped output
    need = (X + 4 * K, B + 2 * N * (K // group), OUT + 4 * N)
    try:
        t = big_transport(min(G, 256), *need)
    except ValueError:
        if not nocarrier:
            raise
        # A DIRECTLY BOUND PIPELINE HAS NO ADMISSION VIEW TO SATISFY: the view describes the largest K-loop GEMM the
        # validator admits, and the buffers are sized from the data (the lm_head block's reason, at 2 x 8192 rows)
        t = big_transport(min(G, 256), need[0], 4096 * 4096 * 2, need[2])
        t = dict(t, b_bytes=_align(need[1]), view_partial=True)
    extra = dict(nocarrier=True) if nocarrier else {}
    return dict(op="qmv", carrier_groups=CG, hoist=hoist, **extra, premask=premask, fma=fma, Nout=N, Kq=K, bits=bits, group=group, rows=rows, groups=G, per_word=per_word,
                X=X, W=W, S=S, B=B, OUT=OUT, **t)


def with_residual(lay, res):
    """Fold a residual add into the epilogue. Binding 3 holds one per-layer region [h fp32 [N] at OUT = 0 |
    x fp16 [N] at RES]: "add16" (wo + residual1) reads x and writes h = fp32(dot + x); "add32_to16" (w2 +
    residual2) reads h and writes the next layer's x = fp16_rne(dot + h) into the same x slot."""
    if res not in ("add16", "add32_to16"):
        raise ValueError(res)
    N = lay["Nout"]
    RES = _align(lay["OUT"] + 4 * N)
    return dict(lay, res=res, RES=RES, c_bytes=max(lay["c_bytes"], _align(RES + 2 * N)))


def with_next_norm(lay, XG, PSO):
    """THE NEXT RMSNORM, SPLIT ACROSS TWO KERNELS (MM 25.139.8): W (x r g) = r W (x g). This residual qmv also
    writes xg = fp16_rne(v g) at binding 3 [XG] (g the next norm's fp16 gain at binding 2 [GN], v the norm's
    input: h for the ffn norm, the fp16 x for the attention norm) and its threadgroup's sum of squares of v at
    binding 3 [PSO + group]; the consumer scales its rows by r from those partials (with_rnorm)."""
    N, K = lay["Nout"], lay["Kq"]
    GN = _align(lay["B"] + 2 * N * (K // lay["group"]))
    return dict(lay, next_norm=True, XG=XG, PSO=PSO, GN=GN, b_bytes=max(lay["b_bytes"], _align(GN + 2 * N)),
                c_bytes=max(lay["c_bytes"], _align(XG + 2 * N), _align(PSO + 4 * lay["groups"])))


def with_rnorm(lay, X, PS, NP, eps=1e-5):
    """The consumer half: x is the producer's xg (fp16) at binding 1 [X]; r = rsqrt_rn(mean + eps) from the NP
    partials at binding 1 [PS] (lane l sums l, l + 32, ... ascending, then the row and column butterflies);
    every output row is scaled by r (before the SwiGLU when fused)."""
    if NP % 32:
        raise ValueError("rnorm: whole lanes of partials")
    return dict(lay, x16=True, rnorm=True, X=X, PS=PS, NP=NP, eps=eps,
                a_bytes=max(lay["a_bytes"], _align(X + 2 * lay["Kq"]), _align(PS + 4 * NP)))


def next_norm_reference(lay, v, gain):
    """The producer's extra outputs from the next norm's input v (fp32 values; the fp16 x widened for the
    attention norm): xg fp16 and the per-threadgroup partials (rows in order, fmul then fadd)."""
    v = np.asarray(v, F32)
    xg = (v * np.asarray(gain, np.float16).astype(F32)).astype(F32).astype(np.float16)
    rows = v.reshape(-1, lay["rows"])
    part = (rows[:, 0] * rows[:, 0]).astype(F32)
    for r in range(1, lay["rows"]):
        part = (part + (rows[:, r] * rows[:, r]).astype(F32)).astype(F32)
    return xg, part


def rnorm_reference(parts, K, eps=1e-5):
    """The consumer's r from the partials (lane chains, then the tensorreduce butterflies)."""
    import g17decodestep as D_
    from agxforge.g17 import tensorreduce as TR
    p = np.asarray(parts, F32).reshape(-1, 32)
    loc = p[0].copy()
    for i in range(1, p.shape[0]):
        loc = (loc + p[i]).astype(F32)
    ss = TR.butterfly([float(x) for x in loc], TR.ROW_BUTTERFLY_MASKS, "sum")
    ss = F32(TR.butterfly(list(ss), TR.COLUMN_BUTTERFLY_MASKS, "sum")[0])
    mean = F32(ss * F32(1.0 / K))
    return F32(D_.rsqrt(np.asarray([F32(mean + F32(eps))], F32))[0])


def residual_reference(lay, y, x16=None, h32=None):
    """The epilogue's add on qmv2's fp32 output y: h = fp32(y + x) or out = fp16_rne(y + h)."""
    if lay["res"] == "add16":
        return (np.asarray(y, F32) + np.asarray(x16, np.float16).astype(F32)).astype(F32)
    return (np.asarray(y, F32) + np.asarray(h32, F32)).astype(F32).astype(np.float16)


def with_norm(lay, norm):
    """Fold an RMSNorm into the x load: buffer 1 holds the row v (norm "half" or "float") and buffer 2 the fp16
    gain g after B, at G. The reference is the qmv's on g17decodestep.rmsnorm(v, g)."""
    N = 2 * lay["ffn"] if lay.get("swiglu") else lay["Nout"]
    G = _align(lay["B"] + 2 * N * (lay["Kq"] // lay["group"]))
    b_bytes = max(lay["b_bytes"], _align(G + 2 * lay["Kq"]))
    a_bytes = max(lay["a_bytes"], _align(lay["X"] + 4 * lay["Kq"]))
    return dict(lay, norm=norm, G=G, b_bytes=b_bytes, a_bytes=a_bytes, eps=1e-5)


def with_batch(lay, nb):
    """BATCHED DECODE (MM 25.144.3): nb activation vectors against the SAME weights, VECTOR-MAJOR - sequence b sees
    exactly the single-vector layout shifted by a fixed stride: x_b fp32 at X + 4 K b; y_b (fp32; the act32 act for
    the fused FFN) at OUT + 4 n b, n = Nout (F when fused); with a residual, h_b at OUT + 4 N b and the fp16 x_b at
    RES + 2 N b, RES after all nb h rows. Apply it last (after with_residual)."""
    if nb < 2:
        raise ValueError("with_batch: two or more vectors")
    K = lay["Kq"]
    n = lay["ffn"] if lay.get("swiglu") else lay["Nout"]
    out = dict(lay, batch=nb, a_bytes=max(lay["a_bytes"], _align(lay["X"] + 4 * K * nb)))
    if lay.get("res"):
        N = lay["Nout"]
        RES = _align(lay["OUT"] + 4 * N * nb)
        out.update(RES=RES, c_bytes=max(lay["c_bytes"], _align(RES + 2 * N * nb)))
    else:
        width = 4 if (not lay.get("swiglu") or lay.get("act32")) else 2
        out["c_bytes"] = max(lay["c_bytes"], _align(lay["OUT"] + width * n * nb))
    return out


def qmv_batch_reference(lay, xs, q, s16, b16):
    """[nb, n]: each vector's single-vector value (the batched kernel computes each vector in exactly that order)."""
    single = dict(lay)
    single.pop("batch", None)
    ref = qmv_dq_reference if lay.get("dequant_once") else qmv2_ksplit_reference if lay.get("ksplit") else qmv2_reference_fast
    return np.stack([ref(single, x, q, s16, b16) for x in xs])


def qmv_swiglu_layout(F, K, bits=4, rows=4, nocarrier=True):
    """w1 and w3 fused with the SwiGLU: W, S and B are laid out as ONE qmv of 2F rows (gate rows then up
    rows); the launch is F / rows threadgroups; y is act, fp16 [F], at OUT."""
    lay = qmv_layout(2 * F, K, bits=bits, rows=rows, nocarrier=nocarrier)
    G = F // rows
    return dict(lay, swiglu=True, ffn=F, groups=G, carrier_groups=min(G, 256))


def qmv_swiglu_reference(lay, x, q, s16, b16):
    """act = fp16(silu(g) u), g and u the two halves of qmv2's fp32 output (g17decodestep's silu)."""
    import g17decodestep as D
    y = (qmv_dq_reference if lay.get("dequant_once") else qmv2_ksplit_reference if lay.get("ksplit") else qmv2_reference_fast)(lay, x, q, s16, b16)
    F = lay["ffn"]
    return D.narrow(D.fmul(D.silu(y[:F]), y[F:])).astype(np.float16)


def build_qmv(lay, fault=None):
    """The kernel. fault="skip_bias" (the control) drops the bias term."""
    from agxforge.g17 import cc, ir, tensorreduce as TR
    N, K, bits, grp, R = lay["Nout"], lay["Kq"], lay["bits"], lay["group"], lay["rows"]
    pw = lay["per_word"]
    words = K // pw
    lane_words = words // 32                  # words per lane (trips)
    fn, b, a, bb, c = O._function()
    O._carrier(b, a, bb, c, dict(lay, groups=lay["carrier_groups"]))
    lane = b.builtin("thread_index_in_simdgroup", name="lane")
    group = b.builtin("threadgroup_position_in_grid", name="group")
    mask = O._c(b, (1 << bits) - 1, "qmask")
    # per-lane bases
    xw0 = b.add(b.mul(lane, O._c(b, lane_words * pw, "lane_elems"), name="lx"), O._c(b, lay["X"] // 4, "xbase"), name="xw0")
    row0 = b.mul(group, O._c(b, R, "rows"), name="row0")
    # ONE base per stream (row r adds the constant r * words / r * gpr inside the loop): per-row bases held
    # across the loop ran a 32-row threadgroup out of registers
    wbase0 = b.add(b.mul(row0, O._c(b, words, "words"), name="nw"),
                   b.add(b.mul(lane, O._c(b, lane_words, "lw"), name="lwo"), O._c(b, lay["W"] // 4, "wb"), name="lwb"),
                   name="w0")
    gpr = K // grp                            # groups per row
    sbase0 = b.add(b.mul(row0, O._c(b, gpr, "gpr"), name="sng"), O._c(b, lay["S"] // 2, "sb"), name="s0")
    bdiff = O._c(b, (lay["B"] - lay["S"]) // 2, "bdiff")
    wpg = grp // pw                           # words per group
    # counted loop over this lane's words
    hdr, post = fn.block("qmv_loop"), fn.block("qmv_done")
    k0 = O._c(b, 0, "k0")
    zeros = [O._cf(b, F32(0.0), "fzero%d" % r) for r in range(R)]
    b.br(hdr)
    b.at(hdr)
    k = b.phi(k0, name="k")
    # one zero constant PER accumulator: phis sharing an initial value coalesce into one register
    accs = [b.phi(zeros[r], type=ir.F32, name="acc%d" % r) for r in range(R)]
    # x for this word: element index lane*lane_elems + k*pw + j
    xk = b.add(xw0, b.mul(k, O._c(b, pw, "pw"), name="kx"), name="xk")
    xs = [b.load(a, b.add(xk, O._c(b, j, "xj%d" % j), name="xi%d" % j), type=ir.I32, name="x%d" % j) for j in range(pw)]
    sx = xs[0]
    for j in range(1, pw):
        sx = b.fadd(sx, xs[j], type=ir.F32, name="sx%d" % j)
    premask, use_fma = lay.get("premask", False), lay.get("fma", False)
    if premask:
        # x_j / 2^(bits j), once per trip and shared by every row: (q 2^(bits j)) (x_j 2^-(bits j)) = q x_j exactly
        xq = [xs[0]] + [b.fmul(xs[j], O._cf(b, F32(2.0 ** (-bits * j)), "ps%d" % j), type=ir.F32, name="xq%d" % j)
                        for j in range(1, pw)]
    # the group index of this word, lane-relative: (lane*lane_words + k) // wpg
    gw = b.add(b.mul(lane, O._c(b, lane_words, "lw2"), name="lwk"), k, name="gw")
    gidx = b.shr(gw, O._c(b, wpg.bit_length() - 1, "gsh"), name="gidx")
    wk = b.add(wbase0, k, name="wk")
    sg = b.add(sbase0, gidx, name="sg")
    # LOADS FIRST (lay["hoist"]): every row's word, scale and bias for this trip is issued before any
    # arithmetic, so no row waits out its own load round trip (the K-loop and norm lesson, 25.137.2)
    hoist = lay.get("hoist", False)
    if hoist:
        W_ = [b.load(bb, b.add(wk, O._c(b, r * words, "rw%d" % r), name="wi%d" % r) if r else wk, type=ir.I32,
                     name="w%d" % r) for r in range(R)]
        SI_ = [b.add(sg, O._c(b, r * gpr, "rg%d" % r), name="si%d" % r) if r else sg for r in range(R)]
        S_ = [b.load(bb, SI_[r], width="half", name="sh%d" % r) for r in range(R)]
        B_ = ([b.load(bb, b.add(SI_[r], bdiff, name="bi%d" % r), width="half", name="bh%d" % r) for r in range(R)]
              if fault != "skip_bias" else None)
    newacc = []
    for r in range(R):
        w = W_[r] if hoist else b.load(bb, b.add(wk, O._c(b, r * words, "rw%d" % r), name="wi%d" % r) if r else wk,
                                       type=ir.I32, name="w%d" % r)
        t = None
        for j in range(pw):
            if premask:
                q = getattr(b, "and")(w, O._c(b, ((1 << bits) - 1) << (bits * j), "m%d_%d" % (r, j)) if j else mask,
                                      name="q%d_%d" % (r, j))
                xj = xq[j]
            else:
                q = getattr(b, "and")(b.shr(w, O._c(b, bits * j, "sh%d_%d" % (r, j)), name="ws%d_%d" % (r, j)) if j else w,
                                      mask, name="q%d_%d" % (r, j))
                xj = xs[j]
            qf = b.u32_to_f32(q, name="qf%d_%d" % (r, j))
            if t is None:
                t = b.fmul(qf, xj, type=ir.F32, name="p%d_%d" % (r, j))
            elif use_fma:
                t = b.fma(qf, xj, t, name="t%d_%d" % (r, j))
            else:
                t = b.fadd(t, b.fmul(qf, xj, type=ir.F32, name="p%d_%d" % (r, j)), type=ir.F32, name="t%d_%d" % (r, j))
        si = SI_[r] if hoist else (b.add(sg, O._c(b, r * gpr, "rg%d" % r), name="si%d" % r) if r else sg)
        s = b.shl(S_[r] if hoist else b.load(bb, si, width="half", name="sh%d" % r), O._c(b, 16, "s16_%d" % r),
                  name="s%d" % r)
        u = b.fmul(s, t, type=ir.F32, name="u%d" % r)
        acc = b.fadd(accs[r], u, type=ir.F32, name="au%d" % r)
        if fault != "skip_bias":
            bi = b.shl(B_[r] if hoist else b.load(bb, b.add(si, bdiff, name="bi%d" % r), width="half", name="bh%d" % r),
                       O._c(b, 16, "b16_%d" % r), name="bv%d" % r)
            v = b.fmul(bi, sx, type=ir.F32, name="v%d" % r)
            acc = b.fadd(acc, v, type=ir.F32, name="av%d" % r)
        newacc.append(acc)
    kn = b.add(k, ir.Imm(1), name="k_next")
    ir.Builder.phi_latch(k, kn)
    for r in range(R):
        ir.Builder.phi_latch(accs[r], newacc[r])
    b.br_cond(b.cmp(kn, lane_words, "lt", name="more"), hdr, post)
    b.at(post)
    for r in range(R):
        v = TR.emit_butterfly(b, newacc[r], TR.ROW_BUTTERFLY_MASKS, operation="sum")
        v = TR.emit_butterfly(b, v, TR.COLUMN_BUTTERFLY_MASKS, operation="sum")
        b.store_at(c, b.add(row0, O._c(b, r + lay["OUT"] // 4, "o%d" % r), name="oi%d" % r), v)
    b.ret()
    ir.verify(fn)
    return cc.compile_function(fn)


class _XI:
    """xi[e], formed on first read."""
    def __init__(self, f): self.f = f
    def __getitem__(self, e): return self.f(e)


def build_qmv2(lay, _into=None):
    """qmv with WIDE trips: each trip covers wpt = lay["wpt"] packed words per row (one vector load of 2 or 4
    words), its wpt * per_word activations (vector loads), and applies the row's scale and bias ONCE per
    trip. All loads of a trip are issued first; nibbles are masked in place against pre-scaled x and
    accumulated with fma. Order (qmv2_reference): per trip, per row, t = q0 x0 then t = fma(q_e, x_e, t)
    over the trip's elements in order; acc = (acc + s t) + b sx, sx the trip's x summed in order."""
    if lay.get("wide"):
        return build_qmv_wide(lay)
    if lay.get("batch", 1) > 1 or lay.get("dequant_once"):
        # the dequantize-once order exists only in the multi-vector builder; at one vector (no "batch") it is the
        # single-vector kernel in that order, so a batched dq graph's single runs share its fp32 order
        return build_qmv2_batch(lay)
    from agxforge.g17 import cc, ir, tensorreduce as TR
    N, K, bits, grp, R = lay["Nout"], lay["Kq"], lay["bits"], lay["group"], lay["rows"]
    pw, wpt = lay["per_word"], lay["wpt"]
    # SWIGLU FUSED (lay["swiglu"]): W holds the gate rows [0, F) then the up rows [F, 2F); a simdgroup computes
    # R gate rows and the R up rows at the same positions, and writes act = fp16(silu(g) u) - one dispatch for
    # w1, w3 and the SwiGLU, whose fp16 act is w2's x. ROFF maps the computed row to its W row.
    R0 = R
    if lay.get("swiglu"):
        F = lay["ffn"]
        ROFF = [r if r < R0 else F + r - R0 for r in range(2 * R0)]
    else:
        ROFF = list(range(R0))
    R = len(ROFF)
    words = K // pw
    lane_words = words // 32
    if lane_words % wpt or (grp // pw) % wpt or wpt not in (1, 2, 4):
        raise ValueError("qmv2: a trip must hold whole vector loads inside one group")
    trips = lane_words // wpt
    if lay.get("ksplit"):
        if lay.get("sgs") not in (2, 4, 8) or trips % lay["sgs"]:
            raise ValueError("ksplit: 2, 4 or 8 simdgroups dividing the trips")
        trips //= lay["sgs"]
    E = wpt * pw                                  # elements per trip
    if _into is not None:
        # ONE ARM OF A MULTI-OP PROGRAM (build_uber): emit into the caller's function from its current block,
        # labels prefixed, and branch to its exit instead of returning
        if lay.get("coop") or lay.get("last_norm") or lay.get("norm") or lay.get("hoist_consts"):
            raise ValueError("qmv2: an uber arm takes the plain nocarrier forms only")
        fn, b, a, bb, c = _into["fn"], _into["b"], _into["a"], _into["bb"], _into["c"]
    elif lay.get("coop"):
        # THE COOPERATIVE THREE-BINDING CLASS (MM 25.140.3): slot 0 written (rank 0, where the device atomics are
        # measured), slot 1 x, slot 2 the weights; 32-thread threadgroups with a scratchpad; SR 156 and 164 only
        c = ir.Buffer("C", 0, elem=ir.F32)
        a = ir.Buffer("A", 1, elem=ir.F16)
        bb = ir.Buffer("B", 2, elem=ir.F16)
        fn = ir.Function("tensor_gemm_generic_runtime_demo", [c, a, bb])
        fn.declare_threadgroup(lay["sgs"] * R if lay.get("ksplit") else 4, size=(32 * lay["sgs"] if lay.get("ksplit") else 32, 1, 1))
        b = ir.Builder(fn, fn.block("entry"))
    else:
        fn, b, a, bb, c = O._function()
    if not lay.get("nocarrier") and not lay.get("coop") and _into is None:
        # the carrier exists only for the common worker's tensor-program admission (ABI v5); a runtime that
        # binds the pipeline itself (the timing harnesses, a decode executor) needs none (MM 25.138)
        O._carrier(b, a, bb, c, dict(lay, groups=lay["carrier_groups"]))
    lane = b.builtin("thread_position_in_threadgroup" if lay.get("coop") else "thread_index_in_simdgroup", name="lane")
    if lay.get("ksplit"):
        if not lay.get("coop"):
            raise ValueError("ksplit: the cooperative three-binding class (threadgroup memory)")
        lane = getattr(b, "and")(lane, ir.Imm(31), name="lane31")
    group = b.builtin("threadgroup_position_in_grid", name="group")
    sgs = lay.get("sgs", 1)
    if sgs > 1:
        # SEVERAL SIMDGROUPS PER THREADGROUP (MLX's qmv runs 2): simdgroup s of threadgroup t takes the rows of
        # unit t sgs + s; the launch is the same threads in threadgroups of 32 sgs
        sgi = b.shr(b.builtin("thread_position_in_threadgroup", name="tpt"), O._c(b, 5, "sg_sh"), name="sgi")
        # SPLIT-K IN THE THREADGROUP (lay["ksplit"], MM 25.141.15): simdgroup s of S = sgs instead takes the
        # threadgroup's SAME rows over trips [s T/S, (s+1) T/S); the partials meet in threadgroup memory after the loop
        if not lay.get("ksplit"):
            group = b.add(b.mul(group, O._c(b, sgs, "sgs"), name="gsg"), sgi, name="unit")
    row0 = b.mul(group, O._c(b, R0, "rows"), name="row0")
    gpr = K // grp
    # vector-load indices count wpt-word units; scalar half loads count halves
    coal = lay.get("coalesced", False)
    # COALESCED with wpt words: in trip k lane l reads words 32 wpt k + wpt l .. + wpt - 1 (adjacent words per
    # lane, one contiguous 128 wpt-byte span per SIMD load)
    # CONTIGUOUS (default): lane l owns words [l lane_words, (l+1) lane_words). COALESCED: in trip k lane l
    # reads word 32 k + l, so one SIMD load is 128 contiguous bytes (and x likewise)
    wv0 = b.add(b.mul(row0, O._c(b, words, "wv_row"), name="wvr"),
                b.add((lane if wpt == 1 else b.mul(lane, O._c(b, wpt, "wv_lanew"), name="wvlw")) if coal
                      else b.mul(lane, O._c(b, lane_words, "wv_lane"), name="wvl"),
                      O._c(b, lay["W"] // 4, "wvb"), name="wvlb"),
                name="wv0")
    xv0 = b.add(b.mul(lane, O._c(b, wpt * pw if coal else lane_words * pw, "xv_lane"), name="xvl"),
                O._c(b, lay["X"] // (2 if (lay.get("x16") or lay.get("norm") == "half") else 4), "xvb"), name="xv0")
    s0 = b.add(b.mul(row0, O._c(b, gpr, "gpr"), name="sng"), O._c(b, lay["S"] // 2, "sb"), name="s0")
    bdiff = O._c(b, (lay["B"] - lay["S"]) // 2, "bdiff")
    tpg = (grp // pw) // wpt                      # trips per group
    lean = lay.get("lean", False)
    # invariant per-row bases cost 3 registers a row: only below 9 rows (32 rows ran out of registers)
    inv = lean and R <= 8
    if inv:
        # LOOP-INVARIANT per-row bases, formed once: inside the loop each row's word and scale index is ONE add
        WB = [b.add(wv0, O._c(b, ROFF[r] * words, "wrb%d" % r), name="wrow%d" % r) if r else wv0 for r in range(R)]
        SB = [b.add(s0, O._c(b, ROFF[r] * gpr, "srb%d" % r), name="srow%d" % r) if r else s0 for r in range(R)]
        BB = None if lay.get("ptr_addr") else [b.add(SB[r], bdiff, name="brow%d" % r) for r in range(R)]   # ptr_addr carries its own
    pparts = None
    if lay.get("rnorm"):
        # the producer's partials, loaded before the loop (their latency overlaps it); r is formed after it
        pparts = [b.load(a, b.add(lane, O._c(b, 32 * i + lay["PS"] // 4, "rpq%d" % i), name="rpi%d" % i), type=ir.I32,
                         name="rpv%d" % i) for i in range(lay["NP"] // 32)]
    norm = lay.get("norm")
    if norm:
        # RMSNORM FOLDED INTO THE x LOAD (lay["norm"] = "half" | "float": the row v's type). Every simdgroup
        # recomputes r = rsqrt_rn(mean(v^2) + eps) in g17decodestep.rmsnorm's order - lane l's ascending chain
        # over v[l + 32 i], the row then column butterflies, mean = ss (1/K) - and each x element is then
        # fp16_rne((v r) g), exactly the h the separate norm dispatch would have written.
        xunit = 2 if norm == "half" else 4
        def _v(idx, tag):
            if norm == "half":
                return b.f16_to_f32(b.load(a, idx, width="half", name=tag + "h"), name=tag)
            return b.load(a, idx, type=ir.I32, name=tag)
        local = None
        for i in range(K // 32):
            v = _v(b.add(lane, O._c(b, 32 * i + lay["X"] // xunit, "nq%d" % i), name="nqi%d" % i), "nv%d" % i)
            sq = b.fmul(v, v, type=ir.F32, name="nsq%d" % i)
            local = sq if local is None else b.fadd(local, sq, type=ir.F32, name="nacc%d" % i)
        ss = TR.emit_butterfly(b, local, TR.ROW_BUTTERFLY_MASKS, operation="sum")
        ss = TR.emit_butterfly(b, ss, TR.COLUMN_BUTTERFLY_MASKS, operation="sum")
        mean = b.fmul(ss, O._cf(b, F32(1.0 / K), "inv_d"), name="nmean")
        KN = O.emit_constants(b)
        rnorm = O.emit_rn(b, "rsqrt", b.fadd(mean, O._cf(b, F32(lay.get("eps", 1e-5)), "eps"), type=ir.I32, name="nvar"), KN, "nrs")
        gdiff = O._c(b, lay["G"] // 2 - lay["X"] // xunit, "gdiff")
    P = _into["prefix"] if _into is not None else ""
    hdr, post = fn.block(P + "qmv_loop"), fn.block(P + "qmv_done")
    k0 = O._c(b, 0, "k0")
    zeros = [O._cf(b, F32(0.0), "fzero%d" % r) for r in range(R)]
    scales = [O._cf(b, F32(2.0 ** (-bits * j)), "ps%d" % j) for j in range(1, pw)]
    MR = {}
    if lay.get("pool_masks") and not (lay.get("a16") and lay.get("coop")):
        raise ValueError("qmv2 pool_masks: the and16 extraction (a16) in the cooperative three-binding class, the "
                         "only class whose mask pool is witnessed (MM 25.141.16)")
    if lay.get("a16") and not lay.get("pool_masks"):
        # and16's masks above 8 bits, in the low half of loop-invariant registers
        for pos in range((32 // bits) // 2):
            m = ((1 << bits) - 1) << (bits * pos)
            if m > 0xFF:
                MR[m] = O._c(b, m, "hmask%d" % pos)
    if lean:
        # masks above 8 bits live in loop-invariant registers (and-immediate is an 8-bit slot)
        MK = [ir.Imm(((1 << bits) - 1) << (bits * j)) if ((1 << bits) - 1) << (bits * j) <= 0xFF
              else O._c(b, ((1 << bits) - 1) << (bits * j), "mask%d" % j) for j in range(pw)]
    if lay.get("ksplit"):
        koff = b.mul(sgi, O._c(b, trips, "ktrips"), name="koff")
    if lay.get("hi16_scales"):
        zero_hi = [O._c(b, 0, "hz%d" % i) for i in range(2 * R)]
    ptr = lay.get("ptr_addr", False)
    if ptr:
        # LOOP-CARRIED ADDRESSES (MM 25.141.3 item 4): the x vector index, each row's weight-vector index and each
        # row's scale index advance by a constant per trip instead of being re-formed from k (k * stride, + base,
        # >> shift, every trip). Their entry values - the split-K slice's first trip included - are formed once here.
        # The group index distributes over the trip term because 32 wpt words are whole groups:
        #     (k 32 wpt + l wpt) >> gsh == k ((32 wpt) >> gsh) + ((l wpt) >> gsh)
        wpg = grp // pw
        if not (coal and lay.get("xvec") and lay.get("vload") and wpt > 1 and inv and (32 * wpt) % wpg == 0
                and not norm and not lay.get("x16") and 16 * (E // 4 - 1) <= 252):
            raise ValueError("qmv2 ptr_addr: coalesced xvec + vload (wpt > 1) with invariant row bases, and 32 wpt "
                             "words a whole number of groups")
        wsh, gsh = wpt.bit_length() - 1, wpg.bit_length() - 1
        XSTEP, VSTEP, GSTEP = 8 * E, 32, (32 * wpt) >> gsh
        def _imm(v, tag):
            return ir.Imm(v) if v <= 0xFF else O._c(b, v, tag)
        def _plus_koff(v, step, tag):
            if not lay.get("ksplit"):
                return v
            return b.add(v, b.mul(koff, O._c(b, step, tag + "_ks"), name=tag + "_kso"), name=tag + "_k0")
        x4_0 = _plus_koff(b.shr(xv0, O._c(b, 2, "x4sh0"), name="x4b"), XSTEP, "x4e")
        lgrp = b.shr(b.mul(lane, O._c(b, wpt, "lwp0"), name="lwp0m"), O._c(b, gsh, "gsh0"), name="lgrp")
        vi_0 = [_plus_koff(b.shr(WB[r], O._c(b, wsh, "vsh0_%d" % r), name="vb%d" % r), VSTEP, "vie%d" % r) for r in range(R)]
        si_0 = [_plus_koff(b.add(SB[r], lgrp, name="sb%d" % r), GSTEP, "sie%d" % r) for r in range(R)]
    pf = lay.get("prefetch_w", False)
    if pf:
        # WEIGHT PREFETCH (MM 25.144.4): the weight stream one trip ahead of the compute. Each row's wpt-word vector is
        # a loop phi; trip 0 loads here, trip k+1 loads at the top of trip k. The last trip's extra load reads the next
        # words of the same buffer (the scale region follows the weights), and its value is never used.
        if not (coal and lay.get("vload") and wpt > 1 and inv and not ptr and not lay.get("pool_masks")):
            raise ValueError("qmv2 prefetch_w: coalesced vector weight loads (wpt > 1) with invariant row bases, "
                             "without ptr_addr or pool_masks")
        _vsh = O._c(b, wpt.bit_length() - 1, "pvsh")
        _wt = O._c(b, 32 * wpt, "pwtrip")
        def _wvec(r, t, tag):
            return b.load_vec_at(bb, b.shr(b.add(WB[r], b.mul(t, _wt, name=tag + "m%d" % r), name=tag + "a%d" % r),
                                           _vsh, name=tag + "i%d" % r), n=wpt, name=tag + "w%d_" % r)
        W0 = [_wvec(r, koff if lay.get("ksplit") else O._c(b, 0, "pt0"), "p0") for r in range(R)]
    b.br(hdr)
    b.at(hdr)
    k = b.phi(k0, name="k")
    kl = k                                             # the loop counter (its compare bound is the split trips)
    accs = [b.phi(zeros[r], type=ir.F32, name="acc%d" % r) for r in range(R)]
    if pf:
        WP = [[b.phi(W0[r][i], name="wp%d_%d" % (r, i)) for i in range(wpt)] for r in range(R)]
    hi16 = lay.get("hi16_scales", False)
    if hi16:
        # SCALES AND BIASES INTO HIGH HALVES (MM 25.141.3 item 3, Apple's form): each row's scale and bias is a loop
        # register whose low half is zeroed once before the loop and never written again; every trip's bf16 lands in
        # its high half (ir.load_hi16, tied), which IS the fp32 value - no shift per trip
        SP = [b.phi(zero_hi[r], name="sp%d" % r) for r in range(R)]
        BP = [b.phi(zero_hi[R + r], name="bp%d" % r) for r in range(R)]
    if ptr:
        XP = b.phi(x4_0, name="xp")
        VP = [b.phi(vi_0[r], name="vp%d" % r) for r in range(R)]
        IP = [b.phi(si_0[r], name="ip%d" % r) for r in range(R)]
    if lay.get("ksplit") and not ptr:
        k = b.add(k, koff, name="kk")
    if pf:
        WN = [_wvec(r, b.add(k, ir.Imm(1), name="pkn%d" % r), "pn") for r in range(R)]
    # ---- the trip's x (shared by every row), its ordered sum and its pre-scaled copies
    # SCALAR loads, issued before any arithmetic. Vector loads now work any number per program (the 14-byte
    # form, MM 25.139.2) and moved the kernel +-5 percent: what cuts the latency is fewer trips with many
    # independent loads in flight, not the vector form
    xidx = None if ptr else b.add(xv0, b.mul(k, O._c(b, 32 * E if coal else E, "xe_trip"), name="xk"), name="xi")
    # per-element indices only where a path reads them: the xvec path reads none, and 15 unread adds per trip stayed
    # in its loop (cc keeps an unread pure op; MM 25.141.3)
    _xi = {}
    def xi_(e):
        if e not in _xi:
            _xi[e] = b.add(xidx, O._c(b, e, "xo%d" % e), name="xie%d" % e) if e else xidx
        return _xi[e]
    xi = _XI(xi_)
    if lay.get("xi_eager"):
        # the old eager form, kept as the control for the dead-add measurement (MM 25.141.3)
        for e in range(E):
            xi[e]
    if norm:
        xs = []
        for e in range(E):
            v = _v(xi[e], "xv%d" % e)
            g = b.f16_to_f32(b.load(bb, b.add(xi[e], gdiff, name="xgi%d" % e), width="half", name="xgh%d" % e), name="xg%d" % e)
            y = b.fmul(b.fmul(v, rnorm, type=ir.I32, name="xvr%d" % e), g, type=ir.I32, name="xy%d" % e)
            xs.append(b.f16_to_f32(b.f32_to_f16_rte(y, name="xyh%d" % e), name="x%d" % e))
    elif lay.get("xvec"):
        # FP32 x BY VECTOR LOADS (MM 25.139.7): the lane's E consecutive fp32 x values are 16-byte aligned (coalesced
        # layout), so E/4 four-component loads fetch them with no per-element address and no conversion - what
        # Apple's compile of MLX's qmv does. Index units are 16 bytes.
        if not coal or E % 4 or lay["X"] % 16:
            raise ValueError("qmv2 xvec: coalesced, E a multiple of 4, X 16-byte aligned")
        xs = []
        x4 = XP if ptr else b.shr(xidx, O._c(b, 2, "x4sh"), name="x4i")
        for j in range(E // 4):
            if ptr:
                # the trip's later vector loads at a BYTE displacement off the one carried index (MM 25.141.17)
                xs += b.load_vec_at(a, x4, n=4, name="xv%d_" % j, offset_bytes=16 * j)
            else:
                xs += b.load_vec_at(a, b.add(x4, O._c(b, j, "x4o%d" % j), name="x4j%d" % j) if j else x4, n=4, name="xv%d_" % j)
    elif lay.get("x16"):
        # FP16 ACTIVATIONS (the model graph's rows): half loads, widened exactly to fp32; the arithmetic after
        # is unchanged, so the reference is qmv2_reference on the widened x
        xs = [b.f16_to_f32(b.load(a, xi[e], width="half", name="xh16_%d" % e), name="x%d" % e) for e in range(E)]
    else:
        xs = [b.load(a, xi[e], type=ir.I32, name="x%d" % e) for e in range(E)]
    if ptr or pf:
        # ptr_addr carries its indices; prefetch_w loads the next trip's weights from its own index (forming this trip's
        # here as well would leave two unread ops per trip, which cc keeps)
        kw = wk = None
    else:
        kw = b.mul(k, O._c(b, 32 * wpt, "wtrip32"), name="kw") if coal else (b.mul(k, O._c(b, wpt, "wtrip"), name="kw") if wpt > 1 else k)
        wk = b.add(wv0, kw, name="wk")
    # the group of this lane's words: word index / words-per-group (coalesced), trip / trips-per-group (contiguous)
    gidx = None if ptr else (b.shr(b.add(b.mul(k, O._c(b, 32 * wpt, "k32"), name="k32m"),
                        lane if wpt == 1 else b.mul(lane, O._c(b, wpt, "lwp"), name="lwpm"), name="gt"),
                  O._c(b, (grp // pw).bit_length() - 1, "gsh"), name="gidx") if coal else
            b.shr(b.add(b.mul(lane, O._c(b, trips, "lt"), name="ltk"), k, name="gt"),
                  O._c(b, tpg.bit_length() - 1, "gsh"), name="gidx"))
    sg = None if (inv or ptr) else b.add(s0, gidx, name="sg")
    if lay.get("sx_in"):
        # THE TRIP'S x SUM, READ INSTEAD OF FORMED (MM 25.152's x-sum hoist). sum(x) over the trip's E consecutive
        # values is the same for every row, so a producer can write it once per token at lay["SX"] (fp32 per trip of
        # E values, in the same left-to-right order) and each row reads one word instead of E-1 fadds. The trip's
        # element index is XP*4 (16-byte units), its chunk XP*4 / E.
        if not lay.get("xvec") or E % 4 or E & (E - 1):
            raise ValueError("qmv2 sx_in: the xvec form with E a power of two")
        if ptr:     # the carried index is in 16-byte units: chunk = XP * 4 / E
            ch = b.shr(XP, O._c(b, (E // 4).bit_length() - 1, "sxsh"), name="sxch") if E > 4 else XP
        else:       # the element index is a multiple of E: chunk = xidx / E
            ch = b.shr(xidx, O._c(b, E.bit_length() - 1, "sxsh"), name="sxch")
        sx = b.load(a, b.add(ch, O._c(b, lay["SX"] // 4, "sxo"), name="sxi"), type=ir.I32, name="sx")
    else:
        sx = xs[0]
        for e in range(1, E):
            sx = b.fadd(sx, xs[e], type=ir.F32, name="sx%d" % e)
    # per-position pre-scaled x only where a path reads it: the a16 path reads xq16 instead, and cc keeps an unread
    # pure op, so the unconditional list ran 12-14 dead fmuls per trip in every shipped loop (MM 25.141.17)
    xq = None if lay.get("a16") else [xs[e] if e % pw == 0 else b.fmul(xs[e], scales[e % pw - 1], type=ir.F32, name="xq%d" % e)
                                      for e in range(E)]
    chains = lay.get("chains", False)
    if chains and not lay.get("a16"):
        raise ValueError("qmv2 chains: the and16 extraction (a16) only")
    if lay.get("a16"):
        # the field's position within what and16 reads: its byte (w and w >> 8), or its half with pool masks
        fb = 16 // bits if lay.get("pool_masks") else 8 // bits
        # POSITION CHAINS (lay["chains"], MM 25.144.4): no pre-scaled x at all - the masked field q 2^(bits pos)
        # meets the raw x in one fma chain per position, and the chains fold once per trip, t = t0 then
        # t = fma(t_pos, 2^-(bits pos), t) (qmv2_chain_reference)
        xq16 = (list(xs) if chains else
                [xs[e] if (e % pw) % fb == 0 else b.fmul(xs[e], scales[(e % pw) % fb - 1], type=ir.F32, name="xh%d" % e)
                 for e in range(E)])
    # ---- rows in CHUNKS of 8: a chunk's loads are all issued, then its arithmetic (interleaved across the
    # chunk's rows when lay["interleave"]); only one chunk's words, scales and biases are live at a time.
    # Each row's operations keep their order, so the values are unchanged.
    newacc = [None] * R
    CH = lay.get("chunk", 8)
    for c0 in range(0, R, CH):
        rows_c = list(range(c0, min(R, c0 + CH)))
        Wv, Sv, Bv = {}, {}, {}
        for r in rows_c:
            base = None if (ptr or pf) else ((b.add(WB[r], kw, name="wi%d" % r) if r else wk) if inv else
                                     (b.add(wk, O._c(b, ROFF[r] * words, "rw%d" % r), name="wi%d" % r) if r else wk))
            if lay.get("vload") and wpt > 1:
                # ONE VECTOR LOAD of the row's wpt adjacent words (coalesced: lane l's words start at a multiple
                # of wpt), indexed in wpt-word units - 8 or 16 bytes per lane per row, MLX's access width
                if not coal or lay["W"] % (4 * wpt) or words % wpt:
                    raise ValueError("qmv2 vload: coalesced, wpt-aligned words only")
                if pf:
                    Wv[r] = WP[r]
                else:
                    vi = VP[r] if ptr else b.shr(base, O._c(b, wpt.bit_length() - 1, "vsh"), name="vi%d" % r)
                    Wv[r] = b.load_vec_at(bb, vi, n=wpt, name="w%d_" % r)
            else:
                Wv[r] = [b.load(bb, b.add(base, O._c(b, i, "wo%d_%d" % (r, i)), name="wj%d_%d" % (r, i)) if i else base,
                                type=ir.I32, name="w%d_%d" % (r, i)) for i in range(wpt)]
        for r in rows_c:
            if ptr:
                si = IP[r]
                bi = b.add(si, bdiff, name="bi%d" % r)
            elif inv:
                si, bi = b.add(SB[r], gidx, name="si%d" % r), b.add(BB[r], gidx, name="bi%d" % r)
            else:
                si = b.add(sg, O._c(b, ROFF[r] * gpr, "rg%d" % r), name="si%d" % r) if r else sg
                bi = b.add(si, bdiff, name="bi%d" % r)
            if hi16:
                Sv[r] = b.load_hi16(SP[r], bb, si, name="sh%d" % r)
                Bv[r] = b.load_hi16(BP[r], bb, bi, name="bh%d" % r)
            else:
                Sv[r] = b.load(bb, si, width="half", name="sh%d" % r)
                Bv[r] = b.load(bb, bi, width="half", name="bh%d" % r)
        ts = {r: None for r in rows_c}
        tc = {}                                        # chains: (row, position) -> that position's running sum
        order = ([(r, e) for e in range(E) for r in rows_c] if lay.get("interleave")
                 else [(r, e) for r in rows_c for e in range(E)])
        shand, a16 = lay.get("shand", False), lay.get("a16", False)
        W8 = {}
        for r, e in order:
            w, jj = Wv[r][e // pw], e % pw
            if lay.get("ablate_nodeq"):
                # TIMING ABLATION (MM 25.200; wrong values by construction, never delivered): every weight and scale
                # load stays, each word folded in with one fp32 add, and no field extraction, conversion or fma
                if jj == 0:
                    ts[r] = w if ts[r] is None else b.fadd(ts[r], w, type=ir.F32, name="nd%d_%d" % (r, e))
                continue
            if a16:
                # ONE-INSTRUCTION EXTRACTION with op426 only (and16, 8-bit immediate, sources kept): byte
                # bi of the word is w.L, w8.L, w.H, w8.H for bi = 0..3 (w8 = w >> 8, one keeping shift per
                # word); a 4-bit field at position pos in its byte is masked 0x0f << 4 pos and met by x
                # pre-scaled 16^-pos. op428 (register mask) read its mask as zero on hardware: its second
                # operand is not an ordinary register (MM 25.138).
                if lay.get("pool_masks"):
                    # POOL MASKS (MM 25.141.16): every field straight from its word's half - masks above 8 bits
                    # are op428 against the constant pool (a uniform), so no w >> 8 per word; x is pre-scaled by
                    # the field's position in the HALF (1, 1/16, 1/256, 1/4096 at q4), Apple's own form
                    fph = 16 // bits
                    half = "L" if jj < fph else "H"
                    m = ((1 << bits) - 1) << (bits * (jj % fph))
                    q = b.and16(w, half, m, name="q%d_%d" % (r, e), pool=m > 0xFF, direct=lay.get("and16_direct", False))
                    xe = xq16[e]
                else:
                    fb = 8 // bits                                   # fields per byte
                    bi, pos = jj // fb, jj % fb
                    if (r, e // pw) not in W8:
                        W8[(r, e // pw)] = b.shr(w, ir.Imm(8), name="w8_%d_%d" % (r, e // pw))
                    src = w if bi in (0, 2) else W8[(r, e // pw)]
                    half = "L" if bi < 2 else "H"
                    m = ((1 << bits) - 1) << (bits * pos)
                    q = b.and16(src, half, m, name="q%d_%d" % (r, e), direct=lay.get("and16_direct", False))
                    xe = xq16[e]
            elif shand:
                # SHIFT THEN AND-IMMEDIATE: the shift forms keep their source (a lifetime field), so the
                # multiply-read word needs no isolation copy per nibble (op424, the reg-reg and, releases
                # both sources and forced two copies per weight); the 8-bit and-immediate mask takes the
                # low field; the top field needs no mask at all
                lo = (1 << bits) - 1
                if jj == pw - 1:
                    q = b.shr(w, ir.Imm(bits * jj), name="q%d_%d" % (r, e))
                elif jj == 0:
                    q = getattr(b, "and")(w, ir.Imm(lo), name="q%d_%d" % (r, e))
                else:
                    q = getattr(b, "and")(b.shr(w, ir.Imm(bits * jj), name="s%d_%d" % (r, e)),
                                          ir.Imm(lo), name="q%d_%d" % (r, e))
                xe = xs[e]
            else:
                m = ((1 << bits) - 1) << (bits * jj)
                q = getattr(b, "and")(w, MK[jj] if lean else O._c(b, m, "m%d_%d" % (r, e)), name="q%d_%d" % (r, e))
                xe = xq[e]
            qf = b.u32_to_f32(q, name="qf%d_%d" % (r, e))
            if chains:
                ck = (r, (e % pw) % fb)
                tc[ck] = (b.fmul(qf, xe, type=ir.F32, name="p%d_%d" % (r, e)) if ck not in tc
                          else b.fma(qf, xe, tc[ck], name="t%d_%d" % (r, e)))
                continue
            ts[r] = (b.fmul(qf, xe, type=ir.F32, name="p%d_%d" % (r, e)) if ts[r] is None
                     else b.fma(qf, xe, ts[r], name="t%d_%d" % (r, e)))
        if chains and not lay.get("ablate_nodeq"):
            for r in rows_c:
                ts[r] = tc[(r, 0)]
                for pos in range(1, fb):
                    ts[r] = b.fma(tc[(r, pos)], scales[pos - 1], ts[r], name="tc%d_%d" % (r, pos))
        for r in rows_c:
            if hi16:
                sv, bv = Sv[r], Bv[r]
            else:
                sv = b.shl(Sv[r], ir.Imm(16) if lay.get("a16") else O._c(b, 16, "s16_%d" % r), name="s%d" % r)
                bv = b.shl(Bv[r], ir.Imm(16) if lay.get("a16") else O._c(b, 16, "b16_%d" % r), name="bv%d" % r)
            if lay.get("epi_fma"):
                # FUSED EPILOGUE (lay["epi_fma"]): acc = fma(b, sx, fma(s, t, acc)), two roundings instead of four
                newacc[r] = b.fma(bv, sx, b.fma(sv, ts[r], accs[r], name="au%d" % r), name="av%d" % r)
                newacc[r].type = ir.F32
                continue
            acc = b.fadd(accs[r], b.fmul(sv, ts[r], type=ir.F32, name="u%d" % r), type=ir.F32, name="au%d" % r)
            newacc[r] = b.fadd(acc, b.fmul(bv, sx, type=ir.F32, name="v%d" % r), type=ir.F32, name="av%d" % r)
    if ptr:
        ir.Builder.phi_latch(XP, b.add(XP, _imm(XSTEP, "xstep"), name="xp_next"))
        for r in range(R):
            ir.Builder.phi_latch(VP[r], b.add(VP[r], _imm(VSTEP, "vstep%d" % r), name="vp%d_next" % r))
            ir.Builder.phi_latch(IP[r], b.add(IP[r], _imm(GSTEP, "gstep%d" % r), name="ip%d_next" % r))
    kn = b.add(kl, ir.Imm(1), name="k_next")
    ir.Builder.phi_latch(kl, kn)
    if pf:
        for r in range(R):
            for i in range(wpt):
                ir.Builder.phi_latch(WP[r][i], WN[r][i])
    for r in range(R):
        ir.Builder.phi_latch(accs[r], newacc[r])
    if hi16:
        for r in range(R):
            ir.Builder.phi_latch(SP[r], Sv[r])
            ir.Builder.phi_latch(BP[r], Bv[r])
    b.br_cond(b.cmp(kn, trips, "lt", name="more"), hdr, post)
    b.at(post)
    rn = None
    if pparts is not None:
        loc = None
        for i, pv in enumerate(pparts):
            pv.type = ir.F32
            loc = pv if loc is None else b.fadd(loc, pv, type=ir.F32, name="rpacc%d" % i)
        ss = TR.emit_butterfly(b, loc, TR.ROW_BUTTERFLY_MASKS, operation="sum")
        ss = TR.emit_butterfly(b, ss, TR.COLUMN_BUTTERFLY_MASKS, operation="sum")
        mean = b.fmul(ss, O._cf(b, F32(1.0 / K), "rinv_d"), name="rmean")
        KRN = O.emit_constants(b)
        rn = O.emit_rn(b, "rsqrt", b.fadd(mean, O._cf(b, F32(lay.get("eps", 1e-5)), "reps"), type=ir.I32, name="rvar"), KRN, "rrs")
    nxt = []                                           # (row, the next norm's input v) for with_next_norm
    ys = []
    VV = []
    for r in range(R):
        v = TR.emit_butterfly(b, newacc[r], TR.ROW_BUTTERFLY_MASKS, operation="sum")
        VV.append(TR.emit_butterfly(b, v, TR.COLUMN_BUTTERFLY_MASKS, operation="sum"))
    if lay.get("ksplit"):
        # every simdgroup publishes its R sums at words s R + r; after one barrier EVERY simdgroup forms the total
        # and runs the epilogue, so all store identical values and nothing needs predication
        for r in range(R):
            b.store_tg(VV[r], b.add(b.mul(sgi, O._c(b, R, "ksR"), name="kst_m%d" % r), O._c(b, r, "ksr%d" % r), name="kst%d" % r))
        b.barrier("threadgroup")
        # summed in simdgroup order: ((p0 + p1) + p2) + p3
        NV = []
        for r in range(R):
            v = b.load_tg(O._c(b, r, "ksl0_%d" % r), type=ir.F32, name="ksp0_%d" % r)
            for sgn in range(1, sgs):
                v = b.fadd(v, b.load_tg(O._c(b, sgn * R + r, "ksl%d_%d" % (sgn, r)), type=ir.F32, name="ksp%d_%d" % (sgn, r)),
                           type=ir.F32, name="ksv%d_%d" % (sgn, r))
            NV.append(v)
        VV = NV
    for r in range(R):
        v = VV[r]
        if rn is not None:
            v = b.fmul(v, rn, type=ir.F32, name="yr%d" % r)
        if lay.get("swiglu"):
            ys.append(v)
            continue
        res = lay.get("res")
        if res == "add16":
            # wo + residual1: h = fp32(dot + x), x the layer input fp16 at binding 3 byte RES; h fp32 at OUT
            xr = b.f16_to_f32(b.load(c, b.add(row0, O._c(b, r + lay["RES"] // 2, "ri%d" % r), name="rii%d" % r),
                                     width="half", name="rh%d" % r), name="rx%d" % r)
            v = b.fadd(v, xr, type=ir.F32, name="hres%d" % r)
            nxt.append((r, v))
        elif res == "add32_to16":
            # w2 + residual2: out = fp16_rne(dot + h), h fp32 at binding 3 byte OUT; out fp16 at byte RES
            hr = b.load(c, b.add(row0, O._c(b, r + lay["OUT"] // 4, "hi%d" % r), name="hii%d" % r), type=ir.I32, name="hr%d" % r)
            o32 = b.fadd(v, hr, type=ir.I32, name="ores%d" % r)
            oh = b.f32_to_f16_rte(o32, name="oh%d" % r)
            b.store_at(c, b.add(row0, O._c(b, r + lay["RES"] // 2, "ro%d" % r), name="roi%d" % r), oh, width="half")
            if lay.get("next_norm"):
                nxt.append((r, b.f16_to_f32(oh, name="ohw%d" % r)))   # the next norm reads the fp16 x
            continue
        b.store_at(c, b.add(row0, O._c(b, r + lay["OUT"] // 4, "o%d" % r), name="oi%d" % r), v)
    if lay.get("next_norm"):
        # xg = fp16_rne(v g) per row, and the threadgroup's sum of squares of v in row order
        part = None
        for r, v in nxt:
            gh = b.load(bb, b.add(row0, O._c(b, r + lay["GN"] // 2, "gn%d" % r), name="gni%d" % r), width="half", name="gnh%d" % r)
            gv = b.f16_to_f32(gh, name="gnv%d" % r)
            xg = b.f32_to_f16_rte(b.fmul(v, gv, type=ir.I32, name="xgm%d" % r), name="xg%d" % r)
            b.store_at(c, b.add(row0, O._c(b, r + lay["XG"] // 2, "xgo%d" % r), name="xgi%d" % r), xg, width="half")
            q2 = b.fmul(v, v, type=ir.I32, name="nsq%d" % r)
            part = q2 if part is None else b.fadd(part, q2, type=ir.I32, name="nps%d" % r)
        b.store_at(c, b.add(group, O._c(b, lay["PSO"] // 4, "psob"), name="psoi"), part)
    if lay.get("last_norm"):
        _emit_last_norm(fn, b, ir, TR, c, bb, lane, lay)
    if lay.get("argmax_chunks"):
        _emit_chunk_argmax(fn, b, ir, c, lane, group, lay)
    if lay.get("swiglu"):
        # g17decodeops.build_swiglu's exact sequence (Piece A's rounding points): t = g (-1/ln 2); exp2_soft;
        # + 1; correctly rounded recip; s = g r; act = fp16_rne(s u)
        import g17tensorcommonruntime as TCR
        K_ = O.emit_constants(b)
        K2 = O.emit_exp2_constants(b)
        nil2 = O._cf(b, TCR.GELU_NEG_INV_LN2, "neg_inv_ln2")
        one = O._cf(b, F32(1.0), "fone")
        for r in range(R0):
            g, u = ys[r], ys[R0 + r]
            t = b.fmul(g, nil2, type=ir.I32, name="st%d" % r)
            e = O.emit_exp2_soft(b, t, K2, "sx%d" % r)
            den = b.fadd(e, one, type=ir.I32, name="sd%d" % r)
            rc = O.emit_rn(b, "recip", den, K_, "sr%d" % r)
            sl = b.fmul(g, rc, type=ir.I32, name="silu%d" % r)
            y = b.fmul(sl, u, type=ir.I32, name="act%d" % r)
            if lay.get("act32"):
                # fp32 of the fp16-rounded act (exact widening): the next qmv reads it as xvec fp32 x
                ah = b.f32_to_f16_rte(y, name="acth%d" % r)
                b.store_at(c, b.add(row0, O._c(b, r + lay["OUT"] // 4, "o%d" % r), name="oi%d" % r),
                           b.f16_to_f32(ah, name="actw%d" % r))
            else:
                b.store_at(c, b.add(row0, O._c(b, r + lay["OUT"] // 2, "o%d" % r), name="oi%d" % r),
                           b.f32_to_f16_rte(y, name="acth%d" % r), width="half")
    if _into is not None:
        b.br(_into["exit"])
        return None
    b.ret()
    if lay.get("hoist_consts"):
        _hoist_loop_constants(fn, ir)
    ir.verify(fn)
    fn.compact_registers = bool(lay.get("compact", False))
    return cc.compile_function(fn)


def build_qmv_wide(lay):
    """MLX qmv_wide layout (MM 25.144.9): KL K-lanes per row, rows_per_sg rows per simdgroup, wide_sgs simdgroups
    (rows_per_sg * wide_sgs rows) per threadgroup, 32 * wide_sgs threads; the grid is (N / rows_tg) row-blocks by
    ceil(NB / vp) vector passes. K-lane l walks its ROW's words {l, l + KL, l + 2KL, ...}; per word the pw fields are
    DEQUANTIZED ONCE (w = q * s + b in fp32) and reused by the pass's vp vectors as a plain fp32 dot,
    acc_v = fma(x_v[e], w, acc_v); then the KL lanes sum through threadgroup memory in lane order. A new fp32
    order (qmv_wide_reference): not the 32-lane split-K butterfly. Scope for now: plain qmv (no residual, no swiglu)."""
    from agxforge.g17 import cc, ir
    if lay.get("res") or lay.get("swiglu"):
        raise ValueError("qmv wide: plain qmv only")
    N, K, bits, grp, pw = lay["Nout"], lay["Kq"], lay["bits"], lay["group"], lay["per_word"]
    NB = lay.get("batch", 1)
    KL, RSG, SGS = lay.get("klanes", 8), lay.get("rows_per_sg", 4), lay.get("wide_sgs", 1)
    if SGS != 1:
        raise ValueError("qmv wide: one simdgroup per threadgroup (the carrier sets 32 threads; SR_SIMD_GRP needs "
                         "declared threadgroup memory the tensor class refuses without cooperative sharing)")
    vp = min(NB, lay.get("wide_pass", 4))
    if NB % vp:
        raise ValueError("qmv wide: vp divides NB")
    if KL & (KL - 1) or (grp // pw) & (grp // pw - 1):
        raise ValueError("qmv wide: KL and words-per-group are powers of two")
    passes, rows_tg, words = NB // vp, RSG * SGS, K // pw
    if words % KL or N % rows_tg:
        raise ValueError("qmv wide: KL divides words and rows_tg divides N")
    wpk, wpg = words // KL, grp // pw
    row_blocks = N // rows_tg
    if row_blocks & (row_blocks - 1):
        raise ValueError("qmv wide: N / rows_tg is a power of two")
    if pw % 4 and pw != 4:
        raise ValueError("qmv wide: pw is 4 or a multiple of 4")
    G = row_blocks * passes
    # one simdgroup (32 threads), C at binding 0, base 0, NO MMA carrier (as build_qmv2_batch). The KL K-lanes of a row
    # reduce THROUGH threadgroup memory (32 * vp words): this both sums the lanes and puts the program in the measured
    # simdgroup-split metadata class (a scalar SR-using program without threadgroup memory has no slot-29 vector). The
    # lane (thread_index_in_simdgroup, SR_SIMD_ELEM) splits into KL K-lanes and RSG = 32 / KL rows.
    c = ir.Buffer("C", 0, elem=ir.F32)
    a = ir.Buffer("A", 1, elem=ir.F16)
    bb = ir.Buffer("B", 2, elem=ir.F16)
    fn = ir.Function("tensor_gemm_generic_runtime_demo", [c, a, bb])
    fn.declare_threadgroup(32 * vp, size=(32, 1, 1))
    b = ir.Builder(fn, fn.block("entry"))
    mask = O._c(b, (1 << bits) - 1, "qmask")
    group = b.builtin("threadgroup_position_in_grid", name="group")
    # the threadgroup metadata class is witnessed at system registers (156 grid, 164 thread_position_in_threadgroup),
    # as build_qmv2_batch: derive the lane from thread_position_in_threadgroup (one simdgroup, so lane = tpt & 31).
    lane = getattr(b, "and")(b.builtin("thread_position_in_threadgroup", name="tpt"), O._c(b, 31, "l31"), name="lane")
    klane = getattr(b, "and")(lane, O._c(b, KL - 1, "klm"), name="klane")
    rinsg = b.shr(lane, O._c(b, KL.bit_length() - 1, "rsh"), name="rinsg")
    pass_i = b.shr(group, O._c(b, row_blocks.bit_length() - 1, "pbsh"), name="pass") if passes > 1 else O._c(b, 0, "pass0")
    rblock = getattr(b, "and")(group, O._c(b, row_blocks - 1, "rbm"), name="rblock") if passes > 1 else group
    row = b.add(b.mul(rblock, O._c(b, rows_tg, "rtg"), name="rb"), rinsg, name="row")
    v0 = b.mul(pass_i, O._c(b, vp, "vp"), name="v0") if passes > 1 else O._c(b, 0, "v0")
    gpr = K // grp
    wbase = b.add(b.add(b.mul(row, O._c(b, words, "words"), name="rw"), klane, name="rwk"), O._c(b, lay["W"] // 4, "wb"), name="wbase")
    sbase = b.add(b.mul(row, O._c(b, gpr, "gpr"), name="sr"), O._c(b, lay["S"] // 2, "sb"), name="sbase")
    bbase = b.add(sbase, O._c(b, (lay["B"] - lay["S"]) // 2, "bd"), name="bbase")
    kpw = b.mul(klane, O._c(b, pw, "pw"), name="kpw")
    gv = [b.add(v0, O._c(b, vl, "vl%d" % vl), name="gv%d" % vl) if vl else v0 for vl in range(vp)]
    xbases = [b.add(b.mul(gv[vl], O._c(b, K, "K%d" % vl), name="vK%d" % vl), O._c(b, lay["X"] // 4, "xb%d" % vl), name="xbase%d" % vl)
              for vl in range(vp)]
    hdr, post = fn.block("wide_loop"), fn.block("wide_done")
    z = [O._cf(b, F32(0.0), "z%d" % vl) for vl in range(vp)]
    step0 = O._c(b, 0, "step0")                       # the loop seeds live in the pre-header, not the header
    b.br(hdr)
    b.at(hdr)
    step = b.phi(step0, name="step")
    accs = [b.phi(z[vl], type=ir.F32, name="acc%d" % vl) for vl in range(vp)]
    kKL = b.mul(step, O._c(b, KL, "KL"), name="kKL")
    W_ = b.load(bb, b.add(wbase, kKL, name="waddr"), type=ir.I32, name="W")
    gidx = b.shr(b.add(kKL, klane, name="wglob"), O._c(b, wpg.bit_length() - 1, "wpgsh"), name="gidx")
    s_ = b.shl(b.load(bb, b.add(sbase, gidx, name="si"), width="half", name="sh"), O._c(b, 16, "s16"), name="sf")
    bv_ = b.shl(b.load(bb, b.add(bbase, gidx, name="bi"), width="half", name="bh"), O._c(b, 16, "b16"), name="bf")
    wdq = []
    for e in range(pw):
        q = getattr(b, "and")(b.shr(W_, O._c(b, bits * e, "esh%d" % e), name="wsh%d" % e) if e else W_, mask, name="q%d" % e)
        wdq.append(b.fma(b.u32_to_f32(q, name="qf%d" % e), s_, bv_, name="wdq%d" % e))
    xstep = b.add(b.mul(step, O._c(b, KL * pw, "KLpw"), name="xk"), kpw, name="xstep")
    newacc = list(accs)
    for vl in range(vp):
        xi = b.shr(b.add(xbases[vl], xstep, name="xa%d" % vl), O._c(b, 2, "two"), name="xi%d" % vl)
        chunks = []
        for blk in range(pw // 4):
            chunks += b.load_vec_at(a, b.add(xi, O._c(b, blk, "blk%d" % blk), name="xib%d_%d" % (vl, blk)) if blk else xi,
                                    n=4, name="xv%d_%d_" % (vl, blk))
        for e in range(pw):
            newacc[vl] = b.fma(chunks[e], wdq[e], newacc[vl], name="d%d_%d" % (vl, e))
    kn = b.add(step, ir.Imm(1), name="step_next")
    ir.Builder.phi_latch(step, kn)
    for vl in range(vp):
        ir.Builder.phi_latch(accs[vl], newacc[vl])
    b.br_cond(b.cmp(kn, wpk, "lt", name="more"), hdr, post)
    b.at(post)
    # publish each lane's vp partials at tg[lane * vp + vl], then each lane sums its row's KL klanes IN LANE ORDER
    for vl in range(vp):
        b.store_tg(newacc[vl], b.add(b.mul(lane, O._c(b, vp, "vptg"), name="lvp"), O._c(b, vl, "tgvl%d" % vl), name="tgi%d" % vl) if vl
                   else b.mul(lane, O._c(b, vp, "vptg0"), name="lvp0"))
    b.barrier("threadgroup")
    row8 = b.mul(rinsg, O._c(b, KL, "rinsgKL"), name="row8")     # lane index of klane 0 of this row
    for vl in range(vp):
        red = None
        for kl in range(KL):
            lk = b.add(row8, O._c(b, kl, "klc%d" % kl), name="lk%d_%d" % (vl, kl)) if kl else row8
            idx = b.add(b.mul(lk, O._c(b, vp, "vpr%d_%d" % (vl, kl)), name="lkv%d_%d" % (vl, kl)), O._c(b, vl, "pvl%d_%d" % (vl, kl)), name="pidx%d_%d" % (vl, kl)) if vl \
                else b.mul(lk, O._c(b, vp, "vpr0_%d" % kl), name="lkv0_%d" % kl)
            p = b.load_tg(idx, type=ir.F32, name="tp%d_%d" % (vl, kl))
            red = p if red is None else b.fadd(red, p, type=ir.F32, name="red%d_%d" % (vl, kl))
        oidx = b.add(O._c(b, lay["OUT"] // 4, "ob%d" % vl), b.add(b.mul(gv[vl], O._c(b, N, "N%d" % vl), name="vN%d" % vl), row, name="orow%d" % vl), name="oidx%d" % vl)
        b.store_at(c, oidx, red)
    b.ret()
    ir.verify(fn)
    return cc.compile_function(fn)



def build_qmv2_batch(lay):
    """THE MULTI-VECTOR qmv (MM 25.144.3): lay["batch"] = nb activation vectors against one weight stream. Each trip loads
    the rows' weight words, scales and biases ONCE, extracts and converts every field ONCE (qf), then for each vector
    loads its x, forms its sum and pre-scaled copies and runs the single-vector fma chain and accumulation - so every
    vector's value is exactly the single-vector split-K kernel's (qmv_batch_reference). The vectors are processed
    one after another inside the trip so only one vector's x is live at a time. Scope: the split-K cooperative form
    (ksplit, coop), a16 extraction, xvec fp32 x, vload, lean invariant row bases; hi16_scales optional; residuals and
    the fused act32 SwiGLU as in build_qmv2; layout from with_batch (vector-major)."""
    from agxforge.g17 import cc, ir, tensorreduce as TR
    NB = lay.get("batch", 1)                 # 1: the single-vector dequantize-once kernel
    need = ("ksplit", "coop", "a16", "xvec", "vload", "coalesced", "lean")
    refuse = ("norm", "rnorm", "next_norm", "last_norm", "x16", "ptr_addr", "pool_masks", "shand", "xpre")
    if not all(lay.get(k) for k in need) or any(lay.get(k) for k in refuse):
        raise ValueError("qmv2 batch: the split-K cooperative a16 xvec vload lean form only")
    if lay.get("swiglu") and not lay.get("act32"):
        raise ValueError("qmv2 batch: the fused FFN writes act32")
    N, K, bits, grp, R0 = lay["Nout"], lay["Kq"], lay["bits"], lay["group"], lay["rows"]
    pw, wpt, sgs = lay["per_word"], lay["wpt"], lay["sgs"]
    if lay.get("swiglu"):
        F = lay["ffn"]
        ROFF = [r if r < R0 else F + r - R0 for r in range(2 * R0)]
        NOUT = F
    else:
        ROFF = list(range(R0))
        NOUT = N
    R = len(ROFF)
    if R > 8:
        raise ValueError("qmv2 batch: at most 8 computed rows (invariant row bases)")
    words = K // pw
    lane_words = words // 32
    if lane_words % wpt or (grp // pw) % wpt or wpt not in (2, 4):
        raise ValueError("qmv2 batch: a trip must hold whole vector loads inside one group")
    trips = lane_words // wpt
    if sgs not in (2, 4, 8) or trips % sgs:
        raise ValueError("qmv2 batch: 2, 4 or 8 simdgroups dividing the trips")
    trips //= sgs
    E = wpt * pw
    if E % 4 or lay["X"] % 16 or lay["W"] % (4 * wpt):
        raise ValueError("qmv2 batch: aligned vector loads")
    c = ir.Buffer("C", 0, elem=ir.F32)
    a = ir.Buffer("A", 1, elem=ir.F16)
    bb = ir.Buffer("B", 2, elem=ir.F16)
    fn = ir.Function("tensor_gemm_generic_runtime_demo", [c, a, bb])
    fn.declare_threadgroup(sgs * R * NB, size=(32 * sgs, 1, 1))
    b = ir.Builder(fn, fn.block("entry"))
    tpt = b.builtin("thread_position_in_threadgroup", name="lane")
    lane = getattr(b, "and")(tpt, ir.Imm(31), name="lane31")
    group = b.builtin("threadgroup_position_in_grid", name="group")
    sgi = b.shr(b.builtin("thread_position_in_threadgroup", name="tpt"), O._c(b, 5, "sg_sh"), name="sgi")
    row0 = b.mul(group, O._c(b, R0, "rows"), name="row0")
    gpr = K // grp
    wv0 = b.add(b.mul(row0, O._c(b, words, "wv_row"), name="wvr"),
                b.add(b.mul(lane, O._c(b, wpt, "wv_lanew"), name="wvlw"), O._c(b, lay["W"] // 4, "wvb"), name="wvlb"),
                name="wv0")
    xv0 = b.add(b.mul(lane, O._c(b, E, "xv_lane"), name="xvl"), O._c(b, lay["X"] // 4, "xvb"), name="xv0")
    s0 = b.add(b.mul(row0, O._c(b, gpr, "gpr"), name="sng"), O._c(b, lay["S"] // 2, "sb"), name="s0")
    bdiff = O._c(b, (lay["B"] - lay["S"]) // 2, "bdiff")
    WB = [b.add(wv0, O._c(b, ROFF[r] * words, "wrb%d" % r), name="wrow%d" % r) if r else wv0 for r in range(R)]
    SB = [b.add(s0, O._c(b, ROFF[r] * gpr, "srb%d" % r), name="srow%d" % r) if r else s0 for r in range(R)]
    BB = [b.add(SB[r], bdiff, name="brow%d" % r) for r in range(R)]
    k0 = O._c(b, 0, "k0")
    zeros = [[O._cf(b, F32(0.0), "fzero%d_%d" % (v, r)) for r in range(R)] for v in range(NB)]
    scales = [O._cf(b, F32(2.0 ** (-bits * j)), "ps%d" % j) for j in range(1, pw)]
    koff = b.mul(sgi, O._c(b, trips, "ktrips"), name="koff")
    hi16 = lay.get("hi16_scales", False)
    if hi16:
        zero_hi = [O._c(b, 0, "hz%d" % i) for i in range(2 * R)]
    fb = 8 // bits
    dq = bool(lay.get("dequant_once"))
    ns = lay.get("acc_split", 1) if dq else 1                    # independent partial accumulators per (vector, row)
    if ns not in (1, 2, 4):
        raise ValueError("qmv2 batch: acc_split is 1, 2 or 4 (dequantize-once only)")
    per = lay.get("batch_step", 4 if R == 1 else 2)
    if per not in (2, 4) or E % per:
        raise ValueError("qmv2 batch: a step of 2 or 4 elements dividing the trip")
    psh = per.bit_length() - 1
    # VECTOR PASSES (lay["batch_pass"], MM 25.144.3): the vectors split into groups of `pv`, each its own full K loop
    # over the same weights - the weight stream is read nb / pv times, but only pv vectors' accumulators and x streams
    # are live at once (nb = 8 in one pass ran out of registers or needed its constants in the loop)
    pv = lay.get("batch_pass", NB)
    if pv < 1 or NB % pv:
        raise ValueError("qmv2 batch: batch_pass divides the batch")
    newacc = [[None] * R for _ in range(NB)]
    # every pass takes its OWN entry constants, all formed here in the entry block: two loops' phis coalesced with one
    # shared entry value intersect, and so did a later pass's constants formed in the earlier pass's exit block
    PASS_K0 = [k0] + [O._c(b, 0, "p%d_k0" % pi) for pi in range(1, NB // pv)]
    PASS_ZH = [zero_hi if hi16 else None] + [[O._c(b, 0, "p%d_hz%d" % (pi, i)) for i in range(2 * R)] if hi16 else None
                                             for pi in range(1, NB // pv)]

    def emit_pass(pi, vs):
        P = "" if pv == NB else "p%d_" % pi
        hdr, post = fn.block(P + "qmv_loop"), fn.block(P + "qmv_done")
        k0p, zh = PASS_K0[pi], PASS_ZH[pi]
        zerox = ({v: [[O._cf(b, F32(0.0), P + "zx%d_%d_%d" % (v, r, kk)) for kk in range(1, ns)] for r in range(R)]
                  for v in vs} if ns > 1 else None)                 # a DISTINCT seed per extra accumulator (no coalescing)
        b.br(hdr)
        b.at(hdr)
        kl = b.phi(k0p, name=P + "k")
        accs = {v: [b.phi(zeros[v][r], type=ir.F32, name="acc%d_%d" % (v, r)) for r in range(R)] for v in vs}
        # acc_split: ns - 1 EXTRA accumulator phis per (v, r); each takes a disjoint set of the lane's elements
        # (position mod ns), so the critical dependency chain per accumulator is 1/ns as long. Summed at loop exit.
        accx = ({v: [[b.phi(zerox[v][r][kk - 1], type=ir.F32, name="accx%d_%d_%d" % (v, r, kk)) for kk in range(1, ns)]
                     for r in range(R)] for v in vs} if ns > 1 else None)
        if hi16:
            SP = [b.phi(zh[r], name=P + "sp%d" % r) for r in range(R)]
            BP = [b.phi(zh[R + r], name=P + "bp%d" % r) for r in range(R)]
        k = b.add(kl, koff, name=P + "kk")
        kw = b.mul(k, O._c(b, 32 * wpt, P + "wtrip32"), name=P + "kw")
        gidx = b.shr(b.add(kw, b.mul(lane, O._c(b, wpt, P + "lwp"), name=P + "lwpm"), name=P + "gt"),
                     O._c(b, (grp // pw).bit_length() - 1, P + "gsh"), name=P + "gidx")
        # ---- the shared half of the trip: weights, scales, biases, every field extracted and converted ONCE
        Wv, Sv, Bv = {}, {}, {}
        for r in range(R):
            vi = b.shr(b.add(WB[r], kw, name=P + "wi%d" % r), O._c(b, wpt.bit_length() - 1, P + "vsh"), name=P + "vi%d" % r)
            Wv[r] = b.load_vec_at(bb, vi, n=wpt, name=P + "w%d_" % r)
        for r in range(R):
            si, bi = b.add(SB[r], gidx, name=P + "si%d" % r), b.add(BB[r], gidx, name=P + "bi%d" % r)
            if hi16:
                Sv[r] = b.load_hi16(SP[r], bb, si, name=P + "sh%d" % r)
                Bv[r] = b.load_hi16(BP[r], bb, bi, name=P + "bh%d" % r)
            else:
                Sv[r] = b.load(bb, si, width="half", name=P + "sh%d" % r)
                Bv[r] = b.load(bb, bi, width="half", name=P + "bh%d" % r)
        W8 = {}

        def field(r, e):
            # the (r, e) field extracted and converted: the fma's first operand must sit in a NARROW register (r4..r15,
            # eight free), so fields are formed a few at a time, just before every vector consumes them
            w, jj = Wv[r][e // pw], e % pw
            bi_, pos = jj // fb, jj % fb
            if (r, e // pw) not in W8:
                W8[(r, e // pw)] = b.shr(w, ir.Imm(8), name=P + "w8_%d_%d" % (r, e // pw))
            src = w if bi_ in (0, 2) else W8[(r, e // pw)]
            m = ((1 << bits) - 1) << (bits * pos)
            q = b.and16(src, "L" if bi_ < 2 else "H", m, name=P + "q%d_%d" % (r, e))
            return b.u32_to_f32(q, name=P + "qf%d_%d" % (r, e))
        if hi16:
            SV, BV = [Sv[r] for r in range(R)], [Bv[r] for r in range(R)]
        else:
            SV = [b.shl(Sv[r], ir.Imm(16), name=P + "s%d" % r) for r in range(R)]
            BV = [b.shl(Bv[r], ir.Imm(16), name=P + "bv%d" % r) for r in range(R)]
        # ---- in STEPS of `per` elements: the step's fields first (per R of them live, in narrow registers), then each
        # vector in turn loads its `per` x values (one vector load), advances its running sum and meets the fields with its
        # fma chain - so only one vector's x slice is live at a time. Per vector the chain and the sum still run in element
        # order, so each vector's value is the single-vector kernel's exactly.
        xp = b.shr(b.add(xv0, b.mul(k, O._c(b, 32 * E, P + "xe_trip"), name=P + "xk"), name=P + "xi"),
                   O._c(b, psh, P + "xpsh"), name=P + "xpi")
        # each vector's load index is ONE add off the shared base at its use: a load's index sits in a narrow register,
        # so nb per-vector bases held across the trip exhausted them at nb = 8
        T = {v: [None] * R for v in vs}
        SXv = {v: None for v in vs}
        acc_latch = {v: [None] * R for v in vs}                  # what each (v, r) primary phi latches (dq: kk=0 chain)
        accx_latch = {v: [None] * R for v in vs}
        if dq:
            # DEQUANTIZE ONCE (MLX's qmv_wide idea, MM 25.144.3): per trip and row, the scale pre-scaled for each field
            # position in its byte (and16 leaves the field shifted by bits pos: q 2^(bits pos) s 2^-(bits pos) = q s,
            # exact), then every element's w = fma(qf, s', b) ONCE, then every vector a plain fp32 chain acc = fma(w, x,
            # acc) - no per-vector x sum, pre-scale or bias fold (qmv_dq_reference)
            SPOS = [[SV[r]] + [b.fmul(SV[r], scales[p - 1], type=ir.F32, name=P + "sp%d_%d" % (r, p)) for p in range(1, fb)]
                    for r in range(R)]
            Tx = {v: [[None] * (ns - 1) for _ in range(R)] for v in vs} if ns > 1 else None
            for j in range(E // per):
                es = [per * j + i for i in range(per)]
                WQ = {}
                for e in es:
                    for r in range(R):
                        WQ[(r, e)] = b.fma(field(r, e), SPOS[r][(e % pw) % fb], BV[r], name=P + "wq%d_%d" % (r, e))
                for v in vs:
                    off = v * K // per + j
                    xsl = b.load_vec_at(a, b.add(xp, O._c(b, off, "xpo%d_%d" % (v, j)), name="xpj%d_%d" % (v, j)) if off else xp,
                                        n=per, name="xv%d_%d_" % (v, j))
                    for i, e in enumerate(es):
                        kk = (per * j + i) % ns
                        for r in range(R):
                            if kk == 0:
                                T[v][r] = b.fma(WQ[(r, e)], xsl[i], accs[v][r] if T[v][r] is None else T[v][r],
                                                name="d%d_%d_%d" % (v, r, e))
                            else:
                                cur = accx[v][r][kk - 1] if Tx[v][r][kk - 1] is None else Tx[v][r][kk - 1]
                                Tx[v][r][kk - 1] = b.fma(WQ[(r, e)], xsl[i], cur, name="dx%d_%d_%d" % (v, r, e))
            for v in vs:
                for r in range(R):
                    acc_latch[v][r] = T[v][r]                     # kk = 0 chain feeds this (v, r)'s primary phi
                    if ns > 1:
                        accx_latch[v][r] = list(Tx[v][r])         # kk = 1..ns-1 chains feed the extra phis
                    tot = T[v][r]
                    for kk in range(1, ns):                       # the exit value is the partials summed IN ORDER
                        tot = b.fadd(tot, Tx[v][r][kk - 1], type=ir.F32, name="as%d_%d_%d" % (v, r, kk))
                    newacc[v][r] = tot
                    newacc[v][r].type = ir.F32
        for j in range(0 if dq else E // per):
            es = [per * j + i for i in range(per)]
            QF = {(r, e): field(r, e) for e in es for r in range(R)}
            for v in vs:
                off = v * K // per + j
                xsl = b.load_vec_at(a, b.add(xp, O._c(b, off, "xpo%d_%d" % (v, j)), name="xpj%d_%d" % (v, j)) if off else xp,
                                    n=per, name="xv%d_%d_" % (v, j))
                for i, e in enumerate(es):
                    x = xsl[i]
                    SXv[v] = x if SXv[v] is None else b.fadd(SXv[v], x, type=ir.F32, name="sx%d_%d" % (v, e))
                    xe = x if (e % pw) % fb == 0 else b.fmul(x, scales[(e % pw) % fb - 1], type=ir.F32, name="xh%d_%d" % (v, e))
                    for r in range(R):
                        T[v][r] = (b.fmul(QF[(r, e)], xe, type=ir.F32, name="p%d_%d_%d" % (v, r, e)) if T[v][r] is None
                                   else b.fma(QF[(r, e)], xe, T[v][r], name="t%d_%d_%d" % (v, r, e)))
        for v in (() if dq else vs):
            for r in range(R):
                acc = b.fadd(accs[v][r], b.fmul(SV[r], T[v][r], type=ir.F32, name="u%d_%d" % (v, r)), type=ir.F32,
                             name="au%d_%d" % (v, r))
                newacc[v][r] = b.fadd(acc, b.fmul(BV[r], SXv[v], type=ir.F32, name="bs%d_%d" % (v, r)), type=ir.F32,
                                      name="av%d_%d" % (v, r))
                acc_latch[v][r] = newacc[v][r]
        kn = b.add(kl, ir.Imm(1), name=P + "k_next")
        ir.Builder.phi_latch(kl, kn)
        for v in vs:
            for r in range(R):
                ir.Builder.phi_latch(accs[v][r], acc_latch[v][r])
                for kk in range(1, ns):
                    ir.Builder.phi_latch(accx[v][r][kk - 1], accx_latch[v][r][kk - 1])
        if hi16:
            for r in range(R):
                ir.Builder.phi_latch(SP[r], Sv[r])
                ir.Builder.phi_latch(BP[r], Bv[r])
        b.br_cond(b.cmp(kn, trips, "lt", name=P + "more"), hdr, post)
        b.at(post)
        # this pass's vectors reduced and published now, so their values need not live through the later passes
        for v in vs:
            for r in range(R):
                t = TR.emit_butterfly(b, newacc[v][r], TR.ROW_BUTTERFLY_MASKS, operation="sum")
                t = TR.emit_butterfly(b, t, TR.COLUMN_BUTTERFLY_MASKS, operation="sum")
                b.store_tg(t, b.add(b.mul(sgi, O._c(b, R * NB, "ksRB%d_%d" % (v, r)), name="kst_m%d_%d" % (v, r)),
                                    O._c(b, v * R + r, "ksr%d_%d" % (v, r)), name="kst%d_%d" % (v, r)))
    for pi in range(NB // pv):
        emit_pass(pi, list(range(pi * pv, (pi + 1) * pv)))
    # ---- the split-K partials (published per pass at word s R nb + v R + r) meet after one barrier
    b.barrier("threadgroup")
    VV = [[None] * R for _ in range(NB)]
    for v in range(NB):
        for r in range(R):
            t = b.load_tg(O._c(b, v * R + r, "ksl0_%d_%d" % (v, r)), type=ir.F32, name="ksp0_%d_%d" % (v, r))
            for sgn in range(1, sgs):
                t = b.fadd(t, b.load_tg(O._c(b, sgn * R * NB + v * R + r, "ksl%d_%d_%d" % (sgn, v, r)), type=ir.F32,
                                        name="ksp%d_%d_%d" % (sgn, v, r)), type=ir.F32, name="ksv%d_%d_%d" % (sgn, v, r))
            VV[v][r] = t
    # ---- the epilogue, per vector at its stride
    res = lay.get("res")
    if lay.get("swiglu"):
        import g17tensorcommonruntime as TCR
        K_ = O.emit_constants(b)
        K2 = O.emit_exp2_constants(b)
        nil2 = O._cf(b, TCR.GELU_NEG_INV_LN2, "neg_inv_ln2")
        one = O._cf(b, F32(1.0), "fone")
    for v in range(NB):
        for r in range(R0 if lay.get("swiglu") else R):
            if lay.get("swiglu"):
                g, u = VV[v][r], VV[v][R0 + r]
                t = b.fmul(g, nil2, type=ir.I32, name="st%d_%d" % (v, r))
                ex = O.emit_exp2_soft(b, t, K2, "sx%d_%d" % (v, r))
                den = b.fadd(ex, one, type=ir.I32, name="sd%d_%d" % (v, r))
                rc = O.emit_rn(b, "recip", den, K_, "sr%d_%d" % (v, r))
                sl = b.fmul(g, rc, type=ir.I32, name="silu%d_%d" % (v, r))
                y = b.fmul(sl, u, type=ir.I32, name="act%d_%d" % (v, r))
                ah = b.f32_to_f16_rte(y, name="acth%d_%d" % (v, r))
                b.store_at(c, b.add(row0, O._c(b, r + lay["OUT"] // 4 + v * NOUT, "o%d_%d" % (v, r)), name="oi%d_%d" % (v, r)),
                           b.f16_to_f32(ah, name="actw%d_%d" % (v, r)))
                continue
            y = VV[v][r]
            if res == "add16":
                xr = b.f16_to_f32(b.load(c, b.add(row0, O._c(b, r + lay["RES"] // 2 + v * N, "ri%d_%d" % (v, r)),
                                                  name="rii%d_%d" % (v, r)), width="half", name="rh%d_%d" % (v, r)),
                                  name="rx%d_%d" % (v, r))
                y = b.fadd(y, xr, type=ir.F32, name="hres%d_%d" % (v, r))
            elif res == "add32_to16":
                hr = b.load(c, b.add(row0, O._c(b, r + lay["OUT"] // 4 + v * N, "hi%d_%d" % (v, r)), name="hii%d_%d" % (v, r)),
                            type=ir.I32, name="hr%d_%d" % (v, r))
                oh = b.f32_to_f16_rte(b.fadd(y, hr, type=ir.I32, name="ores%d_%d" % (v, r)), name="oh%d_%d" % (v, r))
                b.store_at(c, b.add(row0, O._c(b, r + lay["RES"] // 2 + v * N, "ro%d_%d" % (v, r)), name="roi%d_%d" % (v, r)),
                           oh, width="half")
                continue
            b.store_at(c, b.add(row0, O._c(b, r + lay["OUT"] // 4 + v * NOUT, "o%d_%d" % (v, r)), name="oi%d_%d" % (v, r)), y)
    b.ret()
    if lay.get("hoist_consts"):
        _hoist_loop_constants(fn, ir)
    ir.verify(fn)
    return cc.compile_function(fn)


def build_uber(arms, SEL):
    """SEVERAL OPS AS ONE PROGRAM, so a chain of them is one pipeline (MM 25.139.6: a pipeline change drains the
    GPU, consecutive dispatches of one pipeline overlap). The op is the uint32 at binding 3 byte SEL (0, 1, ...),
    the same for every lane; the dispatch launches that arm's grid. An arm is a qmv layout (build_qmv2's plain
    nocarrier form) or a callable emit(into) that emits from into["b"]'s block and branches to into["exit"].
    cc lowers a conditional as PREDICATION (an if-then region under the exec mask, no else and no skip branch), so
    the program is one guarded region per arm in sequence, each guard (sel == i) as an equality value tested
    against 0 (the last-threadgroup norm's measured idiom), and every dispatch also walks the other arms masked
    off: their straight code and one masked pass of each loop."""
    from agxforge.g17 import cc, ir
    fn, b, a, bb, c = O._function()
    tg = b.builtin("threadgroup_position_in_grid", name="u_tg")             # read so the program authors
    si = b.add(b.mul(tg, O._c(b, 0, "u_z"), name="u_z0"), O._c(b, SEL // 4, "u_selw"), name="u_si")
    sel = b.add(b.load(c, si, type=ir.I32, name="u_sel_ld"), O._c(b, 0, "u_sz"), name="u_sel")
    order = [fn.blocks[0]]
    for i, arm_ in enumerate(arms):
        exit_ = ir.Block("u%d_exit" % i)
        arm = ir.Block("u%d_entry" % i)
        eq = b.icmp(sel, O._c(b, i, "u_i%d" % i), "eq", name="u_eq%d" % i)
        b.br_cond(b.cmp(eq, 0, "gt", name="u_is%d" % i), arm, exit_)
        n0 = len(fn.blocks)
        fn.blocks.append(arm)
        b.at(arm)
        into = dict(fn=fn, b=b, a=a, bb=bb, c=c, exit=exit_, prefix="u%d_" % i)
        if callable(arm_):
            arm_(into)
        else:
            build_qmv2(dict(arm_), _into=into)
        order += fn.blocks[n0:]
        fn.blocks.append(exit_); order.append(exit_)
        b.at(exit_)
    fn.blocks[:] = order
    b.ret()
    ir.verify(fn)
    return cc.compile_function(fn)


def _emit_last_norm(fn, b, ir, TR, c, bb, lane, lay):
    """THE NEXT RMSNORM IN THE LAST THREADGROUP (MM 25.140.4). Every threadgroup, its rows stored, fences device
    memory and counts itself (lane 0, op10094 at binding 0 + CNT; its old value reaches every lane through the
    scratchpad). The threadgroup that reads G - 1 is the last: it reads the whole row v (the fp16 x at RES for the
    attention norm, the fp32 h at OUT for the FFN norm), forms r in g17decodestep.rmsnorm's order (lane l's
    ascending chain over v[l + 32 i], the row then column butterflies, mean = ss / N, rsqrt_rn(mean + eps)),
    writes xn = fp16_rne((v r) g) at binding 0 [XN] (g fp16 at binding 2 [GN]), and resets the count to 0."""
    import g17decodeops as O_
    N, G = lay["Nout"], lay["groups"]
    I = ir.I32
    b.barrier("fence_device")
    one = O_._c(b, 1, "ln_one")
    cl, cj = fn.block("ln_count"), fn.block("ln_counted")
    b.br_cond(b.cmp(lane, 1, "lt", name="ln_lane0"), cl, cj)
    b.at(cl)
    old = b.atomic_uniform("add", c, one, name="ln_old", slot6=lay["CNT"])
    b.store_tg(b.add(old, O_._c(b, 0, "ln_z"), name="ln_oc"), O_._c(b, 0, "ln_t0"))
    b.br(cj)
    b.at(cj)
    b.barrier("threadgroup")
    seen = b.load_tg(O_._c(b, 0, "ln_t0r"), name="ln_seen")
    mg, jn = fn.block("ln_last"), fn.block("ln_done")
    # the compare immediate is 8 bits: G - 1 goes through an equality VALUE, and the branch tests it against 0
    last = b.icmp(b.add(seen, O_._c(b, 0, "ln_z2"), name="ln_s"), O_._c(b, G - 1, "ln_gm1"), "eq", name="ln_eq")
    b.br_cond(b.cmp(last, 0, "gt", name="ln_is_last"), mg, jn)
    b.at(mg)
    b.barrier("fence_device")
    half = lay["last_norm"] == "half"
    def _v(i, tag):
        idx = b.add(lane, O_._c(b, 32 * i + (lay["RES"] // 2 if half else lay["OUT"] // 4), "%s_vi%d" % (tag, i)), name="%s_vx%d" % (tag, i))
        return b.f16_to_f32(b.load(c, idx, width="half", name="%s_vh%d" % (tag, i)), name="%s_v%d" % (tag, i)) if half else \
            b.load(c, idx, type=I, name="%s_v%d" % (tag, i))
    # two passes over the row, each value loaded where it is used: holding all N / 32 of them across the rsqrt
    # was what ran the register file out
    # LOADS IN BATCHES, each batch issued before any of its arithmetic: one load and its use at a time paid a full
    # memory latency 64 times in a row in the single threadgroup that does this (+37 us per dispatch)
    BATCH = 16
    loc = None
    for i0 in range(0, N // 32, BATCH):
        vs = [_v(i, "ln") for i in range(i0, min(N // 32, i0 + BATCH))]
        for j, v in enumerate(vs):
            i = i0 + j
            sq = b.fmul(v, v, type=ir.F32, name="ln_sq%d" % i)
            loc = sq if loc is None else b.fadd(loc, sq, type=ir.F32, name="ln_acc%d" % i)
    ss = TR.emit_butterfly(b, loc, TR.ROW_BUTTERFLY_MASKS, operation="sum")
    ss = TR.emit_butterfly(b, ss, TR.COLUMN_BUTTERFLY_MASKS, operation="sum")
    mean = b.fmul(ss, O_._cf(b, F32(1.0 / N), "ln_invd"), name="ln_mean")
    KN = O_.emit_constants(b)
    r = O_.emit_rn(b, "rsqrt", b.fadd(mean, O_._cf(b, F32(lay.get("eps", 1e-5)), "ln_eps"), type=I, name="ln_var"), KN, "ln_rs")
    for i0 in range(0, N // 32, BATCH // 2):
        idx = list(range(i0, min(N // 32, i0 + BATCH // 2)))
        vs = [_v(i, "lw") for i in idx]
        gs = [b.f16_to_f32(b.load(bb, b.add(lane, O_._c(b, 32 * i + lay["GN"] // 2, "ln_gi%d" % i), name="ln_gx%d" % i),
                                  width="half", name="ln_gh%d" % i), name="ln_g%d" % i) for i in idx]
        for i, v, g in zip(idx, vs, gs):
            y = b.fmul(b.fmul(v, r, type=I, name="ln_vr%d" % i), g, type=I, name="ln_y%d" % i)
            b.store_at(c, b.add(lane, O_._c(b, 32 * i + lay["XN"] // 2, "ln_xo%d" % i), name="ln_xi%d" % i),
                       b.f32_to_f16_rte(y, name="ln_yh%d" % i), width="half")
    rl, rj = fn.block("ln_reset"), fn.block("ln_reset_done")
    b.br_cond(b.cmp(lane, 1, "lt", name="ln_lane0r"), rl, rj)
    b.at(rl)
    b.atomic_uniform("and", c, O_._c(b, 0, "ln_zero"), name="ln_clr", slot6=lay["CNT"])
    b.br(rj)
    b.at(rj)
    b.br(jn)
    fn.blocks.remove(jn)
    fn.blocks.append(jn)
    b.at(jn)


def _emit_chunk_argmax(fn, b, ir, c, lane, group, lay):
    """THE LM_HEAD WITH ARGMAX PASS 1 FUSED (MM 25.144.7). g17gen's pass 1 has threadgroup t scan logit rows
    [t C, (t + 1) C) and write (max, first index) at PAIRS + 2 t. Here each head threadgroup of chunk t = row0 / C
    fences its stored logits and adds 1 to the chunk's word at CNT + 4 t (op10090, one thread, the returned value
    DISCARDED: cc refuses to read it, 25.144.7), fences again and LOADS the word. A threadgroup that sees a multiple
    of C / R knows every threadgroup of the chunk has stored and fenced, and runs pass 1's scan for t: the same
    loads, combines and butterflies in the same order, so the pair is pass 1's bit for bit. Several threadgroups may
    see it; they write identical pairs. The words only grow (C / R per dispatch); the graph zeroes them with the
    rest of the region per sequence. The multiple-of-384 test ((w >> 7) / 3 by multiply) holds below 12.58 M,
    i.e. 32,768 dispatches per sequence. gen_step reads the pairs unchanged."""
    import g17decodeops as O_
    import g17gen as G_
    I = ir.I32
    C, R0 = lay["argmax_C"], lay["rows"]
    if C != 384 or R0 != 1 or lay["Nout"] % C:
        raise ValueError("argmax_chunks: rows 1 and pass 1's chunk of 384 (the index divides by 384 via (g >> 7) / 3)")

    def div384(x, tag):
        # x / 384 = ((x >> 7) * 43691) >> 17, exact for x >> 7 < 98,304 (checked over every head row)
        return b.shr(b.mul(b.shr(x, O_._c(b, 7, tag + "_s7"), name=tag + "_g7"), O_._c(b, 43691, tag + "_m3"),
                           name=tag + "_gm"), O_._c(b, 17, tag + "_s17"), name=tag + "_q")
    b.barrier("fence_device")
    b.barrier("threadgroup")
    t = div384(group, "am_t")
    widx = b.add(t, O_._c(b, lay["CNT"] // 4, "am_cnt"), name="am_w")
    tpt = b.builtin("thread_position_in_threadgroup", name="am_tpt")
    cl, cj = fn.block("am_count"), fn.block("am_counted")
    b.br_cond(b.cmp(tpt, 1, "lt", name="am_t0"), cl, cj)
    b.at(cl)
    b.atomic_add(c, widx, O_._c(b, 1, "am_one"), name="am_inc")
    b.br(cj)
    b.at(cj)
    b.barrier("threadgroup")
    b.barrier("fence_device")
    seen = b.add(b.load(c, widx, type=I, name="am_ld"), O_._c(b, 0, "am_z"), name="am_seen")
    rem = b.sub(seen, b.mul(div384(seen, "am_d"), O_._c(b, 384, "am_384"), name="am_q384"), name="am_rem")
    mg, jn = fn.block("am_last"), fn.block("am_done")
    full = b.icmp(rem, O_._c(b, 0, "am_z0"), "eq", name="am_full")
    b.br_cond(b.cmp(full, 0, "gt", name="am_is_full"), mg, jn)
    b.at(mg)
    b.barrier("fence_device")
    # pass 1's scan of chunk t (g17gen.build_pass1), the logits at OUT
    rbase = b.add(b.mul(t, O_._c(b, C, "am_C"), name="am_tC"), lane, name="am_rbase")
    base = b.add(rbase, O_._c(b, lay["OUT"] // 4, "am_out"), name="am_base")
    v = i = None
    for k in range(C // 32):
        idx = b.add(base, O_._c(b, 32 * k, "am_o%d" % k), name="am_ix%d" % k) if k else base
        ridx = b.add(rbase, O_._c(b, 32 * k, "am_ro%d" % k), name="am_rx%d" % k) if k else rbase
        x = b.load(c, idx, type=I, name="am_x%d" % k)
        xv = b.fadd(x, O_._cf(b, F32(0.0), "am_zf%d" % k), type=I, name="am_xv%d" % k)
        fi = b.u32_to_f32(ridx, name="am_fi%d" % k)
        if v is None:
            v, i = xv, fi
        else:
            v, i = G_._combine(b, ir, v, i, xv, fi, "am_s%d" % k)
    v, i = G_._lane_reduce(b, ir, v, i, "am_r")
    pb = b.add(b.shl(t, O_._c(b, 1, "am_one2"), name="am_t2"), O_._c(b, lay["PAIRS"] // 4, "am_pb"), name="am_pw")
    b.store_at(c, pb, v)
    b.store_at(c, b.add(pb, O_._c(b, 1, "am_one3"), name="am_pw1"), i)
    b.br(jn)
    fn.blocks.remove(jn)
    fn.blocks.append(jn)
    b.at(jn)


def with_argmax_chunks(lay, C=384):
    """The lm_head writing g17gen's pass-1 pairs itself (_emit_chunk_argmax): binding 0 gains the chunk counters
    (int32 [N / C], all 0 between dispatches) at CNT and the (value, index) pairs at PAIRS, after the logits."""
    if not lay.get("coop"):
        raise ValueError("argmax_chunks: the cooperative class (the counters are on binding 0)")
    G = lay["Nout"] // C
    CNT = _align(lay["OUT"] + 4 * lay["Nout"])
    PAIRS = _align(CNT + 4 * G)
    return dict(lay, argmax_chunks=True, argmax_C=C, argmax_G=G, CNT=CNT, PAIRS=PAIRS,
                c_bytes=max(lay["c_bytes"], _align(PAIRS + 8 * G)))


def with_last_norm(lay, kind, XN):
    """The residual qmv in the cooperative class, computing the next RMSNorm in its last threadgroup: binding 0 is
    the layer region R = [count word at CNT = 0 | h fp32 at OUT = 256 | x fp16 at RES | xn fp16 at XN]; the
    norm's input is the fp16 x (kind "half", after w2 + residual2) or the fp32 h ("float", after wo + residual1);
    g is the norm's fp16 gain at binding 2 [GN] (after B)."""
    N, K = lay["Nout"], lay["Kq"]
    OUT = 256
    RES = _align(OUT + 4 * N)
    GN = _align(lay["B"] + 2 * N * (K // lay["group"]))
    return dict(lay, coop=True, last_norm=kind, CNT=0, OUT=OUT, RES=RES, XN=XN, GN=GN,
                b_bytes=max(lay["b_bytes"], _align(GN + 2 * N)), c_bytes=max(lay["c_bytes"], _align(XN + 2 * N)))


def _hoist_loop_constants(fn, ir):
    """Constants the loop body creates move to the entry block (before its branch), one per (value, type):
    a literal made inside the loop is re-materialised every trip (28 movimm per trip in the q4 kernel,
    MM 25.139.7). Uses of a merged duplicate are rewritten to the kept value."""
    entry = fn.blocks[0]
    keep = {}
    for o in entry.ops:
        if o.kind == "const":
            keep.setdefault((o.args[0].v, o.dest.type), o.dest)
    alias = {}
    for blk in fn.blocks[1:]:
        moved = []
        for o in list(blk.ops):
            if o.kind != "const":
                continue
            key = (o.args[0].v, o.dest.type)
            if key in keep:
                alias[o.dest] = keep[key]
            else:
                keep[key] = o.dest
                moved.append(o)
            blk.ops.remove(o)
        entry.ops[-1:-1] = moved                      # before the entry's branch
    if alias:
        for blk in fn.blocks:
            for o in blk.ops:
                o.args = [alias.get(x, x) if isinstance(x, ir.Value) else x for x in o.args]


def qmv2_reference(lay, x, q, s16, b16):
    N, K, pw, grp, wpt = lay["Nout"], lay["Kq"], lay["per_word"], lay["group"], lay["wpt"]
    E = wpt * pw
    lane_elems = K // 32
    trips = lane_elems // E
    s32, b32 = _from_bf16(s16), _from_bf16(b16)
    x = np.asarray(x, F32)
    y = np.zeros(N, F32)
    for n in range(N):
        acc = np.zeros(32, F32)
        for l in range(32):
            a_ = F32(0.0)
            for k in range(trips):
                e0 = (k * 32 * wpt + l * wpt) * pw if lay.get("coalesced") else l * lane_elems + k * E
                xs = x[e0:e0 + E]
                sx = xs[0]
                for e in range(1, E):
                    sx = F32(sx + xs[e])
                t = F32(F32(q[n, e0]) * xs[0])
                for e in range(1, E):
                    t = _fma32(F32(q[n, e0 + e]), xs[e], t)
                gi = e0 // grp
                a_ = F32(a_ + F32(s32[n, gi] * t))
                a_ = F32(a_ + F32(b32[n, gi] * sx))
            acc[l] = a_
        for m in MASKS:
            acc = (acc + acc[np.arange(32) ^ m]).astype(F32)
        y[n] = acc[0]
    return y


def _fma32v(a, b, c):
    """_fma32 over arrays, vectorised and exact: p = a b is exact in f64 (two fp32 significands, 48 bits); p + c is
    rounded to ODD in f64 (TwoSum's error term decides the sticky bit), then to fp32. Rounding to odd at 53 >= 24 + 2
    bits makes that second rounding the correct single rounding of the exact a b + c (Boldo and Melquiond), so no
    double rounding enters. This replaced a per-element fall back to _fma32's exact rationals, which made
    test_g17attnlongctx the gate's slowest module (405 s); it agrees with _fma32 bit for bit on 370,010 cases
    including 17,173 inexact f64 sums with near-ties, subnormal results and signed zeros."""
    a, b, c = (np.asarray(v, np.float64) for v in (a, b, c))
    p = a * b
    s = p + c
    if not isinstance(s, np.ndarray) or s.ndim == 0:
        return _fma32v_ref(a, b, c)
    # the same TwoSum and odd fix as ever, written in place (the same values; fewer temporaries, and the rare odd fix
    # applied only where it is needed, test_g17simspeed)
    bb = s - p
    err = s - bb
    np.subtract(p, err, out=err)
    np.subtract(c, bb, out=bb)
    err += bb                                  # TwoSum: s + err == p + c exactly
    odd_fix = err != 0
    odd_fix &= (s.view(np.uint64) & 1) == 0
    odd_fix &= np.isfinite(s)
    if odd_fix.any():
        i = np.nonzero(odd_fix)
        s[i] = np.nextafter(s[i], np.where(err[i] > 0, np.inf, -np.inf))
    return s.astype(F32)


def _fma32v_ref(a, b, c):
    """_fma32v as it was written before the in-place form (its equality oracle in test_g17simspeed)."""
    a, b, c = (np.asarray(v, np.float64) for v in (a, b, c))
    p = a * b
    s = p + c
    bb = s - p
    err = (p - (s - bb)) + (c - bb)
    odd_fix = (err != 0) & ((s.view(np.uint64) & 1) == 0) & np.isfinite(s)
    s = np.where(odd_fix, np.nextafter(s, np.where(err > 0, np.inf, -np.inf)), s)
    return s.astype(F32)


def qmv2_reference_fast(lay, x, q, s16, b16):
    """qmv2_reference vectorised over rows, lanes AND trips: the same fp32 operations in the same order.

    Each trip's x sum and dot chain t restart per trip, so they are formed for every trip at once ([rows, trips, 32]
    arrays, rows in blocks to bound memory); only the accumulation acc over trips is serial, and it stays a loop.
    Every element sees exactly the operations _qmv2_reference_fast_trips (the per-trip form it replaced) applies, so
    the outputs are bit-identical (test_g17simspeed checks that, with a control that the check can fail)."""
    N, K, pw, grp, wpt = lay["Nout"], lay["Kq"], lay["per_word"], lay["group"], lay["wpt"]
    E = wpt * pw
    lane_elems = K // 32
    trips = lane_elems // E
    s32, b32 = _from_bf16(s16), _from_bf16(b16)
    x = np.asarray(x, F32)
    lanes = np.arange(32)
    ks = np.arange(trips)[:, None]
    e0 = (ks * 32 * wpt + lanes[None, :] * wpt) * pw if lay.get("coalesced") else lanes[None, :] * lane_elems + ks * E
    # trip k, lane l, element e is ONE reshape of the row (coalesced: k 32E + l E + e; else l K/32 + k E + e), so
    # these are views, not gathers
    xs = x.reshape(trips, 32, E) if lay.get("coalesced") else x.reshape(32, trips, E).transpose(1, 0, 2)   # [T, 32, E]
    sx = xs[..., 0].copy()
    for e in range(1, E):
        sx = (sx + xs[..., e]).astype(F32)
    gi = e0 // grp                                                # [T, 32]
    x64 = xs.astype(np.float64)
    out = np.empty(N, F32)
    # row blocks are independent (every row's operations stay in order inside its block), so they run on a thread
    # pool: numpy's elementwise kernels release the GIL. G17_REF_THREADS=1 runs them in order.
    B = max(1, (1 << 20) // max(K, 1))

    def block(r0):
        r1 = min(N, r0 + B)
        qr = np.asarray(q[r0:r1])
        # float64 holds the (small integer) fields exactly, so every product below is the same value it was in fp32
        # and _fma32v converts nothing twice
        qe = (qr.reshape(-1, trips, 32, E) if lay.get("coalesced") else
              qr.reshape(-1, 32, trips, E).transpose(0, 2, 1, 3)).astype(np.float64)      # [n, T, 32, E]
        if lay.get("chains"):
            fb = 16 // lay["bits"] if lay.get("pool_masks") else 8 // lay["bits"]
            tp = {}
            for e in range(E):
                c = (e % pw) % fb
                qs = qe[..., e] * (2.0 ** (lay["bits"] * c))           # exact: a field times a power of two
                xb = np.broadcast_to(x64[None, :, :, e], qs.shape)
                tp[c] = (qs * xb).astype(F32) if c not in tp else _fma32v(qs, xb, tp[c])
            t = tp[0]
            for c in range(1, fb):
                t = _fma32v(tp[c], np.full(t.shape, 2.0 ** (-lay["bits"] * c), F32), t)
        else:
            t = (qe[..., 0] * x64[None, :, :, 0]).astype(F32)     # exact in f64, then the one fp32 rounding
            for e in range(1, E):
                t = _fma32v(qe[..., e], np.broadcast_to(x64[None, :, :, e], t.shape), t)
        sr, br = s32[r0:r1], b32[r0:r1]
        acc = np.zeros((r1 - r0, 32), F32)
        for k in range(trips):
            if lay.get("epi_fma"):
                acc = _fma32v(sr[:, gi[k]], t[:, k], acc)
                acc = _fma32v(br[:, gi[k]], np.broadcast_to(sx[k][None, :], acc.shape), acc)
                continue
            acc = (acc + (sr[:, gi[k]] * t[:, k]).astype(F32)).astype(F32)
            acc = (acc + (br[:, gi[k]] * sx[k][None, :]).astype(F32)).astype(F32)
        for m in MASKS:
            acc = (acc + acc[:, lanes ^ m]).astype(F32)
        out[r0:r1] = acc[:, 0]
    starts = range(0, N, B)
    nt = min(int(os.environ.get("G17_REF_THREADS", "8")), len(starts))
    if nt > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(nt) as ex:
            list(ex.map(block, starts))
    else:
        for r0 in starts:
            block(r0)
    return out


def _qmv2_reference_fast_trips(lay, x, q, s16, b16):
    """The per-trip form qmv2_reference_fast replaced (kept as its equality oracle): rows and lanes vectorised, one
    trip at a time."""
    N, K, pw, grp, wpt = lay["Nout"], lay["Kq"], lay["per_word"], lay["group"], lay["wpt"]
    E = wpt * pw
    lane_elems = K // 32
    trips = lane_elems // E
    s32, b32 = _from_bf16(s16), _from_bf16(b16)
    x = np.asarray(x, F32)
    acc = np.zeros((N, 32), F32)
    lanes = np.arange(32)
    for k in range(trips):
        e0 = (k * 32 * wpt + lanes * wpt) * pw if lay.get("coalesced") else lanes * lane_elems + k * E
        xs = x[e0[:, None] + np.arange(E)]                      # [32, E]
        sx = xs[:, 0].copy()
        for e in range(1, E):
            sx = (sx + xs[:, e]).astype(F32)
        qe = q[:, e0[:, None] + np.arange(E)].astype(F32)       # [N, 32, E]
        if lay.get("chains"):
            # build_qmv2 lay["chains"]: one chain per field position over the masked (scaled) field and the raw x,
            # folded t = t0 then t = fma(t_pos, 2^-(bits pos), t)
            fb = 16 // lay["bits"] if lay.get("pool_masks") else 8 // lay["bits"]
            tp = {}
            for e in range(E):
                c = (e % pw) % fb
                qs = (qe[:, :, e] * F32(2.0 ** (lay["bits"] * c))).astype(F32)
                xb = np.broadcast_to(xs[None, :, e], qs.shape)
                tp[c] = (qs * xb).astype(F32) if c not in tp else _fma32v(qs, xb, tp[c])
            t = tp[0]
            for c in range(1, fb):
                t = _fma32v(tp[c], np.full(t.shape, 2.0 ** (-lay["bits"] * c), F32), t)
        else:
            t = (qe[:, :, 0] * xs[None, :, 0]).astype(F32)
            for e in range(1, E):
                t = _fma32v(qe[:, :, e], np.broadcast_to(xs[None, :, e], t.shape), t)
        gi = e0 // grp
        if lay.get("epi_fma"):
            acc = _fma32v(s32[:, gi], t, acc)
            acc = _fma32v(b32[:, gi], np.broadcast_to(sx[None, :], acc.shape), acc)
            continue
        acc = (acc + (s32[:, gi] * t).astype(F32)).astype(F32)
        acc = (acc + (b32[:, gi] * sx[None, :]).astype(F32)).astype(F32)
    for m in MASKS:
        acc = (acc + acc[:, lanes ^ m]).astype(F32)
    return acc[:, 0].copy()


def qmv_dq_reference(lay, x, q, s16, b16):
    """build_qmv2_batch with lay["dequant_once"], one vector (MM 25.144.3): per split-K slice (sgs contiguous K ranges),
    lane l keeps ONE fp32 chain over its trips and each trip's elements in order: w = fma(q, s, b) (the field times its
    group's scale is exact, so this is the one rounding of q s + b), acc = fma(w, x, acc) from 0; then the lane
    butterflies; then the slices summed in slice order. A new fp32 order: not qmv2_ksplit_reference's.

    Vectorised as qmv2_reference_fast is: every w is independent, so all of a row block's are formed at once; only the
    acc chain is serial, over (trip, element), with the slices and lanes as array axes; row blocks on a thread pool
    (G17_REF_THREADS). Bit-identical to _qmv_dq_reference_loop, the written-out form (test_g17qmvbatch)."""
    S = lay["sgs"] if lay.get("ksplit") else 1
    N, K, pw, grp, wpt = lay["Nout"], lay["Kq"], lay["per_word"], lay["group"], lay["wpt"]
    if lay.get("swiglu"):
        N = 2 * lay["ffn"]
    E = wpt * pw
    H = K // S
    T = (H // 32) // E
    if not lay.get("coalesced") or H % (32 * E) or grp % E:
        return _qmv_dq_reference_loop(lay, x, q, s16, b16)
    s32, b32 = _from_bf16(s16).astype(F32), _from_bf16(b16).astype(F32)
    # element j H + k 32 E + l E + e is slice j, trip k, lane l, element e: one reshape of the row
    xs = np.asarray(x, F32).astype(np.float64).reshape(S, T, 32, E)
    gi = ((np.arange(S)[:, None, None] * H + np.arange(T)[None, :, None] * 32 * E
           + np.arange(32)[None, None, :] * E) // grp)                     # [S, T, 32]
    lanes = np.arange(32)
    out = np.empty(N, F32)
    B = max(1, (1 << 19) // max(K, 1))

    def block(r0):
        r1 = min(N, r0 + B)
        qe = np.asarray(q[r0:r1]).reshape(-1, S, T, 32, E).astype(np.float64)
        w = _fma32v(qe, s32[r0:r1][:, gi][..., None].astype(np.float64),
                    b32[r0:r1][:, gi][..., None].astype(np.float64))      # [n, S, T, 32, E], each fp32-rounded once
        acc = np.zeros((r1 - r0, S, 32), F32)
        for k in range(T):
            for e in range(E):
                acc = _fma32v(w[:, :, k, :, e], np.broadcast_to(xs[None, :, k, :, e], acc.shape), acc)
        for m in MASKS:
            acc = (acc + acc[..., lanes ^ m]).astype(F32)
        y = acc[:, 0, 0].copy()
        for j in range(1, S):
            y = (y + acc[:, j, 0]).astype(F32)
        out[r0:r1] = y
    starts = range(0, N, B)
    nt = min(int(os.environ.get("G17_REF_THREADS", "8")), len(starts))
    if nt > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(nt) as ex:
            list(ex.map(block, starts))
    else:
        for r0 in starts:
            block(r0)
    return out


def _qmv_dq_reference_loop(lay, x, q, s16, b16):
    """qmv_dq_reference written out slice by slice and trip by trip (its equality oracle, and the fallback for layouts
    the vectorised form does not reshape)."""
    S = lay["sgs"] if lay.get("ksplit") else 1
    N, K, pw, grp, wpt = lay["Nout"], lay["Kq"], lay["per_word"], lay["group"], lay["wpt"]
    if lay.get("swiglu"):
        N = 2 * lay["ffn"]
    E = wpt * pw
    H = K // S
    trips = (H // 32) // E
    s32, b32 = _from_bf16(s16).astype(F32), _from_bf16(b16).astype(F32)
    x = np.asarray(x, F32)
    lanes = np.arange(32)
    y = None
    for j in range(S):
        acc = np.zeros((N, 32), F32)
        for k in range(trips):
            e0 = j * H + (k * 32 * wpt + lanes * wpt) * pw
            gi = e0 // grp
            sc, bi = s32[:, gi], b32[:, gi]                              # [N, 32]
            for e in range(E):
                w = _fma32v(q[:, e0 + e].astype(F32), sc, bi)
                acc = _fma32v(w, np.broadcast_to(x[e0 + e][None, :], acc.shape), acc)
        for m in MASKS:
            acc = (acc + acc[:, lanes ^ m]).astype(F32)
        p = acc[:, 0].copy()
        y = p if y is None else (y + p).astype(F32)
    return y


def qmv_wide_reference(lay, x, q, s16, b16):
    """build_qmv_wide, one vector (MM 25.144.9): row n, K-lane l holds one fp32 chain over its words {l, l+KL, ...}
    in word order and each word's elements in order (w = q*s+b once, acc = fma(x, w, acc) from 0), then the KL lanes
    sum through threadgroup memory in lane order (kl = 0..KL-1). A new fp32 order. Vectorised over rows and lanes;
    bit-identical to _qmv_wide_reference_loop (its oracle)."""
    N, K, pw, grp, bits = lay["Nout"], lay["Kq"], lay["per_word"], lay["group"], lay["bits"]
    KL = lay.get("klanes", 8)
    words, wpg = K // pw, grp // pw
    wpk = words // KL
    s32, b32 = _from_bf16(s16).astype(F32), _from_bf16(b16).astype(F32)
    x = np.asarray(x, F32)
    kl = np.arange(KL)
    acc = np.zeros((N, KL), F32)
    xk = x.astype(np.float64)
    for step in range(wpk):
        wk = kl + KL * step                                   # each lane's word this step
        g = (KL * step + kl) // wpg                           # [KL]
        for e in range(pw):
            elem = wk * pw + e                                # [KL]
            wdq = _fma32v(np.asarray(q[:, elem], np.float64), s32[:, g], b32[:, g])   # [N, KL], q*s+b
            acc = _fma32v(np.broadcast_to(xk[elem][None, :], acc.shape), wdq, acc)    # x*w+acc
    red = acc[:, 0].copy()                               # reduce the KL lanes in lane order (the tg-memory sum)
    for kl in range(1, KL):
        red = (red + acc[:, kl]).astype(F32)
    return red


def _qmv_wide_reference_loop(lay, x, q, s16, b16):
    """qmv_wide_reference written out row by row and lane by lane (its equality oracle)."""
    N, K, pw, grp, bits = lay["Nout"], lay["Kq"], lay["per_word"], lay["group"], lay["bits"]
    KL = lay.get("klanes", 8)
    words, wpg = K // pw, grp // pw
    wpk = words // KL
    s32, b32 = _from_bf16(s16).astype(F32), _from_bf16(b16).astype(F32)
    x = np.asarray(x, F32)
    y = np.empty(N, F32)
    for n in range(N):
        part = np.zeros(KL, F32)
        for l in range(KL):
            acc = F32(0.0)
            for step in range(wpk):
                w = l + KL * step
                g = (KL * step + l) // wpg
                for e in range(pw):
                    elem = w * pw + e
                    wdq = _fma32v(np.float32(q[n, elem]), s32[n, g], b32[n, g])
                    acc = _fma32v(np.float32(x[elem]), wdq, acc)
            part[l] = acc
        red = part[0]
        for kl in range(1, KL):
            red = np.float32(red + part[kl])
        y[n] = red
    return y



def qmv2_ksplit_reference(lay, x, q, s16, b16):
    """build_qmv2 with lay["ksplit"]: qmv2_reference_fast over each of the sgs contiguous K slices (in the coalesced
    layout trip k covers elements [32 E k, 32 E (k+1)), so the trip ranges are K ranges on group boundaries), summed
    in simdgroup order ((p0 + p1) + p2) + ... in fp32."""
    S, K, grp = lay["sgs"], lay["Kq"], lay["group"]
    H = K // S
    part = dict(lay, Kq=H)
    y = None
    for j in range(S):
        k0, k1 = j * H, (j + 1) * H
        p = qmv2_reference_fast(part, np.asarray(x)[k0:k1], q[:, k0:k1], s16[:, k0 // grp:k1 // grp],
                                b16[:, k0 // grp:k1 // grp])
        y = p if y is None else (y + p).astype(F32)
    return y


def quantize(Wf, bits=4, group=64, seed=None):
    """MLX-style affine quantization of fp32 weights [N, K] -> (packed uint32 [N, K*bits/32], scales bf16
    bits [N, K/group] as uint16, biases likewise, and the dequantized fp32 matrix)."""
    N, K = Wf.shape
    g = Wf.reshape(N, K // group, group)
    lo, hi = g.min(-1), g.max(-1)
    levels = (1 << bits) - 1
    scale = np.where(hi > lo, (hi - lo) / levels, 1.0).astype(F32)
    s16 = _to_bf16(scale)
    b16 = _to_bf16(lo.astype(F32))
    s32, b32 = _from_bf16(s16), _from_bf16(b16)
    q = np.clip(np.rint((g - b32[..., None]) / s32[..., None]), 0, levels).astype(np.uint32).reshape(N, K)
    pw = 32 // bits
    qw = q.reshape(N, K // pw, pw)
    packed = np.zeros((N, K // pw), np.uint32)
    for j in range(pw):
        packed |= qw[:, :, j] << np.uint32(bits * j)
    return packed, s16, b16, q


def _to_bf16(x):
    u = np.asarray(x, F32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def _from_bf16(h):
    return (np.asarray(h, np.uint32) << 16).view(F32)


def _fma32(a, b, c):
    """fp32 fma: one rounding of the exact a b + c. The f64 sum is exact unless its exponents are far
    apart; then fall back to exact rationals, so no double rounding enters."""
    a, b, c = float(a), float(b), float(c)
    p = a * b                                  # exact: a is a 4/8-bit integer, b an fp32
    s = p + c
    if s - p == c and s - c == p:
        return F32(s)
    from fractions import Fraction
    exact = Fraction(a) * Fraction(b) + Fraction(c)
    lo = F32(float(exact))
    for cand in (lo, np.nextafter(lo, F32(np.inf)), np.nextafter(lo, F32(-np.inf))):
        pass
    cands = sorted({float(lo), float(np.nextafter(lo, F32(np.inf))), float(np.nextafter(lo, F32(-np.inf)))},
                   key=lambda v: (abs(Fraction(v) - exact), int(np.float32(v).view(np.uint32)) & 1))
    return F32(cands[0])


def qmv_reference(lay, x, q, s16, b16, skip_bias=False):
    """The kernel's exact fp32 order."""
    N, K, pw, grp = lay["Nout"], lay["Kq"], lay["per_word"], lay["group"]
    lane_words = K // pw // 32
    s32, b32 = _from_bf16(s16), _from_bf16(b16)
    x = np.asarray(x, F32)
    y = np.zeros(N, F32)
    for n in range(N):
        acc = np.zeros(32, F32)
        for l in range(32):
            a = F32(0.0)
            for k in range(lane_words):
                e = (l * lane_words + k) * pw
                xs = x[e:e + pw]
                sx = xs[0]
                for j in range(1, pw):
                    sx = F32(sx + xs[j])
                t = F32(F32(q[n, e]) * xs[0])
                for j in range(1, pw):
                    if lay.get("fma"):
                        t = _fma32(F32(q[n, e + j]), xs[j], t)
                    else:
                        t = F32(t + F32(F32(q[n, e + j]) * xs[j]))
                gi = (l * lane_words + k) // (grp // pw)
                a = F32(a + F32(s32[n, gi] * t))
                if not skip_bias:
                    a = F32(a + F32(b32[n, gi] * sx))
            acc[l] = a
        for m in MASKS:
            acc = (acc + acc[np.arange(32) ^ m]).astype(F32)
        y[n] = acc[0]
    return y


def qmv_io(lay, x, packed, s16, b16, want_y, gain=None):
    a, b, c = O._buffers(lay)
    cl = dict(lay, groups=lay["carrier_groups"])
    if not lay.get("nocarrier"):
        ca, cb = O._with_carrier(cl, (a, b, c))
    O._place(a, lay["X"], np.asarray(x, "<f2" if (lay.get("x16") or lay.get("norm") == "half") else "<f4"))
    if lay.get("norm"):
        O._place(b, lay["G"], np.asarray(gain, "<f2"))
    O._place(b, lay["W"], packed.astype("<u4"))
    O._place(b, lay["S"], s16.astype("<u2"))
    O._place(b, lay["B"], b16.astype("<u2"))
    want = bytearray(c)
    O._place(want, lay["OUT"], np.asarray(want_y, "<f2" if lay.get("swiglu") else "<f4"))
    if not lay.get("nocarrier"):
        O._place(want, 0, O.carrier_reference(cl, ca, cb).astype("<f4"))
    return bytes(a), bytes(b), bytes(c), bytes(want)


def case(N, K, bits, rows, seed=7, hoist=False, premask=False, fma=False, nocarrier=False):
    rng = np.random.default_rng(seed)
    Wf = (rng.standard_normal((N, K)) * 0.02).astype(F32)
    x = rng.standard_normal(K).astype(F32)
    lay = qmv_layout(N, K, bits=bits, rows=rows, hoist=hoist, premask=premask, fma=fma, nocarrier=nocarrier)
    packed, s16, b16, q = quantize(Wf, bits=bits)
    return lay, x, packed, s16, b16, q


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("build", "check"))
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--N", type=int, default=2048)
    ap.add_argument("--K", type=int, default=2048)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "g17-qmv-v1")
    ap.add_argument("--fault", default=None)
    ap.add_argument("--hoist", action="store_true")
    ap.add_argument("--premask", action="store_true")
    ap.add_argument("--fma", action="store_true")
    ap.add_argument("--wpt", type=int, default=0, help="words per trip: selects build_qmv2 (1, 2 or 4)")
    ap.add_argument("--interleave", action="store_true")
    ap.add_argument("--lean", action="store_true")
    ap.add_argument("--coalesced", action="store_true")
    ap.add_argument("--shand", action="store_true")
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--a16", action="store_true")
    args = ap.parse_args(argv)
    lay, x, packed, s16, b16, q = case(args.N, args.K, args.bits, args.rows, hoist=args.hoist, premask=args.premask,
                                       fma=args.fma)
    if args.wpt:
        lay = dict(lay, wpt=args.wpt, interleave=args.interleave, lean=args.lean, coalesced=args.coalesced,
                   shand=args.shand, chunk=args.chunk, a16=args.a16)
    prog = build_qmv2(lay) if args.wpt else build_qmv(lay, fault=args.fault)
    import hashlib
    name = "qmv%s_b%d_n%d_k%d_r%d%s%s%s%s" % ("2w%d%s%s%s%s" % (args.wpt, "i" if args.interleave else "", "L" if args.lean else "",
                                                           "C" if args.coalesced else "", "S" if args.shand else "") + ("c%d" % args.chunk if args.chunk != 8 else "")
                                       + ("A" if args.a16 else "")
                                       if args.wpt else "",
                                       args.bits, args.N, args.K, args.rows,
                                       "_h" if args.hoist else "",
                                       "_pm" if args.premask else "", "_fma" if args.fma else "",
                                       "_" + args.fault if args.fault else "")
    print(name, len(prog.code), "bytes", hashlib.sha256(prog.code).hexdigest()[:16], flush=True)
    if args.cmd == "build":
        return 0
    y = qmv2_reference(lay, x, q, s16, b16) if args.wpt else qmv_reference(lay, x, q, s16, b16)
    a, b, c, want = qmv_io(lay, x, packed, s16, b16, y)
    bundle = args.out / name
    if not bundle.exists():
        args.out.mkdir(parents=True, exist_ok=True)
        O.author(bundle, lay, prog, a, b, c, extra={"arm": name})
    worker = args.out / "common-worker"
    if not worker.exists():
        import g17tensorcommonruntime as TCR
        TCR.build_worker(worker)
    outs = O.dispatch(bundle, worker, queries=3, inputs=(a, b, c))
    res = [O.compare_words(o, want) for o in outs]
    got = np.frombuffer(outs[0], "<f4", lay["Nout"], lay["OUT"])
    exact = np.asarray(q, np.float64) * np.repeat(_from_bf16(s16), lay["group"], 1) + np.repeat(_from_bf16(b16), lay["group"], 1)
    ref64 = exact @ np.asarray(x, np.float64)
    print(json.dumps(dict(words_differing=res, max_rel_vs_f64=float(np.max(np.abs(got - ref64)) / np.max(np.abs(ref64))))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
