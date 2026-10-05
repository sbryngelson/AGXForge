#!/usr/bin/env python3
"""Grid-scaled elementwise kernels over M prompt rows, for prefill (MM 25.144.3).

Decode's SwiGLU and residual programs (g17decodeops) are fully unrolled over one row. These take M rows of N
elements with a FIXED small unroll: lane l of threadgroup t handles elements t 32 U + l + 32 i, i < U. The grid
is M N / (32 U) threadgroups of 32 lanes, and every offset is formed in a register, so no 8-bit immediate bounds
M. The plain class: bindings 1 and 2 read, binding 3 written, every stream at offset 0 of its binding (a runtime
binds each at its own offset).

  swiglu_rows    b1 gate fp32 [M N], b2 up fp32 [M N]  ->  b3 act fp16 [M N] = fp16(silu(g) u)
                 in16 (MM 25.183): gate and up fp16 [M N] - the GEMMs' narrowed outputs - widened exactly, then the same
                 fast (MM 25.183): the hardware exp2 (op1272) and the raw reciprocal (op3658) in place of exp2_soft and
                 the corrected reciprocal - not CPU-reproducible, so it is checked by an enclosure (fast_check)
                 (g17decodestep.stage_ffn_swiglu: t = g (-1/ln 2), exp2_soft, + 1, corrected recip, g r, then u)
  residual_rows  b1 y fp32 [M N], b2 x fp16 [M N]      ->  b3 h fp32 [M N] = y + x at 0, and fp16(h) at H16
  fold_rows      b1 split-K partials fp32 [2][M N] (p0 then p1), b2 h fp32 [M N]
                                                        ->  b3 x fp16 [M N] = fp16((p0 + p1) + h), that order
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import g17decodeops as O  # noqa: E402

F32 = np.float32
U = 4
KINDS = ("swiglu", "residual", "fold")


def _align(v, a=256):
    return -(-v // a) * a


def rows_layout(kind, M, N, in16=False, fast=False, out32=False):
    """Buffer sizes and offsets for one kernel over M rows of N elements. in16: swiglu's gate and up are fp16."""
    if kind not in KINDS:
        raise ValueError("rows: kind is one of %r" % (KINDS,))
    n = M * N
    if n % (32 * U):
        raise ValueError("rows: M N must be a multiple of %d" % (32 * U))
    lay = dict(op="rows_" + kind, kind=kind, M=M, N=N, unroll=U, groups=n // (32 * U))
    if in16 and kind != "swiglu":
        raise ValueError("rows: in16 is a swiglu input form")
    if kind == "swiglu":
        w = 2 if in16 else 4
        lay.update(a_bytes=_align(w * n), b_bytes=_align(w * n), c_bytes=_align((4 if out32 else 2) * n))
        if out32:
            # MM 25.207: the act as fp32 OF THE fp16-ROUNDED value (decode's act32), for a reader of fp32 rows (g17qmvw)
            lay["out32"] = True
        if in16:
            lay["in16"] = True
        if fast:
            lay["fast"] = True
    elif kind == "residual":
        H16 = _align(4 * n)
        lay.update(H16=H16, a_bytes=_align(4 * n), b_bytes=_align(2 * n), c_bytes=H16 + _align(2 * n))
    else:
        lay.update(a_bytes=_align(8 * n), b_bytes=_align(4 * n), c_bytes=_align(2 * n))
    return lay


def build_rows(lay):
    from agxforge.g17 import cc, ir
    import g17tensorcommonruntime as TCR
    kind, n = lay["kind"], lay["M"] * lay["N"]
    fn, b, a, bb, c = O._function()
    I = ir.I32
    lane = b.builtin("thread_index_in_simdgroup", name="lane")
    t = b.builtin("threadgroup_position_in_grid", name="t")
    base = b.add(b.mul(t, O._c(b, 32 * U, "tU"), name="tb"), lane, name="base")
    idx = [b.add(base, O._c(b, 32 * i, "o%d" % i), name="e%d" % i) if i else base for i in range(U)]
    if kind == "swiglu":
        K = O.emit_constants(b)
        K2 = O.emit_exp2_constants(b)
        nil2 = O._cf(b, TCR.GELU_NEG_INV_LN2, "neg_inv_ln2")
        one = O._cf(b, F32(1.0), "fone")
        if lay.get("in16"):
            gs = [b.f16_to_f32(b.load(a, e, width="half", name="gh%d" % i), name="g%d" % i) for i, e in enumerate(idx)]
            us = [b.f16_to_f32(b.load(bb, e, width="half", name="uh%d" % i), name="u%d" % i) for i, e in enumerate(idx)]
        else:
            gs = [b.load(a, e, type=I, name="g%d" % i) for i, e in enumerate(idx)]
            us = [b.load(bb, e, type=I, name="u%d" % i) for i, e in enumerate(idx)]
        for i in range(U):
            tt = b.fmul(gs[i], nil2, type=I, name="st%d" % i)
            if lay.get("fast"):
                ex = b.exp2(tt, type=I, name="sx%d" % i)
                den = b.fadd(ex, one, type=I, name="sd%d" % i)
                rc = b.recip(den, type=I, name="sr%d" % i)
            else:
                ex = O.emit_exp2_soft(b, tt, K2, "sx%d" % i)
                den = b.fadd(ex, one, type=I, name="sd%d" % i)
                rc = O.emit_rn(b, "recip", den, K, "sr%d" % i)
            sl = b.fmul(gs[i], rc, type=I, name="silu%d" % i)
            y = b.fmul(sl, us[i], type=I, name="act%d" % i)
            if lay.get("out32"):
                b.store_at(c, idx[i], b.f16_to_f32(b.f32_to_f16_rte(y, name="ah%d" % i), name="aw%d" % i))
            else:
                b.store_at(c, idx[i], b.f32_to_f16_rte(y, name="ah%d" % i), width="half")
    elif kind == "residual":
        ys = [b.load(a, e, type=I, name="y%d" % i) for i, e in enumerate(idx)]
        xs = [b.f16_to_f32(b.load(bb, e, width="half", name="xh%d" % i), name="x%d" % i) for i, e in enumerate(idx)]
        for i in range(U):
            h = b.fadd(ys[i], xs[i], type=I, name="h%d" % i)
            b.store_at(c, idx[i], h)
            b.store_at(c, b.add(idx[i], O._c(b, lay["H16"] // 2, "h16o%d" % i), name="hi%d" % i),
                       b.f32_to_f16_rte(h, name="hh%d" % i), width="half")
    else:
        p0 = [b.load(a, e, type=I, name="p0_%d" % i) for i, e in enumerate(idx)]
        p1 = [b.load(a, b.add(e, O._c(b, n, "p1o%d" % i), name="p1i%d" % i), type=I, name="p1_%d" % i)
              for i, e in enumerate(idx)]
        hs = [b.load(bb, e, type=I, name="hv%d" % i) for i, e in enumerate(idx)]
        for i in range(U):
            s = b.fadd(p0[i], p1[i], type=I, name="s%d" % i)
            v = b.fadd(s, hs[i], type=I, name="v%d" % i)
            b.store_at(c, idx[i], b.f32_to_f16_rte(v, name="xo%d" % i), width="half")
    b.ret()
    ir.verify(fn)
    return cc.compile_function(fn)


def fast_check(got16, g, u, base="e"):
    """The fast SwiGLU's enclosure: the largest distance in fp16 ulps between the kernel's fp16 outputs and
    fp16(silu(g) u) computed in float64. base "2" is the WRONG-BASE control (2^-g in place of e^-g), which a correct
    kernel must sit far outside."""
    g64, u64 = np.asarray(g, np.float64), np.asarray(u, np.float64)
    sig = 1.0 / (1.0 + (np.exp(-g64) if base == "e" else np.exp2(-g64)))
    want = (g64 * sig * u64).astype(np.float16)
    a = np.asarray(got16, np.float16).view(np.int16).astype(np.int64)
    w = want.view(np.int16).astype(np.int64)
    # a monotone integer order of fp16 bit patterns, so a sign-crossing pair counts its true ulp distance
    a = np.where(a < 0, -32768 - a, a); w = np.where(w < 0, -32768 - w, w)
    return int(np.abs(a - w).max())


def rows_reference(lay, ins):
    """{output name: numpy array} for the inputs `ins` (see the module docstring)."""
    import g17decodestep as D
    kind = lay["kind"]
    if kind == "swiglu":
        g, u = ((np.asarray(x, np.float16).astype(F32) if lay.get("in16") else np.asarray(x, F32)) for x in ins)
        return dict(act=D.narrow(D.fmul(D.silu(g), u)).astype(np.float16))
    if kind == "residual":
        y, x = np.asarray(ins[0], F32), np.asarray(ins[1], np.float16).astype(F32)
        h = (y + x).astype(F32)
        return dict(h=h, h16=h.astype(np.float16))
    p, h = np.asarray(ins[0], F32).reshape(2, -1), np.asarray(ins[1], F32)
    return dict(x=((p[0] + p[1]).astype(F32) + h).astype(F32).astype(np.float16))
