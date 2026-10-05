#!/usr/bin/env python3
"""The decode step's three elementwise and row stages as GPU programs compiled by this repository's
compiler (MM 25.136): RMSNorm, RoPE with the 1-row KV-cache append, and SwiGLU. Each is bit-exact
against tools/g17decodestep.py's stage reference by construction, and each replaces a host stub in
tools/g17decodestep_gpu.py (the stub stays as the fallback).

    rmsnorm      one simdgroup; lane l owns elements l, l + 32, ... (a sequential fp32 chain of
                 squares), then the measured row butterfly (1, 8) and column butterfly (2, 4, 16),
                 mean = ss * (1/d), rsqrt(mean + eps) rounded ONCE (below), times the weight, op1016
                 narrowed to half. Two input forms: a half row (attn_norm's x) and an fp32 row
                 (ffn_norm's h)
    rope_append  RoPE on q and k at position kv_len (per output two fmuls and one fadd, no fma), q
                 times log2(e)/sqrt(head_dim), narrowed; k and v appended at cache row kv_len, READ
                 AT RUN TIME from the uint32 at buffer-1 byte KV_LENGTH_BYTE (P9's uniform, MM
                 25.131). The cache is P9's operand layout per head: keys x head_dim halves,
                 row-major, heads stacked (K at KC, V after it), in buffer 3. Four threadgroups,
                 four heads each
    swiglu       a = silu(gate) * up over the FFN width, a separate elementwise body (the gate and
                 up rows are host-summed K partials today, so no GEMM epilogue could read them);
                 eight threadgroups

ROUNDED ONCE, ON THE GPU (the rsqrt the reference fixed, and SiLU's recip). op3850 (rsqrt) and op3658
(recip) are within one ulp of the exact value but NOT correctly rounded (MM 25.136 measures both on the
cc form). Each result is corrected to the correctly rounded value by an exact integer test: for a
candidate a, the midpoint between a and its successor is (2 Ma + 1) 2^(Ea - 151), and 1/sqrt(x) lies
below it exactly when Mx (2 Ma + 1)^2 > 2^(452 - Ex - 2 Ea) (1/g: Mg (2 Ma + 1) > 2^(301 - Eg - Ea)).
The products are formed exactly in 15-bit limbs with 32-bit integer multiplies, every intermediate
below 2^31, and only the bits from 2^60 (2^30) up are kept, which is all the comparison with a power
of two at least 2^60 needs. No midpoint is ever hit exactly (the product would be a power of two, and
2 Ma + 1 is odd and at least 2^24), so the test never ties. Given a seed within one ulp, the result is
the correctly rounded value; the seed's error is measured (the probe program).

exp2 IS NOT CORRECTED, AND SILU'S REFERENCE CHANGED. op1272 is within one ulp both ways (172 of 264
exact, docs/archive/g17-settle-20260923.md item 2) and no cheap exact test decides 2^t against a
midpoint, so no reference that rounds exp2 once can be matched bit for bit (the GELU register step
passes by an enclosure for exactly this reason, MM 25.109). SiLU's exp2 is now `exp2_soft`, a fixed
sequence of RNE fp32 and integer operations that the GPU reproduces by construction: a clamp to
[-125, 125], t + 1.5 * 2^23 (whose low bits are rint(t)), the exact reduction f = t - n, a degree-7
Taylor polynomial in Horner form, and n added to the exponent field. It is within one ulp of 2^t.

THE TRANSPORT. The common worker admits tensor programs only (ABI v5, `execution.tensor`, system
registers 130 and 156), so each program opens with a CARRIER: one 16 x 16 x 16 MMA per threadgroup
from buffer-2 tiles into its own region of buffer 3, whose output is checked bit for bit against the
MMA reference as a dispatch-level positive control. Buffers: 1 = A (inputs), 2 = B (weights and the
carrier tiles), 3 = C (outputs; the KV cache), sized by gemm_generic's rule from M, N and K.
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

import g17decodestep as D                  # noqa: E402

F32 = np.float32
M15 = 0x7FFF
KV_LENGTH_BYTE = 6144                        # runtime.KV_LENGTH_BYTE (P9's uniform); asserted below
CARRIER = 16                                 # the carrier MMA's M, N and K per threadgroup
LOG2E = D.LOG2E
EXP2_MAGIC, EXP2_LO, EXP2_HI, EXP2_COEF = D.EXP2_MAGIC, D.EXP2_LO, D.EXP2_HI, D.EXP2_COEF


def _bits(x):
    return int(np.asarray(x, F32).reshape(()).view(np.uint32))


# ---------------------------------------------------------------------------------------------------
# host models (numpy, vectorised), instruction for instruction what the programs emit

def correct_rsqrt(x, y0):
    """The program's correction of a seed (within one ulp) to the correctly rounded 1/sqrt(x)."""
    return D.round_once(D.mid_below_rsqrt, x, y0)


def correct_recip(g, y0):
    return D.round_once(D.mid_below_recip, g, y0)


rsqrt_rn, recip_rn, exp2_soft = D.rsqrt, D.recip, D.exp2_soft


# ---------------------------------------------------------------------------------------------------
# IR emission

def _c(b, v, name):
    from agxforge.g17 import ir
    return b.const(int(v) & 0xFFFFFFFF, type=ir.I32, name=name)


def _cf(b, v, name):
    from agxforge.g17 import ir
    return b.const(_bits(v), type=ir.F32, name=name)


def _and(b, x, m, name):
    return getattr(b, "and")(x, m, name=name)


def _emit_fields(b, bits, K, tag):
    """(M, E): the 24-bit significand with its hidden bit, and the biased exponent."""
    m = getattr(b, "or")(_and(b, bits, K["mant"], tag + "_m23"), K["hidden"], name=tag + "_m")
    e = _and(b, b.shr(bits, K["s23"], name=tag + "_esh"), K["ff"], name=tag + "_e")
    return m, e


def _emit_mid_top(b, kind, x_m, x_e, cand, K, tag):
    """1 where f(x) < mid(cand, cand+), as the 0/1 value of an unsigned compare (see the models)."""
    a_m, a_e = _emit_fields(b, cand, K, tag + "a")
    q = b.add(b.shl(a_m, K["s1"], name=tag + "_2m"), K["one"], name=tag + "_q")
    q0 = _and(b, q, K["m15"], tag + "_q0")
    q1 = b.shr(q, K["s15"], name=tag + "_q1")
    m0 = _and(b, x_m, K["m15"], tag + "_x0")
    m1 = b.shr(x_m, K["s15"], name=tag + "_x1")
    if kind == "rsqrt":
        t0 = b.mul(q0, q0, name=tag + "_t0")
        s0 = _and(b, t0, K["m15"], tag + "_s0")
        c = b.shr(t0, K["s15"], name=tag + "_c0")
        t1 = b.add(b.shl(b.mul(q0, q1, name=tag + "_q01"), K["s1"], name=tag + "_2q01"), c, name=tag + "_t1")
        s1 = _and(b, t1, K["m15"], tag + "_s1")
        c = b.shr(t1, K["s15"], name=tag + "_c1")
        t2 = b.add(b.mul(q1, q1, name=tag + "_q11"), c, name=tag + "_t2")
        s2 = _and(b, t2, K["m15"], tag + "_s2")
        s3 = b.shr(t2, K["s15"], name=tag + "_s3")
        k = b.shr(b.mul(s0, m0, name=tag + "_p00"), K["s15"], name=tag + "_k0")
        col = b.add(b.add(b.mul(s1, m0, name=tag + "_p10"), b.mul(s0, m1, name=tag + "_p01"), name=tag + "_c1s"), k, name=tag + "_col1")
        k = b.shr(col, K["s15"], name=tag + "_k1")
        col = b.add(b.add(b.mul(s2, m0, name=tag + "_p20"), b.mul(s1, m1, name=tag + "_p11"), name=tag + "_c2s"), k, name=tag + "_col2")
        k = b.shr(col, K["s15"], name=tag + "_k2")
        col = b.add(b.add(b.mul(s3, m0, name=tag + "_p30"), b.mul(s2, m1, name=tag + "_p21"), name=tag + "_c3s"), k, name=tag + "_col3")
        k = b.shr(col, K["s15"], name=tag + "_k3")
        top = b.add(b.mul(s3, m1, name=tag + "_p31"), k, name=tag + "_top")
        # sh = 392 - Ex - 2 Ea
        sh = b.sub(b.sub(K["c392"], x_e, name=tag + "_sh1"), b.shl(a_e, K["s1"], name=tag + "_2ea"), name=tag + "_sh")
    else:
        k = b.shr(b.mul(q0, m0, name=tag + "_p00"), K["s15"], name=tag + "_k0")
        col = b.add(b.add(b.mul(q1, m0, name=tag + "_p10"), b.mul(q0, m1, name=tag + "_p01"), name=tag + "_c1s"), k, name=tag + "_col1")
        k = b.shr(col, K["s15"], name=tag + "_k1")
        top = b.add(b.mul(q1, m1, name=tag + "_p11"), k, name=tag + "_top")
        # sh = 271 - Eg - Ea
        sh = b.sub(b.sub(K["c271"], x_e, name=tag + "_sh1"), a_e, name=tag + "_sh")
    bound = b.shl(K["one"], sh, name=tag + "_bound")
    below = b.icmp(top, bound, rel="ult", name=tag + "_below")        # 1 where P < 2^S
    return below


def emit_constants(b):
    from agxforge.g17 import ir
    return dict(mant=_c(b, 0x7FFFFF, "k_mant"), hidden=_c(b, 0x800000, "k_hidden"), s23=_c(b, 23, "k_s23"),
                ff=_c(b, 0xFF, "k_ff"), s1=_c(b, 1, "k_s1"), one=_c(b, 1, "k_one"), m15=_c(b, M15, "k_m15"),
                s15=_c(b, 15, "k_s15"), c392=_c(b, 392, "k_c392"), c271=_c(b, 271, "k_c271"),
                zero=_c(b, 0, "k_zero"))


def emit_rn(b, kind, x, K, tag):
    """The correctly rounded rsqrt(x) or recip(x): the hardware seed, then the exact midpoint test."""
    from agxforge.g17 import ir
    y0 = (b.rsqrt if kind == "rsqrt" else b.recip)(x, type=ir.I32, name=tag + "_seed")
    x_m, x_e = _emit_fields(b, x, K, tag + "x")
    ym1 = b.sub(y0, K["one"], name=tag + "_ym1")
    below_lo = _emit_mid_top(b, kind, x_m, x_e, ym1, K, tag + "L")      # 1: f >= mid(y0-1, y0)
    below_hi = _emit_mid_top(b, kind, x_m, x_e, y0, K, tag + "H")       # 1: f > mid(y0, y0+1)
    # result = y0 - 1 + below_lo + below_hi: below_hi implies below_lo (the midpoints are ordered)
    return b.add(b.add(ym1, below_lo, name=tag + "_lo"), below_hi, name=tag + "_rn")


def emit_exp2_soft(b, t, K2, tag):
    from agxforge.g17 import ir
    I = ir.I32                                  # float bits carried in integer-typed values
    t = b.fmin(b.fmax(t, K2["lo"], type=I, name=tag + "_cl"), K2["hi"], type=I, name=tag + "_ch")
    s = b.fadd(t, K2["magic"], type=I, name=tag + "_s")
    nf = b.fadd(s, K2["nmagic"], type=I, name=tag + "_nf")
    f = b.fadd(t, b.fneg(nf, type=I, name=tag + "_nnf"), type=I, name=tag + "_f")
    p = K2["c7"]
    for i in range(6, -1, -1):
        p = b.fadd(b.fmul(p, f, type=I, name="%s_m%d" % (tag, i)), K2["c%d" % i], type=I, name="%s_p%d" % (tag, i))
    n = b.sub(s, K2["mbits"], name=tag + "_n")
    return b.add(p, b.shl(n, K2["s23"], name=tag + "_n23"), name=tag + "_e")


def emit_exp2_constants(b):
    K2 = dict(lo=_cf(b, EXP2_LO, "e_lo"), hi=_cf(b, EXP2_HI, "e_hi"), magic=_cf(b, EXP2_MAGIC, "e_magic"),
              nmagic=_cf(b, -EXP2_MAGIC, "e_nmagic"), mbits=_c(b, 0x4B400000, "e_mbits"), s23=_c(b, 23, "e_s23"))
    for i, cf in enumerate(EXP2_COEF):
        K2["c%d" % i] = _cf(b, cf, "e_c%d" % i)
    return K2


def _function(name="tensor_gemm_generic_runtime_demo"):
    from agxforge.g17 import ir
    a = ir.Buffer("A", 1, elem=ir.F16)
    bb = ir.Buffer("B", 2, elem=ir.F16)
    c = ir.Buffer("C", 3, elem=ir.F32)
    fn = ir.Function(name, [a, bb, c])
    return fn, ir.Builder(fn, fn.block("entry")), a, bb, c


def _carrier(b, a, bb, c, lay):
    """One 16 x 16 x 16 MMA per threadgroup, at offset 0 of every buffer (the grid split applies no
    operand offsets): A = buffer-1 rows [0, 16 G) of 16 halves, B = buffer-2's first 16 x 16 halves, D
    into buffer-3 rows [0, 16 G) of 16 floats. This path reads A from buffer 1 whatever buffer the IR
    names (the probe's first carrier named buffer 2 and read buffer 1, MM 25.136.3). Runs first: no
    scalar is live across a tensor body."""
    b.tensor_matmul(a, bb, c, M=CARRIER * lay["groups"], N=CARRIER, K=CARRIER, threadgroups=lay["groups"])


def _transport(groups, a_bytes, b_bytes, c_bytes, N=256):
    """gemm_generic's buffer rule (the worker's): A = M K 2, B = K N 2, C = M N 4, M a multiple of 16
    groups and at most 1,024 rows per group."""
    unit = 16 * groups
    M = max(unit, -(-(-(-c_bytes // (4 * N))) // unit) * unit)
    if M // groups > 1024:
        raise ValueError("transport: %d rows per threadgroup exceed the worker's 1,024" % (M // groups))
    K = max(16, -(-(-(-a_bytes // (2 * M))) // 16) * 16)
    if K * N * 2 < b_bytes:
        K = max(K, -(-(-(-b_bytes // (2 * N))) // 16) * 16)
    # gemm_generic's class table refuses K above 256 without a K loop (the wide_n rule at N 256), so
    # grow M instead until A and B fit with K <= 256
    while K > 256:
        M += unit
        if M // groups > 1024:
            raise ValueError("transport: no M x N x K with K <= 256 holds these buffers")
        K = max(16, -(-(-(-a_bytes // (2 * M))) // 16) * 16, -(-(-(-b_bytes // (2 * N))) // 16) * 16)
    return dict(M=M, N=N, K=K, a_bytes=M * K * 2, b_bytes=K * N * 2, c_bytes=M * N * 4)


def _align(v, a=256):
    return -(-v // a) * a


def _carrier_bytes(groups):
    """Bytes the carrier occupies at the start of buffer 3 (16 G rows of 16 floats); its A tile at the
    start of buffer 1 (16 G rows of 16 halves) and its B tile in buffer 2 are smaller, so one offset
    clears all three."""
    return _align(4 * CARRIER * CARRIER * groups)


# ---- RMSNorm ---------------------------------------------------------------------------------------

def rmsnorm_layout(d, in_dtype):
    if d % 32:
        raise ValueError("rmsnorm: the row must fill 32 lanes")
    if in_dtype not in ("half", "float"):
        raise ValueError("rmsnorm: the input row is half or float")
    groups = 1
    X = G = OUT = _carrier_bytes(groups)        # every buffer: the carrier's tile, then the row
    t = _transport(groups, X + (2 if in_dtype == "half" else 4) * d, G + 2 * d, OUT + 2 * d, N=128)
    return dict(op="rmsnorm", d=d, in_dtype=in_dtype, groups=groups, X=X, G=G, OUT=OUT, **t)


def build_rmsnorm(lay, eps):
    """attn_norm / ffn_norm (the reference's rmsnorm, g17decodestep.rmsnorm)."""
    from agxforge.g17 import cc, ir, tensorreduce as TR
    d = lay["d"]
    fn, b, a, bb, c = _function()
    _carrier(b, a, bb, c, lay)
    lane = b.builtin("thread_index_in_simdgroup", name="lane")
    group = b.builtin("threadgroup_position_in_grid", name="group")      # the row (one row: 0)
    row = b.mul(group, _c(b, d, "row_elems"), name="row_base")
    base = b.add(row, lane, name="elem_base")

    xunit = 2 if lay["in_dtype"] == "half" else 4

    def value(i, tag):
        idx = b.add(base, _c(b, 32 * i, "%s_off%d" % (tag, i)), name="%s_i%d" % (tag, i))
        src = b.add(idx, _c(b, lay["X"] // xunit, "%s_xb%d" % (tag, i)), name="%s_xi%d" % (tag, i))
        if lay["in_dtype"] == "half":
            return b.f16_to_f32(b.load(a, src, width="half", name="%s_h%d" % (tag, i)), name="%s_v%d" % (tag, i)), idx
        return b.load(a, src, type=ir.I32, name="%s_v%d" % (tag, i)), idx

    local = None
    for i in range(d // 32):
        v, _ = value(i, "sq")
        sq = b.fmul(v, v, type=ir.F32, name="sq%d" % i)
        local = sq if local is None else b.fadd(local, sq, type=ir.F32, name="acc%d" % i)
    ss = TR.emit_butterfly(b, local, TR.ROW_BUTTERFLY_MASKS, operation="sum")
    ss = TR.emit_butterfly(b, ss, TR.COLUMN_BUTTERFLY_MASKS, operation="sum")
    mean = b.fmul(ss, _cf(b, F32(1.0 / d), "inv_d"), name="mean")
    K = emit_constants(b)
    r = emit_rn(b, "rsqrt", b.fadd(mean, _cf(b, F32(eps), "eps"), type=ir.I32, name="var"), K, "rs")
    for i in range(d // 32):
        v, idx = value(i, "o")
        g = b.f16_to_f32(b.load(bb, b.add(idx, _c(b, lay["G"] // 2, "g_base%d" % i), name="g_i%d" % i),
                                width="half", name="g_h%d" % i), name="g%d" % i)
        y = b.fmul(b.fmul(v, r, name="vr%d" % i), g, name="y%d" % i)
        b.store_at(c, b.add(idx, _c(b, lay["OUT"] // 2, "out_base%d" % i), name="out_i%d" % i),
                   b.f32_to_f16_rte(y, name="yh%d" % i), width="half")
    b.ret()
    ir.verify(fn)
    return cc.compile_function(fn)


def build_rmsnorm_part(lay, eps, part):
    """DIAGNOSIS ONLY (MM 25.136.6): one piece of build_rmsnorm's straight-line program, to time the
    pieces apart. "carrier": the carrier and ret; "sum": carrier, sum pass, rsqrt, and each lane's
    narrowed r stored at out[lane] (so nothing is dead); "scale": carrier and the scale-and-store pass
    with r = 1.0 (no sum, no rsqrt). Not bit-exact to anything but its own model; never routed."""
    from agxforge.g17 import cc, ir, tensorreduce as TR
    if part not in ("carrier", "sum", "scale"):
        raise ValueError(part)
    d = lay["d"]
    fn, b, a, bb, c = _function()
    _carrier(b, a, bb, c, lay)
    if part == "carrier":
        b.ret()
        ir.verify(fn)
        return cc.compile_function(fn)
    lane = b.builtin("thread_index_in_simdgroup", name="lane")
    base = lane
    xunit = 2 if lay["in_dtype"] == "half" else 4

    def value(i, tag):
        idx = b.add(base, _c(b, 32 * i, "%s_off%d" % (tag, i)), name="%s_i%d" % (tag, i))
        src = b.add(idx, _c(b, lay["X"] // xunit, "%s_xb%d" % (tag, i)), name="%s_xi%d" % (tag, i))
        if lay["in_dtype"] == "half":
            return b.f16_to_f32(b.load(a, src, width="half", name="%s_h%d" % (tag, i)), name="%s_v%d" % (tag, i)), idx
        return b.load(a, src, type=ir.I32, name="%s_v%d" % (tag, i)), idx

    if part == "sum":
        local = None
        for i in range(d // 32):
            v, _ = value(i, "sq")
            sq = b.fmul(v, v, type=ir.F32, name="sq%d" % i)
            local = sq if local is None else b.fadd(local, sq, type=ir.F32, name="acc%d" % i)
        ss = TR.emit_butterfly(b, local, TR.ROW_BUTTERFLY_MASKS, operation="sum")
        ss = TR.emit_butterfly(b, ss, TR.COLUMN_BUTTERFLY_MASKS, operation="sum")
        mean = b.fmul(ss, _cf(b, F32(1.0 / d), "inv_d"), name="mean")
        K = emit_constants(b)
        r = emit_rn(b, "rsqrt", b.fadd(mean, _cf(b, F32(eps), "eps"), type=ir.I32, name="var"), K, "rs")
        b.store_at(c, b.add(lane, _c(b, lay["OUT"] // 2, "out_base"), name="out_i"),
                   b.f32_to_f16_rte(r, name="rh"), width="half")
    else:
        r = _cf(b, F32(1.0), "r_one")
        for i in range(d // 32):
            v, idx = value(i, "o")
            g = b.f16_to_f32(b.load(bb, b.add(idx, _c(b, lay["G"] // 2, "g_base%d" % i), name="g_i%d" % i),
                                    width="half", name="g_h%d" % i), name="g%d" % i)
            y = b.fmul(b.fmul(v, r, name="vr%d" % i), g, name="y%d" % i)
            b.store_at(c, b.add(idx, _c(b, lay["OUT"] // 2, "out_base%d" % i), name="out_i%d" % i),
                       b.f32_to_f16_rte(y, name="yh%d" % i), width="half")
    b.ret()
    ir.verify(fn)
    return cc.compile_function(fn)


# ---- RMSNorm on counted loops (MM 25.136.6) -------------------------------------------------------

def rmsnorm_loop_layout(d, in_dtype, groups=1, unroll=8, hoist=False):
    """rmsnorm_layout for build_rmsnorm_loop: `groups` threadgroups each REDUNDANTLY form the whole
    sum of squares (the reference's order, lane l over i = 0..d/32-1, then the butterflies) and each
    scales and stores its own d/groups slice. `unroll` elements per lane per loop trip."""
    if d % 32:
        raise ValueError("rmsnorm: the row must fill 32 lanes")
    if in_dtype not in ("half", "float"):
        raise ValueError("rmsnorm: the input row is half or float")
    per = d // 32                                   # elements per lane in the sum pass
    if groups < 1 or per % groups:
        raise ValueError("rmsnorm_loop: %d elements per lane do not split over %d threadgroups" % (per, groups))
    slice_ = per // groups                          # elements per lane in the scale pass
    if unroll < 1 or per % unroll or slice_ % unroll and unroll % slice_:
        raise ValueError("rmsnorm_loop: unroll %d does not tile %d / %d elements per lane" % (unroll, per, slice_))
    # each buffer: the carrier's tile, then the row. The carrier's A tile is 16 G rows of 16 halves, its B
    # tile one 16 x 16 (so the weight row sits at 512 whatever G), its D tile 16 G rows of 16 floats
    X, G, OUT = _align(2 * CARRIER * CARRIER * groups), _align(2 * CARRIER * CARRIER), _carrier_bytes(groups)
    t = _transport(groups, X + (2 if in_dtype == "half" else 4) * d, G + 2 * d, OUT + 2 * d, N=128)
    return dict(op="rmsnorm_loop", d=d, in_dtype=in_dtype, groups=groups, unroll=unroll, X=X, G=G, OUT=OUT,
                **({"hoist": True} if hoist else {}), **t)


def build_rmsnorm_loop(lay, eps, fault=None, _into=None):
    """attn_norm / ffn_norm with the SAME arithmetic as build_rmsnorm, in the same order, on small code.

    The straight-line program is 19.9 KB (half) and runs once per dispatch in one simdgroup, so every
    byte is a cold instruction fetch (MM 25.136.6 measures it). Here:
      * sum pass: lane l's chain over i = 0 .. d/32-1 at element l + 32 i, `unroll` elements per trip.
        The first `unroll` elements are peeled straight-line (acc = sq0, then + sq1 ...), exactly the
        reference's first additions, so no +0.0 seed enters the chain; the counted loop adds the rest
        in ascending i. Then the row and column butterflies, mean, eps, the corrected rsqrt: unchanged.
      * scale pass: threadgroup t scales elements [t s, (t + 1) s) of each lane's d/32 (s = d/32/groups),
        in a counted loop of s/unroll trips (straight-line when that is one trip). Elementwise, so the
        split changes no value.
    Every threadgroup computes the identical sum (same inputs, same order, same instructions).
    `fault` is for the failing control only: "drop_last" (unroll 1) runs the sum loop one fewer trip,
    so element l + 32 (d/32 - 1) never enters lane l's chain."""
    from agxforge.g17 import cc, ir, tensorreduce as TR
    d, U, Gn = lay["d"], lay["unroll"], lay["groups"]
    per = d // 32
    s = per // Gn
    if _into is not None:
        # ONE ARM OF A MULTI-OP PROGRAM (g17qmv.build_uber): the caller's function and block, labels prefixed,
        # no carrier, a branch to its exit instead of the return
        fn, b, a, bb, c = _into["fn"], _into["b"], _into["a"], _into["bb"], _into["c"]
    else:
        fn, b, a, bb, c = _function()
        if not lay.get("nocarrier"):
            # the carrier exists only for the common worker's admission; a runtime that binds the pipeline
            # itself needs none (as build_qmv2's nocarrier)
            _carrier(b, a, bb, c, lay)
    P = _into["prefix"] if _into is not None else ""
    xunit = 2 if lay["in_dtype"] == "half" else 4
    lane = b.builtin("thread_index_in_simdgroup", name="lane")
    group = b.builtin("threadgroup_position_in_grid", name="group")
    k32 = _c(b, 32, "k32")

    def load_v(p, tag):
        if lay["in_dtype"] == "half":
            return b.f16_to_f32(b.load(a, p, width="half", name=tag + "_h"), name=tag + "_v")
        return b.load(a, p, type=ir.I32, name=tag + "_v")

    def sum_body(p, acc, tag, first=False):
        """`unroll` elements from pointer p (element index into buffer 1, in xunit units), each loaded,
        squared and added in ascending i - the straight-line program's per-element order, which keeps
        the narrow registers (12; load indices and cvt destinations need them) to a handful.
        -> (pointer past them, acc)."""
        q = p
        if lay.get("hoist"):
            # HOISTED: every load of the trip, then every square, then the adds in the SAME ascending
            # order. The loads and the squares are independent of the chain, so the values are
            # bit-identical; only the fadd fold is ordered (Piece A, MM 25.136.6).
            raw = []
            for u in range(U):
                if u:
                    q = b.add(q, k32, name="%s_p%d" % (tag, u))
                if lay["in_dtype"] == "half":
                    raw.append(b.load(a, q, width="half", name="%s_%d_h" % (tag, u)))
                else:
                    raw.append(b.load(a, q, type=ir.I32, name="%s_%d_v" % (tag, u)))
            # the half input's conversions come after ALL the trip's loads, so none waits on its own
            vs = ([b.f16_to_f32(h, name="%s_%d_v" % (tag, u)) for u, h in enumerate(raw)]
                  if lay["in_dtype"] == "half" else raw)
            sqs = [b.fmul(v, v, type=ir.F32, name="%s_sq%d" % (tag, u)) for u, v in enumerate(vs)]
            for u, sq in enumerate(sqs):
                acc = sq if (first and u == 0) else b.fadd(acc, sq, type=ir.F32, name="%s_acc%d" % (tag, u))
            return b.add(q, k32, name=tag + "_next"), acc
        for u in range(U):
            if u:
                q = b.add(q, k32, name="%s_p%d" % (tag, u))
            v = load_v(q, "%s_%d" % (tag, u))
            sq = b.fmul(v, v, type=ir.F32, name="%s_sq%d" % (tag, u))
            acc = sq if (first and u == 0) else b.fadd(acc, sq, type=ir.F32, name="%s_acc%d" % (tag, u))
        return b.add(q, k32, name=tag + "_next"), acc

    # ---- sum pass
    x0 = b.add(lane, _c(b, lay["X"] // xunit, "x_base"), name="x0")
    if fault not in (None, "drop_last"):
        raise ValueError("fault: None or 'drop_last'")
    if fault and U != 1:
        raise ValueError("drop_last is built at unroll 1 (the loop runs one fewer trip)")
    trips = per // U - 1 - (1 if fault else 0)
    p, acc = sum_body(x0, None, "s0", first=True)
    if trips > 0:
        hdr, post = fn.block(P + "sum_loop"), fn.block(P + "sum_done")
        k0 = _c(b, 0, "k0")
        b.br(hdr)
        b.at(hdr)
        k = b.phi(k0, name="k")
        pp = b.phi(p, name="sp")
        pa = b.phi(acc, type=ir.F32, name="sacc")
        np_, na = sum_body(pp, pa, "s")
        kn = b.add(k, ir.Imm(1), name="k_next")
        ir.Builder.phi_latch(k, kn)
        ir.Builder.phi_latch(pp, np_)
        ir.Builder.phi_latch(pa, na)
        b.br_cond(b.cmp(kn, trips, "lt", name="sum_more"), hdr, post)
        b.at(post)
        acc = na
    ss =TR.emit_butterfly(b, acc, TR.ROW_BUTTERFLY_MASKS, operation="sum")
    ss = TR.emit_butterfly(b, ss, TR.COLUMN_BUTTERFLY_MASKS, operation="sum")
    mean = b.fmul(ss, _cf(b, F32(1.0 / d), "inv_d"), name="mean")
    K = emit_constants(b)
    r = emit_rn(b, "rsqrt", b.fadd(mean, _cf(b, F32(eps), "eps"), type=ir.I32, name="var"), K, "rs")

    # ---- scale pass over this threadgroup's slice: element e = lane + 32 (t s + j), j = 0 .. s-1
    e0 = b.add(lane, b.mul(group, _c(b, 32 * s, "slice_elems"), name="slice0"), name="e0")
    kx, kg, ko = _c(b, lay["X"] // xunit, "xo_base"), _c(b, lay["G"] // 2, "g_base"), _c(b, lay["OUT"] // 2, "out_base")

    def scale_body(e, tag, n, advance=True):
        """n elements from element index e (same per-element order as build_rmsnorm). -> e past them, formed
        only when `advance` (a single trip never reads it: cc keeps an unread op, MM 25.144.6)."""
        for u in range(n):
            if u:
                e = b.add(e, k32, name="%s_e%d" % (tag, u))
            v = load_v(b.add(e, kx, name="%s_xi%d" % (tag, u)), "%s_x%d" % (tag, u))
            g = b.f16_to_f32(b.load(bb, b.add(e, kg, name="%s_gi%d" % (tag, u)), width="half", name="%s_gh%d" % (tag, u)),
                             name="%s_g%d" % (tag, u))
            y = b.fmul(b.fmul(v, r, name="%s_vr%d" % (tag, u)), g, name="%s_y%d" % (tag, u))
            b.store_at(c, b.add(e, ko, name="%s_oi%d" % (tag, u)), b.f32_to_f16_rte(y, name="%s_yh%d" % (tag, u)),
                       width="half")
        return b.add(e, k32, name=tag + "_en") if advance else None

    n = min(U, s)
    otrips = s // n
    if otrips == 1:
        scale_body(e0, "o", n, advance=False)
    else:
        hdr2, post2 = fn.block(P + "scale_loop"), fn.block(P + "scale_done")
        j0 = _c(b, 0, "j0")
        b.br(hdr2)
        b.at(hdr2)
        j = b.phi(j0, name="j")
        pe = b.phi(e0, name="pe")
        ne = scale_body(pe, "o", n)
        jn = b.add(j, ir.Imm(1), name="j_next")
        ir.Builder.phi_latch(j, jn)
        ir.Builder.phi_latch(pe, ne)
        b.br_cond(b.cmp(jn, otrips, "lt", name="scale_more"), hdr2, post2)
        b.at(post2)
    if _into is not None:
        b.br(_into["exit"])
        return None
    b.ret()
    ir.verify(fn)
    return cc.compile_function(fn)


def rmsnorm_drop_last_model(v, g, spec):
    """The failing control's value (build_rmsnorm_loop fault="drop_last"): the reference rmsnorm with
    element l + 32 (d/32 - 1) left out of lane l's chain, everything else identical."""
    from agxforge.g17 import tensorreduce as TR
    v = np.asarray(v, F32)
    d = v.size
    sq = D.fmul(v, v).reshape(d // 32, 32)
    local = sq[0].copy()
    for i in range(1, d // 32 - 1):
        local = D.fadd(local, sq[i])
    lanes = TR.butterfly([float(x) for x in local], TR.ROW_BUTTERFLY_MASKS, "sum")
    lanes = TR.butterfly(list(lanes), TR.COLUMN_BUTTERFLY_MASKS, "sum")
    r = D.rsqrt(D.fadd(D.fmul(F32(lanes[0]), F32(1.0 / d)), F32(spec.norm_eps)))
    return D.narrow(D.fmul(D.fmul(v, r), np.asarray(g, F32)), spec.storage)


# ---- RoPE + append ---------------------------------------------------------------------------------

# THE ATTENTION GRID's CACHE LAYOUT (MM 25.138.2): per head 131,072 bytes (65,536 halves); K row-major (key x 128),
# V in 16 x 16 tiles (block j, slice sl at element 2048 j + 256 sl, row key % 16, column dim % 16), as
# g17tensorcommonruntime._grid_buffers places them in buffers 3 and 2
GRID_HEAD_ELEMS = 65536


def rope_layout(n_heads, head_dim, capacity, groups=4, cache="p9", kv_heads=None, chained=False):
    if cache == "flash":
        return _rope_flash_layout(n_heads, head_dim, capacity, groups, kv_heads)
    """`groups` threadgroups (default 4, the pinned MM 25.136 program: four heads each). Up to n_heads,
    each takes n_heads/groups whole heads; past it (MM 25.136.6), each takes one head and a power-of-two
    share of its head_dim/64 lane-pair chunks. Elementwise: the split changes no value.
    cache "grid" (MM 25.138.2) writes the new row straight into the attention grid's layout: K at KC with a
    65,536-half head stride, V at VC in 16 x 16 tiles, so a chained runtime aliases the attention's buffers
    onto them and copies nothing."""
    if cache not in ("p9", "grid"):
        raise ValueError("rope_append: cache is p9 or grid")
    if cache == "grid" and (head_dim != 128 or capacity * head_dim > GRID_HEAD_ELEMS):
        raise ValueError("rope_append: the grid cache is head 128 and at most %d keys" % (GRID_HEAD_ELEMS // 128))
    if head_dim % 64 or n_heads % 4:
        raise ValueError("refused: rope_append needs head_dim a multiple of 64 (32-lane dim pairs) and heads a "
                         "multiple of 4 (four threadgroups)")
    if capacity % 16:
        raise ValueError("rope_append: capacity is whole 16-key blocks")
    chunks = head_dim // 64
    if groups <= n_heads:
        if n_heads % groups:
            raise ValueError("rope_append: %d heads do not split over %d threadgroups" % (n_heads, groups))
    else:
        gph = groups // n_heads
        if groups % n_heads or gph & (gph - 1) or chunks % gph:
            raise ValueError("rope_append: %d threadgroups are not a power-of-two share of %d heads x %d chunks"
                             % (groups, n_heads, chunks))
    d = n_heads * head_dim
    QKV = 8192                                  # buffer 1: the length word at 6144 (P9), then qkv, cos, sin
    LEN = KV_LENGTH_BYTE
    carrier_a = 2 * CARRIER * CARRIER * groups  # the carrier's A tile: 512 bytes per threadgroup from byte 0
    if carrier_a > LEN:
        # MM 25.138.2: from 12 threadgroups the carrier's A tile covers byte 6144, so writing the length word
        # there overwrote group 12's carrier rows and its tile came out 16 words off (25.136.6's "grouped RoPE
        # g16 incorrect"). Past that the length word and qkv follow the carrier.
        LEN = _align(carrier_a)
        QKV = LEN + 256
    kv_heads = kv_heads or n_heads
    if n_heads % kv_heads or (n_heads // kv_heads) & (n_heads // kv_heads - 1):
        raise ValueError("rope_append: %d query heads do not share %d KV heads a power of two each" % (n_heads, kv_heads))
    COS = QKV + (d + 2 * kv_heads * head_dim) * 4
    if chained:
        # THE CHAINED LAYOUT (MM 25.138.3; cache "grid" only): qkv at 64 KiB, past any fold's carrier (whose zone
        # lies just before the region it writes), then cos, sin and the length word AFTER qkv, so no carrier
        # reaches them; q written in the attention split's A layout (head stride 8,192 bytes, row 0) at QA
        if cache != "grid":
            raise ValueError("rope_append: the chained layout is the grid cache's")
        QKV = 65536
        COS = QKV + (d + 2 * kv_heads * head_dim) * 4
        LEN = COS + head_dim * 4                    # cos, sin (half each), then the length word
    SIN = COS + head_dim // 2 * 4
    a_bytes = max(SIN + head_dim // 2 * 4, LEN + 4)
    Q16 = _carrier_bytes(groups)                # buffer 3: carrier, q16, then P9-layout K and V caches
    stride = GRID_HEAD_ELEMS * 2 if cache == "grid" else capacity * head_dim * 2
    QA = None
    if chained:
        # q as the split's buffer 1 (head h's row 0 at QA + 8,192 h; rows 1..31 zero, never written), then K and V;
        # V starts past the whole of the split's buffer 3, which the runtime aliases onto KC
        import agxforge.g17.runtime as R
        QA = _align(Q16, 8192)
        KC = _align(QA + n_heads * 8192, 8192)
        split_c = R.kv_split_transport_rows(R.kv_split_c_bytes(n_heads, 8), n_heads * 8) * 256 * 4
        VC = KC + _align(max(n_heads * stride, split_c), 8192)
    else:
        KC = _align(Q16 + 2 * d, 8192)
        VC = KC + n_heads * stride
    c_bytes = VC + n_heads * stride
    t = _transport(groups, a_bytes, _carrier_bytes(groups), c_bytes)
    return dict(op="rope_append", n_heads=n_heads, head_dim=head_dim, capacity=capacity, groups=groups,
                LEN=LEN, QKV=QKV, COS=COS, SIN=SIN, Q16=Q16, KC=KC, VC=VC, cache_end=c_bytes,
                **({"cache": "grid"} if cache == "grid" else {}), **({"QA": QA} if chained else {}),
                **({"kv_heads": kv_heads} if kv_heads != n_heads else {}), **t)


def _rope_flash_layout(n_heads, head_dim, capacity, groups, kv_heads):
    """THE FLASH-DECODE LAYOUT (MM 25.138.5, Piece B's attention, 25.140): buffer 1 as the chained layout (qkv at
    64 KiB, then cos, sin and the length word). Buffer 3: the carrier, then q fp16 [heads][head_dim] contiguous at
    QF (the attention's binding 1; the attention's own length word sits at QF + 4096 and is never written here),
    then the cache PER KV HEAD, row-major: K [kv][capacity][head_dim] at KC and V at KC + VOFF (the attention's
    binding 2, V at its VOFF)."""
    kv_heads = kv_heads or n_heads
    lay = rope_layout(n_heads, head_dim, capacity, groups=groups, cache="grid", kv_heads=kv_heads, chained=True)
    QF = _align(_carrier_bytes(groups), 8192)
    KC = QF + 8192
    VOFF = _align(kv_heads * capacity * head_dim * 2)
    VC = KC + VOFF
    c_bytes = VC + VOFF
    t = _transport(groups, lay["a_bytes"], _carrier_bytes(groups), c_bytes)
    lay.update(cache="flash", QF=QF, KC=KC, VC=VC, VOFF=VOFF, cache_end=c_bytes, kv_heads=kv_heads, **t)
    lay.pop("QA", None)
    return lay


def build_rope_append(lay, q_scale):
    from agxforge.g17 import cc, ir
    H, hd, cap = lay["n_heads"], lay["head_dim"], lay["capacity"]
    d, half = H * hd, hd // 2
    split = lay["groups"] > H                   # MM 25.136.6: one head per threadgroup, a share of its chunks
    per_group = 1 if split else H // lay["groups"]
    gph = lay["groups"] // H if split else 1    # threadgroups per head
    fn, b, a, bb, c = _function()
    _carrier(b, a, bb, c, lay)
    lane = b.builtin("thread_index_in_simdgroup", name="lane")
    group = b.builtin("threadgroup_position_in_grid", name="group")
    length = b.load(a, _c(b, lay["LEN"] // 4, "len_word"), type=ir.I32, name="kv_length")
    if split:
        head0 = b.shr(group, _c(b, gph.bit_length() - 1, "log2_gph"), name="head0")
        share = half // 32 // gph               # chunks of 32 lane pairs per threadgroup
        jlane = b.add(lane, b.mul(_and(b, group, _c(b, gph - 1, "gph_mask"), "gsub"),
                                  _c(b, 32 * share, "share_pairs"), name="gsub_pairs"), name="jlane")
    else:
        head0 = b.mul(group, _c(b, per_group, "heads_per_group"), name="head0")
    flash = lay.get("cache") == "flash"
    grid = lay.get("cache") == "grid"
    hstride = GRID_HEAD_ELEMS if grid else cap * hd           # halves between heads in the cache
    if grid:
        # K: head0 * 65536 + length * hd; V: head0 * 65536 + (length >> 4) * 2048 + (length & 15) * 16
        h0 = b.shl(head0, _c(b, 16, "h0_shift"), name="h0_base")
        row = b.add(h0, b.shl(length, _c(b, 7, "len_x128"), name="len_row"), name="row0_elems")
        vrow = b.add(h0, b.add(b.shl(b.shr(length, _c(b, 4, "len_blk_sh"), name="len_blk"), _c(b, 11, "blk_x2048"),
                                     name="len_blk_elems"),
                               b.shl(_and(b, length, _c(b, 15, "len_lo_mask"), "len_lo"), _c(b, 4, "lo_x16"),
                                     name="len_lo_elems"), name="len_v_elems"), name="vrow0_elems")
    else:
        # the new cache row's first half-element: (head0 * cap + length) * hd
        row = b.mul(b.add(b.mul(head0, _c(b, cap, "cap"), name="h0cap"), length, name="row0"),
                    _c(b, hd, "hd"), name="row0_elems")
    qs = _cf(b, q_scale, "q_scale")
    for j in range(half // 32 // gph):
        # lane pair index i = 32 j + lane: this lane owns dims i and i + hd/2 of each head
        i = b.add(jlane if split else lane, _c(b, 32 * j, "pair%d" % j), name="pair_i%d" % j)
        cs = b.load(a, b.add(i, _c(b, lay["COS"] // 4, "cos_base%d" % j), name="cos_i%d" % j), type=ir.I32, name="cos%d" % j)
        sn = b.load(a, b.add(i, _c(b, lay["SIN"] // 4, "sin_base%d" % j), name="sin_i%d" % j), type=ir.I32, name="sin%d" % j)
        for hh in range(per_group):
            hoff = b.mul(b.add(head0, _c(b, hh, "hh%d_%d" % (j, hh)), name="head%d_%d" % (j, hh)),
                         _c(b, hd, "hd%d_%d" % (j, hh)), name="hoff%d_%d" % (j, hh))
            hi = b.add(hoff, i, name="hi%d_%d" % (j, hh))                # head h, dim i (element index)
            kvh = lay.get("kv_heads", H)
            if kvh != H:
                # GQA (MM 25.138.2): k and v of query head h are KV head h >> log2(H / kv_heads)'s
                kvoff = b.mul(b.shr(b.add(head0, _c(b, hh, "kh%d_%d" % (j, hh)), name="kqh%d_%d" % (j, hh)),
                                    _c(b, (H // kvh).bit_length() - 1, "kvsh%d_%d" % (j, hh)), name="kvh%d_%d" % (j, hh)),
                              _c(b, hd, "kvhd%d_%d" % (j, hh)), name="kvoff%d_%d" % (j, hh))
                khi = b.add(kvoff, i, name="khi%d_%d" % (j, hh))
            else:
                khi = hi
            if flash:
                # the per-KV-head cache row: (kv * cap + length) * hd; both query heads of a pair write the same row
                kvq = b.shr(b.add(head0, _c(b, hh, "fkh%d_%d" % (j, hh)), name="fqh%d_%d" % (j, hh)),
                            _c(b, (H // kvh).bit_length() - 1, "fkvsh%d_%d" % (j, hh)), name="fkv%d_%d" % (j, hh))
                frow = b.mul(b.add(b.mul(kvq, _c(b, cap, "fcap%d_%d" % (j, hh)), name="fkc%d_%d" % (j, hh)), length,
                                   name="fkl%d_%d" % (j, hh)), _c(b, hd, "fhd%d_%d" % (j, hh)), name="frow%d_%d" % (j, hh))
            for part, tag in ((0, "q"), (1, "k"), (2, "v")):
                base = lay["QKV"] // 4 + (0 if part == 0 else d + (part - 1) * kvh * hd)
                src = b.add(hi if part == 0 else khi, _c(b, base, "%s_base%d_%d" % (tag, j, hh)), name="%s_i%d_%d" % (tag, j, hh))
                x1 = b.load(a, src, type=ir.I32, name="%s_x1_%d_%d" % (tag, j, hh))
                x2 = b.load(a, b.add(src, _c(b, half, "%s_h%d_%d" % (tag, j, hh)), name="%s_i2_%d_%d" % (tag, j, hh)),
                            type=ir.I32, name="%s_x2_%d_%d" % (tag, j, hh))
                if tag == "v":
                    o1, o2 = x1, x2
                else:
                    o1 = b.fadd(b.fmul(x1, cs, name="%s_ac%d_%d" % (tag, j, hh)),
                                b.fneg(b.fmul(x2, sn, name="%s_bs%d_%d" % (tag, j, hh)), name="%s_nbs%d_%d" % (tag, j, hh)),
                                name="%s_o1_%d_%d" % (tag, j, hh))
                    o2 = b.fadd(b.fmul(x2, cs, name="%s_bc%d_%d" % (tag, j, hh)), b.fmul(x1, sn, name="%s_as%d_%d" % (tag, j, hh)),
                                name="%s_o2_%d_%d" % (tag, j, hh))
                if tag == "q":
                    o1 = b.fmul(o1, qs, name="q_s1_%d_%d" % (j, hh))
                    o2 = b.fmul(o2, qs, name="q_s2_%d_%d" % (j, hh))
                if flash:
                    if tag == "q":
                        dst = b.add(hi, _c(b, lay["QF"] // 2, "fq_base%d_%d" % (j, hh)), name="fq_i%d_%d" % (j, hh))
                    else:
                        dst = b.add(b.add(frow, i, name="%s_fri%d_%d" % (tag, j, hh)),
                                    _c(b, (lay["KC"] if tag == "k" else lay["VC"]) // 2, "%s_fbase%d_%d" % (tag, j, hh)),
                                    name="%s_fdi%d_%d" % (tag, j, hh))
                elif tag == "q" and lay.get("QA") is not None:
                    # the split's A layout: head h's row 0 at QA + 8,192 h bytes (4,096 halves), dim i
                    qh = b.shl(b.add(head0, _c(b, hh, "qhh%d_%d" % (j, hh)), name="qh%d_%d" % (j, hh)),
                               _c(b, 12, "qx4096_%d_%d" % (j, hh)), name="qhe%d_%d" % (j, hh))
                    dst = b.add(b.add(qh, i, name="qhi%d_%d" % (j, hh)), _c(b, lay["QA"] // 2, "qa_base%d_%d" % (j, hh)),
                                name="qa_i%d_%d" % (j, hh))
                elif tag == "q":
                    dst = b.add(hi, _c(b, lay["Q16"] // 2, "q16_base%d_%d" % (j, hh)), name="q16_i%d_%d" % (j, hh))
                elif grid and tag == "v":
                    # dims i and i + 64 in V's tiles: (d >> 4) * 256 + (d & 15); the second is 4 tiles further
                    tile = b.add(b.shl(b.shr(i, _c(b, 4, "v_dsh%d_%d" % (j, hh)), name="v_dt%d_%d" % (j, hh)),
                                       _c(b, 8, "v_dx256_%d_%d" % (j, hh)), name="v_dte%d_%d" % (j, hh)),
                                 _and(b, i, _c(b, 15, "v_dm%d_%d" % (j, hh)), "v_dl%d_%d" % (j, hh)),
                                 name="v_d%d_%d" % (j, hh))
                    dst = b.add(b.add(vrow, tile, name="v_ri%d_%d" % (j, hh)),
                                _c(b, lay["VC"] // 2 + hh * hstride, "v_cbase%d_%d" % (j, hh)), name="v_di%d_%d" % (j, hh))
                    b.store_at(c, dst, b.f32_to_f16_rte(o1, name="v_n1_%d_%d" % (j, hh)), width="half")
                    b.store_at(c, b.add(dst, _c(b, (half // 16) * 256, "v_d2_%d_%d" % (j, hh)), name="v_di2_%d_%d" % (j, hh)),
                               b.f32_to_f16_rte(o2, name="v_n2_%d_%d" % (j, hh)), width="half")
                    continue
                else:
                    # cache element (head h, row length, dim i): row0 + hh * cap * hd + i
                    dst = b.add(b.add(row, i, name="%s_ri%d_%d" % (tag, j, hh)),
                                _c(b, (lay["KC"] if tag == "k" else lay["VC"]) // 2 + hh * hstride,
                                   "%s_cbase%d_%d" % (tag, j, hh)), name="%s_di%d_%d" % (tag, j, hh))
                b.store_at(c, dst, b.f32_to_f16_rte(o1, name="%s_n1_%d_%d" % (tag, j, hh)), width="half")
                b.store_at(c, b.add(dst, _c(b, half, "%s_d2_%d_%d" % (tag, j, hh)), name="%s_di2_%d_%d" % (tag, j, hh)),
                           b.f32_to_f16_rte(o2, name="%s_n2_%d_%d" % (tag, j, hh)), width="half")
    b.ret()
    ir.verify(fn)
    return cc.compile_function(fn)


# ---- SwiGLU ------------------------------------------------------------------------------------------

def swiglu_layout(ffn, groups=8):
    """`groups` threadgroups of one simdgroup each; each lane runs ffn / (32 groups) unrolled bodies.
    The default 8 is the pinned milestone program (MM 25.136); 25.132.4 times 32 to 256."""
    if groups < 1 or ffn % (32 * groups):
        raise ValueError("swiglu: the width is a multiple of %d (%d threadgroups of 32 lanes)" % (32 * groups, groups))
    GATE = OUT = _carrier_bytes(groups)
    UP = GATE + 4 * ffn
    # buffer 2 holds only the carrier's 16 x 16 B tile (512 bytes). _carrier_bytes(groups) (buffer 3's
    # carrier rows) over-asked for it and, at 256 groups, forced K to 512 (no transport)
    t = _transport(groups, UP + 4 * ffn, _align(2 * CARRIER * CARRIER), OUT + 2 * ffn)
    return dict(op="swiglu", ffn=ffn, groups=groups, GATE=GATE, UP=UP, OUT=OUT, **t)


def build_swiglu(lay):
    from agxforge.g17 import cc, ir
    import g17tensorcommonruntime as TCR
    f = lay["ffn"]
    per = f // lay["groups"]
    fn, b, a, bb, c = _function()
    _carrier(b, a, bb, c, lay)
    lane = b.builtin("thread_index_in_simdgroup", name="lane")
    group = b.builtin("threadgroup_position_in_grid", name="group")
    base = b.add(b.mul(group, _c(b, per, "per_group"), name="gbase"), lane, name="base")
    K = emit_constants(b)
    K2 = emit_exp2_constants(b)
    neg_inv_ln2 = _cf(b, TCR.GELU_NEG_INV_LN2, "neg_inv_ln2")
    one = _cf(b, F32(1.0), "fone")
    for i in range(per // 32):
        idx = b.add(base, _c(b, 32 * i, "off%d" % i), name="i%d" % i)
        x = b.load(a, b.add(idx, _c(b, lay["GATE"] // 4, "gate_base%d" % i), name="gi%d" % i), type=ir.I32, name="gate%d" % i)
        t = b.fmul(x, neg_inv_ln2, name="t%d" % i)
        e = emit_exp2_soft(b, t, K2, "x%d" % i)
        u = b.fadd(e, one, type=ir.I32, name="u%d" % i)
        r = emit_rn(b, "recip", u, K, "r%d" % i)
        s = b.fmul(x, r, name="silu%d" % i)
        up = b.load(a, b.add(idx, _c(b, lay["UP"] // 4, "up_base%d" % i), name="ui%d" % i), type=ir.I32, name="up%d" % i)
        y = b.fmul(s, up, name="y%d" % i)
        b.store_at(c, b.add(idx, _c(b, lay["OUT"] // 2, "out_base%d" % i), name="oi%d" % i),
                   b.f32_to_f16_rte(y, name="yh%d" % i), width="half")
    b.ret()
    ir.verify(fn)
    return cc.compile_function(fn)


# ---- the split-K fold (Set C's tensorreduce.emit_split_k_fold, #197) ------------------------------

class _Offset:
    """A builder whose load and store_at add fixed element offsets to every index, and which delegates
    everything else. emit_split_k_fold indexes its partials and its output from element 0; the carrier's
    tiles occupy the start of buffers 1 and 3, so the fold's regions start after them. The fold's own
    arithmetic (its order, its lane and threadgroup indexing) is main's, unchanged."""

    def __init__(self, b, partials, out, p_off, o_off):
        self._b, self._p, self._o, self._po, self._oo = b, partials, out, p_off, o_off

    def __getattr__(self, name):
        return getattr(self._b, name)

    def load(self, buf, idx, **kw):
        if buf is self._p and self._po:
            idx = self._b.add(idx, _c(self._b, self._po, "fold_p_base"), name="fold_pi")
        return self._b.load(buf, idx, **kw)

    def store_at(self, buf, idx, v, **kw):
        if buf is self._o and self._oo:
            idx = self._b.add(idx, _c(self._b, self._oo, "fold_o_base"), name="fold_oi")
        return self._b.store_at(buf, idx, v, **kw)


def fold_layout(G, M, N, M_live=None):
    """G stacked M x N fp32 partials in buffer 1 at P; the M x N fp32 fold in buffer 3 at OUT, of which
    rows [0, M_live) are written (Set C's row-limited fold; decode's live row is row 0); one threadgroup
    per 32 columns (emit_split_k_fold's column grid), each also running the carrier."""
    if G < 2 or N % 32:
        raise ValueError("split-K fold: G >= 2 and N a multiple of 32")
    M_live = M if M_live is None else M_live
    if not 1 <= M_live <= M:
        raise ValueError("split-K fold: M_live is 1..M")
    groups = N // 32
    P = OUT = _carrier_bytes(groups)
    t = _transport(groups, P + 4 * G * M * N, _carrier_bytes(groups), OUT + 4 * M * N)
    return dict(op="split_k_fold", G=G, MF=M, NF=N, M_live=M_live, groups=groups, P=P, OUT=OUT, **t)


def build_split_k_fold(lay):
    """The carrier FIRST (its output is at buffer 3's start, outside the fold's region), then Set C's
    fold over buffer 1's partials into buffer 3: exactly fold_split_k without C (the caller adds C)."""
    from agxforge.g17 import cc, ir, tensorreduce
    fn, b, a, bb, c = _function()
    _carrier(b, a, bb, c, lay)
    tensorreduce.emit_split_k_fold(_Offset(b, a, c, lay["P"] // 4, lay["OUT"] // 4), a, c,
                                   G=lay["G"], M=lay["MF"], N=lay["NF"],
                                   **({} if lay.get("M_live", lay["MF"]) == lay["MF"] else dict(M_live=lay["M_live"])))
    b.ret()
    ir.verify(fn)
    return cc.compile_function(fn)


def host_fold(partials, G, M, reverse=False):
    """fold_split_k (tools/g17decodestep_gpu.py) without C: an fp32 left fold in ascending t. reverse=True
    is THE CONTROL: the same adds in descending t, which rounds differently."""
    p = np.asarray(partials, F32).reshape(G, M, -1)
    order = range(G - 1, -1, -1) if reverse else range(G)
    it = iter(order)
    acc = p[next(it)].copy()
    with np.errstate(over="ignore", invalid="ignore"):
        for t in it:
            acc = (acc + p[t]).astype(F32)
    return acc


def fold_io(lay, partials, reverse=False):
    a, b, c = _buffers(lay)
    ca, cb = _with_carrier(lay, (a, b, c))
    _place(a, lay["P"], np.asarray(partials, "<f4"))
    want = bytearray(c)
    live = lay.get("M_live", lay["MF"])          # rows past M_live are not written: they keep c's zeros
    _place(want, lay["OUT"], host_fold(partials, lay["G"], lay["MF"], reverse=reverse)[:live])
    # the carrier reads buffer 1's first 16 G rows of 16 halves, which the partials do not reach (P is past them)
    _place(want, 0, carrier_reference(lay, ca, cb).astype("<f4"))
    return bytes(a), bytes(b), bytes(c), bytes(want)


def fold_stage(partials, G, M, M_live=None):
    N = np.asarray(partials).shape[-1]
    lay = fold_layout(G, M, N, M_live)
    key = "fold_g%d_m%d_n%d%s" % (G, M, N, "" if lay["M_live"] == M else "_r%d" % lay["M_live"])
    return key, lay, (lambda: build_split_k_fold(lay)), fold_io(lay, partials)


def fold_out(lay, c):
    return np.frombuffer(c, "<f4", lay["MF"] * lay["NF"], lay["OUT"]).reshape(lay["MF"], lay["NF"]).copy()


# ---- the glue a chained token needs on the GPU (MM 25.138.3) -----------------------------------------
# What the host pipeline did between dispatches, as programs: the residual add (the fold's C, added last) with
# the half narrowing of the layer output, and the attention output gathered from the merge's O tiles and narrowed
# into o_proj's A row. Elementwise, one fadd or one narrowing each: the reference's own rounding points.

def residual_layout(n, r_dtype="float", out32=True, outh=False, groups=8):
    """out = F + R (RNE fp32; R is fp32 or half, widened exactly), written as fp32 at OUT32 and/or narrowed to half
    at OUTH. F and R in buffer 1, outputs in buffer 3; `groups` threadgroups of 32 lanes, n / (32 groups) each."""
    if n % (32 * groups):
        raise ValueError("residual: n is a multiple of %d" % (32 * groups))
    if r_dtype not in ("float", "half") or not (out32 or outh):
        raise ValueError("residual: R is float or half, and at least one output")
    F = _carrier_bytes(groups)
    R = F + 4 * n
    a_end = R + (4 if r_dtype == "float" else 2) * n
    OUT32 = _carrier_bytes(groups)
    OUTH = OUT32 + (4 * n if out32 else 0)
    t = _transport(groups, a_end, _align(2 * CARRIER * CARRIER), OUTH + (2 * n if outh else 0))
    return dict(op="residual", n=n, r_dtype=r_dtype, out32=out32, outh=outh, groups=groups, F=F, R=R,
                OUT32=OUT32 if out32 else None, OUTH=OUTH if outh else None, **t)


def build_residual(lay):
    from agxforge.g17 import cc, ir
    n, per = lay["n"], lay["n"] // lay["groups"]
    fn, b, a, bb, c = _function()
    _carrier(b, a, bb, c, lay)
    lane = b.builtin("thread_index_in_simdgroup", name="lane")
    group = b.builtin("threadgroup_position_in_grid", name="group")
    base = b.add(b.mul(group, _c(b, per, "per_group"), name="gbase"), lane, name="base")
    for i in range(per // 32):
        idx = b.add(base, _c(b, 32 * i, "off%d" % i), name="i%d" % i)
        f = b.load(a, b.add(idx, _c(b, lay["F"] // 4, "f_base%d" % i), name="fi%d" % i), type=ir.I32, name="f%d" % i)
        if lay["r_dtype"] == "half":
            r = b.f16_to_f32(b.load(a, b.add(idx, _c(b, lay["R"] // 2, "r_base%d" % i), name="ri%d" % i), width="half",
                                    name="rh%d" % i), name="r%d" % i)
        else:
            r = b.load(a, b.add(idx, _c(b, lay["R"] // 4, "r_base%d" % i), name="ri%d" % i), type=ir.I32, name="r%d" % i)
        y = b.fadd(f, r, type=ir.I32, name="y%d" % i)
        if lay["out32"]:
            b.store_at(c, b.add(idx, _c(b, lay["OUT32"] // 4, "o32_base%d" % i), name="o32i%d" % i), y)
        if lay["outh"]:
            b.store_at(c, b.add(idx, _c(b, lay["OUTH"] // 2, "oh_base%d" % i), name="ohi%d" % i),
                       b.f32_to_f16_rte(y, name="yh%d" % i), width="half")
    b.ret()
    ir.verify(fn)
    return cc.compile_function(fn)


def residual_io(lay, f32, r):
    a, b, c = _buffers(lay)
    ca, cb = _with_carrier(lay, (a, b, c))
    _place(a, lay["F"], np.asarray(f32, "<f4"))
    _place(a, lay["R"], np.asarray(r, "<f4" if lay["r_dtype"] == "float" else np.float16))
    y = (np.asarray(f32, F32) + np.asarray(r, F32)).astype(F32)
    want = bytearray(c)
    if lay["out32"]:
        _place(want, lay["OUT32"], y)
    if lay["outh"]:
        _place(want, lay["OUTH"], y.astype(np.float16))
    _place(want, 0, carrier_reference(lay, ca, cb).astype("<f4"))
    return bytes(a), bytes(b), bytes(c), bytes(want)


def residual_stage(f32, r, r_dtype="float", out32=True, outh=False):
    n = np.asarray(f32).size
    lay = residual_layout(n, r_dtype, out32, outh)
    key = "residual_n%d_r%s%s%s" % (n, r_dtype[0], "_o32" if out32 else "", "_oh" if outh else "")
    return key, lay, (lambda: build_residual(lay)), residual_io(lay, f32, r)


def attn_gather_layout(heads=16, value=128, groups=8):
    """o32's row 0 of each head's value/16 O tiles, from buffer 1 laid out as the attention grid's buffer 3
    (head h's tile t at byte h * 131072 + O_tiles[t], 32 x 16 fp32, row 0 = 16 words), narrowed to half into
    buffer 3 at OUTH as one contiguous heads x value row (o_proj's A row 0)."""
    import agxforge.g17.runtime as R
    n = heads * value
    if n % (32 * groups) or value % 16:
        raise ValueError("attn_gather: heads x value a multiple of %d, value of 16" % (32 * groups))
    O0 = R.ATTENTION_GRID_C["O"]
    a_end = (heads - 1) * R.ATTENTION_GRID_STRIDE["C"] + O0 + 2048 * (value // 16)
    OUTH = _carrier_bytes(groups)
    t = _transport(groups, a_end, _align(2 * CARRIER * CARRIER), OUTH + 2 * n)
    return dict(op="attn_gather", heads=heads, value=value, groups=groups, O0=O0,
                HSTRIDE=R.ATTENTION_GRID_STRIDE["C"], OUTH=OUTH, **t)


def build_attn_gather(lay):
    from agxforge.g17 import cc, ir
    n = lay["heads"] * lay["value"]
    per = n // lay["groups"]
    fn, b, a, bb, c = _function()
    _carrier(b, a, bb, c, lay)
    lane = b.builtin("thread_index_in_simdgroup", name="lane")
    group = b.builtin("threadgroup_position_in_grid", name="group")
    base = b.add(b.mul(group, _c(b, per, "per_group"), name="gbase"), lane, name="base")
    vshift = lay["value"].bit_length() - 1
    for i in range(per // 32):
        e = b.add(base, _c(b, 32 * i, "off%d" % i), name="e%d" % i)             # output element h * value + d
        h = b.shr(e, _c(b, vshift, "vsh%d" % i), name="h%d" % i)
        d = _and(b, e, _c(b, lay["value"] - 1, "vmask%d" % i), "d%d" % i)
        # word: h * HSTRIDE/4 + O0/4 + (d >> 4) * 512 + (d & 15)
        src = b.add(b.add(b.shl(h, _c(b, (lay["HSTRIDE"] // 4).bit_length() - 1, "hsh%d" % i), name="hw%d" % i),
                          b.shl(b.shr(d, _c(b, 4, "dsh%d" % i), name="dt%d" % i), _c(b, 9, "t512_%d" % i), name="dtw%d" % i),
                          name="hd%d" % i),
                    b.add(_and(b, d, _c(b, 15, "dm%d" % i), "dl%d" % i), _c(b, lay["O0"] // 4, "o0_%d" % i), name="dl_o%d" % i),
                    name="src%d" % i)
        x = b.load(a, src, type=ir.I32, name="x%d" % i)
        b.store_at(c, b.add(e, _c(b, lay["OUTH"] // 2, "oh_base%d" % i), name="ohi%d" % i),
                   b.f32_to_f16_rte(x, name="xh%d" % i), width="half")
    b.ret()
    ir.verify(fn)
    return cc.compile_function(fn)


def attn_gather_io(lay, grid_c):
    """grid_c: the merge's buffer 3 (fp32 words); the want is o32 narrowed, per g17decodestep.narrow."""
    a, b, c = _buffers(lay)
    ca, cb = _with_carrier(lay, (a, b, c))
    g = np.asarray(grid_c, "<f4").ravel()
    nbytes = min(len(a) - 0, g.nbytes)
    raw = bytearray(a)
    raw[:nbytes] = g.tobytes()[:nbytes]
    a = bytes(raw)
    # buffer 1 IS the grid's buffer 3 (a chained runtime aliases it), so the carrier's A tile is head 0's first K rows
    shape = carrier_tiles(lay)[0].shape
    ca = np.frombuffer(a, np.float16, int(np.prod(shape))).reshape(shape).copy()
    H, V = lay["heads"], lay["value"]
    o32 = np.zeros((H, V), F32)
    for h in range(H):
        for t in range(V // 16):
            w = (h * lay["HSTRIDE"] + lay["O0"] + 2048 * t) // 4
            o32[h, 16 * t:16 * t + 16] = g[w:w + 16]
    want = bytearray(c)
    _place(want, lay["OUTH"], o32.reshape(-1).astype(np.float16))
    _place(want, 0, carrier_reference(lay, ca, cb).astype("<f4"))
    return a, bytes(b), bytes(c), bytes(want)


def attn_gather_stage(grid_c, heads=16, value=128):
    lay = attn_gather_layout(heads, value)
    key = "attn_gather_h%d_v%d" % (heads, value)
    return key, lay, (lambda: build_attn_gather(lay)), attn_gather_io(lay, grid_c)


# ---- the transcendental probe ---------------------------------------------------------------------

PROBE_OUT = ("rsqrt_raw", "rsqrt_rn", "recip_raw", "recip_rn", "exp2_raw", "exp2_soft")


def probe_layout(per_lane=4):
    groups = 8
    n = 32 * groups * per_lane                 # inputs per function
    OUT = RS = _carrier_bytes(groups)
    RC, EX = RS + 4 * n, RS + 8 * n
    t = _transport(groups, EX + 4 * n, _carrier_bytes(groups), OUT + 4 * n * len(PROBE_OUT))
    return dict(op="probe", groups=groups, per_lane=per_lane, n=n, RS=RS, RC=RC, EX=EX, OUT=OUT, **t)


def build_probe(lay):
    from agxforge.g17 import cc, ir
    n, per = lay["n"], 32 * lay["per_lane"]
    fn, b, a, bb, c = _function()
    _carrier(b, a, bb, c, lay)
    lane = b.builtin("thread_index_in_simdgroup", name="lane")
    group = b.builtin("threadgroup_position_in_grid", name="group")
    base = b.add(b.mul(group, _c(b, per, "per_group"), name="gbase"), lane, name="base")
    K = emit_constants(b)
    K2 = emit_exp2_constants(b)
    ob = lay["OUT"] // 4
    for i in range(lay["per_lane"]):
        idx = b.add(base, _c(b, 32 * i, "off%d" % i), name="i%d" % i)
        ld = lambda region, tag: b.load(a, b.add(idx, _c(b, region // 4, "%s_b%d" % (tag, i)), name="%s_i%d" % (tag, i)),
                                        type=ir.I32, name="%s%d" % (tag, i))
        outs = []
        x = ld(lay["RS"], "rs")
        outs.append(b.rsqrt(x, type=ir.I32, name="rs_raw%d" % i))
        outs.append(emit_rn(b, "rsqrt", x, K, "rs%d" % i))
        g = ld(lay["RC"], "rc")
        outs.append(b.recip(g, type=ir.I32, name="rc_raw%d" % i))
        outs.append(emit_rn(b, "recip", g, K, "rc%d" % i))
        t = ld(lay["EX"], "ex")
        outs.append(b.exp2(t, type=ir.I32, name="ex_raw%d" % i))
        outs.append(emit_exp2_soft(b, t, K2, "ex%d" % i))
        for k, v in enumerate(outs):
            b.store_at(c, b.add(idx, _c(b, ob + k * n, "o%d_%d" % (k, i)), name="oi%d_%d" % (k, i)), v)
    b.ret()
    ir.verify(fn)
    return cc.compile_function(fn)


# ---------------------------------------------------------------------------------------------------
# inputs, references and the bundles

def carrier_tiles(lay, seed=4242):
    """The carrier's A tile (16 G x 16 halves) and B tile (16 x 16), drawn."""
    rng = np.random.default_rng(seed)
    ca = rng.uniform(-2, 2, (CARRIER * lay["groups"], CARRIER)).astype(np.float16)
    cb = rng.uniform(-2, 2, (CARRIER, CARRIER)).astype(np.float16)
    return ca, cb


def carrier_reference(lay, ca, cb):
    """Threadgroup t's rows [16 t, 16 t + 16) of A times B: one MMA over the whole tile."""
    return D.gemm(ca.astype(F32), cb.astype(F32))


def _place(buf, off, arr):
    raw = np.ascontiguousarray(arr).tobytes()
    buf[off:off + len(raw)] = raw


def _buffers(lay):
    return bytearray(lay["a_bytes"]), bytearray(lay["b_bytes"]), bytearray(lay["c_bytes"])


def _with_carrier(lay, bufs):
    a, b, c = bufs
    ca, cb = carrier_tiles(lay)
    _place(a, 0, ca)
    _place(b, 0, cb)
    return ca, cb


def rmsnorm_io(lay, v, g, spec):
    """(a, b, c) inputs and the expected C for one row (the reference rmsnorm)."""
    a, b, c = _buffers(lay)
    ca, cb = _with_carrier(lay, (a, b, c))
    _place(a, lay["X"], np.asarray(v, np.float16 if lay["in_dtype"] == "half" else "<f4"))
    _place(b, lay["G"], np.asarray(g, np.float16))
    want = bytearray(c)
    _place(want, lay["OUT"], D.rmsnorm(v, g, spec).astype(np.float16))
    _place(want, 0, carrier_reference(lay, ca, cb).astype("<f4"))
    return bytes(a), bytes(b), bytes(c), bytes(want)


def grid_cache_images(k, v):
    """(K image, V image), each heads x 65,536 halves, from heads x keys x 128 arrays: the attention grid's layout
    (K row-major; V in 16 x 16 tiles), zero past the keys."""
    H, n, hd = np.asarray(k).shape
    ki = np.zeros((H, GRID_HEAD_ELEMS), np.float16)
    vi = np.zeros((H, GRID_HEAD_ELEMS), np.float16)
    ki[:, :n * hd] = np.asarray(k, np.float16).reshape(H, -1)
    nb = -(-n // 16)
    vp = np.zeros((H, nb * 16, hd), np.float16); vp[:, :n] = v
    tiles = vp.reshape(H, nb, 16, hd // 16, 16).transpose(0, 1, 3, 2, 4)     # h, block, slice, row, col
    vi[:, :tiles[0].size] = tiles.reshape(H, -1)
    return ki, vi


def grid_cache_arrays(ki, vi, n):
    """The inverse of grid_cache_images: heads x n x 128 K and V."""
    H = ki.shape[0]
    nb = -(-n // 16)
    k = ki[:, :n * 128].reshape(H, n, 128).astype(F32)
    v = vi[:, :nb * 16 * 128].reshape(H, nb, 8, 16, 16).transpose(0, 1, 3, 2, 4).reshape(H, nb * 16, 128)[:, :n]
    return k, v.astype(F32)


def rope_cache_image(lay, k_cache, v_cache):
    """P9's operand layout per head: K at KC and V at VC, each heads x capacity x head_dim halves,
    rows beyond the prefill zero (KVCache.zero_fill's rule: masked keys still pass through PV)."""
    H, cap, hd = lay["n_heads"], lay["capacity"], lay["head_dim"]
    kc = np.zeros((H, cap, hd), np.float16)
    vc = np.zeros((H, cap, hd), np.float16)
    n = k_cache.shape[1]
    kc[:, :n] = k_cache
    vc[:, :n] = v_cache
    return kc, vc


def rope_io(lay, spec, qkv32, rope_cos, rope_sin, k_cache, v_cache, length):
    a, b, c = _buffers(lay)
    ca, cb = _with_carrier(lay, (a, b, c))
    _place(a, lay["LEN"], np.array([length], "<u4"))
    _place(a, lay["QKV"], np.asarray(qkv32, "<f4"))
    _place(a, lay["COS"], np.asarray(rope_cos, "<f4"))
    _place(a, lay["SIN"], np.asarray(rope_sin, "<f4"))
    kc, vc = rope_cache_image(lay, k_cache, v_cache)
    ref = D.stage_rope_append(spec, qkv32, rope_cos, rope_sin, k_cache[:, :length], v_cache[:, :length])
    kw, vw = kc.copy(), vc.copy()
    kw[:, length] = ref["k_new"]
    vw[:, length] = ref["v_new"]
    if lay.get("cache") == "grid":
        kc, vc = grid_cache_images(kc, vc)
        kw, vw = grid_cache_images(kw, vw)
    elif lay.get("cache") == "flash":
        # per KV head: KV head g is query head g * (H / kv_heads)'s row (the GQA duplicates are equal)
        gsz = lay["n_heads"] // lay["kv_heads"]
        kc, vc, kw, vw = (np.ascontiguousarray(x[::gsz]) for x in (kc, vc, kw, vw))
    _place(c, lay["KC"], kc)
    _place(c, lay["VC"], vc)
    want = bytearray(c)
    _place(want, lay["KC"], kw)
    _place(want, lay["VC"], vw)
    if lay.get("cache") == "flash":
        _place(want, lay["QF"], ref["q16"].astype(np.float16))
    elif lay.get("QA") is not None:
        for h in range(lay["n_heads"]):
            _place(want, lay["QA"] + 8192 * h, ref["q16"][h].astype(np.float16))
    else:
        _place(want, lay["Q16"], ref["q16"].astype(np.float16))
    _place(want, 0, carrier_reference(lay, ca, cb).astype("<f4"))
    return bytes(a), bytes(b), bytes(c), bytes(want)


def swiglu_io(lay, spec, gate32, up32):
    a, b, c = _buffers(lay)
    ca, cb = _with_carrier(lay, (a, b, c))
    _place(a, lay["GATE"], np.asarray(gate32, "<f4"))
    _place(a, lay["UP"], np.asarray(up32, "<f4"))
    want = bytearray(c)
    _place(want, lay["OUT"], D.stage_ffn_swiglu(spec, gate32, up32)["act"].astype(np.float16))
    _place(want, 0, carrier_reference(lay, ca, cb).astype("<f4"))
    return bytes(a), bytes(b), bytes(c), bytes(want)


def probe_inputs(lay, extra_rsqrt=(), seed=136):
    """Dense sweeps: rsqrt over positive normals 1e-3..1e4 (settle item 2's range) with binade edges
    and the given extra arguments; recip over [1, 2^20] (SiLU's 1 + 2^t) with binade edges; exp2
    over [-30, 30] with integers and half-integers."""
    rng = np.random.default_rng(seed)
    n = lay["n"]
    def fill(fixed, lo, hi, log):
        fixed = np.asarray(fixed, F32)[:n]
        m = n - fixed.size
        r = np.exp(rng.uniform(np.log(lo), np.log(hi), m)) if log else rng.uniform(lo, hi, m)
        return np.concatenate([fixed, r.astype(F32)])
    p2 = F32(2.0) ** np.arange(-9, 13, dtype=F32)
    rs = fill(np.concatenate([np.asarray(extra_rsqrt, F32), p2, np.nextafter(p2, F32(0)), np.nextafter(p2, F32(np.inf))]),
              1e-3, 1e4, True)
    q2 = F32(2.0) ** np.arange(0, 20, dtype=F32)
    rc = fill(np.concatenate([q2, np.nextafter(q2[1:], F32(0)), np.nextafter(q2, F32(np.inf))]), 1.0, 2.0 ** 20, True)
    ex = fill(np.concatenate([np.arange(-30, 31, dtype=F32), np.arange(-30, 30, dtype=F32) + F32(0.5)]), -30.0, 30.0, False)
    return rs, rc, ex


def probe_io(lay, rs, rc, ex):
    a, b, c = _buffers(lay)
    ca, cb = _with_carrier(lay, (a, b, c))
    _place(a, lay["RS"], rs.astype("<f4"))
    _place(a, lay["RC"], rc.astype("<f4"))
    _place(a, lay["EX"], ex.astype("<f4"))
    return bytes(a), bytes(b), bytes(c), carrier_reference(lay, ca, cb)


def probe_score(lay, rs, rc, ex, out_words):
    """The measurement: each raw function's ulp distance from the correctly rounded value, and
    whether the corrected / soft outputs equal their models bit for bit."""
    n = lay["n"]
    o = np.frombuffer(out_words, "<u4")[lay["OUT"] // 4:lay["OUT"] // 4 + n * len(PROBE_OUT)].reshape(len(PROBE_OUT), n)
    rn = {"rsqrt": rsqrt_rn(rs).view(np.uint32), "recip": recip_rn(rc).view(np.uint32),
          "exp2": D.exp2(ex).view(np.uint32)}
    def hist(got, want):
        dd = got.astype(np.int64) - want.astype(np.int64)
        vals, counts = np.unique(dd, return_counts=True)
        return {str(int(v)): int(k) for v, k in zip(vals, counts)}
    return {
        "rsqrt_raw_ulp_vs_rn": hist(o[0], rn["rsqrt"]),
        "rsqrt_rn_equals_rn": int(np.sum(o[1] == rn["rsqrt"])),
        "recip_raw_ulp_vs_rn": hist(o[2], rn["recip"]),
        "recip_rn_equals_rn": int(np.sum(o[3] == rn["recip"])),
        "exp2_raw_ulp_vs_rn": hist(o[4], rn["exp2"]),
        "exp2_soft_equals_model": int(np.sum(o[5] == exp2_soft(ex).view(np.uint32))),
        "exp2_soft_model_ulp_vs_rn": hist(exp2_soft(ex).view(np.uint32), rn["exp2"]),
        "n": n,
    }


# ---------------------------------------------------------------------------------------------------
# authoring and dispatch (one worker process per dispatch, under the machine-wide GPU lock)

def generic_view(lay):
    """What manifest_for reads for gemm_generic's transport. A layout may carry its own `view` (a K-loop
    GEMM whose buffers are large enough; tools/g17qmv.py), else the default below."""
    if "view" in lay:
        return dict(lay["view"])
    return dict(M=lay["M"], N=lay["N"], K=lay["K"], a="half", b="half", simdgroups=1, threadgroups=lay["groups"],
                grid_n=1, split_k=1,                   # Set C's column and K grids (25.134): neither is used here
                epilogue=[], stages=[], accumulate=False, saturate=False)


def author(bundle: Path, lay, program, a, b, c, extra=None):
    """Write a bundle the common worker loads: the image, its manifest, the three inputs."""
    import g17tensorcommonruntime as TCR
    from agxforge.g17 import scanlink
    bundle = Path(bundle)
    if bundle.exists():
        raise ValueError("refusing to overwrite an existing bundle: %s" % bundle)
    if (len(a), len(b), len(c)) != (lay["a_bytes"], lay["b_bytes"], lay["c_bytes"]):
        raise ValueError("inputs do not match the transport")
    image = scanlink.author(program)
    manifest = TCR.manifest_for(program, generic=generic_view(lay)).model_copy(update={
        "sha256": {"archive": TCR.sha(image.archive), "library": TCR.sha(image.library),
                   "object": TCR.sha(image.object), "code": TCR.sha(program.code)},
        "field_ledger": image.field_ledger})
    expected = [(bd.index, bd.offset, bd.written) for bd in manifest.abi.bindings]
    if scanlink.verify_contract(image.archive, image.library, expected) != image.object:
        raise ValueError("authored archive does not contain its delivered object")
    bundle.mkdir(parents=True)
    (bundle / "decodeop.json").write_text(json.dumps(dict(layout=lay, **(extra or {})), indent=1, sort_keys=True) + "\n")
    for name, data in (("a.f16", a), ("b.f16", b), ("c.f32", c)):
        (bundle / name).write_bytes(data)
    (bundle / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n")
    for name, data in (("scan.arc.metallib", image.archive), ("scan.lib.metallib", image.library),
                       ("scan.o", image.object), ("program.bin", program.code)):
        (bundle / name).write_bytes(data)
    return {"program_sha256": TCR.sha(program.code), "code_bytes": len(program.code),
            "manifest_sha256": TCR.sha((bundle / "manifest.json").read_bytes())}


def dispatch(bundle: Path, worker: Path, queries=3, inputs=None):
    """Load-only check, then one dispatch process of `queries` queries; every query's C returned.
    `inputs` (a, b, c) replaces the bundle's input files first (same sizes)."""
    import g17commonstage
    import g17packeddispatch
    import g17tensorcommonruntime as TCR
    bundle = Path(bundle)
    if inputs is not None:
        for name, data in zip(("a.f16", "b.f16", "c.f32"), inputs):
            if len(data) != (bundle / name).stat().st_size:
                raise ValueError("%s: size differs from the authored bundle" % name)
            (bundle / name).write_bytes(data)
    with g17commonstage.lock_gpu():
        before = g17packeddispatch.gpu_events()
        load = subprocess.run([str(worker), str(bundle), "--tensor-load-approved"], capture_output=True, timeout=60)
        if load.returncode or json.loads(load.stdout) != {"status": 0, "load_only": True, "gpu_dispatched": False}:
            raise RuntimeError("tensor load-only failed: " + load.stderr.decode(errors="replace")[-2000:])
        run = subprocess.run([str(worker), str(bundle), "tensor-inputs", str(queries), "--tensor-dispatch-approved"],
                             capture_output=True, timeout=120)
        after = g17packeddispatch.gpu_events()
    if run.returncode:
        raise RuntimeError("tensor dispatch failed: " + run.stderr.decode(errors="replace")[-2000:])
    if before != after:
        raise RuntimeError("GPU diagnostics changed during the dispatch")
    frames = TCR._frames(run.stdout)
    if len(frames) != queries + 1 or frames[0][0].get("sequence") != 0:
        raise ValueError("tensor worker handshake or frame count differs from the contract")
    out = []
    for q, (header, payload) in enumerate(frames[1:], 1):
        if (header.get("sequence"), header.get("status"), header.get("gpu_dispatched"),
                header.get("boundary_guard"), header.get("readonly_inputs")) != (q, 0, True, True, True):
            raise ValueError("tensor worker response %d is not fully checked" % q)
        out.append(bytes(payload))
    return out


def compare_words(got, want):
    g = np.frombuffer(got, "<u4")
    w = np.frombuffer(want, "<u4")
    return int(np.sum(g != w))


def compare_halves(got, want, off, count):
    g = np.frombuffer(got, "<u2")[off // 2:off // 2 + count]
    w = np.frombuffer(want, "<u2")[off // 2:off // 2 + count]
    return int(np.sum(g != w))


# ---------------------------------------------------------------------------------------------------
# stages for the pipeline (tools/g17decodestep_gpu.py): (program key, layout, build, io) from a stage's
# actual inputs, and the stage outputs read back out of buffer 3. A layout refusal (ValueError) is the
# pipeline's cue to fall back to the host stub.

# THE DECODE STEP's NORMS (Piece B, MM 25.136.6): 32 threadgroups each forming the whole sum of squares in the
# reference's order with the loads (and the half conversions) issued first, each scaling its 1/32 of the row;
# bit-exact, 43 -> 6.6 us. in_dtype -> (groups, unroll, hoist); a width they do not tile keeps the straight line
RMSNORM_LOOP = {"half": (32, 16, True), "float": (32, 8, True)}


def rmsnorm_stage(spec, v, g, in_dtype, loop=True):
    if loop and in_dtype in RMSNORM_LOOP:
        groups, unroll, hoist = RMSNORM_LOOP[in_dtype]
        try:
            lay = rmsnorm_loop_layout(spec.d_model, in_dtype, groups=groups, unroll=unroll, hoist=hoist)
        except ValueError:
            lay = None
        if lay is not None:
            key = "rmsnorm_%s_d%d_g%d_u%d%s" % (in_dtype, spec.d_model, groups, unroll, "_h" if hoist else "")
            return key, lay, (lambda: build_rmsnorm_loop(lay, spec.norm_eps)), rmsnorm_io(lay, v, g, spec)
    lay = rmsnorm_layout(spec.d_model, in_dtype)
    key = "rmsnorm_%s_d%d" % (in_dtype, spec.d_model)
    return key, lay, (lambda: build_rmsnorm(lay, spec.norm_eps)), rmsnorm_io(lay, v, g, spec)


def rmsnorm_out(lay, c):
    return np.frombuffer(c, np.float16, lay["d"], lay["OUT"]).astype(F32)


def rope_stage(spec, qkv32, rope_cos, rope_sin, k_cache, v_cache, cache="p9"):
    kvh = spec.kv_heads
    if cache == "flash":
        cap = 272
        lay = rope_layout(spec.n_heads, spec.head_dim, cap, groups=16, cache="flash", kv_heads=kvh)
        key = "rope_h%d_d%d_c%d_flash_kv%d" % (spec.n_heads, spec.head_dim, cap, kvh)
        kc = np.zeros((spec.n_heads, cap, spec.head_dim), F32); kc[:, :spec.kv_len] = k_cache[:, :spec.kv_len]
        vc = np.zeros_like(kc); vc[:, :spec.kv_len] = v_cache[:, :spec.kv_len]
        qs = D.q_scale(spec)
        return key, lay, (lambda: build_rope_append(lay, qs)), rope_io(lay, spec, qkv32, rope_cos, rope_sin, kc, vc,
                                                                     spec.kv_len)
    chained = cache == "chained"
    cache = "grid" if chained else cache
    if cache == "grid":
        # the attention grid's layout at its whole capacity: one program for every length (MM 25.138.2)
        import agxforge.g17.runtime as R
        cap = R.ATTENTION_GRID_CAPACITY * D.KEY_BLOCK
        lay = rope_layout(spec.n_heads, spec.head_dim, cap, groups=16, cache="grid", kv_heads=kvh, chained=chained)
        key = "rope_h%d_d%d_c%d_%s%s" % (spec.n_heads, spec.head_dim, cap, "chained" if chained else "grid",
                                          "_kv%d" % kvh if kvh != spec.n_heads else "")
        kc = np.zeros((spec.n_heads, cap, spec.head_dim), F32); kc[:, :spec.kv_len] = k_cache[:, :spec.kv_len]
        vc = np.zeros_like(kc); vc[:, :spec.kv_len] = v_cache[:, :spec.kv_len]
        k_cache, v_cache = kc, vc
    else:
        cap = spec.key_blocks * D.KEY_BLOCK
        lay = rope_layout(spec.n_heads, spec.head_dim, cap, kv_heads=kvh)
        key = "rope_h%d_d%d_c%d%s" % (spec.n_heads, spec.head_dim, cap, "_kv%d" % kvh if kvh != spec.n_heads else "")
    qs = D.q_scale(spec)
    return key, lay, (lambda: build_rope_append(lay, qs)), rope_io(lay, spec, qkv32, rope_cos, rope_sin, k_cache,
                                                                 v_cache, spec.kv_len)


def rope_out(lay, c, length):
    """q16, k_new, v_new, k_all and v_all (the cache's first length + 1 rows), as the reference names them."""
    H, hd, cap = lay["n_heads"], lay["head_dim"], lay["capacity"]
    if lay.get("cache") == "flash":
        q16 = np.frombuffer(c, np.float16, H * hd, lay["QF"]).astype(F32).reshape(H, hd)
        kvh, gsz = lay["kv_heads"], H // lay["kv_heads"]
        kc = np.repeat(np.frombuffer(c, np.float16, kvh * cap * hd, lay["KC"]).astype(F32).reshape(kvh, cap, hd), gsz, 0)
        vc = np.repeat(np.frombuffer(c, np.float16, kvh * cap * hd, lay["VC"]).astype(F32).reshape(kvh, cap, hd), gsz, 0)
        return dict(q16=q16, k_new=kc[:, length].copy(), v_new=vc[:, length].copy(),
                    k_all=kc[:, :length + 1].copy(), v_all=vc[:, :length + 1].copy())
    if lay.get("QA") is not None:
        q16 = np.stack([np.frombuffer(c, np.float16, hd, lay["QA"] + 8192 * h).astype(F32) for h in range(H)])
    else:
        q16 = np.frombuffer(c, np.float16, H * hd, lay["Q16"]).astype(F32).reshape(H, hd)
    if lay.get("cache") == "grid":
        ki = np.frombuffer(c, np.float16, H * GRID_HEAD_ELEMS, lay["KC"]).reshape(H, GRID_HEAD_ELEMS)
        vi = np.frombuffer(c, np.float16, H * GRID_HEAD_ELEMS, lay["VC"]).reshape(H, GRID_HEAD_ELEMS)
        kc, vc = grid_cache_arrays(ki, vi, cap)
    else:
        kc = np.frombuffer(c, np.float16, H * cap * hd, lay["KC"]).astype(F32).reshape(H, cap, hd)
        vc = np.frombuffer(c, np.float16, H * cap * hd, lay["VC"]).astype(F32).reshape(H, cap, hd)
    return dict(q16=q16, k_new=kc[:, length].copy(), v_new=vc[:, length].copy(),
                k_all=kc[:, :length + 1].copy(), v_all=vc[:, :length + 1].copy())


def swiglu_stage(spec, gate32, up32, groups=8):
    lay = swiglu_layout(spec.ffn_dim, groups)
    key = "swiglu_f%d" % spec.ffn_dim + ("" if groups == 8 else "_g%d" % groups)
    return key, lay, (lambda: build_swiglu(lay)), swiglu_io(lay, spec, gate32, up32)


def swiglu_out(lay, c):
    return np.frombuffer(c, np.float16, lay["ffn"], lay["OUT"]).astype(F32)


# ---------------------------------------------------------------------------------------------------
# the milestone arms: programs, inputs, expected outputs and the wrong-input controls

RESULTS = ROOT / "results" / "g17-decodeops-v1"


_MILESTONE_CACHE = {}


def milestone_inputs(spec=D.MILESTONE, seed=20260924):
    """The layer inputs and every stage input the reference step computes (its env). Computed once per
    (spec, seed) in a process: the milestone reference takes tens of seconds."""
    key = (spec, seed)
    if key not in _MILESTONE_CACHE:
        inp = D.make_inputs(spec, seed)
        _MILESTONE_CACHE[key] = (inp, D.reference(spec, inp)["env"])
    return _MILESTONE_CACHE[key]


def _rms_arg(v, spec):
    """The argument the rmsnorm's rsqrt receives (mean of squares + eps), for the probe."""
    from agxforge.g17 import tensorreduce as TR
    v = np.asarray(v, F32)
    sq = D.fmul(v, v).reshape(v.size // 32, 32)
    local = sq[0].copy()
    for i in range(1, v.size // 32):
        local = D.fadd(local, sq[i])
    lanes = TR.butterfly([float(x) for x in local], TR.ROW_BUTTERFLY_MASKS, "sum")
    lanes = TR.butterfly(list(lanes), TR.COLUMN_BUTTERFLY_MASKS, "sum")
    return D.fadd(D.fmul(F32(lanes[0]), F32(1.0 / v.size)), F32(spec.norm_eps))


def arms(spec=D.MILESTONE, seed=20260924, swiglu_groups=8):
    """Every dispatch arm at the milestone shape, in dispatch order. Each: name, the program key (arms
    sharing a key run ONE program), layout, build(), io = (a, b, c, expected C), and controls: the SAME
    GPU output scored against the expected C of a WRONG input (the program unchanged), each of which
    must differ in exactly the preregistered number of halves. `swiglu_groups` sets the SwiGLU arm's
    threadgroup count (swiglu_layout); the default is the pinned program."""
    inp, env = milestone_inputs(spec, seed)
    d, H, hd, f = spec.d_model, spec.n_heads, spec.head_dim, spec.ffn_dim
    out = []
    lay = probe_layout()
    rs, rc, ex = probe_inputs(lay, extra_rsqrt=(_rms_arg(inp["x"], spec), _rms_arg(env["h"], spec)))
    out.append(dict(name="probe", program="probe", layout=lay, build=lambda lay=lay: build_probe(lay),
                    io=probe_io(lay, rs, rc, ex), probe=(rs, rc, ex), controls={}))
    lay = rmsnorm_layout(d, "half")
    out.append(dict(name="attn_norm", program="rmsnorm_half", layout=lay,
                    build=lambda lay=lay: build_rmsnorm(lay, spec.norm_eps),
                    io=rmsnorm_io(lay, inp["x"], inp["g1"], spec),
                    controls={"x_rolled": rmsnorm_io(lay, np.roll(inp["x"], 1), inp["g1"], spec)[3]}))
    lay = rmsnorm_layout(d, "float")
    out.append(dict(name="ffn_norm", program="rmsnorm_float", layout=lay,
                    build=lambda lay=lay: build_rmsnorm(lay, spec.norm_eps),
                    io=rmsnorm_io(lay, env["h"], inp["g2"], spec),
                    controls={"g1_for_g2": rmsnorm_io(lay, env["h"], inp["g1"], spec)[3]}))
    cap = spec.key_blocks * D.KEY_BLOCK
    lay = rope_layout(H, hd, cap)
    qs = D.q_scale(spec)
    tabs = (inp["rope_cos"], inp["rope_sin"])
    cache = (inp["k_cache"], inp["v_cache"])
    out.append(dict(name="rope_append", program="rope_append", layout=lay,
                    build=lambda lay=lay: build_rope_append(lay, qs),
                    io=rope_io(lay, spec, env["qkv32"], *tabs, *cache, spec.kv_len),
                    controls={"stale_length": rope_io(lay, spec, env["qkv32"], *tabs, *cache, spec.kv_len - 1)[3],
                              "cos_sin_swapped": rope_io(lay, spec, env["qkv32"], *tabs[::-1], *cache, spec.kv_len)[3]}))
    # the SAME program at another length: only the buffer-1 length word (and the data) differ
    short = D.LayerSpec(**{**spec.as_dict(), "kv_len": 37})
    sinp = D.make_inputs(short, seed + 1)
    sqkv = D.gemv(D.rmsnorm(sinp["x"], sinp["g1"], short), sinp["wqkv"], k_chunk=short.k_chunk)
    stabs, scache = (sinp["rope_cos"], sinp["rope_sin"]), (sinp["k_cache"], sinp["v_cache"])
    out.append(dict(name="rope_append_len37", program="rope_append", layout=lay, build=None,
                    io=rope_io(lay, short, sqkv, *stabs, *scache, 37),
                    controls={"stale_length": rope_io(lay, short, sqkv, *stabs, *scache, 36)[3]}))
    lay = swiglu_layout(f, swiglu_groups)
    out.append(dict(name="ffn_swiglu", program="swiglu" if swiglu_groups == 8 else "swiglu_g%d" % swiglu_groups, layout=lay, build=lambda lay=lay: build_swiglu(lay),
                    io=swiglu_io(lay, spec, env["gate32"], env["up32"]),
                    controls={"gate_up_swapped": swiglu_io(lay, spec, env["up32"], env["gate32"])[3]}))
    return out


def _sha(data):
    import hashlib
    return hashlib.sha256(bytes(data)).hexdigest()


def halves_differing(x, y):
    return int(np.sum(np.frombuffer(x, "<u2") != np.frombuffer(y, "<u2")))


def preregistration(spec=D.MILESTONE, seed=20260924, swiglu_groups=8):
    """Before any dispatch: each program's hash, each arm's input and expected-output hashes, and each
    control's predicted count (halves of C in which the wrong input's expected C differs from the
    right one's: what the GPU output must differ by, if it equals the right one)."""
    import g17tensorcommonruntime as TCR
    rows, programs = [], {}
    for arm in arms(spec, seed, swiglu_groups):
        if arm["build"] is not None:
            p = arm["build"]()
            programs[arm["program"]] = {"program_sha256": TCR.sha(p.code), "code_bytes": len(p.code),
                                        "layout": {k: v for k, v in arm["layout"].items() if isinstance(v, (int, str))}}
        a, b, c, want = arm["io"]
        row = {"arm": arm["name"], "program": arm["program"],
               "inputs_sha256": _sha(a + b + c)[:16],
               "controls": {k: halves_differing(want, v) for k, v in arm["controls"].items()}}
        if arm["name"] == "probe":
            row["carrier_sha256"] = _sha(np.asarray(want, "<f4").tobytes())[:16]
            row["prediction"] = ("rsqrt_rn and recip_rn equal the correctly rounded value on every input; "
                                 "exp2_soft equals its model on every input; op3850, op3658 and op1272 raw "
                                 "are within one ulp (the corrections' precondition)")
        else:
            row["expected_c_sha256"] = _sha(want)[:16]
            row["prediction"] = "C equals the expected C in every word, 3 of 3 queries"
        rows.append(row)
    return {"spec": spec.as_dict(), "seed": seed, "programs": programs, "arms": rows}


def run(workdir: Path, spec=D.MILESTONE, seed=20260924, only=None, swiglu_groups=8):
    """Author every program once (by key), dispatch every arm (one process each, 3 queries), score."""
    import g17tensorcommonruntime as TCR
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    worker = workdir / "common-worker"
    if not worker.exists():
        TCR.build_worker(worker)
    bundles, report = {}, []
    all_arms = arms(spec, seed, swiglu_groups)
    for arm in all_arms:
        if only and arm["name"] not in only:
            continue
        a, b, c, want = arm["io"]
        key = arm["program"]
        if key not in bundles:
            builder = arm["build"] or next(x["build"] for x in all_arms if x["program"] == key and x["build"])
            meta = author(workdir / key, arm["layout"], builder(), a, b, c, extra={"program": key})
            bundles[key] = (workdir / key, meta)
        path, meta = bundles[key]
        outs = dispatch(path, worker, queries=3, inputs=(a, b, c))
        same = all(o == outs[0] for o in outs)
        row = {"arm": arm["name"], "program": key, "program_sha256": meta["program_sha256"],
               "queries": len(outs), "queries_identical": same}
        if arm["name"] == "probe":
            rs, rc, ex = arm["probe"]
            lay = arm["layout"]
            carrier = np.frombuffer(outs[0], "<f4")[:CARRIER * CARRIER * lay["groups"]]
            row["carrier_bit_exact"] = bool(np.array_equal(carrier.view(np.uint32),
                                                           np.asarray(want, "<f4").ravel().view(np.uint32)))
            row["score"] = probe_score(lay, rs, rc, ex, outs[0])
            (workdir / "probe-out.bin").write_bytes(outs[0])
        else:
            row["words_differing"] = [compare_words(o, want) for o in outs]
            row["bit_exact"] = all(n == 0 for n in row["words_differing"])
            row["controls"] = {k: halves_differing(outs[0], v) for k, v in arm["controls"].items()}
            (workdir / ("%s-out.bin" % arm["name"])).write_bytes(outs[0])
        report.append(row)
        print(json.dumps(row), file=sys.stderr, flush=True)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("prereg", help="print the preregistration (hashes and predicted control counts)")
    r = sub.add_parser("run", help="author, dispatch and score every arm (GPU)")
    for x in (pr, r):
        x.add_argument("--swiglu-groups", type=int, default=8, help="the SwiGLU arm's threadgroups (default 8)")
    r.add_argument("workdir", type=Path)
    r.add_argument("--only", nargs="*")
    r.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    if args.cmd == "prereg":
        print(json.dumps(preregistration(swiglu_groups=args.swiglu_groups), indent=1))
        return 0
    rep = run(args.workdir, only=args.only, swiglu_groups=args.swiglu_groups)
    text = json.dumps({"spec": D.MILESTONE.as_dict(), "arms": rep}, indent=1)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)
    ok = all(r.get("bit_exact", True) and r["queries_identical"] for r in rep)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())


# ---- a wide-threadgroup RMSNorm (MM 25.141.4) -----------------------------------------------------------------------

def build_rmsnorm_wide(lay, eps, tpg=1024):
    """RMSNorm as ONE threadgroup of `tpg` threads (a multiple of 32, d = 2 tpg or a multiple of tpg): thread t owns
    elements t, t + tpg, ...; its squares are summed in that order (the first a product, then fadd of each next
    product); the simdgroup's lanes combine with the row then column butterflies; lane 0 of simdgroup s stores the
    partial at threadgroup word s; after one barrier EVERY simdgroup reads the tpg/32 partials (lane l reads word l,
    lanes past tpg/32 read word 0 and are not summed... they read zero) and butterflies them again, so every lane
    holds the same total without a second barrier; then mean, eps, the corrected rsqrt and the scale, as
    build_rmsnorm. Offsets as build_rmsnorm_loop (x at X, gain at G, the fp16 row out at OUT), no carrier; the
    bindings are the cooperative three-binding class's (the harness binds [c, a, b] at 0, 1, 2)."""
    from agxforge.g17 import cc, ir, tensorreduce as TR
    d = lay["d"]
    if tpg % 32 or d % tpg or tpg > 1024:
        raise ValueError("rmsnorm_wide: tpg a multiple of 32, at most 1024, dividing d")
    per, nsg = d // tpg, tpg // 32
    # THE COOPERATIVE THREE-BINDING CLASS (MM 25.140.3): slot 0 written (the row out), slot 1 x, slot 2 the gain;
    # system registers 156 and 164 only, so the lane is t & 31 rather than its own register
    c = ir.Buffer("C", 0, elem=ir.F32); a = ir.Buffer("A", 1, elem=ir.F16); bb = ir.Buffer("B", 2, elem=ir.F16)
    fn = ir.Function("tensor_gemm_generic_runtime_demo", [c, a, bb])
    fn.declare_threadgroup(33 if lay.get("rs_once") else 32, size=(tpg, 1, 1))
    b = ir.Builder(fn, fn.block("entry"))
    xunit = 2 if lay["in_dtype"] == "half" else 4
    t0 = b.builtin("thread_position_in_threadgroup", name="t0")
    tg = b.builtin("threadgroup_position_in_grid", name="tg")          # 0; read so the class's registers match
    t = b.add(t0, b.mul(tg, _c(b, 0, "tgz"), name="tg0"), name="t")
    lane = getattr(b, "and")(t, ir.Imm(31), name="lane")
    sg = b.shr(t, _c(b, 5, "five"), name="sg")
    # BATCHED (lay["batch"], MM 25.144.3): threadgroup b normalizes row b - its x and its out are offset b d elements
    # (vector-major, the multi-vector qmv's layout); the gain is shared
    tr = b.add(t, b.mul(tg, _c(b, d, "rowd"), name="rowoff"), name="tr") if lay.get("batch", 1) > 1 else t

    def load_v(i, tag):
        idx = b.add(tr, _c(b, i * tpg + lay["X"] // xunit, tag + "_o"), name=tag + "_i")
        if lay["in_dtype"] == "half":
            return b.f16_to_f32(b.load(a, idx, width="half", name=tag + "_h"), name=tag)
        return b.load(a, idx, type=ir.I32, name=tag)
    vs = [load_v(i, "v%d" % i) for i in range(per)]
    acc = None
    for i, v in enumerate(vs):
        sq = b.fmul(v, v, type=ir.F32, name="sq%d" % i)
        acc = sq if acc is None else b.fadd(acc, sq, type=ir.F32, name="acc%d" % i)
    s = TR.emit_butterfly(b, acc, TR.ROW_BUTTERFLY_MASKS, operation="sum")
    s = TR.emit_butterfly(b, s, TR.COLUMN_BUTTERFLY_MASKS, operation="sum")
    # lane 0 of each simdgroup publishes its partial
    pub, joined = fn.block("publish"), fn.block("published")
    b.br_cond(b.cmp(lane, 1, "lt", name="lane0"), pub, joined)
    b.at(pub)
    b.store_tg(b.fadd(s, _cf(b, F32(0.0), "pz"), type=ir.F32, name="part"), sg)
    b.br(joined)
    b.at(joined)
    b.barrier("threadgroup")
    # every simdgroup reads all partials: lane l < nsg reads word l, the rest contribute exact zeros
    word = b.csel(lane, _c(b, nsg - 1, "nsgm1"), _c(b, 0, "w0"), lane, rel="gt", name="word")
    p = b.load_tg(word, type=ir.I32, name="p")
    p = b.fadd(p, _cf(b, F32(0.0), "pz2"), type=ir.I32, name="pv")
    p = b.csel(lane, _c(b, nsg - 1, "nsgm1b"), _cf(b, F32(0.0), "zero"), p, rel="gt", name="pm")
    p = b.fadd(p, _cf(b, F32(0.0), "pz3"), type=ir.F32, name="pf")
    tot = TR.emit_butterfly(b, p, TR.ROW_BUTTERFLY_MASKS, operation="sum")
    tot = TR.emit_butterfly(b, tot, TR.COLUMN_BUTTERFLY_MASKS, operation="sum")
    mean = b.fmul(tot, _cf(b, F32(1.0 / d), "inv_d"), name="mean")
    if lay.get("rs_seed"):
        # THE HARDWARE SEED ITSELF (op3850, MM 25.141.16): an exact function, modelled by g17decodestep.rsqrt_seed,
        # so every thread forms r in one instruction with no region and no second barrier
        r = b.rsqrt(b.fadd(mean, _cf(b, F32(eps), "eps"), type=ir.I32, name="var"), type=ir.I32, name="rs_seed")
    elif lay.get("rs_once"):
        # THE CORRECTED RSQRT ONCE, by simdgroup 0 (about 110 of the program's 196 IR ops): it publishes r at word
        # 32, and after a second barrier every thread reads it. The other simdgroups SKIP the region (cc's skip
        # branch, 25.141.2) rather than walking it masked. Same arithmetic, so the same value.
        fn.skip_regions = True
        rsb, rsj = fn.block("rs_one"), fn.block("rs_done")
        b.br_cond(b.cmp(sg, 1, "lt", name="sg0"), rsb, rsj)
        b.at(rsb)
        K = emit_constants(b)
        r0 = emit_rn(b, "rsqrt", b.fadd(mean, _cf(b, F32(eps), "eps"), type=ir.I32, name="var"), K, "rs")
        b.store_tg(b.fadd(r0, _cf(b, F32(0.0), "rz"), type=ir.F32, name="rpub"), _c(b, 32, "w32"))
        b.br(rsj)
        b.at(rsj)
        b.barrier("threadgroup")
        r = b.fadd(b.load_tg(_c(b, 32, "w32r"), type=ir.I32, name="r_ld"), _cf(b, F32(0.0), "rz2"), type=ir.I32, name="r")
    else:
        K = emit_constants(b)
        r = emit_rn(b, "rsqrt", b.fadd(mean, _cf(b, F32(eps), "eps"), type=ir.I32, name="var"), K, "rs")
    # the scale pass as a COUNTED LOOP over the thread's elements (the class is witnessed above 31 instructions
    # only with a back edge, MM 25.140.4); element i is reloaded, so nothing is carried but the index
    hdr, post = fn.block("scale_loop"), fn.block("scale_done")
    e0 = t
    j0 = _c(b, 0, "j0")
    b.br(hdr)
    b.at(hdr)
    j = b.phi(j0, name="j")
    e = b.phi(e0, name="e")
    batched = lay.get("batch", 1) > 1
    er = b.add(e, b.mul(tg, _c(b, d, "rowd2"), name="rowoff2"), name="er") if batched else e
    xi = b.add(er, _c(b, lay["X"] // xunit, "xs_o"), name="xs_i")
    if lay["in_dtype"] == "half":
        v = b.f16_to_f32(b.load(a, xi, width="half", name="xs_h"), name="xs")
    else:
        v = b.fadd(b.load(a, xi, type=ir.I32, name="xs_l"), _cf(b, F32(0.0), "xs_z"), type=ir.I32, name="xs")
    g = b.f16_to_f32(b.load(bb, b.add(e, _c(b, lay["G"] // 2, "g_o"), name="g_i"), width="half", name="g_h"), name="g")
    y = b.fmul(b.fmul(v, r, name="vr"), g, name="y")
    if lay.get("out32"):
        # fp32 of the fp16-rounded row (exact widening): the next qmv reads it as xvec fp32 x
        b.store_at(c, b.add(er, _c(b, lay["OUT"] // 4, "o_o"), name="o_i"),
                   b.f16_to_f32(b.f32_to_f16_rte(y, name="yh"), name="yw"))
    else:
        b.store_at(c, b.add(er, _c(b, lay["OUT"] // 2, "o_o"), name="o_i"), b.f32_to_f16_rte(y, name="yh"), width="half")
    jn = b.add(j, ir.Imm(1), name="j_next")
    en = b.add(e, _c(b, tpg, "tpg"), name="e_next")
    ir.Builder.phi_latch(j, jn)
    ir.Builder.phi_latch(e, en)
    b.br_cond(b.cmp(jn, per, "lt", name="more"), hdr, post)
    b.at(post)
    b.ret()
    ir.verify(fn)
    return cc.compile_function(fn)


def rmsnorm_wide_rows_reference(V, g, spec, tpg=1024, seed=False):
    """rmsnorm_wide_reference of every row of V [R, d] at once: the same operations per row (the lane butterflies as
    tensorreduce.butterfly_array). A row whose butterfly values are not all finite is recomputed by
    rmsnorm_wide_reference itself, so overflow behaves as it does there. test_g17simspeed checks bit-identity."""
    from agxforge.g17 import tensorreduce as TR
    V = np.asarray(V, F32)
    R, d = V.shape
    per, nsg = d // tpg, tpg // 32
    sq = D.fmul(V, V).reshape(R, per, tpg)
    acc = sq[:, 0].copy()
    for i in range(1, per):
        acc = D.fadd(acc, sq[:, i])
    lanes = TR.butterfly_array(TR.butterfly_array(acc.reshape(R, nsg, 32), TR.ROW_BUTTERFLY_MASKS, "sum"),
                               TR.COLUMN_BUTTERFLY_MASKS, "sum")
    pl = np.zeros((R, 32), F32)
    pl[:, :nsg] = lanes[:, :, 0]
    tot = TR.butterfly_array(TR.butterfly_array(pl, TR.ROW_BUTTERFLY_MASKS, "sum"), TR.COLUMN_BUTTERFLY_MASKS, "sum")[:, 0]
    ok = np.isfinite(acc).all(1) & np.isfinite(lanes).all((1, 2)) & np.isfinite(tot)
    out = np.zeros(V.shape, F32)
    if ok.any():
        r = (D.rsqrt_seed if seed else D.rsqrt)(D.fadd(D.fmul(tot[ok].astype(F32), F32(1.0 / d)), F32(spec.norm_eps)))
        out[ok] = D.narrow(D.fmul(D.fmul(V[ok], r[:, None]), np.asarray(g, F32)[None, :]), spec.storage)
    for i in np.nonzero(~ok)[0]:
        out[i] = rmsnorm_wide_reference(V[i], g, spec, tpg=tpg, seed=seed)
    return out


def rmsnorm_wide_reference(v, g, spec, tpg=1024, seed=False):
    """build_rmsnorm_wide's value: per-thread ordered squares, the lane butterflies per simdgroup, then the
    butterflies over the partials (zeros past tpg/32), mean, eps, rsqrt, scale."""
    from agxforge.g17 import tensorreduce as TR
    v = np.asarray(v, F32); d = v.size; per, nsg = d // tpg, tpg // 32
    sq = D.fmul(v, v).reshape(per, tpg)
    acc = sq[0].copy()
    for i in range(1, per):
        acc = D.fadd(acc, sq[i])
    parts = []
    for s in range(nsg):
        lanes = TR.butterfly([float(x) for x in acc[32 * s:32 * s + 32]], TR.ROW_BUTTERFLY_MASKS, "sum")
        lanes = TR.butterfly(list(lanes), TR.COLUMN_BUTTERFLY_MASKS, "sum")
        parts.append(F32(lanes[0]))
    pl = [float(parts[l]) if l < nsg else 0.0 for l in range(32)]
    tot = TR.butterfly(pl, TR.ROW_BUTTERFLY_MASKS, "sum")
    tot = TR.butterfly(list(tot), TR.COLUMN_BUTTERFLY_MASKS, "sum")
    r = (D.rsqrt_seed if seed else D.rsqrt)(D.fadd(D.fmul(F32(tot[0]), F32(1.0 / d)), F32(spec.norm_eps)))
    return D.narrow(D.fmul(D.fmul(v, r), np.asarray(g, F32)), spec.storage)
