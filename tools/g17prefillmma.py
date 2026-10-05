#!/usr/bin/env python3
"""Causal prefill attention on the TENSOR UNITS (MM 25.144.2), with its own stated order, bit-exact against
prefill_mma_reference.

One simdgroup runs one KV head's pair of q heads over one 16-query block: the 32-row tile holds q head 2 kh's 16
queries in rows 0..15 and q head 2 kh + 1's in rows 16..31 (GQA folded into the tile). Per key block j (16 keys):
  QK   S [32 x 16] fp32 = Q [32 x 128] @ K_j^T   one tensor body, K from buffer 3 under transB, eight 16-wide issues
                                                 in ascending K (g17tensorcommonruntime._mma16, C first)
  row  lane r owns row r (exact scalar code, the order stated here):
         valid_k = key j*16 + k <= q0_r (q0_r = p0 + (r & 15));  mb = fmax over valid s_k, k ascending (NEG_MAX if none)
         m' = fmax(m, mb);  al = exp2_soft(m + (m' * -1));  p_k = valid ? exp2_soft(s_k + (m' * -1)) : +0 (written over S)
         l = (l * al) + (((p_0 + p_1) + p_2) ... + p_15);  O[r, :] = O[r, :] * al;  m = m'   (m, l in the stats tiles)
  PV   O_sl [32 x 16] = P @ V_j,sl + O_sl for sl = 0..7: eight tensor bodies, P fp32 (the fp32-A mode truncates it to
       10 mantissa bits, modelled), V from buffer 2 BLOCK-MAJOR (block j, slice sl: a contiguous 16 x 16), ACCUMULATE
       (tlower: the chain, then one fadd of C)
then out[r, d] = fp16_rne(O[r, d] * recip_rn(l_r)).
State written before block 0: O = 0, m = NEG_MAX, l = 0.

This file holds the PROBE (one threadgroup, straight-line blocks) that established the route on hardware; the
bucketed grid form builds on it.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import g17attn as A  # noqa: E402
import g17decodeops as O  # noqa: E402

F32 = np.float32
D = 128


def probe_layout(nb, p0):
    """Buffer 3 (fp32 words unless said): S at 0 (32 x 16), eight O tiles at 2048 + 2048 sl, M at 18432, L at 18560,
    out fp16 [32][128] at 18688, K cache blocks [nb][16][128] fp16 at 26880 (+ 4096 j). Buffer 1: Q fp16 [32][128].
    Buffer 2: V block-major [nb][8][16][16] fp16."""
    return dict(nb=nb, p0=p0, S=0, OT=2048, M=18432, L=18560, OUT=18688, KC=26880,
                c_bytes=26880 + nb * 4096, a_bytes=32 * D * 2, b_bytes=nb * 8 * 512)


def _row_stage(b, ir, c, lane, lay, j, K2, KR=None, last=False):
    """The exact row stage for key block j (lane r = row r). Emits the normalisation too when `last`."""
    I = ir.I32
    C = lambda v, n: O._c(b, v, "j%d_%s" % (j, n))
    CF = lambda v, n: O._cf(b, F32(v), "j%d_%s" % (j, n))
    q0 = b.add(getattr(b, "and")(lane, ir.Imm(15), name="j%d_l15" % j), C(lay["p0"], "p0"), name="j%d_q0" % j)
    rowS = b.add(b.shl(lane, C(4, "s4"), name="j%d_r16" % j), C(lay["S"] // 4, "Sb"), name="j%d_rowS" % j)
    s = [b.load(c, b.add(rowS, C(k, "k%d" % k), name="j%d_si%d" % (j, k)) if k else rowS, type=I, name="j%d_s%d" % (j, k))
         for k in range(16)]
    negmax = CF(A.NEG_MAX, "negmax")
    mb = None
    for k in range(16):
        # valid when key j*16 + k <= q0, i.e. q0 > j*16 + k - 1
        v = b.csel(C(16 * j + k, "key%d" % k), q0, negmax, s[k], rel="gt", name="j%d_sv%d" % (j, k))
        mb = v if mb is None else b.fmax(mb, v, type=I, name="j%d_mb%d" % (j, k))
    mi = b.add(lane, C(lay["M"] // 4, "Mb"), name="j%d_mi" % j)
    li = b.add(lane, C(lay["L"] // 4, "Lb"), name="j%d_li" % j)
    m = b.load(c, mi, type=I, name="j%d_m" % j)
    l = b.load(c, li, type=I, name="j%d_l" % j)
    mn = b.fmax(m, mb, type=I, name="j%d_mn" % j)
    nmn = b.fmul(mn, CF(-1.0, "neg1"), type=I, name="j%d_nmn" % j)
    al = O.emit_exp2_soft(b, b.fadd(m, nmn, type=I, name="j%d_dm" % j), K2, "j%d_al" % j)
    zero = CF(0.0, "fz")
    ps = []
    for k in range(16):
        p = O.emit_exp2_soft(b, b.fadd(s[k], nmn, type=I, name="j%d_ds%d" % (j, k)), K2, "j%d_pe%d" % (j, k))
        p = b.csel(C(16 * j + k, "kp%d" % k), q0, zero, p, rel="gt", name="j%d_p%d" % (j, k))
        ps.append(p)
        b.store_at(c, b.add(rowS, C(k, "kw%d" % k), name="j%d_pw%d" % (j, k)) if k else rowS, p)
    ssum = ps[0]
    for k in range(1, 16):
        ssum = b.fadd(ssum, ps[k], type=I, name="j%d_ps%d" % (j, k))
    ln = b.fadd(b.fmul(l, al, type=I, name="j%d_la" % j), ssum, type=I, name="j%d_ln" % j)
    b.store_at(c, mi, mn)
    b.store_at(c, li, ln)
    # O[r, :] *= al: row r of tile sl is words OT/4 + sl*512 + r*16 + (0..15)
    rowO = b.add(b.shl(lane, C(4, "o4"), name="j%d_ro16" % j), C(lay["OT"] // 4, "OTb"), name="j%d_rowO" % j)
    for sl in range(8):
        for d in range(16):
            idx = b.add(rowO, C(sl * 512 + d, "o%d_%d" % (sl, d)), name="j%d_oi%d_%d" % (j, sl, d)) if (sl or d) else rowO
            ov = b.load(c, idx, type=I, name="j%d_ov%d_%d" % (j, sl, d))
            b.store_at(c, idx, b.fmul(ov, al, type=I, name="j%d_os%d_%d" % (j, sl, d)))


def _init_state(b, ir, c, lane, lay):
    I = ir.I32
    rowO = b.add(b.shl(lane, O._c(b, 4, "i_o4"), name="i_ro16"), O._c(b, lay["OT"] // 4, "i_OTb"), name="i_rowO")
    for sl in range(8):
        for d in range(16):
            idx = b.add(rowO, O._c(b, sl * 512 + d, "i_o%d_%d" % (sl, d)), name="i_oi%d_%d" % (sl, d)) if (sl or d) else rowO
            b.store_at(c, idx, O._cf(b, F32(0.0), "i_z%d_%d" % (sl, d)))
    b.store_at(c, b.add(lane, O._c(b, lay["M"] // 4, "i_Mb"), name="i_mi"), O._cf(b, A.NEG_MAX, "i_negmax"))
    b.store_at(c, b.add(lane, O._c(b, lay["L"] // 4, "i_Lb"), name="i_li"), O._cf(b, F32(0.0), "i_lz"))


def _finish(b, ir, c, lane, lay):
    I = ir.I32
    l = b.load(c, b.add(lane, O._c(b, lay["L"] // 4, "f_Lb"), name="f_li"), type=I, name="f_l")
    KR = O.emit_constants(b)
    rl = O.emit_rn(b, "recip", l, KR, "f_rl")
    rowO = b.add(b.shl(lane, O._c(b, 4, "f_o4"), name="f_ro16"), O._c(b, lay["OT"] // 4, "f_OTb"), name="f_rowO")
    rowY = b.add(b.shl(lane, O._c(b, 7, "f_y7"), name="f_ry"), O._c(b, lay["OUT"] // 2, "f_OUTb"), name="f_rowY")
    for sl in range(8):
        for d in range(16):
            idx = b.add(rowO, O._c(b, sl * 512 + d, "f_o%d_%d" % (sl, d)), name="f_oi%d_%d" % (sl, d)) if (sl or d) else rowO
            ov = b.load(c, idx, type=I, name="f_ov%d_%d" % (sl, d))
            y = b.fmul(ov, rl, type=I, name="f_y%d_%d" % (sl, d))
            b.store_at(c, b.add(rowY, O._c(b, sl * 16 + d, "f_yo%d_%d" % (sl, d)), name="f_yi%d_%d" % (sl, d)) if (sl or d) else rowY,
                       b.f32_to_f16_rte(y, name="f_yh%d_%d" % (sl, d)), width="half")


def build_probe(lay):
    """The straight-line probe: init, then per block QK, row stage, eight PV bodies; then the normalisation."""
    from agxforge.g17 import cc, ir
    fn, b, a, bb, c = O._function()
    lane = b.builtin("thread_index_in_simdgroup", name="lane")
    K2 = O.emit_exp2_constants(b)
    _init_state(b, ir, c, lane, lay)
    for j in range(lay["nb"]):
        b.tensor_matmul(a, c, c, M=32, N=16, K=D, transB=True, offsetA=0, offsetB=lay["KC"] + j * 4096, offsetC=lay["S"])
        _row_stage(b, ir, c, lane, lay, j, K2)
        for sl in range(8):
            b.tensor_matmul(c, bb, c, M=32, N=16, K=16, a_dtype="float", b_dtype="half", accumulate=True,
                            offsetA=lay["S"], offsetB=j * 4096 + sl * 512, offsetC=lay["OT"] + sl * 2048)
    _finish(b, ir, c, lane, lay)
    b.ret()
    ir.verify(fn)
    return cc.compile_function(fn)


# ------------------------------------------------------------------------------------------------------ reference
def mma_reference(Q16, Kb, Vb, p0):
    """Q16 [32][128], Kb [nb][16][128], Vb [nb][8][16][16] (fp16 values). Returns (out fp16 [32][128], the final O, l)."""
    import g17decodestep as DS
    import g17tensorcommonruntime as T
    nb = Kb.shape[0]
    Q = np.asarray(Q16, F32)
    Ot = np.zeros((8, 32, 16), F32)
    m = np.full(32, A.NEG_MAX, F32)
    l = np.zeros(32, F32)
    q0 = p0 + (np.arange(32) & 15)
    for j in range(nb):
        S = T._gemm_mma(Q, np.asarray(Kb[j], F32).T, None, 32, 16, D).astype(F32)
        keys = 16 * j + np.arange(16)
        valid = keys[None, :] <= q0[:, None]
        sv = np.where(valid, S, A.NEG_MAX).astype(F32)
        mb = sv[:, 0].copy()
        for k in range(1, 16):
            mb = np.maximum(mb, sv[:, k])
        mn = np.maximum(m, mb)
        nmn = (mn * F32(-1.0)).astype(F32)
        al = DS.exp2_soft((m + nmn).astype(F32)).astype(F32)
        P = DS.exp2_soft((S + nmn[:, None]).astype(F32)).astype(F32)
        P = np.where(valid, P, F32(0.0)).astype(F32)
        ssum = P[:, 0].copy()
        for k in range(1, 16):
            ssum = (ssum + P[:, k]).astype(F32)
        l = ((l * al).astype(F32) + ssum).astype(F32)
        m = mn
        Ot = (Ot * al[None, :, None]).astype(F32)
        for sl in range(8):
            Ot[sl] = T._gemm_mma(P, np.asarray(Vb[j, sl], F32), Ot[sl], 32, 16, 16, truncate_a=True)
    rl = DS.recip(l).astype(F32)
    O_ = np.concatenate([Ot[sl] for sl in range(8)], axis=1)             # [32][128]
    return (O_ * rl[:, None]).astype(F32).astype(np.float16), O_, l


def gemm_mma_v(a, b, c=None, truncate_a=False):
    """g17tensorcommonruntime._gemm_mma, vectorised over leading axes: a [..., M, K], b [..., K, N] (fp16 or fp32
    values; a truncated to 10 mantissa bits when truncate_a), 16-wide issues in ascending K, each: exact products
    rounded to fp32, P_i = rne(p_2i + p_2i+1), Q_j = rne(P_j + P_j+4), acc = C (the previous issue) FIRST then + Q_0..3;
    the first issue starts at Q_0; then `c` added once (tlower's ACCUMULATE fadd)."""
    a = np.asarray(a, F32)
    if truncate_a:
        a = (a.view(np.uint32) & np.uint32(0xFFFFE000)).view(F32)
    a = a.astype(np.float64)
    b = np.asarray(b, F32).astype(np.float64)
    K = a.shape[-1]
    acc = None
    for s0 in range(0, K, 16):
        prod = (a[..., :, s0:s0 + 16, None] * b[..., None, s0:s0 + 16, :]).astype(F32).astype(np.float64)   # [.., M, 16, N]
        P = (prod[..., 0::2, :] + prod[..., 1::2, :]).astype(F32).astype(np.float64)                           # [.., M, 8, N]
        Qs = (P[..., 0:4, :] + P[..., 4:8, :]).astype(F32).astype(np.float64)                                  # [.., M, 4, N]
        if acc is None:
            acc = Qs[..., 0, :]
            rest = range(1, 4)
        else:
            rest = range(4)
        for jq in rest:
            acc = (acc + Qs[..., jq, :]).astype(F32).astype(np.float64)
    if c is not None:
        acc = (acc + np.asarray(c, F32).astype(np.float64)).astype(F32).astype(np.float64)
    return acc.astype(F32)


# ------------------------------------------------------------------------------------------------ the bucketed grid
SCRSZ = 20480                 # per-threadgroup scratch: S 2048 | 8 O tiles 16384 | M 128 | L 128 (padded)
T_S, T_OT, T_M, T_L, T_N = 0, 2048, 18432, 18560, 18688
RSCR = 1536                   # the register-O route's scratch: S 1024 | M 64 | L 64 | AL 64 | RL 64 | N (padded)
R_S, R_M, R_L, R_AL, R_RL, R_N = 0, 1024, 1088, 1152, 1216, 1280


def rego_scratch(sg, bk=16):
    """The register-O route's per-threadgroup scratch for `sg` simdgroups (bytes): S [16 sg][bk] fp32, then the per-row
    words M, L, AL, RL (16 sg each) and the trip-bound word N. sg 1 at bk 16 is RSCR / R_* exactly."""
    SB = 64 * bk * sg
    return dict(S=0, M=SB, L=SB + 64 * sg, AL=SB + 128 * sg, RL=SB + 192 * sg, N=SB + 256 * sg,
                size=A._align(SB + 256 * sg + 64))


def mma_layout(cap, M, p0, out16=False, rego=False, sg=1, bk=16):
    """A shape bucket: M prompt rows (a power-of-two multiple of 16, 128..2048) starting at p0 (a multiple of 16),
    NB = (p0 + M) / 16 key blocks. Every offset is REGION-3 ABSOLUTE (decode's region, extended), so the graph binds
    the one region at slots 1, 2 and 3 of the attention: Q tiles QT (slot 1), block-major V VB (slot 2), the K cache
    and the per-threadgroup scratch SCR (slot 3), output at PATTN."""
    import g17prefillattn as P
    if M % 16 or (M // 16) & (M // 16 - 1) or not 128 <= M <= 2048 or p0 % 16:
        raise ValueError("mma bucket: M a power-of-two multiple of 16 (128..2048), p0 a multiple of 16")
    lay = P.prefill_layout(cap, M, out16=out16)
    NB = (p0 + M) // 16
    if NB > 255 or p0 + M > cap:
        raise ValueError("mma bucket: p0 + M within the cache, at most 255 key blocks")
    KVH = lay["kv_heads"]
    nq = M // 16
    QT = A._align(lay["prefill_region3_bytes"])
    VB = QT + A._align(KVH * nq * 32 * D * 2)
    SCR = VB + A._align(KVH * cap * D * 2)
    if rego:
        # the register-O route: one threadgroup per (q head, sg query blocks), sg simdgroups of 16 rows each; V read
        # from the decode cache
        if sg not in (1, 2, 4) or nq % sg:
            raise ValueError("register-O: 1, 2 or 4 simdgroups, dividing the query blocks")
        if 2 * nq // sg > 128:
            raise ValueError("register-O: %d slices per KV head exceed head_slices' 128; use more simdgroups" % (2 * nq // sg))
        # bk: keys per block of the key loop (MM 25.162): 16, or 32 (half the trips, one rescale per 32 keys)
        if bk not in (16, 32) or (p0 + M) % bk:
            raise ValueError("register-O: bk is 16 or 32, dividing p0 + M")
        RS = rego_scratch(sg, bk)
        grid = KVH * 2 * nq // sg
        end = SCR + grid * RS["size"]
        return dict(lay, mma=True, rego=True, sg=sg, RS=RS, p0=p0, M=M, nq=nq, NB=NB, QT=QT, VB=VB, SCR=SCR,
                    mma_region3_bytes=A._align(end), grid=grid, **({"bk": bk} if bk != 16 else {}))
    end = SCR + KVH * nq * SCRSZ
    return dict(lay, mma=True, p0=p0, M=M, nq=nq, NB=NB, QT=QT, VB=VB, SCR=SCR, mma_region3_bytes=A._align(end),
                grid=KVH * nq)


def build_mma(lay):
    """Threadgroup t = kvh * nq + qb (head_slices = nq): one simdgroup, the 32-row tile of kv head kvh's two q heads over
    query block qb. A counted loop of NB key blocks (compile-time): QK (K from the decode cache, register "k" +
    kvh * CAP * 256 bytes), the exact row stage (runtime causal mask; a block past every row's edge is an exact identity
    step), eight PV bodies (block-major V, register "v"). Then the normalisation, written as attn32 or fp16 at PATTN."""
    from agxforge.g17 import cc, ir
    fn, b, a, bb, c = O._function()
    I = ir.I32
    CAP, nq, NB = lay["cap"], lay["nq"], lay["NB"]
    lane = b.builtin("thread_index_in_simdgroup", name="lane")
    t = b.builtin("threadgroup_position_in_grid", name="tg")
    lgq = nq.bit_length() - 1
    kvh = b.shr(t, O._c(b, lgq, "lgq"), name="kvh")
    qb = getattr(b, "and")(t, O._c(b, nq - 1, "qmask"), name="qb")
    base = b.add(b.mul(t, O._c(b, SCRSZ // 4, "scrw"), name="tscr"), O._c(b, lay["SCR"] // 4, "scrb"), name="base")
    q0 = b.add(b.add(b.shl(qb, O._c(b, 4, "qb16"), name="qb16v"), getattr(b, "and")(lane, ir.Imm(15), name="l15"), name="qrel"),
               O._c(b, lay["p0"], "p0c"), name="q0")
    rowS = b.add(base, b.shl(lane, O._c(b, 4, "s4"), name="r16"), name="rowS")
    rowO = b.add(rowS, O._c(b, T_OT // 4, "otw"), name="rowO")
    mi = b.add(base, b.add(lane, O._c(b, T_M // 4, "mw"), name="lm"), name="mi")
    li = b.add(base, b.add(lane, O._c(b, T_L // 4, "lw"), name="ll"), name="li")
    K2 = O.emit_exp2_constants(b)
    for sl in range(8):
        for d in range(16):
            b.store_at(c, b.add(rowO, O._c(b, sl * 512 + d, "i%d_%d" % (sl, d)), name="io%d_%d" % (sl, d)) if (sl or d) else rowO,
                       O._cf(b, F32(0.0), "iz%d_%d" % (sl, d)))
    b.store_at(c, mi, O._cf(b, A.NEG_MAX, "inegmax"))
    b.store_at(c, li, O._cf(b, F32(0.0), "ilz"))
    if lay.get("runtime_trips"):
        b.store_at(c, b.add(base, O._c(b, T_N // 4, "nw0"), name="ni0"),
                   b.add(qb, O._c(b, lay["p0"] // 16 + 1, "nbase"), name="trips0"))
    b.tensor_index_init("k", lay["KOFF"])
    b.tensor_index_init("v", lay["VB"])
    counter0 = b.const(0, name="blk0")
    kb0 = O._c(b, 0, "kb0")
    hdr, ex = fn.block("mma_keys"), fn.block("mma_done")
    b.br(hdr)
    b.at(hdr)
    i = b.phi(counter0, name="blk")
    kbase = b.phi(kb0, name="kbase")
    b.tensor_matmul(a, c, c, M=32, N=16, K=D, transB=True, offsetA=lay["QT"], offsetB=0, offsetC=lay["SCR"] + T_S,
                    head_stride=(nq * 32 * D * 2, CAP * D * 2, nq * SCRSZ), head_slices=nq, slice_stride=(32 * D * 2, 0, SCRSZ),
                    offsetB_register="k", offsetB_step=16 * D * 2)
    s = [b.load(c, b.add(rowS, O._c(b, k, "sk%d" % k), name="si%d" % k) if k else rowS, type=I, name="s%d" % k) for k in range(16)]
    negmax = O._cf(b, A.NEG_MAX, "negmax")
    zero = O._cf(b, F32(0.0), "fz")
    # e = the number of this block's keys the row sees: q0 + 1 - kbase when kbase <= q0, else 0 (never negative, so
    # the unsigned compares below are exact); key k is masked iff k + 1 > e
    e = b.csel(kbase, q0, O._c(b, 0, "ez"), b.sub(b.add(q0, O._c(b, 1, "q01"), name="q0p1"), kbase, name="e_raw"), rel="gt", name="e")
    mb = None
    for k in range(16):
        v = b.csel(O._c(b, k + 1, "kk%d" % k), e, negmax, s[k], rel="gt", name="sv%d" % k)
        mb = v if mb is None else b.fmax(mb, v, type=I, name="mb%d" % k)
    m = b.load(c, mi, type=I, name="m")
    l = b.load(c, li, type=I, name="l")
    mn = b.fmax(m, mb, type=I, name="mn")
    nmn = b.fmul(mn, O._cf(b, F32(-1.0), "neg1"), type=I, name="nmn")
    al = O.emit_exp2_soft(b, b.fadd(m, nmn, type=I, name="dm"), K2, "al")
    ssum = None
    for k in range(16):
        p = O.emit_exp2_soft(b, b.fadd(s[k], nmn, type=I, name="ds%d" % k), K2, "pe%d" % k)
        p = b.csel(O._c(b, k + 1, "kp%d" % k), e, zero, p, rel="gt", name="p%d" % k)
        b.store_at(c, b.add(rowS, O._c(b, k, "pk%d" % k), name="pw%d" % k) if k else rowS, p)
        ssum = p if ssum is None else b.fadd(ssum, p, type=I, name="ps%d" % k)
    ln = b.fadd(b.fmul(l, al, type=I, name="la"), ssum, type=I, name="ln")
    b.store_at(c, mi, mn)
    b.store_at(c, li, ln)
    for sl in range(8):
        for d in range(16):
            idx = b.add(rowO, O._c(b, sl * 512 + d, "o%d_%d" % (sl, d)), name="oi%d_%d" % (sl, d)) if (sl or d) else rowO
            b.store_at(c, idx, b.fmul(b.load(c, idx, type=I, name="ov%d_%d" % (sl, d)), al, type=I, name="os%d_%d" % (sl, d)))
    for sl in range(8):
        b.tensor_matmul(c, bb, c, M=32, N=16, K=16, a_dtype="float", b_dtype="half", accumulate=True,
                        offsetA=lay["SCR"] + T_S, offsetB=0, offsetC=lay["SCR"] + T_OT + sl * 2048,
                        head_stride=(nq * SCRSZ, (CAP // 16) * 8 * 512, nq * SCRSZ), head_slices=nq,
                        slice_stride=(SCRSZ, 0, SCRSZ), offsetB_register="v", offsetB_step=512)
    kn = b.add(kbase, O._c(b, 16, "k16"), name="kbase_next")
    if lay.get("runtime_trips"):
        # THE CAUSAL SKIP (cc's capped runtime trip count, M8): this threadgroup's query block qb sees key blocks
        # 0 .. p0/16 + qb only; the later blocks are exact identity steps, so skipping them changes no value. The bound
        # is RELOADED here from the threadgroup's scratch word T_N (written before the loop): a waited load of a
        # per-threadgroup uniform value. (An SR-derived bound would also be correct: an unwaited slot-0 SR read is
        # measured correct at distance 0, MM 25.117 / 25.121; this form is the one receipted on hardware.)
        n = b.load(c, b.add(base, O._c(b, T_N // 4, "nw"), name="ni_latch"), type=I, name="trips_rt")
        nxt = b.add(i, ir.Imm(1), name="blk_next")
        ir.Builder.phi_latch(i, nxt)
        ir.Builder.phi_latch(kbase, kn)
        b.br_cond(b.cmp(nxt, n, "lt", cap=NB, name="more"), hdr, ex)
    else:
        nxt = b.add(i, ir.Imm(1), name="blk_next")          # last before the latch (the static latch check's shape)
        ir.Builder.phi_latch(i, nxt)
        ir.Builder.phi_latch(kbase, kn)
        b.br_cond(b.cmp(nxt, NB, "lt", name="more"), hdr, ex)
    b.at(ex)
    lf = b.load(c, li, type=I, name="lf")
    rl = O.emit_rn(b, "recip", lf, O.emit_constants(b), "rl")
    qrow = b.add(b.shl(qb, O._c(b, 4, "f16"), name="fqb"), getattr(b, "and")(lane, ir.Imm(15), name="fl15"), name="fqrow")
    head = b.add(b.shl(kvh, O._c(b, 1, "f2"), name="fkv2"), b.shr(lane, O._c(b, 4, "f4"), name="fhh"), name="fhead")
    orow = b.shl(b.add(b.shl(qrow, O._c(b, 4, "fH"), name="fqH"), head, name="fqh"), O._c(b, 7, "fD"), name="forow")
    attn32 = not lay.get("out16")
    ob = b.add(orow, O._c(b, lay["PATTN"] // (4 if attn32 else 2), "fpat"), name="fob")
    for sl in range(8):
        for d in range(16):
            idx = b.add(rowO, O._c(b, sl * 512 + d, "fo%d_%d" % (sl, d)), name="foi%d_%d" % (sl, d)) if (sl or d) else rowO
            y = b.fmul(b.load(c, idx, type=I, name="fov%d_%d" % (sl, d)), rl, type=I, name="fy%d_%d" % (sl, d))
            yh = b.f32_to_f16_rte(y, name="fyh%d_%d" % (sl, d))
            oi = b.add(ob, O._c(b, sl * 16 + d, "fw%d_%d" % (sl, d)), name="fwi%d_%d" % (sl, d)) if (sl or d) else ob
            if attn32:
                b.store_at(c, oi, b.f16_to_f32(yh, name="fyw%d_%d" % (sl, d)))
            else:
                b.store_at(c, oi, yh, width="half")
    b.ret()
    ir.verify(fn)
    return cc.compile_function(fn)


def build_mma_rego(lay):
    """THE REGISTER-O ROUTE (M8's register accumulator, MM 25.144.8). Threadgroup t = kvh * 2nq + hh * nq + qb
    (head_slices = 2 nq): one simdgroup, q head h = t / nq over query block qb, a 16-row tile. O [16 x 128] lives in
    eight accumulator tiles (64 registers) for the whole key loop. Per key block: QK (M 16), the exact row stage
    (lane r and r + 16 both run row r & 15, identical values), al published per row through scratch and read back by
    the lanes that hold that row's accumulator registers (tensor_acc_position), the rescale in registers, then ONE
    PV body, N = 128, V straight from the decode cache (register "v"), acc = "O". Every row and column sees the same
    operations in the same order as build_mma, so mma_prefill_reference is this route's reference too.

    lay["sg"] = 2 or 4 (mma_layout(..., rego=True, sg=S)): one threadgroup per (q head, S query blocks), simdgroup s
    owning rows 16 s .. 16 s + 15 of the threadgroup's tile (M8's simdgroups: A and C split by rows, K and V read by
    all), threadgroup t = kvh * 2nq/S + hh * nq/S + qg. Each simdgroup's row stage touches only its own rows, so
    nothing crosses simdgroups; the causal skip bounds the threadgroup by its LAST query block."""
    from agxforge.g17 import cc, ir
    fn, b, a, bb, c = O._function()
    I = ir.I32
    CAP, nq, NB, H = lay["cap"], lay["nq"], lay["NB"], lay["heads"]
    C = lambda v, n: O._c(b, v, n)
    sg = lay.get("sg", 1)
    lgs = sg.bit_length() - 1
    ng = nq // sg                       # threadgroups per (kv head, head half): each takes sg query blocks
    lgq = ng.bit_length() - 1
    BK = lay.get("bk", 16)              # keys per block (MM 25.162); 32 needs the row2 stage
    if BK not in (16, 32) or (BK == 32 and not lay.get("row2")):
        raise ValueError("register-O: bk is 16, or 32 with the row2 stage")
    lbk = BK.bit_length() - 1
    NBL = NB * 16 // BK                 # key-loop trips when every block runs
    RS = lay.get("RS") or rego_scratch(1, BK)
    R_S, R_M, R_L, R_AL, R_RL, R_N, RSZ = (RS[k] for k in ("S", "M", "L", "AL", "RL", "N", "size"))
    sgkw = dict(simdgroups=sg) if sg > 1 else {}
    # lay["fold"]: M8's fold_offsets - each body's large region offsets are added to its index registers once in the
    # prologue, so every tile displacement fits (instead of a per-tile, per-trip base re-derivation)
    if lay.get("fold"):
        sgkw["fold_offsets"] = True
    # lay["hoist"]: M8's hoist_prologue - each looped body's loop-invariant prologue runs once before the loop
    if lay.get("hoist"):
        sgkw["hoist_prologue"] = True
    # TIMING ABLATIONS (never delivered; the output is wrong by construction): "exp" makes exp2_soft the identity,
    # "rescale" drops the 64-register O rescale, "rowstage" drops the whole row stage (PV reads S as P)
    ablate = set(lay.get("ablate", ()))
    # lay["hw_exp2"]: the row stage's 2^t is the HARDWARE exp2 (op1272, b.exp2) in ONE instruction, not the ~14-op
    # exp2_soft (MM 25.144.2). op1272 is within 1 ulp of true exp2 but not bit-exactly reproducible on the CPU
    # (isa/g17-exp2-op1272-characterization.json), so a hw_exp2 kernel is NOT bit-identical to the exp2_soft
    # reference - it is validated by an enclosure of true softmax (mma_prefill_reference_trueexp) and by model
    # tokens. OFF by default: every delivered bit-exact kernel keeps exp2_soft.
    if lay.get("hw_exp2"):
        exp2 = lambda b_, x, K2, tag: b_.exp2(x, type=I, name=tag)
    elif "exp" in ablate:
        exp2 = lambda b_, x, K2, tag: x
    else:
        exp2 = O.emit_exp2_soft

    def where(g):
        """This lane's addresses, recomputed from the builtins in each region (g = a name prefix), each made only when
        first asked for: nothing is held across the key loop but its two counters, so the scalar code fits beside the
        64 accumulator registers."""
        made = {}

        def get(k):
            if k not in made:
                made[k] = make[k]()
            return made[k]
        make = dict(
            lane=lambda: b.builtin("thread_index_in_simdgroup", name=g + "lane"),
            t=lambda: b.builtin("threadgroup_position_in_grid", name=g + "tg"),
            qb=lambda: getattr(b, "and")(get("t"), C(ng - 1, g + "qmask"), name=g + "qb"),
            base=lambda: b.add(b.mul(get("t"), C(RSZ // 4, g + "scrw"), name=g + "tscr"), C(lay["SCR"] // 4, g + "scrb"),
                               name=g + "base"),
            sgi=lambda: getattr(b, "and")(b.builtin("simdgroup_index_in_threadgroup", name=g + "sgr"), ir.Imm(sg - 1),
                                          name=g + "sgi"),
            # the simdgroup's first row within the threadgroup (0 for one simdgroup)
            sg16=lambda: b.shl(get("sgi"), C(4, g + "sg4"), name=g + "sg16"),
            r0=lambda: getattr(b, "and")(get("lane"), ir.Imm(15), name=g + "r0"),
            r=lambda: get("r0") if sg == 1 else b.add(get("sg16"), get("r0"), name=g + "r"),
            # this lane's accumulator rows: ra (slots 0..3) and ra + 8 (slots 4..7) (ir.tensor_acc_position)
            ra=lambda: b.add(b.shl(getattr(b, "and")(b.shr(get("lane"), C(4, g + "l4"), name=g + "lb4"), ir.Imm(1), name=g + "lb4m"),
                                   C(2, g + "x4r"), name=g + "ra4"),
                             getattr(b, "and")(b.shr(get("lane"), C(1, g + "l1"), name=g + "lb1"), ir.Imm(3), name=g + "lb21"),
                             name=g + "ra0" if sg > 1 else g + "ra") if sg == 1 else
               b.add(get("sg16"), b.add(b.shl(getattr(b, "and")(b.shr(get("lane"), C(4, g + "l4"), name=g + "lb4"), ir.Imm(1),
                                                                  name=g + "lb4m"), C(2, g + "x4r"), name=g + "ra4"),
                                         getattr(b, "and")(b.shr(get("lane"), C(1, g + "l1"), name=g + "lb1"), ir.Imm(3),
                                                           name=g + "lb21"), name=g + "ra0"), name=g + "ra"),
            mi=lambda: b.add(get("base"), b.add(get("r"), C(R_M // 4, g + "mw"), name=g + "lm"), name=g + "mi"),
            li=lambda: b.add(get("base"), b.add(get("r"), C(R_L // 4, g + "lw"), name=g + "ll"), name=g + "li"))

        class W:
            def __getitem__(self, k):
                return get(k)
        return W()

    w0 = where("i_")
    base, mi, li = w0["base"], w0["mi"], w0["li"]
    zero = O._cf(b, F32(0.0), "iz")
    for tt in range(8):
        for s in range(8):
            b.tensor_acc_write("O", s, zero, tile=(0, tt))
    b.store_at(c, mi, O._cf(b, A.NEG_MAX, "inegmax"))
    b.store_at(c, li, zero)
    if lay.get("runtime_trips"):
        # the threadgroup's last query block is qb * sg + sg - 1: it sees key blocks 0 .. p0/16 + qb * sg + sg - 1
        qs = w0["qb"] if sg == 1 else b.shl(w0["qb"], C(lgs, "qsg"), name="qsgv")
        ni0 = b.add(base, C(R_N // 4, "nw0"), name="ni0")        # made first: the bk 16 bytes are unchanged
        t16 = b.add(qs, C(lay["p0"] // 16 + sg, "nbase"), name="trips0")
        if BK == 32:                    # the 16-key block count rounded up to 32-key blocks
            t16 = b.shr(b.add(t16, ir.Imm(1), name="trips0r"), C(1, "trips0s"), name="trips0h")
        b.store_at(c, ni0, t16)
    b.tensor_index_init("k", lay["KOFF"])
    b.tensor_index_init("v", lay["VOFF"])
    counter0 = b.const(0, name="blk0")
    kb0 = C(0, "kb0")
    # lay["holdk"]: the exp2 constants made once before the loop and held across it (fits only with the prologue hoisted)
    K2_held = O.emit_exp2_constants(b) if (lay.get("holdk") and not lay.get("hw_exp2")) else None
    hdr, ex = fn.block("mma_keys"), fn.block("mma_done")
    b.br(hdr)
    b.at(hdr)
    i = b.phi(counter0, name="blk")
    kbase = b.phi(kb0, name="kbase")
    b.tensor_matmul(a, c, c, M=16 * sg, N=BK, K=D, transB=True, offsetA=lay["QT"], offsetB=0, offsetC=lay["SCR"] + R_S,
                    head_stride=(2 * nq * 16 * D * 2, CAP * D * 2, 2 * ng * RSZ), head_slices=2 * ng,
                    slice_stride=(sg * 16 * D * 2, 0, RSZ), offsetB_register="k", offsetB_step=BK * D * 2, **sgkw)
    # everything the row stage needs is made AFTER the QK body (the exp2 constants, fourteen immediates a trip, and
    # the addresses from the builtins): live across no tensor body, it may share the bodies' working registers (cc,
    # M8); made before QK it would be live across it and cannot fit beside the accumulator
    rowstage = "rowstage" not in ablate
    K2 = None if lay.get("hw_exp2") else ((K2_held if lay.get("holdk") else O.emit_exp2_constants(b)) if rowstage and "exp" not in ablate else None)
    zero = O._cf(b, F32(0.0), "fz") if rowstage else None
    w = where("k_")
    base, r, ra, mi, li = w["base"], w["r"], w["ra"], w["mi"], w["li"]
    def _row_stage_rego():
        q0 = b.add(b.add(b.shl(w["qb"], C(4 + lgs, "qb16"), name="qb16v"), r, name="qrel"), C(lay["p0"], "p0c"), name="q0")
        rowS = b.add(base, b.shl(r, C(4, "s4"), name="r16"), name="rowS")
        # the scores are loaded twice (mask/max, then exp) rather than held: 16 live scores do not fit beside the 64
        # accumulator registers
        sidx = lambda k, tag: b.add(rowS, C(k, "%s%d" % (tag, k)), name="%si%d" % (tag, k)) if k else rowS
        s = [None] * 16
        negmax = O._cf(b, A.NEG_MAX, "negmax")
        e = b.csel(kbase, q0, C(0, "ez"), b.sub(b.add(q0, C(1, "q01"), name="q0p1"), kbase, name="e_raw"), rel="gt", name="e")
        mb = None
        for k in range(16):
            v = b.csel(C(k + 1, "kk%d" % k), e, negmax, b.load(c, sidx(k, "sa"), type=I, name="sa%d" % k), rel="gt", name="sv%d" % k)
            mb = v if mb is None else b.fmax(mb, v, type=I, name="mb%d" % k)
        m = b.load(c, mi, type=I, name="m")
        l = b.load(c, li, type=I, name="l")
        mn = b.fmax(m, mb, type=I, name="mn")
        nmn = b.fmul(mn, O._cf(b, F32(-1.0), "neg1"), type=I, name="nmn")
        al = exp2(b, b.fadd(m, nmn, type=I, name="dm"), K2, "al")
        ssum = None
        for k in range(16):
            p = exp2(b, b.fadd(b.load(c, sidx(k, "sb"), type=I, name="sb%d" % k), nmn, type=I, name="ds%d" % k), K2, "pe%d" % k)
            p = b.csel(C(k + 1, "kp%d" % k), e, zero, p, rel="gt", name="p%d" % k)
            b.store_at(c, b.add(rowS, C(k, "pk%d" % k), name="pw%d" % k) if k else rowS, p)
            ssum = p if ssum is None else b.fadd(ssum, p, type=I, name="ps%d" % k)
        ln = b.fadd(b.fmul(l, al, type=I, name="la"), ssum, type=I, name="ln")
        b.store_at(c, mi, mn)
        b.store_at(c, li, ln)
        b.store_at(c, b.add(base, b.add(r, C(R_AL // 4, "alw"), name="lal"), name="ali"), al)
        ala = b.add(base, b.add(ra, C(R_AL // 4, "alwa"), name="lala"), name="alia")
        als = (b.load(c, ala, type=I, name="al_a"), b.load(c, b.add(ala, C(8, "al8"), name="alib"), type=I, name="al_b"))
        for tt in range(8 if "rescale" not in ablate else 0):
            for sl in range(8):
                if lay.get("scale"):        # M8's in-place register multiply: the same IEEE product, one instruction
                    b.tensor_acc_scale("O", sl, als[sl >> 2], tile=(0, tt))
                else:
                    b.tensor_acc_write("O", sl, b.fmul(b.tensor_acc_read("O", sl, tile=(0, tt), name="o%d_%d" % (tt, sl)),
                                                        als[sl >> 2], type=I, name="os%d_%d" % (tt, sl)), tile=(0, tt))
    def _row_stage_rego2():
        """The same row stage, cheaper per trip, every value and order unchanged (lay["row2"]):
          - address offsets as add immediates, not a materialised constant each;
          - the 16 exps split between the two lanes that run a row: lane half h computes p_k for k in 8h .. 8h + 7 and
            stores it over s_k (only its own keys, after BOTH halves have read all 16 scores for the max); then every
            lane reloads p_0 .. p_15 and sums them in the stated order, so l is bit-identical.
        The row max stays whole in each lane (fmax of +0 and -0 is not order-free)."""
        IM = ir.Imm
        q0 = b.add(b.add(b.shl(w["qb"], C(4 + lgs, "qb16"), name="qb16v"), r, name="qrel"), C(lay["p0"], "p0c"), name="q0")
        rowS = b.add(base, b.shl(r, C(lbk, "s4"), name="r16"), name="rowS")
        sidx = lambda k, tag: b.add(rowS, IM(k), name="%si%d" % (tag, k)) if k else rowS
        negmax = O._cf(b, A.NEG_MAX, "negmax")
        e = b.csel(kbase, q0, C(0, "ez"), b.sub(b.add(q0, IM(1), name="q0p1"), kbase, name="e_raw"), rel="gt", name="e")
        mb = None
        for k in range(BK):
            v = b.csel(C(k + 1, "kk%d" % k), e, negmax, b.load(c, sidx(k, "sa"), type=I, name="sa%d" % k), rel="gt", name="sv%d" % k)
            mb = v if mb is None else b.fmax(mb, v, type=I, name="mb%d" % k)
        m = b.load(c, mi, type=I, name="m")
        l = b.load(c, li, type=I, name="l")
        mn = b.fmax(m, mb, type=I, name="mn")
        nmn = b.fmul(mn, O._cf(b, F32(-1.0), "neg1"), type=I, name="nmn")
        al = exp2(b, b.fadd(m, nmn, type=I, name="dm"), K2, "al")
        hk8 = (b.shr(getattr(b, "and")(w["lane"], IM(16), name="lh16"), IM(1), name="hk8") if BK == 16 else
               getattr(b, "and")(w["lane"], IM(16), name="hk8"))   # this lane half's first key: 0 or BK / 2
        rowH = b.add(rowS, hk8, name="rowH")
        for j in range(BK // 2):
            at = b.add(rowH, IM(j), name="ph%d" % j) if j else rowH
            p = exp2(b, b.fadd(b.load(c, at, type=I, name="sb%d" % j), nmn, type=I, name="ds%d" % j), K2, "pe%d" % j)
            p = b.csel(b.add(hk8, IM(j + 1), name="kp%d" % j), e, zero, p, rel="gt", name="p%d" % j)
            b.store_at(c, at, p)
        ssum = None
        for k in range(BK):
            p = b.load(c, sidx(k, "pr"), type=I, name="pr%d" % k)
            ssum = p if ssum is None else b.fadd(ssum, p, type=I, name="ps%d" % k)
        ln = b.fadd(b.fmul(l, al, type=I, name="la"), ssum, type=I, name="ln")
        b.store_at(c, mi, mn)
        b.store_at(c, li, ln)
        b.store_at(c, b.add(base, b.add(r, IM(R_AL // 4), name="lal"), name="ali"), al)
        ala = b.add(base, b.add(ra, IM(R_AL // 4), name="lala"), name="alia")
        als = (b.load(c, ala, type=I, name="al_a"), b.load(c, b.add(ala, IM(8), name="alib"), type=I, name="al_b"))
        for tt in range(8 if "rescale" not in ablate else 0):
            for sl in range(8):
                if lay.get("scale"):
                    b.tensor_acc_scale("O", sl, als[sl >> 2], tile=(0, tt))
                else:
                    b.tensor_acc_write("O", sl, b.fmul(b.tensor_acc_read("O", sl, tile=(0, tt), name="o%d_%d" % (tt, sl)),
                                                        als[sl >> 2], type=I, name="os%d_%d" % (tt, sl)), tile=(0, tt))
    if rowstage:
        (_row_stage_rego2 if lay.get("row2") else _row_stage_rego)()
    b.tensor_matmul(c, bb, c, M=16 * sg, N=D, K=BK, a_dtype="float", b_dtype="half", accumulate=True, acc="O",
                    offsetA=lay["SCR"] + R_S, offsetB=0,
                    head_stride=(2 * ng * RSZ, CAP * D * 2, 0), head_slices=2 * ng, slice_stride=(RSZ, 0, 0),
                    offsetB_register="v", offsetB_step=BK * D * 2, **sgkw)
    kn = b.add(kbase, C(BK, "k16"), name="kbase_next")
    if lay.get("runtime_trips"):
        n = b.load(c, b.add(base, C(R_N // 4, "nw"), name="ni_latch"), type=I, name="trips_rt")
        nxt = b.add(i, ir.Imm(1), name="blk_next")
        ir.Builder.phi_latch(i, nxt)
        ir.Builder.phi_latch(kbase, kn)
        b.br_cond(b.cmp(nxt, n, "lt", cap=NBL, name="more"), hdr, ex)
    else:
        nxt = b.add(i, ir.Imm(1), name="blk_next")
        ir.Builder.phi_latch(i, nxt)
        ir.Builder.phi_latch(kbase, kn)
        b.br_cond(b.cmp(nxt, NBL, "lt", name="more"), hdr, ex)
    b.at(ex)
    w = where("f_")
    lane, base, r, ra, li, qb = w["lane"], w["base"], w["r"], w["ra"], w["li"], w["qb"]
    h = b.shr(w["t"], C(lgq, "lgq"), name="h")
    lf = b.load(c, li, type=I, name="lf")
    rl = O.emit_rn(b, "recip", lf, O.emit_constants(b), "rl")
    b.store_at(c, b.add(base, b.add(r, C(R_RL // 4, "rlw"), name="lrl"), name="rli"), rl)
    rla = b.add(base, b.add(ra, C(R_RL // 4, "rlwa"), name="lrla"), name="rlia")
    rls = (b.load(c, rla, type=I, name="rl_a"), b.load(c, b.add(rla, C(8, "rl8"), name="rlib"), type=I, name="rl_b"))
    # out [M][H][D]: row qb*16 + ra (+ 8), head h, column 16 t + 8 lane[3] + 4 lane[0] + (slot & 3)
    cb = b.add(b.shl(getattr(b, "and")(b.shr(lane, C(3, "l3"), name="lb3"), ir.Imm(1), name="lb3m"), C(3, "x8c"), name="cb8"),
               b.shl(getattr(b, "and")(lane, ir.Imm(1), name="lb0"), C(2, "x4c"), name="cb4"), name="cb")
    qrow = b.add(b.shl(qb, C(4 + lgs, "f16"), name="fqb"), ra, name="fqrow")
    orow = b.add(b.shl(b.add(b.mul(qrow, C(H, "fH"), name="fqH"), h, name="fqh"), C(7, "fD"), name="forow0"), cb, name="forow")
    attn32 = not lay.get("out16")
    oba = b.add(orow, C(lay["PATTN"] // (4 if attn32 else 2), "fpat"), name="foba")
    obs = (oba, b.add(oba, C(8 * H * D, "f8r"), name="fobb"))
    for tt in range(8):
        for sl in range(8):
            y = b.fmul(b.tensor_acc_read("O", sl, tile=(0, tt), name="fo%d_%d" % (tt, sl)), rls[sl >> 2], type=I, name="fy%d_%d" % (tt, sl))
            yh = b.f32_to_f16_rte(y, name="fyh%d_%d" % (tt, sl))
            w = 16 * tt + (sl & 3)
            oi = b.add(obs[sl >> 2], C(w, "fw%d_%d" % (tt, sl)), name="fwi%d_%d" % (tt, sl)) if w else obs[sl >> 2]
            if attn32:
                b.store_at(c, oi, b.f16_to_f32(yh, name="fyw%d_%d" % (tt, sl)))
            else:
                b.store_at(c, oi, yh, width="half")
    b.ret()
    ir.verify(fn)
    return cc.compile_function(fn)


def build_mma_sreg(lay):
    """THE REGISTER-S ROUTE (MM 25.163): build_mma_rego with S and P never leaving registers. Per 16-key block:
      - S (16 x 16 fp32) is a register accumulator "S" (the ninth accumulator group, cc.TENSOR_ACC_EXTRA_GROUPS):
        zeroed, then QK accumulates onto it, so S = +0 + the MMA chain (a -0 score reads +0);
      - the row stage reads the lane's own eight slots (rows ra and ra + 8, columns 8 lane[3] + 4 lane[0] + slot[1:0],
        ir.tensor_acc_position), masks the causal edge per slot, and reduces each row over its four lanes: the four
        slots in order, then simd_shuffle_xor 1 and 8 (fmax; for the sum ((c0 + c1) + c2) + c3 per lane, + the xor-1
        partner, + the xor-8 partner - every lane of a row ends with the same value);
      - P = exp2(s - m) is written back into S, the O rescale reads the lane's own al (O's rows are S's rows), and PV
        takes A from "S" (a_acc: D fed as A is the identity, fp32 with no instruction).
    Only m and l stay in scratch, one word per row (loop-carried scalars are not admitted), written alike by the four
    lanes of a row. One simdgroup, bk 16 or 32 (two S tiles: per tile its four slots, the tiles in order, then the
    shuffles). The stated order is mma_prefill_reference(lay) with lay["sreg"]."""
    from agxforge.g17 import cc, ir
    BK = lay.get("bk", 16)
    if lay.get("sg", 1) != 1 or BK not in (16, 32):
        raise ValueError("register-S: one simdgroup, 16- or 32-key blocks")
    if BK == 32:
        # measured (MM 25.163): the tenth accumulator group leaves the 16 x 32 QK body no register plan in 126 registers
        raise ValueError("refused: register-S at bk 32 - ten accumulator groups leave QK (1 x 2 tiles) no register plan")
    NT = BK // 16                       # S tiles per block: at bk 32 the tenth accumulator group
    fn, b, a, bb, c = O._function()
    I = ir.I32
    IM = ir.Imm
    CAP, nq, NB, H = lay["cap"], lay["nq"], lay["NB"], lay["heads"]
    NBL = NB * 16 // BK
    C = lambda v, n: O._c(b, v, n)
    ng = nq
    lgq = ng.bit_length() - 1
    RS = lay.get("RS") or rego_scratch(1, BK)
    R_M, R_L, R_N, RSZ = (RS[k] for k in ("M", "L", "N", "size"))
    qkw = {}
    if lay.get("fold"):
        qkw["fold_offsets"] = True
    pvkw = dict(qkw)                    # PV takes A from registers: no hoisted prologue (it needs an A index)
    if lay.get("hoist"):
        qkw["hoist_prologue"] = True
    if lay.get("hw_exp2"):
        exp2 = lambda b_, x, K2, tag: b_.exp2(x, type=I, name=tag)
    else:
        exp2 = O.emit_exp2_soft

    def lanes(g, with_ra=True, with_qb=True):
        lane = b.builtin("thread_index_in_simdgroup", name=g + "lane")
        t = b.builtin("threadgroup_position_in_grid", name=g + "tg")
        qb = getattr(b, "and")(t, C(ng - 1, g + "qmask"), name=g + "qb") if with_qb else None
        base = b.add(b.mul(t, C(RSZ // 4, g + "scrw"), name=g + "tscr"), C(lay["SCR"] // 4, g + "scrb"), name=g + "base")
        ra = None if not with_ra else b.add(
            b.shl(getattr(b, "and")(b.shr(lane, C(4, g + "l4"), name=g + "lb4"), IM(1), name=g + "lb4m"), C(2, g + "x4r"),
                  name=g + "ra4"),
            getattr(b, "and")(b.shr(lane, C(1, g + "l1"), name=g + "lb1"), IM(3), name=g + "lb21"), name=g + "ra")
        return lane, t, qb, base, ra

    # the per-row m / l words: every lane of a row stores them, alike; lane r < 16 initialises row r
    lane0, t0, qb0, base0, _ = lanes("i_", with_ra=False, with_qb=bool(lay.get("runtime_trips")))
    r0 = getattr(b, "and")(lane0, IM(15), name="i_r0")
    zero = O._cf(b, F32(0.0), "iz")
    for tt in range(8):
        for sl in range(8):
            b.tensor_acc_write("O", sl, zero, tile=(0, tt))
    b.store_at(c, b.add(base0, b.add(r0, C(R_M // 4, "i_mw"), name="i_lm"), name="i_mi"), O._cf(b, A.NEG_MAX, "inegmax"))
    b.store_at(c, b.add(base0, b.add(r0, C(R_L // 4, "i_lw"), name="i_ll"), name="i_li"), zero)
    if lay.get("runtime_trips"):
        ni0 = b.add(base0, C(R_N // 4, "nw0"), name="ni0")
        t16 = b.add(qb0, C(lay["p0"] // 16 + 1, "nbase"), name="trips0")
        if BK == 32:                    # the 16-key block count rounded up to 32-key blocks
            t16 = b.shr(b.add(t16, IM(1), name="trips0r"), C(1, "trips0s"), name="trips0h")
        b.store_at(c, ni0, t16)
    b.tensor_index_init("k", lay["KOFF"])
    b.tensor_index_init("v", lay["VOFF"])
    counter0 = b.const(0, name="blk0")
    kb0 = C(0, "kb0")
    K2_held = O.emit_exp2_constants(b) if (lay.get("holdk") and not lay.get("hw_exp2")) else None
    hdr, ex = fn.block("mma_keys"), fn.block("mma_done")
    b.br(hdr)
    b.at(hdr)
    i = b.phi(counter0, name="blk")
    kbase = b.phi(kb0, name="kbase")
    zs = O._cf(b, F32(0.0), "sz")
    for tn in range(NT):
        for sl in range(8):
            b.tensor_acc_write("S", sl, zs, tile=(0, tn))
    b.tensor_matmul(a, c, c, M=16, N=BK, K=D, transB=True, offsetA=lay["QT"], offsetB=0, accumulate=True, acc="S",
                    head_stride=(2 * nq * 16 * D * 2, CAP * D * 2, 0), head_slices=2 * ng,
                    slice_stride=(16 * D * 2, 0, 0), offsetB_register="k", offsetB_step=BK * D * 2, **qkw)
    K2 = None if lay.get("hw_exp2") else (K2_held if lay.get("holdk") else O.emit_exp2_constants(b))
    fz = O._cf(b, F32(0.0), "fz")
    lane, t, qb, base, ra = lanes("k_")
    cb = b.add(b.shl(getattr(b, "and")(b.shr(lane, C(3, "c3"), name="cb3"), IM(1), name="cb3m"), C(3, "c8"), name="cb8"),
               b.shl(getattr(b, "and")(lane, IM(1), name="cb0"), C(2, "c4"), name="cb4"), name="cb")
    negmax = O._cf(b, A.NEG_MAX, "negmax")
    q0 = b.add(b.add(b.shl(qb, C(4, "qb16"), name="qb16v"), ra, name="qrel"), C(lay["p0"], "p0c"), name="q0")
    rows = []                                  # (half, row index, valid-key bound e) for rows ra and ra + 8
    for hf in range(2):
        q = q0 if hf == 0 else b.add(q0, IM(8), name="q8")
        e = b.csel(kbase, q, C(0, "ez%d" % hf), b.sub(b.add(q, IM(1), name="qp1_%d" % hf), kbase, name="er%d" % hf),
                   rel="gt", name="e%d" % hf)
        # the slot's key offset in the block is cb + j; it is valid while cb + j + 1 <= e
        rows.append((hf, ra if hf == 0 else b.add(ra, IM(8), name="ra8"), e))
    for hf, rr, e in rows:
        sv = []
        for tn in range(NT):
            for j in range(4):
                sl = 4 * hf + j
                v = b.tensor_acc_read("S", sl, tile=(0, tn), name="s%d_%d" % (tn, sl))
                v = b.csel(b.add(cb, IM(16 * tn + j + 1), name="kj%d_%d" % (tn, sl)), e, negmax, v, rel="gt",
                           name="sv%d_%d" % (tn, sl))
                sv.append(v)
        mx = sv[0]
        for j in range(1, len(sv)):
            mx = b.fmax(mx, sv[j], type=I, name="mx%d_%d" % (hf, j))
        for mask in (1, 8):
            mx.type = ir.F32
            mx = b.fmax(mx, b.simd_shuffle_xor(mx, mask, name="mxs%d_%d" % (hf, mask)), type=I, name="mxb%d_%d" % (hf, mask))
        mi = b.add(base, b.add(rr, C(R_M // 4, "mw%d" % hf), name="lm%d" % hf), name="mi%d" % hf)
        li = b.add(base, b.add(rr, C(R_L // 4, "lw%d" % hf), name="ll%d" % hf), name="li%d" % hf)
        m = b.load(c, mi, type=I, name="m%d" % hf)
        l = b.load(c, li, type=I, name="l%d" % hf)
        mn = b.fmax(m, mx, type=I, name="mn%d" % hf)
        nmn = b.fmul(mn, O._cf(b, F32(-1.0), "neg1_%d" % hf), type=I, name="nmn%d" % hf)
        al = exp2(b, b.fadd(m, nmn, type=I, name="dm%d" % hf), K2, "al%d" % hf)
        ps = None                       # per S tile its four slots in order, then the tiles in order
        for tn in range(NT):
            pt = None
            for j in range(4):
                sl = 4 * hf + j
                p_ = exp2(b, b.fadd(b.tensor_acc_read("S", sl, tile=(0, tn), name="sr%d_%d" % (tn, sl)), nmn, type=I,
                                    name="ds%d_%d" % (tn, sl)), K2, "pe%d_%d" % (tn, sl))
                p_ = b.csel(b.add(cb, IM(16 * tn + j + 1), name="kp%d_%d" % (tn, sl)), e, fz, p_, rel="gt",
                            name="p%d_%d" % (tn, sl))
                b.tensor_acc_write("S", sl, p_, tile=(0, tn))
                pt = p_ if pt is None else b.fadd(pt, p_, type=I, name="pt%d_%d_%d" % (hf, tn, j))
            ps = pt if ps is None else b.fadd(ps, pt, type=I, name="ps%d_%d" % (hf, tn))
        for mask in (1, 8):
            ps.type = ir.F32
            ps = b.fadd(ps, b.simd_shuffle_xor(ps, mask, name="pss%d_%d" % (hf, mask)), type=I, name="psb%d_%d" % (hf, mask))
        ln = b.fadd(b.fmul(l, al, type=I, name="la%d" % hf), ps, type=I, name="ln%d" % hf)
        b.store_at(c, mi, mn)
        b.store_at(c, li, ln)
        for tt in range(8):
            for j in range(4):
                b.tensor_acc_scale("O", 4 * hf + j, al, tile=(0, tt))
    b.tensor_matmul(c, bb, c, M=16, N=D, K=BK, a_dtype="float", b_dtype="half", accumulate=True, acc="O", a_acc="S",
                    offsetA=0, offsetB=0, head_stride=(0, CAP * D * 2, 0), head_slices=2 * ng, slice_stride=(0, 0, 0),
                    offsetB_register="v", offsetB_step=BK * D * 2, **pvkw)
    kn = b.add(kbase, C(BK, "k16"), name="kbase_next")
    nxt = b.add(i, IM(1), name="blk_next")
    ir.Builder.phi_latch(i, nxt)
    ir.Builder.phi_latch(kbase, kn)
    if lay.get("runtime_trips"):
        n = b.load(c, b.add(base, C(R_N // 4, "nw"), name="ni_latch"), type=I, name="trips_rt")
        b.br_cond(b.cmp(nxt, n, "lt", cap=NBL, name="more"), hdr, ex)
    else:
        b.br_cond(b.cmp(nxt, NBL, "lt", name="more"), hdr, ex)
    b.at(ex)
    lane, t, qb, base, ra = lanes("f_")
    h = b.shr(t, C(lgq, "lgq"), name="h")
    KC = O.emit_constants(b)
    rls = []
    for hf in range(2):
        rr = ra if hf == 0 else b.add(ra, IM(8), name="fra8")
        lf = b.load(c, b.add(base, b.add(rr, C(R_L // 4, "flw%d" % hf), name="fll%d" % hf), name="fli%d" % hf), type=I,
                    name="lf%d" % hf)
        rls.append(O.emit_rn(b, "recip", lf, KC, "rl%d" % hf))
    cbf = b.add(b.shl(getattr(b, "and")(b.shr(lane, C(3, "l3"), name="lb3"), IM(1), name="lb3m"), C(3, "x8c"), name="cb8f"),
                b.shl(getattr(b, "and")(lane, IM(1), name="lb0"), C(2, "x4c"), name="cb4f"), name="cbf")
    qrow = b.add(b.shl(qb, C(4, "f16"), name="fqb"), ra, name="fqrow")
    orow = b.add(b.shl(b.add(b.mul(qrow, C(H, "fH"), name="fqH"), h, name="fqh"), C(7, "fD"), name="forow0"), cbf, name="forow")
    attn32 = not lay.get("out16")
    oba = b.add(orow, C(lay["PATTN"] // (4 if attn32 else 2), "fpat"), name="foba")
    obs = (oba, b.add(oba, C(8 * H * D, "f8r"), name="fobb"))
    for tt in range(8):
        for sl in range(8):
            y = b.fmul(b.tensor_acc_read("O", sl, tile=(0, tt), name="fo%d_%d" % (tt, sl)), rls[sl >> 2], type=I,
                       name="fy%d_%d" % (tt, sl))
            yh = b.f32_to_f16_rte(y, name="fyh%d_%d" % (tt, sl))
            w = 16 * tt + (sl & 3)
            oi = b.add(obs[sl >> 2], C(w, "fw%d_%d" % (tt, sl)), name="fwi%d_%d" % (tt, sl)) if w else obs[sl >> 2]
            if attn32:
                b.store_at(c, oi, b.f16_to_f32(yh, name="fyw%d_%d" % (tt, sl)))
            else:
                b.store_at(c, oi, yh, width="half")
    b.ret()
    ir.verify(fn)
    return cc.compile_function(fn)


def mma_prefill_reference(lay, q16, K, V, expf=None):
    """The stated MMA order for every (kv head, query block): q16 [M][H][D], K and V [KVH][CAP][D] (fp16 values, the
    cache after the append). Returns attn fp16 [M][H][D]. `expf` is the 2^x used for al and P (default exp2_soft, the
    bit-exact kernel's; pass true exp2 for the hw_exp2 enclosure golden, or a wrong exp for the control)."""
    import g17decodestep as DS
    if expf is None:
        expf = DS.exp2_soft
    KVH, H, nq, NB, p0 = lay["kv_heads"], lay["heads"], lay["nq"], lay["NB"], lay["p0"]
    Mrows = lay["M"]
    q16 = np.asarray(q16, F32)
    qr = q16.reshape(nq, 16, H, D)
    Qt = np.concatenate([qr[:, :, 0::2].transpose(2, 0, 1, 3), qr[:, :, 1::2].transpose(2, 0, 1, 3)], axis=2)   # [KVH][nq][32][D]
    BK = lay.get("bk", 16)                     # keys per block (MM 25.162): the online softmax's grouping
    sreg = bool(lay.get("sreg"))               # the register-S route's order (MM 25.163, build_mma_sreg)
    NBL = NB * 16 // BK
    Kc = np.asarray(K, F32)[:, :NB * 16].reshape(KVH, NBL, BK, D)
    Vc = np.asarray(V, F32)[:, :NB * 16].reshape(KVH, NBL, BK, 8, 16).transpose(0, 1, 3, 2, 4)                # [KVH][NBL][8][BK][16]
    q0 = p0 + 16 * np.arange(nq)[:, None] + (np.arange(32) & 15)[None, :]
    Ot = np.zeros((KVH, nq, 8, 32, 16), F32)
    m = np.full((KVH, nq, 32), A.NEG_MAX, F32)
    l = np.zeros((KVH, nq, 32), F32)
    for j in range(NBL):
        S = gemm_mma_v(Qt, np.swapaxes(Kc[:, j], -1, -2)[:, None])
        if sreg:                                   # the register-S route: S accumulates onto +0 (a -0 score is +0)
            S = (F32(0.0) + S).astype(F32)
        keys = BK * j + np.arange(BK)
        valid = (keys[None, None, :] <= q0[:, :, None])[None]
        sv = np.where(valid, S, A.NEG_MAX).astype(F32)
        mb = sv[..., 0].copy()
        for k in range(1, BK):
            mb = np.maximum(mb, sv[..., k])
        mn = np.maximum(m, mb)
        nmn = (mn * F32(-1.0)).astype(F32)
        al = expf((m + nmn).astype(F32)).astype(F32)
        Pm = expf((S + nmn[..., None]).astype(F32)).astype(F32)
        Pm = np.where(valid, Pm, F32(0.0)).astype(F32)
        if sreg:                                   # four in-lane columns in order, then the xor-1 and xor-8 partners
            quad = lambda c0: (((Pm[..., c0] + Pm[..., c0 + 1]).astype(F32) + Pm[..., c0 + 2]).astype(F32)
                               + Pm[..., c0 + 3]).astype(F32)
            g = [quad(4 * q) for q in range(4)]
            for tn in range(1, BK // 16):          # a lane's later S tiles, each its own quad, in tile order
                g = [(g[q] + quad(16 * tn + 4 * q)).astype(F32) for q in range(4)]
            ssum = ((g[0] + g[1]).astype(F32) + (g[2] + g[3]).astype(F32)).astype(F32)
        else:
            ssum = Pm[..., 0].copy()
            for k in range(1, BK):
                ssum = (ssum + Pm[..., k]).astype(F32)
        l = ((l * al).astype(F32) + ssum).astype(F32)
        m = mn
        Ot = (Ot * al[:, :, None, :, None]).astype(F32)
        for sl in range(8):
            Ot[:, :, sl] = gemm_mma_v(Pm, Vc[:, j, sl][:, None], Ot[:, :, sl], truncate_a=True)
    rl = DS.recip(l).astype(F32)
    Of = np.concatenate([Ot[:, :, sl] for sl in range(8)], axis=-1)
    y = (Of * rl[..., None]).astype(F32).astype(np.float16)
    out = np.zeros((Mrows, H, D), np.float16)
    for kvh in range(KVH):
        for hh in range(2):
            out[:, 2 * kvh + hh] = y[kvh, :, 16 * hh:16 * hh + 16].reshape(Mrows, D)
    return out
