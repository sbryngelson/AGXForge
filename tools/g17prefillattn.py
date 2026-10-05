#!/usr/bin/env python3
"""Causal prefill attention (MM 25.144.2): M prompt rows at positions p0 .. p0 + M - 1, written into the decode cache and
attended over keys 0 .. p0 + i, in two dispatches, so a graph can prefill a prompt and then DECODE on the same cache.

The contract is decode's (tools/g17attn.py, the delivered wide + butterfly-merge + attn32 form at the same cap):
  region 3 (the decode layout's written region, bound at slot 3 by the append and slot 0 by the attention)
            K fp16 [KVH][CAP][D] at KOFF, V at VOFF (decode's), then q16 fp16 [M][H][D] at Q16 and the prefill
            output attn fp32 (of the fp16-rounded value) [M][H][D] at PATTN (both past decode's region3_bytes)
  region 2  decode's rope region: word 0 = p0 (decode's q0 word), cos [CAP][D/2] at COST, sin at SINT
  region 1  the qkv projection's fp32 rows [M][(H + 2 KVH) D] at 0

  append    M KVH threadgroups of 32: threadgroup i KVH + kh rotates and writes row p = min(p0 + i, CAP - 1) of KV head kh
            (k16 = fp16(rope(k)), v16 = fp16(v)) and q16 of q heads 2 kh, 2 kh + 1 (fp16(rope(q) * log2(e)/sqrt(D))),
            the arithmetic of the decode kernel's rope and append exactly
  attention M H threadgroups of 1,024 (the cooperative class): threadgroup i H + h runs the decode wide kernel's key loop
            and butterfly merge with q0 = min(p0 + i, CAP - 1), reading q16 and the cache; its mask of keys past q0 IS the
            causal mask, and the value it writes is bit for bit the one decode writes for that token at that position.
prefill_reference replays both in that order (vectorised; `attn_reference_one` cross-checks it against g17attn.attn_reference).

    python3 tools/g17prefillattn.py check --m 16 --p0 0 [--cap 272]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import g17attn as A  # noqa: E402
import g17decodeops as O  # noqa: E402

F32 = np.float32
H_, KVH_, D_ = 16, 8, 128
ROW_MASKS, COL_MASKS = (1, 8), (2, 4, 16)


def decode_layout(cap, heads=16, kv_heads=8):
    """The delivered decode attention layout (g17deliver.attn_layout) this prefill shares its cache with. heads / kv_heads:
    16 / 8 for InternLM2 and Qwen3-0.6B, 32 / 8 for Qwen3-8B (GQA 4)."""
    return A.with_attn32(A.with_bfly_merge(A.with_wide(A.with_fused_merge(A.with_rope_tables(
        A.attn_rope_layout(cap=cap, heads=heads, kv_heads=kv_heads))))))


def prefill_layout(cap, mmax, out16=False, heads=16, kv_heads=8):
    """out16: the attention writes fp16 [M][H][D] at PATTN (M1's wo GEMM reads fp16 A) instead of attn32."""
    lay = decode_layout(cap, heads, kv_heads)
    H, D = lay["heads"], lay["head_dim"]
    Q16 = A._align(lay["region3_bytes"])
    PATTN = Q16 + A._align(mmax * H * D * 2)
    return dict(lay, prefill=True, mmax=mmax, out16=bool(out16), QKVROW=(H + 2 * lay["kv_heads"]) * D, Q16=Q16, PATTN=PATTN,
                prefill_region3_bytes=PATTN + A._align(mmax * H * D * 4), qkv_bytes=A._align(mmax * (H + 2 * lay["kv_heads"]) * D * 4))


# --------------------------------------------------------------------------------------------------------- kernels
def build_prefill_append(lay):
    from agxforge.g17 import cc, ir
    import g17decodestep as DS
    H, KVH, D, CAP = lay["heads"], lay["kv_heads"], lay["head_dim"], lay["cap"]
    GQ = H // KVH                       # q heads per KV head: 2 (InternLM2, Qwen3-0.6B) or 4 (Qwen3-8B)
    if H % KVH or GQ & (GQ - 1) or KVH & (KVH - 1):
        raise ValueError("prefill append: H / KVH and KVH powers of two")
    if lay.get("mma") and not lay.get("rego") and GQ != 2:
        raise ValueError("prefill append: the MMA route's Q tile holds two q heads per KV head")
    fn, b, a, bb, c = O._function()
    I = ir.I32
    lane = b.builtin("thread_index_in_simdgroup", name="lane")
    d0 = b.shl(lane, O._c(b, 2, "two"), name="d0")
    g = b.builtin("threadgroup_position_in_grid", name="g")
    i = b.shr(g, O._c(b, KVH.bit_length() - 1, "three"), name="row")
    kh = getattr(b, "and")(g, ir.Imm(KVH - 1), name="kh")
    p0 = b.load(bb, O._c(b, 0, "p0w"), type=I, name="p0")
    capm1 = O._c(b, CAP - 1, "capm1")
    p = b.add(p0, i, name="praw")
    p = b.csel(p, capm1, capm1, p, rel="gt", name="p")
    dm = b.shl(b.sub(lane, b.shl(b.shr(lane, O._c(b, 4, "f4"), name="lhi"), O._c(b, 4, "f4b"), name="lhi16"), name="llo"),
               O._c(b, 2, "two_"), name="dm0")
    dm = b.add(dm, b.mul(p, O._c(b, D // 2, "halfD"), name="prow"), name="dmrow")
    cs = [b.load(bb, b.add(dm, O._c(b, lay["COST"] // 4 + k, "co%d" % k), name="ci%d" % k), type=I, name="cos%d" % k) for k in range(4)]
    sn = [b.load(bb, b.add(dm, O._c(b, lay["SINT"] // 4 + k, "so%d" % k), name="si%d" % k), type=I, name="sin%d" % k) for k in range(4)]
    # rotate-half sign as a value (the decode wide form's): -1 on lanes 0..15, +1 on 16..31
    sgn = b.fadd(b.fmul(b.u32_to_f32(b.shr(lane, O._c(b, 4, "sg4"), name="lh4"), name="lh4f"),
                        O._cf(b, F32(2.0), "ftwo"), type=I, name="lh8"), O._cf(b, F32(-1.0), "fneg_s"), type=ir.F32, name="rsgn")

    def rope(vals, tag):
        out = []
        for k in range(4):
            own = vals[k]
            own.type = ir.F32
            part = b.simd_shuffle_xor(own, 16, name="%s_pt%d" % (tag, k))
            ac = b.fmul(own, cs[k], type=I, name="%s_ac%d" % (tag, k))
            bs = b.fmul(part, sn[k], type=I, name="%s_bs%d" % (tag, k))
            out.append(b.fadd(ac, b.fmul(bs, sgn, type=I, name="%s_sb%d" % (tag, k)), type=I, name="%s_r%d" % (tag, k)))
        return out

    rowb = b.mul(i, O._c(b, lay["QKVROW"], "qkvrow"), name="rowb")
    kin_b = b.add(b.add(rowb, b.mul(kh, O._c(b, D, "Dk"), name="khD"), name="rk"), b.add(d0, O._c(b, H * D, "kbase"), name="kd0"), name="kin_b")
    kin = [b.load(a, b.add(kin_b, O._c(b, k, "ko%d" % k), name="kii%d" % k) if k else kin_b, type=I, name="kin%d" % k) for k in range(4)]
    vin = [b.load(a, b.add(kin_b, O._c(b, KVH * D + k, "vo%d" % k), name="vii%d" % k), type=I, name="vin%d" % k) for k in range(4)]
    kr = rope(kin, "k")
    kw = b.add(b.add(b.mul(b.add(b.mul(kh, O._c(b, CAP, "CAPk"), name="khC"), p, name="khCp"), O._c(b, D, "Dkw"), name="krow"), d0, name="kw0"),
               O._c(b, lay["KOFF"] // 2, "koff"), name="kw")
    vw = b.add(kw, O._c(b, (lay["VOFF"] - lay["KOFF"]) // 2, "voff"), name="vw")
    # only a row INSIDE the cache writes K/V: rows past CAP - 1 would all clamp to the last row and race (different
    # threadgroups, different values). They still write their own q16 and attend with q0 clamped, as decode does.
    fn.skip_regions = True
    wkv, wdone = fn.block("write_kv"), fn.block("write_kv_done")
    # "inside" = the clamp left the position unchanged (the compare immediate is 8 bits, so not praw < CAP)
    inside = b.icmp(b.add(p0, i, name="praw2"), p, "eq", name="inside")
    b.br_cond(b.cmp(inside, 0, "gt", name="incache"), wkv, wdone)
    b.at(wkv)
    for k in range(4):
        b.store_at(c, b.add(kw, O._c(b, k, "kwo%d" % k), name="kwi%d" % k) if k else kw, b.f32_to_f16_rte(kr[k], name="k16_%d" % k), width="half")
        b.store_at(c, b.add(vw, O._c(b, k, "vwo%d" % k), name="vwi%d" % k) if k else vw, b.f32_to_f16_rte(vin[k], name="v16_%d" % k), width="half")
    if lay.get("mma") and not lay.get("rego"):
        # THE TENSOR ROUTE'S V (g17prefillmma): also block-major, [KVH][CAP/16][8 slices][16 keys][16 dims], so every
        # PV body reads one contiguous 16 x 16 tile (the memory stream admits no B stride)
        vbw = b.add(b.shl(b.add(b.shl(b.add(b.mul(kh, O._c(b, CAP // 16, "nbc"), name="khnb"), b.shr(p, O._c(b, 4, "p4"), name="pblk"),
                                              name="kvblk"), O._c(b, 3, "x8"), name="kvb8"),
                                b.shr(lane, O._c(b, 2, "l2"), name="vsl"), name="kvbs"), O._c(b, 8, "x256"), name="vbase"),
                    b.add(b.shl(getattr(b, "and")(p, ir.Imm(15), name="pin"), O._c(b, 4, "x16"), name="pin16"),
                          b.shl(getattr(b, "and")(lane, ir.Imm(3), name="lq"), O._c(b, 2, "x4"), name="lq4"), name="voff_in"),
                    name="vbw0")
        vbw = b.add(vbw, O._c(b, lay["VB"] // 2, "vbo"), name="vbw")
        for k in range(4):
            b.store_at(c, b.add(vbw, O._c(b, k, "vbk%d" % k), name="vbi%d" % k) if k else vbw,
                       b.f32_to_f16_rte(vin[k], name="vb16_%d" % k), width="half")
    b.br(wdone)
    b.at(wdone)
    qsc = O._cf(b, F32(DS.q_scale(DS.MILESTONE)), "qscale")
    for hh in range(GQ):
        h = b.add(b.shl(kh, O._c(b, GQ.bit_length() - 1, "one%d" % hh), name="kh2_%d" % hh), O._c(b, hh, "hh%d" % hh), name="h%d" % hh)
        qb = b.add(b.add(rowb, b.mul(h, O._c(b, D, "Dq%d" % hh), name="hD%d" % hh), name="rq%d" % hh), d0, name="qb%d" % hh)
        qin = [b.load(a, b.add(qb, O._c(b, k, "qo%d_%d" % (hh, k)), name="qii%d_%d" % (hh, k)) if k else qb, type=I, name="qin%d_%d" % (hh, k))
               for k in range(4)]
        qr = rope(qin, "q%d" % hh)
        qw = b.add(b.add(b.mul(b.add(b.mul(i, O._c(b, H, "Hq%d" % hh), name="iH%d" % hh), h, name="iHh%d" % hh), O._c(b, D, "Dqw%d" % hh),
                                name="qrow%d" % hh), d0, name="qw0_%d" % hh), O._c(b, lay["Q16"] // 2, "q16o%d" % hh), name="qw%d" % hh)
        q16s = [b.f32_to_f16_rte(b.fmul(qr[k], qsc, type=I, name="qs%d_%d" % (hh, k)), name="q16_%d_%d" % (hh, k)) for k in range(4)]
        for k in range(4):
            b.store_at(c, b.add(qw, O._c(b, k, "qwo%d_%d" % (hh, k)), name="qwi%d_%d" % (hh, k)) if k else qw, q16s[k], width="half")
        if lay.get("rego"):
            # THE REGISTER-O ROUTE'S Q TILE: [H][M/16][16 rows][128], row i & 15 of tile (h, i >> 4)
            qt = b.add(b.shl(b.add(b.shl(b.add(b.mul(h, O._c(b, lay["nq"], "rnq%d" % hh), name="hnq%d" % hh),
                                                 b.shr(i, O._c(b, 4, "ri4_%d" % hh), name="riblk%d" % hh), name="rqtile%d" % hh),
                                           O._c(b, 4, "x16_%d" % hh), name="rqt16_%d" % hh),
                                     getattr(b, "and")(i, ir.Imm(15), name="riin%d" % hh), name="rqtr%d" % hh),
                               O._c(b, 7, "rx128_%d" % hh), name="rqtw0_%d" % hh), d0, name="rqtw1_%d" % hh)
            qt = b.add(qt, O._c(b, lay["QT"] // 2, "rqto%d" % hh), name="rqtw%d" % hh)
            for k in range(4):
                b.store_at(c, b.add(qt, O._c(b, k, "rqtk%d_%d" % (hh, k)), name="rqti%d_%d" % (hh, k)) if k else qt, q16s[k], width="half")
        elif lay.get("mma"):
            # THE TENSOR ROUTE'S Q TILE: [KVH][M/16][32 rows][128], row hh*16 + (i & 15) of tile (kh, i >> 4)
            qt = b.add(b.shl(b.add(b.shl(b.add(b.mul(kh, O._c(b, lay["nq"], "nq%d" % hh), name="khnq%d" % hh),
                                                 b.shr(i, O._c(b, 4, "i4_%d" % hh), name="iblk%d" % hh), name="qtile%d" % hh),
                                           O._c(b, 5, "x32_%d" % hh), name="qt32_%d" % hh),
                                     b.add(getattr(b, "and")(i, ir.Imm(15), name="iin%d" % hh), O._c(b, 16 * hh, "hh16_%d" % hh),
                                           name="qtrow%d" % hh), name="qtr%d" % hh),
                               O._c(b, 7, "x128_%d" % hh), name="qtw0_%d" % hh), d0, name="qtw1_%d" % hh)
            qt = b.add(qt, O._c(b, lay["QT"] // 2, "qto%d" % hh), name="qtw%d" % hh)
            for k in range(4):
                b.store_at(c, b.add(qt, O._c(b, k, "qtk%d_%d" % (hh, k)), name="qti%d_%d" % (hh, k)) if k else qt, q16s[k], width="half")
    b.ret()
    ir.verify(fn)
    return cc.compile_function(fn)


def build_prefill_attn(lay):
    """The decode wide + butterfly-merge + attn32 key loop, one threadgroup per (row i, head h), q0 = min(p0 + i, CAP - 1),
    q16 read from Q16, no append, the output at PATTN [(i H + h) D + d]."""
    from agxforge.g17 import cc, ir, tensorreduce as TR
    H, KVH, D, CAP = lay["heads"], lay["kv_heads"], lay["head_dim"], lay["cap"]
    S, PW = 32, lay["PW"]
    TC = -(-CAP // S)
    if TC > 255:
        raise ValueError("prefill attention: the latch compares an 8-bit trip cap")
    c = ir.Buffer("C", 0, elem=ir.F32)
    a = ir.Buffer("A", 1, elem=ir.F16)
    bb = ir.Buffer("B", 2, elem=ir.F16)
    fn = ir.Function("tensor_gemm_generic_runtime_demo", [c, a, bb])
    fn.declare_threadgroup(S * PW, size=(32 * S, 1, 1))
    b = ir.Builder(fn, fn.block("entry"))
    I = ir.I32
    tpos = b.builtin("thread_position_in_threadgroup", name="tpos")
    lane = getattr(b, "and")(tpos, ir.Imm(31), name="lane")
    d0 = b.shl(lane, O._c(b, 2, "two"), name="d0")
    g = b.builtin("threadgroup_position_in_grid", name="g")
    s = b.shr(tpos, O._c(b, 5, "five_s"), name="slice")
    i = b.shr(g, O._c(b, H.bit_length() - 1, "four_h"), name="row")
    h = getattr(b, "and")(g, ir.Imm(H - 1), name="head")
    p0 = b.load(bb, O._c(b, 0, "p0w"), type=I, name="p0")
    capm1 = O._c(b, CAP - 1, "capm1")
    q0 = b.add(p0, i, name="q0raw")
    q0 = b.csel(q0, capm1, capm1, q0, rel="gt", name="q0")
    one = O._c(b, 1, "one")
    zero = O._c(b, 0, "zero")
    trips = b.shr(b.sub(b.add(q0, O._c(b, S, "S"), name="q0S"), s, name="q0Ss"), O._c(b, 5, "lgS2"), name="trips")
    trips = b.csel(trips, zero, trips, one, rel="gt", name="trips1")
    kvh = b.shr(h, O._c(b, (lay["heads"] // lay["kv_heads"]).bit_length() - 1, "gqa"), name="kvh")
    qb = b.add(b.add(b.mul(g, O._c(b, D, "Dq"), name="gD"), d0, name="gDd"), O._c(b, lay["Q16"] // 2, "q16o"), name="qb")
    qv = [b.f16_to_f32(b.load(c, b.add(qb, O._c(b, k, "qo%d" % k), name="qi%d" % k) if k else qb, width="half", name="qh%d" % k),
                       name="q%d" % k) for k in range(4)]
    kb = b.add(b.add(b.mul(kvh, O._c(b, CAP * D, "CAPD"), name="kvoff"), d0, name="kb0"), O._c(b, lay["KOFF"] // 2, "koff"), name="kb")
    vb = b.add(kb, O._c(b, (lay["VOFF"] - lay["KOFF"]) // 2, "voff"), name="vb")
    negmax = O._cf(b, A.NEG_MAX, "negmax")
    fzero = O._cf(b, F32(0.0), "fzero")
    neg1 = O._cf(b, F32(-1.0), "neg1")
    K2 = O.emit_exp2_constants(b)
    lzero = O._cf(b, F32(0.0), "lzero")
    ozeros = [O._cf(b, F32(0.0), "ozero%d" % k) for k in range(4)]
    hdr, post = fn.block("keys"), fn.block("out")
    b.br(hdr)
    b.at(hdr)
    t = b.phi(zero, name="t")
    m = b.phi(negmax, type=ir.F32, name="m")
    l = b.phi(lzero, type=ir.F32, name="l")
    o = [b.phi(ozeros[k], type=ir.F32, name="o%d" % k) for k in range(4)]
    j = b.add(s, b.shl(t, O._c(b, 5, "lgS3"), name="tS"), name="j")
    jr = b.csel(j, q0, q0, j, rel="gt", name="jr")
    row = b.mul(jr, O._c(b, D, "D2"), name="row")
    ki = b.add(kb, row, name="ki")
    vi = b.add(vb, row, name="vi")
    if lay.get("kvvec"):
        # DECODE'S kvvec (g17attn.with_kvvec, MM 25.144.5; here MM 25.205): one 8-byte load per K / V row per lane, the
        # lane's four fp16 dims as two words' halves - the same values as the four half loads
        def halves(idx, tag):
            w = b.load_vec_at(c, b.shr(idx, O._c(b, 2, tag + "v2"), name=tag + "vi"), n=2, name=tag + "w")
            out = []
            for wi, word in enumerate(w):
                out.append(b.f16_to_f32(b.low16(word, name="%sl%d" % (tag, wi)), name="%sf%d" % (tag, 2 * wi)))
                out.append(b.f16_to_f32(b.low16(b.shr(word, O._c(b, 16, tag + "s16_%d" % wi), name="%sh%d" % (tag, wi)),
                                                name="%shl%d" % (tag, wi)), name="%sf%d" % (tag, 2 * wi + 1)))
            return out
        kl, vl = halves(ki, "kvk"), halves(vi, "kvv")
    else:
        kl = [b.f16_to_f32(b.load(c, b.add(ki, O._c(b, k, "ko%d" % k), name="kii%d" % k) if k else ki, width="half", name="kh%d" % k),
                           name="kl%d" % k) for k in range(4)]
        vl = [b.f16_to_f32(b.load(c, b.add(vi, O._c(b, k, "vo%d" % k), name="vii%d" % k) if k else vi, width="half", name="vh%d" % k),
                           name="vl%d" % k) for k in range(4)]
    part = b.fmul(qv[0], kl[0], type=ir.F32, name="pd0")
    for k in range(1, 4):
        part = b.fma(qv[k], kl[k], part, name="pd%d" % k)
    part.type = ir.F32
    sc = TR.emit_butterfly(b, part, TR.ROW_BUTTERFLY_MASKS, operation="sum")
    sc = TR.emit_butterfly(b, sc, TR.COLUMN_BUTTERFLY_MASKS, operation="sum")
    sc = b.csel(j, q0, m, sc, rel="gt", name="sc")
    mn = b.fmax(m, sc, type=I, name="mn")
    nmn = b.fmul(mn, neg1, type=I, name="nmn")
    # lay["hw_exp2"] (MM 25.205): decode's hardware exp2 (op1272, g17attn._exp2), as the decode graph's attention runs it
    al = A._exp2(lay, b, b.fadd(m, nmn, type=I, name="dm_"), K2, "al")
    pe = A._exp2(lay, b, b.fadd(sc, nmn, type=I, name="ds"), K2, "pe")
    pe = b.csel(j, q0, fzero, pe, rel="gt", name="p")
    ln = b.fadd(b.fmul(l, al, type=I, name="la"), pe, type=ir.F32, name="ln")
    on = [b.fma(pe, vl[k], b.fmul(o[k], al, type=I, name="oa%d" % k), name="on%d" % k) for k in range(4)]
    tn = b.add(t, one, name="tn")
    ir.Builder.phi_latch(t, tn)
    ir.Builder.phi_latch(m, mn)
    ir.Builder.phi_latch(l, ln)
    for k in range(4):
        ir.Builder.phi_latch(o[k], on[k])
    b.br_cond(b.cmp(tn, trips, "lt", cap=TC), hdr, post)
    b.at(post)
    pw0 = b.mul(s, O._c(b, PW, "PWs"), name="pw0")
    l0b, l0j = fn.block("pub_ml"), fn.block("pub_ml_done")
    b.br_cond(b.cmp(lane, 1, "lt", name="lane0p"), l0b, l0j)
    b.at(l0b)
    b.store_tg(b.fadd(mn, fzero, type=ir.F32, name="mpub"), pw0)
    b.store_tg(b.fadd(ln, fzero, type=ir.F32, name="lpub"), b.add(pw0, one, name="pw1"))
    b.br(l0j)
    b.at(l0j)
    ob_ = b.add(b.add(pw0, O._c(b, 4, "four_w"), name="pw4"), d0, name="opw")
    for k in range(4):
        b.store_tg(b.fadd(on[k], fzero, type=ir.F32, name="opub%d" % k), b.add(ob_, O._c(b, k, "opo%d" % k), name="opi%d" % k) if k else ob_)
    b.barrier("threadgroup")
    # the butterfly merge with "head" = g = i H + h, so its output index (i H + h) D + 4 sg + k lands at PATTN
    A._emit_bfly_merge(b, ir, TR, c, g, s, lane, q0, dict(lay, slices=S, OUT_AT=lay["PATTN"], attn32=not lay.get("out16")),
                       K2, neg1, fzero, zero, fn)
    b.ret()
    ir.verify(fn)
    return cc.compile_function(fn)


def _bfly_merge_row(b, ir, TR, dst, hidx, sg, lane, q0, lay, K2, neg1, fzero, fn, tag, out_at, attn32):
    """g17attn._emit_bfly_merge's arithmetic with every name and block label prefixed by `tag` (so it can be emitted
    once per row of a query block): lane l of simdgroup sg reads slice l's m, l and dims 4 sg .. 4 sg + 3 from the
    scratchpad [l PW]; M, L and the dims are lane butterflies; weights exp2_soft(m - M), 0 for a slice past q0."""
    D, PW = lay["head_dim"], lay["PW"]
    I = ir.I32

    def C(v, n):
        return O._c(b, v, tag + n)
    base = b.mul(lane, C(PW, "bPW"), name=tag + "bbase")

    def ld(idx, name):
        return b.fadd(b.load_tg(idx, name=tag + name + "_t"), fzero, type=I, name=tag + name)
    m_ = ld(base, "bm")
    l_ = ld(b.add(base, C(1, "b1"), name=tag + "bli"), "bl")
    ob = b.add(b.add(base, C(4, "b4"), name=tag + "bo4"), b.shl(sg, C(2, "b2s"), name=tag + "sg4"), name=tag + "bob")
    os_ = [ld(b.add(ob, C(i, "boi%d" % i), name=tag + "bo%d_i" % i) if i else ob, "bo%d" % i) for i in range(4)]
    m_.type = ir.F32
    M = TR.emit_butterfly(b, m_, TR.ROW_BUTTERFLY_MASKS, operation="max")
    M = TR.emit_butterfly(b, M, TR.COLUMN_BUTTERFLY_MASKS, operation="max")
    nM = b.fmul(M, neg1, type=I, name=tag + "bnM")
    w = O.emit_exp2_soft(b, b.fadd(m_, nM, type=I, name=tag + "bdM"), K2, tag + "bw")
    w = b.csel(lane, q0, fzero, w, rel="gt", name=tag + "bwz")
    lw = b.fmul(l_, w, type=ir.F32, name=tag + "blw")
    L = TR.emit_butterfly(b, lw, TR.ROW_BUTTERFLY_MASKS, operation="sum")
    L = TR.emit_butterfly(b, L, TR.COLUMN_BUTTERFLY_MASKS, operation="sum")
    KR = O.emit_constants(b)
    rL = O.emit_rn(b, "recip", L, KR, tag + "brL")
    outb = b.add(b.mul(hidx, C(D, "bD"), name=tag + "bhD"), b.shl(sg, C(2, "b2o"), name=tag + "bsg4"), name=tag + "boutb")
    ys = []
    for i in range(4):
        t = b.fmul(os_[i], w, type=ir.F32, name=tag + "bow%d" % i)
        t = TR.emit_butterfly(b, t, TR.ROW_BUTTERFLY_MASKS, operation="sum")
        t = TR.emit_butterfly(b, t, TR.COLUMN_BUTTERFLY_MASKS, operation="sum")
        ys.append(b.fmul(t, rL, type=I, name=tag + "by%d" % i))
    fn.skip_regions = True
    wb, wj = fn.block(tag + "bfly_write"), fn.block(tag + "bfly_done")
    b.br_cond(b.cmp(lane, 1, "lt", name=tag + "blane0"), wb, wj)
    b.at(wb)
    for i in range(4):
        idx = b.add(outb, C((out_at // 4 if attn32 else out_at // 2) + i, "bout%d" % i), name=tag + "bouti%d" % i)
        if attn32:
            b.store_at(dst, idx, b.f16_to_f32(b.f32_to_f16_rte(ys[i], name=tag + "byh%d" % i), name=tag + "byw%d" % i))
        else:
            b.store_at(dst, idx, b.f32_to_f16_rte(ys[i], name=tag + "byh%d" % i), width="half")
    b.br(wj)
    b.at(wj)


def build_prefill_attn_qb(lay, qb):
    """THE QUERY-BLOCK FORM: threadgroup g = block x 16 + head runs qb consecutive rows at once; every key's K and V are
    loaded ONCE for all qb rows. Per row the arithmetic is build_prefill_attn's (so decode's): slice s takes keys s,
    s + 32, ...; a key past the row's own q0 gives sc = m and p = 0, an exact identity step (alpha = exp2_soft(0) = 1,
    fma(0, v, o 1) = o for a written row v). The block runs the trips of its LAST row and reads key min(j, q0 of the
    last row), always a written row. Then the rows are merged one at a time through the one scratchpad (publish,
    barrier, butterfly merge, barrier). Launch: (M / qb) x 16 threadgroups of 1,024, M a multiple of qb."""
    from agxforge.g17 import cc, ir, tensorreduce as TR
    H, D, CAP = lay["heads"], lay["head_dim"], lay["cap"]
    S, PW = 32, lay["PW"]
    TC = -(-CAP // S)
    if qb not in (1, 2, 4, 8, 16):
        raise ValueError("query block: a power of two, 1..16 rows")
    c = ir.Buffer("C", 0, elem=ir.F32)
    a = ir.Buffer("A", 1, elem=ir.F16)
    bb = ir.Buffer("B", 2, elem=ir.F16)
    fn = ir.Function("tensor_gemm_generic_runtime_demo", [c, a, bb])
    fn.declare_threadgroup(S * PW, size=(32 * S, 1, 1))
    b = ir.Builder(fn, fn.block("entry"))
    I = ir.I32
    tpos = b.builtin("thread_position_in_threadgroup", name="tpos")
    lane = getattr(b, "and")(tpos, ir.Imm(31), name="lane")
    d0 = b.shl(lane, O._c(b, 2, "two"), name="d0")
    g = b.builtin("threadgroup_position_in_grid", name="g")
    s = b.shr(tpos, O._c(b, 5, "five_s"), name="slice")
    blk = b.shr(g, O._c(b, H.bit_length() - 1, "four_h"), name="blk")
    h = getattr(b, "and")(g, ir.Imm(H - 1), name="head")
    row0 = b.shl(blk, O._c(b, qb.bit_length() - 1, "lgqb"), name="row0") if qb > 1 else blk
    p0 = b.load(bb, O._c(b, 0, "p0w"), type=I, name="p0")
    capm1 = O._c(b, CAP - 1, "capm1")
    base0 = b.add(p0, row0, name="q0base")
    q0s = []
    for r in range(qb):
        q = b.add(base0, O._c(b, r, "ro%d" % r), name="q0raw%d" % r) if r else base0
        q0s.append(b.csel(q, capm1, capm1, q, rel="gt", name="q0_%d" % r))
    qlast = q0s[-1]
    one = O._c(b, 1, "one")
    zero = O._c(b, 0, "zero")
    trips = b.shr(b.sub(b.add(qlast, O._c(b, S, "S"), name="q0S"), s, name="q0Ss"), O._c(b, 5, "lgS2"), name="trips")
    trips = b.csel(trips, zero, trips, one, rel="gt", name="trips1")
    kvh = b.shr(h, O._c(b, (lay["heads"] // lay["kv_heads"]).bit_length() - 1, "gqa"), name="kvh")
    rowsH = b.mul(row0, O._c(b, H, "Hrow0"), name="row0H")
    qv = []
    for r in range(qb):
        gr = b.add(b.add(rowsH, O._c(b, r * H, "rHo%d" % r), name="rH%d" % r) if r else rowsH, h, name="rh%d" % r)
        qbase = b.add(b.add(b.mul(gr, O._c(b, D, "Dq%d" % r), name="gD%d" % r), d0, name="gDd%d" % r),
                      O._c(b, lay["Q16"] // 2, "q16o%d" % r), name="qb%d" % r)
        qv.append([b.f16_to_f32(b.load(c, b.add(qbase, O._c(b, k, "qo%d_%d" % (r, k)), name="qi%d_%d" % (r, k)) if k else qbase,
                                       width="half", name="qh%d_%d" % (r, k)), name="q%d_%d" % (r, k)) for k in range(4)])
    kb = b.add(b.add(b.mul(kvh, O._c(b, CAP * D, "CAPD"), name="kvoff"), d0, name="kb0"), O._c(b, lay["KOFF"] // 2, "koff"), name="kb")
    vb = b.add(kb, O._c(b, (lay["VOFF"] - lay["KOFF"]) // 2, "voff"), name="vb")
    negmax = O._cf(b, A.NEG_MAX, "negmax")
    fzero = O._cf(b, F32(0.0), "fzero")
    neg1 = O._cf(b, F32(-1.0), "neg1")
    K2 = O.emit_exp2_constants(b)
    # every loop phi takes its own entry constant (a shared one would coalesce the phis into one register)
    ent = [(O._cf(b, A.NEG_MAX, "m0c%d" % r), O._cf(b, F32(0.0), "l0c%d" % r),
            [O._cf(b, F32(0.0), "o0c%d_%d" % (r, k)) for k in range(4)]) for r in range(qb)]
    t0 = O._c(b, 0, "t0c")
    hdr, post = fn.block("keys"), fn.block("out")
    b.br(hdr)
    b.at(hdr)
    t = b.phi(t0, name="t")
    st = []
    for r in range(qb):
        m = b.phi(ent[r][0], type=ir.F32, name="m%d" % r)
        l = b.phi(ent[r][1], type=ir.F32, name="l%d" % r)
        o = [b.phi(ent[r][2][k], type=ir.F32, name="o%d_%d" % (r, k)) for k in range(4)]
        st.append((m, l, o))
    j = b.add(s, b.shl(t, O._c(b, 5, "lgS3"), name="tS"), name="j")
    jr = b.csel(j, qlast, qlast, j, rel="gt", name="jr")
    rowk = b.mul(jr, O._c(b, D, "D2"), name="rowk")
    ki = b.add(kb, rowk, name="ki")
    vi = b.add(vb, rowk, name="vi")
    kl = [b.f16_to_f32(b.load(c, b.add(ki, O._c(b, k, "ko%d" % k), name="kii%d" % k) if k else ki, width="half", name="kh%d" % k),
                       name="kl%d" % k) for k in range(4)]
    vl = [b.f16_to_f32(b.load(c, b.add(vi, O._c(b, k, "vo%d" % k), name="vii%d" % k) if k else vi, width="half", name="vh%d" % k),
                       name="vl%d" % k) for k in range(4)]
    tn = b.add(t, one, name="tn")
    ir.Builder.phi_latch(t, tn)
    fin = []
    for r in range(qb):
        m, l, o = st[r]
        q0 = q0s[r]
        part = b.fmul(qv[r][0], kl[0], type=ir.F32, name="pd%d_0" % r)
        for k in range(1, 4):
            part = b.fma(qv[r][k], kl[k], part, name="pd%d_%d" % (r, k))
        part.type = ir.F32
        sc = TR.emit_butterfly(b, part, TR.ROW_BUTTERFLY_MASKS, operation="sum")
        sc = TR.emit_butterfly(b, sc, TR.COLUMN_BUTTERFLY_MASKS, operation="sum")
        sc = b.csel(j, q0, m, sc, rel="gt", name="sc%d" % r)
        mn = b.fmax(m, sc, type=I, name="mn%d" % r)
        nmn = b.fmul(mn, neg1, type=I, name="nmn%d" % r)
        al = O.emit_exp2_soft(b, b.fadd(m, nmn, type=I, name="dm%d" % r), K2, "al%d" % r)
        pe = O.emit_exp2_soft(b, b.fadd(sc, nmn, type=I, name="ds%d" % r), K2, "pe%d" % r)
        pe = b.csel(j, q0, fzero, pe, rel="gt", name="p%d" % r)
        ln = b.fadd(b.fmul(l, al, type=I, name="la%d" % r), pe, type=ir.F32, name="ln%d" % r)
        on = [b.fma(pe, vl[k], b.fmul(o[k], al, type=I, name="oa%d_%d" % (r, k)), name="on%d_%d" % (r, k)) for k in range(4)]
        ir.Builder.phi_latch(m, mn)
        ir.Builder.phi_latch(l, ln)
        for k in range(4):
            ir.Builder.phi_latch(o[k], on[k])
        fin.append((mn, ln, on))
    b.br_cond(b.cmp(tn, trips, "lt", cap=TC), hdr, post)
    b.at(post)
    pw0 = b.mul(s, O._c(b, PW, "PWs"), name="pw0")
    ob_ = b.add(b.add(pw0, O._c(b, 4, "four_w"), name="pw4"), d0, name="opw")
    attn32 = not lay.get("out16")
    for r in range(qb):
        mn, ln, on = fin[r]
        if r:
            b.barrier("threadgroup")                    # the previous row's merge has read the scratchpad
        l0b, l0j = fn.block("pub_ml%d" % r), fn.block("pub_ml_done%d" % r)
        b.br_cond(b.cmp(lane, 1, "lt", name="lane0p%d" % r), l0b, l0j)
        b.at(l0b)
        b.store_tg(b.fadd(mn, fzero, type=ir.F32, name="mpub%d" % r), pw0)
        b.store_tg(b.fadd(ln, fzero, type=ir.F32, name="lpub%d" % r), b.add(pw0, one, name="pw1_%d" % r))
        b.br(l0j)
        b.at(l0j)
        for k in range(4):
            b.store_tg(b.fadd(on[k], fzero, type=ir.F32, name="opub%d_%d" % (r, k)),
                       b.add(ob_, O._c(b, k, "opo%d_%d" % (r, k)), name="opi%d_%d" % (r, k)) if k else ob_)
        b.barrier("threadgroup")
        hidx = b.add(b.add(rowsH, O._c(b, r * H, "hrH%d" % r), name="hrow%d" % r) if r else rowsH, h, name="hidx%d" % r)
        _bfly_merge_row(b, ir, TR, c, hidx, s, lane, q0s[r], lay, K2, neg1, fzero, fn, "r%d_" % r, lay["PATTN"], attn32)
    b.ret()
    ir.verify(fn)
    return cc.compile_function(fn)


# ------------------------------------------------------------------------------------------------------ reference
def _bfly(vals, masks, op):
    """The measured lane butterfly (tensorreduce.butterfly), vectorised over leading axes: sums through float64 then
    rounded (tensorreduce._fadd), maxima as np.maximum (no NaN reaches it)."""
    lanes = np.arange(32)
    v = np.asarray(vals, F32)
    for mk in masks:
        other = v[..., lanes ^ mk]
        v = (v.astype(np.float64) + other.astype(np.float64)).astype(F32) if op == "sum" else np.maximum(v, other)
    return v


def rope_append_reference(lay, qkv32, cos_t, sin_t, Kc, Vc, p0, M):
    """q16 [M, H, D] and the caches with rows p0 .. p0 + M - 1 written (g17decodestep's rope and rounding)."""
    import g17decodestep as DS
    H, KVH, D, CAP = lay["heads"], lay["kv_heads"], lay["head_dim"], lay["cap"]
    rows = np.asarray(qkv32, F32).reshape(-1, lay["QKVROW"])[:M]
    pos = np.minimum(p0 + np.arange(M), CAP - 1)
    cos, sin = np.asarray(cos_t, F32)[pos][:, None, :], np.asarray(sin_t, F32)[pos][:, None, :]
    q = rows[:, :H * D].reshape(M, H, D)
    k = rows[:, H * D:(H + KVH) * D].reshape(M, KVH, D)
    v = rows[:, (H + KVH) * D:].reshape(M, KVH, D)
    q16 = DS.narrow(DS.fmul(DS.rope_rotate(q, cos, sin), DS.q_scale(DS.MILESTONE)))
    k16, v16 = DS.narrow(DS.rope_rotate(k, cos, sin)), DS.narrow(v)
    K, V = np.array(Kc, F32), np.array(Vc, F32)
    for r in range(M):                                   # only rows inside the cache write K/V (build_prefill_append)
        if p0 + r < CAP:
            K[:, pos[r]] = k16[r]
            V[:, pos[r]] = v16[r]
    return q16, K, V


def attn_reference_many(lay, q16, K, V, q0s, chunk=32):
    """attn_reference (wide, 32 slices, butterfly merge) for every row: q16 [M, H, D], q0s [M]; returns fp16 [M, H, D]."""
    import g17decodestep as DS
    from g17qmv import _fma32v
    H, KVH, D, CAP = lay["heads"], lay["kv_heads"], lay["head_dim"], lay["cap"]
    S = 32
    q16 = np.asarray(q16, F32)
    M = q16.shape[0]
    out = np.zeros((M, H, D), np.float16)
    kvh = np.arange(H) // (H // KVH)
    sl = np.arange(S)
    for c0 in range(0, M, chunk):
        q0 = np.minimum(np.asarray(q0s[c0:c0 + chunk]), CAP - 1)
        Q = len(q0)
        qd = q16[c0:c0 + Q].reshape(Q, H, 1, 32, 4)
        trips = np.maximum((q0[:, None] + S - sl[None, :]) >> 5, 1)          # [Q, S]
        m = np.full((Q, H, S), A.NEG_MAX, F32)
        l = np.zeros((Q, H, S), F32)
        o = np.zeros((Q, H, S, 32, 4), F32)
        for t in range(int(trips.max())):
            act = (t < trips)[:, None, :]                                    # [Q, 1, S]
            j = sl[None, :] + S * t                                          # [1, S]
            jr = np.minimum(j, q0[:, None])                                  # [Q, S]
            past = (j > q0[:, None])[:, None, :]                             # [Q, 1, S]
            k = K[kvh[None, :, None], jr[:, None, :]].reshape(Q, H, S, 32, 4)
            v = V[kvh[None, :, None], jr[:, None, :]].reshape(Q, H, S, 32, 4)
            part = (qd[..., 0] * k[..., 0]).astype(F32)
            for ii in range(1, 4):
                part = _fma32v(np.broadcast_to(qd[..., ii], part.shape), k[..., ii], part)
            sc = _bfly(_bfly(part, ROW_MASKS, "sum"), COL_MASKS, "sum")[..., 0]
            sc = np.where(past, m, sc)
            mn = np.maximum(m, sc)
            nmn = (mn * F32(-1.0)).astype(F32)
            al = DS.exp2_soft((m + nmn).astype(F32)).astype(F32)
            p = DS.exp2_soft((sc + nmn).astype(F32)).astype(F32)
            p = np.where(past, F32(0.0), p).astype(F32)
            ln = ((l * al).astype(F32) + p).astype(F32)
            on = _fma32v(np.broadcast_to(p[..., None, None], o.shape), v, (o * al[..., None, None]).astype(F32))
            m = np.where(act, mn, m); l = np.where(act, ln, l); o = np.where(act[..., None, None], on, o)
        # the butterfly merge: lane s holds slice s
        Mb = _bfly(_bfly(m, ROW_MASKS, "max"), COL_MASKS, "max")[..., :1]
        nM = (Mb * F32(-1.0)).astype(F32)
        w = DS.exp2_soft((m + nM).astype(F32)).astype(F32)
        w = np.where(sl[None, None, :] <= q0[:, None, None], w, F32(0.0)).astype(F32)
        lw = (l * w).astype(F32)
        L = _bfly(_bfly(lw, ROW_MASKS, "sum"), COL_MASKS, "sum")[..., 0]
        rL = DS.recip(L).astype(F32)
        ow = (o.reshape(Q, H, S, D) * w[..., None]).astype(F32)             # [Q, H, S, D]
        ob = _bfly(_bfly(np.moveaxis(ow, 2, -1), ROW_MASKS, "sum"), COL_MASKS, "sum")[..., 0]   # [Q, H, D]
        out[c0:c0 + Q] = (ob * rL[..., None]).astype(F32).astype(np.float16)
    return out


def prefill_reference(lay, qkv32, cos_t, sin_t, Kc, Vc, p0, M):
    """(attn fp16 [M, H, D], q16, K, V): the append, then attention for every row."""
    q16, K, V = rope_append_reference(lay, qkv32, cos_t, sin_t, Kc, Vc, p0, M)
    q0s = np.minimum(p0 + np.arange(M), lay["cap"] - 1)
    return attn_reference_many(lay, q16, K, V, q0s), q16, K, V


def attn_reference_one(lay, q16_row, K, V, q0):
    """The decode reference for one row (g17attn.attn_reference, wide butterfly form): the cross-check."""
    return A.attn_reference(dict(lay, slices=32), q16_row, K, V, q0)


# ------------------------------------------------------------------------------------------------------------ io
def rope_tables(cap, D=128, theta=1e6):
    th = np.float64(theta) ** (-np.arange(D // 2) * 2 / D)
    pos = np.arange(cap)[:, None] * th[None, :]
    return np.cos(pos).astype(F32), np.sin(pos).astype(F32)


def case(lay, M, p0, seed=11):
    """qkv32 rows, rope tables, and caches whose rows at and past p0 are NaN (a kernel reading one it should not fails)."""
    rng = np.random.default_rng(seed)
    H, KVH, D, CAP = lay["heads"], lay["kv_heads"], lay["head_dim"], lay["cap"]
    qkv = (rng.standard_normal((M, lay["QKVROW"])) * 1.5).astype(F32)
    cos, sin = rope_tables(CAP, D)
    Kc = rng.standard_normal((KVH, CAP, D)).astype(np.float16)
    Vc = rng.standard_normal((KVH, CAP, D)).astype(np.float16)
    Kc[:, p0:] = np.float16(np.nan)
    Vc[:, p0:] = np.float16(np.nan)
    return qkv, cos, sin, Kc, Vc


def io(lay, M, p0, qkv, cos, sin, Kc, Vc):
    """(a, b, c) for both dispatches: c carries the caches, q16 and the output pre-filled with a 0x7f sentinel."""
    a = bytearray(max(lay["qkv_bytes"], 256))
    O._place(a, 0, np.asarray(qkv, "<f4").reshape(-1))
    b = bytearray(lay["rope_bytes"])
    O._place(b, 0, np.asarray([p0], "<u4"))
    O._place(b, lay["COST"], np.asarray(cos, "<f4").reshape(-1))
    O._place(b, lay["SINT"], np.asarray(sin, "<f4").reshape(-1))
    c = bytearray(lay["prefill_region3_bytes"])
    O._place(c, lay["KOFF"], np.asarray(Kc, np.float16).reshape(-1))
    O._place(c, lay["VOFF"], np.asarray(Vc, np.float16).reshape(-1))
    c[lay["Q16"]:lay["PATTN"]] = b"\x7f" * (lay["PATTN"] - lay["Q16"])
    c[lay["PATTN"]:lay["PATTN"] + M * lay["heads"] * lay["head_dim"] * 4] = b"\x7f" * (M * lay["heads"] * lay["head_dim"] * 4)
    return bytes(a), bytes(b), bytes(c)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("check",))
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--p0", type=int, default=0)
    ap.add_argument("--cap", type=int, default=272)
    args = ap.parse_args(argv)
    lay = prefill_layout(args.cap, args.m)
    print("append %d bytes, attention %d bytes" % (len(build_prefill_append(lay).code), len(build_prefill_attn(lay).code)))


if __name__ == "__main__":
    main()
