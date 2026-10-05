#!/usr/bin/env python3
"""Flash-decoding attention for one query row (MM 25.140): two scalar dispatches per layer in place of the
tensor grid's split + merge + gather (113 us per layer at kv_len 128, Set C's profile). 11.9 us at 128 keys.

Contract (carrier-free, bound directly by the decode executor; S = 16 slices by default):
  split   H S threadgroups of 32; threadgroup h S + s runs key slice s of q head h
          binding 1  q fp16 [H][D] at 0 (already scaled by log2(e)/sqrt(D): the softmax is base 2), and the
                     uint32 q0 (the new token's position; keys 0..q0 are valid) at byte LEN
          binding 2  K fp16 [KVH][CAP][D] at 0, V fp16 [KVH][CAP][D] at VOFF (per KV head, row-major)
          binding 3  partials fp32 [H][S][PW] at 0: m, l, two pad words, o[D] (un-normalised)
  merge   H threadgroups of 32
          binding 1  the same q region (q0 at LEN); binding 2 the partials; binding 3 attn fp16 [H][D] at 0
  GQA: q head h reads KV head h // (H / KVH). ONE threadgroup per slice, not one threadgroup of S
  simdgroups merging through threadgroup memory: the object author has measured only 32-thread cooperative
  classes (cooperativemetadata: "unmeasured cooperative required_size").

Order (attn_reference replays it exactly):
  slice s takes keys j = s, s + S, ... <= q0; lane l owns dims 4l..4l+3 of D = 128
  score  = butterfly_sum(fma chain q_d k_d over the lane's 4 dims), rows then columns (every lane agrees)
  per key: m' = max(m, s); a = exp2_soft(m - m'); p = exp2_soft(s - m'); l = l a + p; o_d = o_d a + p v_d
           (a masked key - past q0, run only so an empty slice's loop has one trip - takes s = m and p = 0,
           and reads key min(j, q0) so no unwritten cache row is ever multiplied)
  merge, slices ascending: M = max m_s; w_s = exp2_soft(m_s - M) (0 for an empty slice s > q0);
           L = sum l_s w_s; O_d = sum o_s,d w_s; attn_d = fp16_rne(O_d recip_rn(L))
q0 is clamped to CAP - 1, and the loop latch states its cap: a bad word truncates, never hangs.

    python3 tools/g17attn.py check
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
NEG_MAX = F32(np.frombuffer(np.uint32(0xFF7FFFFF).tobytes(), F32)[0])


def _align(v, a=256):
    return -(-v // a) * a


def attn_layout(heads=16, kv_heads=8, head_dim=128, cap=272, slices=16):
    if head_dim != 128:
        raise ValueError("attn: lanes own 4 of 128 dims")
    if heads % kv_heads or slices not in (2, 4, 8, 16, 32):
        raise ValueError("attn: whole GQA groups, and a power-of-two slice count")
    LEN = heads * head_dim * 2
    VOFF = _align(kv_heads * cap * head_dim * 2)
    PW = 4 + head_dim                                  # partial words per (head, slice): m, l, 2 pad, o[D]
    return dict(op="attn", heads=heads, kv_heads=kv_heads, head_dim=head_dim, cap=cap, slices=slices, PW=PW,
                Q=0, LEN=LEN, K=0, VOFF=VOFF, P=0, OUT=0, trips_cap=-(-cap // slices),
                split_groups=heads * slices, merge_groups=heads,
                q_bytes=_align(LEN + 4), kv_bytes=VOFF + _align(kv_heads * cap * head_dim * 2),
                partial_bytes=_align(heads * slices * PW * 4), out_bytes=_align(heads * head_dim * 2), nocarrier=True)


def _exp2(lay, b, x, K2, tag):
    """2^x for the softmax. Default: the ~14-op exp2_soft (MM 25.144.2), bit-exact against the reference. With
    lay["hw_exp2"]: the HARDWARE exp2 (op1272, b.exp2) in ONE instruction (M2's lowering, MM 25.144.2 / the exp2 PR).
    op1272 is within 1 ulp of true exp2 but not bit-exactly reproducible on the CPU
    (isa/g17-exp2-op1272-characterization.json), so a hw_exp2 attention kernel is NOT bit-identical to the exp2_soft
    reference - it is validated by a softmax enclosure (attn_reference with hw_exp2 uses true exp2) and model tokens.
    Off by default, so every delivered bit-exact attention kernel keeps exp2_soft."""
    from agxforge.g17 import ir as _ir
    if lay.get("hw_exp2"):
        return b.exp2(x, type=_ir.I32, name=tag)
    return O.emit_exp2_soft(b, x, K2, tag)


def _lane_dims(b, ir):
    lane = b.builtin("thread_index_in_simdgroup", name="lane")
    return lane, b.shl(lane, O._c(b, 2, "two"), name="d0")


def build_attn_split(lay):
    """Dispatch 1: threadgroup g = h S + s runs slice s of q head h and writes its un-normalised partial
    (m, l, o[D]) at binding 3 [P + (h S + s) PW]. Bindings: 1 q + the q0 word, 2 the KV cache, 3 partials."""
    from agxforge.g17 import cc, ir, tensorreduce as TR
    H, KVH, D, CAP, S, PW = lay["heads"], lay["kv_heads"], lay["head_dim"], lay["cap"], lay["slices"], lay["PW"]
    TC = lay["trips_cap"]
    fn, b, a, bb, c = O._function()
    I = ir.I32
    lane, d0 = _lane_dims(b, ir)
    g = b.builtin("threadgroup_position_in_grid", name="g")
    lgS = S.bit_length() - 1
    h = b.shr(g, O._c(b, lgS, "lgS"), name="head")
    s = b.sub(g, b.shl(h, O._c(b, lgS, "lgS1"), name="hS"), name="slice")
    q0 = b.load(a, O._c(b, lay["LEN"] // 4, "len_w"), type=I, name="q0raw")
    capm1 = O._c(b, CAP - 1, "capm1")
    q0 = b.csel(q0, capm1, capm1, q0, rel="gt", name="q0")
    trips = b.shr(b.sub(b.add(q0, O._c(b, S, "S"), name="q0S"), s, name="q0Ss"), O._c(b, lgS, "lgS2"), name="trips")
    one = O._c(b, 1, "one")
    zero = O._c(b, 0, "zero")
    trips = b.csel(trips, zero, trips, one, rel="gt", name="trips1")
    qb = b.add(b.mul(h, O._c(b, D, "D"), name="hD"), d0, name="qb")
    qv = [b.f16_to_f32(b.load(a, b.add(qb, O._c(b, i, "qo%d" % i), name="qi%d" % i) if i else qb, width="half", name="qh%d" % i),
                       name="q%d" % i) for i in range(4)]
    kvh = b.shr(h, O._c(b, (H // KVH).bit_length() - 1, "gqa"), name="kvh")
    kb = b.add(b.mul(kvh, O._c(b, CAP * D, "CAPD"), name="kvoff"), d0, name="kb")
    vb = b.add(kb, O._c(b, lay["VOFF"] // 2, "voff"), name="vb")
    negmax = O._cf(b, NEG_MAX, "negmax")
    fzero = O._cf(b, F32(0.0), "fzero")
    neg1 = O._cf(b, F32(-1.0), "neg1")
    K2 = O.emit_exp2_constants(b)
    # one zero per loop-carried value: phis seeded from one shared constant coalesce into one register
    lzero = O._cf(b, F32(0.0), "lzero")
    ozeros = [O._cf(b, F32(0.0), "ozero%d" % i) for i in range(4)]
    hdr, post = fn.block("keys"), fn.block("out")
    b.br(hdr)
    b.at(hdr)
    t = b.phi(zero, name="t")
    m = b.phi(negmax, type=ir.F32, name="m")
    l = b.phi(lzero, type=ir.F32, name="l")
    o = [b.phi(ozeros[i], type=ir.F32, name="o%d" % i) for i in range(4)]
    j = b.add(s, b.shl(t, O._c(b, lgS, "lgS3"), name="tS"), name="j")
    jr = b.csel(j, q0, q0, j, rel="gt", name="jr")              # the row read: never past q0
    row = b.mul(jr, O._c(b, D, "D2"), name="row")
    ki = b.add(kb, row, name="ki")
    vi = b.add(vb, row, name="vi")
    kv = [b.f16_to_f32(b.load(bb, b.add(ki, O._c(b, i, "ko%d" % i), name="kii%d" % i) if i else ki, width="half", name="kh%d" % i),
                       name="k%d" % i) for i in range(4)]
    vv = [b.f16_to_f32(b.load(bb, b.add(vi, O._c(b, i, "vo%d" % i), name="vii%d" % i) if i else vi, width="half", name="vh%d" % i),
                       name="v%d" % i) for i in range(4)]
    part = b.fmul(qv[0], kv[0], type=ir.F32, name="pd0")
    for i in range(1, 4):
        part = b.fma(qv[i], kv[i], part, name="pd%d" % i)
    part.type = ir.F32                                          # fma's value is fp32 bits; the reduction asks the type
    sc = TR.emit_butterfly(b, part, TR.ROW_BUTTERFLY_MASKS, operation="sum")
    sc = TR.emit_butterfly(b, sc, TR.COLUMN_BUTTERFLY_MASKS, operation="sum")
    sc = b.csel(j, q0, m, sc, rel="gt", name="sc")              # masked: s = m
    mn = b.fmax(m, sc, type=I, name="mn")
    nmn = b.fmul(mn, neg1, type=I, name="nmn")
    al = _exp2(lay, b, b.fadd(m, nmn, type=I, name="dm"), K2, "al")
    p = _exp2(lay, b, b.fadd(sc, nmn, type=I, name="ds"), K2, "pe")
    p = b.csel(j, q0, fzero, p, rel="gt", name="p")             # masked: p = 0
    ln = b.fadd(b.fmul(l, al, type=I, name="la"), p, type=ir.F32, name="ln")
    on = [b.fma(p, vv[i], b.fmul(o[i], al, type=I, name="oa%d" % i), name="on%d" % i) for i in range(4)]
    tn = b.add(t, one, name="tn")
    ir.Builder.phi_latch(t, tn)
    ir.Builder.phi_latch(m, mn)
    ir.Builder.phi_latch(l, ln)
    for i in range(4):
        ir.Builder.phi_latch(o[i], on[i])
    b.br_cond(b.cmp(tn, trips, "lt", cap=TC), hdr, post)
    b.at(post)
    pb = b.add(b.mul(g, O._c(b, PW, "PW"), name="gPW"), O._c(b, lay["P"] // 4, "Pb"), name="pb")
    b.store_at(c, pb, mn)
    b.store_at(c, b.add(pb, one, name="pb1"), ln)
    ob = b.add(b.add(pb, O._c(b, 4, "four"), name="pb4"), d0, name="ob")
    for i in range(4):
        b.store_at(c, b.add(ob, O._c(b, i, "oo%d" % i), name="obi%d" % i) if i else ob, on[i])
    b.ret()
    ir.verify(fn)
    return cc.compile_function(fn)


def attn_rope_layout(heads=16, kv_heads=8, head_dim=128, cap=272, slices=16):
    """The split with RoPE and the cache append folded in (MM 25.140.1): binding 1 is the qkv projection's fp32
    output [q H D | k KVH D | v KVH D] with q0 at byte LEN; binding 2 the position's rope rows, cos fp32 [D/2]
    at 0 and sin at SIN; binding 3 [partials at 0 | K cache at KOFF | V cache at VOFF] (the new row WRITTEN)."""
    lay = attn_layout(heads, kv_heads, head_dim, cap, slices)
    LEN = (heads + 2 * kv_heads) * head_dim * 4
    KOFF = lay["partial_bytes"]
    VOFF = KOFF + _align(kv_heads * cap * head_dim * 2)
    return dict(lay, rope=True, LEN=LEN, SIN=_align(head_dim // 2 * 4), KOFF=KOFF, VOFF=VOFF,
                q_bytes=_align(LEN + 4), rope_bytes=2 * _align(head_dim // 2 * 4),
                region3_bytes=VOFF + _align(kv_heads * cap * head_dim * 2))


def with_rope_tables(lay):
    """The device-resident split: binding 2 = [q0 word | cos [CAP][D/2] at COST | sin at SINT] (MM 25.140.2)."""
    D, CAP = lay["head_dim"], lay["cap"]
    # after the q0 word and the token log: g17gen.gen_layout puts the log int32 [CAP] at GEN + 4, so it ends at byte
    # 4 + 4 CAP of this region. A fixed 2048 held only for CAP <= 511; at CAP 2048 the log (to 8196) would have
    # overwritten the cos table. 2048 is kept as the floor so every CAP <= 511 layout is byte-unchanged.
    COST = max(2048, _align(4 + 4 * CAP))
    SINT = COST + _align(CAP * D // 2 * 4)
    return dict(lay, rope_tables=True, COST=COST, SINT=SINT, rope_bytes=SINT + _align(CAP * D // 2 * 4))


def with_wide(lay):
    """ATTENTION AS ONE 1,024-THREAD THREADGROUP PER HEAD (MM 25.141.11), MLX sdpa_vector's shape: simdgroup s of head
    h's threadgroup does what split threadgroup (h, s) did (keys s, s + 32, ...), the 32 partials meet in threadgroup
    memory after one barrier, and simdgroup 0 merges them with build_attn_merge's arithmetic. No device atomics, no
    fence, no last-threadgroup merge. Takes the fused split's with_fused_merge layout (the same binding-0 region, so
    it is a drop-in; the counter and partial words go unused)."""
    # the REGION stays the fused split's (its offsets come from the layout it is given, normally 16 slices), so the
    # kernel is a drop-in; only the kernel's own slice count - its simdgroups - becomes 32
    if not lay.get("fused_merge"):
        raise ValueError("wide attention takes a with_fused_merge layout")
    return dict(lay, wide=True, slices=32, trips_cap=-(-lay["cap"] // 32), split_groups=lay["heads"])


def with_bfly_merge(lay):
    """The wide form's merge in PARALLEL (MM 25.141.14): simdgroup s' merges output dims 4 s' .. 4 s' + 3, lane l holding
    slice l; the maximum, L and the four dims are lane butterflies. The reference (attn_reference with this flag) sums in
    the butterfly order, not the slice order."""
    if not lay.get("wide"):
        raise ValueError("the butterfly merge is an option of the wide form")
    return dict(lay, bfly_merge=True)


def with_keyblock(lay, kb):
    """The wide form's key loop with kb keys per trip (MM 25.144.5): the block's loads, dots and reductions overlap, and
    one rescale exp2 serves the block. A new order: attn_reference with the flag follows it."""
    if not lay.get("wide") or kb not in (1, 2, 4, 8):
        raise ValueError("keyblock: the wide form, 1, 2, 4 or 8 keys per trip")
    return dict(lay, keyblock=kb, trips_cap=-(-lay["trips_cap"] // kb))


def with_gqapair(lay):
    """The wide form with each GQA pair's two q heads in one threadgroup walking the same keys (MM 25.144.5): threadgroup g =
    (KV head g >> 1, key half p = g & 1), simdgroup s serves q head 2 kvh + (s & 1) at slice 16 p + (s >> 1) of 32, so
    the second read of each K/V row is a near hit (MLX's sdpa_vector_2pass_1 pairs its GQA heads the same way). Each
    threadgroup merges its 16 slices per head unnormalized (lane butterflies over masks 2, 4, 8, 16, which keep a lane's
    head parity); the last threadgroup of the pair merges the two halves per head. A new order: attn_reference follows it."""
    if not lay.get("wide") or not lay.get("bfly_merge") or lay["heads"] != 2 * lay["kv_heads"]:
        raise ValueError("gqapair: the wide butterfly form with a GQA ratio of 2")
    if lay.get("tgsplit"):
        raise ValueError("gqapair and tgsplit are not composed")
    return dict(lay, gqapair=True, split_groups=lay["heads"])


def with_tgsplit(lay, n):
    """The wide form with n threadgroups per head (MM 25.144.5): threadgroup (h, p)'s simdgroup s takes slice 32 p + s of
    32 n, so each walks 1/n of the keys; each threadgroup merges its 32 slices unnormalized into a device partial and the
    last of the head (the fused merge's counters) merges the n partials. A new order: attn_reference follows it."""
    if not lay.get("wide") or not lay.get("bfly_merge") or n not in (2, 4, 8):
        raise ValueError("tgsplit: the wide butterfly form, 2, 4 or 8 threadgroups per head")
    tc = -(-lay["cap"] // (32 * n))
    return dict(lay, tgsplit=n, kslices=32 * n, split_groups=lay["heads"] * n,
                trips_cap=-(-tc // lay.get("keyblock", 1)))


def with_hwexp2(lay):
    """The softmax's 2^x by the HARDWARE exp2 (op1272) in one instruction, not the ~14-op exp2_soft (MM 25.144.2). MLX's
    sdpa uses the same hardware exp. It is NOT bit-exact to the exp2_soft reference (op1272 is within 1 ulp of true exp2,
    not CPU-reproducible), so a hwexp2 kernel is validated by a softmax enclosure against attn_float64 (true base-2
    softmax) and by model tokens, never bit-exactly. Off by default; composes with any wide form (kvvec, keyblock, ...)."""
    if not lay.get("wide"):
        raise ValueError("hwexp2: the wide form")
    return dict(lay, hw_exp2=True)


def with_qknorm(lay, eps=1e-6):
    """QK-NORM IN THE ATTENTION (Qwen3, MM 25.188): the per-head RMSNorm of k and q, applied to the loaded rows before
    RoPE. A simdgroup holds one head across its 32 lanes (4 dims a lane), so it is g17qwen3.build_headnorm's arithmetic
    in its order - ordered squares, the row and column butterflies, mean, eps, the corrected rsqrt, (x r) g - and the
    kernel's values equal QK-norm then this attention. The gains, fp16 q_norm [128] then k_norm [128], sit in binding 1
    at QG (after the qkv row and its q0 word)."""
    if not lay.get("wide"):
        raise ValueError("qknorm: the wide form")
    QG = _align(lay["LEN"] + 4)
    return dict(lay, qknorm=True, qknorm_eps=float(eps), QG=QG, q_bytes=_align(QG + 4 * lay["head_dim"]))


def with_kvvec(lay):
    """The wide form's key loop reading each K and V row with ONE 8-byte load per lane (two words, split into four fp16
    halves) instead of four 2-byte loads (MM 25.144.5). The values and the order are unchanged: with no other M5 flag it is
    the base order exactly."""
    if not lay.get("wide"):
        raise ValueError("kvvec: the wide form")
    return dict(lay, kvvec=True)


def with_probe(lay, which):
    """TIMING-ONLY key-loop probes on the M5 loop (wrong values): which = "kv1" (one half load per K and V row) or "nokv"."""
    if which not in ("kv1", "nokv", "kvshare"):
        raise ValueError("probe: kv1, nokv or kvshare")
    lay = dict(lay, **{"probe_" + which: True})
    return lay if lay.get("keyblock") else dict(lay, keyblock=1, nsum=lay.get("nsum", False), m5probe=True)


def with_nsum(lay):
    """The wide form's per-key score reduced by the native simd_sum (op16842; the xor butterfly 1, 2, 4, 8, 16, MM
    25.141.16), instead of the row and column shuffle butterflies (MM 25.144.5). A new order: attn_reference follows it."""
    if not lay.get("wide"):
        raise ValueError("nsum: the wide form")
    return dict(lay, nsum=True)


def with_attn32(lay):
    """The fused merge writing attn as fp32 of its fp16-rounded value (exact widening) at ATTN, [H][D] x 4 bytes, so
    wo reads it as xvec fp32 x (MM 25.141.9)."""
    if not lay.get("fused_merge"):
        raise ValueError("attn32 is an option of the fused merge")
    grow = _align(lay["heads"] * lay["head_dim"] * 4) - _align(lay["heads"] * lay["head_dim"] * 2)
    return dict(lay, attn32=True, region3_bytes=lay["region3_bytes"] + grow)


def with_batch(lay, nb, SS):
    """BATCHED DECODE ATTENTION (MM 25.144.3): nb sequences in ONE dispatch of nb H threadgroups (the wide butterfly
    form), threadgroup g = sequence (g >> log2 H), head (g & (H-1)). Sequence b, vector-major:
      binding 1: its qkv32 row at b LEN bytes (the batched qkv qmv's out stride, 4 N);
      binding 2: its q0 word at b SS bytes (g17gen.gen_batch_layout's state block), the rope tables SHARED after all
                 nb blocks (COST = max(2048, align(nb SS)));
      binding 0: its K and V caches at KOFF + b KVS, KVS = the single form's K + V span; its attn32 row [H][D] fp32 at
                 ATTN + b H D 4, ATTN after all caches (the batched wo's x stride, 4 K).
    Each sequence's value is the single-sequence kernel's (the same program per threadgroup, offsets aside)."""
    if not (lay.get("wide") and lay.get("bfly_merge") and lay.get("attn32") and lay.get("rope_tables")):
        raise ValueError("batched attention: the wide butterfly attn32 form with rope tables")
    H, D, KVH, CAP = lay["heads"], lay["head_dim"], lay["kv_heads"], lay["cap"]
    if H & (H - 1):
        raise ValueError("batched attention: a power-of-two head count")
    KVS = 2 * _align(KVH * CAP * D * 2)
    if lay["VOFF"] - lay["KOFF"] != KVS // 2:
        raise ValueError("batched attention: K and V caches adjacent")
    COST = max(2048, _align(nb * SS))
    SINT = COST + _align(CAP * D // 2 * 4)
    ATTN = _align(lay["KOFF"] + nb * KVS)
    out = dict(lay, batch=nb, SS=SS, KVS=KVS, COST=COST, SINT=SINT, rope_bytes=SINT + _align(CAP * D // 2 * 4),
               ATTN=ATTN, OUT_AT=ATTN, region3_bytes=ATTN + nb * H * D * 4)
    if lay.get("gqapair"):
        # the pairs' cross-threadgroup partials, per sequence (4 per pair, PW words each), after the attn rows; the pair
        # counters are fields (sequence, KV head) = pair index nb KVH, 6 per 16-byte word below byte 256 (at most 96)
        pairs = nb * (H // 2)
        if pairs > 64:
            raise ValueError("batched gqapair: at most 64 pairs (the counter field arithmetic)")
        PB = _align(ATTN + nb * H * D * 4)
        out.update(PB=PB, region3_bytes=PB + pairs * 4 * lay["PW"] * 4)
    return out


def with_fused_merge(lay):
    """The split merging in its last threadgroup, in the cooperative class: binding 0 (written) is [three count
    words at 0, 16, 32 (6 heads x 5 bits each, zero at rest) | partials | K | V | attn fp16 [H][D] at ATTN];
    binding 1 is qkv32; binding 2 is the q0 + rope-table region. The counters sit in the first 256 bytes
    because op10094's byte offset (operand 6) is eight bits."""
    sh = 256
    return dict(lay, fused_merge=True, coop=True, CNTB=0, P=lay["P"] + sh, KOFF=lay["KOFF"] + sh, VOFF=lay["VOFF"] + sh,
                ATTN=lay["region3_bytes"] + sh, OUT_AT=lay["region3_bytes"] + sh,
                region3_bytes=lay["region3_bytes"] + sh + _align(lay["heads"] * lay["head_dim"] * 2))

def build_attn_split_rope(lay):
    """build_attn_split with q rotated and scaled in-kernel and the new k, v row made, used from registers and
    written to the cache. Rotate-half pairs dim d with d + D/2, which is lane l ^ 16 at the same slot."""
    from agxforge.g17 import cc, ir, tensorreduce as TR
    import g17decodestep as D_
    H, KVH, D, CAP, S, PW = lay["heads"], lay["kv_heads"], lay["head_dim"], lay["cap"], lay["slices"], lay["PW"]
    TC = lay["trips_cap"]
    wide = lay.get("wide")
    if lay.get("coop"):
        # THE COOPERATIVE THREE-BINDING CLASS (slot 0 written, 1 and 2 read; 32-thread threadgroups with a
        # declared scratchpad; system registers 156 and 164 only): its written buffer is rank 0, where the device
        # atomics are measured (MM 25.140.3)
        c = ir.Buffer("C", 0, elem=ir.F32)
        a = ir.Buffer("A", 1, elem=ir.F16)
        bb = ir.Buffer("B", 2, elem=ir.F16)
        fn = ir.Function("tensor_gemm_generic_runtime_demo", [c, a, bb])
        if wide:
            # the 32 partials [m, l, 2 pad, o[D]] in the scratchpad; 1,024 threads (MEASURED_SIZES, MM 25.141.4)
            fn.declare_threadgroup(S * PW, size=(32 * S, 1, 1))
        else:
            fn.declare_threadgroup(4, size=(32, 1, 1))
        b = ir.Builder(fn, fn.block("entry"))
        tpos = b.builtin("thread_position_in_threadgroup", name="tpos")
        lane = getattr(b, "and")(tpos, ir.Imm(31), name="lane") if wide else tpos
        d0 = b.shl(lane, O._c(b, 2, "two"), name="d0")
    else:
        fn, b, a, bb, c = O._function()
        lane, d0 = _lane_dims(b, ir)
    I = ir.I32
    g = b.builtin("threadgroup_position_in_grid", name="g")
    lgS = S.bit_length() - 1
    NBT = lay.get("batch", 1)
    seqb = None
    if wide and NBT > 1:
        if lay.get("tgsplit"):
            raise ValueError("attn: batch and tgsplit are not composed (neither is verified with the other)")
        # BATCHED (with_batch): threadgroup g is (sequence g >> log2 H, head g & (H - 1))
        seqb = b.shr(g, O._c(b, H.bit_length() - 1, "lgH"), name="seqb")
        s = b.shr(tpos, O._c(b, 5, "five_s"), name="slice")
        if lay.get("gqapair"):
            # BATCHED GQA PAIRS (M3 on M5's with_gqapair): the local index gl = g & (H - 1) is the single-sequence pair
            # threadgroup (KV head gl >> 1, key half gl & 1)
            gl = getattr(b, "and")(g, O._c(b, H - 1, "hmask"), name="gl")
            tsp = getattr(b, "and")(gl, O._c(b, 1, "gp1"), name="tsp")
            h = b.add(b.sub(gl, tsp, name="g2k"), getattr(b, "and")(s, O._c(b, 1, "sp1"), name="spar"), name="head")
        else:
            h = getattr(b, "and")(g, O._c(b, H - 1, "hmask"), name="head")
    elif wide and lay.get("gqapair"):
        # GQA PAIRED (with_gqapair, MM 25.144.5): threadgroup g = (KV head g >> 1, key half p = g & 1); simdgroup s serves q
        # head 2 kvh + (s & 1) at slice 16 p + (s >> 1), so the two q heads of the pair walk the same keys together
        s = b.shr(tpos, O._c(b, 5, "five_s"), name="slice")
        tsp = getattr(b, "and")(g, O._c(b, 1, "gp1"), name="tsp")
        h = b.add(b.sub(g, tsp, name="g2k"), getattr(b, "and")(s, O._c(b, 1, "sp1"), name="spar"), name="head")
    elif wide:
        if lay.get("tgsplit"):
            lgN = lay["tgsplit"].bit_length() - 1
            h = b.shr(g, O._c(b, lgN, "lgN"), name="head")
            tsp = b.sub(g, b.shl(h, O._c(b, lgN, "lgN1"), name="hN"), name="tsp")
        else:
            h = b.add(g, O._c(b, 0, "gz"), name="head")
        s = b.shr(tpos, O._c(b, 5, "five_s"), name="slice")
    else:
        h = b.shr(g, O._c(b, lgS, "lgS"), name="head")
        s = b.sub(g, b.shl(h, O._c(b, lgS, "lgS1"), name="hS"), name="slice")
    tables = lay.get("rope_tables")
    # DEVICE-RESIDENT GENERATION (MM 25.140.2): q0 and the rope rows come from binding 2 - the word at [0] and the
    # constant tables cos [CAP][D/2] at COST, sin at SINT - so no host write precedes a token
    if seqb is not None:
        q0 = b.load(bb, b.mul(seqb, O._c(b, lay["SS"] // 4, "ssw"), name="q0wi"), type=I, name="q0raw")
    else:
        q0 = b.load(bb if tables else a, O._c(b, 0 if tables else lay["LEN"] // 4, "len_w"), type=I, name="q0raw")
    capm1 = O._c(b, CAP - 1, "capm1")
    q0 = b.csel(q0, capm1, capm1, q0, rel="gt", name="q0")
    one = O._c(b, 1, "one")
    zero = O._c(b, 0, "zero")
    # the register-select key path reads q0 - 1; the wide form reads row q0 back. Under the M5 flags it is not formed (the
    # base wide form keeps its unread q0m1 because test_g17cap2048 pins those bytes)
    m5 = wide and (lay.get("keyblock", 1) > 1 or lay.get("nsum") or lay.get("tgsplit") or lay.get("m5probe") or lay.get("kvvec")
                   or lay.get("gqapair"))
    # the batched form (M3) never forms it either: its bytes are new
    q0m1 = None if (m5 or seqb is not None or lay.get("hw_exp2")) else b.sub(q0, one, name="q0m1")
    ksig, klgS = s, lgS
    if wide and lay.get("gqapair"):
        # slice 16 p + (s >> 1) of the head's 32: the keys it walks are the base's for that slice
        ksig = b.add(b.shl(tsp, O._c(b, 4, "f4k"), name="tsp16"), b.shr(s, O._c(b, 1, "s1k"), name="shalf"), name="ksig")
        trips = b.shr(b.sub(b.add(q0, O._c(b, S, "S"), name="q0S"), ksig, name="q0Ss"), O._c(b, lgS, "lgS2"), name="trips")
    elif wide and lay.get("tgsplit"):
        # threadgroup p's simdgroup s is slice 32 p + s of 32 N: the keys it walks step by 32 N
        SS = lay["kslices"]
        klgS = SS.bit_length() - 1
        ksig = b.add(b.shl(tsp, O._c(b, 5, "f5k"), name="tsp32"), s, name="ksig")
        trips = b.shr(b.sub(b.add(q0, O._c(b, SS, "SS"), name="q0SS"), ksig, name="q0SSs"), O._c(b, klgS, "lgSS2"), name="tripsS")
    else:
        trips = b.shr(b.sub(b.add(q0, O._c(b, S, "S"), name="q0S"), s, name="q0Ss"), O._c(b, lgS, "lgS2"), name="trips")
    trips = b.csel(trips, zero, trips, one, rel="gt", name="trips1")
    kvh = b.shr(h, O._c(b, (H // KVH).bit_length() - 1, "gqa"), name="kvh")
    if lay.get("probe_kvshare"):
        # TIMING-ONLY (wrong values, MM 25.144.5): every head reads KV head 0, so the unique KV bytes fall 8x while the
        # instructions and load requests stay the same
        kvh = b.mul(kvh, O._c(b, 0, "kvz"), name="kvh0")
    # the rope rows at this lane's dims mod D/2: dm = 4 (l & 15) + i
    dm = b.shl(b.sub(lane, b.shl(b.shr(lane, O._c(b, 4, "f4"), name="lhi"), O._c(b, 4, "f4b"), name="lhi16"), name="llo"),
               O._c(b, 2, "two_"), name="dm0")
    if tables:
        dm = b.add(dm, b.mul(q0, O._c(b, D // 2, "halfD"), name="q0row"), name="dmrow")
    cbase, sbase = (lay["COST"] // 4, lay["SINT"] // 4) if tables else (0, lay["SIN"] // 4)
    cs = [b.load(bb, b.add(dm, O._c(b, cbase + i, "co%d" % i), name="ci%d" % i) if (i or cbase) else dm, type=I, name="cos%d" % i) for i in range(4)]
    sn = [b.load(bb, b.add(dm, O._c(b, sbase + i, "so%d" % i), name="si%d" % i), type=I, name="sin%d" % i) for i in range(4)]
    upper = O._c(b, 15, "l15")
    if wide:
        # the rotate-half sign as a value, sign = 2 (lane >> 4) - 1 = -1 (lanes 0..15) or +1 (16..31), so the rope is
        # ac + bs sign = ac -/+ bs exactly (a multiply by +-1 is exact, and the one fadd rounds as before)
        # and needs no select - the four selects on `lane` held narrow registers this program does not have
        sgn = b.fadd(b.fmul(b.u32_to_f32(b.shr(lane, O._c(b, 4, "sg4"), name="lh4"), name="lh4f"),
                            O._cf(b, F32(2.0), "ftwo"), type=I, name="lh8"), O._cf(b, F32(-1.0), "fneg_s"), type=ir.F32, name="rsgn")

    def rope(vals, tag):
        out = []
        for i in range(4):
            own = vals[i]
            own.type = ir.F32
            part = b.simd_shuffle_xor(own, 16, name="%s_pt%d" % (tag, i))
            ac = b.fmul(own, cs[i], type=I, name="%s_ac%d" % (tag, i))
            bs = b.fmul(part, sn[i], type=I, name="%s_bs%d" % (tag, i))
            if wide:
                out.append(b.fadd(ac, b.fmul(bs, sgn, type=I, name="%s_sb%d" % (tag, i)), type=I, name="%s_r%d" % (tag, i)))
                continue
            # lanes 0..15 hold x1: x1 cos - x2 sin; lanes 16..31 hold x2: x2 cos + x1 sin (both formed, one
            # selected: fneg is a source modifier and cannot feed a select)
            lo = b.fadd(ac, b.fneg(bs, type=I, name="%s_nbs%d" % (tag, i)), type=I, name="%s_lo%d" % (tag, i))
            hi = b.fadd(ac, bs, type=I, name="%s_hi%d" % (tag, i))
            out.append(b.csel(lane, upper, hi, lo, rel="gt", name="%s_r%d" % (tag, i)))
        return out

    QW = lay.get("QKV", 0) // 4                        # where qkv32 starts in binding 1 (after the counters)

    def qknorm(vals, goff, tag):
        """with_qknorm: this head's RMSNorm across the simdgroup (g17qwen3.build_headnorm's order)."""
        from agxforge.g17 import tensorreduce as TR
        acc = None
        for i, x in enumerate(vals):
            sq = b.fmul(x, x, type=ir.F32, name="%s_sq%d" % (tag, i))
            acc = sq if acc is None else b.fadd(acc, sq, type=ir.F32, name="%s_acc%d" % (tag, i))
        ss = TR.emit_butterfly(b, acc, TR.ROW_BUTTERFLY_MASKS, operation="sum")
        ss = TR.emit_butterfly(b, ss, TR.COLUMN_BUTTERFLY_MASKS, operation="sum")
        mean = b.fmul(ss, O._cf(b, F32(1.0 / D), tag + "_invd"), name=tag + "_mean")
        KN = O.emit_constants(b)
        r = O.emit_rn(b, "rsqrt", b.fadd(mean, O._cf(b, F32(lay["qknorm_eps"]), tag + "_eps"), type=I, name=tag + "_var"),
                      KN, tag + "_rs")
        gi = b.add(d0, O._c(b, lay["QG"] // 2 + goff, tag + "_go"), name=tag + "_gi")
        out = []
        for i, x in enumerate(vals):
            g = b.f16_to_f32(b.load(a, b.add(gi, O._c(b, i, "%s_gio%d" % (tag, i)), name="%s_gii%d" % (tag, i)) if i else gi,
                                    width="half", name="%s_gh%d" % (tag, i)), name="%s_g%d" % (tag, i))
            out.append(b.fmul(b.fmul(x, r, name="%s_xr%d" % (tag, i)), g, name="%s_n%d" % (tag, i)))
        return out
    if lay.get("qknorm") and not wide:
        raise ValueError("qknorm: the wide form")
    if wide:
        # the WIDE form appends k, v FIRST: nothing of them is kept past the append (the key loop reads row q0 back),
        # so their registers are free before q is rotated - the rope point was this program's register peak
        # batched: this sequence's qkv row (LEN bytes each) and its K/V caches (KVS bytes each)
        qsb = b.mul(seqb, O._c(b, lay["LEN"] // 4, "lenw"), name="qsb") if seqb is not None else None
        d0q = b.add(d0, qsb, name="d0q") if qsb is not None else d0
        kin_b = b.add(b.mul(kvh, O._c(b, D, "D_k"), name="kvD"), b.add(d0q, O._c(b, H * D + QW, "kbase"), name="kd0"), name="kin_b")
        kin = [b.load(a, b.add(kin_b, O._c(b, i, "ko_%d" % i), name="kii_%d" % i) if i else kin_b, type=I, name="kin%d" % i) for i in range(4)]
        vin = [b.load(a, b.add(kin_b, O._c(b, KVH * D + i, "vo_%d" % i), name="vii_%d" % i), type=I, name="vin%d" % i) for i in range(4)]
        if lay.get("qknorm"):
            kin = qknorm(kin, D, "kn")
        kr = rope(kin, "k")
        if not wide:
            knew = [b.f16_to_f32(x, name="kn%d" % i) for i, x in enumerate(k16)]
            vnew = [b.f16_to_f32(x, name="vn%d" % i) for i, x in enumerate(v16)]
        kvoff = b.mul(kvh, O._c(b, CAP * D, "CAPD"), name="kvoff")
        if seqb is not None:
            kvoff = b.add(kvoff, b.mul(seqb, O._c(b, lay["KVS"] // 2, "kvsh"), name="kvsb"), name="kvoffb")
        kb = b.add(b.add(kvoff, d0, name="kb0"), O._c(b, lay["KOFF"] // 2, "koff"), name="kb")
        vb = b.add(kb, O._c(b, (lay["VOFF"] - lay["KOFF"]) // 2, "voff"), name="vb")
        # the append: every slice of both q heads writes the same new row (a benign duplicate); readers of row q0
        # take the registers, never the cache
        rowq0 = b.mul(q0, O._c(b, D, "Dq0"), name="rowq0")
        kw, vw = b.add(kb, rowq0, name="kw"), b.add(vb, rowq0, name="vw")
        # one fp16 value live at a time: fp16 values take the narrow register file, which this program has 12 of
        for i in range(4):
            b.store_at(c, b.add(kw, O._c(b, i, "kwo%d" % i), name="kwi%d" % i) if i else kw,
                       b.f32_to_f16_rte(kr[i], name="k16_%d" % i), width="half")
            b.store_at(c, b.add(vw, O._c(b, i, "vwo%d" % i), name="vwi%d" % i) if i else vw,
                       b.f32_to_f16_rte(vin[i], name="v16_%d" % i), width="half")
        qin = [b.load(a, b.add(b.add(b.mul(h, O._c(b, D, "D_"), name="hDq"), d0q, name="qd0"), O._c(b, QW + i, "qo%d" % i), name="qi%d" % i) if (i or QW)
                      else b.add(b.mul(h, O._c(b, D, "D_0"), name="hDq0"), d0q, name="qi0"), type=I, name="qin%d" % i) for i in range(4)]
        if lay.get("qknorm"):
            qin = qknorm(qin, 0, "qn")
        qsc = O._cf(b, F32(D_.q_scale(D_.MILESTONE)), "qscale")
        qv = [b.f16_to_f32(b.f32_to_f16_rte(b.fmul(r, qsc, type=I, name="qs%d" % i), name="q16_%d" % i), name="q%d" % i)
              for i, r in enumerate(rope(qin, "q"))]
    else:
        qin = [b.load(a, b.add(b.add(b.mul(h, O._c(b, D, "D_"), name="hDq"), d0, name="qd0"), O._c(b, QW + i, "qo%d" % i), name="qi%d" % i) if (i or QW)
                      else b.add(b.mul(h, O._c(b, D, "D_0"), name="hDq0"), d0, name="qi0"), type=I, name="qin%d" % i) for i in range(4)]
        qsc = O._cf(b, F32(D_.q_scale(D_.MILESTONE)), "qscale")
        qv = [b.f16_to_f32(b.f32_to_f16_rte(b.fmul(r, qsc, type=I, name="qs%d" % i), name="q16_%d" % i), name="q%d" % i)
              for i, r in enumerate(rope(qin, "q"))]
        kin_b = b.add(b.mul(kvh, O._c(b, D, "D_k"), name="kvD"), b.add(d0, O._c(b, H * D + QW, "kbase"), name="kd0"), name="kin_b")
        kin = [b.load(a, b.add(kin_b, O._c(b, i, "ko_%d" % i), name="kii_%d" % i) if i else kin_b, type=I, name="kin%d" % i) for i in range(4)]
        vin = [b.load(a, b.add(kin_b, O._c(b, KVH * D + i, "vo_%d" % i), name="vii_%d" % i), type=I, name="vin%d" % i) for i in range(4)]
        k16 = [b.f32_to_f16_rte(r, name="k16_%d" % i) for i, r in enumerate(rope(kin, "k"))]
        v16 = [b.f32_to_f16_rte(vin[i], name="v16_%d" % i) for i in range(4)]
        if not wide:
            knew = [b.f16_to_f32(x, name="kn%d" % i) for i, x in enumerate(k16)]
            vnew = [b.f16_to_f32(x, name="vn%d" % i) for i, x in enumerate(v16)]
        kb = b.add(b.add(b.mul(kvh, O._c(b, CAP * D, "CAPD"), name="kvoff"), d0, name="kb0"), O._c(b, lay["KOFF"] // 2, "koff"), name="kb")
        vb = b.add(kb, O._c(b, (lay["VOFF"] - lay["KOFF"]) // 2, "voff"), name="vb")
        # the append: every slice of both q heads writes the same new row (a benign duplicate); readers of row q0
        # take the registers, never the cache
        rowq0 = b.mul(q0, O._c(b, D, "Dq0"), name="rowq0")
        kw, vw = b.add(kb, rowq0, name="kw"), b.add(vb, rowq0, name="vw")
        for i in range(4):
            b.store_at(c, b.add(kw, O._c(b, i, "kwo%d" % i), name="kwi%d" % i) if i else kw, k16[i], width="half")
            b.store_at(c, b.add(vw, O._c(b, i, "vwo%d" % i), name="vwi%d" % i) if i else vw, v16[i], width="half")
    negmax = O._cf(b, NEG_MAX, "negmax")
    fzero = O._cf(b, F32(0.0), "fzero")
    neg1 = O._cf(b, F32(-1.0), "neg1")
    K2 = O.emit_exp2_constants(b)
    lzero = O._cf(b, F32(0.0), "lzero")
    ozeros = [O._cf(b, F32(0.0), "ozero%d" % i) for i in range(4)]
    KB = lay.get("keyblock", 1)
    if m5:
        mn, ln, on = _emit_key_block_loop(b, ir, fn, O, lay, c, ksig, q0, kb, vb, qv, trips, m0=negmax, fzero=fzero,
                                          neg1=neg1, K2=K2, lzero=lzero, ozeros=ozeros, zero=zero, one=one, lgS=klgS, TC=TC)
    else:
        hdr, post = fn.block("keys"), fn.block("out")
        b.br(hdr)
        b.at(hdr)
        t = b.phi(zero, name="t")
        m = b.phi(negmax, type=ir.F32, name="m")
        l = b.phi(lzero, type=ir.F32, name="l")
        o = [b.phi(ozeros[i], type=ir.F32, name="o%d" % i) for i in range(4)]
        j = b.add(s, b.shl(t, O._c(b, lgS, "lgS3"), name="tS"), name="j")
        jr = b.csel(j, q0, q0, j, rel="gt", name="jr")
        row = b.mul(jr, O._c(b, D, "D2"), name="row")
        ki = b.add(kb, row, name="ki")
        vi = b.add(vb, row, name="vi")
        kl = [b.f16_to_f32(b.load(c, b.add(ki, O._c(b, i, "ko%d" % i), name="kii%d" % i) if i else ki, width="half", name="kh%d" % i),
                           name="kl%d" % i) for i in range(4)]
        vl = [b.f16_to_f32(b.load(c, b.add(vi, O._c(b, i, "vo%d" % i), name="vii%d" % i) if i else vi, width="half", name="vh%d" % i),
                           name="vl%d" % i) for i in range(4)]
        # key q0 (and every masked key, which reads row q0) takes the new row from registers
        if wide:
            # the WIDE form reads row q0 back from the cache: this very thread wrote those four dims of it (every
            # simdgroup appends the full row, lane l its dims 4l..4l+3), so program order makes the store visible, and
            # the fp16 row is exactly the registers' value - 8 registers and 8 selects fewer across the key loop
            kv_, vv = kl, vl
        else:
            kv_ = [b.csel(j, q0m1, knew[i], kl[i], rel="gt", name="k%d" % i) for i in range(4)]
            vv = [b.csel(j, q0m1, vnew[i], vl[i], rel="gt", name="v%d" % i) for i in range(4)]
        part = b.fmul(qv[0], kv_[0], type=ir.F32, name="pd0")
        for i in range(1, 4):
            part = b.fma(qv[i], kv_[i], part, name="pd%d" % i)
        part.type = ir.F32
        sc = TR.emit_butterfly(b, part, TR.ROW_BUTTERFLY_MASKS, operation="sum")
        sc = TR.emit_butterfly(b, sc, TR.COLUMN_BUTTERFLY_MASKS, operation="sum")
        sc = b.csel(j, q0, m, sc, rel="gt", name="sc")
        mn = b.fmax(m, sc, type=I, name="mn")
        nmn = b.fmul(mn, neg1, type=I, name="nmn")
        al = _exp2(lay, b, b.fadd(m, nmn, type=I, name="dm_"), K2, "al")
        p = _exp2(lay, b, b.fadd(sc, nmn, type=I, name="ds"), K2, "pe")
        p = b.csel(j, q0, fzero, p, rel="gt", name="p")
        ln = b.fadd(b.fmul(l, al, type=I, name="la"), p, type=ir.F32, name="ln")
        on = [b.fma(p, vv[i], b.fmul(o[i], al, type=I, name="oa%d" % i), name="on%d" % i) for i in range(4)]
        tn = b.add(t, one, name="tn")
        ir.Builder.phi_latch(t, tn)
        ir.Builder.phi_latch(m, mn)
        ir.Builder.phi_latch(l, ln)
        for i in range(4):
            ir.Builder.phi_latch(o[i], on[i])
        b.br_cond(b.cmp(tn, trips, "lt", cap=TC), hdr, post)
        b.at(post)
    if wide:
        # the slice's partial into the scratchpad: lane 0 stores m and l, every lane its four o dims
        pw0 = b.mul(s, O._c(b, PW, "PWs"), name="pw0")
        l0b, l0j = fn.block("pub_ml"), fn.block("pub_ml_done")
        b.br_cond(b.cmp(lane, 1, "lt", name="lane0p"), l0b, l0j)
        b.at(l0b)
        b.store_tg(b.fadd(mn, fzero, type=ir.F32, name="mpub"), pw0)
        b.store_tg(b.fadd(ln, fzero, type=ir.F32, name="lpub"), b.add(pw0, one, name="pw1"))
        b.br(l0j)
        b.at(l0j)
        ob_ = b.add(b.add(pw0, O._c(b, 4, "four_w"), name="pw4"), d0, name="opw")
        for i in range(4):
            b.store_tg(b.fadd(on[i], fzero, type=ir.F32, name="opub%d" % i), b.add(ob_, O._c(b, i, "opo%d" % i), name="opi%d" % i) if i else ob_)
        b.barrier("threadgroup")
        if lay.get("gqapair"):
            _emit_gqapair_merge(b, ir, c, tsp, s, lane, tpos, q0, lay, K2, neg1, fzero, zero, one, fn, bseq=seqb)
            b.ret()
            ir.verify(fn)
            return cc.compile_function(fn)
        if lay.get("tgsplit"):
            _emit_tg_split_merge(b, ir, TR, c, h, tsp, s, lane, tpos, q0, lay, K2, neg1, fzero, zero, one, fn)
            b.ret()
            ir.verify(fn)
            return cc.compile_function(fn)
        if lay.get("bfly_merge"):
            _emit_bfly_merge(b, ir, TR, c, h, s, lane, q0, lay, K2, neg1, fzero, zero, fn,
                             row_off=(b.mul(seqb, O._c(b, H * D, "HD_b"), name="orow") if seqb is not None else None))
            b.ret()
            ir.verify(fn)
            return cc.compile_function(fn)
        # simdgroup 0 merges; the others skip the region (cc's skip branch, MM 25.141.2)
        fn.skip_regions = True
        mgb, mgj = fn.block("wide_merge"), fn.block("wide_done")
        b.br_cond(b.cmp(s, 1, "lt", name="sg0m"), mgb, mgj)
        b.at(mgb)
        _emit_merge(b, ir, None, c, h, d0, q0, lay, K2, neg1, fzero, zero, tg=True)
        b.br(mgj)
        b.at(mgj)
        b.ret()
        ir.verify(fn)
        return cc.compile_function(fn)
    pb = b.add(b.mul(g, O._c(b, PW, "PW"), name="gPW"), O._c(b, lay["P"] // 4, "Pb"), name="pb")
    b.store_at(c, pb, mn)
    b.store_at(c, b.add(pb, one, name="pb1"), ln)
    ob = b.add(b.add(pb, O._c(b, 4, "four"), name="pb4"), d0, name="ob")
    for i in range(4):
        b.store_at(c, b.add(ob, O._c(b, i, "oo%d" % i), name="obi%d" % i) if i else ob, on[i])
    if lay.get("fused_merge"):
        # THE MERGE IN THE LAST THREADGROUP (MM 25.140.3): each slice's threadgroup publishes its partial, fences
        # device memory, and counts itself in its head's counter at binding 3 [CNT + h] (lane 0 adds 1). The
        # threadgroup that reads S - 1 is the last of its head: it merges the S partials exactly as
        # build_attn_merge does, writes attn fp16 at binding 3 [ATTN + h D + d], and resets the counter to 0.
        # The count is op10094, the UNIFORM device atomic - the one whose returned old value is measured (the
        # per-lane form's is not). Its address is binding 1 + an immediate, so the 16 heads share three words of
        # six 5-bit fields each (a head counts to 16 < 32); a threadgroup adds 1 << 5 (h mod 6) to word h / 6,
        # reads its own field from the old value, and the one that finds S - 1 is its head's last. It clears
        # only its field (and with the complement), so the word is zero again for the next token.
        b.barrier("fence_device")
        words = -(-lay["heads"] // 6)
        hq = b.shr(b.mul(h, O._c(b, 11, "eleven"), name="h11"), O._c(b, 6, "six"), name="hq")    # h / 6, h < 16
        sh = b.mul(b.sub(h, b.mul(hq, O._c(b, 6, "six2"), name="hq6"), name="hr"), O._c(b, 5, "five"), name="sh")
        eqs = [b.icmp(hq, O._c(b, w, "wq%d" % w), "eq", name="inw%d" % w) for w in range(words)]
        incs = [b.shl(eqs[w], sh, name="inc%d" % w) for w in range(words)]
        # op10094 runs once PER LANE (measured: +32 and 32 distinct old values), so lane 0 alone counts; its own
        # word's old value reaches every lane through the scratchpad. The address is binding 0 + operand 6, a BYTE
        # offset (measured: 4, 8, 16 hit words 1, 2, 4); the words sit 16 bytes apart.
        cl, cj = fn.block("cnt_lane0"), fn.block("cnt_done")
        b.br_cond(b.cmp(lane, 1, "lt", name="lane0"), cl, cj)
        b.at(cl)
        own = None
        for w in range(words):
            old_ = b.atomic_uniform("add", c, incs[w], name="cnt_old%d" % w, slot6=lay["CNTB"] + 16 * w)
            pick = b.mul(b.add(old_, O._c(b, 0, "oz%d" % w), name="oc%d" % w), eqs[w], name="pick%d" % w)
            own = pick if own is None else b.add(own, pick, name="own%d" % w)
        b.store_tg(own, O._c(b, 0, "tg0"))
        b.br(cj)
        b.at(cj)
        b.barrier("threadgroup")
        ownall = b.load_tg(O._c(b, 0, "tg0r"), name="ownall")
        field = getattr(b, "and")(b.shr(b.add(ownall, O._c(b, 0, "ow0"), name="owa"), sh, name="fs"), O._c(b, 31, "m31"), name="field")
        mg, jn = fn.block("merge_last"), fn.block("merge_done")
        b.br_cond(b.cmp(field, S - 2, "gt", name="is_last"), mg, jn)
        b.at(mg)
        b.barrier("fence_device")
        _emit_merge(b, ir, c, c, h, d0, q0, lay, K2, neg1, fzero, zero)
        ml, mj = fn.block("clr_lane0"), fn.block("clr_done")
        b.br_cond(b.cmp(lane, 1, "lt", name="lane0c"), ml, mj)
        b.at(ml)
        for w in range(words):
            m31 = b.shl(b.mul(eqs[w], O._c(b, 31, "f31_%d" % w), name="e31_%d" % w), sh, name="fm%d" % w)
            clear = getattr(b, "xor")(m31, O._c(b, 0xFFFFFFFF, "ones%d" % w), name="clr%d" % w)
            b.atomic_uniform("and", c, clear, name="cnt_clr%d" % w, slot6=lay["CNTB"] + 16 * w)
        b.br(mj)
        b.at(mj)
        b.br(jn)
        fn.blocks.remove(jn)
        fn.blocks.append(jn)                           # nested order: merge, clear, clear_done, merge_done (ret)
        b.at(jn)
    b.ret()
    ir.verify(fn)
    return cc.compile_function(fn)



def _emit_key_block_loop(b, ir, fn, O, lay, c, s, q0, kb, vb, qv, trips, m0, fzero, neg1, K2, lzero, ozeros, zero, one, lgS, TC):
    """The wide form's key loop with KB = lay["keyblock"] keys per trip (with_keyblock) and/or the native score sum
    (with_nsum), MM 25.144.5. Trip t of simdgroup s takes keys j_k = s + S (KB t + k), k = 0 .. KB - 1: their loads,
    dots and reductions are independent, so they overlap; then one running maximum over the block, ONE rescale exp2 for
    the block, and each key's p in key order. The order (attn_reference with the same flags): sc_k as before (a masked
    key j_k > q0 takes the block-start m), mn = fmax(m, sc_0, sc_1, ...), al = exp2_soft(m - mn), p_k =
    exp2_soft(sc_k - mn) (0 when masked), l = ((l al) + p_0) + p_1 ..., o = fma(p_k, v_k, ...) from o al in key order.
    KB = 1 without nsum is the unblocked loop's arithmetic exactly. Returns the final (m, l, o[4])."""
    S, D = lay.get("kslices", lay["slices"]), lay["head_dim"]
    KB = lay.get("keyblock", 1)
    lgKB = KB.bit_length() - 1
    I = ir.I32
    tripsB = b.shr(b.add(trips, O._c(b, KB - 1, "kbm1"), name="tkb"), O._c(b, lgKB, "lgKB"), name="tripsB") if KB > 1 else trips
    hdr, post = fn.block("keys"), fn.block("out")
    b.br(hdr)
    b.at(hdr)
    t = b.phi(zero, name="t")
    m = b.phi(m0, type=ir.F32, name="m")
    l = b.phi(lzero, type=ir.F32, name="l")
    o = [b.phi(ozeros[i], type=ir.F32, name="o%d" % i) for i in range(4)]
    j0 = b.add(s, b.shl(t, O._c(b, lgS + lgKB, "lgSKB"), name="tSK"), name="j0")
    js, kls, vls = [], [], []
    for k in range(KB):
        jk = b.add(j0, O._c(b, S * k, "jo%d" % k), name="jk%d" % k) if k else j0
        jr = b.csel(jk, q0, q0, jk, rel="gt", name="jr%d" % k)
        row = b.mul(jr, O._c(b, D, "Dr%d" % k), name="row%d" % k)
        ki = b.add(kb, row, name="ki%d" % k)
        vi = b.add(vb, row, name="vi%d" % k)
        if lay.get("kvvec"):
            # ONE 8-BYTE LOAD PER ROW PER LANE (with_kvvec, MM 25.144.5): the lane's four fp16 dims are two words at the
            # row's 8-byte-aligned address (stride 8 bytes for a two-component load, so the index is the half index >> 2);
            # dims 4l, 4l+1 are word 0's low and high halves, 4l+2, 4l+3 word 1's. The same fp16 values as the four
            # half loads, so the arithmetic after is unchanged.
            def halves(idx, tag):
                w = b.load_vec_at(c, b.shr(idx, O._c(b, 2, tag + "v2"), name=tag + "vi"), n=2, name=tag + "w")
                out = []
                for wi, word in enumerate(w):
                    out.append(b.f16_to_f32(b.low16(word, name="%sl%d" % (tag, wi)), name="%sf%d" % (tag, 2 * wi)))
                    out.append(b.f16_to_f32(b.low16(b.shr(word, O._c(b, 16, tag + "s16_%d" % wi), name="%sh%d" % (tag, wi)),
                                                    name="%shl%d" % (tag, wi)), name="%sf%d" % (tag, 2 * wi + 1)))
                return out
            kls.append(halves(ki, "kv%dk" % k))
            vls.append(halves(vi, "kv%dv" % k))
            js.append(jk)
            continue
        if lay.get("probe_nokv") or lay.get("probe_kv1"):
            # TIMING-ONLY PROBES (wrong values, MM 25.144.5): kv1 keeps one half load per row for all four dims (a quarter
            # of the load requests, the same arithmetic); nokv loads nothing (the arithmetic floor)
            if lay.get("probe_nokv"):
                kx = b.f16_to_f32(b.low16(b.add(jr, O._c(b, 0x3c00, "k1h%d" % k), name="kfake%d" % k), name="kf16_%d" % k),
                                  name="kx%d" % k)
                vx = kx
            else:
                kx = b.f16_to_f32(b.load(c, ki, width="half", name="kh1_%d" % k), name="kx%d" % k)
                vx = b.f16_to_f32(b.load(c, vi, width="half", name="vh1_%d" % k), name="vx%d" % k)
            kls.append([kx] * 4)
            vls.append([vx] * 4)
            js.append(jk)
            continue
        kls.append([b.f16_to_f32(b.load(c, b.add(ki, O._c(b, i, "ko%d_%d" % (k, i)), name="kii%d_%d" % (k, i)) if i else ki,
                                        width="half", name="kh%d_%d" % (k, i)), name="kl%d_%d" % (k, i)) for i in range(4)])
        vls.append([b.f16_to_f32(b.load(c, b.add(vi, O._c(b, i, "vo%d_%d" % (k, i)), name="vii%d_%d" % (k, i)) if i else vi,
                                        width="half", name="vh%d_%d" % (k, i)), name="vl%d_%d" % (k, i)) for i in range(4)])
        js.append(jk)
    parts = []
    for k in range(KB):
        part = b.fmul(qv[0], kls[k][0], type=ir.F32, name="pd0_%d" % k)
        for i in range(1, 4):
            part = b.fma(qv[i], kls[k][i], part, name="pd%d_%d" % (i, k))
        part.type = ir.F32
        parts.append(part)
    if lay.get("nsum"):
        # the native simd_sum (op16842): the adjacent pairwise tree = the xor butterfly 1, 2, 4, 8, 16 (measured on 64 of
        # 64 rounding-sensitive vectors, MM 25.141.16)
        scs = [b.machine(16842, parts[k], name="nsum%d" % k) for k in range(KB)]
        for x in scs:
            x.type = ir.F32
    else:
        # the measured row then column butterflies, one step across all KB keys at a time so their shuffles overlap
        from agxforge.g17 import tensorreduce as TR
        scs = list(parts)
        for mask in TR.ROW_BUTTERFLY_MASKS + TR.COLUMN_BUTTERFLY_MASKS:
            scs = [b.fadd(v, b.simd_shuffle_xor(v, mask, name="bf%d_%d" % (mask, k)), name="bs%d_%d" % (mask, k))
                   for k, v in enumerate(scs)]
    scs = [b.csel(js[k], q0, m, scs[k], rel="gt", name="sc%d" % k) for k in range(KB)]
    mn = m
    for k in range(KB):
        mn = b.fmax(mn, scs[k], type=I, name="mn%d" % k)
    nmn = b.fmul(mn, neg1, type=I, name="nmn")
    al = _exp2(lay, b, b.fadd(m, nmn, type=I, name="dm_"), K2, "al")
    ps = [b.csel(js[k], q0, fzero, _exp2(lay, b, b.fadd(scs[k], nmn, type=I, name="ds%d" % k), K2, "pe%d" % k),
                 rel="gt", name="p%d" % k) for k in range(KB)]
    ln = b.fmul(l, al, type=I, name="la")
    for k in range(KB):
        ln = b.fadd(ln, ps[k], type=ir.F32, name="ln%d" % k)
    on = [b.fmul(o[i], al, type=I, name="oa%d" % i) for i in range(4)]
    for k in range(KB):
        on = [b.fma(ps[k], vls[k][i], on[i], name="on%d_%d" % (k, i)) for i in range(4)]
    tn = b.add(t, one, name="tn")
    ir.Builder.phi_latch(t, tn)
    ir.Builder.phi_latch(m, mn)
    ir.Builder.phi_latch(l, ln)
    for i in range(4):
        ir.Builder.phi_latch(o[i], on[i])
    b.br_cond(b.cmp(tn, tripsB, "lt", cap=TC), hdr, post)
    b.at(post)
    return mn, ln, on


def _emit_tg_split_merge(b, ir, TR, c, h, p, sg, lane, tpos, q0, lay, K2, neg1, fzero, zero, one, fn):
    """with_tgsplit's merge (MM 25.144.5), after the slices' partials are in the scratchpad:
    A) every threadgroup (h, p) merges its 32 slices WITHOUT normalizing: lane l holds slice 32 p + l; M_p is the lane
       butterfly max of m, w = exp2_soft(m - M_p) (0 for a slice past q0), L_p and each dim O_p[d] are lane butterfly sums
       of l w and o w. It writes (M_p, L_p, O_p) to its device partial [P + (h N + p) PW].
    B) fence, then thread 0 counts the threadgroup in its head's counter (op10094, the fused merge's fields); the
       threadgroup that sees N - 1 is the head's last.
    C) the last merges the N partials: lane l reads partial min(l, N - 1); M is the lane butterfly max (lanes l >= N take
       NEG_MAX), w_l = exp2_soft(M_l - M) (0 for l >= N or 32 l > q0), L and each dim are lane butterfly sums of L_l w_l and
       O_l w_l, times the corrected reciprocal of L; attn as the butterfly merge writes it. Then it clears its field."""
    D, PW, N, H = lay["head_dim"], lay["PW"], lay["tgsplit"], lay["heads"]
    I = ir.I32
    base = b.mul(lane, O._c(b, PW, "tPW"), name="tbase")

    def ld(idx, name):
        return b.fadd(b.load_tg(idx, name=name + "_t"), fzero, type=I, name=name)
    m_ = ld(base, "tm")
    l_ = ld(b.add(base, one, name="tli"), "tl")
    ob = b.add(b.add(base, O._c(b, 4, "t4"), name="to4"), b.shl(sg, O._c(b, 2, "t2s"), name="tsg4"), name="tob")
    os_ = [ld(b.add(ob, O._c(b, i, "toi%d" % i), name="to%d_i" % i) if i else ob, "to%d" % i) for i in range(4)]
    m_.type = ir.F32
    Mp = TR.emit_butterfly(b, m_, TR.ROW_BUTTERFLY_MASKS, operation="max")
    Mp = TR.emit_butterfly(b, Mp, TR.COLUMN_BUTTERFLY_MASKS, operation="max")
    nMp = b.fmul(Mp, neg1, type=I, name="tnM")
    w = _exp2(lay, b, b.fadd(m_, nMp, type=I, name="tdM"), K2, "tw")
    sig_l = b.add(b.shl(p, O._c(b, 5, "t5"), name="tp32"), lane, name="tsig")
    w = b.csel(sig_l, q0, fzero, w, rel="gt", name="twz")
    Lp = TR.emit_butterfly(b, b.fmul(l_, w, type=ir.F32, name="tlw"), TR.ROW_BUTTERFLY_MASKS, operation="sum")
    Lp = TR.emit_butterfly(b, Lp, TR.COLUMN_BUTTERFLY_MASKS, operation="sum")
    Op = []
    for i in range(4):
        t = TR.emit_butterfly(b, b.fmul(os_[i], w, type=ir.F32, name="tow%d" % i), TR.ROW_BUTTERFLY_MASKS, operation="sum")
        Op.append(TR.emit_butterfly(b, t, TR.COLUMN_BUTTERFLY_MASKS, operation="sum"))
    # (M_p, L_p, O_p) to the device partial: lane 0 of each simdgroup its four dims, thread 0 also M_p and L_p
    pbase = b.add(b.mul(b.add(b.shl(h, O._c(b, N.bit_length() - 1, "tlgN"), name="thN"), p, name="thNp"), O._c(b, PW, "tPW2"),
                        name="tpw"), O._c(b, lay["P"] // 4, "tPb"), name="tpb")
    fn.skip_regions = True
    wb, wj = fn.block("tsplit_pub"), fn.block("tsplit_pub_done")
    b.br_cond(b.cmp(lane, 1, "lt", name="tlane0"), wb, wj)
    b.at(wb)
    odb = b.add(b.add(pbase, O._c(b, 4, "tpo4"), name="tpb4"), b.shl(sg, O._c(b, 2, "t2p"), name="tsgd"), name="tpod")
    for i in range(4):
        b.store_at(c, b.add(odb, O._c(b, i, "tpoi%d" % i), name="tpoi_%d" % i) if i else odb, Op[i])
    b.br(wj)
    b.at(wj)
    mb, mj = fn.block("tsplit_ml"), fn.block("tsplit_ml_done")
    b.br_cond(b.cmp(tpos, 1, "lt", name="tt0"), mb, mj)
    b.at(mb)
    b.store_at(c, pbase, Mp)
    b.store_at(c, b.add(pbase, one, name="tpb1"), Lp)
    b.br(mj)
    b.at(mj)
    # B) publish, then count
    b.barrier("fence_device")
    b.barrier("threadgroup")
    words = -(-H // 6)
    hq = b.shr(b.mul(h, O._c(b, 11, "televen"), name="th11"), O._c(b, 6, "tsix"), name="thq")    # h / 6, h < 16
    sh = b.mul(b.sub(h, b.mul(hq, O._c(b, 6, "tsix2"), name="thq6"), name="thr"), O._c(b, 5, "tfive"), name="tsh")
    eqs = [b.icmp(hq, O._c(b, w_, "twq%d" % w_), "eq", name="tinw%d" % w_) for w_ in range(words)]
    incs = [b.shl(eqs[w_], sh, name="tinc%d" % w_) for w_ in range(words)]
    cl, cj = fn.block("tsplit_cnt"), fn.block("tsplit_cnt_done")
    b.br_cond(b.cmp(tpos, 1, "lt", name="tt0c"), cl, cj)
    b.at(cl)
    own = None
    for w_ in range(words):
        old_ = b.atomic_uniform("add", c, incs[w_], name="tcnt_old%d" % w_, slot6=lay["CNTB"] + 16 * w_)
        pick = b.mul(b.add(old_, O._c(b, 0, "toz%d" % w_), name="toc%d" % w_), eqs[w_], name="tpick%d" % w_)
        own = pick if own is None else b.add(own, pick, name="town%d" % w_)
    b.store_tg(own, O._c(b, 0, "ttg0"))
    b.br(cj)
    b.at(cj)
    b.barrier("threadgroup")
    ownall = b.load_tg(O._c(b, 0, "ttg0r"), name="townall")
    field = getattr(b, "and")(b.shr(b.add(ownall, O._c(b, 0, "tow0"), name="towa"), sh, name="tfs"), O._c(b, 31, "tm31"),
                              name="tfield")
    lg, ljn = fn.block("tsplit_last"), fn.block("tsplit_last_done")
    b.br_cond(b.cmp(field, N - 2, "gt", name="tislast"), lg, ljn)
    b.at(lg)
    b.barrier("fence_device")
    # C) the last threadgroup of head h merges its N partials
    pl = b.csel(lane, O._c(b, N - 1, "tNm1"), O._c(b, N - 1, "tNm1b"), lane, rel="gt", name="tpl")
    hb = b.add(b.mul(b.add(b.shl(h, O._c(b, N.bit_length() - 1, "tlgN2"), name="thN2"), pl, name="thNl"),
                     O._c(b, PW, "tPW3"), name="thpw"), O._c(b, lay["P"] // 4, "tPb2"), name="thb")
    Ml = b.load(c, hb, type=I, name="tMl")
    Ll = b.load(c, b.add(hb, one, name="thb1"), type=I, name="tLl")
    od2 = b.add(b.add(hb, O._c(b, 4, "thb4"), name="thb4_"), b.shl(sg, O._c(b, 2, "t2m"), name="tsgm"), name="thod")
    Ol = [b.load(c, b.add(od2, O._c(b, i, "thoi%d" % i), name="thoi_%d" % i) if i else od2, type=I, name="tOl%d" % i)
          for i in range(4)]
    negmax = O._cf(b, NEG_MAX, "tnegmax")
    Mv = b.csel(lane, O._c(b, N - 1, "tNm1c"), negmax, Ml, rel="gt", name="tMv")
    Mv.type = ir.F32
    M = TR.emit_butterfly(b, Mv, TR.ROW_BUTTERFLY_MASKS, operation="max")
    M = TR.emit_butterfly(b, M, TR.COLUMN_BUTTERFLY_MASKS, operation="max")
    nM = b.fmul(M, neg1, type=I, name="tnMf")
    wl = _exp2(lay, b, b.fadd(Ml, nM, type=I, name="tdMf"), K2, "twf")
    wl = b.csel(lane, O._c(b, N - 1, "tNm1d"), fzero, wl, rel="gt", name="twfz")
    wl = b.csel(b.shl(lane, O._c(b, 5, "t5b"), name="tl32"), q0, fzero, wl, rel="gt", name="twfq")
    L = TR.emit_butterfly(b, b.fmul(Ll, wl, type=ir.F32, name="tLw"), TR.ROW_BUTTERFLY_MASKS, operation="sum")
    L = TR.emit_butterfly(b, L, TR.COLUMN_BUTTERFLY_MASKS, operation="sum")
    KR = O.emit_constants(b)
    rL = O.emit_rn(b, "recip", L, KR, "trL")
    ys = []
    for i in range(4):
        t = TR.emit_butterfly(b, b.fmul(Ol[i], wl, type=ir.F32, name="tOw%d" % i), TR.ROW_BUTTERFLY_MASKS, operation="sum")
        t = TR.emit_butterfly(b, t, TR.COLUMN_BUTTERFLY_MASKS, operation="sum")
        ys.append(b.fmul(t, rL, type=I, name="ty%d" % i))
    outb = b.add(b.mul(h, O._c(b, D, "tD"), name="thD"), b.shl(sg, O._c(b, 2, "t2o"), name="tsg4o"), name="toutb")
    ob2, oj2 = fn.block("tsplit_write"), fn.block("tsplit_write_done")
    b.br_cond(b.cmp(lane, 1, "lt", name="tlane0w"), ob2, oj2)
    b.at(ob2)
    for i in range(4):
        idx = b.add(outb, O._c(b, (lay["OUT_AT"] // 4 if lay.get("attn32") else lay["OUT_AT"] // 2) + i, "tout%d" % i),
                    name="touti%d" % i)
        if lay.get("attn32"):
            b.store_at(c, idx, b.f16_to_f32(b.f32_to_f16_rte(ys[i], name="tyh%d" % i), name="tyw%d" % i))
        else:
            b.store_at(c, idx, b.f32_to_f16_rte(ys[i], name="tyh%d" % i), width="half")
    b.br(oj2)
    b.at(oj2)
    clb, clj = fn.block("tsplit_clr"), fn.block("tsplit_clr_done")
    b.br_cond(b.cmp(tpos, 1, "lt", name="tt0z"), clb, clj)
    b.at(clb)
    for w_ in range(words):
        m31 = b.shl(b.mul(eqs[w_], O._c(b, 31, "tf31_%d" % w_), name="te31_%d" % w_), sh, name="tfm%d" % w_)
        clear = getattr(b, "xor")(m31, O._c(b, 0xFFFFFFFF, "tones%d" % w_), name="tclr%d" % w_)
        b.atomic_uniform("and", c, clear, name="tcnt_clr%d" % w_, slot6=lay["CNTB"] + 16 * w_)
    b.br(clj)
    b.at(clj)
    b.br(ljn)
    fn.blocks.remove(ljn)
    fn.blocks.append(ljn)                          # nested order: last, write, clear, then the join
    b.at(ljn)


PAIR_MASKS = (2, 4, 8, 16)     # xor masks that keep a lane's parity: they reduce over the lanes of one q head of the pair


def _emit_pair_bfly(b, ir, v, op, tag):
    """A lane butterfly over PAIR_MASKS (sum or max), in that order: lanes of equal parity end equal."""
    v.type = ir.F32
    for mask in PAIR_MASKS:
        peer = b.simd_shuffle_xor(v, mask, name="%s_x%d" % (tag, mask))
        v = b.fadd(v, peer, name="%s_s%d" % (tag, mask)) if op == "sum" else b.fmax(v, peer, name="%s_m%d" % (tag, mask))
        v.type = ir.F32
    return v


def _emit_gqapair_merge(b, ir, c, p, sg, lane, tpos, q0, lay, K2, neg1, fzero, zero, one, fn, bseq=None):
    """with_gqapair's merge (MM 25.144.5), after the 32 slice partials are in the scratchpad (simdgroup s at [s PW], head
    parity s & 1, slice 16 p + (s >> 1)):
    A) every threadgroup merges each head's 16 slices WITHOUT normalizing: lane l holds slot l; M is the pair butterfly
       max (masks 2, 4, 8, 16), w = exp2_soft(m - M) (0 for a slice past q0), L and each dim O[d] are pair butterfly sums
       of l w and o w. Lane a (0 or 1) then holds head a's result; it writes (M, L, O) to partial (a, p).
    B) fence; thread 0 counts the threadgroup in its KV head's counter field; the one that sees 1 is the pair's last.
    C) the last merges each head's two halves: lane l reads partial (l & 1, min(l >> 1, 1)); lanes with l >> 1 > 1 take
       NEG_MAX and weight 0, as does a half with 16 (l >> 1) > q0; M, L, O by pair butterflies; times the corrected
       reciprocal of L; lanes 0 and 1 write heads 2 kvh and 2 kvh + 1.
    All cross-threadgroup state is addressed from one register, pair_base (the pair's four partials), and one counter
    field indexed by kvh, so a batched form adds its sequence offset in one place."""
    D, PW, H = lay["head_dim"], lay["PW"], lay["heads"]
    I = ir.I32
    g = b.builtin("threadgroup_position_in_grid", name="pg")
    if bseq is not None:
        # BATCHED (M3): the local pair threadgroup g & (H - 1); pair index pidx = sequence (H / 2) + KV head addresses the
        # sequence's partials (after its attn rows, at PB) and its counter field
        kvh = b.shr(getattr(b, "and")(g, O._c(b, H - 1, "pgl"), name="pglv"), one, name="pkvh")
        pidx = b.add(b.mul(bseq, O._c(b, H // 2, "pH2"), name="pseq8"), kvh, name="pidx")
        pair_base = b.add(b.mul(pidx, O._c(b, 4 * PW, "p4PW"), name="pkv4"), O._c(b, lay["PB"] // 4, "pPb"), name="pair_base")
    else:
        kvh = b.shr(g, one, name="pkvh")
        pidx = kvh
        # the pair's four partials [(a 2 + p) PW], a = q head parity, p = key half
        pair_base = b.add(b.mul(kvh, O._c(b, 4 * PW, "p4PW"), name="pkv4"), O._c(b, lay["P"] // 4, "pPb"), name="pair_base")
    base = b.mul(lane, O._c(b, PW, "pPW"), name="pbase")

    def ld(idx, name):
        return b.fadd(b.load_tg(idx, name=name + "_t"), fzero, type=I, name=name)
    m_ = ld(base, "pm")
    l_ = ld(b.add(base, one, name="pli"), "pl")
    ob = b.add(b.add(base, O._c(b, 4, "p4"), name="po4"), b.shl(sg, O._c(b, 2, "p2s"), name="psg4"), name="pob")
    os_ = [ld(b.add(ob, O._c(b, i, "poi%d" % i), name="po%d_i" % i) if i else ob, "po%d" % i) for i in range(4)]
    M = _emit_pair_bfly(b, ir, m_, "max", "pM")
    nM = b.fmul(M, neg1, type=I, name="pnM")
    w = _exp2(lay, b, b.fadd(m_, nM, type=I, name="pdM"), K2, "pw")
    sig_l = b.add(b.shl(p, O._c(b, 4, "p16"), name="pp16"), b.shr(lane, one, name="plh"), name="psig")
    w = b.csel(sig_l, q0, fzero, w, rel="gt", name="pwz")
    Lp = _emit_pair_bfly(b, ir, b.fmul(l_, w, type=ir.F32, name="plw"), "sum", "pL")
    Op = [_emit_pair_bfly(b, ir, b.fmul(os_[i], w, type=ir.F32, name="pow%d" % i), "sum", "pO%d" % i) for i in range(4)]
    # lane a (a = 0, 1) writes head a's half-p partial: [pair_base + (a 2 + p) PW]
    par = b.add(pair_base, b.mul(b.add(b.shl(lane, one, name="pl2"), p, name="pl2p"), O._c(b, PW, "pPW2"), name="pparw"),
                name="ppar")
    fn.skip_regions = True
    wb, wj = fn.block("pair_pub"), fn.block("pair_pub_done")
    b.br_cond(b.cmp(lane, 2, "lt", name="plane01"), wb, wj)
    b.at(wb)
    odb = b.add(b.add(par, O._c(b, 4, "ppo4"), name="ppb4"), b.shl(sg, O._c(b, 2, "p2p"), name="psgd"), name="ppod")
    for i in range(4):
        b.store_at(c, b.add(odb, O._c(b, i, "ppoi%d" % i), name="ppoi_%d" % i) if i else odb, Op[i])
    b.br(wj)
    b.at(wj)
    mb, mj = fn.block("pair_ml"), fn.block("pair_ml_done")
    b.br_cond(b.cmp(tpos, 2, "lt", name="pt01"), mb, mj)          # simdgroup 0's lanes 0 and 1
    b.at(mb)
    b.store_at(c, par, M)
    b.store_at(c, b.add(par, one, name="ppar1"), Lp)
    b.br(mj)
    b.at(mj)
    # B) publish, then count this threadgroup in its KV head's field
    b.barrier("fence_device")
    b.barrier("threadgroup")
    if bseq is not None:
        # nb (H / 2) fields: pidx / 6 = (pidx 43) >> 8, exact for pidx < 64 (with_batch caps the pairs)
        words = -(-(lay["batch"] * (H // 2)) // 6)
        hq = b.shr(b.mul(pidx, O._c(b, 43, "p43"), name="ph43"), O._c(b, 8, "peight"), name="phq")
    else:
        words = -(-(H // 2) // 6)
        hq = b.shr(b.mul(kvh, O._c(b, 11, "peleven"), name="ph11"), O._c(b, 6, "psix"), name="phq")    # kvh / 6, kvh < 16
    sh = b.mul(b.sub(pidx, b.mul(hq, O._c(b, 6, "psix2"), name="phq6"), name="phr"), O._c(b, 5, "pfive"), name="psh")
    if bseq is None:
        eqs = [b.icmp(hq, O._c(b, w_, "pwq%d" % w_), "eq", name="pinw%d" % w_) for w_ in range(words)]
        incs = [b.shl(eqs[w_], sh, name="pinc%d" % w_) for w_ in range(words)]

    def eq_w(w_, tag=""):
        # batched: each word's compare formed at its use (nb 8 has 11 words; all of them live at once exhausted the
        # registers the uniform device atomic (op10094) can take as its destination)
        if bseq is None:
            return eqs[w_]
        return b.icmp(hq, O._c(b, w_, "pwq%s%d" % (tag, w_)), "eq", name="pinw%s%d" % (tag, w_))
    cl, cj = fn.block("pair_cnt"), fn.block("pair_cnt_done")
    b.br_cond(b.cmp(tpos, 1, "lt", name="pt0c"), cl, cj)
    b.at(cl)
    own = None
    for w_ in range(words):
        e_ = eq_w(w_)
        inc_ = incs[w_] if bseq is None else b.shl(e_, sh, name="pinc%d" % w_)
        old_ = b.atomic_uniform("add", c, inc_, name="pcnt_old%d" % w_, slot6=lay["CNTB"] + 16 * w_)
        pick = b.mul(b.add(old_, O._c(b, 0, "poz%d" % w_), name="poc%d" % w_), e_, name="ppick%d" % w_)
        own = pick if own is None else b.add(own, pick, name="pown%d" % w_)
    b.store_tg(own, O._c(b, 0, "ptg0"))
    b.br(cj)
    b.at(cj)
    b.barrier("threadgroup")
    ownall = b.load_tg(O._c(b, 0, "ptg0r"), name="pownall")
    field = getattr(b, "and")(b.shr(b.add(ownall, O._c(b, 0, "pow0"), name="powa"), sh, name="pfs"), O._c(b, 31, "pm31"),
                              name="pfield")
    lg, ljn = fn.block("pair_last"), fn.block("pair_last_done")
    b.br_cond(b.cmp(field, 0, "gt", name="pislast"), lg, ljn)
    b.at(lg)
    b.barrier("fence_device")
    # C) the pair's last threadgroup: lane l reads partial (l & 1, min(l >> 1, 1))
    lh = b.shr(lane, one, name="plh2")
    pl = b.csel(lh, one, one, lh, rel="gt", name="ppl")
    hb = b.add(pair_base, b.mul(b.add(b.shl(getattr(b, "and")(lane, one, name="pla"), one, name="pla2"), pl, name="plap"),
                                O._c(b, PW, "pPW3"), name="phpw"), name="phb")
    Ml = b.load(c, hb, type=I, name="pMl")
    Ll = b.load(c, b.add(hb, one, name="phb1"), type=I, name="pLl")
    od2 = b.add(b.add(hb, O._c(b, 4, "phb4"), name="phb4_"), b.shl(sg, O._c(b, 2, "p2m"), name="psgm"), name="phod")
    Ol = [b.load(c, b.add(od2, O._c(b, i, "phoi%d" % i), name="phoi_%d" % i) if i else od2, type=I, name="pOl%d" % i)
          for i in range(4)]
    negmax = O._cf(b, NEG_MAX, "pnegmax")
    Mv = b.csel(lh, one, negmax, Ml, rel="gt", name="pMv")
    Mf = _emit_pair_bfly(b, ir, Mv, "max", "pMf")
    nMf = b.fmul(Mf, neg1, type=I, name="pnMf")
    wl = _exp2(lay, b, b.fadd(Ml, nMf, type=I, name="pdMf"), K2, "pwf")
    wl = b.csel(lh, one, fzero, wl, rel="gt", name="pwfz")
    wl = b.csel(b.shl(lh, O._c(b, 4, "p16b"), name="plh16"), q0, fzero, wl, rel="gt", name="pwfq")
    L = _emit_pair_bfly(b, ir, b.fmul(Ll, wl, type=ir.F32, name="pLw"), "sum", "pLf")
    KR = O.emit_constants(b)
    rL = O.emit_rn(b, "recip", L, KR, "prL")
    ys = [b.fmul(_emit_pair_bfly(b, ir, b.fmul(Ol[i], wl, type=ir.F32, name="pOw%d" % i), "sum", "pOf%d" % i), rL, type=I,
                 name="py%d" % i) for i in range(4)]
    # lanes 0 and 1 write q heads 2 kvh and 2 kvh + 1, dims 4 sg .. 4 sg + 3
    outb = b.add(b.mul(b.add(b.shl(kvh, one, name="pkv2"), lane, name="phd"), O._c(b, D, "pD"), name="phD"),
                 b.shl(sg, O._c(b, 2, "p2o"), name="psg4o"), name="poutb")
    if bseq is not None:
        outb = b.add(outb, b.mul(bseq, O._c(b, H * D, "pHD"), name="porow"), name="poutbr")   # the sequence's attn row
    ob2, oj2 = fn.block("pair_write"), fn.block("pair_write_done")
    b.br_cond(b.cmp(lane, 2, "lt", name="plane01w"), ob2, oj2)
    b.at(ob2)
    for i in range(4):
        idx = b.add(outb, O._c(b, (lay["OUT_AT"] // 4 if lay.get("attn32") else lay["OUT_AT"] // 2) + i, "pout%d" % i),
                    name="pouti%d" % i)
        if lay.get("attn32"):
            b.store_at(c, idx, b.f16_to_f32(b.f32_to_f16_rte(ys[i], name="pyh%d" % i), name="pyw%d" % i))
        else:
            b.store_at(c, idx, b.f32_to_f16_rte(ys[i], name="pyh%d" % i), width="half")
    b.br(oj2)
    b.at(oj2)
    clb, clj = fn.block("pair_clr"), fn.block("pair_clr_done")
    b.br_cond(b.cmp(tpos, 1, "lt", name="pt0z"), clb, clj)
    b.at(clb)
    for w_ in range(words):
        m31 = b.shl(b.mul(eq_w(w_, "c"), O._c(b, 31, "pf31_%d" % w_), name="pe31_%d" % w_), sh, name="pfm%d" % w_)
        clear = getattr(b, "xor")(m31, O._c(b, 0xFFFFFFFF, "pones%d" % w_), name="pclr%d" % w_)
        b.atomic_uniform("and", c, clear, name="pcnt_clr%d" % w_, slot6=lay["CNTB"] + 16 * w_)
    b.br(clj)
    b.at(clj)
    b.br(ljn)
    fn.blocks.remove(ljn)
    fn.blocks.append(ljn)                          # nested order: last, write, clear, then the join
    b.at(ljn)


def _attn_gqapair_reference(lay, q16, K, V, q0):
    """with_gqapair's order exactly: slice partials as the base (blocked loop with keyblock / nsum), each threadgroup's
    per-head merge and the pair's final merge by lane butterflies over PAIR_MASKS (see _emit_gqapair_merge)."""
    import g17decodestep as D_
    from g17qmv import _fma32v
    from agxforge.g17 import tensorreduce as TR
    H, KVH, Dm, CAP = lay["heads"], lay["kv_heads"], lay["head_dim"], lay["cap"]
    S, KB = 32, lay.get("keyblock", 1)
    q0 = min(int(q0), CAP - 1)
    q = np.asarray(q16, F32)
    out = np.zeros((H, Dm), np.float16)
    lanes = np.arange(32)

    def pbf(vals, op):
        w = np.asarray(vals, F32).copy()
        for mask in PAIR_MASKS:
            w = (np.maximum(w, w[lanes ^ mask]) if op == "max" else (w + w[lanes ^ mask])).astype(F32)
        return w

    def slice_partial(h, sig):
        kv = h // 2
        qd = q[h].reshape(32, 4)
        trips = max((q0 + S - sig) >> 5, 1)
        m, l, o = NEG_MAX, F32(0.0), np.zeros((32, 4), F32)
        for tb in range(-(-trips // KB)):
            scs, vs, js = [], [], []
            for kk in range(KB):
                j = sig + S * (KB * tb + kk)
                jr = min(j, q0)
                k = np.asarray(K[kv, jr], F32).reshape(32, 4)
                vs.append(np.asarray(V[kv, jr], F32).reshape(32, 4))
                part = (qd[:, 0] * k[:, 0]).astype(F32)
                for i in range(1, 4):
                    part = _fma32v(qd[:, i], k[:, i], part)
                if lay.get("nsum"):
                    w = np.asarray(part, F32)
                    for mask in (1, 2, 4, 8, 16):
                        w = (w + w[lanes ^ mask]).astype(F32)
                    sc = F32(w[0])
                else:
                    r = TR.butterfly([float(x) for x in part], TR.ROW_BUTTERFLY_MASKS, "sum")
                    sc = F32(TR.butterfly(list(r), TR.COLUMN_BUTTERFLY_MASKS, "sum")[0])
                scs.append(m if j > q0 else sc)
                js.append(j)
            mn = m
            for sc in scs:
                mn = F32(max(mn, sc))
            nmn = F32(mn * F32(-1.0))
            al = F32(D_.exp2_soft(F32(m + nmn)))
            ps = [F32(0.0) if js[kk] > q0 else F32(D_.exp2_soft(F32(scs[kk] + nmn))) for kk in range(KB)]
            l = F32(l * al)
            for pp in ps:
                l = F32(l + pp)
            o = (o * al).astype(F32)
            for kk in range(KB):
                o = _fma32v(np.full((32, 4), ps[kk], F32), vs[kk], o)
            m = mn
        return m, l, o.reshape(-1)
    for kvh in range(KVH):
        halves = {}
        for p in range(2):
            slots = [slice_partial(2 * kvh + (s_ & 1), 16 * p + (s_ >> 1)) for s_ in range(32)]
            ms = np.array([x[0] for x in slots], F32)
            M = pbf(ms, "max")
            nM = (M * F32(-1.0)).astype(F32)
            w = np.array([F32(0.0) if 16 * p + (l_ >> 1) > q0 else F32(D_.exp2_soft(F32(ms[l_] + nM[l_]))) for l_ in range(32)], F32)
            Lp = pbf(np.array([F32(slots[l_][1] * w[l_]) for l_ in range(32)], F32), "sum")
            Op = np.array([pbf(np.array([F32(slots[l_][2][d] * w[l_]) for l_ in range(32)], F32), "sum") for d in range(Dm)], F32)
            for a in (0, 1):
                halves[(a, p)] = (M[a], Lp[a], Op[:, a])
        Ml = np.array([halves[(l_ & 1, min(l_ >> 1, 1))][0] for l_ in range(32)], F32)
        Mv = np.array([Ml[l_] if (l_ >> 1) <= 1 else NEG_MAX for l_ in range(32)], F32)
        Mf = pbf(Mv, "max")
        nMf = (Mf * F32(-1.0)).astype(F32)
        wl = np.array([F32(0.0) if ((l_ >> 1) > 1 or 16 * (l_ >> 1) > q0) else F32(D_.exp2_soft(F32(Ml[l_] + nMf[l_])))
                       for l_ in range(32)], F32)
        L = pbf(np.array([F32(halves[(l_ & 1, min(l_ >> 1, 1))][1] * wl[l_]) for l_ in range(32)], F32), "sum")
        for a in (0, 1):
            rL = F32(D_.recip(np.asarray([L[a]], F32))[0])
            ob = np.array([pbf(np.array([F32(halves[(l_ & 1, min(l_ >> 1, 1))][2][d] * wl[l_]) for l_ in range(32)], F32),
                               "sum")[a] for d in range(Dm)], F32)
            out[2 * kvh + a] = (ob * rL).astype(F32).astype(np.float16)
    return out

def _emit_bfly_merge(b, ir, TR, dst, h, sg, lane, q0, lay, K2, neg1, fzero, zero, fn, row_off=None):
    """with_bfly_merge's arithmetic: every simdgroup sg merges dims 4 sg .. 4 sg + 3; lane l reads slice l's m, l and those
    four o dims from the scratchpad [l PW]; M is the lane butterfly max, w = exp2_soft(m - M) (0 for a slice past q0),
    L and the four dims are lane butterfly sums of l w and o w, then times the corrected reciprocal of L."""
    D, S, PW = lay["head_dim"], lay["slices"], lay["PW"]
    I = ir.I32
    base = b.mul(lane, O._c(b, PW, "bPW"), name="bbase")
    def ld(idx, name):
        return b.fadd(b.load_tg(idx, name=name + "_t"), fzero, type=I, name=name)
    m_ = ld(base, "bm")
    l_ = ld(b.add(base, O._c(b, 1, "b1"), name="bli"), "bl")
    ob = b.add(b.add(base, O._c(b, 4, "b4"), name="bo4"), b.shl(sg, O._c(b, 2, "b2s"), name="sg4"), name="bob")
    os_ = [ld(b.add(ob, O._c(b, i, "boi%d" % i), name="bo%d_i" % i) if i else ob, "bo%d" % i) for i in range(4)]
    m_.type = ir.F32
    M = TR.emit_butterfly(b, m_, TR.ROW_BUTTERFLY_MASKS, operation="max")
    M = TR.emit_butterfly(b, M, TR.COLUMN_BUTTERFLY_MASKS, operation="max")
    nM = b.fmul(M, neg1, type=I, name="bnM")
    w = _exp2(lay, b, b.fadd(m_, nM, type=I, name="bdM"), K2, "bw")
    w = b.csel(lane, q0, fzero, w, rel="gt", name="bwz")
    lw = b.fmul(l_, w, type=ir.F32, name="blw")
    L = TR.emit_butterfly(b, lw, TR.ROW_BUTTERFLY_MASKS, operation="sum")
    L = TR.emit_butterfly(b, L, TR.COLUMN_BUTTERFLY_MASKS, operation="sum")
    KR = O.emit_constants(b)
    rL = O.emit_rn(b, "recip", L, KR, "brL")
    outb = b.add(b.mul(h, O._c(b, D, "bD"), name="bhD"), b.shl(sg, O._c(b, 2, "b2o"), name="bsg4"), name="boutb")
    if row_off is not None:
        outb = b.add(outb, row_off, name="boutbr")          # the batched sequence's attn row (with_batch)
    ys = []
    for i in range(4):
        t = b.fmul(os_[i], w, type=ir.F32, name="bow%d" % i)
        t = TR.emit_butterfly(b, t, TR.ROW_BUTTERFLY_MASKS, operation="sum")
        t = TR.emit_butterfly(b, t, TR.COLUMN_BUTTERFLY_MASKS, operation="sum")
        ys.append(b.fmul(t, rL, type=I, name="by%d" % i))
    # lane 0 of each simdgroup writes its four dims (every lane holds the same values)
    fn.skip_regions = True
    wb, wj = fn.block("bfly_write"), fn.block("bfly_done")
    b.br_cond(b.cmp(lane, 1, "lt", name="blane0"), wb, wj)
    b.at(wb)
    for i in range(4):
        idx = b.add(outb, O._c(b, (lay["OUT_AT"] // 4 if lay.get("attn32") else lay["OUT_AT"] // 2) + i, "bout%d" % i), name="bouti%d" % i)
        if lay.get("attn32"):
            b.store_at(dst, idx, b.f16_to_f32(b.f32_to_f16_rte(ys[i], name="byh%d" % i), name="byw%d" % i))
        else:
            b.store_at(dst, idx, b.f32_to_f16_rte(ys[i], name="byh%d" % i), width="half")
    b.br(wj)
    b.at(wj)


def _emit_merge(b, ir, src, dst, h, d0, q0, lay, K2, neg1, fzero, zero, tg=False):
    """build_attn_merge's arithmetic for head h: partials at src [P + (h S + s) PW] (or, tg, the scratchpad at
    [s PW]), attn fp16 at dst [OUT_AT]."""
    D, S, PW = lay["head_dim"], lay["slices"], lay["PW"]
    I = ir.I32
    if tg:
        hb = O._c(b, 0, "mhb0")
        def ld(idx, name):
            return b.fadd(b.load_tg(idx, name=name + "_t"), fzero, type=I, name=name)
    else:
        hb = b.add(b.mul(h, O._c(b, S * PW, "mSPW"), name="mhSPW"), O._c(b, lay["P"] // 4, "mPb"), name="mhb")
        def ld(idx, name):
            return b.load(src, idx, type=I, name=name)
    ms = [ld(b.add(hb, O._c(b, k * PW, "mmw%d" % k), name="mmi%d" % k) if k else hb, "mms%d" % k) for k in range(S)]
    ls = [ld(b.add(hb, O._c(b, k * PW + 1, "mlw%d" % k), name="mli%d" % k), "mls%d" % k) for k in range(S)]
    M = ms[0]
    for k in range(1, S):
        M = b.fmax(M, ms[k], type=I, name="mM%d" % k)
    nM = b.fmul(M, neg1, type=I, name="mnM")
    ws = []
    for k in range(S):
        w = _exp2(lay, b, b.fadd(ms[k], nM, type=I, name="mdM%d" % k), K2, "mw%d" % k)
        ws.append(b.csel(O._c(b, k, "msk%d" % k), q0, fzero, w, rel="gt", name="mwz%d" % k) if k else w)
    L = b.fmul(ls[0], ws[0], type=I, name="mL0")
    for k in range(1, S):
        L = b.fadd(L, b.fmul(ls[k], ws[k], type=I, name="mlw%d_" % k), type=I, name="mL%d" % k)
    KR = O.emit_constants(b)
    rL = O.emit_rn(b, "recip", L, KR, "mrL")
    ob = b.add(b.add(hb, O._c(b, 4, "mfour"), name="mhb4"), d0, name="mob")
    outb = b.add(b.mul(h, O._c(b, D, "mD3"), name="mhD3"), d0, name="moutb")
    for i in range(4):
        od = None
        for k in range(S):
            ok = ld(b.add(ob, O._c(b, k * PW + i, "mow%d_%d" % (k, i)), name="moi%d_%d" % (k, i)) if (k or i) else ob,
                    "mos%d_%d" % (k, i))
            wk = b.fmul(ok, ws[k], type=I, name="mowk_%d_%d" % (k, i))
            od = wk if od is None else b.fadd(od, wk, type=I, name="mod%d_%d" % (k, i))
        y = b.fmul(od, rL, type=I, name="my%d" % i)
        if lay.get("attn32"):
            b.store_at(dst, b.add(outb, O._c(b, lay["OUT_AT"] // 4 + i, "mout%d" % i), name="mouti%d" % i),
                       b.f16_to_f32(b.f32_to_f16_rte(y, name="myh%d" % i), name="myw%d" % i))
        else:
            b.store_at(dst, b.add(outb, O._c(b, lay["OUT_AT"] // 2 + i, "mout%d" % i), name="mouti%d" % i),
                       b.f32_to_f16_rte(y, name="myh%d" % i), width="half")


def attn_rope_reference(lay, qkv32, cos, sin, Kc, Vc, q0, partials=False):
    """RoPE and the append (g17decodestep.stage_rope_append's rounding), then attn_reference. Returns
    (out[, partials], K, V) with the new row in the caches."""
    import g17decodestep as D_
    H, KVH, Dm = lay["heads"], lay["kv_heads"], lay["head_dim"]
    q0c = min(int(q0), lay["cap"] - 1)
    qkv32 = np.asarray(qkv32, F32)
    q = qkv32[:H * Dm].reshape(H, Dm)
    k = qkv32[H * Dm:(H + KVH) * Dm].reshape(KVH, Dm)
    v = qkv32[(H + KVH) * Dm:(H + 2 * KVH) * Dm].reshape(KVH, Dm)
    q16 = D_.narrow(D_.fmul(D_.rope_rotate(q, cos, sin), D_.q_scale(D_.MILESTONE)))
    k16, v16 = D_.narrow(D_.rope_rotate(k, cos, sin)), D_.narrow(v)
    K = np.array(Kc, F32); V = np.array(Vc, F32)
    K[:, q0c] = k16; V[:, q0c] = v16
    r = attn_reference(lay, q16, K, V, q0, partials=partials)
    return (r if partials else (r,)) + (K, V)


def build_attn_merge(lay):
    """Dispatch 2: threadgroup h combines q head h's S partials (binding 2, [P + (h S + s) PW]) and writes
    attn fp16 [h D + d] at binding 3. Binding 1 holds the q0 word (an empty slice s > q0 is weighted 0)."""
    from agxforge.g17 import cc, ir
    D, S, PW, CAP = lay["head_dim"], lay["slices"], lay["PW"], lay["cap"]
    fn, b, a, bb, c = O._function()
    I = ir.I32
    lane, d0 = _lane_dims(b, ir)
    h = b.builtin("threadgroup_position_in_grid", name="head")
    q0 = b.load(a, O._c(b, lay["LEN"] // 4, "len_w"), type=I, name="q0raw")
    capm1 = O._c(b, CAP - 1, "capm1")
    q0 = b.csel(q0, capm1, capm1, q0, rel="gt", name="q0")
    neg1 = O._cf(b, F32(-1.0), "neg1")
    fzero = O._cf(b, F32(0.0), "fzero")
    K2 = O.emit_exp2_constants(b)
    hb = b.add(b.mul(h, O._c(b, S * PW, "SPW"), name="hSPW"), O._c(b, lay["P"] // 4, "Pb"), name="hb")
    ms = [b.load(bb, b.add(hb, O._c(b, k * PW, "mw%d" % k), name="mi%d" % k) if k else hb, type=I, name="ms%d" % k) for k in range(S)]
    ls = [b.load(bb, b.add(hb, O._c(b, k * PW + 1, "lw%d" % k), name="li%d" % k), type=I, name="ls%d" % k) for k in range(S)]
    M = ms[0]
    for k in range(1, S):
        M = b.fmax(M, ms[k], type=I, name="M%d" % k)
    nM = b.fmul(M, neg1, type=I, name="nM")
    ws = []
    for k in range(S):
        w = _exp2(lay, b, b.fadd(ms[k], nM, type=I, name="dM%d" % k), K2, "w%d" % k)
        ws.append(b.csel(O._c(b, k, "sk%d" % k), q0, fzero, w, rel="gt", name="wz%d" % k) if k else w)
    L = b.fmul(ls[0], ws[0], type=I, name="L0")
    for k in range(1, S):
        L = b.fadd(L, b.fmul(ls[k], ws[k], type=I, name="lw%d_" % k), type=I, name="L%d" % k)
    KR = O.emit_constants(b)
    rL = O.emit_rn(b, "recip", L, KR, "rL")
    ob = b.add(b.add(hb, O._c(b, 4, "four"), name="hb4"), d0, name="ob")
    outb = b.add(b.mul(h, O._c(b, D, "D3"), name="hD3"), d0, name="outb")
    for i in range(4):
        od = None
        for k in range(S):
            ok = b.load(bb, b.add(ob, O._c(b, k * PW + i, "ow%d_%d" % (k, i)), name="oi%d_%d" % (k, i)) if (k or i) else ob,
                        type=I, name="os%d_%d" % (k, i))
            wk = b.fmul(ok, ws[k], type=I, name="ow_%d_%d" % (k, i))
            od = wk if od is None else b.fadd(od, wk, type=I, name="od%d_%d" % (k, i))
        y = b.fmul(od, rL, type=I, name="y%d" % i)
        b.store_at(c, b.add(outb, O._c(b, lay["OUT"] // 2 + i, "out%d" % i), name="outi%d" % i) if (i or lay["OUT"]) else outb,
                   b.f32_to_f16_rte(y, name="yh%d" % i), width="half")
    b.ret()
    ir.verify(fn)
    return cc.compile_function(fn)


def _bfly_f32(v, masks, operation):
    """tensorreduce.butterfly_array (every lane of every row at once); callers check the results are finite."""
    from agxforge.g17 import tensorreduce as TR
    return TR.butterfly_array(v, masks, operation)


def _attn_wide_fast(lay, q, K, V, q0):
    """attn_reference's one-key-per-trip loop and its merge (plain or bfly_merge), vectorised over heads and slices:
    every (head, slice) runs exactly the scalar path's operations, trip by trip (a slice past its trip count keeps
    its state). Returns (out, PT), or None when a value is not finite, so the scalar path (and its overflow
    behaviour) decides those. test_g17simspeed checks bit-identity against the scalar path."""
    import g17decodestep as D_
    from agxforge.g17 import tensorreduce as TR
    from g17qmv import _fma32v
    H, KVH, Dm, CAP, S = lay["heads"], lay["kv_heads"], lay["head_dim"], lay["cap"], lay["slices"]
    sidx = np.arange(S)
    trips = np.maximum((q0 + S - sidx) >> (S.bit_length() - 1), 1)            # [S]
    kv = np.arange(H) // (H // KVH)
    qd = q.reshape(H, 1, 32, 4)
    m = np.full((H, S), NEG_MAX, F32)
    l = np.zeros((H, S), F32)
    o = np.zeros((H, S, 32, 4), F32)
    Kf, Vf = np.asarray(K, F32), np.asarray(V, F32)
    for t in range(int(trips.max())):
        act = (t < trips)[None, :]                                            # [1, S]
        j = sidx + S * t
        jr = np.minimum(j, q0)
        k = Kf[kv[:, None], jr[None, :]].reshape(H, S, 32, 4)
        v = Vf[kv[:, None], jr[None, :]].reshape(H, S, 32, 4)
        part = (np.broadcast_to(qd[..., 0], k.shape[:-1]) * k[..., 0]).astype(F32)
        for i in range(1, 4):
            part = _fma32v(np.broadcast_to(qd[..., i], k.shape[:-1]), k[..., i], part)
        if not np.all(np.isfinite(part)):
            return None
        sc = _bfly_f32(_bfly_f32(part, TR.ROW_BUTTERFLY_MASKS, "sum"), TR.COLUMN_BUTTERFLY_MASKS, "sum")[..., 0]
        if not np.all(np.isfinite(sc)):
            return None
        masked = (j > q0)[None, :]
        sc = np.where(masked, m, sc).astype(F32)
        mn = np.where(sc > m, sc, m).astype(F32)
        nmn = (mn * F32(-1.0)).astype(F32)
        al = D_.exp2_soft((m + nmn).astype(F32)).astype(F32)
        p = np.where(masked, F32(0.0), D_.exp2_soft((sc + nmn).astype(F32))).astype(F32)
        ln = ((l * al).astype(F32) + p).astype(F32)
        on = _fma32v(np.broadcast_to(p[..., None, None], o.shape), v, (o * al[..., None, None]).astype(F32))
        m = np.where(act, mn, m).astype(F32)
        l = np.where(act, ln, l).astype(F32)
        o = np.where(act[..., None, None], on, o).astype(F32)
    PT = np.zeros((H, S, lay["PW"]), F32)
    PT[:, :, 0], PT[:, :, 1], PT[:, :, 4:] = m, l, o.reshape(H, S, -1)
    live = (sidx <= q0)[None, :]
    if lay.get("bfly_merge"):
        assert S == 32, "the butterfly merge is 32 slices, one per lane"
        if not np.all(np.isfinite(m)) or not np.all(np.isfinite(l)) or not np.all(np.isfinite(o)):
            return None
        Mb = _bfly_f32(_bfly_f32(m, TR.ROW_BUTTERFLY_MASKS, "max"), TR.COLUMN_BUTTERFLY_MASKS, "max")[:, 0]   # [H]
        nM = (Mb * F32(-1.0)).astype(F32)
        ws = np.where(live, D_.exp2_soft((m + nM[:, None]).astype(F32)), F32(0.0)).astype(F32)            # [H, S]
        lw = (l * ws).astype(F32)
        Lb = _bfly_f32(_bfly_f32(lw, TR.ROW_BUTTERFLY_MASKS, "sum"), TR.COLUMN_BUTTERFLY_MASKS, "sum")[:, 0]
        terms = (o.reshape(H, S, Dm) * ws[..., None]).astype(F32).transpose(0, 2, 1)                     # [H, Dm, S]
        if not (np.all(np.isfinite(Lb)) and np.all(np.isfinite(lw)) and np.all(np.isfinite(terms))):
            return None
        ob = _bfly_f32(_bfly_f32(terms, TR.ROW_BUTTERFLY_MASKS, "sum"), TR.COLUMN_BUTTERFLY_MASKS, "sum")[..., 0]
        if not np.all(np.isfinite(ob)):
            return None
        rL = D_.recip(Lb.astype(F32)).astype(F32)
        out = (ob * rL[:, None]).astype(F32).astype(np.float16)
        return out, PT
    M = m[:, 0].copy()
    for k_ in range(1, S):
        M = np.where(m[:, k_] > M, m[:, k_], M).astype(F32)
    nM = (M * F32(-1.0)).astype(F32)
    ws = np.where(live, D_.exp2_soft((m + nM[:, None]).astype(F32)), F32(0.0)).astype(F32)
    L = (l[:, 0] * ws[:, 0]).astype(F32)
    for k_ in range(1, S):
        L = (L + (l[:, k_] * ws[:, k_]).astype(F32)).astype(F32)
    rL = D_.recip(L.astype(F32)).astype(F32)
    od = (o[:, 0] * ws[:, 0, None, None]).astype(F32)
    for k_ in range(1, S):
        od = (od + (o[:, k_] * ws[:, k_, None, None]).astype(F32)).astype(F32)
    out = (od * rL[:, None, None]).astype(F32).reshape(H, -1).astype(np.float16)
    return out, PT


def attn_reference(lay, q16, K, V, q0, partials=False, _scalar=False):
    """The kernel's order exactly. q16 [H, D], K and V [KVH, CAP, D] (fp16 values); returns fp16 [H, D].

    The one-key-per-trip form (no gqapair / tgsplit / key blocks / nsum) runs _attn_wide_fast, the same operations
    vectorised over heads and slices, unless a value is not finite; _scalar=True forces the scalar path (the
    equality oracle)."""
    import g17decodestep as D_
    from agxforge.g17 import tensorreduce as TR
    from g17qmv import _fma32v
    if lay.get("gqapair"):
        out = _attn_gqapair_reference(lay, q16, K, V, q0)
        return (out, np.zeros((lay["heads"], lay["slices"], lay["PW"]), F32)) if partials else out
    if lay.get("tgsplit"):
        out = _attn_tgsplit_reference(lay, q16, K, V, q0)
        return (out, np.zeros((lay["heads"], lay["slices"], lay["PW"]), F32)) if partials else out
    H, KVH, Dm, CAP, S = lay["heads"], lay["kv_heads"], lay["head_dim"], lay["cap"], lay["slices"]
    q0 = min(int(q0), CAP - 1)
    q = np.asarray(q16, F32)
    if not _scalar and not (lay.get("wide") and (lay.get("keyblock", 1) > 1 or lay.get("nsum"))):
        with np.errstate(over="ignore", invalid="ignore"):
            r = _attn_wide_fast(lay, q, K, V, q0)
        if r is not None:
            return r if partials else r[0]
    out = np.zeros((H, Dm), np.float16)
    PT = np.zeros((H, S, lay["PW"]), F32)
    for h in range(H):
        kv = h // (H // KVH)
        qd = q[h].reshape(32, 4)
        parts = []
        for s in range(S):
            trips = max((q0 + S - s) >> (S.bit_length() - 1), 1)
            m, l, o = NEG_MAX, F32(0.0), np.zeros((32, 4), F32)
            KB = lay.get("keyblock", 1)
            if lay.get("wide") and (KB > 1 or lay.get("nsum")):
                # the M5 key loop (_emit_key_block_loop): KB keys per trip, one rescale per block, keys in order
                for tb in range(-(-trips // KB)):
                    scs, vs, js = [], [], []
                    for kk in range(KB):
                        j = s + S * (KB * tb + kk)
                        jr = min(j, q0)
                        k = np.asarray(K[kv, jr], F32).reshape(32, 4)
                        vs.append(np.asarray(V[kv, jr], F32).reshape(32, 4))
                        part = (qd[:, 0] * k[:, 0]).astype(F32)
                        for i in range(1, 4):
                            part = _fma32v(qd[:, i], k[:, i], part)
                        if lay.get("nsum"):
                            w = np.asarray(part, F32)
                            for mask in (1, 2, 4, 8, 16):
                                w = (w + w[np.arange(32) ^ mask]).astype(F32)
                            sc = F32(w[0])
                        else:
                            sc = TR.butterfly([float(x) for x in part], TR.ROW_BUTTERFLY_MASKS, "sum")
                            sc = F32(TR.butterfly(list(sc), TR.COLUMN_BUTTERFLY_MASKS, "sum")[0])
                        scs.append(m if j > q0 else sc)
                        js.append(j)
                    mn = m
                    for sc in scs:
                        mn = F32(max(mn, sc))
                    nmn = F32(mn * F32(-1.0))
                    al = F32(D_.exp2_soft(F32(m + nmn)))
                    ps = [F32(0.0) if js[kk] > q0 else F32(D_.exp2_soft(F32(scs[kk] + nmn))) for kk in range(KB)]
                    l = F32(l * al)
                    for p in ps:
                        l = F32(l + p)
                    o = (o * al).astype(F32)
                    for kk in range(KB):
                        o = _fma32v(np.full((32, 4), ps[kk], F32), vs[kk], o)
                    m = mn
                trips = 0
            for t in range(trips):
                j = s + S * t
                jr = min(j, q0)
                k = np.asarray(K[kv, jr], F32).reshape(32, 4)
                v = np.asarray(V[kv, jr], F32).reshape(32, 4)
                part = (qd[:, 0] * k[:, 0]).astype(F32)
                for i in range(1, 4):
                    part = _fma32v(qd[:, i], k[:, i], part)
                sc = TR.butterfly([float(x) for x in part], TR.ROW_BUTTERFLY_MASKS, "sum")
                sc = TR.butterfly(list(sc), TR.COLUMN_BUTTERFLY_MASKS, "sum")
                sc = F32(sc[0])
                if j > q0:
                    sc = m
                mn = F32(max(m, sc))
                nmn = F32(mn * F32(-1.0))
                al = F32(D_.exp2_soft(F32(m + nmn)))
                p = F32(D_.exp2_soft(F32(sc + nmn)))
                if j > q0:
                    p = F32(0.0)
                l = F32(F32(l * al) + p)
                o = _fma32v(np.full((32, 4), p, F32), v, (o * al).astype(F32))
                m = mn
            parts.append((m, l, o))
            PT[h, s, 0], PT[h, s, 1], PT[h, s, 4:] = m, l, o.reshape(-1)
        if lay.get("bfly_merge"):
            # THE PARALLEL MERGE (with_bfly_merge): lane s of every simdgroup holds slice s; the maximum, L and each
            # output dim are the measured lane butterflies (row then column masks) over the 32 slices' terms
            assert S == 32, "the butterfly merge is 32 slices, one per lane"
            ms = [float(parts[k_][0]) for k_ in range(S)]
            Mb = TR.butterfly(ms, TR.ROW_BUTTERFLY_MASKS, "max"); Mb = F32(TR.butterfly(list(Mb), TR.COLUMN_BUTTERFLY_MASKS, "max")[0])
            nM = F32(Mb * F32(-1.0))
            ws = [F32(D_.exp2_soft(F32(parts[k_][0] + nM))) if k_ <= q0 else F32(0.0) for k_ in range(S)]
            lw = [float(F32(parts[k_][1] * ws[k_])) for k_ in range(S)]
            Lb = TR.butterfly(lw, TR.ROW_BUTTERFLY_MASKS, "sum"); Lb = F32(TR.butterfly(list(Lb), TR.COLUMN_BUTTERFLY_MASKS, "sum")[0])
            rL = F32(D_.recip(np.asarray([Lb], F32))[0])
            ob = np.zeros(Dm, F32)
            for d in range(Dm):
                terms = [float(F32(parts[k_][2].reshape(-1)[d] * ws[k_])) for k_ in range(S)]
                t1 = TR.butterfly(terms, TR.ROW_BUTTERFLY_MASKS, "sum"); ob[d] = F32(TR.butterfly(list(t1), TR.COLUMN_BUTTERFLY_MASKS, "sum")[0])
            out[h] = (ob * rL).astype(F32).astype(np.float16)
            continue
        M = parts[0][0]
        for k_ in range(1, S):
            M = F32(max(M, parts[k_][0]))
        nM = F32(M * F32(-1.0))
        ws = [F32(D_.exp2_soft(F32(parts[k_][0] + nM))) if k_ <= q0 else F32(0.0) for k_ in range(S)]
        L = F32(parts[0][1] * ws[0])
        for k_ in range(1, S):
            L = F32(L + F32(parts[k_][1] * ws[k_]))
        rL = F32(D_.recip(np.asarray([L], F32))[0])
        od = (parts[0][2] * ws[0]).astype(F32)
        for k_ in range(1, S):
            od = (od + (parts[k_][2] * ws[k_]).astype(F32)).astype(F32)
        out[h] = (od * rL).astype(F32).reshape(-1).astype(np.float16)
    return (out, PT) if partials else out


def _attn_tgsplit_reference(lay, q16, K, V, q0):
    """with_tgsplit's order exactly (MM 25.144.5): slice sigma = 32 p + s of 32 N walks keys sigma, sigma + 32 N, ... with
    the blocked key loop (keyblock, nsum); threadgroup p merges its 32 slices unnormalized (lane butterflies); the last
    merges the N partials (lane l holds partial min(l, N - 1); lanes l >= N and partials with 32 l > q0 weigh 0)."""
    import g17decodestep as D_
    from agxforge.g17 import tensorreduce as TR
    from g17qmv import _fma32v
    H, KVH, Dm, CAP, N = lay["heads"], lay["kv_heads"], lay["head_dim"], lay["cap"], lay["tgsplit"]
    SS, KB = 32 * N, lay.get("keyblock", 1)
    lgSS = SS.bit_length() - 1
    q0 = min(int(q0), CAP - 1)
    q = np.asarray(q16, F32)
    out = np.zeros((H, Dm), np.float16)

    def bf(vals, op):
        r = TR.butterfly([float(x) for x in vals], TR.ROW_BUTTERFLY_MASKS, op)
        return F32(TR.butterfly(list(r), TR.COLUMN_BUTTERFLY_MASKS, op)[0])
    for h in range(H):
        kv = h // (H // KVH)
        qd = q[h].reshape(32, 4)
        partials = []
        for p in range(N):
            parts = []
            for s_ in range(32):
                sig = 32 * p + s_
                trips = max((q0 + SS - sig) >> lgSS, 1)
                m, l, o = NEG_MAX, F32(0.0), np.zeros((32, 4), F32)
                for tb in range(-(-trips // KB)):
                    scs, vs, js = [], [], []
                    for kk in range(KB):
                        j = sig + SS * (KB * tb + kk)
                        jr = min(j, q0)
                        k = np.asarray(K[kv, jr], F32).reshape(32, 4)
                        vs.append(np.asarray(V[kv, jr], F32).reshape(32, 4))
                        part = (qd[:, 0] * k[:, 0]).astype(F32)
                        for i in range(1, 4):
                            part = _fma32v(qd[:, i], k[:, i], part)
                        if lay.get("nsum"):
                            w = np.asarray(part, F32)
                            for mask in (1, 2, 4, 8, 16):
                                w = (w + w[np.arange(32) ^ mask]).astype(F32)
                            sc = F32(w[0])
                        else:
                            sc = bf(part, "sum")
                        scs.append(m if j > q0 else sc)
                        js.append(j)
                    mn = m
                    for sc in scs:
                        mn = F32(max(mn, sc))
                    nmn = F32(mn * F32(-1.0))
                    al = F32(D_.exp2_soft(F32(m + nmn)))
                    ps = [F32(0.0) if js[kk] > q0 else F32(D_.exp2_soft(F32(scs[kk] + nmn))) for kk in range(KB)]
                    l = F32(l * al)
                    for pp in ps:
                        l = F32(l + pp)
                    o = (o * al).astype(F32)
                    for kk in range(KB):
                        o = _fma32v(np.full((32, 4), ps[kk], F32), vs[kk], o)
                    m = mn
                parts.append((m, l, o.reshape(-1)))
            Mp = bf([x[0] for x in parts], "max")
            nMp = F32(Mp * F32(-1.0))
            ws = [F32(0.0) if 32 * p + s_ > q0 else F32(D_.exp2_soft(F32(parts[s_][0] + nMp))) for s_ in range(32)]
            Lp = bf([F32(parts[s_][1] * ws[s_]) for s_ in range(32)], "sum")
            Op = np.array([bf([F32(parts[s_][2][d] * ws[s_]) for s_ in range(32)], "sum") for d in range(Dm)], F32)
            partials.append((Mp, Lp, Op))
        Ml = [partials[min(l_, N - 1)][0] for l_ in range(32)]
        M = bf([Ml[l_] if l_ < N else NEG_MAX for l_ in range(32)], "max")
        nM = F32(M * F32(-1.0))
        wl = [F32(0.0) if (l_ >= N or 32 * l_ > q0) else F32(D_.exp2_soft(F32(Ml[l_] + nM))) for l_ in range(32)]
        L = bf([F32(partials[min(l_, N - 1)][1] * wl[l_]) for l_ in range(32)], "sum")
        rL = F32(D_.recip(np.asarray([L], F32))[0])
        ob = np.array([bf([F32(partials[min(l_, N - 1)][2][d] * wl[l_]) for l_ in range(32)], "sum") for d in range(Dm)], F32)
        out[h] = (ob * rL).astype(F32).astype(np.float16)
    return out


def attn_float64(q16, K, V, q0, H, KVH):
    """The plain softmax in float64 (base 2): the tolerance yardstick, not the contract."""
    out = np.zeros((H, K.shape[2]))
    for h in range(H):
        kv = h // (H // KVH)
        k = K[kv, :q0 + 1].astype(np.float64); v = V[kv, :q0 + 1].astype(np.float64)
        s = k @ q16[h].astype(np.float64)
        p = np.exp2(s - s.max())
        out[h] = (p @ v) / p.sum()
    return out


def attn_io(lay, q16, K, V, q0, PT, want):
    """(split a, b, c, want_c), (merge a, b, c, want_c): the split writes the partials the merge reads."""
    qa = bytearray(lay["q_bytes"])
    O._place(qa, lay["Q"], np.asarray(q16, "<f2").reshape(-1))
    O._place(qa, lay["LEN"], np.asarray([q0], "<u4"))
    kv = bytearray(lay["kv_bytes"])
    O._place(kv, lay["K"], np.asarray(K, "<f2").reshape(-1))
    O._place(kv, lay["VOFF"], np.asarray(V, "<f2").reshape(-1))
    pt = bytearray(lay["partial_bytes"])
    wp = bytearray(pt)
    O._place(wp, lay["P"], np.asarray(PT, "<f4").reshape(-1))
    out = bytearray(lay["out_bytes"])
    wo = bytearray(out)
    O._place(wo, lay["OUT"], np.asarray(want, "<f2").reshape(-1))
    return (bytes(qa), bytes(kv), bytes(pt), bytes(wp)), (bytes(qa), bytes(wp), bytes(out), bytes(wo))


def case(lay, q0, seed=5, poison=True):
    """q, K, V as fp16 values; cache rows past q0 are POISONED with NaN (a kernel that reads them fails)."""
    rng = np.random.default_rng(seed)
    H, KVH, Dm, CAP = lay["heads"], lay["kv_heads"], lay["head_dim"], lay["cap"]
    import g17decodestep as D_
    qs = D_.q_scale(D_.MILESTONE)
    q = np.asarray(D_.narrow((rng.standard_normal((H, Dm)) * 2.0).astype(F32) * F32(qs)), F32).astype(np.float16)
    K = rng.standard_normal((KVH, CAP, Dm)).astype(np.float16)
    V = rng.standard_normal((KVH, CAP, Dm)).astype(np.float16)
    if poison and q0 + 1 < CAP:
        K[:, q0 + 1:] = np.float16(np.nan); V[:, q0 + 1:] = np.float16(np.nan)
    return q, K, V


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("check",))
    ap.add_argument("--q0", type=int, default=128)
    ap.add_argument("--cap", type=int, default=272)
    args = ap.parse_args(argv)
    lay = attn_layout(cap=args.cap)
    print("split: %d bytes, merge: %d bytes" % (len(build_attn_split(lay).code), len(build_attn_merge(lay).code)))


if __name__ == "__main__":
    main()
