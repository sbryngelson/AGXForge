#!/usr/bin/env python3
"""The batched-decode q4 projection on the tensor units (MM 25.166): y[m][n] = sum_k x[m][k] W[k][n] for up to 16
sequences m, W the model's affine q4 weights (group 64, bf16 scales and biases, the qmv / qmm layout), dequantized
ONCE per weight in registers and multiplied against every sequence by op5106.

It computes the transpose, Y^T = W^T X^T, because only B can advance through the key-block index register:
  A  W^T tiles (16 n x 16 k) held in register accumulator "W" (ir.tensor_matmul a_acc). A lane's eight slots are rows
     n = ra, ra + 8 and columns k = cb .. cb + 3 (ir.tensor_acc_position): two weight rows, four consecutive k - one
     packed u32 each in the unrepacked [N][K/8] layout. The lane dequantizes them as g17qmm.build_dequant does,
     fp32(q) * s + b, then rounds to fp16 (RNE) and back, so the MMA's 10-bit truncation of an fp32 A is exact: A is
     exactly the qmm path's W16.
  B  X^T fp16 [K][16] (row k holds the 16 sequences' x[k]), read from memory, advanced K_S rows a trip.
  C  Y^T tiles in register accumulator "Y", zeroed before the loop, accumulated every trip (c_inplace), stored at the
     end as y fp32 [16][N].
One simdgroup per threadgroup, threadgroup t owning n-tiles [t NT, (t + 1) NT); K_S 16-slices a trip (a K/(16 K_S)-trip
counted loop). The stated order is `reference`: per trip one MMA chain over its K_S slices (the pinned MMA model,
g17prefillmma.gemm_mma_v), then + Y.

    python3 tools/g17qsm.py check --role qkv [--nt 2 --ks 4]      # compile + g17emu bit-exact (CPU only)
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
ROLES = {"qkv": (4096, 2048), "wo": (2048, 2048), "w1": (8192, 2048), "w3": (8192, 2048), "w2": (2048, 8192)}
MB = 16                                # the batch rows of one MMA tile: sequences, padded with zeros


def _align(v, a=256):
    return -(-v // a) * a


def layout(N, K, nt=2, ks=4, sk=1, ahalf=False, xrows=False, offsets=None, mb=16, h16=False):
    """Offsets (bytes): buffer 1 the q4 weights W u32 [N][K/8] at 0, bf16 scales S and biases B [N][K/64] (read by
    scalar code only); buffer 2 X^T fp16 [K][16] at 0 (the MMA's B: a float x half stream body reads B from buffer 2);
    buffer 3 y fp32 [16][N] at 0."""
    if N % (16 * nt) or K % (64 * max(1, ks // 4)) or K % (16 * ks) or (16 * ks) % 64 not in (0, 16, 32):
        raise ValueError("qsm: N a multiple of 16 nt, K of 16 ks, and a trip's slices inside one scale group or whole groups")
    # mb 32 (MM 25.178): ONE n-tile a threadgroup and two bodies over the batch's two 16-row halves, both reading the
    # one dequantized W - its own Y and index register each (the second's starting 16 rows in), ks + 2 accumulator tiles
    if mb not in (16, 32) or (mb == 32 and (nt != 1 or not xrows or ks + 2 > 10)):
        raise ValueError("qsm: mb 16, or mb 32 with nt 1 and xrows (two batch-half bodies over one W)")
    if mb == 16 and (nt != 2 or nt * ks + nt > 10):
        # two bodies, one per n-tile (the memory-stream route takes two or more), each with its own index register
        # (cc.TENSOR_STREAM_INDEX_REGISTERS has two) and its own W and Y accumulators, within the ten groups
        raise ValueError("qsm: nt 2 (one body per n-tile, two index registers), nt * ks + nt <= 10 accumulator tiles")
    # SPLIT-K (sk > 1): threadgroup t = kg * G + ng over the head grid (head = the K group kg, slice = the n group ng),
    # each K group's partial y written at C + kg * 16 N * 4; the consumer sums the sk partials in ascending kg
    G = N // (16 * nt)
    if sk not in (1, 2, 4, 8) or K % (sk * 16 * ks) or (sk > 1 and (G > 256 or G & (G - 1))):
        raise ValueError("qsm: sk 1, 2, 4 or 8 dividing K into whole trips; with sk > 1 the n groups are a power of "
                         "two <= 256 (head_slices)")
    trips = K // (sk * 16 * ks)
    if trips > 255:
        raise ValueError("qsm: %d trips exceed the counted loop's 255" % trips)
    wbytes = N * (K // 8) * 4
    gbytes = N * (K // 64) * 2
    # offsets (MM 25.172): (W, S, B) byte offsets inside buffer 1 for a matrix that is part of a larger block - the
    # batched graph's FFN block holds w1 then w3 rows under one W, one S and one B array
    Wo, So, Bo = offsets if offsets is not None else (0, wbytes, wbytes + gbytes)
    # ahalf: W holds the fp32 dequant and the MMA body narrows it to half (op1016 RNE, eight a tile: a_converted_from),
    # where the scalar code otherwise rounds each weight to fp16 and back (two instructions) for an fp32 A - the same A
    # xrows (MM 25.171): B is x itself, fp16 [16][K] rows as the batched graph holds them, read under transB with the
    # row stride 2K bytes (cc's strideB beside an A from an accumulator), where the default reads X^T [K][16]
    # h16 (MM 25.196): the dequant on the fp16 pipe. Integer code packs two nibbles a word as the halves 1024 + 2^k q
    # (the 0x6400 exponent), one op798 each recovers q exactly (h 2^-k - 2^(10-k)) and a second, fma16(q, s, b) with
    # one rounding, writes the weight straight into a HALF A accumulator the MMA reads with no narrowing
    if h16 and ahalf:
        raise ValueError("qsm: h16 writes a half A itself; ahalf narrows an fp32 one")
    return dict(N=N, K=K, nt=nt, ks=ks, sk=sk, G=G, ahalf=bool(ahalf), h16=bool(h16), xrows=bool(xrows), trips=trips, groups=G * sk, W=Wo, S=So, B=Bo,
                mb=mb, a_bytes=_align(max(Wo + wbytes, So + gbytes, Bo + gbytes)), b_bytes=_align(K * mb * 2),
                c_bytes=_align(sk * mb * N * 4))


def build(lay):
    from agxforge.g17 import cc, ir
    N, K, NT, KS = lay["N"], lay["K"], lay["nt"], lay["ks"]
    SK, G = lay.get("sk", 1), lay.get("G", N // (16 * lay["nt"]))
    # TIMING ABLATIONS (never delivered; the output is wrong by construction): "noload" dequantizes a constant word
    # instead of loading the weights, "nodeq" writes the loaded word's bits to W without the dequant; "intonly" writes the
    # extracted nibble (shr, and) and "nofma" its fp32 conversion, splitting the dequant's cost (MM 25.196)
    ablate = set(lay.get("ablate", ()))
    KG = K // SK                                          # a K group's contraction length
    fn, b, a, bb, c = O._function()       # the default entry name: the bundle manifest looks the function up by it
    I, IM = ir.I32, ir.Imm
    C = lambda v, n: O._c(b, v, n)
    zero = O._cf(b, F32(0.0), "z")
    MBL = lay.get("mb", MB)
    halves = 2 if MBL == 32 else 1                          # batch-half bodies (mb 32) or n-tile bodies (mb 16)
    bodies = [(mi, 0) for mi in range(NT)] if halves == 1 else [(0, bh) for bh in range(2)]
    for j, (mi, bh) in enumerate(bodies):
        for sl in range(8):
            b.tensor_acc_write("Y%d" % j, sl, zero)
        # the second half's x rows start 16 rows in (16 K halves), past the immediate's measured domain: in the register
        b.tensor_index_init("k%d" % j, bh * 16 * K * 2)
    if lay.get("h16"):
        # (scale, offset) half pairs recovering q from 1024 + 2^k q: k 0 (1, -1024), k 4 (1/16, -64), k 6 (1/64, -16),
        # and the packing constants: the magic exponent in both halves and the two nibble masks
        hb = lambda v: int(np.float16(v).view(np.uint16))
        kq = {k: C(hb(2.0 ** -k) | hb(-(2.0 ** (10 - k))) << 16, "kq%d" % k) for k in (0, 4, 6)}
        magic = C(0x64006400, "magic")
        mask_a, mask_b, m16 = C(0x000F00F0, "mska"), C(0x03C000F0, "mskb"), C(0xFFFF, "m16")
    counter0 = b.const(0, name="trip0")
    t0 = b.builtin("threadgroup_position_in_grid", name="tg0") if SK > 1 else None
    kg0 = b.shr(t0, IM(G.bit_length() - 1), name="kg0") if SK > 1 else None
    kw_init = b.mul(kg0, C(KG // 8, "kgw"), name="kw_init") if SK > 1 else C(0, "kw_init")   # the K group's first word
    hdr, ex = fn.block("qsm_k"), fn.block("qsm_done")
    b.br(hdr)
    b.at(hdr)
    i = b.phi(counter0, name="trip")
    # the trip's first packed word along k (16 KS i / 8), carried as its own self-add: the latch check admits the
    # counter only in its increment and compare
    kw = b.phi(kw_init, name="kw")
    lane = b.builtin("thread_index_in_simdgroup", name="lane")
    t = b.builtin("threadgroup_position_in_grid", name="tg")
    ng = getattr(b, "and")(t, C(G - 1, "gmask"), name="ng") if SK > 1 else t
    # this lane's rows ra, ra + 8 and first column cb of every 16 x 16 tile (ir.tensor_acc_position)
    ra = b.add(b.shl(getattr(b, "and")(b.shr(lane, IM(4), name="l4"), IM(1), name="l4m"), IM(2), name="ra4"),
               getattr(b, "and")(b.shr(lane, IM(1), name="l1"), IM(3), name="l21"), name="ra")
    l3 = getattr(b, "and")(b.shr(lane, IM(3), name="l3s"), IM(1), name="l3")
    l0 = getattr(b, "and")(lane, IM(1), name="l0")
    sh0 = b.shl(l0, IM(4), name="sh0")                    # the first nibble's bit offset: 16 lane[0]
    nrow0 = b.add(b.mul(ng, C(16 * NT, "ntg"), name="tn"), ra, name="nrow0")    # global n of the lane's first row
    kw0 = b.add(kw, l3, name="kw0")                        # the lane's first word along k: kw + l3
    g = b.shr(kw, IM(3), name="g")                          # the trip's scale group: 8 kw / 64
    for mi in range(NT):
        for hf in range(2):
            n = b.add(nrow0, C(16 * mi + 8 * hf, "no%d%d" % (mi, hf)), name="n%d_%d" % (mi, hf)) if (mi or hf) else nrow0
            si = b.add(b.add(b.mul(n, C(K // 64, "grow%d%d" % (mi, hf)), name="ng%d_%d" % (mi, hf)), g,
                             name="nsg%d_%d" % (mi, hf)), C(lay["S"] // 2, "sb%d%d" % (mi, hf)), name="si%d_%d" % (mi, hf))
            s32 = b.shl(b.load(a, si, width="half", name="sh%d_%d" % (mi, hf)), IM(16), name="s%d_%d" % (mi, hf))
            b32 = b.shl(b.load(a, b.add(si, C((lay["B"] - lay["S"]) // 2, "bd%d%d" % (mi, hf)), name="bi%d_%d" % (mi, hf)),
                               width="half", name="bh%d_%d" % (mi, hf)), IM(16), name="b%d_%d" % (mi, hf))
            if lay.get("h16"):
                # bf16 -> fp32 as a 32-bit value (the shift of a half load types I16 by default), then fp16: exact in range
                s16h = b.f32_to_f16_rte(b.shl(s32.op.args[0], IM(16), type=I, name="sw%d_%d" % (mi, hf)), name="s16_%d_%d" % (mi, hf))
                b16h = b.f32_to_f16_rte(b.shl(b32.op.args[0], IM(16), type=I, name="bw%d_%d" % (mi, hf)), name="b16_%d_%d" % (mi, hf))
            wrow = b.add(b.mul(n, C(K // 8, "wr%d%d" % (mi, hf)), name="nw%d_%d" % (mi, hf)), kw0, name="wrow%d_%d" % (mi, hf))
            if lay["W"]:
                wrow = b.add(wrow, C(lay["W"] // 4, "wo%d%d" % (mi, hf)), name="wrowo%d_%d" % (mi, hf))
            for kk in range(KS):
                wi = b.add(wrow, IM(2 * kk), name="wi%d_%d_%d" % (mi, hf, kk)) if kk else wrow
                raw = (C(0x76543210 + kk, "wconst%d_%d_%d" % (mi, hf, kk)) if "noload" in ablate else
                       b.load(a, wi, type=I, name="w%d_%d_%d" % (mi, hf, kk)))
                if "nodeq" in ablate:
                    for j in range(4):
                        b.tensor_acc_write("W%d" % mi, 4 * hf + j, raw, tile=(0, kk))
                    continue
                word = b.shr(raw, sh0, name="ws%d_%d_%d" % (mi, hf, kk))
                if lay.get("h16"):
                    tg = "%d_%d_%d" % (mi, hf, kk)
                    tc = getattr(b, "and")(word, m16, name="tc" + tg)   # the lane's four nibbles alone
                    # pa: low 16 q0 (k 4), high q3 (k 0); pb: low 16 q1 (k 4), high 64 q2 (k 6)
                    pa = getattr(b, "or")(getattr(b, "and")(b.shl(tc, IM(4), name="t4" + tg), mask_a, name="pa0" + tg),
                                          magic, name="pa" + tg)
                    pb = getattr(b, "or")(getattr(b, "and")(getattr(b, "or")(tc, b.shl(tc, IM(14), name="t14" + tg),
                                                                              name="tb" + tg), mask_b, name="pb0" + tg),
                                          magic, name="pb" + tg)
                    for j, (p, half, k) in enumerate(((pa, "lo", 4), (pb, "lo", 4), (pb, "hi", 6), (pa, "hi", 0))):
                        qh = b.fma16(p, kq[k], kq[k], halves=(half, "lo", "hi"), name="qh%s_%d" % (tg, j))
                        b.tensor_acc_fma16("W%d" % mi, 4 * hf + j, qh, s16h, b16h, tile=(0, kk))
                    continue
                for j in range(4):
                    q = getattr(b, "and")(b.shr(word, IM(4 * j), name="q%d_%d_%d_%d" % (mi, hf, kk, j)) if j else word,
                                          IM(15), name="qm%d_%d_%d_%d" % (mi, hf, kk, j))
                    # fp32(q) * s is exact (a bf16 scale times q < 16), so one fma rounds exactly where fmul then fadd
                    # does: bit-identical to g17qmm.build_dequant, one instruction fewer
                    if "intonly" in ablate:                   # the extraction alone: the integer nibble to W
                        b.tensor_acc_write("W%d" % mi, 4 * hf + j, q, tile=(0, kk))
                        continue
                    w = b.u32_to_f32(q, name="qf%d_%d_%d_%d" % (mi, hf, kk, j))
                    if "nofma" in ablate:                     # extraction and conversion, no fma
                        b.tensor_acc_write("W%d" % mi, 4 * hf + j, w, tile=(0, kk))
                        continue
                    w = b.fma(w, s32, b32, name="wf%d_%d_%d_%d" % (mi, hf, kk, j))
                    if not lay.get("ahalf"):
                        w = b.f16_to_f32(b.f32_to_f16_rte(w, name="wh%d_%d_%d_%d" % (mi, hf, kk, j)),
                                         name="ww%d_%d_%d_%d" % (mi, hf, kk, j))
                    b.tensor_acc_write("W%d" % mi, 4 * hf + j, w, tile=(0, kk))
    for j, (mi, bh) in enumerate(bodies):
        xr = lay.get("xrows")
        grid = (dict(head_stride=(0, KG * (2 if xr else MB * 2), 0), head_slices=G, slice_stride=(0, 0, 0)) if SK > 1 else {})
        bkw = dict(transB=True, strideB=2 * K) if xr else {}
        akw = (dict(a_dtype="half", a_converted_from="float") if lay.get("ahalf") else dict(a_dtype="half") if lay.get("h16")
               else dict(a_dtype="float"))
        b.tensor_matmul(c, bb, c, M=16, N=MB, K=16 * KS, b_dtype="half", accumulate=True, **akw,
                        acc="Y%d" % j, a_acc="W%d" % mi, offsetA=0, offsetB=0, offsetB_register="k%d" % j,
                        offsetB_step=16 * KS * (2 if xr else MB * 2), **grid, **bkw)
    kwn = b.add(kw, IM(2 * KS), name="kw_next")             # before the counter's increment: the latch pattern
    nxt = b.add(i, IM(1), name="trip_next")
    ir.Builder.phi_latch(i, nxt)
    ir.Builder.phi_latch(kw, kwn)
    b.br_cond(b.cmp(nxt, lay["trips"], "lt", name="more"), hdr, ex)
    b.at(ex)
    lane = b.builtin("thread_index_in_simdgroup", name="f_lane")
    t = b.builtin("threadgroup_position_in_grid", name="f_tg")
    ra = b.add(b.shl(getattr(b, "and")(b.shr(lane, IM(4), name="fl4"), IM(1), name="fl4m"), IM(2), name="fra4"),
               getattr(b, "and")(b.shr(lane, IM(1), name="fl1"), IM(3), name="fl21"), name="fra")
    cb = b.add(b.shl(getattr(b, "and")(b.shr(lane, IM(3), name="fl3"), IM(1), name="fl3m"), IM(3), name="fcb8"),
               b.shl(getattr(b, "and")(lane, IM(1), name="fl0"), IM(2), name="fcb4"), name="fcb")
    fng = getattr(b, "and")(t, C(G - 1, "fgmask"), name="fng") if SK > 1 else t
    n0 = b.add(b.mul(fng, C(16 * NT, "fntg"), name="ftn"), ra, name="fn0")
    ob = b.add(b.mul(cb, C(N, "fN"), name="fcbN"), n0, name="fob")        # y[m][n], m = cb + j, n = n0 (+ 8, + 16 mi)
    if SK > 1:                                            # the K group's partial: C + kg * mb N
        ob = b.add(ob, b.mul(b.shr(t, IM(G.bit_length() - 1), name="fkg"), C(MBL * N, "fpart"), name="fkgo"), name="fobk")
    for j, (mi, bh) in enumerate(bodies):
        for sl in range(8):
            off = ((sl & 3) + 16 * bh) * N + 16 * mi + 8 * (sl >> 2)
            oi = b.add(ob, C(off, "fo%d_%d" % (j, sl)), name="foi%d_%d" % (j, sl)) if off else ob
            b.store_at(c, oi, b.tensor_acc_read("Y%d" % j, sl, name="fy%d_%d" % (j, sl)))
    b.ret()
    ir.verify(fn)
    return cc.compile_function(fn)


def reference(lay, x, q, s16, b16):
    """y fp32 [16][N] in the stated order: A = W^T (the qmm dequant, fp16 exact), B = X^T fp16, per trip one MMA chain
    over the trip's K_S slices, then + Y (zero first)."""
    import g17qmm as QM
    import g17prefillmma as MM
    N, K, KS, SK = lay["N"], lay["K"], lay["ks"], lay.get("sk", 1)
    Wt = QM.dequant_reference(q, s16, b16, 4).T.astype(F32)         # [N][K] fp16 values
    if lay.get("h16"):                                               # fp16(q s + b) with one rounding, s and b in fp16
        import g17emu as EMU
        import g17qmv as Q4
        sh = np.repeat(Q4._from_bf16(s16).astype(np.float16), 64, axis=1).astype(np.float64)
        bh = np.repeat(Q4._from_bf16(b16).astype(np.float16), 64, axis=1).astype(np.float64)
        Wt = EMU.fma16_exact(q.astype(np.float64), sh, bh).view(np.float16).astype(F32)
    Xt = np.asarray(x, np.float16).astype(F32).T                     # [K][16]
    parts = []
    for kg in range(SK):                                             # each K group from zero, its trips in order
        Y = np.zeros((N, lay.get("mb", MB)), F32)
        for tr in range(lay["trips"]):
            k0 = kg * (K // SK) + 16 * KS * tr
            Y = MM.gemm_mma_v(Wt[:, k0:k0 + 16 * KS], Xt[k0:k0 + 16 * KS], Y, truncate_a=True)
        parts.append(Y.T)
    return np.stack(parts)[0].copy() if SK == 1 else np.stack(parts)


def case(lay, seed=11, batch=None):
    """Random q4 weights (g17qmm.weights) and x fp16 [16][K]; rows past `batch` are zero (padding)."""
    import g17qmm as QM
    packed, s16, b16, q = QM.weights(lay["N"], lay["K"], 4, seed)
    rng = np.random.default_rng(seed + 1)
    mb = lay.get("mb", MB)
    batch = mb if batch is None else batch
    x = np.zeros((mb, lay["K"]), np.float16)
    x[:batch] = rng.standard_normal((batch, lay["K"])).astype(np.float16)
    return x, packed, s16, b16, q


def io(lay, x, packed, s16, b16):
    a = bytearray(lay["a_bytes"])
    for off, arr in ((lay["W"], np.asarray(packed, "<u4")), (lay["S"], np.asarray(s16, "<u2")), (lay["B"], np.asarray(b16, "<u2"))):
        raw = np.ascontiguousarray(arr).tobytes(); a[off:off + len(raw)] = raw
    xs = np.asarray(x, np.float16) if lay.get("xrows") else np.asarray(x, np.float16).T
    bb = bytearray(lay["b_bytes"]); bb[:K2(lay)] = np.ascontiguousarray(xs).tobytes()
    return bytes(a), bytes(bb), b"\x7f" * lay["c_bytes"]


def K2(lay):
    return lay["K"] * lay.get("mb", MB) * 2


def author(d, prog, a, bb, c, lay):
    """g17prefillattn_run.author's form: g17deliver.author with placeholder inputs (its manifest transport cannot size
    these buffers; the runner reads only the entry name from the manifest), the real inputs written over them."""
    import g17prefillattn_run as PR
    return PR.author(Path(d), prog, a, bb, c, lay)


def emulate(lay, prog, x, packed, s16, b16, work):
    """Run the program on g17emu through a bundle (the delivered form's own authoring) and return y fp32 [16][N]."""
    import g17deliver as D
    import g17emu as EMU
    a, bb, c = io(lay, x, packed, s16, b16)
    d = author(Path(work) / ("qsm_n%d_k%d_nt%d_ks%d" % (lay["N"], lay["K"], lay["nt"], lay["ks"])), prog, a, bb, c, lay)
    out, _m = EMU.run_bundle(d, lay["groups"] * 32, 32, 1, tier="wp")
    mb = lay.get("mb", MB)
    y = np.frombuffer(out, "<f4", lay.get("sk", 1) * mb * lay["N"])
    return y.reshape(mb, lay["N"]) if lay.get("sk", 1) == 1 else y.reshape(lay["sk"], mb, lay["N"])


def bench(lay, work, rounds=20, warm=5, batch=MB, seed=11):
    """On hardware: author the bundle, check y bit-exact against `reference` over a sentinel (the first dispatch), then
    time `rounds` dispatches with the runner (GPU time per dispatch, the first `warm` dropped). Returns a dict."""
    import json, statistics, subprocess
    import g17deliver as D
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    prog = build(lay)
    x, packed, s16, b16, q = case(lay, seed=seed, batch=batch)
    want = reference(lay, x, q, s16, b16)
    a, bb, c = io(lay, x, packed, s16, b16)
    d = author(work / ("qsm_n%d_k%d_nt%d_ks%d_sk%d" % (lay["N"], lay["K"], lay["nt"], lay["ks"], lay.get("sk", 1))),
               prog, a, bb, c, lay)
    run = work / "run"
    (run / "out").mkdir(parents=True, exist_ok=True)
    plan = run / "plan.json"
    plan.write_text(json.dumps(dict(configs=[dict(tag="qsm", bundle=str(d), threads=lay["groups"] * 32, group=32, base=1,
                                                  rounds=rounds + warm)])))
    r = subprocess.run([str(D.runner()), str(plan), str(run / "out")], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit("qsm run failed: " + r.stderr[-500:])
    got = np.frombuffer((run / "out" / "qsm.out").read_bytes(), "<f4", want.size).reshape(want.shape)
    diff = int((got.view("<u4") != want.view("<u4")).sum())
    us = [float(line.split()[3]) for line in r.stdout.splitlines() if line.startswith("time ")][warm:]
    return dict(N=lay["N"], K=lay["K"], nt=lay["nt"], ks=lay["ks"], batch=batch, code_bytes=len(prog.code),
                mismatched=diff, outputs=want.size, us_median=statistics.median(us), us_range=[min(us), max(us)])


def main(argv=None):
    import tempfile
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("check", "bench"))
    ap.add_argument("--work", default=None, help="bench: the bundle directory")
    ap.add_argument("--role", default="wo", choices=sorted(ROLES))
    ap.add_argument("--nt", type=int, default=2)
    ap.add_argument("--ks", type=int, default=4)
    ap.add_argument("--sk", type=int, default=1)
    ap.add_argument("--n", type=int, default=None, help="override N (a smaller emulated case)")
    ap.add_argument("--k", type=int, default=None, help="override K")
    a = ap.parse_args(argv)
    N, K = ROLES[a.role]
    lay = layout(a.n or N, a.k or K, a.nt, a.ks, a.sk)
    if a.cmd == "bench":
        import json
        if not a.work:
            raise SystemExit("bench needs --work DIR")
        print(json.dumps(bench(lay, a.work)))
        return 0
    prog = build(lay)
    x, packed, s16, b16, q = case(lay)
    want = reference(lay, x, q, s16, b16)
    with tempfile.TemporaryDirectory() as t:
        got = emulate(lay, prog, x, packed, s16, b16, t)
    diff = int((got.view("<u4") != want.view("<u4")).sum())
    print("qsm N %d K %d nt %d ks %d: %d bytes, %d of %d outputs differ" % (lay["N"], lay["K"], lay["nt"], lay["ks"],
                                                                       len(prog.code), diff, want.size))
    return 0 if diff == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
