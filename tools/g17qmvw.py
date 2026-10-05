#!/usr/bin/env python3
"""The wide multi-vector q4 projection (MM 25.207): mlx-lm's qmv_wide shape for a small batch of nb vectors - the
speculative check of k <= 4 drafts. One weight stream, each weight dequantized ONCE and met by every vector's plain
fp32 dot.

Shape: threadgroups of 64 (2 simdgroups); simdgroup s of threadgroup t owns rows 8 t + 4 s .. + 3; lane l is row
l >> 3 of those, K-lane kl = l & 7. The K-lane walks whole 64-weight groups g = kl, kl + 8, ... Weights are the qmv layout:
W u32 [N][K/8] (8 nibbles a word, element 8 w + j at bits 4 j), bf16 scales S and biases B [N][K/64], all in buffer 1 at
byte offsets (W, S, B). x fp32 [nb][K] in buffer 2; y fp32 [nb][N] in buffer 3.

Order (reference): per row and K-lane, groups ascending, words 0..7, nibbles 0..7: w = f32(q s + b) (q s is exact, one
rounding), then for each vector v: acc_v = fma32(w, x_v, acc_v) from 0. Then the 8 K-lanes are summed by xor butterflies
over lane masks 1, 2, 4: ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7)).

    python3 tools/g17qmvw.py check     the emulated program against the reference, nb 1, 2, 4
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


def _align(v, a=256):
    return -(-v // a) * a


def layout(N, K, nb=1, offsets=None, form="kl8"):
    if form == "r4":
        return _layout_r4(N, K, nb, offsets)
    if N % 8 or K % 512 or nb not in (1, 2, 3, 4):
        raise ValueError("qmvw: N a multiple of 8, K of 512 (8 K-lanes of 64-weight groups), nb 1..4")
    trips = K // 512
    if trips > 255:
        raise ValueError("qmvw: the counted loop's 255 trips")
    wbytes, gbytes = N * (K // 8) * 4, N * (K // 64) * 2
    Wo, So, Bo = offsets if offsets is not None else (0, wbytes, wbytes + gbytes)
    return dict(op="qmvw", N=N, K=K, nb=nb, trips=trips, W=Wo, S=So, B=Bo, groups=N // 8, threads_per_group=64,
                a_bytes=_align(max(Wo + wbytes, So + gbytes, Bo + gbytes)), b_bytes=_align(4 * K * nb),
                c_bytes=_align(4 * N * nb))


def _layout_r4(N, K, nb, offsets):
    """THE ROWS FORM (MM 25.207): MLX's qmv_fast shape. Simdgroup s of threadgroup t (64 threads) owns rows
    8 t + 4 s .. + 3; lane l takes packed word 32 i + l of each of them in trip i (8 weights, group 4 i + l >> 3) and the
    trip's 8 x values of every vector ONCE, shared by the 4 rows - the kl8 form loads each vector's x per row."""
    if N % 8 or K % 256 or nb not in (1, 2, 3, 4):
        raise ValueError("qmvw r4: N a multiple of 8, K of 256 (32 lanes of one word), nb 1..4")
    trips = K // 256
    if trips > 255:
        raise ValueError("qmvw: the counted loop's 255 trips")
    wbytes, gbytes = N * (K // 8) * 4, N * (K // 64) * 2
    Wo, So, Bo = offsets if offsets is not None else (0, wbytes, wbytes + gbytes)
    return dict(op="qmvw", form="r4", N=N, K=K, nb=nb, trips=trips, W=Wo, S=So, B=Bo, groups=N // 8, threads_per_group=64,
                a_bytes=_align(max(Wo + wbytes, So + gbytes, Bo + gbytes)), b_bytes=_align(4 * K * nb),
                c_bytes=_align(4 * N * nb))


def _build_r4(lay):
    from agxforge.g17 import cc, ir
    N, K, NB, T, R = lay["N"], lay["K"], lay["nb"], lay["trips"], 4
    fn, b, a, bb, c = O._function()
    I, IM = ir.I32, ir.Imm
    C = lambda v, n: O._c(b, v, n)
    tpos = b.builtin("thread_position_in_threadgroup", name="tpos")
    t = b.builtin("threadgroup_position_in_grid", name="tg")
    lane = getattr(b, "and")(tpos, IM(31), name="lane")
    # row0 = 8 t + 4 (tpos >> 5)
    row0 = b.add(b.shl(t, IM(3), name="t8"), b.shl(b.shr(tpos, IM(5), name="sg"), IM(2), name="sg4"), name="row0")
    w00 = b.add(b.add(b.mul(row0, C(K // 8, "wrow"), name="rw"), lane, name="rwl"), C(lay["W"] // 4, "wb"), name="w00")
    s00 = b.add(b.add(b.mul(row0, C(K // 64, "grow"), name="rg"), b.shr(lane, IM(3), name="l8"), name="rgl"),
                C(lay["S"] // 2, "sb"), name="s00")
    w0 = [w00] + [b.add(w00, C(r * (K // 8), "wr%d" % r), name="w0_%d" % r) for r in range(1, R)]
    s0 = [s00] + [b.add(s00, C(r * (K // 64), "sr%d" % r), name="s0_%d" % r) for r in range(1, R)]
    bdiff = C((lay["B"] - lay["S"]) // 2, "bdiff")
    x0 = b.shl(lane, IM(1), name="x0")                      # vec4 units: this lane's 8 x of vector 0 (2 lane)
    zeros = [[O._cf(b, F32(0.0), "z%d_%d" % (r, v)) for v in range(NB)] for r in range(R)]
    counter0 = b.const(0, name="trip0")
    hdr, ex = fn.block("qmvw_k"), fn.block("qmvw_done")
    b.br(hdr)
    b.at(hdr)
    i = b.phi(counter0, name="trip")
    wi = [b.phi(w0[r], name="wi%d" % r) for r in range(R)]
    si = [b.phi(s0[r], name="si%d" % r) for r in range(R)]
    xi = b.phi(x0, name="xi")
    acc = [[b.phi(zeros[r][v], type=ir.F32, name="acc%d_%d" % (r, v)) for v in range(NB)] for r in range(R)]
    xs = []
    for v in range(NB):
        xq = b.add(xi, C(v * K // 4, "xo%d" % v), name="xq%d" % v) if v else xi
        xs.append(b.load_vec_at(bb, xq, n=4, name="xa%d" % v) + b.load_vec_at(bb, xq, n=4, name="xb%d" % v, offset_bytes=16))
    cur = [list(row) for row in acc]
    for r in range(R):
        word = b.load(a, wi[r], name="wd%d" % r)
        s32 = b.shl(b.load(a, si[r], width="half", name="sh%d" % r), IM(16), type=I, name="s32_%d" % r)
        b32 = b.shl(b.load(a, b.add(si[r], bdiff, name="bi%d" % r), width="half", name="bh%d" % r), IM(16), type=I,
                    name="b32_%d" % r)
        for j in range(8):
            q = getattr(b, "and")(b.shr(word, IM(4 * j), name="ws%d_%d" % (r, j)) if j else word, IM(15), name="q%d_%d" % (r, j))
            w = b.fma(b.u32_to_f32(q, name="qf%d_%d" % (r, j)), s32, b32, name="w%d_%d" % (r, j))
            for v in range(NB):
                cur[r][v] = b.fma(w, xs[v][j], cur[r][v], name="a%d_%d_%d" % (r, v, j))
    nxt = b.add(i, IM(1), name="trip_next")
    for r in range(R):
        ir.Builder.phi_latch(wi[r], b.add(wi[r], IM(32), name="wi%d_next" % r))
        ir.Builder.phi_latch(si[r], b.add(si[r], IM(4), name="si%d_next" % r))
    ir.Builder.phi_latch(xi, b.add(xi, C(64, "xstep"), name="xi_next"))
    ir.Builder.phi_latch(i, nxt)
    for r in range(R):
        for v in range(NB):
            ir.Builder.phi_latch(acc[r][v], cur[r][v])
    b.br_cond(b.cmp(nxt, T, "lt", name="more"), hdr, ex)
    b.at(ex)
    outs = {}
    for r in range(R):
        for v in range(NB):
            y = cur[r][v]
            y.type = ir.F32
            for m in MASKS_R4:
                y = b.fadd(y, b.simd_shuffle_xor(y, m, name="sh%d_%d_%d" % (r, v, m)), type=ir.F32, name="r%d_%d_%d" % (r, v, m))
            outs[r, v] = y
    fn.skip_regions = True
    wb, wj = fn.block("qmvw_write"), fn.block("qmvw_wdone")
    b.br_cond(b.cmp(lane, 1, "lt", name="l0"), wb, wj)
    b.at(wb)
    for r in range(R):
        for v in range(NB):
            b.store_at(c, b.add(row0, C(v * N + r, "yo%d_%d" % (r, v)), name="yi%d_%d" % (r, v)) if (r or v) else row0, outs[r, v])
    b.br(wj)
    b.at(wj)
    b.ret()
    ir.verify(fn)
    return cc.compile_function(fn)


MASKS_R4 = (1, 2, 4, 8, 16)


def _reference_r4(lay, q, s16, b16, x):
    import g17qmv as Q
    N, K, NB = lay["N"], lay["K"], lay["nb"]
    s = Q._from_bf16(s16).astype(np.float64)
    bi = Q._from_bf16(b16).astype(np.float64)
    acc = np.zeros((NB, N, 32))                              # [vector][row][lane]
    lanes = np.arange(32)
    for tr in range(lay["trips"]):
        g = 4 * tr + (lanes >> 3)                            # [32]
        for j in range(8):
            k = 8 * (32 * tr + lanes) + j                    # [32]
            w = _fma32(q[:, k].astype(np.float64), s[:, g], bi[:, g])       # [N][32]
            for v in range(NB):
                acc[v] = _fma32(w, x[v, k].astype(np.float32).astype(np.float64)[None, :], acc[v])
    y = acc
    for m in MASKS_R4:
        y = (y.astype(np.float32) + y[:, :, lanes ^ m].astype(np.float32)).astype(np.float64)
    return y[:, :, 0].astype(np.float32)


def build(lay):
    if lay.get("form") == "r4":
        return _build_r4(lay)
    from agxforge.g17 import cc, ir
    N, K, NB, T = lay["N"], lay["K"], lay["nb"], lay["trips"]
    fn, b, a, bb, c = O._function()
    I, IM = ir.I32, ir.Imm
    C = lambda v, n: O._c(b, v, n)
    tpos = b.builtin("thread_position_in_threadgroup", name="tpos")
    t = b.builtin("threadgroup_position_in_grid", name="tg")
    lane = getattr(b, "and")(tpos, IM(31), name="lane")
    kl = getattr(b, "and")(lane, IM(7), name="kl")
    # row = 8 t + 4 (tpos >> 5) + (lane >> 3) = 8 t + (tpos >> 3): the simdgroup's 4 rows and the lane's row in them
    row = b.add(b.shl(t, IM(3), name="t8"), b.shr(tpos, IM(3), name="tp8"), name="row")
    # per-trip bases, each carried as its own self-add (the latch admits the counter only in its increment and compare)
    w0 = b.add(b.add(b.mul(row, C(K // 8, "wrow"), name="rw"), b.shl(kl, IM(3), name="kl8"), name="rwk"),
               C(lay["W"] // 4, "wb"), name="w0")
    s0 = b.add(b.add(b.mul(row, C(K // 64, "grow"), name="rg"), kl, name="rgk"), C(lay["S"] // 2, "sb"), name="s0")
    bdiff = C((lay["B"] - lay["S"]) // 2, "bdiff")
    x0 = b.shl(kl, IM(6), name="x0")                       # this K-lane's first x element of vector 0
    zeros = [O._cf(b, F32(0.0), "z%d" % v) for v in range(NB)]   # one each: phis sharing an entry value would share a register
    counter0 = b.const(0, name="trip0")
    hdr, ex = fn.block("qmvw_k"), fn.block("qmvw_done")
    b.br(hdr)
    b.at(hdr)
    i = b.phi(counter0, name="trip")
    wi = b.phi(w0, name="wi")
    si = b.phi(s0, name="si")
    xi = b.phi(x0, name="xi")
    acc = [b.phi(zeros[v], type=ir.F32, name="acc%d" % v) for v in range(NB)]
    s32 = b.shl(b.load(a, si, width="half", name="sh"), IM(16), type=I, name="s32")
    b32 = b.shl(b.load(a, b.add(si, bdiff, name="bi"), width="half", name="bh"), IM(16), type=I, name="b32")
    wq = b.shr(wi, IM(2), name="wq")                         # vec4 units: 8 words are two of them
    words = b.load_vec_at(a, wq, n=4, name="wa") + b.load_vec_at(a, b.add(wq, IM(1), name="wq1"), n=4, name="wb")
    cur = list(acc)
    for wd in range(8):
        # this word's 8 x values of each vector, two vec4 loads each (x fp32 [nb][K])
        xs = []
        for v in range(NB):
            xb = b.shr(b.add(xi, C(v * K + 8 * wd, "xo%d_%d" % (v, wd)), name="xa%d_%d" % (v, wd)), IM(2), name="xq%d_%d" % (v, wd))
            xs.append(b.load_vec_at(bb, xb, n=4, name="xl%d_%da" % (v, wd)) +
                      b.load_vec_at(bb, b.add(xb, IM(1), name="xq%d_%d1" % (v, wd)), n=4, name="xl%d_%db" % (v, wd)))
        for j in range(8):
            q = getattr(b, "and")(b.shr(words[wd], IM(4 * j), name="ws%d_%d" % (wd, j)) if j else words[wd], IM(15),
                                  name="q%d_%d" % (wd, j))
            w = b.fma(b.u32_to_f32(q, name="qf%d_%d" % (wd, j)), s32, b32, name="w%d_%d" % (wd, j))
            for v in range(NB):
                cur[v] = b.fma(w, xs[v][j], cur[v], name="a%d_%d_%d" % (v, wd, j))
    nxt = b.add(i, IM(1), name="trip_next")
    ir.Builder.phi_latch(wi, b.add(wi, IM(64), name="wi_next"))
    ir.Builder.phi_latch(si, b.add(si, IM(8), name="si_next"))
    ir.Builder.phi_latch(xi, b.add(xi, C(512, "xstep"), name="xi_next"))
    ir.Builder.phi_latch(i, nxt)
    for v in range(NB):
        ir.Builder.phi_latch(acc[v], cur[v])
    b.br_cond(b.cmp(nxt, T, "lt", name="more"), hdr, ex)
    b.at(ex)
    outs = []
    for v in range(NB):
        y = cur[v]
        y.type = ir.F32
        for m in (1, 2, 4):
            y = b.fadd(y, b.simd_shuffle_xor(y, m, name="sh%d_%d" % (v, m)), type=ir.F32, name="r%d_%d" % (v, m))
        outs.append(y)
    fn.skip_regions = True
    wb, wj = fn.block("qmvw_write"), fn.block("qmvw_wdone")
    b.br_cond(b.cmp(kl, 1, "lt", name="kl0"), wb, wj)
    b.at(wb)
    for v in range(NB):
        b.store_at(c, b.add(row, C(v * N, "yo%d" % v), name="yi%d" % v) if v else row, outs[v])
    b.br(wj)
    b.at(wj)
    b.ret()
    ir.verify(fn)
    return cc.compile_function(fn)


def _fma32(a, b, c):
    """f32(a b + c) with one rounding, from f32-valued float64 arrays: the product is exact in binary64 (48 bits); TwoSum
    gives the sum's error and rounding to odd makes the binary64 -> binary32 rounding the correct one (53 >= 2 24 + 2)."""
    a, b, c = (np.asarray(v, np.float64) for v in (a, b, c))
    with np.errstate(invalid="ignore", over="ignore"):
        p = a * b
        s = p + c
        bb = s - p
        e = (p - (s - bb)) + (c - bb)
        fix = np.isfinite(s) & np.isfinite(e) & (e != 0) & ((s.view(np.uint64) & np.uint64(1)) == 0)
        s = np.where(fix, np.nextafter(s, np.where(e > 0, np.inf, -np.inf)), s)
        return s.astype(np.float32).astype(np.float64)


def reference(lay, q, s16, b16, x):
    """y fp32 [nb][N] in the stated order (q [N][K] nibbles, s16 / b16 [N][K/64] bf16 bits, x fp32 [nb][K])."""
    if lay.get("form") == "r4":
        return _reference_r4(lay, q, s16, b16, x)
    import g17qmv as Q
    N, K, NB = lay["N"], lay["K"], lay["nb"]
    s = Q._from_bf16(s16).astype(np.float64)
    bi = Q._from_bf16(b16).astype(np.float64)
    G = K // 64
    acc = np.zeros((NB, N, 8))                               # [vector][row][K-lane]
    for kl in range(8):
        for g in range(kl, G, 8):
            for e in range(64):
                k = 64 * g + e
                w = _fma32(q[:, k].astype(np.float64), s[:, g], bi[:, g])          # f32(q s + b) per row
                for v in range(NB):
                    acc[v, :, kl] = _fma32(w, np.float64(np.float32(x[v, k])), acc[v, :, kl])
    y = acc
    for m in (1, 2, 4):
        y = (y.astype(np.float32) + y[:, :, np.arange(8) ^ m].astype(np.float32)).astype(np.float64)
    return y[:, :, 0].astype(np.float32)


def case(lay, seed=5):
    import g17qmm as QM
    packed, s16, b16, q = QM.weights(lay["N"], lay["K"], 4, seed)
    x = np.random.default_rng(seed + 1).standard_normal((lay["nb"], lay["K"])).astype(np.float32)
    return packed, s16, b16, q, x


def io(lay, packed, s16, b16, x):
    a = bytearray(lay["a_bytes"])
    for off, arr in ((lay["W"], np.asarray(packed, "<u4")), (lay["S"], np.asarray(s16, "<u2")), (lay["B"], np.asarray(b16, "<u2"))):
        raw = np.ascontiguousarray(arr).tobytes(); a[off:off + len(raw)] = raw
    bb = bytearray(lay["b_bytes"]); raw = np.ascontiguousarray(x, "<f4").tobytes(); bb[:len(raw)] = raw
    return bytes(a), bytes(bb), b"\x7f" * lay["c_bytes"]


def author(d, prog, a, bb, c, lay):
    import g17prefillattn_run as PR
    return PR.author(Path(d), prog, a, bb, c, lay)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("check",))
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--k", type=int, default=1024)
    ap.add_argument("--form", default="kl8", choices=("kl8", "r4"))
    args = ap.parse_args(argv)
    import tempfile
    import g17emu as E
    for nb in (1, 2, 4):
        lay = layout(args.n, args.k, nb, form=args.form)
        prog = build(lay)
        packed, s16, b16, q, x = case(lay)
        want = reference(lay, q, s16, b16, x)
        a, bb, c = io(lay, packed, s16, b16, x)
        with tempfile.TemporaryDirectory() as t:
            d = author(Path(t) / "w", prog, a, bb, c, lay)
            out = E.run_bundle(d, 64 * lay["groups"], 64, 1, tier="wp")
        out = out[0] if isinstance(out, tuple) else out
        got = np.frombuffer(out, "<f4", nb * lay["N"]).reshape(nb, lay["N"])
        print("nb %d: %d instructions, differ %d of %d" % (nb, len(prog.code) // 4, int((got.view(np.uint32) != want.view(np.uint32)).sum()), want.size))
    return 0


if __name__ == "__main__":
    sys.exit(main())
