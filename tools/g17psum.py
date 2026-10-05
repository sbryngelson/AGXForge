#!/usr/bin/env python3
"""The split-K partial sum for the batched decode (MM 25.172): g17qsm's sk partials [sk][rows][N] fp32 summed in
ascending order, then one of three epilogues, element-wise over rows x N:
  sum     out32 = ((p0 + p1) + p2) + ...                      (qkv: the fp32 the attention reads)
  add16   h32  = sum + fp32(x16)                              (wo: g17qmv.residual_reference's add16, y then + x)
  fold16  x16  = fp16_rne(sum + h32)                          (w2: the other residual, y then + h, then rounded)
  half    x16  = fp16_rne(sum)                                (the attention's fp32 rows narrowed for the next g17qsm)
  swiglu  act  = swiglu(sum_w1, sum_w3)                       (MM 25.185: w1's partials at binding 1, w3's at binding
                                                               2, each summed in ascending order, then g17rows' SwiGLU
                                                               order: fp16(silu(g) u); one pass where two psum sums and
                                                               a rows kernel were three)
The partials are `pstride` fp32 elements apart (g17qsm writes 16-row partials: 16 N), of which the first rows x N
are summed - a batch below 16 reads only its own rows.
Buffers: 1 the partials at 0; 2 the residual input at IN (bytes); 3 the output at OUT (bytes). The batched graph binds
2 and 3 to its residual region (x16 rows and h rows), so the epilogue reads and writes R directly.

    python3 tools/g17psum.py check        # compile + g17emu bit-exact on the three modes (CPU only)
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import g17decodeops as O  # noqa: E402

F32 = np.float32
U = 4
MODES = ("sum", "add16", "fold16", "half", "swiglu")


def _align(v, a=256):
    return -(-v // a) * a


def layout(N, sk, mode, rows=16, IN=0, OUT=0, pstride=None):
    if mode not in MODES or sk < 1 or (rows * N) % (32 * U):
        raise ValueError("psum: mode one of %r, sk >= 1, rows x N a multiple of %d" % (MODES, 32 * U))
    n = rows * N
    pstride = n if pstride is None else int(pstride)
    if pstride < n:
        raise ValueError("psum: the partial stride %d is below rows x N %d" % (pstride, n))
    in_w = {"sum": 0, "add16": 2, "fold16": 4, "half": 0, "swiglu": 0}[mode]
    out_w = 2 if mode in ("fold16", "half", "swiglu") else 4
    return dict(N=N, sk=sk, mode=mode, rows=rows, n=n, pstride=pstride, IN=IN, OUT=OUT, groups=n // (32 * U),
                a_bytes=_align(((sk - 1) * pstride + n) * 4),
                b_bytes=(_align(((sk - 1) * pstride + n) * 4) if mode == "swiglu" else _align(IN + in_w * n) if in_w else 256),
                c_bytes=_align(OUT + out_w * n))


def build(lay):
    from agxforge.g17 import cc, ir
    fn, b, a, bb, c = O._function()
    I = ir.I32
    n, sk, mode, ps = lay["n"], lay["sk"], lay["mode"], lay.get("pstride", lay["n"])
    lane = b.builtin("thread_index_in_simdgroup", name="lane")
    t = b.builtin("threadgroup_position_in_grid", name="t")
    base = b.add(b.mul(t, O._c(b, 32 * U, "tU"), name="tb"), lane, name="base")
    idx = [b.add(base, O._c(b, 32 * i, "o%d" % i), name="e%d" % i) if i else base for i in range(U)]
    if mode == "swiglu":
        import g17tensorcommonruntime as TCR
        K = O.emit_constants(b)
        K2 = O.emit_exp2_constants(b)
        nil2 = O._cf(b, TCR.GELU_NEG_INV_LN2, "neg_inv_ln2")
        one = O._cf(b, F32(1.0), "fone")

    def total(buf, e, i, tag):
        s = b.load(buf, e, type=I, name="%s0_%d" % (tag, i))
        for p in range(1, sk):
            s = b.fadd(s, b.load(buf, b.add(e, O._c(b, p * ps, "%so%d_%d" % (tag, p, i)), name="%si%d_%d" % (tag, p, i)),
                                 type=I, name="%s%d_%d" % (tag, p, i)), type=I, name="%ss%d_%d" % (tag, p, i))
        return s
    for i, e in enumerate(idx):
        s = total(a, e, i, "p")
        if mode == "swiglu":
            g, u = s, total(bb, e, i, "q")
            tt = b.fmul(g, nil2, type=I, name="st%d" % i)
            ex = O.emit_exp2_soft(b, tt, K2, "sx%d" % i)
            den = b.fadd(ex, one, type=I, name="sd%d" % i)
            rc = O.emit_rn(b, "recip", den, K, "sr%d" % i)
            y = b.fmul(b.fmul(g, rc, type=I, name="silu%d" % i), u, type=I, name="act%d" % i)
            oi = b.add(e, O._c(b, lay["OUT"] // 2, "ob%d" % i), name="oi%d" % i) if lay["OUT"] else e
            b.store_at(c, oi, b.f32_to_f16_rte(y, name="ah%d" % i), width="half")
            continue
        if mode == "sum":
            oi = b.add(e, O._c(b, lay["OUT"] // 4, "ob%d" % i), name="oi%d" % i) if lay["OUT"] else e
            b.store_at(c, oi, s)
        elif mode == "add16":
            xi = b.add(e, O._c(b, lay["IN"] // 2, "ib%d" % i), name="xi%d" % i) if lay["IN"] else e
            h = b.fadd(s, b.f16_to_f32(b.load(bb, xi, width="half", name="xh%d" % i), name="x%d" % i), type=I, name="h%d" % i)
            oi = b.add(e, O._c(b, lay["OUT"] // 4, "ob%d" % i), name="oi%d" % i) if lay["OUT"] else e
            b.store_at(c, oi, h)
        elif mode == "half":
            oi = b.add(e, O._c(b, lay["OUT"] // 2, "ob%d" % i), name="oi%d" % i) if lay["OUT"] else e
            b.store_at(c, oi, b.f32_to_f16_rte(s, name="xo%d" % i), width="half")
        else:
            hi = b.add(e, O._c(b, lay["IN"] // 4, "ib%d" % i), name="hi%d" % i) if lay["IN"] else e
            v = b.fadd(s, b.load(bb, hi, type=I, name="hv%d" % i), type=I, name="v%d" % i)
            oi = b.add(e, O._c(b, lay["OUT"] // 2, "ob%d" % i), name="oi%d" % i) if lay["OUT"] else e
            b.store_at(c, oi, b.f32_to_f16_rte(v, name="xo%d" % i), width="half")
    b.ret()
    ir.verify(fn)
    return cc.compile_function(fn)


def reference(lay, parts, resid=None):
    """parts [sk][rows][N] fp32 -> the mode's output, in the kernel's order."""
    ps = lay.get("pstride", lay["n"])
    flat = np.asarray(parts, F32).reshape(-1)
    p = np.stack([flat[k * ps:k * ps + lay["n"]] for k in range(lay["sk"])])
    s = p[0].copy()
    for k in range(1, lay["sk"]):
        s = (s + p[k]).astype(F32)
    if lay["mode"] == "sum":
        return s
    if lay["mode"] == "half":
        return s.astype(np.float16)
    if lay["mode"] == "swiglu":
        import g17rows as RW
        flat2 = np.asarray(resid, F32).reshape(-1)
        p2 = np.stack([flat2[k * ps:k * ps + lay["n"]] for k in range(lay["sk"])])
        u = p2[0].copy()
        for k in range(1, lay["sk"]):
            u = (u + p2[k]).astype(F32)
        return RW.rows_reference(dict(kind="swiglu"), (s, u))["act"]
    if lay["mode"] == "add16":
        return (s + np.asarray(resid, np.float16).reshape(-1).astype(F32)).astype(F32)
    return (s + np.asarray(resid, F32).reshape(-1)).astype(F32).astype(np.float16)


def check(N=512, sk=4, seed=3, rows=16):
    import g17emu as EMU
    import g17prefillattn_run as PR
    rng = np.random.default_rng(seed)
    bad = {}
    for mode in MODES:
        lay = layout(N, 1 if mode == "half" else sk, mode, rows=rows, IN=256, OUT=512, pstride=16 * N)
        prog = build(lay)
        parts = rng.standard_normal((lay["sk"], 16, N)).astype(F32)
        resid = (rng.standard_normal((rows, N)).astype(np.float16) if mode == "add16" else
                 rng.standard_normal((rows, N)).astype(F32) if mode == "fold16" else
                 rng.standard_normal((lay["sk"], 16, N)).astype(F32) if mode == "swiglu" else None)
        a = bytearray(lay["a_bytes"]); raw = parts.tobytes(); a[:len(raw)] = raw
        bb = bytearray(lay["b_bytes"])
        if resid is not None:
            at = 0 if mode == "swiglu" else lay["IN"]
            raw = resid.tobytes(); bb[at:at + len(raw)] = raw
        c = b"\x7f" * lay["c_bytes"]
        want = reference(lay, parts, resid)
        with tempfile.TemporaryDirectory() as t:
            d = PR.author(Path(t) / ("psum_" + mode), prog, bytes(a), bytes(bb), c, lay)
            out, _m = EMU.run_bundle(d, lay["groups"] * 32, 32, 1, tier="wp")
        w = 2 if mode in ("fold16", "half", "swiglu") else 4
        got = np.frombuffer(out, "<u2" if w == 2 else "<u4", lay["n"], lay["OUT"])
        bad[mode] = int((got != want.view("<u2" if w == 2 else "<u4")).sum())
        print("psum %-6s N %d sk %d rows %d: %d bytes, %d of %d differ" % (mode, N, lay["sk"], rows, len(prog.code), bad[mode],
                                                                     lay["n"]))
    return bad


if __name__ == "__main__":
    if sys.argv[1:] != ["check"]:
        sys.exit("usage: g17psum.py check")
    sys.exit(0 if not any(check().values()) and not any(check(rows=8).values()) else 1)
