#!/usr/bin/env python3
"""The SwiGLU-fused w3 GEMM (MM 25.144.12, the prefill lever): w3's quantized-prefill GEMM with the SwiGLU in its tail,
so the graph's separate swiglu_rows dispatch (7.2 ms of the 1,024-token prefill, 25.142.11) disappears.

One program, the measured three-binding tensor ABI: 1 A = x fp16 [M][K] rows (w3's input, the ffn norm's output), 2 B = w3's persistent W16 [K][N]
fp16, 3 = one region holding
    U    fp32 [M][N] at U_OFF = 0  this GEMM's own output (the up projection), a scratch the tail rereads; at 0
                                   because the K-loop GEMM path applies no C offset (cc refuses one there)
    G    fp32 [M][N] at G_OFF      w1's output (the gate): the graph binds w1's C here
    ACT  fp16 [M][N] at ACT_OFF    act = fp16(silu(G) U), what w2 reads
1. The GEMM: w3's delivered form (g17qmm.FORM at M 256: 8 simdgroups, grid_n 256, 4 K slices a trip, the 2 x 2 tile),
   the same `tensor_matmul` as g17tensorcommonruntime's generic build, storing U at U_OFF.
2. The tail: threadgroup t owns columns [32 t, 32 t + 32) and simdgroup s rows [s M/sg, (s+1) M/sg), exactly the block its
   own tensor stores wrote (tlower's row split), so each simdgroup rereads only what it stored. Lane l takes column
   32 t + l and the 32 rows in a counted loop (`rows` a trip), and applies g17rows' swiglu per element in its order:
   t = g (-1/ln 2), exp2_soft, + 1, the corrected reciprocal, g r, then u, fp16 RNE.
So the result is bit-identical to the unfused pipeline (w3's GEMM, then swiglu_rows) on the same G: the reference is
`swiglu(G, mma(x, W16))`, g17rows.rows_reference over the pinned MMA model (_gemm_mma's single K chain).

NOTHING HERE DISPATCHES. `build` compiles; `check` runs cc's guards and the counted-loop and hazard checks on the
bytes; the hardware verify is g17deliver's kind `qmm_swiglu` (one dispatch per bundle, when the GPU is free); the CPU
verify is g17emu's (`verify_emu`): bit-exact at M 128 and 256 (MM 25.144.12).

    python3 tools/g17swigluqmm.py check [--M 256]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import g17decodeops as O  # noqa: E402

N_FFN, K_D = 8192, 2048
ROWS = 1                                     # tail rows a loop trip (the smallest program; MM 25.144.12 prices 1-4)


def _align(v, a=256):
    return -(-v // a) * a


def layout(M=256, N=N_FFN, K=K_D, unroll=None, rows=None):
    """The region offsets and w3's delivered GEMM form at M; `unroll` overrides its K slices per trip (the tail needs
    registers the deepest pipelining holds)."""
    import g17qmm as Q
    sp = dict(Q.role_spec("w3", M))
    if unroll is not None:
        sp["kloop_unroll"] = unroll
        if unroll == 1:
            sp.pop("kloop_unroll")
    sg, gn = sp["simdgroups"], sp["grid_n"]
    if sp["split_k"] != 1:
        raise ValueError("qmm_swiglu: w3's form at M %d splits K; the tail needs whole sums" % M)
    R = rows or ROWS
    if N // gn != 32 or (M // sg) % R or (M // sg) // R > 255:
        raise ValueError("qmm_swiglu: the tail maps lane = column (32 columns a threadgroup) and runs the simdgroup's "
                         "rows R at a time in at most 255 trips; w3's form at M %d gives %d columns per threadgroup and "
                         "%d rows per simdgroup" % (M, N // gn, M // sg))
    # ONE REGION at binding 3 (the three-binding tensor ABI is the measured one; a fourth binding is refused by the
    # linker as an unmeasured signature): U at 0 (the K-loop path applies no C offset), then G, then act
    U_OFF = 0
    G_OFF = _align(4 * M * N)
    ACT_OFF = G_OFF + _align(4 * M * N)
    return dict(op="qmm_swiglu", M=M, N=N, K=K, rows=rows or ROWS, simdgroups=sg, grid_n=gn, kloop_unroll=sp.get("kloop_unroll", 1),
                kloop_bases=bool(sp.get("kloop_bases", False)), G_OFF=G_OFF, U_OFF=U_OFF, ACT_OFF=ACT_OFF,
                c_bytes=ACT_OFF + _align(2 * M * N), threadgroups=gn, threads_per_group=32 * sg,
                a_bytes=2 * M * K, b_bytes=2 * K * N, gemm_spec=sp)


class _Lazy(dict):
    """A constant map (g17decodeops' emit_constants / emit_exp2_constants keys) that emits each constant AT ITS USE,
    under a fresh name: nothing is held across the reciprocal's midpoint test, so the narrow registers stay free beside
    the tensor body's reservation. The values are the same constants, so the arithmetic is unchanged."""

    def __init__(self, b, make, tag):
        super().__init__()
        self.b, self.make, self.tag, self.n = b, make, tag, 0

    def __getitem__(self, key):
        self.n += 1
        return self.make(self.b, key, "%s_%s_%d" % (self.tag, key, self.n))


def _k1(b, key, name):
    return O._c(b, dict(mant=0x7FFFFF, hidden=0x800000, s23=23, ff=0xFF, s1=1, one=1, m15=O.M15, s15=15, c392=392,
                        c271=271, zero=0)[key], name)


def _k2(b, key, name):
    if key in ("mbits", "s23"):
        return O._c(b, 0x4B400000 if key == "mbits" else 23, name)
    v = dict(lo=O.EXP2_LO, hi=O.EXP2_HI, magic=O.EXP2_MAGIC, nmagic=-O.EXP2_MAGIC)
    v.update({"c%d" % i: cf for i, cf in enumerate(O.EXP2_COEF)})
    return O._cf(b, v[key], name)


def build(lay, hold=None):
    """The fused program (cc), compiled with the release and dead-op guards REFUSING (MM 25.144.6).

    THE TAIL IS A COUNTED LOOP of 32 / ROWS trips, `lay["rows"]` rows a trip (ROWS by default), each row applying
    g17rows' swiglu. Until cc's interval attempt (MM 25.144.12) this could not allocate beside the K-looped tensor body:
    the narrow pool was r4..r15 less the body's registers (5 left) and the loop's phis and read_sr values were
    pre-coloured into it for the whole program, so the tail had to be straight-line (49.7 KB). cc now shares the
    body's narrow registers after its last row and holds a pre-coloured register only over its live interval.
    `hold`: the exp2 polynomial's constants made once before the loop and held, or made at each use; None tries
    holding first. The reciprocal's integer constants are made at each use either way."""
    if hold is None:
        from agxforge.g17 import cc
        try:
            return build(lay, hold=True)
        except cc.Unsupported:
            return build(lay, hold=False)
    from agxforge.g17 import cc, ir
    from agxforge.g17 import tlower as _tl
    import g17tensorcommonruntime as TCR
    M, N, K, sg = lay["M"], lay["N"], lay["K"], lay["simdgroups"]
    R = lay.get("rows", ROWS)
    rows_per_sg = M // sg
    if rows_per_sg % R:
        raise ValueError("qmm_swiglu: %d rows a trip do not divide the simdgroup's %d rows" % (R, rows_per_sg))
    I = ir.I32
    a = ir.Buffer("A", 1, elem=ir.F16)
    bb = ir.Buffer("B", 2, elem=ir.F16)
    c = ir.Buffer("C", 3, elem=ir.F32)
    fn = ir.Function("tensor_gemm_generic_runtime_demo", [a, bb, c])
    b = ir.Builder(fn, fn.block("entry"))
    b.tensor_matmul(a, bb, c, M=M, N=N, K=K, threadgroups=1, grid_n=lay["grid_n"], kloop=True,
                    kloop_unroll=lay["kloop_unroll"])
    op = next(o for blk in fn.blocks for o in blk.ops if o.kind == "tensor_matmul")
    if sg > 1:
        op.attrs["simdgroups"] = sg                  # as g17tensorcommonruntime's generic build sets it (8 admitted)
    # ---- the tail: this simdgroup's own block, lane l = column 32 t + l, rows s M/sg .. (s+1) M/sg - 1
    lane = b.builtin("thread_index_in_simdgroup", name="lane")
    t = b.builtin("threadgroup_position_in_grid", name="t")
    s = b.builtin("simdgroup_index_in_threadgroup", name="s")
    col = b.add(b.shl(t, O._c(b, 5, "k5"), name="t32"), lane, name="col")
    e0 = b.add(b.mul(s, O._c(b, rows_per_sg * N, "sgrows"), name="s_rows"), col, name="e0")   # (row s M/sg, col)
    K2h = O.emit_exp2_constants(b) if hold else None
    hdr, post = fn.block("tail_loop"), fn.block("tail_done")
    j0 = O._c(b, 0, "j0")
    b.br(hdr)
    b.at(hdr)
    j = b.phi(j0, name="j")
    pe = b.phi(e0, name="pe")
    e = pe
    for r in range(R):
        tag = "w%d" % r
        K1 = _Lazy(b, _k1, tag + "_k")
        K2 = K2h if hold else _Lazy(b, _k2, tag + "_e")
        g = b.load(c, b.add(e, O._c(b, lay["G_OFF"] // 4, tag + "_gb"), name=tag + "_gi"), type=I, name=tag + "_g")
        u = b.load(c, e, type=I, name=tag + "_u")
        # g17rows' swiglu, operation for operation (g17decodestep.stage_ffn_swiglu)
        tt = b.fmul(g, O._cf(b, TCR.GELU_NEG_INV_LN2, tag + "_nil2"), type=I, name=tag + "_st")
        ex = O.emit_exp2_soft(b, tt, K2, tag + "_sx")
        den = b.fadd(ex, O._cf(b, np.float32(1.0), tag + "_one"), type=I, name=tag + "_sd")
        rc = O.emit_rn(b, "recip", den, K1, tag + "_sr")
        sl = b.fmul(g, rc, type=I, name=tag + "_silu")
        y = b.fmul(sl, u, type=I, name=tag + "_act")
        b.store_at(c, b.add(e, O._c(b, lay["ACT_OFF"] // 2, tag + "_ab"), name=tag + "_ai"),
                   b.f32_to_f16_rte(y, name=tag + "_ah"), width="half")
        e = b.add(e, O._c(b, N, tag + "_kN"), name=tag + "_en")
    jn = b.add(j, ir.Imm(1), name="j_next")
    ir.Builder.phi_latch(j, jn)
    ir.Builder.phi_latch(pe, e)
    b.br_cond(b.cmp(jn, rows_per_sg // R, "lt", name="tail_more"), hdr, post)
    b.at(post)
    b.ret()
    ir.verify(fn)
    saved = _tl.KLOOP_BASES
    _tl.KLOOP_BASES = lay["kloop_bases"]
    try:
        prog = cc.compile_function(fn, guards="refuse")
    finally:
        _tl.KLOOP_BASES = saved
    prog.hold = bool(hold)
    return prog


def reference(lay, x16, w16, g32):
    """(U fp32 [M][N], act fp16 [M][N]): U = the pinned MMA model of x16 @ w16 (_gemm_mma, one K chain, the order
    g17deliver's qmm entries at M 128 / 256 are checked in), act = g17rows' swiglu of (G, U)."""
    import g17rows as RW
    import g17tensorcommonruntime as TCR
    M, N, K = lay["M"], lay["N"], lay["K"]
    u = np.asarray(TCR._gemm_mma(np.asarray(x16, np.float16), np.asarray(w16, np.float16), None, M, N, K), np.float32)
    rl = RW.rows_layout("swiglu", M, N)
    act = RW.rows_reference(rl, (np.asarray(g32, np.float32).reshape(-1), u.reshape(-1)))["act"].reshape(M, N)
    return u, act


def check(lay, prog=None):
    """Compile-time checks on the bytes, no dispatch: cc's guards ran in compile; here the loops (the GEMM's static K
    loop and the tail's), the load-use hazards across both, and the code size. -> dict."""
    from agxforge.g17 import tensorview as TV
    prog = prog or build(lay)
    v = TV.view(prog.code)
    return dict(code_bytes=len(prog.code), instructions=len(v), loops=TV.loops(v), hazards=TV.hazards(v),
                guards=dict((k, prog.guards.get(k)) for k in ("mode", "dead", "release", "release_checked")))


def verify_emu(bundle_dir, lay, check, tier="wp"):
    """Bit-exactness on the CPU through g17emu (Claude A's Goal 1, linker/g17-emu): run the authored bundle's program
    over its own inputs and return check(region 3 after the run), 0 when U and act are both bit-exact. Tier "wp": the
    reciprocal's 10-byte OR (op13575) has whole-program evidence only; "strict" refuses it until a hardware receipt of
    that length exists (MM 25.144.12). Measured: M 128 and 256 bit-exact, and g17emu.control's mutant rejected."""
    from pathlib import Path
    try:
        import g17emu
    except ImportError:
        raise NotImplementedError("g17emu is not in this checkout (linker/g17-emu); the CPU verify waits for it")
    d = Path(bundle_dir)
    bufs = {1: (d / "a.f16").read_bytes(), 2: (d / "b.f16").read_bytes(), 3: (d / "c.f32").read_bytes()}
    out = g17emu.run(d, bufs, dict(threadgroups=lay["threadgroups"], threads_per_group=lay["threads_per_group"]), tier=tier)
    return check(bytes(out[3]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--M", type=int, default=256)
    c.add_argument("--unroll", type=int, default=None)
    a = ap.parse_args(argv)
    lay = layout(a.M, unroll=a.unroll)
    print(check(lay))


if __name__ == "__main__":
    main()
