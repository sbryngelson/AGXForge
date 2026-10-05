#!/usr/bin/env python3
"""The quantized prefill GEMM (MM 25.144.1): y[M][N] = x[M][K] . W^T, W the model's affine q4/q8 weights (group 64,
bf16 scales and biases, the layout g17qmv reads), on the tensor units.

Form 1, two dispatches:
  dequant   one thread per packed word, n fastest: w[k][n] = fp16_rne(fp32(q) * s + b) written as W16[K][N] (the
            GEMM's row-major B; the K loop refuses transposed operands). fp32(q) * s is exact (a bf16 scale times an
            integer below 256 fits fp32), the + b rounds once, then fp16 rounds: the reference does the same three
            steps in numpy.
  gemm      gemm_generic (tlower) on A = x fp16 [M][K] rows and B = W16, C = [M][N] fp32 rows (ldc = N), the
            K loop in 1-4 simdgroups, columns over grid_n, K over split_k (w2). The reference is the pinned MMA
            model (_gemm_mma / _gemm_mma_fast) on the dequant's own output.

    python3 tools/g17qmm.py verify --bits 4 --m 512 [--roles qkv,wo,w1,w2]
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import g17qmv as Q  # noqa: E402
import g17decodeops as O  # noqa: E402

SENT = b"\x7f"
# role: (N, K) of the milestone model (d 2048, ffn 8192, wqkv N 4096)
ROLES = {"qkv": (4096, 2048), "wo": (2048, 2048), "w1": (8192, 2048), "w3": (8192, 2048), "w2": (2048, 8192)}


def dequant_layout(N, K, bits):
    """Offsets: the B buffer holds packed words at W (bytes), bf16 scales at S and biases at B (bytes); the C buffer
    holds W16[K][N] fp16 at OUT (bytes)."""
    pw = 32 // bits
    wbytes = N * (K // pw) * 4
    gbytes = N * (K // 64) * 2
    return dict(N=N, K=K, bits=bits, pw=pw, W=0, S=wbytes, B=wbytes + gbytes, b_bytes=wbytes + 2 * gbytes,
                OUT=0, c_bytes=K * N * 2, threads=N * (K // pw))


def build_dequant(lay):
    """The dequant kernel (cc IR): thread gid = tg * 32 + lane; n = gid mod N, j = gid / N (the packed word of row n);
    its pw values land at W16[(j pw + i) N + n], i = 0..pw-1."""
    from agxforge.g17 import cc, ir
    N, K, bits, pw = lay["N"], lay["K"], lay["bits"], lay["pw"]
    if N & (N - 1) or K % 64:
        raise ValueError("dequant: N a power of two, K a multiple of 64")
    fn, b, a, bb, c = O._function()
    lane = b.builtin("thread_index_in_simdgroup", name="lane")
    tg = b.builtin("threadgroup_position_in_grid", name="tg")
    gid = b.add(b.shl(tg, O._c(b, 5, "k5"), name="tg32"), lane, name="gid")
    n = getattr(b, "and")(gid, O._c(b, N - 1, "nmask"), name="n")
    j = b.shr(gid, O._c(b, N.bit_length() - 1, "nsh"), name="j")
    wi = b.add(b.add(b.mul(n, O._c(b, K // pw, "wrow"), name="nw"), j, name="nwj"), O._c(b, lay["W"] // 4, "wb"), name="wi")
    word = b.load(bb, wi, type=ir.I32, name="word")
    g = b.shr(j, O._c(b, (64 // pw).bit_length() - 1, "gsh"), name="g")           # group = (j pw) / 64
    si = b.add(b.add(b.mul(n, O._c(b, K // 64, "grow"), name="ng"), g, name="ngg"), O._c(b, lay["S"] // 2, "sb"), name="si")
    bi = b.add(si, O._c(b, (lay["B"] - lay["S"]) // 2, "bdiff"), name="bi")
    s32 = b.shl(b.load(bb, si, width="half", name="sh"), O._c(b, 16, "s16"), name="s32")
    b32 = b.shl(b.load(bb, bi, width="half", name="bh"), O._c(b, 16, "b16"), name="b32")
    base = b.add(b.add(b.mul(j, O._c(b, pw * N, "jrow"), name="jN"), n, name="jNn"), O._c(b, lay["OUT"] // 2, "ob"),
                 name="obase")
    mask = (1 << bits) - 1
    for i in range(pw):
        q = getattr(b, "and")(b.shr(word, ir.Imm(bits * i), name="ws%d" % i) if i else word, ir.Imm(mask), name="q%d" % i)
        v = b.fmul(b.u32_to_f32(q, name="qf%d" % i), s32, type=ir.F32, name="v%d" % i)
        w = b.fadd(v, b32, type=ir.I32, name="w%d" % i)   # float bits typed I32, as the norms carry them into the fp16 convert
        oi = b.add(base, O._c(b, i * N, "io%d" % i), name="oi%d" % i) if i else base
        b.store_at(c, oi, b.f32_to_f16_rte(w, name="h%d" % i), width="half")
    b.ret()
    ir.verify(fn)
    return cc.compile_function(fn)


def dequant_reference(q, s16, b16, bits):
    """W16[K][N]: fp16_rne(fp32(q) * s + b), each fp32 op rounded to nearest even (numpy float32)."""
    N, K = q.shape
    s = Q._from_bf16(s16).astype(np.float32)
    bb = Q._from_bf16(b16).astype(np.float32)
    srep = np.repeat(s, 64, axis=1)
    brep = np.repeat(bb, 64, axis=1)
    w = (q.astype(np.float32) * srep).astype(np.float32)
    w = (w + brep).astype(np.float32)
    return w.astype(np.float16).T.copy()


def weights(N, K, bits, seed):
    rng = np.random.default_rng(seed)
    Wf = (rng.standard_normal((N, K)) * 0.02).astype(np.float32)
    return Q.quantize(Wf, bits=bits)          # packed [N][K/pw] u32, s16, b16 [N][K/64], q [N][K]


def dequant_io(lay, packed, s16, b16):
    b = bytearray(lay["b_bytes"])
    O._place(b, lay["W"], packed.astype("<u4"))
    O._place(b, lay["S"], s16.astype("<u2"))
    O._place(b, lay["B"], b16.astype("<u2"))
    c = SENT * lay["c_bytes"]
    return bytes(1024), bytes(b), c


# THE DELIVERED FORM per (role, M) (MM 25.144.1): K slices per trip (kloop_unroll) and loop-carried row bases, chosen
# by the warm-clock protocol (g17projwarm, 9 rounds, bit-exact every unit). M 512 bodies split register groups, so a
# longer prompt runs faster as M 256 chunks than as one M 512 dispatch.
FORM = {"qkv": {128: dict(u=4, gn=128), 256: dict(u=4, sg=4, gn=128, tg=2), 512: dict()},
        "wo": {128: dict(u=4, gn=64), 256: dict(u=4, sg=2, gn=64, tg=4), 512: dict()},
        "w1": {128: dict(u=4), 256: dict(u=4, sg=8), 512: dict()},
        "w3": {128: dict(u=4), 256: dict(u=4, sg=8), 512: dict()},
        "w2": {128: dict(u=4, sk=1, gn=64), 256: dict(u=4, sg=2, sk=1, gn=64, tg=4), 512: dict()}}
# MLX's per-simdgroup tile, 2 x 2 (MM 25.144.1): simdgroups x grid_n chosen so each simdgroup owns 2 row tiles and 2
# column tiles, 4 slices per trip; 1.1-1.3x faster warm than the 4 x 1 / 2 x 1 tiles at every role but w1 (already 2 x 2).
# sk overrides gemm_shape's split_k: w2 (K 8192) runs its whole K in one threadgroup (127 trips at 4 slices).
# tg is the M-block grid (MM 25.157): tg row groups beside the grid_n column groups (rows the low bits of the id, so
# the row groups sharing a column block of W16 run adjacently), each simdgroup still a 2 x 2 tile (32 rows; tlower's
# simdgroup split is along rows, so tg x sg x 32 = M). qkv, wo and w2 reach 256 threadgroups at M 256: -2.5 percent of
# the 1,024-token prefill's GPU time, -3.9 ms wall. Timed in the graph, not alone: alone the weights stay cached.


def gemm_spec(M, N, K, sg, grid_n, split_k, unroll=1, bases=False, tg=1):
    spec = dict(M=M, N=N, K=K, threadgroups=tg, simdgroups=sg, grid_n=grid_n, split_k=split_k, kloop=True)
    if unroll > 1:
        spec["kloop_unroll"] = unroll
    if bases:
        spec["kloop_bases"] = True
    return spec


def role_spec(role, M):
    """The delivered spec for (role, M): gemm_shape's launch plus FORM's loop form (M outside FORM: the plain loop)."""
    N, K = ROLES[role]
    sg, gn, sk = gemm_shape(M, N, K)
    f = FORM.get(role, {}).get(M, {})
    if "sk" in f:
        sk = f["sk"]
        gn = min(N // 16, 256 // sk)
    gn = f.get("gn", gn)
    return gemm_spec(M, N, K, f.get("sg", sg), gn, sk, f.get("u", 1), f.get("bases", False), f.get("tg", 1))


def gemm_shape(M, N, K):
    """(sg, grid_n, split_k) for a projection: 16 rows x 16 columns per MMA tile; 4 simdgroups own M/4 rows each
    (M/64 tile rows per simdgroup), one tile column per threadgroup where N allows <= 256 threadgroups (w1 two),
    and K over two threadgroup groups when K/16 exceeds the loop's 256 slices."""
    sg = 4 if M >= 64 else 1
    split_k = 2 if K // 16 > 256 else 1
    grid_n = min(N // 16, 256 // split_k)
    return sg, grid_n, split_k


def verify(bits, M, roles, work, seed=11):
    """Dequant then GEMM on hardware for each role; returns {role: dict}. Every output is sentinel-filled first."""
    import g17deliver as D
    import g17tensorcommonruntime as R
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    out = {}
    for role in roles:
        N, K = ROLES[role]
        lay = dequant_layout(N, K, bits)
        packed, s16, b16, q = weights(N, K, bits, seed + bits * 10 + len(role))
        prog = build_dequant(lay)
        a, bbuf, c = dequant_io(lay, packed, s16, b16)
        d = D.author(work / ("dequant_q%d_%s" % (bits, role)), prog, a, bbuf, c, lay)
        (work / "dq_run").mkdir(exist_ok=True)
        got = D.dispatch([dict(tag="dq", dir=d, threads=lay["threads"], group=32, base=1)], work / "dq_run")["dq"]
        w16 = np.frombuffer(got, "<u2", K * N, lay["OUT"]).reshape(K, N)
        want = dequant_reference(q, s16, b16, bits).view("<u2")
        dq_diff = int((w16 != want).sum())
        _sp = role_spec(role, M)
        sg, gn, sk = _sp["simdgroups"], _sp["grid_n"], _sp["split_k"]
        g = work / ("gemm_q%d_%s_m%d" % (bits, role, M))
        if g.exists():
            shutil.rmtree(g)
        R.author_generic(g, role_spec(role, M))
        rng = np.random.default_rng(seed + M)
        x = np.asarray(rng.standard_normal((M, K)), np.float16)
        (g / "a.f16").write_bytes(x.astype("<f2").tobytes())
        (g / "b.f16").write_bytes(np.asarray(w16).tobytes())          # the dequant's OWN output feeds the GEMM
        rep = R.run(g, queries=1, composition="generic")
        qrec = rep["queries"][0]
        out[role] = dict(N=N, K=K, M=M, dequant_mismatch=dq_diff, dequant_code_bytes=len(prog.code),
                         gemm_status=rep["status"], gemm_mismatch=qrec.get("mismatched_elements"),
                         gemm_shape=dict(sg=sg, grid_n=gn, split_k=sk),
                         gemm_code_bytes=len((g / "program.bin").read_bytes()))
        print("q%d %-3s M %d: dequant %d mismatched (%d B), gemm %s %s mismatched (sg %d gn %d sk %d, %d B)" % (
            bits, role, M, dq_diff, len(prog.code), rep["status"], qrec.get("mismatched_elements"), sg, gn, sk,
            out[role]["gemm_code_bytes"]), flush=True)
    return out


def bench(bits, ms, roles, work, rounds=15, warm=5, seed=11):
    """GPU time of ours per (role, M): the dequant (runner rounds, first `warm` dropped) and the GEMM (common-worker
    queries after `warm` warm-ups), medians. Every GEMM bundle is verified (status passed) before its time counts."""
    import statistics
    import subprocess
    import g17deliver as D
    import g17tensorcommonruntime as R
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    rows = []
    for role in roles:
        N, K = ROLES[role]
        lay = dequant_layout(N, K, bits)
        packed, s16, b16, q = weights(N, K, bits, seed + bits * 10 + len(role))
        prog = build_dequant(lay)
        a, bbuf, c = dequant_io(lay, packed, s16, b16)
        d = D.author(work / ("dequant_q%d_%s" % (bits, role)), prog, a, bbuf, c, lay)
        run = work / "dq_bench"
        run.mkdir(exist_ok=True)
        (run / "out").mkdir(exist_ok=True)
        plan = run / "plan.json"
        plan.write_text(json.dumps(dict(configs=[dict(tag="dq", bundle=str(d), threads=lay["threads"], group=32, base=1,
                                                      rounds=rounds + warm)])))
        r = subprocess.run([str(D.runner()), str(plan), str(run / "out")], capture_output=True, text=True)
        if r.returncode:
            raise SystemExit("dequant run failed: " + r.stderr[-500:])
        dq_us = statistics.median([float(line.split()[3]) for line in r.stdout.splitlines() if line.startswith("time ")][warm:])
        w16 = np.frombuffer((run / "out" / "dq.out").read_bytes(), "<u2", K * N, lay["OUT"]).reshape(K, N)
        if int((w16 != dequant_reference(q, s16, b16, bits).view("<u2")).sum()):
            raise SystemExit("dequant not bit-exact: nothing timed")
        for M in ms:
            _sp = role_spec(role, M)
            sg, gn, sk = _sp["simdgroups"], _sp["grid_n"], _sp["split_k"]
            g = work / ("gemm_q%d_%s_m%d" % (bits, role, M))
            if g.exists():
                shutil.rmtree(g)
            R.author_generic(g, role_spec(role, M))
            rng = np.random.default_rng(seed + M)
            (g / "a.f16").write_bytes(np.asarray(rng.standard_normal((M, K)), np.float16).astype("<f2").tobytes())
            (g / "b.f16").write_bytes(np.asarray(w16).tobytes())
            rep = R.run(g, queries=rounds + warm, composition="generic")
            if rep["status"] != "passed":
                raise SystemExit("gemm %s M %d not bit-exact: nothing timed" % (role, M))
            gemm_us = statistics.median([qq["gpu_seconds"] * 1e6 for qq in rep["queries"]][warm:])
            tf = 2.0 * M * N * K / ((dq_us + gemm_us) * 1e-6) / 1e12
            rows.append(dict(bits=bits, role=role, M=M, N=N, K=K, dequant_us=dq_us, gemm_us=gemm_us,
                             total_us=dq_us + gemm_us, tflops_total=tf, gemm_shape=dict(sg=sg, grid_n=gn, split_k=sk)))
            print("ours q%d %-3s M %4d: dequant %8.1f us  gemm %8.1f us  total %8.1f us  %5.2f TFLOP/s (gemm alone %5.2f)" % (
                bits, role, M, dq_us, gemm_us, dq_us + gemm_us, tf, 2.0 * M * N * K / (gemm_us * 1e-6) / 1e12), flush=True)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("verify", "bench"))
    ap.add_argument("--ms", default="128,256,512")
    ap.add_argument("--out")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--m", type=int, default=512)
    ap.add_argument("--roles", default="qkv,wo,w1,w2")
    ap.add_argument("--work", default=None)
    a = ap.parse_args(argv)
    work = a.work or os.path.join("/tmp", "g17qmm-%d" % os.getpid())
    if a.cmd == "bench":
        rows = bench(a.bits, [int(v) for v in a.ms.split(",")], a.roles.split(","), work)
        if a.out:
            with open(a.out, "w") as fh:
                json.dump(dict(clock="GPU (command buffer GPUEnd - GPUStart), medians", rows=rows), fh, indent=1)
        return
    rep = verify(a.bits, a.m, a.roles.split(","), work)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
