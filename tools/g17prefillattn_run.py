#!/usr/bin/env python3
"""Hardware verification of the scalar prefill attention (MM 25.144.2): the append, checked against
rope_append_reference (every written K/V row and every q16 value; the rest of the region untouched), then the attention
fed the append's ACTUAL output region, checked against prefill_reference over a 0x7f sentinel.

    python3 tools/g17prefillattn_run.py --cap 272 --m 16 --p0 0 [--work DIR]
"""
import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import g17deliver as G  # noqa: E402
import g17prefillattn as P  # noqa: E402


def author(d, prog, a, b, c, lay):
    """g17deliver.author with placeholder inputs (its manifest transport cannot size prefill-scale buffers; g17bundlerun
    reads only the entry name from the manifest), then the real inputs written over them."""
    d = G.author(d, prog, bytes(256), bytes(256), bytes(256), lay)
    for name, data in (("a.f16", a), ("b.f16", b), ("c.f32", c)):
        (d / name).write_bytes(bytes(data))
    return d


def verify(cap, M, p0, work, seed=11, progs=None, out16=False, qb=0):
    lay = P.prefill_layout(cap, M, out16=out16)
    H, KVH, D = lay["heads"], lay["kv_heads"], lay["head_dim"]
    qkv, cos, sin, Kc, Vc = P.case(lay, M, p0, seed)
    want, q16, K, V = P.prefill_reference(lay, qkv, cos, sin, Kc, Vc, p0, M)
    a, b, c = P.io(lay, M, p0, qkv, cos, sin, Kc, Vc)
    pa, pt = progs or (P.build_prefill_append(lay), P.build_prefill_attn_qb(lay, qb) if qb else P.build_prefill_attn(lay))
    tag = "c%d_m%d_p%d_qb%d" % (cap, M, p0, qb)
    da = author(work / ("append_" + tag), pa, a, b, c, lay)
    out1 = G.dispatch([dict(tag="append_" + tag, dir=da, threads=M * KVH * 32, group=32, base=1, rounds=2)], work)["append_" + tag]
    reg = bytearray(out1[:len(c)])
    kc = np.frombuffer(bytes(reg), np.float16, KVH * cap * D, lay["KOFF"]).reshape(KVH, cap, D)
    vc = np.frombuffer(bytes(reg), np.float16, KVH * cap * D, lay["VOFF"]).reshape(KVH, cap, D)
    qg = np.frombuffer(bytes(reg), np.float16, M * H * D, lay["Q16"]).reshape(M, H, D)
    rows = [r for r in range(p0, min(p0 + M, cap))]
    diff_append = (int((kc[:, rows].view(np.uint16) != K[:, rows].astype(np.float16).view(np.uint16)).sum())
                   + int((vc[:, rows].view(np.uint16) != V[:, rows].astype(np.float16).view(np.uint16)).sum())
                   + int((qg.view(np.uint16) != q16.astype(np.float16).view(np.uint16)).sum()))
    # nothing outside the written rows, q16 and the output moved
    untouched = [r for r in range(cap) if r not in rows]
    diff_append += int((kc[:, untouched].view(np.uint16) != np.asarray(Kc, np.float16)[:, untouched].view(np.uint16)).sum())
    dt = author(work / ("attn_" + tag), pt, a, b, bytes(reg), lay)
    out2 = G.dispatch([dict(tag="attn_" + tag, dir=dt, threads=(M // qb if qb else M) * H * 1024, group=1024, base=0, rounds=2)], work)["attn_" + tag]
    if out16:
        got = np.frombuffer(out2, "<u2", M * H * D, lay["PATTN"])
        wv = want.view(np.uint16).reshape(-1)
        sentinel = int((got == 0x7f7f).sum())
    else:
        got = np.frombuffer(out2, "<u4", M * H * D, lay["PATTN"])
        wv = want.astype(np.float32).view(np.uint32).reshape(-1)
        sentinel = int((got == 0x7f7f7f7f).sum())
    diff_attn = int((got != wv).sum())
    same_cache = out2[lay["KOFF"]:lay["Q16"]] == bytes(reg[lay["KOFF"]:lay["Q16"]])
    return dict(cap=cap, M=M, p0=p0, append_differing=diff_append, attn_differing=diff_attn, sentinel_left=sentinel,
                cache_unchanged_by_attention=bool(same_cache), append_bytes=len(pa.code), attn_bytes=len(pt.code))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=272)
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--p0", type=int, default=0)
    ap.add_argument("--work")
    args = ap.parse_args(argv)
    work = Path(args.work or tempfile.mkdtemp(prefix="g17prefill_"))
    work.mkdir(parents=True, exist_ok=True)
    print(verify(args.cap, args.m, args.p0, work))


if __name__ == "__main__":
    main()
