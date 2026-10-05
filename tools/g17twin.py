#!/usr/bin/env python3
"""The matched study (MM 25.211): would the decode graph's algorithms, fusion and execution structure run as fast if
Apple's compiler built the kernels? Each kernel of the InternLM2 q4 decode graph has a Metal twin in tools/twins/ -
the same threads, the same work partition and the same fp32 operation order - compiled by `xcrun metal`
(-fno-fast-math -ffp-contract=off). Our bundle and its twin run on the same buffers (tools/g17twinrun), and a twin
graph swaps bundles for twins with everything else unchanged (tools/g17decodegen reads twin.json).

    g17twin.py kernels --graph G --out DIR             qmv family and attention: bit compare + ABBA chained timing
    g17twin.py graph --graph G --kinds all|K,K --tag T --out DIR    writes graph_twin_T.json beside G
    g17twin.py e2e --ctx NAME=GRAPHDIR=IDS.json ... --out DIR       ours / twins / mlx-lm decode, alternating
"""
import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

TWINS = ROOT / "tools" / "twins"
FLAGS = ["-fno-fast-math", "-ffp-contract=off"]
KINDS = {"qkv": "qkv", "wo_res1": "wo_res1", "ffn": "ffn", "w2_res2": "w2_res2", "head.lm": "lm", "attn_fused": "attn",
         "attn_norm": "norm", "ffn_norm": "norm", "head.final_norm": "norm", "head.argmax_pass1": "argmax",
         "head.gen_step": "gen"}
QMV = ("qkv", "wo_res1", "ffn", "w2_res2", "lm")


def kind_of(name):
    return KINDS[re.sub(r"^L\d+\.", "", name)]


def metallib(src, out_dir, defs):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    air, lib = out_dir / "k.air", out_dir / "k.metallib"
    subprocess.run(["xcrun", "-sdk", "macosx", "metal", "-c", "-O3", *FLAGS, *["-D%s=%d" % kv for kv in defs.items()],
                    str(TWINS / src), "-o", str(air)], check=True, stderr=subprocess.DEVNULL)
    subprocess.run(["xcrun", "-sdk", "macosx", "metallib", str(air), "-o", str(lib)], check=True)
    return lib


def twin_spec(kind, lay):
    """(source, function, macros) of the twin of a delivered bundle with layout `lay`; refuses a form it does not twin."""
    if kind in QMV:
        if not (lay["wpt"] == 2 and lay["rows"] == 1 and lay.get("coalesced") and lay.get("ksplit") and lay["group"] == 64):
            raise ValueError("qmv twin: the cooperative split-K wpt 2 one-row coalesced form only")
        epi = {"add16": 1, "add32_to16": 2}.get(lay.get("res"), 3 if lay.get("swiglu") else 0)
        defs = dict(KQ=lay["Kq"], NOUT=lay["Nout"], SGS=lay["sgs"], SOFF=lay["S"], BOFF=lay["B"],
                    CHAINS=int(bool(lay.get("chains"))), EPI=epi)
        if lay.get("res"):
            defs["RESOFF"] = lay["RES"]
        if lay.get("swiglu"):
            defs["FFN"] = lay["ffn"]
        return "qmv_twin.metal", "qmv_twin", defs
    if kind == "attn":
        if not (lay.get("wide") and lay.get("bfly_merge") and lay.get("kvvec") and lay.get("attn32") and lay["slices"] == 32
                and not lay.get("qknorm") and not lay.get("keyblock") and not lay.get("nsum")):
            raise ValueError("attention twin: the wide bfly kvvec attn32 form only")
        return "attn_twin.metal", "attn_twin", dict(A_H=lay["heads"], A_KVH=lay["kv_heads"], A_CAP=lay["cap"],
                                                     A_KOFF=lay["KOFF"], A_VOFF=lay["VOFF"], A_OUTAT=lay["OUT_AT"],
                                                     A_COST=lay["COST"], A_SINT=lay["SINT"])
    if kind == "norm":
        if not (lay["d"] == 2048 and lay.get("out32") and lay.get("rs_seed")):
            raise ValueError("norm twin: d 2048, out32, the rsqrt seed")
        return "misc_twin.metal", "norm_twin", dict(N_IN_HALF=int(lay["in_dtype"] == "half"), N_X=lay["X"],
                                                     N_OUT=lay["OUT"], N_G=lay["G"])
    defs = dict(A_C=lay["C"], A_PL=lay["per_lane"], A_G=lay["G"], A_CAP=lay["cap"], A_D=lay["d"], A_RX16=lay["R_X16"],
                A_GEN=lay["GEN"], A_LOG=lay["LOG"])
    return "misc_twin.metal", "argmax1_twin" if kind == "argmax" else "gen_twin", defs


def _layout(bundle):
    return json.loads((Path(bundle) / "decodeop.json").read_text())["layout"]


def cmd_graph(a):
    g = json.loads(Path(a.graph).read_text())
    want = None if a.kinds == "all" else set(a.kinds.split(","))
    made = {}
    for d in g["dispatches"]:
        k = kind_of(d["name"])
        if want is not None and k not in want:
            continue
        key = (d["bundle"], k)
        if key not in made:
            src, fn, defs = twin_spec(k, _layout(d["bundle"]))
            tdir = Path(a.out).resolve() / "twins" / ("%s_%s" % (Path(d["bundle"]).name, k))
            lib = metallib(src, tdir, defs)
            (tdir / "twin.json").write_text(json.dumps(dict(metallib=str(lib), fn=fn)))
            made[key] = str(tdir)
        d["bundle"] = made[key]
    out = Path(a.graph).parent / ("graph_twin_%s.json" % a.tag)
    out.write_text(json.dumps(g))
    print(out, len(made), "twin variants")


def _run_arms(out, bundle, lib, fn, d, files, rotate, copies, n):
    out = Path(out)
    arms = [dict(kind="bundle", dir=str(Path(bundle).resolve()), dump={"0": str(out / "dump_ours.bin")}),
            dict(kind="apple", metallib=str(lib), fn=fn, dump={"0": str(out / "dump_apple.bin")})]
    spec = dict(arms=arms, threads=d["threads"], group=d["group"], buffers={k: dict(file=str(p)) for k, p in files.items()},
                rotate=rotate, copies=copies, chains=16, n=n, warm_s=0.6)
    (out / "spec.json").write_text(json.dumps(spec))
    subprocess.run([str(ROOT / "tools" / "g17twinrun"), str(out / "spec.json"), str(out / "run.json")], check=True)
    per = json.loads((out / "run.json").read_text())["per_dispatch_us"]
    return per, np.fromfile(out / "dump_ours.bin", np.uint8), np.fromfile(out / "dump_apple.bin", np.uint8)


def _qmv_inputs(out, kind, lay, d):
    rng = np.random.default_rng(11)
    K = lay["Kq"]
    rows = lay["Nout"]
    w2 = bytearray(d["binds"]["2"]["bytes"])
    raw = rng.integers(0, 2 ** 32, rows * K // 8, dtype=np.uint64).astype("<u4").tobytes()
    w2[0:len(raw)] = raw
    sc = (np.abs(rng.standard_normal(rows * K // 64)) * 0.01 + 1e-3).astype(np.float32).view(np.uint32) >> 16
    bi = (rng.standard_normal(rows * K // 64) * 0.05).astype(np.float32).view(np.uint32) >> 16
    w2[lay["S"]:lay["S"] + 2 * sc.size] = sc.astype("<u2").tobytes()
    w2[lay["B"]:lay["B"] + 2 * bi.size] = bi.astype("<u2").tobytes()
    b0 = bytearray(d["binds"]["0"]["bytes"])
    if lay.get("res"):
        N = lay["Nout"]
        h = rng.standard_normal(N).astype("<f4").tobytes(); b0[0:len(h)] = h
        x16 = rng.standard_normal(N).astype("<f2").tobytes(); b0[lay["RES"]:lay["RES"] + len(x16)] = x16
    files = {}
    for k, data in (("0", bytes(b0)), ("1", rng.standard_normal(K).astype("<f4").tobytes()), ("2", bytes(w2))):
        files[k] = Path(out) / ("in_%s.bin" % k)
        files[k].write_bytes(data)
    return files


def _attn_inputs(out, lay, d, q0):
    rng = np.random.default_rng(3 + q0)
    H, KVH, D, CAP = lay["heads"], lay["kv_heads"], lay["head_dim"], lay["cap"]
    b0 = bytearray(d["binds"]["0"]["bytes"])
    for off in (lay["KOFF"], lay["VOFF"]):
        c = np.zeros((KVH, CAP, D), np.float16)
        c[:, :q0] = (rng.standard_normal((KVH, q0, D)) * 1.5).astype(np.float16)
        raw = c.tobytes(); b0[off:off + len(raw)] = raw
    b2 = bytearray(d["binds"]["2"]["bytes"])
    b2[0:4] = np.uint32(q0).tobytes()
    ang = np.outer(np.arange(CAP), 1.0 / (1000000.0 ** (np.arange(0, D, 2) / D)))
    for off, tab in ((lay["COST"], np.cos(ang)), (lay["SINT"], np.sin(ang))):
        raw = tab.astype("<f4").tobytes(); b2[off:off + len(raw)] = raw
    files = {}
    for k, data in (("0", bytes(b0)), ("1", (rng.standard_normal((H + 2 * KVH) * D) * 2).astype("<f4").tobytes()), ("2", bytes(b2))):
        files[k] = Path(out) / ("ain_%s.bin" % k)
        files[k].write_bytes(data)
    return files


def cmd_kernels(a):
    g = json.loads(Path(a.graph).read_text())
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    first = {}
    for d in g["dispatches"]:
        first.setdefault(kind_of(d["name"]), d)
    res = dict(qmv={}, attn={})
    for kind in QMV:
        d = first[kind]
        lay = _layout(d["bundle"])
        src, fn, defs = twin_spec(kind, lay)
        lib = metallib(src, out / ("lib_" + kind), defs)
        files = _qmv_inputs(out, kind, lay, d)
        copies = max(2, -(-(96 << 20) // files["2"].stat().st_size))
        per, a0, a1 = _run_arms(out, d["bundle"], lib, fn, d, files, "2", copies, 48)
        nb = d["binds"]["0"]["bytes"]
        row = dict(ours_us=statistics.median(per[0]), apple_us=statistics.median(per[1]), chains=per,
                   bytes_differ=int((a0[:nb] != a1[:nb]).sum()), bytes=int(nb), bundle=Path(d["bundle"]).name, macros=defs)
        row["apple_over_ours"] = row["apple_us"] / row["ours_us"]
        res["qmv"][kind] = row
        print("%-8s ours %7.1f us  apple %7.1f us  apple/ours %.3f  bytes differ %d of %d" % (
            kind, row["ours_us"], row["apple_us"], row["apple_over_ours"], row["bytes_differ"], nb), flush=True)
    d = first["attn"]
    lay = _layout(d["bundle"])
    src, fn, defs = twin_spec("attn", lay)
    lib = metallib(src, out / "lib_attn", defs)
    H, D = lay["heads"], lay["head_dim"]
    for q0 in a.q0:
        files = _attn_inputs(out, lay, d, q0)
        per, a0, a1 = _run_arms(out, d["bundle"], lib, fn, d, files, "0", 4, 24)
        o0 = a0[lay["OUT_AT"]:lay["OUT_AT"] + H * D * 4]; o1 = a1[lay["OUT_AT"]:lay["OUT_AT"] + H * D * 4]
        kv = slice(lay["KOFF"], lay["VOFF"] + lay["kv_heads"] * lay["cap"] * D * 2)
        row = dict(ours_us=statistics.median(per[0]), apple_us=statistics.median(per[1]), chains=per,
                   out_bytes_differ=int((o0 != o1).sum()), kv_cache_identical=bool((a0[kv] == a1[kv]).all()),
                   out_finite=bool(np.isfinite(o0.view("<f4")).all()), bundle=Path(d["bundle"]).name)
        row["apple_over_ours"] = row["apple_us"] / row["ours_us"]
        res["attn"][str(q0)] = row
        print("attn q0 %4d  ours %6.1f us  apple %6.1f us  apple/ours %.3f  out bytes differ %d  kv identical %s" % (
            q0, row["ours_us"], row["apple_us"], row["apple_over_ours"], row["out_bytes_differ"], row["kv_cache_identical"]), flush=True)
    (out / "kernels.json").write_text(json.dumps(res, indent=1))


MLX_STEP = r'''
import json, sys, time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler
model, tok = load(sys.argv[1], tokenizer_config={"trust_remote_code": True})
ids = json.load(open(sys.argv[2])); n = int(sys.argv[3])
for rep in range(2):
    out, t0 = [], None
    for i, (t, _) in enumerate(generate_step(mx.array(ids), model, max_tokens=n, sampler=make_sampler(temp=0.0))):
        if i == 0: t0 = time.perf_counter()
        out.append(int(t))
    t1 = time.perf_counter()
print(json.dumps(dict(decode_tok_s=(len(out) - 1) / (t1 - t0), out_ids=out)))
'''


def cmd_e2e(a):
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    ctxs = [c.split("=") for c in a.ctx]
    arms = a.arms.split(",")
    gfile = dict(ours="graph.json", twin_all="graph_twin_all.json", twin_attn="graph_twin_attn.json",
                 twin_qmv="graph_twin_qmv.json")
    mlx_py = out / "mlx_step.py"
    mlx_py.write_text(MLX_STEP)
    rows = []
    for rep in range(a.reps):
        for name, gdir, ids in ctxs:
            for arm in (arms if rep % 2 == 0 else arms[::-1]):
                if not (arm in ("ours", "mlx") or (Path(gdir) / gfile.get(arm, arm + ".json")).exists()):
                    continue
                l0 = os.getloadavg()[0]
                if arm == "mlx":
                    p = subprocess.run(["python3", str(mlx_py), a.mlx_model, ids, str(a.tokens)], capture_output=True, text=True)
                    r = json.loads(p.stdout.strip().splitlines()[-1])
                    r = dict(tok_s=r["decode_tok_s"], out_ids=r["out_ids"])
                else:
                    o = out / ("dg_%s_%s.json" % (name, arm))
                    # an arm that names no known graph is a graph file stem in GRAPHDIR (graph_forced: the
                    # token log pre-filled with a shared history, so every arm decodes the same tokens)
                    p = subprocess.run([str(ROOT / "tools" / "g17decodegen"), str(Path(gdir) / gfile.get(arm, arm + ".json")), str(o), str(a.tokens)],
                                       cwd=str(ROOT), capture_output=True, text=True)
                    m = re.search(r"FULL ([\d.]+) ms/tok \(([\d.]+) tok/s\) \| GPU ([\d.]+)", p.stderr + p.stdout)
                    r = dict(tok_s=float(m.group(2)), ms_tok=float(m.group(1)), gpu_ms_tok=float(m.group(3)),
                             out_ids=json.loads(o.read_text())["out_ids"])
                r.update(rep=rep, ctx=name, arm=arm, load=l0, t=time.time())
                rows.append(r)
                print("rep %d %-6s %-10s %6.1f tok/s  load %.1f" % (rep, name, arm, r["tok_s"], l0), flush=True)
                (out / "e2e_rows.json").write_text(json.dumps(rows))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("kernels")
    k.add_argument("--graph", required=True)
    k.add_argument("--out", required=True)
    k.add_argument("--q0", type=int, nargs="+", default=[127, 1023, 1791])
    g = sub.add_parser("graph")
    g.add_argument("--graph", required=True)
    g.add_argument("--kinds", required=True)
    g.add_argument("--tag", required=True)
    g.add_argument("--out", required=True)
    e = sub.add_parser("e2e")
    e.add_argument("--ctx", nargs="+", required=True, help="NAME=GRAPHDIR=IDS.json")
    e.add_argument("--out", required=True)
    e.add_argument("--mlx-model", required=True)
    e.add_argument("--tokens", type=int, default=128)
    e.add_argument("--reps", type=int, default=4)
    e.add_argument("--arms", default="ours,twin_all,mlx,twin_attn,twin_qmv")
    a = ap.parse_args(argv)
    return {"kernels": cmd_kernels, "graph": cmd_graph, "e2e": cmd_e2e}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
