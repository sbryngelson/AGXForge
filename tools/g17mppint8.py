#!/usr/bin/env python3
"""Apple's own int8 GEMM against this project's: the public Metal Performance Primitives matmul2d at the four
InternLM2 prefill shapes of MM 25.161, timed the way that section timed ours.

MM 25.161 compares this project's exact int8 GEMM with MLX, which has no int8 matmul, so its baseline is MLX's
FP32 route. The baseline an Apple engineer would ask for is Apple's own: `matmul2d` compiles int8 x int8 -> int32 to
the same widening MMA (op10384, MM 25.40). This instrument gives Apple its best configuration. Phase 1 sweeps the
per-threadgroup output tile (TM x TN) and execution_simdgroups for each shape, with K dynamic; phase 2 re-times the
three fastest per shape with seven interleaved chains of 40 dispatches (tools/g17mppint8.m, the method of
tools/g17chainwarm.m: clock ramp, one command buffer per chain, B cache-warm). Every chain's C is compared byte for
byte with an int64 NumPy reference.

    python3 tools/g17mppint8.py OUT            build, sweep, re-time; writes OUT/mpp-int8.json
"""
import concurrent.futures as cf
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
SHAPES = [("wo", 256, 2048, 2048), ("qkv", 256, 4096, 2048), ("w1", 256, 8192, 2048), ("w2", 256, 2048, 8192)]
TILES = [16, 32, 64, 128]
SIMDGROUPS = [1, 2, 4, 8]
# this project's chained int8 times, cache-warm B (evidence/g17-three-threads-v1/int8-gemm-chained.json, MM 25.161)
OURS = os.path.join(ROOT, "evidence", "g17-three-threads-v1", "int8-gemm-chained.json")

HEAD = """#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal; using namespace mpp; using namespace mpp::tensor_ops;
kernel void %(fn)s(device int8_t *A [[buffer(0)]], device int8_t *B [[buffer(1)]], device int32_t *C [[buffer(2)]],
                   constant uint3 &dims [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]]) {
  int M = dims.x, N = dims.y, K = dims.z;
  tensor<device int8_t, dextents<int,2>, tensor_inline> tA(A, dextents<int,2>(K, M), array<int,2>{1, K});
  tensor<device int8_t, dextents<int,2>, tensor_inline> tB(B, dextents<int,2>(N, K), array<int,2>{1, N});
  tensor<device int32_t, dextents<int,2>, tensor_inline> tC(C, dextents<int,2>(N, M), array<int,2>{1, N});
  constexpr auto desc = matmul2d_descriptor(%(tm)d, %(tn)d, dynamic_length_v<int>, false, false, false,
                                            matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroups<%(sg)d>> op;
  auto sA = tA.slice(0, int(tg.y) * %(tm)d);
  auto sB = tB.slice(int(tg.x) * %(tn)d, 0);
  auto sC = tC.slice(int(tg.x) * %(tn)d, int(tg.y) * %(tm)d);
  op.run(sA, sB, sC);
}
"""


def build(out, tm, tn, sg):
    fn = "mm_%d_%d_%d" % (tm, tn, sg)
    src, air, lib = (os.path.join(out, "lib", fn + x) for x in (".metal", ".air", ".metallib"))
    open(src, "w").write(HEAD % dict(fn=fn, tm=tm, tn=tn, sg=sg))
    r = subprocess.run(["xcrun", "-sdk", "macosx", "metal", "-std=metal4.0", "-O3", "-c", src, "-o", air],
                       capture_output=True, text=True)
    if r.returncode:
        return fn, None, r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "compile failed"
    subprocess.run(["xcrun", "-sdk", "macosx", "metallib", air, "-o", lib], check=True, capture_output=True)
    return fn, lib, None


def data(out):
    rng = np.random.default_rng(161)
    files = {}
    for tag, M, N, K in SHAPES:
        a = rng.integers(-128, 128, size=(M, K), dtype=np.int8)
        b = rng.integers(-128, 128, size=(K, N), dtype=np.int8)
        c = a.astype(np.int64) @ b.astype(np.int64)
        assert np.abs(c).max() < 2 ** 31
        paths = [os.path.join(out, "data", "%s.%s" % (tag, x)) for x in ("a.i8", "b.i8", "c.i32")]
        a.tofile(paths[0]); b.tofile(paths[1]); c.astype("<i4").tofile(paths[2])
        files[tag] = paths
    return files


def run(binary, out, name, configs, chains, n):
    plan = os.path.join(out, name + "-plan.json")
    recs = os.path.join(out, name + "-records.json")
    json.dump({"chains": chains, "n": n, "seed": 161, "configs": configs}, open(plan, "w"), indent=1)
    subprocess.run([binary, plan, recs], check=True)
    return json.load(open(recs))


def summarize(records):
    by = {}
    for r in records["records"]:
        by.setdefault(r["tag"], []).append(r)
    return {t: {"per_dispatch_us_median": float(np.median([r["per_dispatch_us"] for r in rs])),
                "per_dispatch_us_range": [min(r["per_dispatch_us"] for r in rs), max(r["per_dispatch_us"] for r in rs)],
                "bit_exact_all": all(r["bit_exact"] for r in rs), "chains": len(rs)} for t, rs in by.items()}


def main(out):
    os.makedirs(os.path.join(out, "lib"), exist_ok=True)
    os.makedirs(os.path.join(out, "data"), exist_ok=True)
    binary = os.path.join(out, "g17mppint8")
    subprocess.run(["clang", "-fobjc-arc", "-O2", "-framework", "Foundation", "-framework", "Metal", "-I", TOOLS,
                    os.path.join(TOOLS, "g17mppint8.m"), "-o", binary], check=True)
    with cf.ThreadPoolExecutor(8) as ex:
        built = list(ex.map(lambda t: build(out, *t), [(tm, tn, sg) for tm in TILES for tn in TILES for sg in SIMDGROUPS]))
    libs = {fn: lib for fn, lib, _ in built if lib}
    refused = {fn: err for fn, _, err in built if err}
    print("built %d kernels, %d refused by the compiler" % (len(libs), len(refused)))
    files = data(out)

    def config(tag, M, N, K, fn):
        tm, tn, sg = map(int, fn.split("_")[1:])
        a, b, c = files[tag]
        return {"tag": "%s/%s" % (tag, fn), "lib": libs[fn], "fn": fn, "M": M, "N": N, "K": K,
                "tm": tm, "tn": tn, "sg": sg, "a": a, "b": b, "expected": c}

    sweep_cfg = [config(tag, M, N, K, fn) for tag, M, N, K in SHAPES for fn in sorted(libs)
                 if M % int(fn.split("_")[1]) == 0 and N % int(fn.split("_")[2]) == 0]
    sweep = summarize(run(binary, out, "sweep", sweep_cfg, chains=2, n=20))
    best = []
    for tag, M, N, K in SHAPES:
        rows = sorted((v["per_dispatch_us_median"], t) for t, v in sweep.items()
                      if t.startswith(tag + "/") and v["bit_exact_all"])
        best += [config(tag, M, N, K, t.split("/")[1]) for _, t in rows[:3]]
    final = summarize(run(binary, out, "final", best, chains=7, n=40))
    ours = json.load(open(OURS))["configs"]
    table = []
    for tag, M, N, K in SHAPES:
        cand = sorted((v["per_dispatch_us_median"], t) for t, v in final.items() if t.startswith(tag + "/"))
        us, t = cand[0]
        mine = ours["%s_int8_c1" % tag]["per_dispatch_us_median"]
        table.append({"shape": tag, "M": M, "N": N, "K": K, "apple_best": t.split("/")[1], "apple_us": round(us, 2),
                      "apple_tops": round(2 * M * N * K / us / 1e6, 1), "ours_us_mm25161": mine,
                      "apple_over_ours": round(us / mine, 3)})
    receipt = {"section": "MM 25.161 (Apple's matmul2d baseline)",
               "instrument": "tools/g17mppint8.py and tools/g17mppint8.m",
               "what": "Apple's MPP matmul2d int8 x int8 -> int32, K dynamic, best of a TM x TN x execution_simdgroups "
                       "sweep; chained timing as tools/g17chainwarm.m (cache-warm B); C compared byte for byte with an "
                       "int64 NumPy reference",
               "table": table, "final": final, "sweep": sweep, "refused_by_compiler": refused}
    json.dump(receipt, open(os.path.join(out, "mpp-int8.json"), "w"), indent=1)
    for r in table:
        print("%-4s Apple %-14s %8.2f us %5.1f TOPS | ours %7.2f us | Apple/ours %.2f" % (
            r["shape"], r["apple_best"], r["apple_us"], r["apple_tops"], r["ours_us_mm25161"], r["apple_over_ours"]))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
