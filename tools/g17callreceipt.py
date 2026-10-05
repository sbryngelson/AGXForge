#!/usr/bin/env python3
"""A function call from Metal source to silicon: `o[t] = f(a[t]) + f(a[t] * 2)` with
`__attribute__((noinline)) float f(float x) { return x * 1.5f + 1.0f; }`.

Apple keeps a noinline callee as a separate function and calls it with op450/10. This compiler
has no call/return lowering; its front end INLINES a single-block callee at each call site
(g17front.inline_calls), which is the source's meaning, and the receipt is that the program runs
and computes it. It is not a receipt for op450. 32 lanes, inputs multiples of 0.25 so every
product and sum is exact in float32. Output at buffer 1, dispatched through ac_run_ps_rb.

    python3 tools/g17callreceipt.py          compile (offline)
    python3 tools/g17callreceipt.py --run    dispatch, retain, record
"""
import json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "spike", "accel", "re"))
T = 32
N = 64
RETAINED = "g17-call-inline-runtime-v1"
OUT = os.path.join(ROOT, "isa", "g17-execution-callinline-results.json")
SENTINEL = 0xDEADBEEF
SOURCE = """#include <metal_stdlib>
using namespace metal;
__attribute__((noinline)) float f(float x) { return x * 1.5f + 1.0f; }
kernel void k(device const float *a [[buffer(0)]], device float *o [[buffer(1)]],
              uint t [[thread_position_in_grid]]) {
  o[t] = f(a[t]) + f(a[t] * 2.0f);
}
"""


def source():
    return SOURCE


def function():
    import tempfile, g17air, g17front
    d = tempfile.mkdtemp(prefix="callinline-")
    p = os.path.join(d, "call.metal")
    open(p, "w").write(source())
    air = g17air.air_of(p)
    assert "call fast float @_Z1ff" in air, "the source no longer keeps the call; nothing is tested"
    return g17front.to_ir(air)


def compiled():
    import g17ref
    from agxforge.g17 import cc
    code = bytes(cc.compile_function(function()).code)
    return code, {"%d/%d" % (o, n) for _a, n, o in g17ref.walk(code, 0)}, None


def inputs():
    import numpy as np
    a = np.zeros(N * N, np.float32)
    a[:T] = [(t - 15) * 0.25 for t in range(T)]
    return a, np.zeros(N * N, np.float32)


def reference(a, c):
    import numpy as np
    f = lambda x: np.float32(np.float32(x * np.float32(1.5)) + np.float32(1.0))
    return [int(np.float32(f(a[t]) + f(np.float32(a[t] * np.float32(2.0)))).view(np.uint32)) for t in range(T)]


def _child(code_hex):
    import ctypes, numpy as np
    import g17program, g17oracle, g17endtoend as E2
    import g17imgconst_scalar as K
    code = bytes.fromhex(code_hex)
    text = bytes.fromhex("0e000000") + g17oracle.FILLER * ((g17oracle.ENTRY - 4) // 2) + code
    if len(text) % 16:
        text += g17oracle.FILLER * ((16 - len(text) % 16) // 2)
    P = g17program.G17Program(text=text, entry=g17oracle.ENTRY, buffers=[0, 1], stats_md=K.STATS_MD)
    d = os.path.join(E2.SCRATCH, "callinline-%d" % os.getpid())
    os.makedirs(d, exist_ok=True)
    open(d + "/k.arc", "wb").write(P.image()); open(d + "/k.lib", "wb").write(P.library())
    L = E2._lib()
    L.ac_run_ps_rb.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_uint] * 5
    L.ac_run_ps_rb.restype = ctypes.c_int
    assert L.ac_lib_from_data((d + "/k.lib").encode()) == 0
    ps = L.ac_pipeline_from_archive((d + "/k.arc").encode(), b"k")
    assert ps
    a, c = inputs()
    o = np.full(N * N, SENTINEL, np.uint32)
    st = L.ac_run_ps_rb(ctypes.c_void_p(ps), a.ctypes.data, o.ctypes.data, c.ctypes.data, N, 4, T, 1, 1)
    print(json.dumps({"status": int(st), "out": [int(v) for v in o[:T]],
                      "beyond": int(sum(1 for v in o[T:] if v != SENTINEL))}))


def main(argv):
    code, ours, _ = compiled()
    print("forms %s (no op450: the call is inlined)" % sorted(ours))
    if "--run" not in argv:
        return 0
    p = subprocess.run([sys.executable, os.path.abspath(__file__), "--child", code.hex()],
                       capture_output=True, text=True, timeout=120)
    r = json.loads(p.stdout.strip().splitlines()[-1])
    a, c = inputs()
    want = reference(a, c)
    match = [x == y for x, y in zip(r["out"], want)]
    ok = r["status"] == 0 and all(match) and r["beyond"] == 0
    print("status %s  %d of %d lanes = f(a[t]) + f(2 a[t])  (%d words written past the grid)"
          % (r["status"], sum(match), T, r["beyond"]))
    json.dump([dict(id="callinline.noinline_f", status="ok" if ok else "mismatch", cb_status=r["status"],
                    finished=True, values=r["out"], expect=want, forms=sorted(ours), program=code.hex(),
                    note="a noinline single-block callee, inlined by the front end at both call sites")],
              open(OUT, "w"), indent=1, sort_keys=True)
    d = os.path.join(ROOT, "results", RETAINED)
    os.makedirs(os.path.join(d, "programs", "call_inline"), exist_ok=True)
    open(os.path.join(d, "programs", "call_inline", "program.bin"), "wb").write(code)
    with open(os.path.join(d, "campaign.json"), "w") as fh:
        json.dump(dict(status="passed" if ok else "failed", generator="tools/g17callreceipt.py --run",
                       source=source(), lanes=T, correct=sum(match), beyond_grid=r["beyond"]),
                  fh, indent=1, sort_keys=True)
        fh.write("\n")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--child":
        _child(sys.argv[2])
    else:
        sys.exit(main(sys.argv[1:]))
