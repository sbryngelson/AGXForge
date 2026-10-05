#!/usr/bin/env python3
"""A switch from Metal source to silicon: four arms on `s[t] & 3`, each computing a value.

Apple lowers a small switch to compare-and-branch with exec-mask updates. This front end
IF-CONVERTS a switch whose arms are pure single blocks (g17front._if_convert_switch): every arm is
computed and the phi becomes an `icmp eq` / `select` chain, lowered to op11372 and op11375. The
receipt is that the program runs and computes the switch; s[t] = t, so every arm and the default
run on eight lanes each. Inputs are multiples of 0.25, so every value is exact in float32.

    python3 tools/g17switchreceipt.py          compile (offline)
    python3 tools/g17switchreceipt.py --run    dispatch, retain, record
"""
import json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "spike", "accel", "re"))
T = 32
N = 64
RETAINED = "g17-switch-runtime-v1"
OUT = os.path.join(ROOT, "isa", "g17-execution-switch-results.json")
SENTINEL = 0xDEADBEEF
SOURCE = """#include <metal_stdlib>
using namespace metal;
kernel void k(device const float *a [[buffer(0)]], device float *o [[buffer(1)]],
              device const uint *s [[buffer(2)]], uint t [[thread_position_in_grid]]) {
  float v = a[t];
  switch (s[t] & 3u) {
    case 0: v = v * 2.0f; break;
    case 1: v = v + 3.0f; break;
    case 2: v = v * v; break;
    default: v = 0.5f;
  }
  o[t] = v;
}
"""


def source():
    return SOURCE


def function():
    import tempfile, g17air, g17front
    d = tempfile.mkdtemp(prefix="switch-")
    p = os.path.join(d, "switch.metal")
    open(p, "w").write(source())
    air = g17air.air_of(p)
    assert "switch i32" in air, "the source no longer compiles to a switch; nothing is tested"
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
    sel = np.zeros(N * N, np.uint32)
    sel[:T] = range(T)
    return a, sel


def reference(a, c):
    import numpy as np
    def arm(x, k):
        return [np.float32(x * np.float32(2.0)), np.float32(x + np.float32(3.0)), np.float32(x * x),
                np.float32(0.5)][k]
    return [int(np.float32(arm(a[t], int(c[t]) & 3)).view(np.uint32)) for t in range(T)]


def _child(code_hex):
    import ctypes, numpy as np
    import g17program, g17oracle, g17endtoend as E2
    import g17imgconst_scalar as K
    code = bytes.fromhex(code_hex)
    text = bytes.fromhex("0e000000") + g17oracle.FILLER * ((g17oracle.ENTRY - 4) // 2) + code
    if len(text) % 16:
        text += g17oracle.FILLER * ((16 - len(text) % 16) // 2)
    P = g17program.G17Program(text=text, entry=g17oracle.ENTRY, buffers=[0, 1, 2], stats_md=K.STATS_MD)
    d = os.path.join(E2.SCRATCH, "switch-%d" % os.getpid())
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
    print("forms %s (if-converted: op11372 and op11375, no branch)" % sorted(ours))
    if "--run" not in argv:
        return 0
    p = subprocess.run([sys.executable, os.path.abspath(__file__), "--child", code.hex()],
                       capture_output=True, text=True, timeout=120)
    r = json.loads(p.stdout.strip().splitlines()[-1])
    a, c = inputs()
    want = reference(a, c)
    match = [x == y for x, y in zip(r["out"], want)]
    ok = r["status"] == 0 and all(match) and r["beyond"] == 0
    print("status %s  %d of %d lanes = the switch  (%d words written past the grid)"
          % (r["status"], sum(match), T, r["beyond"]))
    json.dump([dict(id="switch.four_arms", status="ok" if ok else "mismatch", cb_status=r["status"],
                    finished=True, values=r["out"], expect=want, forms=sorted(ours), program=code.hex(),
                    note="a four-arm switch, if-converted by the front end to icmp/select")],
              open(OUT, "w"), indent=1, sort_keys=True)
    d = os.path.join(ROOT, "results", RETAINED)
    os.makedirs(os.path.join(d, "programs", "switch"), exist_ok=True)
    open(os.path.join(d, "programs", "switch", "program.bin"), "wb").write(code)
    with open(os.path.join(d, "campaign.json"), "w") as fh:
        json.dump(dict(status="passed" if ok else "failed", generator="tools/g17switchreceipt.py --run",
                       source=source(), lanes=T, correct=sum(match), beyond_grid=r["beyond"]),
                  fh, indent=1, sort_keys=True)
        fh.write("\n")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--child":
        _child(sys.argv[2])
    else:
        sys.exit(main(sys.argv[1:]))
