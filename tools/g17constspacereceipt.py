#!/usr/bin/env python3
"""A constant-address-space read, from Metal source to silicon: `o[t] = a[t] * c[t & 63]` with
`constant float *c [[buffer(2)]]`.

The capability gap (compiler.gap.mem.address_spaces) says what the constant space is on this GPU,
from Apple's compiler: the read lowers to the ORDINARY device load, op12682, with no
address-space-specific form. So the front end's constant pointer goes through the existing load
path, and what is missing is proof it runs. The source is the census's own arm
(tools/g17featurecensus.py ARMS["mem.address_spaces.constant"]), so the receipt is about exactly
the program the gap names, compiled by this compiler; its forms must be a subset of the forms
Apple's compilation of the same source carries (isa/g17-feature-census.json).

Three buffers, output at index 1: dispatched through ac_run_ps_rb, the runner that copies buffer 1
back. 32 lanes; `a` and `c` are distinct per lane and every product is exact in float32, so the
reference is plain arithmetic. One pipeline per process.

    python3 tools/g17constspacereceipt.py          compile and compare forms (offline)
    python3 tools/g17constspacereceipt.py --run    dispatch, retain, record
"""
import json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "spike", "accel", "re"))
ARM = "mem.address_spaces.constant"
T = 32
N = 64                      # buffers are N*N words
RETAINED = "g17-constant-space-runtime-v1"
OUT = os.path.join(ROOT, "isa", "g17-execution-constspace-results.json")
SENTINEL = 0xDEADBEEF


def source():
    import g17featurecensus as C
    return C.ARMS[ARM][1]


def function():
    """The census arm's source through the Metal toolchain's AIR and this project's front end."""
    import tempfile, g17air, g17front
    d = tempfile.mkdtemp(prefix="constspace-")
    p = os.path.join(d, "constant.metal")
    open(p, "w").write(source())
    return g17front.to_ir(g17air.air_of(p))            # air_of returns the AIR text itself


def compiled():
    import g17ref
    from agxforge.g17 import cc
    code = bytes(cc.compile_function(function()).code)
    ours = {"%d/%d" % (o, n) for _a, n, o in g17ref.walk(code, 0)}
    census = json.load(open(os.path.join(ROOT, "isa", "g17-feature-census.json")))
    apple = set(census["rows"][ARM]["forms"])
    return code, ours, apple


def inputs():
    import numpy as np
    a = np.zeros(N * N, np.float32)
    c = np.zeros(N * N, np.float32)
    a[:T] = [(t - 15) * 0.5 for t in range(T)]
    c[:64] = [1.0 + k * 0.25 for k in range(64)]
    return a, c


def reference(a, c):
    import numpy as np
    return [int(np.float32(a[t] * c[t & 63]).view(np.uint32)) for t in range(T)]


def _child(code_hex):
    import ctypes, numpy as np
    import g17program, g17oracle, g17endtoend as E2
    import g17imgconst_scalar as K
    code = bytes.fromhex(code_hex)
    text = bytes.fromhex("0e000000") + g17oracle.FILLER * ((g17oracle.ENTRY - 4) // 2) + code
    if len(text) % 16:
        text += g17oracle.FILLER * ((16 - len(text) % 16) // 2)
    P = g17program.G17Program(text=text, entry=g17oracle.ENTRY, buffers=[0, 1, 2], stats_md=K.STATS_MD)
    d = os.path.join(E2.SCRATCH, "constspace-%d" % os.getpid())
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
    code, ours, apple = compiled()
    print("forms %s\nApple's arm %s\nsubset of Apple's: %s" % (sorted(ours), sorted(apple), ours <= apple))
    if not ours <= apple:
        print("REFUSED: this compile carries forms Apple's does not: %s" % sorted(ours - apple))
        return 2
    if "--run" not in argv:
        return 0
    p = subprocess.run([sys.executable, os.path.abspath(__file__), "--child", code.hex()],
                       capture_output=True, text=True, timeout=120)
    r = json.loads(p.stdout.strip().splitlines()[-1])
    a, c = inputs()
    want = reference(a, c)
    match = [x == y for x, y in zip(r["out"], want)]
    ok = r["status"] == 0 and all(match) and r["beyond"] == 0
    print("status %s  %d of %d lanes = a[t] * c[t & 63]  (%d words written past the grid)"
          % (r["status"], sum(match), T, r["beyond"]))
    json.dump([dict(id="constspace.census_arm", status="ok" if ok else "mismatch", cb_status=r["status"],
                    finished=True, values=r["out"], expect=want, forms=sorted(ours), program=code.hex(),
                    note="the census's constant-space source through the front end and cc; op12682 is "
                         "the load, as in Apple's compilation")],
              open(OUT, "w"), indent=1, sort_keys=True)
    d = os.path.join(ROOT, "results", RETAINED)
    os.makedirs(os.path.join(d, "programs", "constant_space"), exist_ok=True)
    open(os.path.join(d, "programs", "constant_space", "program.bin"), "wb").write(code)
    with open(os.path.join(d, "campaign.json"), "w") as fh:
        json.dump(dict(status="passed" if ok else "failed", generator="tools/g17constspacereceipt.py --run",
                       source=source(), lanes=T, correct=sum(match), beyond_grid=r["beyond"]),
                  fh, indent=1, sort_keys=True)
        fh.write("\n")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--child":
        _child(sys.argv[2])
    else:
        sys.exit(main(sys.argv[1:]))
