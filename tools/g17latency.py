#!/usr/bin/env python3
"""Per-opcode latency and issue cost, from programs this compiler authors, as SLOPES.

Two arms per operation, each at several chain lengths R:

    latency   ONE dependent chain, x = op(x, y) R times, in ONE simdgroup (32 threads, one
              threadgroup). Nothing else is resident to hide the wait, so the slope of GPU time
              against R is the time from one result to the next instruction that reads it.
    issue     EIGHT independent chains interleaved, R ops in total, over 262,144 threads. The GPU
              is full, so the slope is what one more instruction costs when latency is hidden:
              reported per simdgroup-instruction across the whole GPU.

WHY AUTHORED PROGRAMS AND NOT METAL. tools/g17l2timing.py times Apple-compiled source, which is
right for a matrix op and wrong here: fast-math reassociates `x = x + y` chains into trees and
folds integer ones into a multiply, so the chain a latency measurement needs does not survive the
compiler. This compiler does not reassociate, and the emitted op count is still CHECKED - every
program is decoded and must hold exactly R instances of the op, or the row is refused.

Slopes, not absolutes: fixed dispatch overhead is the intercept and cancels. Each arm's fit
reports its intercept and its worst residual, and a row whose times do not rise with R is
reported as flat rather than as a cost. Nanoseconds, not cycles: the clock is not measured here.

Straight-line code only - no loop, no branch - in-bounds loads and stores, one pipeline per
process (ac_time_ps_es, spike/accel/accel.mm).

    python3 tools/g17latency.py                 compile and check counts (offline)
    python3 tools/g17latency.py --run [OP ...]  time every arm; --write records the table
    python3 tools/g17latency.py --reinterpret   re-derive the interpretation from the table
    python3 tools/g17latency.py --loop [--run [--write]]   the hot-cache loop arms (latency)

WHAT IT FOUND (interpret()): the one-simdgroup arms are FETCH-bound, so they do not measure
latency; the issue arm is the table.
"""
import json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "spike", "accel", "re"))
OUT = os.path.join(ROOT, "isa", "g17-latency.json")
RS = (16, 32, 64, 128)
CHAINS = 8
ISSUE_GRID = 8192             # threadgroups of 32 -> 262,144 threads
N_ISSUE = 512                 # buffers are N*N words: exactly one per thread
N_LAT = 64
REPS = 20
# THE FETCH CONTROL. The first run's one-simdgroup dependent slopes tracked instruction BYTES
# (10-12 byte ops ~22 ns, 14-byte ~28, the 16-byte ffma ~32), which is what a fetch-bound
# straight-line program looks like, not an ALU latency. "fetch" is the same program shape with
# eight INDEPENDENT chains in the same single simdgroup: equal bytes per op, no dependence. Only
# the difference between the two arms is latency the dependence exposes.
ARMS = ("latency", "fetch", "issue")
ONE_F = 0x3F800000            # 1.0f: float chains stay finite at every R

# name: (builder(b, ir, x, y) -> next x, float inputs?)
OPS = {
    "load":     (None, False),                   # the pointer chase, built in build_loop_ir only
    "iadd":     (lambda b, ir, x, y: b.add(x, y), False),
    "iadd_imm": (lambda b, ir, x, y: b.add(x, ir.Imm(3)), False),
    "imul":     (lambda b, ir, x, y: b.mul(x, y), False),
    "shl_imm":  (lambda b, ir, x, y: b.shl(x, ir.Imm(1)), False),
    "xor_imm":  (lambda b, ir, x, y: b.xor(x, ir.Imm(0x55)), False),
    "icmp":     (lambda b, ir, x, y: b.icmp(x, y, rel="ult"), False),
    "fadd":     (lambda b, ir, x, y: b.fadd(x, y), True),
    "fmul":     (lambda b, ir, x, y: b.fmul(x, y), True),
    "ffma":     (lambda b, ir, x, y: b.fma(x, y, y), True),
    "fmax":     (lambda b, ir, x, y: b._def("fmax", [x, y], ir.F32, None), True),
    # THE PREREGISTERED ROWS (agxforge/g17/schedmodel.PREDICTIONS): committed as predictions by
    # scheduling class before they were timed
    "msb":      (lambda b, ir, x, y: b._def("msb", [x], ir.I32, None), False),
    "reverse":  (lambda b, ir, x, y: b._def("reverse", [x], ir.I32, None), False),
    "sar_imm":  (lambda b, ir, x, y: b._def("sar", [x, ir.Imm(1)], ir.I32, None), False),
    "addsat":   (lambda b, ir, x, y: b._def("addsat", [x, y], ir.I32, None), False),
    "subsat":   (lambda b, ir, x, y: b._def("subsat", [x, y], ir.I32, None), False),
    # THE UNARY FLOAT ROWS (conversions, rounding, transcendental, saturate), timed like the rest
    "recip":    (lambda b, ir, x, y: b._def("recip", [x], ir.F32, None), True),
    "rsqrt":    (lambda b, ir, x, y: b._def("rsqrt", [x], ir.F32, None), True),
    "exp2":     (lambda b, ir, x, y: b._def("exp2", [x], ir.F32, None), True),
    "log2":     (lambda b, ir, x, y: b._def("log2", [x], ir.F32, None), True),
    "rint":     (lambda b, ir, x, y: b._def("rint", [x], ir.F32, None), True),
    "fsat":     (lambda b, ir, x, y: b._def("fsat", [x], ir.F32, None), True),
}


def build_ir(op, arm, r):
    from agxforge.g17 import ir
    fn, isf = OPS[op]
    f = ir.Function("lat_%s_%s_%d" % (op, arm, r), [ir.Buffer("S", 1), ir.Buffer("O", 2)])
    b = ir.Builder(f, f.block("entry"))
    t = b.builtin("thread_position_in_grid", name="t")
    v = b.load(f.buffers[0], t, name="v")
    y = b.add(v, ir.Imm(0), name="y")
    k = 1 if arm == "latency" else CHAINS          # "fetch" and "issue" interleave independent chains
    xs = [b.add(v, ir.Imm(0), name="x%d" % c) for c in range(k)]
    for i in range(r):
        c = i % k
        xs[c] = fn(b, ir, xs[c], y)
    acc = xs[0]
    for c in range(1, k):
        acc = b.xor(acc, xs[c])
    b.store_at(f.buffers[1], t, acc)
    b.ret()
    return f


# THE LOOP ARMS, for a hot instruction cache. The straight-line arms above are fetch-bound in one
# simdgroup, so the dependence hides under fetch. Here a body of B ops runs TRIPS times inside a
# counted loop (constant bound, proved by the compiler), so after the first trip the body is
# cached and the slope of GPU time against B, divided by TRIPS, is the cost of one op in the body.
# "loop_dep" chains every op on the previous one; "loop_indep" splits the same B ops over eight
# independent chains. Their difference is the latency the dependence exposes.
LOOP_ARMS = ("loop_dep", "loop_indep")
# THE UNARY, SHIFT AND SATURATING FORMS REACH ONLY THE NARROW FILE (sixteen registers, some pre-coloured), and eight
# loop-carried chains do not fit there. Four still hide a slow op: 4 chains x 2 issue intervals = 8
# intervals between dependent ops, against a measured slow latency of 4.8.
LOOP_CHAINS = {"msb": 4, "reverse": 4, "sar_imm": 4, "addsat": 4, "subsat": 4,
               "recip": 4, "rsqrt": 4, "exp2": 4, "log2": 4, "rint": 4, "fsat": 4}
LOOP_BS = (8, 16, 32, 64)


def chase_expected(nops, lanes=32):
    """Lane t's final value: S[t], then nops * TRIPS more hops of x = S[x], S[i] = 5i + 1 mod N_LAT^2."""
    size = N_LAT * N_LAT
    out = []
    for t in range(lanes):
        x = (5 * t + 1) % size
        for _ in range(nops * TRIPS):
            x = (5 * x + 1) % size
        out.append(x)
    return out


def arms_for(op):
    """A LOAD HAS NO INDEPENDENT ARM. Every load fills slot 7, and a wait on slot 7 waits for every
    load outstanding on it - the other chains' included - so 'eight independent chains' of waiting
    loads are partly serialised through the shared slot (measured: 34 ns/load, between the
    dependent 57 and the unwaited 9). The load's control is the same single chain WITHOUT its
    waits: then nothing waits, and the slope is the cost of issuing a load."""
    return ("loop_dep", "loop_unwaited") if op == "load" else LOOP_ARMS
TRIPS = 128


def build_loop_ir(op, arm, nops):
    from agxforge.g17 import ir
    fn, isf = OPS[op]
    f = ir.Function("lat_%s_%s_%d" % (op, arm, nops), [ir.Buffer("S", 1), ir.Buffer("O", 2)])
    pre, hdr, ex = f.block("pre"), f.block("header"), f.block("exit")
    b = ir.Builder(f, pre)
    t = b.builtin("thread_position_in_grid", name="t")
    v = b.load(f.buffers[0], t, name="v")
    y = b.add(v, ir.Imm(0), name="y")
    k = 1 if arm in ("loop_dep", "loop_unwaited") else LOOP_CHAINS.get(op, CHAINS)
    x0 = [b.add(v, ir.Imm(0), name="x%d" % c) for c in range(k)]
    z = b.const(0, name="z")
    b.br(hdr); b.at(hdr)
    i = b.phi(z, name="i")
    ph = [b.phi(x0[c], name="p%d" % c) for c in range(k)]
    xs = list(ph)
    for n in range(nops):
        c = n % k
        # A LOAD CHAIN reads the input buffer, which OPS' (b, ir, x, y) signature cannot reach:
        # x = S[x], a pointer chase. S holds 7 everywhere, so every address is in range and every
        # load hits the same cached line - the latency measured is a cached device load's.
        xs[c] = b.load(f.buffers[0], xs[c], name="ld%d" % n) if op == "load" else fn(b, ir, xs[c], y)
    for c in range(k):
        ir.Builder.phi_latch(ph[c], xs[c])
    nxt = b.add(i, ir.Imm(1), name="i_next")
    ir.Builder.phi_latch(i, nxt)
    b.br_cond(b.cmp(nxt, TRIPS, "lt", name="p"), hdr, ex)
    b.at(ex)
    acc = xs[0]
    for c in range(1, k):
        acc = b.xor(acc, xs[c])
    b.store_at(f.buffers[1], t, acc)
    b.ret()
    return f


def compiled_loop(op, arm, nops, opc):
    """Refused unless doubling the body adds exactly nops of the op and nothing else."""
    import collections, g17ref
    from agxforge.g17 import cc
    cnt = lambda c: collections.Counter(o for _a, _n, o in g17ref.walk(c, 0))
    a = bytes(cc.compile_function(build_loop_ir(op, arm, nops)).code)
    b2 = bytes(cc.compile_function(build_loop_ir(op, arm, 2 * nops)).code)
    if op == "load" and arm != "loop_unwaited":
        # THE CHAIN MUST WAIT, WITHOUT A COPY IN IT. cc's loads did not wait for a loaded index
        # (tools/g17indirectload.py: index 0 on every lane), so a chain compiled that way measures
        # nothing; the compiler's fix routes the index through a waiting copy, whose own latency
        # would be added to every step. Each chained load instead gets its predecessor's slot in its
        # own wait mask (tensorview.add_waits, loop back edge included), and the hazard check must
        # then come back empty.
        from agxforge.g17 import tensorview as TV
        a, b2 = TV.add_waits(a), TV.add_waits(b2)
    grew = cnt(b2)[opc] - cnt(a)[opc], sum(cnt(b2).values()) - sum(cnt(a).values())
    if grew != (nops, nops):
        raise SystemExit("REFUSED: %s %s B=%d->%d adds %d op%d and %d instructions" % (op, arm, nops, 2 * nops, grew[0], opc, grew[1]))
    back = [o for _a, _n, o in g17ref.walk(a, 0)].count(458)
    if back != 1:
        raise SystemExit("REFUSED: %s %s B=%d has %d back edges" % (op, arm, nops, back))
    return a


def loop_main(argv):
    ops = [x for x in argv if not x.startswith("--")] or list(OPS)
    rows = {}
    for op in ops:
        opc = opcode_of(op)
        codes = {(arm, n): compiled_loop(op, arm, n, opc) for arm in arms_for(op) for n in LOOP_BS}
        print("%-9s op%-5d  loop bodies compile, one back edge each" % (op, opc))
        if "--run" not in argv:
            continue
        row = {"opcode": opc}
        for arm in arms_for(op):
            ts = []
            for n in LOOP_BS:
                p = subprocess.run([sys.executable, os.path.abspath(__file__), "--child",
                                    codes[(arm, n)].hex(), "latency",
                                    "2" if op == "load" else ("1" if OPS[op][1] else "0")],
                                   capture_output=True, text=True, timeout=300)
                res = json.loads(p.stdout.strip().splitlines()[-1])
                if res["status"] != 0 or res["written"] != res["threads"]:
                    raise SystemExit("REFUSED: %s %s B=%d: %s" % (op, arm, n, res))
                if op == "load" and arm == "loop_dep" and res["values"] != chase_expected(n):
                    raise SystemExit("REFUSED: load chain B=%d did not make every hop: lanes wrong %d of %d"
                                     % (n, sum(a != b for a, b in zip(res["values"], chase_expected(n))),
                                        len(res["values"])))
                ts.append(res["seconds"])
            s_, c_, worst = fit(list(LOOP_BS), ts)
            row[arm] = dict(seconds=ts, ns_per_op=s_ * 1e9 / TRIPS, intercept_us=c_ * 1e6,
                            worst_residual_us=worst * 1e6)
            # A SLOPE IS A CLAIM ONLY IF THE TIMES RISE WITH THE BODY. The load chain's dependent arm
            # did not (120.6, 232.6, 456.8, 384.7 us one run; 152.0, 275.8, 223.5, 319.2 the next),
            # and fitting through such points printed four different "latencies" in four runs.
            if any(b <= a for a, b in zip(ts, ts[1:])):
                row[arm]["nonlinear"] = True
                print("  %-9s %s: times do not rise with the body (%s us): NO slope is claimed"
                      % (op, arm, ", ".join("%.1f" % (t * 1e6) for t in ts)))
        base = "loop_unwaited" if op == "load" else "loop_indep"
        row["exposed_ns"] = (None if row["loop_dep"].get("nonlinear") or row[base].get("nonlinear")
                             else row["loop_dep"]["ns_per_op"] - row[base]["ns_per_op"])
        from agxforge.g17 import schedmodel as SM
        if opc in SM.PREDICTIONS:
            unit = 3.105                     # the iadd issue interval the model is stated in
            got = (round(row["loop_dep"]["ns_per_op"] / unit), round(row["loop_indep"]["ns_per_op"] / unit))
            row["prediction"] = dict(predicted=list(SM.PREDICTIONS[opc][2]), measured_rounded=list(got),
                                     holds=tuple(got) == SM.PREDICTIONS[opc][2])
            print("  %-9s PREDICTED %s by class %d, measured %s: %s" % (op, SM.PREDICTIONS[opc][2],
                  SM.PREDICTIONS[opc][1], got, "HOLDS" if row["prediction"]["holds"] else "FAILS"))
        if row["exposed_ns"] is None:
            print("  %-9s in a hot loop: %s %.3f ns/op; dependent arm not linear - no latency claimed"
                  % (op, base, row[base]["ns_per_op"]))
        else:
            print("  %-9s in a hot loop: dependent %.3f  %s %.3f ns/op  => exposed %.3f ns"
                  % (op, row["loop_dep"]["ns_per_op"], base, row[base]["ns_per_op"], row["exposed_ns"]))
        rows[op] = row
    if "--write" in argv and rows:
        doc = json.load(open(OUT)) if os.path.exists(OUT) else {}
        prev = (doc.get("loop") or {}).get("rows", {})
        doc["loop"] = dict(generator="tools/g17latency.py --loop --run --write", bodies=list(LOOP_BS),
                           trips=TRIPS, chains=CHAINS, rows=dict(prev, **rows))
        doc["loop"]["interpretation"] = interpret_loop(doc["loop"])
        with open(OUT, "w") as fh:
            json.dump(doc, fh, indent=1, sort_keys=True)
            fh.write("\n")
        print("wrote %s (loop)" % os.path.relpath(OUT, ROOT))
    return 0


def opcode_of(op):
    """The opcode whose count scales with R: decoded, not assumed."""
    import collections, g17ref
    from agxforge.g17 import cc
    cs = []
    rs = (LOOP_BS[0], LOOP_BS[1]) if op == "load" else (RS[0], RS[1])
    for r in rs:
        fn = build_loop_ir(op, "loop_dep", r) if op == "load" else build_ir(op, "latency", r)
        code = bytes(cc.compile_function(fn).code)
        cs.append(collections.Counter(o for _a, _n, o in g17ref.walk(code, 0)))
    grew = {o: cs[1][o] - cs[0][o] for o in cs[1] if cs[1][o] - cs[0][o]}
    per = {o: d for o, d in grew.items() if d == rs[1] - rs[0]}
    if len(per) != 1:
        raise SystemExit("REFUSED: %s: no single opcode grows one per written op: %s" % (op, grew))
    return next(iter(per))


def compiled(op, arm, r, opc):
    """The program's bytes, refused unless it holds exactly R more of the op than the same arm
    with no chain - the loads' copies may be the same opcode, so the count is relative."""
    import collections, g17ref
    from agxforge.g17 import cc
    ops = lambda c: collections.Counter(o for _a, _n, o in g17ref.walk(c, 0))
    code = bytes(cc.compile_function(build_ir(op, arm, r)).code)
    base = ops(bytes(cc.compile_function(build_ir(op, arm, 0)).code))
    n, total = ops(code)[opc] - base[opc], sum(ops(code).values()) - sum(base.values())
    # and NOTHING ELSE grows: a copy riding along with each op would be timed as part of it
    if (n, total) != (r, r):
        raise SystemExit("REFUSED: %s %s R=%d adds %d op%d and %d instructions, not %d" % (op, arm, r, n, opc, total, r))
    return code


def _child(code_hex, arm, isf):
    """One pipeline, timed; prints JSON."""
    import ctypes, numpy as np
    import g17program, g17oracle, g17endtoend as E2
    import g17imgconst_scalar as K
    code = bytes.fromhex(code_hex)
    text = bytes.fromhex("0e000000") + g17oracle.FILLER * ((g17oracle.ENTRY - 4) // 2) + code
    if len(text) % 16:
        text += g17oracle.FILLER * ((16 - len(text) % 16) // 2)
    P = g17program.G17Program(text=text, entry=g17oracle.ENTRY, buffers=[1, 2], stats_md=K.STATS_MD)
    d = os.path.join(E2.SCRATCH, "latency-%d" % os.getpid())
    os.makedirs(d, exist_ok=True)
    open(d + "/k.arc", "wb").write(P.image()); open(d + "/k.lib", "wb").write(P.library())
    L = E2._lib()
    L.ac_time_ps_es.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_uint] * 5 + [ctypes.c_int,
                                                                              ctypes.POINTER(ctypes.c_double)]
    L.ac_time_ps_es.restype = ctypes.c_int
    assert L.ac_lib_from_data((d + "/k.lib").encode()) == 0
    ps = L.ac_pipeline_from_archive((d + "/k.arc").encode(), b"k")
    assert ps
    n = N_ISSUE if arm == "issue" else N_LAT
    # "busy": the latency program on 64 threadgroups. Each simdgroup's chain is still serial, so a
    # latency that holds only when the GPU is otherwise idle (a lowered clock) shows as a difference
    gw = ISSUE_GRID if arm == "issue" else (64 if arm == "busy" else 1)
    A = np.zeros(n * n, np.uint32)
    # "2": the POINTER-CHASE input - a permutation, S[i] = 5i + 1 mod n*n (5 is odd, so a
    # bijection), so the chain's final value says exactly how many hops ran. With a constant input
    # a chain that skipped hops or read a stale index would still end at the same value.
    B = ((np.arange(n * n, dtype=np.uint64) * 5 + 1) % (n * n)).astype(np.uint32) if isf == "2" \
        else np.full(n * n, ONE_F if isf == "1" else 7, np.uint32)
    C = np.full(n * n, 0xDEADBEEF, np.uint32)
    best = ctypes.c_double()
    st = L.ac_time_ps_es(ctypes.c_void_p(ps), A.ctypes.data, B.ctypes.data, C.ctypes.data,
                         n, 4, 32, gw, 1, REPS, ctypes.byref(best))
    threads = 32 * gw
    import hashlib
    print(json.dumps({"status": int(st), "seconds": best.value,
                      "written": int(sum(1 for v in C[:threads] if v != 0xDEADBEEF)), "threads": threads,
                      "values": [int(x) for x in C[:min(threads, 64)]],
                      "digest": hashlib.sha256(C[:threads].tobytes()).hexdigest()}))


def fit(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    s = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    c = my - s * mx
    return s, c, max(abs(y - (s * x + c)) for x, y in zip(xs, ys))


def interpret(rows):
    """What the recorded times do and do not show - a pure function of them, so it can be
    re-derived from the committed table (--reinterpret) without the GPU.

    MEASURED on 2026-09-23, first run and its control: the one-simdgroup dependent slope equals
    the one-simdgroup INDEPENDENT slope within noise for every op (differences -3.3..+5.5 ns, both
    signs), and both scale with instruction bytes. So a single simdgroup running straight-line code
    is bound by instruction FETCH (~2 ns per byte), and the dependence is hidden under it: those
    arms bound nothing about ALU latency. A latency needs a hot instruction cache - a short body
    repeated in a loop - which the capped runtime loops now make possible.

    The ISSUE arm is the table: 262,144 threads amortise fetch, its slopes reproduce across runs,
    and its ratios separate the ops."""
    ref = rows.get("iadd", {}).get("issue", {}).get("slope_ns")
    out = {"latency_measured": False,
           "why_not": "one-simdgroup dependent and independent chains cost the same within noise and "
                      "both scale with instruction bytes: fetch-bound, so the dependence is hidden",
           "issue_relative_to_iadd": {}}
    for op, r in sorted(rows.items()):
        if ref and "issue" in r:
            out["issue_relative_to_iadd"][op] = round(r["issue"]["slope_ns"] / ref, 3)
        r.pop("exposed_latency_ns", None)
        for arm in ("latency", "fetch"):
            if arm in r:
                r[arm]["means"] = "instruction fetch of a single simdgroup; not an ALU latency"
    return out


# THE SCHEDULER'S TEST. Two dependence chains of B/2 ops each, WRITTEN one after the other, in the
# same hot loop and one simdgroup: unscheduled, every link waits on the previous one; scheduled
# (agxforge/g17/sched.py), the two interleave. The outputs must be bit-identical - the pass reorders
# and adds nothing - and the time is the scheduler's measured effect.
SCHED_OPS = ("iadd", "imul", "fmul")


def build_sched_ir(op, nops, scheduled):
    from agxforge.g17 import ir, sched
    fn, isf = OPS[op]
    f = ir.Function("sched_%s_%d_%d" % (op, nops, scheduled), [ir.Buffer("S", 1), ir.Buffer("O", 2)])
    pre, hdr, ex = f.block("pre"), f.block("header"), f.block("exit")
    b = ir.Builder(f, pre)
    t = b.builtin("thread_position_in_grid", name="t")
    v = b.load(f.buffers[0], t, name="v")
    y = b.add(v, ir.Imm(0), name="y")
    x0 = [b.add(v, ir.Imm(c), name="x%d" % c) for c in range(2)]
    z = b.const(0, name="z")
    b.br(hdr); b.at(hdr)
    i = b.phi(z, name="i")
    ph = [b.phi(x0[c], name="p%d" % c) for c in range(2)]
    xs = list(ph)
    for c in range(2):                              # chain 0 entirely, then chain 1
        for _n in range(nops // 2):
            xs[c] = fn(b, ir, xs[c], y)
    for c in range(2):
        ir.Builder.phi_latch(ph[c], xs[c])
    nxt = b.add(i, ir.Imm(1), name="i_next")
    ir.Builder.phi_latch(i, nxt)
    b.br_cond(b.cmp(nxt, TRIPS, "lt", name="p"), hdr, ex)
    b.at(ex)
    b.store_at(f.buffers[1], t, b.xor(xs[0], xs[1]))
    b.ret()
    if scheduled:
        sched.schedule(f)
    return f


def sched_main(argv):
    from agxforge.g17 import cc
    rows = {}
    for op in [x for x in argv if not x.startswith("--")] or list(SCHED_OPS):
        codes = {(s_, n): bytes(cc.compile_function(build_sched_ir(op, n, s_)).code)
                 for s_ in (False, True) for n in LOOP_BS}
        same_len = all(len(codes[(False, n)]) == len(codes[(True, n)]) for n in LOOP_BS)
        differ = all(codes[(False, n)] != codes[(True, n)] for n in LOOP_BS)
        print("%-6s scheduled bodies: same length %s, reordered %s" % (op, same_len, differ))
        if not (same_len and differ):
            raise SystemExit("REFUSED: the scheduled program must be a reordering of the same instructions")
        if "--run" not in argv:
            continue
        row = {}
        for s_ in (False, True):
            ts, dig = [], []
            for n in LOOP_BS:
                p = subprocess.run([sys.executable, os.path.abspath(__file__), "--child",
                                    codes[(s_, n)].hex(), "latency", "1" if OPS[op][1] else "0"],
                                   capture_output=True, text=True, timeout=300)
                res = json.loads(p.stdout.strip().splitlines()[-1])
                if res["status"] != 0 or res["written"] != res["threads"]:
                    raise SystemExit("REFUSED: %s sched=%s B=%d: %s" % (op, s_, n, res))
                ts.append(res["seconds"]); dig.append(res["digest"])
            sl, c_, worst = fit(list(LOOP_BS), ts)
            row["scheduled" if s_ else "as_written"] = dict(seconds=ts, digests=dig, ns_per_op=sl * 1e9 / TRIPS,
                                                              worst_residual_us=worst * 1e6)
        row["identical_outputs"] = row["scheduled"]["digests"] == row["as_written"]["digests"]
        row["speedup"] = row["as_written"]["ns_per_op"] / row["scheduled"]["ns_per_op"]
        print("  %-6s as written %.3f ns/op, scheduled %.3f ns/op: %.2fx; outputs identical: %s"
              % (op, row["as_written"]["ns_per_op"], row["scheduled"]["ns_per_op"], row["speedup"],
                 row["identical_outputs"]))
        rows[op] = row
    if "--write" in argv and rows:
        doc = json.load(open(OUT))
        doc["scheduler"] = dict(generator="tools/g17latency.py --sched --run --write", bodies=list(LOOP_BS),
                                trips=TRIPS, rows=rows)
        with open(OUT, "w") as fh:
            json.dump(doc, fh, indent=1, sort_keys=True)
            fh.write("\n")
    return 0 if all(r["identical_outputs"] for r in rows.values()) else 1


def interpret_loop(loop):
    """The hot-loop arms, read: per op, dependent and independent ns, and both in units of the
    simple-op issue interval (iadd's independent cost), which is the unit a scheduler needs.

    MEASURED 2026-09-23: every single-cycle-class op (iadd, iadd-imm, xor, icmp, fadd, fmul, ffma,
    fmax) costs ~6.05 ns dependent and ~3.1 ns independent in one simdgroup - a result is ready two
    issue intervals after its producer issues. imul and shl cost ~14.9 dependent and ~6.1
    independent: half-rate issue and a ~4.8-interval latency. Every fit is linear to within 0.7 us
    on 20-135 us times. No wait bit is involved: the 52 adjacent pairs of
    ledger/g17-alu-pairs-read-the-previous-result.toml are correct, so ALU dependences are
    interlocked by the hardware and latency shows as time, never as a wrong value."""
    rows = loop["rows"]
    unit = rows["iadd"]["loop_indep"]["ns_per_op"]
    return {"latency_measured": True, "issue_interval_ns": round(unit, 3),
            "per_op": {op: dict(latency_ns=(None if r["loop_dep"].get("nonlinear") else round(r["loop_dep"]["ns_per_op"], 3)),
                                issue_ns=round(r[_base(r)]["ns_per_op"], 3),
                                issue_arm=_base(r),
                                latency_intervals=(None if r["loop_dep"].get("nonlinear") else round(r["loop_dep"]["ns_per_op"] / unit, 2)),
                                issue_intervals=round(r[_base(r)]["ns_per_op"] / unit, 2))
                       for op, r in sorted(rows.items())},
            "load": ("a cached device load (op12682, every lane one address) in a pointer chase whose "
                     "loads wait on their predecessor's slot: its latency is the load-to-use time; its "
                     "issue arm is the same chain WITHOUT waits (arms_for explains why a load has no "
                     "independent arm)") if "load" in rows else None}


def _base(row):
    return "loop_unwaited" if "loop_unwaited" in row else "loop_indep"


def main(argv):
    if "--loop" in argv:
        return loop_main(argv)
    if "--sched" in argv:
        return sched_main(argv)
    if "--reinterpret" in argv:
        doc = json.load(open(OUT))
        doc["interpretation"] = interpret(doc["rows"])
        if "loop" in doc:
            doc["loop"]["interpretation"] = interpret_loop(doc["loop"])
            # the straight-line verdict names the loop arms as where latency WAS measured
            doc["interpretation"]["where_latency_is"] = "loop.interpretation (a hot instruction cache)"
        with open(OUT, "w") as fh:
            json.dump(doc, fh, indent=1, sort_keys=True)
            fh.write("\n")
        print(json.dumps(doc["interpretation"], indent=1))
        return 0
    ops = [a for a in argv if not a.startswith("--")] or list(OPS)
    rows = {}
    for op in ops:
        opc = opcode_of(op)
        codes = {(arm, r): compiled(op, arm, r, opc) for arm in ARMS for r in RS}
        print("%-9s op%-5d  %s" % (op, opc, " ".join("%s/%d:%dB" % (a[0], r, len(c))
                                                       for (a, r), c in sorted(codes.items()))))
        if "--run" not in argv:
            continue
        row = {"opcode": opc}
        for arm in ARMS:
            ts = []
            for r in RS:
                p = subprocess.run([sys.executable, os.path.abspath(__file__), "--child",
                                    codes[(arm, r)].hex(), arm, "1" if OPS[op][1] else "0"],
                                   capture_output=True, text=True, timeout=300)
                res = json.loads(p.stdout.strip().splitlines()[-1])
                if res["status"] != 0 or res["written"] != res["threads"]:
                    raise SystemExit("REFUSED: %s %s R=%d: %s" % (op, arm, r, res))
                ts.append(res["seconds"])
            s, c, worst = fit(list(RS), ts)
            per = s * 1e9 / (ISSUE_GRID if arm == "issue" else 1)
            row[arm] = dict(seconds=ts, slope_ns=per, intercept_us=c * 1e6, worst_residual_us=worst * 1e6,
                            rises=ts[-1] > ts[0])
        print("  %-9s one simdgroup: dependent %.2f  independent %.2f ns/op (fetch);  issue %.5f ns/simdgroup-op GPU-wide"
              % (op, row["latency"]["slope_ns"], row["fetch"]["slope_ns"], row["issue"]["slope_ns"]))
        rows[op] = row
    if "--write" in argv and rows:
        with open(OUT, "w") as fh:
            json.dump({"generator": "tools/g17latency.py --run --write", "rs": list(RS), "chains": CHAINS,
                       "issue_threads": 32 * ISSUE_GRID, "reps": REPS, "interpretation": interpret(rows),
                       "rows": rows}, fh, indent=1, sort_keys=True)
            fh.write("\n")
        print("wrote %s" % os.path.relpath(OUT, ROOT))
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 4 and sys.argv[1] == "--child":
        _child(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        sys.exit(main(sys.argv[1:]))
