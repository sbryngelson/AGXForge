#!/usr/bin/env python3
"""Instruction-pair adjacency: does op B read op A's result correctly when it is the VERY NEXT
instruction? Checked for every ordered pair in each domain, bit-exact, on 32 distinct lanes.

The load-use hazard (a load's result read too early) and the read_sr hazard are measured and
repaired in cc.py. Whether an ALU result is ready for the instruction right after it is a
separate question, and every straight-line program this compiler emits already assumes yes. Here
it is asked directly: a chain x = B(A(B(A(x, y), y), y), y)... of LENGTH steps, each consuming the
previous one's result with nothing in between, for every ordered pair (A, B) of the scalar ops
tools/g17latency.py times - integer ops with integer ops, float ops with float ops.

The reference is exact: integer ops mod 2^32, float ops in exact rational arithmetic rounded once
to float32 (round-to-nearest-even), so a fused multiply-add is not double-rounded. Inputs differ
per lane, so a lane reading a stale register or a neighbour's value shows. Every program must
emit exactly LENGTH more instructions than its empty twin, so the pair really is adjacent.

    python3 tools/g17pairhazard.py            compile and check adjacency (offline)
    python3 tools/g17pairhazard.py --run      dispatch each pair, one process each, record
"""
import fractions, itertools, json, os, struct, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "spike", "accel", "re"))
import g17latency as LAT
T = 32
LENGTH = 16
M32 = 0xFFFFFFFF
OUT = os.path.join(ROOT, "isa", "g17-execution-pairs-results.json")
INTS = ["iadd", "iadd_imm", "imul", "shl_imm", "xor_imm", "icmp"]
FLOATS = ["fadd", "fmul", "ffma", "fmax"]
f2u = lambda v: struct.unpack("<I", struct.pack("<f", v))[0]
u2f = lambda u: struct.unpack("<f", struct.pack("<I", u & M32))[0]

# lane inputs: distinct words; floats in [1, 1.03) so a 16-step product or sum stays normal
INT_IN = [(t * 0x9E3779B1 + 12345) & M32 for t in range(T)]
FLOAT_IN = [f2u(1.0 + t / 1024.0) for t in range(T)]


def round_f32(q):
    """Exact rational -> the nearest float32 bits, ties to even. Normal range only (asserted)."""
    if q == 0:
        return 0
    sign = 0x80000000 if q < 0 else 0
    q = abs(q)
    e = q.numerator.bit_length() - q.denominator.bit_length()
    if fractions.Fraction(2) ** e > q:
        e -= 1
    assert -126 <= e <= 127, "outside the normal range: %r" % q
    m = q / fractions.Fraction(2) ** e * (1 << 23)          # in [2^23, 2^24)
    n, rem = divmod(m.numerator, m.denominator)
    if 2 * rem > m.denominator or (2 * rem == m.denominator and n & 1):
        n += 1
    if n == 1 << 24:
        n, e = 1 << 23, e + 1
    return sign | (e + 127) << 23 | (n - (1 << 23))


def F(u):
    return fractions.Fraction(u2f(u))


REF = {
    "iadd": lambda x, y: (x + y) & M32,
    "iadd_imm": lambda x, y: (x + 3) & M32,
    "imul": lambda x, y: (x * y) & M32,
    "shl_imm": lambda x, y: (x << 1) & M32,
    "xor_imm": lambda x, y: x ^ 0x55,
    "icmp": lambda x, y: int(x < y),
    "fadd": lambda x, y: round_f32(F(x) + F(y)),
    "fmul": lambda x, y: round_f32(F(x) * F(y)),
    "ffma": lambda x, y: round_f32(F(x) * F(y) + F(y)),
    "fmax": lambda x, y: x if u2f(x) > u2f(y) else y,
}


def pairs():
    return [(a, b) for dom in (INTS, FLOATS) for a, b in itertools.product(dom, dom)]


def build_ir(a, b, length):
    from agxforge.g17 import ir
    f = ir.Function("pair_%s_%s" % (a, b), [ir.Buffer("S", 1), ir.Buffer("O", 2)])
    bb = ir.Builder(f, f.block("entry"))
    t = bb.builtin("thread_position_in_grid", name="t")
    v = bb.load(f.buffers[0], t, name="v")
    y = bb.add(v, ir.Imm(0), name="y")
    x = bb.add(v, ir.Imm(0), name="x")
    for i in range(length):
        x = LAT.OPS[(a, b)[i % 2]][0](bb, ir, x, y)
    bb.store_at(f.buffers[1], t, x)
    bb.ret()
    return f


def reference(a, b, v):
    x = v
    for i in range(LENGTH):
        x = REF[(a, b)[i % 2]](x, v)
    return x


def compiled(a, b):
    import g17ref
    from agxforge.g17 import cc
    code = bytes(cc.compile_function(build_ir(a, b, LENGTH)).code)
    base = bytes(cc.compile_function(build_ir(a, b, 0)).code)
    grew = len(list(g17ref.walk(code, 0))) - len(list(g17ref.walk(base, 0)))
    if grew != LENGTH:
        raise SystemExit("REFUSED: %s,%s adds %d instructions, not %d - the pair is not adjacent"
                         % (a, b, grew, LENGTH))
    return code


def main(argv):
    rows, bad = [], []
    for a, b in pairs():
        code = compiled(a, b)
        ins = FLOAT_IN if a in FLOATS else INT_IN
        want = [reference(a, b, v) for v in ins]
        if "--run" not in argv:
            continue
        inp = os.path.join(ROOT, "isa", ".pair-inputs-%d.json" % os.getpid())
        json.dump(ins, open(inp, "w"))
        try:
            p = subprocess.run([sys.executable, os.path.join(HERE, "g17formlowerrun.py"), "--dispatch",
                                code.hex(), "file:" + inp], capture_output=True, text=True, timeout=120)
        finally:
            os.remove(inp)
        r = json.loads(p.stdout.strip().splitlines()[-1])
        out = r["out"][:T]
        ok = r["status"] == 0 and not r["untouched"] and out == want
        rows.append(dict(pair=[a, b], status="ok" if ok else "mismatch", cb_status=r["status"],
                         values=out, expect=want, program=code.hex()))
        if not ok:
            bad.append((a, b))
            print("%-9s -> %-9s MISMATCH on %d lanes" % (a, b, sum(o != w for o, w in zip(out, want))))
    n = len(pairs())
    if "--run" not in argv:
        print("%d ordered pairs compile adjacent (%d integer, %d float), %d steps each"
              % (n, len(INTS) ** 2, len(FLOATS) ** 2, LENGTH))
        return 0
    with open(OUT, "w") as fh:
        json.dump({"generator": "tools/g17pairhazard.py --run", "length": LENGTH, "lanes": T,
                   "rows": rows}, fh, indent=1, sort_keys=True)
        fh.write("\n")
    print("%d of %d ordered pairs read the previous result correctly%s"
          % (n - len(bad), n, "" if not bad else "; hazards: %s" % bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
