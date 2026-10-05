#!/usr/bin/env python3
"""The execution-mask family: measured field map, canonical encoder, and round-trip check.

Apple's conditional is PREDICATION, not branching. A single-sided `if` never emits a control-flow
instruction at all - the compiler predicates the store (op17229 instead of op17235) or if-converts
the whole thing to a `csel`. The mask machinery only appears once a region has two sides, side
effects at more than one level, or a loop.

Everything below was measured, not assumed. Shapes were compiled from source this file writes, so
the control flow is known exactly; fields were then recovered by flipping one bit at a time and
asking the decoder. Nothing here was executed - see UNPROVEN at the bottom for what that leaves.

THE FAMILY IS ONE ENCODING. `end`, the four exec kinds and the branches are the same instruction
word with different selector bits:

    b0.4 = 0                      -> `end`   (op684); every other field becomes its immediate
    b0.4 = 1, b2.4 = 0            -> exec
    b0.4 = 1, b2.4 = 1, kind != 0 -> branch
    b0.5 + 2*b0.6 = kind            0 if / 1 pop / 2 else / 3 while

    exec kinds   if   op582 (b2.1=0) op583 (b2.1=1)
                 else op575          op576
                 pop  op577          op577      (b2.1 is dead here)
                 whil op578          op579
    branch kinds pop-slot -> op462   else-slot -> op458   while-slot -> op450

FIELDS (4-byte exec form):

    count       b0.7 (high) and b2.3 (low), and the two bits are a CODE TABLE, not an integer:
                    00 -> 1     01 -> 0     10 -> 2     11 -> 3
                An encoder that writes the count as a plain 2-bit integer emits `pop 0` when it
                means `pop 1`. Encoding 00 is all-zeros for the commonest value, which is why.
    predicate   b1[4:6], an 8-entry LOOKUP TABLE - [74,75,76,77,78,79,3,2]. Indices 0-5 are the
                six predicate registers; 6 and 7 cross into a different register file, so a
                linear base+index*slope model is wrong for exactly the two entries Apple uses to
                push a loop nesting level unconditionally. b1[0:3] and b1.7 are dead.
    invert      b2.1. `if (x == 0)` and `if (x != 0)` compile to the SAME compare (cc=12) and
                differ only in this bit, so it inverts the predicate rather than selecting a
                comparison.
    displace    branches only: signed two's complement, sign at b9.6, weights 2..2^47 scattered
                over bytes 0-9 in the order below. No weight-1 bit - displacements are even.
                Target is measured as `own offset + displacement`, verified on 65 of 65 branches.

WHAT THE COUNTS MEAN. Nesting depth 1..6 with a store at every level emits one `if count=1` per
level and pops that sum to exactly the number of pushes (6 levels -> pop 2, pop 2, pop 2). `else`
and `while` are net-neutral. Apple never emits count 3 even where it would fit, and never a
single pop deeper than 2, though the field encodes both - so 3 is mapped, not witnessed.

    python3 tools/g17cf.py --build     compile the control-flow probe corpus
    python3 tools/g17cf.py --show      per-probe control-flow listing
    python3 tools/g17cf.py --verify    re-encode every instruction the probe corpus emitted
    python3 tools/g17cf.py --safety    which of these probes must never be dispatched
    python3 tools/g17cf.py --corpus    the same against all 14,195 in isa/g17-corpus-programs.jsonl
"""
import os, subprocess, sys, tempfile
# siblings come from the package
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agxforge.g17 import corpus as g17corpus, metal as g17metal, slice as g17slice

KIND = {"if": 0, "pop": 1, "else": 2, "while": 3}
# The three branch opcodes, named once. A backward displacement on any of them is what makes a
# program loop - which decides a metadata slot, and which is also what makes a kernel unsafe to
# dispatch with foreign buffer contents.
BRANCH_OPCODES = {450, 458, 462}
COUNT_CODE = {1: 0, 0: 1, 2: 2, 3: 3}          # value -> the two-bit field, measured
PRED_TABLE = [74, 75, 76, 77, 78, 79, 3, 2]     # b1[4:6] -> printed register
# THE AUXILIARY FIELD. Operand 0 of the exec family, and the one part of it nobody understands.
# Across 14,195 real control-flow instructions it takes only a handful of values per kind - three
# for `pop`, three for `else` - so it is a small witnessed enumeration, not an address, despite
# printing as one. It is carried explicitly with a measured default rather than pretended away:
# an assembler that hard-codes one value writes an instruction Apple never emitted 4.6% of the
# time, which is exactly what the probe corpus was too narrow to show.
AUX_BITS = [(1, 0), (1, 1), (2, 5), (2, 6), (2, 7), (3, 5), (3, 7), (4, 5), (4, 6)]
AUX_DEFAULT = {"if": 0, "pop": 11, "else": 11, "while": 0}

DISP_BITS = [(0, 7), (1, 4), (1, 5), (1, 6), (2, 1), (2, 3), (2, 5), (2, 6), (2, 7),
             (1, 0), (1, 1), (1, 2), (1, 3), (3, 4), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4),
             (6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7),
             (7, 0), (7, 1), (7, 2), (7, 6), (7, 7),
             (8, 0), (8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (8, 6), (8, 7),
             (9, 0), (9, 1), (9, 2), (9, 3), (9, 4), (9, 5), (9, 6)]   # weight 2<<i, last = sign


def _put(buf, byte, bit, v):
    if v:
        buf[byte] |= 1 << bit
    else:
        buf[byte] &= ~(1 << bit)


def encode_end():
    """op684. b0.4 clear is what makes it `end`; the rest of the word is its immediate, zero."""
    return bytes([0x0e, 0x00, 0x00, 0x00])


def encode_exec(kind, count=1, pred=74, invert=False, aux=None):
    """A 4-byte exec-mask instruction built from the field map, no witness bits inherited."""
    if count not in COUNT_CODE:
        raise ValueError("count %r outside the two-bit field" % count)
    b = bytearray([0x1e, 0x00, 0x00, 0x0e])     # length + the bits mutation proved forced
    k = KIND[kind]
    _put(b, 0, 5, k & 1)
    _put(b, 0, 6, (k >> 1) & 1)
    code = COUNT_CODE[count]
    _put(b, 0, 7, (code >> 1) & 1)
    _put(b, 2, 3, code & 1)
    a = AUX_DEFAULT[kind] if aux is None else aux
    for j, (by, bit) in enumerate(AUX_BITS):
        if by < len(b):
            _put(b, by, bit, (a >> j) & 1)
    if kind != "pop":                            # pop reads no predicate; leave the field dead
        if pred not in PRED_TABLE:
            raise ValueError("register %r is not one of %s" % (pred, PRED_TABLE))
        idx = PRED_TABLE.index(pred)
        for j in range(3):
            _put(b, 1, 4 + j, (idx >> j) & 1)
        _put(b, 2, 1, 1 if invert else 0)
    return bytes(b)


def aux_of(raw):
    """The auxiliary field's value as Apple wrote it."""
    v = 0
    for j, (b, i) in enumerate(AUX_BITS):
        if b < len(raw) and (raw[b] >> i) & 1:
            v |= 1 << j
    return v


def encode_branch(kind, disp, aux=0):
    """A 10-byte branch. `disp` is relative to the branch's own offset and must be even."""
    if kind == "if":
        raise ValueError("kind 0 with b2.4 set is not a branch")
    if disp % 2:
        raise ValueError("displacement %d is odd; the field has no weight-1 bit" % disp)
    b = bytearray(10)
    b[0] = 0x1e
    b[2] = 0x01                                  # b2.0 is forced set in the 10-byte form
    b[3] = 0x0e
    _put(b, 2, 4, 1)                             # b2.4 is what turns exec into branch
    k = KIND[kind]
    _put(b, 0, 5, k & 1)
    _put(b, 0, 6, (k >> 1) & 1)
    for j, (by, bit) in enumerate(AUX_BITS):
        if by < len(b) and not any(by == db and bit == di for db, di in DISP_BITS):
            _put(b, by, bit, (aux >> j) & 1)
    v = disp >> 1
    if not -(1 << 46) <= v < (1 << 46):
        raise ValueError("displacement %d out of range" % disp)
    v &= (1 << 47) - 1
    for i, (by, bit) in enumerate(DISP_BITS):
        _put(b, by, bit, (v >> i) & 1)
    return bytes(b)


# THESE PROBES ARE FOR DECODING. Several must never be dispatched, and on 2026-09-05 six of them
# were, through a harness that feeds every kernel the same synthetic buffers - the GPU firmware
# workloop wedged, WindowServer watchdogged twice, and the machine rebooted. I wrote these
# kernels to make Apple's compiler EMIT control flow so I could read the encoding; I never
# intended them to run, and I did not say so anywhere, which is the defect.
#
# The trip count is the whole issue. `for (j = 0; j < u[i]; ++j)` with an arbitrary buffer is up
# to 2^32 iterations, and loop2/loop3 nest that two and three deep - u[i] * u[i+1] * u[i+2]
# iterations, which is not a slow kernel but a permanently wedged GPU.
#
#   unbounded   trip count comes from buffer data and has no ceiling
#   bounded     contains a loop, but one that terminates regardless of input
#   safe        no backward branch at all
DISPATCH = {
    "loop_if":   ("unbounded", "for j < u[i]"),
    "loopbrk":   ("unbounded", "for j < u[i], break inside"),
    "loop2":     ("unbounded", "u[i] * u[i+1] iterations"),
    "loop3":     ("unbounded", "u[i] * u[i+1] * u[i+2] iterations"),
    "while_div": ("bounded",   "while (j > 1) j >>= 1 - at most 32 iterations for any input"),
}


def safety():
    """What each probe does to a GPU that runs it. Sources are in probes(), so this is ground
    truth rather than inference: I wrote them."""
    for name in sorted(probes()):
        verdict, why = DISPATCH.get(name, ("safe", "no backward branch"))
        print("  %-11s %-10s %s" % (name, verdict, why))
    bad = [n for n, (v, _) in DISPATCH.items() if v == "unbounded"]
    print("\n  NEVER DISPATCH: %s" % ", ".join(sorted(bad)))
    print("  A guard keyed on the loop-back edge - any branch with a NEGATIVE displacement -")
    print("  catches every one of them, and is structural rather than an idiom that can change.")


HEAD = """#include <metal_stdlib>
using namespace metal;
kernel void k(device uint *u [[buffer(0)]], device int *s [[buffer(2)]],
              constant uint &n [[buffer(1)]], uint3 tp [[thread_position_in_threadgroup]]) {
  uint i = tp.x;
"""


def probes():
    """Shapes chosen so the control flow in the source is unambiguous."""
    P, B = {}, " { u[500+i]=1u; u[600+i]=2u; }\n"
    for d in range(1, 7):                        # nesting depth, side effect at every level
        s = ""
        for k in range(d):
            s += "  " * (k + 1) + "if (u[i+%d] > %du) {\n" % (k, k * 3)
            s += "  " * (k + 2) + "u[500+%d*64+i] = %du;\n" % (k, k + 1)
        P["deep%d" % d] = s + "".join("  " * k + "}\n" for k in range(d, 0, -1))
    P["seq3"] = "".join("  if (u[i]>%du){u[%d+i]=%du;}\n" % (k + 1, 500 + 100 * k, k + 1)
                        for k in range(3))
    P["ifelse"] = "  if (u[i] > 3u) { u[500+i]=1u; } else { u[500+i]=2u; }\n"
    P["elsedeep2"] = ("  if (u[i] > 3u) { u[500+i]=1u; if (u[i]>7u) { u[600+i]=2u; }\n"
                      "    else { u[600+i]=3u; } } else { u[500+i]=4u; }\n")
    P["while_div"] = "  { uint j=u[i]; uint a=0; while (j>1u){ a+=j; j>>=1; } u[500+i]=a; }\n"
    P["loop_if"] = "  for (uint j=0;j<u[i];++j){ if ((j&1u)==0u) { u[500+i]+=j; } }\n"
    P["loop2"] = "  for (uint j=0;j<u[i];++j) for (uint m=0;m<u[i+1];++m) u[500+i]+=m;\n"
    P["loop3"] = ("  for (uint j=0;j<u[i];++j) for (uint m=0;m<u[i+1];++m)\n"
                  "    for (uint q=0;q<u[i+2];++q) u[500+i]+=q;\n")
    P["loopbrk"] = "  for (uint j=0;j<u[i];++j){ if (u[i+1]==j) break; u[500+i]+=j; }\n"
    P["switch4"] = ("  switch (u[i] & 3u) { case 0: u[500+i]=1u; break; case 1: u[500+i]=2u;\n"
                    "    break; case 2: u[500+i]=3u; break; default: u[500+i]=4u; }\n")
    for nm, c in [("ugt", "u[i] >  3u"), ("ule", "u[i] <= 3u"), ("une", "u[i] != 0u"),
                  ("ueq", "u[i] == 0u"), ("ult", "u[i] <  3u"), ("uge", "u[i] >= 3u"),
                  ("sgt", "s[i] >  3"), ("sle", "s[i] <= 3"), ("slt", "s[i] <  3"),
                  ("sge", "s[i] >= 3")]:
        P["pol_" + nm] = "  if (" + c + ")" + B
    return P


def build():
    ok = 0
    for name, body in sorted(probes().items()):
        r = g17corpus.build("cf-" + name, HEAD + body + "}\n")
        if r in ("built", "cached"):
            ok += 1
        else:
            print("   FAIL %-12s %s" % (name, r[:70]))
    print("control-flow probes built or cached: %d of %d" % (ok, len(probes())))


def rows(tag):
    from agxforge.g17 import machobj, agxdis, ref as g17ref
    g17ref.binary()
    d = os.path.join(g17metal.CACHE, tag)
    if not os.path.exists(d + "/out/object/0-0"):
        return []
    loc = machobj.locate(d + "/s.arc.metallib", d + "/out/object/0-0")
    fo, sz = agxdis.sections(loc["obj"])
    t = bytes(loc["obj"][fo:fo + sz])
    e = loc["syms"]["_agc.main"]
    with tempfile.NamedTemporaryFile(suffix=".bin") as fh:
        fh.write(t); fh.flush()
        r = subprocess.run([g17metal.DIS, fh.name, str(e), str(len(t) - e), "--pc", str(e),
                            "--expr"], capture_output=True, text=True)
    out = []
    for line in r.stdout.splitlines():
        p = line.split()
        if len(p) < 3 or p[1] == "bad":
            continue
        off = int(p[0], 16)
        out.append((off, int(p[2]), p[3:], t[off:off + int(p[1])]))
    return out


def _decode(raw):
    """What each emitted instruction says it is, read back through the field map."""
    kind = ["if", "pop", "else", "while"][((raw[0] >> 5) & 1) | (((raw[0] >> 6) & 1) << 1)]
    code = (((raw[0] >> 7) & 1) << 1) | ((raw[2] >> 3) & 1)
    count = {v: k for k, v in COUNT_CODE.items()}[code]
    pred = PRED_TABLE[(raw[1] >> 4) & 7]
    return kind, count, pred, bool((raw[2] >> 1) & 1), aux_of(raw)


def verify():
    """Rebuild every control-flow instruction Apple emitted, from the spec and nothing else."""
    names = g17slice.KNOWN_OPS
    tags = sorted(d for d in os.listdir(g17metal.CACHE) if d.startswith("cf-"))
    ok = bad = 0
    fails = []
    for tag in tags:
        rs = rows(tag)
        offs = {r[0] for r in rs}
        for off, op, ops, raw in rs:
            nm = names.get(op, "")
            if nm == "end" and len(raw) == 4:
                mine = encode_end()
            elif nm == "exec" and len(raw) == 4:
                k, c, p, inv, a = _decode(raw)
                mine = encode_exec(k, c, p, inv, a)
            elif nm == "branch" and len(raw) == 10:
                tgt = [t for t in offs if t == off + _disp(raw)]
                mine = encode_branch(["if", "pop", "else", "while"][
                    ((raw[0] >> 5) & 1) | (((raw[0] >> 6) & 1) << 1)], _disp(raw))
                if not tgt:
                    fails.append((tag, off, "branch target %d not an instruction" % (off + _disp(raw))))
            else:
                continue
            if mine == raw:
                ok += 1
            else:
                bad += 1
                fails.append((tag, off, "%s -> %s" % (raw.hex(), mine.hex())))
    print("canonical re-encode of Apple's own control flow: %d byte-exact, %d differ" % (ok, bad))
    for f in fails[:12]:
        print("   %-14s @%04x  %s" % (f[0], f[1], f[2]))
    return bad == 0


def _disp(raw):
    v = 0
    for i, (by, bit) in enumerate(DISP_BITS):
        v |= ((raw[by] >> bit) & 1) << i
    if v >> 46:
        v -= 1 << 47
    return v << 1


def show():
    names = g17slice.KNOWN_OPS
    for tag in sorted(d for d in os.listdir(g17metal.CACHE) if d.startswith("cf-")):
        rs = [r for r in rows(tag) if names.get(r[1], "") in ("exec", "branch", "end")]
        if not rs:
            continue
        print("=== %s" % tag[3:])
        for off, op, ops, raw in rs:
            nm = names.get(op, "")
            if nm == "exec":
                k, c, p, inv, a = _decode(raw)
                print("   %04x  %-6s count=%d pred=r%-3d%s aux=%-3d %s"
                      % (off, k, c, p, " INV" if inv else "    ", a, raw.hex()))
            elif nm == "branch":
                print("   %04x  branch %+d -> %04x            %s"
                      % (off, _disp(raw), off + _disp(raw), raw.hex()))
            else:
                print("   %04x  end                          %s" % (off, raw.hex()))
        print("")


def corpus():
    """The real test. The probe corpus is 284 instructions I caused to exist by writing the
    source, so it cannot show a form my probes never provoke - and it did not: measured against
    every control-flow instruction Apple actually ships, the encoder was 94.34% before the
    auxiliary field was carried explicitly."""
    import json
    path = os.path.join(ROOT, "isa", "g17-corpus-programs.jsonl")
    EXEC = {573, 574, 575, 576, 577, 578, 579, 582, 583}
    BR = BRANCH_OPCODES
    ok = bad = 0
    for line in open(path):
        d = json.loads(line)
        code = bytes.fromhex(d["text"])
        for o, l, op in d["spans"]:
            raw = code[o:o + l]
            if len(raw) != l:
                continue
            if op == 684 and l == 4:
                mine = encode_end()
            elif op in EXEC and l == 4:
                k, c, p, inv, a = _decode(raw)
                mine = encode_exec(k, c, p, inv, a)
            elif op in BR and l == 10:
                k = ["if", "pop", "else", "while"][((raw[0] >> 5) & 1) | (((raw[0] >> 6) & 1) << 1)]
                mine = encode_branch(k, _disp(raw), aux_of(raw))
            else:
                continue
            if mine == raw:
                ok += 1
            else:
                bad += 1
    print("every control-flow instruction Apple ships: %d byte-exact, %d differ (%.2f%%)"
          % (ok, bad, 100.0 * ok / (ok + bad) if ok + bad else 0))
    return bad == 0


def main(argv=None):
    """The command-line entry, callable. It was inline under `if __name__`, so after the
    move neither the compatibility module nor the library could reach it: a shim IMPORTS
    this file, it does not execute it, and the result is an entry point that exits zero
    having done nothing. Sixth, seventh and eighth instance of that shape in this
    migration, which is why it is now the first thing checked per module.
    """
    argv = list(sys.argv if argv is None else argv)
    if "--safety" in argv:
        safety()
        sys.exit(0)
    if "--corpus" in argv:
        sys.exit(0 if corpus() else 1)
    if "--build" in argv:
        build()
    elif "--verify" in argv:
        sys.exit(0 if verify() else 1)
    else:
        show()


if __name__ == "__main__":
    main()
