#!/usr/bin/env python3
"""Recover an opcode's FIELD LAYOUT from Apple's decoder, by correlating bits with operands.

tools/agx3dis.c reads the MCInst that Apple's decoder builds, so for every instruction in the
corpus we have both the bytes and the operand VALUES the ISA description says those bytes mean.
That is enough to solve the layout without mutating anything: collect every instance of an
opcode, and for each operand ask which instruction bits equal which bit of its value across ALL
instances. A bit that matches everywhere is a field bit; a bit that varies and matches nothing
is unexplained, and unexplained bits are exactly what the mission says to eliminate rather than
inherit.

This does not replace causal evidence. It says where a field is under Apple's own description of
its own encoding; it does not say what the instruction DOES, and 2026-09-04 is the day this
project learned that a decode-clean instruction can still compute the wrong thing. Execution is
still what settles semantics.

    python3 tools/g17fields.py 11666            solve one opcode
    python3 tools/g17fields.py --control        rediscover a field that is already modelled
    python3 tools/g17fields.py --top 12         solve the most common opcodes
"""
import collections, glob, os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
# ANCHORED ON THE CHECKOUT ROOT: two levels up from agxforge/g17/, where one sufficed from
# tools/. The native helpers do not move - the Makefile keeps building them into tools/.
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS = os.path.join(ROOT, "tools")
DIS = os.path.join(TOOLS, "agx3dis")
CACHE = os.path.expanduser("~/.cache/agxforge/agx/*")

# LLVM register numbers. R_n = 105 + n, recovered from MCRegisterDesc; the decoder prints the
# LLVM number, the instruction encodes the index.
REG_BASE = 105
# Values above this are addresses and relocations, not fields of the instruction.
FIELD_MAX = 1 << 20


def instances(opcodes, limit_objects=None):
    """{opcode: [(bytes, [(kind, value)...]), ...]} over every cached object."""
    from agxforge.g17 import machobj, agxdis
    from agxforge.g17 import ref as g17ref
    g17ref.binary()
    out = collections.defaultdict(list)
    objs = 0
    for d in sorted(glob.glob(CACHE)):
        arc, obj = d + "/s.arc.metallib", d + "/out/object/0-0"
        if not (os.path.exists(arc) and os.path.exists(obj)):
            continue
        try:
            loc = machobj.locate(arc, obj)
            f, sz = agxdis.sections(loc["obj"])
            t = bytes(loc["obj"][f:f + sz])
            e = loc["syms"]["_agc.main"]
        except Exception:
            continue
        objs += 1
        if limit_objects and objs > limit_objects:
            break
        # WALK THE CONSTANT PROGRAM TOO. It is a real instruction stream - the prologue that
        # publishes the relocations main then reads - and walking only _agc.main made every
        # opcode that lives there report ZERO instances and show as unsolvable. op592, the
        # publish, has 389 instances and the solver could not see one of them. Defect found by
        # the ISA agent in its own census first, and it is the same defect here.
        cp = loc["syms"].get("_agc.main.constant_program")
        spans = [(e, len(t))] + ([(cp, e)] if cp is not None and cp < e else [])
        out_lines = []
        for start, stop in spans:
            with tempfile.NamedTemporaryFile(suffix=".bin") as fh:
                fh.write(t)
                fh.flush()
                r = subprocess.run([DIS, fh.name, str(start), str(stop - start),
                                    "--pc", str(start)], capture_output=True, text=True)
            out_lines += r.stdout.splitlines()
        for line in out_lines:
            p = line.split()
            if len(p) < 3 or p[1] == "bad":
                continue
            op = int(p[2])
            if opcodes and op not in opcodes:
                continue
            off, ln = int(p[0], 16), int(p[1])
            ops = []
            for tok in p[3:]:
                if ":" in tok:
                    k, v = tok.split(":", 1)
                    ops.append((k, int(v, 0)))
            out[op].append((t[off:off + ln], ops))
    return out, objs


def _owned_mask(opcode):
    """What g17asm already authors for this opcode, as {byte: mask}. Empty when the opcode has no
    encoder yet, which is itself the answer to "is this authorable"."""
    from agxforge.g17 import asm as g17asm
    if opcode in getattr(g17asm, "ALU_FORM", {}):
        m = dict(g17asm.OWNED_ALU)
        for bits in (g17asm.ALU_DEST, g17asm.SLOT_A, g17asm.SLOT_B):
            for b, i in bits: m[b] = m.get(b, 0) | (1 << i)
        return m
    if opcode in getattr(g17asm, "BITWISE_FORM", {}):
        m = {}
        for bits in (g17asm.ALU_DEST, g17asm.BW_SRC, g17asm.BW_IMM):
            for b, i in bits: m[b] = m.get(b, 0) | (1 << i)
        return m
    if opcode in getattr(g17asm, "BITWISE_REG_FORM", {}):
        m = {}
        for bits in (g17asm.BW_R_DEST, g17asm.BW_R_SRCA, g17asm.BW_R_SRCB, g17asm.BW_R_OP):
            for b, i in bits: m[b] = m.get(b, 0) | (1 << i)
        return m
    return {}


def _columns(rows):
    """[(byte, bit, column)] for every bit position, column parallel to rows."""
    n = min(len(b) for b, _ in rows)
    return [(byte, bit, [(b[byte] >> bit) & 1 for b, _ in rows])
            for byte in range(n) for bit in range(8)]


def solve(opcode, rows, verbose=True):
    """Field map for one opcode. Returns (fields, unexplained, note)."""
    sig = collections.Counter(tuple(k for k, _ in ops) for _, ops in rows)
    kinds, nsig = sig.most_common(1)[0]
    rows = [r for r in rows if tuple(k for k, _ in r[1]) == kinds]
    lens = collections.Counter(len(b) for b, _ in rows)
    ln, nlen = lens.most_common(1)[0]
    rows = [r for r in rows if len(r[0]) == ln]
    if verbose:
        print("opcode %-6d %d instances, %d bytes, operands %s"
              % (opcode, len(rows), ln, " ".join(kinds)))
        if len(sig) > 1:
            print("  NOTE %d operand signatures; solving the modal one (%d of %d)"
                  % (len(sig), nsig, sum(sig.values())))
    cols = _columns(rows)
    varying = [(b, i, c) for b, i, c in cols if len(set(c)) > 1]
    frozen = [(b, i, c[0]) for b, i, c in cols if len(set(c)) == 1]
    explained, fields = set(), []
    for k in range(len(kinds)):
        vals = [ops[k][1] for _, ops in rows]
        kind = kinds[k]
        if len(set(vals)) == 1:
            fields.append((k, kind, "constant %d" % vals[0], []))
            continue
        if max(abs(v) for v in vals) > FIELD_MAX:
            fields.append((k, kind, "values too large to be a field (max %d)" % max(vals), []))
            continue
        # A register operand carries an LLVM number; the instruction holds the index. Try the
        # documented base and the observed minimum, and keep whichever leaves nothing over.
        best = None
        for base in ([REG_BASE, min(vals), 0] if kind == "reg" else [0, min(vals)]):
            res = [v - base for v in vals]
            if min(res) < 0:
                continue
            width = max(r.bit_length() for r in res)
            bits, ok = [], True
            for j in range(width):
                want = [(r >> j) & 1 for r in res]
                m = [(b, i, False) for b, i, c in varying if c == want]
                m += [(b, i, True) for b, i, c in varying if c == [1 - x for x in want]]
                if not m:
                    # a value bit that is constant across the sample is carried by a frozen bit
                    if len(set(want)) == 1:
                        m = [(b, i, want[0] != v) for b, i, v in frozen if v == want[0]][:0]
                        bits.append((j, None))
                        continue
                    ok = False
                    bits.append((j, None))
                else:
                    bits.append((j, m))
            score = (ok, sum(1 for _, m in bits if m), -base)
            if best is None or score > best[0]:
                best = (score, base, bits, ok)
        _, base, bits, ok = best
        desc = []
        for j, m in bits:
            if m is None:
                desc.append("bit%d=?" % j)
            else:
                desc.append("bit%d=%s" % (j, "|".join(("~" if inv else "") + "b%d[%d]" % (b, i)
                                                      for b, i, inv in m)))
                for b, i, _ in m:
                    explained.add((b, i))
        fields.append((k, kind, ("base %d, " % base if base else "") + " ".join(desc), bits))
    unexplained = [(b, i) for b, i, _ in varying if (b, i) not in explained]
    # NET RESIDUE. A bit the decoder's operands do not explain may still be one this project
    # AUTHORS from semantics - the ALU's liveness bit and the fused-shift scale are both like
    # that, recovered causally by execution rather than by correlation with an operand. The
    # metric that matters for a compiler is what it can neither author nor attribute.
    owned = _owned_mask(opcode)
    net = [(b, i) for b, i in unexplained if not (owned.get(b, 0) >> i) & 1]
    if verbose:
        for k, kind, desc, _ in fields:
            print("  operand %-2d %-4s %s" % (k, kind, desc))
        ent = {}
        for b, i, c in varying:
            if (b, i) in unexplained:
                ent[(b, i)] = sum(c) / float(len(c))
        print("  frozen bits    : %d of %d" % (len(frozen), len(cols)))
        print("  field bits     : %d" % len(explained))
        print("  UNEXPLAINED    : %d  %s" % (len(unexplained),
              " ".join("b%d[%d]=%.2f" % (b, i, ent[(b, i)]) for b, i in unexplained) or "none"))
        if owned:
            print("  NET, after the encoder's authored bits: %d  %s"
                  % (len(net), " ".join("b%d[%d]" % x for x in net) or "none"))
    return fields, unexplained, frozen


def roundtrip(rows, min_n=20):
    """Can an opcode's instructions be REBUILT from their operands alone?

    For each opcode: solve the layout, take one instance as the template, and for every other
    instance write only the solved field bits onto it. Byte-exact means the operand values plus
    that one template account for the whole instruction. This is the measure the mission asks to
    shrink the complement of - the residue is inherited bits, and no hand model is involved, so
    it cannot be tuned.

    POSITIVE DISCRIMINATION CONTROL: the same loop scored with NO fields written - the template
    alone. If authoring the fields did not raise the score, the fields are not doing the work and
    the number is measuring how repetitive the corpus is instead.
    """
    tot = exact = base_exact = 0
    per = []
    for op in sorted(rows, key=lambda o: -len(rows[o])):
        rs = rows[op]
        if len(rs) < min_n:
            continue
        sig = collections.Counter(tuple(k for k, _ in o) for _, o in rs).most_common(1)[0][0]
        rs = [r for r in rs if tuple(k for k, _ in r[1]) == sig]
        ln = collections.Counter(len(b) for b, _ in rs).most_common(1)[0][0]
        rs = [r for r in rs if len(r[0]) == ln]
        if len(rs) < min_n:
            continue
        fields, _, _ = solve(op, rs, verbose=False)
        bits = sorted({(b, i) for _, _, _, bl in fields for j, m in (bl or [])
                       if m for b, i, inv in m if not inv})
        tmpl = collections.Counter(b for b, _ in rs).most_common(1)[0][0]
        ok = base = 0
        for u, _ in rs:
            v = bytearray(tmpl)
            for b, i in bits:
                v[b] = (v[b] & ~(1 << i)) | (u[b] & (1 << i))
            ok += bytes(v) == u
            base += bytes(tmpl) == u
        per.append((op, len(rs), ln, len(bits), ok, base))
        tot += len(rs); exact += ok; base_exact += base
    per.sort(key=lambda r: -r[1])
    print("%-8s %6s %5s %6s %10s %10s" % ("opcode", "n", "bytes", "fields", "rebuilt", "template"))
    for op, n, ln, nb, ok, base in per[:24]:
        print("%-8d %6d %5d %6d %6d %4.0f%% %6d %4.0f%%"
              % (op, n, ln, nb, ok, 100.0 * ok / n, base, 100.0 * base / n))
    print("\n%d opcodes, %d instructions" % (len(per), tot))
    print("REBUILT from operands + one template : %d (%.1f%%)" % (exact, 100.0 * exact / tot))
    print("CONTROL, template alone, no fields   : %d (%.1f%%)" % (base_exact, 100.0 * base_exact / tot))
    return exact, base_exact, tot


CONTROL = 10279   # add reg,imm - operand and width fields modelled in tools/g17asm.py


def _control(rows):
    """The method has to rediscover a field that is already known, or its answer for an unknown
    opcode means nothing. g17asm.decode_alu reads 10279's immediate as
        (b1[1:0]) | (b3[7:5] << 2) | (b5[7] << 5) | (b8[1:0] << 6)
    and its src1 as 2*b9[6:0] + b8[7]. Both must come back out of the correlation."""
    fields, unexplained, _ = solve(CONTROL, rows)
    want_imm = ["b1[0]", "b1[1]", "b3[5]", "b3[6]", "b3[7]", "b5[7]", "b8[0]", "b8[1]"]
    want_src1 = ["b8[7]", "b9[0]", "b9[1]", "b9[2]", "b9[3]", "b9[4]", "b9[5]", "b9[6]"]
    found = {}
    for k, kind, desc, bits in fields:
        got = []
        for j, m in bits:
            got.append("b%d[%d]" % m[0][:2] if m and len(m) == 1 else "?")
        found[k] = got
    # A field bit only shows up if the sample VARIES it: no corpus instance uses r128, so
    # src1's top bit is not observable here and demanding it would fail a correct answer. The
    # control accepts a prefix and reports how much of the field the sample could reach.
    ok = 0
    for want, name in ((want_imm, "immediate"), (want_src1, "src1")):
        hit = [(k, got) for k, got in found.items()
               if got and got == want[:len(got)] and len(got) >= len(want) - 1]
        if hit:
            k, got = hit[0]
            print("  CONTROL %-9s operand %d, %d of %d bits observed: %s"
                  % (name, k, len(got), len(want), " ".join(got)))
        else:
            print("  CONTROL %-9s NOT RECOVERED (expected %s)" % (name, " ".join(want)))
        ok += bool(hit)
    return ok == 2


def main(argv=None):
    """The command-line entry, callable. It was inline under `if __name__`, so after the
    move neither the shim nor the library could reach it - a compatibility module imports
    this file, it does not execute it. Fifth instance of that defect class in this
    migration; it exits zero having done nothing, which no return-code test can see.
    """
    argv = list(sys.argv if argv is None else argv)   # mirrors sys.argv; the body slices it
    args = [a for a in argv[1:]]
    if "--control" in args:
        rows, objs = instances({CONTROL})
        print("corpus: %d objects" % objs)
        sys.exit(0 if _control(rows[CONTROL]) else 1)
    if "--roundtrip" in args:
        rows, objs = instances(None)
        print("corpus: %d objects, %d opcodes\n" % (objs, len(rows)))
        roundtrip(rows)
        sys.exit(0)
    if "--top" in args:
        n = int(args[args.index("--top") + 1])
        rows, objs = instances(None)
        print("corpus: %d objects, %d opcodes" % (objs, len(rows)))
        for op, _ in collections.Counter({o: len(v) for o, v in rows.items()}).most_common(n):
            solve(op, rows[op])
            print()
        sys.exit(0)
    want = {int(a) for a in args if a.isdigit()}
    rows, objs = instances(want)
    print("corpus: %d objects" % objs)
    for op in sorted(want):
        if op not in rows:
            print("opcode %d: no instances" % op)
            continue
        solve(op, rows[op])
        print()


if __name__ == "__main__":
    main()
