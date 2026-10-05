#!/usr/bin/env python3
"""Do our bytes DECODE to the instruction Apple wrote, even when they are not Apple's bytes?

Byte-exactness is one question and it is the strict one, but it answers two different situations
with the same "no". A field can have redundant encodings - two bit patterns that decode to the
same operand, which this project has already found in three forms - and then reproducing Apple's
choice is a matter of taste rather than of understanding. A field we simply do not model produces
a DIFFERENT instruction, which is a matter of understanding.

So decode what we assemble and compare the operand tuples. Three outcomes, and the middle one is
the point:

    exact        our bytes are Apple's bytes
    equivalent   different bytes, same opcode and same printed operands
    different    a different instruction, or the decoder refuses it

`equivalent` is authored. `different` is not, and the count of it is the honest size of what is
left to recover. Apple's decoder is the oracle for its own instruction description, which is the
same standing this project gives it everywhere else: it says what the bytes mean, not what the
silicon does, and nothing here is dispatched.

    python3 tools/g17same.py [--limit N] [--report]
"""
import collections, json, os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import g17as, g17metal, g17slice
from g17corpus import corpus_programs

ISA = os.path.join(os.path.dirname(HERE), "isa")
CORPUS = os.path.join(ISA, "g17-corpus-programs.jsonl")
STRIDE = 32


def decode_many(blobs, _retry=True):
    """[(opcode, (token, ...)) or None] for a list of single instructions.

    ONE BAD INSTRUCTION TRUNCATES THE BATCH. The disassembler walks the file and stops when it
    cannot decode, so everything after the first illegal instruction comes back missing and reads
    as illegal too - which is how 30 genuinely illegal encodings measured as 328. So anything that
    comes back missing is re-decoded ON ITS OWN, where nothing else can have stopped the walk.
    """
    body = bytearray()
    for b in blobs:
        body += bytes(b).ljust(STRIDE, b"\x06")
    out = [None] * len(blobs)
    if not blobs:
        return out
    with tempfile.NamedTemporaryFile(suffix=".bin") as fh:
        fh.write(body)
        fh.flush()
        r = subprocess.run([g17metal.DIS, fh.name, "0", str(len(body)), "--pc", "0",
                            "--stride", str(STRIDE), "--expr"], capture_output=True, text=True)
    for line in r.stdout.splitlines():
        p = line.split()
        if len(p) < 2:
            continue
        try:
            off = int(p[0].rstrip(":"), 16)
        except ValueError:
            continue
        if off % STRIDE or off // STRIDE >= len(blobs):
            continue
        if p[1] != "bad" and len(p) >= 3:
            out[off // STRIDE] = (int(p[2]), tuple(p[3:]))
    if _retry:
        for j, v in enumerate(out):
            if v is None:
                out[j] = decode_many([blobs[j]], _retry=False)[0]
    return out


def main():
    lim = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    tally = collections.Counter()
    per_op = collections.defaultdict(collections.Counter)
    pend_ours, pend_theirs, pend_op = [], [], []

    def flush():
        if not pend_ours:
            return
        a = decode_many(pend_ours)
        b = decode_many(pend_theirs)
        for got, want, op in zip(a, b, pend_op):
            # THE NULL CONTROL. Apple's OWN bytes go through the same padding and the same
            # stride, so if they do not decode here the harness is what failed, not the encoder.
            if want is None:
                k = "harness lost Apple's own bytes"
            elif got is None:
                # THE ASSEMBLER NEVER EMITS THIS. render compares bytes and falls back to raw
                # ones, so what is counted here is the CANDIDATE encoding - what would have gone
                # out without that guard. Measured over 120 programs on the two opcodes that
                # produce them: 0 illegal instructions emitted, 83 refused and emitted as bytes.
                # The distinction matters because the mission's own wording said "we emit bytes
                # that are not a legal instruction at all", and we do not.
                k = "candidate is not legal (render refuses it)"
            elif got == want:
                k = "equivalent"
            else:
                k = "different"
            tally[k] += 1
            per_op[op][k] += 1
        del pend_ours[:], pend_theirs[:], pend_op[:]

    for line in corpus_programs(lim):
        d = json.loads(line)
        code = bytes.fromhex(d["text"])
        spans = [tuple(s) for s in d["spans"]]
        notes = {}
        try:
            prog = g17as.assemble(g17as.render(code, spans, notes))
        except Exception:
            tally["program failed"] += 1
            continue
        for it, (off, ln, op) in zip(prog.insts, spans):
            raw = code[off:off + ln]
            if it.mnem != ".byte" and it.bytes == raw:
                tally["exact"] += 1
                per_op[op]["exact"] += 1
                continue
            if notes.get(off) == "near miss":
                # what render WOULD have emitted, re-derived so the comparison is against a real
                # candidate encoding rather than against the raw bytes it fell back to
                one = {}
                g17as.render(raw, [(0, ln, op)], one)
                cand = _candidate(op, ln, raw)
                if cand is None:
                    tally["no candidate"] += 1
                    per_op[op]["no candidate"] += 1
                    continue
                pend_ours.append(cand)
                pend_theirs.append(raw)
                pend_op.append(op)
                if len(pend_ours) >= 400:
                    flush()
            else:
                tally["no form"] += 1
                per_op[op]["no form"] += 1
    flush()
    tot = sum(tally.values())
    print("WHAT WE PRODUCE, decoded and compared with what Apple wrote (%d instructions)" % tot)
    for k in ("exact", "equivalent", "different", "candidate is not legal (render refuses it)",
              "harness lost Apple's own bytes", "no candidate", "no form", "program failed"):
        if tally[k]:
            print("   %-14s %7d  (%.1f%%)" % (k, tally[k], 100.0 * tally[k] / tot))
    authored = tally["exact"] + tally["equivalent"]
    print("   %-14s %7d  (%.1f%%)   exact or equivalent"
          % ("AUTHORED", authored, 100.0 * authored / tot if tot else 0))
    if "--report" in sys.argv:
        rows = sorted(per_op.items(), key=lambda kv: -kv[1]["different"])
        print("\n  the opcodes that produce a DIFFERENT instruction:")
        for op, c in rows[:20]:
            if not c["different"]:
                break
            print("     op%-6d %-12s different %5d   equivalent %5d   exact %5d"
                  % (op, g17slice.KNOWN_OPS.get(op, "-"), c["different"], c["equivalent"],
                     c["exact"]))


def _candidate(op, ln, raw):
    """The bytes render would have emitted for this instruction if it did not insist on exact."""
    # THE SAME FORM RENDER WOULD HAVE CHOSEN. More than one form can describe one (opcode,
    # length) and only some can read a given instruction; taking whichever was registered first
    # made this diagnostic disagree with the assembler it is measuring, which is the one thing a
    # diagnostic must never do. It reported 11 instructions as having no candidate that the
    # assembler renders perfectly well.
    tbl = g17as.forms_table()
    by = collections.defaultdict(list)
    for m, f in tbl.items():
        by[(f["op"], f["len"])].append(m)
    mn = args = None
    for cand in by.get((op, ln), []):
        got = g17as._read_operands(op, ln, tbl[cand], raw)
        if got is not None:
            mn, args = cand, got
            break
    if mn is None:
        return None
    for idx in tbl[mn].get("mods", []):
        r = g17as.maps().get((op, ln, idx, "imm")) or g17as.maps().get((op, ln, idx, "reg"))
        if not r or r["verdict"] not in ("verified", "table"):
            continue
        v = g17as.field_decode(op, idx, r["kind"], raw)
        if v is None or v == r.get("modal"):
            continue
        args.append("op%d=%s" % (idx, ("r%d" % v) if r["kind"] == "reg" else "#%d" % v))
    try:
        it = g17as.Inst("x", 1, mn, [g17as.parse_operand(t, 1)
                                     for t in g17as.SPLIT.split(", ".join(args)) if t.strip()])
        it.addr = 0
        return g17as.encode_item(it, {}, tbl)
    except Exception:
        return None


if __name__ == "__main__":
    main()
