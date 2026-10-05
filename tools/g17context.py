#!/usr/bin/env python3
"""What an unexplained bit is CONTAINED BY: differential against the instruction's context.

tools/g17fields.py explains a bit by matching it against the operand values Apple's decoder
reports, and tools/g17layout.py does the same against slot-encoded values and subtracts the
fields the encode side established causally. What is left is NET RESIDUE: bits the compiler can
neither author nor attribute. Both tools look only INSIDE the instruction, so a bit that encodes
something about the instruction's neighbours is invisible to them by construction.

byte4[5] is why this exists. Five readings of it were eliminated - all about the instruction's
own operands and their lifetimes - and the bit turned out to be contained by the PREVIOUS
instruction: an op10282 that reads a register its predecessor defined has byte4[5]=0 in 2 of
3285, and one that does not has it in 697 of 3744. No test that looks inside the instruction
could have found that.

    python3 tools/g17context.py 10282           residual bits of one opcode
    python3 tools/g17context.py --top 20        the most common opcodes with residue
    python3 tools/g17context.py 10282 --bit 4,5 one bit, every feature printed

WHAT THIS DOES NOT DO. It reports CONTAINMENT, never semantics. A feature that separates a bit
says the compiler's choice of that bit covaries with that feature in Apple's shaders; it does not
say the hardware reads the bit, and byte4[3] is the standing proof that it might not - it
correlates with result liveness 31 of 31 and flipping it changes no result. Every rate here is
printed against the bit's MARGINAL rate, because a "94.8% agreement" that turned out to be a
bit's own base rate has already cost this project a day.
"""
import collections, glob, os, pickle, re, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "spike", "accel", "re"))
import g17layout

DIS = os.path.join(HERE, "agx3dis")
CACHE = os.path.expanduser("~/.cache/agxforge/agx/*")
WALK = os.path.expanduser("~/.cache/agxforge/g17context-walk.pkl")
REG_BASE = 105
OPCODES_TOML = os.path.join(os.path.dirname(HERE), "isa", "g17-opcodes.toml")
# How far back and forward to look for the nearest producer and consumer. Beyond this the answer
# is reported as "none", which is honest: it is not that the distance is large, it is that this
# walk did not find one.
WINDOW = 12
MIN_CELL = 40          # a cell smaller than this is not reported; it cannot separate anything


def num_defs():
    """{opcode: NumDefs} from Apple's own MCInstrDesc, via isa/g17-opcodes.toml."""
    out, cur = {}, {}
    for line in open(OPCODES_TOML):
        m = re.match(r"(id|defs) = (\d+)", line.strip())
        if m:
            cur[m.group(1)] = int(m.group(2))
        if "defs" in cur and "id" in cur:
            out[cur["id"]] = cur["defs"]
            cur = {}
    return out


def walk(refresh=False):
    """[(object, span, [(offset, length, opcode, bytes, operands)])] over the whole corpus.

    Walks the CONSTANT PROGRAM as well as _agc.main. Walking only main once made every opcode
    living in the prologue report zero instances, and the same defect would hide any bit whose
    behaviour differs between the two streams - byte4[5] is 0 in 790 of 17935 main instructions
    and 0 of 214 constant-program ones, which is a fact only a walker that sees both can state.
    """
    if not refresh and os.path.exists(WALK):
        with open(WALK, "rb") as fh:
            return pickle.load(fh)
    import machobj, agxdis, g17ref
    g17ref.binary()
    streams = []
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
        cp = loc["syms"].get("_agc.main.constant_program")
        spans = [("main", e, len(t))]
        if cp is not None and cp < e:
            spans.append(("constprog", cp, e))
        for name, start, stop in spans:
            with tempfile.NamedTemporaryFile(suffix=".bin") as fh:
                fh.write(t)
                fh.flush()
                r = subprocess.run([DIS, fh.name, str(start), str(stop - start), "--pc", str(start)],
                                   capture_output=True, text=True)
            st = []
            for line in r.stdout.splitlines():
                p = line.split()
                if len(p) < 3 or p[1] == "bad":
                    continue
                off, ln, op = int(p[0], 16), int(p[1]), int(p[2])
                ops = [(k, int(v, 0)) for k, v in
                       (tok.split(":", 1) for tok in p[3:] if ":" in tok)]
                st.append((off, ln, op, t[off:off + ln], ops))
            streams.append((os.path.basename(d), name, st))
    os.makedirs(os.path.dirname(WALK), exist_ok=True)
    # temporary + rename: WALK is shared by every worktree on the machine, and concurrent ledger
    # checks must never load a pickle another process is halfway through writing
    tmp = "%s.%d.tmp" % (WALK, os.getpid())
    with open(tmp, "wb") as fh:
        pickle.dump(streams, fh)
    os.replace(tmp, WALK)
    return streams


def features(streams, defs):
    """{opcode: [(bytes, operands, {feature: value})]} for every instruction in the corpus."""
    def dset(op, ops):
        return {v - REG_BASE for k, v in ops[:defs.get(op, 1)] if k == "reg" and v >= REG_BASE}
    def uset(op, ops):
        return {v - REG_BASE for k, v in ops[defs.get(op, 1):] if k == "reg" and v >= REG_BASE}
    out = collections.defaultdict(list)
    for name, span, st in streams:
        du = [(dset(op, ops), uset(op, ops)) for _, _, op, _, ops in st]
        for i, (off, ln, op, b, ops) in enumerate(st):
            d, u = du[i]
            back = next((i - j for j in range(i - 1, max(-1, i - 1 - WINDOW), -1)
                         if du[j][0] & u), None)
            fwd = next((j - i for j in range(i + 1, min(len(st), i + 1 + WINDOW))
                        if du[j][1] & d), None)
            f = {
                "span": span,
                "object": name,
                "prev_op": st[i - 1][2] if i else None,
                "next_op": st[i + 1][2] if i + 1 < len(st) else None,
                "prev_len": st[i - 1][1] if i else None,
                "next_len": st[i + 1][1] if i + 1 < len(st) else None,
                "align4": off % 4,
                "reads_predecessor": bool(i and (du[i - 1][0] & u)),
                "successor_reads_it": bool(i + 1 < len(st) and (du[i + 1][1] & d)),
                "producer_distance": back if back is not None else "none",
                "consumer_distance": fwd if fwd is not None else "none",
                "in_place": bool(d & u),
                "n_reg_uses": len(u),
            }
            out[op].append((b, ops, f))
    return out


def separate(rows, bit, feature):
    """[(value, n, zeros, rate)] for one feature against one bit, biggest cells first."""
    byte, index = bit
    tab = collections.defaultdict(lambda: [0, 0])
    for b, _, f in rows:
        if len(b) <= byte:
            continue
        tab[f[feature]][(b[byte] >> index) & 1] += 1
    out = []
    for value, (z, o) in tab.items():
        if z + o >= MIN_CELL:
            out.append((value, z + o, z, 100.0 * z / (z + o)))
    return sorted(out, key=lambda r: -r[1])


def report(opcode, rows, bits, show_all=False):
    """Print, for each residual bit, the features that separate it and by how much."""
    print("\n=== opcode %d, %d instances ===" % (opcode, len(rows)))
    for bit in sorted(bits):
        byte, index = bit
        col = [(b[byte] >> index) & 1 for b, _, _ in rows if len(b) > byte]
        if not col or len(set(col)) < 2:
            print("\n  byte%d[%d]: does not vary in this walk" % bit)
            continue
        marginal = 100.0 * col.count(0) / len(col)
        print("\n  byte%d[%d]  marginal: 0 in %d of %d (%.1f%%)"
              % (byte, index, col.count(0), len(col), marginal))
        found = []
        for feature in ("reads_predecessor", "successor_reads_it", "in_place", "span",
                        "align4", "prev_len", "next_len", "producer_distance",
                        "consumer_distance", "n_reg_uses", "prev_op", "next_op"):
            cells = separate(rows, bit, feature)
            if len(cells) < 2:
                continue
            # A feature is worth printing when some cell departs from the bit's OWN base rate.
            # Ranking by rate alone reproduces the mistake that made a bit's marginal look like a
            # result; the deviation is what carries information.
            best = max(cells, key=lambda c: abs(c[3] - marginal))
            if abs(best[3] - marginal) < 15 and not show_all:
                continue
            found.append((abs(best[3] - marginal), feature, cells))
        if not found:
            print("    no feature departs from the marginal by 15 points")
        for _, feature, cells in sorted(found, reverse=True)[:5]:
            print("    %s:" % feature)
            for value, n, z, rate in sorted(cells, key=lambda c: -abs(c[3] - marginal))[:6]:
                print("      %-14s n=%-6d 0 in %-6d %5.1f%%  %+6.1f vs marginal"
                      % (value, n, z, rate, rate - marginal))


def main():
    args = sys.argv[1:]
    show_all = "--all" in args
    refresh = "--refresh" in args
    args = [a for a in args if not a.startswith("--")]
    streams = walk(refresh)
    print("corpus: %d object spans, %d instructions"
          % (len(streams), sum(len(s) for _, _, s in streams)))
    defs = num_defs()
    rows = features(streams, defs)

    if "--top" in sys.argv:
        n = int(sys.argv[sys.argv.index("--top") + 1])
        targets = [op for op, _ in
                   collections.Counter({k: len(v) for k, v in rows.items()}).most_common(n)]
    else:
        targets = [int(a) for a in args]

    for opcode in targets:
        rs = rows.get(opcode, [])
        if len(rs) < 8:
            print("\n=== opcode %d: %d instances, too few ===" % (opcode, len(rs)))
            continue
        # Net residue from the layout solver, so this tool asks about the SAME bits the
        # authorability table calls unexplained rather than inventing its own list.
        a = g17layout.analyse(opcode, [(b, ops) for b, ops, _ in rs])
        if a is None:
            print("\n=== opcode %d: layout solver declined ===" % opcode)
            continue
        _, ln, kinds, varying, explained, unexplained, net = a
        modal = [r for r in rs if len(r[0]) == ln
                 and tuple(k for k, _ in r[1]) == tuple(kinds)]
        if "--bit" in sys.argv:
            b, i = sys.argv[sys.argv.index("--bit") + 1].split(",")
            net = {(int(b), int(i))}
        if not net:
            print("\n=== opcode %d: no net residue ===" % opcode)
            continue
        report(opcode, modal, net, show_all)


if __name__ == "__main__":
    main()
