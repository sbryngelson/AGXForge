#!/usr/bin/env python3
"""Verify each operand field against MANY of Apple's instructions, not the one it was fitted to.

`g17encode.extended_fields` derives a field's base offset from a single witness: base is whatever
the field's bits fail to account for in that one instruction. That is unfalsifiable by
construction. Reconstruction never caught it because it reads values back through the same map,
and the first oracle batch did catch it - ten of eleven records assembled to bytes Apple's own
decoder rejected, because a register number was written into a field that counts from somewhere
else.

So fit the model on one pair of instances and TEST it on the rest:

    printed = base + encoded * step

VERDICTS. `verified` means at least two distinct encoded values were seen and every instance
agrees - the field is safe to author with a value that did not come from a witness. `degenerate`
means every instance carries the same value, so step and base are not separately determined and
the map predicts nothing away from that point. `refuted` means instances disagree with any single
linear model, so the bit set is wrong, the field is a lookup table, or it is coupled to another.

Only `verified` fields may be offered to an assembler. A `degenerate` field is exactly as
trustworthy as the single-witness base it replaced, which is to say not at all.

    python3 tools/g17opmap.py --ops 10295,13588 --report
    python3 tools/g17opmap.py --set proven          the execution-validated instruction set
    python3 tools/g17opmap.py --all --limit 400     everything with enough instances
"""
import collections, fractions, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g17canon, g17encode, g17fields, g17slice

CORPUS_INSTANCES = True   # see corpus_instances()

ISA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "isa")
OUT = os.path.join(ISA, "g17-operand-maps.jsonl")
FORMBITS = os.path.join(ISA, "g17-form-bits.json")
# How many real instances of a form to flip bits on. One is not enough; see form_bits.
FORMBASES = 8

# Names this project has actually executed, or that the peer's oracle batch matched exactly on
# solver-chosen inputs. Everything else stays decode-only until something runs it.
PROVEN = {10295: "add", 10864: "mul", 11680: "sub", 13473: "nand", 13588: "or", 17784: "xor",
          410: "andn", 17757: "xnor", 437: "and", 423: "and", 424: "and", 13574: "or",
          13575: "or", 17770: "xor", 17771: "xor", 902: "fadd.sat", 3098: "fmul.sat",
          586: "mov", 555: "movimm", 11842: "movimm"}


def _encoded(positions, raw):
    v = 0
    for w, b, i, inv in positions:
        if b < len(raw):
            v |= (((raw[b] >> i) & 1) ^ inv) << w
    return v


def _subvalues(kind, val):
    """The numeric sub-fields one printed operand carries, as (kind, value) pairs.

    An address expression is NOT one number. Apple's decoder prints operand 3 of a load as
    `bin(op0,const(0),8)`: a base operand, a constant and a scale, each encoded in its own bits.
    Fitting the token as a whole gives the field no observations at all - which is why 141
    register slots across loads, stores and the fp16 forms had no map of any kind, and why 122
    (opcode, length) pairs could not be offered a form. Splitting it gives each sub-field its own
    map and its own verdict.
    """
    if kind == "expr" and isinstance(val, str):
        m = g17canon.EXPR_RE.match("expr:" + val)
        if not m:
            return []
        return [("expr.base", int(m.group(1))), ("expr.const", int(m.group(2))),
                ("expr.scale", int(m.group(3)))]
    if isinstance(val, int):
        return [(kind, val)]
    return []


def corpus_instances(ops):
    """Instances read from the PRECOMPUTED `spans` INDEX of isa/g17-corpus-programs.jsonl.

    NOT A DECODE OF THE FILE, and this docstring said it was for as long as the file has existed.
    `spans` holds 184,349 of the 365,590 instructions those same bytes decode to - it drops every
    `end` and 142,452 op13483 - and every `host` in it is one of this project's own probe names.
    It is NOT "Apple's whole shipped set": that is the vendor shader corpus, 4,644,438
    instructions, and the union of the two is 5,010,028. docs/archive/g17-map-holdout.md measures the maps
    fitted here at 91.4% on this population and 55.4% on Apple's held-out encodings.

    THE BEHAVIOUR IS DELIBERATELY UNCHANGED. This is the population every shipped operand map was
    fitted on and ten tools read it; widening it here would move published fits with no marker.
    The widening is done in its own file with the before and after published - see
    tools/g17maprefit.py and docs/archive/g17-map-refit.md, which fit on the union and score both maps on
    a fresh disjoint split. Nothing that calls this function inherits that change.

    The reason it reads a corpus at all: the maps used to be fitted on a scan of the build cache
    and then MEASURED against this corpus, which is a different population. That gap is not
    academic: op554's four-byte form was `verified` at explains=1.0 on the cache while being
    unable to encode register 117, which this corpus contains 169 times. Fit and measurement have
    to see the same instructions.
    """
    import subprocess, tempfile, g17metal, g17ref
    g17ref.binary()
    path = os.path.join(ISA, "g17-corpus-programs.jsonl")
    want = set(ops)
    seen = {}
    for line in open(path):
        d = json.loads(line)
        code = bytes.fromhex(d["text"])
        for o, l, op in d["spans"]:
            if op in want and len(code[o:o + l]) == l:
                seen.setdefault((op, code[o:o + l]), 0)
                seen[(op, code[o:o + l])] += 1
    keys = sorted(seen)
    STRIDE, PAD = 32, bytes.fromhex("0600")
    blob = bytearray()
    for _, raw in keys:
        blob += raw + PAD * ((STRIDE - len(raw)) // 2)
    out = collections.defaultdict(list)
    with tempfile.NamedTemporaryFile(suffix=".bin") as fh:
        fh.write(blob)
        fh.flush()
        r = subprocess.run([g17metal.DIS, fh.name, "0", str(len(blob)), "--pc", "0", "--expr"],
                           capture_output=True, text=True)
    for line in r.stdout.splitlines():
        p = line.split()
        if len(p) < 3 or p[1] == "bad":
            continue
        off = int(p[0], 16)
        if off % STRIDE or off // STRIDE >= len(keys):
            continue
        op, raw = keys[off // STRIDE]
        ops_out = []
        for tok in p[3:]:
            kind, _, val = tok.partition(":")
            try:
                ops_out.append((kind, int(val)))
            except ValueError:
                ops_out.append((kind, val))
        # weight each distinct encoding by how often it occurs, so the fit sees the real
        # distribution rather than one vote per distinct byte string
        for _ in range(min(seen[(op, raw)], 40)):
            out[op].append((raw, ops_out))
    return out, None


def _bits_of(rec):
    """Every instruction bit this map reads, from all four places a record can name one."""
    s = set()
    for pos in (rec.get("positions") or []):
        t = tuple(pos)
        if len(t) >= 3:
            s.add((t[1], t[2]))
    for b, i, _w in (rec.get("extra") or []):
        s.add((b, i))
    for b, i in (rec.get("order") or []):
        s.add((b, i))
    for t in ((rec.get("poly") or {}).get("lin") or []):
        s.add((t[0], t[1]))
    return s


def _overlap_against(rec, siblings):
    """How many bits this record shares with the OTHER operands of its form."""
    mine = _bits_of(rec)
    n = 0
    for other in siblings:
        if other["operand"] == rec["operand"]:
            continue
        n += len(mine & _bits_of(other))
    return n


def _no_new_overlap(cand, cur, siblings):
    """A challenger may not read more of its neighbours' bits than the map it replaces.

    NOT a non-overlap rule. Overlap is a REAL property of this encoding - 266 forms of the shipped
    map have it - so refusing it outright would refuse ground truth, which is the mirror image of
    the defect this precondition exists to fix. What is refused is an INCREASE: a swap that gives a
    field bits belonging to another operand of the same form, which is what makes writing one field
    corrupt its neighbour. Every shipped form passes this by construction, because it is compared
    against itself.
    """
    return _overlap_against(cand, siblings) <= _overlap_against(cur, siblings)


def _explains_union(rec, insts):
    """Does this map predict what the decoder reads, on every union instance it applies to?

    Only meaningful for a numeric linear field; anything else is left alone (returns True) so the
    caller does not replace a table or an expression on the strength of a check that cannot see it.
    """
    step = str(rec.get("step", ""))
    if not step.lstrip("-").isdigit() or not isinstance(rec.get("operand"), int):
        return True
    idx, kind, L = rec["operand"], rec.get("kind"), rec.get("length")
    seen = False
    for raw, ops in insts:
        if len(raw) != L or idx >= len(ops):
            continue
        k, v = ops[idx]
        if k != kind or not isinstance(v, int):
            continue
        try:
            enc = _encoded(rec["positions"], raw)
        except Exception:
            continue
        seen = True
        if rec["base"] + enc * int(step) != v:
            return False
    return True if seen else True


def insts_for_kind(insts, key, kind):
    """The instances whose operand `key` is of `kind`, in the same order `obs` was built."""
    out = []
    for raw, ops in insts:
        if key >= len(ops):
            continue
        for k2, v in _subvalues(*ops[key]):
            if k2 == kind:
                out.append((bytes(raw), v))
    return out


def by_length(insts):
    """Split an opcode's instances by ENCODED LENGTH before anything else looks at them.

    170 of the 557 opcodes Apple emits appear at more than one length, and those account for
    57.8% of all instructions. A field map pooled across lengths is describing two different
    instructions at once: bit 5 of byte 8 is an operand bit in the fourteen-byte form of op12682
    and does not exist in the eight-byte one. Pooling made op554's destination register look
    refuted - 174 encodings collapsing onto 17 varying bits, which is arithmetically impossible
    and was the tell.
    """
    out = collections.defaultdict(list)
    for raw, ops in insts:
        out[len(bytes(raw))].append((raw, ops))
    return out


def rescore(rec, insts):
    """Make the verdict describe the map that is actually stored.

    Scoring happened before the positions were filtered to this form, so a record could say
    `verified, explains=1.0` while carrying a map that reproduces eight of sixty-three of its own
    instances - op554's four-byte destination register did exactly that. A verdict that does not
    describe its own map is the same unfalsifiable pass this whole file exists to prevent, so the
    last thing that happens to a record is being measured against the instances it came from.
    """
    key, kind = rec["operand"], rec.get("kind")
    obs = insts_for_kind(insts, key, kind)
    if not obs:
        return
    if rec["verdict"] == "pinned":
        # A PINNED FIELD HAS NO MAP TO SCORE. Its claim is that the value never varies, which is
        # checked directly - scoring it as a linear map refuted all 977 of them.
        vals = {v for _raw, v in obs}
        if vals and vals != {rec.get("value")}:
            rec.update(verdict="refuted", why="pinned, but the corpus gives %d values" % len(vals))
        else:
            rec["explains"] = 1.0
        return
    if rec["verdict"] == "table":
        order = rec.get("order") or []
        if not order:
            rec.update(verdict="refuted", why="the table has no bits inside this form")
            return
        ok = sum(1 for raw, v in obs
                 if rec["table"].get("".join(str((raw[b] >> i) & 1) for b, i in order)) == v)
    elif rec.get("poly"):
        ok = sum(1 for raw, v in obs if g17encode.poly_value(rec["poly"], raw) == v)
    else:
        pos = [tuple(p) for p in rec.get("positions", [])]
        st = fractions.Fraction(rec["step"]) if rec.get("step") is not None else None
        if not pos or st is None:
            rec.update(verdict="degenerate" if rec["verdict"] == "degenerate" else "refuted")
            return
        bs = rec.get("base", 0)
        ok = 0
        for raw, v in obs:
            e = 0
            for w, b, i, inv in pos:
                if b < len(raw):
                    e |= (((raw[b] >> i) & 1) ^ inv) << w
            extra = sum(c for b, i, c in (rec.get("extra") or [])
                        if b < len(raw) and (raw[b] >> i) & 1)
            if bs + e * st + extra == v:
                ok += 1
    frac = ok / float(len(obs))
    rec["explains"] = round(frac, 4)
    if rec["verdict"] in ("verified", "table"):
        if frac < 1.0:
            rec["verdict"] = "conditional" if frac >= 0.6 else "refuted"


def spec_class_bits(op, width, cls):
    """The (byte, bit) pairs the specification gives this class for this form."""
    spec = g17encode.spec_for(op, width) or {}
    out = set()
    for k, d in spec.items():
        c = d.get("class") if isinstance(d, dict) else d
        if c == cls:
            b, i = (int(x) for x in k.split("."))
            out.add((b, i))
    return out


MOVEDBITS = os.path.join(ISA, "g17-moved-bits.json")

_MOVES = None


def moves(op, width):
    """{(byte, bit): {operand index it MOVES}}, measured by flipping it in real instructions.

    The specification attributes each bit to an operand and g17diff.candidates() trusts that
    attribution for the bits it gives to somebody ELSE - deliberately, because widening the search
    to every neighbour's bit was tried and reverted: a map that owns a bit it merely CORRELATES
    with reads correctly and corrupts its neighbour when it writes.

    A flip is not a correlation. op998's b5.1 is attributed to another operand and moves operand 5
    at eleven of sixteen base points, so operand 5 - a modifier field carrying 0, 2, 16, 18 and 34
    - was unmeasurable: every instrument refused it, `field` because two bits showed more than one
    coefficient, `witness_table` because a pattern carried more than one value. Both are what an
    incomplete bit set looks like from the inside. That is 148 of op998's 162 six-byte instances
    rendering as raw bytes.

    Built by `python3 tools/g17opmap.py --moved-bits` and read here. It is NEVER written by an
    ordinary fit: the ban file was, without a decoder, and that quietly emptied it.
    """
    global _MOVES
    if _MOVES is None:
        try:
            _MOVES = json.load(open(MOVEDBITS))
        except (IOError, ValueError):
            _MOVES = {}
    d = _MOVES.get("%d,%d" % (op, width)) or {}
    return {tuple(int(x) for x in k.split(".")): set(v) for k, v in d.items()}


def moved_by_flip(ops, rows=None, measured=None, bases=None):
    """Measure, per form, which operand each bit moves. See moves()."""
    if rows is None:
        rows, _ = corpus_instances(ops)
    out = {}
    for op in ops:
        for length, group in by_length(rows.get(op) or []).items():
            raws = [bytes(r) for r, _ in group]
            if not raws:
                continue
            step = max(1, len(raws) // (bases or FORMBASES))
            pts = raws[::step][:(bases or FORMBASES)]
            probe, order = [], []
            for raw in pts:
                for b in range(length):
                    for i in range(8):
                        m = bytearray(raw)
                        m[b] ^= 1 << i
                        probe.append(bytes(m))
                        order.append((b, i))
            ref = measured(pts)
            got = measured(probe)
            hit = collections.defaultdict(set)
            for (b, i), d, base in zip(order, got,
                                       [g for g in ref for _ in range(length * 8)]):
                if base is None or d is None or d[0] != base[0]:
                    continue
                if len(d[1]) != len(base[1]):
                    continue
                for k, (x, y) in enumerate(zip(base[1], d[1])):
                    if x != y:
                        hit[(b, i)].add(k)
            if hit:
                out["%d,%d" % (op, length)] = {"%d.%d" % k: sorted(v) for k, v in sorted(hit.items())}
    return out


_SELECTS = None


def selects_opcode(op, width):
    """The bits that MEASURABLY select an opcode for this form, whatever the specification calls
    them.

    op10369's b1.0 is classed `forced` and flipping it decodes as op10372. 1,373 bits of the
    shipped file are like that - they choose the instruction while being labelled something else,
    883 `mode`, 459 `forced`, 22 `invisible` and 9 `operand` - and an operand map that owns one
    produces a different instruction whenever it writes a value needing that bit. This docstring
    said 460, which is neither that total nor any subset of it; see docs/archive/g17-map-refit.md. Six fields fail the encode-direction check for exactly this reason, every
    one reporting "opcode changed" rather than a wrong value.

    Measured by isa/g17-form-bits.json (`python3 tools/g17opmap.py --form-bits`), which flips
    every bit of a real instance and watches the decoded opcode. Absent that file the ban falls
    back to the classification, which is the behaviour this had before and is weaker rather than
    wrong.
    """
    global _SELECTS
    if _SELECTS is None:
        try:
            _SELECTS = json.load(open(FORMBITS))
        except (IOError, ValueError):
            _SELECTS = {}
    d = _SELECTS.get("%d,%d" % (op, width)) or {}
    return {(b, i) for b, i in (d.get("selects_opcode") or [])}


def opcode_bits(op, width):
    """The (byte, bit) pairs the specification calls opcode bits for this form.

    No search may take one. Flipping one decodes as a different instruction, so it cannot be part
    of this instruction's operand, and an encoder that writes it destroys the opcode - which is
    what the assembler's identity assertion caught on op554, whose four-byte destination map had
    quietly acquired b0.3 and made 755 of its instances unrenderable.
    """
    return spec_class_bits(op, width, "opcode") | selects_opcode(op, width)


def length_of(pairs):
    return min(len(r) for r, _ in pairs) if pairs else 0


def _solve_by_gcd(obs, raws, banned=()):
    """(step, base, {weight: (byte, bit, inverted)}) that reproduces every instance, or None.

    The step is the gcd of the differences between the observed values - exact, not searched. The
    base is then the one offset that makes every encoded value a non-negative integer whose bits
    can be found; weights that never vary are folded into the base, because a bit that is 1 in
    every instance is not evidence of a position.
    """
    vals = [v for _, v in obs]
    if len(set(vals)) < 2 or not raws or len(raws) != len(obs):
        return None
    lo = min(vals)
    g = 0
    for v in vals:
        d = fractions.Fraction(v - lo)
        if d.denominator != 1:
            return None
        g = _gcd(g, int(d))
    if g <= 0:
        return None
    width = min(len(r) for r in raws)
    for c in range(0, 33):
        base = lo - c * g
        enc = [(v - base) // g for v in vals]
        if any((v - base) % g for v in vals) or any(e < 0 for e in enc):
            continue
        place, folded, ok = {}, 0, True
        for k in range(max(enc).bit_length()):
            want = [(e >> k) & 1 for e in enc]
            if len(set(want)) < 2:
                if want[0]:
                    folded += g << k
                continue
            found = None
            for b in range(width):
                for i2 in range(8):
                    if (b, i2) in banned:
                        continue
                    bit = [(r[b] >> i2) & 1 for r in raws]
                    if bit == want:
                        found = (b, i2, 0)
                    elif bit == [1 - x for x in want]:
                        found = (b, i2, 1)
                    if found:
                        break
                if found:
                    break
            if not found:
                ok = False
                break
            place[k] = found
        if not ok or not place:
            continue
        b2 = base + folded
        if all(b2 + g * sum((((r[b] >> i2) & 1) ^ iv) << k
                            for k, (b, i2, iv) in place.items()) == v
               for r, v in zip(raws, vals)):
            return fractions.Fraction(g), b2, place
    return None


def moved_bits(op, width, key, sub=""):
    """The bits the specification says move THIS operand, as (byte, bit).

    Evidence for shape, never for value: it decides which bits the solver may spend, and the
    solution it produces is still checked against every instance. Narrowing the columns this way
    is what stops an underdetermined system paying for a bit that merely correlates.
    """
    spec = g17encode.spec_for(op, width) or {}
    out = set()
    for k, d in spec.items():
        if not isinstance(d, dict):
            continue
        for mv in (d.get("moved") or []):
            if mv and mv[0] == key and (mv[1] or "") == (sub or ""):
                b, i = (int(x) for x in k.split("."))
                out.add((b, i))
    return out


def _solve_linear(obs, raws, banned=(), prefer=None):
    """value = base + SUM(coefficient x bit), solved exactly over the instruction's own bits.

    The gcd solve assumes the field is one number on a grid: base plus a step times a binary
    weight. Some fields are not. op12682's operand 1 at fourteen bytes takes 22 values spanning
    four million steps, so no weight assignment reaches them and both the linear fit and the code
    table refuse it - while the value is in fact an exact integer combination of six of the
    instruction's own bits.

    So solve for the combination directly: one unknown coefficient per candidate bit plus a
    constant, one equation per instance, eliminated over the rationals. Free variables are set to
    zero, which puts the solution on the pivot columns and keeps its support at most the rank -
    a bit that merely correlates is not spent unless it is needed. The result is then checked
    against EVERY instance, so an underdetermined system cannot certify itself.
    """
    if not raws or len(raws) != len(obs):
        return None
    width = min(len(r) for r in raws)
    cols = [(b, i) for b in range(width) for i in range(8)
            if (b, i) not in banned and len({(r[b] >> i) & 1 for r in raws}) > 1]
    if prefer:
        narrowed = [c for c in cols if c in prefer]
        if narrowed:
            cols = narrowed
    if not cols or len(cols) > 96:
        return None
    seen, rows = set(), []
    for r, (_e, v) in zip(raws, obs):
        key = tuple((r[b] >> i) & 1 for b, i in cols)
        if key in seen:
            continue
        seen.add(key)
        rows.append([fractions.Fraction(x) for x in key] + [fractions.Fraction(1),
                                                            fractions.Fraction(v)])
        if len(rows) > 400:
            break
    m = len(cols) + 1
    piv = []
    r0 = 0
    for c in range(m):
        p = next((k for k in range(r0, len(rows)) if rows[k][c]), None)
        if p is None:
            continue
        rows[r0], rows[p] = rows[p], rows[r0]
        f = rows[r0][c]
        rows[r0] = [x / f for x in rows[r0]]
        for k in range(len(rows)):
            if k != r0 and rows[k][c]:
                g = rows[k][c]
                rows[k] = [a - g * b for a, b in zip(rows[k], rows[r0])]
        piv.append((c, r0))
        r0 += 1
        if r0 == len(rows):
            break
    for k in range(r0, len(rows)):
        if rows[k][m] and not any(rows[k][:m]):
            return None                    # inconsistent: no such combination exists
    sol = [fractions.Fraction(0)] * m
    for c, k in piv:
        sol[c] = rows[k][m]
    if any(x.denominator != 1 for x in sol):
        return None
    coef = [(cols[j], int(sol[j])) for j in range(len(cols)) if sol[j]]
    base = int(sol[m - 1])
    if not coef:
        return None
    for r, (_e, v) in zip(raws, obs):
        if base + sum(c for (b, i), c in coef if (r[b] >> i) & 1) != v:
            return None
    g = 0
    for _, c in coef:
        g = _gcd(g, c)
    if g <= 0:
        return None
    place, extra, used = {}, [], set()
    for (b, i), c in coef:
        q = c // g
        k = q.bit_length() - 1
        if q > 0 and q == (1 << k) and k not in used:
            used.add(k)
            place[k] = (b, i, 0)
        else:
            extra.append([b, i, c])
    if len(extra) > 8:
        return None
    return fractions.Fraction(g), base, place, extra


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return abs(a)


def _discriminate(rec, insts, byslot):
    """When one slot is a register in some instances and an expression in others, find the bit.

    op12682's operand 3 is an address expression in 3,776 of its eight-byte instances and a plain
    register in 22. That is not two maps of one field, it is a MODE, and without the bit that
    selects it a renderer has to guess which reading a given instruction wants - so it renders 22
    instructions as expressions or 3,776 as registers, and either way the bytes come back wrong.
    The bit is measured, not assumed: it has to partition every instance of this form perfectly,
    and if no single bit does, the record says so instead of pretending.
    """
    key, kind = rec["operand"], rec.get("kind")
    fam = (kind or "").split(".", 1)[0]
    if len(byslot.get(key) or ()) < 2:
        return
    mine, raws = [], []
    for raw, ops in insts:
        if key >= len(ops):
            continue
        fams = {k2.split(".", 1)[0] for k2, _v in _subvalues(*ops[key])}
        if not fams:
            continue
        raws.append(bytes(raw))
        mine.append(fam in fams)
    if not raws or all(mine) or not any(mine):
        return
    width = min(len(r) for r in raws)
    for b in range(width):
        for i in range(8):
            bit = [(r[b] >> i) & 1 for r in raws]
            for want in (0, 1):
                if all((v == want) == m for v, m in zip(bit, mine)):
                    rec["kind_bit"] = [b, i, want]
                    return
    rec["kind_bit"] = None


# A PINNED field ranks with a mapped one: it is fully determined, just not by any bit.
# Below this many instances of a form, a fitted verdict is weak evidence and a measured map is
# preferred outright. See the note at the preference below for what that was priced at.
THIN = 20

RANK = {"refuted": 0, "ungated": 0, "undetermined": 0, "degenerate": 1,
        "conditional": 2, "pinned": 3, "table": 3, "verified": 3}


def fit_all(op, insts, gate=None):
    """One set of records per (opcode, length).

    `gate`, when given, is a SECOND population the record is scored against after being fitted -
    the corpus, when the fit was allowed the wider union of corpus and build cache. The union is
    there to break degeneracy: 1,910 fields carry one value across the corpus alone, so their step
    and base are not separately determined and they block 229 (opcode, length) pairs covering
    35.5% of every instruction Apple ships. The cache scan holds encodings the corpus does not.
    But a map determined by the cache must still be CONSISTENT with the corpus to be authorable,
    so the corpus is scored last and it is what the verdict reports.
    """
    rows = []
    gate_by_len = by_length(gate) if gate else {}
    for length, group in sorted(by_length(insts).items()):
        start = len(rows)
        byslot = collections.defaultdict(set)
        for _raw, ops in group:
            for i, o in enumerate(ops):
                for k2, _v in _subvalues(*o):
                    byslot[i].add(k2.split(".", 1)[0])
        for rec in fit(op, group):
            rec["length"] = length
            # Bits outside this form do not exist. They contributed nothing to the fit either -
            # every read guards on the instruction's length - so dropping them changes no value
            # and stops an encoder being asked to write past the end of the instruction.
            rec["positions"] = [p for p in rec.get("positions", []) if p[1] < length]
            if rec.get("order"):
                rec["order"] = [o for o in rec["order"] if o[0] < length]
            rescore(rec, group)
            # THE CORPUS IS NOT THE ONLY INSTRUMENT. A field that carries one value across every
            # instruction Apple ships cannot be determined by any amount of scoring - the
            # population is the limit, not the fitter - but its bits can still be FLIPPED and the
            # result read back through the decoder. 1,431 of the 2,321 fields the corpus cannot
            # determine turn out to be measurable this way. The measured map is then held to the
            # same standard as a fitted one: it has to reproduce every instance of its form.
            # ALSO FOR A MAP THAT ALREADY EXPLAINS EVERY INSTANCE. Explaining the corpus is not
            # the same as being complete: op12674's operand 9 takes two values across 3,453
            # instances, so one bit reproduces all of them while three more bits of the same
            # field go unclaimed - and an encoder that does not know about them writes zeros
            # where Apple wrote something. The measured map is taken only when it reproduces
            # every instance too AND accounts for strictly more bits.
            if rec["verdict"] != "table":
                import g17diff
                kind = rec.get("kind") or ""
                got, _why = g17diff.field(op, length, rec["operand"],
                                          kind.split(".", 1)[1] if kind.startswith("expr.") else "",
                                          group)
                if got is None and "--modes" in sys.argv:
                    # A mode-selected map is real - g17diff.moded finds thirteen of them and
                    # refuses the rest - but offering one changes the SHAPE of its form, and
                    # measured end to end that cost 47 instructions more than it bought. It stays
                    # opt-in until the acceptance test is byte-level rather than value-level.
                    got, _why = g17diff.moded(op, length, rec["operand"],
                                              kind.split(".", 1)[1]
                                              if kind.startswith("expr.") else "", group)
                if got is None and rec["verdict"] not in ("verified", "table"):
                    # ONLY WHERE THERE IS NOTHING TO LOSE. A third-order map that reads every
                    # instance can still encode worse than the linear map it replaces, and the
                    # "strictly more bits" guard does not apply to a polynomial - so it was
                    # replacing good maps as well as recovering bad ones. Offered only to fields
                    # that have no usable map at all, its 45 recoveries cost nothing.
                    # A THIRD ORDER, bounded by the interaction graph's triangles rather than by
                    # the bit count: a triple can only carry a term when all three of its pairs
                    # already interact. 45 fields need exactly this - and shipping them cost 549
                    # byte-exact instructions that are not yet explained, so it is behind a flag.
                    got, _why = g17diff.cubic(op, length, rec["operand"],
                                              kind.split(".", 1)[1]
                                              if kind.startswith("expr.") else "", group)
                if got is None:
                    # NOT EVERY FIELD IS A SUM OF ITS BITS. When a bit's measured coefficient
                    # depends on the other bits, that is a product term, and the second
                    # difference measures it directly.
                    got, _why = g17diff.quadratic(op, length, rec["operand"],
                                                  kind.split(".", 1)[1]
                                                  if kind.startswith("expr.") else "", group)
                if got is None and rec["verdict"] == "degenerate" and "--pinned" in sys.argv:
                    # NOTHING IS UNDETERMINED HERE: the form pins the operand, and a field with no
                    # encoding cannot be encoded wrongly.
                    got, _why = g17diff.pinned(op, length, rec["operand"],
                                               kind.split(".", 1)[1]
                                               if kind.startswith("expr.") else "", group)
                if got is None:
                    # THE LINEAR MAP CAN BE THE WRONG MAP even when it explains every instance.
                    # The differential says when to suspect it - the constant it measures differs
                    # between base points - and enumerating the field's patterns through the
                    # decoder answers it exactly.
                    got, _why = g17diff.code_table(op, length, rec["operand"],
                                                   kind.split(".", 1)[1]
                                                   if kind.startswith("expr.") else "", group)
                    if got is None:
                        # LAST, BECAUSE IT MAKES THE SMALLEST CLAIM. A table over the patterns
                        # Apple actually witnesses is exact on all of them and refuses every
                        # other, so it cannot invent an encoding for a value nobody has shown it.
                        # It reaches fields no polynomial does: op10279's operand 1 at twelve
                        # bytes has seventeen moving bits, no interacting pairs, and fifteen
                        # patterns carrying fifteen values with no ambiguity. Priced with g17ab
                        # across every unrecovered field before being wired in: 232 byte-exact
                        # gained, 0 lost, 10 forms improved, none worsened.
                        got, _why = g17diff.witness_table(op, length, rec["operand"],
                                                          kind.split(".", 1)[1]
                                                          if kind.startswith("expr.") else "",
                                                          group)
                elif got is not None and not got.get("poly") and rec["verdict"] == "verified":
                    # A VERDICT EARNED ON TWELVE INSTANCES IS WEAK EVIDENCE. `verified` means the
                    # map explains every instance of its form, and where a form has few of them
                    # that is a small claim: op590's eight-byte form reads all 29 of its witnesses
                    # and re-encodes a held-out instance to different bytes. The differential does
                    # not care how many witnesses there are - it measures a coefficient by
                    # flipping a bit - so where the fitter is weakest the measurement is not.
                    # Priced with g17ab over all 329 thin forms before being made a rule:
                    # 35 instructions gained, 1 lost, 11 forms improved and 1 worsened.
                    if len(group) >= THIN:
                        have = len(rec.get("positions") or []) + len(rec.get("extra") or [])
                        if (len(got.get("positions") or [])
                                + len(got.get("extra") or [])) <= have:
                            got = None
                if got:
                    # THE BAN APPLIES TO THE DIFFERENTIAL TOO. Its table search chooses bits by
                    # what the decoder does when they move, which is the right instrument and no
                    # reason to be exempt: four of its tables had adopted a bit that selects the
                    # opcode, and a map that owns one writes a different instruction rather than
                    # a wrong value. Refusing the candidate leaves the fitted map standing, which
                    # is the weaker claim and the correct one.
                    taken = {(b, i) for b, i in (got.get("order") or [])}
                    taken |= {(p[1], p[2]) for p in (got.get("positions") or [])}
                    taken |= {(e[0], e[1]) for e in (got.get("extra") or [])}
                    if taken & opcode_bits(op, length):
                        got = None
                if got:
                    if got.get("verdict") == "table":
                        rec.update(step=None, base=None, positions=[], extra=[])
                    rec.update(got)
                    rec["verdict"] = got.get("verdict", "verified")
                    if rec["verdict"] == "pinned":
                        rec.update(step=None, base=None, positions=[], extra=[])
                    rec["source"] = "differential"
                    # certified against the same instances as everything else
                    rescore(rec, group)
            _discriminate(rec, group, byslot)
            if gate is not None:
                rec["union_n"] = len(group)
                rec["union_explains"] = rec.get("explains")
                g = gate_by_len.get(length) or []
                if g:
                    rescore(rec, g)
                    rec["gate_n"] = len(g)
                else:
                    # Determined by the cache and never seen by the corpus. That is not a corpus
                    # verdict and must not be reported as one.
                    rec["gate_n"] = 0
                    if rec["verdict"] in ("verified", "table"):
                        rec["verdict"] = "ungated"
            rows.append(rec)
        _unforced(op, length, group, rows[start:])
    return rows


def form_bits(ops, rows=None, measured=None):
    """What Apple's own instances say about the bits the specification calls fixed.

    Two different things, both measured per (opcode, length) and neither inferred:

    `unforced` - a bit classed `forced` that Apple VARIES across the instances of this form. It
    is an operand or modifier bit that the classification got wrong, and an encoder that leaves
    it at the specified value reproduces a different instruction: op10282's b4.6 is one, and it
    is 44 of its instances in the first 400 programs.

    `constant` - a bit classed forced or opcode whose value across every instance CONTRADICTS the
    value the specification declares. There are four in the whole corpus, which is the right
    order of magnitude for a genuine error and the reason this is recorded rather than assumed.
    """
    if rows is None:
        rows, _ = corpus_instances(ops)
    out = {}
    for op in ops:
        for length, group in by_length(rows.get(op) or []).items():
            raws = [bytes(r) for r, _ in group]
            if not raws:
                continue
            # A PER-FORM SPEC OR NOTHING, because `spec_for(op, length)` FALLS BACK to the
            # opcode-wide specification when no per-form entry exists - and that fallback was
            # derived from a DIFFERENT length, so comparing this form's bits against it compares
            # bits that are not the same bit. `unforced` and `constant` are the two findings that
            # read the spec, and both are meaningless under the fallback: an `unforced` bit tells
            # an encoder that a bit Apple fixes is free to search, which is the wrong direction to
            # be wrong in.
            #
            # IT CHANGES NOTHING ABOUT THE SHIPPED FILE. All 784 forms of the `spans` population
            # have a per-form entry, so the fallback never fires there, and the control run in
            # docs/archive/g17-map-refit.md reproduces isa/g17-form-bits.json key for key and bit for bit.
            # On the union population 763 of 1,662 forms fall back, and the fallback manufactured
            # 1,315 of 1,323 `constant` contradictions and 282 of 785 `unforced` bits.
            # `selects_opcode` below never reads the spec and is unaffected either way.
            spec = (g17encode.forms().get((op, length)) or {}).get("bits") or {}
            unf, const = [], {}
            for k, d in spec.items():
                if not isinstance(d, dict) or d.get("class") not in ("forced", "opcode"):
                    continue
                b, i = (int(x) for x in k.split("."))
                if b >= length:
                    continue
                vals = {(r[b] >> i) & 1 for r in raws}
                if len(vals) > 1:
                    if d["class"] == "forced":
                        unf.append([b, i])
                elif next(iter(vals)) != d.get("value"):
                    const[k] = next(iter(vals))
            # A BIT THAT SELECTS AN OPCODE, WHATEVER THE SPECIFICATION CALLS IT. op10369's b1.0
            # is classed `forced`, and flipping it decodes as op10372 - so an operand map that
            # owns it changes the instruction whenever it writes a value needing that bit. Six
            # fields fail the encode-direction check for exactly this reason, every one of them
            # reporting "opcode changed" rather than a wrong value. The ban on opcode bits was
            # keyed on the classification; this is keyed on what the bit DOES.
            sel = []
            if measured is not None:
                # SEVERAL BASE POINTS, because whether a bit selects the opcode DEPENDS ON THE
                # REST OF THE INSTRUCTION. This flipped raws[0] only, and op586's b3.3 is what
                # showed that up: flipped across 24 real instances it moves 586 -> 585 at eight
                # of them and merely re-reads operand 2 as an address expression at the other
                # sixteen. One base point had roughly a one-in-three chance of seeing it, and did
                # not - so 221 of op586's 222 near misses differ at exactly that bit.
                #
                # It is the same discipline the differential prober already uses (g17diff spreads
                # BASES=12 across the corpus) and it had been dropped here. Strided rather than
                # taken from the front, for the reason in g17corpus.corpus_programs.
                step = max(1, len(raws) // FORMBASES)
                bases = raws[::step][:FORMBASES]
                probe, order = [], []
                for raw in bases:
                    for b in range(length):
                        for i in range(8):
                            m = bytearray(raw)
                            m[b] ^= 1 << i
                            probe.append(bytes(m))
                            order.append((b, i))
                got = measured(bases)
                # WHAT THE BIT DOES, over every base point, in four outcomes: it changes the
                # opcode, it changes an operand's KIND (a register slot becoming an address
                # expression), it changes only a VALUE, or nothing.
                #
                # The ban is the bits that NEVER carry a value. "Changes the opcode somewhere"
                # is not the same test and it over-bans: op586's b1.7 moves the opcode at 21 of
                # 24 base points and carries operand 3's value at the other three, so banning it
                # leaves that operand with no home at all. Its b3.3 changes an operand's kind at
                # 22 and the opcode at 2 and carries a value at NONE - that is a selector, and an
                # operand map holding it writes a different instruction rather than a wrong
                # number. A bit that carries a value at even one base point is an operand bit
                # somewhere, and the honest reading is that (586,4) is more than one form.
                roles = collections.defaultdict(collections.Counter)
                for (b, i), d, base in zip(order, measured(probe),
                                           [g for g in got for _ in range(length * 8)]):
                    if base is None or d is None:
                        roles[(b, i)]["refused"] += 1
                    elif d[0] != base[0]:
                        roles[(b, i)]["opcode"] += 1
                    elif len(d[1]) != len(base[1]):
                        roles[(b, i)]["arity"] += 1
                    elif any(x.split(":")[0] != y.split(":")[0] for x, y in zip(base[1], d[1])):
                        roles[(b, i)]["kind"] += 1
                    elif d[1] != base[1]:
                        roles[(b, i)]["value"] += 1
                    else:
                        roles[(b, i)]["same"] += 1
                for (b, i), c in sorted(roles.items()):
                    if (c["opcode"] or c["kind"] or c["arity"]) and not c["value"]:
                        sel.append([b, i])
            if unf or const or sel:
                out["%d,%d" % (op, length)] = {"unforced": unf, "constant": const,
                                               "selects_opcode": sel, "instances": len(raws)}
    return out


def _unforced(op, length, insts, recs):
    """Mark the `forced` bits that Apple's own instances do not agree on.

    A bit the specification calls forced is one an encoder must leave alone, and the assembler
    asserts exactly that - which refused 192 of op17257's 265 instances, because several of its
    operand bits are classed forced and Apple varies them. A forced bit that VARIES across the
    instances of its own form is not forced; the classification is wrong and the measurement says
    so. One that never varies is left alone, so the assertion still does its job.
    """
    raws = [bytes(r) for r, _ in insts]
    if not raws:
        return
    width = min(len(r) for r in raws)
    varying = {(b, i) for b in range(width) for i in range(8)
               if len({(r[b] >> i) & 1 for r in raws}) > 1}
    forced = spec_class_bits(op, length, "forced") & varying
    if not forced:
        return
    for rec in recs:
        took = {(b, i) for _, b, i, _ in (tuple(p) for p in (rec.get("positions") or []))}
        took |= {(b, i) for b, i, _ in (rec.get("extra") or [])}
        took |= {(b, i) for b, i in (rec.get("order") or [])}
        hit = sorted(took & forced)
        if hit:
            rec["unforced"] = [list(x) for x in hit]


def fit(op, insts):
    """One record per operand field of `op`, each with a verdict against every instance.

    Two corrections the single-witness fit could not make. First the step is RATIONAL: a register
    field whose printed number advances by one for every two encoded units carries a low bit that
    is not part of the register index, and forcing an integer step refutes a map that is merely
    scaled wrong. Second, observations are grouped by operand KIND - a slot that is a register in
    one instruction and an immediate in the next has two maps, and fitting one line through both
    refutes a field that is fine.
    """
    width = len(bytes(insts[0][0])) if insts else None
    ext, steps, _, _ = g17encode.extended_fields(op, length_hint=width)
    # THE CLASSIFIER'S SEED CAN CONTAIN AN OPCODE BIT, and a seeded one is as damaging as a
    # searched one. Decoding survives it - the bit never varies, so the fitted base absorbs it -
    # but encoding scatters the value across the same positions and writes the opcode bit with
    # whatever the value says, which destroys the instruction. Dropping it costs nothing: a
    # constant bit carries no information, and the base absorbs it either way.
    banned = opcode_bits(op, width or 0)
    # EVERY OPERAND THE DECODER PRINTS, not only the ones the classifier found bits for. op12674
    # prints ten operands at sixteen bytes and the classifier has bits for nine; operand 9 had no
    # record of any kind, so it was neither a named operand nor a modifier and its four bits were
    # left to whatever the modal fill wrote - 137 of its instances, every one of them a near
    # miss. A field with no seed still has a value to explain, and the exact solve and the
    # differential can both find its bits from nothing.
    printed = set()
    for _raw, ops in insts:
        for i2, o in enumerate(ops):
            if _subvalues(*o):
                printed.add(i2)
    rows = []
    for key in sorted({k for k in ext if isinstance(k, int)} | printed):
        bykind = collections.defaultdict(list)
        for raw, ops in insts:
            if key >= len(ops):
                continue
            for k2, v in _subvalues(*ops[key]):
                bykind[k2].append((bytes(raw), v))
        for kind, rows_k in sorted(bykind.items()):
            # A sub-field of an expression has its own bits when the classifier found them; the
            # whole operand's bits are the fallback, and either way the positions are re-derived
            # from measurement below if the seeded set does not explain the instances.
            sub = kind.split(".", 1)[1] if kind.startswith("expr.") else None
            positions = [tuple(p) for p in
                         ((ext.get((key, sub)) or ext.get(key) or []) if sub
                          else (ext.get(key) or []))
                         if (p[1], p[2]) not in banned]
            obs = [(_encoded(positions, raw), v) for raw, v in rows_k]
            counts = collections.Counter(v for _, v in obs)
            rec = {"opcode": op, "operand": key, "kind": kind, "bits": len(positions),
                   "positions": [list(p) for p in positions], "n": len(obs),
                   # what Apple actually puts here, so a modifier slot the assembler does not
                   # model gets a witnessed default instead of an invented zero
                   "modal": counts.most_common(1)[0][0] if counts else None,
                   "seen": [v for v, _ in counts.most_common(8)]}
            uniq = sorted(set(obs))
            # DEGENERACY IS ABOUT THE VALUE, not about the pair. A field whose bits vary while
            # its printed value never does fits a line of slope ZERO, and slope zero scored
            # explains=1.0 and came out `verified` - a map that predicts one value everywhere,
            # certified against the instances that made it. That is the unfalsifiable pass this
            # file exists to prevent, and seeding an expression sub-field with the whole
            # operand's bits walked straight into it.
            if len({v for _, v in obs}) < 2:
                rec["verdict"] = "degenerate" if obs else "undetermined"
                rows.append(rec)
                continue
            # Candidate lines from pairs, scored by how many observations they explain. Both
            # loops are bounded: the pair search is quadratic in distinct values and the scoring
            # linear in instances, so on an opcode with a thousand distinct operand values and
            # five thousand instances it is a hundred million operations for ONE field. Score
            # candidates on a sample, then score the winner against every instance.
            probe = uniq if len(uniq) <= 60 else uniq[::max(1, len(uniq) // 60)][:60]
            score = obs if len(obs) <= 500 else obs[::max(1, len(obs) // 500)][:500]
            best = None
            for i in range(len(probe)):
                for j in range(i + 1, min(len(probe), i + 25)):
                    (e0, v0), (e1, v1) = probe[i], probe[j]
                    if e1 == e0:
                        continue
                    st = fractions.Fraction(v1 - v0, e1 - e0)
                    bs = v0 - e0 * st
                    if bs.denominator != 1:
                        continue
                    hit = sum(1 for e, v in score if bs + e * st == v)
                    if best is None or hit > best[0]:
                        best = (hit, st, int(bs))
            if best is not None:
                st, bs = best[1], best[2]
                best = (sum(1 for e, v in obs if bs + e * st == v), st, bs)
            if best is None:
                # NO PAIR TO FIT means the seeded bits do not move while the value does - which
                # is a statement about the seed, not about the field. Carry on to the exact
                # solve and the table search rather than calling the field refuted here.
                best = (0, fractions.Fraction(0), 0)
            hit, st, bs = best
            extra = []
            if hit < len(obs) and st != 0:
                # BITS THE CLASSIFIER NEVER FOUND, and not necessarily binary weights of the
                # field. op10864's fourth operand needs one bit worth 144 - a register-bank
                # selector, which no power-of-two weight can express and which leaves the map
                # looking like two different linear fields. Fix the slope, drop the base to the
                # lowest any instance implies so every residual is non-negative, then buy the
                # residual down one bit at a time with whatever integer coefficient fits.
                low = min(v - e * st for e, v in obs)
                if fractions.Fraction(low).denominator == 1:
                    bs2 = int(low)
                    # Bound the search. It is O(rounds x bytes x bits x candidates x n), which
                    # on an opcode with five thousand instances runs for ten seconds a field and
                    # would take hours over the whole table. A few hundred instances carry the
                    # same signal; the verdict is still scored against every one of them below.
                    pairs = list(zip([v - bs2 - e * st for e, v in obs],
                                     [r for r, _ in insts_for_kind(insts, key, kind)]))

                    SAMPLE = 300
                    if len(pairs) > SAMPLE:
                        stride = len(pairs) // SAMPLE
                        pairs = pairs[::stride][:SAMPLE]
                    resid = [x for x, _ in pairs]
                    used = set(opcode_bits(op, min(len(r) for _, r in pairs) if pairs else 0))
                    raws = [r for _, r in pairs]
                    for _ in range(3):
                        if not any(resid):
                            break
                        pick = None
                        width = min(len(r) for r in raws) if raws else 0
                        for b in range(width):
                            for i2 in range(8):
                                if (b, i2) in used:
                                    continue
                                bit = [(r[b] >> i2) & 1 for r in raws]
                                if len(set(bit)) < 2:
                                    continue
                                cand = collections.Counter(
                                    rz for rz, bt in zip(resid, bit) if bt and rz)
                                for c, _ in cand.most_common(6):
                                    if fractions.Fraction(c).denominator != 1:
                                        continue
                                    z = sum(1 for rz, bt in zip(resid, bit) if rz - c * bt == 0)
                                    if pick is None or z > pick[0]:
                                        pick = (z, b, i2, int(c), bit)
                        if pick is None or pick[0] <= sum(1 for rz in resid if rz == 0):
                            break
                        _, b, i2, c, bit = pick
                        resid = [rz - c * bt for rz, bt in zip(resid, bit)]
                        used.add((b, i2))
                        extra.append([b, i2, c])
                    if not any(resid) and extra:
                        # re-score the completed map against EVERY instance, not the sample
                        full = 0
                        for (e, v), (raw, _) in zip(obs, insts_for_kind(insts, key, kind)):
                            add = sum(c for b, i2, c in extra
                                      if b < len(raw) and (raw[b] >> i2) & 1)
                            if bs2 + e * st + add == v:
                                full += 1
                        if full > hit:
                            hit, bs = full, bs2
                        else:
                            extra = []
            # SOLVE THE STEP FROM THE VALUES, then find the bits by measurement. The pair fit can
            # only work with the bits the classifier already found, and when that set is too small
            # it cannot reach the field at all: op12682's operand 1 takes 8,388,608 and 1,048,576
            # and 7,340,032 through ONE seeded bit, so the best line through them has slope zero
            # and the record came out degenerate - while the specification itself names three more
            # bytes that move it. The value set determines the step exactly, as the gcd of the
            # differences; the base is then fixed by requiring every encoded value to be a
            # non-negative integer; and each weight is a bit that must agree with some instruction
            # bit on EVERY instance. Nothing here is chosen greedily and nothing is fitted twice.
            if hit < len(obs):
                pairs = insts_for_kind(insts, key, kind)
                ban = opcode_bits(op, min(len(r) for r, _ in pairs) if pairs else 0)
                got = _solve_by_gcd(obs, [r for r, _ in pairs], ban)
                if got:
                    st, bs, place = got
                    rec["positions"] = [[k, b, i2, iv] for k, (b, i2, iv) in sorted(place.items())]
                    extra, hit = [], len(obs)
                else:
                    got = _solve_linear(obs, [r for r, _ in pairs], ban,
                                        moved_bits(op, length_of(pairs), key, sub or ""))
                    if got:
                        st, bs, place, extra = got
                        rec["positions"] = [[k, b, i2, iv]
                                            for k, (b, i2, iv) in sorted(place.items())]
                        hit = len(obs)
            # A FIELD THAT IS A TABLE, not an integer. op10279's operand 4 takes fifteen values
            # and its bits are a code whose ALL-ZEROS pattern means 1024, the largest value - so
            # every linear model refutes it, and that one field is 12,096 near misses in the
            # corpus. Two things are needed to see it. The mapping has to be read as a table
            # rather than a sum, AND the bit set has to be searched for, because the classifier's
            # set is incomplete: it had three bits in byte 10 and the field also uses byte 11.
            # Greedily add whichever bit most reduces the patterns that map to more than one
            # value, and stop when the pattern determines the value.
            if hit < len(obs):
                raws = [r for r, _ in insts_for_kind(insts, key, kind)]
                vv = [v for _, v in obs]
                if raws and len({v for v in vv}) <= 64:
                    width = min(len(r) for r in raws)
                    # Never let the search take an OPCODE bit. Flipping one decodes as a
                    # different instruction, so it cannot be part of this instruction's operand,
                    # and an encoder that writes it destroys the opcode - which is what the
                    # identity assertion in the assembler caught on op554. `forced` bits are
                    # fair game and deliberately so: b11.0 is forced by the specification and is
                    # genuinely part of op10279's operand 4.
                    # THE SAME BAN AS EVERY OTHER SEARCH. This filtered on the CLASSIFICATION
                    # while the linear searches had moved to the measurement, so the table path
                    # was the one way a selector could still become an operand: op586's b3.3
                    # changes operand 2 from a register to an address expression at 22 of 24 base
                    # points and carries a value at none, and the table search took it for
                    # operand 3 anyway. That is 222 of op586's instructions - every one of them a
                    # near miss, all differing at exactly that bit.
                    banned = opcode_bits(op, width)
                    cand = [(b, i2) for b in range(width) for i2 in range(8)
                            if len({(r[b] >> i2) & 1 for r in raws}) > 1
                            and (b, i2) not in banned]
                    # Search on a sample; the chosen bits are scored against every instance
                    # below, so a table only survives if it is exact on the whole corpus.
                    if len(raws) > 600:
                        st = len(raws) // 600
                        sr, sv = raws[::st][:600], vv[::st][:600]
                    else:
                        sr, sv = raws, vv

                    def impurity(bits):
                        g = collections.defaultdict(collections.Counter)
                        for r, v in zip(sr, sv):
                            g[tuple((r[b] >> i2) & 1 for b, i2 in bits)][v] += 1
                        return sum(sum(c.values()) - c.most_common(1)[0][1]
                                   for c in g.values()), g
                    # Minimise MISCLASSIFIED INSTANCES, not the count of impure groups. The group
                    # count is not monotone - splitting can raise it while making progress - and
                    # greedily minimising it wanders off into unrelated bits.
                    chosen, impure, groups = [], None, {}
                    for _ in range(10):
                        best = None
                        for c in cand:
                            if c in chosen:
                                continue
                            imp, g = impurity(chosen + [c])
                            if best is None or imp < best[0]:
                                best = (imp, c, g)
                        if best is None:
                            break
                        impure, groups = best[0], best[2]
                        chosen.append(best[1])
                        if impure == 0:
                            break
                    if chosen and impure == 0 and len(groups) <= 64:
                        # score the candidate table against EVERY instance
                        full = collections.defaultdict(collections.Counter)
                        for r, v in zip(raws, vv):
                            full[tuple((r[b] >> i2) & 1 for b, i2 in chosen)][v] += 1
                        if any(len(c) > 1 for c in full.values()):
                            impure = 1
                        else:
                            groups = full
                    if chosen and impure == 0 and len(groups) <= 64:
                        tab = {"".join(str(x) for x in k): c.most_common(1)[0][0]
                               for k, c in sorted(groups.items())}
                        # WHICH SPELLING TO WRITE. Several patterns can carry one value and the
                        # encoder took whichever came first, which is sorted order and therefore
                        # arbitrary. op10372's operand 2 has eight patterns for 12; SIX of them
                        # read back as 12 in all 49 instances and the first one does so in 6.
                        # That single choice made a sound map look like a field that writes the
                        # wrong value 113 times in 120, and it cost the map its verdict.
                        #
                        # The pattern Apple writes MOST OFTEN for a value is the one whose other
                        # bits agree with the rest of the form, and the counts are already here.
                        n_pat = {k: sum(c.values()) for k, c in groups.items()}
                        prefer = {}
                        # `val` AND NOT `key`: this loop used to bind `key`, which is the
                        # ENCLOSING loop's operand index, and left it holding a stringified
                        # operand VALUE for the rest of that iteration. Every later use of the
                        # operand index in the same pass then got '63' instead of 3 - the record
                        # it built carried operand='63', and g17diff compared a str against an
                        # int. It survived because only the FULL run reaches an opcode with a
                        # prefer table, and the full run crashed earlier on an unset `lim`.
                        for k, c in sorted(groups.items()):
                            val, pat = str(c.most_common(1)[0][0]), "".join(str(x) for x in k)
                            if val not in prefer or n_pat[k] > n_pat[tuple(int(x) for x in prefer[val])]:
                                prefer[val] = pat
                        rec.update(verdict="table", explains=1.0,
                                   order=[[b, i2] for b, i2 in chosen],
                                   table=tab, prefer=prefer)
                        rows.append(rec)
                        continue
            # RE-DERIVE THE BIT POSITIONS INSIDE THIS FORM. The classifier's map is not keyed by
            # length, so it can place a weight in a byte this form does not have: op554's
            # four-byte movimm was told weights 2, 3 and 5 live in byte 7, which made registers
            # 113 to 118 unencodable in a form Apple encodes them in constantly. Given the fitted
            # step and base, each weight of enc = (value - base) / step is a bit that must agree
            # with some in-form instruction bit on EVERY instance. That is a measurement, not a
            # fit, and it either finds the bit or says there is none.
            if st and best is not None:
                pairs = insts_for_kind(insts, key, kind)
                width = min(len(r) for r, _ in pairs) if pairs else 0
                enc = []
                for (raw, _), (e, v) in zip(pairs, obs):
                    q = fractions.Fraction(v - bs) / st
                    enc.append((raw, int(q)) if q.denominator == 1 and q >= 0 else (raw, None))
                if enc and all(e is not None for _, e in enc):
                    mx = max(e for _, e in enc)
                    banned = opcode_bits(op, width)
                    place = {}
                    for k in range(mx.bit_length()):
                        want = [(e >> k) & 1 for _, e in enc]
                        if len(set(want)) < 2:
                            continue
                        for b in range(width):
                            for i2 in range(8):
                                if (b, i2) in banned:
                                    continue
                                got = [(r[b] >> i2) & 1 for r, _ in enc]
                                if got == want:
                                    place[k] = (b, i2, 0)
                                    break
                                if got == [1 - x for x in want]:
                                    place[k] = (b, i2, 1)
                                    break
                            if k in place:
                                break
                    if place:
                        good = sum(1 for r, e in enc
                                   if sum((((r[b] >> i2) & 1) ^ iv) << k
                                          for k, (b, i2, iv) in place.items()) == e)
                        if good > hit:
                            hit = good
                            rec["positions"] = [[k, b, i2, iv] for k, (b, i2, iv)
                                                in sorted(place.items())]
            frac = hit / float(len(obs))
            rec.update(step=str(st), base=bs, explains=round(frac, 4), extra=extra,
                       distinct_encoded=len({e for e, _ in obs}))
            if st == 0:
                # A LINE OF SLOPE ZERO predicts one value everywhere. It scored explains=1.0 on
                # the instances that made it and came out `verified`, which is the unfalsifiable
                # pass this file exists to prevent.
                rec.update(verdict="degenerate",
                           why="the best line through these instances has slope zero")
            elif frac == 1.0:
                rec["verdict"] = "verified"
            elif frac >= 0.6:
                rec["verdict"] = "conditional"     # a mode bit selects between two meanings
            else:
                rec["verdict"] = "refuted"
            rows.append(rec)
    return rows


def main():
    if "--moved-bits" in sys.argv:
        ops = sorted(g17slice.KNOWN_OPS)
        import g17same
        d = moved_by_flip(ops, measured=g17same.decode_many)
        with open(MOVEDBITS, "w") as fh:
            json.dump(d, fh, indent=1, sort_keys=True)
        print("%d forms have a bit whose operand was measured -> %s" % (len(d), MOVEDBITS))
        return
    if "--form-bits" in sys.argv:
        ops = sorted(g17slice.KNOWN_OPS)
        import g17same
        d = form_bits(ops, measured=g17same.decode_many)
        with open(FORMBITS, "w") as fh:
            json.dump(d, fh, indent=1, sort_keys=True)
        print("%d forms carry a bit the specification describes wrongly -> %s"
              % (len(d), FORMBITS))
        return
    if "--set" in sys.argv and sys.argv[sys.argv.index("--set") + 1] == "proven":
        ops = sorted(PROVEN)
    elif "--ops" in sys.argv:
        ops = [int(x) for x in sys.argv[sys.argv.index("--ops") + 1].split(",")]
    else:
        ops = sorted(g17slice.KNOWN_OPS)
        if "--limit" in sys.argv:
            lim = int(sys.argv[sys.argv.index("--limit") + 1])
            # ACROSS THE OPCODES, not the first N of them. A capped run is a probe, but a probe
            # that always looks at the same low-numbered opcodes answers a question about those
            # opcodes and gets quoted as an answer about the set. Striding costs nothing here.
            #
            # INSIDE the --limit branch, which is where it always belonged: the stride ran
            # unconditionally and read `lim` when no limit was given, so the FULL run - the only
            # one permitted to write the shared map - raised UnboundLocalError every time. The one
            # path that matters was the one path nobody took.
            ops = ops[::max(1, len(ops) // lim)][:lim]
    # A partial run is a probe, not a release. Writing it to the shared map file is how the
    # assembler silently lost 510 of its 511 opcodes twice today.
    out_path = OUT if set(ops) == set(g17slice.KNOWN_OPS) else OUT + ".partial"
    corpus, _ = corpus_instances(ops)
    union = None
    if "--union" in sys.argv:
        cache, _ = g17fields.instances(set(ops))
        union = {}
        for op in ops:
            merged = list(corpus.get(op) or [])
            seen = {bytes(raw) for raw, _ in merged}
            for raw, o in (cache.get(op) or []):
                if bytes(raw) not in seen:
                    merged.append((raw, o))
            union[op] = merged
    rows = corpus if CORPUS_INSTANCES else g17fields.instances(set(ops))[0]
    out, tally = [], collections.Counter()
    for op in ops:
        recs = fit_all(op, rows.get(op) or [])
        if union is not None:
            # The union fit is a CHALLENGER, never a replacement. A field the corpus already
            # determines is left alone; only a degenerate or refuted one is offered the wider
            # population, and the challenger has to survive being scored on the corpus to take
            # the slot. Otherwise the cache's extra encodings could quietly overwrite a map the
            # corpus had proven.
            have = {(r.get("length"), r["operand"], r.get("kind")): r for r in recs}
            for cand in fit_all(op, union.get(op) or [], corpus.get(op) or []):
                k = (cand.get("length"), cand["operand"], cand.get("kind"))
                cur = have.get(k)
                take = cur is None or RANK[cand["verdict"]] > RANK[cur["verdict"]]
                # A MAP ITS OWN WIDER POPULATION REFUTES IS NOT VERIFIED, whatever its verdict says.
                # The rank guard above exists so extra encodings cannot quietly overwrite a map the
                # corpus proved - but it also protects a map the corpus could not prove, because a
                # field fitted from ONE encoded value passes explains=1.0 trivially and is marked
                # verified. Those are the maps that mispredict: 68 of 352 verified numeric fields on
                # the compiler's own opcodes plus the atomic family disagree with the decoder on
                # instances the union contains, including op10090's operand 0, whose base is 425 and
                # whose union of 54 instances says 105.
                #
                # Opt-in, because it moves numbers the class table and slot-set derivation are
                # denominated in. Off by default: the audit is the deliverable until someone
                # decides to re-baseline.
                if not take and "--challenge-contradicted" in sys.argv and cur is not None:
                    take = not _explains_union(cur, union.get(op) or [])
                    if take:
                        cand["replaced_contradicted"] = True
                # THE OVERLAP LAW IS A PRECONDITION NOW, not a report after the swap. The first
                # re-baseline was refused because it moved overlap from 266 forms to 333 - a law
                # that had been MEASURED and left unenforced, so nothing stopped the swaps that
                # caused it.
                #
                # IT APPLIES TO A NEW RECORD TOO, and that is not a detail: of the 156 records it
                # refuses, 66 are fields the shipped map does not have at all. Gating only the
                # SWAPS leaves the increase at 276 forms; gating both, judged against the map as it
                # accumulates rather than a snapshot, gives 261 - below the 266 it started from.
                # A new field has no predecessor to be no worse than, so what it must not do is
                # take bits that already belong to one of its own siblings.
                # ONLY IN THE RE-BASELINE MODE, which is the mode the separation was measured in.
                # The same defect could be gated on the default path too, but widening a guard past
                # the population it was measured on is the failure this whole exercise is about, so
                # that needs its own number before it happens.
                if take and "--challenge-contradicted" in sys.argv:
                    sibs = [r for r in have.values() if r.get("length") == cand.get("length")]
                    if not (_no_new_overlap(cand, cur, sibs) if cur is not None
                            else _overlap_against(cand, sibs) == 0):
                        take = False
                        cand["refused_new_overlap"] = True
                if take:
                    cand["from_union"] = True
                    have[k] = cand
            recs = [have[k] for k in sorted(have, key=str)]
        for rec in recs:
            out.append(rec)
            tally[rec["verdict"]] += 1
    if out_path == OUT:
        # MEASURED, because this write is what erased the measurement. form_bits() only fills
        # `selects_opcode` when it is given a decoder; called without one it writes [] for every
        # form. So a full fit read the ban at its start and destroyed it at its end, and the ban
        # applied to exactly one fit - the one immediately after `--form-bits` - and to no other,
        # silently. The file that shipped had 102 forms and not a single selecting bit in it.
        import g17same
        with open(FORMBITS, "w") as fh:
            json.dump(form_bits(ops, corpus, measured=g17same.decode_many), fh,
                      indent=1, sort_keys=True)
    with open(out_path, "w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    print("operand fields checked against every cached instance of their opcode")
    for k in ("verified", "table", "pinned", "conditional", "degenerate", "refuted", "ungated",
              "undetermined"):
        print("   %-14s %d" % (k, tally[k]))
    print("   -> %s" % out_path)
    if "--report" in sys.argv:
        for r in out:
            nm = g17slice.KNOWN_OPS.get(r["opcode"], "?")
            print("   op%-6d %-9s l%-3s op%-2s %-5s n=%-5d %-12s %s"
                  % (r["opcode"], nm, r.get("length"), r["operand"], r.get("kind", ""),
                     r["n"], r["verdict"],
                     "step=%s base=%s explains=%s" % (r.get("step"), r.get("base"),
                                                      r.get("explains", ""))))


if __name__ == "__main__":
    main()
