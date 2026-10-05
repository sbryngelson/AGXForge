#!/usr/bin/env python3
"""Measure an operand field by FLIPPING its bits, instead of fitting a line to what Apple shipped.

Correlation can only explain what the corpus varies. 2,627 fields carry ONE value across all
184,349 instructions Apple ships, so their step and base are not separately determined and no
amount of scoring will determine them - the population is the limit, not the fitter. And a field
the corpus does vary can still be fitted wrong, because a bit that merely correlates costs nothing
to spend.

A differential has neither problem. Hold one of Apple's own instructions fixed, flip one bit,
decode again, and read what the operand became. The change is that bit's contribution, measured.
Doing it from several different base instructions is what separates a coefficient from a
coincidence: a linear bit gives the SAME normalised change from every base point, and a bit whose
change depends on the base is telling you the field has a mode, which is a finding rather than a
failure.

Normalised, because the sign is not the coefficient: flipping a bit that was 0 adds c and
flipping one that was 1 subtracts it.

WHAT THIS IS. A statement about Apple's instruction description, which is what tools/agx3dis was
built from - the same standing rule as everywhere else here. Nothing is dispatched, nothing is
patched, and every map this produces is checked against every instance of its form in the corpus
before it is allowed to be a map.

    python3 tools/g17diff.py 12682 14 1        one field
    python3 tools/g17diff.py --degenerate 40   the fields the corpus cannot determine
"""
import collections, fractions, itertools, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import g17canon, g17encode, g17opmap

MAXBITS = 48
# How many coefficient bits an encoder can be asked to choose between. The choice is a subset
# search, so this is the exponent in its cost; the cache below is what makes it affordable.
MAXEXTRA = 14
BASES = 12


def _value(tok, sub):
    """The numeric sub-value this printed operand carries, or None."""
    vals = g17canon.token_values(tok)
    return vals.get(sub or "")


# WIDENING THE SEARCH PAST THE SPECIFICATION'S ATTRIBUTION WAS TRIED AND REVERTED, and the reason
# is structural rather than empirical. candidates() already offers every bit the specification
# attributes to THIS operand plus every bit it attributes to nobody, so the only bits a wider
# search can add are ones it gives to a DIFFERENT operand - and a map that owns a neighbour's bit
# reads correctly, because the bits do correlate, and corrupts the neighbour when it writes.
# Measured: 162 more fields became measurable, 32 more forms were offered, not one of them
# encoded correctly, and instructions that are not instructions at all went from 9 to 13.


def candidates(op, length, key, sub=""):
    """The bits worth flipping: what the specification attributes to this operand, PLUS the bits
    it attributes to nobody.

    The attribution is incomplete, and incompletely in one direction. op10282's b4.6 is classed
    `forced` and named as moving no operand at all - and flipping it moves operand 1, which is why
    44 of its instances re-encoded to bytes Apple never wrote. A bit the specification gives to a
    DIFFERENT operand is still excluded; a bit it gives to no one is fair game, and the
    measurement decides. A bit that moves some other operand simply measures as zero here.
    """
    # BITS OUTSIDE THE FORM DO NOT EXIST. spec_for can fall back to a pooled specification whose
    # keys name bytes a shorter form does not have, and flipping one is an index error rather than
    # a measurement. It never fired while the differential ran on a few hundred fields; preferring
    # it on every thin form found it immediately.
    own = {c for c in g17opmap.moved_bits(op, length, key, sub) if c[0] < length}
    spec = g17encode.spec_for(op, length) or {}
    claimed = set()
    for k, d in spec.items():
        if isinstance(d, dict) and d.get("moved"):
            b, i = (int(x) for x in k.split("."))
            if b < length:
                claimed.add((b, i))
    ban = g17opmap.opcode_bits(op, length)
    # A BIT THE SPECIFICATION GIVES TO SOMEONE ELSE, when a flip says otherwise. This is not the
    # widening that was reverted above: that offered every neighbour's bit to every field and let
    # correlation decide, and correlation is exactly what reading cannot distinguish. A flip is
    # causal - g17opmap.moves() records which operand each bit MOVED when it was flipped in real
    # instructions - so a bit is added here only where the measurement puts it on THIS operand.
    # op998's b5.1 is the case: attributed elsewhere, moves operand 5 at eleven of sixteen base
    # points, and without it operand 5 has no complete bit set and every instrument refuses it.
    # PRICED AND OPT-IN. Measured end to end it is 9,748 byte-exact against 9,755 without: it
    # costs seven instructions and buys nothing on that axis, though it does convert 64 verified
    # maps into tables and takes near misses from 413 to 380. A change that costs more than it
    # buys does not ship, and the measurement is kept so the next person does not re-run it.
    moved = {c for c, keys in g17opmap.moves(op, length).items()
             if key in keys and c[0] < length and c not in ban} \
        if "--attributed" in sys.argv else set()
    free = [(b, i) for b in range(length) for i in range(8)
            if (b, i) not in claimed and (b, i) not in ban and (b, i) not in own]
    return (sorted(own | moved) + [c for c in free if c not in moved])[:MAXBITS]


def spread(op, length, key, sub, bases, cand):
    """{bit: {base index: normalised coefficient}} and the value at each base."""
    blob, index = [], []
    for bi, base in enumerate(bases):
        blob.append(base)
        index.append((bi, None))
        for c in cand:
            m = bytearray(base)
            m[c[0]] ^= 1 << c[1]
            blob.append(bytes(m))
            index.append((bi, c))
    got = g17encode.decode(blob)
    ref, out, dropped = {}, collections.defaultdict(dict), set()
    for j, (bi, c) in enumerate(index):
        if c is None:
            d = got.get(j)
            ref[bi] = (_value(d[1][key], sub)
                       if d and d[0] == op and key < len(d[1]) else None)
    for j, (bi, c) in enumerate(index):
        if c is None or ref[bi] is None:
            continue
        d = got.get(j)
        if d is None or d[0] != op or key >= len(d[1]):
            dropped.add(c)
            continue
        v = _value(d[1][key], sub)
        if v is None:
            dropped.add(c)
            continue
        was = (bases[bi][c[0]] >> c[1]) & 1
        out[c][bi] = (v - ref[bi]) if was == 0 else (ref[bi] - v)
    return out, ref, dropped


def mode_bits(op, length, key, sub, bases, cand, depth=4, pool=None):
    """The smallest set of bits whose value decides the other coefficients.

    A bit whose normalised coefficient differs between base instructions is selected by something.
    Add the bit that explains the most disagreement, and look again - checking INSIDE every
    assignment of what has been chosen so far, not inside one of them. Filtering the base points
    down to a single mode is how a two-bit answer looked complete when op12682's operand 1 needs
    three: b5.5, b5.6 and b5.7, which are adjacent.
    """
    sp, _ref, _drop = spread(op, length, key, sub, bases, cand)

    def ok(c, chosen):
        groups = collections.defaultdict(set)
        for bi, v in sp[c].items():
            groups[tuple((bases[bi][p] >> q) & 1 for p, q in chosen)].add(v)
        return all(len(g) == 1 for g in groups.values())

    chosen = []
    for _ in range(depth):
        bad = [c for c in sp if not ok(c, chosen)]
        if not bad:
            return chosen
        votes = collections.Counter()
        for y in (pool or cand):
            if y in chosen:
                continue
            votes[y] = sum(1 for c in bad if c != y and ok(c, chosen + [y]))
        y, n = votes.most_common(1)[0]
        if not n:
            return None
        chosen.append(y)
    return None


def measure(op, length, key, sub, bases, cand=None):
    """{(byte, bit): coefficient} plus the constant, measured; or (None, reason).

    A bit whose flip makes the decoder refuse the instruction, or decode it as a different
    opcode, is not an operand bit of this form and is dropped rather than counted - a rejection
    is information, not noise.
    """
    cand = cand or candidates(op, length, key, sub)
    if not cand or not bases:
        return None, "nothing to probe"
    blob, index = [], []
    for bi, base in enumerate(bases):
        blob.append(base)
        index.append((bi, None))
        for c in cand:
            m = bytearray(base)
            m[c[0]] ^= 1 << c[1]
            blob.append(bytes(m))
            index.append((bi, c))
    got = g17encode.decode(blob)
    ref = {}
    for j, (bi, c) in enumerate(index):
        if c is None:
            d = got.get(j)
            ref[bi] = (_value(d[1][key], sub)
                       if d and d[0] == op and key < len(d[1]) else None)
    if not any(v is not None for v in ref.values()):
        return None, "the operand does not read back from its own bytes"
    coef, dropped = collections.defaultdict(dict), set()
    for j, (bi, c) in enumerate(index):
        if c is None or ref[bi] is None:
            continue
        d = got.get(j)
        if d is None or d[0] != op or key >= len(d[1]):
            dropped.add(c)                      # not an operand bit: it selects the form
            continue
        v = _value(d[1][key], sub)
        if v is None:
            dropped.add(c)
            continue
        was = (bases[bi][c[0]] >> c[1]) & 1
        coef[c][bi] = (v - ref[bi]) if was == 0 else (ref[bi] - v)
    out, modal = {}, []
    for c in cand:
        if c in dropped:
            continue
        seen = set((coef.get(c) or {}).values())
        if len(seen) == 1:
            v = next(iter(seen))
            if v:
                out[c] = v
        elif len(seen) > 1:
            modal.append(c)                     # a mode selects this bit's coefficient
    if modal:
        return None, "%d bits have more than one coefficient: %s" % (
            len(modal), ["b%d.%d" % c for c in modal[:6]])
    if not out:
        return None, "no bit moves this operand"
    # The constant is what is left when every measured bit is accounted for, and it has to be the
    # same constant from every base instruction or the model is not additive.
    consts = {ref[bi] - sum(v for c, v in out.items() if (base[c[0]] >> c[1]) & 1)
              for bi, base in enumerate(bases) if ref[bi] is not None}
    if len(consts) != 1:
        return None, "the constant differs between base instructions: %s" % sorted(consts)[:4]
    return (out, consts.pop()), None


def _constant(insts, key, sub):
    """A map for a field that takes ONE value across these instances, or None."""
    seen = set()
    for _raw, ops in insts:
        if key >= len(ops):
            continue
        v = dict(g17opmap._subvalues(*ops[key])).get(("expr." + sub) if sub else ops[key][0])
        if v is None:
            return None
        seen.add(v)
        if len(seen) > 1:
            return None
    if len(seen) != 1:
        return None
    return {"step": "0", "base": next(iter(seen)), "positions": [], "extra": []}


def as_record(coef, const):
    """A measured coefficient set as this project's map: step, base, positions, extra."""
    g = 0
    for c in coef.values():
        g = g17opmap._gcd(g, c)
    if g <= 0:
        return None
    place, extra, used = {}, [], set()
    for (b, i), c in sorted(coef.items()):
        q = c // g
        k = q.bit_length() - 1
        if q > 0 and q == (1 << k) and k not in used:
            used.add(k)
            place[k] = (b, i)
        else:
            extra.append([b, i, c])
    if len(extra) > MAXEXTRA:
        return None
    return {"step": str(fractions.Fraction(g)), "base": const,
            "positions": [[k, b, i, 0] for k, (b, i) in sorted(place.items())],
            "extra": extra}


def explains(rec, insts, key, sub):
    """How many of these instances the measured map reproduces."""
    st = fractions.Fraction(rec["step"])
    ok = 0
    for raw, ops in insts:
        if key >= len(ops):
            continue
        v = g17opmap._subvalues(*ops[key])
        want = dict(v).get(("expr." + sub) if sub else ops[key][0])
        if want is None:
            continue
        e = 0
        for w, b, i, inv in (tuple(p) for p in rec["positions"]):
            if b < len(raw):
                e |= (((raw[b] >> i) & 1) ^ inv) << w
        add = sum(c for b, i, c in rec["extra"] if b < len(raw) and (raw[b] >> i) & 1)
        if rec["base"] + e * st + add == want:
            ok += 1
    return ok


def field(op, length, key, sub, insts):
    """A measured map for one field, or (None, why). `insts` are its corpus instances."""
    # SPREAD THE BASE POINTS. Taking the first few distinct encodings takes the first few
    # instructions of the first few programs, which resemble each other; a mode the corpus only
    # enters later is then invisible and the measurement certifies a map that reproduces 382 of
    # 6,193 instances. Sampling across the whole set is what makes the disagreement show up.
    seen, distinct = set(), []
    for raw, _ops in insts:
        b = bytes(raw)
        if b not in seen:
            seen.add(b)
            distinct.append(b)
    stride = max(1, len(distinct) // BASES)
    raws = distinct[::stride][:BASES]
    got, why = measure(op, length, key, sub, raws)
    if got is None:
        return None, why
    rec = as_record(*got)
    if rec is None:
        return None, "the coefficients do not fit this project's map shape"
    n = sum(1 for raw, ops in insts if key < len(ops))
    ok = explains(rec, insts, key, sub)
    if ok != n:
        return None, "measured, but reproduces %d of %d instances" % (ok, n)
    rec["explains"] = 1.0
    rec["measured"] = True
    return rec, None


MAXTABLE = 10
# How many bits a second-order expansion may span. The cost is one decode per pair, so this is
# quadratic in the bound, and the encoder has to solve a quadratic pseudo-Boolean equation over
# the same set.
MAXQUAD = 34


def quadratic(op, length, key, sub, insts):
    """The field as an exact SECOND-ORDER polynomial in its own bits, or (None, why).

    A single-bit differential measures how one bit moves the operand. When that answer depends on
    the other bits, the field is not a sum of bits - and "depends on the other bits" is precisely
    the signature of a product term. So measure the second difference:

        I(x,y) = f(b^x^y) - f(b^x) - f(b^y) + f(b)

    which is zero exactly when x and y do not interact. op12682's operand 1 at fourteen bytes -
    22 values, no linear map, no code table, and no set of mode bits that accounts for it - has 26
    first-order terms and 83 interaction terms, and the interaction table is IDENTICAL measured
    from two different base instructions. It reproduces all 1,359 of its instances.

    The expansion is taken around one real instruction and then rewritten in absolute terms, so
    the stored map does not carry a base instruction with it: with u_i = x_i XOR b_i and
    u_i = a_i + s_i x_i, a quadratic in u is a quadratic in x.
    """
    bases = _bases(insts, 3)
    if not bases:
        return None, "no instances"
    cand = candidates(op, length, key, sub)
    sp, _ref, _drop = spread(op, length, key, sub, bases, cand)
    bits = sorted(c for c, v in sp.items() if set(v.values()) != {0})
    if not bits:
        return None, "no bit moves this operand"
    if len(bits) > MAXQUAD:
        return None, "%d bits move this operand; too many to expand" % len(bits)
    base = bases[0]

    def val(d):
        if d is None or d[0] != op or key >= len(d[1]):
            return None
        return _value(d[1][key], sub)

    blob, index = [base], [()]
    for x in bits:
        m = bytearray(base)
        m[x[0]] ^= 1 << x[1]
        blob.append(bytes(m))
        index.append((x,))
    for x, y in itertools.combinations(bits, 2):
        m = bytearray(base)
        m[x[0]] ^= 1 << x[1]
        m[y[0]] ^= 1 << y[1]
        blob.append(bytes(m))
        index.append((x, y))
    got = decode = g17encode.decode(blob)
    f = {k: val(got.get(j)) for j, k in enumerate(index)}
    f0 = f[()]
    if f0 is None:
        return None, "the operand does not read back from its own bytes"
    # A FLIP THE DECODER REFUSES MEASURES NOTHING, and refusing the whole field over it throws
    # away the answer: op12682's operand 1 has 26 usable bits and pairs among them that do not
    # decode, and the expansion over what does decode still reproduces all 1,359 instances. So
    # skip what cannot be measured and let the check against every instance decide.
    lin = {x: f[(x,)] - f0 for x in bits if f.get((x,)) is not None}
    bits = [x for x in bits if x in lin]
    if not bits:
        return None, "every single flip made the instruction undecodable"
    inter = {}
    for x, y in itertools.combinations(bits, 2):
        c = f.get((x, y))
        if c is None:
            continue
        v = c - lin[x] - lin[y] - f0
        if v:
            inter[(x, y)] = v
    # rewrite around zero: u_i = a_i + s_i x_i with (a, s) = (0, 1) or (1, -1)
    const = f0
    linx = collections.Counter()
    quadx = collections.Counter()
    A = {x: (base[x[0]] >> x[1]) & 1 for x in bits}
    S = {x: (1 if not A[x] else -1) for x in bits}
    for x, c in lin.items():
        const += c * A[x]
        linx[x] += c * S[x]
    for (x, y), c in inter.items():
        const += c * A[x] * A[y]
        linx[x] += c * S[x] * A[y]
        linx[y] += c * S[y] * A[x]
        quadx[(x, y)] += c * S[x] * S[y]
    rec = {"verdict": "verified", "measured": True, "source": "differential",
           "poly": {"const": int(const),
                    "lin": [[x[0], x[1], int(c)] for x, c in sorted(linx.items()) if c],
                    "quad": [[x[0], x[1], y[0], y[1], int(c)]
                             for (x, y), c in sorted(quadx.items()) if c]},
           "step": None, "base": None, "positions": [], "extra": [], "explains": 1.0}
    n = ok = 0
    for raw, ops in insts:
        if key >= len(ops):
            continue
        want = dict(g17opmap._subvalues(*ops[key])).get(("expr." + sub) if sub else ops[key][0])
        if want is None:
            continue
        n += 1
        if g17encode.poly_value(rec["poly"], bytes(raw)) == want:
            ok += 1
    if ok != n:
        return None, "expanded, but reproduces %d of %d instances" % (ok, n)
    return rec, None


def pinned(op, length, key, sub, insts):
    """A field NO bit moves and whose value never varies, or (None, why).

    977 of the 1,182 fields still unrecovered are of this kind, and calling them `degenerate` -
    "step and base are not separately determined" - describes them wrongly. Nothing is
    undetermined: the form pins the operand. There is no encoding to get right because there is no
    encoding at all, so such a field cannot make an authored instruction wrong and must not block
    the form that contains it.

    Proved both ways: every bit the differential can reach leaves the operand alone, AND every
    instance of the form carries the same value. Either alone would be weaker - a bit outside the
    candidate set could move it, or the corpus could simply be narrow - and together they are the
    same two-sided argument used everywhere else here.

    MEASURED AND NOT SHIPPED, behind --pinned. Classifying 1,056 fields this way is more honest
    than calling them degenerate, and it took byte-exact reconstruction from 93.5% to 77.8% and
    forms with no offer from 84 to 182. A classification that is truer and costs 629 instructions
    is not ready, and the cost has not been explained yet, so it stays off rather than being
    argued for.
    """
    bases = _bases(insts, 8)
    if not bases:
        return None, "no instances"
    cand = candidates(op, length, key, sub)
    sp, ref, _drop = spread(op, length, key, sub, bases, cand)
    movers = [c for c, v in sp.items() if set(v.values()) != {0}]
    if movers:
        return None, "%d bits move this operand" % len(movers)
    want = ("expr." + sub) if sub else None
    vals = set()
    for raw, ops in insts:
        if key >= len(ops):
            continue
        v = dict(g17opmap._subvalues(*ops[key])).get(want or ops[key][0])
        if v is not None:
            vals.add(v)
    if len(vals) != 1:
        return None, "no bit moves it and yet it takes %d values" % len(vals)
    return {"verdict": "pinned", "value": vals.pop(), "measured": True,
            "source": "differential", "explains": 1.0}, None


def cubic(op, length, key, sub, insts):
    """The field as an exact THIRD-ORDER polynomial, or (None, why).

    Some fields are second order and some are not. The second difference says which pairs carry a
    term; the third says which TRIPLES do:

        I(x,y,z) = f(b^x^y^z) - f(b^x^y) - f(b^x^z) - f(b^y^z) + f(b^x) + f(b^y) + f(b^z) - f(b)

    The search is bounded by the interaction graph rather than by the bit count. A triple can only
    carry a term when all three of its pairs already interact - if x and y do not interact, no
    term containing both survives - so the candidates are the TRIANGLES of the pair graph, and on
    the fields that need this there are hundreds rather than the thousands a blind search would
    face.

    This is what the last recoverable inherited bits need: 114 of them move an operand whose map
    is `verified` and does not own them, which is a map that explains every instance and is still
    incomplete.
    """
    bases = _bases(insts, 3)
    if not bases:
        return None, "no instances"
    cand = candidates(op, length, key, sub)
    base = bases[0]
    sp, _ref, _drop = spread(op, length, key, sub, bases, cand)
    bits = sorted(c for c, v in sp.items() if set(v.values()) != {0})
    if not bits:
        return None, "no bit moves this operand"
    if len(bits) > MAXQUAD:
        return None, "%d bits move this operand; too many to expand" % len(bits)

    def val(d):
        if d is None or d[0] != op or key >= len(d[1]):
            return None
        return _value(d[1][key], sub)

    def probe(sets):
        blob = []
        for st in sets:
            m = bytearray(base)
            for b, i in st:
                m[b] ^= 1 << i
            blob.append(bytes(m))
        got = g17encode.decode(blob)
        return {frozenset(st): val(got.get(j)) for j, st in enumerate(sets)}

    singles = [(x,) for x in bits]
    pairs = list(itertools.combinations(bits, 2))
    f = probe([()] + singles + pairs)
    f0 = f[frozenset()]
    if f0 is None:
        return None, "the operand does not read back from its own bytes"
    lin = {x: f[frozenset((x,))] - f0 for x in bits if f.get(frozenset((x,))) is not None}
    bits = [x for x in bits if x in lin]
    inter = {}
    for x, y in itertools.combinations(bits, 2):
        c = f.get(frozenset((x, y)))
        if c is not None:
            v = c - lin[x] - lin[y] - f0
            if v:
                inter[(x, y)] = v
    if not inter:
        return None, "no pair interacts, so there is nothing a third order can add"
    adj = collections.defaultdict(set)
    for x, y in inter:
        adj[x].add(y)
        adj[y].add(x)
    tri = [t for t in itertools.combinations(bits, 3)
           if t[1] in adj[t[0]] and t[2] in adj[t[0]] and t[2] in adj[t[1]]]
    if not tri:
        return None, "the interaction graph has no triangle"
    if len(tri) > 4000:
        return None, "%d triangles; too many to probe" % len(tri)
    g = probe(tri)
    cube = {}
    for t in tri:
        c = g.get(frozenset(t))
        if c is None:
            continue
        x, y, z = t
        v = (c - f[frozenset((x, y))] - f[frozenset((x, z))] - f[frozenset((y, z))]
             + lin[x] + lin[y] + lin[z] + 2 * f0)
        if v:
            cube[t] = v
    if not cube:
        return None, "every triangle measures zero, so the field is second order"
    # rewrite around zero, exactly as quadratic() does but one degree further
    A = {x: (base[x[0]] >> x[1]) & 1 for x in bits}
    S = {x: (1 if not A[x] else -1) for x in bits}
    const = f0
    linx, quadx, cubx = collections.Counter(), collections.Counter(), collections.Counter()
    for x, c in lin.items():
        const += c * A[x]
        linx[x] += c * S[x]
    for (x, y), c in inter.items():
        const += c * A[x] * A[y]
        linx[x] += c * S[x] * A[y]
        linx[y] += c * S[y] * A[x]
        quadx[(x, y)] += c * S[x] * S[y]
    for (x, y, z), c in cube.items():
        const += c * A[x] * A[y] * A[z]
        linx[x] += c * S[x] * A[y] * A[z]
        linx[y] += c * S[y] * A[x] * A[z]
        linx[z] += c * S[z] * A[x] * A[y]
        quadx[(x, y)] += c * S[x] * S[y] * A[z]
        quadx[(x, z)] += c * S[x] * S[z] * A[y]
        quadx[(y, z)] += c * S[y] * S[z] * A[x]
        cubx[(x, y, z)] += c * S[x] * S[y] * S[z]
    rec = {"verdict": "verified", "measured": True, "source": "differential",
           "poly": {"const": int(const),
                    "lin": [[x[0], x[1], int(c)] for x, c in sorted(linx.items()) if c],
                    "quad": [[x[0], x[1], y[0], y[1], int(c)]
                             for (x, y), c in sorted(quadx.items()) if c],
                    "cube": [[x[0], x[1], y[0], y[1], z[0], z[1], int(c)]
                             for (x, y, z), c in sorted(cubx.items()) if c]},
           "step": None, "base": None, "positions": [], "extra": [], "explains": 1.0}
    n = ok = 0
    for raw, ops in insts:
        if key >= len(ops):
            continue
        want = dict(g17opmap._subvalues(*ops[key])).get(("expr." + sub) if sub else ops[key][0])
        if want is None:
            continue
        n += 1
        if g17encode.poly_value(rec["poly"], bytes(raw)) == want:
            ok += 1
    if ok != n:
        return None, "third order, but reproduces %d of %d instances" % (ok, n)
    return rec, None


def code_table(op, length, key, sub, insts):
    """The field's whole value table, ENUMERATED through the decoder rather than fitted.

    A linear map that explains the corpus can still be the wrong map. op12674's operand 9 takes
    exactly two values across 3,453 instances - 0 and 15 - so `value = 15 x bits` reproduces every
    one of them, and it is not what the field does: the four bits are a CODE whose all-zeros
    pattern means 15 and whose single-bit patterns mean 1, 2, 4 and 8. Encoding 15 as a set bit
    writes an instruction Apple never wrote, 137 times.

    The differential says when to suspect this - the constant it measures differs between base
    points - and the decoder answers it exactly: walk the field's bits through every combination
    from a real instruction and record what each pattern decodes to. Bounded, because the field
    is small, and checked against every instance afterwards like everything else.
    """
    cand = candidates(op, length, key, sub)
    bases = _bases(insts, 4)
    if not bases:
        return None, "no instances"
    sp, ref, _drop = spread(op, length, key, sub, bases, cand)
    bits = sorted(c for c, v in sp.items() if set(v.values()) != {0})
    if not bits:
        return None, "no bit moves this operand"
    if len(bits) > MAXTABLE:
        return None, "%d bits move this operand; too many to enumerate" % len(bits)
    base = bases[0]
    blob = []
    for pat in range(1 << len(bits)):
        m = bytearray(base)
        for j, (b, i) in enumerate(bits):
            m[b] = (m[b] & ~(1 << i)) | (((pat >> j) & 1) << i)
        blob.append(bytes(m))
    got = g17encode.decode(blob)
    table = {}
    for pat in range(1 << len(bits)):
        d = got.get(pat)
        if d is None or d[0] != op or key >= len(d[1]):
            continue
        v = _value(d[1][key], sub)
        if v is None:
            continue
        table["".join(str((pat >> j) & 1) for j in range(len(bits)))] = v
    if len(table) < 2 or len(set(table.values())) < 2:
        return None, "the enumeration gives one value"
    rec = {"verdict": "table", "order": [list(b) for b in bits], "table": table,
           "explains": 1.0, "measured": True}
    n = sum(1 for _raw, ops in insts if key < len(ops))
    ok = 0
    for raw, ops in insts:
        if key >= len(ops):
            continue
        want = dict(g17opmap._subvalues(*ops[key])).get(("expr." + sub) if sub else ops[key][0])
        code = "".join(str((raw[b] >> i) & 1) for b, i in bits)
        if want is not None and table.get(code) == want:
            ok += 1
    if ok != n:
        return None, "enumerated, but reproduces %d of %d instances" % (ok, n)
    return rec, None


def witness_table(op, length, key, sub, insts):
    """The field as a table over the bits that move it, built from the patterns Apple witnesses.

    code_table() enumerates every combination of the moving bits, which needs the field to be
    small - ten bits at most - and gives up when most combinations are illegal. Some fields are
    neither small nor linear: op10279's operand 1 at twelve bytes has seventeen moving bits, no
    interacting pairs, and no polynomial of any degree that reproduces it.

    But the seventeen bits DETERMINE it. Grouping every corpus instance by its pattern across
    them, no pattern carries two values - fifteen patterns, fifteen values, no ambiguity. So the
    field is a function of those bits and the corpus exhibits the whole of what Apple uses.

    The map is therefore exact on every witnessed pattern and REFUSES every other, which is the
    honest shape for a field recovered this way: it can reproduce anything Apple writes and will
    not invent an encoding for a value nobody has shown it. That is a smaller claim than the
    enumerated table makes and it is the claim the evidence supports.
    """
    bases = _bases(insts, 12)
    if not bases:
        return None, "no instances"
    cand = candidates(op, length, key, sub)
    sp, _ref, _drop = spread(op, length, key, sub, bases, cand)
    bits = sorted(c for c, v in sp.items() if set(v.values()) != {0})
    if not bits:
        return None, "no bit moves this operand"
    if len(bits) > 24:
        return None, "%d bits move this operand" % len(bits)
    want = ("expr." + sub) if sub else None
    table, n = {}, 0
    for raw, ops in insts:
        if key >= len(ops):
            continue
        v = dict(g17opmap._subvalues(*ops[key])).get(want or ops[key][0])
        if v is None:
            continue
        n += 1
        code = "".join(str((raw[b] >> i) & 1) for b, i in bits)
        if table.setdefault(code, v) != v:
            return None, "pattern %s carries more than one value" % code
    if len(table) < 2:
        return None, "the witnessed patterns give one value"
    return {"verdict": "table", "order": [list(b) for b in bits], "table": table,
            "explains": 1.0, "measured": True, "source": "witnessed patterns",
            "step": None, "base": None, "positions": [], "extra": []}, None


def _bases(insts, want=BASES, mode=None):
    seen, distinct = set(), []
    for raw, _ops in insts:
        b = bytes(raw)
        if b in seen:
            continue
        if mode and any(((b[p] >> q) & 1) != v for (p, q), v in mode):
            continue
        seen.add(b)
        distinct.append(b)
    if not distinct:
        return []
    return distinct[::max(1, len(distinct) // want)][:want]


def moded(op, length, key, sub, insts, depth=4):
    """A map PER MODE, when one map does not fit, or (None, why).

    Not a fallback and not a relaxation: each mode's coefficients are measured inside that mode,
    every instance has to fall into a recorded mode, and every instance has to be reproduced by
    the map of the mode it falls into. A field that needs a mode and does not get one is refused
    either way; the difference is that this one says what the mode is.

    The mode is found by iteration, and the base points for each round are drawn from the
    partition that FAILED. Twenty-four instructions sampled across the whole set do not always
    contain the disagreement; the ones that broke the last attempt always do.
    """
    cand = candidates(op, length, key, sub)
    # A SELECTOR IS THE RIGHT THING TO CONDITION A MODE ON, and it was the one thing excluded.
    # The mode search voted only over the VALUE candidates, which never contain a bit that
    # changes the shape of the instruction - so a field whose map depends on the shape could not
    # be moded at all. op998's operand 5 is exactly that: b5.1 decides whether operand 4 is a
    # register or an address expression, and operand 5 reads differently on each side. Offered as
    # a mode it partitions the field; offered as a value bit it would corrupt operand 4, so it is
    # only ever in `pool` and never in `inner`.
    pool = cand + [c for c in sorted(g17opmap.opcode_bits(op, length))
                   if c[0] < length and c not in cand]
    probe = _bases(insts, 24)
    if not probe:
        return None, "no instances"
    mb = mode_bits(op, length, key, sub, probe, cand, pool=pool)
    if mb is None:
        return None, "no set of bits accounts for the disagreement"
    why = "no mode was needed"
    for _ in range(depth):
        if not mb:
            return None, "no set of bits accounts for the disagreement"
        inner = [c for c in cand if c not in mb]
        parts = collections.defaultdict(list)
        for raw, ops in insts:
            b = bytes(raw)
            parts["".join(str((b[p] >> q) & 1) for p, q in mb)].append((raw, ops))
        out, worst = {}, None
        for k, part in sorted(parts.items()):
            got, w = measure(op, length, key, sub, _bases(part, BASES), inner)
            rec = as_record(*got) if got else None
            if rec is None:
                # A MODE CAN PIN THE FIELD. "no bit moves this operand" inside a partition is not
                # a failure to measure - it is the measurement, and it says the field takes one
                # value on this side of the mode. That is encodable and witnessed on every
                # instance of the partition; it is refused below if it is not.
                rec = _constant(part, key, sub)
            if rec is None:
                worst, why = part, "mode %s: %s" % (k, w or "map shape")
                break
            ok = explains(rec, part, key, sub)
            if ok != len(part):
                worst, why = [x for x in part
                              if explains(rec, [x], key, sub) == 0], \
                             "mode %s reproduces %d of %d" % (k, ok, len(part))
                break
            out[k] = rec
        if worst is None:
            big = max(out, key=lambda k: len(parts[k]))
            rec = dict(out[big])
            rec.update(explains=1.0, measured=True,
                       mode_bits=[list(x) for x in mb], modes=out)
            return rec, None
        more = mode_bits(op, length, key, sub, _bases(worst, 24),
                         [c for c in cand if c not in mb],
                         pool=[c for c in pool if c not in mb])
        if not more:
            return None, why
        mb = mb + more
    return None, why


def main():
    if "--degenerate" in sys.argv:
        lim = int(sys.argv[sys.argv.index("--degenerate") + 1])
        want = collections.defaultdict(list)
        for line in open(g17opmap.OUT):
            d = json.loads(line)
            if d["verdict"] in ("degenerate", "refuted", "conditional"):
                want[d["opcode"]].append(d)
        done = fixed = 0
        ops = sorted(want)
        ops = ops[::max(1, len(ops) // lim)][:lim]
        rows, _ = g17opmap.corpus_instances(ops)
        for op in ops:
            by = g17opmap.by_length(rows.get(op) or [])
            for d in want[op]:
                insts = by.get(d.get("length")) or []
                if not insts:
                    continue
                kind = d.get("kind") or ""
                sub = kind.split(".", 1)[1] if kind.startswith("expr.") else ""
                rec, why = field(op, d["length"], d["operand"], sub, insts)
                if rec is None:
                    rec, why2 = code_table(op, d["length"], d["operand"], sub, insts)
                    why = why if rec else "%s; %s" % (why, why2)
                done += 1
                if rec:
                    fixed += 1
                    how = ("a table of %d values" % len(rec["table"]) if rec.get("table")
                           else "second order, %d+%d terms" % (len(rec["poly"]["lin"]),
                                                               len(rec["poly"]["quad"]))
                           if rec.get("poly")
                           else "step=%s base=%s" % (rec.get("step"), rec.get("base")))
                    print("   op%-6d l%-3d op%-2d %-11s MEASURED %s"
                          % (op, d["length"], d["operand"], kind, how))
                elif "--why" in sys.argv:
                    print("   op%-6d l%-3d op%-2d %-11s %s"
                          % (op, d["length"], d["operand"], kind, why))
        print("%d of %d fields the corpus could not determine are measurable" % (fixed, done))
        return
    op, length, key = (int(x) for x in sys.argv[1:4])
    sub = sys.argv[4] if len(sys.argv) > 4 else ""
    rows, _ = g17opmap.corpus_instances([op])
    insts = g17opmap.by_length(rows.get(op) or []).get(length) or []
    rec, why = field(op, length, key, sub, insts)
    print(json.dumps(rec, indent=2) if rec else "not measurable: %s" % why)


if __name__ == "__main__":
    main()
