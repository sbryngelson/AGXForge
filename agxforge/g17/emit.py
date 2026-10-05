#!/usr/bin/env python3
"""THE OFFSET MAP, CONSTRUCTED RATHER THAN RECALLED.

Two mined tables stood between the builder and a shape nobody had compiled: g17-vtable-layout-pk
keyed 46 slot sets to offset maps for the per-kernel table, and the record maps were keyed on
(vector, slot set). Neither computes an answer for a key it does not hold, and together they were
the two largest refusal causes - 2,056 and 2,999 Apple sections.

Both collapse to ONE EMISSION ORDER PER TABLE KIND. FlatBuffers writes an object back to front, so
the field added FIRST lands at the highest offset; if the writer's schema fixes an order over the
slots, then the map for ANY subset is a walk:

    pos = declared length
    for each slot in emission order, if present:
        pos = align_down(pos - width, width)
        offset[slot] = pos

That is the same back-to-front rule that explained declared lengths in epoch 6, applied one level
down. It reproduces all 46 mined per-kernel keys exactly, offsets and declared length both, and it
generalises: on Apple slot sets the corpus NEVER compiled it is right 3,717 of 4,741 where a
remembered table is right zero times.

THE ORDER IS PARTIAL, AND THE GAPS REFUSE. Two slots the corpus never places relative to each
other have no constructed order between them, and guessing is how a construction that "generalises"
produces a wrong build. Such a pair is a named refusal:

    slots 17 and 18 never ordered by the corpus     870 Apple sections
    slots 38 and 41 never ordered by the corpus     737

Those are not defects in the method, they are two kernels nobody has compiled. Two witnesses would
decide 1,607 sections.

DERIVED FROM THE CORPUS, SCORED ON APPLE, NEVER FITTED TO IT. Pooling both populations gives 0
conflicting pairs, so a single writer order is consistent with both - the relation is a property of
Apple's schema, recovered once, not a per-shape memory.

    per-kernel   corpus-derived order on all Apple   6,907 decided   1,777 refused   0 WRONG
                 Apple half A -> half B              4,398 decided       0 refused   0 WRONG
                 Apple half B -> half A              4,282 decided       4 refused   0 WRONG
    records      corpus-derived, scored on Apple   105,142 decided     256 refused   0 WRONG
                 the mined table it replaces        17,061 (16.2%)   88,286 refused  51 WRONG

    python3 tools/g17emit.py --build     re-derive the relation from the corpus and store it
    python3 tools/g17emit.py             report what the stored relation decides and refuses
"""
import collections
import json
import os
import struct
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STORE = os.path.join(ROOT, "isa", "g17-emission-order.json")

# The semantic width of each record field, by vector. A field's width is what the walk subtracts,
# and it is the one thing here that is not derived from the order.
RECW = {4: {0: 1, 1: 4, 2: 4, 3: 1}, 2: {0: 1, 1: 4, 2: 4, 3: 4, 4: 4}, 26: {0: 4, 1: 1, 2: 4, 3: 4},
        10: {0: 4, 1: 4, 2: 4, 3: 4}}


class Refuse(KeyError):
    """The construction cannot decide this shape, and says which pair it cannot order."""


# ---------------------------------------------------------------------------------------------
# the relation
# ---------------------------------------------------------------------------------------------
def precedence(maps):
    """{a: {b, ...}} where a is at a HIGHER offset than b - a was emitted first - wherever both
    appear. Returns the relation and any pair the evidence orders both ways."""
    prec = collections.defaultdict(set)
    for want, _tlen in maps:
        for a in want:
            for b in want:
                if a != b and want[a] > want[b]:
                    prec[a].add(b)
    conflict = {tuple(sorted((a, b))) for a in list(prec) for b in prec[a] if a in prec.get(b, ())}
    return {a: sorted(b) for a, b in prec.items()}, sorted(conflict)


def widths(maps):
    """A field's width is the gap to the next field up, capped at four; taken as the mode."""
    wc = collections.defaultdict(collections.Counter)
    for want, tlen in maps:
        seq = sorted(want, key=lambda s: want[s])
        for i, s in enumerate(seq):
            nxt = want[seq[i + 1]] if i + 1 < len(seq) else tlen
            wc[s][min(nxt - want[s], 4)] += 1
    return {s: c.most_common(1)[0][0] for s, c in wc.items()}


def _closure(prec, slots):
    reach = {s: set(prec.get(s, ())) & slots for s in slots}
    for k in slots:
        for i in slots:
            if k in reach[i]:
                reach[i] |= reach[k] & slots
    return reach


def order_for(prec, slots):
    """The emission order restricted to `slots`, or Refuse naming the pair it cannot order."""
    slots = set(slots)
    reach = _closure(prec, slots)
    for a in sorted(slots):
        for b in sorted(slots):
            if a < b and b not in reach[a] and a not in reach[b]:
                raise Refuse("slots %d and %d are never ordered by the corpus" % (a, b))
    return sorted(slots, key=lambda s: -len(reach[s]))


def walk(order, slots, wid, tlen):
    pos, off = tlen, {}
    for s in order:
        if s not in slots:
            continue
        w = wid.get(s, 4)
        pos = (pos - w) - ((pos - w) % w)
        off[s] = pos
    return off


def natural_tlen(order, slots, wid):
    """The smallest declared length that leaves the last field at or above four.

    Four because the table's first word is the offset to its vtable. Not EXACTLY four: a byte-wide
    field can land at five with the alignment padding below it, and demanding four refused 99
    per-kernel tables that are perfectly well formed.
    """
    base = 4 + sum(wid.get(s, 4) for s in slots)
    for t in range(base, base + 32):
        o = walk(order, slots, wid, t)
        if o and min(o.values()) >= 4:
            return t
    raise Refuse("no declared length leaves room for the vtable offset")


# ---------------------------------------------------------------------------------------------
# the stored relation
# ---------------------------------------------------------------------------------------------
_S = None


def _store():
    global _S
    if _S is None:
        try:
            raw = json.load(open(STORE))
        except Exception:
            raise Refuse("no emission order stored; run g17emit.py --build")
        _S = {"pk_prec": {int(k): set(v) for k, v in raw["pk"]["prec"].items()},
              "pk_wid": {int(k): v for k, v in raw["pk"]["widths"].items()},
              "rec_prec": {k: {int(a): set(b) for a, b in v.items()}
                           for k, v in raw["rec"]["prec"].items()}}
    return _S


def pk_layout(slots):
    """(offset map, declared length) for the per-kernel table. Raises Refuse rather than guess."""
    s = _store()
    slots = set(int(x) for x in slots)
    missing = sorted(x for x in slots if x not in s["pk_wid"])
    if missing:
        raise Refuse("no width for per-kernel slot %s" % missing)
    order = order_for(s["pk_prec"], slots)
    tlen = natural_tlen(order, slots, s["pk_wid"])
    return walk(order, slots, s["pk_wid"], tlen), tlen


def record_layout(vector, kind, slots):
    """(offset map, declared length) for one record. The KIND is the table's schema - vector 2
    holds a union of record types, and one order for the vector describes none of them: keyed on
    the vector alone it got 35,289 of 36,953 wrong, and keyed on the kind it gets 0 wrong."""
    s = _store()
    key = "%d|%s" % (vector, kind)
    prec = s["rec_prec"].get(key)
    if prec is None:
        raise Refuse("the corpus holds no record of kind %s in vector %d" % (kind, vector))
    slots = set(int(x) for x in slots)
    wid = RECW.get(vector, {})
    order = order_for(prec, slots)
    tlen = natural_tlen(order, slots, wid)
    return walk(order, slots, wid, tlen), tlen


def natural_record_tlen(vector, kind, slots):
    """The declared length a record would have if it absorbed no slack from its successor."""
    st = _store()
    prec = st["rec_prec"].get("%d|%s" % (vector, kind))
    if prec is None:
        raise Refuse("the corpus holds no record of kind %s in vector %d" % (kind, vector))
    slots = set(int(x) for x in slots)
    return natural_tlen(order_for(prec, slots), slots, RECW.get(vector, {}))


def record_offsets_at(vector, kind, slots, tlen):
    """The offset map for a record whose DECLARED length is already known.

    A record's fields hang from the object's END - epoch 6's start == align_down(end - extent, 4) -
    so the slack an object absorbs moves its fields down inside it. The declared length is not known
    until the blocks are placed, which is why the builder constructs offsets twice: once from the
    natural extent to place the block, and once more from the final length. Recomputing moves no
    block, only where the fields sit inside one.
    """
    s = _store()
    prec = s["rec_prec"].get("%d|%s" % (vector, kind))
    if prec is None:
        raise Refuse("the corpus holds no record of kind %s in vector %d" % (kind, vector))
    slots = set(int(x) for x in slots)
    return walk(order_for(prec, slots), slots, RECW.get(vector, {}), tlen)


# ---------------------------------------------------------------------------------------------
# building the relation from the corpus
# ---------------------------------------------------------------------------------------------
def _pk_maps(pop):
    from . import gpumd as GM
    from . import mdgen as M
    out = []
    for _tag, md in pop:
        try:
            t = M.describe(md)["tables"][GM.kernel_table(md)]
            out.append(({int(a): b for a, b in t["slots"].items()}, t["tlen"]))
        except Exception:
            pass
    return out


def records(md, vectors=(26, 4, 2)):
    """(vector, kind, offset map, declared length) for every record in the named vectors.

    Vector 10 is not in the default set - it holds the threadgroup records, which the builder does
    not compose - but it is where the kind-43 and kind-93 records live, so slot 18's rule reads it."""
    from . import gpumd as GM
    pk = GM.kernel_table(md)
    if pk is None:
        return
    try:
        sl, _ = GM.table_at(md, pk)
    except Exception:
        return
    for v in vectors:
        if len(sl) <= v or not sl[v]:
            continue
        a = pk + sl[v]
        if a + 4 > len(md):
            continue
        va = a + struct.unpack_from("<I", md, a)[0]
        if va + 4 > len(md):
            continue
        n = struct.unpack_from("<I", md, va)[0]
        if n > 4096 or va + 4 + 4 * n > len(md):
            continue
        for k in range(n):
            p = va + 4 + 4 * k
            t = p + struct.unpack_from("<I", md, p)[0]
            if not (0 < t < len(md) - 1):
                continue
            try:
                s2, tsz = GM.table_at(md, t)
            except Exception:
                continue
            sm = {i: x for i, x in enumerate(s2) if x}
            if not sm:
                continue
            # THE KIND'S WIDTH IS THE GAP TO THE NEXT FIELD, NOT A TABLE ENTRY. Reading it as
            # four bytes because RECW says four disagreed with the validated reader on 631 Apple
            # records and 117 corpus ones, and made slot 18's rule look broken when it was the
            # reader. The gap is what the writer left, so it is what the field occupies.
            occ = sorted(sm.values())
            kind = None
            if 0 in sm:
                nxt = next((x for x in occ if x > sm[0]), tsz)
                w = max(1, min(nxt - sm[0], 4))
                o = t + sm[0]
                if o + w <= len(md):
                    kind = int.from_bytes(md[o:o + w], "little")
            yield v, kind, sm, tsz


def build(population=None):
    if population is None:
        raise ValueError(
            "build() derives its tables from the corpus population, which this package does not "
            "import: pass population=..., or run tools/g17emit.py, whose legacy entry supplies it "
            "from g17classregress. The regression harness is a corpus experiment, not part of "
            "authoring, and none of the seven functions the common author calls reaches this one.")
    pop = population
    pk = _pk_maps(pop)
    pk_prec, pk_conf = precedence(pk)
    rec = collections.defaultdict(list)
    for _tag, md in pop:
        for v, kind, sm, tsz in records(md) or []:
            rec[(v, kind)].append((sm, tsz))
    rp, rconf = {}, 0
    for (v, kind), maps in rec.items():
        p, c = precedence(maps)
        rconf += len(c)
        rp["%d|%s" % (v, kind)] = {str(a): b for a, b in p.items()}
    out = {"pk": {"prec": {str(a): b for a, b in pk_prec.items()},
                  "widths": {str(a): b for a, b in widths(pk).items()},
                  "tables": len(pk), "conflicts": pk_conf},
           "rec": {"prec": rp, "kinds": len(rec), "conflicts": rconf}}
    json.dump(out, open(STORE, "w"), indent=1, sort_keys=True)
    print("derived from %d corpus per-kernel tables and %d record kinds"
          % (len(pk), len(rec)))
    print("conflicting pairs: per-kernel %d, records %d  (a conflict means no single order exists)"
          % (len(pk_conf), rconf))
    print("stored in %s" % os.path.relpath(STORE, ROOT))


def main(argv=None, population=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if "--build" in argv:
        return build(population)
    s = _store()
    print("per-kernel: %d slots with a width, %d with outgoing order constraints"
          % (len(s["pk_wid"]), len(s["pk_prec"])))
    print("records: %d (vector, kind) schemas" % len(s["rec_prec"]))
    un = collections.Counter()
    for a in s["pk_wid"]:
        for b in s["pk_wid"]:
            if a < b:
                try:
                    order_for(s["pk_prec"], {a, b})
                except Refuse:
                    un[(a, b)] += 1
    print("per-kernel slot pairs the corpus never orders: %d" % len(un))
    for k in sorted(un)[:12]:
        print("   %d and %d" % k)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
