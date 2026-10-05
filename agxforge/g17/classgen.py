#!/usr/bin/env python3
"""Lay a metadata section out from block CONTENTS, so a class can be generated rather than copied.

g17mdgen.build_from() writes each table at the absolute position its description records. That is
enough to re-emit a blob that was measured and not enough to build one for content nobody has
compiled - its own comment says so. Transplanting a longer constant program into another kernel's
description therefore produces a section that is short by exactly the difference.

The rule that closes it, measured over 11,226 tables in 1,500 cached kernels and holding in every
one of them:

    a table's position is its vtable position plus its vtable length      pos == vtpos + vlen

and the blocks - each table's vtable, body and tail, and the vectors belonging to that region -
lie consecutively, aligned to two bytes. So a layout is a walk: take the blocks in order, give
each one the space its contents need, and let the offsets fall out.

    python3 tools/g17classgen.py            relayout every cached kernel and check the offsets
"""
import os, sys
import struct
import re as _re

from . import classbytes as CB
from . import mdgen as M


def body_len(t):
    """The bytes a table's own record occupies, before its tail.

    tlen, NOT max(tlen, highest slot + 4). A slot offset can exceed the declared table length -
    c4probe has a table with tlen 12 whose highest slot reaches 15 - and taking the larger makes
    the block overrun the next one by three bytes, which is the -12 gap that put 73 kernels twelve
    bytes over. The declared length is the length; a slot pointing past it points into what
    follows, the same way a table's body runs to what comes next rather than to a declared size.
    """
    # MUST MATCH build_from(), which writes a table's tail at pos + max(tlen, highest slot + 4).
    # Sizing blocks with tlen alone measured better - 98.6% against 97.7% - but it desynchronised
    # the layout from the writer, so a table whose slots reach past tlen had its tail written
    # further out than the layout reserved. A number that improves by breaking an invariant is
    # not an improvement, so the two are kept consistent and the c4probe overlap is handled by
    # trimming the tail instead.
    return t["tlen"]


def blocks(desc):
    """The section as an ordered list of (kind, key, size), in the order they appear."""
    items = []
    for pos, t in desc["tables"].items():
        if t.get("shared"):
            # A table that shares another's vtable owns no vtable bytes; its block is its body.
            items.append((pos, "table", pos, body_len(t) + len(t.get("tail") or b"")))
        else:
            items.append((t["vtpos"], "table", pos,
                          t["vlen"] + body_len(t) + len(t.get("tail") or b"")))
    for off, recs in desc["vectors"].items():
        items.append((off, "vector", off, 4 + 4 * len(recs)))
    items.sort()
    return items


# The constant program's symbol, as it appears in the tail that carries it.
SYMBOL = b"agc.main.constant_program"


def table_index(desc):
    """Every table, in layout order - the canonical enumeration the contract keys by.

    describe()'s `order` is the walk that found the tables, and it stops at what the root's slots
    reach. Layout order reaches all of them and is the same sequence before and after relayout,
    which is what an override key has to be.
    """
    return [key for _s, kind, key, _sz in blocks(desc) if kind == "table"]


def trim_tails(desc):
    """Cut a table's tail back to the space it actually owns in the original layout.

    describe() attributes a tail by reading forward, and in nineteen of fifteen hundred kernels it
    reads past the next block: the table at 406 in a6-tgcalc has vlen 6, body 11 and a nine-byte
    tail, which would end at 432 while the next vtable is at 420. A tail length is DERIVED - it
    runs until the next block - and treating it as an independent quantity is the same error this
    format has produced twice before with declared lengths.
    """
    items = blocks(desc)
    starts = [s for s, _k, _key, _sz in items]
    out = dict(desc)
    out["tables"] = {}
    for i, (start, kind, key, _sz) in enumerate(items):
        if kind != "table":
            continue
        t = dict(desc["tables"][key])
        room = (starts[i + 1] if i + 1 < len(starts) else desc["size"]) - start
        # A table that shares another's vtable has no vtable bytes of its own in its block, so
        # subtracting vlen here cut ten bytes off its tail and the class donor carried the loss.
        keep = room - body_len(t) - (0 if t.get("shared") else t["vlen"])
        tail = t.get("tail") or b""
        if keep < len(tail):
            t["tail"] = tail[:max(0, keep)]
        out["tables"][key] = t
    for pos, t in desc["tables"].items():
        out["tables"].setdefault(pos, t)
    return out


def gaps_of(desc):
    """The space between one block's end and the next block's start, in the original layout.

    A gap is part of the class's layout, not noise: a6-tgcalc has eight zero bytes between two
    vectors, and rebuilding it from its OWN description without them comes out eight bytes short.
    Eighteen kernels fail as self-donors for exactly this reason, all short by a multiple of four.
    """
    items = blocks(desc)
    out = []
    for i, (start, _k, _key, size) in enumerate(items):
        nxt = items[i + 1][0] if i + 1 < len(items) else desc["size"]
        out.append(max(0, nxt - (start + size)))
    return out


def relayout(desc, align=2, trailing=None, base=None, trim=True, gaps=None):
    """Recompute every offset from the block contents, keeping their order.

    Returns a new description. The first block keeps its start - the root pointer at offset 0 and
    whatever precedes the first vtable are not part of this walk - and everything after it is
    packed consecutively with `align` padding.
    """
    # Trimming uses the description's OWN absolute offsets to decide what a tail owns, so it must
    # not run on a description whose contents came from one kernel and whose offsets came from
    # another - it would trim the target's tails against the donor's spacing. The generate path
    # trims the target first and passes trim=False.
    if trim:
        desc = trim_tails(desc)
    items = blocks(desc)
    if not items:
        return desc
    out = {"size": desc["size"], "root": desc["root"], "order": [],
           "tables": {}, "vectors": {}, "extra": dict(desc.get("extra") or {})}
    cur = items[0][0]
    moved = {}
    gaps = gaps if gaps is not None else gaps_of(desc)
    for gi, (_start, kind, key, size) in enumerate(items):
        if cur % align:
            cur += align - (cur % align)
        if kind == "table":
            t = dict(desc["tables"][key])
            # THE TABLE'S POSITION IS FOUR-ALIGNED, not its vtable's. pos == vtpos + vlen, so the
            # vtable starts wherever it must for the body to land on a four-boundary. That is the
            # whole of the mysterious two-byte gap: it appears only before a table, and only when
            # the body would otherwise land two off.
            if t.get("shared"):
                # No vtable of its own: the body starts here, and its vtpos is fixed up below to
                # wherever the table that owns that vtable moved it.
                if cur % 4:
                    cur += 4 - (cur % 4)
                newpos = cur
            else:
                want = cur + t["vlen"]
                if want % 4:
                    cur += 4 - (want % 4)
                t["vtpos"] = cur
                newpos = cur + t["vlen"]          # the law
            moved[key] = newpos
            out["tables"][newpos] = t
        else:
            moved[key] = cur
            out["vectors"][cur] = list(desc["vectors"][key])
        cur += size
        if gi < len(gaps):
            cur += gaps[gi]
    # A shared vtable moved with the table that owns it; the sharers have to follow.
    vtmoved = {}
    for newpos, t in out["tables"].items():
        if not t.get("shared"):
            vtmoved[desc["tables"][{v: k for k, v in moved.items()}[newpos]]["vtpos"]] = t["vtpos"]
    for newpos, t in out["tables"].items():
        if t.get("shared"):
            t["vtpos"] = vtmoved.get(t["vtpos"], t["vtpos"])
    # vector records point at tables; move the targets with them
    for off, recs in list(out["vectors"].items()):
        out["vectors"][off] = [moved.get(r, r) for r in recs]
    out["root"] = moved.get(desc["root"], desc["root"])
    out["order"] = [moved.get(p, p) for p in desc["order"]]
    # WHERE EVERY BLOCK WENT. A reference field's value is the distance from the field to what it
    # points at, so recomputing one after a relayout needs the map from old position to new. It
    # was not exposed, which is why those fields have been carried from the donor and are correct
    # only while the layout does not move.
    out["_moved"] = dict(moved)
    # The section's length is the original's, moved by however much the blocks grew. Taking the
    # walk's end as the size overshoots by whatever trailed the last block in the original.
    old_end = max(start + size for start, _k, _key, size in items)
    # The bytes after the last block. For a kernel laid out from its OWN description that is
    # whatever its own section trailed by; when a DONOR of a different structure supplies the
    # template it is the target's trailing region, not the donor's, and the caller passes it.
    # SIZE. For a kernel laid out from its own description the section is its own size shifted by
    # however much the blocks moved - `size + (cur - old_end)` - and the two ends cancel. When a
    # DONOR of another kernel supplies the template that cancellation is against the wrong
    # section, so the caller passes the target's (size, block end) and the arithmetic is the same.
    base_size, base_end = base if base else (desc["size"], old_end)
    out["size"] = base_size + (cur - base_end)
    if trailing is not None:
        out["size"] = max(out["size"], cur + trailing)
    return out


def check(limit=1500):
    """Relayout each cached kernel's own description and compare with where Apple put things."""
    same = shifted = err = 0
    detail = []
    n = 0
    for d in sorted(os.listdir(CB.g17metal.CACHE)):
        md = CB.metadata(d)
        if not md:
            continue
        n += 1
        if n > limit:
            break
        try:
            desc = M.describe(md)
            re_ = relayout(desc)
        except Exception as exc:
            err += 1
            continue
        if sorted(re_["tables"]) == sorted(desc["tables"]) and \
           sorted(re_["vectors"]) == sorted(desc["vectors"]):
            same += 1
        else:
            shifted += 1
            if len(detail) < 4:
                a, b = sorted(desc["tables"]), sorted(re_["tables"])
                first = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), None)
                detail.append((d, a[first] if first is not None else None,
                               b[first] if first is not None else None))
    print("RELAYOUT from block contents, compared with Apple's own offsets")
    print("  identical offsets : %d" % same)
    print("  differ            : %d" % shifted)
    print("  describe failed   : %d" % err)
    for d, a, b in detail:
        print("     %-22s first difference: Apple %s, computed %s" % (d, a, b))
    return shifted == 0


def _main():
    if "--matrix" in sys.argv:
        o, b, r, fam = matrix()
        tot = o + b + r
        print("BUILT FROM CONTRACT INPUTS over the whole cache: %d kernels" % tot)
        print("  byte-identical to Apple's own metadata : %d  (%.1f%%)" % (o, 100.0 * o / tot))
        print("  differ                                 : %d" % b)
        print("  refused (no donor for the group)       : %d" % r)
        print("\n  by family:")
        for f in sorted(fam):
            a, n = fam[f]
            print("     %-20s %5d of %-5d %5.1f%%" % (f, a, n, 100.0 * a / n if n else 0))
        return 0
    if "--linker" in sys.argv:
        return 0 if linker_report() else 1
    return 0 if check() else 1


def verify_built(out):
    """Check a built section against the laws before returning it. Returns None if it holds.

    THE GUARD HELD ON EVERY SHAPE THE MODEL WAS GIVEN AND NOT OTHERWISE. All four wrong builds in
    the unfiltered linker path are syn- search residue - kernels outside the corpus - and each came
    out walkable but wrong, which is the worst way for a gap to show. These laws are measured at
    100% on both populations, so a built section that breaks one is not a section Apple's compiler
    would have written and the builder should refuse rather than emit it.

        the pointer block is dense in rank order over the recorded bindings
        a record slot is present only when its value is not the default
        kind 6's destination plus its length equals per-kernel slot 1
        texture state implies at least one internal binding

    THE LAST ONE WAS ADDED AFTER A SECTION THIS GUARD PASSED HUNG THE GPU. An authored texture
    kernel was given a section with slot 38 = 8 and two user bindings and no internal ones. It
    loaded, the archive accepted it, the pipeline was created, and the dispatch hung the device -
    kIOGPUCommandBufferCallbackErrorHang, and the machine took a GPU restart. Every texture section
    in both populations has an internal binding: 6,495 of 6,495 Apple and 399 of 399 corpus, no
    exceptions. A texture fetch reaches its descriptor through one, so a section that declares
    texture state and names no internal binding points the hardware at something that is not there.

    That is the limit of a structural guard, and it belongs in the docstring rather than only in
    the notes: this section walked, every reader returned the intended value, and the laws above
    all held. Passing them says the document is well formed, not that the GPU can run it.
    """
    import struct
    from . import facts as _F
    from . import gpumd as _GM
    try:
        pk = _GM.kernel_table(out)
        if pk is None:
            return "no per-kernel table"
        sl, _t = _GM.table_at(out, pk)
        binds = _F.binding_records(out)
        f = _GM.fields(out)
    except Exception as exc:
        return "unreadable: %s" % (str(exc)[:40],)
    # "CANNOT EVALUATE" IS NOT "FINE". binding_records returns None both for a section with no
    # binding vector and for one whose vector cannot be read, and accepting the second is the peer's
    # overlay-loader defect in different clothes - theirs asked "does this record carry a linear fit"
    # when the question was "is this record usable", and silently dropped 343 table records.
    #
    # Found by sabotage, not by reading: pointing the binding vector out of bounds produced a
    # section this guard ACCEPTED, while the other two sabotages were caught. A branch that never
    # fires on real data is not the same as a branch that would catch the thing it is for.
    if binds is None:
        if len(sl) > 4 and sl[4]:
            return "the section has a binding vector that cannot be read"
        return None
    # the record vector, read the way the laws are stated
    recs = []
    if len(sl) > 2 and sl[2]:
        a = pk + sl[2]
        if a + 4 <= len(out):
            v = a + struct.unpack_from("<I", out, a)[0]
            if v + 4 <= len(out):
                n = struct.unpack_from("<I", out, v)[0]
                if n <= 4096 and v + 4 + 4 * n <= len(out):
                    for k in range(n):
                        p2 = v + 4 + 4 * k
                        t2 = p2 + struct.unpack_from("<I", out, p2)[0]
                        # A RECORD THAT CANNOT BE READ IS A DEFECT, NOT A RECORD TO SKIP. This used
                        # to `continue` on both branches, so a record pointing out of bounds or with
                        # an unreadable vtable silently exempted itself from every law below. Same
                        # shape as the binding vector this guard once accepted because
                        # binding_records returned None, which was found by sabotage; the peer's
                        # name for it is a check that reads as passing when it never ran.
                        if not (0 < t2 < len(out) - 1):
                            return "a record in the argument vector points outside the section"
                        try:
                            s2, tsz = _GM.table_at(out, t2)
                        except Exception as exc:
                            return "a record in the argument vector cannot be read: %s" % (
                                str(exc)[:40],)
                        sm = {i: x for i, x in enumerate(s2) if x}
                        occ = sorted(set(sm.values()))
                        fl = {}
                        for i2, off in sm.items():
                            nx = min([o for o in occ if o > off], default=tsz)
                            w = nx - off
                            if t2 + off + min(w, 4) > len(out):
                                continue
                            fl[i2] = (out[t2 + off] if w == 1 else
                                      struct.unpack_from("<H", out, t2 + off)[0] if w == 2 else
                                      struct.unpack_from("<I", out, t2 + off)[0])
                        recs.append(fl)
    # THE ELISION CHECK APPLIES TO BOTH VECTORS. Checking only the record vector let two wrong
    # builds through whose BINDING record was {0: 5, 1: 2, 3: 0} - slot 3 present and holding the
    # default, which the law forbids on 457,367 present slots across both populations. A guard that
    # tests a law on half the places it holds is a guard with a hole in it.
    for r in list(recs) + list(binds or []):
        for i2, val in r.items():
            if val == 0:
                return "a record slot holds the default (slot %d)" % i2
    k6 = [r for r in recs if r.get(0) == 6 and 3 in r]
    if k6 and f.get(1) is not None and k6[0][3] + k6[0].get(2, 0) != f[1]:
        return "kind 6 does not end at slot 1 (%d + %d != %d)" % (
            k6[0][3], k6[0].get(2, 0), f[1])
    if binds:
        offs = sorted(b.get(2, 0) for b in binds)
        if offs != sorted(set(offs)):
            return "two bindings share a pointer-block slot"
    # EVERY LAW THAT GOVERNS THIS OPERATION, not the three that were easy to reach. The peer's
    # re-baseline was stopped by a case that had measured operand-bit overlap and reported it AFTER
    # the swap instead of conditioning the swap on it; a law you have measured and do not enforce is
    # the same defect as a guard aimed at half its domain, one level up. These are all at 100% on
    # both populations, so a correct build satisfies them and only a broken one is refused.
    k3 = [r for r in recs if r.get(0) == 3]
    k5 = sorted([r for r in recs if r.get(0) == 5], key=lambda r: r.get(3, 0))
    if binds and k3:
        rec_i = [b.get(1, 0) for b in binds]
        users = [i for i in rec_i if i <= 30]
        promo = {r.get(1, 0) for r in k5}
        # THE RANK LAW'S OWN TWO CONDITIONALS, which the first version of this guard dropped: the
        # acceleration-structure family orders its internals differently, and a promoted buffer
        # below the lowest recorded user may or may not take a slot. Outside those the law is
        # 100% on both populations; inside them it is not a law and must not be a guard.
        if (not any(i > 0xFFFF for i in rec_i)
                and not (users and any(p <= 30 and p < min(users) for p in promo))):
            pu = {p for p in promo if p <= 30}
            if users:
                pu = {p for p in pu if p >= min(users)}
            order = sorted({i for i in rec_i if i > 30}) + sorted(set(users) | pu)
            rank = {i: k for k, i in enumerate(order)}
            for b in binds:
                i = b.get(1, 0)
                if i in rank and b.get(2, 0) != 2 * rank[i]:
                    return "binding %d is not at 2*rank" % i
            if rank:
                want = 2 * (max(rank[i] for i in rec_i if i in rank) + 1)
                if k3[0].get(2, 0) != want:
                    return "kind 3 is not the pointer block size (%d vs %d)" % (
                        k3[0].get(2, 0), want)
        # a promoted range never starts inside the pointer block
        if k5 and k3 and k5[0].get(3, 0) < k3[0].get(2, 0):
            return "a promoted range starts inside the pointer block"
    for a2, b2 in zip(k5, k5[1:]):
        L = a2.get(2, 0)
        if b2.get(3, 0) - a2.get(3, 0) != max(2, L + (L % 2)):
            return "promoted ranges do not advance by the rounded length"
    for r in k5:
        if r.get(4, 0) % 2:
            return "a promoted range has an odd source offset"
    # ASCENDING ORDER WAS MEASURED ON USER-INDEX SECTIONS ONLY - 100% there, and false on 5,069 of
    # Apple's own sections once internal indices are present. Enforcing it everywhere refused 368
    # builds that were byte-identical to Apple's, which is the same mistake as applying the rank law
    # outside its domain: a law is only a guard inside the population it was measured on.
    if binds and not any(b.get(1, 0) > 30 for b in binds):
        bo = [b.get(2, 0) for b in binds]
        if bo != sorted(bo):
            return "the binding vector is not in ascending offset order"
    # TEXTURE STATE IMPLIES AN INTERNAL BINDING - 6,495 of 6,495 Apple and 399 of 399 corpus.
    # Uses the f and binds this function already read. The first version of this check called
    # GM.fields and g17facts.binding_records, which are imported here as _GM and _F, so it raised
    # NameError into a bare except and passed the very section it was written for - a branch that
    # never fires, in the function whose docstring warns about branches that never fire.
    if f.get(38) and not any(30 < b.get(1, 0) <= 0xFFFF for b in binds):
        return ("the section declares texture state and names no internal binding; "
                "a fetch has no descriptor to reach")

    return None


def linker_report():
    """Score the linker path across every dimension the matrix is supposed to cover.

    One number over the whole cache hides which KINDS of kernel the gate can build. This breaks it
    out by resource count, binding pattern, binding kinds, atomics, control flow, threadgroup
    memory and tensor use, so a dimension with no coverage is visible instead of averaged away.
    """
    import collections
    from . import obj as g17obj
    from . import gpumd as GM
    rows = []
    for d in sorted(os.listdir(CB.g17metal.CACHE)):
        sig = CB.source_signature(d)
        md = CB.metadata(d) if sig else None
        obj = os.path.join(CB.g17metal.CACHE, d, "out", "object", "0-0")
        if not md or not os.path.exists(obj):
            continue
        md = bytes(md)
        try:
            desc = M.describe(md)
            raw = open(obj, "rb").read()
            sects, syms = g17obj.sections_of(raw)
            off, size = sects["__TEXT,__text"]
            ni, loop = code_facts(bytes(raw[off:off + size]), syms["_agc.main"])
        except Exception:
            continue
        verdict = "refused"
        try:
            out, _e, _key = build_for(d, md, desc, sig, ni, loop)
            if out is not None:
                bad = verify_built(out)
                if bad is not None:
                    verdict = "refused"
                    out = None
                else:
                    verdict = "exact" if out == md else "WRONG"
        except (Ambiguous, Unsupported):
            pass
        except Exception:
            pass
        src = os.path.join(CB.g17metal.CACHE, d, "s.metal")
        text = open(src, errors="replace").read() if os.path.exists(src) else ""
        rows.append(dict(tag=d, verdict=verdict, bound=sig[0], declared=sig[3],
                         corpus=in_corpus(d),
                         origin=("manufactured" if d.startswith("sib-") else "original"),
                         at_zero=sig[1], const=sig[2], loop=loop, insts=ni,
                         atomics="atomic_" in text, tgmem=bool(GM.fields(md).get(28)),
                         tensor="simdgroup_" in text or "tensor" in text,
                         texture="texture" in text, kinds=tuple(sorted(set(sig[4] or ())))[:1]))

    # WHERE THE EVIDENCE CAME FROM, because the instrument can move its own score in both
    # directions. A manufactured witness that lands on its parent's key CONFIRMS a claim; one that
    # lands on a key of its own creates a new untested claim and is then refused, which drags the
    # headline down without anything about the model having changed. Reported separately so that
    # neither reading can hide behind the other: refusals over the corpus this project did not
    # make are the number that measures the model.
    def table(name, keyf):
        t = collections.defaultdict(collections.Counter)
        for r in rows:
            t[keyf(r)][r["verdict"]] += 1
        print("\n  by %s:" % name)
        for k in sorted(t, key=str):
            c = t[k]
            n = sum(c.values())
            print("     %-16s %5d kernels  exact %5d (%5.1f%%)  refused %5d  wrong %d"
                  % (str(k)[:16], n, c["exact"], 100.0 * c["exact"] / n, c["refused"],
                     c["WRONG"]))

    # DENOMINATE IT TWICE, because one number here was answering two questions and getting both
    # wrong. This walks every cache directory with a signature and an object - which includes
    # probe- kernels and syn- SEARCH RESIDUE, neither of which is corpus: a synthesized kernel is
    # corpus only if it is a recorded witness. So the headline fell from 98.5% over 10,153 kernels
    # to 69.6% over 18,693 without the model changing at all; the corpus did not get harder, the
    # denominator filled up with kernels nobody ever told the model about.
    #
    #     unfiltered   18,693 kernels   13,010 exact (69.6%)   5,679 refused   4 WRONG
    #     corpus       11,905 kernels   11,859 exact (99.6%)      46 refused   0 WRONG
    #
    # Both are worth printing and they answer different questions. The corpus line is the model's
    # score and is where "wrong stays zero" is denominated. The unfiltered line is where the
    # builder BUILDS instead of refusing on a shape it was never given - all four wrong builds are
    # non-corpus, and a guard that only holds on familiar shapes is a guard with a measured limit.
    n = len(rows)
    ex = sum(1 for r in rows if r["verdict"] == "exact")
    wr = sum(1 for r in rows if r["verdict"] == "WRONG")
    cn = sum(1 for r in rows if r["corpus"])
    cex = sum(1 for r in rows if r["corpus"] and r["verdict"] == "exact")
    cwr = sum(1 for r in rows if r["corpus"] and r["verdict"] == "WRONG")
    print("THE LINKER PATH over the CORPUS: %d kernels, %d byte-exact (%.1f%%), %d refused, %d wrong"
          % (cn, cex, 100.0 * cex / max(1, cn), cn - cex - cwr, cwr))
    print("  and over every cached kernel including probes and search residue: %d kernels, "
          "%d byte-exact (%.1f%%), %d refused, %d wrong"
          % (n, ex, 100.0 * ex / n, n - ex - wr, wr))
    if wr > cwr:
        print("  the %d wrong build(s) are ALL outside the corpus: %s"
              % (wr - cwr, ", ".join(r["tag"] for r in rows
                                     if r["verdict"] == "WRONG" and not r["corpus"])[:200]))
    table("where the evidence came from", lambda r: r["origin"])
    table("bound resources", lambda r: r["bound"])
    table("declared resources", lambda r: r["declared"])
    table("first binding at zero", lambda r: r["at_zero"])
    table("a constant binding", lambda r: r["const"])
    table("control flow", lambda r: "loops" if r["loop"] else "straight-line")
    table("size class", lambda r: size_class(r["insts"], r["loop"]))
    table("atomics", lambda r: r["atomics"])
    table("threadgroup memory", lambda r: r["tgmem"])
    table("textures", lambda r: r["texture"])
    table("simdgroup/tensor", lambda r: r["tensor"])
    return wr == 0


# ---------------------------------------------------------------------------------------------
# THE CLASS TABLE, KEYED BY (SIGNATURE, STRUCTURE).
#
# The recorded classes are keyed by signature alone, and that is why c16-originA32 could not be
# built: it binds three buffers and needs eight descriptor tables, and every recorded eight-table
# class is a two-binding one. Transplanting its content into one of those produces 458 bytes where
# Apple emits 468, because the binding vector is a different length. Transplanting into a donor
# that matches on BOTH the signature and the structure produces 468 bytes, byte-identical.
#
# So the key is the pair. One representative description per (signature key, structure) is enough,
# and the descriptions are structural templates - the program's own content is transplanted over
# them and the layout recomputed, so no byte of the donor's program survives into the image.

def shape_of(desc):
    """The structural key: table count, vector count, and the ORDER the blocks appear in.

    Counts alone are not enough. Two kernels with the same table and vector counts can interleave
    their vectors differently between the tables, and the layout walk then pads differently - which
    is why 231 of 234 generation failures used a donor that was not the kernel itself and every
    length delta was a multiple of four.
    """
    # THE VLEN/TLEN SEQUENCE IS THE CLASS. Table and vector counts are too coarse: bar_both and
    # b45p_prev_load share (2 bound, not at zero, no constant, 3 declared) and (8 tables, 3
    # vectors) and are different classes - their first four tables differ in vtable length, table
    # length and tail length. Keyed only by counts, a barrier kernel gets a buffer-pattern
    # donor and the section comes out sixteen bytes wrong.
    # TAIL LENGTHS BELONG IN THE KEY TOO. cf-deep1 and at-store-uint share their whole vlen/tlen
    # sequence and are still different classes - the sections differ in 87 bytes from offset 188
    # and in total length by four - because their tails are different lengths. A tail length is
    # contract-derived (the symbol name and the constant program are both Kernel fields), so
    # keying on it asks the backend for nothing it does not have.
    # THE SLOT MAP IS STRUCTURE TOO. It lives in the vtable, from_contract does not carry it, and
    # cf-loop_if took a donor from cf-loop2 whose slots differ - two bytes at 128 and 130, inside
    # a vtable. A slot map is a property of the class's shape, so it belongs in the key rather
    # than in the contract.
    # AND `order` IS NOT EVERY TABLE. The root's slots reach four of mx-w.2d-1's eight tables;
    # the other four are found by the scan and were outside the key, outside contract_inputs and
    # outside from_contract's overrides - so their fields were class constants copied from a
    # donor, whatever the kernel being built. Keying and keying-in on table_index() covers all of
    # them and costs nothing: the enumeration is layout order, which both sides already agree on.
    # AND WHETHER THE TABLE OWNS ITS VTABLE. A table that shares another's occupies no vtable
    # bytes, so two sections with the same vlen/tlen/tail/slot sequence lay out differently if one
    # of them shares - which is what put 77 bytes of difference into mr-tex.buffer-2.
    return (tuple((desc["tables"][p]["vlen"], desc["tables"][p]["tlen"],
                   len(desc["tables"][p].get("tail") or b""),
                   bool(desc["tables"][p].get("shared")),
                   tuple(sorted((int(k), v) for k, v in
                                (desc["tables"][p].get("slots") or {}).items())))
                  for p in table_index(desc)),
            "".join(k[0] for _s, k, _key, _sz in blocks(desc)))


def class_table(limit=None):
    """{(signature key, structure): a representative description} built from the cache."""
    import collections
    out = {}
    seen = collections.Counter()
    n = 0
    for d in sorted(os.listdir(CB.g17metal.CACHE)):
        sig = CB.source_signature(d)
        md = CB.metadata(d) if sig else None
        if not md:
            continue
        n += 1
        if limit and n > limit:
            break
        try:
            desc = M.describe(md)
        except Exception:
            continue
        # THE WHOLE SIGNATURE, not its first four fields. The type set and the resource kinds
        # were being computed and then dropped: a cube texture took its donor from a 1d texture
        # because both keyed to (2 bound, at zero, no constant, 2 declared).
        key = (tuple(sig), shape_of(desc))
        seen[key] += 1
        # The donor is a TEMPLATE and must be trimmed: describe() reads a tail forward and can
        # read past the section, and relayout is called with trim=False on this path, so an
        # untrimmed donor carries its overrun into every image built from it - twelve bytes over,
        # 73 times.
        out.setdefault(key, (d, trim_tails(desc)))
    return out, seen


def generate(desc_target, table, sig=None):
    """Build a section for a program, from the donor that matches its signature AND structure.

    Both halves of the key matter and the structure alone is not enough: c16-originA32 needs an
    eight-table donor that also binds three buffers, and every eight-table donor that binds two
    produces 458 bytes where Apple emits 468, because the binding vector is a different length.
    """
    import copy
    struct = shape_of(desc_target)
    best = None
    if sig is not None:
        best = table.get((tuple(sig), struct))
    if best is None:
        for (key, st), cand in table.items():
            if st == struct:
                best = cand
                break
    if best is None:
        return None, None
    tag, donor = best
    desc_target = trim_tails(desc_target)
    gen = copy.deepcopy(donor)
    for i in range(min(len(gen["order"]), len(desc_target["order"]))):
        bo, do = gen["order"][i], desc_target["order"][i]
        for w in ("vlen", "tlen", "fields", "slots", "tail"):
            gen["tables"][bo][w] = copy.deepcopy(desc_target["tables"][do].get(w))
    gb, db = sorted(gen["vectors"]), sorted(desc_target["vectors"])
    if len(gb) == len(db):
        for a, b in zip(gb, db):
            gen["vectors"][a] = copy.deepcopy(desc_target["vectors"][b])
    gen["extra"] = copy.deepcopy(desc_target.get("extra") or {})
    # Measure the target's trailing region on its TRIMMED description. describe() over-attributes
    # the last table's tail, so the untrimmed blocks can end past the section itself and the
    # trailing comes out negative - it was -3 for a6-2d, which then shortened every generated
    # section by three bytes.
    # A VECTOR MUST PRECEDE THE TABLES IT POINTS AT. build_from stores each record as an unsigned
    # delta from its own slot, so a vector laid out after its target cannot be encoded at all -
    # it raises "'I' format requires 0 <= number", which is 37 of the generation failures. Detect
    # it here and refuse, because emitting a section with a wrapped-around record would produce an
    # image that loads and reads the wrong table.
    for off, recs in gen["vectors"].items():
        for k, r in enumerate(recs):
            if r < off + 4 + 4 * k:
                return None, "vector at %d points backward at %d" % (off, r)
    t_items = blocks(desc_target)
    t_end = max(s + sz for s, _k, _key, sz in t_items) if t_items else desc_target["size"]
    return bytes(M.build_from(
        relayout(gen, base=(desc_target["size"], t_end), trim=False,
                 gaps=gaps_of(desc_target)))), tag


# ---------------------------------------------------------------------------------------------
# THE PER-KERNEL TABLE, COMPUTED RATHER THAN TRANSPLANTED.
#
# A donor supplies a shape and the contract writes values into it, and that means 97.9% of a
# byte-exact section is copied. Splitting that showed 47.1% of it is determined by rules already
# pinned here and simply not computed by the builder - which is code that has not been written,
# not knowledge that is missing.
#
# This is the first piece of writing it. The per-kernel table's vtable is a pure function of its
# SLOT SET: 43 distinct sets across the corpus and each has exactly one offset map and one body
# length, 7,594 of 7,594. And three of the optional slots follow from the contract exactly -
# slot 28 iff the kernel allocates threadgroup memory, slot 32 iff it has more than 30
# instructions, slot 33 iff it branches backward - so the set is partly contract-derived and the
# vtable is then arithmetic on it.
#
# isa/g17-pk-vtable.json is that function, 43 rows keyed by slot set. It is a derived table, not
# an Apple artifact: every row is reproducible from the rule that a FlatBuffers vtable's length is
# 4 + 2*(highest slot + 1) and its offsets follow the writer's field order.
_PKV = None


def pk_vtable(slots):
    """(offset map, body length) for a per-kernel table with this slot set, or None."""
    global _PKV
    if _PKV is None:
        import json
        p = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         "isa", "g17-pk-vtable.json")
        _PKV = {tuple(int(x) for x in k.split(",")): v for k, v in json.load(open(p)).items()}
    e = _PKV.get(tuple(sorted(slots)))
    if e is None:
        # FALL BACK TO THE MINED TABLE. g17-pk-vtable.json was written by hand and holds 43 slot
        # sets; g17vtmine now derives 46 from the same corpus, and the three it adds were
        # unanswerable in the build path for no reason but that nothing had re-derived the file.
        # The mined table is a superset on every shared key - checked, not assumed.
        offs = pk_layout(slots)
        tlen = pk_layout_tlen(slots)
        return None if offs is None or tlen is None else (offs, tlen)
    return {int(k): v for k, v in e["offsets"].items()}, e["tlen"]


def pk_slot_set(donor_slots, threadgroup=None, insts=None, loop=None):
    """The per-kernel slot set the contract asks for, starting from the donor's.

    Only the slots with an exact rule are decided here; the rest stay as the class has them,
    because a slot set with a member guessed is a vtable with a member guessed. Slot 31 is the one
    that matters and has no rule - present in 2,799 kernels, no candidate predicate above 93.8% -
    so it is left alone rather than invented.
    """
    s = set(int(x) for x in donor_slots)
    if threadgroup is not None:
        s.add(28) if threadgroup else s.discard(28)
    if insts is not None:
        s.add(32) if insts > 30 else s.discard(32)
    if loop is not None:
        s.add(33) if loop else s.discard(33)
    return tuple(sorted(s))


def apply_pk_rules(desc, threadgroup=None, insts=None, loop=None):
    """Recompute the per-kernel table's vtable from its slot set, in place on a copy.

    Returns (description, "computed"/"unchanged") - and never a half-applied table: if the slot
    set the contract asks for has no recorded layout the description comes back untouched, which
    is a refusal to guess rather than a vtable assembled from parts.
    """
    import copy
    out = copy.deepcopy(desc)
    pk = None
    for pos in table_index(out):
        sl = {int(k) for k in (out["tables"][pos].get("slots") or {})}
        if {26, 27, 29} <= sl:
            pk = pos
            break
    if pk is None:
        return out, "unchanged"
    t = out["tables"][pk]
    want = pk_slot_set(t["slots"], threadgroup, insts, loop)
    lay = pk_vtable(want)
    if lay is None:
        return out, "unchanged"
    offs, tlen = lay
    old = {int(k): v for k, v in (t.get("fields") or {}).items()}
    t["slots"] = dict(offs)
    t["vlen"] = 4 + 2 * ((max(offs) + 1) if offs else 0)
    t["tlen"] = tlen
    t["fields"] = {k: (offs[k], w, v) for k, (o, w, v) in old.items() if k in offs}
    return out, "computed"


# EVERY VTABLE, not only the per-kernel one. The slot set alone determines the layout for 35.4% of
# tables; adding the field widths and the declared body length takes it to 90.59%, in 74 keys of
# which 71 are single-valued. The record tables are where the remaining transplanted bytes live,
# and their slot arrays are computable from three small facts about them.
_VTLX = None


_VETO = None


def layout_veto():
    """Ambiguous keys, AT EACH TABLE'S OWN GRANULARITY - {"base": set, "back": set, "x": set}.

    THE FIX FOR A BUG THAT COST THE HARD INVARIANT, and the second attempt at it. The three layout
    tables were hand-mined once and nothing re-derived them, so a key that acquired a second
    layout as the corpus grew was still answered - confidently, with whichever had been installed.
    Five kernels built wrong that way, two slot offsets swapped.

    A detector that reports it afterwards is not enough; the lookup has to stop answering. But the
    first version vetoed all THREE tables on the BASE key's ambiguity, which is exactly backwards:
    -back and -x exist because their finer keys resolve what the base one cannot. Vetoing them
    took the corpus from 12,958 byte-exact to 4,080 - trading five wrong builds for 8,878
    refusals, which is the wholesale mistake in a different costume.

    So each table is vetoed on ITS OWN key. A key that is ambiguous at (slots, widths, tlen) and
    unique once the distance from the end is added is answered by -back and refused by the base,
    which is what those tables were built to do.
    """
    global _VETO
    if _VETO is not None:
        return _VETO
    import collections
    from . import cache as g17cache
    from . import gpumd as GM
    got = g17cache.load("layout-veto")
    if got is not None:
        _VETO = got
        return _VETO
    base = collections.defaultdict(set)
    back = collections.defaultdict(set)
    xk = collections.defaultdict(set)
    val = collections.defaultdict(set)
    for d in sorted(os.listdir(CB.g17metal.CACHE)):
        if not in_corpus(d):
            continue
        md = CB.metadata(d)
        if not md:
            continue
        md = bytes(md)
        try:
            desc = M.describe(md)
            pk = GM.kernel_table(md)
            idx = table_index(desc)
            cp, _c, _t, _s = contract_inputs(desc)
        except Exception:
            continue
        cpl = len(bytes(cp or b""))
        for j2, pos in enumerate(idx):
            t = desc["tables"][pos]
            sl = {int(k): v for k, v in (t.get("slots") or {}).items()}
            f = {int(k): v for k, v in (t.get("fields") or {}).items()}
            if pos == pk or not sl or set(sl) != set(f):
                continue
            keys = tuple(sorted(sl))
            widths = tuple(f[k][1] for k in keys)
            offs = tuple(sorted(sl.items()))
            bk = len(idx) - 1 - j2
            base[(keys, widths, t["tlen"])].add(offs)
            back[(keys, widths, t["tlen"], bk)].add(offs)
            xk[(keys, widths, t["tlen"], bk, cpl)].add(offs)
            val[(keys, widths, t["tlen"], f[keys[0]][2])].add(offs)
    _VETO = {"base": {k for k, v in base.items() if len(v) > 1},
             "back": {k for k, v in back.items() if len(v) > 1},
             "x": {k for k, v in xk.items() if len(v) > 1},
             "val": {k for k, v in val.items() if len(v) > 1}}
    with g17cache._Lock("layout-veto"):
        g17cache.save("layout-veto", _VETO)
    return _VETO


_VTLPK = None


def pk_layout(slots, widths=None):
    """The PER-KERNEL table's offset map, from its slot set and field widths.

    THE ONE TABLE NOTHING MINED. g17vtmine skipped it and g17mdgen.compose carried a hand-written
    nine-slot map instead, which is why composing a section from the contract rather than from a
    probe refused: the composed per-kernel table had slots [1,2,3,4,6,8,10,13,26] where Apple's
    has fifteen, and no rule knew the shape.

    It is the most determined table in the document, not the least. Over 21,453 per-kernel tables -
    11,897 corpus and 9,556 Apple - the offset map is a function of the SLOT SET with no ambiguity
    anywhere: 175 keys, none with two maps. Mined from the corpus alone it holds 46 keys, and those
    reproduce 4,512 of Apple's 9,556 exactly with 0 wrong; the other 5,044 carry a slot set this
    corpus has never compiled and REFUSE.

    A single emission order explains most of it and is written down in the notes, but the four
    byte-wide fields at the top do not follow it in 7% of tables, so what ships is the measured
    table rather than the law that nearly holds.

    `widths` IS ACCEPTED AND IGNORED. The first version keyed on it, and the widths in a mined key
    are gaps to the next field, not semantic widths - a caller reasoning about field types builds
    a key that is not in the table and gets a silent refusal, indistinguishable from a slot set
    nobody has compiled. The peer session hit that on all five of its tables. The slot set alone
    determines the map in 175 of 175 keys over both populations, so the widths were never carrying
    anything.
    """
    global _VTLPK
    if _VTLPK is None:
        import json
        p = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         "isa", "g17-vtable-layout-pk.json")
        try:
            _VTLPK = json.load(open(p))
        except Exception:
            _VTLPK = {}
    e = _VTLPK.get(",".join(map(str, sorted(int(x) for x in slots))))
    return None if e is None else {int(i): o for i, o in e["offsets"].items()}


def pk_layout_tlen(slots):
    """The declared body length that goes with pk_layout's offset map, or None."""
    pk_layout(slots)
    e = _VTLPK.get(",".join(map(str, sorted(int(x) for x in slots))))
    return None if e is None else e["tlen"]


_VTLV = None


def vtable_layout_val(slots, widths, tlen, first_value):
    """The offset map keyed additionally on the value of the table's LOWEST-NUMBERED FIELD.

    THE KEY THAT BLOCKED THE NO-PROBE BUILD. Composing a section from the contract rather than
    from a probe refused on every one of the four Apple contracts that build byte-exactly, and on
    the same table each time: slots [0,2,3], widths [1,4,4], body length 16. It is vetoed because
    the corpus holds two offset maps for it, and neither the distance from the end nor the
    constant program's length separates them.

    The FIELD VALUE does, exactly:

        field 0 = 3    10 tables    {0: 11, 2: 12, 3: 4}
        field 0 = 6 8,281 tables    {0: 11, 2: 12, 3: 4}
        field 0 = 9    72 tables    {0: 15, 2:  8, 3: 4}     the mp-precise_* family

        0 of 8,363 tables still ambiguous once that value is in the key

    This is not a donor read. The builder is WRITING that field's value - it comes from the
    contract - so keying the layout on it asks the same question the other three tables ask, one
    component finer. Same standing as -back and -x, and vetoed on its own key like both.
    """
    if (tuple(slots), tuple(widths), tlen, first_value) in layout_veto()["val"]:
        return None                      # the corpus shows two layouts for this key; refuse
    global _VTLV
    if _VTLV is None:
        import json
        p = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         "isa", "g17-vtable-layout-val.json")
        try:
            _VTLV = json.load(open(p))
        except Exception:
            _VTLV = {}
    e = _VTLV.get("%s|%s|%d|%s" % (",".join(map(str, slots)), ",".join(map(str, widths)),
                                   tlen, first_value))
    return None if e is None else {int(i): o for i, o in e.items()}


def vtable_layout_x(slots, widths, tlen, back, env):
    """The offset map for the two keys that (slot set, widths, body length, distance from the end)
    leaves ambiguous - 4,732 tables between them, and BOTH are settled by the constant program's
    length, which is already a contract input."""
    if (tuple(slots), tuple(widths), tlen, back, env.get("CP")) in layout_veto()["x"]:
        return None                      # the corpus shows two layouts for this key; refuse
    global _VTLX
    if _VTLX is None:
        import json
        p = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         "isa", "g17-vtable-layout-x.json")
        _VTLX = json.load(open(p))
    base = "%s|%s|%d|%d" % (",".join(map(str, slots)), ",".join(map(str, widths)), tlen, back)
    for q in ("CP", "NI", "B", "D", "NB", "NT", "TG", "SC", "SZ", "NVEC", "LOOP"):
        e = _VTLX.get("%s|%s|%s" % (base, q, env.get(q)))
        if e is not None:
            return {int(i): o for i, o in e.items()}
    return None


_VTLB = None


def vtable_layout_back(slots, widths, tlen, back):
    """The offset map, with where the table sits from the END added to the key.

    (slot set, widths, body length) determines 90.58% of tables; adding the distance from the end
    takes it to 91.49% in 200 keys, and from the front only to 90.76%. Records are appended, so
    the end is the stable reference - the same reason the record field and tail rules are keyed
    that way.
    """
    if (tuple(slots), tuple(widths), tlen, back) in layout_veto()["back"]:
        return None                      # the corpus shows two layouts for this key; refuse
    global _VTLB
    if _VTLB is None:
        import json
        p = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         "isa", "g17-vtable-layout-back.json")
        _VTLB = json.load(open(p))
    e = _VTLB.get("%s|%s|%d|%d" % (",".join(map(str, slots)), ",".join(map(str, widths)),
                                   tlen, back))
    return None if e is None else {int(i): o for i, o in e.items()}


_VTL = None


def vtable_layout(slots, widths, tlen):
    """The offset map for a table with this slot set, these field widths and this body length."""
    if (tuple(slots), tuple(widths), tlen) in layout_veto()["base"]:
        return None                      # the corpus shows two layouts for this key; refuse
    global _VTL
    if _VTL is None:
        import json
        p = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         "isa", "g17-vtable-layout.json")
        _VTL = json.load(open(p))
    k = "%s|%s|%d" % (",".join(map(str, slots)), ",".join(map(str, widths)), tlen)
    e = _VTL.get(k)
    return None if e is None else {int(i): o for i, o in e.items()}


def apply_vtable_rules(desc, env=None):
    """Recompute every table's slot array from its slot set, widths and body length.

    A table whose key has no single recorded layout is left exactly as it was - three of the 74
    keys are like that - because a slot array assembled from a guess is a document the loader
    walks into the wrong place.
    """
    import copy
    out = copy.deepcopy(desc)
    done = 0
    undecided = []
    idx = table_index(out)
    for j, pos in enumerate(idx):
        t = out["tables"][pos]
        sl = {int(k): v for k, v in (t.get("slots") or {}).items()}
        f = {int(k): v for k, v in (t.get("fields") or {}).items()}
        if set(sl) != set(f) or not sl:
            continue
        offs = vtable_layout(sorted(sl), [f[k][1] for k in sorted(sl)], t["tlen"])
        if offs is None:
            offs = vtable_layout_back(sorted(sl), [f[k][1] for k in sorted(sl)], t["tlen"],
                                      len(idx) - 1 - j)
        if offs is None and env:
            offs = vtable_layout_x(sorted(sl), [f[k][1] for k in sorted(sl)], t["tlen"],
                                   len(idx) - 1 - j, env)
        if offs is None:
            offs = vtable_layout_val(sorted(sl), [f[k][1] for k in sorted(sl)], t["tlen"],
                                     f[sorted(sl)[0]][2])
        if offs is None or set(offs) != set(sl):
            # THE BUILDER MUST NOT INHERIT AN OFFSET MAP IT CANNOT DERIVE. This used to leave the
            # table exactly as the donor had it, which is byte-exact for the obvious reason and
            # silent about the fact that those bytes came from Apple rather than from a rule. The
            # ablation found it: destroy the donor's offsets and one kernel in 284 builds WRONG
            # instead of refusing, because nothing downstream knew the map had been inherited.
            #
            # 24 vtables corpus-wide cannot be derived, 0.030% of 80,444, and every one is the
            # same key - slots [0,2,3], widths [1,4,4], tlen 16, which is one of the ambiguous
            # shapes already recorded here. They cost 24 kernels, 0.19%, which now refuse.
            undecided.append(pos)
            continue
        t["slots"] = dict(offs)
        t["fields"] = {k: (offs[k], w, v) for k, (_o, w, v) in f.items()}
        t["vlen"] = 4 + 2 * ((max(offs) + 1) if offs else 0)
        done += 1
    # AND EVERY VTABLE LENGTH, whether or not its offset map was reproduced. The length is a
    # function of the slot set alone - 84,688 tables, 0 exceptions - and it was being set only
    # where the layout lookup happened to succeed, so everywhere else the donor's value survived.
    #
    # That distinction was invisible until the honest arm was run. Overwriting vlen with the rule
    # scores 493 of 493, which proves the rule and nothing about the builder: the value written is
    # the value that was there. DESTROYING vlen scores 0 of 493, and that is the measurement -
    # the builder was not computing it. Same trap as reading a body length back out of a table
    # built from the corpus, one function along.
    for pos in idx:
        t = out["tables"][pos]
        sl = [int(k) for k in (t.get("slots") or {})]
        t["vlen"] = 4 + 2 * ((max(sl) + 1) if sl else 0)
    return out, done, undecided
def contract_written_tail(tail):
    """Is this tail the constant program's, and therefore contract-written end to end?

    THE BYTE OFFSET IS NOT A STABLE ADDRESS IN THIS TAIL. It holds the identity vector - N words
    of 0..N-1 - then the symbol's length, then the symbol, then padding, then the payload. N is a
    contract input and it VARIES, so the same byte offset means different things in two sections
    that happen to share a tail length. Keying a class constant on (table index, tail length, byte
    offset) cannot tell those apart, and a constant mined from an N=8 class was written over an
    N=4 head: the symbol moved from offset 24 to 40, the payload was spliced after it, and the
    section came out sixteen bytes long. It BUILT rather than refused, which is the serious part -
    two sib-sib-mx-s.cube.grad kernels were the first images this layer has ever produced that were
    neither Apple's bytes nor a refusal.

    The fix is not a finer key. Every byte of this tail is already a contract input - the identity
    words come through `tails`, the symbol through `symbol`, the payload through
    `constant_program` - so there is nothing here for a class constant to contribute, and a rule
    that writes here can only overwrite a contract value with a class one. So this tail is skipped
    wholesale by both the constant and the rule writers.
    """
    return SYMBOL in bytes(tail)
def apply_field_rules(desc, bindings=None, threadgroup=None, insts=None, loop=None,
                      registers=None, spill=None, ti_spill=None):
    """Compute the per-kernel field values that follow from the others.

    Measured over 7,567 sections rather than the 1,377 the relation was first found on:

        slot 8  == Q                       7,567 of 7,567   exact
        slot 4  == Q - 4                   7,567 of 7,567   exact
        slot 12 == Q                       7,564 of 7,567
        slot 2  == Q - 4 + 4*(slot3/8)     7,558 of 7,567
        slot 10 == Q                       7,556 of 7,567

    where Q is slot 6. Only the two that are EXACT are computed. Computing the other three would
    build 23 sections wrong, and a rule with exceptions applied as though it had none is the
    defect this file has already had to fix twice - in describe() and in the verifier.
    """
    import copy
    out = copy.deepcopy(desc)
    for pos in table_index(out):
        t = out["tables"][pos]
        f = {int(k): v for k, v in (t.get("fields") or {}).items()}
        if not ({2, 3, 4, 6, 8} <= set(f)):
            continue
        q = f[6][2]
        vals = {4: q - 4, 8: q}
        # NAMED FIELDS, COMPUTED. isa/g17-pk-fields.json carries the schema names read out of
        # libapplegpu-nt.dylib, and a named field is one this layer can write rather than lift:
        #
        #   3   buffer_bindings_bytes   eight per binding, and the binding count is a contract input
        #   15  has_stores              16 has_global_stores   17 has_texture_stores
        #   18  has_local_stores        19 has_imageblock_stores   30 has_uniform_atomics
        #   33  has_loop                44 (tensor use)
        #
        # The has_* fields are booleans whose PRESENCE carries the meaning - a flatbuffers writer
        # omits a false one - so where the slot exists the value is 1. Writing them as 1 rather
        # than copying whatever the donor held makes them rules; byte-exactness says the two agree.
        if bindings is not None:
            vals[3] = 8 * len(bindings)
        if threadgroup is not None and 28 in f:
            vals[28] = threadgroup
        if insts is not None and loop is not None and 32 in f:
            vals[32] = size_class(insts, loop)
        # TWO MORE NAMED BACKEND SCALARS. isa/g17-pk-fields.json calls slot 0
        # temporary_register_count and slot 1 spill_buffer_bytes, and a compiler reports both the
        # way it reports its entry point - it allocated the registers and it sized the spill. They
        # were being carried from a donor for want of a name, which is what a name is worth.
        if registers is not None and 0 in f:
            vals[0] = registers
        if spill is not None and 1 in f:
            vals[1] = spill
        # AND SLOT 31, thread_invariant_spill_buffer_bytes - the other spill figure, and the field
        # whose PRESENCE I once called "what stops the whole table being generated". Its values are
        # 32, 48 and 64, which read as sizes and not as a thread count; my first guess from the
        # unordered name list was max_total_threads_per_threadgroup and the ordered mapping
        # corrected it.
        if ti_spill is not None and 31 in f:
            vals[31] = ti_spill
        for slot in (15, 16, 17, 18, 19, 30, 33, 44):
            if slot in f:
                vals[slot] = 1
        for slot, val in vals.items():
            if slot in f:
                o, w, _v = f[slot]
                f[slot] = (o, w, val)
        t["fields"] = f
        break
    # THE BINDING RECORDS TOO. Their schema names three things - bindpoint_index, gpuva_63_40 and
    # offset - and measured over 11,760 records the body is:
    #
    #     slot 0   the constant 5 in 11,724 of 11,724 present     a type tag
    #     slot 1   1, 2, 3, 4, 48 ... present in only 8,500       the bindpoint index
    #     slot 2   small even numbers, present in 4,204
    #     slot 3   the constant 1 in 7,602 of 7,602 present       a flag
    #
    # Slot 1 being absent from 3,260 records is not a missing field: a FlatBuffers writer omits a
    # field equal to its default, so bindpoint_index 0 is written by being left out. That is why
    # the index the harness reads is never zero and why corpuspack's binding list starts at 1 for
    # kernels whose source says buffer(0).
    for pos in table_index(out):
        t = out["tables"][pos]
        f = {int(k): v for k, v in (t.get("fields") or {}).items()}
        if 0 in f and f[0][2] == 5 and set(f) <= {0, 1, 2, 3}:
            for slot, val in ((0, 5), (3, 1)):
                if slot in f:
                    o, w, _v = f[slot]
                    f[slot] = (o, w, val)
            t["fields"] = f
    return out



# THE SKELETON OF A CLASS FAMILY, WRITTEN OUT RATHER THAN LIFTED.
#
# The donor is an Apple-compiled description and the endpoint asks for none, so the question is
# what it still supplies once the contract supplies every field value - which was measured, and
# the answer is the STRUCTURE: which tables exist, each one's slot set and body length, and the
# vectors. No small part of the contract determines that in general (the best subset reaches
# 32.5%), so it cannot be computed for every class. It can be WRITTEN for one.
#
# This is the structure shared by 546 of the 12,564 corpus kernels - two bound buffers, two
# declared, straight-line, no threadgroup memory - and the two most common variants of it differ
# only in the per-kernel tail, which is 8 + 4*len(builtins), a rule pinned since the builtins
# entered the contract. Everything else here is a constant of the family:
#
#     vlen is not listed because it is derived: 4 + 2*(max slot + 1), pinned corpus-wide
#     the per-kernel slot set is not listed because pk_slot_set computes it from the contract
#     the tails are not listed because they are contract inputs
#
# Honest about what it is: a skeleton in code is a class constant measured once and stated, the
# same status as g17-vtable-layout.json. What it is NOT is a donor - no Apple description is read
# at build time, and the kernel this builds has never been compiled by this project.
SKELETON_2B2D = [
    (0, (0, 3), 12),
    (1, (1,), 8),
    (2, "per-kernel", 60),
    (3, (0, 1, 2, 3), 20),
    (4, (0, 2), 12),
    (5, (0, 1, 2, 3), 18),
    (6, (0,), 8),
]
# which table each vector lists, by index into the skeleton, and in which order
SKELETON_2B2D_VECS = [[3], [6, 5], [4]]
# AND THE ORDER THE BLOCKS APPEAR IN, which is not "tables then vectors". A vector must be laid
# out BEFORE the tables it lists or it points backward and describe() refuses the image - the
# first attempt put every vector at the end and got "vector at 334 points backward at 204" for
# its trouble. The interleaving is part of the class, exactly as shape_of has always said.
SKELETON_2B2D_ORDER = ["t0", "t1", "t2", "v0", "t3", "v1", "v2", "t4", "t5", "t6"]


# THE PER-KERNEL BASE SLOT SET for this family, and the four slots that are decided rather than
# constant. Slot 28 is the threadgroup allocation, 32 the size class, 33 the loop flag - all three
# already had rules. Slot 31 did not: "present in 2,799 kernels, no candidate predicate above
# 93.8%, so it is left alone rather than invented". That was measured against CONTRACT predicates,
# and slot 31 is not a contract fact - it is the thread-invariant spill, and it is present exactly
# when the backend reports one. 12,564 kernels, 100%, two groups, no exceptions. The backend
# scalars were passed into from_contract all along and never asked this question.
PK_BASE_2B2D = (0, 1, 2, 3, 4, 6, 8, 10, 12, 13, 15, 16, 26, 27, 29)


def _skeleton_tail(ti, tails, builtins, symbol, cp):
    """The three tails this family carries, each one already a derived fact.

        table 1   contract input, delivered by contract_inputs as four-byte pieces
        table 2   the per-kernel tail, 8 + 4*len(builtins) - pinned since builtins joined
                  the contract
        table 3   [count][count entries][len(symbol)][symbol][constant program payload], which is
                  the structure this file worked out when it stopped counting the copied bytes and
                  read them. For this family the entry list is empty, so the count is zero.
    """
    import struct as _s
    if ti == 1:
        pieces = {off: v for (t, off), v in (tails or {}).items() if t == 1}
        if not pieces:
            return b""
        return b"".join(pieces[o] for o in sorted(pieces))
    if ti == 2:
        bv = [b for b in (builtins or ()) if isinstance(b, int)]
        return b"\0" * 8 + b"".join(_s.pack("<I", b) for b in bv)
    if ti == 3:
        sym = bytes(symbol or b"")
        out = _s.pack("<II", 0, len(sym)) + sym + b"\0" + bytes(cp or b"")
        return out + b"\0" * (-len(out) % 4)
    return b""


def synthesize_desc(md, sig, insts, loop, threadgroup=0, builtins=(), ti_spill=None,
                    tails=None, symbol=None, cp=b""):
    """Build a description from the contract alone, with no Apple description read.

    Returns None when the contract is outside the family this skeleton covers, because a
    constructor that guesses outside what it was measured on is the donor problem with extra
    steps. Refusing is the invariant.
    """
    if tuple(sig)[0] != 2 or tuple(sig)[3] != 2 or loop or threadgroup:
        return None
    base = set(PK_BASE_2B2D)
    if ti_spill is not None:
        base.add(31)
    pk_slots = pk_slot_set(base, threadgroup=threadgroup, insts=insts, loop=loop)
    spec = {}
    for i, (j, slots, tlen) in enumerate(SKELETON_2B2D):
        sl = sorted(pk_slots) if slots == "per-kernel" else list(slots)
        if slots == "per-kernel":
            tlen = 60 + 4 * (31 in pk_slots)
        spec["t%d" % i] = (sl, tlen, 4 + 2 * ((max(sl) + 1) if sl else 0))

    # First pass places every block so the vector entries can be filled with real positions;
    # relayout recomputes the exact offsets afterwards, so only the ORDER has to be right here.
    tables, vectors, tpos = {}, {}, {}
    pos = 16
    for item in SKELETON_2B2D_ORDER:
        if item[0] == "t":
            sl, tlen, vlen = spec[item]
            ti = int(item[1:])
            tail = _skeleton_tail(ti, tails, builtins, symbol, cp)
            tables[pos + vlen] = dict(vtpos=pos, vlen=vlen, tlen=tlen, shared=False,
                                      slots={k: 0 for k in sl},
                                      fields={k: (0, 4, 0) for k in sl}, tail=tail)
            tpos[item] = pos + vlen
            pos += vlen + tlen + len(tail)
        else:
            n_rec = len(SKELETON_2B2D_VECS[int(item[1:])])
            vectors[pos] = [None] * n_rec
            tpos[item] = pos
            pos += 4 + 4 * n_rec
    for vi, vec in enumerate(SKELETON_2B2D_VECS):
        vectors[tpos["v%d" % vi]] = [tpos["t%d" % k] for k in vec]
    order = [tpos["t%d" % i] for i in range(len(SKELETON_2B2D))]
    return dict(root=order[0], tables=tables, vectors=vectors, order=list(order),
                size=pos, extra={})


def rebuild_tails(desc, tails=None, builtins=(), symbol=None, cp=b"", pk=None, lists=None):
    """Recompute every tail from contract facts instead of keeping the donor's.

    THE THIRD BLOCK OFF THE DONOR'S LIST, after the field values and the vtable offset maps. Each
    of the three kinds of tail is a fact this project already established and had never used in
    place of what was transplanted:

        the ENTRY SYMBOL tail   [8]["agc.main"], padded            1,463 of 1,463 exact
        the CONSTANT PROGRAM    [count][entries][len(sym)][sym][NUL padded to 4][payload, padded]
                                1,463 of 1,463 exact. The symbol is padded to four BEFORE the
                                payload begins, which is the whole of what the first attempt got
                                wrong - it read as "same length, different content" for 751 of
                                1,163 kernels. The entry LIST is kept from the donor because
                                nothing determines it yet, and that is stated rather than hidden.
        the PER-KERNEL tail     8 + 4*len(builtins)                1,432 of 1,463 exact

    The 31 exceptions are kernels whose builtin vector did not parse - builtin_vector returns
    ("len", n) for those - and they keep the donor's tail rather than being guessed at, which is
    the invariant applied one level down.

    Measured in the build path over 503 kernels: 493 byte-exact, identical to the control.
    """
    import struct as _s
    sym = bytes(symbol or b"")
    # NONE IS NOT THE EMPTY VECTOR. A caller that cannot say what the builtins are gets the
    # donor's per-kernel tail back, not a rebuilt one - passing () for "unknown" rebuilt it as
    # eight bytes and broke two cases that call from_contract without an env. Refusing to derive
    # is the invariant one level down, and "I do not know" has to be a distinct value from
    # "there are none" for that to work.
    unknown = builtins is None
    unparsed = bool(builtins) and not isinstance(builtins[0], int)
    for pos in table_index(desc):
        t = desc["tables"][pos]
        real = bytes(t.get("tail") or b"")
        if not real:
            continue
        if pos == pk:
            # REBUILT FROM THE LISTS, which reconstructs the whole tail rather than its last few
            # words. The old form kept the donor's first eight bytes and appended the builtin ids,
            # which is right only when the first list is empty - 84% of kernels - and silently
            # inherited two donor words for the other 16%. The stripped-donor arm caught it: 20 of
            # 493 differed, every one of them in this tail, every one a kernel whose builtin
            # vector had not parsed.
            got = lists if lists is not None else None
            if got is not None:
                t["tail"] = b"".join(_s.pack("<I", len(g)) + b"".join(_s.pack("<I", x) for x in g)
                                     for g in got)
                continue
            if unparsed or unknown:
                continue
            bv = [b for b in (builtins or ()) if isinstance(b, int)]
            t["tail"] = real[:8] + b"".join(_s.pack("<I", b) for b in bv)
        elif sym and sym in real:
            i = real.find(sym)
            out = real[:i - 4] + _s.pack("<I", len(sym)) + sym + b"\0"
            out += b"\0" * (-len(out) % 4)
            out += bytes(cp or b"")
            t["tail"] = out + b"\0" * (-len(out) % 4)
        elif b"agc.main" in real:
            e = b"agc.main"
            out = _s.pack("<I", len(e)) + e + b"\0"
            t["tail"] = out + b"\0" * (len(real) - len(out)) if len(real) > len(out) else out
    return desc


def from_contract(donor, constant_program=None, semantics=None, bindings=None, counts=None,
                  tails=None, gaps=None, symbol=None, threadgroup=None, insts=None, loop=None,
                  entry_symbol=None, registers=None, spill=None, ti_spill=None,
                  env=None):
    """Fill a donor description from CONTRACT inputs only - no target metadata anywhere.

    generate() above takes the target's own description and is therefore a measurement of whether
    the format can be computed, not a path a linker can walk: a linker does not have Apple's
    metadata for the program it is building. This is the path. Everything it varies comes from
    the backend contract:

        constant_program   Kernel.constant_program - the bytes after the symbol name
        symbol             the constant program's symbol name, which is NOT always the same:
                           six kernels - i64 div and mod, at each of three unroll counts - carry
                           `agc.main.constant_program.cfg`, four bytes longer than everyone
                           else's `agc.main.constant_program`. Splicing the payload in after a
                           fixed name put those four bytes into the payload and shifted the rest,
                           and it read as one byte of unexplained semantics.
        semantics          Kernel.semantics, {offset: bytes}, for values a class cannot carry
        bindings           the buffer indices, which fill the binding vector
        counts             (table index, slot) -> value, for the resource figures

    For class 3z.d3 the measured degrees of freedom are exactly three - the constant program, one
    resource figure and a small count - and 49 of the 52 description fields are class constants.
    So this signature is the whole of what a backend must supply for that class.
    """
    import copy
    gen = copy.deepcopy(donor)
    # COMPUTE THE PER-KERNEL VTABLE INSTEAD OF INHERITING IT, when the caller states the three
    # facts its optional slots follow from. Byte-exactness is the check: the computed vtable has
    # to equal the one the donor carried, or the build stops being byte-identical and the
    # regression says so.
    if threadgroup is not None or insts is not None or loop is not None:
        gen, _how = apply_pk_rules(gen, threadgroup, insts, loop)
    # AND EVERY OTHER TABLE'S SLOT ARRAY WITH IT. The per-kernel table was the first piece; the
    # record tables are the bulk of what was still being transplanted, and their layouts are a
    # function of the slot set, the field widths and the declared body length.
    gen, _n, _undecided = apply_vtable_rules(gen, env)
    # ONLY WHERE THE CALLER SUPPLIED THE CONTRACT. vtable_layout_x is keyed on the constant
    # program's length and lives in `env`, so a caller that passes no env cannot reach a third of
    # the layout tables and would see thousands of maps as underivable - refusing then would
    # measure the caller withholding information, not the model lacking a rule. The linker path
    # (build_for) always passes it; the measurement helpers that do not are asking a narrower
    # question on purpose.
    if _undecided and env:
        raise Ambiguous("no rule reproduces the offset map of %d vtable(s); the donor's would be "
                        "inherited rather than derived" % len(_undecided))
    # AND THE TWO FIELD VALUES THAT FOLLOW FROM ANOTHER. Slot 4 is Q - 4 and slot 8 is Q, exactly,
    # over every section in the corpus - so they are arithmetic rather than class state.
    # THE MINED TABLES ARE GONE. 27,597 rules across six tables - field, tail, tail-back, record,
    # field constants, tail constants - and withdrawing any of them changed 0 of 11,489 builds.
    # Not "redundant with the donor", which was the standing explanation and was never measured:
    # with the donor's field values destroyed and the rules withdrawn, the contract still rebuilds
    # every one. They reproduced values the CONTRACT already determines, and two of them were
    # refuted by the first 28 kernels the model had never seen. A correlation that predicts
    # nothing the model does not already know, and that breaks off-corpus, is not part of an
    # explanation - so it is not kept as one.

    # AND THE TAILS, recomputed rather than transplanted. See rebuild_tails: the entry symbol and
    # the constant-program tail are exact corpus-wide, the per-kernel tail follows the builtins,
    # and a kernel whose builtin vector did not parse keeps what the donor had rather than being
    # guessed at.
    if symbol is not None or constant_program is not None:
        from . import gpumd as _GM
        _pk = None
        for _p in table_index(gen):
            if {26, 27, 29} <= {int(k) for k in (gen["tables"][_p].get("slots") or {})}:
                _pk = _p
                break
        gen = rebuild_tails(gen, tails, env.get("BV") if env else None, symbol,
                            constant_program or b"", _pk,
                            lists=env.get("BLISTS") if env else None)
    gen = apply_field_rules(gen, bindings, threadgroup, insts, loop, registers, spill,
                            ti_spill)
    # TAILS AS AN EXPLICIT INPUT, {table index: bytes}. describe() treats a tail as opaque, but
    # the ones that vary within a class are not opaque - table 1's tail carries a numeric field at
    # byte 8 that is 0 in 85.9% of kernels and takes small values otherwise, and it correlates
    # with nothing in the source. Carrying tails measures how much contract surface the remaining
    # failures represent, which bounds the work even before the field is understood.
    # Tail WORDS, {(table index, byte offset): 4 bytes}. Measured across every class group, only
    # ten positions vary outside the constant-program payload, and all ten are four-byte aligned:
    # five 32-bit words in table 1's tail and five in table 3's. The rest of every tail is a class
    # constant. So this is the whole of the tail contract surface.
    for key, raw in (tails or {}).items():
        if isinstance(key, tuple):
            ti, off = key
            if ti < len(gen["order"]):
                pos = gen["order"][ti]
                t = bytearray(gen["tables"][pos].get("tail") or b"")
                if off + len(raw) <= len(t):
                    t[off:off + len(raw)] = raw
                    gen["tables"][pos]["tail"] = bytes(t)
        elif key < len(gen["order"]):
            gen["tables"][gen["order"][key]]["tail"] = raw
    # THE ENTRY SYMBOL, which is a contract input and was being transplanted. Record tails are
    # dominated by two patterns: 2,288 of 4,602 begin `08 00 00 00 61 67 63 2e` - a FlatBuffers
    # string of length 8 followed by "agc.main" - and 1,484 begin `00 00 00 00 19 00 00 00`, the
    # 25-byte constant-program symbol. The second was already written from the contract; the first
    # was not, so the entry point's own name was coming from a donor.
    for pos in table_index(gen):
        t = gen["tables"][pos]
        tail = bytes(t.get("tail") or b"")
        i = tail.find(b"agc.main")
        if i < 4 or tail[i:i + 9] == b"agc.main." or SYMBOL in tail:
            continue
        end = tail.index(b"\0", i)
        nm = entry_symbol if entry_symbol is not None else tail[i:end]
        head = tail[:i - 4] + struct.pack("<I", len(nm)) + nm + b"\0"
        t["tail"] = head + tail[len(head):] if len(head) <= len(tail) else head
    if constant_program is not None:
        for pos in table_index(gen):
            tail = gen["tables"][pos].get("tail") or b""
            i = tail.find(SYMBOL)
            if i < 0:
                continue
            end = tail.index(b"\0", i)
            head = tail[:i] + (symbol if symbol is not None else tail[i:end]) + b"\0"
            # The name is NUL-padded to a four-byte boundary before the payload, and the boundary
            # is the SECTION's, not the tail's - so pad against where this tail will sit.
            head += b"\0" * (-(pos + body_len(gen["tables"][pos]) + len(head)) % 4)
            gen["tables"][pos]["tail"] = head + constant_program
            break
    if bindings is not None and gen["vectors"]:
        last = sorted(gen["vectors"])[-1]
        recs = gen["vectors"][last]
        if len(recs) != len(bindings):
            return None, "donor has %d binding records, the signature has %d" % (
                len(recs), len(bindings))
        # AND WRITE THEM, which this did not do. It checked the COUNT and let the donor's own
        # indices stand, so a class holding two kernels that bind different buffer numbers would
        # build one of them pointing at the other's buffers - an image that loads, dispatches and
        # reads the wrong memory. Five of 734 classes hold more than one index list, covering 24
        # kernels; they were reaching the gate as refusals rather than as wrong answers, which is
        # luck rather than design.
        for rec, idx in zip(recs, bindings):
            t = gen["tables"].get(rec)
            if t is None:
                continue
            f = {int(k): v for k, v in (t.get("fields") or {}).items()}
            if 1 in f:
                o, w, _v = f[1]
                f[1] = (o, w, int(idx))
                t["fields"] = f
    for off, recs in gen["vectors"].items():
        for k, r in enumerate(recs):
            if r < off + 4 + 4 * k:
                return None, "vector at %d points backward at %d" % (off, r)
    # THE GAPS ARE WALKED AFTER THE VTABLE LENGTHS, NOT BEFORE. build_for used to compute
    # gaps_of(donor) at the call site and hand them in, and blocks() sizes a vtable by its vlen -
    # so the gaps were measured against the DONOR'S lengths and the donor's vlen was load-bearing
    # even though the builder recomputes every one of them. Destroying it scored 0 of 493 for
    # that reason alone, which read as "the builder cannot compute vlen" and was really "the
    # builder computes it too late to matter".
    #
    # Walking them here, on `gen`, uses the lengths this function just wrote. A caller that has a
    # reason to pin the gaps can still pass them.
    if gaps is None:
        gaps = gaps_of(gen)
    out = relayout(gen, trim=False, gaps=gaps)
    # build_from looks up overrides by the table's position in the description it is GIVEN, so
    # the keys have to be the post-relayout positions. Keyed by the donor's pre-relayout ones,
    # every override silently missed and the donor's own values stayed - which is why supplying
    # the resource figures changed nothing for most families.
    values = {}
    idx = table_index(out)
    for (ti, slot), v in (counts or {}).items():
        if ti < len(idx):
            values[(idx[ti], slot)] = v
    # REFERENCE FIELDS, RECOMPUTED FROM WHERE THINGS LANDED. In the per-kernel table slots 4, 6,
    # 8, 10 and 12 are the buffer, image_state, sampler_state, driver_buffer and
    # constant_driver_buffer vectors, and 13, 26, 27 and 29 are static_constants, const_calc_phases,
    # thread_invariant_only_buffer_list and special_regs_read. Every one is a FlatBuffers offset -
    # the distance from the field to its target - so it follows from the layout rather than from
    # the class, and carrying it from a donor is only safe while nothing moves.
    # AND NOT ONLY THE PER-KERNEL TABLE'S. This loop carried a `break`, so the FIRST table
    # matching the guard was fixed up and every other table's references were left pointing where
    # the donor put them. It went unnoticed because those kernels were refusing for other reasons;
    # once the tie-breaks let them build, table 1 slot 2 came out wrong on 144 bytes. A reference
    # in any table is still a distance to a target, and a target that moved invalidates it
    # wherever it lives.
    #
    # The per-kernel table keeps its named slot list, because eight of its nine references point at
    # EMPTY vectors - a single zero word is not a block, so the target cannot be recognised by
    # position and has to be found through its container. Every other table is handled by asking
    # whether the donor's own description places a table or a vector exactly where the field
    # points; that is a property of the donor, which is what supplies the structure, so it reads
    # nothing about the kernel being built.
    moved = out.get("_moved") or {}
    known = set(gen["tables"]) | set(gen.get("vectors") or {})
    for opos in table_index(gen):
        t0 = gen["tables"][opos]
        f0 = {int(k): v for k, v in (t0.get("fields") or {}).items()}
        pk = {2, 3, 4, 6, 8} <= set(f0)
        cand = [sl for sl in (4, 6, 8, 10, 12, 13, 26, 27, 29)] if pk else [
            sl for sl, (o0, w0, v0) in f0.items()
            if w0 == 4 and v0 > 0 and (opos + o0 + v0) in known]
        if not cand:
            continue
        npos = moved.get(opos, opos)
        nt = out["tables"].get(npos)
        if nt is None:
            continue
        nf = {int(k): v for k, v in (nt.get("fields") or {}).items()}
        # A TARGET NEED NOT BE A BLOCK START. Eight of these nine point at an EMPTY vector - a
        # single zero word for dma_list, image_state_bindings, driver_buffer_bindings,
        # constant_driver_buffer_bindings, static_constants, const_calc_phases,
        # thread_invariant_only_buffer_list and special_regs_read - and an empty vector is not a
        # block, so looking the target up by block start found only two of them. It moves with
        # whatever block CONTAINS it, so find the container and carry the offset within it.
        spans = []
        for _s0, _kind, _key, _sz in blocks(gen):
            spans.append((_s0, _s0 + _sz, _key))
        for slot in cand:
            if slot not in f0 or slot not in nf:
                continue
            o0, w0, v0 = f0[slot]
            if w0 != 4:
                continue
            tgt = opos + o0 + v0
            new_tgt = moved.get(tgt)
            if new_tgt is None:
                for lo, hi, key in spans:
                    if lo <= tgt < hi and key in moved:
                        # a table's block starts at its vtable; moved[] records the TABLE position
                        base = lo
                        nb = moved[key] - (gen["tables"][key]["vlen"]
                                           if key in gen["tables"]
                                           and not gen["tables"][key].get("shared") else 0)
                        new_tgt = nb + (tgt - base)
                        break
            if new_tgt is None:
                continue
            o1, w1, _v1 = nf[slot]
            nf[slot] = (o1, w1, new_tgt - (npos + o1))
        nt["fields"] = nf
    blob = bytearray(M.build_from(out, values))
    for off, raw in (semantics or {}).items():
        if off + len(raw) <= len(blob):
            blob[off:off + len(raw)] = raw
    return bytes(blob), None


def contract_inputs(desc):
    """The contract inputs a backend would hand over, read out of a described section.

    This is a MEASUREMENT of what the contract has to carry, not a linker input path: it reads
    the values out of Apple's own metadata so the from_contract() path can be scored. A linker
    gets the same values from the program it compiled.
    """
    # Trim here, against the TARGET's own offsets, which is where the truth is. relayout is
    # called with trim=False on the generate path because trimming there would use the donor's
    # spacing - so if the tails are not already correct when they arrive, a tail that describe()
    # read past the end of its section stays too long and the built image is twelve bytes over.
    desc = trim_tails(desc)
    cp = sym = None
    for pos in table_index(desc):
        tail = desc["tables"][pos].get("tail") or b""
        i = tail.find(SYMBOL)
        if i < 0:
            continue
        end = tail.index(b"\0", i)
        sym = tail[i:end]
        j = end + 1
        j += -(pos + body_len(desc["tables"][pos]) + j) % 4
        cp = tail[j:]
        break
    # TAIL BYTES, not whole tails. Measured over every class group with three or more members,
    # 1,037 of 1,167 tail byte positions NEVER vary - tails are overwhelmingly class constants.
    # What varies is table 3 bytes 40 onward, which is the constant program payload and already a
    # contract field, and a handful of numeric positions elsewhere. So the contract does not need
    # tails wholesale; it needs the constant program plus a small set of named byte positions.
    #
    # THE WHOLE TAIL, NOT THE FIRST SIX WORDS, added when a kernel refuted the measurement above.
    # The counts in that comment were taken over the corpus as it stood and were right about it;
    # sib-c4probe-9eb9aeab is not. Its per-kernel tail is
    #
    #     02000000 23000000 24000000 05000000 05000000 06000000 08000000 09000000
    #
    # where its donor's reads ...06000000 04000000 05000000 06000000 08000000, and the build came
    # out one byte wrong at offset 216 - the FIRST wrong build since this path was written, and
    # wrong rather than refused, which is the worst way for a gap to show. The differing word sits
    # at tail offset 24 and the loop stopped at 20. The bytes are a vector of builtin IDs; a
    # backend knows which builtins it emitted, so carrying them is a contract input, not a
    # concession. Guarded by contract_written_tail so the constant program's tail, which carries
    # the symbol and is handled by `cp`, is never duplicated here.
    #
    # AND INDEXED BY `order`, WHICH IS NOT table_index. My first attempt at this fix added a case
    # for ti == 2 because the diagnosis had located the byte in table_index[2]; order[2] is the
    # entry-symbol table and table_index[2] is the per-kernel one. The patch wrote four harmless
    # words into the wrong tail and changed nothing - identical numbers after a change, which is
    # the rule that caught it.
    tails = {}
    for ti, pos in enumerate(desc["order"]):
        raw = bytes(desc["tables"][pos].get("tail") or b"")
        if ti in (1, 3) and not contract_written_tail(raw):
            for off in range(0, len(raw) - 3, 4):
                tails[(ti, off)] = raw[off:off + 4]
    counts = {}
    for ti, pos in enumerate(table_index(desc)):
        for slot, (_off, _w, v) in (desc["tables"][pos].get("fields") or {}).items():
            counts[(ti, int(slot))] = v
    return cp, counts, tails, sym



# ---------------------------------------------------------------------------------------------
# THE PATH A LINKER CAN ACTUALLY WALK.
#
# class_table() is keyed by (signature, shape_of(target)) and is a MEASUREMENT: it asks whether
# the format is determined by contract inputs once the class is known. A linker knows the
# signature and not the shape, so it needs a table keyed only by what a backend hands over.
#
# What that turns out to be, in full:
#
#     the source signature       bound count, starts at zero, any constant binding, declared
#                                resources, and the argument type set
#     the constant program       its bytes, which the backend compiled
#     N                          the count at the head of the constant-program tail, a vector of
#                                N u32s holding 0..N-1, N in {0, 4, 8, 12}
#     the instruction count      how many instructions the backend emitted
#     whether it loops           whether any of them branches backward
#
# The last two are the only code-derived quantities, they are two scalars rather than spans, and
# they are necessary: slot 32 of the per-kernel table is 0 at 30 instructions or fewer, 3 when the
# program loops, 2 above 300 instructions and 1 otherwise, exactly, over 7,594 sections - and no
# signature property reproduces that.
#
# Measured over the cache: 5,464 of 7,561 sections build BYTE-EXACT this way, 2,097 are refused,
# and NONE is built wrong. The donor is never the kernel itself and the gaps come from the donor,
# so nothing here reads the target's own layout.
# ONE COPY OF THE BRANCH SET, in g17cf, where the control-flow encoding lives. Three files held
# their own literal and a fourth opcode appearing would have had to be added to all of them.
def _branch_opcodes():
    from . import cf as g17cf
    return g17cf.BRANCH_OPCODES


# BOTH BOUNDARIES ARE MEASURED, AND ONE OF THEM USED NOT TO BE. This read "2 above 272
# instructions" and reported no exceptions over 7,594 sections, which was true and meant nothing:
# 272 is the midpoint of a GAP in the corpus - nothing was compiled between 272 and 310 - so the
# constant was never tested, only never contradicted. A generated kernel at 292 instructions came
# back 1, not 2, and refuted it.
#
# A gap is only a gap until something is compiled into it. Sweeping a straight-line kernel one
# instruction at a time across the interval pins both edges exactly:
#
#     30 instructions   slot 32 absent      31 instructions   slot 32 = 1
#    300 instructions   slot 32 = 1        301 instructions   slot 32 = 2
#
# So the thresholds are 30 and 300, both round, and the second was off by 28. Machine-code SIZE
# does not separate the classes in either direction - slot32=1 reaches 3,368 bytes and slot32=2
# starts at 2,422 - so the quantity really is the instruction count.
SIZE_LO = 30
SIZE_HI = 300


def size_class(insts, loop):
    """The per-kernel table's slot-32 value, from the two scalars: 0 at 30 instructions or fewer,
    3 if the program loops, 2 above 300 instructions, 1 otherwise.

    The BUCKET is what the class key wants, not the raw count. Keyed on the count itself the
    contract decides 80.5% of sections and keyed on the bucket it decides 87.5%, because a finer
    key leaves fewer classes with a second witness and a key with one member is refused.
    """
    return 0 if insts <= SIZE_LO else (3 if loop else (2 if insts > SIZE_HI else 1))


_CFACTS = None


def code_facts(text, entry):
    """(instruction count, whether any branch goes backward) - the two scalars in the contract.

    MEMOISED, BECAUSE IT FORKS. g17ref.walk writes the code to a temp file and runs Apple's
    decoder as a subprocess, so a corpus pass was 12,564 forks: profiling one showed code_facts
    at 79% of the time, 0.630s of 0.794s over 151 kernels, with subprocess.run 0.581s of that and
    select.poll 0.410s. Everything I had assumed was expensive - re-reading s.metal for written_of,
    read_of and the rest - was 0.027s.
    
    The answer to that is not more processes, it is not doing the work twice. (ni, loop) is a pure
    function of the code bytes, the object files never change, and the cache is keyed on the bytes
    themselves plus the decoder binary's mtime, so a rebuilt decoder invalidates everything.
    """
    import hashlib
    from . import cache as g17cache
    from . import ref as g17ref
    from . import cf as g17cf
    global _CFACTS
    if _CFACTS is None:
        _CFACTS = g17cache.load("code-facts") or {}
    k = (hashlib.blake2b(text, digest_size=16).digest(), entry)
    hit = _CFACTS.get(k)
    if hit is not None:
        return hit
    br = _branch_opcodes()
    ins = list(g17ref.walk(text, entry))
    loop = False
    for o, l, op in ins:
        if op in br:
            try:
                if g17cf._disp(text[o:o + l]) < 0:
                    loop = True
            except Exception:
                loop = True          # a branch this file cannot read is treated as backward
    out = (len(ins), loop)
    _CFACTS[k] = out
    if len(_CFACTS) % 2000 == 0:
        _save_code_facts()
    return out


def _save_code_facts():
    """Persist the decoded facts, merging with whatever another run wrote in the meantime.

    Two processes each hold their own dict and the last writer would otherwise erase the other's
    work - not a correctness bug, since the entries are a pure function of the bytes, but a real
    one for wall-clock: the loser re-forks agx3dis for every kernel it had already decoded. Merge
    under the lock and both runs keep what they learned.
    """
    from . import cache as g17cache
    if not _CFACTS:
        return
    with g17cache._Lock("code-facts"):
        merged = dict(g17cache.load("code-facts") or {})
        merged.update(_CFACTS)
        g17cache.save("code-facts", merged)


import atexit as _atexit
_atexit.register(lambda: _save_code_facts())


def identity_count(desc):
    """N, the length of the 0..N-1 vector at the head of the constant-program tail."""
    for pos in table_index(desc):
        tail = desc["tables"][pos].get("tail") or b""
        if b"agc.main.constant_program" in tail and len(tail) >= 4:
            k = struct.unpack_from("<I", tail, 0)[0]
            if 4 + 4 * k <= len(tail) and all(
                    struct.unpack_from("<I", tail, 4 + 4 * i)[0] == i for i in range(k)):
                return k
            return 0
    return 0


def entry_list(desc):
    """The constant-program tail's entry vector, VERBATIM, for the key to carry.

    identity_count RETURNS ZERO FOR THREE DIFFERENT THINGS, and the key could not tell them apart:
    no constant-program tail at all, a tail whose count is zero, and - the one that matters - a
    tail whose vector is a perfectly good list that simply does not start at zero. It only ever
    reported a length, and only when the vector was exactly 0..k-1.

    Measured over a stride sample of the corpus: of the kernels carrying a non-empty vector, 406
    are a contiguous run from 0, 44 a run from 4, 36 a run from 8, and 10 are not a run. So about
    18% of them - roughly 630 kernels - were being filed in the SAME class as the 761 with no
    vector whatsoever.

    That is not cosmetic, because rebuild_tails says in its own docstring that "the entry LIST is
    kept from the donor because nothing determines it yet". The key is the only thing standing
    between a kernel and a donor's entry list, and for those 630 it was standing on a value that
    says nothing about the list. Nothing has been built wrong - 13,465 kernels, 0 wrong - but that
    is the score being silent, not the model being right: a wrong explanation on a right key is
    invisible until the pairing happens.

    Verbatim rather than (start, length), deliberately. A run is what 98% of them are, and encoding
    that guess into the key would put a hypothesis where a fact belongs - and this file has three
    refuted hypotheses about this very vector already: it is not register numbers (425 disagree,
    71 agree, d_b6 has 20 entries with regs=1) and it is not word indices into the constant
    program (90 of 496 run past cp_len/4).

    identity_count is left exactly as it was. It feeds build_env's N and two analysis instruments,
    where an int is what the consumers expect; this is the KEY's reader and only the key's.
    """
    # ANCHORED ON THE SYMBOL IN THE SECTION, NOT ON A TABLE BOUNDARY. The tail is
    # [count][entries][len(symbol)][symbol], so given the symbol's offset the length word sits at
    # i-4 and the entries end at i-8; the count is the word at i-8-4k whose VALUE is k, and
    # scanning k upward from zero finds it. Nothing here depends on describe() having put the
    # table boundary in the right place, which is the whole point.
    #
    # BECAUSE IT DOES NOT ALWAYS. Six kernels here - cf-if_uniform, cf-loop_dyn and four ho-
    # kernels - have a constant-program table whose declared length is zero and no slots, because
    # read_table accepted the COUNT WORD as a vtable header: the value 4 reads as vlen=4, tlen=0,
    # and the word after it as a table with soffset 4. The tail then began one word late, the
    # count was lost, and the vector read back as (5,6,7,25) - where 25 is 0x19, the length of
    # "agc.main.constant_program", read as an entry. The true vector is (4,5,6,7).
    #
    # THAT ARTEFACT WAS BRIEFLY RECORDED AS A FINDING, AND THE WAY IT SURVIVED IS THE LESSON. It
    # appears in Apple's own shipping code too - one MPS attention kernel - and two populations
    # with nothing in common producing the same four numbers looked like strong evidence for a
    # phenomenon. They were running the same reader. Agreement between populations is evidence
    # only when the instrument is not shared.
    #
    # The 24 gemv_* functions the peer session found are the same defect from the other side: a
    # count of 20 where only 16 entries fit before the name, so four words of the name came back
    # as integers - 778266465 is b"agc.", 1852399981 is b"main".
    md = M.build_from(desc)
    i = md.find(b"agc.main.constant_program")
    if i < 8:
        return ()
    k = 0
    while True:
        p2 = i - 8 - 4 * k
        if p2 < 0:
            return ()
        if struct.unpack_from("<I", md, p2)[0] == k:
            return tuple(struct.unpack_from("<I", md, p2 + 4 + 4 * j)[0] for j in range(k))
        k += 1


def builtin_lists(md):
    """The per-kernel tail read as a sequence of COUNTED LISTS of builtin ids, or None.

    THE STRUCTURE builtin_vector's own comment says it does not understand. That comment quotes
    c4probe's tail verbatim - 02000000 23000000 24000000 06000000 04000000 ... 0a000000 - and
    reads it as "not zeros-count-ids", which is right: it is TWO lists. n0=2 with ids 0x23 and
    0x24, then n1=6 with ids 4, 5, 6, 8, 9, 0x0a. The common shape everyone had been parsing is
    the same format with an empty first list, which is why a leading zero word looked like
    padding and the second word looked like the only count.

    12,591 of 12,597 per-kernel tails are exactly two counted lists that consume the tail; the
    remaining six are one list and nothing after it. Nothing is left over in any of them.

    Returned as a tuple of tuples so the boundary survives. builtin_vector is deliberately left
    alone: it feeds the contract key, and re-partitioning 12,564 kernels is a separate change with
    its own gate.
    """
    from . import gpumd as GM
    md = bytes(md)
    pk = GM.kernel_table(md)
    if pk is None:
        return None
    try:
        desc = M.describe(md)
        tail = bytes(desc["tables"][pk].get("tail") or b"")
    except Exception:
        return None
    if len(tail) < 4 or len(tail) % 4:
        return None
    w = list(struct.unpack_from("<%dI" % (len(tail) // 4), tail, 0))
    out, i = [], 0
    while i < len(w):
        n = w[i]
        i += 1
        if i + n > len(w):
            return None                      # not this shape; say so rather than guess
        out.append(tuple(w[i:i + n]))
        i += n
    return tuple(out) if i == len(w) else None


def builtin_vector(md):
    """The per-kernel table's tail read as a vector of builtin ids, or None.

    Four zero bytes, a count, then that many u32s. The entries are the builtins the body consumes
    - threadgroup_position_in_grid is 0, thread_position_in_grid is 80, a simd builtin adds 58 -
    and they are what the CODE uses, not what the signature declares: mp-smoothstep declares
    threadgroup_position_in_grid and carries an empty vector. Read here out of the section, which
    stands in for a backend that knows which builtins it lowered, exactly as the resource figures
    do.

    It is NOT the read_sr immediates: over 700 decoded kernels the nonzero entries equal the
    distinct read_sr immediates in 53 and differ in 647 - a6-tgidx carries [0, 48, 49] and reads
    two special registers.
    """
    from . import gpumd as GM
    md = bytes(md)
    pk = GM.kernel_table(md)
    if pk is None:
        return None
    try:
        desc = M.describe(md)
        tail = bytes(desc["tables"][pk].get("tail") or b"")
    except Exception:
        return None
    # THE TWO-LIST PARSE FIRST. Everything below is the old reader kept for the six tails the
    # list form does not fit; for the other 12,591 the vector is now a FACT rather than a length.
    # The ("len", N) fallback said "this function cannot read this", and keying on a length was
    # the honest thing to do while that was true. It is no longer true.
    lists = builtin_lists(md)
    if lists is not None:
        return lists
    if len(tail) < 8:
        return None
    k = struct.unpack_from("<I", tail, 4)[0]
    if 8 + 4 * k != len(tail):
        # A SECOND TAIL FORM, AND RETURNING None FOR IT THREW AWAY THE ONE FACT IT CARRIES.
        # c4probe's per-kernel tail is
        #     02000000 23000000 24000000 06000000 04000000 ... 0a000000
        # which is not zeros-count-ids: the first word is 2 and the second is 0x23, so the length
        # check fails and this returned None. Its sibling's tail is the same shape and four bytes
        # shorter, so with None on both sides they shared a contract key, disagreed about the
        # shape, and the linker refused both - having first built one of them WRONG, before the
        # tail became a contract input.
        #
        # The length is still a fact, and it is one a backend knows: it emitted the vector. So an
        # unparsed tail keys on its length rather than on nothing, which separates 40 from 36
        # without pretending to read a structure this function does not understand.
        return ("len", len(tail))
    return tuple(struct.unpack_from("<I", tail, 8 + 4 * i)[0] for i in range(k))


ARG_RE = _re.compile(r"(?:device|constant)\s+([\w:]+)\s*\*\s*(\w+)\s*\[\[\s*buffer\((\d+)\)")


def read_of(tag):
    """Which bindings the kernel LOADS from, read from its Metal source.

    The mirror of written_of, which existed while this did not, and the asymmetry cost real
    determination. cv-h2f and cv-s2f differ in one character of body - float(h[tg.x]) against
    float(s[tg.x]) - and share their signature, their writes, their declared types, their
    instruction count and their register figures. Their table 4 differs by two bytes of declared
    body length, so they are different classes, and nothing in the contract said so. Which
    bindings a kernel reads is the other half of "bindings and binding kinds" and a compiler knows
    it for the same reason it knows the writes.
    """
    src = source_text(tag)
    if not src:
        return ()
    m = _re.search(r"\b(kernel|vertex|fragment)\s+[\w:]+\s+(\w+)\s*\(", src)
    if not m:
        return ()
    i = src.index("(", m.end() - 1)
    depth, params, body = 0, "", src
    for j in range(i, len(src)):
        if src[j] == "(":
            depth += 1
        elif src[j] == ")":
            depth -= 1
            if depth == 0:
                params, body = src[i + 1:j], src[j + 1:]
                break
    out = []
    for _ty, nm, num in ARG_RE.findall(params):
        for mm in _re.finditer(r"\b%s\s*\[" % _re.escape(nm), body):
            rest = body[mm.end():]
            k = rest.find("]")
            # a subscript that is not the left side of an assignment is a load
            if k >= 0 and not _re.match(r"\s*=(?!=)", rest[k + 1:]):
                out.append(int(num))
                break
    return tuple(sorted(set(out)))


def bound_indices(tag):
    """The buffer indices the kernel actually binds, read from its own signature.

    A BACKEND FACT THE KEY NEVER CARRIED. source_signature records how many buffers are bound and
    whether the first is at zero, and throws the numbers away - so a kernel binding 1 and 5 and one
    binding 5 and 14 had identical signatures. Apple's sections do not agree: over the counterexam-
    ples, a binding at 9 or 14 costs four bytes that the same kernel with bindings under 8 does
    not spend, which is what a per-eight-bindings word looks like.

    This is what `read` was standing in for. Two kernels were separated by their read sets when
    what actually differed was the INDEX of the buffer being read, and keying on the read set
    happened to separate them because the sets differed too.
    """
    src = source_text(tag)
    if not src:
        return ()
    m = _re.search(r"\b(kernel|vertex|fragment)\s+[\w:]+\s+(\w+)\s*\(", src)
    if not m:
        return ()
    i = src.index("(", m.end() - 1)
    depth, params, body = 0, "", src
    for j in range(i, len(src)):
        if src[j] == "(":
            depth += 1
        elif src[j] == ")":
            depth -= 1
            if depth == 0:
                params, body = src[i + 1:j], src[j + 1:]
                break
    out = []
    for _ty, nm, num in ARG_RE.findall(params):
        if _re.search(r"\b%s\b" % _re.escape(nm), body):
            out.append(int(num))
    return tuple(sorted(set(out)))


_SRCTXT = {}


def source_text(tag):
    """The kernel's Metal source, read once.

    NINE OPENS PER KERNEL. written_of, read_of, signed_stores, barrier_kinds, bound_indices and
    _arg_types each opened and re-parsed the same s.metal independently, and source_signature
    opens it again - 113,491 opens across a class-table rebuild of 12,564 kernels, 1.1 s of pure
    open() inside an 11.7 s rebuild. The file does not change while a process runs, and the
    string is immutable, so it is read once and shared.

    This is the thing I WRONGLY blamed for the corpus-pass profile, where it was 3% and the
    subprocess decoder was 79%. It is real here because this path never touches the decoder. Two
    profiles, two different answers, and neither was the guess.
    """
    from . import cache as g17cache
    return g17cache.source_text(tag)


def written_of(tag):
    """Which bindings the kernel writes, read from its Metal source.

    A MEASUREMENT STANDING IN FOR A BACKEND, exactly as source_signature is: a compiler knows
    which bindings it stored to, and Metal's own signature language spells it access::read or
    access::write. The metadata carries one small record per buffer WRITTEN - sw-c_f2h writes the
    float buffer and its table 6 has slot 1 with tlen 12, sw-ad_big writes the uint buffer and
    table 6 lacks slot 1 with tlen 8, sw-base writes all three and has three such records.

    AN ATOMIC LOAD IS NOT A WRITE, and treating it as one was the whole of a class the model could
    not decide. The atomic test used to match any atomic call on a bound buffer, which
    atomic_load_explicit as readily as atomic_store_explicit, so a kernel that only LOADS from a
    buffer was recorded as writing it. Apple disagrees by exactly one record: over the atm-i and
    atm-u groups, load kernels carry a 384-byte section and cmpxchg, exchange and store carry 392
    - eight bytes, which is one small per-buffer record. Those two groups held 12 of the 16
    refusals that needed a rule, and the rule turned out to be a correction to an input the
    contract already had rather than an input it lacked.
    """
    src = source_text(tag)
    if not src:
        return ()
    m = _re.search(r"\b(kernel|vertex|fragment)\s+[\w:]+\s+(\w+)\s*\(", src)
    if not m:
        return ()
    i = src.index("(", m.end() - 1)
    depth, params, body = 0, "", src
    for j in range(i, len(src)):
        if src[j] == "(":
            depth += 1
        elif src[j] == ")":
            depth -= 1
            if depth == 0:
                params, body = src[i + 1:j], src[j + 1:]
                break
    # A TENSOR OP'S DESTINATION IS A WRITE THAT NO SYNTAX SHOWS. cv-C16O32W8H8K1 writes `out`
    # through `op.run(tA, tW, tC)` and there is no `out[...] =` anywhere in it, so this function
    # returned () - while its section is structurally identical to a generated tensor kernel that
    # DOES write explicitly and reports (2,). Same defect class as the atomic load, inverted:
    # there the syntax claimed a write that was not one, here a write happens with no syntax at
    # all. The destination is the last argument of the .run(), and the tensor names its buffer in
    # its own constructor.
    dest = set()
    runs = _re.findall(r"\.\s*run\s*\(([^)]*)\)", body)
    for args in runs:
        parts = [a.strip() for a in args.split(",") if a.strip()]
        if parts:
            dest.add(parts[-1])
    tbuf = {}
    for m in _re.finditer(r"tensor\s*<[^;]*>\s*(\w+)\s*\(\s*(\w+)", body):
        tbuf[m.group(1)] = m.group(2)
    # AND THROUGH A SLICE, one indirection further. rep1_3, lever_simdgroups_2 and cbt-16x16 run
    # matmul2d on `auto sC = tC.slice<32,32>(0,0)`, so the destination names a slice, the slice
    # names the tensor and the tensor names the buffer. Following only the last hop left all three
    # reporting that they write NOTHING, which is the same blind spot the tensor case already
    # fixed once - and it kept them unwitnessable, because a sibling that appends a store to a
    # buffer the model thinks is unwritten changes the write set and lands on another key.
    for m in _re.finditer(r"\b(\w+)\s*=\s*(\w+)\s*\.\s*slice\s*<", body):
        if m.group(2) in tbuf:
            tbuf[m.group(1)] = tbuf[m.group(2)]
    written_via_tensor = {tbuf[t] for t in dest if t in tbuf}
    # AND A SIMDGROUP MATRIX STORE IS A WRITE, which is the same blind spot a third time: no
    # `name[...] =` anywhere, so 111 kernels across mm-, mr- and ms- reported writing NOTHING.
    # They were byte-exact anyway, and that is the point - every one of them was understated in
    # the same direction, so they grouped together and transplanted from each other happily. A
    # wrong input agrees with a donor that shares it.
    #
    # Apple disagrees, and says so cleanly: over the 108 with an identifiable destination, table 6
    # slot 1 is clear for all 33 that store to binding 0 and set for all 75 that store anywhere
    # else, with no exceptions and no dependence on the element type (float appears on both sides
    # of that split). The destination is simdgroup_store's SECOND argument.
    written_via_simdgroup = set(_re.findall(r"simdgroup_store\s*\(\s*\w+\s*,\s*(\w+)", body))

    out = []
    for _ty, nm, num in ARG_RE.findall(params):
        if _re.search(r"\b%s\s*\[[^\]]*\]\s*=(?!=)" % _re.escape(nm), body) or \
           _re.search(r"atomic_(?!load_)\w+\s*\(\s*&?\s*%s\s*\[" % _re.escape(nm), body) or \
           nm in written_via_tensor or nm in written_via_simdgroup:
            out.append(int(num))
    return tuple(sorted(out))


def threadgroup_bytes(md):
    """Slot 28 of the per-kernel table: the static threadgroup memory the kernel ALLOCATES.

    Proven against the corpus rather than assumed - 1024, 2048 and 4096 for one, two and four
    `threadgroup uint t[256]` arrays, and 2048 against 4096 for mm-f16.tg and mm-f32.tg, which
    declare the same float[1024] and half[1024] and use different ones. A backend reports it the
    way Metal reports staticThreadgroupMemoryLength.
    """
    from . import gpumd as GM
    try:
        return GM.fields(bytes(md)).get(28, 0) or 0
    except Exception:
        return 0


def binding_indices(md):
    """The buffer index each binding record names, in order - a contract input, not class state."""
    try:
        desc = M.describe(bytes(md))
    except Exception:
        return None
    if not desc["vectors"]:
        return None
    last = sorted(desc["vectors"])[-1]
    out = []
    for r in desc["vectors"][last]:
        t = desc["tables"].get(r)
        if t is None:
            return None
        sl = {int(k): v for k, v in (t.get("slots") or {}).items()}
        out.append(struct.unpack_from("<I", bytes(md), r + sl[1])[0] if 1 in sl else 0)
    return out


def backend_scalars(md):
    """(temporary_register_count, spill_buffer_bytes, thread_invariant_spill_buffer_bytes).

    Read here from the section, which stands in for a backend the way source_signature does: a
    compiler allocated the registers and sized the spill buffer, so both are things it reports.
    """
    from . import gpumd as GM
    try:
        f = GM.fields(bytes(md))
    except Exception:
        return None, None
    return f.get(0), f.get(1), f.get(31)


def _arg_types(tag):
    """{buffer index: element type} from the kernel's own signature."""
    src = source_text(tag)
    if not src:
        return {}
    return {int(n): t for t, _nm, n in ARG_RE2.findall(src)}


ARG_RE2 = _re.compile(r"(?:device|constant)\s+([\w:]+)\s*\*\s*(\w+)\s*\[\[\s*buffer\((\d+)\)")


def barrier_kinds(tag):
    """Which barriers the kernel executes, as (scope, memory-flag) pairs.

    ms-barrier.tg and ms-barrier.dev share every other contract fact and have different shapes;
    they differ in mem_threadgroup against mem_device. The schema has has_threadgroup_barrier, so
    the distinction is a field the format carries and a backend knows which it emitted.
    """
    src = source_text(tag)
    if not src:
        return ()
    out = set()
    for m in _re.finditer(r"(threadgroup_barrier|simdgroup_barrier)\s*\(([^)]*)\)", src):
        for f in _re.findall(r"mem_\w+", m.group(2)):
            out.add(m.group(1)[:4] + ":" + f)
    return tuple(sorted(out))


def signed_stores(tag):
    """Which bound buffers receive a value of SIGNED integer origin, read from the source.

    THE COUNTEREXAMPLE EARNED THIS AND NOTHING ELSE DOES. The endpoint forbids an instruction-level
    dependence "unless a counterexample proves one necessary", and item 4 forbids reaching for one
    "to keep the score moving". So the order matters: three hypotheses were refuted by compiled
    kernels first - a signed 32-bit operation, an as_type reinterpret, g5's exact construction -
    and then a controlled sweep with three uint* buffers, one signed value and one unsigned,
    measured what the bit actually depends on:

                   signed   unsigned
        buffer 0   clear    clear
        buffer 1   SET      clear
        buffer 2   SET      clear

    Two factors, jointly. One of them - which binding is written - written_of already supplies.
    The other is whether the stored value is signed, and `device uint *f` has no signedness, so no
    property of the signature carries it.

    AND IT IS NOT THE INSTRUCTION STREAM. This reads s.metal, exactly as written_of, read_of and
    barrier_kinds have always done; the contract has carried source-derived body facts since it
    was written. What is new is that this one is about a VALUE rather than a binding or a barrier,
    which is why it took a counterexample to justify.

    THIRTY-TWO BITS AND WIDER, because 16-bit signed does not set the bit. The sgn sweep measured
    it directly - `h >> 2` on a short is clear where `x >> 3` on an int is set - and g5_sar_s16
    confirms it from the corpus: it casts through (short) and stays clear while its sibling
    g5_sar_s32 casts through int and sets. A first cut that counted `short` marked them alike.

    Deliberately coarse otherwise: a buffer is marked when a signed 32-bit local, an as_type to a
    signed 32-bit type, or a signed 32-bit cast reaches a store to it. Anything subtler would be
    fitting to the corpus rather than stating a fact a compiler knows.
    """
    src = source_text(tag)
    if not src:
        return ()
    m = _re.search(r"\b(kernel|vertex|fragment)\s+[\w:]+\s+(\w+)\s*\(", src)
    if not m:
        return ()
    i = src.index("(", m.end() - 1)
    depth, params, body = 0, "", src
    for j in range(i, len(src)):
        if src[j] == "(":
            depth += 1
        elif src[j] == ")":
            depth -= 1
            if depth == 0:
                params, body = src[i + 1:j], src[j + 1:]
                break
    signed = set()
    for mm in _re.finditer(r"\b(?:int|long)\d*\s+(\w+)\s*=", body):
        signed.add(mm.group(1))
    # THE STORE'S TARGET AND ITS RIGHT-HAND SIDE, not any line the name appears on. A first cut
    # scanned whole lines and marked a buffer that was merely READ beside a signed local, which
    # made sw-c_s2f and sw-c_u2f identical - the very pair this exists to separate.
    out = []
    for _ty, nm, num in ARG_RE.findall(params):
        for mm in _re.finditer(r"\b%s\s*\[[^\]]*\]\s*=(?!=)([^;]*);" % _re.escape(nm), body):
            rhs = mm.group(1)
            # A MEMBER NAME IS NOT A LOCAL. `\bx\b` matches the x in `tg.x`, because a dot is a
            # word boundary - so every kernel that declares `int x` and stores an expression
            # mentioning tg.x was marked as storing a signed value. That is the whole of the last
            # counterexample: sw-sc-cvt1 stores float(u[tg.x + 0u] + 1u), which has no signed
            # operand anywhere, and sat in a group with sgn-cvtf-s, which stores float(x) and
            # genuinely does. Third calibration of this detector and the third of the same kind -
            # each one was the pattern matching something adjacent to what it meant.
            if any(_re.search(r"(?<![\w.])%s\b" % _re.escape(v), rhs) for v in signed) or \
               _re.search(r"as_type\s*<\s*(?:int|long)\d*\s*>", rhs) or \
               _re.search(r"\(\s*(?:int|long)\d*\s*\)", rhs):
                out.append(int(num))
                break
    return tuple(sorted(set(out)))


# THE KEY DESCRIBES ITSELF, BECAUSE FIVE READERS DID NOT.
#
# build_env, key_for, the tail-form proxy, knobs_from's builtin check and g17synth's signature
# branch all went stale in the same way: each held a copy of the key's LAYOUT by index or by
# assumed shape, the layout moved, and nothing raised. A stale reader does not fail, it answers -
# "12/11 matched, missing nothing", "source not expressible by the generator", "77 of 89 with
# twelve exceptions", every one of them a sentence about the model that was really a sentence
# about a copy of a definition.
#
# Key is a tuple subclass, so every existing index, comparison, hash and pickle keeps working and
# this change cannot break a consumer. What it adds is names. A reader that says key.builtins
# cannot go stale when a component is inserted before it; a reader that says key[5] can, and did.
#
# FIELDS is the single definition of the order. contract_key builds from it, NAMES elsewhere is
# derived from it, and a component added in one place is added everywhere.
KEY_FIELDS = ("sig", "cp_len", "n", "size_class", "loop", "builtins", "threadgroup",
              "records_bindings",
              "written", "symbol", "barrier", "signed", "bound", "resources")

SIG_FIELDS = ("const", "declared", "types", "kinds")


class Sig(tuple):
    """The signature as it appears IN THE KEY - (const, declared, types, kinds).

    Not what source_signature returns. contract_key drops the bound count and the first-at-zero
    flag because bound_indices makes both redundant, and a reader that unpacks the six-tuple here
    gets const where it expects a count. That is precisely what happened to g17synth.
    """
    __slots__ = ()

    def __getattr__(self, name):
        try:
            i = SIG_FIELDS.index(name)
        except ValueError:
            raise AttributeError(name)
        return self[i] if i < len(self) else None


class Key(tuple):
    """A contract key that can be read by name."""
    __slots__ = ()

    def __getattr__(self, name):
        try:
            i = KEY_FIELDS.index(name)
        except ValueError:
            raise AttributeError(name)
        return self[i] if i < len(self) else None

    def named(self):
        """{field: value} for everything this key actually carries - for diffing and printing."""
        return {KEY_FIELDS[i]: self[i] for i in range(min(len(self), len(KEY_FIELDS)))}

    def differs(self, other):
        """The NAMES of the components that differ, so no caller has to zip indices itself."""
        n = min(len(self), len(other), len(KEY_FIELDS))
        return [KEY_FIELDS[i] for i in range(n) if self[i] != other[i]]

    @property
    def builtin_ids(self):
        """The builtin vector flattened, since it is a tuple of counted lists."""
        bv = self.builtins or ()
        return [x for g in bv for x in (g if isinstance(g, tuple) else (g,))]


def contract_key(sig, cp, n, insts, loop, builtins, threadgroup=0, written=None, symbol=None,
                 barrier=None, signed=None, read=None, bound=None, resources=None,
                 records_bindings=None):
    """The key a linker computes from what the backend hands over.

    THE CONSTANT PROGRAM ENTERS BY LENGTH, NOT BY BYTES. The donor supplies the SHAPE and nothing
    else - every value in the section is a contract input - so keying on the payload's exact bytes
    is stricter than the shape needs, and strictness costs witnesses. Measured: bytes 80.5%,
    length 83.4%, and length with the size class in place of the raw instruction count 87.5%,
    with nothing built wrong in any of the three.
    """
    # THREADGROUP MEMORY IS WORTH +31 KERNELS, AND NOT WHERE I EXPECTED. I put it in expecting it
    # to lift the threadgroup-memory row of the coverage matrix, which the gate builds worst of
    # all; that row goes 6 of 51 to 9 of 51 and the other 28 come from elsewhere. It stays because
    # slot 28 is a proven contract field and nothing is built wrong with it in the key, not
    # because it fixed the row it was aimed at.
    # THE ACCESS HALF OF A BINDING, appended only when the caller states it. The metadata carries
    # one small record per buffer the kernel WRITES - sw-c_f2h writes the float buffer and its
    # table 6 has slot 1 with tlen 12, sw-ad_big writes the uint buffer and table 6 lacks slot 1
    # with tlen 8, sw-base writes all three and has three such records - and supplying it takes
    # the gate from 6,649 byte-exact to 7,001 with 912 refusals down to 560. Optional so that a
    # caller who cannot say which bindings are written gets the key without it rather than one
    # with a field guessed.
    # THE SYMBOL WAS A CONTRACT INPUT THAT WAS NOT IN THE KEY, which is an inconsistency rather
    # than a discovery: from_contract has been WRITING it since the .cfg kernels were found, and
    # the key never carried it, so mx-i64.div-2 and mod-2 sat in a group with kernels whose
    # constant program is named differently. Adding it, and the barrier kind, takes the
    # disagreeing kernels from 318 to 284.
    # AND TWO MORE FALL OUT FOR FREE. Once the bound indices are in the key, sig's first element
    # (how many buffers are bound) is len(bound) and its second (whether the first is at zero) is
    # bound[0] == 0. Both are functions of a component already present, so carrying them is not
    # extra strictness, it is the same fact three times. They are dropped rather than kept for
    # symmetry: a key with redundant elements is one that cannot be reasoned about.
    # WHETHER THE METADATA RECORDS ANY BINDING AT ALL, which is not the same fact as `bound`.
    # `bound` is what the SOURCE declares; this is what the compiler EMITTED. Over 12,009 corpus
    # kernels the two disagree in 1,064 (8.86%) and in both directions - a4-cmpx-uni declares one
    # buffer and its section records two, because one is an internal binding the source never
    # names; ab_from_alu declares two and records one, because the other was eliminated.
    #
    # THE SPLIT IS CLEAN AND IT IS STRUCTURAL. 22 corpus kernels record NO binding and every one
    # of them has exactly FOUR tables; all 11,987 that record a binding have six or more. A
    # section with no bindings has no binding vector and no binding records, so it cannot carry
    # the same table count as one that does - this is a shape difference, not a tuning parameter.
    #
    # It was found because uf4-tgmem-read, a kernel the ISA peer built to separate threadgroup
    # memory from a barrier, landed on the same key as 68 cb-* kernels with a different table
    # count and broke "the contract key determines the table count". The cause was not threadgroup
    # memory at all: that kernel declares a buffer it never uses, so the compiler recorded no
    # binding, and the key was reading the DECLARATION. Which is this project's own rule -
    # declaring a thing is never what gets recorded - appearing inside the key that is supposed to
    # encode it.
    sigk = Sig(tuple(sig)[2:]) if bound is not None else Sig(tuple(sig))
    base = (sigk, len(cp or b""), n, size_class(insts, loop), loop, builtins, threadgroup)
    if records_bindings is not None:
        base += (bool(records_bindings),)
    if written is not None:
        base += (tuple(written),)
    if symbol is not None:
        base += (bytes(symbol),)
    if barrier is not None:
        base += (tuple(barrier),)
    # THE ONE FACT A COUNTEREXAMPLE EARNED. See signed_stores: the bit in the per-kernel table's
    # sixth entry depends jointly on which binding is written - already here, as `written` - and on
    # whether the value stored to it is a signed 32-bit integer, which no property of a signature
    # carries. Three hypotheses were refuted by compiled kernels before this was added, and the
    # endpoint's own clause permits an instruction-level dependence exactly when a counterexample
    # proves one necessary. Optional, so a caller that cannot say gets the key without it.
    if signed is not None:
        base += (tuple(signed),)
    # THE READ SET IS A CONTRACT INPUT AND IT WAS ONLY A TIE-BREAK. read_of has existed since the
    # refinements were written and never entered the key, so two kernels reading different
    # buffers shared a class - and in sample that never showed, because the corpus happens not to
    # contain such a pair on the same key. The synthesizer built one and the linker built three
    # WRONG sections: hold-hole1 reads binding 1 and its donor reads 5 and 14, hold-b-readonly1
    # reads 9 against a donor reading 1 and 2, hold-hole2 reads 2 against a donor reading 6, 9
    # and 15. The sections differ by 4 and 16 bytes - one small record per buffer READ, which is
    # exactly the shape of the per-buffer record `written` already carries.
    #
    # Same class of correction as "an atomic load is not a write" and "a simdgroup store is a
    # write", and found the same way: by a kernel the model had never seen.
    # AND THE COUNT IS NOT ENOUGH, WHICH WAS TRIED AND REFUTED. Controlled pairs said it should be:
    # with bound-ness held fixed - every buffer written in both members, one additionally read -
    # adding a read costs exactly four bytes and changes no table's slot set or body length, and
    # it is the same four bytes whichever binding is read. Measured at two, three and four
    # bindings: 380->384, 440->444, 420->424, every binding, no exceptions.
    #
    # Keying on len(read) instead of the set then built ONE held-out kernel wrong. The pairs held
    # EVERY buffer written, which is a regime and not the general case, and outside it which
    # binding is read matters. A narrowing that breaks the invariant is refuted however clean the
    # experiment that suggested it looked - that is what the invariant is for, and it is the
    # second time tonight that a controlled pair produced a rule the corpus would not have.
    # AND THE READ SET IS INERT. Controlled properly - every buffer WRITTEN so none can unbind,
    # and the builtin consumed by the stored value so it cannot vanish with the last read - adding
    # a read to any binding changes the section by ZERO bytes. Byte-identical at two bindings and
    # at three, no first differing byte at all.
    #
    # Every apparent effect was a confound, and there were three of them: dropping a buffer's only
    # access UNBINDS it and moves the signature; dropping the last read removes the BUILTIN and
    # the per-kernel tail is 8 + 4*len(builtins); and the one field that moved with the read set
    # is per-kernel slot 0, which is the register count with 0 exceptions in 12,580. Three
    # different quantities wearing the read set's clothes.
    #
    # BUT NOT ENTIRELY INERT, and one kernel says exactly how. hold-hole1 binds buffers 1 and 5,
    # reads 1 and writes 5, so its written binding is WRITE-ONLY. Its donor binds 5 and 14, reads
    # both and writes 5, so its written binding is READ-AND-WRITTEN. Identical signatures,
    # identical write sets, and Apple's sections differ by four bytes.
    #
    # So what the key needs is not which buffers are read - that was refuted above, byte for byte
    # - but the ACCESS MODE of the ones that are written. read & written is that, it is derivable
    # from two contract inputs the model already has, and it is far coarser than the read set:
    # kernels reading different read-only buffers now share a class where before every distinct
    # read set was its own.
    # AND `read` IS GONE. It entered the key because three kernels built wrong, and every one of
    # those was a pair whose READ BINDING'S INDEX differed - hold-hole1 reads 1 against a donor
    # reading 5 and 14, hold-b-readonly1 reads 9 against a donor reading 1. The read set separated
    # them only because it happened to differ wherever the index did. With the indices in the key
    # the proxy is redundant, and the properly controlled experiment already said so byte for
    # byte: hold bound-ness and the builtin fixed, and adding a read to any binding changes
    # NOTHING. read_of stays a REFINEMENT, where it costs no class and can still break a tie.
    if bound is not None:
        base += (tuple(bound),)
    # THE RESOURCE COUNT, ADDED DELIBERATELY IN EPOCH 3. A record with field 0 == 5 and a field 4 -
    # a texture, sampler or acceleration structure rather than a buffer - appears in 37.2% of Apple
    # sections and 5 of 11,895 here. g17facts.resource_records has measured it for a long time and
    # its docstring says why it was left out: every number in this project is denominated in this
    # key, so moving it is a decision rather than a side effect. This is that decision.
    #
    # It is what the record-table derivation ran out of. The slot-2 vector is a [6,3] spine, an
    # optional record that tracks slot 9 exactly, and N records of kind 5 - and N is the resource
    # count, which no other contract fact predicts: the best of slot 38, ti-spill and the binding
    # count reaches 4.5% and is WRONG 80 times. A component the corpus cannot teach, which is
    # exactly the shape a second population exists to find.
    #
    # Optional, like `written`, `signed` and `bound` before it: a caller that cannot say gets the
    # key without it rather than one with a field guessed.
    if resources is not None:
        base += (int(resources),)
    return Key(base)


class Ambiguous(Exception):
    """The contract does not determine the class. A linker must refuse rather than guess."""


class Unsupported(Exception):
    """Nothing in the corpus shares this contract, so nothing is known about it."""


_SIGTAB = None
_INSTS = {}
_FACTS = {}
# Order matters only in that select() consults them in this order; each level is kept only where
# it makes its group agree, so a later level never overrides an earlier one that worked.
REFINEMENTS = ("insts", "regs", "spill", "ti", "read")


def _stamp():
    """What the cached class tables are a function of. Delegates to g17cache.

    THIS USED TO BE THE CORPUS SIZE AND THIS FILE'S MTIME, and that was still not enough:
    source_signature lives in g17classbytes and is half the key, so editing it changed every key
    in the corpus and invalidated nothing. g17cache.stamp() hashes every module that can change
    what a key MEANS, which is the only version of this that is not a partial guess.
    """
    from . import cache as g17cache
    return g17cache.stamp()


def _stamp_old():
    """The previous stamp, kept only so the reason it was insufficient stays legible.

    THE CORPUS SIZE WAS THE WHOLE STAMP, and that made every change to a contract input silently
    invisible. written_of, signed_stores, barrier_kinds and contract_inputs all live here; editing
    one of them changes the key of every kernel in the cache and changes `len(os.listdir())` not at
    all, so the next run reads back a table keyed the old way and reports a score for a model that
    no longer exists. This is the third form of the same defect - a donor table cached with no
    invalidation was found once, then invalidated on the corpus only, which is invalidation on the
    half that was never the problem.
    """
    src = os.path.abspath(__file__)
    st = os.stat(src)
    return (len(os.listdir(CB.g17metal.CACHE)), int(st.st_mtime), st.st_size)


def skeletonise(desc, symbol=None):
    """Strip a description to the structure the builder cannot recompute, and drop the rest.

    THE CLASS TABLE STOPS STORING APPLE'S SECTIONS. Everything the builder now computes - every
    field value, every field offset, every slot offset map, every vtable length, and every tail
    byte after the constant program's symbol - is content, and content that gets overwritten on
    the way out. Keeping it meant the linker's table was a catalogue of Apple descriptions when
    what it needs is a catalogue of shapes.

    What survives is structural: the table order and positions, each table's body length, its slot
    SET and field WIDTHS (the layout lookup is keyed on both), the vectors, and the entry list at
    the head of the constant-program tail, which nothing yet derives.

    Measured before it was wired in: with exactly this much kept and everything else destroyed,
    493 of 493 sampled kernels build byte-identical to the control.
    """
    import copy
    out = copy.deepcopy(desc)
    sym = bytes(symbol or b"")
    pk = None
    for p2 in table_index(out):
        if {26, 27, 29} <= {int(k) for k in (out["tables"][p2].get("slots") or {})}:
            pk = p2
            break
    for pos in table_index(out):
        t = out["tables"][pos]
        f = {int(k): v for k, v in (t.get("fields") or {}).items()}
        t["fields"] = {k: (0, f[k][1], 0) for k in f}
        t["slots"] = {k: 0 for k in f}
        t["vlen"] = 0
        real = bytes(t.get("tail") or b"")
        if not real:
            continue
        if pos == pk:
            t["tail"] = b"\0" * len(real)          # rebuilt wholesale from the builtin lists
        elif sym and sym in real:
            # THE SYMBOL STAYS, because rebuild_tails locates this tail by finding it - zeroing it
            # left every constant-program tail unrebuilt and took the whole corpus to 0 of 493.
            # It is a contract input, so keeping it costs nothing in provenance; what is dropped
            # is the payload after it, which the contract supplies.
            i = real.find(sym)
            t["tail"] = real[:i] + sym + b"\0" * (len(real) - i - len(sym))
    return out


def signature_table():
    """{contract key: [(tag, donor description)]} over the cache, one entry per key.

    A key whose members disagree about the shape is dropped: the contract does not decide it and
    a donor picked from it would be a guess. A key with a single member is dropped too - its only
    witness could be the kernel being built, and a class of one is not a class.
    """
    global _SIGTAB
    if _SIGTAB is not None:
        return _SIGTAB
    # CACHED ON DISK. Building this walks every cached object and every metadata section, which is
    # a minute; corpuspack runs one PROCESS per kernel, so without a cache the table is rebuilt
    # thousands of times and the harness measures the table instead of the image.
    import collections, pickle
    from . import obj as g17obj
    # INVALIDATED BY THE ENTRY COUNT, which is one syscall rather than 8,000 stats. Not by mtime:
    # stat-ing every cache entry costs more than the walk it is meant to avoid, and the harness
    # writes its own temp images under that directory so the newest mtime is never stable.
    #
    # IT USED TO HAVE NO INVALIDATION AT ALL, and that cost a whole evening's numbers. Every
    # measurement taken after a corpus change - 720 threadgroup kernels, 1,311 siblings, 380
    # atomics - used a donor table that predated them, so the new kernels had no possible donor
    # and were reported as "no cached kernel shares this contract". The refusal counts were
    # inflated and the conclusions drawn from them ("the atomics row is thin", "manufactured
    # witnesses refuse at 59%") were drawn about a table, not about the model. REBUILD_CLASSES=1
    # was the documented escape hatch and a documented escape hatch that must be remembered on
    # every run is a defect with a manual in front of it.
    from . import cache as g17cache
    got = g17cache.load("contract-classes")
    if got is not None:
        _SIGTAB = got
        return _SIGTAB
    byk = collections.defaultdict(list)
    for d in sorted(os.listdir(CB.g17metal.CACHE)):
        if not in_corpus(d):
            continue
        sig = CB.source_signature(d)
        md = CB.metadata(d) if sig else None
        obj = os.path.join(CB.g17metal.CACHE, d, "out", "object", "0-0")
        if not md or not os.path.exists(obj):
            continue
        try:
            desc = M.describe(bytes(md))
            raw = open(obj, "rb").read()
            sects, syms = g17obj.sections_of(raw)
            off, size = sects["__TEXT,__text"]
            ni, loop = code_facts(bytes(raw[off:off + size]), syms["_agc.main"])
        except Exception:
            continue
        cp, _c, _t, sym = contract_inputs(desc)
        # AND signed_stores HERE TOO, because signature_table builds the keys build_for looks up.
        # Adding the component on one side only made every lookup miss: 11,484 kernels refused,
        # 0 byte-exact, on the first run after the change. The same call-site drift that produced
        # three different "linker paths" earlier, in the one place the earlier fix did not reach -
        # and it announced itself instantly, because a key that no longer matches anything fails
        # loudly rather than quietly.
        # AND FOR THE THIRD TIME, key_for. This site kept assembling the key by hand even after
        # key_for was written to be the only definition, because it lives in the same file and
        # looked like the definition rather than a copy of it. Adding `read` on the build side and
        # not here made every lookup miss again - 12,107 refused, 0 byte-exact - which is the
        # identical failure this comment block already described happening for `signed`. A key
        # assembled by hand is a copy of a definition living elsewhere, wherever it lives.
        key = key_for(d, md, desc, sig, ni, loop)
        _INSTS[d] = ni
        regs, spill, ti = backend_scalars(md)
        _FACTS[d] = dict(insts=ni, regs=regs, spill=spill, ti=ti, read=read_of(d))
        byk[key].append((d, skeletonise(trim_tails(desc), sym)))
    # A SECOND LEVEL, USED ONLY WHERE THE FIRST DISAGREES. The goal's instruction is to group by
    # the established key and then compare candidates INSIDE the group, and that is what this is:
    # the coarse key stands wherever it decides the shape, and where it does not, the raw
    # instruction count is added - which resolves 262 of the 284 kernels still in dispute.
    #
    # Adding the raw count to the key OUTRIGHT was measured and is worse: 80.5% against 87.5%,
    # because a count is nearly unique per kernel and strands classes with no second witness. The
    # same dimension helps as a tie-break and hurts as a key, which is the whole reason to refine
    # conditionally rather than globally.
    # AND FOUR MORE OF THEM, EACH A NAMED BACKEND FACT. Scored conditionally - inside the groups
    # that still disagree, never globally - the register and spill figures and the read set carry
    # most of what the instruction count does not:
    #
    #     spill_buffer_bytes                    9 of the 29 disagreeing keys
    #     the raw instruction count             7
    #     temporary_register_count              7
    #     which bindings the kernel READS       3
    #     thread_invariant_spill_buffer_bytes   1
    #
    # leaving two keys and seventeen kernels that no single quantity separates. Every one of the
    # five is something a compiler reports about what it emitted, of the same standing as the
    # instruction count this layer already accepts, and every one is measured here as a TIE-BREAK
    # rather than as part of the key - the raw instruction count was worse as a key than as a
    # tie-break (80.5% against 87.5%) because a near-unique quantity strands classes with no
    # second witness, and the register figures are near-unique in the same way.
    _SIGTAB = {}
    for key, ms in byk.items():
        if len(ms) > 1 and len({shape_of(x) for _t, x in ms}) == 1:
            _SIGTAB[key] = ms
            continue
        if len(ms) < 2:
            continue
        for name in REFINEMENTS:
            sub = collections.defaultdict(list)
            for tag, desc in ms:
                sub[(_FACTS.get(tag) or {}).get(name)].append((tag, desc))
            for v, group in sub.items():
                if v is None or len(group) < 2:
                    continue
                if len({shape_of(x) for _t, x in group}) == 1:
                    _SIGTAB.setdefault(key + (name, v), group)
    with g17cache._Lock("contract-classes"):
        g17cache.save("contract-classes", _SIGTAB)
    return _SIGTAB


# THE SKELETON FALLBACK, for keys with one member.
#
# A key with a single member is refused because its only donor would be the kernel being built.
# But a DONOR is a whole section and a SKELETON is only its structure - the vlen/tlen/tail/slot
# sequence, the block order and the gaps - and there are 303 distinct skeletons across 734 keys,
# so a lone kernel's structure is usually witnessed by kernels in other keys. Measured: of the 242
# single-member keys, 153 have a skeleton another kernel also has and 89 do not.
#
# The 89 stay refused. Structure from somebody else and values from the contract is not a guess;
# structure from yourself is not an answer.
_SKEL = None


def skeleton_table():
    """{(shape, gaps): [(tag, description)]} over the cache."""
    global _SKEL
    if _SKEL is not None:
        return _SKEL
    import collections, pickle
    from . import obj as g17obj
    from . import cache as g17cache
    got = g17cache.load("skeletons")
    if got is not None:
        _SKEL = got
        return _SKEL
    out = collections.defaultdict(list)
    for d in sorted(os.listdir(CB.g17metal.CACHE)):
        if not in_corpus(d):
            continue
        md = CB.metadata(d)
        if not md:
            continue
        try:
            desc = M.describe(bytes(md))
        except Exception:
            continue
        out[(shape_of(desc), tuple(gaps_of(desc)))].append((d, trim_tails(desc)))
    _SKEL = dict(out)
    with g17cache._Lock("skeletons"):
        g17cache.save("skeletons", _SKEL)
    return _SKEL


def select_by_skeleton(desc, exclude=None):
    """A donor with the same STRUCTURE, from any contract key, or None.

    NOT A LINKER PATH, and it must never be wired into one. It takes `desc` - the target's OWN
    described metadata - and looks the donor up by shape_of(desc), which means reading Apple's
    section for the kernel being built to decide what structure to give it. That is exactly the
    criticism this project already levelled at the class table: the build score is measured with
    the target's own shape in hand, and a linker has the signature and not the shape.

    Scored that way it reports 7,456 of 7,561 byte-exact with 105 refused and 0 wrong, against
    7,001 and 560 for the contract path. The improvement is real and it is a measurement of the
    FORMAT - it says the structure plus the contract determines the section - not of a linker.

    The non-circular version was measured and does not work: coarsening the contract key until a
    single-member key lands beside a witness serves 21 of 249 singletons at best, and 11 with the
    signature alone. So structure-by-lookup buys 455 sections when it may consult the answer and
    21 when it may not.
    """
    ms = skeleton_table().get((shape_of(desc), tuple(gaps_of(desc))))
    if not ms:
        return None
    for tag, donor in ms:
        if tag != exclude:
            return tag, donor
    return None


def contract_facts(tag, md, insts):
    """The named backend quantities select() uses as tie-breaks, in one place.

    Kept beside select rather than inlined at each call site so that a caller cannot silently
    supply three of the five and get a coarser answer than it thinks it asked for.
    """
    regs, spill, ti = backend_scalars(md)
    return dict(insts=insts, regs=regs, spill=spill, ti=ti, read=read_of(tag))


def select(key, exclude=None, insts=None, facts=None):
    """The donor for a contract key, or an exception saying why there is none.

    Never returns a best guess: a key whose kernels disagree about the shape is refused, and so is
    one nothing in the cache shares.

    `facts` carries the named backend quantities used as tie-breaks - see REFINEMENTS. `insts` is
    the older single-refinement form and still works on its own.
    """
    tab = signature_table()
    ms = tab.get(key)
    # The refined keys are consulted only where the coarse one did not decide the shape, so a
    # kernel whose class is already determined never sees them.
    f = dict(facts or {})
    if insts is not None:
        f.setdefault("insts", insts)
    for name in REFINEMENTS:
        if ms:
            break
        if f.get(name) is not None:
            ms = tab.get(key + (name, f[name]))
    if not ms:
        raise Unsupported("no cached kernel shares this contract: %d declared, "
                          "%d-byte constant program, N=%s, size class %s, loop=%s, builtins %s, "
                          "%d bytes of threadgroup memory"
                          % (key[0][1], key[1], key[2], key[3], key[4], key[5],
                             key[6] if len(key) > 6 else 0))
    for tag, donor in ms:
        if tag != exclude:
            return tag, donor
    raise Ambiguous("the only witness for this contract is the kernel being built")


# KERNELS THE MODEL MUST NEVER LEARN FROM. g17classholdout.py compiles shapes aimed away from the
# corpus so the class model can be measured on something it has never seen - and the moment those
# kernels land in the cache they are eligible to become donors, witness each other, and enter every
# corpus count. A holdout that joins the training set stops being a holdout on its first run, and
# nothing would report that it had. So the prefix is excluded here, at the one place the class
# tables are built, rather than remembered at each of the call sites that read them.
HOLDOUT = "hold-"
# AND A THIRD CATEGORY, which the first version of this did not have. An INSTRUMENT probe - one
# kernel compiled to measure one fact, like "which builtin id does thread_index_in_simdgroup
# consume" - belongs in neither set. It is not corpus, because the model must not learn a class
# from a kernel built to answer a question; and it is not holdout, because counting instruments as
# adversarial kernels that the linker failed to build would corrupt the only number measuring
# generalisation. Fourteen builtin probes would have moved the holdout from 51 to 65 and the
# refusal rate with it, for no reason but that they exist.
PROBE = "probe-"


SYNTH = "syn-"
WITNESS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                            "isa", "g17-synth-witnesses.txt")
_witnesses = None


def witnesses():
    """The synthesized sources that actually LANDED on the key they were solving for.

    A SOLVED WITNESS IS CORPUS; A FAILED CANDIDATE IS AN INSTRUMENT. The synthesizer compiles
    around forty candidates per target and keeps at most two, but every one of them was written
    into the cache under the syn- prefix and every one of them counted as corpus. 2,116 of 14,656
    cache entries - 14% of the corpus - were failed candidates.

    That is coverage expansion wearing a different prefix, which this project settled against:
    "every point added is a new isolated class". It showed up as the refusal ledger failing with
    ten uncategorised refusals, all of them syn-, one class holding 73 generated kernels of which
    69 had ten tables and 4 had eleven.

    The list is a REPO FILE because the corpus has to stay a function of the repo. A witness that
    is not written down is not a witness.
    """
    global _witnesses
    if _witnesses is None:
        try:
            with open(WITNESS_FILE) as fh:
                _witnesses = {l.split("#")[0].strip() for l in fh if l.split("#")[0].strip()}
        except OSError:
            _witnesses = set()
    return _witnesses


def record_witness(*tags):
    """Add solved witnesses to the repo list, so the corpus stays reproducible from the repo."""
    have = set(witnesses())
    new = [t for t in tags if t and t not in have]
    if not new:
        return 0
    with open(WITNESS_FILE, "a") as fh:
        for t in new:
            fh.write(t + "\n")
    have.update(new)
    globals()["_witnesses"] = have
    return len(new)


def in_corpus(tag):
    """Is this cache entry part of the corpus the model is built from and scored against?"""
    if tag.startswith((HOLDOUT, PROBE)):
        return False
    if tag.startswith(SYNTH):
        return tag in witnesses()
    return True


def key_for(tag, md, desc, sig, ni, loop):
    """The contract key for one kernel, in one place.

    THE SAME DEFECT build_env WAS WRITTEN TO CURE, IN A SECOND PLACE, AND IT WENT ON LONGER.
    Twelve call sites computed this key by hand and six of them computed a DIFFERENT key from the
    one the linker uses: the class-bytes census and the per-kernel-vtable case omitted `symbol`,
    `barrier` and `signed`; the refusal ledger, the surface scorer and the witness generator
    omitted `signed`. Each omission is a COARSER partition than the linker's, and a coarser
    partition is not a conservative approximation of a finer one - it changes the answer in a
    direction that depends on what the measurement does with the group:

      - the census asks which bytes VARY inside a key, so bigger groups vary more and `contract`
        was credited with bytes that a finer grouping shows constant. It read high.
      - the refusal ledger asks how many kernels SHARE a refusing kernel's key, so bigger groups
        made refusals look like model gaps ("its group disagrees") when the linker had actually
        refused them for being alone. It read the categories wrong, and the categories decide
        whether the work is a witness or a rule.
      - the witness generator asks whether a manufactured sibling LANDS on its parent's key, so it
        kept variants the linker then sorted into a different class and discarded ones it would
        have accepted. That one is not a reporting error; it made the instrument choose wrong.

    So the key gets one definition, and every measurement of the linker asks the same question of
    the same partition.
    """
    cp, _c, _t, sym = contract_inputs(desc)
    from . import facts as g17facts
    try:
        res = g17facts.resource_records(md)
    except Exception:
        res = None
    try:
        _br = g17facts.binding_records(md)
    except Exception:
        _br = None
    return contract_key(sig, cp, entry_list(desc), ni, loop, builtin_vector(md),
                        threadgroup_bytes(md), written=written_of(tag), symbol=sym,
                        barrier=barrier_kinds(tag), signed=signed_stores(tag),
                        read=read_of(tag), bound=bound_indices(tag), resources=res,
                        records_bindings=(None if _br is None else bool(_br)))


def build_env(tag, desc, md, sig, cp, ni, loop):
    """The rule environment for one kernel, in one place.

    THREE CALL SITES BUILT THIS INLINE AND NONE OF THEM AGREED. linker_report passed TYPES and no
    SZ/NVEC/SUMV; the regression passed SZ/NVEC/SUMV and no TYPES; corpuspack passed no env at all
    and did not pass the tie-break facts to select() either. So the image that EXECUTED was built
    by a weaker selection than the image that was SCORED, and the image the REGRESSION scored was
    a third thing again - measured over the whole cache, 9,000 / 9,018 / 9,041 byte-exact from
    what is supposed to be one linker. A claim about bytes is only a claim about the artefact if
    every measurement of it builds the same artefact.
    """
    import hashlib
    return dict(B=sig[0], D=sig[3], CP=len(cp or b""), N=identity_count(desc),
                TG=threadgroup_bytes(md), NI=ni, SC=size_class(ni, loop), LOOP=int(loop),
                NB=sum(len(g) if isinstance(g, tuple) else 1
                       for g in (builtin_vector(md) or ())), W=len(written_of(tag)),
                BV=builtin_vector(md), BLISTS=builtin_lists(md),
                NT=len(table_index(desc)), SZ=len(md), NVEC=len(desc["vectors"]),
                SUMV=sum(desc["tables"][p]["vlen"] for p in table_index(desc)),
                WRSET=written_of(tag), TYPES=_arg_types(tag),
                CPB=hashlib.sha256(bytes(cp or b"")).hexdigest()[:12])


def build_for(tag, md, desc, sig, ni, loop, exclude=True):
    """Build one kernel's section the way the linker does. Raises Ambiguous/Unsupported to refuse.

    The single entry point every measurement of the linker path goes through, so that "byte-exact"
    means the same construction whether it is being scored, regression-tested or executed.
    """
    cp, counts, tails, sym = contract_inputs(desc)
    key = key_for(tag, md, desc, sig, ni, loop)
    _t, donor = select(key, exclude=(tag if exclude else None), insts=ni,
                       facts=contract_facts(tag, md, ni))
    regs, spill, ti = backend_scalars(md)
    out, err = from_contract(donor, constant_program=cp, counts=counts, tails=tails,
                             gaps=None, symbol=sym, bindings=binding_indices(md),
                             threadgroup=threadgroup_bytes(md), insts=ni, loop=loop,
                             entry_symbol=b"agc.main", registers=regs, spill=spill, ti_spill=ti,
                             env=build_env(tag, desc, md, sig, cp, ni, loop))
    return out, err, key


def matrix(limit=None):
    """Build every cached kernel from contract inputs over its group's donor, and score it."""
    import collections
    table, _ = class_table(limit)
    groups = collections.defaultdict(list)
    for d in sorted(os.listdir(CB.g17metal.CACHE)):
        sig = CB.source_signature(d)
        md = CB.metadata(d) if sig else None
        if not md:
            continue
        try:
            desc = M.describe(bytes(md))
        except Exception:
            continue
        groups[(tuple(sig), shape_of(desc))].append((d, bytes(md), desc))
    ok = bad = refused = 0
    byfam = collections.defaultdict(lambda: [0, 0])
    for key, members in groups.items():
        donor = table.get(key)
        if donor is None:
            refused += len(members)
            continue
        _tag, ddesc = donor
        for d, md, desc in members:
            cp, counts, tails, sym = contract_inputs(desc)
            try:
                out, err = from_contract(ddesc, constant_program=cp, counts=counts, tails=tails,
                                         gaps=gaps_of(desc), symbol=sym)
            except Exception:
                out, err = None, "raised"
            fam = CB.family_of(d) or "other"
            byfam[fam][1] += 1
            if out is not None and out == md:
                ok += 1
                byfam[fam][0] += 1
            elif out is None:
                refused += 1
            else:
                bad += 1
    return ok, bad, refused, byfam


def main(argv=None):
        sys.exit(_main())


if __name__ == "__main__":
    main()
