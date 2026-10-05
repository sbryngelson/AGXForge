#!/usr/bin/env python3
"""Contract facts read from the SECTION and the TEXT instead of from s.metal.

The contract key is defined over Metal source facts - read_of, written_of, signed_stores,
bound_indices and barrier_kinds all parse s.metal with regexes - and that is the single reason
Apple's own compiled functions cannot be scored for byte-exactness: they ship AIR and no Metal.
Every fact recovered here is one step toward a population drawn from a different imagination than
this corpus.

WHAT IS RECOVERED, AND HOW WELL, measured against the source-derived versions on the corpus where
both are known. None of these is exact, so none of them replaces its source counterpart yet; they
are readers, and the numbers are what says how far they can be trusted.

    bindings   91.7% of kernels, from the metadata      (93.5% against a source parse with its
                                                         own gap closed - on 227 kernels the
                                                         METADATA is right and the source is not)
    written    98.7% exact, and 100.0% of source-written sets are a SUBSET of the metadata's, so
               it never misses a write - it sometimes marks one the source does not make
    barriers   93.8% on presence, from the text; the gap is entirely SIMD-scope barriers, which
               op447's byte1 scope field does not name

NOT RECOVERED: the argument TYPES and KINDS that `sig` carries, and `signed`. Binding-record slot 2
is not the element type - the same type appears against slot2 absent, 2 and 4 - so whatever slot 2
is, it is not what `sig` needs.
"""
import os
import sys

# NO AMBIENT PATH MUTATION: the siblings this module reads are package modules now, and a library
# that edits sys.path on import decides what its callers can import.
from . import mdgen as M


def binding_records(md):
    r"""The BINDING vector's records, as {slot: value} dicts, or None.

    ANCHORED ON PER-KERNEL SLOT 4, which IS the binding vector - established over 26,205 sections
    where slots 2, 4, 6, 8, 10, 12 and 26 all read as length-prefixed vectors with no exception.

    THE PREVIOUS READER SEARCHED FOR IT and that is what made 882 Apple sections unkeyable. It took
    the vector whose records ALL carry field 0 == 5, so a vector holding other kinds beside them
    was rejected - and the mixing is in the slot-2 vector, not this one. Slot 4's vector is
    HOMOGENEOUS in both populations: field 0 is 5 or elided and never another kind, 21,452 of
    21,452. So no filter is applied here; filtering a homogeneous vector can only drop real
    members, and the first attempt at this did exactly that - rejecting fully-elided records, where
    kind and index are both at their defaults, and losing binding 0 in 5,395 corpus sections.

    Measured against the searching reader:

        corpus   0 regressions, 0 disagreements,   9 newly keyable
        apple    0 regressions, 857 newly keyable, 22 disagreements

    and slot 3 arbitrates the 22: slot 3 is 8 times the binding count and is ABSENT in every one of
    them, so the binding set is empty, which is what this reader says and the searching one did not.

    An earlier attempt widened the filter instead and reached 234 while regressing 9 corpus
    kernels. Reading the vector the table names is not a broader filter, it is a different question,
    and it regresses nothing.
    """
    import struct
    from . import gpumd as GM
    md = bytes(md)
    pk = GM.kernel_table(md)
    if pk is None:
        return None
    sl, _ = GM.table_at(md, pk)
    if len(sl) <= 4 or not sl[4]:
        return None
    a = pk + sl[4]
    if a + 4 > len(md):
        return None
    v = a + struct.unpack_from("<I", md, a)[0]
    if v + 4 > len(md):
        return None
    n = struct.unpack_from("<I", md, v)[0]
    if n > 4096 or v + 4 + 4 * n > len(md):
        return None
    out = []
    for k in range(n):
        p = v + 4 + 4 * k
        t = p + struct.unpack_from("<I", md, p)[0]
        if not (0 < t < len(md)):
            return None
        try:
            s2, tsz = GM.table_at(md, t)
        except Exception:
            return None
        occ = sorted({x for x in s2 if x})
        w = {occ[j]: occ[j + 1] - occ[j] for j in range(len(occ) - 1)}
        if occ:
            w[occ[-1]] = tsz - occ[-1]
        f = {}
        for i2, x in enumerate(s2):
            if not x:
                continue
            ww = w.get(x, 4)
            # THE BOUNDS CHECK MUST USE THE FIELD'S OWN WIDTH. Requiring four bytes for a
            # one-byte field silently drops any narrow field near the end of the section, and
            # a6-2d's binding record is at 388 in a 396-byte section with both its fields one byte
            # wide - so it read as {} and the kernel lost its write flag. Fifth instrument this
            # session to assume a shape instead of reading it.
            if t + x + ww > len(md):
                continue
            q = t + x
            f[i2] = (md[q] if ww == 1 else struct.unpack_from("<H", md, q)[0] if ww == 2
                     else struct.unpack_from("<I", md, q)[0])
        out.append(f)
    return out


def bound_from_md(md):
    """The buffer indices the kernel binds. Slot 1, ELIDED when zero.

    NO BINDING VECTOR IS NOT THE SAME AS NO BINDINGS, and conflating them cost 225 Apple sections.
    Per-kernel slot 3 is 8 times the number of buffer bindings, so a section with no binding vector
    AND no slot 3 has an empty binding set - a fact, not a failure. The 225 are the second-largest
    class of the 1,109 Apple sections this vocabulary could not key at all.

    "THESE KERNELS BIND NO BUFFERS" WAS THE WRONG REASON AND IS CORRECTED HERE. mi-h_h.big.add-1
    DECLARES two buffers and lands in this class: `h[400] = (half)(h[0] + 1e30h)`, whose store the
    compiler removes, leaving 68 bytes of text against 86 for the saturating variant that keeps it.
    Slot 3 and slot 16 are both absent because nothing survived, not because nothing was written in
    the source. So the empty tuple is a statement about the OPTIMIZED PROGRAM - the consumes rule
    again - and describing it as a property of the kernel's signature is the exact conflation this
    project keeps having to undo.

    None is still returned when slot 3 says there ARE bindings and no vector was found, because
    that is a reader failing rather than a kernel binding nothing.
    """
    rs = binding_records(md)
    if rs is None:
        from . import gpumd as GM
        try:
            if GM.fields(bytes(md)).get(3) is None:
                return ()
        except Exception:
            return None
        return None
    return tuple(sorted({fl.get(1, 0) for fl in rs}))


# Metal's front end refuses buffer(31) upward, so a binding index above 30 was never written by a
# programmer. The compiler uses them for its own bindings and their records carry the write flag
# whatever the kernel does.
MAX_USER_BINDING = 30


def written_from_md(md):
    """The buffer indices the kernel WRITES. Slot 3 == 1 on a record marks it, EXCLUDING indices
    Metal cannot declare.

    THE EXCLUSION IS THE CORRECTION OF 2026-09-07 AND IT MOVED A LOT. Without it this reader names
    binding 45 in 5,222 of 8,438 Apple sections, and per-kernel slot 16 - which the peer session
    named as "a device buffer is written" - says no buffer is written in every one of them:

        agreement with slot 16, before   3,167 of 8,438   37.5%   5,256 false positives
        agreement with slot 16, after    8,422 of 8,438   99.8%       0 false positives

    THE CORPUS COULD NOT HAVE FOUND IT. Only 61 corpus kernels name an index above 30 at all, so
    against the SOURCE - which is ground truth here and cannot name buffer 45 because Metal will not
    compile it - the correction is a small strict improvement:

        exact agreement with the source, before   11,705 of 11,875   98.6%
        exact agreement with the source, after    11,766 of 11,875   99.1%
        the source set stays a subset of this one, 11,875 of 11,875, both before and after

    A defect worth 0.5 points internally and 62 points out of sample is the clearest instance in
    this project of why a second population is not another holdout.
    """
    rs = binding_records(md)
    if rs is None:
        return None
    return tuple(sorted({fl.get(1, 0) for fl in rs
                         if fl.get(3) == 1 and fl.get(1, 0) <= MAX_USER_BINDING}))


def written_all_from_md(md):
    """Every binding index whose record is marked written, INCLUDING the compiler's own.

    SLOT 15 AND SLOT 16 ARE THE SAME FACT OVER TWO DIFFERENT BINDING SETS, and correcting
    written_from_md is what made the pair visible:

        slot 15 <-> ANY binding is written        corpus 11,875 / 11,875 = 100.00%
                                                  Apple   8,423 /  8,438 =  99.82%
        slot 16 <-> a USER binding is written     corpus 11,875 / 11,875 = 100.00%
                                                  Apple   8,422 /  8,438 =  99.81%

    Swapped, each collapses: slot 15 against user writes is 37.52% on Apple and slot 16 against all
    writes is 37.53%, because 5,256 Apple sections write the compiler's binding 45 and no user
    buffer at all. Those are the texture writers, which is why binding 45 was accidentally acting as
    a texture-write marker before the correction.

    THE CORPUS CANNOT DISTINGUISH THE TWO. Both readings score 100.00% there, because only 61
    corpus kernels ever write an internal binding. A pair of slots that are provably different on
    9,547 Apple sections are indistinguishable on 11,875 of this project's own, which is the same
    lesson as the imageblock read forms one level down.
    """
    rs = binding_records(md)
    return None if rs is None else tuple(sorted({fl.get(1, 0) for fl in rs if fl.get(3) == 1}))


def barrier_scopes_from_text(text):
    """The barrier SCOPES the program executes, read from op447's byte1.

    Threadgroup (0x51) and device (0x69) only. A SIMD-scope barrier is not this instruction and is
    not found here, which is the whole of the 6.2% disagreement with barrier_kinds.
    """
    from . import dis as g17dis
    from . import cover as g17cover
    out = set()
    for o, l, k in g17dis.walk(text, 0):
        if g17cover.family(text, o, l, k) == "barrier":
            out.add(text[o + 1])
    return tuple(sorted(out))


def text_of(cachedir):
    """-> (text bytes, symbol table) for a cached kernel."""
    # THE ONE IMPORT MY REWRITE MISSED. This was a bare `import machobj` while the agxdis line
    # beside it became relative, so `from agxforge.g17 import facts; facts.text_of(...)` in a fresh
    # root process raised ModuleNotFoundError before it read its argument. Root found it; my own
    # checks were import-only, and importing a module never executes a deferred import inside a
    # function - the function has to be called.
    from . import machobj
    from . import agxdis as agxdis
    loc = machobj.locate(cachedir + "/s.arc.metallib", cachedir + "/out/object/0-0")
    f, sz = agxdis.sections(loc["obj"])
    return bytes(loc["obj"][f:f + sz]), loc["syms"]


def size_class_from_md(md):
    """The recorded size class: the per-kernel table's slot 32, ABSENT MEANING ZERO.

    FlatBuffers elides a field equal to its default, which this format uses for the binding index
    too - a buffer at index 0 is written with the index field absent. Slot 32 is the same: over
    11,889 corpus kernels the field is missing on 8,868 and the size class computed from (insts,
    loop) is 0 on ALL of them, with no exceptions.

    Reading absence as "no size class" is what put 1,197 of Apple's sections in the unkeyable
    bucket. They were keyable all along.
    """
    from . import gpumd as GM
    desc = M.describe(bytes(md))
    pk = GM.kernel_table(bytes(md))
    t = desc["tables"].get(pk)
    if t is None:
        return None
    f = (t.get("fields") or {}).get(32)
    return 0 if f is None else f[2]


def loop_from_md(md):
    """Whether the program branches backward: the per-kernel table carries slot 33 iff it does.

    Exact over the corpus - 11,889 of 11,889, with slot 33 present on all 846 looping kernels and
    absent on all 11,043 that do not. So `loop` never had to be measured from the code either.
    """
    from . import gpumd as GM
    desc = M.describe(bytes(md))
    t = desc["tables"].get(GM.kernel_table(bytes(md)))
    return None if t is None else (33 in {int(k) for k in (t.get("slots") or {})})


def insts_loop_for(size_class, loop=False):
    """An (insts, loop) pair producing this size class, or None.

    size_class is 0 at <=30 instructions, 3 if the program loops, 2 above 300, 1 otherwise. Only
    class 0 is ambiguous in `loop` - the <=30 test short-circuits, so a SHORT LOOPING kernel is
    class 0 too - and slot 33 resolves it.

    ASSUMING loop=False FOR CLASS 0 BUILT FOUR APPLE SECTIONS WRONG. refitHeaderInPlaceKernel and
    initializeBestBuffer are short and they loop; the key said "class 0, no loop", selected a donor
    whose per-kernel table has no slot 33, and the section came out 28 bytes short with 122 bytes
    differing. That was the first WRONG build on a population this project did not write, and it
    was a guess standing where a readable field was.
    """
    if size_class == 0:
        return (1, bool(loop))
    return {1: (100, False), 2: (400, False), 3: (100, True)}.get(size_class)


def resource_records(md):
    """How many PROMOTED CONSTANT RANGES the section carries.

    EPOCH 6. The previous reader counted records with slot 0 == 5 AND slot 4 present, in ANY
    vector, and both halves were heuristics from before these records had names.

    "SLOT 4 PRESENT" WAS A WAY TO TELL A RESOURCE RECORD FROM A BINDING RECORD, because both carry
    kind 5. But slot 4 is the range's SOURCE offset inside its buffer, and a record slot is present
    exactly when its value is not the default - 456,969 present slots across both vectors and both
    populations, none holding the default. So requiring slot 4 drops every range whose source
    offset is zero, which is the FIRST range promoted out of each buffer. The undercount is
    systematic, never random, and never in the other direction.

    "IN ANY VECTOR" is what forced the first heuristic. The two record types are told apart by
    WHICH VECTOR they live in: promoted ranges are in the slot-2 vector, bindings in slot 4. Read
    the right vector and no field-count test is needed at all.

    WHAT THE CORRECTION MOVES, measured before it was made:

        apple    3,551 of 9,556 sections nonzero (37.2%)  ->  6,283 (65.7%);  51.5% change value
        corpus      96 of 18,693 nonzero (0.5%)           ->    189 (1.0%);    0.6% change value

    The corpus barely moves because this project's generator could not emit more than one such
    record until it was given a knob for it - summing the reads into one store folds the loads and
    the promotion disappears - so the corpus has almost nothing for the correction to correct.
    That is the same blind spot the component was added to cover, one level down.
    """
    import struct
    from . import gpumd as GM
    md = bytes(md)
    pk = GM.kernel_table(md)
    if pk is None:
        return 0
    try:
        sl, _ = GM.table_at(md, pk)
    except Exception:
        return 0
    if len(sl) <= 2 or not sl[2]:
        return 0
    a = pk + sl[2]
    if a + 4 > len(md):
        return 0
    v = a + struct.unpack_from("<I", md, a)[0]
    if v + 4 > len(md):
        return 0
    n = struct.unpack_from("<I", md, v)[0]
    if n > 4096 or v + 4 + 4 * n > len(md):
        return 0
    out = 0
    for k in range(n):
        p = v + 4 + 4 * k
        t = p + struct.unpack_from("<I", md, p)[0]
        if not (0 < t < len(md) - 1):
            continue
        try:
            s2, _tsz = GM.table_at(md, t)
        except Exception:
            continue
        # slot 0 is the kind and is one byte wide wherever it is present
        if s2 and s2[0] and t + s2[0] < len(md) and md[t + s2[0]] == 5:
            out += 1
    return out


def threadgroup_records(md):
    """The records in the vector per-kernel slot 10 points at - one per threadgroup array the
    OPTIMIZED PROGRAM actually touches - as a list of their field-2 values.

    Anchored on the per-kernel table rather than on g17mdgen.describe's vector scan, which finds
    this vector in 469 of the 1,416 Apple sections that have one. Same shape of defect as the
    binding-vector selection and the entry-vector boundary: the scan is a heuristic and the table
    slot is a fact.

    THE CONSUMES RULE, MEASURED RATHER THAN INFERRED. probe-s9-tg1of3 declares three threadgroup
    arrays and touches one; it gets one record. probe-s9-tg2of4 declares four and touches two; it
    gets two. Declaring a threadgroup array is not using it, and the metadata records use.
    """
    import struct
    from . import gpumd as GM
    md = bytes(md)
    pk = GM.kernel_table(md)
    if pk is None:
        return None
    slots, _ = GM.table_at(md, pk)
    if len(slots) <= 10 or not slots[10]:
        return None
    a = pk + slots[10]
    if a + 4 > len(md):
        return None
    v = a + struct.unpack_from("<I", md, a)[0]
    if v + 4 > len(md):
        return None
    n = struct.unpack_from("<I", md, v)[0]
    if n > 4096 or v + 4 + 4 * n > len(md):
        return None
    out = []
    for i in range(n):
        p = v + 4 + 4 * i
        t = p + struct.unpack_from("<I", md, p)[0]
        try:
            sl, tsz = GM.table_at(md, t)
        except Exception:
            return None
        f2 = 0
        if len(sl) > 2 and sl[2]:
            occ = sorted({s for s in sl if s})
            w = {occ[j]: occ[j + 1] - occ[j] for j in range(len(occ) - 1)}
            w[occ[-1]] = tsz - occ[-1]
            q, ww = t + sl[2], w.get(sl[2], 4)
            f2 = (md[q] if ww == 1 else struct.unpack_from("<H", md, q)[0] if ww == 2
                  else struct.unpack_from("<I", md, q)[0])
        out.append(f2)
    return out


def slot9_from_md(md):
    """Per-kernel slot 9, DERIVED. Not a value to predict - the size of the slot-10 vector's
    contents:

        slot 9 = 4 * (1 + max field-2 over those records)      absent when the vector is empty

    16,649 of 16,649 corpus sections and 9,556 of 9,556 Apple sections, no exceptions.

    Counting the records gives two different laws - 8K-4 on probes whose records step field 2 by
    two, 4K on Apple's, which step by one - because K is the wrong quantity. Field 2 is a running
    offset in four-byte units and slot 9 is where it ends, so the law does not care which record
    kind (43 or 93) the vector holds.

    THIS IS WHY SLOT 9 REFUSED. It was searched as a free value against contract flags and stalled
    at 658 of 4,278; it is arithmetic over the structure, and a derived slot never refuses.
    """
    recs = threadgroup_records(md)
    if not recs:
        return None
    return 4 * (1 + max(recs))
