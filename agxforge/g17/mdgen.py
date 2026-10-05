"""__GPU_METADATA for the scalar object class, GENERATED rather than carried as a blob.

tools/g17imgconst_scalar.py holds this section as 500 measured bytes with the inert ones zeroed,
and tools/g17bindenc.py writes the buffer indices into it. That left 59 non-zero bytes belonging
to Apple. Walked as FlatBuffers - the reading the peer ISA session supplied for __GPU_LD_MD, which
applies here too - all but ONE of them is structure or a value that follows from the kernel:

    root table        pointer, vtable, two slot offsets
    per-kernel table  vtable with nine slots present, and three values:
                          slot 3 = 8 * bindings          the pointer table's size
                          slot 4 = Q - 4                 exact: only the right value executes
                          slot 2 = Q - 4 + 4 * bindings
    binding records   one per bound buffer, carrying the buffer index
    a second vector   two records carrying (6, 20, 4) and (3, 4)

Q is the one number neither session can name. It is per-kernel, a multiple of 4, takes 2 to 4
distinct values inside every metadata size class, and correlates exactly with nothing: not with
the entry PC, not with the binding count, not with the text length, and not with any of the 43
readable fields in the four metadata sections other than its own copies in slots 6, 8, 10 and 12,
which are themselves inert. It is carried here by class and asserted.

THE LOADER DOES NOT REACH THE RECORDS THROUGH FLATBUFFERS. The root's slot 3 - the only pointer
that could reach the two vectors at 376 and 388 - is ZERO in this blob, zeroed by the inertness
sweep, and the binding records are still read: swapping the two buffer indices moves the store off
the buffer the kernel means to write and the answer becomes the untouched sentinel. So whatever
walks this section finds those records some other way, and the FlatBuffers reading above is a
description of the BYTES rather than of the loader's traversal.

The layout - which vtable sits where - is Apple's, reproduced. build() asserts byte-identity
against the measured blob, so the description is checked rather than asserted.
"""
import os, struct

# TWO OBJECT CLASSES, each a measured layout. A class is (section sizes, table positions); the
# free values inside one are the binding indices and Q. A third class needs its own walk.
#
# A binding record comes in three shapes, and which one a buffer gets is Apple's choice, not a
# free parameter: "long" carries four fields, "short" two, and "elided" omits the index field
# entirely - which is how buffer 0 is written, since a zero index and an absent one encode the
# same way.

SCALAR = dict(
    size=500, root=16, rvt=4, root_vlen=12, root_tlen=12,
    pk=128, pkvt=58, pk_vlen=70, pk_tlen=0,
    pk_slots={26: 12, 13: 16, 10: 24, 8: 28, 6: 32, 3: 36, 4: 40, 2: 44, 1: 48},
    pk_extra={},
    vec_bind=376, bind=[(488, 480, "short"), (464, 452, "long")],
    vec2=388, v2=[(436, 424, "long"), (412, 402, "short")],
    v2_vals=[(6, 20, 4), (3, 4, None)],
    vec0=200, v0=(220, 208),
    q=212,
)

TENSOR = dict(
    size=412, root=16, rvt=4, root_vlen=12, root_tlen=12,
    pk=152, pkvt=58, pk_vlen=94, pk_tlen=0,
    pk_slots={26: 12, 13: 16, 10: 24, 8: 28, 6: 32, 3: 36, 4: 40, 2: 44, 1: 48, 44: 55},
    pk_extra={44: ("<B", 1)},
    vec_bind=324, bind=[(404, 398, "elided"), (380, 368, "long", 18)],
    vec2=336, v2=[(356, 346, "short")],
    v2_vals=[(3, 4, None)],
    vec0=232, v0=(252, 240),
    q=136,
)

# THE VECTOR THE INERTNESS SWEEP ERASED. Every real object has THREE vectors and the two measured
# classes were built from swept blobs that had lost one: a single-record vector the per-kernel
# table's slot 26 locates, twelve bytes early. Zeroing it costs nothing when a kernel stores to a
# device buffer, which is why it survived the sweep - and it is what made a third binding record
# segfault the loader, because slot 26 was then pointing at the wrong vector. Its record is
# byte-identical in every class measured.
V0REC = (12, 20, {0: 16, 1: 15, 2: 8, 3: 4})
V0_VALS = {3: ("<I", 16), 2: ("<I", 1), 1: ("<B", 3), 0: ("<I", 8)}

# Binding-record shapes: vtable length, declared inline length, slot -> body offset.
_REC = {
    # A WRITABLE BUFFER AT POINTER OFFSET 0. It carries the index and the written flag and elides
    # the offset, which no earlier shape does: "long" has all four fields and "short" has no
    # written flag. Measured from the two-writable-buffer witness, where the first bound buffer is
    # written and sits at offset 0 (results/g17-two-writable-class-v1).
    "written": (12, 12, {0: 10, 1: 4, 3: 11}),
    "long":   (12, 16, {0: 10, 1: 4, 2: 12, 3: 11}),
    "short":  (8, 12, {0: 11, 1: 4}),
    "elided": (6, 8, {0: 7}),
    # A FOURTH SHAPE, measured from buf3 and needed by every three-binding class in the corpus:
    # three fields where "long" carries four and "short" two. Derived from the delivered record
    # rather than interpolated between the two neighbours it sits between.
    "mid":    (10, 18, {0: 11, 1: 4, 2: 12}),
    # A WRITTEN BUFFER 0. The elided shape carries no index field, so it can only name buffer 0 -
    # and when buffer 0 is the written one it gains slot 3 and nothing else: no index, no offset.
    # Measured from the four-active write-0 control; the two-writable class's "written" shape is a
    # different thing, carrying the index.
    "elided_written": (12, 8, {0: 6, 3: 7}),
}
# The second vector's records use their own shapes; "long" here carries a third value.
_V2REC = {
    "long":  (12, 16, {0: 11, 2: 12, 3: 4}),
    "short": (10, 12, {0: 7, 2: 8}),
    # THE PROMOTED-RANGE RECORD, measured from the six-buffer family (g17promotedranges). It is the
    # only slot-2 record that carries slot 1, the promoted buffer's binding index, and that is why
    # it needs a shape of its own rather than a longer `long`. Note slot 0 at body offset 15,
    # one byte, immediately before slot 2's four-byte length at 16 - the adjacency that made a
    # four-byte read of slot 0 return `kind | (length << 8)` and produced a law that could not fail.
    "promoted": (12, 20, {0: 15, 1: 8, 2: 16, 3: 4}),
}


def _put_table(buf, pos, vtpos, vlen, tlen, slots, values, shared=False):
    """One FlatBuffers table. `slots` maps slot -> body offset, `values` slot -> (fmt, value).

    A vtable normally abuts its table and the assertion below keeps that measured invariant. A
    SHARED vtable does not: the five-user-buffer class emits two binding records - identical in
    shape, differing only in field values - that reference ONE vtable, and the second table sits
    BEFORE it, so its soffset is negative. That is legal FlatBuffers and it is what Apple's
    compiler emits. `shared=True` is how a caller declares it, so the invariant still holds
    everywhere it was measured to hold.
    """
    if not shared:
        assert vtpos + vlen == pos, ("vtable at %d (len %d) must abut the table at %d"
                                     % (vtpos, vlen, pos))
    struct.pack_into("<HH", buf, vtpos, vlen, tlen)
    struct.pack_into("<i", buf, pos, pos - vtpos)
    for slot, off in slots.items():
        struct.pack_into("<H", buf, vtpos + 4 + 2 * slot, off)
    for slot, (fmt, val) in values.items():
        struct.pack_into(fmt, buf, pos + slots[slot], val)


def _put_vector(buf, pos, recs):
    """A FlatBuffers vector of table references: a count then one relative offset each."""
    struct.pack_into("<I", buf, pos, len(recs))
    for k, r in enumerate(recs):
        struct.pack_into("<I", buf, pos + 4 + 4 * k, r - (pos + 4 + 4 * k))


def build(bindings, q=None, layout=None, second=None, restore_swept=True, words=None,
          offsets=None, system_registers=None, register_count=None, pk13=None, pk27=None):
    """The whole section, from the bound buffer indices and Q.

    restore_swept=False reproduces the measured zeroed class exactly, including
    its empty optional vectors. The explicit scalar compatibility ABI uses this
    mode; the existing restored-table behavior remains the default.

    register_count is per-PROGRAM, not per-class. Slot 0 is the highest 32-bit
    register index named in _agc.main plus one - 2,022 of 2,022 corpus objects
    once the count excludes the constant program. Each class carries a witness's
    value in pk_extra, which is right for that witness and wrong for every other
    program in the class, so a caller that knows the program's own count passes
    it here and it wins over the class constant. Callers that do not pass it keep
    the class value, which is what the class-reproduction witnesses require.
    """
    # ONE MEASURED LAYOUT PER BINDING COUNT. A count with no measured layout falls through to
    # SCALAR and hits its "exactly N binding records" refusal, which is the intended answer: an
    # unmeasured count is refused rather than approximated by stretching a neighbour.
    L = layout or layout_for(bindings) or SCALAR
    # THE LAYOUT FOLLOWS THE DECLARED REGISTER COUNT, and callers must not have to know that.
    # Each register past the first adds an entry to the slot-29 vector and moves everything laid
    # out after it. Widening here rather than at the call site is what keeps a two-register
    # contract from being serialized into a one-register class, which writes the second entry
    # over the structure that follows and yields a section of the ORIGINAL size that decodes as
    # if nothing were wrong. Classes with no slot-29 vector are untouched.
    if L.get("slot29_vector") is not None and system_registers:
        L = with_system_registers(L, len(list(system_registers)),
                                  measured_counts=L.get("slot29_counts"))
    # A CLASS WHOSE VECTOR IS A CONSTANT FILL STILL HAS A WITNESSED SET. Such a layout carries no
    # `slot29_vector` to derive from, so the checks above never run for it, and the fill writes
    # the witness's vector whatever was declared - including for a caller that declares nothing.
    # Where the layout states its sets, a direct request outside them is refused by name here.
    # A caller that STATES a set - the empty one included - is held to the witness; a caller
    # that states nothing (None) is describing the layout, as _pk_slot1 does, and is not.
    if (L.get("slot29_sets") is not None and L.get("slot29_vector") is None
            and system_registers is not None):
        declared = tuple(int(r) for r in system_registers)
        if declared not in tuple(tuple(s) for s in L["slot29_sets"]):
            raise ValueError(
                "this class carries its slot-29 vector as a measured constant and is witnessed only "
                "for the system-register sets %s; %s was requested. It cannot carry that set."
                % ([list(s) for s in L["slot29_sets"]], list(declared)))
    q = L["q"] if q is None else q
    second = L["v2_vals"] if second is None else second
    if len(bindings) != len(L["bind"]):
        raise ValueError("this class's layout has exactly %d binding records; %d requested. "
                         "A different count needs its own measured layout."
                         % (len(L["bind"]), len(bindings)))
    b = bytearray(L["size"])
    struct.pack_into("<I", b, 0, L["root"])
    _put_table(b, L["root"], L["rvt"], L["root_vlen"], L["root_tlen"], {0: 8, 3: 4},
               {0: ("<I", L["pk"] - (L["root"] + 8))})   # slot 3 stays zero: measured inert
    # THE PER-KERNEL TABLE'S OFFSETS ARE ORDINARY FLATBUFFERS REFERENCES: the u32 in a field is
    # relative to THAT FIELD'S ADDRESS, not to the table. Written table-relative the arithmetic
    # happens to agree whenever the serialiser put slot 4 at offset 40, which is 7,519 of the
    # 7,557 cached objects and not the other 38. So each is computed from where its own field
    # lands, and Q is just the value slot 4 carries.
    sl = L["pk_slots"]
    ref = lambda slot, target: target - (L["pk"] + sl[slot])
    pk_vals = {3: ("<I", 8 * len(bindings)),
               4: ("<I", ref(4, L["vec_bind"])),
               2: ("<I", ref(2, L["vec2"]))}
    # THE PER-KERNEL TABLE IS A TABLE OF OFFSETS, and the inertness sweep zeroed four of them
    # because a store to a device buffer never reaches them. Walking 7,403 objects gives their
    # laws exactly - slots 6, 8 and 10 carry Q itself, and slot 26 locates the second vector
    # twelve bytes early - so they are written from the layout rather than left at zero. Left at
    # zero the loader computes the second vector at pk+12 and a three-binding section segfaults it.
    # Slots 6, 8 and 10 hold the SAME NUMBER in every object and therefore refer to three
    # different places, four bytes apart, because their fields are. Apple's value is Q, so they are
    # written that way rather than pointed anywhere this file has a name for.
    for s6 in ((6, 8, 10, 12) if restore_swept else ()):
        if s6 in sl:
            pk_vals[s6] = ("<I", q)
    if restore_swept and 26 in sl and L.get("vec0") is not None:
        pk_vals[26] = ("<I", ref(26, L["vec0"]))
    # THE OTHER POINTER SLOTS, when the class measures them. 13, 27 and 29 hold references like 26
    # does, and leaving them zero is what stood between this serializer and a byte-exact
    # out-of-sample section: four bytes, three of them these.
    for slot, target in (L.get("ptrs") or {}).items():
        if slot in sl:
            pk_vals[slot] = ("<I", ref(slot, target))
    pk_vals.update(L["pk_extra"])
    # THE PROGRAM'S OWN REGISTER COUNT OVERRIDES THE CLASS WITNESS'S. It is a fact about the code,
    # so a constant is correct only for the object the class was measured from.
    if register_count is not None:
        if type(register_count) is not int or register_count < 1:
            raise ValueError("register count must be a positive integer, not %r" % (register_count,))
        if 0 not in sl:
            raise ValueError("this measured class has no slot 0 to carry a register count")
        pk_vals[0] = ("<I", register_count)
    _put_table(b, L["pk"], L["pkvt"], L["pk_vlen"], L["pk_tlen"], L["pk_slots"], pk_vals)

    _put_vector(b, L["vec_bind"], [r[0] for r in L["bind"]])
    # A vtable named by more than one binding record is SHARED; the later table need not abut it.
    _counts = {}
    for _rec in L["bind"]:
        _counts[_rec[1]] = _counts.get(_rec[1], 0) + 1
    _shared_vtables = {_v for _v, _n in _counts.items() if _n > 1}
    for rank, (rec, idx) in enumerate(zip(L["bind"], bindings)):
        pos, vtpos, shape = rec[:3]
        vlen, tlen, slots = _REC[shape]
        if len(rec) > 3: tlen = rec[3]      # the declared inline length is per class
        vals = {0: ("<B", 5)}
        if shape.startswith("elided"):
            if idx: raise ValueError("record %d elides its index field, so it can only name "
                                     "buffer 0; %d requested" % (pos, idx))
        else:
            vals[1] = ("<I", idx)
        if 2 in slots:
            # FIELD 2 IS THE BINDING'S POINTER-BLOCK OFFSET, which the compiler states, and it is
            # only 2*rank when the offsets happen to run 0, 2, 4. The four-binding witness settles
            # it: c4probe's records carry 6, -, 2 and 4 against ranks 0, 2 and 3, so a rank-derived
            # value writes the wrong buffer's offset into three of them. 2*rank remains the
            # fallback because it reproduces the two- and three-binding classes byte-for-byte,
            # where every measured offset equals it.
            vals[2] = ("<I", (offsets[rank] if offsets is not None else 2 * rank))
        # THE WRITTEN FLAG FOLLOWS THE SHAPE'S OWN SLOT MAP, not the shape's NAME. Keying on "long"
        # meant a record shape that carries slot 3 without an offset field - which is what a
        # writable buffer at pointer offset 0 emits - silently left the flag clear, and the section
        # differed from its witness in exactly that byte while everything else reproduced.
        if 3 in slots:
            vals[3] = ("<B", 1)
        _put_table(b, pos, vtpos, vlen, tlen, slots, vals, shared=vtpos in _shared_vtables)

    # THE SYMBOL NAMES ARE NOT OBJECT-SPECIFIC. The kernel is agc.main in 21,001 of 21,001 corpus
    # objects and its constant program is agc.main.constant_program, so a class that carries them
    # can emit them: they are a property of the format, not of the program. The swept class has
    # these regions zeroed, which is why it needs none of this.
    if restore_swept:
        for key, name in (("name", SYMBOLS[0]), ("cpname", SYMBOLS[1])):
            spot = L.get(key)
            if not spot:
                continue
            lenpos, strpos = spot
            enc = name.encode()
            _struct.pack_into("<I", b, lenpos, len(enc))
            b[strpos:strpos + len(enc)] = enc
        for slot, (pos, length) in (L.get("word_vectors") or {}).items():
            entries = (words or {}).get(slot)
            if entries is None:
                raise ValueError(
                    "this class carries a word vector at per-kernel slot %d with %d entries and "
                    "their VALUES are the compiler's, not the class's - in the measured witness "
                    "slot 27 holds binding indices. State them (pk_vectors[%d]) rather than "
                    "inheriting the witness's numbers." % (slot, length, slot))
            if len(entries) != length:
                raise ValueError("slot %d's vector is %d entries in this class; %d given"
                                 % (slot, length, len(entries)))
            _struct.pack_into("<I", b, pos, length)
            for j, v in enumerate(entries):
                _struct.pack_into("<I", b, pos + 4 + 4 * j, v)
        spot = L.get("slot29_vector")
        if spot is not None:
            if not system_registers:
                raise ValueError(
                    "this class's slot-29 vector is derived from the program's declared system "
                    "register and none was stated. Supply abi['system_registers'].")
            # Classes that state their sets are held to them; the call shape for every other
            # class is unchanged, so a checker that stands in for this function still fits.
            entries = (slot29_entries(system_registers, L.get("slot29_counts"), L["slot29_sets"])
                       if L.get("slot29_sets") is not None
                       else slot29_entries(system_registers, L.get("slot29_counts")))
            _struct.pack_into("<I", b, spot, len(entries))
            for _j, entry in enumerate(entries):
                _struct.pack_into("<I", b, spot + 4 + 4 * _j, entry)
        for pos, length in (L.get("sequence_vectors") or ()):
            _struct.pack_into("<I", b, pos, length)
            for j in range(length):
                _struct.pack_into("<I", b, pos + 4 + 4 * j, j)
        bv = L.get("byte_vector")
        if bv:
            _slot, pos = bv
            _struct.pack_into("<I", b, pos, len(bindings))
        nt = L.get("nametab")
        if nt:
            npos, nvt, nvlen, ntlen, nslots, nvals = nt
            _put_table(b, npos, nvt, nvlen, ntlen, nslots, nvals)
        for pos, fmt, val in (L.get("fills") or ()):
            _struct.pack_into(fmt, b, pos, val)
    if restore_swept and L.get("vec0") is not None:
        pos, vtpos = L["v0"]
        _put_vector(b, L["vec0"], [pos])
        v0vals = dict(V0_VALS)
        if L.get("v0_field0") is not None:
            v0vals[0] = ("<I", L["v0_field0"])
        if L.get("v0_field2") is not None:
            v0vals[2] = ("<I", L["v0_field2"])
        _put_table(b, pos, vtpos, V0REC[0], V0REC[1], V0REC[2], v0vals)
    _put_vector(b, L["vec2"], [r[0] for r in L["v2"]])
    # A SLOT-2 VTABLE CAN BE SHARED TOO. The six-buffer witness's kind-6 and kind-3 records name one
    # vtable between them, exactly as binding records 2 and 3 do in the five-buffer class; without
    # this the second table's abutment assertion fires on a layout Apple actually emits.
    _v2_counts = {}
    for _rec in L["v2"]:
        _v2_counts[_rec[1]] = _v2_counts.get(_rec[1], 0) + 1
    _v2_shared = {_v for _v, _n in _v2_counts.items() if _n > 1}
    for rec2, vals in zip(L["v2"], second):
        pos, vtpos, shape = rec2[:3]
        vlen, tlen, slots = _V2REC[shape]
        if len(rec2) > 3:
            tlen = rec2[3]          # the declared inline length is per class, as for bindings
        # VALUES BY SLOT, not by position. The tuple form is (kind, field 2, field 3) and every
        # measured class before the six-buffer one used it; a promoted-range record also states
        # slot 1 and may state slot 4, so it names its slots instead of growing the tuple.
        if isinstance(vals, dict):
            v = {}
            for slot, value in sorted(vals.items()):
                if slot not in slots:
                    raise ValueError("this slot-2 record shape has no slot %d; the class states %s"
                                     % (slot, sorted(slots)))
                v[slot] = ("<B", value) if slot in (0, 4) else ("<I", value)
        else:
            v = {0: ("<B", vals[0]), 2: ("<I", vals[1])}
            if shape == "long":
                v[3] = ("<I", vals[2])
        _put_table(b, pos, vtpos, vlen, tlen, slots, v, shared=vtpos in _v2_shared)
    # SLOT-13/27 CONTENT TRANSPLANT (item 11, Set C). These per-kernel vectors carry constant-program
    # data (slot 13: K-loop trips, offsets, division magics; slot 27: binding-index words) that are
    # Piece A's value computation, which this side does not yet GENERATE. When the caller supplies a
    # vector (recovered from Apple's own object, so TRANSPLANTED not generated), overwrite its bytes.
    # The vector is located by resolving the slot's flatbuffers field pointer in the built bytes, NOT
    # by a layout position (ptrs[slot] and the grown field-resolved address differ). Slot 13's count is
    # a BYTE length; slot 27's count is a WORD count, so its data byte length is 4x the count word. The
    # guard checks the built count word against the recovered vector's byte length in the slot's own
    # units. Absent the argument, the output is byte-identical, so every existing caller (including
    # emit()) is unaffected.
    for _slot, _pk in ((13, pk13), (27, pk27)):
        if _pk is None or _slot not in L["pk_slots"]:
            continue
        _w = 4 if _slot == 27 else 1                       # slot 27 counts words; slot 13 bytes
        at = L["pk"] + L["pk_slots"][_slot]
        vec = at + struct.unpack_from("<I", b, at)[0]
        if struct.unpack_from("<I", b, vec)[0] * _w == len(_pk):   # data byte length match
            b[vec + 4:vec + 4 + len(_pk)] = _pk
    return bytes(b)



# THREE BINDINGS, measured from buf3 - 452 bytes, and 188 corpus objects carry its shape. Every one
# of the 3,258 three-binding objects in the corpus carries a RICHER per-kernel class than the
# two-binding scan: fifteen slots here against the scan's nine, and zero three-binding objects have
# a slot set inside the scan's. So a three-buffer scan in the scan's own minimal class has no
# witness, and this is the class that does.
#
# VERIFIED ON THE PART A LAYOUT DETERMINES: built against the delivered buf3 section, the binding
# vector and all three of its records are byte-identical, 0 differing of 92, as is the header. What
# differs is object-specific content - the name string agc.main, the v0 record, the
# constant-program name - which this generator does not invent for any class. The executed
# two-binding class has those regions SWEPT to zero, which is why restore_swept=False reproduces it
# whole and cannot reproduce an unswept object whole.
THREE = dict(
    size=452, root=16, rvt=4, root_vlen=12, root_tlen=12,
    pk=124, pkvt=60, pk_vlen=64, pk_tlen=60,
    pk_slots={29: 4, 27: 8, 26: 12, 13: 16, 12: 20, 10: 24, 8: 28, 6: 32,
              3: 36, 4: 40, 2: 44, 1: 48, 16: 54, 15: 55, 0: 56},
    pk_extra={1: ("<I", 8), 15: ("<B", 1), 16: ("<B", 1), 0: ("<I", 3)},
    vec_bind=300, bind=[(444, 438, "elided"), (420, 410, "mid", 18), (392, 380, "long", 18)],
    vec2=316, v2=[(364, 352, "long"), (340, 330, "short")],
    v2_vals=[(6, 2, 6), (3, 6, None)],
    vec0=196, v0=(216, 204), q=140,
    name=(44, 48), cpname=(240, 244),
    # __GPU_ARCH_LD_MD is a CLASS fact too. Two 32-byte forms exist: the executed scalar class
    # zeroes the tail, and 141 of 200 sampled corpus objects carry this populated one. Emitting the
    # executed class's shape for every class was four of the differing bytes.
    arch32=bytes.fromhex("0c00000000000600080004000600000008000000040004000400000000000000"),
    ptrs={13: 272, 27: 184, 29: 188},
    # THE SLOT-29 VECTOR HERE IS A CONSTANT FILL, NOT AN ENCODING. `fills` writes a count of 1 at
    # 188 and the entry at 192 is the witness's zero: buf3 reads threadgroup_position_in_grid.x,
    # SR156, whose entry IS 0. So this class carries exactly one witnessed set and nothing in the
    # build reads the declaration - a contract declaring [160] got [0] where [80] belongs, and
    # [156, 160] was truncated to [0] (pr_eq4; syn-s447197ee47), silently, because the count
    # gate saw a mapped register and stopped. Stating the witnessed length and set is what makes
    # `slot29_entries` refuse those by name before anything is built. No vector support is added;
    # a second set on this shape needs its own witness.
    slot29_counts=(1,), slot29_sets=((156,),),
    # The kernel-name table: vtable at 28, table at 36, its one field, and the two residual
    # values that reference them. Measured from the witness like every other layout number.
    nametab=(36, 28, 8, 8, {1: 4}, {1: ("<I", 4)}),
    fills=((0x14, "<I", 16), (0x110, "<I", 8), (188, "<I", 1)),
)

# Three END-only bindings 0/1/2, with only buffer 2 written and
# thread_position_in_grid.x (SR160) read. Two independently compiled Apple
# objects, apple-three-binding-witness.o and apple-three-binding-holdout.o,
# have this same 452-byte shape and slot-29 vector [80]. Replacing THREE's
# constant SR156 fill with the witnessed vector reproduces both metadata
# sections byte for byte, including their distinct register counts.
THREE_SR160 = dict(THREE)
THREE_SR160.update(
    fills=tuple(f for f in THREE["fills"] if f[0] != 188),
    slot29_vector=188, slot29_counts=(1,), slot29_sets=((160,),))

# THE TILE-COORDINATE SET ON THE THREE-BUFFER SHAPE. An imageblock program builds its coordinate from
# SR_LOCAL_X and SR_LOCAL_Y (164, 165), so its slot-29 vector has TWO entries, [48, 49], where THREE's
# witness has one: the vector at 188 gains a word and everything after it moves by four, which is
# exactly what `with_system_registers` derives. WITNESSED ON HARDWARE 2026-09-23: Apple's explicit-
# imageblock compute kernel run with this section (plus agxforge.g17.imageblock's declaration) returned
# every lane's neighbour exactly, three runs of three (isa/g17-imageblock-receipt.json). Only this set
# is admitted; THREE itself is unchanged, so every section it already authors is byte-identical.
THREE_COORDINATE = dict(THREE)
for _k in ("slot29_counts", "slot29_sets"):
    THREE_COORDINATE.pop(_k, None)
THREE_COORDINATE.update(
    fills=tuple(f for f in THREE["fills"] if f[0] != 188),
    slot29_vector=188, slot29_counts=(2,), slot29_sets=((164, 165),))
COORDINATE_REGISTERS = (164, 165)

# THREE BINDINGS THAT DO NOT INCLUDE BUFFER 0, measured from abi-v1-separate-indexed-93c46ccf38:
# 456 bytes, and 54 corpus objects bind exactly (1, 2, 3) without sharing a vtable between binding
# records. It is THREE with its first record changed from "elided" to "short", because an elided
# record carries no index field and so can only ever name buffer 0. That is not a variant to choose
# by taste: a contract binding 1, 2 and 3 CANNOT be authored by the elided-first layout, and the
# generator says so rather than emitting a record that names the wrong buffer.
#
# Verified the same way as THREE: built against the delivered section, the binding vector and all
# three records are byte-identical, 0 differing of 92, as is the header.
SEPARATE = dict(THREE)
# THREE's witnessed set is THREE's. This layout and everything derived from it are witnessed
# separately, so the constant-fill gate is not inherited by copying; each states its own.
for _k in ("slot29_counts", "slot29_sets"):
    SEPARATE.pop(_k, None)
SEPARATE.update(
    size=456,
    # One residual this class carries and buf3's does not, measured the same way: a u32 at 192.
    fills=THREE["fills"] + ((192, "<I", 80),),
    pk_extra={1: ("<I", 8), 15: ("<B", 1), 16: ("<B", 1), 0: ("<I", 2)},
    bind=[(444, 436, "short"), (420, 410, "mid", 16), (392, 380, "long", 18)],
)

# ONE BINDING, BUFFER 0, WRITTEN. Measured from the corrected r-shr64a's own object and from an
# ordinary sibling with the same signature and a different body; the two are BYTE-IDENTICAL, so the
# 64-bit shift does not reach this section and the signature does. It is THREE's structure with one
# binding record instead of three - same root, same per-kernel table, same slot set - and three
# class-specific values:
#
#   the single record is "elided_written", the only shape with no index field and so the only one
#   that can name buffer 0, carrying slot 3 for the write;
#   the second vector's kind-6 record states f2 = 2 (the 8-byte pool over four) and f3 = 2;
#   PER-KERNEL SLOT 1 IS 4, and that is a fact of THIS class and not a one-binding formula. The
#   ledger's own line records that one binding at offset 0 takes 4, 8, 12 AND 16 across 21,020
#   corpus sections, so nothing derives it - which is exactly why the public author refused to
#   guess. Three controls at this signature show 4, at register counts 2 and 12.
#
# The stated form is the ONE-register base, as _tg_three's is: `build` derives the two-register form
# the delivery needs, and slot29_counts admits one or two entries and nothing else.
#
# WITNESSED BELOW THE SLOT-32 BOUNDARY ONLY. A third control with a longer body carries slot 32 and
# is 404 bytes - a different layout - so `layout_for` refuses this class for a program that crosses
# the boundary rather than serving the short form to a long program.
ONE_WRITTEN_ZERO_REGISTERS = (156, 157)
ONE_WRITTEN_ZERO = dict(THREE)
# THREE's witnessed set is THREE's. This layout and everything derived from it are witnessed
# separately, so the constant-fill gate is not inherited by copying; each states its own.
for _k in ("slot29_counts", "slot29_sets"):
    ONE_WRITTEN_ZERO.pop(_k, None)
ONE_WRITTEN_ZERO.update(
    size=392,
    bind=[(384, 372, "elided_written")],
    vec_bind=300, vec2=308, vec0=196,
    v2=[(356, 344, "long"), (332, 322, "short")],
    v2_vals=[(6, 2, 2), (3, 2, None)],
    v0=(216, 204),
    ptrs={13: 272, 27: 184, 29: 188},
    cpname=(240, 244),
    slot29_counts=(1, 2), slot29_vector=188,
    fills=((20, "<I", 16), (272, "<I", 8)),
    q=140,
    pk_extra={1: ("<I", 4), 15: ("<B", 1), 16: ("<B", 1), 0: ("<I", 2)},
)


# THREE BINDINGS WITH BUFFER 0 WRITTEN, measured from probe-aq-0-before's own object and from an
# ordinary control with the same signature and the same write mask. It is SEPARATE with ONE record
# changed: position 0 carries "elided_written" rather than "short", because the elided shape is the
# only record with no index field - so it is the only one that can name buffer 0 - and when buffer 0
# is the written one it gains slot 3. Records 1 and 2 are SEPARATE's unchanged, the same shapes at
# the same offsets, which is why the size does not move.
#
# TWO CONTROLS, AND THE PAIR IS WHAT MAKES IT A CLASS FACT RATHER THAN AN ATOMIC ONE. The delivered
# source declares `device atomic_uint` at buffer 0 and reads-modifies-writes it; the control
# declares an ordinary `device uint` there and reads and writes it. Their sections are IDENTICAL
# except at offset 180 - slot 0, register count 2 against 3 - so the atomic declaration does not
# reach this section and the write mask does. Each reproduces byte-for-byte from its OWN register
# count; neither is built from the other's.
THREE_WITH_WRITTEN_ZERO = dict(SEPARATE)
THREE_WITH_WRITTEN_ZERO.update(
    bind=[(448, 436, "elided_written"), (420, 410, "mid", 16), (392, 380, "long", 18)],
)

# ALL THREE BUFFERS WRITTEN, buffer 0 bound: the (0,1,2) class Apple emits for 518 corpus MMA
# programs (Set C, item 11). Buffer 0's index field is elided (the flatbuffer default, since a zero
# index and an absent one encode the same), every non-const buffer is marked written, and the two
# identical `long` records share one vtable. Measured byte-for-byte from ac2-128x32x64 through
# TM.layout's growth. Its record region is 12 bytes shorter than SEPARATE's [short, mid, long]
# (elided_written+long+long with a shared vtable is 64 bytes against 76), so the free-tail base is
# 444. layout_for selects it only for the exact (True, True, True) mask with buffer 0 bound.
SEPARATE_ALL_WRITTEN = dict(SEPARATE)
SEPARATE_ALL_WRITTEN.update(
    size=444,
    bind=[(436, 424, "elided_written", 8), (408, 396, "long", 16), (380, 396, "long", 16)],
)

# Retained syn-s7f595f1cd1 binds 26/28/29 at offsets 0/2/4 and carries
# slot 29 [0,1,2,48]; syn-s2390eeee22 independently carries that vector.
# Compose that length-prefixed vector with the existing unpromoted separate
# class. Their promoted constant-program fields are NOT inherited. This is a
# structural composition, not a claim of execution for the new combination.
# The old class lacks a relocation descriptor and writes [80] through fills;
# leave that historical domain byte-identical. The new descriptor makes build
# move every later node by twelve bytes and write all four entries instead.
SEPARATE_FOUR_REGISTERS = dict(SEPARATE,
    slot29_counts=(4,), slot29_vector=SEPARATE["ptrs"][29],
    fills=tuple(row for row in SEPARATE["fills"]
                if row[0] not in (SEPARATE["ptrs"][29], SEPARATE["ptrs"][29] + 4)))


# FOUR RECORDS WITH DRIVER-INTERNAL BINDINGS, measured from c4probe: 552 bytes. Kept because it is
# a real class and the measurement stands, but it is NOT a four-user-buffer class: c4probe's source
# declares three buffers at 0, 1 and 2 plus threadgroups_per_grid and threads_per_threadgroup, and
# indices 35-37 are resources the driver adds for those builtins. Its slot 27 lists them, which is
# why it needs a word vector no compiler holds. layout_for does not select it., and 39 corpus objects carry four binding records
# that each hold an index field, which is what lets a class name indices 1, 2, 3 and 4. No corpus
# object binds exactly (1,2,3,4) - the LayerNorm contract's shape - and it does not need to: the
# index is a FIELD VALUE, not a position, which the three-binding case proved when a layout measured
# at (1,2,3) reproduced (2,4,6) byte-for-byte.
#
# THIS CLASS CARRIES WORD VECTORS WHOSE CONTENTS ARE PROGRAM FACTS, not class facts. Slot 27 holds
# a two-entry vector that in the witness is literally binding indices [35, 36], and slot 29 a
# six-entry one. Their POSITIONS and LENGTHS are class data and live here; their CONTENTS have to
# come from the compiler, so a contract that does not state them is refused rather than filled with
# the witness's numbers - that would be a donor copy.
FOUR_INTERNAL = dict(
    size=552, root=16, rvt=4, root_vlen=12, root_tlen=12,
    pk=128, pkvt=60, pk_vlen=68, pk_tlen=64,
    pk_slots={29: 4, 27: 8, 26: 12, 13: 16, 12: 20, 10: 24, 8: 28, 6: 32, 3: 36, 4: 40, 2: 44,
              1: 48, 31: 52, 16: 58, 15: 59, 0: 60},
    pk_extra={15: ("<B", 1), 16: ("<B", 1), 0: ("<I", 10), 1: ("<I", 12)},
    vec_bind=364, bind=[(536, 524, "long"), (512, 504, "short"),
                        (488, 478, "mid", 16), (460, 450, "mid", 18)],
    vec2=384, v2=[(432, 420, "long", 18), (408, 398, "short")],
    v2_vals=[(6, 1, 11), (3, 8, None)],
    vec0=232, v0=(252, 240), q=200, v0_field0=40, v0_field2=7,
    sequence_vectors=((272, 8),),
    ptrs={13: 340, 27: 192, 29: 204, 31: 212},
    name=(44, 48), cpname=(308, 312),
    nametab=(36, 28, 8, 8, {1: 4}, {1: ("<I", 4)}),
    fills=((0x14, "<I", 16), (188, "<I", 10), (0x104, "<I", 7), (0x108, "<I", 50331648)),
    # slot -> (position, length). The length is class data; the entries are the compiler's.
    word_vectors={27: (192, 2), 29: (204, 6)},
    # slot 13 is a BYTE vector whose length is the binding count and whose entries are zero in the
    # witness - both derivable, so it is emitted rather than required.
    byte_vector=(13, 340),
    arch32=bytes.fromhex("0c00000000000600080004000600000008000000040004000400000000000000"),
)


# THE CLASS THIS CONTRACT NEEDS HAS NO WITNESS, and that is a different problem from a missing fact.
#
# The LayerNorm delivery binds four FP32 buffers at indices 1, 2, 3, 4 with pointer-block offsets
# 0, 2, 4, 6 and the fourth written. By the elision law - a field is present exactly when its value
# is not the default - that requires binding records carrying, in member order:
#
#     {0,1}      index 1, offset 0 elided as the default
#     {0,1,2}    index 2, offset 2
#     {0,1,2}    index 3, offset 4
#     {0,1,2,3}  index 4, offset 6, and field 3 marking it written
#
# pr-buf4 is that class one field away: four user buffers, three read and one written, the SAME
# fifteen per-kernel slots as buf3 and separate-indexed, and word vectors that need nothing from the
# compiler - slot 13 and 27 empty, slot 29 a single derivable entry. Its member 0 is the ELIDED
# shape because it binds buffer 0, so both its index and its offset are defaults. It cannot name
# index 1.
#
# No object in this corpus carries the short-first sequence ('01','012','012','0123'): zero of the
# four-record classes have every record carrying index AND offset, and the sequence is absent. The
# three-binding case had both variants only because a probe was compiled for the second -
# abi-v1-separate-indexed exists precisely so that (1,2,3) could be named where buf3 names (0,1,2).
#
# So the requirement is a WITNESS, not a fact: one Metal kernel with four device buffers at
# [[buffer(1)]] through [[buffer(4)]], three read and one written, no texture, sampler, threadgroup
# or simd construct. Measuring that gives this class the way separate-indexed gave the last one.
# Constructing it by promoting pr-buf4's first record from elided to short would move every
# subsequent record and invent a layout, which is the donor-free equivalent of guessing.


# FOUR USER BUFFERS AT NON-ZERO INDICES, measured from ln-buf4-indexed, 448 bytes. This is the class
# the MiniLM LayerNorm delivery needs and it reproduces its witness byte-for-byte.
#
# The witness had to be compiled: no cached object bound four buffers at 1, 2, 3, 4, so the record
# sequence this contract requires - by the elision law, {0,1} then {0,1,2} then {0,1,2} then
# {0,1,2,3} - existed nowhere. That is the same gap abi-v1-separate-indexed filled for three
# bindings, and it was closed the same way: one Metal kernel through Apple's compiler, CPU only.
#
# It carries the SAME fifteen per-kernel slots and the same slot offsets as buf3 and
# separate-indexed, and it needs nothing from the compiler that this side cannot derive: slots 13
# and 27 are empty vectors and slot 29 is a single entry of 80, exactly as in the three-binding
# class. The pk_vectors requirement c4probe produced was an artifact of that wrong class.
FOUR = dict(SEPARATE)
FOUR.update(
    size=448, pk=124, pkvt=60, pk_vlen=64, pk_tlen=60,
    pk_extra={15: ("<B", 1), 16: ("<B", 1), 0: ("<I", 5), 1: ("<I", 8)},
    vec_bind=292, bind=[(436, 428, "short"), (412, 402, "mid", 16),
                        (384, 374, "mid", 18), (356, 344, "long", 18)],
    vec2=312, v2=[(332, 322, "short")], v2_vals=[(3, 8, None)],
    vec0=196, v0=(216, 204), q=132,
    ptrs={13: 272, 27: 184, 29: 188},
    name=(44, 48), cpname=(240, 244),
    fills=((0x14, "<I", 16),),
    # position of the slot-29 vector; its LENGTH and ENTRY come from the declared register
    slot29_vector=188,
)
for _k in ("word_vectors", "byte_vector", "sequence_vectors", "v0_field0", "v0_field2"):
    FOUR.pop(_k, None)

# SLOT 29'S ENTRY IS DERIVED FROM THE DECLARED SYSTEM REGISTER, not carried as a constant.
# Measured over 690 objects: the first entry of the slot-29 vector tracks which special register the
# program reads. reg:61 - the decoder's print for SR 160, thread_position_in_grid - gives (80,) in
# all 34 objects that read it alone, and reg:54 gives a first entry of 0. So a class that reads one
# system register derives its slot-29 vector from the ABI's system_registers, and a register outside
# the measured map is refused rather than defaulted.
# FIVE USER BUFFERS AT NON-ZERO INDICES, 508 bytes, measured from b5-indexed and corroborated by
# three further compiled witnesses. This is the class a fused feed-forward stage needs - source,
# two weights, two biases, one output - and it was refused here until it was measured.
#
#     b5-indexed      508 B  slot 29 [80]      (1,0,r)(2,2,r)(3,4,r)(4,6,r)(5,8,w)
#     b5-indexed-2    508 B  DIFFERENT ARITHMETIC, byte-identical section - the class is a
#                            function of the contract, not of the program
#     b5-renumbered   508 B  indices 2..6, byte-identical - the index is a FIELD VALUE, not a
#                            position, the same law the three- and four-binding classes obey
#     b5-xy           512 B  slot 29 [80, 81] - the four-byte widening law carries here too
#
# TWO STRUCTURAL FACTS THIS CLASS ADDED, neither visible in any smaller class:
#   - records 2 and 3 SHARE one vtable, and the second table sits BEFORE it, so its soffset is
#     negative. _put_table's abutment assertion is measured and still holds everywhere else; a
#     shared vtable is declared rather than assumed.
#   - its slot-13 fill lives at 272, above the slot-29 vector, so widening has to move fills as
#     well. Every smaller class's fills sit below the cut, which is why nothing caught it before.
FIVE = dict(SEPARATE)
FIVE.update(
    size=508, vec_bind=300, q=140,
    pk_extra={15: ("<B", 1), 16: ("<B", 1), 0: ("<I", 6), 1: ("<I", 12)},
    bind=[(496, 488, "short"), (472, 462, "mid", 16), (444, 434, "mid", 18),
          (416, 434, "mid", 18), (400, 388, "long", 16)],
    vec2=324, v2=[(372, 360, "long"), (348, 338, "short")],
    v2_vals=[(6, 2, 10), (3, 10, None)],
    vec0=196, v0=(216, 204), ptrs={13: 272, 27: 184, 29: 188},
    name=(44, 48), cpname=(240, 244), slot29_vector=188,
    fills=((0x14, "<I", 16), (272, "<I", 8)),
)


# FOUR USER BUFFERS WITH A THREADGROUP SCRATCHPAD - the cooperative class. 512 bytes for one
# declared system register, 516 for two, measured from the register-chain witness integration
# retained and reproducing it byte-for-byte.
#
# It carries three slots no earlier class has:
#   slot 18  threadgroup memory used, static or dynamic  (uses_threadgroup in the ABI)
#   slot 28  the statically declared byte count          (threadgroup.static_memory_bytes)
#   slot 32  DERIVED, not stated: absent at <= 30 instructions, 3 with a back edge, 1 up to 300,
#            2 from 301 - measured on 43 witnesses across both boundaries.
#
# SLOT 13 IS THE PROGRAM'S CONSTANT POOL, NOT A CLASS CONSTANT. Integration's controls settle it:
# changing one literal moves slot-13 bytes while __TEXT stays byte-identical. The witness this
# class is measured from carries an EMPTY pool of sixteen zero bytes, which is why it is the right
# witness; an earlier attempt used a 185-literal FMA chain and 126 bytes differed, all of them that
# program's own constants. A contract with real constants must supply them; they are not class data.
FOUR_THREADGROUP = dict(SEPARATE)
FOUR_THREADGROUP.update(
    size=512, root=16, rvt=4, root_vlen=12, root_tlen=14,
    pk=136, pkvt=66, pk_vlen=70, pk_tlen=64,
    pk_slots={0: 60, 1: 48, 2: 44, 3: 36, 4: 40, 6: 32, 8: 28, 10: 24, 12: 20, 13: 16,
              15: 59, 16: 58, 18: 57, 26: 12, 27: 8, 28: 52, 29: 4, 32: 56},
    pk_extra={15: ("<B", 1), 16: ("<B", 1), 18: ("<B", 1), 0: ("<I", 4), 1: ("<I", 12),
              28: ("<I", 128), 32: ("<B", 2)},
    vec_bind=324, q=152,
    nametab=(40, 30, 10, 12, {1: 8, 2: 4}, {1: ("<I", 4), 2: ("<I", 32)}),
    bind=[(500, 492, "short"), (476, 466, "mid", 16), (448, 438, "mid", 18), (420, 408, "long", 18)],
    vec2=344, v2=[(392, 380, "long"), (368, 358, "short")], v2_vals=[(6, 4, 8), (3, 8, None)],
    vec0=212, v0=(232, 220), ptrs={13: 288, 27: 200, 29: 204},
    name=(52, 56), cpname=(256, 260), slot29_vector=204,
    fills=((0x14, "<I", 20), (288, "<I", 16)),
)

# THE LOOP CLASS, DERIVED FROM THE ONE ABOVE BY A MEASURED SHIFT AND CHECKED AGAINST ITS WITNESS.
#
# coop4-loop-nolt is coop4-flat-b with ONE source line changed - the thirty-two unrolled adds
# replaced by a loop - and the delta between the two sections was measured at
# results/g17-coop-loop-witness-v1 rather than guessed:
#
#     the per-kernel table gains EXACTLY slot 33 and no other
#     pk 136 -> 140, its vtable 66 -> 68, vlen 70 -> 72, tlen 64 -> 68, section 516 -> 524
#     slot 33 takes offset 59; slots 0, 15, 16, 18 and 32 move up four; thirteen do not move
#     every structure AFTER the per-kernel table moves by eight; the two tables before it do not
#
# THE SHIFT IS A CONSTRUCTION, NOT A WARRANT. What makes this a class rather than a neighbour
# stretched to fit is that `g17cooperativemetadata` authors coop4-loop-nolt's 524 bytes exactly
# from a contract, and its test asserts that. If the shift were wrong the section would still be
# 524 bytes - which is precisely the failure mode the cooperative gate exists to prevent - so the
# byte check is the claim and this arithmetic is only how the candidate was produced.
#
# `q` IS NOT AN OFFSET AND IS NOT SHIFTED. It is a value derived from vec_bind, pk and slot 4's
# offset, and shifting it as though it were an address left four words wrong in the first attempt.
LOOP_PK_SLOT = 33
LOOP_PK_SLOT_OFFSET = 59
LOOP_SLOTS_THAT_MOVE = (0, 15, 16, 18, 32)
LOOP_SHIFT_AFTER_PK = 8


def _four_threadgroup_loop():
    import copy as _copy
    base = FOUR_THREADGROUP
    fixed = {"pk", "pkvt", "pk_vlen", "pk_tlen", "pk_slots", "pk_extra", "size", "root", "rvt",
             "root_vlen", "root_tlen", "arch32", "v2_vals", "nametab", "name", "q"}

    def move(value):
        if isinstance(value, dict):
            return {k: move(v) for k, v in value.items()}
        if isinstance(value, tuple):
            return tuple(move(v) for v in value)
        if isinstance(value, list):
            return [move(v) for v in value]
        if isinstance(value, int) and value >= base["pk"]:
            return value + LOOP_SHIFT_AFTER_PK
        return value

    out = {k: (_copy.deepcopy(v) if k in fixed else move(v)) for k, v in base.items()}
    out["pk"] = base["pk"] + 4
    out["pkvt"] = base["pkvt"] + 2
    out["pk_vlen"] = base["pk_vlen"] + 2
    out["pk_tlen"] = base["pk_tlen"] + 4
    slots = dict(base["pk_slots"])
    for slot in LOOP_SLOTS_THAT_MOVE:
        slots[slot] += 4
    slots[LOOP_PK_SLOT] = LOOP_PK_SLOT_OFFSET
    out["pk_slots"] = slots
    out["pk_extra"] = dict(base["pk_extra"])
    out["pk_extra"][LOOP_PK_SLOT] = ("<B", 1)
    out["size"] = base["size"] + LOOP_SHIFT_AFTER_PK
    # Its own formula, not a shift: q is a value, not an address.
    out["q"] = out["vec_bind"] - (out["pk"] + out["pk_slots"][4]) + 4
    return out


FOUR_THREADGROUP_LOOP = _four_threadgroup_loop()

# THE LOOP CLASS WITHOUT SLOT 32. A program that loops carries slot 33, and slot 32 is absent when
# its instruction count is one this side's measured law maps to nothing - `slot32_for` returns None
# below thirty-one. That combination is a THIRD layout, not the loop class with a field left out:
# the per-kernel table is four bytes shorter and everything after it moves.
#
# MEASURED on coop4-tg2, and on cn32-trip31 which is coop4-tg with ONE NUMBER changed - the loop
# bound 32 to 31 - and is byte-identical to it (results/g17-coop-noslot32-v1):
#
#     slot 32 is dropped and no slot is added; pk, its vtable and vlen are unchanged
#     tlen 68 -> 64, section 528 -> 524, and every structure after the table moves down four
#     slots 0, 15, 16 and 18 move down FOUR and slot 33 moves down THREE, 59 -> 56
#
# SLOT 33 MOVING BY THREE IS WHY THESE OFFSETS ARE WRITTEN OUT RATHER THAN SHIFTED. A blanket -4
# left the vtable entry and two table bytes wrong, and the section was still 524 bytes - the
# plausible-size wrong section this class's gate exists to prevent.
LOOP_NO32_PK_SLOTS = {0: 60, 15: 59, 16: 58, 18: 57, 33: 56}


def _four_threadgroup_loop_no32():
    import copy as _copy
    base = FOUR_THREADGROUP_LOOP
    fixed = {"pk", "pkvt", "pk_vlen", "pk_tlen", "pk_slots", "pk_extra", "size", "root", "rvt",
             "root_vlen", "root_tlen", "arch32", "v2_vals", "nametab", "name", "q"}
    after = base["pk"] + base["pk_tlen"]

    def move(value):
        if isinstance(value, dict):
            return {k: move(v) for k, v in value.items()}
        if isinstance(value, tuple):
            return tuple(move(v) for v in value)
        if isinstance(value, list):
            return [move(v) for v in value]
        if isinstance(value, int) and value >= after:
            return value - 4
        return value

    out = {k: (_copy.deepcopy(v) if k in fixed else move(v)) for k, v in base.items()}
    out["pk_tlen"] = base["pk_tlen"] - 4
    slots = {k: v for k, v in base["pk_slots"].items() if k != 32}
    slots.update(LOOP_NO32_PK_SLOTS)
    out["pk_slots"] = slots
    out["pk_extra"] = {k: v for k, v in base["pk_extra"].items() if k != 32}
    out["size"] = base["size"] - 4
    out["q"] = out["vec_bind"] - (out["pk"] + out["pk_slots"][4]) + 4
    return out


FOUR_THREADGROUP_LOOP_NO32 = _four_threadgroup_loop_no32()

FOUR_THREADGROUP_REQUIRED_SHAPES = ("01", "012", "012", "0123")


def slot32_for(instructions, has_back_edge):
    """Slot 32 from the two facts the delivered contract already carries.

    Measured on 43 compiled witnesses, including both boundaries the corpus could not show: a
    straight-line program of 30 instructions carries no slot 32 and one of 31 carries 1; 300 gives
    1 and 301 gives 2. This is a law about what Apple's compiler emits on this toolchain, not a
    device guarantee, so a program landing near either edge deserves its own witness.
    """
    if instructions <= 30:
        return None
    if has_back_edge:
        return 3
    return 1 if instructions <= 300 else 2


# THE ENTRY IS A PER-BASE CONSTANT PLUS THE AXIS, AND THE BASES ARE A LOOKUP. Each entry below is
# measured on a four-user-buffer control reading that register ALONE, the standard 160 and 161 were
# set by (results/g17-system-register-bases-v1):
#
#     threadgroup_position_in_grid    0x9c  156,157,158  ->   0,  1,  2
#     thread_position_in_grid         0xa0  160,161,162  ->  80, 81, 82
#     thread_position_in_threadgroup  0xa4  164,165,166  ->  48, 49, 50
#
# THE BASE ENTRIES ARE NOT MONOTONE IN THE SR INDEX - 156 -> 0, 160 -> 80, 164 -> 48 - so there is
# no arithmetic across bases and `entry = register - 80`, which fits the middle row exactly, is
# refuted by both others. Within a base the axis adds one; across bases nothing is derived.
#
# 48 is the entry `results/g17-cooperative-class-census-v1` found unmapped on coop4-tg, established
# there by a discriminating pair (coop4-tg reads the threadgroup position, coop4-tg-nolt indexes
# from the grid instead). srb-local-x measures it directly.
SLOT29_BY_SYSTEM_REGISTER = {130: 52, 133: 53, 152: 4, 156: 0, 157: 1, 158: 2, 160: 80, 161: 81,
                             162: 82, 164: 48, 165: 49, 166: 50}

# 152 -> 4 IS MEASURED TO THE MAP'S OWN STANDARD, by one build-cache object that reads 152 alone and carries the
# single entry 4 (tools/g17slot29census.py, isa/g17-slot29-entries.json). Five fixed197 sources whose compiled
# contract declares 152 beside other registers agree, in Apple's own compile of the same source:
#     syn-scc8f7ad4bf (152, 156) [0, 4]; syn-sd356798b7b, syn-sfbb623ecab (152, 164) [4, 48];
#     syn-sc5af8687d5 (152, 156, 164) [0, 4, 48]; syn-sebe0ba9cb3 (152, 157, 160) [1, 4, 80].
# Each of those sources then matches Apple's compiler on hardware (tools/g17sourceexec.py, MM 25.149.1).

# 133 -> 53 (SR_SIMD_GRP, `simdgroup_index_in_threadgroup`) IS MEASURED TO THE SAME STANDARD: Apple's
# compile of a kernel reading that builtin ALONE carries slot 29 [53]; with thread_position_in_grid
# beside it, [53, 80]; `quadgroup_index_in_threadgroup` reads SR_SIMD_ELEM and SR_SIMD_GRP (and the
# grid position it is stored by) and carries [52, 53, 80] (results/g17-sr-simdgrp-v1, compile only). It agrees with the 51 cached tensor objects
# whose decoded set is SR130, SR_SIMD_GRP and SR156, all [0, 52, 53]
# (docs/archive/g17-tensorops-sr130-slot29-recount.md). 133 is byte 1 of the FOUR-byte read form, the
# numbering every key here uses. Apple's EIGHT-byte op14060 names the same register with byte 1 = 0x83,
# which g17asm.decode_sr reports as 131: key an eight-byte read by the decoder's name, not by byte 1.

# SR130 is a tensor-specific map entry measured from 1,039 exact (130,156) objects, all carrying
# [0, 52]. It is not inferred from the three coordinate bases below; those base entries remain
# independently measured and non-monotone.

# HOW MANY REGISTERS THE VECTOR CARRIES IS MEASURED, NOT ASSUMED. Compiled controls, four user
# buffers at 1..4 with identical bindings, differing only in which grid coordinates the program
# reads:
#
#     reads x        448 bytes   slot 29 [80]        reproduces the ln-buf4-indexed witness exactly
#     reads y        448 bytes   slot 29 [81]        the entry follows the REGISTER, not a position
#     reads x and y  452 bytes   slot 29 [80, 81]
#
# 161 -> 81 is measured on sr-y-only, an object reading that register ALONE, which is the standard
# the 160 -> 80 entry was measured to.
#
# A LITERAL IN THE SOURCE MOVES SLOT 13, NOT SLOT 29, and it confounded the first reading of this.
# An x-only kernel carrying the literal 384 is 496 bytes with slot 13 holding 16 bytes; an x-and-y
# kernel with no literal is 452 with slot 13 empty. Comparing the two-coordinate program against
# the one-coordinate LayerNorm therefore appears to show a 52-byte class change that is mostly the
# literal. The system-value difference alone is FOUR BYTES: one more entry in this vector.
# 162 -> 82 IS MEASURED TO THE SAME STANDARD, on sr3-z-only: a four-user-buffer control reading
# `thread_position_in_grid.z` ALONE, 448 bytes with slot 29 [82], compiled by Apple's own metal
# through the corpus harness. The three-register control sr3-xyz reads all three and carries
# [80, 81, 82] at 456 bytes - the one-register class plus two entries of four bytes, which is the
# derivation `with_system_registers` already performs. Both reproduce byte-for-byte from the
# serializer, and the three controls the map was built on reproduce unchanged in the same run,
# which is what makes the new number calibrated rather than merely new
# (results/g17-three-coordinate-controls).
#
# THE MAP NOW COVERS ALL THREE BASES, each axis measured on a control reading that register ALONE
# (results/g17-system-register-bases-v1): `threadgroup_position_in_grid` 0x9c -> 0, 1, 2;
# `thread_position_in_grid` 0xa0 -> 80, 81, 82; `thread_position_in_threadgroup` 0xa4 -> 48, 49, 50.
# The paragraph above this one used to say the other two bases had no measured entry for any axis;
# that was true when it was written and is not now. `entry = register - 80` stays dead: it fits the
# middle base exactly and is refuted by both others, which is why nothing here is derived across
# bases.
#
# THE COUNT IS THE VECTOR'S LENGTH, AND IT IS A PROPERTY OF THE CLASS. This is the default for
# classes that state none. Each entry past the first adds four bytes and moves every structure laid
# out after the vector, so a count admitted where no witness carries it writes over that structure
# and yields a section of an entirely plausible size. A class with a wider witness states its own
# counts in its layout (`slot29_counts`); the cooperative class does, at four
# (results/g17-coop-group-base-v1), and no other class has witnessed a fourth entry.
# THE DEFAULT IS (1, 2, 3) AND IT IS WITNESSED, WHICH I GOT WRONG ONCE AND AM RECORDING HERE.
# I narrowed this to (1, 2) on the argument that a three-entry vector was witnessed only in the
# cooperative class, so the four-buffer class was authoring 456 bytes by extrapolating the +4 law
# one step past its own witnesses. The premise was false. `sr3-xyz` - four user buffers reading x,
# y and z, compiled by Apple through the corpus harness - is 456 bytes with a THREE-entry slot-29
# vector, and this class reproduces it byte for byte with zero differing. So 456 is measured, not
# extrapolated, and `results/g17-three-coordinate-controls` is the witness that says so.
#
# WHY I MISSED IT: I scanned the retained population by globbing the filesystem while the evidence
# archive was only PARTLY extracted, so the three-coordinate controls were not on disk to be found
# and a real witness read as an absence. Integration's queue had already said to use the library's
# logical tracked_paths rather than the filesystem for archived evidence; a population measured
# over a partially materialised tree is the same defect as a census capped at the front of a corpus.
# FIVE is witnessed once, by Apple's compile of syn-s09b3736f39 (157, 158, 160, 161, 162), whose vector is
# [1, 2, 80, 81, 82] - the map's own entries, entry-ordered, with no extra entry. Four is not yet witnessed.
MEASURED_SYSTEM_REGISTER_COUNTS = (1, 2, 3, 5)


def slot29_entries(system_registers, measured_counts=None, measured_sets=None):
    """The slot-29 vector's entries for a declared register set, or a refusal.

    One place for the register-to-entry map and every refusal around it, so the two serializers -
    g17mdgen.build for the four-buffer class and g17metaclass.graph for the three-buffer one -
    cannot drift apart on which sets are admitted.

    `measured_counts` is the calling CLASS's witnessed vector lengths; classes that state none get
    the module default. A class widens this only by carrying a witness of its own.

    `measured_sets` is narrower still: the exact register SETS the class was compiled for. A count
    gates length and the map gates each register, and between them they admitted write-0 with
    [160] or [164] and write-3 with [156, 157] or [160, 164] on classes whose only witnesses are
    [156] and [156], [160], [156, 160] - plausible sections no control ever produced. A class that
    states its sets is held to them; a class that states none keeps the count-and-map rule.
    """
    counts = tuple(measured_counts or MEASURED_SYSTEM_REGISTER_COUNTS)
    declared = [int(r) for r in (system_registers or ())]
    if len(declared) not in counts:
        raise ValueError(
            "the slot-29 vector is measured for %s declared system registers; %d were stated (%s). "
            "Wider forms carry further entries this side has not measured."
            % (" or ".join(str(n) for n in counts), len(declared), declared))
    if declared != sorted(set(declared)):
        raise ValueError("declared system registers must be distinct and ascending; got %s."
                         % (declared,))
    for sr in declared:
        if sr not in SLOT29_BY_SYSTEM_REGISTER:
            raise ValueError(
                "no measured slot-29 entry for system register %d; the map covers %s, each measured "
                "on objects reading that register alone." % (sr, sorted(SLOT29_BY_SYSTEM_REGISTER)))
    # THE SET GATE RUNS AFTER THE MAP, so an unmapped register keeps the map's own reason and a
    # mapped-but-uncompiled set gets this one. Both are reasons a reader can act on.
    if measured_sets is not None and tuple(declared) not in tuple(tuple(s) for s in measured_sets):
        raise ValueError(
            "this class is witnessed only for the system-register sets %s; %s was declared. A set "
            "whose registers are each mapped is not thereby a set any control compiled."
            % ([list(s) for s in measured_sets], declared))
    # THE VECTOR IS ORDERED BY ENTRY, NOT BY DECLARED REGISTER, and this line said register order
    # until a cross-base control could tell them apart. Within one base the two agree, because the
    # entries ascend with the registers - so every witness this rule was written on was one where
    # it could not be wrong. srb-mixed-local-grid-x declares 160 and 164 and carries [48, 80];
    # register order would emit [80, 48]. coop4-tg declares 160, 161 and 164 and carries
    # [48, 80, 81], where register order would emit [80, 81, 48]. Two witnesses, both cross-base,
    # both entry-ordered.
    return sorted(SLOT29_BY_SYSTEM_REGISTER[sr] for sr in declared)


def with_system_registers(layout, count, vector_at=None, measured_counts=None):
    """`layout` measured for one declared system register, re-derived for `count` of them.

    The slot-29 vector holds a length word then one entry per declared register, so each register
    past the first adds four bytes and every structure laid out after the vector moves by that
    much. Slots 27 and 29 sit at or before it and do not move; slots 2, 4, 6, 8, 10, 12, 13 and 26
    do. This is a DERIVATION from the measured one-register class, checked byte-for-byte against a
    compiled two-register control, not a second class transcribed by hand.

    `measured_counts` is the calling class's witnessed vector lengths, as in `slot29_entries`: the
    derivation is mechanical, but a length no class witness carries is admitted by no class that
    does not state it. THIS GATE IS THE ONE THAT RUNS FIRST - the entries are written after the
    layout is widened - so widening only `slot29_entries` would have left it refusing.
    """
    counts = tuple(measured_counts or MEASURED_SYSTEM_REGISTER_COUNTS)
    if count not in counts:
        raise ValueError(
            "the slot-29 vector is measured for %s declared system registers; %d were stated. "
            "Wider forms exist in the corpus and this side has not measured their layout."
            % (" or ".join(str(n) for n in counts), count))
    spot = layout.get("slot29_vector") if vector_at is None else vector_at
    if spot is None:
        raise ValueError("this class carries no slot-29 vector to widen")
    extra = 4 * (count - 1)
    if not extra:
        return layout
    cut = spot + 8                      # length word, one measured entry: the first byte past it

    def moved(value):
        return value + extra if type(value) is int and value >= cut else value

    out = dict(layout)
    out["_widened_from"] = layout.get("_widened_from", layout)
    out["size"] = layout["size"] + extra
    # Q IS A RELATIVE OFFSET, NOT A POSITION, so the "does it lie past the vector" test does not
    # apply to it and it still has to move. Slots 6, 8, 10 and 12 store Q at field positions in the
    # per-kernel table, and it addresses the empty vectors that sit after the slot-29 vector: slot
    # 6's field at 156 holds 132 and so points at 288. Those vectors move, the fields do not, so
    # the offset between them grows. Leaving Q alone left exactly these four words wrong.
    if "q" in layout and layout["q"] is not None:
        out["q"] = layout["q"] + extra
    for key in ("vec_bind", "vec2", "vec0"):
        if key in layout:
            out[key] = moved(layout[key])
    if "v0" in layout:
        out["v0"] = tuple(moved(v) for v in layout["v0"])
    for key in ("name", "cpname"):
        if key in layout:
            out[key] = tuple(moved(v) for v in layout[key])
    for key in ("v2", "bind"):
        if key in layout:
            out[key] = [tuple(moved(v) if i < 2 else v for i, v in enumerate(rec))
                        for rec in layout[key]]
    if "ptrs" in layout:
        out["ptrs"] = {slot: moved(pos) for slot, pos in layout["ptrs"].items()}
    # FILLS ARE POSITIONS TOO. A class whose only fill sits below the cut - the four-buffer one -
    # hides this; the five-buffer class has a slot-13 length at 272 and widening left it written
    # at the old address, two bytes wrong in an otherwise byte-exact section.
    if "fills" in layout and layout["fills"]:
        out["fills"] = tuple((moved(pos),) + tuple(rest) for pos, *rest in layout["fills"])
    return out

FOUR_REQUIRED_SHAPES = ("01", "012", "012", "0123")
# The five-buffer class's record shapes in member order, measured from b5-indexed. A different
# sequence is a different class and needs its own witness, exactly as for four.
FIVE_REQUIRED_SHAPES = ("01", "012", "012", "012", "0123")
REQUIRED_SHAPES_BY_SIZE = {448: FOUR_REQUIRED_SHAPES, 508: FIVE_REQUIRED_SHAPES}


# SIX USER BUFFERS AT NON-ZERO INDICES, and the first class in this file whose slot-2 vector
# states a resource the CONTRACT owns. 488 bytes with no promoted range, 540 with one, measured
# from the six-buffer promotion family (results/g17-promoted-ranges-v1) and reproducing both
# witnesses byte-for-byte across the binding vector, all of its records, the slot-2 vector and all
# of its records - 196 bytes of 488 and 212 of 540. What differs is the object-specific content no
# class in this file invents: the name string, the v0 record, the constant-program name and the
# per-program per-kernel values.
#
# TWO FACTS THIS CLASS ADDED:
#   - THE POINTER OFFSET IS NOT 2 * RANK. It is 2 * (binding index - the lowest bound index), so a
#     promoted buffer ABOVE the lowest leaves its pointer slot empty and the buffers above it keep
#     their own. sixp-idx4 is the arm that separates the two rules: it promotes buffer 4 out of
#     1,2,3,4,5,6 and its five records carry 0, 2, 4, 8, 10 - a rank-derived value would write 6
#     and 8 into the last two and bind the wrong buffers. Where the promoted buffer IS the lowest,
#     as in sixp-one, the block simply starts at the next one and both rules agree, which is why
#     one arm could not have found this.
#   - THE SLOT-2 VECTOR CARRIES A PROMOTED-RANGE RECORD, kind 5, with the promoted buffer's binding
#     index in slot 1 - the only slot-2 record that has one.
# `q` AND THE SECOND FILL ARE THE SIX-BUFFER WITNESS'S, NOT FIVE'S, and inheriting them produced an
# image that SEGFAULTED the driver at pipeline creation (root's
# results/g17-six-buffer-runtime-v1/baseline/loader.json, returncode -11, phase=create_pipelines).
# FIVE's q is 140 and this class's witness carries 132 in per-kernel slots 6, 8, 10 and 12; FIVE
# fills 8 at byte 272 and the witness has 0 there. Six bytes wrong, four of them pointers into the
# section, and the class-region reproduction could not see any of them - the same blind spot the
# pointer targets sit in, this time with the witness disagreeing outright rather than my arithmetic.
SIX = dict(FIVE)
SIX.update(
    size=488, vec_bind=292, vec2=320, q=132, fills=((0x14, "<I", 16),),
    bind=[(476, 468, "short"), (452, 442, "mid", 16), (424, 414, "mid", 18),
          (396, 414, "mid", 18), (380, 442, "mid", 16), (364, 352, "long", 16)],
    v2=[(340, 330, "short")], v2_vals=[(3, 12, None)],
)
SIX_PROMOTED = dict(FIVE)
SIX_PROMOTED.update(
    size=540, vec_bind=304, vec2=328, q=144, fills=((0x14, "<I", 16), (272, "<I", 12)),
    bind=[(528, 520, "short"), (504, 494, "mid", 16), (476, 466, "mid", 18),
          (448, 466, "mid", 18), (432, 420, "long", 16)],
    # The kind-6 and kind-3 records SHARE vtable 392, as binding records do in the five-buffer
    # class; the promoted record has its own at 344.
    v2=[(404, 392, "long"), (376, 392, "long"), (356, 344, "promoted")],
    v2_vals=[{0: 6, 2: 3, 3: 13}, {0: 3, 2: 10, 3: 2}, {0: 5, 1: 1, 2: 1, 3: 12}],
)


# TWO WRITABLE USER BUFFERS at non-zero indices, 388 bytes for one declared system register and
# 392 for two. Measured from three BYTE-IDENTICAL corpus witnesses binding 10 and 11, both written,
# with no per-kernel slot 32 and no promoted range - the shape root's spill-resource control needs,
# which has none of its own in the corpus at indices 1 and 2.
#
# THE CLASS IS THE 388-BYTE ONE AND THE WITNESS IS ITS WIDENING. The witness declares TWO system
# registers; the four-byte widening law carries it to 392 and reproduces all 392 bytes exactly,
# zero differing. So the base is stated at one register - which is what the control declares - and
# the two-register witness is the evidence for it rather than the class itself.
#
# EVERY KEY IS DERIVED FROM THE WITNESS, none inherited on the strength of looking similar. That
# is the discipline the six-buffer class's `q` cost: it inherited FIVE's 140 against its witness's
# 132 and the image it built segfaulted the driver. Here the header keys happen to equal FIVE's -
# vec0, v0, cpname, name and the pointer targets all match - and that is a RESULT of reading them
# off the witness, not a reason for having assumed them.
#
# SCALAR IS NOT THIS CLASS. Two bindings alone select SCALAR, the nine-slot minimal shape with no
# slot 0, which is why the author refused root's control with "this measured class has no slot 0 to
# carry a register count". What separates them is that BOTH buffers are written.
TWO_WRITABLE = dict(FIVE)
TWO_WRITABLE.update(
    size=388, vec_bind=292, vec2=304, q=132,
    ptrs={13: 272, 27: 184, 29: 188}, slot29_vector=188,
    vec0=196, v0=(216, 204), name=(44, 48), cpname=(240, 244),
    fills=((0x14, "<I", 16),),
    pk_extra={15: ("<B", 1), 16: ("<B", 1), 0: ("<I", 2), 1: ("<I", 4)},
    bind=[(376, 364, "written"), (348, 336, "long", 16)],
    v2=[(324, 314, "short")], v2_vals=[(3, 4, None)],
)


# TWO WRITABLE USER BUFFERS, LONG ENOUGH TO CARRY A PER-KERNEL SLOT 32. 440 bytes at one declared
# system register. The retained corpus has NO section of this shape - zero of 9,547, and none
# satisfying more than five of its six conditions - so the witnesses were COMPILED through Apple's
# own compiler rather than found: straight-line kernels, both buffers written at indices 1 and 2,
# every read at a varying index so nothing is staged into the argument buffer, swept over length.
#
# FIVE OF THEM REPRODUCE BYTE FOR BYTE, across both slot-32 bands, once their own per-program
# values are supplied - the register count, the argument word count, the slot-32 value and the
# constant pool. That is what makes this a class rather than a composition: the earlier refusal was
# right that composing slot 32 into the short class would invent a table shape nothing witnessed,
# and this table is witnessed.
#
# SLOT 32 IS A VALUE, NOT A CLASS. `tw-fine-6-3` (31 instructions, slot 32 = 1) and `tw-fine-86-3`
# (301, slot 32 = 2) differ in exactly TWO bytes: the register count and slot 32 itself. One class
# covers both bands.
#
# AND THE COUNT RULE WAS TESTED HERE RATHER THAN ASSUMED. `slot32_for` came from a different
# population; on this class it agreed with the emitted value on 39 of 39 compiled variants, with
# both boundaries pinned by adjacent counts - 30 absent against 31 = 1, and 300 = 1 against
# 301 = 2 - rather than inferred from the rule it was being checked against.
TWO_WRITABLE_LONG = dict(FIVE)
TWO_WRITABLE_LONG.update(
    size=440, vec_bind=312, vec2=324, q=148,
    pk=128, pkvt=58, pk_vlen=70, pk_tlen=60,
    pk_slots={0: 56, 1: 48, 2: 44, 3: 36, 4: 40, 6: 32, 8: 28, 10: 24, 12: 20, 13: 16,
              15: 55, 16: 54, 26: 12, 27: 8, 29: 4, 32: 53},
    ptrs={13: 276, 27: 188, 29: 192}, slot29_vector=192,
    vec0=200, v0=(220, 208), name=(44, 48), cpname=(244, 248),
    fills=((0x14, "<I", 16),),
    pk_extra={15: ("<B", 1), 16: ("<B", 1), 0: ("<I", 11), 1: ("<I", 8), 32: ("<B", 1)},
    bind=[(428, 416, "written"), (400, 388, "long", 16)],
    v2=[(372, 360, "long"), (348, 338, "short")], v2_vals=[(6, 4, 4), (3, 4, None)],
)


def pointer_offsets(indices):
    """The pointer-block offset each bound buffer's record carries: 2 * (index - the lowest bound).

    NOT 2 * RANK, which is the general builder's fallback and is right only when the bound indices
    are contiguous. A promoted buffer above the lowest leaves its slot empty and the buffers above
    keep theirs - measured on sixp-idx4, whose records carry 0, 2, 4, 8, 10 for buffers 1, 2, 3, 5
    and 6. Rank would write 6 and 8 into the last two and point them at the wrong buffers.
    """
    indices = list(indices)
    if not indices:
        return []
    base = min(indices)
    return [2 * (index - base) for index in indices]


# FOUR USER BUFFERS INCLUDING BUFFER 0, measured from the four-active volatile control retained at
# results/g17-source-admission-v3 (object 720a94a0b7c8b2ce): 428 bytes, indices 0/1/2/3 all ACTIVE,
# register_count 2, no system registers. It is FOUR with its first record changed from "short" to
# "elided" - the same relation SEPARATE has to THREE, in the other direction - because the elided
# shape is the only one with no index field and so the only one that can name buffer 0. `layout_for`
# refused this contract outright before this class existed.
#
# WHAT IT DOES NOT SHARE WITH FOUR, each measured rather than shifted: its two middle records SHARE
# one vtable at 386, its slot-29 vector is EMPTY and present rather than derived from a declared
# register, and its __GPU_ARCH_LD_MD is a THIRD form - 40 bytes, where both classes measured before
# it carry 32 and neither carries these bytes.
#
# VERIFIED THE SAME WAY AS EVERY CLASS ABOVE: built against the delivered section from the contract
# alone, all 428 bytes are identical, 0 differing.
FOUR_WITH_ZERO = dict(FOUR)
FOUR_WITH_ZERO.pop("slot29_vector", None)
FOUR_WITH_ZERO.update(
    size=428, vec_bind=288, vec2=308, vec0=192, v0=(212, 204 - 4), cpname=(236, 240),
    ptrs={13: 268, 27: 184, 29: 188},
    v2=[(328, 318, "short")],
    bind=[(420, 414, "elided"), (396, 386, "mid", 18), (368, 386, "mid", 18),
          (352, 340, "long", 16)],
    arch32=bytes.fromhex("0c0000000000060008000400060000000c00000008000800"
                         "00000700080000000000000100000000"),
)
FOUR_WITH_ZERO["q"] = (FOUR_WITH_ZERO["vec_bind"]
                       - (FOUR_WITH_ZERO["pk"] + FOUR_WITH_ZERO["pk_slots"][4]) + 4)


# THE SAME FOUR BINDINGS WITH THE WRITE IN THE MIDDLE, measured from a control designed for this
# gap and compiled on CPU: four active buffers at 0/1/2/3 where buffer 2 is the written one. The
# write mask is a LAYOUT fact here, not a field value - only the "long" shape carries slot 3 - so a
# mask the class does not place cannot be expressed by setting a byte, and the previous refusal was
# correct but incomplete. 440 bytes against the last-written class's 428, because "long" is two
# bytes wider than the "mid" it swaps with and every record after it moves.
#
# The per-kernel table, all seven of its vectors, both strings and the 40-byte ARCH form are
# IDENTICAL to FOUR_WITH_ZERO; what moves is the four binding records and slot 2's inline length,
# 12 -> 14. Verified the same way: built from the contract alone, all 440 bytes identical.
FOUR_WITH_ZERO_MIDWRITE = dict(FOUR_WITH_ZERO)
FOUR_WITH_ZERO_MIDWRITE.update(
    size=440,
    v2=[(328, 318, "short", 14)],
    bind=[(432, 426, "elided"), (408, 398, "mid", 18), (380, 368, "long", 18),
          (352, 342, "mid", 16)],
)
FOUR_WITH_ZERO_MIDWRITE["q"] = (FOUR_WITH_ZERO_MIDWRITE["vec_bind"]
                                - (FOUR_WITH_ZERO_MIDWRITE["pk"]
                                   + FOUR_WITH_ZERO_MIDWRITE["pk_slots"][4]) + 4)


# THE OTHER TWO WRITE POSITIONS, each from its own CPU control compiled for this gap.
#
# WRITE AT BUFFER 1: 440 bytes, the long record second. Same shapes as the mid-write class with the
# long and the mids rearranged, and slot 2 keeps the 14-byte inline length those two share.
#
# WRITE AT BUFFER 0: 432 bytes, and it needs a shape nothing else uses. Buffer 0's record elides
# its index, so a written buffer 0 cannot be expressed by setting a flag on any existing shape -
# it carries slot 0 and slot 3 and NOTHING else. Its last two records also share one vtable, and
# slot 2 returns to the 12-byte inline length the write-last class uses. Neither of these is a
# field value in another class; each is where a shape sits.
FOUR_WITH_ZERO_WRITE1 = dict(FOUR_WITH_ZERO)
FOUR_WITH_ZERO_WRITE1.update(
    size=440,
    v2=[(328, 318, "short", 14)],
    bind=[(432, 426, "elided"), (408, 396, "long", 18), (380, 370, "mid", 16),
          (352, 342, "mid", 18)],
)
FOUR_WITH_ZERO_WRITE1["q"] = (FOUR_WITH_ZERO_WRITE1["vec_bind"]
                              - (FOUR_WITH_ZERO_WRITE1["pk"]
                                 + FOUR_WITH_ZERO_WRITE1["pk_slots"][4]) + 4)

FOUR_WITH_ZERO_WRITE0 = dict(FOUR_WITH_ZERO)
FOUR_WITH_ZERO_WRITE0.update(
    size=432,
    bind=[(424, 412, "elided_written"), (396, 386, "mid", 16), (368, 358, "mid", 18),
          (340, 358, "mid", 18)],
)
FOUR_WITH_ZERO_WRITE0["q"] = (FOUR_WITH_ZERO_WRITE0["vec_bind"]
                              - (FOUR_WITH_ZERO_WRITE0["pk"]
                                 + FOUR_WITH_ZERO_WRITE0["pk_slots"][4]) + 4)


# THE EMPTY VECTOR IS A WITNESSED LENGTH, NOT AN ABSENCE. Every buffer-zero class above was
# measured from a control that reads no system value, and each carries a slot-29 vector of length
# ZERO - present, and empty. Stating that length here is what lets `slot29_entries` refuse a
# contract that declares registers against one of these classes, instead of the class placing an
# empty vector under a declaration it never witnessed. That silent placement is exactly what six
# delivered images did (results/g17-execution150-v1/empty-sr-vector-review) before the classes
# below were measured.
for _empty in (FOUR_WITH_ZERO, FOUR_WITH_ZERO_MIDWRITE, FOUR_WITH_ZERO_WRITE1, FOUR_WITH_ZERO_WRITE0):
    _empty["slot29_counts"] = (0,)


# THE SAME TWO SHAPES WITH A DECLARED SYSTEM REGISTER, each based on its ONE-ENTRY witness.
#
# Six controls compiled for this question (results/g17-fourbinding-sr-v1): four active buffers at
# 0/1/2/3, one written, and ONLY the coordinate term of the stored value varied. The two no-SR
# arms reproduce the retained four-active-volatile-nosr and four-active-write0 witnesses byte for
# byte, so the SR arms are one term away from a known anchor. Apple emits, from section bytes:
#
#     fb0-w3-gx     write 3, SR[156]        432   slot 29 count 1  [0]
#     fb0-w3-tx     write 3, SR[160]        432   slot 29 count 1  [80]
#     fb0-w3-gxtx   write 3, SR[156, 160]   436   slot 29 count 2  [0, 80]
#     fb0-w0-gx     write 0, SR[156]        436   slot 29 count 1  [0]
#
# fb0-w3-gx against fb0-w3-tx differ at ONE byte offset in the whole section, 192, the entry: the
# entry follows the register. The vector sits at 188 in every arm, and each entry moves every
# structure after it by four - the derivation `with_system_registers` already performs, which is
# why the two-entry form below is DERIVED from the one-entry base and checked against fb0-w3-gxtx
# rather than transcribed a third time.
#
# THE BASE IS THE ONE-ENTRY WITNESS. `with_system_registers` re-derives a layout measured for one
# declared register, so a zero-entry class cannot be handed a vector position and asked for one
# entry: it returns the zero-entry section, four bytes short. These dicts are the one-entry
# positions - FOUR_WITH_ZERO's with every field past the vector moved by four - and every one of
# them is MEASURED in the sense that matters: built from the contract alone, all 432 and 436 bytes
# are identical to the compiled controls, zero differing, and the derived 436-byte two-entry form
# is identical to fb0-w3-gxtx.
#
# WHAT THESE DO NOT WITNESS, and refuse: a two-entry vector on the write-0 shape (no control), a
# third entry on either (none), the middle and write-1 masks with any register (none), a register
# outside the measured map, AND ANY SET THAT IS NOT ONE OF THE COMPILED ONES. Each class states
# `slot29_sets`, the exact sets its controls declared, because the count and the map together
# still admitted write-0 [160] and write-3 [156, 157] - the map knows those registers, and nothing
# in this family ever compiled them. The ARCH form is NOT a class fact here: Apple's SR156 arms
# carry the 32-byte elided ARCH section and the SR160 arm the 40-byte set one, so it follows the
# contract's own arch_flag through the author's ordinary route and no `arch32` is carried.
FOUR_WITH_ZERO_SR = dict(FOUR_WITH_ZERO)
FOUR_WITH_ZERO_SR.pop("arch32", None)
FOUR_WITH_ZERO_SR.update(
    size=432, vec_bind=292, vec2=312, vec0=196, v0=(216, 204), cpname=(240, 244),
    ptrs={13: 272, 27: 184, 29: 188}, slot29_vector=188, slot29_counts=(1, 2),
    slot29_sets=((156,), (160,), (156, 160)),
    v2=[(332, 322, "short")],
    bind=[(424, 418, "elided"), (400, 390, "mid", 18), (372, 390, "mid", 18),
          (356, 344, "long", 16)],
)
FOUR_WITH_ZERO_SR["q"] = (FOUR_WITH_ZERO_SR["vec_bind"]
                          - (FOUR_WITH_ZERO_SR["pk"] + FOUR_WITH_ZERO_SR["pk_slots"][4]) + 4)

FOUR_WITH_ZERO_WRITE0_SR = dict(FOUR_WITH_ZERO_SR)
FOUR_WITH_ZERO_WRITE0_SR.update(
    size=436, slot29_counts=(1,), slot29_sets=((156,),),
    bind=[(428, 416, "elided_written"), (400, 390, "mid", 16), (372, 362, "mid", 18),
          (344, 362, "mid", 18)],
)
FOUR_WITH_ZERO_WRITE0_SR["q"] = (FOUR_WITH_ZERO_WRITE0_SR["vec_bind"]
                                 - (FOUR_WITH_ZERO_WRITE0_SR["pk"]
                                    + FOUR_WITH_ZERO_WRITE0_SR["pk_slots"][4]) + 4)


# THE FOUR BUFFER-ZERO CLASSES AS ONE NAME. The author gates the measured route on membership
# here; listing them again at the call site is how the write-1 class reached the generic serializer
# after being measured, which is the same "a list written twice drifts" defect this tree keeps
# finding. Adding a fifth write position adds it here and the author follows.
FOUR_WITH_ZERO_CLASSES = (FOUR_WITH_ZERO, FOUR_WITH_ZERO_MIDWRITE,
                          FOUR_WITH_ZERO_WRITE1, FOUR_WITH_ZERO_WRITE0,
                          FOUR_WITH_ZERO_SR, FOUR_WITH_ZERO_WRITE0_SR)


# THREE BOUND BUFFERS WITH A THREADGROUP ARRAY, measured from two CPU controls compiled from the
# delivered sources with every declaration kept active so Apple retains all three. The cooperative
# four-binding class refuses these by name; this is the same family at three records.
#
# WHAT EACH CONTROL FIXES, and what it does not. Slot 28 is the DECLARED threadgroup bytes - 1024
# for float[256], 256 for uint[64] - and the name table's second field is the ELEMENT count, 256
# and 64, so the class carries both the extent and the count and neither is derived from the other.
# The register count moves slot 0 alone: a control at rc10 is four bytes larger only because its
# instruction count crosses the slot-32 boundary, which `slot32_for` already owns, and both
# delivered contracts sit below it at fifteen instructions.
#
# STATED AS THE ONE-REGISTER BASE so `with_system_registers` derives the two-register form rather
# than a second layout being transcribed; the derivation reproduces both controls byte for byte.
_TG3_SLOTS = {0: 60, 1: 48, 2: 44, 3: 36, 4: 40, 6: 32, 8: 28, 10: 24, 12: 20, 13: 16,
              15: 59, 16: 58, 18: 57, 26: 12, 27: 8, 28: 52, 29: 4}

# THE THREADGROUP NAME TABLE STATES A FIXED 32-BIT SCRATCH WORD AND A COUNT OF THOSE WORDS.
# Field 1 is 4 and field 2 is the count; the count is the DECLARED EXTENT OVER FOUR, and it is
# neither the source array's element count nor the extent over the declared alignment. Five CPU
# controls separate the three readings, each changing one declaration and nothing else:
#
#     threadgroup float s0[256]                1024 bytes   field2 = 256   count 256   words 256
#     threadgroup half  s0[512]                1024 bytes   field2 = 256   count 512   words 256
#     threadgroup half  s0[256]                 512 bytes   field2 = 128   count 256   words 128
#     threadgroup uchar s0[1024]               1024 bytes   field2 = 256   count 1024  words 256
#     threadgroup float s0[256] aligned(16)    1024 bytes   field2 = 256   256/16 = 64
#
# Rows 2, 3 and 4 refute the element count; row 5 refutes the extent over the alignment, and its
# section is BYTE-IDENTICAL to row 1's, so the declared alignment is not encoded here at all.
# FOUR_THREADGROUP's own pair - 128 bytes, field 2 = 32 - reads the same way. The earlier version
# of this file computed the count as extent//alignment, which agreed with every witness only
# because each one declared a four-byte element at alignment four.
SCRATCH_WORD_BYTES = 4


def _tg_three(size, bind, threadgroup_bytes):
    import copy as _copy
    k = _copy.deepcopy(FOUR_THREADGROUP)
    k.update(size=size, pk=132, pkvt=68, pk_vlen=64, pk_tlen=64, pk_slots=dict(_TG3_SLOTS),
             vec_bind=316, vec2=332, vec0=212, v0=(232, 220), cpname=(256, 260), name=(52, 56),
             ptrs={13: 288, 27: 196, 29: 200},
             v2=[(380, 368, "long", 18), (356, 346, "short")],
             v2_vals=[[6, 2, 6], [3, 6, None]],
             pk_extra={15: ("<B", 1), 16: ("<B", 1), 18: ("<B", 1), 0: ("<I", 1),
                       1: ("<I", 8), 28: ("<I", threadgroup_bytes)},
             nametab=(40, 30, 10, 12, {1: 8, 2: 4},
                      {1: ("<I", SCRATCH_WORD_BYTES),
                       2: ("<I", threadgroup_bytes // SCRATCH_WORD_BYTES)}),
             fills=((20, "<I", 20), (288, "<I", 8)), bind=bind)

    def down(value):
        if isinstance(value, dict):
            return {a: down(b) for a, b in value.items()}
        if isinstance(value, tuple):
            return tuple(down(b) for b in value)
        if isinstance(value, list):
            return [down(b) for b in value]
        if isinstance(value, int) and not isinstance(value, bool) and value > 200:
            return value - 4
        return value

    for key in ("vec_bind", "vec2", "vec0", "size", "v0", "cpname", "bind", "v2", "ptrs", "fills"):
        k[key] = down(k[key])
    k["slot29_vector"] = 200
    k["slot29_counts"] = (1, 2)
    k["q"] = k["vec_bind"] - (k["pk"] + k["pk_slots"][4]) + 4
    return k


# THE THREE-BINDING CLASS WITH A BACK EDGE (MM 25.140.4), derived from the flat layout by the delta MEASURED
# between each straight-line control and its one-edit loop variant (results/g17-coop3-loopclass-v1: the final
# store becomes a runtime-bounded loop; 31 and 32 instructions, so both carry slot 32 = 3 and slot 33 = 1):
#
#     the flat class carries NO slot 32 (its controls sit below the boundary); the loop adds slots 32 AND 33
#     the vtable grows to 34 slots (vlen 64 -> 72), so the table moves 132 -> 140; the table grows by four
#     (tlen 64 -> 68): slot 32 at 60, slot 33 at 59, slots 0, 15, 16, 18 up four; every structure at or after
#     the flat table's end (196) moves up twelve; section 468 -> 480 (write at 1), 472 -> 484 (write at 0)
#
# As for the four-binding loop class, the arithmetic is only how the candidate is produced; the claim is that
# tools/g17coop3loopclass.py authors both Apple sections byte-for-byte.
THREE_LOOP_PK_SLOTS = {0: 64, 15: 63, 16: 62, 18: 61, 32: 60, 33: 59}
THREE_LOOP_SHIFT_AFTER = 196
THREE_LOOP_SHIFT = 12


def _tg_three_loop(flat):
    import copy as _copy
    fixed = {"pk", "pkvt", "pk_vlen", "pk_tlen", "pk_slots", "pk_extra", "size", "root", "rvt", "root_vlen",
             "root_tlen", "arch32", "v2_vals", "nametab", "name", "q", "slot29_counts"}

    def move(value):
        if isinstance(value, dict):
            return {k: move(v) for k, v in value.items()}
        if isinstance(value, tuple):
            return tuple(move(v) for v in value)
        if isinstance(value, list):
            return [move(v) for v in value]
        if isinstance(value, int) and not isinstance(value, bool) and value >= THREE_LOOP_SHIFT_AFTER:
            return value + THREE_LOOP_SHIFT
        return value

    out = {k: (_copy.deepcopy(v) if k in fixed else move(v)) for k, v in flat.items()}
    out["pk_vlen"] = flat["pk_vlen"] + 8
    out["pk"] = flat["pk"] + 8
    out["pk_tlen"] = flat["pk_tlen"] + 4
    slots = dict(flat["pk_slots"])
    slots.update(THREE_LOOP_PK_SLOTS)
    out["pk_slots"] = slots
    out["pk_extra"] = dict(flat["pk_extra"])
    out["pk_extra"][32] = ("<B", 3)
    out["pk_extra"][33] = ("<B", 1)
    out["size"] = flat["size"] + THREE_LOOP_SHIFT
    out["q"] = out["vec_bind"] - (out["pk"] + out["pk_slots"][4]) + 4
    return out


def threadgroup_three(threadgroup_bytes, written_index, instructions=None, back_edge=False):
    """The three-binding threadgroup class, selected by which buffer the contract writes.

    `written_index` is 0 or 1; each is witnessed by its own control. Any other write position has
    no witness and is refused rather than served by whichever shape is nearer.

    The scratch-word count is DERIVED from the declared extent here rather than accepted from a
    caller, so the two numbers the section carries for one declaration cannot disagree.
    """
    if threadgroup_bytes % SCRATCH_WORD_BYTES:
        raise ValueError("this class states the threadgroup extent as a count of %d-byte scratch "
                         "words; %d bytes is not a whole number of them"
                         % (SCRATCH_WORD_BYTES, threadgroup_bytes))
    if written_index == 1:
        bind = [(460, 454, "elided"), (436, 424, "long", 18), (408, 398, "mid", 16)]
        size = 468
    elif written_index == 0:
        bind = [(464, 452, "elided_written"), (436, 426, "mid", 16), (408, 398, "mid", 18)]
        size = 472
    else:
        raise ValueError("the three-binding threadgroup class is measured for the write at buffer "
                         "0 or buffer 1; %r is witnessed by neither control" % (written_index,))
    flat = _tg_three(size, bind, threadgroup_bytes)
    if not back_edge:
        return flat
    # the loop form is witnessed ABOVE the slot-32 boundary only (slot 32 = 3); a looping program below it
    # would carry slot 33 without slot 32 - a shape no control here has, so it is refused
    if instructions is None or slot32_for(instructions, True) != 3:
        raise ValueError("the three-binding loop class is witnessed at slot 32 = 3 (31 or more instructions); "
                         "%r instructions with a back edge is unwitnessed" % (instructions,))
    return _tg_three_loop(flat)


def layout_for(indices, promoted_ranges=None, written=None, instructions=None,
               back_edge=False, system_registers=None):
    """The measured layout for these binding indices, or None to refuse.

    `system_registers` is the contract's declared set, consulted only where two measured classes
    share a shape and differ in whether the program reads a system value - the buffer-zero
    four-binding pair. Left None it selects as it always did.

    Binding COUNT alone does not determine the class. Two three-binding classes are measured and
    they differ in whether buffer 0 is bound, because the elided record - the only shape with no
    index field - can name nothing else. Picking by count alone would author a record naming
    buffer 0 for a contract that binds 1, 2 and 3.

    `promoted_ranges` is the contract's own statement, not a count: six DECLARED buffers with one
    promoted emit FIVE binding records, so the six-buffer promoted class is selected by the range
    the contract declares and never by the length of the binding list. Selecting it from the count
    alone would hand a five-buffer contract the promoted class and emit a resource record for a
    range nothing declared.
    """
    n = len(indices)
    if n == 1:
        # ONE MEASURED ONE-BINDING CLASS, and it is measured for one signature: buffer 0, written.
        # Anything else at this count - a different index, a read-only binding - has no witness and
        # is refused rather than served this one, because the record shape here can name no buffer
        # but 0 and marks a write it might not have.
        if tuple(indices) != (0,):
            return None
        # THE MASK MUST BE STATED. This class's single record carries slot 3, which MARKS A WRITE,
        # so selecting it for a contract that has not said its buffer is written would author a
        # declaration the contract never made. `None` here means "not stated", not "no writes".
        if written is None or tuple(bool(w) for w in written) != (True,):
            return None
        # THE BOUNDARY IS NOT AN EXTRAPOLATION. The longer control crosses it and is a different,
        # 404-byte layout carrying slot 32; this class is witnessed below it only. A BACK EDGE is a
        # third layout again - slot 33 present - and all three controls here are straight-line, so
        # it is refused rather than served the flat form.
        if back_edge:
            return None
        if instructions is None or slot32_for(instructions, back_edge) is not None:
            return None
        return ONE_WRITTEN_ZERO
    if n == 2:
        # BOTH WRITTEN IS A DIFFERENT CLASS, and the count cannot tell. SCALAR is the nine-slot
        # minimal shape with no slot 0, so a contract needing a per-program register count is
        # refused there by name; the two-writable class carries one. `written` is the contract's
        # own statement, not a guess from the binding count.
        if written is not None and tuple(written) == (True, True):
            if 0 in tuple(indices):
                return None
            # THE LENGTH SELECTS BETWEEN TWO MEASURED CLASSES, and the contract's own instruction
            # count is what says which. A program past the slot-32 boundary carries one and the
            # short class has nowhere to put it; a program that loops carries a slot 33 and neither
            # class is witnessed for that.
            if instructions is None:
                return TWO_WRITABLE
            band = slot32_for(instructions, back_edge)
            if band is None:
                return TWO_WRITABLE
            if back_edge:
                return None
            return TWO_WRITABLE_LONG
        return SCALAR
    if n == 3:
        # Both measured layouts place the only write-capable record last. A
        # different write pattern needs another measured layout; selecting these
        # would give downstream contract validation the wrong write mask.
        # Apple's syn-s5ed5557afe witness has a middle-written 488-byte class,
        # including Apple-specific spill state. That does not establish the
        # appropriate no-spill layout for our independently compiled program.
        mask = None if written is None else tuple(bool(w) for w in written)
        # A SECOND MASK IS WITNESSED NOW, and only where buffer 0 is bound. `elided_written` is the
        # only shape that can both name buffer 0 and mark it written, so the class it belongs to is
        # the class for a contract that writes buffer 0 and buffer 2. Its controls are
        # results/g17-threebinding-write02-v1; a contract binding 1, 2, 3 has no such record and
        # nothing witnesses this mask there.
        if mask == (True, False, True) and 0 in tuple(indices):
            return THREE_WITH_WRITTEN_ZERO
        # ALL THREE WRITTEN with buffer 0 bound: Apple's (0,1,2) class (item 11). elided_written names
        # buffer 0 and marks its write; the two longs carry index+offset+written. Measured byte-exact.
        if mask == (True, True, True) and 0 in tuple(indices):
            return SEPARATE_ALL_WRITTEN
        if mask is not None and mask != (False, False, True):
            return None
        if (tuple(indices) == (0, 1, 2) and mask == (False, False, True)
                and tuple(system_registers or ()) == (160,)):
            return THREE_SR160
        if 0 in tuple(indices) and tuple(system_registers or ()) == COORDINATE_REGISTERS:
            return THREE_COORDINATE
        return THREE if 0 in tuple(indices) else SEPARATE
    if n == 4:
        # Four USER buffers. Buffer 0 needs the ELIDED-first variant, because the elided shape is
        # the only record with no index field and so the only one that can name index 0; FOUR's
        # first record is "short" and would name the wrong buffer. That variant is measured now -
        # FOUR_WITH_ZERO, 428 bytes, reproduced byte-for-byte from the four-active volatile control.
        if 0 in tuple(indices):
            # ONE WRITE MASK IS WITNESSED, the last. Both four-binding layouts fix a record SHAPE
            # per position and only "long" carries slot 3, so a contract whose written buffer is
            # not the last one would author a section naming the WRONG buffer as written - the
            # same defect the three-binding branch above refuses. The control writes buffer 3 and
            # establishes that mask and no other.
            # TWO MASKS ARE WITNESSED NOW, each by its own control, and the mask selects the
            # class rather than a field inside one: only "long" carries slot 3, so where the
            # write sits is where that shape sits. Any other mask still has no witness.
            # ALL FOUR SINGLE-WRITE POSITIONS ARE WITNESSED NOW, each by its own CPU control,
            # and the mask selects the CLASS rather than a field inside one: only a shape carrying
            # slot 3 marks a write, so where the write sits is where that shape sits. Two or more
            # writes, or none, still have no witness and are refused.
            mask = None if written is None else tuple(bool(w) for w in written)
            if mask is None:
                return FOUR_WITH_ZERO
            # THE DECLARED REGISTER SET SELECTS BETWEEN TWO MEASURED CLASSES of the same shape,
            # because Apple emits different sections for them: the empty-vector class was
            # measured from controls reading no system value, and the one-entry class from
            # controls that do. Choosing by mask alone put every register-reading program into
            # the empty class. A mask with no SR witness returns its empty class regardless, and
            # that class's stated count of zero is what refuses the declaration downstream.
            declared = bool(system_registers)
            return {(False, False, False, True): FOUR_WITH_ZERO_SR if declared else FOUR_WITH_ZERO,
                    (False, False, True, False): FOUR_WITH_ZERO_MIDWRITE,
                    (False, True, False, False): FOUR_WITH_ZERO_WRITE1,
                    (True, False, False, False): (FOUR_WITH_ZERO_WRITE0_SR if declared
                                                  else FOUR_WITH_ZERO_WRITE0)}.get(mask)
        return FOUR
    if n == 5:
        # FIVE RECORDS ARE TWO CLASSES NOW, and the contract says which. Five bound buffers is the
        # measured five-buffer class; five bound buffers with a declared promoted range is the
        # SIX-buffer class, which emits no record for the promoted buffer. The count cannot tell
        # them apart and is not asked to.
        if 0 in tuple(indices):
            return None
        return SIX_PROMOTED if promoted_ranges else FIVE
    if n == 6:
        # Six bound buffers and no promoted range. A contract that declares six AND promotes one
        # binds five, which is the branch above; reaching here with a range means the contract
        # states a promotion whose buffer it also binds, and that is refused rather than resolved.
        return SIX if 0 not in tuple(indices) and not promoted_ranges else None
    return None


# One measured layout per binding count. A count with no entry is REFUSED rather than approximated.
BY_BINDINGS = {2: SCALAR, 3: THREE}

Q_SCALAR = SCALAR["q"]
Q_TENSOR = TENSOR["q"]


def main(argv=None):
    from . import imgconst_scalar as g17imgconst_scalar
    from . import imgconst as g17imgconst
    from . import bindenc as g17bindenc
    for name, layout, want, binds in (
            ("scalar", SCALAR, g17bindenc.encode(g17imgconst_scalar.GPU_METADATA, [2, 1]), [2, 1]),
            ("tensor", TENSOR, g17imgconst.GPU_METADATA, [0, 2])):
        got = build(binds, layout=layout)
        # The blob these are checked against was SWEPT - four of the per-kernel table's offset
        # slots were zeroed because a store to a device buffer never reaches them. build() writes
        # them from the layout now, so those four positions are expected to differ and are named
        # rather than excluded silently.
        filled = set()
        for sl in (6, 8, 10, 12, 26):
            if sl in layout["pk_slots"]:
                off = layout["pk"] + layout["pk_slots"][sl]
                filled.update(range(off, off + 4))
        if layout.get("vec0") is not None:          # the vector the sweep erased entirely
            filled.update(range(layout["vec0"], layout["vec0"] + 8))
            filled.update(range(layout["v0"][1], layout["v0"][0] + V0REC[1]))
        bad = [i for i in range(layout["size"]) if got[i] != want[i] and i not in filled]
        refilled = sorted(i for i in range(layout["size"]) if got[i] != want[i] and i in filled)
        print("__GPU_METADATA %-7s %s; %d bytes restored where the sweep zeroed offsets" % (name,
              "byte-identical to the measured blob (%d bytes, %d non-zero)"
              % (len(want), sum(1 for v in want if v)) if not bad
              else "DIFFERS at %s\n  got  %s\n  want %s"
              % (bad[:16], bytes(got[i] for i in bad[:16]).hex(" "),
                 bytes(want[i] for i in bad[:16]).hex(" ")), len(refilled)))


# ---------------------------------------------------------------------------------------------
# ANY CLASS, DESCRIBED AND RE-EMITTED.
#
# SCALAR and TENSOR above are hand-written layouts, which was fine for two classes and does not
# scale: the convolution objects are a third shape with three bindings and a richer per-kernel
# table, and the goal as set covers whatever the ISA allows rather than two chosen paths. So this
# WALKS a measured blob into a description and re-emits from the description.
#
# The description is structure, not bytes: table positions, vtable lengths, slot maps, scalar
# widths and the vectors' record lists. Nothing is copied - build_from writes each field from the
# description's value, and byte-identity against the blob it was described from is the check that
# the walk saw everything.
import struct as _struct


def describe(md):
    """-> a layout description of a __GPU_METADATA blob: every table, vector and scalar in it."""
    tables, vectors, seen = {}, {}, set()

    def read_table(pos):
        if pos in seen or not (4 <= pos < len(md)):
            return None
        seen.add(pos)
        vt = _struct.unpack_from("<i", md, pos)[0]
        vtpos = pos - vt
        if not (0 <= vtpos < len(md) - 4):
            return None
        vlen, tlen = _struct.unpack_from("<HH", md, vtpos)
        if not (4 <= vlen <= 400) or vtpos + vlen > len(md):
            return None
        # A VTABLE CAN BE SHARED, AND THEN IT DOES NOT PRECEDE ITS TABLE. FlatBuffers dedupes
        # identical vtables, so a second table with the same field layout points BACK at the first
        # one's - a negative soffset, and vtpos + vlen != pos. Six vertex sections do exactly that:
        # mg-vs_basevertex-1 has a vector of three at 280 whose second record is at 328 with
        # soffset -16, sharing the vtable at 344 that the table at 356 owns. Rejecting it rejected
        # the whole vector, and the 22 bytes it describes stayed residual - which is where the last
        # byte of carried semantics lived.
        shared = vtpos + vlen != pos
        if shared and vtpos not in {t["vtpos"] for t in tables.values()}:
            return None
        slots = [_struct.unpack_from("<H", md, vtpos + 4 + 2 * i)[0] for i in range((vlen - 4) // 2)]
        fields = {}
        for i, off in enumerate(slots):
            if not off:
                continue
            nxt = sorted(x for x in slots if x > off)
            gap = (nxt[0] - off) if nxt else (tlen - off)
            # THE GAP TO THE NEXT SLOT IS NOT THE FIELD'S WIDTH. FlatBuffers aligns fields, so a
            # one-byte value followed by padding shows a gap of six or seven and used to be skipped
            # entirely - which is where most of the bytes this walk could not describe came from.
            #
            # NARROWING EVERY INEXACT GAP TO ONE BYTE IS ALSO WRONG, and it hid a field. The
            # per-kernel table's slot 28 sits at a four-aligned offset with a gap of five - four
            # bytes of value and one of padding before the byte-wide flags that follow - and
            # reading it as one byte left its top three bytes in `extra`, where they became class
            # constants copied from a donor. They are the STATIC THREADGROUP ALLOCATION: 1024,
            # 2048 and 4096 for one, two and four `threadgroup uint t[256]` arrays, and 2048 for
            # mm-f16.tg against 4096 for mm-f32.tg, which declare the same two arrays and use
            # different ones. Alignment is what decides the width: a field is as wide as the
            # widest of four, two and one that fits the gap AND is legal at its own offset.
            w = next((c for c in (4, 2, 1) if c <= gap and (pos + off) % c == 0), 1)
            if pos + off + w <= len(md):
                fields[i] = (off, w, _struct.unpack_from({1: "<B", 2: "<H", 4: "<I"}[w], md, pos + off)[0])
        tables[pos] = dict(vtpos=vtpos, vlen=vlen, tlen=tlen, shared=shared,
                           slots={i: o for i, o in enumerate(slots) if o}, fields=fields,
                           tail=b"")
        return tables[pos]

    root = _struct.unpack_from("<I", md, 0)[0]
    rt = read_table(root)
    order = [root]
    for i, off in sorted(rt["slots"].items()):
        target = root + off + _struct.unpack_from("<I", md, root + off)[0]
        if read_table(target):
            order.append(target)
        else:                                    # not a table: a vector of table references
            n = _struct.unpack_from("<I", md, target)[0]
            if 1 <= n <= 31:
                recs = [target + 4 + 4 * k + _struct.unpack_from("<I", md, target + 4 + 4 * k)[0]
                        for k in range(n)]
                if all(read_table(r) for r in recs):
                    vectors[target] = recs
                    order.extend(recs)
    # Any vector the root does not reach - the binding table is reached this way in the swept
    # blobs, whose root slot 3 was zeroed - is found by scanning for a plausible count followed by
    # offsets that all land on tables.
    for pos in range(0, len(md) - 8, 4):
        if pos in vectors or pos in tables:
            continue
        n = _struct.unpack_from("<I", md, pos)[0]
        if not (1 <= n <= 31) or pos + 4 + 4 * n > len(md):
            continue
        recs = []
        for k in range(n):
            off = _struct.unpack_from("<I", md, pos + 4 + 4 * k)[0]
            q = pos + 4 + 4 * k + off
            if off == 0 or not (pos < q < len(md)):
                recs = None; break
            recs.append(q)
        if not recs or len(set(recs)) != n:
            continue
        if all(r in tables or read_table(r) for r in recs):
            vectors[pos] = recs
            order.extend(r for r in recs if r not in order)
    # A TABLE'S DECLARED LENGTH UNDER-STATES ITS BODY. The per-kernel table declares zero and
    # occupies sixty bytes; the constant-program symbol's table declares twenty and is followed by
    # twenty more bytes that belong to it. Trusting the declaration leaves those bytes undescribed,
    # so each table's body is extended to whatever comes next and the extra is carried as a tail.
    edges = sorted(set([len(md)] + [t["vtpos"] for t in tables.values()]
                       + list(vectors) + list(tables)))
    for pos, t in tables.items():
        # THE BODY IS tlen. A slot offset can exceed the declared table length - c4probe has a
        # table with tlen 12 whose highest slot reaches 15 - and taking the larger makes the table
        # overrun the one after it by three bytes. A slot past tlen points into what FOLLOWS,
        # which is the same rule as everywhere else in this format: a declared length is the
        # length, and what comes after belongs to what comes after.
        body = pos + t["tlen"]
        # A TABLE WHOSE BODY IS EMPTY still owns what follows it. Taking the first edge at or
        # after the body finds the table's own position and yields no tail at all, which left the
        # bytes after a zero-length table undescribed. The edge has to be one that comes AFTER this
        # table starts.
        nxt = min([e for e in edges if e > pos and e >= body] or [len(md)])
        if nxt > body:
            t["tail"] = bytes(md[body:nxt])
    desc = dict(size=len(md), root=root, tables=tables, vectors=vectors, order=order, extra={})
    # RESIDUAL BYTES, COUNTED RATHER THAN HIDDEN. A field whose width is not 1, 2 or 4 - the last
    # slot of a table, whose width comes from the declared inline length - and anything the walk
    # does not reach are recorded as raw bytes with their offsets. Byte-identity is then guaranteed
    # and the number of residual bytes is the honest measure of how much of the class the
    # description actually explains.
    got = build_from(desc)
    desc["extra"] = {i: md[i] for i in range(len(md)) if got[i] != md[i]}
    return desc


def build_from(desc, values=None):
    """Re-emit a described blob. `values` overrides (table_pos, slot) -> value."""
    values = values or {}
    b = bytearray(desc["size"])
    _struct.pack_into("<I", b, 0, desc["root"])
    for pos, t in desc["tables"].items():
        _struct.pack_into("<HH", b, t["vtpos"], t["vlen"], t["tlen"])
        # The soffset is the distance BACK to the vtable, which equals vlen only when the vtable
        # immediately precedes the table. A shared vtable can follow it, and then this is negative.
        _struct.pack_into("<i", b, pos, pos - t["vtpos"])
        for slot, off in t["slots"].items():
            _struct.pack_into("<H", b, t["vtpos"] + 4 + 2 * slot, off)
        for slot, (off, w, v) in t["fields"].items():
            _struct.pack_into({1: "<B", 2: "<H", 4: "<I"}[w], b, pos + off,
                              values.get((pos, slot), v))
        tail = t.get("tail") or b""
        if tail:
            # The body is tlen; see describe(). A slot past tlen points into what follows.
            body = pos + t["tlen"]
            b[body:body + len(tail)] = tail
    for vec, recs in desc["vectors"].items():
        _struct.pack_into("<I", b, vec, len(recs))
        for k, r in enumerate(recs):
            _struct.pack_into("<I", b, vec + 4 + 4 * k, r - (vec + 4 + 4 * k))
    for off, v in desc.get("extra", {}).items():
        b[off] = v
    return bytes(b)


# ---------------------------------------------------------------------------------------------
# THE SECTION COMPOSED, for any number of bindings.
#
# SCALAR and TENSOR are measured positions, and `describe`/`build_from` re-emit a blob that was
# already measured. Neither can produce a section for a binding count nobody has compiled, which is
# the whole of what a general linker needs.
#
# What makes composing possible is that the loader WALKS this document rather than reading fixed
# offsets: relocating every table, vector and record sixteen bytes later inside a larger section
# leaves the kernel correct, while exchanging the two binding indices sends the store to the wrong
# buffer and the answer comes back as the untouched sentinel - so the observable is reading the
# records, and it found them where they moved to (spike/accel/re/mdshift.py).
#
# AND Q IS NOT A MYSTERY, IT IS THIS LAYOUT. The number neither session could name - per-kernel, a
# multiple of four, correlated with nothing semantic, and exact in the sense that only the right
# value executes - is the DISTANCE FROM THE PER-KERNEL TABLE TO THE BINDING VECTOR:
#
#     binding vector = per-kernel table + Q + 36        7,403 of 7,414 cached objects
#
# which is why moving the vector alone breaks the binding, moving the per-kernel table alone
# crashes the loader, moving the root alone is harmless, and moving everything together is fine.
# The loader does not follow a pointer to the records; it computes their position from a field.
#
# So this lays the document out forward from the root, sized by the signature, and Q is COMPUTED
# from where the vector landed rather than carried by class.
PK_SLOTS = {26: 12, 13: 16, 10: 24, 8: 28, 6: 32, 3: 36, 4: 40, 2: 44, 1: 48}
PK_VLEN, PK_BODY = 70, 52
# Apple declares the per-kernel table's inline length as ZERO while its body is 52 bytes. That is
# not what a FlatBuffers writer emits and it is what the loader accepts, so it is reproduced rather
# than corrected.
PK_TLEN = 0
ROOT_VLEN, ROOT_BODY = 12, 12
_BODY = {"long": 16, "short": 12, "elided": 8}


def _align(n, a=4):
    return (n + a - 1) & ~(a - 1)


# Q's relation to the binding vector, written the way FlatBuffers defines a reference. The
# table-relative reading (pk + Q + 36) is this same equation with slot 4's offset of 40 folded in,
# and it is wrong for the 38 objects whose serialiser put that field somewhere else.
QOFF = 36            # kept only for readers of the older ledger entries


def for_bindings(bindings, base=None, shapes=None):
    """A class layout re-laid for a DIFFERENT number of bindings.

    Two positions are not free and both are laws measured across the cache, not choices:

        slot 4  refers to the binding vector        address of slot 4 + the u32 it holds
        slot 2  refers to the last vector            all 7,557 cached objects
        slot 26 refers to the first vector

    all three being ordinary FlatBuffers references - the stored u32 is relative to the FIELD, not
    the table. The last vector immediately follows the binding vector in every cached object, which
    is an observed adjacency rather than a requirement, and it is kept because moving it while the
    binding count changed is what made three bindings segfault a document that verified clean. The
    first vector, which slot 26 locates, keeps its position below them.
    """
    L = dict(SCALAR if base is None else base)
    shapes = ["elided" if b == 0 else "long" for b in bindings] if shapes is None else list(shapes)
    body = {"long": 16, "short": 12, "elided": 8}
    nb = len(bindings)
    L["vec2"] = L["vec_bind"] + 4 + 4 * nb
    cur = _align(L["vec2"] + 4 + 4 * len(L["v2"]))
    v2 = []
    for r in L["v2"]:
        shape = r[2]
        vlen = _V2REC[shape][0]
        cur = ((cur + vlen + 3) & ~3) - vlen
        vt, cur = cur, cur + vlen
        pos, cur = cur, cur + (16 if shape == "long" else 12)
        v2.append(tuple([pos, vt] + list(r[2:])))
    L["v2"] = v2
    bind = []
    for shape in shapes:
        vlen = _REC[shape][0]
        cur = ((cur + vlen + 3) & ~3) - vlen        # so the record BODY lands on four
        vt, cur = cur, cur + vlen
        pos, cur = cur, cur + body[shape]
        bind.append((pos, vt, shape))
    L["bind"] = bind
    L["size"] = max((cur + 15) & ~15, _align(L["vec_bind"], 16))
    # Q is the value slot 4 carries, and slot 4 is a reference: target minus the field's address.
    L["q"] = L["vec_bind"] - (L["pk"] + L["pk_slots"][4]) + 4
    return L


def compose(bindings, q=None, second=None, shapes=None, pad=0, root_reaches=True):
    """__GPU_METADATA for an arbitrary signature: one binding record per bound buffer.

    `shapes` picks each record's shape; the default is the one Apple uses for a non-zero index.
    `root_reaches` writes the root's slot-3 pointer at the binding vector - Apple leaves it zero
    and the records are still read, so this says whether the pointer is used or ignored.
    `q` defaults to whatever the layout requires; passing one is how a wrong Q is tested.
    """
    second = [(6, 20, 4), (3, 4, None)] if second is None else list(second)
    shapes = ["elided" if i == 0 and b == 0 else "long"
              for i, b in enumerate(bindings)] if shapes is None else list(shapes)

    # EVERY TABLE BODY LANDS ON FOUR. The scalar values inside a table are u32 at a slot offset,
    # so a table body at an odd multiple of two makes every one of them a misaligned read - which
    # is what the first composed document did, and it bound nothing.
    cur = 4
    rvt, cur = cur, cur + ROOT_VLEN
    root, cur = cur, cur + ROOT_BODY
    cur = _align(cur + PK_VLEN) - PK_VLEN
    pkvt, cur = cur, cur + PK_VLEN
    pk, cur = cur, cur + PK_BODY
    cur = _align(cur)
    vec_bind, cur = cur, cur + 4 + 4 * len(bindings)
    bind = []
    for shape in shapes:
        vlen = _REC[shape][0]
        cur = _align(cur + vlen) - vlen          # so the record BODY lands on four
        vt, cur = cur, cur + vlen
        pos, cur = cur, cur + _BODY[shape]
        bind.append((pos, vt, shape))
    cur = _align(cur)
    vec2, cur = cur, cur + 4 + 4 * len(second)
    v2 = []
    for vals in second:
        shape = "long" if vals[2] is not None else "short"
        vlen = _V2REC[shape][0]
        cur = _align(cur + vlen) - vlen
        vt, cur = cur, cur + vlen
        pos, cur = cur, cur + (16 if shape == "long" else 12)
        v2.append((pos, vt, shape))
    size = _align(cur + pad, 16)
    q = (vec_bind - pk - QOFF) if q is None else q

    b = bytearray(size)
    struct.pack_into("<I", b, 0, root)
    _put_table(b, root, rvt, ROOT_VLEN, ROOT_BODY, {0: 8, 3: 4},
               {0: ("<I", pk - (root + 8)),
                3: ("<I", (vec_bind - (root + 4)) if root_reaches else 0)})
    _put_table(b, pk, pkvt, PK_VLEN, PK_TLEN, PK_SLOTS,
               {3: ("<I", 8 * len(bindings)), 4: ("<I", q - 4),
                2: ("<I", q - 4 + 4 * len(bindings))})
    _put_vector(b, vec_bind, [r[0] for r in bind])
    for (pos, vtpos, shape), idx in zip(bind, bindings):
        vlen, tlen, slots = _REC[shape]
        vals = {0: ("<B", 5)}
        if shape == "elided":
            if idx:
                raise ValueError("an elided record can only name buffer 0; %d requested" % idx)
        else:
            vals[1] = ("<I", idx)
        if shape == "long":
            vals.update({2: ("<I", 2), 3: ("<B", 1)})
        _put_table(b, pos, vtpos, vlen, tlen, slots, vals)
    _put_vector(b, vec2, [r[0] for r in v2])
    for (pos, vtpos, shape), vals in zip(v2, second):
        vlen, tlen, slots = _V2REC[shape]
        v = {0: ("<B", vals[0]), 2: ("<I", vals[1])}
        if shape == "long":
            v[3] = ("<I", vals[2])
        _put_table(b, pos, vtpos, vlen, tlen, slots, v)
    return bytes(b)


# ---------------------------------------------------------------------------------------------
# THE SAME SECTION AS AN OBJECT GRAPH. No offset formulas: tables, vectors and records with typed
# references between them, placed by g17schema and resolved from each field's own address. What
# used to be "binding vector = per-kernel table + Q + 36" is now slot 4 holding a Ref, and Q is
# whatever that reference happens to serialise to.
from . import schema as _sch

# Each binding record shape as (slot -> field offset, declared inline length, body size).
_REC_SHAPE = {
    "long":   ({0: 10, 1: 4, 2: 12, 3: 11}, 16, 16),
    "short":  ({0: 11, 1: 4}, 12, 12),
    "elided": ({0: 7}, 8, 8),
}
_V2_SHAPE = {
    "long":  ({0: 11, 2: 12, 3: 4}, 16, 16),
    "short": ({0: 7, 2: 8}, 12, 12),
}
# THE FIRST VECTOR'S RECORD IS READ PAST ITS FIELDS. Its declared inline length is 20 and its
# fields end there, and a document packed with no space after it segfaults the loader; giving that
# ONE node room runs, and giving the same room to any other node does not - checked node by node at
# two binding counts. Twelve bytes past its fields is enough for two and three bindings and
# twenty-four for one, so the extent is 44 here. WHY it varies with the binding count is not
# explained; what is explained is that the slack belongs to this node and not to the layout, which
# is what the old twelve-bytes-between-everything gap was hiding.
V0_SHAPE = ({0: 16, 1: 15, 2: 8, 3: 4}, 20, 44)


def graph(bindings, klass=None, shapes=None, second=None, q_slots=(6, 8, 10, 12),
          pk_values=None):
    """The document for one signature: a root, a per-kernel table, three vectors and their records.

    Nothing in here is a position. The only inputs are the signature and which class's per-kernel
    table shape to use, and the class contributes its slot map rather than a set of addresses.
    """
    L = SCALAR if klass is None else klass
    # THE RECORD SHAPES ARE THE CLASS'S when the count matches. "long" for everything looked free -
    # a record with more fields can say anything a shorter one can - and the corpus comparison says
    # otherwise: packaged with two long records, kernels that agree with Apple's archive under the
    # class's own [short, long] stop agreeing.
    if shapes is None:
        shapes = ([r[2] for r in L["bind"]] if len(L["bind"]) == len(bindings)
                  else ["elided" if b == 0 else "long" for b in bindings])
        shapes = [("elided" if b == 0 and sh != "elided" else sh)
                  for sh, b in zip(shapes, bindings)]
    else:
        shapes = list(shapes)
    second = L["v2_vals"] if second is None else list(second)

    recs = []
    for shape, idx in zip(shapes, bindings):
        slots, tlen, size = _REC_SHAPE[shape]
        fields = {0: ("<B", 5)}
        if shape != "elided":
            fields[1] = ("<I", idx)
        elif idx:
            raise ValueError("an elided record can only name buffer 0; %d requested" % idx)
        if shape == "long":
            fields.update({2: ("<I", 2), 3: ("<B", 1)})
        recs.append(_sch.Table(slots, fields, size, tlen, name="bind%d" % idx))
    bindvec = _sch.Vector([_sch.Ref(r) for r in recs], name="bindings")

    v2recs = []
    for vals, shape in zip(second, [r[2] for r in L["v2"]]):
        slots, tlen, size = _V2_SHAPE[shape]
        fields = {0: ("<B", vals[0]), 2: ("<I", vals[1])}
        if shape == "long":
            fields[3] = ("<I", vals[2])
        v2recs.append(_sch.Table(slots, fields, size, tlen, name="v2"))
    lastvec = _sch.Vector([_sch.Ref(r) for r in v2recs], name="last")

    slots, tlen, size = V0_SHAPE
    v0rec = _sch.Table(slots, {3: ("<I", 16), 2: ("<I", 1), 1: ("<B", 3), 0: ("<I", 8)},
                       size, tlen, name="v0rec")
    firstvec = _sch.Vector([_sch.Ref(v0rec)], name="first")

    pk_fields = {3: ("<I", 8 * len(bindings)), 4: ("<I", _sch.Ref(bindvec)),
                 2: ("<I", _sch.Ref(lastvec))}
    if 26 in L["pk_slots"]:
        pk_fields[26] = ("<I", _sch.Ref(firstvec))
    # THE PER-KERNEL TABLE DECLARES SIXTY BYTES, not zero. The swept blob says zero and the class
    # layout gets away with it because the class leaves seventy-two bytes of empty space after that
    # table anyway; a packed graph does not, and whatever reads the declared length walks into the
    # next node's vtable. Across the cache the declared length is 60 in 1,108 objects of 1,500, 64
    # in 361, and the gap to the next structure is never less than 68.
    # THE PER-KERNEL TABLE NEEDS ROOM PAST ITS FIELDS. It declares sixty bytes (the swept blob's
    # zero is another thing the sweep destroyed) and every object in the cache leaves at least
    # sixty-eight before the next structure, 72 being modal. Packed to its 52 bytes of fields the
    # loader segfaults; a gap sweep on the schema-placed document puts the boundary between 8 and
    # 12 bytes of slack, which is 68 and 72 - the corpus's own two numbers. So the node reserves 72
    # rather than every node being padded.
    # PER-KERNEL VALUES THE CALLER SUPPLIES. Slots 2, 3, 4 and 26 are references this function
    # resolves; everything else in the class's slot map got its offset here and no VALUE, so it
    # serialised as zero. A texture kernel needs slot 38 and its copy 42 to carry the texture-state
    # byte count, and a write set needs 15/16/17 - all of them facts a backend reports, none of them
    # derivable here. Passing them is how a caller says what its program is, and a slot named in
    # pk_values but absent from the class's slot map is a caller error rather than a silent drop.
    for _s, _v in (pk_values or {}).items():
        if _s not in L["pk_slots"]:
            raise KeyError("slot %d is not in this class's per-kernel slot map" % _s)
        pk_fields[_s] = ("<B", _v) if _s in (15, 16, 17, 18, 19, 30, 32, 33, 40, 43) else ("<I", _v)
    pk = _sch.Table(L["pk_slots"], pk_fields, 60, 60, name="perkernel")
    root = _sch.Table({0: 8, 3: 4}, {0: ("<I", _sch.Ref(pk)), 3: ("<I", 0)}, 12, 12, name="root")

    nodes = [root, pk, firstvec, v0rec, bindvec, lastvec] + v2recs + recs
    doc = _sch.Doc(root, nodes)
    # Slots 6, 8 and 10 hold the same NUMBER in every object and so refer to three different
    # places, four bytes apart, because their fields do. Apple's number is what slot 4 serialises
    # to, so they are filled after placement rather than pointed at anything named here.
    doc.q_slots = tuple(s for s in q_slots if s in L["pk_slots"])
    doc.pk = pk
    doc.bindvec = bindvec
    return doc


# NO GLOBAL SLACK IS NEEDED any more. It used to be twelve bytes between every node, which ran and
# explained nothing; reserving room on one node at a time and dispatching says it is the FIRST
# VECTOR'S RECORD and nothing else - see V0_SHAPE.
GRAPH_GAP = 0


def build_graph(bindings, klass=None, gap=GRAPH_GAP, order=None, size=None, reserve=None, **kw):
    """Place and serialise the graph. `gap`/`order`/`reserve` produce different legal layouts."""
    doc = graph(bindings, klass, **kw)
    place(doc, gap=gap, order=order, reserve=reserve)
    return _sch.emit(doc, size=size)


def place(doc, gap=0, order=None, reserve=None):
    end = _sch.place(doc, gap=gap, order=order, reserve=reserve)
    at = doc.pk.addr + doc.pk.slots[4]
    q = doc.bindvec.addr - at
    for s in doc.q_slots:
        doc.pk.fields[s] = ("<I", q)
    return end


# ---------------------------------------------------------------------------------------------
# CLASSES RECORDED AS STRUCTURE, one per binding count, in isa/g17-mdclass.json.
#
# SCALAR and TENSOR are hand-written layouts with two binding records each, and re-laying their
# binding region for another count executes a store-only kernel correctly and gets a DIFFERENT
# ANSWER from Apple's archive on the kernels that actually compute something. The store probe could
# not see it: one slot, one value, no way to tell a correct binding from a wrong one.
#
# So a class is recorded the way describe() reads it - tables, vectors, slot maps, field values and
# the residual bytes - and re-emitted with this kernel's own binding indices. It is a measurement,
# not a byte blob copied at build time, and it is the same standing SCALAR and TENSOR already have.
_CLASSES = None


def classes():
    global _CLASSES
    if _CLASSES is None:
        import json
        # THE ROOT ANCHOR, kept beside the read it serves. This file sits two directories
        # below the repository root (agxforge/g17/), where tools/g17mdgen.py sat one, so the walk is
        # three dirname() calls rather than two. Getting it wrong is silent: the path simply does
        # not exist, `classes()` returns an empty map, and every class it describes defaults away.
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "isa", "g17-mdclass.json")
        _CLASSES = {}
        if os.path.exists(path):
            for n, ser in json.load(open(path)).items():
                _CLASSES[n] = dict(
                    size=ser["size"], root=ser["root"], order=ser["order"],
                    tables={int(k): dict(vtpos=v["vtpos"], vlen=v["vlen"], tlen=v["tlen"],
                                         slots={int(a): b for a, b in v["slots"].items()},
                                         fields={int(a): tuple(b) for a, b in v["fields"].items()},
                                         tail=bytes.fromhex(v.get("tail", "")))
                            for k, v in ser["tables"].items()},
                    vectors={int(k): v for k, v in ser["vectors"].items()},
                    extra={int(k): v for k, v in ser["extra"].items()},
                    records=ser["binding_records"], key_opcodes=ser.get("key_opcodes"),
                    threadgroup=bool(ser.get("threadgroup")),
                    index_slot={int(k): v for k, v in ser["index_slot"].items()})
    return _CLASSES


# The two symbol names that appear in every class's residual bytes, length-prefixed. They are the
# entry point and the constant program - the same symbols this project's own object declares - so
# they are written from the program's names rather than carried as somebody else's bytes.
SYMBOLS = ("agc.main", "agc.main.constant_program")


def _write_names(blob, extra, names):
    """Replace the symbol names sitting in a class's residual bytes, and say which bytes they were.

    They are stored as a length-prefixed string but the prefix is a described field rather than a
    residual byte, so the run is found by its CONTENT: the residual bytes that spell a name this
    project also emits are that name, and they are ours to write.
    """
    out = bytearray(blob)
    left = dict(extra)
    for want, name in zip(SYMBOLS, names):
        enc = want.encode()
        # SEARCH THE BLOB, NOT THE RESIDUE. Part of a name can already be covered by a table's
        # tail, and requiring the whole run to be residual then matches nothing and leaves the
        # rest of the name counted as unexplained - which is a census reporting the wrong number
        # for a reason that has nothing to do with the name.
        for start in [i for i in range(len(out) - len(enc) + 1) if bytes(out[i:i + len(enc)]) == enc]:
            new = name.encode()
            if len(new) != len(enc):
                raise ValueError("a replacement symbol name must be the same length for now")
            out[start:start + len(new)] = new
            for i in range(len(enc)):
                left.pop(start + i, None)
            # The length prefix sits four bytes before the text in every class measured, and it is
            # the name's length, so it is ours to write too.
            if left.get(start - 4) == len(enc):
                out[start - 4] = len(new)
                left.pop(start - 4)
            break
    return bytes(out), left


class ClassMismatch(Exception):
    """The class the key selects is not the shape the program needs, so building it would produce
    an image that loads, runs and computes the wrong answer."""


def for_class(bindings, names=SYMBOLS, patch=None, kinds=None, counts=None, want=None,
              memory_opcodes=None, declared=None, structure=None, register_count=None):
    """The recorded class for this binding count, re-emitted with these buffer indices.

    -> bytes, or None if no class of that shape has been recorded.

    `structure` is (table count, vector count) when the backend knows what its program needs, and
    it exists because of an executed counterexample rather than on principle. c16-originA32 binds
    three buffers starting at zero, declares three, and its argument types are float/half/uint -
    identical on every one of those to ac2-32x32x64 - yet its metadata has EIGHT descriptor tables
    where ac2's has nine. No key over the signature separates them. Handed the nine-table class it
    loads, dispatches and writes a different answer than Apple's own archive, and nothing refuses
    it. With `structure` supplied, a mismatch raises instead.

    The mission this layer serves permits exactly this: an input beyond the signature "unless a
    real counterexample proves one necessary". This is that counterexample, and the input is one
    integer pair the backend has by construction.
    """
    # THE CLASS KEY IS THE SIGNATURE, and the second half of it is derivable rather than mysterious.
    #
    # Two shapes of per-kernel table exist per binding count - one carries a slot 1 the other lacks -
    # and nothing about the kernel predicted which Apple emits. It does not have to: the richer
    # shape runs for both groups, and with it 90 of 90 corpus kernels agree where the common shape
    # got 87.
    #
    # What DOES have to be selected is whether the first binding record carries an index field at
    # all. A record without one can only name buffer 0 - an absent index and a zero index encode the
    # same way - so a class recorded from a kernel whose first bound buffer is 0 cannot describe a
    # signature that starts at 1. That is a property of the signature this layer is handed, not of
    # anything hidden in Apple's compiler, so the key is (binding count, does it start at buffer 0).
    # AND THE ADDRESS SPACE OF THE BOUND ARGUMENTS. A `constant` argument gets a first binding
    # record with a longer vtable - across the cache the shape is fixed by (count, starts at zero)
    # everywhere EXCEPT where a constant is bound, and that is where it deviates: (2 bindings, not
    # starting at zero) is (8,12) in 405 kernels and (12,12) in the 50 that bind a constant, and
    # (3, starting at zero) is (6,8) in 142 and (12,8) in the 9 with a constant. So the kind of
    # each binding is part of the key, and Binding.kind in the contract is where it comes from.
    # THE KEY DOES NOT DETERMINE THE CLASS, and pretending it does just moves which kernels fail:
    # picking the modal shape for (2, starts at zero, a constant bound) fixed ad-dynamic and broke
    # cf-if_uniform, because that key holds five distinct shapes across 40 kernels. So a caller who
    # KNOWS - a backend that compiled the kernel and can see its whole resource set - may name the
    # class, and the derived key is the default for a caller who does not.
    def _checked(cls, why):
        if structure is not None:
            got = (len(cls.get("tables") or ()), len(cls.get("vectors") or ()))
            if tuple(structure) != got:
                raise ClassMismatch(
                    "%s gives a class with %d tables and %d vectors, but the program needs "
                    "%d and %d. Building it would run and compute the wrong answer."
                    % (why, got[0], got[1], structure[0], structure[1]))
        return _emit_class(cls, bindings, names, patch, counts, register_count)

    def _tg_ok(cls, why):
        # THE OTHER DIRECTION OF THE ASSERTION, AND IT WAS OPEN. The regression checked that a
        # signature with no threadgroup kind cannot reach the threadgroup class; nothing checked
        # that a signature WITH one cannot reach a class without it. It could, silently:
        #
        #   for_class([1,2,0], [device, device, threadgroup])  ->  456, class "2"
        #   for_class([1,2],   [device, device])               ->  456, class "2"
        #
        # Byte-identical. The threadgroup kind was filtered out for the key and then the fallback
        # chain dropped the "t" without a word, so the image declared no threadgroup pointer,
        # setThreadgroupMemoryLength had nothing to attach to, and a round trip read zero - which
        # is uninterpretable rather than wrong, and that is the expensive kind.
        #
        # Refusing is the point. A caller who asked for a threadgroup binding and is handed a class
        # that has none has been given a wrong answer, and the linker's whole discipline is to
        # decline instead.
        if tgroup and not cls.get("threadgroup"):
            raise ClassMismatch(
                "this signature binds threadgroup memory and %s has no threadgroup binding. "
                "Building it would load and dispatch, and setThreadgroupMemoryLength would have "
                "nothing to attach to - the round trip reads zero and the zero means nothing. "
                "Record a class from a kernel with this signature instead." % why)
        return cls

    # HOISTED ABOVE ITS FIRST USE. _tg_ok reads `tgroup`, and the caller-named branch below calls
    # _tg_ok, so naming a class raised NameError before the assignment was reached. Nothing caught
    # it because md_class had never been passed - every caller let the key be derived from the
    # signature. The extraction itself is unchanged; only its position is.
    if kinds and any(k == "threadgroup" for k in kinds):
        keep = [i for i, k in enumerate(kinds) if k != "threadgroup"]
        bindings = [bindings[i] for i in keep]
        kinds = [kinds[i] for i in keep]
        tgroup = "t"
    else:
        tgroup = ""

    if want and want in classes():
        return _checked(_tg_ok(classes()[want], "the class the caller named (%r)" % want),
                        "the class the caller named")
    # THE CLASS IS DERIVABLE FROM THE CODE. Within one signature key the metadata comes in several
    # shapes, and what separates them is WHICH MEMORY INSTRUCTIONS the program uses: across the 40
    # cached kernels that bind two buffers, start at zero and bind a constant, the store opcode
    # alone partitions 36 of them - op17235 to a 468-byte class, op17244 to 464, op17262 to 424,
    # op17229 to 456. The backend emitted those instructions and can say what they are, so this is
    # a property of the code it hands over rather than an extra thing it has to know.

    # AND THE NUMBER OF RESOURCES THE SIGNATURE DECLARES, not just the ones the code binds. Within
    # the key that was ambiguous - two bindings, starting at zero, one of them constant - the
    # declared count separates the shapes that the bound set cannot: 2 declared is a 456-byte
    # class, 5 declared is 468. It is a signature property, so it needs no instruction decoding.
    # A THREADGROUP BINDING IS A BINDING, AND NOTHING HERE COULD SAY SO. Binding.kind has carried
    # the address space all along and this function branched on exactly one value of it,
    # "constant"; none of the nine recorded classes came from a signature with a threadgroup
    # pointer. So an image generated for such a kernel declared no threadgroup usage at all, and
    # setThreadgroupMemoryLength:atIndex:0 had nothing to attach to - which is the whole of a
    # threadgroup round trip that returned zero on silicon and was NOT evidence about the three
    # opcodes it was meant to test.
    #
    # WHAT IT COSTS IN THE SECTION, measured over the 25 cached kernels that take a threadgroup
    # POINTER as a parameter (a static `threadgroup T s[N]` array is a different thing and 2,368
    # kernels have one):
    #
    #     the pointer is declared and never used in the body   3 vectors   11 kernels
    #     the pointer is actually used                         4 vectors   14 kernels
    #
    # A used threadgroup pointer adds a whole vector, holding one small record (vlen 6, tlen 8)
    # that sits AHEAD of the buffer bindings and displaces them - which is also why
    # corpuspack.signature raises "no binding vector" on a6-tgidx: its pk + F[4][2] + 4 + 36 lands
    # at 308 where the vectors are at 296, 312 and 320.
    #
    # Declared-and-unused costing nothing is the third independent appearance of the same rule,
    # after the builtin lists and the compile-time constant: this section counts what the code
    # CONSUMES, not what the signature mentions.
    #
    # The threadgroup binding is structure rather than an index to rewrite - its record has no
    # slot 1, so it can only ever name threadgroup(0) - so it is taken out of the binding list and
    # into the key, and _emit_class still receives only the buffer bindings it knows how to place.
    zero = "z" if bindings and bindings[0] == 0 else ""
    const = "c" if kinds and any(k == "constant" for k in kinds) else ""
    key = str(len(bindings)) + os.environ.get("CLASS_VARIANT", zero + const + tgroup)
    if declared:
        keyed = "%s.d%d" % (key, declared)
        if keyed in classes():
            return _checked(_tg_ok(classes()[keyed], "the refined key %r" % keyed),
                            "the declared-count refined key %r" % keyed)
    # WITHIN one signature key, the memory instructions choose between the shapes. Matching them
    # across keys is too strong - it hands a two-binding kernel a class recorded for a different
    # signature - so the opcode set only disambiguates classes that share the key.
    if memory_opcodes:
        want_ops = sorted(memory_opcodes)
        for nm, c in classes().items():
            if (nm == key or nm.startswith(key + ".")) and c.get("key_opcodes") \
                    and sorted(c["key_opcodes"]) == want_ops:
                return _checked(_tg_ok(c, "the opcode-disambiguated class %r" % nm),
                                "the memory-opcode disambiguation")
    cls = (classes().get(key) or classes().get(str(len(bindings)) + zero)
           or classes().get(str(len(bindings))))
    if cls is None:
        return None
    return _checked(_tg_ok(cls, "the class for key %r" % key), "the signature key")


def _emit_class(cls, bindings, names, patch, counts, register_count=None):
    values = {}
    # THE FIRST VECTOR'S RECORD CARRIES TWO COUNTS the class cannot supply. Two kernels alike in
    # every structural respect - binding count, starts-at-zero, a constant binding, the per-kernel
    # slots, the record shapes - differ in them, and no function of the buffer signature explains
    # either: ten readings tested across 7,419 kernels, best 46%. So they are values the backend
    # supplies, the same standing as the word at 280.
    if counts:
        import struct as _s
        root = cls["root"]; rt = cls["tables"][root]
        blob0 = build_from(cls)
        pk = root + rt["slots"][0] + _s.unpack_from("<I", blob0, root + rt["slots"][0])[0]
        first = pk + cls["tables"][pk]["fields"][26][2] + 12
        rec0 = cls["vectors"].get(first, [None])[0]
        if rec0 is not None:
            for slot, v in counts.items():
                if slot in cls["tables"][rec0]["fields"]:
                    values[(rec0, slot)] = v
    for rec, idx in zip(cls["records"], bindings):
        slot = cls["index_slot"].get(rec)
        if slot is not None:
            values[(rec, 1)] = idx
        elif idx:
            raise ValueError("this class's record at %d has no index field, so it can only name "
                             "buffer 0; %d requested" % (rec, idx))
    # THE RESIDUAL BYTES ARE APPLE'S and the census counts them as such, so the first question is
    # whether they are needed. MD_NOEXTRA drops them; if the corpus still agrees, they are emitted
    # as zeros and the image goes back to owning every byte.
    if os.environ.get("MD_NOEXTRA") == "1":
        cls = dict(cls, extra={})
    # SLOT 1 IS A NUMBER, NOT A SHAPE. Two kernels with the same record count and the same table
    # shape carry 4 and 8 there, and the one whose value differs from the recorded class computes a
    # different answer. CLASS_SLOT1 overrides it so the question is settled by execution.
    over = os.environ.get("CLASS_SLOT1")
    if over:
        root = cls["root"]; rt = cls["tables"][root]
        import struct as _s
        blob0 = build_from(cls, values=values)
        pk = root + rt["slots"][0] + _s.unpack_from("<I", blob0, root + rt["slots"][0])[0]
        if 1 in cls["tables"][pk]["slots"]:
            values = dict(values); values[(pk, 1)] = int(over)
    # Slot 0 is per-program, while the rest of the recorded class is structural. A class witness
    # count must never leak into a different compiled program's image.
    if register_count is not None:
        if type(register_count) is not int or register_count < 1:
            raise ValueError("register count must be a positive integer, not %r" % (register_count,))
        import struct as _s
        root = cls["root"]; rt = cls["tables"][root]
        blob0 = build_from(cls, values=values)
        pk = root + rt["slots"][0] + _s.unpack_from("<I", blob0, root + rt["slots"][0])[0]
        if 0 not in cls["tables"][pk]["slots"]:
            raise ValueError("this measured class has no slot 0 to carry a register count")
        values[(pk, 0)] = register_count
    blob = build_from(cls, values=values)
    blob, _left = _write_names(blob, cls.get("extra", {}), names)
    # A VALUE THE CLASS CANNOT CARRY. Two atomic kernels of the SAME class shape differ in one byte
    # of this section - at 280, fetch_and has 0xFFFFFFFF and fetch_min 0x7FFFFFFF - and packaging
    # them from one class gets one of them wrong. So the section holds at least one value that
    # varies within a class, and it is an input this layer takes.
    #
    # It looked like the identity of the reduction and that reading is WITHDRAWN: predicting the
    # rest of that table failed on 20 of 23 atomic kernels, and the two extra points I thought
    # supported it (fetch_or, fetch_xor) are in 388- and 392-byte classes where byte 280 is a
    # different field altogether. What is proven is that the value must be reproduced.
    if patch:
        u = bytearray(blob)
        for off, val in patch.items():
            u[off:off + len(val)] = val
        blob = bytes(u)
    return blob


def class_residue(bindings, kinds=None, want=None):
    """How many of a class's residual bytes are NOT the two symbol names - the honest count of
    what this file re-emits without being able to say what it is.

    It takes the same key as for_class, because counting the residue of a class the program will
    not use is a census that reports the wrong number.
    """
    zero = "z" if bindings and bindings[0] == 0 else ""
    const = "c" if kinds and any(k == "constant" for k in kinds) else ""
    cls = (classes().get(want) or classes().get(str(len(bindings)) + zero + const)
           or classes().get(str(len(bindings)) + zero) or classes().get(str(len(bindings))))
    if cls is None:
        return 0
    # Count against the REAL bytes: the names have to be findable, and a zeroed buffer contains
    # nothing to find, so counting against one reports every name byte as unexplained.
    _blob, left = _write_names(build_from(cls), cls.get("extra", {}), SYMBOLS)
    return len(left)


def shift_desc(desc, d):
    """The same described document with every node moved `d` bytes later.

    A relocation-invariance test with a real observable behind it: if a described class re-emitted
    at a different address still computes what Apple's archive computes, placement is free and what
    matters is the content. Every reference in build_from is written relative to its own field, so
    moving everything together should change nothing - and "should" is why it is dispatched.
    """
    out = dict(desc)
    out["size"] = desc["size"] + d
    out["root"] = desc["root"] + d
    out["order"] = [x + d for x in desc.get("order", [])]
    out["tables"] = {k + d: dict(v, vtpos=v["vtpos"] + d) for k, v in desc["tables"].items()}
    out["vectors"] = {k + d: [x + d for x in v] for k, v in desc["vectors"].items()}
    out["extra"] = {k + d: v for k, v in desc.get("extra", {}).items()}
    out["records"] = [x + d for x in desc.get("records", [])]
    out["index_slot"] = {k + d: v for k, v in desc.get("index_slot", {}).items()}
    return out


# ---------------------------------------------------------------------------------------------
# THE ONE VALUE THAT WAS SUPPLIED RATHER THAN DERIVED, and it is derivable after all.
#
# In the 436-byte class the word at offset 280 tracks the ATOMIC OPERATION exactly. The operation
# is four bits in the instruction - b4[5], b5[3], b6[3], b7[7] of op10094/10095/10022/10023, from
# the ISA session's isa/g17-atomic-family.toml - and across the five cached kernels of that class:
#
#     code  1   and     0xFFFFFFFF     all ones, the identity of AND
#     code  4   smax    0x80000000     INT_MIN, the identity of signed MAX
#     code  5   smin    0x7FFFFFFF     INT_MAX, the identity of signed MIN
#     code 13   umin    0xFFFFFFFF     UINT_MAX, the identity of unsigned MIN
#
# Four operations, four correct identities, no exceptions.
#
# THE EARLIER RETRACTION WAS RIGHT TOO, and both can be. The ISA session refuted "offset 280 is the
# identity" by testing the 392, 396 and 400-byte classes, where NO offset distinguishes the
# operations - and that is true, because the field is not present in those classes. My mistake was
# never the reading; it was comparing byte 280 across documents of different sizes, where it is a
# different field. Grouping by class first is what makes the two results agree.
#
# Only the four measured codes are emitted. A code nobody has seen keeps the class's own value,
# because predicting the rest of the table is exactly what was refuted once already.
ATOMIC_OPCODES = (10094, 10095, 10022, 10023)
ATOMIC_IDENTITY = {1: 0xFFFFFFFF, 4: 0x80000000, 5: 0x7FFFFFFF, 13: 0xFFFFFFFF}


def atomic_operation(code, spans):
    """The operation code of the first atomic instruction in a program, or None."""
    for off, ln, op in spans:
        if op in ATOMIC_OPCODES and off + 8 <= len(code):
            b = code[off:off + ln]
            return (((b[4] >> 5) & 1) | (((b[5] >> 3) & 1) << 1)
                    | (((b[6] >> 3) & 1) << 2) | (((b[7] >> 7) & 1) << 3))
    return None


ATOMIC_IDENTITY_AT = {436: 280}          # class size -> where the identity sits


def derived_patch(code, spans, class_size):
    """{offset: bytes} for values this layer can compute from the program itself."""
    at = ATOMIC_IDENTITY_AT.get(class_size)
    if at is None:
        return {}
    op = atomic_operation(code, spans)
    v = ATOMIC_IDENTITY.get(op)
    return {at: v.to_bytes(4, "little")} if v is not None else {}


# THE CONSTANT PROGRAM'S RECORD (goal item 11, docs/g17-tensorops-machine-model.md 25.120). A program
# whose constant program writes uniforms carries per-kernel slot 31, an argument map after the
# kind-3 (12/20) record's body, and kind-3 / kind-6 values that follow from that program. These are
# ADDITIVE: no class above changes, and a layout without a constant program never reaches them.

def relocate_layout(layout, cuts):
    """`layout` with bytes inserted (positive) or deleted (negative) at each (position, delta) in
    `cuts`: every position at or past a cut moves by that cut's delta, as `with_system_registers`
    moves them for its one cut. Q is the binding vector's distance from the per-kernel table
    (vec_bind = pk + Q + 36), so it is recomputed rather than moved."""
    def moved(value):
        return value + sum(d for c, d in cuts if value >= c) if type(value) is int else value

    out = dict(layout)
    out["size"] = layout["size"] + sum(d for _c, d in cuts)
    for key in ("vec_bind", "vec2", "vec0", "slot29_vector"):
        if layout.get(key) is not None:
            out[key] = moved(layout[key])
    for key in ("v0", "name", "cpname"):
        if layout.get(key):
            out[key] = tuple(moved(v) for v in layout[key])
    for key in ("v2", "bind"):
        if key in layout:
            out[key] = [tuple(moved(v) if i < 2 else v for i, v in enumerate(rec)) for rec in layout[key]]
    if "ptrs" in layout:
        out["ptrs"] = {slot: moved(pos) for slot, pos in layout["ptrs"].items()}
    if layout.get("fills"):
        out["fills"] = tuple((moved(pos),) + tuple(rest) for pos, *rest in layout["fills"])
    if layout.get("nametab"):
        table = list(layout["nametab"])
        table[0], table[1] = moved(table[0]), moved(table[1])
        out["nametab"] = tuple(table)
    if layout.get("q") is not None:
        out["q"] = layout["q"] + (moved(layout["vec_bind"]) - layout["vec_bind"]) - (moved(layout["pk"]) - layout["pk"])
    return out


# Apple's per-kernel tail with slot 31, in both measured tails (straight-line 64 and looping 64):
# slot 44 at 55, slot 31 (a u32) at 56, the byte slots 33 (looping only), 32, 16 and 15
# right-aligned to end at 63, slot 0 at 64; the table is 68 bytes.
CONSTANT_PROGRAM_TAIL_BYTES = (33, 32, 16, 15)
CONSTANT_PROGRAM_PK_TLEN = 68


def with_constant_program(layout, slot31, argument_map, register_count, kind6):
    """`layout` (a class's, before `build`) for a program whose constant program writes uniforms.

    slot31          per-kernel slot 31's value (constprog.slot31)
    argument_map    the words after the kind-3 record's body (constprog.argument_map)
    register_count  the constant program's register count: kind-3 field 2
    kind6           None to drop the class's kind-6 record (Apple omits it when its field 2 would
                    be 0), or (field 2, field 3) to keep it with those values

    Refuses a class with no kind-6 record when one is asked for: that record's placement is not
    measured here."""
    import copy
    layout = copy.deepcopy(layout)
    cuts = []
    pk_end = layout["pk"] + layout["pk_tlen"]
    slots = dict(layout["pk_slots"])
    tail = [s for s in CONSTANT_PROGRAM_TAIL_BYTES if s in slots]
    if 44 in slots:
        slots[44] = 55
    slots[31] = 56
    for j, s in enumerate(tail):
        slots[s] = 64 - len(tail) + j
    slots[0] = 64
    layout["pk_slots"] = slots
    layout["pk_extra"] = dict(layout["pk_extra"])
    layout["pk_extra"][31] = ("<I", slot31)
    cuts.append((pk_end, CONSTANT_PROGRAM_PK_TLEN - layout["pk_tlen"]))
    layout["pk_tlen"] = CONSTANT_PROGRAM_PK_TLEN
    count_at = layout["v0"][0] + 20
    cuts.append((count_at + 4, 4 * len(argument_map)))
    layout["v0_field0"] = 8 + 4 * len(argument_map)
    layout["v0_field2"] = register_count
    kinds = [i for i, v in enumerate(layout["v2_vals"]) if (v.get(0) if isinstance(v, dict) else v[0]) == 6]
    if kind6 is None and kinds:
        i = kinds[0]
        pos, vtpos = layout["v2"][i][:2]
        lo, hi = min(pos, vtpos), max(pos + 16, vtpos + 12)
        cuts.append((lo, lo - hi))
        cuts.append((layout["vec2"] + 4 * len(layout["v2"]), -4))     # its entry: the vector's last
        layout["v2"] = [r for j, r in enumerate(layout["v2"]) if j != i]
        layout["v2_vals"] = [r for j, r in enumerate(layout["v2_vals"]) if j != i]
    elif kind6 is not None and kinds:
        vals = list(layout["v2_vals"])
        vals[kinds[0]] = (6, kind6[0], kind6[1])
        layout["v2_vals"] = vals
    elif kind6 is not None:
        raise ValueError("this class has no kind-6 record to set")
    cuts.sort()
    out = relocate_layout(layout, cuts)
    at = relocate_layout(dict(size=0, pk=0, vec_bind=0, q=0, v0=(count_at,)), cuts)["v0"][0]
    out["fills"] = tuple(out.get("fills") or ()) + ((at, "<I", len(argument_map)),) + tuple(
        (at + 4 + 4 * j, "<I", word) for j, word in enumerate(argument_map))
    return out


if __name__ == "__main__":
    main()
