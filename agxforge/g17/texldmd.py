#!/usr/bin/env python3
"""The texture class's __GPU_LD_MD, stated as a structure rather than carried as a blob.

    python3 tools/g17texldmd.py build results/g17-texture-ldmd-class-v1
    python3 tools/g17texldmd.py check results/g17-texture-ldmd-class-v1

`g17teximagesections` measured that this section is a CLASS CONSTANT: 336 bytes, byte-identical
across two independently compiled texture kernels that differ in their dynamic threadgroup buffer,
their instruction count and their register count, and 224 bytes with a different slot set on the
buffer twin. The common-runtime assignment asks the adapter to "state or derive, in checked-in
data, the texture class's __GPU_LD_MD ... rule", and a 336-byte literal is the weakest thing that
could satisfy that sentence. This is the stronger form: the section is written down as the NODE
GRAPH it is, and rebuilt from that statement.

WHAT IS CHECKED. `build(entry)` emits 336 bytes with ZERO differing from both retained texture
witnesses. Nothing is read from a witness at runtime - the positions, slot maps, declared lengths
and values below are the statement, and the comparison afterwards is a test rather than an input.
Every one of the section's 83 non-zero bytes lies inside a named node; the 22 bytes outside one are
zero, and `padding()` lists their ranges.

WHAT THE STATEMENT SAYS THE BUFFER CLASS DOES NOT. Read against the retained buffer twin, the
texture class's main table ADDS slots 3, 4, 9 and 30, DROPS slot 1, declares 60 bytes rather than
48, and reaches a thirteen-slot sub-table that the buffer class has no analogue for. That is the
shape of the 2026-09-08 defect stated positively: those are the bytes a buffer-shaped LD_MD in a
texture image would be missing, and no load-time error reports their absence.

WHAT IS REPRODUCED RATHER THAN DERIVED, listed rather than buried - the same standing as the
RESTORE constants in `g17ldmd`, and `unnamed()` returns it:

    the added slots 3, 4, 9, 30   present and non-default in both texture witnesses and absent from
                                  the buffer twin. What they MEAN is not established here.
    the 8-byte pair (8192, 0x807bff00)   appears twice: as the single element of slot 4's vector and
                                  again inside the sub-table. Two occurrences of the same eight
                                  bytes is why it is stated as one struct rather than two scalars.
    the sub-table's eight fields  including 0x477fe000, which is 65504.0 read as a float - the
                                  largest finite half. Suggestive, and suggestive is not measured.
    the serialiser's padding      22 zero bytes in five runs, reproduced at Apple's positions

THE ENTRY PC IS THE ONE PARAMETER. Slot 6 is the entry PC by the evidence that named it for the
buffer class - equal to the `_agc.main` offset in 21,001 of 21,001 corpus objects, and a multiple
of 64 in every one - and it is 64 in both texture witnesses. `build` takes it and enforces the
alignment law; it does not pretend a different entry has been witnessed for THIS class.

THE GUARD DOES NOT BORROW THE METADATA CLASS'S. `g17texemit.admit` refuses a dynamic threadgroup,
because slot 10 of __GPU_METADATA moves when one is declared. This section DOES NOT MOVE: one of
the two witnesses carries a dynamic threadgroup and the 336 bytes are identical. Reusing that guard
here would refuse a contract this class has a witness for, so `admit` states the facts the two
witnesses SHARE, and `not_constrained()` names the three they differ in and why each is absent.

No GPU dispatch, no image built, no loader claim. Reproducing a section is not authoring an image.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MEASURED = "results/g17-texture-image-sections-v1/measured"
WITNESSES = ("tex2d-read", "tex-dyn-tg")
CONTROL = "buffer-twin"

SECTION_BYTES = 336
ENTRY_ALIGN = 64                 # g17ldmd: 2-, 4- and 32-aligned entry PCs all fail to execute
WITNESSED_ENTRY = 64

# THE STATEMENT. Positions are Apple's, reproduced rather than derived, exactly as `g17ldmd` does
# for the buffer class: FlatBuffers leaves the serialiser free to place a vtable, and this emits
# the placement the measured objects use.
ROOT_TABLE, ROOT_VT, ROOT_SLOTS, ROOT_TLEN = 20, 8, 4, 12
T3, T3_VT, T3_SLOTS, T3_TLEN = 44, 32, 4, 6
MAIN, MAIN_VT, MAIN_SLOTS, MAIN_TLEN = 136, 50, 41, 60
SUB, SUB_VT, SUB_SLOTS, SUB_TLEN = 308, 278, 13, 28

# slot -> (offset in the table body, kind, value). "ref" is resolved against the FIELD'S address.
MAIN_FIELDS = {
    40: (4, "ref", 232),         # an empty vector
    29: (8, "<Q", 1),
    9: (16, "<Q", 1),            # ADDED by this class
    30: (24, "ref", 236),        # ADDED by this class
    4: (28, "ref", 252),         # ADDED by this class
    3: (32, "ref", 268),         # ADDED by this class - reaches the sub-table
    18: (39, "<B", 1),
    6: (40, "<I", None),         # THE ENTRY PC, the one parameter
    5: (44, "ref", 212),
    2: (48, "ref", 220),         # the stage name
    38: (52, "ref", 196),
    7: (56, "ref", 204),
}
SUB_FIELDS = {
    12: (4, "<I", 8192),
    8: (18, "<B", 1),
    6: (19, "<B", 1),
    10: (20, "<I", 0x477FE000),  # 65504.0 as a float: the largest finite half
    5: (24, "<B", 1),
    2: (25, "<B", 4),
    1: (26, "<B", 4),
    0: (27, "<B", 4),
}
STAGE = b"compute"
STAGE_AT = 220
# The eight bytes that appear twice. Stated once, written at both places it is measured.
PAIR = bytes.fromhex("0020000000ff7b80")
PAIR_AT = (256, 312)
VECTORS = ((196, []), (204, [107]), (212, [0]), (232, []), (236, [0]))
PAIR_VECTOR_AT = 252             # count 1, then PAIR
REF_VECTOR_AT = 268              # count 1, then a reference to the sub-table

# The bytes no node covers. All zero, all the serialiser's, all reproduced at Apple's positions.
PADDING = ((4, 8), (200, 204), (244, 252), (264, 268), (276, 278))


class Refused(ValueError):
    """A contract outside the shape both witnesses share. Refusing beats authoring a silent one."""


def read_mask_for(public_indices):
    """Slot 9: a bitmask over the PUBLIC `[[texture(n)]]` indices a program actually reads.

    Measured, not assumed. Over the retained r32float pair compiles the mask is the bitmask of the
    source indices READ - tx1f-solo reads 0 and carries 1, tx1f-at4 reads 4 and carries 16,
    tx2f-pair reads 0 and 1 and carries 3, tx2f-3and7 reads 3 and 7 and carries 136 - and a texture
    DECLARED but not read is absent from it, which tx2f-dead settles by declaring two and carrying
    1. The indices are an authoring input: three programs naming different slots compile to one
    `main.bin` and one `__GPU_METADATA`, so nothing in the delivered bytes states them.
    """
    indices = tuple(public_indices)
    if not indices:
        raise Refused("a texture class with no read texture has no slot 9 to author")
    if len(set(indices)) != len(indices):
        raise Refused("public texture indices repeat: %s" % (list(indices),))
    for index in indices:
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < 64:
            raise Refused("public texture index %r is outside the 64-bit slot-9 mask" % (index,))
    return sum(1 << index for index in indices)


def build(entry=WITNESSED_ENTRY, allow_unwitnessed=False, read_mask=None):
    """The whole 336-byte section from the statement above and the entry PC.

    `read_mask` is slot 9. It defaults to None, which authors the one-texture value this class was
    witnessed at, so every existing caller emits the same bytes it always did. Supplying it authors
    the same class for another set of read textures: with 3 this reproduces the retained tx2f-pair
    section byte for byte, and with 136 the retained tx2f-3and7 one, neither of which this side
    compiled. The 344-byte members of that family are a DIFFERENT class and are not reachable by
    changing this field - `admit` keeps them out rather than letting a mask author the wrong shape.

    THE UNWITNESSED ENTRY IS REFUSED HERE AS WELL AS IN `admit`, and the duplication is the point.
    A caller holds the entry PC and the contract separately - `g17authorobj._ld_md` takes `entry`
    as an argument and the contract as another - so a contract stating 64 can admit while the
    argument carries something else, and the section would then be emitted for an entry no witness
    covers with nothing between it and the object. Checking it at the only place that writes the
    field closes that regardless of the call site. `allow_unwitnessed` exists so a test can show
    what the field does, not so an author can opt out.
    """
    if entry % ENTRY_ALIGN:
        raise Refused("entry PC %d is not a multiple of %d; the shader does not start"
                      % (entry, ENTRY_ALIGN))
    if entry != WITNESSED_ENTRY and not allow_unwitnessed:
        raise Refused("entry PC %d is outside this class's witnesses, which both carry %d; pass "
                      "allow_unwitnessed=True only to demonstrate the field, never to author"
                      % (entry, WITNESSED_ENTRY))
    b = bytearray(SECTION_BYTES)

    def table(pos, vt, nslots, tlen, fields):
        vlen = 4 + 2 * nslots
        assert vt + vlen == pos, "the vtable must abut the table it describes"
        struct.pack_into("<HH", b, vt, vlen, tlen)
        struct.pack_into("<i", b, pos, vlen)
        for slot, (off, kind, value) in fields.items():
            struct.pack_into("<H", b, vt + 4 + 2 * slot, off)
            if kind == "ref":
                struct.pack_into("<I", b, pos + off, value - (pos + off))
            else:
                struct.pack_into(kind, b, pos + off, value)

    def vector(pos, items):
        struct.pack_into("<I", b, pos, len(items))
        for k, value in enumerate(items):
            struct.pack_into("<I", b, pos + 4 + 4 * k, value)

    struct.pack_into("<I", b, 0, ROOT_TABLE)
    table(ROOT_TABLE, ROOT_VT, ROOT_SLOTS, ROOT_TLEN,
          {0: (8, "ref", MAIN), 3: (4, "ref", T3)})
    table(T3, T3_VT, T3_SLOTS, T3_TLEN, {3: (5, "<B", 1)})
    main = dict(MAIN_FIELDS)
    main[6] = (main[6][0], main[6][1], entry)
    if read_mask is not None:
        if read_mask <= 0:
            raise Refused("slot 9 must name at least one read texture; %r names none" % (read_mask,))
        main[9] = (main[9][0], main[9][1], read_mask)
    table(MAIN, MAIN_VT, MAIN_SLOTS, MAIN_TLEN, main)
    for pos, items in VECTORS:
        vector(pos, items)
    struct.pack_into("<I", b, STAGE_AT, len(STAGE))
    b[STAGE_AT + 4:STAGE_AT + 4 + len(STAGE)] = STAGE
    struct.pack_into("<I", b, PAIR_VECTOR_AT, 1)
    struct.pack_into("<I", b, REF_VECTOR_AT, 1)
    struct.pack_into("<I", b, REF_VECTOR_AT + 4, SUB - (REF_VECTOR_AT + 4))
    table(SUB, SUB_VT, SUB_SLOTS, SUB_TLEN, SUB_FIELDS)
    for at in PAIR_AT:
        b[at:at + len(PAIR)] = PAIR
    return bytes(b)


def retained(name, section="ld_md"):
    with open(os.path.join(ROOT, MEASURED, "%s.%s.bin" % (name, section)), "rb") as handle:
        return handle.read()


def against_the_witnesses():
    """Differing byte counts, per retained texture witness. The comparison is a test, not an input."""
    section = build()
    rows = {}
    for name in WITNESSES:
        want = retained(name)
        rows[name] = {
            "bytes": len(want),
            "differing": (SECTION_BYTES if len(want) != SECTION_BYTES
                          else sum(1 for i in range(SECTION_BYTES) if section[i] != want[i])),
        }
    return rows


def padding():
    """The byte ranges no named node covers, with the check that every one of them is zero."""
    want = retained(WITNESSES[0])
    return {
        "ranges": [list(r) for r in PADDING],
        "bytes": sum(b - a for a, b in PADDING),
        "all zero in the witness": all(not any(want[a:b]) for a, b in PADDING),
        "non-zero bytes in the section": sum(1 for v in want if v),
        "non-zero bytes outside a named node": sum(
            1 for a, b in PADDING for v in want[a:b] if v),
    }


def _main_slots(blob):
    """(slot -> offset, declared length) for the main table of a retained section."""
    root = struct.unpack_from("<I", blob, 0)[0]
    rvt = root - struct.unpack_from("<i", blob, root)[0]
    off = struct.unpack_from("<H", blob, rvt + 4)[0]
    main = root + off + struct.unpack_from("<I", blob, root + off)[0]
    vt = main - struct.unpack_from("<i", blob, main)[0]
    vlen, tlen = struct.unpack_from("<HH", blob, vt)
    slots = {}
    for i in range((vlen - 4) // 2):
        at = struct.unpack_from("<H", blob, vt + 4 + 2 * i)[0]
        if at:
            slots[i] = at
    return slots, tlen


def the_class_diff():
    """What this class carries that the buffer twin does not, read off both retained sections."""
    texture, t_len = _main_slots(retained(WITNESSES[0]))
    buffer_, b_len = _main_slots(retained(CONTROL))
    return {
        "texture main slots": sorted(texture),
        "buffer main slots": sorted(buffer_),
        "added by the texture class": sorted(set(texture) - set(buffer_)),
        "dropped by the texture class": sorted(set(buffer_) - set(texture)),
        "texture main declared length": t_len,
        "buffer main declared length": b_len,
        "texture section bytes": len(retained(WITNESSES[0])),
        "buffer section bytes": len(retained(CONTROL)),
        "the sub-table": ("the texture class's slot 3 reaches a %d-slot table the buffer class has "
                          "no analogue for" % SUB_SLOTS),
        "why it matters": ("these are the bytes a buffer-shaped __GPU_LD_MD in a texture image "
                           "would be missing, and no load-time error reports their absence"),
    }


def unnamed():
    """Reproduced rather than derived. Stated so the next reader does not mistake it for a rule."""
    return {
        "the added slots": {
            "slots": sorted(set(MAIN_FIELDS) - {40, 29, 18, 6, 5, 2, 38, 7}),
            "standing": "present and non-default in both texture witnesses, absent from the buffer "
                        "twin; what they mean is not established here",
        },
        "the eight-byte pair": {
            "bytes": PAIR.hex(),
            "at": list(PAIR_AT),
            "standing": "the same eight bytes twice - slot 4's single vector element and again "
                        "inside the sub-table - which is why it is stated as one struct",
        },
        "the sub-table's fields": {
            "slots": sorted(SUB_FIELDS),
            "standing": "reproduced; 0x477FE000 is 65504.0 read as a float, the largest finite "
                        "half, which is suggestive and suggestive is not measured",
        },
        "the padding": {
            "ranges": [list(r) for r in PADDING],
            "standing": "the serialiser's, reproduced at Apple's positions",
        },
    }


def not_constrained():
    """The three facts the guard deliberately does NOT check, each with its witness."""
    return {
        "a dynamic threadgroup buffer": (
            "one of the two witnesses declares one and the 336 bytes are identical. "
            "g17texemit.admit refuses it because slot 10 of __GPU_METADATA moves; this section "
            "does not move, so borrowing that guard would refuse a contract with a witness"),
        "the main instruction count": "the witnesses differ in it and the section is identical",
        "the register count": "the witnesses differ in it and the section is identical",
    }


def admit(abi):
    """Refuse a contract outside the shape BOTH witnesses share. Returns a list of reasons."""
    resources = abi.get("resources") or {}
    reasons = []
    textures = resources.get("textures") or []
    public = resources.get("texture_public_indices")
    # ONE OR TWO, AND THE PAIR HAS TO SAY WHICH SLOTS IT READS. Both of this class's own witnesses
    # declare one texture, but the retained r32float compiles carry two-texture sections of this
    # exact 336-byte shape - tx2f-pair and tx2f-3and7 - and `build` reproduces both byte for byte
    # from slot 9 alone. The pair is admitted on that evidence and on nothing else: the public
    # indices are an authoring input, because three programs naming different slots compile to one
    # main.bin and one __GPU_METADATA and the delivered bytes therefore do not state them.
    if len(textures) not in (1, 2):
        reasons.append("this class is witnessed for one or two read textures; the contract "
                       "declares %d" % len(textures))
    elif any(t.get("access") != "read" or t.get("dimension") != "2d" for t in textures):
        reasons.append("every witness of this class declares read-only 2d textures")
    elif len(textures) == 2:
        if not public:
            reasons.append("a two-texture contract must state resources.texture_public_indices, "
                           "the [[texture(n)]] slot each texture is read at: slot 9 is a mask over "
                           "them and no delivered byte states it")
        elif len(public) != len(textures):
            reasons.append("resources.texture_public_indices states %d slot(s) for %d texture(s)"
                           % (len(public), len(textures)))
    if resources.get("samplers"):
        reasons.append("no witness of this class declares a sampler")
    entry = abi.get("entry")
    if entry is None:
        reasons.append("the entry PC is this class's one parameter and the contract does not "
                       "state it")
    elif entry % ENTRY_ALIGN:
        reasons.append("the entry PC must be a multiple of %d; the shader does not start otherwise"
                       % ENTRY_ALIGN)
    elif entry != WITNESSED_ENTRY:
        reasons.append("entry PC %d is outside this class's witnesses, which both carry %d; the "
                       "field is the start PC by the buffer class's evidence, but no texture "
                       "witness exercises another value" % (entry, WITNESSED_ENTRY))
    return reasons


def a_mutation_is_visible():
    """The statement is load-bearing, and how SMALL the visible mutation is, is the point.

    Dropping the added slot 3 changes TWO bytes - its vtable entry - because the sub-table it
    reaches stays placed and stays structurally valid. A section that has lost one of the four
    slots this class adds differs from a correct one by two bytes in a table nobody reads at load
    time, which is the 2026-09-08 failure shape at its smallest. The larger number below is the
    same defect at its largest: the whole buffer-shaped section in a texture image.
    """
    base = build()
    saved = MAIN_FIELDS.pop(3)
    try:
        without = build()
    finally:
        MAIN_FIELDS[3] = saved
    control = retained(CONTROL)
    return {
        "dropping the added slot 3": sum(1 for i in range(SECTION_BYTES)
                                         if base[i] != without[i]),
        "and that is only its vtable entry": ("the sub-table stays placed and stays structurally "
                                              "valid; two bytes is the whole visible difference"),
        "the buffer twin is a different section": base != control,
        "bytes the buffer-shaped section would be short": SECTION_BYTES - len(control),
        "differing over the buffer twin's own length": sum(
            1 for i in range(len(control)) if base[i] != control[i]),
    }


def _document():
    return {
        "against_the_witnesses": against_the_witnesses(),
        "padding": padding(),
        "the_class_diff": the_class_diff(),
        "unnamed": unnamed(),
        "not_constrained": not_constrained(),
        "a_mutation_is_visible": a_mutation_is_visible(),
        "sha256 of the emitted section": hashlib.sha256(build()).hexdigest(),
    }


def build_dir(destination):
    os.makedirs(destination, exist_ok=True)
    path = os.path.join(destination, "ldmd-class.json")
    with open(path, "w") as handle:
        json.dump(_document(), handle, indent=2, sort_keys=True)
        handle.write("\n")
    with open(os.path.join(destination, "rebuilt.ld_md.bin"), "wb") as handle:
        handle.write(build())
    return path


def check(destination=None):
    findings = []
    rows = against_the_witnesses()
    for name, row in rows.items():
        if row["differing"]:
            findings.append("%s: %d differing bytes" % (name, row["differing"]))
    pad = padding()
    if not pad["all zero in the witness"]:
        findings.append("a byte outside every named node is non-zero")
    if pad["non-zero bytes outside a named node"]:
        findings.append("%d non-zero bytes lie outside a named node"
                        % pad["non-zero bytes outside a named node"])
    diff = the_class_diff()
    if not diff["added by the texture class"]:
        findings.append("the class diff against the buffer twin found nothing added, which would "
                        "mean the two classes share a slot set")
    mutation = a_mutation_is_visible()
    if not mutation["dropping the added slot 3"]:
        findings.append("dropping a stated slot changed no byte, so the statement is not "
                        "load-bearing")
    if not mutation["the buffer twin is a different section"]:
        findings.append("the emitted section equals the buffer twin's")
    if destination:
        path = os.path.join(destination, "ldmd-class.json")
        if not os.path.exists(path):
            findings.append("%s is not retained" % path)
    return findings


def main(argv):
    if len(argv) > 1 and argv[1] == "build":
        print(build_dir(argv[2] if len(argv) > 2 else "results/g17-texture-ldmd-class-v1"))
        return 0
    if len(argv) > 1 and argv[1] == "check":
        findings = check(argv[2] if len(argv) > 2 else None)
        for finding in findings:
            print(finding)
        print("%d finding(s)" % len(findings))
        return 1 if findings else 0
    print(json.dumps(_document(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
