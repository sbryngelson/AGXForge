"""__GPU_LD_MD and __GPU_ARCH_LD_MD, GENERATED from the program's own layout.

These two sections were the last opaque bytes in the scalar object - 18 and 5 non-zero bytes
carried by class, with no handle beyond "select the blob by exact size". The peer ISA session
supplied the structural handle: LD_MD is a FlatBuffers table of the same shape as __GPU_METADATA,
a slot-0 subtable and a slot-3 subtable. Read as a table rather than as a blob, every one of the
23 bytes is either FlatBuffers structure or one of two values, and both values are now named.

    [0,6]  u32   THE ENTRY PC: the offset in __text at which the shader begins executing,
                 which must be a multiple of 64. Established causally by moving it - see below.
    [3,3]  u8    1. RE-MEASURED at 21,001 of 21,001 corpus objects, up from the 1,377 this
                 line was written on; writing 0 makes the loader
                 abort the process, writing 2 is accepted, so it is read and only 0 is rejected.

WHAT THE LOADER DOES NOT NEED, MEASURED ON BUFFER KERNELS ONLY. Apple's own LD_MD for the same
kernel carries 25 more non-zero
bytes: fields [0,2]=8, [0,5]=36, [0,38]=12, [0,40]=72, several u8 flags, and the string
"compute". All of them are ZERO here and the kernel still executes correctly. That includes every
field the peer catalogued as free or derived - the [0,5]=[0,2]-4 and [0,38]=[0,7]-4 relations are
real in the corpus and do not have to hold for a program to run, because neither side is read.

HOW [0,6] WAS NAMED. It equals the offset of the _agc.main symbol in all 21,001 corpus objects
- RE-MEASURED, up from the 1,376 this line was written on, and the entry is a multiple of 64 in
every one. The sub-table also carries slots beyond 3 in 1,904 objects, which this header does not
describe. Originally: in all 1376 corpus objects,
which is a correlation, and the novel kernel needed exactly 64 while this project places its
entry at 0x40, which is a coincidence of two numbers. spike/accel/re/entryoff.py moves the code
and the field independently, and the eleven rows separate the readings:

    entry  64  field  64   ok          the default
    entry 128  field 128   ok          both moved together
    entry 192  field 192   ok
    entry 128  field  64   ok          field points at FILLER 64 bytes before the code, which
    entry 192  field  64   ok          executes as no-ops until control reaches the code
    entry 192  field 128   ok
    entry  64  field 128   sentinel    field points into the middle of the code
    entry  66  field  66   sentinel    2-aligned
    entry  68  field  68   sentinel    4-aligned
    entry  96  field  96   sentinel    32-aligned
    entry  66  field  64   ok          code at 66 is fine; only the START PC is constrained

So the field is the start PC, not a description of where the code is: any 64-byte-aligned offset
from which control reaches the code works, and no offset that is not a multiple of 64 does. The
prediction written into that probe said the two had to be equal, and the rows that refuted it are
the ones that named the field.

The LAYOUT is Apple's, reproduced rather than derived: FlatBuffers has freedom in where it places
a vtable, and this emits the placement the measured objects use. build() asserts byte-identity
against the measured blob for the default arguments, so the description above is checked.
"""
import struct

# Positions Apple's serialiser chose for this object class. Reproduced, not derived.
_ROOT, _RVT = 20, 8            # root table and its vtable
_T3, _T3VT = 44, 32            # slot-3 subtable and its vtable
_T0, _T0VT = 136, 50           # slot-0 subtable and its vtable
_T0_SLOTS, _T3_SLOTS = 41, 4   # vtable slot counts (vlen = 4 + 2*slots)

ENTRY_ALIGN = 64               # measured: 2-, 4- and 32-aligned entry PCs all fail to execute


def _put_table(buf, pos, vtpos, nslots, tlen, fields):
    """One FlatBuffers table: vtable at vtpos, body at pos, back-pointer = vtable length.

    `fields` maps slot index -> (offset within the body, struct format, value). The vtable runs
    from vtpos up to pos exactly, which is what makes the back-pointer equal to the length.
    """
    vlen = 4 + 2 * nslots
    assert vtpos + vlen == pos, "vtable must abut the table it describes"
    struct.pack_into("<HH", buf, vtpos, vlen, tlen)
    struct.pack_into("<i", buf, pos, vlen)
    for slot, (off, fmt, val) in fields.items():
        struct.pack_into("<H", buf, vtpos + 4 + 2 * slot, off)
        struct.pack_into(fmt, buf, pos + off, val)


# THE SWEPT REGIONS OF __GPU_LD_MD, and they are format constants rather than program facts.
# Constant across all 33 sampled 216-byte sections: 36, 8, 12, 1, 107, 1, and a length-7 name that
# is "compute" in every one of them. The executed scalar class has all of them ZEROED, exactly as it
# zeroes the metadata symbol names, so emitting them is restoring a swept class rather than
# inventing content - and the swept class keeps its zeros.
RESTORE = ((0xa4, 36), (0xa8, 8), (0xac, 12), (0xb8, 1), (0xbc, 107), (0xc0, 1), (0xc8, 7))
# Eleven more, each constant across all 33 sampled 216-byte sections and several at odd offsets,
# so they are written as bytes rather than words. Same standing as the seven above: swept to zero in
# the executed class, present in every unswept one.
RESTORE_BYTES = ((0x34, 0x28), (0x3a, 0x1c), (0x44, 0x24), (0x5a, 0x13), (0x70, 0x08),
                 (0x82, 0x20), (0x86, 0x04), (0x8c, 0x48), (0x90, 0x01), (0x9b, 0x01),
                 (0xa0, 0x20))
STAGE = b"compute"


# THE ATOMIC DECLARATION IS FOUR BYTES AND IT MOVES NOTHING. CORRECTED 2026-09-17, AND THE
# CORRECTION COST A CRASH.
#
# What stood here said "a device atomic is NOT this section plus a flag: six of the eight shared
# fields move by four bytes, two new slots appear, and tlen grows 40 -> 48." That was measured on
# Apple's 461 atomic objects, every one of which has a 224-BYTE section - so it compared two
# variables at once, the declaration and the section shape, and attributed both to the atomic.
# Building on it, `build(atomic=True, size=216)` wrote the 224-byte shape's offsets into a
# 216-byte section. Those offsets are section-relative: slot 40 at 80 instead of 72, slot 2 at 40
# instead of 36, and so on, each eight bytes past where the data actually is. The result is the
# one failure mode a structural check cannot see - it described correctly, round-tripped
# byte-identically, and reported slot 24 present - and it SEGFAULTED the host inside
# AGX::DynamicLoader at newComputePipelineStateWithDescriptor. No GPU submission, no gpu event:
# the driver reads this table, follows these fields, and faulted on the address they named.
#
# The population that separates the two variables was in the cache all along: census table 136 by
# SECTION SIZE and there are 20 objects at 216 bytes, tlen 40, DECLARING the atomic - all 20
# compiled from Metal source by Apple's own toolchain, all 20 byte-identical to each other, and
# field-for-field identical to this backend's own non-atomic 216-byte section. tlen and the field
# offsets track the section's SIZE (216 -> 40, 224 -> 48, 232 -> 44 or 56, ...); the declaration is
# orthogonal to both. Against the witness, the whole difference is four bytes:
#
#     byte  40  t44  vtable slot 2  = 4      the two vtable entries that make the slots present
#     byte 102  t136 vtable slot 24 = 18
#     byte  48  t44  body   slot 2  = 1      and the two one-byte values themselves
#     byte 154  t136 body   slot 24 = 1
#
# `build(entry=64, size=216, f0_5=32, restore=True)` already reproduced this host's section with
# zero differing bytes; with `atomic=True` it reproduces the VENDOR-LINKED witness, byte for byte.
# Slot 24 is the declaration: present in 511 of 511 objects whose text carries a device atomic and
# 0 of 21,463 others. t44 slot 2 accompanies it in 481 of 481 declaring objects (20 at 216 bytes,
# 461 at 224) and is written with it; it is NOT exclusive to atomics - 20,282 non-declaring 224-byte
# sections carry it too - so it is emitted because every witness has it, not as evidence.
#
# The 224-byte shape's layout is kept below because it is what Apple's 461 atomic objects hold and
# it is measured at 461/461 - but it belongs to that SIZE, and asking for it at any other size is
# refused rather than rescaled. Slot 1 stays omitted in both: a program fact with 218 distinct
# values in that class, absent from every section this backend emits, meaning unmeasured.
#
#     slot   216-shape   224-shape          slot 29 and slot 40 do not move between the shapes;
#       40        4          4              slots 18/6/5/2/38/7 shift +4, and so does slot 24
#       29        8          8              (18 -> 22), which is why 22 looked like "the atomic
#       24       18         22              offset" from the 224 class alone. At 216 offset 22
#       18       19         23              lands inside slot 6, the four-byte entry PC at 20.
#        6       20         24
#        5       24         28
#        2       28         32
#       38       32         36
#        7       36         40
#        1        -         44  (omitted)
#     tlen       40         48
#
# ledger/g17-the-atomic-declaration-is-slot-24-of-table-136.toml
ATOMIC_DECL_T3 = {2: (4, "<B", 1)}          # both shapes, 481 of 481 declaring objects
ATOMIC_DECL_T0 = {216: {24: (18, "<B", 1)}, # 20 vendor witnesses, byte-for-byte reproduced
                  224: {24: (22, "<B", 1)}} # 461 vendor objects
ATOMIC_WIDE_T0 = {40: (4, "<I", 80), 29: (8, "<I", 1), 18: (23, "<B", 1),
                  6: (24, "<I", None), 5: (28, "<I", None), 2: (32, "<I", 40),
                  38: (36, "<I", 12), 7: (40, "<I", 16)}
ATOMIC_SIZE, ATOMIC_TLEN = 224, 48
ATOMIC_SIZES = (216, 224)


def build(entry, size=None, f0_5=0, f3_3=1, check_align=True, restore=False, atomic=False):
    """The whole section, from the entry PC. Everything else is structure or a constant.

    `atomic` adds the device-scope atomic declaration a program carrying one needs. At 216 bytes
    that is four bytes and nothing moves; at 224 it is Apple's wider body layout, which belongs to
    that size. Off by default and the default path is byte-identical to before, which is the
    control - adding the declaration must not move the sections that already execute.
    """
    if check_align and entry % ENTRY_ALIGN:
        raise ValueError("entry PC %d is not a multiple of %d; the shader does not start"
                         % (entry, ENTRY_ALIGN))
    # A DEFAULT IS NOT A REQUEST. This read `if atomic and size == 216: size = ATOMIC_SIZE`,
    # which silently gave 224 bytes to a caller who explicitly asked for 216 - the same shape as
    # a `.get(dtype, 2)` swallowing an unknown type. And 216 is a real request: the atomic body
    # occupies 136..184, so it FITS, which is what makes an in-place splice into a 216-byte host
    # section possible at all. Apple's 224 is what Apple puts after the body, not what the body
    # needs.
    if size is None:
        size = ATOMIC_SIZE if atomic else 216
    # A LAYOUT BELONGS TO A SIZE. These offsets are section-relative, and using one shape's
    # offsets at another shape's size points the loader past the data - which segfaulted the
    # driver's metadata loader, silently, behind a section that described and round-tripped
    # perfectly. So an unwitnessed size is refused rather than rescaled.
    if atomic and size not in ATOMIC_SIZES:
        raise ValueError("no atomic layout is measured for a %d-byte section; the witnessed "
                         "sizes are %s, and these field offsets are section-relative - the "
                         "224-byte shape's offsets in a 216-byte section fault the driver's "
                         "metadata loader" % (size, ", ".join(str(x) for x in ATOMIC_SIZES)))
    b = bytearray(size)
    struct.pack_into("<I", b, 0, _ROOT)
    _put_table(b, _ROOT, _RVT, 4, 12,
               {0: (8, "<I", _T0 - (_ROOT + 8)), 3: (4, "<I", _T3 - (_ROOT + 4))})
    t3 = {3: (5, "<B", f3_3)}
    if atomic:
        t3.update(ATOMIC_DECL_T3)
    _put_table(b, _T3, _T3VT, _T3_SLOTS, 6, t3)
    if atomic and size == 224:
        fields = {slot: (off, fmt,
                         entry if slot == 6 else f0_5 if slot == 5 else val)
                  for slot, (off, fmt, val) in ATOMIC_WIDE_T0.items()}
        fields.update(ATOMIC_DECL_T0[224])
        _put_table(b, _T0, _T0VT, _T0_SLOTS, ATOMIC_TLEN, fields)
    else:
        # The 216-byte shape, atomic or not, is ONE layout: the declaration adds a slot and
        # moves nothing, so the non-atomic path is the atomic path minus four bytes.
        fields = {6: (20, "<I", entry), 5: (24, "<I", f0_5)}
        if atomic:
            fields.update(ATOMIC_DECL_T0[216])
        _put_table(b, _T0, _T0VT, _T0_SLOTS, 0, fields)
    if restore:
        for off, val in RESTORE:
            if off + 4 <= len(b):
                struct.pack_into("<I", b, off, val)
        for off, val in RESTORE_BYTES:
            if off < len(b):
                b[off] = val
        if 0xcc + len(STAGE) <= len(b):
            b[0xcc:0xcc + len(STAGE)] = STAGE
    return bytes(b)


def build_arch(size=32, f0=0):
    """__GPU_ARCH_LD_MD: one table, one present-but-zero field. All five non-zero bytes are
    structure - the field's value is inert, 8 and 999 both execute."""
    b = bytearray(size)
    struct.pack_into("<I", b, 0, 12)
    _put_table(b, 12, 6, 1, 8, {0: (4, "<I", f0)})
    return bytes(b)


def main(argv=None):
    from . import imgconst_scalar as K
    for got, want, name in ((build(entry=0x40), K.GPU_LD_MD, "__GPU_LD_MD"),
                            (build_arch(), K.GPU_ARCH_LD_MD, "__GPU_ARCH_LD_MD")):
        bad = [i for i in range(len(want)) if got[i] != want[i]]
        print("%-18s %s" % (name, "byte-identical to the measured blob (%d bytes, %d non-zero)"
                            % (len(want), sum(1 for v in want if v)) if not bad
                            else "DIFFERS at %s" % bad[:12]))


if __name__ == "__main__":
    main()
