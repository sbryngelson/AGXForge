#!/usr/bin/env python3
"""THE ARCHIVE CONTAINER'S HEAD, written field by field instead of copied.

Slice 1 of the archive is a Mach-O bundle, and its first 1,472 bytes were the last thing in the
image still taken from Apple whole: a 32-byte header, 848 bytes of load commands, and a 592-byte
table region the loader segfaults without. g17arc copied that block and patched the offsets it knew
about, which is not the same as owning it.

Diffing the block across three archives built from unrelated kernels says how much of it is a
kernel fact at all. Twenty-two spans differ, and every one of them is either a size, an offset, a
UUID, or the module hash:

    456/472/536/608/616/624/688/704/768/776/784   section and segment sizes in the load commands
    864..880                                      LC_UUID, proven inert by blanking
    1008..1026, 1032                              the metallib's UUID, offset and size
    1048..1080, 1196..1228                        the module hash, twice - the same 32 bytes the
                                                  metallib's own function list carries
    1120, 1152, 1156, 1164, 1172, 1444..1458      descriptor, object and reflection placement

Everything else - 1,380 of 1,472 bytes - is identical in all three. That is the class-constant
result the image has shown at every level: what the loader requires is overwhelmingly a shape, and
the kernel-specific part of it is placement. So this file writes the shape from named fields and
computes the placement, and nothing here is a copied blob.

The eight AIR_DATA tables are declared twice: a load command each - a 16-byte name and the
(offset, size) of a DESCRIPTOR - and then the descriptor itself, sixteen bytes at 880 + 16i giving
where that table's entries start and how many there are. The entries are the size-stepping
structure the layout implies:

    AIR_METALLIB      1008  40 bytes   uuid, offset, size-4
    AIR_MODULE        1048  40 bytes   the 32-byte module hash
    AIR_DESCRIPTOR    1088  56 bytes   a 32-byte key identical in every archive, then placement
    AIR_OBJECT        1144  36 bytes   __compute and __reflection placement
    AIR_PIPELINE      1180   8 bytes
    AIR_HASHES        1188  2 entries  the module hash, then SHA-256 of the empty string
    AIR_OBJECT_INDEX     -  absent     offset 0, count 0
    AIR_STRTABLE      1460  4 entries  four zero bytes
"""
import struct

LC_SEGMENT_64, LC_UUID, LC_TABLE, LC_BUILD = 0x19, 0x1B, 0x31, 0x32
MH_MAGIC_64, MH_BUNDLE = 0xFEEDFACF, 13
CPUTYPE, CPUSUBTYPE = 16777235, 0x163

HEAD_LEN = 1472           # the Mach-O header, the load commands, and the table region
TABLES_OFF = 880
SECTION_NAMES = ("__reflection", "__compute", "__descriptor", "__metallib")
TABLES = ("AIR_METALLIB", "AIR_MODULE", "AIR_DESCRIPTOR", "AIR_OBJECT",
          "AIR_OBJECT_INDEX", "AIR_PIPELINE", "AIR_HASHES", "AIR_STRTABLE")

# WHICH OF THE CLASS CONSTANTS ARE ACTUALLY READ, one zeroed at a time and dispatched
# (spike/accel/re/classconst.py). Reproducing a constant is not knowing what it is, and the first
# question about one is whether anything reads it. Of the 148 bytes that were carried as
# "identical in every archive measured", 76 are inert and are emitted as zeros:
#
#     the AIR_DESCRIPTOR key      32   zeroed AND filled with junk, both correct - the value that
#                                      differs in Apple's blit shaders is not checked at all
#     the descriptor's tail       16   inert
#     four of the eight words at 1296  inert
#     three of the four at 1424        inert
#     four of the eight in LC 0x32     inert, including the one holding an OS version
#
# and 72 are REQUIRED, each now a named field rather than an unexplained byte:
#
#     SHA-256 of the empty string 32   value-checked: junk in its place gets no pipeline, so the
#                                      loader is comparing a hash of empty content, not a token
#     1296 words 0,1,2 and 7      16   0x402, 0x403, 0xC0000000, 1
#     1424 word 1                  4   0x408
#     the flag at 1440             4   0x40000004
#     LC 0x32 words 0,3,6,7       16   1, 2, 0x403, 0x20001
DESCRIPTOR_KEY = bytes(32)      # measured value in ledger/g17-class-constants-are-mostly-inert
SHA256_EMPTY = bytes.fromhex("e3b0c44298fc1c149afbf4c8996fb924"
                             "27ae41e4649b934ca495991b7852b855")
BUILD_CMD = bytes.fromhex("01000000" "00000000" "00000000" "02000000"
                          "00000000" "00000000" "03040000" "01000200")
DESCRIPTOR_TAIL = (392, 16)     # the descriptor's recorded size and a step; both proven inert
TAIL_WORDS = (0x402, 0x403, 0xC0000000, 0, 0, 0, 0, 1)   # 1296..1328; words 3..6 proven inert
TAIL2 = (0, 0x408, 0, 0)                                 # 1424..1440; only word 1 is read
TAIL3_FLAG = 0x40000004                                  # 1440
# The bytes above that execution shows are read. Everything else in these runs is zeros.
READ_BYTES = 32 + 16 + 4 + 4 + 16


def _seg(name, vmaddr, vmsize, fileoff, filesize, nsects):
    return (struct.pack("<II", LC_SEGMENT_64, 72 + 80 * nsects)
            + name.encode().ljust(16, b"\0")
            + struct.pack("<QQQQiiII", vmaddr, vmsize, fileoff, filesize, 1, 1, nsects, 0))


def _sect(name, addr, size, align=4):
    return (name.encode().ljust(16, b"\0") + b"__TEXT".ljust(16, b"\0")
            + struct.pack("<QQIIIIIIII", addr, size, addr, align, 0, 0, 0, 0, 0, 0))


def build(sizes, uuid=b"\0" * 16, module_hash=b"\0" * 32, metallib_uuid=b"\0" * 16,
          obj_size=None, refl_size=None):
    """The container's first 1,472 bytes for a given section layout.

    `sizes` maps each of the four __TEXT section names to its byte length; the offsets follow from
    them, since the sections are contiguous from the end of this block with no alignment padding
    in any archive measured.

    `obj_size` and `refl_size` are the payloads' own lengths where those are shorter than their
    sections. Apple pads __compute to eight bytes and __reflection to sixteen and records the
    unpadded length in both cases - across 7,491 archives the reflection section runs 0, 4, 8 or 12
    bytes past what the table says. Nothing here needs to pad, so both default to the section size.
    """
    off, place = HEAD_LEN, {}
    for n in SECTION_NAMES:
        place[n] = (off, sizes[n]); off += sizes[n]
    total = off

    cmds = _seg("__AIR_DATA", 0, HEAD_LEN, 0, HEAD_LEN, 0)
    for i, t in enumerate(TABLES):
        cmds += (struct.pack("<II", LC_TABLE, 40) + t.encode().ljust(16, b"\0")
                 + struct.pack("<QQ", TABLES_OFF + 16 * i, 16))
    cmds += _seg("__TEXT", HEAD_LEN, total - HEAD_LEN, HEAD_LEN, total - HEAD_LEN, 4)
    for n in SECTION_NAMES:
        cmds += _sect(n, place[n][0], place[n][1])
    cmds += struct.pack("<II", LC_BUILD, 40) + BUILD_CMD
    cmds += struct.pack("<II", LC_UUID, 24) + uuid
    head = struct.pack("<IIiIIIII", MH_MAGIC_64, CPUTYPE, CPUSUBTYPE, MH_BUNDLE,
                       len(TABLES) + 4, len(cmds), 0, 0) + cmds
    assert len(head) == TABLES_OFF, (len(head), TABLES_OFF)

    ml_off, ml_size = place["__metallib"]
    ds_off, _ = place["__descriptor"]
    ob_off, ob_size = place["__compute"]
    if obj_size is not None:
        ob_size = obj_size
    rf_off, rf_size = place["__reflection"]
    # The metallib's recorded size is four short in every archive - its own trailing word - and
    # the reflection's is whatever padding was applied.
    rf = rf_size if refl_size is None else refl_size
    # Where each table's entries live, and how many. The offsets follow from the entry sizes;
    # AIR_OBJECT_INDEX is declared with no entries in every archive measured.
    entries = {"AIR_METALLIB": (1008, 1), "AIR_MODULE": (1048, 1), "AIR_DESCRIPTOR": (1088, 1),
               "AIR_OBJECT": (1144, 1), "AIR_OBJECT_INDEX": (0, 0), "AIR_PIPELINE": (1180, 1),
               "AIR_HASHES": (1188, 2), "AIR_STRTABLE": (1460, 4)}
    t = bytearray()
    for name in TABLES:
        t += struct.pack("<QQ", *entries[name])
    t += metallib_uuid + struct.pack("<QQQ", ml_off, ml_size - 4, 0)          # AIR_METALLIB
    t += module_hash + struct.pack("<Q", 0)                                    # AIR_MODULE
    t += DESCRIPTOR_KEY + struct.pack("<QQQ", ds_off, *DESCRIPTOR_TAIL)        # AIR_DESCRIPTOR
    t += struct.pack("<QIIIIIII", 0, ob_off, ob_size, rf_off, rf, rf_off, rf, 0)  # AIR_OBJECT
    t += struct.pack("<II", 1, 0)                                              # AIR_PIPELINE
    t += struct.pack("<II", 1, 0) + module_hash + b"\0" * 32 + SHA256_EMPTY + b"\0" * 4
    t += struct.pack("<8I", *TAIL_WORDS) + b"\0" * 96
    t += struct.pack("<4I", *TAIL2) + struct.pack("<5I", TAIL3_FLAG, ob_off, ob_size, rf_off, rf)
    t += b"\0" * 12                                                            # AIR_STRTABLE
    assert len(t) == HEAD_LEN - TABLES_OFF, (len(t), HEAD_LEN - TABLES_OFF)
    return bytes(head) + bytes(t), place
