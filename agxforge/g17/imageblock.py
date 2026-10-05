"""The explicit-layout IMAGEBLOCK declaration, applied to authored metadata sections.

MEASURED 2026-09-23 from Apple's compiler (explicit-layout compute kernels, each built in its own
process) and proven on hardware (tools/g17imageblock.py, isa/g17-imageblock-receipt.json):

    per-kernel table   slot 19 = 1        a one-byte field at table offset 53, in padding that the
                                          three-buffer class already carries - nothing else moves
    __GPU_LD_MD        t136 slot 23 = 1   vtable byte 100 = 18, body byte 154 = 1: the neighbour of
                                          the device atomic's slot 24 (agxforge.g17.ldmd), and like it
                                          an in-place write into the 216-byte shape
    __GPU_ARCH_LD_MD   40 bytes           the root gains a field pointing at a subtable whose u32 at
                                          byte 32 is the imageblock ELEMENT SIZE in bytes

Seven layouts from a 2-byte half to a 32-byte struct move nothing but byte 32, which equals
sizeof(struct) in 7 of 7. The driver sizes the allocation from it: declared 4 -> 128 bytes and
declared 8 -> 256 bytes at 32x1, and with no declaration the pipeline asks for 0 bytes and every
imageblock read returns 0.

THE LAYOUT MUST BE EXPLICIT. Apple's compute backend itself rejects an implicit-layout imageblock
(ledger/g17-the-implicit-imageblock-is-not-a-compute-construct.toml); that refusal is Apple's and
stays. The declaration also says nothing about the program's slot-29 vector: an imageblock program
reads SR_LOCAL_X and SR_LOCAL_Y (164, 165) to build its tile coordinate, and the author declares
those as it declares any system register.
"""
import struct

PK_SLOT = 19
PK_OFFSET = 53
TENSOR_SLOT, TENSOR_SLOT_OFFSET = 44, 55   # the tensor class's slot-44 byte, where slot 19 goes instead
LD_SIZE = 216
LD_VT_BYTE, LD_VT_VALUE = 100, 18      # t136 vtable entry for slot 23
LD_BODY_BYTE = 154                     # t136 body byte for slot 23
ARCH_SCALAR = bytes.fromhex("0c00000000000600080004000600000008000000040004000400000000000000")
_ARCH_IMAGEBLOCK = bytes.fromhex("0c000000000006000a000400060000000c000000000006000800040006000000"
                                 "0000000000000000")
ARCH_SIZE_BYTE = 32


class Refused(ValueError):
    pass


def arch(element_bytes):
    """__GPU_ARCH_LD_MD for an imageblock whose element is `element_bytes` bytes."""
    if not isinstance(element_bytes, int) or not 0 < element_bytes < 1 << 16:
        raise Refused("imageblock element size %r: state sizeof(the element struct) in bytes"
                      % (element_bytes,))
    b = bytearray(_ARCH_IMAGEBLOCK)
    struct.pack_into("<I", b, ARCH_SIZE_BYTE, element_bytes)
    return bytes(b)


def pk_declared(metadata):
    """True when the per-kernel table already carries slot 19 = 1 (a tensor author wrote it in the tail)."""
    from . import gpumd as GM
    md = bytes(metadata)
    t = GM.kernel_table(md)
    if t is None:
        return False
    vt = t - struct.unpack_from("<i", md, t)[0]
    vsz = struct.unpack_from("<H", md, vt)[0]
    if 4 + 2 * PK_SLOT >= vsz:
        return False
    off = struct.unpack_from("<H", md, vt + 4 + 2 * PK_SLOT)[0]
    return bool(off) and md[t + off] == 1


def declare(metadata, ld_md, arch_ld_md, element_bytes, per_kernel=True):
    """(metadata, ld_md, arch_ld_md) with the imageblock declared, or Refused.

    Applied only where it was measured: the per-kernel table must leave offset 53 unused and slot
    19 absent, the LD_MD must be the 216-byte shape with slot 23 absent, and the ARCH section must
    be the 32-byte one this backend emits. Any other shape is refused rather than guessed at - the
    atomic's 224-byte layout at the wrong size faulted the driver's metadata loader.
    """
    from . import gpumd as GM
    if not per_kernel:
        # THE PER-KERNEL HALF IS ALREADY AUTHORED: the straight-line tensor table has to grow to hold
        # slot 19, which is a serializer's job (tensormetadata.layout), not a patch's. Only confirm it.
        if not pk_declared(metadata):
            raise Refused("per_kernel=False but the per-kernel table carries no slot 19 = 1")
        md, ld, arch_ = bytes(metadata), bytearray(ld_md), arch_ld_md
        return (md,) + _ld_arch(ld, arch_, element_bytes)
    md = bytearray(metadata)
    t = GM.kernel_table(bytes(md))
    if t is None:
        raise Refused("the metadata has no per-kernel table")
    vt = t - struct.unpack_from("<i", md, t)[0]
    vsz, tsz = struct.unpack_from("<HH", md, vt)
    offsets = [struct.unpack_from("<H", md, vt + 4 + 2 * i)[0] for i in range((vsz - 4) // 2)]
    if PK_SLOT >= len(offsets):
        raise Refused("this per-kernel vtable does not reach slot 19; no imageblock layout is measured "
                      "for a table that has to grow")
    if offsets[PK_SLOT]:
        raise Refused("per-kernel slot 19 is already present")
    if PK_OFFSET >= tsz:
        raise Refused("the per-kernel table is %d bytes and has no byte 53 to hold slot 19" % tsz)
    # Per-kernel fields are at most four bytes wide, so only a field starting at 50..53 can cover
    # byte 53; in the measured class slot 1 is a u32 at 48 and 52..53 are padding. (Reading a
    # field's width as "the gap to the next field" would call that padding occupied.)
    if {o for o in offsets if o} & set(range(PK_OFFSET - 3, PK_OFFSET + 1)) or md[t + PK_OFFSET]:
        raise Refused("byte 53 of the per-kernel table is in use; slot 19 is measured only where it "
                      "is free padding")
    # A TENSOR TABLE TAKES APPLE'S OWN PLACEMENT, not the free byte: in the tensor class slot 44 is
    # the single-byte tail at 55, and Apple's compile of a matmul2d kernel with an explicit
    # imageblock puts slot 19 at 55 and moves slot 44 down to 54 (results/g17-tensor-imageblock-
    # witness-v1, both.o), every other byte of the three sections unchanged. Reproduced exactly so
    # the image is Apple's layout, not merely a valid one.
    pos = PK_OFFSET
    if len(offsets) > TENSOR_SLOT and offsets[TENSOR_SLOT] == TENSOR_SLOT_OFFSET and not md[t + TENSOR_SLOT_OFFSET - 1]:
        md[t + TENSOR_SLOT_OFFSET - 1] = md[t + TENSOR_SLOT_OFFSET]
        struct.pack_into("<H", md, vt + 4 + 2 * TENSOR_SLOT, TENSOR_SLOT_OFFSET - 1)
        pos = TENSOR_SLOT_OFFSET
    struct.pack_into("<H", md, vt + 4 + 2 * PK_SLOT, pos)
    md[t + pos] = 1
    return (bytes(md),) + _ld_arch(bytearray(ld_md), arch_ld_md, element_bytes)


def _ld_arch(ld, arch_ld_md, element_bytes):
    if len(ld) != LD_SIZE:
        raise Refused("no imageblock declaration is measured for a %d-byte __GPU_LD_MD; the "
                      "witnessed shape is %d bytes, and these offsets are section-relative"
                      % (len(ld), LD_SIZE))
    if ld[LD_VT_BYTE] or ld[LD_BODY_BYTE]:
        raise Refused("__GPU_LD_MD t136 slot 23 is already occupied")
    ld[LD_VT_BYTE], ld[LD_BODY_BYTE] = LD_VT_VALUE, 1
    if len(arch_ld_md) != 32:
        raise Refused("__GPU_ARCH_LD_MD is not the 32-byte shape this declaration replaces")
    return bytes(ld), arch(element_bytes)
