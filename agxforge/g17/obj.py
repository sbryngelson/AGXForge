#!/usr/bin/env python3
"""Serialize a complete G17 native object - the Mach-O container, not just its code.

Every executed program in this project so far has been the compiler's __text spliced into an
Apple-built container. This builds the container: mach_header_64, one LC_SEGMENT_64 carrying the
six sections, LC_SYMTAB naming the two entry points, and LC_DYSYMTAB, with the section payloads
laid out after them.

WHAT THAT REQUIRES KNOWING, and it is now measured rather than assumed:

  __TEXT,__text            the compiler generates it
  __GPU_STATS_MD           NOT READ BY THE LOADER, and now measured outside buffer kernels rather
                           than assumed from them. It is 96 bytes with 20,999 DISTINCT patterns
                           over 20,999 objects, so a byte census can only say that it varies. The
                           dispatch says more: on the authored TEXTURE kernel, this section set to
                           zeros, to 0xFF, to a buffer kernel's own 96 bytes, and REMOVED ENTIRELY
                           each leave the texture read exact - in a harness that tracks the
                           coordinate and where those same four mutations of each of the other
                           three sections kill Apple's driver inside newComputePipelineState.
                           tools/g17statsprobe.py, isa/g17-loader-reads-probe.json.
  __GPU_ARCH_LD_MD         needed, and NOT byte-identical. "Byte-identical across every kernel
                           measured" is FALSE over the cache: 8 patterns in two sizes. The section
                           is a root table plus a sub-table carrying ONE optional field - 40 bytes
                           when it is set, 32 when it elides - and the flag is set in 6,234 of
                           21,001 objects. No contract fact and no opcode determines it (seven
                           facts tested, best leaving 3,144 exceptions), so it is the compiler's.
                           THE STRUCTURE IS READ AND THE BOOLEAN, ON ONE KERNEL, IS NOT: zeroing,
                           0xFF-filling or removing the section kills the driver, but substituting
                           a6-2d's valid 32-byte form - the OPPOSITE value of the flag - for
                           ty-2d's 40-byte form leaves the texture read exact. One kernel, one
                           dispatch per value: enough to say the flag did not decide correctness
                           here, not enough to say it never does.
  __GPU_LD_MD              needed, and NOT constant, and A WRONG ONE LOADS SILENTLY: substituting
                           tg-16x48x64's buffer-shaped section into the texture kernel's image -
                           the exact defect shipped on 2026-09-08 - BUILDS A PIPELINE with no
                           error. There is no load-time check to catch it, which is why the
                           authoring tool refuses this input rather than defaulting it.
                           Byte-identical across the 431 objects it was
                           measured on - every one of them a BUFFER kernel of one size class. Over
                           the whole cache it is 216 to 248 bytes without a texture and 224 to 416
                           with one; ty-2d and tu-use1 are 336 where g17imgconst.GPU_LD_MD is 224.
                           An image built from that constant is 112 bytes short for a texture
                           kernel, and Apple's own instructions in Apple's own section do not read
                           in it. The original wording is why.
  __GPU_METADATA           needed, and NOT interchangeable. "Another shape's runs this kernel
                           correctly" holds for the buffer-kernel pair it was measured on and is
                           FALSE in general: on 2026-09-08 a section built for a texture kernel
                           without an internal binding loaded, created a pipeline, and HUNG THE
                           GPU. A texture fetch reaches its descriptor through a binding this side
                           now emits, 6,495 of 6,495 Apple sections and 399 of 399 corpus.

SO THE FOUR METADATA SECTIONS ARE NOT A CLASS CONSTANT. That sentence stood on the three claims
above and two of them are false. What survives is the shape of the finding - the object is mostly
not kernel-specific - not the claim that any of it can be transplanted.

THE SYMBOL TABLE, re-measured: _agc.main is present in all 21,057 objects and its value is 64 in
18,582 of them, not all - it ranges to 5,568 and is a multiple of 64 in every one.
_agc.main.constant_program is at 0 wherever it appears and is ABSENT in 14 objects.

The symbol table is what the loader finds the entry points through: _agc.main at the entry offset
and _agc.main.constant_program at 0.
"""
import struct

LC_SEGMENT_64, LC_SYMTAB, LC_DYSYMTAB = 0x19, 0x02, 0x0B
MH_MAGIC_64, MH_OBJECT = 0xFEEDFACF, 1
from . import target as g17target
# from Apple's own objects for this arch; see tools/g17target.py
CPUTYPE, CPUSUBTYPE = g17target.CPUTYPE, g17target.CPUSUBTYPE
S_ATTR_SOME_INSTRUCTIONS = 0x80000400        # __text's section flags

# section name, segment name, alignment (log2), flags
LAYOUT = [("__text", "__TEXT", 6, S_ATTR_SOME_INSTRUCTIONS),
          ("__compute", "__GPU_METADATA", 3, 0),
          ("__compute", "__GPU_REMARKS_MD", 3, 0),
          ("__compute", "__GPU_LD_MD", 3, 0),
          ("__compute", "__GPU_ARCH_LD_MD", 3, 0),
          ("__compute", "__GPU_STATS_MD", 3, 0)]


def _align(n, a):
    return (n + a - 1) & ~(a - 1)


def build(text, metadata, ld_md, arch_ld_md, stats_md, entry, remarks=b"", extra_pad=0):
    """A complete object. `entry` is the offset of _agc.main within __text."""
    payload = {"__TEXT,__text": bytes(text),
               "__GPU_METADATA,__compute": bytes(metadata),
               "__GPU_REMARKS_MD,__compute": bytes(remarks),
               "__GPU_LD_MD,__compute": bytes(ld_md),
               "__GPU_ARCH_LD_MD,__compute": bytes(arch_ld_md),
               "__GPU_STATS_MD,__compute": bytes(stats_md)}
    ncmds = 3
    sizeofcmds = (72 + 80 * len(LAYOUT)) + 24 + 80
    cursor = 32 + sizeofcmds + extra_pad

    # Apple places the symbol table BETWEEN sections; the loader does not care, so this lays the
    # sections out first and the tables after, which is a different layout from the input and is
    # the point: reproducing Apple's byte order would not show the container was constructed.
    # `align` is the ADDRESS alignment, not the file offset's - Apple's own __text sits at 688,
    # which is not 64-aligned though its section header says align 6. Aligning the file offset too
    # was the first version here and it made the object 16 bytes too large to splice in place.
    offs, addr = {}, 0
    for nm, sg, al, _fl in LAYOUT:
        key = sg + "," + nm
        cursor = _align(cursor, 8)
        addr = _align(addr, 1 << al)
        offs[key] = (cursor, addr, len(payload[key]))
        cursor += len(payload[key])
        addr += len(payload[key])

    cursor = _align(cursor, 8)
    symoff = cursor
    names = [("_agc.main", offs["__TEXT,__text"][1] + entry),
             ("_agc.main.constant_program", 0)]
    cursor += 16 * len(names)
    stroff = cursor
    strtab = b"\0"
    stridx = {}
    for n, _v in names:
        stridx[n] = len(strtab); strtab += n.encode() + b"\0"
    strtab += b"\0" * ((-len(strtab)) % 4)
    cursor += len(strtab)
    total = cursor

    out = bytearray(total)
    struct.pack_into("<IiiIIIII", out, 0, MH_MAGIC_64, CPUTYPE, CPUSUBTYPE, MH_OBJECT,
                     ncmds, sizeofcmds, 0, 0)
    o = 32
    seg_file_off = min(v[0] for v in offs.values())
    seg_file_size = max(v[0] + v[2] for v in offs.values()) - seg_file_off
    struct.pack_into("<II", out, o, LC_SEGMENT_64, 72 + 80 * len(LAYOUT))
    out[o+8:o+24] = b"\0" * 16                       # the segment is unnamed in Apple's objects
    struct.pack_into("<QQQQiiII", out, o + 24, 0, _align(addr, 8), seg_file_off, seg_file_size,
                     7, 7, len(LAYOUT), 0)
    so = o + 72
    for nm, sg, al, fl in LAYOUT:
        key = sg + "," + nm
        foff, vaddr, size = offs[key]
        out[so:so+16] = nm.encode().ljust(16, b"\0")
        out[so+16:so+32] = sg.encode().ljust(16, b"\0")
        struct.pack_into("<QQIIIIIIII", out, so + 32, vaddr, size, foff, al, 0, 0, fl, 0, 0, 0)
        so += 80
    o += 72 + 80 * len(LAYOUT)
    struct.pack_into("<IIIIII", out, o, LC_SYMTAB, 24, symoff, len(names), stroff, len(strtab))
    o += 24
    struct.pack_into("<II", out, o, LC_DYSYMTAB, 80)
    struct.pack_into("<II", out, o + 20, len(names), len(names))

    for i, (n, val) in enumerate(names):
        # n_strx, n_type = N_SECT|N_EXT, n_sect = 1 (__text), n_desc, n_value
        struct.pack_into("<IBBHQ", out, symoff + 16 * i, stridx[n], 0x0E | 0x01, 1, 0, val)
    out[stroff:stroff+len(strtab)] = strtab
    for key, (foff, _a, size) in offs.items():
        out[foff:foff+size] = payload[key]
    return bytes(out)


def provenance(text, metadata, ld_md, arch_ld_md, stats_md, entry, remarks=b"", extra_pad=0,
               inert_bytes=None):
    """How much of the object this project generates, byte for byte.

    The mission's progress metric is "the fraction of the native image generated independently",
    and it has never been computed. Every region is one of:

      GENERATED   emitted from semantics or from a rule this project can state
      CONSTANT    a value measured to be architectural, emitted with its evidence
      INHERITED   Apple bytes carried with an explicit dependency marker

    The header, the segment and section table, the symbol table and the string table are all
    computed from the section list, so they are GENERATED. __TEXT is the compiler's. The stats
    section is NOT READ by the loader - and that is now measured on a texture kernel as well as the
    buffer kernels it was first claimed from: zeros, 0xFF, a foreign kernel's own 96 bytes and no
    section at all each leave the authored texture read exact, where the same four mutations of the
    other three sections kill Apple's driver (tools/g17statsprobe.py). So emitting zeros is
    generation, not inheritance.

    __GPU_LD_MD and __GPU_ARCH_LD_MD are now GENERATED. Read as FlatBuffers tables rather than as
    blobs, every non-zero byte in the pair is structure, the entry PC, or the single constant
    [3,3]=1; tools/g17ldmd.py emits them from the entry offset and asserts byte-identity against
    the measured blob. The entry PC was named causally by moving the code and the field
    independently (spike/accel/re/entryoff.py), so this is a construction, not a copy.

    __GPU_METADATA still varies by class and is INHERITED apart from the binding table this
    project writes and the bytes measured inert: a declared dependency
    (ledger/g17-image-region-dependence.toml) rather than an unexamined copy.
    """
    obj = build(text, metadata, ld_md, arch_ld_md, stats_md, entry, remarks, extra_pad)
    # MEASURED-INERT BYTES COUNT AS GENERATED. A 16-byte window sweep over each section, zeroing
    # one window at a time and running the kernel, classifies every window as inert (still
    # correct), state-carrying (runs, wrong answer) or structural (loader segfaults). Zeroing ALL
    # the inert windows at once - which is the test that matters, since a byte can be inert alone
    # and load-bearing in company - leaves 428 of 764 metadata bytes at zero and the kernel
    # computing 768 of 768 cells correctly. Emitting a zero this project has shown the loader does
    # not read is generation, not inheritance.
    # spike/accel/re/mdbisect.py and mdminimal.py, ledger/g17-metadata-inert-fraction.toml
    inert = inert_bytes if inert_bytes is not None else 0
    inherited = len(metadata) - inert
    constant = 0
    generated = len(obj) - inherited
    return {"total": len(obj), "generated": generated, "constant": constant,
            "inherited": inherited,
            "detail": [("mach-o header, segment, sections, symbols", "GENERATED",
                        len(obj) - len(text) - len(ld_md) - len(arch_ld_md) - len(stats_md)
                        - len(metadata)),
                       ("__TEXT,__text", "GENERATED", len(text)),
                       ("__GPU_STATS_MD (measured not read, incl. texture)", "GENERATED",
                        len(stats_md)),
                       ("__GPU_LD_MD (structure + entry PC + one constant)", "GENERATED",
                        len(ld_md)),
                       ("__GPU_ARCH_LD_MD (structure around an inert field)", "GENERATED",
                        len(arch_ld_md)),
                       ("metadata bytes measured INERT (emitted as zeros)", "GENERATED", inert),
                       ("__GPU_METADATA, load-bearing remainder", "INHERITED", inherited)]}


def sections_of(obj):
    """Convenience: the same view machobj.parse gives, for checking a built object."""
    # The sys.path insert this carried is gone: machobj is a package module now, and the insert
    # pointed at ../spike/accel/re, which resolves to agxforge/spike from here and does not exist.
    from . import machobj
    return machobj.parse(obj)
