"""THE ARCHIVE ENVELOPE, generated around our object.

The G17 object is delivered inside an MTLBinaryArchive: a fat Mach-O with two slices.

    slice 0   an MTLB metallib
    slice 1   a Mach-O bundle whose __TEXT holds four sections
                __reflection    __compute  <- the G17 object    __descriptor    __metallib

WHAT THE LOADER READS, measured by blanking one region at a time and running the novel kernel
(spike/accel/re/envblank.py, airmin.py, headbisect.py, hashkey.py):

    __reflection      480 bytes   zeroing it leaves the kernel correct      INERT
    __descriptor      400 bytes   zeroing it leaves the kernel correct      INERT
    LC_UUID            16 bytes   zeroing it leaves the kernel correct      INERT
    both metallibs                every region inside them zeroed one at a time - the bitcode, the
                                  function name, its hash, the UUID, the reflection and source
                                  lists - and the kernel stayed correct. What the loader wants is a
                                  metallib-SHAPED file, and g17mtlb writes one in 360 bytes
    the module hash    32 bytes   in the container's AIR_MODULE and AIR_HASHES tables, and there
                                  only: it is the archive's CACHE KEY. A hash of our own choosing
                                  gets no pipeline; the source library's own runs. The same field
                                  in the metallibs is free                  REQUIRED, and an input
    the container head 1472 bytes generated field by field by g17container - byte-exact against
                                  7,468 of the 7,491 archives in the cache

So nothing in the envelope is copied any more. `emit` builds a whole archive from payloads alone;
`build` is the older path that starts from a host archive's description, kept because it is what
the earlier probes call.
"""
import hashlib, struct

from . import container as g17container
from . import mtlb as g17mtlb

# Apple uses 0xCBFEBABE here, not the familiar 0xCAFEBABE, with 20-byte arch entries. Taken from
# the archive being described rather than hard-coded, so a different magic is carried through.
FAT_MAGIC = 0xCBFEBABE
LC_SEGMENT_64, LC_UUID = 0x19, 0x1B
LC_TABLE, LC_VERSIONISH = 0x31, 0x32          # named-table entries, and a version/flags record


def parse(fat):
    """-> a description of an existing archive: slice offsets, container sections, table region."""
    magic, narch = struct.unpack_from(">II", fat, 0)
    slices = [struct.unpack_from(">IIIII", fat, 8 + 20 * i) for i in range(narch)]
    nat = slices[1][2]
    ncmds, szcmds = struct.unpack_from("<II", fat, nat + 16)
    sects, cmds = {}, []
    p = nat + 32
    for _ in range(ncmds):
        cmd, cs = struct.unpack_from("<II", fat, p)
        cmds.append((cmd, bytes(fat[p:p + cs])))
        if cmd == LC_SEGMENT_64:
            seg = fat[p + 8:p + 24].rstrip(b"\0").decode()
            nsects = struct.unpack_from("<I", fat, p + 64)[0]
            q = p + 72
            for _ in range(nsects):
                nm = fat[q:q + 16].rstrip(b"\0").decode()
                size = struct.unpack_from("<Q", fat, q + 40)[0]
                off = struct.unpack_from("<I", fat, q + 48)[0]
                sects[nm] = (nat + off, size, seg)
                q += 80
        p += cs
    hdr_end = 32 + szcmds
    return dict(magic=magic, slices=slices, nat=nat, ncmds=ncmds, szcmds=szcmds, cmds=cmds,
                sects=sects,
                air=(slices[0][2], slices[0][3]),
                air_data_tables=(nat + hdr_end, 1472 - hdr_end),
                header=bytes(fat[nat:nat + 32]))


# The four sections of the container's __TEXT, in file order, and which of them the loader reads.
INERT_SECTIONS = ("__reflection", "__descriptor")

# THE OBJECT'S SIZE IS RECORDED TWICE in the container's __AIR_DATA table region, at these offsets
# from its start. Found by searching the region for each host's own object length: 2160, 1632 and
# 1912 all appear at exactly 276 and 568 in their own archives.
#
# It is the SIZE and not a content hash. Apple's archive accepts our object - whose bytes are
# entirely different from the one it was built around - as long as the length matches, and rejects
# a correct object of a different length. Writing these two slots is what makes the archive accept
# an object of any size.
OBJ_SIZE_SLOTS = (276, 568)

# THE CONTAINER ALSO RECORDS THE METALLIB'S LENGTH, in two slots of the same table, and a build that
# changes the AIR without changing them produces an archive the loader rejects as "invalid format".
# That is what a first attempt at transplanting another kernel's AIR actually measured - a size
# mismatch this file created - and it would have read as "the AIR is checked against the object".
#
#   tbl - 104   the __metallib section's length
#   tbl + 152   that length minus four
#
# Both are tbl-relative and the table starts at container offset 880 in every host examined
# (tb-host3wide, tb-host3hi, a6-tgcalc), whose metallibs are 5600, 5408 and 4928 bytes.
MLIB_SIZE_SLOTS = (-104, 152)

# THE TABLE DUPLICATES THE SECTION LAYOUT, and a build that changes a payload size must update it.
#
# The load commands are NOT this table: offsets 424-816 are the __TEXT LC_SEGMENT_64 and its four
# section entries, which _rewrite_sections already owns. The AIR_DATA table proper starts at
# container offset 880, and differencing three hosts whose objects and metallibs differ in size
# (tb-host3wide, gen-clamp16, tb-host3hi) shows exactly which of its fields are layout:
#
#   1024   __metallib's container offset          1120   __descriptor's container offset
#   1032   __metallib's length minus four         1156, 1448   the object's length (OBJ_SIZE_SLOTS)
#
# The only other per-kernel fields in it are identity rather than layout - the metallib's UUID at
# 1008 and a 32-byte value at 1048 and 1196 - and a foreign metallib of the SAME size runs
# correctly without either being updated, so neither is checked against the payload.
LAYOUT_SLOTS = dict(mliboff=(1024,), mlibsize4=(1032,), descoff=(1120,))


def _rewrite_sections(container, desc, layout):
    """Rewrite the container's __TEXT section offsets and sizes, and its segment filesize.

    Needed as soon as any payload changes length - a different AIR, a bigger object - because the
    load commands carry every offset. Without this, build() can only reproduce the sizes it was
    described from.
    """
    nat_ncmds = struct.unpack_from("<I", container, 16)[0]
    p = 32
    for _ in range(nat_ncmds):
        cmd, cs = struct.unpack_from("<II", container, p)
        if cmd == LC_SEGMENT_64 and container[p + 8:p + 24].rstrip(b"\0") == b"__TEXT":
            nsects = struct.unpack_from("<I", container, p + 64)[0]
            first = min(off for _n, off, _sz in layout)
            total = sum(sz for _n, _off, sz in layout)
            # LC_SEGMENT_64 field offsets, from the structure rather than from a guess:
            # cmd 0, cmdsize 4, segname 8, vmaddr 24, vmsize 32, fileoff 40, filesize 48,
            # maxprot 56, initprot 60, nsects 64, flags 68.
            #
            # These were written sixteen bytes too far in, so a build that changed any payload size
            # put the segment's first offset over maxprot/initprot and its total over NSECTS - the
            # section count - and the loader then rejected the archive as "invalid format". Every
            # same-size build was unaffected, which is why it survived: the rewrite only runs when
            # a payload length changes, and until a foreign metallib was transplanted none did.
            struct.pack_into("<Q", container, p + 24, first)           # vmaddr
            struct.pack_into("<Q", container, p + 32, total)           # vmsize
            struct.pack_into("<Q", container, p + 40, first)           # fileoff
            struct.pack_into("<Q", container, p + 48, total)           # filesize
            q = p + 72
            for _ in range(nsects):
                nm = container[q:q + 16].rstrip(b"\0").decode()
                for n, off, sz in layout:
                    if n == nm:
                        struct.pack_into("<Q", container, q + 32, off)   # addr
                        struct.pack_into("<Q", container, q + 40, sz)    # size
                        struct.pack_into("<I", container, q + 48, off)   # offset
                q += 80
        p += cs
    return container


# The two slices' architecture words: an AIR slice and the native one. Constant in every archive
# in the cache, which is what makes an archive buildable with no host file to copy them from.
SLICE_ARCHS = ((16777239, 12, 4), (16777235, 0x163, 4))
# __reflection and __descriptor are both proven inert - zeroing either leaves the kernel correct -
# but they are still declared, so a size has to be chosen. These are the modal ones.
REFLECTION_LEN, DESCRIPTOR_LEN = 480, 400


def emit(obj, air, metallib, reflection=None, descriptor=None, obj_size=None, module_hash=None):
    """A complete archive from payloads alone - no host archive to copy a header from.

    This is the whole envelope: the fat header and its two architecture entries, the container's
    Mach-O header and load commands, the eight AIR_DATA tables, and the four __TEXT sections.
    """
    desc = {"magic": FAT_MAGIC,
            "slices": [(ct, cs, 0, 0, al) for ct, cs, al in SLICE_ARCHS]}
    parts = [("__reflection", bytes(REFLECTION_LEN) if reflection is None else bytes(reflection)),
             ("__compute", bytes(obj)),
             ("__descriptor", bytes(DESCRIPTOR_LEN) if descriptor is None else bytes(descriptor)),
             ("__metallib", bytes(metallib))]
    return _assemble(desc, bytes(air), parts, bytes(obj), obj_size, module_hash)


def _assemble(desc, air, parts, obj, obj_size, module_hash=None):
    """The container with a head this project writes, not one copied from a host archive.

    g17container computes every offset, size and table entry from the payload lengths; the only
    values that have to be handed to it are the two the metallib itself carries, so the container
    and the metallib agree by construction rather than by patching.
    """
    ml = dict(parts).get("__metallib", b"")
    h = g17mtlb.parse(ml)
    fl = dict(g17mtlb.tagged(ml, h["funclist_off"] + 8, h["funclist_off"] + h["funclist_size"]))
    ext = dict(g17mtlb.tagged(ml, h["funclist_off"] + h["funclist_size"] + 4, h["pubmd_off"]))
    if module_hash is None:
        module_hash = fl.get(b"HASH", b"\0" * 32)
    sizes = dict((n, len(b)) for n, b in parts)
    head, _place = g17container.build(
        sizes, uuid=hashlib.sha256(obj).digest()[:16],
        module_hash=module_hash, metallib_uuid=ext.get(b"UUID", b"\0" * 16),
        obj_size=len(obj) if obj_size is None else obj_size)
    container = bytearray(head)
    for _n, body in parts:
        container += body
    out = bytearray()
    out += struct.pack(">II", desc.get("magic", FAT_MAGIC), len(desc["slices"]))
    hdr = 8 + 20 * len(desc["slices"])
    a_off = (hdr + 15) & ~15
    n_off = (a_off + len(air) + 15) & ~15
    for i, (ct, cs, _o, _sz, al) in enumerate(desc["slices"]):
        off, size = (a_off, len(air)) if i == 0 else (n_off, len(container))
        out += struct.pack(">IIIII", ct, cs, off, size, al)
    out += bytes(a_off - len(out)); out += air
    out += bytes(n_off - len(out)); out += container
    return bytes(out)


def build(desc, fat, obj, blank_inert=True, air=None, metallib=None, obj_size=None,
          generate_head=True, module_hash=None):
    """Re-emit the whole archive with `obj` as __compute.

    Every offset and size in the fat header, the container header and the load commands is
    recomputed from the payload sizes rather than copied, so a different-sized object or AIR
    produces a correct archive rather than a corrupt one.
    """
    nat, sects = desc["nat"], desc["sects"]
    air_off, air_size = desc["air"]
    air = bytes(fat[air_off:air_off + air_size]) if air is None else bytes(air)
    # Container payload, in file order after the header and load commands.
    order = sorted((v[0], k) for k, v in sects.items())
    parts = []
    for off, name in order:
        o, size, _seg = sects[name]
        if name == "__compute":
            body = obj
        elif blank_inert and name in INERT_SECTIONS:
            body = bytes(size)
        elif name == "__metallib" and metallib is not None:
            body = bytes(metallib)
        else:
            body = bytes(fat[o:o + size])
        parts.append((name, body))
    if generate_head:
        return _assemble(desc, air, parts, obj, obj_size, module_hash)
    head_len = sects[order[0][1]][0] - nat            # header + load commands + __AIR_DATA tail
    container = bytearray(fat[nat:nat + head_len])
    layout = []
    for name, body in parts:
        layout.append((name, len(container), len(body)))
        container += body
    if any(len(b) != sects[n][1] for n, b in parts):
        _rewrite_sections(container, desc, layout)
    # The recorded value is the OBJECT's length, not the section's: Apple pads __compute up to an
    # eight-byte boundary and still records the unpadded size. Reproducing tb-host3hi and
    # tg-16x48x64 byte-identically is what showed the difference - both have an object eight bytes
    # shorter than their section.
    tbl = desc["air_data_tables"][0] - nat
    for slot in OBJ_SIZE_SLOTS:
        struct.pack_into("<I", container, tbl + slot, len(obj) if obj_size is None else obj_size)
    # The AIR_DATA table's layout fields, recomputed from where the payload actually landed.
    place = dict((n, (off, sz)) for n, off, sz in layout)
    vals = {}
    if "__descriptor" in place: vals["descoff"] = place["__descriptor"][0]
    if "__metallib" in place:
        vals["mliboff"] = place["__metallib"][0]
        vals["mlibsize4"] = place["__metallib"][1] - 4
    for name, slots in LAYOUT_SLOTS.items():
        if name not in vals: continue
        for sl in slots:
            struct.pack_into("<Q", container, sl, vals[name])
    # The fat header, computed.
    out = bytearray()
    out += struct.pack(">II", desc.get("magic", FAT_MAGIC), len(desc["slices"]))
    hdr = 8 + 20 * len(desc["slices"])
    a_off = (hdr + 15) & ~15
    n_off = (a_off + len(air) + 15) & ~15
    for i, (ct, cs, _o, _sz, al) in enumerate(desc["slices"]):
        off, size = (a_off, len(air)) if i == 0 else (n_off, len(container))
        out += struct.pack(">IIIII", ct, cs, off, size, al)
    out += bytes(a_off - len(out)); out += air
    out += bytes(n_off - len(out)); out += container
    return bytes(out)
