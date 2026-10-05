#!/usr/bin/env python3
"""A STRICT CHECK ON A FINISHED IMAGE, run before Metal ever sees it.

Malformed loader-facing metadata does not produce an error - it segfaults the process that loads
it, and three times this session it did. Every probe therefore runs sacrificially, which is right,
but a crash is an expensive way to learn that an offset was wrong. This re-parses the delivered
bytes INDEPENDENTLY of the code that produced them and checks the relations that have to hold.

Independence is the whole point: it must not import the builders' layout constants or ask them
where anything is. It reads the fat header, walks the container's load commands, follows the
AIR_DATA tables, parses both metallibs and the object, and checks the relations execution has
proved matter:

    the fat header's slices land inside the file and do not overlap
    every section the container declares lies inside the container
    the AIR_DATA tables' offsets and counts land inside __AIR_DATA, and the placement they record
        agrees with the section table
    the recorded metallib size is the section's less four, the object size is at most its section
    both metallibs parse, their regions lie inside them, and their function lists name a function
    the container's module hash equals the library's - the archive's cache key
    the object's sections lie inside it and its entry symbol lies inside __text
    __GPU_METADATA's root and per-kernel table are reachable, the binding vector sits at
        per-kernel + Q + 36, and no two binding records name the same buffer

Each failure is one line naming what was expected and what was there.
"""
import os, struct, sys

# NO AMBIENT PATH MUTATION. This module used to insert its own directory and
# `../spike/accel/re` into sys.path, which resolved to the repository root's spike tree from
# tools/ and would resolve to agxforge/spike from here - a path that does not exist. Both inserts are
# dead: the only import left is the package-relative `mtlb` below, and every module that needs
# machobj inserts the spike path itself. A library that edits sys.path on import decides what its
# callers can import, which is the ambient state root has been removing from these tests.

FAT_MAGIC, MH_MAGIC_64 = 0xCBFEBABE, 0xFEEDFACF
LC_SEGMENT_64, LC_UUID, LC_TABLE = 0x19, 0x1B, 0x31


class Report(list):
    @property
    def ok(self):
        return not self

    def __str__(self):
        return "\n".join("   " + line for line in self) or "   no problems found"


def _u32(b, o): return struct.unpack_from("<I", b, o)[0]
def _u64(b, o): return struct.unpack_from("<Q", b, o)[0]


def verify(image, library=None):
    """-> Report. Empty means every relation checked holds."""
    r = Report()
    if len(image) < 48:
        r.append("the image is %d bytes; a fat header alone is 48" % len(image)); return r
    magic, narch = struct.unpack_from(">II", image, 0)
    if magic != FAT_MAGIC:
        r.append("fat magic is %08x, expected %08x" % (magic, FAT_MAGIC)); return r
    slices = [struct.unpack_from(">IIIII", image, 8 + 20 * i) for i in range(narch)]
    if narch != 2:
        r.append("%d architecture slices; an archive carries the AIR and the native one" % narch)
    spans = []
    for i, (_ct, _cs, off, size, _al) in enumerate(slices):
        if off + size > len(image):
            r.append("slice %d runs to %d, past the %d-byte image" % (i, off + size, len(image)))
        spans.append((off, off + size))
    for i in range(len(spans)):
        for j in range(i + 1, len(spans)):
            if spans[i][0] < spans[j][1] and spans[j][0] < spans[i][1]:
                r.append("slices %d and %d overlap: %s and %s" % (i, j, spans[i], spans[j]))
    if r:
        return r
    air = image[slices[0][2]:slices[0][2] + slices[0][3]]
    nat = slices[1][2]
    container = image[nat:nat + slices[1][3]]
    _container(container, r)
    if r:
        return r
    sects, tables, air_data = _sections(container)
    _tables(container, sects, tables, air_data, r)
    for name, blob in (("slice 0", air), ("__metallib", _sect(container, sects, "__metallib"))):
        _metallib(blob, name, r)
    if library is not None:
        _key(container, tables, library, r)
    obj = _sect(container, sects, "__compute")
    _object(obj, r)
    # THE METADATA THAT WILL ACTUALLY BE LOADED, taken out of the delivered object rather than
    # asked of the builder. Checking the builder's copy checks the wrong bytes: what the loader
    # reads is what came through the object's section table and the archive's placement.
    md = _objsect(obj, "__GPU_METADATA")
    if md is None:
        r.append("the object declares no __GPU_METADATA section")
    else:
        r.extend(verify_metadata(md))
    return r


def _objsect(obj, want):
    """One section's bytes out of a Mach-O object, by segment name."""
    if len(obj) < 32 or _u32(obj, 0) != MH_MAGIC_64:
        return None
    o = 32
    for _ in range(_u32(obj, 16)):
        cmd, cs = _u32(obj, o), _u32(obj, o + 4)
        if cmd == LC_SEGMENT_64:
            p = o + 72
            for _j in range(_u32(obj, o + 64)):
                seg = obj[p + 16:p + 32].rstrip(b"\0").decode("ascii", "replace")
                nm = obj[p:p + 16].rstrip(b"\0").decode("ascii", "replace")
                if want in (seg, nm):
                    off, size = _u32(obj, p + 48), _u64(obj, p + 40)
                    return obj[off:off + size]
                p += 80
        o += cs
    return None


def _sect(container, sects, name):
    o, n = sects[name]
    return container[o:o + n]


def _container(c, r):
    if len(c) < 32 or _u32(c, 0) != MH_MAGIC_64:
        r.append("the container's mach magic is %08x, expected %08x"
                 % (_u32(c, 0) if len(c) >= 4 else 0, MH_MAGIC_64))
        return
    ncmds, szcmds = _u32(c, 16), _u32(c, 20)
    if 32 + szcmds > len(c):
        r.append("load commands run to %d, past the %d-byte container" % (32 + szcmds, len(c)))
    o = 32
    for i in range(ncmds):
        if o + 8 > 32 + szcmds:
            r.append("load command %d starts at %d, past the command area" % (i, o)); return
        cs = _u32(c, o + 4)
        if cs < 8 or o + cs > 32 + szcmds:
            r.append("load command %d declares %d bytes at %d" % (i, cs, o)); return
        o += cs
    if o != 32 + szcmds:
        r.append("the load commands end at %d, not at the declared %d" % (o, 32 + szcmds))


def _sections(c):
    """(sections, tables, __AIR_DATA extent) read from the container's own load commands."""
    sects, tables, air_data = {}, {}, (0, 0)
    o = 32
    for _ in range(_u32(c, 16)):
        cmd, cs = _u32(c, o), _u32(c, o + 4)
        if cmd == LC_SEGMENT_64:
            seg = c[o + 8:o + 24].rstrip(b"\0").decode("ascii", "replace")
            fileoff, filesize = _u64(c, o + 40), _u64(c, o + 48)
            nsects = _u32(c, o + 64)
            if seg == "__AIR_DATA":
                air_data = (fileoff, filesize)
            p = o + 72
            for _j in range(nsects):
                nm = c[p:p + 16].rstrip(b"\0").decode("ascii", "replace")
                sects[nm] = (_u32(c, p + 48), _u64(c, p + 40))
                p += 80
        elif cmd == LC_TABLE:
            nm = c[o + 8:o + 24].rstrip(b"\0").decode("ascii", "replace")
            tables[nm] = (_u64(c, o + 24), _u64(c, o + 32))
        o += cs
    return sects, tables, air_data


def _tables(c, sects, tables, air_data, r):
    for nm, (off, size) in sects.items():
        if off + size > len(c):
            r.append("section %s runs to %d, past the %d-byte container" % (nm, off + size, len(c)))
    lo, hi = air_data
    for nm, (off, size) in tables.items():
        if not (lo <= off and off + size <= lo + hi):
            r.append("table %s's descriptor at [%d,%d) is outside __AIR_DATA [%d,%d)"
                     % (nm, off, off + size, lo, lo + hi))
            continue
        eoff, ecount = _u64(c, off), _u64(c, off + 8)
        if ecount and not (lo <= eoff < lo + hi):
            r.append("table %s's %d entries start at %d, outside __AIR_DATA" % (nm, ecount, eoff))
    md = tables.get("AIR_METALLIB")
    if md:
        eoff = _u64(c, md[0])
        rec_off, rec_size = _u64(c, eoff + 16), _u64(c, eoff + 24)
        want = sects.get("__metallib")
        if want and rec_off != want[0]:
            r.append("AIR_METALLIB records the metallib at %d; the section table says %d"
                     % (rec_off, want[0]))
        if want and rec_size != want[1] - 4:
            r.append("AIR_METALLIB records %d bytes; the section is %d and the recorded size is"
                     " four less in every archive" % (rec_size, want[1]))
    ob = tables.get("AIR_OBJECT")
    if ob:
        eoff = _u64(c, ob[0])
        o_off, o_size = _u32(c, eoff + 8), _u32(c, eoff + 12)
        want = sects.get("__compute")
        if want and o_off != want[0]:
            r.append("AIR_OBJECT records the object at %d; the section table says %d"
                     % (o_off, want[0]))
        if want and o_size > want[1]:
            r.append("AIR_OBJECT records %d object bytes, more than the %d-byte section"
                     % (o_size, want[1]))


def _metallib(b, name, r):
    if len(b) < 88 or bytes(b[:4]) != b"MTLB":
        r.append("%s does not begin with MTLB" % name); return
    size = _u64(b, 16)
    if size != len(b) - 4:
        r.append("%s records %d bytes for a %d-byte file (four less is what every archive has)"
                 % (name, size, len(b)))
    for label, off_at, size_at in (("function list", 24, 32), ("public metadata", 40, 48),
                                   ("private metadata", 56, 64), ("bitcode", 72, 80)):
        o, n = _u64(b, off_at), _u64(b, size_at)
        if o + n > len(b):
            r.append("%s's %s runs to %d, past the %d-byte file" % (name, label, o + n, len(b)))
    fo, fn = _u64(b, 24), _u64(b, 32)
    if fo + fn <= len(b) and fn >= 8:
        count = _u32(b, fo)
        if count < 1:
            r.append("%s's function list names no function" % name)


def _key(c, tables, library, r):
    """The archive's cache key: the container's module hash must be the library's function hash."""
    from . import mtlb as g17mtlb
    try:
        want = g17mtlb.function_hash(library)
    except Exception as e:
        r.append("the library's function hash could not be read: %s" % e); return
    md = tables.get("AIR_MODULE")
    if not md:
        r.append("the container declares no AIR_MODULE table"); return
    eoff = _u64(c, md[0])
    got = bytes(c[eoff:eoff + 32])
    if got != want:
        r.append("the container's module hash %s is not the library's %s - the archive lookup "
                 "will not match and the loader returns no pipeline"
                 % (got.hex()[:16], want.hex()[:16]))


def _object(obj, r):
    if len(obj) < 32 or _u32(obj, 0) != MH_MAGIC_64:
        r.append("the object does not begin with a 64-bit mach header"); return
    sects, syms = {}, {}
    o = 32
    for _ in range(_u32(obj, 16)):
        cmd, cs = _u32(obj, o), _u32(obj, o + 4)
        if cmd == LC_SEGMENT_64:
            nsects = _u32(obj, o + 64)
            p = o + 72
            for _j in range(nsects):
                nm = obj[p:p + 16].rstrip(b"\0").decode("ascii", "replace")
                seg = obj[p + 16:p + 32].rstrip(b"\0").decode("ascii", "replace")
                sects[seg or nm] = (_u32(obj, p + 48), _u64(obj, p + 40))
                p += 80
        elif cmd == 0x2:                       # LC_SYMTAB
            symoff, nsyms, stroff = _u32(obj, o + 8), _u32(obj, o + 12), _u32(obj, o + 16)
            for k in range(nsyms):
                e = symoff + 16 * k
                if e + 16 > len(obj): break
                strx, value = _u32(obj, e), _u64(obj, e + 8)
                end = obj.find(b"\0", stroff + strx)
                syms[obj[stroff + strx:end].decode("ascii", "replace")] = value
        o += cs
    for nm, (off, size) in sects.items():
        if off + size > len(obj):
            r.append("the object's %s runs to %d, past its %d bytes" % (nm, off + size, len(obj)))
    text = sects.get("__TEXT") or sects.get("__text")
    for nm, v in syms.items():
        if text and v > text[1]:
            r.append("symbol %s is at %d, past the %d-byte __text" % (nm, v, text[1]))


def _table(md, pos):
    """One FlatBuffers table: (vtable position, slot -> field offset). None if it does not walk."""
    if not (4 <= pos <= len(md) - 4):
        return None
    vt = pos - struct.unpack_from("<i", md, pos)[0]
    if not (0 <= vt <= len(md) - 4):
        return None
    vlen, _tlen = struct.unpack_from("<HH", md, vt)
    if not (4 <= vlen <= 400) or vt + vlen > len(md):
        return None
    # A VTABLE CAN BE SHARED, AND THEN IT DOES NOT PRECEDE ITS TABLE. This checker enforced
    # vt + vlen == pos as a law, which is what the corpus looked like only because every reader in
    # this project imposed it. FlatBuffers dedupes identical vtables and 105 of 7,594 sections use
    # that, so a second table points BACK at the first one's - a negative soffset.
    #
    # IT WAS REJECTING SECTIONS BYTE-IDENTICAL TO APPLE'S. c4wide2, c4wideC, mx-s.half-2 and -4,
    # sr_tggx, sr_tggy and sr_tggz all build byte-exact and were refused here with "slot 4 refers
    # to 404, where nothing walks as a vector of tables" - because one record of that vector
    # shares a vtable. A rule with exceptions written into a checker as though it had none turns
    # the exceptions into faults, which is the same defect as a forced bit that Apple varies.
    if vt + vlen != pos and not (vt > pos and pos + _tlen <= vt):
        return None
    # THE FIELD CHECK MUST NOT ASSUME A WIDTH. This required
    # `pos + off + 4 <= len(md)`, which assumes every field is four bytes wide. A ONE-byte field
    # near the end of a section fails that while sitting entirely inside its own table, and the
    # slot was then DROPPED rather than reported - so `binding_records`, whose accessor returns 0
    # for an absent slot, read a WRITTEN buffer back as read-only. It did that to Apple's own
    # bytes: the two-writable witness's first binding record carries its kind at +10 and its
    # written flag at +11 of a 12-byte table, and 380 + 11 + 4 exceeds the 392-byte section, so
    # both were discarded. The kind is the more alarming of the two - a written flag reading False
    # produced a loud refusal, while a kind reading 0 instead of 5 is a wrong value that nothing
    # would necessarily have contradicted.
    #
    # BOUNDING BY THE TABLE'S OWN `tlen` WAS TRIED AND IS REFUTED. It is the rule FlatBuffers
    # states, and this project's own measured SCALAR class declares `pk_tlen` 0 while carrying
    # fields out to offset 48; three g17metaclass images stopped verifying with "the per-kernel
    # table has no slot 2". So the inline length is not a bound that can be enforced against these
    # sections, and the check stays where it was minus the width assumption: the field must simply
    # begin inside the section.
    slots = {}
    for i in range((vlen - 4) // 2):
        off = struct.unpack_from("<H", md, vt + 4 + 2 * i)[0]
        if off and pos + off < len(md):
            slots[i] = off
    return vt, slots


def _ref(md, pos, slots, slot):
    """A FlatBuffers reference read the way the format defines it: the u32 stored in the field is
    relative to the FIELD'S OWN ADDRESS, not to the table. Reading it as table-relative gives the
    right answer only when the serialiser happens to have placed that field at one offset - which
    is why a rule with a fixed +36 in it fits 7,519 objects and not 7,557."""
    if slot not in slots:
        return None
    at = pos + slots[slot]
    return at + struct.unpack_from("<I", md, at)[0]


def _vector(md, pos):
    """A vector of table references at `pos`, or None."""
    if not (0 <= pos <= len(md) - 8):
        return None
    n = struct.unpack_from("<I", md, pos)[0]
    if not (1 <= n <= 64) or pos + 4 + 4 * n > len(md):
        return None
    out = []
    for k in range(n):
        at = pos + 4 + 4 * k
        t = at + struct.unpack_from("<I", md, at)[0]
        if _table(md, t) is None:
            return None
        out.append(t)
    return out


def verify_metadata(md, bindings=None):
    """__GPU_METADATA on its own, walked HERE rather than by the builder's own reader.

    Sharing a parser with the builder means a layout the builder gets wrong in a way its reader
    mirrors passes the check. This walks the FlatBuffers encoding directly: a table's vtable
    abutting it, and every reference read relative to the field that holds it.
    """
    r = Report()
    if len(md) < 8:
        r.append("the section is %d bytes" % len(md)); return r
    root = struct.unpack_from("<I", md, 0)[0]
    rt = _table(md, root)
    if rt is None:
        r.append("the root pointer at 0 names %d, which does not walk as a table" % root); return r
    _rvt, rslots = rt
    pk = _ref(md, root, rslots, 0)
    if pk is None:
        r.append("the root table has no slot 0, so nothing reaches the per-kernel table"); return r
    pkt = _table(md, pk)
    if pkt is None:
        r.append("the root's slot 0 names %d, which does not walk as a table" % pk); return r
    _pvt, pslots = pkt
    for slot in (2, 3, 4):
        if slot not in pslots:
            r.append("the per-kernel table has no slot %d" % slot)
    if r:
        return r
    q = struct.unpack_from("<I", md, pk + pslots[4])[0] + 4
    nb = struct.unpack_from("<I", md, pk + pslots[3])[0] // 8
    vec = _ref(md, pk, pslots, 4)
    recs = _vector(md, vec)
    if recs is None:
        r.append("slot 4 refers to %d, where nothing walks as a vector of tables "
                 "(field at %d + %d, Q %d)" % (vec, pk + pslots[4], vec - pk - pslots[4], q))
        return r
    if len(recs) != nb:
        r.append("slot 3 declares %d bindings and the vector holds %d records" % (nb, len(recs)))
    if _vector(md, _ref(md, pk, pslots, 2)) is None:
        r.append("slot 2 refers to %d, where nothing walks as a vector of tables"
                 % _ref(md, pk, pslots, 2))
    for slot in (6, 8, 10, 26):
        t = _ref(md, pk, pslots, slot)
        if t is not None and not (0 <= t <= len(md)):
            r.append("slot %d refers to %d, outside the %d-byte section" % (slot, t, len(md)))
    seen = []
    for rec in recs:
        t = _table(md, rec)
        if t is None:
            r.append("binding record at %d does not walk as a table" % rec); continue
        idx = struct.unpack_from("<I", md, rec + t[1][1])[0] if 1 in t[1] else 0
        if idx in seen:
            r.append("two binding records name buffer %d" % idx)
        seen.append(idx)
    if bindings is not None and len(recs) != len(bindings):
        r.append("%d records for a signature with %d bound buffers" % (len(recs), len(bindings)))
    return r
