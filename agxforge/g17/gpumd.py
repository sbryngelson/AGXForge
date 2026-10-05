#!/usr/bin/env python3
"""Read __GPU_METADATA as the FLATBUFFER it is, not at a fixed byte offset.

The register-count declaration - the number an image must state before the driver will launch it
with its own allocation instead of a container's - lives in __GPU_METADATA,__compute of the
native object. It was first spotted at byte offset 212, which works on one family of objects and
fails on others, and the reason is that __GPU_METADATA is a FLATBUFFER: a field's byte offset is
whatever that object's vtable puts it at, so a fixed offset is reading a different field the
moment the table shape changes. Four of the sixteen tensor references read 0x01010101 at 212.

Addressed by PATH instead - root table slot 0, then sub-table slot 0 - it reads correctly on all
sixteen, saturated shapes included, and on 1320 of 1376 corpus objects.

    python3 tools/g17gpumd.py <object>        one object
    python3 tools/g17gpumd.py --check         the whole corpus against the decoded maximum
"""
import collections, glob, os, re, struct, sys

# NO AMBIENT PATH MUTATION: the siblings this module reads are package modules now, and a library
# that edits sys.path on import decides what its callers can import.
from . import bind as g17bind

# R_n names the 32-bit register; R_nL and R_nH are its halves, and a tuple names its members. The
# index is what the metadata counts, so a tuple based at R8 spanning R8..R15 reaches 15.
NAME = re.compile(r"R(\d+)([LH]?)$")


def _table_slot(d, tpos, slot):
    """Absolute position of one slot of the flatbuffer table at tpos, or None if absent."""
    vt = tpos - struct.unpack_from("<i", d, tpos)[0]
    if not (0 <= vt <= len(d) - 4):
        return None
    vsz = struct.unpack_from("<H", d, vt)[0]
    if 4 + 2 * slot >= vsz:
        return None
    off = struct.unpack_from("<H", d, vt + 4 + 2 * slot)[0]
    return tpos + off if off else None


def register_count(data):
    """The register-count declaration, or None if this object's table does not carry it."""
    try:
        root = struct.unpack_from("<I", data, 0)[0]
        a = _table_slot(data, root, 0)
        if a is None:
            return None
        sub = a + struct.unpack_from("<I", data, a)[0]
        b = _table_slot(data, sub, 0)
        return None if b is None else struct.unpack_from("<I", data, b)[0]
    except Exception:
        return None


def threadgroup_declaration(data):
    """Read independent memory-use and static-byte fields through the vtable.

    Missing slots remain None: in particular, a missing static declaration is
    not a claim of zero bytes. Slot 18 and slot 28 are independent measurements.
    """
    root=struct.unpack_from('<I',data)[0]
    pointer=_table_slot(data,root,0)
    if pointer is None:raise ValueError('metadata has no per-kernel table')
    pk=pointer+struct.unpack_from('<I',data,pointer)[0]
    use=_table_slot(data,pk,18);size=_table_slot(data,pk,28)
    return dict(memory_use=None if use is None else struct.unpack_from('<B',data,use)[0],
                static_memory_bytes=None if size is None else struct.unpack_from('<I',data,size)[0])


def declared(path):
    d = g17bind.section(path, "__GPU_METADATA", "__compute")
    return None if d is None else register_count(d)


def highest_register(stream, registers):
    """Highest 32-bit register index any operand names, over a decoded instruction stream."""
    hi = -1
    for _, _, _, _, ops in stream:
        for kind, v in ops:
            if kind != "reg":
                continue
            for part in registers.get(v, "").split("_"):
                m = NAME.match(part)
                if m:
                    hi = max(hi, int(m.group(1)))
    return hi


def table_at(data, pos):
    """(slot offsets, table size) for the flatbuffer table at pos, or None."""
    vt = pos - struct.unpack_from("<i", data, pos)[0]
    if not (0 <= vt <= len(data) - 4):
        return None
    vsz, tsz = struct.unpack_from("<HH", data, vt)
    if not (4 <= vsz <= 400) or vt + vsz > len(data):
        return None
    return [struct.unpack_from("<H", data, vt + 4 + 2 * i)[0]
            for i in range((vsz - 4) // 2)], tsz


def kernel_table(data):
    """Position of root field 0, the per-kernel table."""
    root = struct.unpack_from("<I", data, 0)[0]
    t = table_at(data, root)
    if not t or not t[0][0]:
        return None
    a = root + t[0][0]
    return a + struct.unpack_from("<I", data, a)[0]


def fields(data):
    """{slot: value} for the per-kernel table, each read at its own WIDTH.

    Flatbuffer fields are not all 32 bits and reading them as if they were makes one field look
    like several: slot 15 read as u32 returns the register count shifted by 8 because it is a u8
    with the next field behind it. The width is the gap to the next occupied offset, and the last
    field runs to the end of the table.
    """
    t = kernel_table(data)
    if t is None:
        return {}
    slots, tsz = table_at(data, t)
    occupied = sorted({s for s in slots if s})
    if not occupied:
        return {}
    width = {occupied[i]: occupied[i + 1] - occupied[i] for i in range(len(occupied) - 1)}
    width[occupied[-1]] = tsz - occupied[-1]
    out = {}
    for i, s in enumerate(slots):
        if not s or t + s + 4 > len(data):
            continue
        w = width.get(s, 4)
        out[i] = (data[t + s] if w == 1 else
                  struct.unpack_from("<H", data, t + s)[0] if w == 2 else
                  struct.unpack_from("<I", data, t + s)[0])
    return out


# What the per-kernel table's slots hold.
#
# SEVEN OF THESE SLOTS ARE NOT VALUES. THEY ARE FLATBUFFER VECTOR OFFSETS.
# Slots 2, 4, 6, 8, 10, 12 and 26 read as length-prefixed vectors of valid table offsets in
# 16,649 of 16,649 corpus sections and 9,556 of 9,556 Apple sections. Four of them - 6, 8, 12 and
# usually 10 - point at vectors that are EMPTY in nearly every kernel, and empty vectors laid out
# consecutively sit at nearly the same address. That is the whole reason the earlier arithmetic
# closed:
#
#     "slots 6, 8, 10, 12 are EQUAL in every object; call the common value Q"   <- they are
#     "slot 2 = Q - 4 + 4B"    <- the binding vector, B records of 4 bytes, lies between them
#     "slot 4 = Q - 4"         <- likewise, one length word away
#
# Q was never a quantity. Compile a kernel that touches two threadgroup arrays and slot 10 parts
# from the others by exactly the bytes those records occupy: probe-s9-tg1 has 6=140 8=140 10=136
# 12=140, and probe-s9-tg4 has 6=160 8=160 10=144 12=148.
#
# So a generative builder does not predict slots 2, 4, 6, 8, 10, 12 or 26 at all. It lays out the
# vectors and the offsets follow.
KERNEL_FIELDS = {
    0:  "register count, equals the highest register used plus one in 1320 of 1376 - DECLARED but "
        "measured inert, so it is the right value to emit and not the reason anything works",
    1:  "FREE: multiples of 4, correlated with the binding count but not equal to any multiple of "
        "it; unnamed",
    2:  "OFFSET to a vector of records with kinds (3, 6) or (3, 5, 6); never empty",
    3:  "= 8 * the number of buffer bindings; 7,916 of 9,547 Apple sections, and every miss is a "
        "binding-reader miss rather than a failure of the law",
    4:  "OFFSET to the BINDING vector - records of kind 5, one per bound buffer",
    6:  "OFFSET to a vector that is empty in all 26,205 sections measured",
    8:  "OFFSET to a vector that is empty in all 26,205 sections measured",
    9:  "= 4 * (1 + max field-2 over the records slot 10 points at); ABSENT when that vector is "
        "empty. 16,649 of 16,649 corpus and 9,556 of 9,556 Apple, no exceptions. DERIVED",
    10: "OFFSET to the THREADGROUP vector: one record per threadgroup array the optimized program "
        "actually touches. Record kinds 43 (field 2 steps by 2) and 93 (steps by 1)",
    12: "OFFSET to a vector that is empty in all 26,205 sections measured",
    26: "OFFSET to a vector carrying exactly one record in every section measured",
    41: "= slot 14, exactly: 114 of 114 corpus and 900 of 900 Apple sections, and the two are "
        "never present apart. DERIVED",
    42: "= slot 38, exactly: 302 of 302 corpus and 6,495 of 6,495 Apple sections, and the two are "
        "never present apart. DERIVED",
    15: "constant 1", 16: "constant 1", 33: "constant 1", 44: "constant 1",
    32: "1 or 3, nothing else",
}


def main(argv=None, streams=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if "--schema" in argv:
        for path in [a for a in argv if not a.startswith("--")]:
            data = g17bind.section(path, "__GPU_METADATA", "__compute")
            if data is None:
                print("%s: no __GPU_METADATA" % path)
                continue
            root = struct.unpack_from("<I", data, 0)[0]
            slots, tsz = table_at(data, root)
            print("%s\n   %d bytes, root table %d slots, table size %d"
                  % (path, len(data), len(slots), tsz))
            for slot, value in sorted(fields(data).items()):
                print("   slot %-3d = %-12d %s" % (slot, value, KERNEL_FIELDS.get(slot, "")))
        return

    if "--check" not in argv:
        for p in argv:
            print("%-60s %s" % (p, declared(p)))
        return
    from . import model as g17model
    if streams is None:
        raise ValueError(
            "--check needs the corpus stream, which this package does not import: call "
            "main(streams=...) or run tools/g17gpumd.py, whose legacy entry supplies it from "
            "g17context. The diagnostic lives outside the package on purpose; an authoring "
            "dependency must not be hidden here.")
    registers = g17model.registers()
    hi = collections.defaultdict(lambda: -1)
    for name, _, st in streams:
        hi[name] = max(hi[name], highest_register(st, registers))
    ok, bad, none = 0, [], 0
    for d in sorted(glob.glob(os.path.expanduser("~/.cache/agxforge/agx/*"))):
        name = os.path.basename(d)
        obj = d + "/out/object/0-0"
        if not os.path.exists(obj) or name not in hi:
            continue
        v = declared(obj)
        if v is None:
            none += 1
        elif v == hi[name] + 1:
            ok += 1
        else:
            bad.append((name, v, hi[name] + 1))
    print("declared == highest register + 1:  %d of %d   (%d unreadable)"
          % (ok, ok + len(bad), none))
    if bad:
        pref = collections.Counter(re.match(r"[a-z0-9]+", n).group(0) for n, _, _ in bad)
        print("mismatches by object-name prefix: %s" % dict(pref.most_common()))
        for n, v, w in bad[:8]:
            print("   %-22s declared %-6d highest+1 %d" % (n, v, w))


if __name__ == "__main__":
    main()
