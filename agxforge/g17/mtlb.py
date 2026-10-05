#!/usr/bin/env python3
"""The MTLB (metallib) container, parsed well enough to shrink it.

The archive carries two copies of Apple's AIR and neither supplies anything this compiler needs:
an unrelated kernel's AIR, of a different size and a different signature, runs our authored object
correctly (ledger/g17-the-air-is-a-credential.toml). What remains is to stop borrowing it, and the
first question is which PARTS of it the loader reads.

The header is a table of (offset, size) pairs, all little-endian u64 after an eight-byte magic and
version:

    +16  the file size, or that minus four        +24/+32  function list
    +40/+48  public metadata                      +56/+64  private metadata
    +72/+80  the BITCODE - three quarters of the file in the smallest one measured
"""
import struct

MAGIC = b"MTLB"
FIELDS = ("size", "funclist_off", "funclist_size", "pubmd_off", "pubmd_size",
          "privmd_off", "privmd_size", "bitcode_off", "bitcode_size")


def parse(b):
    if bytes(b[:4]) != MAGIC:
        raise ValueError("not an MTLB: %s" % bytes(b[:4]))
    vals = struct.unpack_from("<9Q", b, 16)
    out = dict(zip(FIELDS, vals))
    out["version"] = bytes(b[4:16])
    out["len"] = len(b)
    return out


def blank_bitcode(b, fill=0):
    """The same metallib with its bitcode region overwritten - same length, same header.

    Keeping the length is the point: the container records it in two places and the loader rejects
    a mismatch, so a shorter file would test the size bookkeeping rather than the bitcode.
    """
    h = parse(b)
    o, n = h["bitcode_off"], h["bitcode_size"]
    if o + n > len(b):
        raise ValueError("bitcode region [%d,%d) is outside a %d-byte file" % (o, o + n, len(b)))
    u = bytearray(b)
    u[o:o + n] = bytes([fill]) * n
    return bytes(u)


def region(b, name):
    h = parse(b)
    o, n = h[name + "_off"], h[name + "_size"]
    return bytes(b[o:o + n])


# The rest of the container, past the four (offset,size) pairs the header names. Everything after
# the function list is a stream of tagged records - four ASCII bytes, a u16 length, the value - and
# `ENDT` closes a list. That is the same encoding the function list uses internally, so one reader
# serves both.
def tagged(b, off, end):
    o, out = off, []
    while o + 6 <= end:
        t = bytes(b[o:o + 4])
        if t == b"ENDT":
            out.append((t, b"")); break
        n = struct.unpack_from("<H", b, o + 4)[0]
        out.append((t, bytes(b[o + 6:o + 6 + n]))); o += 6 + n
    return out


def sections(b):
    """Named (offset, size) regions, header pairs and extension tags together.

    HDYN/RLST/SLST are the three regions the header does NOT name: a per-function dynamic header,
    the reflection list (an `RBUF` flatbuffer carrying the argument names and types), and the source
    list (an `MTLP` flatbuffer carrying `alias:<hash>#<name>` and the lib-from-data UUID string).
    """
    h = parse(b)
    out = {"funclist": (h["funclist_off"], h["funclist_size"]),
           "pubmd": (h["pubmd_off"], h["pubmd_size"]),
           "privmd": (h["privmd_off"], h["privmd_size"]),
           "bitcode": (h["bitcode_off"], h["bitcode_size"])}
    # The function list is closed by its own ENDT, which sits just past the size the header gives;
    # the extension tags start after it.
    ext_off = h["funclist_off"] + h["funclist_size"] + 4
    for t, v in tagged(b, ext_off, h["pubmd_off"]):
        if t == b"ENDT":
            continue
        if len(v) == 16 and t in (b"HDYN", b"RLST", b"SLST"):
            o, n = struct.unpack("<QQ", v)
            out[t.decode().lower()] = (o, n if t != b"HDYN" else 11)
        elif t == b"UUID":
            out["uuid"] = (ext_off + _tagpos(b, ext_off, h["pubmd_off"], b"UUID") + 6, len(v))
    return out


def _tagpos(b, off, end, want):
    o = off
    while o + 6 <= end:
        t = bytes(b[o:o + 4])
        if t == b"ENDT":
            break
        n = struct.unpack_from("<H", b, o + 4)[0]
        if t == want:
            return o - off
        o += 6 + n
    raise KeyError(want)


def funcs(b):
    """[(name, {tag: value})] for the function list."""
    h = parse(b)
    o = h["funclist_off"]
    count = struct.unpack_from("<I", b, o)[0]
    end = o + h["funclist_size"]
    o += 8
    out = []
    for _ in range(count):
        d = dict((t.decode(), v) for t, v in tagged(b, o, end))
        out.append((d.get("NAME", b"").rstrip(b"\0").decode("ascii", "replace"), d))
        o = end  # one function in every archive measured; multi-function needs the per-entry size
    return out


def set_tag(b, region_off, region_end, tag, value):
    """Replace one tagged record's value in place. The length must not change - every offset in
    the header is absolute, so a resized record would move six other regions."""
    o = region_off
    while o + 6 <= region_end:
        t = bytes(b[o:o + 4])
        if t == b"ENDT":
            break
        n = struct.unpack_from("<H", b, o + 4)[0]
        if t == tag:
            if len(value) != n:
                raise ValueError("%s is %d bytes, not %d" % (tag, n, len(value)))
            u = bytearray(b); u[o + 6:o + 6 + n] = value; return bytes(u)
        o += 6 + n
    raise KeyError(tag)


def _tag(t, v):
    return t + struct.pack("<H", len(v)) + v


ENDT = b"ENDT"
# The version words Apple emits for air64_v28 on this OS. Nothing in the image depends on them yet;
# they are copied because a version field is exactly the kind of thing a loader range-checks.
VERSION = bytes.fromhex("01800200090000811a000000")
FUNC_VERS = bytes.fromhex("0200080004000000")


def build(name="k", bitcode=b"", uuid=b"\0" * 16, ftype=2, hash=b"\0" * 32, rflt=4):
    """A metallib assembled from the format, not copied from Apple.

    Every region Apple fills was zeroed one at a time and the kernel still ran - the bitcode, the
    function hash, the UUID, the reflection list, the source list, even the function's NAME. So
    this writes the smallest thing the format describes and lets execution say what the loader
    actually reads. MDSZ carries the function's bitcode size in every archive measured, so it is
    computed here rather than copied.
    """
    nm = name.encode() + b"\0"
    fl = (_tag(b"NAME", nm) + _tag(b"TYPE", bytes([ftype])) + _tag(b"HASH", hash)
          + _tag(b"OFFT", b"\0" * 24) + _tag(b"VERS", FUNC_VERS)
          + _tag(b"MDSZ", struct.pack("<Q", len(bitcode)))
          + _tag(b"RFLT", struct.pack("<Q", rflt)))
    funclist = struct.pack("<II", 1, 8 + len(fl)) + fl
    hdyn = _tag(b"NAME", b"\0") + ENDT
    rlst = struct.pack("<I", 0)      # an empty reflection list; the populated one zeroes fine
    slst = struct.pack("<I", 0)
    md = struct.pack("<I", 8) + ENDT  # public and private metadata: a size and an end marker

    # Two passes: the extension tags carry absolute offsets to regions that follow them, so the
    # layout has to be solved before it can be written. The extension block's size is fixed.
    ext_len = 4 + 4 * (6 + 16) + 4            # the function list's ENDT, four tags, the block's ENDT
    fl_off = 88
    pub_off = fl_off + len(funclist) + ext_len
    priv_off = pub_off + len(md)
    bc_off = priv_off + len(md)
    hdyn_off = bc_off + len(bitcode)
    rlst_off = hdyn_off + len(hdyn)
    slst_off = rlst_off + len(rlst)
    total = slst_off + len(slst) + 8          # the closing ENDT and its trailing word

    ext = (ENDT
           + _tag(b"HDYN", struct.pack("<QQ", hdyn_off, 0xB00))
           + _tag(b"RLST", struct.pack("<QQ", rlst_off, len(rlst)))
           + _tag(b"SLST", struct.pack("<QQ", slst_off, len(slst)))
           + _tag(b"UUID", uuid) + ENDT)
    assert len(ext) == ext_len, (len(ext), ext_len)
    head = MAGIC + VERSION + struct.pack("<9Q", total - 4, fl_off, len(funclist), pub_off, len(md),
                                         priv_off, len(md), bc_off, len(bitcode))
    assert len(head) == 88, len(head)
    out = head + funclist + ext + md + md + bitcode + hdyn + rlst + slst + ENDT + b"\0" * 4
    assert len(out) == total, (len(out), total)
    return out


def function_hash(path, name=None):
    """The 32-byte HASH the given metallib records for a function.

    This is the archive's cache key: the container's AIR_MODULE table and its AIR_HASHES entry both
    carry the SOURCE LIBRARY's function hash, and a pipeline built from that library finds its
    compiled form by matching it. It is an identity of the input, not anything the compiler emits -
    but it has to be carried, and with it zeroed the loader returns no pipeline
    (spike/accel/re/headbisect.py).
    """
    b = open(path, "rb").read() if isinstance(path, str) else bytes(path)
    h = parse(b)
    for nm, tags in funcs(b):
        if name is None or nm == name:
            return tags["HASH"]
    raise KeyError(name)
