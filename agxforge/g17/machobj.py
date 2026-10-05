# Locate the applegpu Mach-O object and its sections inside a serialized
# MTLBinaryArchive. metal-lipo/metal-source show the archive is a fat file of
# {air64_v28, applegpu_g17s}; the native slice carries an embedded AIR metallib,
# a pipeline descriptor, and this object. Only the object holds __TEXT,__text.
import struct

LC_SEGMENT_64, LC_SYMTAB = 0x19, 0x02


def parse(obj):
    """-> (sections{name: (fileoff, size)}, syms{name: addr_in_section})."""
    ncmds = struct.unpack_from("<I", obj, 0x10)[0]
    off, sects, syms = 0x20, {}, {}
    for _ in range(ncmds):
        cmd, sz = struct.unpack_from("<II", obj, off)
        if cmd == LC_SEGMENT_64:
            n = struct.unpack_from("<I", obj, off + 64)[0]
            so = off + 72
            for _ in range(n):
                nm = obj[so:so + 16].rstrip(b"\0").decode()
                sg = obj[so + 16:so + 32].rstrip(b"\0").decode()
                addr, size, foff = struct.unpack_from("<QQI", obj, so + 32)
                sects[sg + "," + nm] = (foff, size)
                so += 80
        elif cmd == LC_SYMTAB:
            symoff, nsyms, stroff, _ = struct.unpack_from("<IIII", obj, off + 8)
            for i in range(nsyms):
                b = symoff + i * 16
                strx, _t, _s, _d, val = struct.unpack_from("<IBBHQ", obj, b)
                nm = obj[stroff + strx:obj.index(b"\0", stroff + strx)].decode()
                syms[nm] = val
        off += sz
    return sects, syms


def locate(archive_path, object_path):
    """Map the extracted object onto the archive; refuse if not exactly one copy."""
    fat = open(archive_path, "rb").read()
    obj = open(object_path, "rb").read()
    occ, i = [], fat.find(obj)
    while i >= 0:
        occ.append(i)
        i = fat.find(obj, i + 1)
    if len(occ) != 1:
        raise SystemExit("object appears %d times in %s" % (len(occ), archive_path))
    sects, syms = parse(obj)
    tf, ts = sects["__TEXT,__text"]
    return {"fat": fat, "obj": obj, "base": occ[0], "text_archive": occ[0] + tf,
            "text_size": ts, "sects": sects, "syms": syms,
            "text_obj": tf}
