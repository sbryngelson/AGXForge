#!/usr/bin/env python3
"""Decode a native G17 object's BINDING TABLE.

The load instruction's base field is a COMPACTED index over the buffers a kernel uses
(ledger/g17-no-descriptor-instructions.toml). This resolves that index to a Metal buffer number
by reading the flatbuffer vector in __GPU_METADATA: vector position k IS base index k, and each
record's first field is the Metal buffer index, elided by flatbuffer default when it is 0.

    python3 tools/g17bind.py <native object>
"""
import sys, struct

def section(path, seg, sect):
    b = open(path, "rb").read()
    assert struct.unpack_from("<I", b, 0)[0] == 0xfeedfacf, "not a 64-bit Mach-O"
    ncmds = struct.unpack_from("<I", b, 16)[0]; off = 32
    for _ in range(ncmds):
        cmd, size = struct.unpack_from("<II", b, off)
        if cmd == 0x19:                                   # LC_SEGMENT_64
            nsects = struct.unpack_from("<I", b, off+64)[0]; so = off+72
            for _ in range(nsects):
                sn  = b[so:so+16].rstrip(b"\0").decode()
                sgn = b[so+16:so+32].rstrip(b"\0").decode()
                _, sz, foff = struct.unpack_from("<QQI", b, so+32)
                if sgn == seg and sn == sect: return b[foff:foff+sz]
                so += 80
        off += size
    return None

def bindings(path):
    """-> list of Metal buffer indices, position = the load's base index."""
    d = section(path, "__GPU_METADATA", "__compute")
    if d is None: return None
    best = None                                  # (count, position, indices)
    for p in range(0, len(d)-8, 4):
        n = struct.unpack_from("<I", d, p)[0]
        if not (1 <= n <= 31) or p+4+4*n > len(d): continue
        t = []
        for k in range(n):
            o = struct.unpack_from("<I", d, p+4+4*k)[0]
            q = p+4+4*k+o
            if o == 0 or not (p < q < len(d)): t = None; break
            t.append(q)
        if not t or len(set(t)) != n: continue
        if t != sorted(t, reverse=True): continue      # flatbuffers write the table backwards
        idx = []
        for q in t:
            vt = struct.unpack_from("<I", d, q)[0]
            # the buffer index is the first field, ELIDED by flatbuffer default when it is 0
            idx.append(struct.unpack_from("<I", d, q+4)[0] if vt >= 8 and q+8 <= len(d) else 0)
        if len(set(idx)) != n or idx != sorted(idx) or any(i >= 31 for i in idx): continue
        # Prefer the longest vector, and on a tie the LAST one: a fixed vector early in the
        # section is byte-identical in every kernel measured and so is not a binding table.
        if best is None or n > best[0] or (n == best[0] and p > best[1]): best = (n, p, idx)
    return best[2] if best else None

def main(argv=None):
    for path in (sys.argv[1:] if argv is None else argv):
        b = bindings(path)
        print("%-58s %s" % (path.split("/")[-3], 
              " ".join("base %d -> buffer %d" % (k, v) for k, v in enumerate(b)) if b else "no binding table found"))



if __name__ == "__main__":
    main()
