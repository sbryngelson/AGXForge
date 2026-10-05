#!/usr/bin/env python3
"""AUTHOR a native object's binding table - the encoder side of tools/g17bind.py.

A load or store's base field is a compacted index over the buffers a kernel uses, and the binding
table in __GPU_METADATA maps index k to a Metal buffer number. Decoding it was solved; emitting it
was not, so those 20 bytes were being carried from an Apple container even though they are
per-kernel data and not a constant.

THE STRUCTURE, read out of a real object rather than assumed:

    a flatbuffer vector: u32 count, then one u32 offset per entry, each relative to its own slot
    each entry points FORWARD to a table whose first field is the Metal buffer index
    the index is ELIDED when it is 0, which shortens that record's vtable

Elision is the constraint on writing in place: a record that stores buffer 0 has no field to
overwrite, so it can be left alone or read as 0, but it cannot be raised to a non-zero buffer
without growing the record and moving everything after it. That case raises rather than silently
writing into a neighbour.
"""
import struct


def locate(md):
    """-> (vector position, [record offsets], [buffer indices]) or None.

    Same search as g17bind.bindings, kept here so the encoder and decoder agree on which vector is
    the binding table rather than trusting two independent scans to pick the same one."""
    best = None
    for p in range(0, len(md) - 8, 4):
        n = struct.unpack_from("<I", md, p)[0]
        if not (1 <= n <= 31) or p + 4 + 4 * n > len(md):
            continue
        recs = []
        for k in range(n):
            off = struct.unpack_from("<I", md, p + 4 + 4 * k)[0]
            q = p + 4 + 4 * k + off
            if off == 0 or not (p < q < len(md)):
                recs = None; break
            recs.append(q)
        if not recs or len(set(recs)) != n or recs != sorted(recs, reverse=True):
            continue
        idx = []
        for q in recs:
            vt = struct.unpack_from("<I", md, q)[0]
            idx.append(struct.unpack_from("<I", md, q + 4)[0] if vt >= 8 and q + 8 <= len(md) else 0)
        if len(set(idx)) != n or idx != sorted(idx) or any(i >= 31 for i in idx):
            continue
        if best is None or n > best[0] or (n == best[0] and p > best[1]):
            best = (n, p, recs, idx)
    return None if best is None else (best[1], best[2], best[3])


def encode(md, buffers):
    """Return `md` with base index k naming Metal buffer `buffers[k]`."""
    found = locate(md)
    if found is None:
        raise ValueError("no binding table in this metadata section")
    pos, recs, cur = found
    if len(buffers) != len(recs):
        raise ValueError("this table has %d entries; %d buffers given" % (len(recs), len(buffers)))
    out = bytearray(md)
    for q, want, have in zip(recs, buffers, cur):
        vt = struct.unpack_from("<I", out, q)[0]
        if vt >= 8:
            struct.pack_into("<I", out, q + 4, want)
        elif want != 0:
            raise ValueError("record at %d elides its buffer index, so it names buffer 0; raising "
                             "it to %d needs the record grown, which moves every later offset"
                             % (q, want))
    return bytes(out)
