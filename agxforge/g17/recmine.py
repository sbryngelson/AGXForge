"""Record-table and slot-2 vector readers, vendored so a tracked tool has a tracked dependency.

These two functions were the only things tools/g17nobuild.py used out of two analysis scripts,
recmine.py and vec2.py, that existed ONLY in one session's scratchpad directory. g17nobuild
imported them by absolute path:

    from .recmine import rec_tables
    from vec2 import vec_records

so a tracked tool - and test_g17halfimage's cold/warm subprocess, which imports it - could not run
in any other session, on any other machine, or from a fresh clone. Neither script was ever tracked
and neither is in git history anywhere. Importing them also RAN them: both do a corpus-wide sweep
at module scope and print, which is why importing g17nobuild printed a census.

Only the readers are vendored. The analysis around them was a one-time mining pass whose result
already lives in the class rules; re-running it on import is not what g17nobuild wanted.

Both functions read a metadata section and return what its tables say. Neither decides anything.
"""
import struct

from . import gpumd as GM


def rec_tables(md):
    """Yield (vector slot, slot set, offset map, declared length) for records in {26, 4, 2}.

    Bounds are checked at every step and a malformed table is skipped rather than guessed at, so
    this can be pointed at a whole corpus without a single bad section stopping the sweep.
    """
    pk = GM.kernel_table(md)
    if pk is None:
        return
    try:
        sl, _ = GM.table_at(md, pk)
    except Exception:
        return
    for v in (26, 4, 2):
        if len(sl) <= v or not sl[v]:
            continue
        a = pk + sl[v]
        if a + 4 > len(md):
            continue
        vaddr = a + struct.unpack_from("<I", md, a)[0]
        if vaddr + 4 > len(md):
            continue
        n = struct.unpack_from("<I", md, vaddr)[0]
        if n > 4096 or vaddr + 4 + 4 * n > len(md):
            continue
        for k in range(n):
            p = vaddr + 4 + 4 * k
            t = p + struct.unpack_from("<I", md, p)[0]
            if not (0 < t < len(md) - 1):
                continue
            try:
                s2, tsz = GM.table_at(md, t)
            except Exception:
                continue
            sm = {i: x for i, x in enumerate(s2) if x}
            if sm:
                yield v, tuple(sorted(sm)), tuple(sorted(sm.items())), tsz


def vec_records(md, slot):
    """The field maps of the records in one vector, or None when the vector is absent."""
    pk = GM.kernel_table(md)
    if pk is None:
        return None
    sl, _ = GM.table_at(md, pk)
    if len(sl) <= slot or not sl[slot]:
        return None
    a = pk + sl[slot]
    v = a + struct.unpack_from("<I", md, a)[0]
    if v + 4 > len(md):
        return None
    n = struct.unpack_from("<I", md, v)[0]
    if n > 4096 or v + 4 + 4 * n > len(md):
        return None
    out = []
    for k in range(n):
        p = v + 4 + 4 * k
        t = p + struct.unpack_from("<I", md, p)[0]
        s2, tsz = GM.table_at(md, t)
        occ = sorted({s for s in s2 if s})
        w = {occ[j]: occ[j + 1] - occ[j] for j in range(len(occ) - 1)}
        if occ:
            w[occ[-1]] = tsz - occ[-1]
        f = {}
        for i, s in enumerate(s2):
            ww = w.get(s, 4)
            # WIDTH-AWARE BOUNDS, NOT FOUR. Requiring four bytes for a one-byte field drops every
            # field that sits near the end of the section - a6-2d's record at 388 of 396 read as
            # {} and caused two WRONG builds. This is the fix g17facts.binding_records already
            # carries; the scratchpad copy this came from carried it too, and the comment is kept
            # because the failure it names is the reason the width is consulted at all.
            if not s or t + s + min(ww, 4) > len(md):
                continue
            q = t + s
            f[i] = (md[q] if ww == 1 else struct.unpack_from("<H", md, q)[0] if ww == 2
                    else struct.unpack_from("<I", md, q)[0])
        out.append(f)
    return out
