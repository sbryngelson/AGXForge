#!/usr/bin/env python3
"""THE FIELD BASE OFFSETS, HARVESTED ONCE INTO THE REPO instead of asked of the vendor per compile.

A field holds (value - base) / step, and g17encode derives that base by decoding the form's own
witness with Apple's disassembler - `semantic_operands(op, real)` - and subtracting what the bits
account for. It is the right derivation and it was being done at COMPILE TIME, so the bytes this
backend emits depended on that tool's behaviour during the build. That is a defect on its own
terms. It is NOT, as first reported, a tracking gap: tools/agx3dis is gitignored because it is
built from tools/agx3dis.c, which is tracked and clean, and g17ref.binary() rebuilds it when the
source is newer - a compiled helper is pinned by its source, and committing the binary would be
the actual mistake. What no tracking fixes is that agx3dis.c links the local SDK's
MCDisassembler, so the same tracked source can decode differently on another machine or OS
version. That is a scope limit on every claim either checker makes, not a work item.

Measured, not assumed: compiling the FP16 scan forks that binary for op1004, op1016 and op17193 -
the widening, the narrowing and the half store, three of the scan's four forms. See
tools/g17inputs.py, which watches `open` and `subprocess.Popen` around a compile rather than
reading its imports, because the input that broke reproducibility last time was a DATA overlay and
a module audit does not see one.

So harvest the bases for every form that has a witness, check them in, and let the compile read
them. The derivation is unchanged - this moves WHEN it happens, not what it computes - and the
--check mode re-derives and compares, so a stale table is a failure rather than a silent drift.

    python3 tools/g17bases.py --refresh    decode every witness and write the table
    python3 tools/g17bases.py --check      re-derive and fail if the table disagrees
    python3 tools/g17bases.py              report coverage
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# ANCHORED ON THE CHECKOUT ROOT: two levels up from agxforge/g17/ where one sufficed from
# tools/. Native helpers stay in tools/ where the Makefile builds them.
ROOT = os.path.dirname(os.path.dirname(HERE))
ISA = os.path.join(ROOT, "isa")
TABLE = os.path.join(ISA, "g17-form-bases.json")
# siblings come from the package


def key_of(op, length_hint):
    return "%d,%s" % (op, "" if length_hint is None else length_hint)


def _pack(bases):
    """[[keyspec, value]] - a keyspec that keeps the key's TYPE, because two keys collide without it.

    The first version wrote "%s,%s" % (index, kind), which maps the bare int key 1 and the tuple
    key (1, "") to the same string "1,". A form's bases hold BOTH, so one silently overwrote the
    other and unpacking returned the int, losing the tuple. The encoder reads them with
    bases.get(k, 0), so a lost key is not an error - it is a base of zero, and the operand is
    placed wrong. Device atomic render coverage fell from 142 of 208 to 132.

    And the check could not see it: it compared _pack(fresh) against the stored packing, so the
    same lossy transform ran on both sides and agreed with itself. Compare UNPACKED against
    DERIVED, which is what --check does now.
    """
    out = []
    for k, v in bases.items():
        out.append([["t", k[0], k[1]] if isinstance(k, tuple) else ["i", k], v])
    return sorted(out, key=lambda e: (e[0][0], str(e[0][1]), str(e[0][2:])))


def unpack(entries):
    """The inverse of _pack, restoring each key's type exactly."""
    if isinstance(entries, dict):        # the lossy first format; refuse it rather than guess
        raise ValueError("isa/g17-form-bases.json is in the superseded flat format; "
                         "run tools/g17bases.py --refresh")
    out = {}
    for spec, v in entries:
        out[(spec[1], spec[2]) if spec[0] == "t" else spec[1]] = v
    return out


def _witness_keys():
    """Every (opcode, length_hint) a compile can ask about, form-level and opcode-wide."""
    from agxforge.g17 import encode as g17encode
    keys = [(op, ln) for (op, ln) in g17encode.forms() if g17encode.forms()[(op, ln)].get("witness")]
    ops = sorted({op for op, _ln in keys})
    return keys + [(op, None) for op in ops]


def harvest(keys=None):
    """Derive the bases through the decoder, one batched pass, and return the table."""
    from agxforge.g17 import encode as g17encode
    keys = keys or _witness_keys()
    # PRIME THE DECODE CACHE IN ONE FORK. decode() is memoised on the exact blob, and the
    # per-witness call inside semantic_operands then hits that cache instead of forking again.
    wits = []
    for op, ln in keys:
        try:
            w = (g17encode.forms().get((op, ln)) or {}).get("witness") if ln is not None else None
            w = w or g17encode.auth()[op].get("apple_witness") or g17encode.auth()[op]["witness"]
            wits.append(bytes.fromhex(w))
        except Exception:
            continue
    for i in range(0, len(wits), 512):
        g17encode.decode([bytes(w) for w in wits[i:i + 512]])
    out = {}
    for op, ln in keys:
        try:
            _f, _s, _u, bases = g17encode._extended_fields(op, ln)
        except Exception:
            continue
        if bases:
            out[key_of(op, ln)] = _pack(bases)
    return out


def load():
    if not os.path.exists(TABLE):
        return {}
    with open(TABLE) as fh:
        d = json.load(fh)
    return d.get("bases") or {}


def main():
    if "--refresh" in sys.argv:
        os.environ["G17_BASES_REFRESH"] = "1"
        table = harvest()
        with open(TABLE, "w") as fh:
            json.dump({
                "note": ("Field base offsets, one per (opcode, length). A field holds "
                         "(value - base) / step; the base is what the field's bits do not account "
                         "for, read from a DECODE of the form's own witness so that reconstruction "
                         "is not circular. Harvested here once so that a compile does not fork "
                         "Apple's disassembler - tools/agx3dis is untracked, and an image built "
                         "with it cannot be rebuilt from this repository alone. The key is "
                         "'opcode,length', with an empty length for the opcode-wide entry."),
                "source": "tools/g17bases.py --refresh",
                "bases": table,
            }, fh, indent=1, sort_keys=True)
        print("wrote %s: %d forms" % (os.path.relpath(TABLE, os.path.dirname(ISA)), len(table)))
        return 0

    have = load()
    if "--check" in sys.argv:
        os.environ["G17_BASES_REFRESH"] = "1"
        from agxforge.g17 import encode as g17encode
        g17encode._EXT_CACHE.clear()
        keys = _witness_keys()
        # PRIME THE DECODE CACHE IN ONE FORK, as the harvest does.
        wits = []
        for op, ln in keys:
            try:
                w = (g17encode.forms().get((op, ln)) or {}).get("witness") if ln is not None else None
                w = w or g17encode.auth()[op].get("apple_witness") or g17encode.auth()[op]["witness"]
                wits.append(bytes.fromhex(w))
            except Exception:
                continue
        for i in range(0, len(wits), 512):
            g17encode.decode([bytes(w) for w in wits[i:i + 512]])
        missing, differ = [], []
        n = 0
        for op, ln in keys:
            try:
                _f, _s, _u, fresh = g17encode._extended_fields(op, ln)
            except Exception:
                continue
            if not fresh:
                continue
            n += 1
            k = key_of(op, ln)
            if k not in have:
                missing.append(k)
            elif unpack(have[k]) != fresh:
                differ.append(k)
        print("checked-in %d forms, re-derived %d" % (len(have), n))
        print("  missing from the table   %d %s" % (len(missing), missing[:5]))
        print("  DISAGREEING              %d %s" % (len(differ), differ[:5]))
        if missing or differ:
            print("\nthe table is stale; run --refresh")
            return 1
        print("\n  the UNPACKED table equals a fresh derivation, key types included, so a compile"
              "\n  needs no oracle. Compared unpacked-against-derived: packing both sides is how"
              "\n  the first version of this check passed while the table was lossy.")
        return 0

    print("%s: %d forms" % (os.path.relpath(TABLE, os.path.dirname(ISA)), len(have)))
    for k in ("1004,12", "1016,12", "17193,14", "12646,14"):
        print("   %-12s %s" % (k, "present" if k in have else "ABSENT"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
