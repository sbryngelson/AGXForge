#!/usr/bin/env python3
"""What Apple actually puts in the bits the specification calls invisible, per (opcode, length).

`g17encode.encode` leaves an invisible bit at zero unless the free-bit census found it constant.
That is defensible - Apple's own instances disagree, so no value is required - but it is not what
Apple writes, and zero is a choice too. Measured on 6,594 real programs, op423's b0.5, b6.5 and
b6.7 are set in every single instance, which makes an all-zero encoding an instruction Apple never
emits, on three bits, every time.

So record the modal value of every bit of every (opcode, length) together with how much of the
corpus agrees. An assembler can then fill an invisible bit with the value Apple carries rather
than with zero, and say in the listing that it did.

This is a DEFAULT, not a semantic claim. The add family's residue was separately shown causally
inert - both bits authored in reverse, fifteen preregistered programs, all correct - so agreement
here is about staying on Apple's manifold, not about correctness.

    python3 tools/g17modal.py
"""
import collections, json, os

# ANCHORED ON THE CHECKOUT ROOT, NOT ON THIS FILE'S PARENT. Under tools/ one level up WAS the
# checkout; under agxforge/g17/ it is agxforge/, and `ISA` would have become agxforge/isa/ - which does not
# exist. OUT is WRITTEN by a refresh, so the wrong anchor would not have failed: it would have
# created a second modal table there and left the real one stale.
#
# The sys.path insertion this module carried is gone with it. It added its own directory so that
# siblings under tools/ could be imported; the library must not put tools/ on the import path, and
# this module imports no sibling.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ISA = os.path.join(ROOT, "isa")
CORPUS = os.path.join(ISA, "g17-corpus-programs.jsonl")
OUT = os.path.join(ISA, "g17-modal-bits.jsonl")

_M = None


def modal():
    """{(opcode, length): {"bit b.i": [value, agreement]}}"""
    global _M
    if _M is None:
        _M = {}
        if os.path.exists(OUT):
            for line in open(OUT):
                r = json.loads(line)
                _M[(r["opcode"], r["length"])] = r["bits"]
    return _M


def main():
    ones = collections.defaultdict(collections.Counter)
    total = collections.Counter()
    for line in open(CORPUS):
        d = json.loads(line)
        code = bytes.fromhex(d["text"])
        for o, l, op in d["spans"]:
            raw = code[o:o + l]
            if len(raw) != l:
                continue
            total[(op, l)] += 1
            for b in range(l):
                for i in range(8):
                    if (raw[b] >> i) & 1:
                        ones[(op, l)]["%d.%d" % (b, i)] += 1
    with open(OUT, "w") as fh:
        for key in sorted(total):
            op, l = key
            n = total[key]
            bits = {}
            for b in range(l):
                for i in range(8):
                    k = "%d.%d" % (b, i)
                    c = ones[key].get(k, 0)
                    v = 1 if c * 2 > n else 0
                    bits[k] = [v, round((c if v else n - c) / float(n), 4)]
            fh.write(json.dumps({"opcode": op, "length": l, "n": n, "bits": bits}) + "\n")
    print("modal bit values for %d (opcode, length) forms over %d instructions"
          % (len(total), sum(total.values())))
    print("   -> %s" % OUT)


if __name__ == "__main__":
    main()
