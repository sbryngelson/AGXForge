#!/usr/bin/env python3
"""Attribute a residual bit by FLIPPING it and reading the operand back through Apple's decoder.

Correlation can only explain a bit the corpus varies. Some bits never vary - every offset in the
cache is a multiple of 32 below 8192, so an offset field's high bits are unreachable at any sample
size - and for those the corpus is the wrong instrument entirely.

The decoder is the right one. tools/agx3dis is Apple's own MCDisassembler, constructed from the
target's instruction description, so what it reports for a mutated instruction is what that
description says the bytes mean. Flip one bit, decode again, and see which operand moved.

    python3 tools/g17bitprobe.py 12674          probe every residual bit of one opcode
    python3 tools/g17bitprobe.py --residue      every form that still has residue

WHAT THIS IS AND IS NOT. It is a statement about Apple's INSTRUCTION DESCRIPTION, not about
silicon: the decoder was built from the description and reads back what the description encodes.
That is the direction the oracle holds, and it is the same standing rule as everywhere else here -
a decoder acceptance covers operand fields and nothing about behaviour. Nothing is dispatched.

A REJECTION IS ALSO INFORMATION. If flipping a bit makes the decoder refuse the instruction, that
bit is not a free field: it participates in whatever the description uses to recognise the form.
"""
import collections, os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "spike", "accel", "re"))
import g17context, g17layout, g17slice

DIS = os.path.join(HERE, "agx3dis")


def decode(data):
    """[(opcode, [(kind, value)...])] for a byte string holding one instruction, or None."""
    with tempfile.NamedTemporaryFile(suffix=".bin") as fh:
        fh.write(data)
        fh.flush()
        r = subprocess.run([DIS, fh.name, "0", str(len(data)), "--pc", "0"],
                           capture_output=True, text=True)
    for line in r.stdout.splitlines():
        p = line.split()
        if len(p) < 3 or p[1] == "bad":
            return None
        ops = [(k, int(v, 0)) for k, v in (t.split(":", 1) for t in p[3:] if ":" in t)]
        return int(p[2]), int(p[1]), ops
    return None



FILLER = bytes.fromhex("0407")   # op11842, movimm r105, 7 - two bytes, always decodes


def decode_many(samples, slot=16):
    """Decode many candidate instructions in ONE disassembler run.

    A single-bit mutation can change an instruction's LENGTH, which desynchronises a stream, so
    each candidate gets its own fixed-width slot and only the decode that starts exactly on a
    slot boundary is believed. Anything the decoder finds inside the padding is ignored. This is
    what makes an exhaustive sweep over a 200-opcode family affordable: one subprocess instead of
    twenty thousand.

    THE PADDING HAS TO DECODE. agx3dis stops at the first `bad` line, so zero padding silently
    truncated the run after the first slot and the sweep reported that nothing was reachable -
    a clean false negative that the positive control below catches. FILLER is a two-byte movimm,
    and every instruction length in this ISA is even, so it always lands the next slot on its
    boundary.

    AND THE PRINTER CAN ABORT THE PROCESS. Some encodings make Apple's MCInstPrinter call
    `LLVM ERROR: Unhandled atomic op` and abort, which kills every candidate after it in the
    buffer - 451 lines out of 640, silently, with the loss depending on where the poison landed.
    That produced answers that changed with the batch size. On a non-zero exit with a short
    result the batch is SPLIT and retried, down to single samples, so one poisonous encoding
    costs only its own slot.
    """
    if not samples:
        return []
    buf = bytearray()
    for s in samples:
        b = bytes(s)
        if len(b) > slot:
            raise ValueError("sample longer than slot")
        buf += b + FILLER * ((slot - len(b)) // len(FILLER))
        buf += b"\x00" * ((slot - len(b)) % len(FILLER))
    with tempfile.NamedTemporaryFile(suffix=".bin") as fh:
        fh.write(bytes(buf))
        fh.flush()
        r = subprocess.run([DIS, fh.name, "0", str(len(buf)), "--pc", "0",
                            "--stride", str(slot)], capture_output=True, text=True)
    out = [None] * len(samples)
    seen = 0
    for line in r.stdout.splitlines():
        p = line.split()
        if len(p) < 3 or p[1] == "bad":
            if len(p) >= 2 and p[1] == "bad":
                seen = max(seen, int(p[0], 16) // slot + 1)
            continue
        off = int(p[0], 16)
        if off % slot or off // slot >= len(samples):
            continue
        idx = off // slot
        ln = int(p[1])
        seen = max(seen, idx + 1)
        # THE DECODE MUST BE EXPLAINED BY THE SAMPLE'S OWN BYTES. An instruction longer than the
        # sample is reading the filler, and the answer is then about the padding rather than the
        # candidate - two encodings in one batch came back as 18-byte instructions from 16-byte
        # samples, where the one-at-a-time prober correctly refuses them as a short read.
        if ln > len(samples[idx]):
            continue
        ops = [(k, int(v, 0)) for k, v in (t.split(":", 1) for t in p[3:] if ":" in t)]
        out[idx] = (int(p[2]), ln, ops)
    if r.returncode != 0 and seen < len(samples):
        if len(samples) == 1:
            return [None]                      # this one encoding aborts the printer
        mid = len(samples) // 2
        return decode_many(samples[:mid], slot) + decode_many(samples[mid:], slot)
    return out


def probe(sample, bit):
    """What flipping `bit` does: (verdict, detail). `sample` is one instruction's bytes."""
    base = decode(sample)
    if base is None:
        return "sample does not decode", ""
    byte, index = bit
    mutated = bytearray(sample)
    mutated[byte] ^= 1 << index
    got = decode(bytes(mutated))
    if got is None:
        return "REJECTED", "the decoder refuses the instruction with this bit flipped"
    if got[0] != base[0]:
        return "changes the OPCODE", "op%d -> op%d" % (base[0], got[0])
    if got[1] != base[1]:
        return "changes the LENGTH", "%d -> %d bytes" % (base[1], got[1])
    # EXPR OPERANDS MUST BE IGNORED. Apple's printer allocates a fresh MCExpr on every decode, so
    # its reported value changes when ANY bit is flipped - including bits with a known unrelated
    # role. Measured: flipping the destination's slot bit 3 moves both operand 0 and the expr in
    # 18 of 20 samples, and so does flipping the offset's bit 5, and the packed immediate's. A
    # comparison that includes them reports every bit as belonging to the expr, which is how this
    # tool first read byte1[6] as part of the base register field. It is allocator noise.
    moved = [(k, a[1], b[1]) for k, (a, b) in enumerate(zip(base[2], got[2]))
             if a[1] != b[1] and a[0] != "expr"]
    if not moved:
        return "no operand changes", "the description does not read this bit"
    if len(moved) == 1:
        k, a, b = moved[0]
        d = b - a
        power = ""
        if d and (abs(d) & (abs(d) - 1)) == 0:
            power = " = 2^%d" % (abs(d).bit_length() - 1)
        return "operand %d" % k, "%d -> %d, delta %+d%s" % (a, b, d, power)
    return "operands %s" % [k for k, _, _ in moved], \
           " ".join("op%d:%+d" % (k, b - a) for k, a, b in moved)


def main():
    named = dict(g17slice.KNOWN_OPS)
    streams = g17context.walk()
    rows = g17context.features(streams, g17context.num_defs())
    want = {int(a) for a in sys.argv[1:] if a.isdigit()}
    for op in sorted(want or rows):
        rs = rows.get(op, [])
        if len(rs) < 8:
            continue
        allr = [(b, o) for b, o, _ in rs]
        by = collections.defaultdict(list)
        for b, o, _ in rs:
            by[(tuple(k for k, _ in o), len(b))].append((b, o))
        for key, sub in sorted(by.items(), key=lambda kv: -len(kv[1])):
            if len(sub) < 8:
                continue
            a = g17layout.analyse(op, sub, all_rows=allr)
            if not (a and a[6]):
                continue
            print("\nop%-7d %-9s %2d bytes n=%-5d" % (op, named.get(op, "unnamed"),
                                                      key[1], len(sub)))
            for bit in sorted(a[6]):
                verdicts = collections.Counter()
                for b, _ in sub[:24]:
                    verdicts[probe(bytes(b), bit)] += 1
                for (v, d), n in verdicts.most_common(3):
                    print("   b%d[%d]  %-22s %-42s x%d" % (bit[0], bit[1], v, d, n))


if __name__ == "__main__":
    main()
