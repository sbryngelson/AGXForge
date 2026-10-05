#!/usr/bin/env python3
"""Classify every bit of every instruction, so an encoder can emit one without a template.

`g17asm` encodes by taking an instruction Apple wrote and overwriting the bits it owns. Everything
it does not own is INHERITED, and inherited bits are the reason the assembler cannot author a form
Apple never emitted: nobody knows what they mean, so nobody knows what to put there.

Measured: 90.5% of bits across the 6,718 admitted opcodes are explained by the opcode identity, an
operand field, or being dead - and only NINE opcodes have zero unexplained bits. The remaining
79,338 bits are what a template supplies.

THE CLASSIFICATION, by mutating one bit at a time and asking the decoder:

    opcode      flipping it decodes as a different opcode, so it is part of the identity and its
                value is fixed by the choice of opcode
    operand     flipping it changes a printed operand, so it belongs to a field the prober missed
    invisible   flipping it changes nothing the decoder prints - either genuinely don't-care, or a
                modifier the printer ignores
    illegal     flipping it makes the encoding undecodable, which pins the bit to its witness value

`invisible` is the only class this cannot settle by decoding, and it is settled by EXECUTION: run
the instruction with the bit clear and with it set and compare the result. Same result means
don't-care and the canonical encoder may emit zero; different result means a semantic modifier
that has to be characterised before the form is offered to the selector.

    python3 tools/g17canon.py --classify [--limit N]    decode-level classification, all opcodes
    python3 tools/g17canon.py --report                  coverage after classification
"""
import collections, json, os, re, subprocess, sys, tempfile
# siblings come from the package
from agxforge.g17 import metal as g17metal, opclass as g17opclass, slice as g17slice

HERE = os.path.dirname(os.path.abspath(__file__))
# ANCHORED ON THE CHECKOUT ROOT: two levels up from agxforge/g17/. One level short pointed every
# table in this module at agxforge/isa/, which does not exist - eight paths from one anchor.
ROOT = os.path.dirname(os.path.dirname(HERE))
ISA = os.path.join(ROOT, "isa")
AUTH = os.path.join(ISA, "g17-authoring.jsonl")
OUT = os.path.join(ISA, "g17-bit-classes.jsonl")
OUT_ALL = os.path.join(ISA, "g17-bit-spec.jsonl")
PAD = bytes.fromhex("0600")


EXPR_RE = re.compile(r"^expr:bin\(op(\d+),const\((-?\d+)\),(\d+)\)$")


def token_values(tok):
    """The numeric sub-values a printed operand carries.

    An operand is not always one number. `expr:bin(op0,const(27),2)` is an address expression with
    a BASE OPERAND, a CONSTANT and a SCALE, and each is encoded in its own bits - so a bit that
    moves the constant is a value bit of the constant, not of some opaque whole. Treating the
    token as one value is why 97 reconstructions differed in operand 0: the encoder wrote a
    correct register where Apple wrote an expression with a different constant and scale.

    Returns {sub-field name: value}; a plain operand has the single key "".
    """
    m = EXPR_RE.match(tok)
    if m:
        return {"base": int(m.group(1)), "const": int(m.group(2)), "scale": int(m.group(3))}
    _, _, val = tok.partition(":")
    try:
        return {"": int(val)}
    except ValueError:
        return {}


def pairs(lst):
    for e in lst or []:
        if isinstance(e, (list, tuple)) and len(e) >= 2:
            yield e[0], e[1]


def explained(r, n):
    ok = set()
    for b, bit in pairs(r.get("opcode_bits")):
        ok.add((b, bit))
    for b, bit in pairs(r.get("dead_bits")):
        ok.add((b, bit))
    for b, bit in pairs(r.get("length_bits")):
        ok.add((b, bit))
    for _, v in (r.get("fields") or {}).items():
        for e in v:
            if len(e) >= 3:
                ok.add((e[1], e[2]))
    for _, v in (r.get("solved_operands") or {}).items():
        for e in v.get("bits", []):
            if len(e) >= 3 and e[1] is not None:
                ok.add((e[1], e[2]))
    return {p for p in ok if isinstance(p[0], int) and p[0] < n}


def decode_batch(variants, stride=32):
    blob = b"".join(v.ljust(stride, b"\x06") for v in variants)
    with tempfile.NamedTemporaryFile(suffix=".bin") as fh:
        fh.write(blob)
        fh.flush()
        # --expr OR THE EXPRESSION OPERANDS ARE NOISE. An MCExpr operand prints as the address
        # of the object the decoder just allocated, so it differs on every decode and every bit
        # looks as though it moves it. 6,721 operand bits could not be placed for exactly this
        # reason - the deltas were heap addresses. With --expr the operand prints as its tree.
        r = subprocess.run([g17metal.DIS, fh.name, "0", str(len(blob)), "--pc", "0",
                            "--stride", str(stride), "--expr"], capture_output=True, text=True)
    out = {}
    for line in r.stdout.splitlines():
        p = line.split()
        if len(p) < 2:
            continue
        try:
            off = int(p[0].rstrip(":"), 16)
        except ValueError:
            continue
        if off % stride:
            continue
        out[off // stride] = None if p[1] == "bad" else (int(p[2]), tuple(p[3:]))
    return out


def classify(op, r, witness=None):
    """{(byte, bit): class} for the bits nothing explains, on ONE form of the instruction.

    The unit of specification is (opcode, length), not opcode: 190 of the 717 opcodes Apple emits
    appear at more than one encoded length, and a spec derived from one witness describes only
    that form. Passing the witness explicitly is what lets each form have its own.
    """
    w = witness or r.get("apple_witness") or r["witness"]
    base = bytes.fromhex(w)
    n = len(base)
    # EVERY BIT, not only the ones no field explains. The bits a field DOES explain still take
    # their value from the witness unless something proves it, and "the prober said this is an
    # opcode bit" is not a proof of what to put there. Classifying all of them turns the encoding
    # into a table whose every entry was tested.
    allbits = {(b, i) for b in range(n) for i in range(8)}
    unknown = sorted(allbits if "--all-bits" in sys.argv else allbits - explained(r, n))
    if not unknown:
        return {}, None
    variants = [base]
    for b, i in unknown:
        m = bytearray(base)
        m[b] ^= 1 << i
        variants.append(bytes(m))
    got = decode_batch(variants)
    ref = got.get(0)
    if ref is None or ref[0] != op:
        return {}, "witness does not decode as op%d" % op
    # RECORD WHAT AN ENCODER NEEDS, not just the class. A forced bit needs its proven value; an
    # operand bit needs to know WHICH printed operand it moves, or the encoder cannot place a
    # value in it; an opcode bit needs the opcode the other setting selects, which is what makes
    # the identity a table rather than a witness.
    out = {}
    for k, (b, i) in enumerate(unknown, start=1):
        v = got.get(k)
        held = (base[b] >> i) & 1
        if v is None:
            out[(b, i)] = {"class": "forced", "value": held}
        elif v[0] != op:
            out[(b, i)] = {"class": "opcode", "value": held, "flips_to": v[0]}
        elif v[1] != ref[1] and any(
                x.partition(":")[0] != y.partition(":")[0]
                for x, y in zip(ref[1], v[1])):
            # A MODE BIT, not a value bit. Flipping it changes the KIND of an operand - a
            # register becomes an expression, an immediate becomes a register - so there is no
            # weight to derive and writing zero there silently changes what the instruction takes.
            # 95 reconstructions differed in operand 0 for exactly this reason: Apple's
            # instruction had an expression where the encoder produced a register.
            sw = [[j, x.partition(":")[0], y.partition(":")[0]]
                  for j, (x, y) in enumerate(zip(ref[1], v[1]))
                  if x.partition(":")[0] != y.partition(":")[0]]
            out[(b, i)] = {"class": "mode", "value": held, "switches": sw}
        elif v[1] != ref[1]:
            # RECORD THE DELTA, not just that something moved. An encoder needs to know which
            # VALUE BIT this is, and the size of the change is exactly that: flipping bit k of a
            # field changes the printed value by 2**k. Without it the field map has a hole and the
            # reconstruction writes zero where Apple wrote a bit.
            moved = []
            for j, (x, y) in enumerate(zip(ref[1], v[1])):
                if x == y:
                    continue
                bx, by = token_values(x), token_values(y)
                for key in sorted(set(bx) | set(by)):
                    if bx.get(key) != by.get(key):
                        moved.append([j, key, bx.get(key), by.get(key)])
            out[(b, i)] = {"class": "operand", "value": held, "moved": moved}
        else:
            out[(b, i)] = {"class": "invisible", "value": held}
    return out, None


def main():
    A = {r["opcode"]: r for r in (json.loads(l) for l in open(AUTH))}
    lim = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    if "--report" in sys.argv:
        tot = collections.Counter()
        per = {}
        for line in open(OUT_ALL if "--all-bits" in sys.argv else OUT):
            d = json.loads(line)
            per[d["opcode"]] = d["bits"]
            for c in d["bits"].values():
                tot[c["class"] if isinstance(c, dict) else c] += 1
        print("bits classified: %d" % sum(tot.values()))
        for k, v in tot.most_common():
            print("   %-10s %6d  (%.1f%%)" % (k, v, 100 * v / sum(tot.values())))
        cl = lambda c: c["class"] if isinstance(c, dict) else c
        print("\nopcodes with an INVISIBLE bit (needs execution to settle): %d"
              % sum(1 for cs in per.values() if any(cl(c) == "invisible" for c in cs.values())))
        print("opcodes whose every bit is now determined at the decode level: %d"
              % sum(1 for cs in per.values()
                    if all(cl(c) != "invisible" for c in cs.values())))
        return
    if "--forms" in sys.argv:
        forms = [json.loads(l) for l in open(os.path.join(ISA, "g17-forms.jsonl"))]
        if lim:
            forms = forms[:lim]
        path = os.path.join(ISA, "g17-form-spec.jsonl")
        done = 0
        with open(path, "w") as fh:
            for f in forms:
                op = f["opcode"]
                if op not in A:
                    continue
                cls, err = classify(op, A[op], witness=f["witness"])
                fh.write(json.dumps({"opcode": op, "length": f["length"], "error": err,
                                     "witness": f["witness"],
                                     "bits": {"%d.%d" % k: v for k, v in cls.items()}}) + "\n")
                done += 1
                if done % 200 == 0:
                    print("   %d/%d forms classified" % (done, len(forms)), flush=True)
        print("wrote %s for %d forms" % (path, done))
        return
    ops = sorted(A)
    if lim:
        ops = ops[:lim]
    done = 0
    path = OUT_ALL if "--all-bits" in sys.argv else OUT
    with open(path, "w") as fh:
        for op in ops:
            cls, err = classify(op, A[op])
            fh.write(json.dumps({"opcode": op, "error": err,
                                 "bits": {"%d.%d" % k: v for k, v in cls.items()}}) + "\n")
            done += 1
            if done % 400 == 0:
                print("   %d/%d opcodes classified" % (done, len(ops)), flush=True)
    print("wrote %s for %d opcodes" % (path, done))


if __name__ == "__main__":
    main()
