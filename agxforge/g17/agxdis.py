#!/usr/bin/env python3
"""Disassembler/assembler for the recovered AGX tensor-bound instruction subset.

Driven by isa/tensor-isa.toml so the description and the tool cannot drift apart.
Only fields validated by native-code mutation are decoded; everything else is
printed as raw bytes rather than guessed at.

  python3 tools/agxdis.py dis <applegpu-object>       # decode tensor-bound units
  python3 tools/agxdis.py asm a --width N [--offset K] # emit a tensor.bound.a byte 8/9
  python3 tools/agxdis.py asm b --width N              # emit a tensor.bound.b byte 8/9
"""
import argparse, os, struct, sys, tomllib

# ANCHORED ON THE CHECKOUT ROOT. Under tools/ two dirnames reached the checkout; from agxforge/g17/
# the same expression stops at agxforge/. This one at least fails LOUDLY - the ISA table is loaded at
# module level on the next line, so a wrong anchor raises on import rather than at first use.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ISA = tomllib.load(open(os.path.join(ROOT, "isa", "tensor-isa.toml"), "rb"))
UNIT = ISA["meta"]["unit_bytes"]
OPC = {i["opcode_byte0"]: i for i in ISA["instruction"] if "opcode_byte0" in i}
# bytes 4..7 are identical in both observed forms; used as a signature so random
# bytes that happen to start 0x37 are not decoded as instructions
SIG = {4: 0x21, 5: 0x20, 6: 0xA1}
# The MAC/issue family is 10 bytes, not 12, and carries the operand dtypes. Its
# signature is deliberately tight: byte6 and byte7 also hold the dtype bits, so the
# mask must ignore exactly those and nothing else.
MAC_UNIT = 10


def is_mac(u):
    # byte8 bit4 marks the relaxed-fp32 arithmetic mode; every other bit of byte8 is
    # zero in all catalogued issues. Requiring byte8 == 0x00 silently hid all 48 issues
    # of the f32relax shader from the disassembler.
    return (len(u) >= MAC_UNIT and (u[4] & 0xF7) == 0x22 and (u[6] & 0xFB) == 0xA0
            and (u[8] & 0xEF) == 0x00 and (u[7] & 0x1F) == 0x02)


def decode_mac(u):
    if not is_mac(u):
        return None
    a = "half" if (u[6] >> 2) & 1 else "bfloat"
    b = "half" if (u[7] >> 6) & 1 else "bfloat"
    slice_ = (u[3] >> 4) & 1
    off = (u[3] >> 5) & 1
    return dict(name="tensor.mac", a_dtype=a, b_dtype=b, acc=u[1],
                relaxed_fp32=(u[8] >> 4) & 1,
                k_slice=slice_, enable=0 if off else 1,
                text="tensor.mac %s kslice=%d%s acc=0x%02x"
                     % ("relaxed-fp32" if (u[8] >> 4) & 1 else "a=%-6s b=%-6s" % (a, b),
                        slice_, " DISABLED" if off else "", u[1]))


def encode_mac(u, a_dtype, b_dtype, k_slice=None, enable=None):
    """Rewrite only the recovered bits; every other bit is carried through."""
    b6 = (u[6] & ~0x04) | ((a_dtype == "half") << 2)
    b7 = (u[7] & ~0x40) | ((b_dtype == "half") << 6)
    b3 = u[3]
    if k_slice is not None:
        b3 = (b3 & ~0x10) | ((k_slice & 1) << 4)
    if enable is not None:
        b3 = (b3 & ~0x20) | ((0 if enable else 1) << 5)
    return b6, b7, b3


def decode(u):
    """u: 12 bytes -> recovered fields, or None if not a tensor-bound unit.

    Both families share ONE width field, validated causally on 256/256 values of
    byte 8 for each:  W = 8*bits[5:3] + 2*bit1,  O = bits[7:6] | byte9[1:0]<<2,
    end = 64 + W + O.  They differ only in where their region starts.
    """
    if len(u) < UNIT or u[0] not in OPC or any(u[k] != v for k, v in SIG.items()):
        return None
    name = OPC[u[0]]["name"]
    b8, b9 = u[8], u[9]
    W = 8 * ((b8 >> 3) & 7) + 2 * ((b8 >> 1) & 1)
    O = ((b8 >> 6) & 3) | ((b9 & 3) << 2)
    if name == "tensor.bound.a":
        end = 64 + W + O
        return dict(name=name, W=W, O=O, start=64 + O, end=end,
                    text="%s start=%d end=%d (W=%d O=%d)" % (name, 64 + O, end, W, O))
    end = 64 + W + O + 16 * ((b9 >> 2) & 1)
    return dict(name=name, W=W, O=O, start=80, end=end,
                text="%s start=80 end=%d (W=%d O=%d)" % (name, end, W, O))


def encode(name, W, O, extra16=0):
    """Canonical inverse of decode: returns (byte8, byte9) with inert bits zero."""
    if W % 8 not in (0, 2):
        sys.exit("W=%d not representable: needs 8*u + 2*p with p in {0,1}" % W)
    u8 = W // 8
    if u8 > 7:
        sys.exit("W=%d exceeds the 3-bit unit field" % W)
    b8 = (u8 << 3) | (((W % 8) // 2) << 1) | ((O & 3) << 6)
    b9 = (O >> 2) & 3
    if name == "tensor.bound.b":
        b9 = (b9 & ~0b100) | ((extra16 & 1) << 2)
    return b8, b9


def sections(obj):
    ncmds = struct.unpack_from("<I", obj, 0x10)[0]
    off = 0x20
    for _ in range(ncmds):
        cmd, sz = struct.unpack_from("<II", obj, off)
        if cmd == 0x19:
            n = struct.unpack_from("<I", obj, off + 64)[0]
            so = off + 72
            for _ in range(n):
                nm = obj[so:so + 16].rstrip(b"\0").decode()
                _a, size, foff = struct.unpack_from("<QQI", obj, so + 32)
                if nm == "__text":
                    return foff, size
                so += 80
        off += sz
    sys.exit("no __TEXT,__text")


# Bits the description marks inert: carried through a re-encode unchanged, because
# an inert bit is not something the semantics can regenerate. Listing them here is
# what makes the round-trip a test of the SEMANTIC bits rather than of luck.
INERT = {"tensor.bound.a": {8: 0b0000_0101, 9: 0b1111_1100},
         "tensor.bound.b": {8: 0b0000_0101, 9: 0b1111_1000}}


def reencode(u):
    """Rebuild bytes 8 and 9 from decoded semantics alone; inert bits carried over."""
    r = decode(u)
    if not r:
        return None, None
    name = r["name"]
    b8, b9 = encode(name, r["W"], r["O"], (u[9] >> 2) & 1 if name == "tensor.bound.b" else 0)
    m8, m9 = INERT[name][8], INERT[name][9]
    return (b8 & ~m8) | (u[8] & m8), (b9 & ~m9) | (u[9] & m9)


def roundtrip(paths):
    tot = exact = amb = 0
    for path in paths:
        obj = open(path, "rb").read()
        foff, size = sections(obj)
        text = obj[foff:foff + size]
        for i in range(len(text) - MAC_UNIT + 1):
            um = text[i:i + MAC_UNIT]
            rm = decode_mac(um)
            if rm:
                tot += 1
                b6, b7, b3 = encode_mac(um, rm["a_dtype"], rm["b_dtype"],
                                        rm["k_slice"], rm["enable"])
                if (b6, b7, b3) == (um[6], um[7], um[3]):
                    exact += 1
                else:
                    amb += 1
                    print("  mac +0x%-5x re-encode %02x %02x %02x != %02x %02x %02x"
                          % (i, b6, b7, b3, um[6], um[7], um[3]))
                continue
            if i + UNIT > len(text):
                continue
            u = text[i:i + UNIT]
            r = decode(u)
            if not r:
                continue
            tot += 1
            b8, b9 = reencode(u)
            if (b8, b9) == (u[8], u[9]):
                exact += 1
            else:
                amb += 1
                print("  %-28s +0x%-5x %s  orig b8=%02x b9=%02x -> re-encoded %02x %02x  (%s)"
                      % (os.path.basename(os.path.dirname(os.path.dirname(path))), i,
                         r["name"], u[8], u[9], b8, b9, r["text"]))
    print("  %d tensor instructions, %d re-encode byte-exact, %d divergent"
          % (tot, exact, amb))
    return amb


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dis"); d.add_argument("path")
    rt = sub.add_parser("roundtrip"); rt.add_argument("paths", nargs="+")
    pt = sub.add_parser("patch")
    pt.add_argument("path"); pt.add_argument("--out", required=True)
    pt.add_argument("--instruction", required=True); pt.add_argument("--index", type=int, default=0)
    pt.add_argument("--width", type=int); pt.add_argument("--offset", type=int, default=None)
    a = sub.add_parser("asm"); a.add_argument("which", choices=["a", "b"])
    a.add_argument("--width", type=int, required=True, help="W = end - start")
    a.add_argument("--offset", type=int, default=0)
    ns = ap.parse_args()
    if ns.cmd == "roundtrip":
        sys.exit(1 if roundtrip(ns.paths) else 0)
    if ns.cmd == "patch":
        obj = bytearray(open(ns.path, "rb").read())
        foff, size = sections(obj)
        hits = [i for i in range(foff, foff + size - UNIT + 1)
                if (r := decode(obj[i:i + UNIT])) and r["name"] == ns.instruction]
        if ns.index >= len(hits):
            sys.exit("only %d %s instruction(s)" % (len(hits), ns.instruction))
        i = hits[ns.index]
        cur = decode(obj[i:i + UNIT])
        w = ns.width if ns.width is not None else cur["W"]
        o = ns.offset if ns.offset is not None else cur["O"]
        b8, b9 = encode(ns.instruction, w, o)
        m8, m9 = INERT[ns.instruction][8], INERT[ns.instruction][9]
        obj[i + 8] = (b8 & ~m8) | (obj[i + 8] & m8)
        obj[i + 9] = (b9 & ~m9) | (obj[i + 9] & m9)
        open(ns.out, "wb").write(bytes(obj))
        print("  patched %s[%d] at +0x%x: %s -> width=%d offset=%d  (%s)"
              % (ns.instruction, ns.index, i - foff, cur["text"], w, o, ns.out))
        return
    if ns.cmd == "asm":
        b8, b9 = encode("tensor.bound." + ns.which, ns.width, ns.offset)
        print("byte8=0x%02x byte9=0x%02x" % (b8, b9))
        return
    obj = open(ns.path, "rb").read()
    foff, size = sections(obj)
    text = obj[foff:foff + size]
    n = m = 0
    for i in range(len(text) - MAC_UNIT + 1):
        r = decode(text[i:i + UNIT]) if i + UNIT <= len(text) else None
        if r:
            print("  +0x%-5x %s   %s" % (i, text[i:i + UNIT].hex(" "), r["text"]))
            n += 1
            continue
        r = decode_mac(text[i:i + MAC_UNIT])
        if r:
            m += 1
            if m <= 6:
                print("  +0x%-5x %s         %s" % (i, text[i:i + MAC_UNIT].hex(" "), r["text"]))
    print("  %d tensor-bound + %d tensor.mac instruction(s) in %d bytes of __text%s"
          % (n, m, size, "  (first 6 macs shown)" if m > 6 else ""))


if __name__ == "__main__":
    main()
