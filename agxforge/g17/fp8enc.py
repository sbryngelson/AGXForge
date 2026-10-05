"""op17642/10, the fp8 unpack: two fp8 values in a 16-bit register become two bf16 values in a
32-bit register (recon section 137 part 5). Apple's compiler emits four of these per fp8 operand
fragment before a bf16 op5106 (seta-fp8-{e4m3,e5m2}-v1, compile only, decoded).

No authoring table describes this length. g17auth's record for op17642 comes from a 16-byte
witness, and applied to the 10-byte form it writes bytes that do not decode. So the fields here
are READ FROM APPLE'S DECODER. Each of the eight destination-field bits and nine source-field
bits was found by flipping it alone and watching which operand moved. The tables below are
built by enumerating every pattern of each field through the decoder and keeping the register
name it reads, so no bit arithmetic is assumed. The two formats are separate Apple templates,
because the format is not one mapped field (e4m3 and e5m2 differ in bytes 8 and 9). Every result
is decoded back, and the module reproduces Apple's sixteen instances byte for byte
(test_g17fp8enc).
"""
import itertools

from agxforge.g17 import model

TEMPLATE = {"e4m3": bytes.fromhex("af0000982500ac121002"),     # operand 2 = 97
            "e5m2": bytes.fromhex("af0000982500ac125000")}     # operand 2 = 98
FORMAT_CODE = {"e4m3": 97, "e5m2": 98}
DEST_BITS = ((0, 4), (0, 7), (2, 3), (2, 4), (2, 7), (7, 3), (7, 4), (7, 5))
SRC_BITS = ((1, 0), (1, 1), (3, 5), (3, 6), (3, 7), (5, 7), (8, 0), (8, 1), (8, 2))
# operand 1's scoreboard wait bits: slot s is 1 << (24 + s); these are where the decoder reads them
WAIT_BITS = {0: (1, 2), 1: (1, 3), 2: (1, 4), 3: (1, 5), 4: (1, 6), 5: (7, 7), 6: (2, 6), 7: (0, 3)}
_NAMES = model.registers()


def _apply(u, bits, pattern):
    for i, (byte, bit) in enumerate(bits):
        if pattern >> i & 1:
            u[byte] |= 1 << bit
        else:
            u[byte] &= ~(1 << bit) & 0xFF


def _table(bits, operand):
    out = {}
    for pattern in range(1 << len(bits)):
        u = bytearray(TEMPLATE["e4m3"])
        _apply(u, bits, pattern)
        d = list(model.decode(bytes(u), 0))
        if d and d[0].opcode and d[0].opcode.id == 17642 and len(d[0].raw) == 10:
            name = _NAMES.get(d[0].values[operand][1])
            if name is not None:
                out.setdefault(name, pattern)
    return out


_DEST, _SRC = None, None


def unpack(dest, src_half, fmt="e4m3", wait_slot=None, release=True):
    """R<dest> <- two bf16 from the two fp8 in `src_half` ('R8L', 'R8H', ...). wait_slot: the
    scoreboard slot this instruction waits on (the fragment's load), or None."""
    global _DEST, _SRC
    if _DEST is None:
        _DEST, _SRC = _table(DEST_BITS, 0), _table(SRC_BITS, 3)
    want_d, want_s = "R%d" % dest, src_half
    if want_d not in _DEST or want_s not in _SRC:
        raise ValueError("op17642/10 cannot name %s <- %s" % (want_d, want_s))
    u = bytearray(TEMPLATE[fmt])
    _apply(u, DEST_BITS, _DEST[want_d])
    _apply(u, SRC_BITS, _SRC[want_s])
    for slot, (byte, bit) in WAIT_BITS.items():
        u[byte] &= ~(1 << bit) & 0xFF
    if wait_slot is not None:
        byte, bit = WAIT_BITS[wait_slot]
        u[byte] |= 1 << bit
    if not release:
        raise ValueError("refused: only the released source (Apple's 16) is measured for op17642/10")
    b = bytes(u)
    d = list(model.decode(b, 0))
    got = [_NAMES.get(v) for k, v in d[0].values if k == "reg"] if d and d[0].opcode else None
    flags = d[0].values[1][1] if got else None
    if (got != [want_d, want_s] or d[0].values[2][1] != FORMAT_CODE[fmt] or
            (flags >> 24) & 0xFF != (0 if wait_slot is None else 1 << wait_slot)):
        raise ValueError("op17642/10 does not decode as %s <- %s (%s): %s" % (want_d, want_s, fmt, b.hex()))
    return b
