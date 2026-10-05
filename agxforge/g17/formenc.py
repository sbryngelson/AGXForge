"""Production encoders for the float and immediate forms this backend lowers.

ONE MODULE, SIX FORMS, AND THE PROBES LEFT BEHIND. Each of these lived in a tools/ module that also
held its witness builders and corpus diagnostics - and those halves imported g17cc for
compile_function, while g17cc imported the encoding half for BASE/encode/decode. The compiler and
its own form tables were mutually dependent, and the two directions were narrow and opposite. That
is the seam: 250 of the 2,709 lines in those six modules were on this side of it.

THE SHARED _get/_put ARE PROVEN EQUIVALENT, NOT ASSUMED. Six copies existed, differing in parameter
names, expression spelling and docstrings - and one was NOT equivalent. g17fmul4._put wrote the
whole field, clearing a carrier whose value bit is 0, while the other five only ever set bits. A
randomised sweep over carrier layouts found the divergence in 4,000 trials, after the six had been
read as identical by eye. They agree for these six forms only because no form's BASE carries a 1 in
a carrier position, which was checked per form rather than argued. The write-whole version is kept:
it is correct when a base does carry such a bit, and writing one carrier instead of a whole operand
is how op11666 once emitted a value Apple never writes.

Refused is defined ONCE here. Emitting a per-form copy gives the encoders one exception class and
each legacy module's refusal harness another, so every guard reads as ACCEPTED - which is what the
fmul4 suite caught on the first attempt at this.

Per-form constants keep their own names, prefixed, because they are not shared: REG_MAX is six bits
for one form and four for another, and collapsing them would change what each refuses.
"""


class Refused(Exception):
    """A shape these encoders will not write. Every one names the measurement it lacks."""


def _get(buf, carriers):
    """Read a scattered field: carrier i supplies bit i of the value."""
    return sum(((buf[by] >> pos) & 1) << i for i, (by, pos) in enumerate(carriers))


def _put(buf, carriers, value):
    """Write a scattered field WHOLE, clearing carriers whose value bit is zero."""
    for i, (by, pos) in enumerate(carriers):
        buf[by] = (buf[by] & ~(1 << pos)) | (((value >> i) & 1) << pos)


# ---------------------------------------------------------------- Fadd4 (op998/4)
# from tools/g17fadd4.py; its witness builders and diagnostics stay there.
_Fadd4_OPCODE, _Fadd4_LENGTH = 998, 4
_Fadd4_BASE = bytes.fromhex("09010001")
_Fadd4_REG_BASE, _Fadd4_REG_MAX = 105, 63
_Fadd4_DEST = [(0, 4), (0, 5), (0, 6), (0, 7), (2, 6), (2, 7)]
_Fadd4_SRC0 = [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6)]
_Fadd4_SRC1 = [(3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6)]
_Fadd4_DEST_LIFE = {32: (2, 5)}
_Fadd4_SRC0_LIFE = {16: (2, 3), 32: (1, 7)}
_Fadd4_SRC1_LIFE = {16: (2, 4), 32: (3, 7)}
_Fadd4_LIFETIMES = (0, 16, 32)
_Fadd4_ALL_FIELDS = (_Fadd4_DEST + _Fadd4_SRC0 + _Fadd4_SRC1 + list(_Fadd4_DEST_LIFE.values())
              + list(_Fadd4_SRC0_LIFE.values()) + list(_Fadd4_SRC1_LIFE.values()))
def _Fadd4_encode(dest, src0, src1, dest_life=32, src0_life=16, src1_life=16):
    """Write one op998/4. Registers are RELATIVE to the 105 base."""
    for name, r in (("destination", dest), ("source 0", src0), ("source 1", src1)):
        if not isinstance(r, int) or isinstance(r, bool):
            raise Refused("%s is %r, not a register index" % (name, r))
        if not 0 <= r <= _Fadd4_REG_MAX:
            raise Refused("%s r%d is outside the six-bit 105-based field this form carries "
                          "(0..%d); it is not truncated" % (name, r, _Fadd4_REG_MAX))
    if dest_life not in (0, 32):
        raise Refused("destination lifetime %r: this form's only carrier is the 32 bit at b2[5], "
                      "and the corpus shows 32 and 0 and nothing else" % (dest_life,))
    for name, life in (("source 0", src0_life), ("source 1", src1_life)):
        if life not in _Fadd4_LIFETIMES:
            raise Refused("%s lifetime %r is not one this population shows (%s)"
                          % (name, life, list(_Fadd4_LIFETIMES)))
    u = bytearray(_Fadd4_BASE)
    _put(u, _Fadd4_DEST, dest); _put(u, _Fadd4_SRC0, src0); _put(u, _Fadd4_SRC1, src1)
    for table, life in ((_Fadd4_DEST_LIFE, dest_life), (_Fadd4_SRC0_LIFE, src0_life), (_Fadd4_SRC1_LIFE, src1_life)):
        for value, (by, pos) in table.items():
            if life & value:
                u[by] |= 1 << pos
    return bytes(u)
def _Fadd4_decode(b):
    if len(b) != _Fadd4_LENGTH:
        raise Refused("op%d/%d is %d bytes; got %d" % (_Fadd4_OPCODE, _Fadd4_LENGTH, _Fadd4_LENGTH, len(b)))
    u = bytearray(b)
    out = dict(dest=_get(u, _Fadd4_DEST), src0=_get(u, _Fadd4_SRC0), src1=_get(u, _Fadd4_SRC1))
    for name, table in (("dest_life", _Fadd4_DEST_LIFE), ("src0_life", _Fadd4_SRC0_LIFE),
                        ("src1_life", _Fadd4_SRC1_LIFE)):
        out[name] = sum(v for v, (by, pos) in table.items() if (u[by] >> pos) & 1)
    mask = [0xFF] * _Fadd4_LENGTH
    for by, pos in _Fadd4_ALL_FIELDS:
        mask[by] &= ~(1 << pos)
    out["residue"] = bytes(b[i] & mask[i] for i in range(_Fadd4_LENGTH))
    return out


# ---------------------------------------------------------------- Fadd6 (op998/6)
# from tools/g17fadd6.py; its witness builders and diagnostics stay there.
_Fadd6_OPCODE, _Fadd6_LENGTH = 998, 6
_Fadd6_BASE = bytes.fromhex("090104010000")
_Fadd6_REG_BASE, _Fadd6_REG_MAX = 105, 127
_Fadd6_DEST = [(0, 4), (0, 5), (0, 6), (0, 7), (2, 6), (2, 7), (5, 4)]
_Fadd6_SRC0 = [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (5, 0)]
_Fadd6_SRC1 = [(3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (5, 2)]
_Fadd6_DEST_LIFE = {32: (2, 5)}
_Fadd6_SRC0_LIFE = {16: (2, 3), 32: (1, 7)}
_Fadd6_SRC1_LIFE = {2: (5, 3), 16: (2, 4), 32: (3, 7)}
_Fadd6_INDEX = [(5, 5), (5, 6), (5, 7)]
_Fadd6_INDEX_TABLE = {(0, 0, 0): 0, (1, 0, 0): 1 << 24, (0, 1, 0): 1 << 25,
               (1, 1, 0): 1 << 26, (0, 0, 1): 1 << 27, (1, 0, 1): 1 << 28}
_Fadd6_SRC0_LIFETIMES = (0, 16, 32)
_Fadd6_SRC1_LIFETIMES = (0, 2, 16, 18, 32, 34)
_Fadd6_ALL_FIELDS = (_Fadd6_DEST + _Fadd6_SRC0 + _Fadd6_SRC1 + _Fadd6_INDEX + list(_Fadd6_DEST_LIFE.values())
              + list(_Fadd6_SRC0_LIFE.values()) + list(_Fadd6_SRC1_LIFE.values()))
def _Fadd6_encode(dest, src0, src1, dest_life=32, src0_life=16, src1_life=16, index=0):
    """Write one op998/6. Registers are RELATIVE to the 105 base."""
    for name, r in (("destination", dest), ("source 0", src0), ("source 1", src1)):
        if not isinstance(r, int) or isinstance(r, bool):
            raise Refused("%s is %r, not a register index" % (name, r))
        if not 0 <= r <= _Fadd6_REG_MAX:
            raise Refused("%s r%d is outside the seven-bit 105-based field (0..%d)"
                          % (name, r, _Fadd6_REG_MAX))
    if dest_life not in (0, 32):
        raise Refused("destination lifetime %r: this form's only carrier is the 32 bit at b2[5]"
                      % (dest_life,))
    if src0_life not in _Fadd6_SRC0_LIFETIMES:
        raise Refused("source 0 lifetime %r is not one this population shows (%s); note source 0 "
                      "has NO negate carrier, only source 1 does"
                      % (src0_life, list(_Fadd6_SRC0_LIFETIMES)))
    if src1_life not in _Fadd6_SRC1_LIFETIMES:
        raise Refused("source 1 lifetime %r is not one this population shows (%s)"
                      % (src1_life, list(_Fadd6_SRC1_LIFETIMES)))
    if index != 0:
        raise Refused("index %r: this encoder writes the ZERO state only. The index is a "
                      "three-bit code assigned when the scheduler allocates a slot - ascending "
                      "distinct within a program - and Apple leaves it zero on 1,695 of 1,875 "
                      "rows. Choosing a non-zero slot would be inventing a schedule rather than "
                      "compiling a program." % (index,))
    u = bytearray(_Fadd6_BASE)
    _put(u, _Fadd6_DEST, dest); _put(u, _Fadd6_SRC0, src0); _put(u, _Fadd6_SRC1, src1)
    for table, life in ((_Fadd6_DEST_LIFE, dest_life), (_Fadd6_SRC0_LIFE, src0_life), (_Fadd6_SRC1_LIFE, src1_life)):
        for value, (by, pos) in table.items():
            if life & value:
                u[by] |= 1 << pos
    return bytes(u)
def _Fadd6_decode(b):
    if len(b) != _Fadd6_LENGTH:
        raise Refused("op%d/%d is %d bytes; got %d" % (_Fadd6_OPCODE, _Fadd6_LENGTH, _Fadd6_LENGTH, len(b)))
    key = tuple((b[by] >> pos) & 1 for by, pos in _Fadd6_INDEX)
    if key not in _Fadd6_INDEX_TABLE:
        raise Refused("the index code is %s, which is not one of the six states this corpus "
                      "ships (%s)" % (key, sorted(_Fadd6_INDEX_TABLE)))
    mask = [0xFF] * _Fadd6_LENGTH
    for by, pos in _Fadd6_ALL_FIELDS:
        mask[by] &= ~(1 << pos)
    return dict(dest=_get(b, _Fadd6_DEST), src0=_get(b, _Fadd6_SRC0), src1=_get(b, _Fadd6_SRC1),
                dest_life=sum(v for v, (by, p) in _Fadd6_DEST_LIFE.items() if (b[by] >> p) & 1),
                src0_life=sum(v for v, (by, p) in _Fadd6_SRC0_LIFE.items() if (b[by] >> p) & 1),
                src1_life=sum(v for v, (by, p) in _Fadd6_SRC1_LIFE.items() if (b[by] >> p) & 1),
                index=_Fadd6_INDEX_TABLE[key],
                residue=bytes(b[i] & mask[i] for i in range(_Fadd6_LENGTH)))


# ---------------------------------------------------------------- Ffma4 (op2190/4)
# from tools/g17ffma4.py; its witness builders and diagnostics stay there.
_Ffma4_OPCODE, _Ffma4_LENGTH = 2190, 4
_Ffma4_BASE = bytes.fromhex("09010200")
_Ffma4_DEST = ((0, 4), (0, 5), (0, 6), (0, 7), (2, 6), (2, 7))
_Ffma4_SRC0 = ((1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6))
_Ffma4_SRCN = ((3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6))
_Ffma4_REG_BASE, _Ffma4_REG_MAX = 105, 63   # the same convention g17fmul4 uses: _Ffma4_encode/_Ffma4_decode speak 0-based
_Ffma4_MODE_ACC_AT_4 = (2, 0)
_Ffma4_MODE_ACC_AT_6 = (3, 0)
def _Ffma4_encode(dest, src0, other, accumulator_printed_at_6,
           dest_life=32, src0_life=16, other_life=16):
    """The four bytes. `other` is the source that is NOT the accumulator; the accumulator is the
    destination and is not passed, because it is not independently encoded.

    Registers are 0-BASED allocator indices, as g17fmul4.encode's are - the compiler hands those
    through, and 105 is added only when a register is named in a message."""
    for name, reg in (("dest", dest), ("src0", src0), ("other", other)):
        if not 0 <= reg <= _Ffma4_REG_MAX:
            raise Refused("%s r%d: the field is six bits, so this form reaches r%d..r%d"
                          % (name, _Ffma4_REG_BASE + reg, _Ffma4_REG_BASE, _Ffma4_REG_BASE + _Ffma4_REG_MAX))
    for name, life in (("dest", dest_life), ("src0", src0_life), ("other", other_life)):
        if life not in (0, 16, 32):
            raise Refused("%s lifetime %d is not one of the three values this population shows "
                          "(0, 16, 32)" % (name, life))
    buf = bytearray(_Ffma4_BASE)
    _put(buf, _Ffma4_DEST, dest)
    _put(buf, _Ffma4_SRC0, src0)
    _put(buf, _Ffma4_SRCN, other)
    byte, bit = _Ffma4_MODE_ACC_AT_6 if accumulator_printed_at_6 else _Ffma4_MODE_ACC_AT_4
    buf[byte] |= 1 << bit
    if dest_life & 32:
        buf[2] |= 1 << 5
    if src0_life & 16:
        buf[2] |= 1 << 3
    if src0_life & 32:
        buf[1] |= 1 << 7
    if other_life & 16:
        buf[2] |= 1 << 4
    if other_life & 32:
        buf[3] |= 1 << 7
    return bytes(buf)
def _Ffma4_decode(b):
    """The inverse, for the compiler's read-back check. It returns the RESIDUE too: an instruction
    whose non-field bits are not Apple's base is not this form however its operands read - all 653
    corpus instances share one residue, so a different one means a different instruction wearing
    the same operands. g17fmul4.decode makes the same argument for op3290/4."""
    if len(b) != _Ffma4_LENGTH:
        raise Refused("op%d is being read at %d bytes; this reading is the four-byte form only"
                      % (_Ffma4_OPCODE, len(b)))
    u = bytearray(b)
    at4, at6 = (u[_Ffma4_MODE_ACC_AT_4[0]] >> _Ffma4_MODE_ACC_AT_4[1]) & 1, (u[_Ffma4_MODE_ACC_AT_6[0]] >> _Ffma4_MODE_ACC_AT_6[1]) & 1
    if at4 == at6:
        raise Refused("the mode pair reads (%d,%d); only (1,0) and (0,1) are witnessed, and "
                      "nothing here establishes what a third combination means" % (at4, at6))
    out = dict(dest=_get(u, _Ffma4_DEST), src0=_get(u, _Ffma4_SRC0), other=_get(u, _Ffma4_SRCN),
               accumulator_printed_at_6=bool(at6),
               dest_life=32 if (u[2] >> 5) & 1 else 0,
               src0_life=(16 if (u[2] >> 3) & 1 else 0) + (32 if (u[1] >> 7) & 1 else 0),
               other_life=(16 if (u[2] >> 4) & 1 else 0) + (32 if (u[3] >> 7) & 1 else 0))
    mask = [0xFF] * _Ffma4_LENGTH
    for by, pos in _Ffma4_DEST + _Ffma4_SRCN + _Ffma4_SRC0:
        mask[by] &= ~(1 << pos)
    for by, pos in (_Ffma4_MODE_ACC_AT_4, _Ffma4_MODE_ACC_AT_6, (2, 5), (2, 3), (1, 7), (2, 4), (3, 7)):
        mask[by] &= ~(1 << pos)
    out["residue"] = bytes(b[i] & mask[i] for i in range(_Ffma4_LENGTH))
    return out


# ---------------------------------------------------------------- Ffma6 (op2190/6)
# from tools/g17ffma6.py; its witness builders and diagnostics stay there.
_Ffma6_OPCODE, _Ffma6_LENGTH = 2190, 6
_Ffma6_BASE = bytes.fromhex("090106010000")
_Ffma6_DEST = ((0, 4), (0, 5), (0, 6), (0, 7), (2, 6), (2, 7))
_Ffma6_SRC0 = ((1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6))
_Ffma6_SRC1 = ((3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6))
_Ffma6_SRC2 = ((5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6))
_Ffma6_LIFETIME = {"dest": {32: (2, 5)},
            "src0": {16: (2, 3), 32: (1, 7)},
            "src1": {16: (2, 4), 32: (3, 7)},
            "src2": {16: (4, 7), 32: (5, 7)}}
_Ffma6_REG_BASE, _Ffma6_REG_MAX = 105, 63
def _Ffma6_encode(dest, src0, src1, src2, dest_life=32, src0_life=16, src1_life=16, src2_life=16):
    """r<dest> = fma(r<src0>, r<src1>, r<src2>), six bytes. Registers are 0-BASED allocator indices,
    the convention g17fmul4 and g17ffma4 use and the compiler hands through."""
    for name, reg in (("dest", dest), ("src0", src0), ("src1", src1), ("src2", src2)):
        if not 0 <= reg <= _Ffma6_REG_MAX:
            raise Refused("%s r%d: the field is six bits, so this form reaches r%d..r%d"
                          % (name, _Ffma6_REG_BASE + reg, _Ffma6_REG_BASE, _Ffma6_REG_BASE + _Ffma6_REG_MAX))
    lives = dict(dest=dest_life, src0=src0_life, src1=src1_life, src2=src2_life)
    for name, life in lives.items():
        allowed = set(_Ffma6_LIFETIME[name]) | {0}
        if life not in allowed:
            raise Refused("%s lifetime %r: this form's carriers express %s, and 0 is the value "
                          "Apple's population also holds; nothing else is located"
                          % (name, life, sorted(allowed)))
    buf = bytearray(_Ffma6_BASE)
    _put(buf, _Ffma6_DEST, dest)
    _put(buf, _Ffma6_SRC0, src0)
    _put(buf, _Ffma6_SRC1, src1)
    _put(buf, _Ffma6_SRC2, src2)
    for name, life in lives.items():
        for value, (byte, bit) in _Ffma6_LIFETIME[name].items():
            if life == value:
                buf[byte] |= 1 << bit
    return bytes(buf)
def _Ffma6_decode(b):
    """The inverse, with the RESIDUE, for the compiler's read-back check: an instruction whose
    non-field bits are not Apple's base is not this form however its operands read."""
    if len(b) != _Ffma6_LENGTH:
        raise Refused("op%d is being read at %d bytes; this reading is the six-byte form only"
                      % (_Ffma6_OPCODE, len(b)))
    u = bytearray(b)
    out = dict(dest=_get(u, _Ffma6_DEST), src0=_get(u, _Ffma6_SRC0), src1=_get(u, _Ffma6_SRC1), src2=_get(u, _Ffma6_SRC2))
    for name in ("dest", "src0", "src1", "src2"):
        out[name + "_life"] = sum(v for v, (by, pos) in _Ffma6_LIFETIME[name].items()
                                  if (u[by] >> pos) & 1)
    mask = [0xFF] * _Ffma6_LENGTH
    for by, pos in _Ffma6_DEST + _Ffma6_SRC0 + _Ffma6_SRC1 + _Ffma6_SRC2:
        mask[by] &= ~(1 << pos)
    for d in _Ffma6_LIFETIME.values():
        for by, pos in d.values():
            mask[by] &= ~(1 << pos)
    out["residue"] = bytes(b[i] & mask[i] for i in range(_Ffma6_LENGTH))
    return out


# ---------------------------------------------------------------- Fmul4 (op3290/4)
# from tools/g17fmul4.py; its witness builders and diagnostics stay there.
_Fmul4_OPCODE, _Fmul4_LENGTH = 3290, 4
_Fmul4_BASE = bytes.fromhex("09010101")
_Fmul4_DEST = ((0, 4), (0, 5), (0, 6), (0, 7), (2, 6), (2, 7))
_Fmul4_SRC0 = ((1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6))
_Fmul4_SRC1 = ((3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6))
_Fmul4_LIFETIME = {"dest": {32: (2, 5)}, "src0": {16: (2, 3), 32: (1, 7)}, "src1": {16: (2, 4), 32: (3, 7)}}
_Fmul4_REG_BASE = 105
_Fmul4_REG_MAX = 63                  # the field is six bits, so 105..168 - and 0..63 is what the corpus uses
def _Fmul4_encode(dest, src0, src1, dest_life=32, src0_life=16, src1_life=16):
    """r<dest> = r<src0> * r<src1>, four bytes, with each operand's lifetime WRITTEN.

    The lifetimes are operands, not decoration: memory:g17-modifier-operand-lifetimes records four
    separate occasions on which inheriting one from a template made an authored program read zero.
    """
    for name, v in (("dest", dest), ("src0", src0), ("src1", src1)):
        if not 0 <= v <= _Fmul4_REG_MAX:
            raise Refused("%s r%d: the field is six bits, so this form reaches r%d..r%d"
                          % (name, _Fmul4_REG_BASE + v, _Fmul4_REG_BASE, _Fmul4_REG_BASE + _Fmul4_REG_MAX))
    for name, v in (("dest", dest_life), ("src0", src0_life), ("src1", src1_life)):
        allowed = set(_Fmul4_LIFETIME[name]) | {0}
        if v not in allowed:
            raise Refused("%s lifetime %r: this form's carriers express %s, and 0 is the value "
                          "Apple's population also holds; nothing else is located"
                          % (name, v, sorted(allowed)))
    u = bytearray(_Fmul4_BASE)
    _put(u, _Fmul4_DEST, dest); _put(u, _Fmul4_SRC0, src0); _put(u, _Fmul4_SRC1, src1)
    for name, val in (("dest", dest_life), ("src0", src0_life), ("src1", src1_life)):
        for v, (by, pos) in _Fmul4_LIFETIME[name].items():
            u[by] = (u[by] & ~(1 << pos)) | ((1 if val == v else 0) << pos)
    return bytes(u)
def _Fmul4_decode(b):
    if len(b) != _Fmul4_LENGTH:
        raise Refused("op%d is being read at %d bytes; this reading is the four-byte form only"
                      % (_Fmul4_OPCODE, len(b)))
    u = bytearray(b)
    out = dict(dest=_get(u, _Fmul4_DEST), src0=_get(u, _Fmul4_SRC0), src1=_get(u, _Fmul4_SRC1))
    for name in ("dest", "src0", "src1"):
        out[name + "_life"] = sum(v for v, (by, pos) in _Fmul4_LIFETIME[name].items() if (u[by] >> pos) & 1)
    mask = [0xFF] * 4
    for by, pos in _Fmul4_DEST + _Fmul4_SRC0 + _Fmul4_SRC1: mask[by] &= ~(1 << pos)
    for d in _Fmul4_LIFETIME.values():
        for by, pos in d.values(): mask[by] &= ~(1 << pos)
    out["residue"] = bytes(b[i] & mask[i] for i in range(4))
    return out


# ---------------------------------------------------------------- Movimm2 (op11842/2)
# from tools/g17movimm2.py; its witness builders and diagnostics stay there.
_Movimm2_OPCODE, _Movimm2_LENGTH = 11842, 2
_Movimm2_BASE = bytes.fromhex("0400")
_Movimm2_REG_BASE, _Movimm2_REG_MAX = 105, 15          # four bits: r105..r120
_Movimm2_IMM_MAX = 127                        # seven bits, and b1[7] is REFUSED by the decoder
_Movimm2_DEST = [(0, 4), (0, 5), (0, 6), (0, 7)]
_Movimm2_IMM = [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6)]
_Movimm2_ALL_FIELDS = _Movimm2_DEST + _Movimm2_IMM
def _Movimm2_encode(dest, imm):
    """Write one op11842/2. `dest` is RELATIVE to the 105 base."""
    if not isinstance(dest, int) or isinstance(dest, bool):
        raise Refused("destination is %r, not a register index" % (dest,))
    if not 0 <= dest <= _Movimm2_REG_MAX:
        raise Refused("destination r%d is outside this form's FOUR-bit field (r%d..r%d absolute). "
                      "That is an ALLOCATION constraint, not an encoding one, and it is refused "
                      "rather than truncated" % (dest + _Movimm2_REG_BASE, _Movimm2_REG_BASE, _Movimm2_REG_BASE + _Movimm2_REG_MAX))
    if not isinstance(imm, int) or isinstance(imm, bool):
        raise Refused("immediate is %r, not an integer" % (imm,))
    if not 0 <= imm <= _Movimm2_IMM_MAX:
        raise Refused("immediate %d does not fit this form's SEVEN bits (0..%d). The eighth bit "
                      "b1[7] is not a wider immediate: Apple's decoder REFUSES the instruction "
                      "with it set, on 16 of 16 distinct encodings" % (imm, _Movimm2_IMM_MAX))
    u = bytearray(_Movimm2_BASE)
    _put(u, _Movimm2_DEST, dest); _put(u, _Movimm2_IMM, imm)
    return bytes(u)
def _Movimm2_decode(b):
    if len(b) != _Movimm2_LENGTH:
        raise Refused("op%d/%d is %d bytes; got %d" % (_Movimm2_OPCODE, _Movimm2_LENGTH, _Movimm2_LENGTH, len(b)))
    u = bytearray(b)
    mask = [0xFF] * _Movimm2_LENGTH
    for by, pos in _Movimm2_ALL_FIELDS:
        mask[by] &= ~(1 << pos)
    return dict(dest=_get(u, _Movimm2_DEST), imm=_get(u, _Movimm2_IMM),
                residue=bytes(b[i] & mask[i] for i in range(_Movimm2_LENGTH)))


# BITWISE REGISTER TEMPLATES, moved out of g17cc because they are encoding data rather than
# compiler behaviour. g17const.inventory() needed exactly this one name from the compiler, which
# was the whole of its edge back into g17cc; the library owns encoding constants, so the edge goes
# away rather than being carried. g17cc re-exports it, so its four other readers are unaffected.
BITWISE_REG_TEMPLATE = {424: bytes.fromhex("0b831201"), 13575: bytes.fromhex("1b851403"),
                        17771: bytes.fromhex("2b0d1b05")}


class _Form(object):
    """One form's production surface, so callers ask for a form rather than a global."""

    def __init__(self, base, encode, decode):
        self.BASE, self.encode, self.decode = base, encode, decode


Fadd4 = _Form(_Fadd4_BASE, _Fadd4_encode, _Fadd4_decode)
Fadd6 = _Form(_Fadd6_BASE, _Fadd6_encode, _Fadd6_decode)
Ffma4 = _Form(_Ffma4_BASE, _Ffma4_encode, _Ffma4_decode)
Ffma6 = _Form(_Ffma6_BASE, _Ffma6_encode, _Ffma6_decode)
Fmul4 = _Form(_Fmul4_BASE, _Fmul4_encode, _Fmul4_decode)
Movimm2 = _Form(_Movimm2_BASE, _Movimm2_encode, _Movimm2_decode)
