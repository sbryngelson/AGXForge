"""Measured six-byte encoders used by Apple's int32-to-byte narrowing sequence.

This module is deliberately a small measured boundary.  It does not add a generic authoring rule
for opcodes 11364, 11375, or 10369, and it does not generalize their fields beyond the retained
requantization class.  The constants below preserve the exact templates from the retained compiler
differential and expose only the register fields whose positions were independently measured.  The
fixed stage byte table is selected only by the explicit int32-to-byte IR primitive.

Provenance
----------
The maps and templates come from linker/g17-tensorops-recon at 02ffc5cf.  The source snapshots
used for this handoff are pinned by these SHA-256 values:

* isa/g17-operand-maps.jsonl: c881e7108b4d1922f040f4338208122522bf7c9a90caec8db22f5b514c02d959
* reqz_i8_chain2/result.json: 6526a008608310aef307f9cd3bdc9571389fb9ecde569847fdd36e0a0f784ce1
* reqz_kernelA_u8/result.json: 047afc46374c5e2d362d8f68e65e632e18a4a349da737d854cd2f7f38cbf2525
* reqz_round_probe_u8/result.json: 9d745fd704f53ccdc919ec516873bfd9ca9ffc1568cd6f4967539703a33991f3

The encoder accepts *printed decoder register numbers* (for example 107), rather than compiler
SSA numbers.  This is intentional: translating an allocated compiler register into the opcode's
register class is a separate integration decision and must not be smuggled into this field codec.
"""

from __future__ import annotations


PROVENANCE_COMMIT = "02ffc5cf"

# Exact six-byte Apple templates.  Bytes not covered by a register map remain inherited from the
# measured template; no unmeasured immediate or lifetime bit is synthesized here.
SIGNED_SELECT_TEMPLATE = bytes.fromhex("22052e0d0602")       # op11375
SIGNED_CLAMP_TEMPLATE = bytes.fromhex("32050efe1702")        # op11364
SIGNED_CLAMP_LOW_TEMPLATE = bytes.fromhex("0a8332e00502")    # op10369

UNSIGNED_CLAMP_LOW_TEMPLATE = bytes.fromhex("0a8132e00502")  # op10369
UNSIGNED_CLAMP_HIGH_TEMPLATE = bytes.fromhex("22052e800602") # op11364
UNSIGNED_CLAMP_FINAL_TEMPLATE = bytes.fromhex("22051efe1502") # op11364

# op3290/6 is the scale multiply in the retained constant-program requantization witness.
# Its generic authoring table describes a different six-byte family, so this form is deliberately
# a closed template codec.  The measured operation is two-address: the destination and operand 0
# are the same printed decoder register.  Operand 2 is the fixed expression [op0 + 12*4], and the
# remaining immediate/lifetime fields are fixed by the witness; callers may only choose that tied
# register.
FMUL6_TEMPLATE = bytes.fromhex("090d35018000")
FMUL6_REG_FIELDS = {
    0: (105, ((0, 0, 4, 0), (1, 0, 5, 0), (2, 0, 6, 0), (3, 0, 7, 0),
              (4, 2, 6, 0), (5, 2, 7, 0), (6, 5, 4, 0))),
    4: (105, ((0, 3, 1, 0), (1, 3, 2, 0), (2, 3, 3, 0), (3, 3, 4, 0),
              (4, 3, 5, 0), (5, 3, 6, 0), (6, 5, 2, 0))),
}

# The retained Apple scalar class is a complete, fixed instruction sequence.  These are the
# measured main-body bytes for the 256-element constant-preload witness.  The compiler may select
# this sequence only for the explicit stage primitive below; it must not generalize these bytes to
# an arbitrary csel, conversion, or store.  The sequence is deliberately kept as per-instruction
# records so the compiler can retain ordinary ABI/liveness facts while the unmeasured fields remain
# inherited from the exact witness.
REQUANT_STAGE_SIGNED = {
    "read_sr": bytes.fromhex("1ca01006"),
    "load": bytes.fromhex("0f000302182210c0410080000000"),
    "i32_to_f32": bytes.fromhex("2f0000022a80af026800"),
    "fmul": FMUL6_TEMPLATE,
    "rint": bytes.fromhex("270004182a20a0023000"),
    "narrow": bytes.fromhex("2700001a2a00ae027001"),
    "clamp_low": bytes.fromhex("02012e0f0602"),
    "clamp_high": bytes.fromhex("02010efe1702"),
    "store": bytes.fromhex("0f08030201061040"),
}
REQUANT_STAGE_UNSIGNED = {
    **REQUANT_STAGE_SIGNED,
    "clamp_low": bytes.fromhex("02012e800602"),
    "clamp_high": bytes.fromhex("02011efe1502"),
}


def stage_bytes(signed, step):
    """Return one exact byte string from the measured scalar stage."""
    table = REQUANT_STAGE_SIGNED if signed else REQUANT_STAGE_UNSIGNED
    try:
        return table[step]
    except KeyError as exc:
        raise ValueError("unknown measured requantization stage step %r" % (step,)) from exc


# Each tuple is (operand index, decoder-register base, ((value_bit, byte, bit, inverted), ...)).
# These are the measured register fields from the pinned operand-map snapshot.  The field value is
# decoder_register - base; it is not a compiler register number and it is not a slot value.
_REG_FIELDS = {
    11375: {
        0: (105, ((0, 0, 4, 0), (1, 0, 5, 0), (2, 0, 6, 0), (3, 0, 7, 0),
                  (4, 2, 6, 0), (5, 2, 7, 0), (6, 5, 4, 0))),
        3: (105, ((0, 1, 1, 0), (1, 1, 2, 0), (2, 1, 3, 0), (3, 1, 4, 0),
                  (4, 1, 5, 0), (5, 1, 6, 0), (6, 5, 0, 0))),
        7: (105, ((0, 1, 1, 0), (1, 1, 2, 0), (2, 1, 3, 0), (3, 1, 4, 0),
                  (4, 1, 5, 0), (5, 1, 6, 0), (6, 5, 0, 0))),
    },
    11364: {
        0: (105, ((0, 0, 4, 0), (1, 0, 5, 0), (2, 0, 6, 0), (3, 0, 7, 0),
                  (4, 2, 6, 0), (5, 2, 7, 0), (6, 5, 4, 0))),
        3: (105, ((0, 1, 1, 0), (1, 1, 2, 0), (2, 1, 3, 0), (3, 1, 4, 0),
                  (4, 1, 5, 0), (5, 1, 6, 0), (6, 5, 0, 0))),
        6: (105, ((0, 1, 1, 0), (1, 1, 2, 0), (2, 1, 3, 0), (3, 1, 4, 0),
                  (4, 1, 5, 0), (5, 1, 6, 0), (6, 5, 0, 0))),
    },
    10369: {
        0: (74, ((0, 0, 5, 0), (1, 0, 6, 0), (2, 0, 7, 0))),
        3: (105, ((0, 1, 1, 0), (1, 1, 2, 0), (2, 1, 3, 0), (3, 1, 4, 0),
                  (4, 1, 5, 0), (5, 1, 6, 0), (6, 5, 0, 0))),
    },
}


def _write_registers(opcode, template, registers):
    """Write only measured register operands onto *template*.

    ``registers`` maps operand indices to printed decoder register numbers.  A missing operand is
    left exactly as in the supplied template.  The function refuses a register outside the field's
    measured class and refuses a template of the wrong length, so a caller cannot accidentally use
    this as a generic six-byte opcode author.
    """
    if len(template) != 6:
        raise ValueError("the measured requant templates are exactly six bytes")
    fields = _REG_FIELDS.get(opcode)
    if fields is None:
        raise ValueError("no measured requant register map for opcode %d" % opcode)
    u = bytearray(template)
    occupied = {}
    for operand, printed in registers.items():
        if operand not in fields:
            raise ValueError("opcode %d has no measured register operand %d" % (opcode, operand))
        if type(printed) is not int:
            raise TypeError("decoder register must be an int")
        base, bits = fields[operand]
        value = printed - base
        width = max(j for j, _by, _bi, _inv in bits) + 1
        if value < 0 or value >= (1 << width):
            raise ValueError("decoder register %d is outside operand %d's measured class r%d..r%d"
                             % (printed, operand, base, base + (1 << width) - 1))
        for value_bit, by, bi, inverted in bits:
            bit = ((value >> value_bit) & 1) ^ inverted
            key = (by, bi)
            if key in occupied and occupied[key] != bit:
                raise ValueError("opcode %d register operands overlap with conflicting values at "
                                 "byte %d bit %d" % (opcode, by, bi))
            occupied[key] = bit
            u[by] = (u[by] & ~(1 << bi)) | (bit << bi)
    return bytes(u)


def encode_11375(registers, template=SIGNED_SELECT_TEMPLATE):
    """Encode measured op11375 register operands (normally 0, 3 and 7)."""
    return _write_registers(11375, template, registers)


def encode_11364(registers, template=SIGNED_CLAMP_TEMPLATE):
    """Encode measured op11364 register operands (normally 0, 3 and 6)."""
    return _write_registers(11364, template, registers)


def encode_10369(registers, template=SIGNED_CLAMP_LOW_TEMPLATE):
    """Encode measured op10369 register operands (normally 0 and 3)."""
    return _write_registers(10369, template, registers)


def encode_fmul6(template, defs, uses):
    """Encode the measured op3290/6 scale multiply.

    ``defs`` and ``uses`` are compiler register numbers (the same zero-based numbers used by the
    allocator); the byte template carries the corresponding printed decoder register, which is
    105 plus that number.  Only the measured tied destination/source form is admitted.
    """
    if template is None:
        template = FMUL6_TEMPLATE
    if bytes(template) != FMUL6_TEMPLATE:
        raise ValueError("op3290/6 requantization uses only the retained scale-multiply template")
    if len(defs) != 1 or len(uses) != 1:
        raise ValueError("op3290/6 requantization takes one tied destination and one source")
    if type(defs[0]) is not int or type(uses[0]) is not int:
        raise TypeError("op3290/6 requantization registers must be ints")
    if defs[0] != uses[0]:
        raise ValueError("op3290/6 requantization is measured only as a tied destination/source")
    reg = defs[0] + 105
    return _write_fmul6_register(reg)


def _write_fmul6_register(printed):
    if type(printed) is not int or not 105 <= printed <= 232:
        raise ValueError("op3290/6 requantization register must be in the measured decoder range r105..r232")
    u = bytearray(FMUL6_TEMPLATE)
    occupied = {}
    for operand, (base, bits) in FMUL6_REG_FIELDS.items():
        value = printed - base
        for value_bit, by, bi, inverted in bits:
            bit = ((value >> value_bit) & 1) ^ inverted
            key = (by, bi)
            if key in occupied and occupied[key] != bit:
                raise ValueError("op3290/6 tied register fields conflict")
            occupied[key] = bit
            u[by] = (u[by] & ~(1 << bi)) | (bit << bi)
    return bytes(u)


def decode_fmul6(encoded):
    """Decode the two measured register fields and verify the fixed template residue."""
    if len(encoded) != 6:
        raise ValueError("op3290/6 is exactly six bytes")
    u = bytearray(encoded)
    out = {}
    for operand, (base, bits) in FMUL6_REG_FIELDS.items():
        value = 0
        for value_bit, by, bi, inverted in bits:
            if (((u[by] >> bi) & 1) ^ inverted):
                value |= 1 << value_bit
        out[operand] = base + value
    # All non-register bits are inherited from the measured template.  This catches a caller that
    # accidentally changes the expression, control immediate, or lifetime while patching a register.
    mask = [0xFF] * 6
    for _base, bits in FMUL6_REG_FIELDS.values():
        for _value_bit, by, bi, _inv in bits:
            mask[by] &= ~(1 << bi)
    residue = bytes(u[i] & mask[i] for i in range(6))
    want = bytes(FMUL6_TEMPLATE[i] & mask[i] for i in range(6))
    if residue != want:
        raise ValueError("op3290/6 bytes do not preserve the measured scale-multiply template")
    if out[0] != out[4]:
        raise ValueError("op3290/6 decoded destination/source are not tied")
    return {"register": out[0], "dest": out[4], "src": out[0], "residue": residue}


def decode_measured_registers(opcode, encoded):
    """Decode only the register fields this module is allowed to write.

    This is a structural readback helper for tests.  It deliberately does not claim to decode the
    opcode's immediate/expr operands; those remain template-owned until their authoring maps are
    promoted into the checked-in ISA table.
    """
    if len(encoded) != 6:
        raise ValueError("the measured requant templates are exactly six bytes")
    out = {}
    for operand, (base, bits) in _REG_FIELDS.get(opcode, {}).items():
        value = 0
        for value_bit, by, bi, inverted in bits:
            if (((encoded[by] >> bi) & 1) ^ inverted):
                value |= 1 << value_bit
        out[operand] = base + value
    return out


__all__ = [
    "PROVENANCE_COMMIT",
    "SIGNED_SELECT_TEMPLATE", "SIGNED_CLAMP_TEMPLATE", "SIGNED_CLAMP_LOW_TEMPLATE",
    "UNSIGNED_CLAMP_LOW_TEMPLATE", "UNSIGNED_CLAMP_HIGH_TEMPLATE", "UNSIGNED_CLAMP_FINAL_TEMPLATE",
    "encode_11375", "encode_11364", "encode_10369", "decode_measured_registers",
    "FMUL6_TEMPLATE", "encode_fmul6", "decode_fmul6",
    "REQUANT_STAGE_SIGNED", "REQUANT_STAGE_UNSIGNED", "stage_bytes",
]
