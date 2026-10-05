"""Physical-register encoders for a tensor body's register epilogue (Set A item 6).

The three forms are the ones cc.py emits for the same scalar operations, taken byte for byte from
its output for `fmul`, `fmax` and `const` (fmul op3290/14, fmax op9700/14 with operation 7,
movimm op11842/8). Only register and lifetime operands are rewritten, through the same
`g17auth.encode` path cc uses. Every other bit, the scoreboard word included, stays the
template's. Each result is decoded back and refused unless Apple's decoder reads the intended
registers.

Lifetimes: 16 releases a source after the read, 32 keeps it live. The constants are read by
every tile, so they are kept. An accumulator updated in place is released, exactly as tlower's
accumulate fadd does.
"""
from agxforge.g17 import auth as g17auth, asm as g17asm, model

FMUL = (3290, bytes.fromhex("210205100280a02a1c0820000210"))
FMAX = (9700, bytes.fromhex("3200072a2380a0223608a1640044"))
MOVIMM = (11842, bytes.fromhex("0c80423e60200008"))
# cc's own exp2 (op1272/10): operand 1 is 32 in every instance cc emits, whether the source came
# from a load or an ALU op, so it is a constant of the form; operand 3 is the source lifetime.
EXP2 = (1272, bytes.fromhex("2702042a2a20a51a3000"))
# cc's own recip (op3658/10) and fadd (op998/12), taken from its compile of the memory-stage GELU
# (tensorreduce.emit_row_gelu): recip is exp2's shape (operand 1 fixed at 32, operand 3 the source
# lifetime); fadd's operand 1 is the template's scoreboard word, operands 3 and 5 the lifetimes.
RECIP = (3658, bytes.fromhex("b70004b82a20a10a3000"))
FADD = (998, bytes.fromhex("8102049a0220a00a64041200"))
SHUFFLE_XOR = (14169, bytes.fromhex("2780017a2800a922b800"))     # cc's own simd_shuffle_xor, mask 2
RELEASE, KEEP = 16, 32
_NAMES = model.registers()


def _encode(form, dest, sources, lifetimes, imms=None):
    opc, template = form
    dsts, srcs = g17auth.register_operands(opc)
    if len(sources) != len(srcs):
        raise ValueError("op%d takes %d register sources, got %d" % (opc, len(srcs), len(sources)))
    vals = {dsts[0]: g17auth.field_value(opc, dsts[0], dest)}
    for index, register in zip(srcs, sources):
        vals[index] = g17auth.field_value(opc, index, register)
    vals.update(lifetimes)
    vals.update(imms or {})
    # g17auth pads its result; the instruction is the template's length.
    b = bytes(g17auth.encode(opc, vals, template=template, trusted=tuple(lifetimes) + tuple(imms or ())))[:len(template)]
    ins = list(model.decode(b, 0))
    if not ins or ins[0].opcode is None or ins[0].opcode.id != opc or len(ins[0].raw) != len(template):
        raise ValueError("op%d: %s does not decode as itself" % (opc, b.hex()))
    got = [_NAMES.get(v) for k, v in ins[0].values if k == "reg"]
    want = ["R%d" % r for r in [dest] + list(sources)]
    if got != want:
        raise ValueError("op%d decodes registers %s, not %s" % (opc, got, want))
    return bytes(b)


def fmul(dest, a, b, keep_a=False, keep_b=True):
    """dest = a * b (op3290/14). Operands 3 and 5 are the lifetimes of a and b."""
    return _encode(FMUL, dest, [a, b], {3: KEEP if keep_a else RELEASE, 5: KEEP if keep_b else RELEASE})


def fmax(dest, a, b, keep_a=False, keep_b=True):
    """dest = max(a, b) (op9700/14, operation 7). Apple compares a with b and selects from the same pair."""
    la, lb = KEEP if keep_a else RELEASE, KEEP if keep_b else RELEASE
    # The pair is read twice, as compared values and as choices, and cc writes one lifetime at
    # both positions of an operand: a multi-use constant is 32 at 6 and 10, a released value 16 at
    # 4 and 8 (cc's own output for fmax of a reused zero).
    return _encode(FMAX, dest, [a, b, a, b], {4: la, 6: lb, 8: la, 10: lb})


def movimm(dest, value):
    """dest = the 32-bit pattern `value` (op11842/8), through cc's own movimm.8 encoder: the
    8-byte form's field layout is not the one g17auth maps (that record is a 10-byte witness)."""
    b = bytes(g17asm.encode_movimm(imm=value & 0xFFFFFFFF, template=MOVIMM[1], dest=dest))
    ins = list(model.decode(b, 0))
    if not ins or ins[0].opcode is None or ins[0].opcode.id != MOVIMM[0] or len(ins[0].raw) != 8:
        raise ValueError("movimm R%d = %#x does not decode as itself" % (dest, value))
    got = [_NAMES.get(v) for k, v in ins[0].values if k == "reg"]
    imm = [v for k, v in ins[0].values if k == "imm"][-1]
    if got != ["R%d" % dest] or imm != value & 0xFFFFFFFF:
        raise ValueError("movimm decodes %s = %#x, not R%d = %#x" % (got, imm, dest, value))
    return b


def cvt_f32_to_f16(dest_half, src):
    """R<dest_half> (a 16-bit half such as 'R12L') <- f16(R<src>), RNE: cc's own 12-byte cvt.f32.f16
    (op1016/12, the measured narrowing), releasing the source. Apple's register-direct chain uses
    op1016 for exactly this step (MMA, eight conversions, MMA; docs/archive/g17-tensor-register-chain.md)."""
    from agxforge.g17 import cc, assembler as g17as
    base = int(dest_half[1:-1]); hi = dest_half.endswith("H")
    dest_id = (281 if hi else 425) + base          # GPR16 numbering: low halves 425 + r, high halves 281 + r
    line = cc._as_line("cvt.f32.f16.l12", 1016, 12, {0: "r%d" % dest_id, 2: "r%d" % (src + 105)},
                       pinned={1: 2147483648, 3: cc.MOV_RELEASE})
    b = bytes(g17as.assemble(line).text)
    d = list(model.decode(b, 0))
    got = [_NAMES.get(v) for k, v in d[0].values if k == "reg"] if d and d[0].opcode else None
    if got != [dest_half, "R%d" % src]:
        raise ValueError("op1016 decodes %s, not %s <- R%d" % (got, dest_half, src))
    return b


def exp2(dest, src, keep_src=False):
    """dest = 2**src (op1272/10, cc's own form), releasing the source unless it is kept.
    The hardware function is within one ulp of 2**x in both directions (docs/archive/g17-settle-20260923.md),
    not exactly RNE, so a reference for it is a one-ulp bound, not a bit pattern."""
    return _encode(EXP2, dest, [src], {3: KEEP if keep_src else RELEASE})


def shuffle_xor(dest, src, mask):
    """dest = src read from lane (lane ^ mask) (op14169/10), cc's own form: fixed control words 32 in
    slots 1 and 3, the XOR lane mask in slot 4. The source is kept (32). Masks 1, 2, 4, 8, 16 only,
    the measured one-bit butterfly (recon sections 122, 130)."""
    if mask not in (1, 2, 4, 8, 16):
        raise ValueError("shuffle_xor mask %r is not a measured one-bit butterfly mask" % (mask,))
    return _encode(SHUFFLE_XOR, dest, [src], {1: 32, 3: 32}, {4: mask})


# FP8 QUANTIZE-OUT (Set A item 9b). Both forms are Apple's own bytes from the OS compiler's
# lowering of `air.convert.f.v8f8<fmt>.f.v8f32` and two 32-bit stores at lane-dependent word
# indices (tools/tensorops-model/fp8_store_witness.py, compile only):
#   op13618/12  pack: a 16-bit half <- two fp32 registers, byte 0 from the first. Operand 2 is the
#               format (97 e4m3fn, 98 e5m2); it moves two template bits and g17auth maps one, so each
#               format keeps its own template and only registers and lifetimes are written.
#   op17202/8   one 32-bit word at index x 4 (the width is the template's; Apple's index is the
#               word index lane*5), binding descriptor in byte 1 (4 x rank), displacement 0.
# Semantics of the pack (recon section 138 part 1, all 2^32 fp32 patterns): round to nearest even,
# no saturation, and an overflow KEEPS ITS SIGN (e4m3fn beyond +-464 -> NaN 0x7f / 0xff, e5m2 at
# +-61440 or more -> inf 0x7c / 0xfc; hardware, results/g17-tensor-lowprec-v1); every NaN INPUT gives
# the positive canonical NaN (0x7f, 0x7e); signed zeros kept.
FP8PACK = {"e4m3fn": (13618, bytes.fromhex("2700006b2500ad1290836000")),
           "e5m2": (13618, bytes.fromhex("2700006b2500ad1290832040"))}
STOREW = (17202, bytes.fromhex("47040304210e1040"))


def fp8pack(dest, half, a, b, fmt, keep=False):
    """R<dest><half> ('L' or 'H') <- fp8(R<a>), fp8(R<b>) as bytes 0 and 1 (op13618). The 16-bit
    destination field is a slot, 2 x register + (1 for H)."""
    from agxforge.g17 import auth as g17auth
    opc, template = FP8PACK[fmt]
    life = KEEP if keep else RELEASE
    vals = {0: 2 * dest + (half == "H"), 3: g17auth.field_value(opc, 3, a), 4: life,
            5: g17auth.field_value(opc, 5, b), 6: life}
    out = bytes(g17auth.encode(opc, vals, template=template, trusted=(4, 6)))[:len(template)]
    ins = list(model.decode(out, 0))
    if not ins or ins[0].opcode is None or ins[0].opcode.id != opc or len(ins[0].raw) != len(template):
        raise ValueError("op13618: %s does not decode as itself" % out.hex())
    got = [_NAMES.get(v) if k == "reg" else v for k, v in ins[0].values]
    want = ["R%d%s" % (dest, half), 0, {"e4m3fn": 97, "e5m2": 98}[fmt], "R%d" % a, life, "R%d" % b, life]
    if got != want:
        raise ValueError("op13618 decodes %s, not %s" % (got, want))
    return out


def store_word(src, index, binding, src_last=True, index_last=False):
    """One 32-bit word R<src> to binding `binding` at word index R<index> (op17202)."""
    from agxforge.g17 import auth as g17auth
    opc, template = STOREW
    vals = {0: g17auth.field_value(opc, 0, src), 1: 16 if src_last else 0,
            5: g17auth.field_value(opc, 5, index), 6: 16 if index_last else 0}
    out = bytearray(bytes(g17auth.encode(opc, vals, template=template, trusted=(1, 6)))[:len(template)])
    out[1] = 4 * binding
    out = bytes(out)
    ins = list(model.decode(out, 0))
    if not ins or ins[0].opcode is None or ins[0].opcode.id != opc or len(ins[0].raw) != len(template):
        raise ValueError("op17202: %s does not decode as itself" % out.hex())
    v = ins[0].values
    got = [_NAMES.get(v[0][1]), v[1][1], _NAMES.get(v[5][1]), v[6][1], v[7][1], v[8][1], out[1]]
    want = ["R%d" % src, 16 if src_last else 0, "R%d" % index, 16 if index_last else 0, 0, 4, 4 * binding]
    if got != want:
        raise ValueError("op17202 decodes %s, not %s" % (got, want))
    return out


def recip(dest, src, keep_src=False):
    """dest = 1/src (op3658/10, cc's own form), releasing the source unless it is kept. Its
    accuracy is not settled by a dense sweep (exp2 and rsqrt are, within one ulp both ways)."""
    return _encode(RECIP, dest, [src], {3: KEEP if keep_src else RELEASE})


def fadd(dest, a, b, keep_a=False, keep_b=True):
    """dest = a + b (op998/12, cc's own form). Operands 3 and 5 are the lifetimes of a and b."""
    return _encode(FADD, dest, [a, b], {3: KEEP if keep_a else RELEASE, 5: KEEP if keep_b else RELEASE})


# REQUANTIZATION EPILOGUE (production row P6, machine model 25.130). The int32 -> int8/uint8 step in
# the GEMM's own dispatch. The conversion, rounding and narrowing forms are the EXACT bytes of the
# retained scalar requantization class (requantenc.REQUANT_STAGE_SIGNED, hardware-checked by
# test_g17requant's stage); only their register fields are rewritten, and only in place (destination
# == source), because op11179's and op9320's source-lifetime operands are unmeasured (tools/g17cvtf2i.py,
# cc's i32_to_f32 note) and an in-place conversion reads nothing again. fmin is Apple's own sw-f_min
# template (cc.FSELECT_TEMPLATE), operation 3; fmax (operation 7) is the form above.
from agxforge.g17 import requantenc as _requantenc

I2F = (11179, _requantenc.REQUANT_STAGE_SIGNED["i32_to_f32"])     # cvt.i2f, signed (operand 2 = 5)
RINT = (3770, _requantenc.REQUANT_STAGE_SIGNED["rint"])            # rint: round half to even
F2I = (9320, _requantenc.REQUANT_STAGE_SIGNED["narrow"])           # cvt.f2i of an integer-valued float
FMIN = (9700, bytes.fromhex("2200470b2300a0029400a1600040"))       # sw-f_min, operation 3


def _in_place(form, reg):
    """One of the requantization stage's unary forms rewritten to read and write R<reg>. Every
    non-register bit (the lifetime and mode operands included) stays the retained stage's."""
    opc, template = form
    dsts, srcs = g17auth.register_operands(opc)
    vals = {dsts[0]: g17auth.field_value(opc, dsts[0], reg), srcs[0]: g17auth.field_value(opc, srcs[0], reg)}
    b = bytes(g17auth.encode(opc, vals, template=template))[:len(template)]
    ins = list(model.decode(b, 0))
    if not ins or ins[0].opcode is None or ins[0].opcode.id != opc or len(ins[0].raw) != len(template):
        raise ValueError("op%d: %s does not decode as itself" % (opc, b.hex()))
    got = [_NAMES.get(v) for k, v in ins[0].values if k == "reg"]
    if got != ["R%d" % reg, "R%d" % reg]:
        raise ValueError("op%d decodes registers %s, not R%d in place" % (opc, got, reg))
    # the rewrite must move register bits only: every other operand decodes as the template's
    want = [v for k, v in list(model.decode(template, 0))[0].values if k != "reg"]
    if [v for k, v in ins[0].values if k != "reg"] != want:
        raise ValueError("op%d: the register rewrite moved a non-register operand" % opc)
    return b


def i2f_inplace(reg):
    """R<reg> = float(int32 R<reg>) (op11179, the stage's signed conversion)."""
    return _in_place(I2F, reg)


def rint_inplace(reg):
    """R<reg> = rint(R<reg>), round half to even (op3770, the stage's rounding)."""
    return _in_place(RINT, reg)


def f2i_inplace(reg):
    """R<reg> = int32(R<reg>) (op9320, the stage's narrowing). Fed only integer-valued floats inside
    [-255, 255] by the requantization epilogue, so its unmeasured mode operand (rounding, saturation)
    cannot change the result."""
    return _in_place(F2I, reg)


def fmin(dest, a, b, keep_a=False, keep_b=True):
    """dest = min(a, b) (op9700 operation 3, Apple's sw-f_min), lifetimes as fmax."""
    la, lb = KEEP if keep_a else RELEASE, KEEP if keep_b else RELEASE
    return _encode(FMIN, dest, [a, b, a, b], {4: la, 6: lb, 8: la, 10: lb})
