"""The uniform-preload constant program: what the prologue loads before the main program.

The production half of tools/g17uniformpreload.py - what g17cc asks for when it folds a
constant program. The probes that build witnesses for it stayed behind.
"""
import hashlib, json, os, sys


class Unsupported(Exception): pass
T_SR8 = bytes.fromhex("248021104701a082")        # op14061 8-byte: reg 105, argument offset 8
T_SR4 = bytes.fromhex("1c8a0827")                # op14061 4-byte: reg 106, argument offset 10
T_LD14 = bytes.fromhex("0f0003008044" "00a04100800000" "00")   # op12688: dest 105, pair 105, offset 0
T_PUB8 = bytes.fromhex("c30407023" "6a0a402")    # op592 8-byte: block op0 const 16, source 105, release
END = bytes.fromhex("0e000000"); FILLER = bytes.fromhex("0600"); ENTRY = 64
END = bytes.fromhex("0e000000"); FILLER = bytes.fromhex("0600"); ENTRY = 64
END = bytes.fromhex("0e000000"); FILLER = bytes.fromhex("0600"); ENTRY = 64
def encode_sr4(reg, offset):
    """op14061 4-byte: register byte0[6:4], offset in byte1 and byte3[7:5]. Both located over nine witnesses."""
    if not 105 <= reg <= 112: raise Unsupported("op14061 4-byte register %d: byte0[6:4] holds reg - 105 in 0..7" % reg)
    if offset % 2 or not 10 <= offset <= 18: raise Unsupported("op14061 4-byte argument offset %d: witnessed even offsets 10..18 only; the field is byte1 = 0x80 | offset and byte3[7:5] = (offset >> 2) - 1, and outside the witnessed range those two encodings are not known to agree" % offset)
    u = bytearray(T_SR4); u[0] = (u[0] & 0x8F) | ((reg - 105) << 4); u[1] = 0x80 | offset; u[3] = (u[3] & 0x1F) | (((offset >> 2) - 1) << 5)
    return bytes(u)
def encode_sr8(reg, offset):
    """op14061 8-byte: register bits 1..2 at byte7[4:3] (three even witnesses); bit 0 and the offset are not located."""
    if offset != 8: raise Unsupported("op14061 8-byte argument offset %d: every witness carries 8 and the field is not located" % offset)
    if (reg - 105) % 2 or not 105 <= reg <= 111: raise Unsupported("op14061 8-byte register %d: only bits 1..2 of reg - 105 are located (byte7[4:3], witnesses 105/107/109); an odd register would need bit 0, which no witness varies" % reg)
    u = bytearray(T_SR8); u[7] = (u[7] & 0xE7) | ((((reg - 105) >> 1) & 3) << 3)
    return bytes(u)
def encode_ld14(dest, pair, offset):
    """op12688 14-byte: destination bit 0 at byte0[4] (witnesses 105, 106); pair 105 only; offset 0 only."""
    if dest not in (105, 106): raise Unsupported("op12688 destination %d: only bit 0 of the destination is located (byte0[4]); witnesses 105 and 106" % dest)
    if pair != 105: raise Unsupported("op12688 base pair %d: the 14-byte witnesses all read through pair 105; the pair field is inferred from the 8-byte form only" % pair)
    if offset != 0: raise Unsupported("op12688 load offset %d: not located - S1 carries 0, M4 carries 1604 across bytes 6, 7 and 13; M4's load is the reproducer" % offset)
    u = bytearray(T_LD14); u[0] = (u[0] & 0xEF) | (((dest - 105) & 1) << 4)
    return bytes(u)
def encode_pub8(const, src):
    """op592 8-byte publish into block op0, release lifetime. The constant is the nine-bit field the decoder sweep
    located (CONST9: S1's 16 and M4's 12 are both its readings, and M4's instruction is reproduced from S1's by
    writing 12 into it - the "two points, no rule" of 10z is closed by the decoder, not by hardware); the source
    is PUB8_SRC. The compiler still emits only the witnessed program shapes."""
    if const % 4 or not 0 <= const < 512: raise Unsupported("op592 block-op0 publish const %d: a byte offset of a four-byte word inside the nine-bit field" % const)
    if not 105 <= src <= 105 + 255: raise Unsupported("op592 block-op0 publish source %d: registers are 105 + an eight-bit field" % src)
    u = bytearray(T_PUB8); _put(u, CONST9, const); _put(u, PUB8_SRC, src - 105); return bytes(u)
CONST9 = ((3, 0), (0, 4), (7, 3), (7, 4), (0, 7), (7, 5), (2, 7), (2, 3), (2, 4))
SLOT_A = ((1, 1), (3, 5), (3, 6), (3, 7), (5, 7), (8, 0), (8, 1), (8, 2))
SLOT_B = ((8, 7), (9, 0), (9, 1), (9, 2), (9, 3), (9, 4), (9, 5), (9, 6))
ALU_DEST = ((0, 4), (7, 3), (7, 4), (0, 7), (7, 5), (2, 7), (2, 3), (2, 4))
LD_DEST = ((0, 4), (0, 5), (0, 6), (0, 7), (2, 6), (2, 7), (5, 4))
LD_PAIR = ((1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7))
PUB8_SRC = ((1, 1), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (6, 0), (6, 1))
T_LD8_S2 = bytes.fromhex("2f04030080440020")        # S2 +20: r107 <- [pair 107], first of two loads
T_LD8_S3A = bytes.fromhex("4f08030080440020")       # S3 +28: r109 <- [pair 109], first of three
T_LD8_S3B = bytes.fromhex("2f04030080640020")       # S3 +36: r107 <- [pair 107], second of three
T_LD14_S2 = bytes.fromhex("0f000300806400a04100800000" "00")   # S2 +28: r105 <- [pair 105], last of two
T_LD14_S3 = bytes.fromhex("1f000300888400a04100800000" "00")   # S3 +44: r106 <- [pair 105], last of three
T_ADD12_S3 = bytes.fromhex("2704043a2900a30228822100")         # S3 +58: r105 = r107 + r109
T_PUBADD_S2 = bytes.fromhex("c704041a3100a30a28812100")        # S2 +42: block[20] = r105 + r107
T_PUBADD_S3 = bytes.fromhex("c70a041a3100a31228802100")        # S3 +70: block[24] = r106 + r105
def _put(u, bits, v):
    if v < 0 or v >> len(bits): raise Unsupported("value %d does not fit a %d-bit field" % (v, len(bits)))
    for i, (byte, bit) in enumerate(bits): u[byte] = (u[byte] & ~(1 << bit)) | (((v >> i) & 1) << bit)
def encode_ld8(dest, pair, template):
    """op12688 8-byte: dest and pair through the located carriers; everything else (the composite) from the witness
    at this position, which the caller names."""
    if not 105 <= dest <= 105 + 127 or not 105 <= pair <= 105 + 127: raise Unsupported("op12688 registers are 105 + a seven-bit field")
    u = bytearray(template); _put(u, LD_DEST, dest - 105); _put(u, LD_PAIR, pair - 105); return bytes(u)
def encode_ld14_at(dest, pair, template):
    """op12688 14-byte from a positional witness: dest and pair authored, offset 0 (the offset field is not located)."""
    if not 105 <= dest <= 105 + 127 or not 105 <= pair <= 105 + 127: raise Unsupported("op12688 registers are 105 + a seven-bit field")
    u = bytearray(template); _put(u, LD_DEST, dest - 105); _put(u, LD_PAIR, pair - 105); return bytes(u)
def encode_add12(dest, a, b, template=T_ADD12_S3):
    """op10282 12-byte, register form: dest = a + b (registers 105-based)."""
    u = bytearray(template); _put(u, ALU_DEST, dest - 105); _put(u, SLOT_A, a - 105); _put(u, SLOT_B, b - 105); return bytes(u)
def encode_pubadd(const, a, b, template):
    """op10306: block[const] = a + b, both released. const through the nine-bit field; the template names which
    member's inherited bits (byte1[3:2]) this instruction carries."""
    if const % 4 or not 0 <= const < 512: raise Unsupported("op10306 block constant %d: a byte offset of a four-byte word inside the nine-bit field" % const)
    u = bytearray(template); _put(u, CONST9, const); _put(u, SLOT_A, a - 105); _put(u, SLOT_B, b - 105); return bytes(u)
CONSTANT_PROGRAM_FORMS = ((592, 8), (684, 4), (12688, 14), (13483, 2), (14061, 4), (14061, 8))
CONSTANT_PROGRAM_FORMS_BY_TERMS = {1: CONSTANT_PROGRAM_FORMS,
                                   2: ((684, 4), (10306, 12), (12688, 8), (12688, 14), (13483, 2), (14061, 4), (14061, 8)),
                                   3: ((684, 4), (10282, 12), (10306, 12), (12688, 8), (12688, 14), (13483, 2), (14061, 4), (14061, 8))}
def _pad(body):
    entry = 64 if len(body) <= 64 else 128
    if (entry - len(body)) % 2: raise Unsupported("the constant program does not reach the %d-byte entry by two-byte filler" % entry)
    return body + FILLER * ((entry - len(body)) // 2)
def constant_program_folded(terms, publish_const):
    """S2's (two terms) and S3's (three terms) constant programs from their operands: the argument-table entries of
    the terms in order (units 8/10, 12/14, 16/18 - the entry law witnessed for declared buffers 1..n beside a
    written buffer 0), each loaded through its pair (8-byte loads, the last one 14-byte), summed, and published
    into block op0 at the constant main reads. The register plan is the witness's - Apple's allocation, reproduced,
    not chosen here - and the publish constant must be the S-shape's 4 x (n + 3) records: the entry law is not
    witnessed for any other declaration, so any other shape refuses by name."""
    if terms == 1: return constant_program(publish_const=publish_const)
    if terms not in (2, 3): raise Unsupported("a folded preload of %d terms: only two (S2) and three (S3) are witnessed" % terms)
    if publish_const != 4 * (terms + 3): raise Unsupported("a %d-term preload publishing at %d: the witnessed S-shape declares buffers 0..%d beside two internals, so the block is %d bytes; the argument-table entry law is not witnessed for any other declaration" % (terms, publish_const, terms, 4 * (terms + 3)))
    if terms == 2:
        body = (encode_sr8(107, 8) + encode_sr4(108, 10) + encode_sr4(105, 12) + encode_sr4(106, 14)
                + encode_ld8(107, 107, T_LD8_S2) + encode_ld14_at(105, 105, T_LD14_S2) + encode_pubadd(20, 105, 107, T_PUBADD_S2) + END)
    else:
        body = (encode_sr8(109, 8) + encode_sr4(110, 10) + encode_sr4(107, 12) + encode_sr4(108, 14) + encode_sr4(105, 16) + encode_sr4(106, 18)
                + encode_ld8(109, 109, T_LD8_S3A) + encode_ld8(107, 107, T_LD8_S3B) + encode_ld14_at(106, 105, T_LD14_S3)
                + encode_add12(105, 107, 109) + encode_pubadd(24, 106, 105, T_PUBADD_S3) + END)
    return _pad(body)
def constant_program(pair=105, arg_offset=8, publish_const=16):
    """S1's constant program from its operands: read src's pointer into (pair, pair + 1), load src[0] into pair, publish it."""
    body = encode_sr8(pair, arg_offset) + encode_sr4(pair + 1, arg_offset + 2) + encode_ld14(pair, pair, 0) + encode_pub8(publish_const, pair) + END
    if len(body) > ENTRY or (ENTRY - len(body)) % 2: raise Unsupported("the constant program does not fit the %d-byte entry" % ENTRY)
    return body + FILLER * ((ENTRY - len(body)) // 2)
