import os
#!/usr/bin/env python3
"""Encode G17 scalar instructions from recovered semantics, driven by isa/g17-scalar-isa.toml.

Nothing here copies a compiler instruction wholesale: the semantic fields are computed from
(dest, src1, imm) and written into the bit positions the database records. Bytes the database
classes as structural are carried from a template, and encode() reports exactly which those
are, so the accounting of what is authored versus inherited stays honest.
"""
import os, tomllib

# ANCHORED ON THE CHECKOUT ROOT: two levels up from agxforge/g17/, where one sufficed from tools/.
_here = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_here))
_scalar_isa = os.path.join(ROOT, "isa", "g17-scalar-isa.toml")
ISA = tomllib.load(open(_scalar_isa, "rb")) if os.path.exists(_scalar_isa) else None
ISA = tomllib.load(open(_scalar_isa, "rb"))

# Bits the semantic fields OWN, per byte. Everything else is inherited from a template of the
# same sub-form and reported, never silently inherited. This is a mask, not a byte list: the
# old byte list [2,4,6,10,11] was wrong - byte0, byte1, byte3, byte8 and byte9 each carry
# structural bits alongside their semantic ones, and an encoder that zeroes them emits a
# malformed instruction that still decodes correctly. See ledger/g17-authored-chain.toml.
OWNED_ADD = {0: 0x90, 1: 0x03, 3: 0xE0, 5: 0x80, 7: 0x18, 8: 0x83, 9: 0x07}

def structural_add(template):
    """The bits of `template` this form inherits rather than authors, as {byte: mask}."""
    return {i: (0xFF & ~OWNED_ADD.get(i, 0)) & template[i]
            for i in range(len(template)) if (0xFF & ~OWNED_ADD.get(i, 0)) & template[i]}

def encode_add_imm(dest, src1, imm, template):
    """alu.add.imm: dest = byte0[4] + 2*byte7[3] + 4*byte7[4]
                    src1 = 2*byte9[2:0] + byte8[7]
                    imm  = byte1[1:0] | byte3[7:5]<<2 | byte5[7]<<5 | byte8[1:0]<<6"""
    if not 0 <= dest <= 15:
        raise ValueError("dest r%d out of range: this form encodes r0..r15" % dest)
    if not 0 <= src1 <= 15: raise ValueError("src1 out of range")
    if not 0 <= imm <= 255: raise ValueError("immediate needs the wide form")
    u = bytearray(template)
    u[0] = (u[0] & ~0x90) | ((dest & 1) << 4) | (((dest >> 3) & 1) << 7)   # byte0[7] = weight 8
    u[7] = (u[7] & ~0x18) | (((dest >> 1) & 1) << 3) | (((dest >> 2) & 1) << 4)
    u[9] = (u[9] & ~0x07) | ((src1 >> 1) & 0x07)
    u[8] = (u[8] & ~0x80) | ((src1 & 1) << 7)
    u[1] = (u[1] & ~0x03) | (imm & 3)
    u[3] = (u[3] & ~0xE0) | (((imm >> 2) & 7) << 5)
    u[5] = (u[5] & ~0x80) | (((imm >> 5) & 1) << 7)
    u[8] = (u[8] & ~0x03) | ((imm >> 6) & 3)
    return bytes(u)

def liveness_mark(u):
    """byte4[3]: set by the compiler exactly when the result feeds a later ALU (31/31), but
    causally inert - clearing it on a forwarded producer changes nothing. Metadata, not state."""
    return (u[4] >> 3) & 1

def decode_add_imm(u):
    return dict(dest=((u[0] >> 4) & 1) | (((u[7] >> 3) & 1) << 1) | (((u[7] >> 4) & 1) << 2)
                     | (((u[0] >> 7) & 1) << 3),
                src1=2 * (u[9] & 7) + (u[8] >> 7),
                imm=(u[1] & 3) | (((u[3] >> 5) & 7) << 2) | (((u[5] >> 7) & 1) << 5) | ((u[8] & 3) << 6))

def roundtrip(u):
    """Self-consistency only: decode then re-encode USING u ITSELF as the template.

    WARNING, measured 2026-09-03: this is vacuous as a validity test. Because encode() starts
    from the template and rewrites only the bits the semantic fields own, this returns True for
    ANY 12 bytes - it succeeded at 1254 of 1254 offsets of a real shader, including offsets
    that are plainly mid-instruction. It proves the encoder is its own inverse, nothing more.
    Use conforms() to ask whether bytes are actually an instance of the form."""
    d = decode_add_imm(u)
    return encode_add_imm(d["dest"], d["src1"], d["imm"], u) == bytes(u)

# Bits that legitimately vary between instances of the SAME form and must not count as
# structural mismatches: byte4[3] is the liveness marker (causally inert, see
# ledger/g17-byte4-liveness-metadata.toml), byte10 is the op field whose 0x01/0x11/0x21 are all
# add variants, and byte8[3] varies with operand routing.
VARIABLE = {4: 0x08, 8: 0x08, 10: 0xFF}

def conforms(u, template):
    """Do these bytes belong to the same sub-form as `template`?

    Re-encode u's operands onto the template and compare, ignoring bits known to vary between
    instances of one form. Unlike roundtrip() this CAN fail, so it carries information: measured
    over a real shader it accepts a small fraction of offsets, not all of them."""
    d = decode_add_imm(u)
    enc = encode_add_imm(d["dest"], d["src1"], d["imm"], template)
    for i in range(min(len(enc), len(u))):
        m = 0xFF & ~VARIABLE.get(i, 0)
        if (enc[i] & m) != (u[i] & m): return False
    return True


# --- store.device -----------------------------------------------------------------------
# Recovered 2026-09-03. A store writes N CONSECUTIVE registers to N consecutive slots, which
# is why a kernel with eight scalar stores emits only three instructions.
# byte1[6:2] IS THE BUFFER SELECTOR and the encoder writes it now, so it is owned. Measured by
# flipping each bit of an emitted store and reading the address expression back - const = 4 x
# byte1[6:2], identical on op17229 and op17244. Before this the const came from the template,
# which pinned every store this compiler emits to one buffer.
OWNED_STORE = {0: 0xF0, 1: 0x7C, 4: 0x60, 6: 0x80, 7: 0x1F, 9: 0x20, 13: 0xFF}
_NCOMP = {1: 0, 2: 1, 3: 2, 4: 3}

STORE_SRC = ((0, 4), (0, 5), (0, 6), (0, 7), (2, 6), (2, 7), (5, 4))


# THE BUFFER CONST LIVES IN byte1[6:2] AND IS WORTH 4 PER UNIT. Measured 2026-09-08 by flipping
# every bit of an emitted store and reading the address expression back: bit2 moves the const by 4,
# bit3 by 8, bit4 by 16, bit5 by 32, bit6 by 64. So const = 4 * byte1[6:2], and the const is the
# BUFFER SELECTOR - 4 x the buffer's rank among the ones the kernel binds.
#
# It used to be inherited from the template, which fixed every store this compiler emits at const 4.
# That is correct for a kernel whose output is the second of two buffers and wrong for every other
# shape - right by accident on the population, which is why twenty end-to-end kernels passed with it.
STORE_CONST = (1, 2, 5)          # (byte, low bit, width)

# THE STORE SUB-FORM IS byte5[3:2], AND IT IS WHAT APPLE'S DECODER NUMBERS: on one 14-byte store with
# every other byte held, the decoder walk gives 00 -> op17229, 01 -> op17235, 10 -> op17238, 11 ->
# op17244, unchanged by byte4[6:5] (the component field) - a decoder fact, measured on this side's
# own bytes and matching the nine Apple objects of results/g17-texture-family-compiles-v1, whose
# texel stores all carry 01. What the sub-forms DO is measured only in part: 11 is the store this
# compiler emits (k and k+1 from r<src>, r<src+1> at n=2; k and k+3 at the 0 encoding); 01 is the
# one consumer measured to read a texture fetch (ledger/g17-only-one-consumer-can-read-a-texture-
# fetch.toml) and was measured NOT to read a changed general source (ledger/g17-stage3-full-chain
# .toml: the "byte5 = 0x04 sub-form rejects a source change") - consistent with its source naming
# the fetch's destination rather than a general register. 00 and 10 are not emitted here.
STORE_SUBFORM_OPCODES = {0: 17229, 1: 17235, 2: 17238, 3: 17244}


def encode_store(src, n, slot, template, wait_load=0, const=None, subform=None):
    """Store r<src> .. r<src+n-1> to out[slot ..]. Fields:
         src   = byte0[7:4]                         (first source register)
         n     = byte4[6:5] as 0->1, 1->2, 2->3, 3->4
         slot  = 2*(byte7[6:0]) + byte6[7]
    The template MUST be a multi-capable store (byte5 bit3 set). The byte5=0x04 sub-form
    rejects any source change - see isa/g17-scalar-isa.toml."""
    if n not in _NCOMP: raise ValueError("n=%d: 1..4 components only" % n)
    if not 0 <= src <= 127: raise ValueError("src r%d out of range (7-bit field)" % src)
    if not 0 <= slot <= 64 * 255 + 63: raise ValueError("slot %d out of range" % slot)
    # The byte5 = 0x04 sub-form rejects a SOURCE change. Narrowed twice: first to fire only when
    # src actually moves (it was refusing slot changes too), then to byte5 == 0x04 exactly. The
    # test for the second narrowing is in ledger/g17-stage3-full-chain.toml: a byte5 = 0x06 store
    # had its source changed r1 -> r6 and stored the right value from the right register, so the
    # "not bit3" condition was too broad.
    # THE GUARD THAT USED TO LIVE HERE IS GONE, narrowed a third time and this time to nothing. It refused a
    # source change when template[5] == 0x04, inherited from a finding about the sub-form.
    # ledger/g17-stage3-full-chain.toml records the two earlier narrowings and says of both that the check
    # "was suppressing capability the hardware actually has"; the remaining condition was never tested,
    # because the r1 -> r6 test that produced the second narrowing ran on a byte5 = 0x06 template. Three
    # lines of evidence retire it:
    #
    #   Apple's population   515 of the 525 op17235/8 instances in isa/g17-corpus-programs.jsonl carry
    #                        byte5 = 0x04, and between them they use NINETEEN different source registers
    #   hardware             results/g17-halfslot-runtime-v1 (integration's receipt) executed this side's
    #                        op17199 stores - same sub-form, byte5 = 0x04 - with a source register the
    #                        template did not have, at four displacements, agreeing with the reference
    #   a witness            results/g17-wordslot-compiles-v1/T4 is Apple's own op17235/8 with src = 1
    #
    # A field that takes nineteen values in shipped code and stores correctly on silicon is not a field an
    # encoder may refuse to write. What replaces the guard is the round-trip: every caller's bytes are read
    # back through the decoder in selfcheck, which catches a source this form cannot actually carry.
    u = bytearray(template)
    # SEVEN BITS, NOT FOUR. byte0[7:4] is the low nibble; bits 4, 5 and 6 are byte2[6], byte2[7]
    # and byte5[4], every delta a power of two under mutation. The four-bit field confined every
    # stored value to r0..r15, which is why a program whose value lived above r15 could not be
    # emitted at all.
    _bits_put(u, STORE_SRC, src)
    u[4] = (u[4] & ~0x60) | (_NCOMP[n] << 5)
    if subform is not None:
        if subform not in STORE_SUBFORM_OPCODES: raise ValueError("store sub-form %r: byte5[3:2] takes 0..3" % (subform,))
        u[5] = (u[5] & ~0x0C) | (subform << 2)
    hi, rem = divmod(slot, 64)
    if len(u) > 9: u[9] = (u[9] & ~0x20) | ((wait_load & 1) << 5)
    u[6] = (u[6] & ~0x80) | ((rem & 1) << 7)
    u[7] = (u[7] & ~0x1F) | ((rem >> 1) & 0x1F)
    if const is not None:
        by, lo, w = STORE_CONST
        if const % 4: raise ValueError("store const %d is not a multiple of 4" % const)
        q = const // 4
        if not 0 <= q < (1 << w): raise ValueError("store const %d exceeds the field" % const)
        u[by] = (u[by] & ~(((1 << w) - 1) << lo)) | (q << lo)
    if len(u) > 13: u[13] = hi
    elif hi: raise ValueError("slot %d needs the 14-byte store form" % slot)
    return bytes(u)

# THE HALF-VECTOR SLOT STORES (handoff 10ah; integration's 66cef0b5). op17208 (n=2), op17217 (n=3) and
# op17226 (n=4) are the SAME slot store as the word forms with byte0[3] CLEAR, which is the width bit the
# sweep located (setting it on op17220 gives op17256, this backend's accepted word vector store). The
# component count is byte4[6:5] = n-1 and is part of Apple's opcode NUMBER, exactly as it is for the word
# slot stores - so encode_store already writes every field these forms need and this is a TEMPLATE, not a
# new encoder.
#
# The bytes are the retained fourteen-byte corpus instances (isa/g17-corpus-programs.jsonl: mc-cc.hh.eq-4 at
# 136 and mc-cc.hh.eq-2 at 68), which roundtrip_store reproduces exactly. byte12 is inert on the sweep, and
# bytes 8 and 11's low bits are INHERITED from the witness and stated as such: nothing here measures them.
HALFVEC_SLOT_TEMPLATE_4 = bytes.fromhex("07000300610c10a4010000000003")   # mc-cc.hh.eq-4 +136: out[200..203] <- r0..r3, halves
HALFVEC_SLOT_TEMPLATE_2 = bytes.fromhex("07000300210c10a4010000000003")   # mc-cc.hh.eq-2 +68:  out[200..201] <- r0..r1, halves
HALFVEC_SLOT_TEMPLATES = {2: HALFVEC_SLOT_TEMPLATE_2, 3: HALFVEC_SLOT_TEMPLATE_4, 4: HALFVEC_SLOT_TEMPLATE_4}
HALFVEC_OPCODES = {2: 17208, 3: 17217, 4: 17226}
HALFVEC_WIDTH_BIT = (0, 3)      # clear = sixteen-bit elements, set = thirty-two-bit (op17220 -> op17256)


# THE SINGLE SIXTEEN-BIT ELEMENT SLOT STORE, op17199 (handoff 10aj; integration's d8557330). One half
# element at an immediate address, sub-form 01, and the sixteen-bit twin of op17235 - flipping byte0[3] on
# any witness below gives op17235 under Apple's decoder, the same width bit the half-vector round located.
#
# ITS ADDRESS IS A BYTE DISPLACEMENT, NOT A SLOT, and that is why this is a separate encoder rather than
# another template for encode_store. The decoder sweep locates the field completely, no strays, on all three
# lengths: byte6[5] is bit 0 (worth ONE byte), byte6[6] worth 2, byte6[7] worth 4, byte7[0:4] worth 8..128,
# and the fourteen-byte form adds byte13 for 256..32768. The word store's slot field is the same physical
# bits from byte6[7] up with the low two always zero, which is why decode_store read S0's displacement of 14
# as "slot 3": the same bits at a four-byte scale with no bit 0. A half element at an ODD index lands two
# bytes into its word, and nothing in the word store's field map can say so.
#
# THE LENGTH IS SELECTED BY THE VALUE AND THE ADDRESS, measured on three retained witnesses:
#   8 bytes   no load to wait for                         results/g17-halfslot-compiles-v1/S2 (a zero)
#   10 bytes  the value comes from a load (byte9[5])      S0, S3
#   14 bytes  the displacement does not fit in 8 bits     S4 (element 200 -> 400 bytes)
HALFSLOT_TEMPLATE_8 = bytes.fromhex("070003000104d021")                    # S2 +4:  out[7] <- r425, one half, no wait
HALFSLOT_TEMPLATE_10 = bytes.fromhex("07000300010 4d0a10020".replace(" ", ""))   # S0 +18: the same, waiting on a load
HALFSLOT_TEMPLATE_14 = bytes.fromhex("07000300010410b201200000000 1".replace(" ", ""))  # S4 +18: displacement 400
HALFSLOT_DISP_LOW = ((6, 5), (6, 6), (6, 7), (7, 0), (7, 1), (7, 2), (7, 3), (7, 4))
HALFSLOT_DISP_HIGH = tuple((13, b) for b in range(8))
HALFSLOT_WAIT = (9, 5)
HALFSLOT_OPCODE = 17199
HALFSLOT_MAX_DISP = 32766        # the field is signed: 32768 decodes as -32768, so this is the last even non-negative


# THE WORD TWIN, op17235 (handoff 10ak; integration's f49d43fb). Same store one width up: byte0[3] set,
# operand 0 naming the 105-based file instead of the 425-based one, the SAME displacement field and the same
# length rule. These are retained Apple witnesses from results/g17-wordslot-compiles-v1 rather than the half
# templates with a bit flipped, because a template is evidence and a flipped bit is an inference.
WORDSLOT_TEMPLATE_8 = bytes.fromhex("0f000300010490 3f".replace(" ", ""))          # T4 +4:  out[63] <- r106, disp 252
WORDSLOT_TEMPLATE_10 = bytes.fromhex("0f00030001049 0a30020".replace(" ", ""))      # T0 +18: out[7] <- r105, waiting on a load
WORDSLOT_TEMPLATE_14 = bytes.fromhex("0f000300010410a301000000000 1".replace(" ", ""))  # T4 +12: out[70] <- r105, disp 280
WORDSLOT_OPCODE = 17235


def _element_templates(half):
    return ((HALFSLOT_TEMPLATE_8, HALFSLOT_TEMPLATE_10, HALFSLOT_TEMPLATE_14) if half
            else (WORDSLOT_TEMPLATE_8, WORDSLOT_TEMPLATE_10, WORDSLOT_TEMPLATE_14))


def _halfslot_template(disp, wait_load, half=True):
    t8, t10, t14 = _element_templates(half)
    if disp > 0xFF:
        return t14, 14
    return (t10, 10) if wait_load else (t8, 8)


# THE SIXTEEN-BIT REGISTER MOVE, op590/4 (handoff 10al). 773 instances in 495 corpus programs, and the
# sole missing form of 380 of them. Located by sweeping U0's second instruction, one bit at a time:
#
#   destination   byte0[4:7] + byte2[6:7]   six bits, 425-based
#   dest file     byte3[0]                  clear = the 425 file, set = the 281 file (moves reg:425 to reg:281)
#   source        byte1[1:6] + byte3[4]     seven bits
#   source file   byte1[0]                  clear = 425, set = 281 - the SAME sense as the destination's, which
#                                           is checked rather than assumed: all four combinations were
#                                           authored and read back, giving 425<-425, 425<-281, 281<-425 and
#                                           281<-281. The first draft of this comment said the two bits had
#                                           opposite senses, which was false and would have been a false
#                                           statement about an encoding in a file other people author from
#   lifetime      byte2[3] (16) + byte1[7] (32)   the source's keep/release operand, which Apple prints as an
#                                           immediate and which memory:g17-modifier-operand-lifetimes says
#                                           must be WRITTEN from liveness rather than inherited
#
# byte2[0] turns this into op555, the sixteen-bit zero move - so the zero this project lowered first and this
# move are one bit apart, which is the tidiest confirmation available that both readings are right.
MOVHALF_TEMPLATE = bytes.fromhex("03010900")        # U0 +26: r425 <- r281, releasing its source
MOVHALF_DEST = ((0, 4), (0, 5), (0, 6), (0, 7), (2, 6), (2, 7))
MOVHALF_DEST_FILE = (3, 0)
MOVHALF_SRC = ((1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (3, 4))
MOVHALF_SRC_FILE = (1, 0)
MOVHALF_KEEP16 = (2, 3)
MOVHALF_KEEP32 = (1, 7)
MOVHALF_OPCODE = 590


def _put_lifetime(u, lifetime):
    """THE LIFETIME TAKES THREE VALUES, NOT TWO. Apple's corpus carries 0 as well as 16 (release) and 32
    (keep) - the op586/4 witness ab_store_wide +114 is a 0 - and an encoder that could only write 16 or 32
    failed to reproduce it. What 0 MEANS is not measured here; it is expressible so a witness can be
    round-tripped, and this side's own copies write 16 or 32 from liveness."""
    if lifetime not in (0, 16, 32):
        raise ValueError("the source lifetime operand takes 0, 16 or 32; got %r" % (lifetime,))
    b16, bit16 = MOVHALF_KEEP16
    b32, bit32 = MOVHALF_KEEP32
    u[b16] = (u[b16] & ~(1 << bit16)) | ((1 if lifetime == 16 else 0) << bit16)
    u[b32] = (u[b32] & ~(1 << bit32)) | ((1 if lifetime == 32 else 0) << bit32)


def _get_lifetime(u):
    b16, bit16 = MOVHALF_KEEP16
    b32, bit32 = MOVHALF_KEEP32
    return 16 * ((u[b16] >> bit16) & 1) + 32 * ((u[b32] >> bit32) & 1)


def encode_movhalf(dest, src, keep_src=False, dest_281=False, src_281=True, template=MOVHALF_TEMPLATE,
                   lifetime=None):
    """r<dest> <- r<src>, sixteen bits, at four bytes.

    `dest_281` and `src_281` name which half register FILE each operand is in - set means the 281-based file
    for both, and all four combinations decode as expected. `keep_src` writes the lifetime
    operand: 32 keeps the source, 16 releases it, and it is written from the caller's liveness because a
    lifetime inherited from a template is how an authored program silently reads zero."""
    if not 0 <= dest < 64: raise ValueError("dest r%d out of the six-bit destination field" % dest)
    if not 0 <= src < 128: raise ValueError("src r%d out of the seven-bit source field" % src)
    u = bytearray(template)
    _bits_put(u, MOVHALF_DEST, dest)
    _bits_put(u, MOVHALF_SRC, src)
    b, bit = MOVHALF_DEST_FILE
    u[b] = (u[b] & ~(1 << bit)) | ((1 if dest_281 else 0) << bit)
    b, bit = MOVHALF_SRC_FILE
    u[b] = (u[b] & ~(1 << bit)) | ((1 if src_281 else 0) << bit)
    _put_lifetime(u, (32 if keep_src else 16) if lifetime is None else lifetime)
    return bytes(u)


def decode_movhalf(u):
    b, bit = MOVHALF_DEST_FILE
    dest_281 = bool((u[b] >> bit) & 1)
    b, bit = MOVHALF_SRC_FILE
    src_281 = bool((u[b] >> bit) & 1)
    return dict(dest=_bits_get(u, MOVHALF_DEST), src=_bits_get(u, MOVHALF_SRC),
                dest_281=dest_281, src_281=src_281, lifetime=_get_lifetime(u),
                keep_src=_get_lifetime(u) == 32)


def roundtrip_movhalf(u):
    d = decode_movhalf(u)
    return encode_movhalf(dest=d["dest"], src=d["src"], lifetime=d["lifetime"],
                          dest_281=d["dest_281"], src_281=d["src_281"], template=u) == bytes(u)


# THE THIRTY-TWO-BIT REGISTER MOVE, op586/4, in the same shape one width up (handoff 10al). Swept on a
# corpus witness (ab_store_wide +114, 4,707 instances of this length) and the layout is identical to
# op590/4's: destination byte0[4:7] + byte2[6:7], source byte1[1:6] + byte3[4], lifetime byte2[3] + byte1[7].
# The two FILE bits are not fields here - byte1[0] and byte3[0] are refused by the decoder, which is what a
# word register having no half-file selector looks like - and byte2[0] gives op554, the thirty-two-bit zero,
# exactly as byte2[0] on the half move gives op555.
#
# WHY THIS EXISTS WHEN mov.4 ALREADY EMITTED op586: mov.4 goes through g17as.assemble, which needs Apple's
# decoder at emit time, so a program containing one cannot be part of a DECODER-FREE audited build - the
# audit produced "emitted bytes do not decode" for it while the same program compiled fine outside. This
# encoder is pure, so the repeat copies are audit-safe. mov.4's own bytes are untouched.
MOVWORD_TEMPLATE = bytes.fromhex("1b000100")        # ab_store_wide +114: r106 <- r105
MOVWORD_OPCODE = 586


def encode_movword(dest, src, keep_src=False, template=MOVWORD_TEMPLATE, lifetime=None):
    """r<105+dest> <- r<105+src>, thirty-two bits, at four bytes, with the source lifetime written."""
    if not 0 <= dest < 64: raise ValueError("dest r%d out of the six-bit destination field" % dest)
    if not 0 <= src < 128: raise ValueError("src r%d out of the seven-bit source field" % src)
    u = bytearray(template)
    _bits_put(u, MOVHALF_DEST, dest)
    _bits_put(u, MOVHALF_SRC, src)
    _put_lifetime(u, (32 if keep_src else 16) if lifetime is None else lifetime)
    return bytes(u)


def decode_movword(u):
    return dict(dest=_bits_get(u, MOVHALF_DEST), src=_bits_get(u, MOVHALF_SRC),
                lifetime=_get_lifetime(u), keep_src=_get_lifetime(u) == 32)


def roundtrip_movword(u):
    d = decode_movword(u)
    return encode_movword(dest=d["dest"], src=d["src"], lifetime=d["lifetime"], template=u) == bytes(u)


def encode_element_store(src, disp, half=True, const=None, wait_load=0, template=None):
    """ONE element at an immediate byte displacement: op17199 at sixteen bits, op17235 at thirty-two.

    The two are one bit apart at byte0[3] and share everything else - the displacement field, its signed
    bound, the length rule and sub-form 01 - so they share an encoder, with the WIDTH choosing the retained
    template family. `half=False` writes the word form, whose displacement is a multiple of four in every
    one of Apple's 525 eight-byte instances.

    THE SOURCE FIELD IS WRITTEN, and that is measured rather than assumed. encode_store refuses a source
    change on a byte5 == 0x04 template, a guard narrowed twice before (ledger/g17-stage3-full-chain.toml,
    where both narrowings "were suppressing capability the hardware actually has") and never tested in its
    remaining form. It is wrong here on three counts: 515 of Apple's 525 op17235/8 instances carry
    byte5 = 0x04 and use NINETEEN different source registers between them; integration's hardware receipt
    (results/g17-halfslot-runtime-v1) executed this side's op17199 stores with a source the template did not
    have, at four displacements, agreeing with the reference; and T4 below is Apple's own op17235/8 with
    src = 1. So this encoder writes the source and the guard is narrowed a third time where it lives."""
    return _encode_element(src, disp, half, const, wait_load, template)


def encode_halfslot_store(src, disp, const=None, wait_load=0, template=None):
    """out[disp .. disp+1] <- the sixteen bits of r<425+src>, at 8, 10 or 14 bytes by the rule above.

    `disp` is a BYTE displacement, so a caller storing element e of a half array passes 2*e - which is how
    an odd element is expressible at all. A displacement this side cannot write refuses rather than being
    truncated into another location."""
    return _encode_element(src, disp, True, const, wait_load, template)


def _encode_element(src, disp, half, const, wait_load, template):
    if not 0 <= src <= 127: raise ValueError("src r%d out of range (7-bit field)" % src)
    if disp < 0: raise ValueError("a negative displacement (%d)" % disp)
    step = 2 if half else 4
    if disp % step:
        raise ValueError("a %s element at byte displacement %d: every retained witness is %d-byte aligned "
                         "(all 21 distinct displacements across Apple's 525 op17235/8 instances are multiples "
                         "of four) and nothing measures an unaligned one"
                         % ("sixteen-bit" if half else "thirty-two-bit", disp, step))
    t, ln = _halfslot_template(disp, wait_load, half) if template is None else (bytes(template), len(template))
    # THE FIELD IS SIGNED, WHICH A TEST FOUND AND NOT A GUESS. Apple's decoder reads 32768 back as -32768
    # and 65534 as -2, so anything above 32766 addresses memory BEFORE the buffer base rather than far
    # inside it - a silently wrong location, which is the class this project refuses rather than emits.
    if disp > HALFSLOT_MAX_DISP:
        raise ValueError("displacement %d: the field is SIGNED sixteen bits, so Apple's decoder reads this "
                         "as %d - a write before the buffer base. The largest expressible displacement is %d"
                         % (disp, disp - 0x10000, HALFSLOT_MAX_DISP))
    if disp > 0xFF and ln != 14: raise ValueError("displacement %d needs the fourteen-byte form" % disp)
    b, bit = HALFSLOT_WAIT
    if wait_load and ln < 10:
        raise ValueError("a load-wait needs byte9, which the eight-byte form does not have - the value must "
                         "go through an ALU first, as the word store's path does")
    if bool((t[0] >> 3) & 1) == bool(half):
        raise ValueError("the template's width bit byte0[3] is %s, which is the %s store - and this call asked "
                         "for the %s one" % ("set" if (t[0] >> 3) & 1 else "clear",
                                             "thirty-two-bit" if (t[0] >> 3) & 1 else "sixteen-bit",
                                             "sixteen-bit" if half else "thirty-two-bit"))
    if ((t[5] >> 2) & 3) != 1:
        raise ValueError("the template's sub-form is %d, not 01: every retained witness of this form carries "
                         "01" % ((t[5] >> 2) & 3))
    u = bytearray(t)
    _bits_put(u, STORE_SRC, src)
    _bits_put(u, HALFSLOT_DISP_LOW, disp & 0xFF)
    if ln == 14:
        _bits_put(u, HALFSLOT_DISP_HIGH, (disp >> 8) & 0xFF)
    if ln >= 10:
        u[b] = (u[b] & ~(1 << bit)) | ((1 if wait_load else 0) << bit)
    if const is not None:
        by, lo, w = STORE_CONST
        if const % 4: raise ValueError("store const %d is not a multiple of 4" % const)
        q = const // 4
        if not 0 <= q < (1 << w): raise ValueError("store const %d exceeds the field" % const)
        u[by] = (u[by] & ~(((1 << w) - 1) << lo)) | (q << lo)
    return bytes(u)


def decode_element_store(u):
    """The operands this side writes, read back out of the bytes - so a round-trip can be asserted. Reads
    both widths: `half` comes out of byte0[3] rather than being passed in."""
    d = dict(src=_bits_get(u, STORE_SRC), disp=_bits_get(u, HALFSLOT_DISP_LOW),
             subform=(u[5] >> 2) & 3, half=not ((u[0] >> 3) & 1))
    if len(u) > 13:
        d["disp"] |= _bits_get(u, HALFSLOT_DISP_HIGH) << 8
    b, bit = HALFSLOT_WAIT
    d["wait_load"] = ((u[b] >> bit) & 1) if len(u) > b else 0
    by, lo, w = STORE_CONST
    d["const"] = 4 * ((u[by] >> lo) & ((1 << w) - 1))
    return d


decode_halfslot_store = decode_element_store        # the name the half delivery uses; one decoder, both widths


def roundtrip_halfslot(u):
    d = decode_element_store(u)
    return encode_element_store(src=d["src"], disp=d["disp"], half=d["half"], const=d["const"],
                                wait_load=d["wait_load"], template=u) == bytes(u)


roundtrip_element = roundtrip_halfslot


def encode_halfvec_store(src, n, slot, const=None, template=None):
    """out[slot .. slot+n-1] <- r<425+src> .. r<425+src+n-1>, sixteen bits each, at fourteen bytes.

    A thin, named wrapper over encode_store with the retained half template: the only reason it exists is
    so the caller cannot pass a WORD template by accident and so the width bit is asserted rather than
    assumed. n=1 is refused for the same reason the word form refuses it - byte4[6:5]=0 is the k/k+3
    encoding, not a one-component store - and a single half element goes through store_at(width="half"),
    which is op17193 and has its own witness."""
    if n not in HALFVEC_OPCODES:
        raise ValueError("a half range store takes 2..4 components (n=1 is the k/k+3 encoding, and a "
                         "single half element is op17193 through store_at); got n=%r" % (n,))
    t = HALFVEC_SLOT_TEMPLATES[n] if template is None else bytes(template)
    b, bit = HALFVEC_WIDTH_BIT
    if (t[b] >> bit) & 1:
        raise ValueError("the template's width bit (byte%d[%d]) is set: that is a THIRTY-TWO-bit store, not a "
                         "half one - the sweep on round A's Q0 shows setting it turns op17220 into op17256"
                         % (b, bit))
    if len(t) != 14:
        raise ValueError("a half range store is emitted at fourteen bytes; this template is %d" % len(t))
    if ((t[5] >> 1) & 1):
        raise ValueError("the template's address-mode bit (byte5[1]) is set: that is the per-thread-index "
                         "family op17202/17211/17220, which is not lowered")
    return encode_store(src=src, n=n, slot=slot, template=t, const=const)


def decode_store(u):
    # the buffer const comes back too, so selfcheck can compare it against the rank asked for
    # slot = 64*byte13 + 2*byte7[4:0] + byte6[7].  CORRECTED 2026-09-04.
    # The old reading 2*byte7[6:0] + byte6[7] agreed on slots 100/110/120 and was wrong by 64 on
    # slot 130, which it decoded as 66 - byte7 bits 5 and 6 are structural (bit5 is always set,
    # bit6 always clear in the corpus sweep) and are not part of the value. The high bits live in
    # byte13 of the 14-byte form; the 8-byte form implies 0. ledger/g17-store-slot-corrected.toml
    hi = u[13] if len(u) > 13 else 0
    d = dict(src=_bits_get(u, STORE_SRC), n={0: 1, 1: 2, 2: 3, 3: 4}[(u[4] >> 5) & 3],
             slot=64 * hi + 2 * (u[7] & 0x1F) + ((u[6] >> 7) & 1))
    # byte9[5]: wait for a pending LOAD into the source register before storing.
    # ledger/g17-store-load-dependency.toml
    if len(u) > 9: d["wait_load"] = (u[9] >> 5) & 1
    d["subform"] = (u[5] >> 2) & 3
    by, lo, w = STORE_CONST
    d["const"] = 4 * ((u[by] >> lo) & ((1 << w) - 1))
    return d

def roundtrip_store(u):
    return encode_store(template=u, **decode_store(u)) == bytes(u)

def structural_store(template):
    return {i: (0xFF & ~OWNED_STORE.get(i, 0)) & template[i]
            for i in range(8) if (0xFF & ~OWNED_STORE.get(i, 0)) & template[i]}


# --- ALU, operand-mode aware -------------------------------------------------------------
# Added 2026-09-04 with the operand-slot model: byte1 and byte3 are ONE operand slot read
# either as an 8-bit immediate or as a 3-bit register index, selected by byte4[3:2].
# ledger/g17-milestone1-closed.toml
# byte8 bit5 is src2_keep - written in BOTH operand modes and previously undeclared, which
# made the highest-entropy bit in the family look unexplained. byte8 bits 0-1 and byte5 bit7
# are immediate-mode only, so this mask is the UNION over modes; tools/g17owned.py derives
# the per-mode truth automatically. ledger/g17-owned-mask-drift.toml
OWNED_ALU = {0: 0xB8, 1: 0x03, 2: 0x88, 3: 0xE2, 4: 0x2C, 5: 0x80, 6: 0x2F, 7: 0x38, 8: 0xEB, 9: 0xFF, 10: 0x27, 11: 0x01}   # byte10[2:0] + byte11[0] are the fused-shift scale
MODE_REG, MODE_IMM = 0, 1   # byte4[2] ALONE; byte4[3] is the inert liveness marker

# --- THE OPERAND SLOT ------------------------------------------------------------------------
# An alu.12 operand is ONE scattered bit-field, not two overlapping readings that have to be
# reconciled. Recovered 2026-09-04 by tools/g17fields.py, which correlates each instruction bit
# with the operand VALUES Apple's own decoder reports, over 6314 instances of 10279 and 6254 of
# 10282. The method's positive control is that it rediscovers, bit for bit, the immediate field
# this file already modelled - see g17fields._control.
#
#   slot = (register << 1) | half_selector      when the operand is a register
#   slot = the 8-bit immediate                  when it is an immediate
#
# That single rule replaces three separate hand-recovered facts: the 8-bit immediate spread, the
# 4-bit operand-B register, and byte1[0] as a half-register selector. They were all the same
# field read at different widths, which is why byte1[0] looked like an anomaly needing its own
# ledger entry - it is slot bit 0, and a register just does not use it.
#
# WHAT THIS CORRECTS. The destination was modelled as FIVE bits and is SEVEN; operand A's
# register was modelled as FOUR and is SEVEN. Apple's own shaders use r0..r116, so the compiler
# had 32 registers where the hardware offers 128, and every corpus instruction with a register
# above r15 in slot A was unencodable. byte2[7] alone was 5894 of the round trip's unexplained
# residue.
SLOT_A = ((1, 0), (1, 1), (3, 5), (3, 6), (3, 7), (5, 7), (8, 0), (8, 1))                  # 8 bits
SLOT_B = ((8, 6), (8, 7), (9, 0), (9, 1), (9, 2), (9, 3), (9, 4), (9, 5), (9, 6))          # 9 bits
# EIGHT bits, not seven. byte2[4] is the destination's bit 7, established by mutation - flipping
# it moves the printed destination by exactly 128 on every ALU and bitwise form. Registers run to
# r127 so the compiler always writes it as 0, but writing it is the point: left unwritten it was
# one of the "operand 0" bits frozen at a corpus value.
#
# THE FIELD IS EIGHT BITS AND APPLE USES SEVEN, and those are different claims. The ISA peer's
# census puts a number on the second: across 265,115 instructions of Apple's own code carrying a
# destination this project can decode, the highest is r125 and byte2[4] is never set once. That is
# what Apple EMITS. What the encoding can EXPRESS is settled here instead - by the mutation above,
# and by a standing regression case in which op554's four-byte movimm encodes r129 and reads it
# back. Neither says whether the hardware has those registers; nothing measured on either side
# does. Read "seven bits" as a fact about Apple's shaders, never about the file.
ALU_DEST = ((0, 4), (7, 3), (7, 4), (0, 7), (7, 5), (2, 7), (2, 3), (2, 4))                # 8 bits

# THE HAZARD WORD. Apple's printer renders operand 1 of every ALU and bitwise form as one large
# immediate; it is fourteen scattered bits, and their weights come from mutation:
#
#     bit  5  b4[3]     bit 24..28  b1[2..6]   the wait TAG
#     bit  6  b6[4]     bit 29      b7[7]
#     bit 30  b2[6]     bit 31      b0[3]      the ALU load-use wait (measured causally)
#     bit 33  b0[5]     bit 37      b4[5]      bit 41 b6[5]   bit 47 b6[7]
#
# bits 33, 37 and 41 are byte0[5], byte4[5] and byte6[5], the three this project measured causally
# INERT. The compiler writes the whole word from its own hazard state - zero when no load is
# outstanding - rather than inheriting whatever the template's kernel was waiting for.
# FOUR OF THE FOURTEEN ARE INVERTED, and they are exactly the three this project measured causally
# inert plus byte6[7]. The printed bit is the COMPLEMENT of the instruction bit, so writing the word
# naively sets them and the decoder reports 2^33 + 2^37 + 2^41 + 2^47 for a hazard of zero. The
# polarity is read off the templates directly: instruction bit against printed bit, and it is the
# same on op10279, op423 and op14391.
ALU_HAZARD = (((4, 3), 5, 0), ((6, 4), 6, 0),
              ((1, 2), 24, 0), ((1, 3), 25, 0), ((1, 4), 26, 0), ((1, 5), 27, 0), ((1, 6), 28, 0),
              ((7, 7), 29, 0), ((2, 6), 30, 0), ((0, 3), 31, 0),
              ((0, 5), 33, 1), ((4, 5), 37, 1), ((6, 5), 41, 1), ((6, 7), 47, 1))


def _hazard_put(u, value):
    for (byi, bi), w, inv in ALU_HAZARD:
        if byi < len(u):
            u[byi] = (u[byi] & ~(1 << bi)) | ((((value >> w) & 1) ^ inv) << bi)

def _slot_get(u, bits):
    return sum(((u[b] >> i) & 1) << j for j, (b, i) in enumerate(bits))

def _slot_put(u, bits, v):
    for j, (b, i) in enumerate(bits):
        u[b] = (u[b] & ~(1 << i)) | (((v >> j) & 1) << i)

# WHICH SLOT HOLDS WHAT, per Apple opcode. The family's byte6 gives the OPERATION but not the
# operand roles: 10279 and 10282 are both `a+b` with byte6 0xa3, and 11664 and 11666 are both
# `a-b` with byte6 0xa2, yet each pair differs in which slot carries the immediate. Since
# subtraction is not commutative that difference is the whole meaning:
#
#   opcode 11664   dest = imm(slot A) - reg(slot B)      `5u - x`
#   opcode 11666   dest = reg(slot A) - imm(slot B)      `x - 5u`
#   opcode 11667   dest = reg(slot A) - reg(slot B)      `x - y`
#
# read straight off the operand order Apple's decoder prints, with the Metal source stating which
# operand is the minuend. RESULT = SLOT A - SLOT B in all three, so the slots are ordered, and
# the compiler's old `sub` - which put the immediate in slot A and the register in slot B and so
# computed K - x - was inverted. That is the -10 that execution measured where 7 was predicted.
ALU_FORM = {
    10279: ("add", "imm", "reg"), 10282: ("add", "reg", "reg"),
    11664: ("sub", "imm", "reg"), 11666: ("sub", "reg", "imm"), 11667: ("sub", "reg", "reg"),
    10822: ("mul", "reg", "imm"), 10825: ("mul", "reg", "reg"),
    # The SHIFTS use the same two slots, so they need no new geometry - only their own opcodes
    # and their own length. `x << 3u` is not one of these: Apple strength-reduces small left
    # shifts into the add forms' fused scale (x << 3 is one add with scale 8), and only reaches
    # for a real shift instruction when the scale table cannot express it.
    14391: ("shl", "reg", "imm"), 14392: ("shl", "reg", "reg"),
    17013: ("shr", "reg", "imm"), 17014: ("shr", "reg", "reg"),
    # SATURATING add and subtract, named by the peer's isolation sweep. They use the same two
    # slots as the ordinary forms - checked instance by instance against the printed operands, not
    # assumed - so the geometry is shared and only the opcode differs.
    10239: ("addsat", "reg", "reg"), 11624: ("subsat", "reg", "reg"),
    # ARITHMETIC shift right is its own family, not a bit on the logical one - isa/g17-opmap.toml
    # records the two as separate opcodes from source-level kernels. Same two slots.
    16805: ("sar", "reg", "imm"),
}
# Sizes are per opcode too: the mul forms are FOURTEEN bytes. The compiler emitted twelve, and
# Apple's decoder read its 12-byte mul plus the first two bytes of the next instruction as one
# 14-byte instruction - the same truncation that made every authored taken branch fault.
ALU_FORM_SIZE = {10279: 12, 10282: 12, 11664: 12, 11666: 12, 11667: 12, 10822: 14, 10825: 14,
                 14391: 14, 14392: 14, 17013: 14, 17014: 14, 10239: 12, 11624: 12, 16805: 12}

def decode_alu(u):
    d = dict(dest=_slot_get(u, ALU_DEST),           # EIGHT bits - see ALU_DEST above
             src1=_slot_get(u, SLOT_B) >> 1,        # slot B as a register: (reg << 1) | half
             # OPERAND WIDTH. byte9[7] is src1's, byte8[3] is operand B's; 1 means 32-bit.
             # Recovered by controlled differential, not mutation - both are inert at the two
             # sites where inertness was measured, so they are structural, and they were the two
             # largest causes of ENCODE failure (54.6% and 37.5% of alu.12 re-encodes).
             # Within ONE kernel, register allocation held fixed: a 32-bit int add is (1,1) and
             # a 16-bit int add is (0,0). The discriminating cells are the mixed ones, and both
             # were preregistered and hit: 32-bit src1 with a real immediate operand B gives
             # (1,0), and (uint)s[i] + u[i] - a 16-bit source widened against a 32-bit register -
             # gives (0,1). ledger/g17-alu-operand-width.toml
             src1_w=(u[9] >> 7) & 1, srcb_w=(u[8] >> 3) & 1,
             # DESTINATION width, completing the triple: src1 byte9[7], operand B byte8[3],
             # dest byte3[1]; 1 means 32-bit. Third largest cause of ENCODE failure on Apple's
             # shaders (37.8% of alu.12) and previously never written - OWNED_ALU[3] was 0xE0.
             # The decisive cells hold SOURCE width fixed and vary only the destination, in one
             # kernel, distinguished by opcode so neither can be mistaken for the other:
             #   uint   wide   = (uint)s[i] * (uint)s[i+1]   MUL  src1 16-bit  dest 32  byte3 0x3a
             #   ushort narrow = (ushort)u[i] + (ushort)u[i+1] ADD src1 16-bit dest 16  byte3 0x18
             # Both preregistered, both hit. ledger/g17-alu-dest-width.toml
             dest_w=(u[3] >> 1) & 1,
             # THE OPCODE. byte6 is 0xA0 | op for this family; the low nibble selects the
             # operation. Mapped by sweeping thirty Metal expressions with runtime-loaded operands
             # (ledger/g17-alu-opcode-map.toml): 0 = a*b-c, 1 = a*b, 2 = a-b, 3 = a+b, 9 = mulhi,
             # a = popcount, b = clz, e and f = the division lowering. Owned rather than inherited,
             # so an authored instruction states its operation instead of copying one.
             op=u[6] & 0x0F,
             # THE LOAD-USE WAIT. byte0[3] is the single largest unexplained bit in the corpus -
             # 13330 instructions of the cross-template round trip's residue - and it is not a
             # register field, so correlation against operand values could never name it.
             #
             # A controlled compiler differential names it. Eleven kernels varying result reuse,
             # operand liveness, uniformity, position and register pressure leave it at 1; the ONE
             # cell that changes it is where the add's operands stop coming from a load:
             #
             #     s = x + y   with x, y loaded        byte0 = 0x2f    bit3 = 1
             #     p = i+7, q = i+9, s = p + q         byte0 = 0x37    bit3 = 0
             #     p = i+7, s = p + x  (one loaded)    byte0 = 0x2f    bit3 = 1
             #
             # and the corpus agrees from the other direction: over 188 objects it is set in 19.7%
             # of adds whose PREVIOUS instruction is a load (op12682) and in 0.0% of those
             # following another ALU. Mixed within an object in 75 of 188, so it is per
             # instruction, not a per-shader mode.
             #
             # This project already knew a load-use hazard exists - the compare refuses a loaded
             # operand because "the compare's wait mechanism is not recovered" - and this is that
             # mechanism on the ALU. ledger/g17-alu-load-use-wait.toml
             load_wait=(u[0] >> 3) & 1,
             # UNRESOLVED, and named for what it is rather than for a guess. byte4[5] is the last
             # bit blocking three of the four add-family opcodes. Five readings are eliminated
             # (ledger/g17-byte4-bit5-typed-hole.toml); the standing containment is the ISA
             # agent's, that it tracks the PREDECESSOR's identity - 283 of 284 adds following a
             # movimm carry 0, and 0 of 619 following a load do. It is exposed here so it can be
             # authored in both directions and tested, not because it is understood.
             b4_5=(u[4] >> 5) & 1,
             # ONE FIELD IN TWO PLACES. byte0[5] and byte6[5] are EQUAL in 7135 of 7135 op10279
             # and 7019 of 7031 op10282 (the twelve exceptions all carry byte6[6]), so Apple never
             # separates them and an instruction that separates them is off the manifold. They are
             # written together here for that reason. Meaning unresolved; the standing signal is
             # that an add following a movimm has them CLEAR in 286 of 306 while an add reading its
             # predecessor's result has them set in 3315 of 3329. ISA agent, g17context.py.
             b0_5=(u[0] >> 5) & 1,
             mode=(u[4] >> 2) & 1, live=(u[4] >> 3) & 1, keep=(u[10] >> 5) & 1,
             # fused shift-add address scale, a table not an arithmetic field:
             #   (b10[2],b10[0],b11[0]) = 010 x1, 100 x2, 110 x4, 001 x8, 000 x16
             # isa/g17-scalar-isa.toml alu.shiftadd.imm
             scale_code=(8 * ((u[10] >> 2) & 1) + 4 * ((u[10] >> 1) & 1)
                         + 2 * (u[10] & 1) + (u[11] & 1)) if len(u) > 11 else 0)
    d["src2_keep"] = not ((u[8] >> 5) & 1)      # meaningful in both modes; see encode_alu
    # SLOT A, read at the width the mode calls for. slot bit 0 is the half-register selector,
    # which is why a register is slot >> 1 and an immediate is the slot itself. The bit was
    # recovered on its own first (byte1[0], 105 of 2686 register-mode instructions have it set,
    # and clearing it made an authored operand read as 0 - ledger/g17-operand-b-lifetime.toml);
    # the slot model says WHY it exists, and the two agree.
    a = _slot_get(u, SLOT_A)
    d["src2_half"] = a & 1
    d["src1_half"] = _slot_get(u, SLOT_B) & 1    # the same bit for slot B: byte8[6]
    if d["mode"] == MODE_REG:
        d["src2"] = a >> 1                       # SEVEN bits, was modelled as four
    else:
        d["imm"] = a
    return d

def encode_alu(dest, src1, mode, template, imm=None, src2=None, live=0, keep=0,
               src2_keep=False, scale_code=None, src1_w=None, srcb_w=None, dest_w=None, op=None,
               src2_half=0, src1_half=0, load_wait=None, b4_5=None, b0_5=None, hazard=None):
    # RANGE CHECKS, from the slot model. Slot B holds (register << 1) | half so its register is
    # eight bits; slot A is eight bits total, so its register is seven.
    #
    # THE DESTINATION CAP IS CONSERVATIVE, NOT A FIELD WIDTH, and this comment used to say
    # otherwise. ALU_DEST is EIGHT bits - byte2[4] is its bit 7, established by mutation, and
    # op554's four-byte movimm has a standing regression case encoding r129 and reading it back. So
    # r0..r127 here is a choice: nothing this project has verified needs a destination above r127,
    # and the ISA peer's census finds Apple's own code never emits one (highest r125 across 265,115
    # instructions). Some forms DO reject the eighth bit - the unary and ten-byte families, measured
    # by bit-role sweep, see _UNARY_SRC10 below - and for those seven is the width. For this one it
    # is caution. Do not cite this cap as evidence about how deep the register file is; nothing
    # measured on either side of this project says.
    if not 0 <= dest <= 127: raise ValueError("alu dest r%d out of range (capped at r127; ALU_DEST is 8 bits)" % dest)
    if not 0 <= src1 <= 255: raise ValueError("alu src1 r%d out of range (8-bit field)" % src1)
    if mode == MODE_REG and src2 is not None and not 0 <= src2 <= 127:
        raise ValueError("alu operand A r%d out of range (7-bit field, r0..r127)" % src2)
    if mode == MODE_IMM and imm is not None and not 0 <= imm <= 255:
        raise ValueError("alu immediate %d out of range (8-bit slot)" % imm)
    u = bytearray(template)
    if src1_w is not None: u[9] = (u[9] & ~0x80) | ((src1_w & 1) << 7)   # operand widths
    if srcb_w is not None: u[8] = (u[8] & ~0x08) | ((srcb_w & 1) << 3)
    if dest_w is not None: u[3] = (u[3] & ~0x02) | ((dest_w & 1) << 1)
    if op is not None:     u[6] = (u[6] & 0xF0) | (op & 0x0F)
    if load_wait is not None: u[0] = (u[0] & ~0x08) | ((load_wait & 1) << 3)
    if b4_5 is not None:      u[4] = (u[4] & ~0x20) | ((b4_5 & 1) << 5)
    if b0_5 is not None:                                    # both positions, never separately
        u[0] = (u[0] & ~0x20) | ((b0_5 & 1) << 5)
        u[6] = (u[6] & ~0x20) | ((b0_5 & 1) << 5)
    _slot_put(u, ALU_DEST, dest)
    _slot_put(u, SLOT_B, (src1 << 1) | (src1_half & 1))
    u[4] = (u[4] & ~0x0C) | ((mode & 1) << 2) | ((live & 1) << 3)
    u[10] = (u[10] & ~0x20) | ((keep & 1) << 5)   # source-register lifetime
    if scale_code is not None and len(u) > 11:   # fused shift-add address scale
        u[10] = ((u[10] & ~0x07) | (((scale_code >> 3) & 1) << 2)
                 | (((scale_code >> 2) & 1) << 1) | ((scale_code >> 1) & 1))
        u[11] = (u[11] & ~0x01) | (scale_code & 1)
    # THE OPERAND-B SLOT IS BUILT WHOLE, never patched bit by bit.
    #
    # byte1, byte3, byte5[7] and byte8 together form ONE slot, read as an 8-bit immediate or as a
    # 4-bit register index, and the two readings OVERLAP. Writing only the bits the new reading
    # needs leaves the old reading's bits behind, producing an instruction that decodes correctly
    # and executes wrongly: an authored register-mode add inherited byte1[0] from an immediate
    # template and read 0 instead of its operand (preregistered 44, measured 7). So every bit of
    # the slot is cleared and rewritten from semantic state on every encode.
    # byte8[5] is part of the slot in BOTH modes, so it is always written from semantic state:
    #   register mode   0 keeps the operand-b register, 1 releases it after the read
    #   immediate mode  must be SET; clearing it makes the operand read as 0
    # ledger/g17-operand-b-lifetime.toml
    u[8] = (u[8] & ~0x20) | ((0 if src2_keep else 1) << 5)
    # SLOT A IS WRITTEN WHOLE, at the width the mode calls for, in both modes.
    #
    # The previous code wrote only part of it in register mode and left byte5[7] and byte8[1:0]
    # alone, on the evidence that clearing them halved the corpus round trip. That evidence was
    # real and the conclusion was wrong: those three bits are register bits 4, 5 and 6, so
    # clearing them without writing them destroyed the operand, while leaving them destroyed
    # nothing only because the compiler never allocated a register above r15.
    _slot_put(u, SLOT_A, imm if mode == MODE_IMM else ((src2 << 1) | (src2_half & 1)))
    # THE HAZARD WORD, written last so it subsumes load_wait/b4_5/b0_5 - those three are single
    # bits of the same packed operand, and writing the whole word is what makes the other eleven
    # semantic instead of inherited.
    if hazard is not None:
        _hazard_put(u, hazard)
    return bytes(u)


# THE MULTIPLY IS A MULTIPLY-ADD. op10822's last printed operand is an ADDEND, an 8-bit field
# scattered over three bytes, and it must be 0 for a pure multiply. Apple's own templates carry
# whatever addend the kernel they were cut from needed, so inheriting it silently changes the
# arithmetic: with the corpus's most common addend of 6 instead of 0, this project's novel kernel
# compiled, passed Apple's decoder and returned 118 instead of 32.
#
# The bit weights are from mutation - flip the bit, read the printed operand back - and every
# delta is a power of two, which is what makes this a field rather than a fit.
# ledger/g17-multiply-is-multiply-add.toml
# BYTE8[5] IS NOT A LIFETIME BIT IN EVERY FORM. On op10279, op14391 and op17013 it SELECTS THE
# OPCODE - clearing it turns add-immediate into op10285 and the shifts into their register forms,
# so `x << 8` shifts by whatever a register holds. Apple's decoder accepts the result and a round
# trip agrees, because the instruction is perfectly well formed; only execution disagrees. Tested
# by toggling the bit on a template of every ALU form and reading the decode back.
#
# None means the form has no writable slot-A lifetime bit, so the operand is never released there.
# That is conservative - a value kept alive too long costs a register, a value released too early
# is read as zero.
# For the three forms where byte8[5] selects the opcode, the slot-A lifetime is byte10[5]
# instead - the bit encode_alu calls src1's. Leaving it UNWRITTEN is not conservative: the
# shift templates come from kernels whose operand was dead, so they RELEASE, and the next
# reader gets zero.
# None means NO WRITABLE LIFETIME BIT IS KNOWN for this form's slot A, and the form is
# therefore treated as DESTRUCTIVE by the compiler: byte8[5] selects the opcode here, and
# byte10[5] was tested in both polarities and changes nothing. A value still live after one
# of these must be copied first (g17cc.copy_before_destructive).
# THE SLOT-A LIFETIME BIT, PER FORM, ALL MEASURED BY EXECUTION rather than transferred.
#
# byte8[5] is the bit for the add and multiply forms - the one recovered first - and it is NOT the
# bit for the subtract or the shifts, where byte4[2] governs the source and byte8[5] selects the
# opcode. Setting byte4[2] on a MULTIPLY breaks the multiply instead, so the split is real and not
# a naming choice. Each entry was established the same way: two programs that read one value
# through two different forms, in both orders, with the bit authored both ways.
#
#   1 RELEASES the source, 0 keeps it. Apple's templates carry whichever their donor kernel needed,
#   which is why `x & m` destroyed a live value and `x | m` did not - op423's template releases and
#   op13574's does not.
ALU_KEEP_A = {10279: None,
              14391: (4, 2), 14392: (4, 2), 17013: (4, 2), 17014: (4, 2),
              11664: (4, 2), 11666: (4, 2), 11667: (4, 2),
              # The saturating pair splits the same way the ordinary forms do: addsat keeps at
              # byte8[5] like add and multiply, subsat at byte4[2] like subtract and the shifts.
              # Verified separately - the default carried addsat and lost subsat's source, which a
              # random chain that read one value through a subsat and then a shift caught.
              11624: (4, 2),
              # sar sits with the other shifts.
              16805: (4, 2)}

# The ten-byte bitwise forms carry it at byte8[2], on all three opcodes.
BITWISE_KEEP = (8, 2)

ALU_ADDEND = {10822: ((10, 2), (12, 1), (12, 2), (12, 3), (8, 4), (12, 5), (10, 0), (12, 7))}


_ALU_KEEP_CARRIER = {}


def _alu_keep_carrier(opcode, release_at):
    """The KEEP carrier paired with `release_at`, or None if the two models do not line up.

    ALU_KEEP_A names one bit per opcode and calls it the lifetime. The authoring table models the
    same thing as an OPERAND with two carriers - value-bit 5 keeps, value-bit 4 releases. Where the
    bit ALU_KEEP_A names IS that operand's release carrier, the operand's keep carrier is the other
    half and writing only one of them leaves the operand in a state Apple never emits.
    """
    key = (opcode, tuple(release_at))
    if key in _ALU_KEEP_CARRIER:
        return _ALU_KEEP_CARRIER[key]
    out = None
    try:
        from agxforge.g17 import auth as g17auth
        for i in g17auth.fields(opcode):
            car = g17auth.carriers(opcode, i)
            if g17auth.LIFETIME_KEEP in car and g17auth.LIFETIME_RELEASE in car:
                rel = tuple(car[g17auth.LIFETIME_RELEASE][0][:2])
                if rel == tuple(release_at):
                    out = tuple(car[g17auth.LIFETIME_KEEP][0][:2])
                    break
    except Exception:
        out = None
    _ALU_KEEP_CARRIER[key] = out
    return out


def encode_alu_form(opcode, dest, a, b, template, keep_a=None, keep_b=None, addend=None,
                    hazard=None):
    """Author an alu.12 from its OPERANDS, with the slot roles its OPCODE fixes.

    `a` and `b` are slot A and slot B, in that order, each ("reg", n, half) or ("imm", v). Every
    form measured computes  dest = A op B, so the order is the semantics: for a non-commutative
    operation, putting the operands in the wrong slots computes the wrong thing while decoding
    perfectly. That is exactly how the old `sub` came to negate.

    This is the form-aware path. encode_alu is the older one, which selects the immediate/register
    reading from byte4[2]; that bit is the mode only for the add opcodes, and reading 11666 through
    it gives the immediate as a register index. A form is an OPCODE plus its slot roles, not a
    family name and a mode bit.

    Bits outside dest, the two slots and the operation nibble are INHERITED from the template and
    are not yet authored: the operand modifiers (byte8[5] and byte10[5] in the add forms, byte4[2]
    in the sub forms), the three width bits and the fused-shift scale. tools/g17frozen.py reports
    them against the corpus.
    """
    if opcode not in ALU_FORM:
        raise ValueError("no slot model for opcode %d; isa/g17-opmap.toml lists the measured forms"
                         % opcode)
    name, ra, rb = ALU_FORM[opcode]
    for slot, role, got in (("A", ra, a[0]), ("B", rb, b[0])):
        if role != got:
            raise ValueError("opcode %d (%s) takes %s in slot %s, given %s"
                             % (opcode, name, role, slot, got))
    def value(o, width):
        if o[0] == "imm":
            if not 0 <= o[1] <= 0xFF:
                raise ValueError("alu immediate %d does not fit the 8-bit slot" % o[1])
            return o[1]
        n, half = o[1], (o[2] if len(o) > 2 else 0)
        if not 0 <= n < (1 << (width - 1)):
            raise ValueError("alu register r%d does not fit a %d-bit slot" % (n, width))
        return (n << 1) | (half & 1)
    if not 0 <= dest <= 127:
        # the same conservative cap as encode_alu, for the same reason - see its range-check note
        raise ValueError("alu dest r%d out of range (capped at r127; ALU_DEST is 8 bits)" % dest)
    u = bytearray(template)
    _slot_put(u, ALU_DEST, dest)
    _slot_put(u, SLOT_A, value(a, len(SLOT_A)))
    _slot_put(u, SLOT_B, value(b, len(SLOT_B)))
    # OPERAND LIFETIME, BY SLOT ROLE RATHER THAN BY ARGUMENT NAME.
    #
    # byte8[5] belongs to SLOT B and byte10[5] to SLOT A. That is what encode_alu already says -
    # "byte8[5] is part of the slot in BOTH modes ... immediate mode must be SET; clearing it makes
    # the operand read as 0" - and this function had it the other way round, writing keep_a into
    # byte8[5]. For the add forms, whose slot A is the immediate, the two happen to coincide and
    # nothing showed. For a form whose slot B is the IMMEDIATE - shl reg,imm, shr reg,imm - keeping
    # slot A alive cleared byte8[5] and turned the instruction into the REGISTER form: op14391
    # became op14392, and `x << 8` shifted by whatever was in a register. Apple's decoder accepts
    # it, the round trip agrees, and only execution disagrees. A randomly generated chain that
    # shifted a value still live afterwards found it on the first seed with a non-degenerate
    # answer. ledger/g17-lifetime-bits-belong-to-slots.toml
    ka = ALU_KEEP_A.get(opcode, (8, 5))
    if keep_a is not None and ka is not None:
        u[ka[0]] = (u[ka[0]] & ~(1 << ka[1])) | ((0 if keep_a else 1) << ka[1])
        # BOTH CARRIERS, OR THE OPERAND SAYS TWO THINGS AT ONCE. The bit above is the RELEASE
        # carrier of a modifier operand whose value is 32 (keep) or 16 (release). Writing only it
        # left the KEEP carrier at whatever the template held, and for a template carrying keep
        # that made `release` come out as 32+16 = 48 - a value Apple emits in none of its 682
        # op11666 instances, nor anywhere in the compiled system shaders. op11624 had it too.
        #
        # The pair is DERIVED, not tabulated: the keep carrier is taken from the same operand whose
        # release carrier is already the position ALU_KEEP_A names, so a mismatch between the two
        # models leaves this untouched rather than writing a bit on a guess.
        kb = _alu_keep_carrier(opcode, ka)
        if kb is not None:
            u[kb[0]] = (u[kb[0]] & ~(1 << kb[1])) | ((1 if keep_a else 0) << kb[1])
    if keep_b is not None and len(u) > 10:
        u[10] = (u[10] & ~0x20) | ((0 if keep_b else 1) << 5)
        # THE SAME PAIRING FOR OPERAND B. byte10[5] is a release carrier like byte4[2] is, and
        # leaving its partner at the template's value gave op11624 the same 48 that op11666 had.
        kb2 = _alu_keep_carrier(opcode, (10, 5))
        if kb2 is not None:
            u[kb2[0]] = (u[kb2[0]] & ~(1 << kb2[1])) | ((1 if keep_b else 0) << kb2[1])
    if hazard is not None:
        _hazard_put(u, hazard)
    if opcode in ALU_ADDEND:
        _bits_put(u, ALU_ADDEND[opcode], 0 if addend is None else addend)
    elif addend:
        raise ValueError("opcode %d has no measured addend field" % opcode)
    return bytes(u)


# --- the BITWISE forms, class b, 10 bytes ------------------------------------------------------
# and/or/xor against an 8-bit immediate. Recovered 2026-09-04 by tools/g17fields.py over 1391
# instances of op423, with the operation and the mask stated by the Metal source: `x & 15u`
# produces op423 with the decoder reporting imm 15, and every mask a peer session had catalogued
# as unexplained - 4, 15, 60, 195, 255 - reproduces from a source-level AND.
#
# This retires the `bitwise` blocker, which said the operands "are not in the instruction at all":
# six source expressions and two destinations had left the recovered bytes unchanged. They did,
# because the form being measured was chosen by family name and was the wrong one. The operands
# are in these bytes, and Apple's decoder says where.
#
# The DESTINATION is the same 7-bit field as the 12-byte family, and the IMMEDIATE is the same
# slot B, low 8 bits. Only the register source has its own geometry.
BW_SRC = ((1, 1), (5, 6), (5, 7), (3, 1), (3, 3), (3, 4), (3, 5))      # 7 bits
BW_IMM = SLOT_B[:8]                                                     # b8[6] b8[7] b9[0..5]
BITWISE_FORM = {423: "and", 13574: "or", 17770: "xor"}

# THE FOUR-BYTE MOVE-IMMEDIATE, op554. Its destination is SIX bits - r0..r63 - established by
# mutation: byte0[4..7] then byte2[6] and byte2[7], weights 1, 2, 4, 8, 16, 32, and nothing above.
# That is a real allocation constraint, not a modelling gap: a value materialised by this form
# cannot live above r63, which is why growing a tensor kernel has to place its new accumulator
# below that line and move the tiles that were there.
MOVIMM4_DEST = ((0, 4), (0, 5), (0, 6), (0, 7), (2, 6), (2, 7))


# THE SIXTEEN-BIT ZERO MOVE (handoff 10ag; integration's d076d651): op555 at FOUR bytes, the largest support
# gap on the frontier at 761 of 6,594 corpus programs. What it is, measured rather than assumed:
#   * 35,531 corpus instances, byte1 = 0 on ALL of them and the destination ALWAYS in a sixteen-bit file
#   * the source-faithful witness is `device half *out; out[gid.x] = 0.0h` (results/g17-movimm4-roundB-*/W0),
#     whose op555 is exactly `13 00 00 00` writing r426 and feeding a half store
#   * the ZEROING reading is confirmed and the small-value reading refuted by W3: the same source with 1.0h
#     emits op11843 at EIGHT bytes carrying imm:15360 = 0x3C00 = half(1.0), so the value-carrying sibling is a
#     different opcode and this form moves zero only
#   * W2 (a packed half2 of zeroes) uses op554 - the 32-bit zero - so a packed pair is not this form either
# The destination is six bits, 425-based, located by a decoder sweep of two witnesses; byte3[0] selects the
# OTHER sixteen-bit file (the 281-based one: setting it turns r428 into r284, a 144 step rather than a register
# step), and this encoder refuses that file because no source of ours reaches it.
MOVIMM16_ZERO_TEMPLATE = bytes.fromhex("13000000")       # W0 +4: r426 <- 0, feeding a half store
MOVIMM16_ZERO_DEST = ((0, 4), (0, 5), (0, 6), (0, 7), (2, 6), (2, 7))
MOVIMM16_FILE_BIT = (3, 0)                               # 0 = the 425-based file (every witness of ours), 1 = 281-based


def encode_movimm16_zero(dest, imm=0, template=MOVIMM16_ZERO_TEMPLATE):
    """op555/4: r<425 + dest> <- 0. `imm` exists only to be refused: the value-carrying sibling is op11843/8."""
    if imm: raise ValueError("op555 at four bytes moves ZERO; the value form is op11843 at eight bytes (witness W3 carries 0x3C00 = half(1.0)), which this encoder does not author")
    if not 0 <= dest <= 63: raise ValueError("op555's destination is six bits, 425-based; r%d does not fit" % dest)
    u = bytearray(template); _bits_put(u, MOVIMM16_ZERO_DEST, dest)
    b, bit = MOVIMM16_FILE_BIT
    if (u[b] >> bit) & 1: raise ValueError("this template selects the 281-based sixteen-bit file, which no witness of ours writes")
    return bytes(u)


def decode_movimm16_zero(u):
    b, bit = MOVIMM16_FILE_BIT
    return dict(dest=_bits_get(u, MOVIMM16_ZERO_DEST), file_281=bool((u[b] >> bit) & 1))


def encode_movimm4(dest, template, imm=0):
    if imm: raise ValueError("only the zeroing form is modelled; imm=%d is not" % imm)
    if not 0 <= dest <= 63:
        raise ValueError("op554's destination is six bits; r%d does not fit" % dest)
    u = bytearray(template)
    _bits_put(u, MOVIMM4_DEST, dest)
    return bytes(u)


def decode_movimm4(u):
    return dict(dest=_bits_get(u, MOVIMM4_DEST))


# --- the UNARY integer forms -----------------------------------------------------------------
#
# not, clz and reverse take one register and write one, and they reuse fields already recovered:
# the destination is ALU_DEST and the source is SLOT_A carrying (register << 1), exactly as in the
# ALU forms. Established by mutation - every delta on both operands is a power of two and the bit
# positions coincide with the existing field maps rather than being fitted to them.
#
# popcount (op465) is NOT here: its source sits at byte3[1], byte3[3..6] and byte5[6..7], a
# different layout, and a layout is not carried across forms.
# The source is SLOT_A's shape - (register << 1) - but the eight-byte form has no byte8, so its
# top two bits move to byte6. Per form, never shared.
_UNARY_SRC10 = ((1, 1), (3, 5), (3, 6), (3, 7), (5, 7), (8, 0), (8, 1), (8, 2))
_UNARY_SRC8 = ((1, 1), (3, 5), (3, 6), (3, 7), (5, 7), (6, 0), (6, 1))
# op9986 is NOT clz. The peer named it from an isolation kernel containing only Metal's
# clz(), and execution says it returns 6 for 109 - the index of the highest set bit, where
# clz(109) is 25. So Metal's clz lowers to this plus a subtract from 31, and the opcode
# itself is "position of the most significant set bit". Named msb here and reported back.
# THE UNSIGNED-INTEGER TO FP32 CONVERSION, op11179 at ten bytes (cvt.i2f).
#
# THE FORM IS APPLE'S, AND SO IS THE TEMPLATE. 2,641 instances across 126 of the 6,594 corpus
# programs; the template below is one of them, recorded by identity rather than constructed:
#
#     host   ds_setup_indirect_update_mapping_42_setup_indirect_update_mapping  at +0x152
#     bytes  378000022a80af120800
#     prints 11179 reg:110 imm:32 imm:4 imm:0 reg:105 imm:32
#
# and (32, 4, 0, 32) is the modal operand set - operand 1 is 32 in 587 of 694 sampled instances,
# operands 2/3 are 4/0 in 496, operand 5 is 32 in 342.
#
# THE DESTINATION IS THE ORDINARY ALU DESTINATION, verified rather than assumed: a bit-role sweep
# of all eighty bits through Apple's decoder gives operand 0's eight bits at exactly ALU_DEST's
# positions, in that order.
#
# THE SOURCE FIELD IS A THIRD VARIANT, which is why it is written out here. The ten float unaries
# split into _UNARY_SRC10 and a shifted pair; this form is neither: it takes byte3[4] where
# _UNARY_SRC10 takes byte3[5], and byte3[6] at weight 16 where that map uses byte5[7]. Assuming
# the family would have written the wrong operand, which is the mistake the transcendental comment
# above already warns about.
#
# WHAT IS NOT WRITTEN. Operands 1, 2, 3 and 5 stay exactly as the witness has them. Operand 1's
# fifteen bits ARE locatable by the same sweep, but what the field MEANS is unmeasured, and 32
# appearing there is not evidence that it is the KEEP the lifetime family writes elsewhere. Operand
# 5 is two located bits with three witnessed values and equally unmeasured semantics. So this
# encoder writes the two REGISTERS and nothing else, and UNARY_KEEP carries None for this opcode so
# no lifetime request can reach those bits.
_CVT_I2F_SRC10 = ((1, 1), (3, 4), (3, 5), (3, 7), (3, 6), (8, 0), (8, 1), (8, 2))
CVT_I2F_TEMPLATE = bytes.fromhex("378000022a80af120800")
CVT_I2F_OPCODE = 11179
CVT_I2F_WITNESS = ("ds_setup_indirect_update_mapping_42_setup_indirect_update_mapping", 0x152,
                   "operands 1/2/3/5 = 32/4/0/32, semantics unmeasured")

# OPERAND 2 IS SIGNEDNESS, and at ten bytes its one carrier is byte8[6]: 4 (clear, the template's)
# converts UNSIGNED, 5 (set) SIGNED, both rounding to nearest even. Measured by dispatching the same
# program at both values against eight rivals, one fitting each (tools/g17cvtsignreceipt.py, ledger
# g17-cvt-operand-two-is-signedness). Apple's requantization witness, a signed int32 source,
# carries 5.
CVT_I2F_SIGNED_BIT = (8, 6)

UNARY_FORM = {11190: ("not", 8, _UNARY_SRC8),
              9986: ("msb", 10, _UNARY_SRC10),
              14047: ("reverse", 10, _UNARY_SRC10),
              CVT_I2F_OPCODE: ("cvt.i2f", 10, _CVT_I2F_SRC10)}


def decode_unary(u, opcode):
    return dict(dest=_slot_get(u, ALU_DEST), src=_bits_get(u, UNARY_FORM[opcode][2]))


# The source lifetime is byte4[2], the same bit as the subtract and the shifts - found by
# DIFFING two templates of the same family that behaved differently: clz kept its source alive and
# reverse destroyed it, and the two templates differ in byte4[2] and byte6[1]. Authoring both in
# both polarities settles which: byte4[2]=0 keeps, and byte6[1] breaks the operation itself.
# PER FORM, like everything else here. The ten-byte forms carry it at byte4[2] - the subtract and
# shift bit - and the eight-byte `not` at byte2[5]. On `not`, byte4[2] also keeps the value alive
# but changes what the instruction computes, which is why the sweep tests the RESULT as well as
# the survival: a bit that preserves one and breaks the other is not the lifetime.
# op9986 HAS NO WRITABLE LIFETIME BIT. byte4[2] is the bit for reverse - authored in both
# polarities with the value's survival AND the result as the observable - but on op9986 clearing
# it turns the SOURCE OPERAND INTO AN EXPRESSION rather than a register, so the instruction reads
# an address where a register was meant and the next reader gets nothing.
#
# It was missed because msb does not destroy its source with the template's value, so the
# polarity was never exercised. The rule this session wrote for `not` - a sweep must check the
# RESULT as well as the survival - applies just as much to a bit one never had a reason to write.
# The template keeps, so not writing it is correct and conservative.
UNARY_KEEP = {11190: (2, 5), 9986: None, 14047: (4, 2), CVT_I2F_OPCODE: None}


def encode_unary(opcode, dest, src, template, hazard=None, keep=None, signed=None):
    if opcode not in UNARY_FORM:
        raise ValueError("no unary model for opcode %d" % opcode)
    name, ln, srcmap = UNARY_FORM[opcode]
    if len(template) != ln:
        raise ValueError("opcode %d (%s) is %d bytes, template is %d" % (opcode, name, ln, len(template)))
    if not 0 <= dest <= 255: raise ValueError("unary dest r%d out of range" % dest)
    if src >> len(srcmap): raise ValueError("unary source r%d does not fit a %d-bit field"
                                            % (src, len(srcmap)))
    u = bytearray(template)
    _slot_put(u, ALU_DEST, dest)
    _bits_put(u, srcmap, src)
    if hazard is not None:
        _hazard_put(u, hazard)
    kb = UNARY_KEEP[opcode]
    if keep is not None and kb is not None:
        u[kb[0]] = (u[kb[0]] & ~(1 << kb[1])) | ((0 if keep else 1) << kb[1])
    if signed is not None:
        if opcode != CVT_I2F_OPCODE:
            raise ValueError("opcode %d has no signedness field" % opcode)
        i, bit = CVT_I2F_SIGNED_BIT
        u[i] = (u[i] & ~(1 << bit)) | (int(bool(signed)) << bit)
    return bytes(u)


# --- the FLOAT UNARY forms: the transcendental unit and the rounding modes ---------------------
#
# One ten-byte shape covers ten operations. The peer swept byte6[2:0] through Apple's decoder from
# a sqrt encoding and cross-checked the codes against what Apple's compiler emits for each Metal
# builtin - five of five: exp2 a5, log2 a4, sqrt a2, rsqrt a3, and the sine primitive a6
# (isa/g17-transcendental-unit.toml). Reading the sw-f_* corpus for the single-instruction
# lowerings adds the fourth axis: with byte6 = a0 the unit ROUNDS, and byte8[6:7] picks the mode.
#
#     byte6  a0 round   a1 recip   a2 sqrt   a3 rsqrt   a4 log2   a5 exp2   a6 trig   a7 rejected
#     byte8  0x30 rint    0x70 floor    0xb0 ceil    0xf0 trunc      - and each step is +16 on
#                                                                     the opcode id, in that order
#
# code 6 is NOT sin: it is the one instruction common to sin, cos, sinpi, cospi and both the fast::
# and precise:: forms, and what separates sine from cosine is op9710 against op9711 later in a
# twelve-instruction sequence. Named `trig`, and this backend does not select it, because what it
# computes alone is not established - the same care op9986 needed when it turned out not to be clz.
#
# THE GEOMETRY IS THE INTEGER UNARY GEOMETRY, which is why there is no new field map here: a
# bit-role sweep of a sqrt witness through Apple's decoder gives the destination at ALU_DEST's
# first seven bits and the source at _UNARY_SRC10, exactly. (The eighth destination bit, b2[4], is
# rejected by this form, so the destination is r0..r127.) The decoder prints a register operand as
# its absolute MCRegister id, 105 above the field value, which is why the field and the printed
# number differ by a constant.
# THE SOURCE FIELD IS NOT IN THE SAME PLACE FOR ALL TEN, and the difference is visible in the
# templates: the eight that carry byte3 low nibble 8 put it at _UNARY_SRC10, while log2 and exp2
# carry low nibble a and shift three of its bits. Solved per template by flipping every one of the
# eighty bits through Apple's decoder and reading which printed operand moved by which power of
# two - not assumed from the family, which is the mistake the peer's own caution warns about.
_TRANS_SRC_A = ((1, 1), (3, 5), (3, 6), (3, 7), (5, 7), (8, 0), (8, 1), (8, 2))   # = _UNARY_SRC10
_TRANS_SRC_B = ((1, 1), (3, 4), (3, 5), (3, 7), (3, 6), (8, 0), (8, 1), (8, 2))   # log2, exp2
TRANS_FORM = {3658: ("recip", _TRANS_SRC_A), 3770: ("rint", _TRANS_SRC_A),
              3786: ("floor", _TRANS_SRC_A), 3802: ("ceil", _TRANS_SRC_A),
              3818: ("trunc", _TRANS_SRC_A), 3850: ("rsqrt", _TRANS_SRC_A),
              3946: ("trig", _TRANS_SRC_A), 3978: ("rsqrt2", _TRANS_SRC_A),
              1272: ("exp2", _TRANS_SRC_B), 2570: ("log2", _TRANS_SRC_B)}
# op3978 IS NOT sqrt. The peer named it from Apple's `sqrt(x)` lowering, which is this instruction
# followed by a multiply - and the multiply is not rounding, it is the algorithm: sqrt(x) is
# x * rsqrt(x). Executed on three inputs, op3978 returns the RECIPROCAL square root:
#     2.75 -> 0.60302269   16 -> 0.25   0.25 -> 2
# so it is renamed here. That is the third name this project has had to correct by execution after
# an isolation sweep - op9986 was called clz and returns the index of the highest set bit - and the
# pattern is always the same: isolation names what Apple SELECTS for a source construct, which is
# only the instruction's meaning when the lowering is one instruction long.
# WHAT SEPARATES op3978 FROM op3850 IS DENORMAL HANDLING, established by execution 2026-09-09.
# Both were run on four inputs through spike/accel/re/opsem.py, op998/fadd rediscovered as the
# control in the same session:
#
#     input                    op3850      op3978      1/sqrt(x)
#     2.5                      0.632456    0.632456    0.632456
#     -3.75                    NaN         NaN         NaN
#     2.10e-44  (denormal)     +inf        1.0         6.90e+21
#     1.40e-45  (denormal)     +inf        1.0         2.67e+22
#
# op3850 FLUSHES A DENORMAL INPUT TO ZERO - 1/sqrt(0) is +inf - and op3978 returns exactly 1.0 for
# one. Two independent denormals give the same split, and the two agree everywhere else tested.
#
# SO "rsqrt" DESCRIBES NEITHER OF THEM COMPLETELY. It is right on normal positive inputs and wrong
# below the normal range, which is the fourth time an isolation name has turned out to be a partial
# description here. A checker that asserts 1/sqrt(x) over the whole domain will disagree with the
# silicon on denormals, and a caller that needs a defined answer there has to choose the opcode
# deliberately rather than take whichever the name suggests.
# The destination is ALU_DEST's first seven bits in all ten. The eighth, b2[4], is a field bit on
# eight of them and REJECTED by recip and sqrt, so the backend uses seven everywhere rather than
# a destination range that depends on which function is being applied.
TRANS_DEST = ALU_DEST[:7]
# THE SOURCE FIELD IS EIGHT BITS WIDE AND ITS TOP IS NOT ALL REACHABLE. Authoring every value
# 0..255 onto each of the ten templates and reading it back through Apple's decoder: 0..143 come
# back as the register asked for, and 144..255 are REJECTED outright, identically on all ten. So
# the admitted source range is r0..r143 - 128 registers and sixteen more - and the encoder refuses
# the rest rather than emitting bytes the decoder will not take. spike/accel/re/fpunary.py holds
# both directions of the check, the range and the rejection above it.
# --- THE EIGHT-BIT FLOAT IMMEDIATE ---------------------------------------------------------
# A float instruction's second source can be a constant carried IN the instruction, and the eight
# bits that carry it are a miniature float:
#
#     bit 7    sign
#     bits 6:4 exponent          0 selects a subnormal row, no implicit leading one
#     bits 3:0 mantissa
#     value = sign * (mantissa + 16*(exponent != 0)) * 2**(exponent-1 if exponent else 0) / 64
#
# so the reachable set is 1/64 apart from 0 to 1/4, then doubling: ..., 31 and -31 at the ends, 0
# and -0 at 0x00 and 0x80. MEASURED, not inferred: fifteen values swept through op775 at x=0.25 fix
# the format, and then twelve more chosen to be OUTSIDE that fit - including three exponent rows the
# fit never saw - are predicted exactly, as are six through op1000 at f32. The immediate is a
# property of the source encoding, not of the datapath.
# ledger/g17-the-one-source-float-form.toml
def float_imm_value(code):
    """The float that the eight-bit immediate `code` spells."""
    e, m = (code >> 4) & 7, code & 15
    return ((-1.0 if code & 0x80 else 1.0) * (m + (16 if e else 0))
            * (2.0 ** ((e - 1) if e else 0)) / 64.0)

def float_imm(x, nearest=False):
    """The code that spells x EXACTLY, or None - because a constant the compiler rounded without
    being asked is a silently wrong program. nearest=True asks for the closest code instead."""
    x = float(x)
    best, err = None, None
    for c in range(256):
        v = float_imm_value(c)
        if v == x and (x != 0.0 or (c == 0x80) == (str(x)[0] == "-")): return c
        d = abs(v - x)
        if err is None or d < err: best, err = c, d
    return best if nearest else None

TRANS_SRC_MAX = 143
# THE SOURCE LIFETIME IS AN OPERAND, not a hidden bit, and it is Apple's decoder operand 3.
# Found by differencing two kernels that differ only in whether the value is read again:
#
#     f[200] = rint(p);                    3770 ... reg:105 imm:16    byte8[5]=1
#     f[200] = rint(p); f[201] = floor(p); 3770 ... reg:106 imm:32    byte1[7]=1   <- p read again
#                                          3786 ... reg:106 imm:16    byte8[5]=1   <- last read
#
# So operand 3 = 32 KEEPS the source and 16 RELEASES it, and the two values are one bit each:
# value bit 5 at byte1[7] and value bit 4 at byte8[5]. Both live in the same place for all ten
# opcodes even though the source REGISTER field does not.
#
# This was not a bit anyone would have gone looking for. It was found because a program that read
# one value through nine of these operations got the first two answers right and then read zero
# seven times - the failure the integer forms' lifetime bits produced twice before.
TRANS_KEEP, TRANS_RELEASE = (1, 7), (8, 5)


def decode_trans(u, opcode):
    if opcode not in TRANS_FORM:
        raise ValueError("no float-unary model for opcode %d" % opcode)
    return dict(dest=_slot_get(u, TRANS_DEST), src=_bits_get(u, TRANS_FORM[opcode][1]))


def encode_trans(opcode, dest, src, template, keep=None):
    """dest <- f(src), one instruction, for the ten operations TRANS_FORM names.

    `keep` writes the SOURCE LIFETIME - Apple's decoder operand 3 - from real liveness: True when
    the source is read again later, False when this is its last read. Leaving it None carries the
    template's own value, which is only right when the template came from a kernel with the same
    liveness, and that is exactly the assumption that made nine correct instructions produce seven
    zeros.

    Operand 1, the destination-side modifier, is still the template's. It is the last field of
    this form the backend does not author.
    """
    if opcode not in TRANS_FORM:
        raise ValueError("no float-unary model for opcode %d" % opcode)
    name, srcmap = TRANS_FORM[opcode]
    if len(template) != 10:
        raise ValueError("opcode %d (%s) is 10 bytes, template is %d" % (opcode, name, len(template)))
    if not 0 <= dest <= 127:
        raise ValueError("float-unary dest r%d is outside the seven-bit field" % dest)
    if not 0 <= src <= TRANS_SRC_MAX:
        raise ValueError("float-unary source r%d is outside the source's admitted range" % src)
    u = bytearray(template)
    _slot_put(u, TRANS_DEST, dest)
    _bits_put(u, srcmap, src)
    if keep is not None:
        for (by, bi), v in ((TRANS_KEEP, 1 if keep else 0), (TRANS_RELEASE, 0 if keep else 1)):
            u[by] = (u[by] & ~(1 << bi)) | (v << bi)
    return bytes(u)


def encode_bitwise_imm(opcode, dest, src, imm, template, hazard=None, keep=None):
    if opcode not in BITWISE_FORM:
        raise ValueError("no bitwise model for opcode %d" % opcode)
    if not 0 <= imm <= 0xFF: raise ValueError("bitwise immediate %d does not fit 8 bits" % imm)
    if not 0 <= dest <= 127: raise ValueError("bitwise dest r%d out of range" % dest)
    if not 0 <= src <= 127: raise ValueError("bitwise source r%d out of range" % src)
    u = bytearray(template)
    _slot_put(u, ALU_DEST, dest)
    _slot_put(u, BW_SRC, src)
    _slot_put(u, BW_IMM, imm)
    if hazard is not None:
        _hazard_put(u, hazard)
    if keep is not None:
        u[BITWISE_KEEP[0]] = ((u[BITWISE_KEEP[0]] & ~(1 << BITWISE_KEEP[1]))
                              | ((0 if keep else 1) << BITWISE_KEEP[1]))
    return bytes(u)

# --- the FOUR-BYTE bitwise form, three registers ----------------------------------------------
# The same three operations with both operands in registers, and it is the encoding Apple
# normally uses: 1070 instances of xor, 245 of or, 117 of and, against 11, 19 and 131 for the
# 10-byte form. Solved by tools/g17fields.py with ZERO unexplained bits over those instances -
# every varying bit is one of the three register fields.
#
# NOTE THE DIFFERENT GEOMETRY. This form is not the 10-byte one truncated: its destination is five
# bits in byte0 and byte2 rather than the seven-bit field the 12-byte family uses, and its two
# sources are contiguous runs in byte1 and byte3. A form is an opcode plus its own field table.
# SIX BITS, NOT FIVE. b2[7] was missing, and the narrow confinement on bitwise.reg is what hid it:
# this side never allocated a bitwise destination above r15, so a field that stops at r31 was never
# reached. Apple does reach it - 46 of its 2,188 four-byte bitwise instances carry a destination
# above r31, and every one of them DECODED WRONG through the five-bit map (r35 read back as r3).
# With b2[7] as bit 5 the map reads 2,188 of 2,188 correctly, across op424, op13575 and op17771.
# All three templates carry b2[7] clear, so writing the new bit changes no emitted byte for a
# destination at or below r31. Measured 2026-09-12 against the corpus.
BW_R_DEST = ((0, 4), (0, 5), (0, 6), (0, 7), (2, 6), (2, 7))                    # 6 bits
BW_R_SRCA = ((1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6))            # 6 bits
BW_R_SRCB = ((3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6))            # 6 bits
# THE OPERATION IS byte2[2:0] AND IS AUTHORED, not inherited. Across the corpus each opcode takes
# four different byte2 values - op424 uses 0x32, 0x12, 0x22 and 0x2A - and the low three bits are
# CONSTANT within an opcode and different between them:
#
#     and 2      or 4      xor 3
#
# This is here because leaving it inherited produced a wrong answer on hardware. An `and` built on
# xor's template returned 6 where 8 was predicted - which is the XOR of the same operands - so the
# instruction carried out the template's operation and ignored the opcode it was labelled with.
# A preregistered failure that the field model would never have shown, because every register
# field was correct. ledger/g17-bitwise-reg-selector-was-inherited.toml
BW_R_OP = ((2, 0), (2, 1), (2, 2))
BITWISE_REG_FORM = {424: "and", 13575: "or", 17771: "xor"}
BITWISE_REG_SEL = {"and": 2, "or": 4, "xor": 3}

def encode_bitwise_reg(opcode, dest, a, b, template):
    if opcode not in BITWISE_REG_FORM:
        raise ValueError("no 4-byte bitwise model for opcode %d" % opcode)
    for n, w in ((dest, len(BW_R_DEST)), (a, len(BW_R_SRCA)), (b, len(BW_R_SRCB))):
        if not 0 <= n < (1 << w):
            raise ValueError("bitwise register r%d does not fit a %d-bit field" % (n, w))
    u = bytearray(template)
    _slot_put(u, BW_R_DEST, dest)
    _slot_put(u, BW_R_SRCA, a)
    _slot_put(u, BW_R_SRCB, b)
    _slot_put(u, BW_R_OP, BITWISE_REG_SEL[BITWISE_REG_FORM[opcode]])
    return bytes(u)

# THE TEN-BYTE REGISTER-REGISTER BITWISE. One Apple instance per opcode, each with all-register
# operands, taken whole: the operation selector and every modifier travel with the template and only
# the three register fields are written. BW_R_OP is NOT the selector here - it reads 7, 7 and 6 on
# these three witnesses against the four-byte values 2, 4 and 3 - which is the four-byte layout not
# carrying across, exactly as tools/g17bw10probe.py warned. The field map is the wide one: ALU_DEST
# (8 bits), BW_SRC (7 bits) and BW_IMM carrying register << 1, and those three are what a bit sweep
# of the witness reports as its register carriers.
BITWISE_REG10_TEMPLATE = {424: bytes.fromhex("230007aa3870a4020280"),
                          13575: bytes.fromhex("2300078052f8a40a0481"),
                          17771: bytes.fromhex("230206803278a48a0281")}
BW10_DEST_MAX = (1 << len(ALU_DEST)) - 1
BW10_SRCA_MAX = (1 << len(BW_SRC)) - 1
BW10_SRCB_MAX = (1 << (len(BW_IMM) - 1)) - 1      # the field carries register << 1


def encode_bitwise_reg10(opcode, dest, a, b, template=None):
    """The ten-byte form, for registers the four-byte fields cannot hold.

    Refuses rather than truncating, per operand, because the three fields are different widths and
    a single 'does not fit' would not say which.
    """
    if opcode not in BITWISE_REG10_TEMPLATE:
        raise ValueError("no ten-byte bitwise template for opcode %d" % opcode)
    for n, top, what in ((dest, BW10_DEST_MAX, "destination"), (a, BW10_SRCA_MAX, "source A"),
                         (b, BW10_SRCB_MAX, "source B")):
        if not 0 <= n <= top:
            raise ValueError("ten-byte bitwise %s r%d exceeds its field (r0..r%d)"
                             % (what, n, top))
    u = bytearray(template or BITWISE_REG10_TEMPLATE[opcode])
    _slot_put(u, ALU_DEST, dest)
    _slot_put(u, BW_SRC, a)
    _slot_put(u, BW_IMM, 2 * b)
    return bytes(u)


def decode_bitwise_reg10(u):
    """The three registers as the encoder wrote them; the caller checks the opcode itself."""
    return dict(dest=_slot_get(u, ALU_DEST), a=_slot_get(u, BW_SRC),
                b=_slot_get(u, BW_IMM) >> 1)


# The bit positions the three operand fields own. Everything else in a ten-byte bitwise - the
# opcode among it - comes from the template and must survive encoding untouched.
BW10_OPERAND_BITS = frozenset(ALU_DEST) | frozenset(BW_SRC) | frozenset(BW_IMM)


def bitwise_reg10_departures(opcode, u):
    """Bit positions where `u` differs from opcode's registered template outside the operands.

    THIS IS WHAT A COMPILE CAN ASK WITHOUT FORKING A DECODER. The opcode of a ten-byte bitwise
    lives in bits no operand field owns, so bytes that match BITWISE_REG10_TEMPLATE[opcode]
    everywhere outside ALU_DEST/BW_SRC/BW_IMM carry that opcode's form, and bytes that do not are
    something else - a different template, or an encoder that clobbered a bit it does not own.

    What it deliberately does NOT establish is that the template TABLE is labelled correctly: if
    BITWISE_REG10_TEMPLATE[424] held `xor`'s bytes, this would agree with it. That question is
    about the table rather than about any one compile, it needs an instrument this side does not
    own, and it is answered once by the independent decoder in
    test_g17bw424lower.test_the_opcode_is_read_back_from_the_bytes - post-build, where a
    subprocess is allowed. Asking it on every compile is what made an audited build refuse.
    """
    if opcode not in BITWISE_REG10_TEMPLATE:
        raise ValueError("no ten-byte bitwise template for opcode %d" % opcode)
    t = BITWISE_REG10_TEMPLATE[opcode]
    if len(u) != len(t):
        raise ValueError("ten-byte bitwise is %d bytes, not %d" % (len(u), len(t)))
    return [(by, bi) for by in range(len(t)) for bi in range(8)
            if (by, bi) not in BW10_OPERAND_BITS and ((u[by] >> bi) & 1) != ((t[by] >> bi) & 1)]


def decode_bitwise_reg(opcode, u):
    """The operation is read from the BYTES, not from the opcode argument, so a template carrying
    the wrong operation is visible rather than assumed away."""
    sel = _slot_get(u, BW_R_OP)
    name = next((k for k, v in BITWISE_REG_SEL.items() if v == sel), "selector=%d" % sel)
    return dict(op=name, dest=_slot_get(u, BW_R_DEST),
                a=_slot_get(u, BW_R_SRCA), b=_slot_get(u, BW_R_SRCB))


def decode_bitwise_imm(opcode, u):
    return dict(op=BITWISE_FORM[opcode], dest=_slot_get(u, ALU_DEST),
                src=_slot_get(u, BW_SRC), imm=_slot_get(u, BW_IMM))


def decode_alu_form(opcode, u):
    """The reading encode_alu_form writes: slot roles from the opcode, not from byte4[2]."""
    name, ra, rb = ALU_FORM[opcode]
    def read(bits, role):
        v = _slot_get(u, bits)
        return ("imm", v) if role == "imm" else ("reg", v >> 1, v & 1)
    return dict(op=name, dest=_slot_get(u, ALU_DEST),
                a=read(SLOT_A, ra), b=read(SLOT_B, rb))

# --- the ACCUMULATOR INITIALISER, opcode 554, four bytes --------------------------------------
# 14.6% of the corpus and the single most frequent instruction. It defines one register and takes
# one immediate, and it is read ONLY through phi merges - never directly, in 1231 of 1231 - which
# is what a compiler emits to make a value defined on every path. In a tensor kernel there are
# exactly min(M*N, 2048)/32 + 1 of them, one per accumulator register, and the first MAC on a
# tuple consumes one (ISA agent; ledger/g17-tensor-mac-count-law.toml).
#
# Solved with ZERO unexplained bits over 13117 instances.
TINIT_DEST = ((0, 4), (0, 5), (0, 6), (0, 7), (2, 6), (2, 7))     # 6 bits
TINIT_FLAG = ((2, 5),)      # operand 1 bit 5; the rest of that operand is frozen

def encode_tensor_init(dest, template, flag=None):
    if not 0 <= dest < (1 << len(TINIT_DEST)):
        raise ValueError("op554 destination r%d does not fit its %d-bit field"
                         % (dest, len(TINIT_DEST)))
    u = bytearray(template)
    _slot_put(u, TINIT_DEST, dest)
    if flag is not None:
        _slot_put(u, TINIT_FLAG, flag)
    return bytes(u)

def decode_tensor_init(u):
    return dict(dest=_slot_get(u, TINIT_DEST), flag=_slot_get(u, TINIT_FLAG))


# --- THE FLAG SELECTOR ------------------------------------------------------------------------
# A compact 3-bit index, FLAGn -> n, with 6 meaning FLAGTRUE. NOT the FLAGR class index, which runs
# FLAG0..FLAG14 then FLAGTRUE at 15 and does not fit in three bits. Located by the ISA agent with
# two independent confirmations: op582 varies through 0..6 and Apple's decoder resolves 6 to
# FLAGTRUE, and op575 - which reads FLAGTRUE and nothing else - has those bits frozen at exactly 6.
#
# There are FIFTEEN flag registers, FLAG0..FLAG14; only FLAG0..FLAG5 appear in this corpus.
#
# WHY THIS MATTERS HERE: g17cc emits every compare into FLAG0 and REFUSES a program where a compare
# is not immediately consumed, because with one flag allocated anything else is a clobber
# (_check_flag_discipline). With the selector located, flags become allocatable.
# THE WRITER AND THE READER USE DIFFERENT BYTES. Same compact numbering, different position, and
# conflating them is worse than a no-op: applying the READER's layout to a compare puts the index
# into byte1[4:6], which on that form is the compared REGISTER - so the instruction writes FLAG0
# anyway and compares the wrong operand. That is exactly what my first attempt did, and why
# "cmp FLAG1 / exec FLAG0" failed as well as the matched pairs.
#
#   READER  op582 op579 op583 op578, 4 bytes    byte1[4:6], bit0 at byte1[4]
#   WRITER  op10369 op10370 op10372, SIX bytes  byte0[5:7], bit0 at byte0[5]
#
# TEN-BYTE forms of the writers are a DIFFERENT layout - bit0 is unmatched at length 10 on all
# three - so this must not be used to author a 10-byte compare. ISA agent,
# ledger/g17-flag-selector-writer-and-reader-differ.toml.
FLAG_BITS = {
    582: ((1, 4), (1, 5), (1, 6)),   579: ((1, 4), (1, 5), (1, 6)),
    583: ((1, 4), (1, 5), (1, 6)),   578: ((1, 5), (1, 6)),
    10369: ((0, 5), (0, 6), (0, 7)), 10370: ((0, 5), (0, 6), (0, 7)),
    10372: ((0, 5), (0, 6), (0, 7)),
}
FLAGTRUE = 6

def encode_flag(opcode, u, flag):
    """Write the flag index into an instruction of a form that carries one."""
    if opcode not in FLAG_BITS:
        raise ValueError("opcode %d has no located flag selector" % opcode)
    bits = FLAG_BITS[opcode]
    if not 0 <= flag < (1 << len(bits)):
        raise ValueError("FLAG%d does not fit opcode %d's %d-bit selector"
                         % (flag, opcode, len(bits)))
    v = bytearray(u)
    _slot_put(v, bits, flag)
    return bytes(v)

def decode_flag(opcode, u):
    return _slot_get(u, FLAG_BITS[opcode])


# --- the BARRIER, opcode 447, six bytes -------------------------------------------------------
# byte1 IS THE SCOPE, and it is a SELECTOR in exactly the sense byte2[2:0] is for the bitwise
# family: constant within a barrier kind, different between kinds. Inheriting it gives a
# threadgroup barrier where a device one was asked for, and no single-scope test can see that -
# the same failure shape as `and` returning an xor.
#
# Recovered by the ISA agent from kernels that state their own scope. The discriminating case is a
# kernel asking for BOTH, which emits exactly one of each adjacent to the other, and which neither
# single-scope kernel produces:
#
#     bar_tg    one op447, byte1 = 0x51            bar_dev   one op447, byte1 = 0x69
#     bar_both  two op447, one 0x51 AND one 0x69
#
# NOT EXECUTED. This is a source-level correspondence - a kernel named for a scope emits a
# distinguishable byte - and says nothing about ordering semantics. Corpus-wide the split is 343
# threadgroup to 3 device, and every device instance is in one of those probe kernels, so Apple's
# driver shaders here never use it.
# THE SCOPE SPANS TWO BYTES, not one. Compiling threadgroup_barrier at all five memory flags gives
# five encodings of op447 that differ only in bytes 0 and 1 (isa/g17-barrier-scopes.txt):
#
#     none 07 51    threadgroup 27 51    imageblock 47 51    device 0f 69    texture 87 67
#
# Byte 0's bits 5 and 6 step the threadgroup family - the scope immediate goes 20, 276, 532 - and
# device and texture move byte 1 as well, so they are a DIFFERENT field rather than more of the
# same axis. This encoder wrote byte 1 only, which meant `device` emitted byte 0 from whatever
# template it was handed: 27 69, an encoding Apple never produces. Nothing in this backend had
# selected a device barrier yet, so it cost nothing, and it would have been silent when it did.
BARRIER_SCOPE = {"none": (0x07, 0x51), "threadgroup": (0x27, 0x51), "imageblock": (0x47, 0x51),
                 "device": (0x0f, 0x69), "texture": (0x87, 0x67)}

def encode_barrier(scope, template):
    if scope not in BARRIER_SCOPE:
        raise ValueError("barrier scope %r; recovered scopes are %s"
                         % (scope, sorted(BARRIER_SCOPE)))
    u = bytearray(template)
    u[0], u[1] = BARRIER_SCOPE[scope]
    return bytes(u)

def decode_barrier(u):
    """The scope is read from the BYTES, so a template carrying the wrong one is visible."""
    for k, (b0, b1) in BARRIER_SCOPE.items():
        if u[0] == b0 and u[1] == b1: return dict(scope=k)
    return dict(scope="unrecovered:%02x%02x" % (u[0], u[1]))


# --- the TENSOR MAC, opcode 5106, ten bytes ---------------------------------------------------
# The compiler has treated a tensor sequence as opaque 12-byte "units" with four authored bits
# each. Apple's decoder frames it differently and more usefully: op5106 is a TEN-byte instruction
# with FOUR register operands in two tensor register classes, and operands 0 and 9 are the SAME
# bit-field - the accumulator, read and written, which is the multiply-accumulate.
#
# Field positions from tools/g17fields.py over 6073 instances. The low bits of each register field
# never vary in the corpus because the accumulators are 8-register tuples and the operand tiles
# are 4-aligned, so those bits are UNRESOLVED rather than known-zero, and this encoder refuses a
# register it cannot place instead of truncating one.
# FIVE bits, not four. The peer's mutation dump over a real op5106 - every flip XORing the member
# index by one power of two - gives the accumulator's index bits as b0[7] b7[5] b2[7] b2[3] b2[4],
# and byte2[4] was missing here. Four bits reach member 15 and the class has 18, so the top two
# accumulators were unaddressable.
TMAC_ACC  = ((0, 7), (7, 5), (2, 7), (2, 3), (2, 4))  # accumulator, value bits 3..7
TMAC_A    = ((3, 4), (3, 1), (3, 0), (3, 7), (3, 5))  # tile operand A, value bits 2..6
TMAC_B    = ((9, 1), (9, 2), (9, 3), (9, 4), (5, 2))  # tile operand B, value bits 2..6
TMAC_ACC_SHIFT, TMAC_A_SHIFT, TMAC_B_SHIFT = 3, 2, 2

def _tmac_get(u, bits, shift):
    return _slot_get(u, bits) << shift

def _tmac_put(u, bits, shift, v):
    if v & ((1 << shift) - 1):
        raise ValueError("tensor register %d is not a multiple of %d - the low bits of this "
                         "field never vary in the corpus and are unresolved" % (v, 1 << shift))
    if (v >> shift) >= (1 << len(bits)):
        raise ValueError("tensor register %d does not fit a %d-bit field" % (v, len(bits)))
    _slot_put(u, bits, v >> shift)

# THE FOUR NON-REGISTER FIELDS, read off the corrected references against the composition law
# (spike/accel/re/tgen.py, tools/g17tensor.compose_macs). Without these a composed sequence has the
# right operands and the wrong bytes:
#
#   byte0[3]  FIRST     set on the first MAC of each repeat of the sequence, and only there
#   byte2[5]  COL0      set exactly when the MAC's column block is 0
#   byte5[1]  ROW0      set exactly when the MAC's ROW block is 0 - the exact analogue of COL0,
#                       and the reason 16x32 and 16x48 composed byte-exactly on the first attempt
#                       while every multi-row shape did not: with rows=1 every MAC is row 0, so
#                       the template carried the right value and the field was invisible
#   byte4[3]  KSLICE    the K slice, 1 then 0 within each pair
#   byte1     TOKEN     a saturating one-hot on position within the repeat: 0 at position 0,
#                       then 1 << (pos+1) for positions 1..5, and 0 again beyond - so a sequence
#                       with more than six distinct MACs reuses 0. That saturation is measured,
#                       not assumed: 64x16 has eight and its positions 6 and 7 both carry 0.
TMAC_FIRST, TMAC_COL0, TMAC_KSLICE, TMAC_ROW0 = (0, 3), (2, 5), (4, 3), (5, 1)

# ---------------------------------------------------------------------------------------------
# THE TENSOR STORE, op17257 at 10 and 16 bytes.
#
# The peer's operand classes: NumDefs 0, so operand 0 is the VALUE, a GPR32tup4 - four registers,
# where the loads move a GPR32tup2 pair. Two stores therefore write one column's eight-register
# accumulator, which is exactly what the corpus shows: fields f and f+4 with offsets O and O+4096.
#
#   value tup4   b0[4] b0[5] b0[6] b0[7] b2[6] b2[7] b7[4], weights 1..64. This is the tup4
#                OPERAND CODE, not a register number: the printer's id moves one for one with it,
#                and in the 16xN family it takes the multiples of four, but the corpus also
#                contains odd codes, so any register mapping read off the tensor kernels alone
#                would be an artefact of those kernels. The shape law below gives the code
#                directly, which is what authoring needs.
#   index reg    slot bits 1..7 at b3[1..7]
#   offset       the same layout as the load: b8[5] b8[6] b8[7] b9[0] b9[1] b9[2] b9[3] b9[4],
#                continuing into b15[0..7] on the long form
#   b7[2]        set on the LAST column's long store in every 16xN shape, and nowhere else
#
# The shape law, read off 16x16, 16x32, 16x48 and 16x64 with no exception: for `cols` columns and
# output column block c, the pair is (value 8*(cols-c), offset 64c) and (value 8*(cols-c)+4,
# offset 4096 + 64c).
TSTORE_VALUE = ((0, 4), (0, 5), (0, 6), (0, 7), (2, 6), (2, 7), (7, 4))
TSTORE_INDEX = ((3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7))
TSTORE_BASE  = ((1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6))    # slot bits 1..6
TSTORE_LAST  = (7, 2)


def decode_tensor_store(u):
    return dict(value=_bits_get(u, TSTORE_VALUE),
                index=_bits_get(u, TSTORE_INDEX),
                base=_bits_get(u, TSTORE_BASE),
                offset=_bits_get(u, TLOAD_OFF16 if len(u) >= 16 else TLOAD_OFF12),
                last=(u[TSTORE_LAST[0]] >> TSTORE_LAST[1]) & 1)


def encode_tensor_store(template, value=None, index=None, base=None, offset=None, last=None):
    u = bytearray(template)
    if value is not None: _bits_put(u, TSTORE_VALUE, value)
    if index is not None: _bits_put(u, TSTORE_INDEX, index)
    if base is not None: _bits_put(u, TSTORE_BASE, base)
    if offset is not None:
        _bits_put(u, TLOAD_OFF16 if len(u) >= 16 else TLOAD_OFF12, offset)
    if last is not None:
        u[TSTORE_LAST[0]] = ((u[TSTORE_LAST[0]] & ~(1 << TSTORE_LAST[1]))
                             | ((1 if last else 0) << TSTORE_LAST[1]))
    return bytes(u)


# ---------------------------------------------------------------------------------------------
# THE BRANCH DISPLACEMENT, op458 (backward in every corpus instance) and op462 (forward).
#
# The peer names these two as the whole of scheduling class 6 - NumDefs 0, an imm.t4 target - and
# a composer that inserts or deletes instructions inside a loop has to re-target them or the loop
# body moves under the branch. The displacement is a byte count relative to the instruction, and
# it is SCATTERED across eight bytes in an order that is nowhere near monotonic: bits 2..4 sit
# below bits 10..13 in byte1, and bits 5..9 interleave with two bits that are not part of it.
#
# Recovered by authoring rather than by correlation: flip one bit, hand the instruction to Apple's
# decoder, read the printed target back, and record the delta. Every delta below is a power of
# two, which is what makes the reading a field rather than a fit - and bit 0 never appears, so
# displacements are even, as they must be when every instruction is 2-byte aligned.
BRANCH_DISP = ((0, 7),                                                    # bit 1
               (1, 4), (1, 5), (1, 6),                                    # bits 2..4
               (2, 1), (2, 3), (2, 5), (2, 6), (2, 7),                    # bits 5..9
               (1, 0), (1, 1), (1, 2), (1, 3),                            # bits 10..13
               (3, 4),                                                    # bit 14
               (4, 0), (4, 1), (4, 2), (4, 3), (4, 4),                    # bits 15..19
               (6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7),   # 20..27
               (7, 0), (7, 1), (7, 2),                                    # bits 28..30
               (7, 6), (7, 7),                                            # bits 31..32
               (8, 0), (8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (8, 6), (8, 7),   # 33..40
               (9, 0), (9, 1), (9, 2), (9, 3), (9, 4), (9, 5),            # bits 41..46
               (9, 6))                                                    # bit 47, the sign
BRANCH_OPS = (458, 462)


def decode_branch10(u):
    """-> the signed byte displacement carried by a 10-byte op458 or op462.

    The 4-byte branch forms above are a different, narrower encoding - 12 bits of displacement in
    one word - and share nothing with this one. Same two opcodes, two lengths, two layouts, which
    is the peer's rule that a field map never crosses a length.
    """
    v = _bits_get(u, BRANCH_DISP) << 1
    return v - (1 << 48) if v >> 47 & 1 else v


def encode_branch10(template, disp):
    """Re-target a 10-byte branch; `disp` is what decode_branch10 returns."""
    if disp % 2:
        raise ValueError("displacement %d is odd; instructions are 2-byte aligned and bit 0 of "
                         "this field does not exist" % disp)
    v = (disp >> 1) & ((1 << len(BRANCH_DISP)) - 1)
    u = bytearray(template)
    _bits_put(u, BRANCH_DISP, v)
    return bytes(u)


# ---------------------------------------------------------------------------------------------
# THE TENSOR OPERAND LOAD, op12674 and op12675, at 12 and 16 bytes.
#
# The peer ISA session solved these forms on the sixteen corrected tg-* references and left six
# bits of the 16-byte form unattributed. Re-solved here over 12949 loads in the whole cache - with
# the printer's MCRegister ids converted to SLOTS first, which is what hid the register fields -
# five of those six are accounted for, and the sixth (b6[7]) is not authored.
#
#   dest pair     slot bits 2..7   b0[5] b0[6] b0[7] b2[6] b2[7] b7[4]     slot = 2 * register
#   index reg     slot bits 1..7   b3[1] .. b3[7]
#   base pair     slot bits 1..5   b1[1] .. b1[5]                          (peer, def-use 97.7%)
#   k             1 + (b8[4]<<2 | b6[4]<<1 | b6[3])   EXACT on 8557 of 8557 16- and 12-byte loads
#   hi flag       b10[6] and b12[7] together          the 2^37 bit of the printed operand 1
#   offset        bits 4..7 at b9[1..4], 8..14 at b15[0..6], sign at b15[7]
#
# The k field is the one the corpus could not show as a field at all: the printer renders it as
# operand 1 = k << 20, and only its low bit correlates with a single instruction bit, because the
# other two live in different bytes. Grouping by the VALUE and asking which bits are constant
# within a group is what separated them.
#
# THE OFFSET's HIGH BITS ARE DECODER-VERIFIED, NOT CORPUS-VERIFIED. Every offset in the cache is a
# multiple of 32 below 8192, so bits 7 to 10 never vary and no correlation can reach them. They
# were established by authoring: set the bit, hand the instruction to Apple's decoder, and read
# the operand back - b9[4] gives +128, b15[0] +256, b15[1] +512, b15[2] +1024, b15[7] the sign.
# b9[5] and b9[6] make the decoder reject the instruction outright, so they are not offset bits.
# That is the same oracle the round-trip gate uses, and it is authoritative for decoding.
TLOAD_DEST  = ((0, 5), (0, 6), (0, 7), (2, 6), (2, 7), (7, 4))     # slot bits 2..7
TLOAD_INDEX = ((3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7))   # slot bits 1..7
# SIX bits, not five: the peer found byte1[6] by the same mutate-and-read-back probe, on an
# instance whose base PRINTS as a register. No base in the corpus reaches slot 64, so the bit
# is zero wherever the operand is visible and correlation could never have reached it.
TLOAD_BASE  = ((1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6))     # slot bits 1..6
TLOAD_K     = ((6, 3), (6, 4), (8, 4))                             # k - 1, low bit first
TLOAD_HI    = ((10, 6), (12, 7))
# The offset starts at b8[5] and runs contiguously through b9[4], then continues in byte15 on the
# long form. Bits 0..3 never vary in any corpus kernel - every offset there is a multiple of 16 -
# and were established the same way as the high bits, by authoring and reading the decoder back.
TLOAD_OFF16 = ((8, 5), (8, 6), (8, 7), (9, 0), (9, 1), (9, 2), (9, 3), (9, 4),      # bits 0..7
               (15, 0), (15, 1), (15, 2), (15, 3), (15, 4), (15, 5), (15, 6), (15, 7))
TLOAD_OFF12 = ((8, 5), (8, 6), (8, 7), (9, 0), (9, 1), (9, 2), (9, 3), (9, 4))
# THE "0 OR 16" OPERAND IS THE ADDRESS REGISTER'S LIFETIME. Apple's decoder calls it operand 6 on
# op12674/12675 and operand 10 on op12675's second source, and 16 is RELEASE - the same modifier
# word the ALU and the float unary forms carry (g17auth.LIFETIME_RELEASE). The census says it
# plainly: across the 16x16, 16x32, 16x48 and 16x64 tensor kernels the number of loads carrying 16
# is EXACTLY 3 + 2 + 1 whatever the column count, while the loads carrying 0 grow by eight per
# column. The released ones are the last readers of their address registers and stay last as
# columns are added, so EVERY AUTHORED LOAD MUST CARRY ZERO.
TLOAD_OP6   = (7, 2)                          # operand 6, value bit 4 - release the base address
TLOAD_OP10  = (4, 7)                          # operand 10, value bit 4 - op12675's index register


def _bits_get(u, bits):
    return sum(((u[byte] >> bit) & 1) << i for i, (byte, bit) in enumerate(bits))


def _bits_put(u, bits, v):
    if v >> len(bits):
        raise ValueError("value %d does not fit the %d-bit field" % (v, len(bits)))
    for i, (byte, bit) in enumerate(bits):
        u[byte] = (u[byte] & ~(1 << bit)) | (((v >> i) & 1) << bit)


def decode_tensor_load(u):
    off = TLOAD_OFF16 if len(u) >= 16 else TLOAD_OFF12
    return dict(dest=_bits_get(u, TLOAD_DEST) * 2,          # slot bits 2.. -> register, pair base
                index=_bits_get(u, TLOAD_INDEX),            # slot bits 1.. -> register number
                base=_bits_get(u, TLOAD_BASE),
                k=1 + _bits_get(u, TLOAD_K),
                hi=(u[TLOAD_HI[0][0]] >> TLOAD_HI[0][1]) & 1,
                offset=_bits_get(u, off),
                op6=((u[TLOAD_OP6[0]] >> TLOAD_OP6[1]) & 1) << 4)


# THE RENDEZVOUS PARITY. byte6[3] - which this file also reads as the low bit of the k field -
# carries the COMPLEMENT of the consuming MAC's wait-token parity: the peer measured
# load b6[3] = 1 - (token & 1) in 382 of 384 pairs over sixteen shapes, against 50.0% for a
# shuffled control and 50.3% for the load's own position, so it is a rendezvous the two
# instructions write jointly and not two counters advancing together.
#
# This is a compiler-emission correlation, not a proven hardware rendezvous.
# The same ledger records a control flipping this bit on four loads: all 256
# observed outputs stayed correct. That does not prove the bit inert when load
# latency is exposed, but it also does not establish a causal explanation for
# the redirected column's zeros. Preserve the observed encoding relationship
# when reproducing the compiler schedule; dependency semantics remain open.
# ledger/g17-tensor-wait-token-is-a-rendezvous.toml
TLOAD_TOKEN_PARITY = (6, 3)


def set_load_token_parity(u, token):
    """Write the load's half of the rendezvous with the MAC that consumes it."""
    b = bytearray(u)
    want = 1 - (token & 1)
    b[TLOAD_TOKEN_PARITY[0]] = ((b[TLOAD_TOKEN_PARITY[0]] & ~(1 << TLOAD_TOKEN_PARITY[1]))
                                | (want << TLOAD_TOKEN_PARITY[1]))
    return bytes(b)


def encode_tensor_load(template, dest=None, index=None, base=None, k=None, hi=None,
                       offset=None, op6=None, op10=None):
    """One operand load, from a template of the same opcode and length.

    dest is the first register of the destination pair and must be even; index and base are
    register numbers; k is the 1-based tile selector the printer shows as k << 20; offset is a
    byte offset and must be a multiple of 16.
    """
    u = bytearray(template)
    if dest is not None:
        if dest % 2: raise ValueError("the destination is a register PAIR; r%d is odd" % dest)
        _bits_put(u, TLOAD_DEST, dest // 2)
    if index is not None: _bits_put(u, TLOAD_INDEX, index)
    if base is not None: _bits_put(u, TLOAD_BASE, base)
    if k is not None:
        if not 1 <= k <= 8: raise ValueError("k is a 3-bit field offset by one; %d is outside 1..8" % k)
        _bits_put(u, TLOAD_K, k - 1)
    if hi is not None:
        for byte, bit in TLOAD_HI:
            if byte < len(u): u[byte] = (u[byte] & ~(1 << bit)) | ((hi & 1) << bit)
    if offset is not None:
        _bits_put(u, TLOAD_OFF16 if len(u) >= 16 else TLOAD_OFF12, offset)
    if op6 is not None:
        u[TLOAD_OP6[0]] = (u[TLOAD_OP6[0]] & ~(1 << TLOAD_OP6[1])) | ((1 if op6 else 0) << TLOAD_OP6[1])
    if op10 is not None and len(u) > TLOAD_OP10[0]:
        u[TLOAD_OP10[0]] = ((u[TLOAD_OP10[0]] & ~(1 << TLOAD_OP10[1]))
                            | ((1 if op10 else 0) << TLOAD_OP10[1]))
    return bytes(u)


def decode_tensor_mac(u):
    return dict(acc=_tmac_get(u, TMAC_ACC, TMAC_ACC_SHIFT),
                a=_tmac_get(u, TMAC_A, TMAC_A_SHIFT),
                b=_tmac_get(u, TMAC_B, TMAC_B_SHIFT),
                first=(u[TMAC_FIRST[0]] >> TMAC_FIRST[1]) & 1,
                col0=(u[TMAC_COL0[0]] >> TMAC_COL0[1]) & 1,
                kslice=(u[TMAC_KSLICE[0]] >> TMAC_KSLICE[1]) & 1,
                row0=(u[TMAC_ROW0[0]] >> TMAC_ROW0[1]) & 1,
                tag=next((k for k, bb in enumerate(TMAC_TAG_BITS)
                          if bb and (u[bb[0]] >> bb[1]) & 1), 0))

def encode_tensor_mac(acc, a, b, template, first=None, col0=None, kslice=None, tag=None,
                      row0=None):
    u = bytearray(template)
    _tmac_put(u, TMAC_ACC, TMAC_ACC_SHIFT, acc)
    _tmac_put(u, TMAC_A, TMAC_A_SHIFT, a)
    _tmac_put(u, TMAC_B, TMAC_B_SHIFT, b)
    for v, (byte, bit) in ((first, TMAC_FIRST), (col0, TMAC_COL0), (kslice, TMAC_KSLICE),
                           (row0, TMAC_ROW0)):
        if v is not None:
            u[byte] = (u[byte] & ~(1 << bit)) | ((v & 1) << bit)
    if tag is not None:
        u[1] &= ~0x7C                       # clear the five tag bits that live in byte1
        u[7] &= ~0x80
        u[2] &= ~0x40
        for byte, mask in mac_tag(tag).items():
            u[byte] |= mask
    return bytes(u)

# THE WAIT TAG IS AN EIGHT-STATE FIELD WITH SCATTERED BITS, not a one-hot that overflows - the
# same layout every field in this ISA has, including op5106's own registers. Positions:
#
#     k=0        no bit set          k=1..5   b1[2] b1[3] b1[4] b1[5] b1[6]
#     k=6        b7[7]               k=7      b2[6]
#
# Identified by the ISA agent from the def-use graph. The earlier reading here - a one-hot on
# position that saturates past 5 - fitted the shapes with six or fewer distinct MACs and was wrong
# about every larger one.
TMAC_TAG_BITS = (None, (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (7, 7), (2, 6))

def mac_tag(k):
    """The byte-level effect of wait tag k, as {byte: mask}. k None means no tag."""
    if k is None or k == 0:
        return {}
    if not 0 < k < len(TMAC_TAG_BITS):
        raise ValueError("wait tag %r outside the eight states this field encodes" % k)
    byte, bit = TMAC_TAG_BITS[k]
    return {byte: 1 << bit}


# --- read_sr.direct (class c, 4 bytes) ---------------------------------------------------
# thread/threadgroup/grid indexing. Recovered 2026-09-04.
#   dest = byte0[7:4]   byte1 = special-register index   byte3[6:5] = sequence position
#
# byte2[2] SELECTS WHICH 16-BIT HALF OF THE DESTINATION IS WRITTEN, measured 2026-09-08 by
# sweeping the bit at six destination indices: clear prints the half-register file based at 425
# and set prints the one based at 281, at a constant offset of 144 for every index. Apple's
# imageblock prologue is exactly one pair of these - SR_LOCAL_X into 425+i and SR_LOCAL_Y into
# 281+i with the SAME i - which is how a ushort2 coordinate is assembled in one 32-bit register.
# The 32-bit form leaves the bit alone and is unaffected.
SR_HALF_BIT = (2, 0x04)          # (byte, mask): clear = file@425, set = file@281
# byte2[2] IS OWNED, and writing it without saying so cost one bit of scalar debt on the scorecard
# the first time this shipped. encode_sr writes it on every call - 0 for the 32-bit form, where it
# must be clear or the encoding is refused, and the requested half for the 16-bit one - so it is
# claimed here rather than left to be inherited from whatever template arrives.
OWNED_SR = {0: 0xF0, 1: 0xFF, 2: SR_HALF_BIT[1], 3: 0x60}

def decode_sr(u):
    b, m = SR_HALF_BIT
    return dict(dest=u[0] >> 4, sr=u[1], seq=(u[3] >> 5) & 3,
                half=1 if (len(u) > b and u[b] & m) else 0)

SR_WIDTH_NIBBLE = {32: 0xC, 16: 0x4}   # byte0[3:0], swept over all sixteen values

def encode_sr(dest, sr, seq, template, half=None):
    """half=None writes the 32-bit destination; half=0/1 writes one 16-bit half of it.

    THE HALF BIT BELONGS TO THE 16-BIT FORM AND ONLY TO IT. byte0[3:0] selects the width - 0xC
    decodes as op14059 with a 32-bit destination and 0x4 as op14060 with a 16-bit one - and
    setting byte2[2] on the 32-bit form is not an encoding at all, the decoder rejects it. So a
    half request switches the width nibble too, which is measured rather than inherited: the same
    template with the nibble changed reproduces Apple's own imageblock prologue byte for byte at
    dest 0, and at dest 0/2/6/11/15 the two halves stay at one index.
    """
    if not 0 <= dest <= 15: raise ValueError("dest r%d out of range" % dest)
    u = bytearray(template)
    u[0] = (u[0] & 0x0F) | ((dest & 0xF) << 4)
    u[1] = sr & 0xFF
    u[3] = (u[3] & ~0x60) | ((seq & 3) << 5)
    b, m = SR_HALF_BIT
    if half is None:
        u[b] &= ~m                       # the 32-bit form: the decoder refuses it set
    else:
        u[0] = (u[0] & 0xF0) | SR_WIDTH_NIBBLE[16]
        u[b] = (u[b] | m) if half else (u[b] & ~m)
    return bytes(u)


# --- load (class 7/f, byte2 == 0x03, byte1 not a store selector) --------------------------
# dest = byte0[7:4], causal (ledger/g17-load-family.toml). The load carries NO immediate offset
# and NO buffer selector - both were shown identical across seven compiler variants - so the
# remaining bytes are address-register and form bits not yet recovered.
OWNED_LOAD = {0: 0xF0, 1: 0xFF, 3: 0xFE, 4: 0x64, 6: 0x80, 7: 0x3F, 9: 0x04, 13: 0xFF}

def decode_load(u):
    # Same address field as the store: verified on a kernel loading b2[60], b2[70], b2[80], whose
    # byte7/byte13 decode to 60, 70 and 80 under the store's expression.
    # ledger/g17-loadstore-direction.toml
    hi = u[13] if len(u) > 13 else 0
    # index_reg = byte3 >> 1, RESOLVED. On a kernel with two live index registers holding p and
    # q, byte3 0x08 loads b1[p] and 0x0a loads b1[q]; preregistered on fresh data p=6 q=2 and
    # measured exactly. ledger/g17-load-index-register.toml
    # disp2 (byte4[6:5]) is a SECOND displacement, independent of offset and ADDITIVE with it:
    # byte4[5]+byte6[7] gives +2 and byte4[6]+byte7[0] gives +4, both measured.
    # index_scale (byte7[5]) multiplies the index by 2 - verified at two index values.
    # width/format, both preregistered and exact at four index values:
    #   byte9[2] set -> the load returns only the low BYTE  (1006 -> 238)
    #   byte4[2] set -> the loaded value is shifted left 16 (1006 -> 65929216)
    # byte9 exists only in the 12- and 14-byte forms. Reading it unconditionally made
    # decode_load raise IndexError on every one of the 329 eight-byte loads in Apple's corpus,
    # which g17cover reported as "CLAIMED BUT NEVER ATTEMPTED" - the family was counted as
    # modelled and scored zero, so the defect showed up as a coverage gap rather than as a crash.
    # The `hi` field above was already guarded the same way; `narrow` was not.
    # THE DESTINATION IS SEVEN BITS AND THIS READ FOUR. Reading the nibble alone made selfcheck
    # report "dest decoded 1, selection asked 17" the moment the encoder started writing them -
    # a decoder that cannot see a field the encoder writes turns a correct instruction into a
    # failed round trip. Same three bits, same weights. See encode_load.
    dest = ((u[0] >> 4) & 0xF) | (((u[2] >> 6) & 1) << 4) | (((u[2] >> 7) & 1) << 5)
    if len(u) > 5:
        dest |= ((u[5] >> 4) & 1) << 6
    return dict(dest=dest, base=u[1], index_reg=u[3] >> 1,
                narrow=((u[9] >> 2) & 1) if len(u) > 9 else 0, hi16=(u[4] >> 2) & 1,
                disp2=(u[4] >> 5) & 3, index_scale=2 if (u[7] >> 5) & 1 else 1,
                offset=64 * hi + 2 * (u[7] & 0x1F) + ((u[6] >> 7) & 1))

# WHICH LOAD FORMS ACTUALLY CARRY byte9 bit2, counted over Apple's own instances. A template is
# identified by its opcode, which is what the walk reports for it.
_NARROW_FORMS = {11999, 12646, 12652, 12674, 12675}


def _narrow_witnessed(template):
    """Does Apple ever set byte9 bit2 on the form this template belongs to?"""
    try:
        from agxforge.g17 import ref as g17ref
        r = list(g17ref.walk(bytes(template), 0))
        return bool(r) and r[0][2] in _NARROW_FORMS
    except Exception:
        return False


# THE LOAD'S DESTINATION IS SEVEN BITS, NOT FOUR. This encoder wrote only byte0[4:7] and refused
# anything above r15, which made every loaded value compete for the twelve narrow registers and
# capped the whole backend at FOUR simultaneous live values. The operand map has carried the other
# three bits all along - byte2[6], byte2[7] and byte5[4], weights 16, 32 and 64 on top of the
# nibble - `verified` over 1,359 instances at explains 1.0, and Apple itself emits destinations up
# to r88 (printed 193). The formula reproduces all 66 distinct destinations in the corpus with no
# exceptions. ledger/g17-the-live-value-ceiling-was-three-bits.toml
LOAD_DEST_BITS = ((4, 0, 4), (5, 0, 5), (6, 0, 6), (7, 0, 7),   # (bit of the register, byte, bit)
                  (2, 6, 4), (2, 7, 5), (5, 4, 6))
LOAD_DEST_MAX = 127


def encode_load(dest, base, offset, template, index_reg=None, disp2=0, index_scale=1,
                narrow=0, hi16=0):
    if not 0 <= dest <= LOAD_DEST_MAX:
        raise ValueError("dest r%d out of range for a seven-bit field" % dest)
    u = bytearray(template)
    u[0] = (u[0] & 0x0F) | ((dest & 0xF) << 4)
    # the three high bits, written whole rather than left at whatever the template carried
    u[2] = (u[2] & ~0xC0) | (((dest >> 4) & 1) << 6) | (((dest >> 5) & 1) << 7)
    if len(u) > 5:
        u[5] = (u[5] & ~0x10) | (((dest >> 6) & 1) << 4)
    elif dest > 63:
        raise ValueError("dest r%d needs byte5, and this form is %d bytes" % (dest, len(u)))
    u[1] = base & 0xFF
    if index_reg is not None: u[3] = (u[3] & 0x01) | ((index_reg & 0x7F) << 1)
    # hi16 IS THE SAME STORY AS narrow, ON THE SAME FORMS. byte4 bit2 is set in 208 of op12646's
    # 586 fourteen-byte instances and in 0 of op12682's 1,359. Writing it on a form that does not
    # carry it produces bytes Apple's decoder rejects.
    if hi16 and not _narrow_witnessed(template):
        raise ValueError("hi16 is unwitnessed on this load form (byte4 bit2 is never set in "
                         "Apple's instances of it); select a form that has the field")
    u[4] = (u[4] & ~0x64) | ((disp2 & 3) << 5) | ((hi16 & 1) << 2)
    # NARROW IS NOT A FIELD OF EVERY LOAD FORM. byte9 bit2 is witnessed on op12674 (1,695 of
    # 3,453), op12675 (862 of 1,765), op12646, op12652 and op11999 - and NEVER on op12682, which
    # this backend selects: 0 of 1,359 corpus instances. Setting it there produces bytes Apple's
    # decoder rejects outright, and that is what stopped `addressing_modes` compiling.
    #
    # A form without the field refuses rather than inventing it. The instruction that needs a
    # narrow load needs a form that HAS one, which is a selection question and not an encoding one.
    if narrow:
        if len(u) <= 9:
            raise ValueError("narrow load needs a form with byte9; this one is %d bytes" % len(u))
        if not _narrow_witnessed(template):
            raise ValueError("narrow is unwitnessed on this load form (byte9 bit2 is never set in "
                             "Apple's instances of it); select a form that has the field")
        u[9] = (u[9] & ~0x04) | 0x04
    elif len(u) > 9:
        u[9] &= ~0x04
    hi, rem = divmod(offset, 64)
    u[6] = (u[6] & ~0x80) | ((rem & 1) << 7)
    u[7] = (u[7] & ~0x3F) | ((rem >> 1) & 0x1F) | ((1 if index_scale == 2 else 0) << 5)
    if len(u) > 13: u[13] = hi
    elif hi: raise ValueError("offset %d needs the 14-byte load form" % offset)
    return bytes(u)


# --- conditional branch (4 bytes) --------------------------------------------------------
# Displacement bit 0 is implicit zero (targets are 2-byte aligned); bits 1..8 are scattered
# through the little-endian word. isa/g17-scalar-isa.toml branch.cond.fwd / branch.cond.back
_BR_BITS = [(1, 7), (2, 12), (3, 13), (4, 14), (5, 17), (6, 19), (7, 21), (8, 22),
            (9, 23), (10, 8)]
_BR_SIGN = 28                                     # word[28], i.e. byte3[4]

# The displacement occupies eleven scattered bits of the little-endian word: ten magnitude bits
# plus the sign at word[28]. Everything else in the instruction is the form itself. Owning the
# displacement is what lets a branch be AUTHORED, and the encoding is validated corpus-wide -
# every one of Apple's 825 branch targets lands on an instruction boundary the walk found
# independently (ledger/g17-branch-target-consistency.toml).
def _branch_owned_mask():
    m = 0
    for _db, _wb in _BR_BITS: m |= 1 << _wb
    m |= 1 << _BR_SIGN
    return m

OWNED_BRANCH = {i: (_branch_owned_mask() >> (8 * i)) & 0xFF for i in range(4)}

# The two branch forms, with the displacement masked out. An instruction matching either in every
# other bit is a branch of that form whatever target it carries.
FWD_FORM  = int.from_bytes(bytes.fromhex("3e005b0e"), "little") & ~_branch_owned_mask()
BACK_FORM = int.from_bytes(bytes.fromhex("de6ff31e"), "little") & ~_branch_owned_mask()

def is_branch(u):
    if len(u) != 4: return False
    w = int.from_bytes(bytes(u), "little") & ~_branch_owned_mask()
    return w in (FWD_FORM, BACK_FORM)

def decode_branch(u):
    w = int.from_bytes(u[:4], "little")
    m = 0
    for db, wb in _BR_BITS:
        m |= ((w >> wb) & 1) << db
    return dict(disp=m - 2048 if (w >> _BR_SIGN) & 1 else m)

def encode_branch(disp, template):
    if disp % 2: raise ValueError("branch targets are 2-byte aligned; disp %d is odd" % disp)
    if not -2048 <= disp <= 2046: raise ValueError("disp %d out of 12-bit range" % disp)
    w = int.from_bytes(bytes(template[:4]), "little")
    neg = disp < 0
    m = disp + 2048 if neg else disp
    for db, wb in _BR_BITS:
        w = (w & ~(1 << wb)) | (((m >> db) & 1) << wb)
    w = (w & ~(1 << _BR_SIGN)) | (int(neg) << _BR_SIGN)
    return w.to_bytes(4, "little") + bytes(template[4:])


# --- cmp.value.imm threshold (class-c half of a 6+8 byte compound) ------------------------
# A comparison producing a VALUE is carried by TWO instructions, a 6-byte class-2 and an 8-byte
# class-c. The threshold lives in the class-c: the stored halfword is 1 + 64*K.
# isa/g17-scalar-isa.toml cmp.value.imm ; ledger/g17-comparison-threshold-authored.toml
def decode_cmp_threshold(u):
    return dict(k=((u[3] << 8) | u[2]) >> 6)

def encode_cmp_threshold(k, template):
    if not 0 <= k <= 255: raise ValueError("threshold %d outside the form's range" % k)
    hw = 1 + 64 * k
    u = bytearray(template)
    u[2] = hw & 0xFF
    u[3] = (hw >> 8) & 0xFF
    return bytes(u)


# Bits measured INERT at TWO INDEPENDENT SITES (dep4 consumer and tileptr value-add): each
# was flipped at a working authored site and the program's output was unchanged. They are metadata
# a backend need not reproduce, so a second ENCODE score may exclude them - with this list as the
# justification. Bits NOT in this list are either causal or untested; nothing is excluded on
# suspicion.
INERT_ALU = {(0, 3), (0, 5), (0, 6), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (2, 5), (2, 6), (3, 0), (3, 1), (3, 2), (4, 5), (4, 6), (4, 7), (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (6, 4), (6, 5), (6, 6), (6, 7), (7, 0), (7, 1), (7, 2), (7, 6), (7, 7), (8, 3), (8, 4), (8, 6), (9, 7), (10, 4), (10, 6), (10, 7), (11, 1), (11, 2), (11, 3), (11, 4), (11, 5), (11, 6), (11, 7)}


# Load bits measured INERT at TWO sites (ldindex +0x052 and cbl +0x068). Same rule as
# INERT_ALU: a bit inert at only one site is NOT listed. ledger/g17-load-bit-labelling.toml
INERT_LOAD = {(0, 3), (2, 3), (2, 5), (3, 0), (5, 3), (5, 5), (5, 6), (5, 7), (6, 0), (6, 3), (6, 5), (6, 6), (8, 1), (8, 3), (8, 5), (8, 6), (9, 0), (9, 1), (9, 3), (9, 4), (9, 5), (9, 6), (9, 7), (10, 0), (10, 7), (11, 0), (11, 1), (11, 2), (11, 3), (11, 4), (11, 5), (11, 6), (11, 7), (12, 0), (12, 1), (12, 2), (12, 3), (12, 4), (12, 5), (12, 6), (12, 7)}


# --- class-b: a BYTE load (4 bytes) -------------------------------------------------------
# Element width selects the instruction class: byte loads are class-b, half/word are class-f.
# ledger/g17-load-width-format.toml. dest and base are causal (ledger/g17-classb-is-a-load.toml);
# bytes 2 and 3 are NOT modelled, and the corpus's dominant sub-form (byte1=0x00, byte2 in
# 40/80/00) is a different shape from the byte-load sub-form measured here.
OWNED_CLASSB = {0: 0xF0, 1: 0xFF}

def decode_classb(u):
    return dict(dest=u[0] >> 4, base=u[1])

def encode_classb(dest, base, template):
    if not 0 <= dest <= 15: raise ValueError("dest r%d out of range" % dest)
    u = bytearray(template)
    u[0] = (u[0] & 0x0F) | ((dest & 0xF) << 4)
    u[1] = base & 0xFF
    return bytes(u)


# --- mov.wide.imm: the 8-byte class-c move-immediate -------------------------------------
# A 32-bit constant, every bit located by flipping one immediate bit at a time from a fixed base
# so the varint length could not shift underneath the measurement (spike/accel/re/immmap.py).
# All 32 bits resolved to exactly one position each. byte1[7] is a marker, always set in the wide
# form; byte2 is 0x02 in every instance seen. ledger/g17-mov-wide-immediate.toml
_IMM_MAP = ([(i, 1, i) for i in range(7)] +            # imm[0:6]   -> byte1[0:6]
            [(7+i, 4, 1+i) for i in range(4)] +        # imm[7:10]  -> byte4[1:4]
            [(11+i, 5, 2+i) for i in range(2)] +       # imm[11:12] -> byte5[2:3]
            [(13+i, 6, i) for i in range(8)] +         # imm[13:20] -> byte6[0:7]
            [(21+i, 7, i) for i in range(4)] +         # imm[21:24] -> byte7[0:3]
            [(25+i, 3, 1+i) for i in range(7)])        # imm[25:31] -> byte3[1:7]
OWNED_MOVIMM = {0: 0xF0, 1: 0x7F, 3: 0xFE, 4: 0x1E, 5: 0x0C, 6: 0xFF, 7: 0x0F}  # byte0 = dest

def decode_movimm(u):
    """The destination is read through MOVIMM_DEST, all eight bits of it.

    It used to return `u[0] >> 4` - the low nibble alone - while encode_movimm wrote the full
    eight-bit field the note below describes. So the two directions disagreed above r15: asking
    for r16 wrote byte2[6] and read back r0, and the compiler's own selfcheck reported
    "dest decoded 0, selection asked 16" on affine_index. Apple writes destinations at r121..r128
    in 25 of 400 sampled instances and sets byte2[6] in 47 of 540, so the encoder was right and
    the decoder was four bits short.
    """
    v = 0
    for ib, by, bb in _IMM_MAP: v |= ((u[by] >> bb) & 1) << ib
    return dict(imm=v, dest=_bits_get(u, MOVIMM_DEST))

# THE MOVIMM DESTINATION IS EIGHT BITS, NOT FOUR. byte0[7:4] is only its low nibble; bits 4..7
# live at byte2[6], byte2[7], byte2[3], byte2[4], with weights 16, 32, 64 and 128 - established by
# mutation, every delta a power of two. The four-bit field confined every materialised constant to
# r0..r15 and left the other four bits at the donor's value.
MOVIMM_DEST = ((0, 4), (0, 5), (0, 6), (0, 7), (2, 6), (2, 7), (2, 3), (2, 4))


def encode_movimm(imm, template, dest=None):
    if not 0 <= imm <= 0xFFFFFFFF: raise ValueError("imm %d is not a 32-bit value" % imm)
    u = bytearray(template)
    for ib, by, bb in _IMM_MAP:
        u[by] = (u[by] & ~(1 << bb)) | (((imm >> ib) & 1) << bb)
    if dest is not None:
        if not 0 <= dest <= 255: raise ValueError("movimm dest r%d out of range" % dest)
        _bits_put(u, MOVIMM_DEST, dest)
    return bytes(u)


# --- loop.trip.k: the K-loop trip count -------------------------------------------------
# A 6-byte class-2 instruction whose byte1 carries the matmul2d reduction depth. Located by
# elimination - a K=64 against K=128 differential leaves nine differing bytes and only this one
# moves the result - and authored to reproduce the compiler's own output for depths the source
# never used. isa/g17-scalar-isa.toml loop.trip.k ; ledger/g17-k-loop-trip-count-authored.toml
#
# The observable only tracks K when the accumulation control is cleared; with it set the tile is
# independent of the trip count, which is why a naive patch of byte1 alone appears inert.
OWNED_LOOPTRIP = {1: 0xFF}

def decode_loop_trip(u):
    return dict(k=u[1] - 96)

def encode_loop_trip(k, template):
    if not 0 <= k + 96 <= 0xFF:
        raise ValueError("K=%d needs byte1=%d, which does not fit; K>=192 changes the form" % (k, k+96))
    u = bytearray(template); u[1] = k + 96
    return bytes(u)


# --- cmp.pair.imm: 4-byte class-3 compare + 2-byte class-a, one predicate ------------------
# isa/g17-scalar-isa.toml cmp.pair.imm. The 8-bit constant is interleaved across FOUR bytes of
# TWO instructions, which is why a bit-solver found nothing and it had to be fitted by hand.
#
# THE RELATION IS NOT AUTHORABLE. The ISA entry says so in as many words: the class-a byte0
# separates >, >=, < and <= as 0xca, 0x8a, 0xc8, 0x08, but two of those bits are also immediate
# bits, "a third moves too and is not separated. NOT resolved." So encode_cmp_pair writes the
# immediate and keeps every relation bit from the template. A caller that needs a different
# relation must say so and be refused, not served a guess.
#
# THE COMPARED REGISTER IS NOT MODELLED EITHER - the entry lists only imm. Both gaps are named by
# the compiler when it hits them, which is how Track A tells Track B what to recover.
_CMPP_IMM = [(0, 4, 6), (1, 4, 7), (2, 2, 7), (7, 3, 2)]      # (imm bit, byte index, bit)
_CMPP_MID = (5, 1)                                            # imm[6:3] -> byte5[4:1]

def decode_cmp_pair(u):
    v = 0
    for ib, by, bb in _CMPP_IMM: v |= ((u[by] >> bb) & 1) << ib
    v |= ((u[_CMPP_MID[0]] >> _CMPP_MID[1]) & 0xF) << 3
    return dict(imm=v)

def encode_cmp_pair(imm, template):
    if not 0 <= imm <= 0xFF: raise ValueError("cmp.pair immediate %d is not 8-bit" % imm)
    u = bytearray(template)
    for ib, by, bb in _CMPP_IMM:
        u[by] = (u[by] & ~(1 << bb)) | (((imm >> ib) & 1) << bb)
    u[_CMPP_MID[0]] = (u[_CMPP_MID[0]] & ~0x1E) | (((imm >> 3) & 0xF) << 1)
    return bytes(u)

# --- cmp.imm: compare against an immediate, the form used when the operand is a builtin -----
# isa/g17-scalar-isa.toml cmp.imm.split, confidence causal + EXECUTED.
#
# Two 2-byte instructions. The immediate is EIGHT BITS SPLIT THREE WAYS, and its low bit lives in
# the second instruction alongside the relation - which is why a bit-solver looking at either
# instruction alone found nothing:
#
#     byte0[4]      imm[7]
#     byte1[6:1]    imm[6:1]          byte1[7] is always set, byte1[0] always clear
#     byte2[4]      imm[0]
#     byte2[3:0]    the RELATION      0x4 unsigned >   0x5 unsigned <
#     byte3         the compared REGISTER - NOT modelled, inherited from the template
#
# Reproduces 22 compiler outputs exactly, including the >= and <= cases, which the compiler
# canonicalises to > K-1 and < K+1. Validated by execution over 8 variants x 16 threadgroup
# widths, 128/128, each predicting the exact width at which the branch starts being taken.
# ledger/g17-compare-immediate-executed.toml
# The compared REGISTER is not in this pair at all - byte3 is constant at 0x02 across five
# different physical registers. It is the 2-byte instruction immediately BEFORE the pair:
#
#     0a <(reg << 1) | 1>      r0 -> 0a 01   r1 -> 0a 03   r2 -> 0a 05   r3 -> 0a 07   r5 -> 0a 0b
#
# Executed: with r1 holding tp.x and r0 holding a stale tg.x, `0a 03` and `0a 01` give branch
# thresholds of gw >= 22 and gw >= 43 respectively - two registers, two distinct thresholds, both
# exact over 12 grid sizes. ledger/g17-compare-immediate-executed.toml
OWNED_CMP_SRC = {1: 0xFE}

def is_cmp_src(u): return len(u) == 2 and u[0] == 0x0A and (u[1] & 1)
def decode_cmp_src(u):
    return dict(reg=(u[1] >> 1) & 0x3F, source_modifier=32 if u[1]&0x80 else 0)
def encode_cmp_src(reg, template=b"\x0a\x03", source_modifier=0):
    # Apple's decoder reads byte1[7] as operand4's modifier32, not source
    # register bit6. The former 7-bit writer and reader agreed with each other
    # while encode(70) actually described register6 with modifier32.
    if type(reg) is not int or not 0 <= reg <= 0x3F:
        raise ValueError("cmp source register must be in 0..63; bit6 is a separate source modifier")
    if type(source_modifier) is not int or source_modifier not in (0,32):
        raise ValueError('cmp source modifier must be explicit 0 or 32')
    return bytes([template[0], (reg << 1) | 1 | (0x80 if source_modifier else 0)])

CMP_REL = {"gt": 0x4, "lt": 0x5}

# THE COMPARE'S SOURCE LIFETIME, byte0[3] of the relation word - byte 2 of the six.
#
# Set, the compare RELEASES the register it read and the next reader gets zero. Clear, the register
# survives. Executed both ways in one program: `if (t > 8) C[t] = v*3; else C[t] = v+100;` with the
# index register reused across the compare gives every guarded lane slot 0 with the bit set, and
# matches Apple's compilation of the same source on all 32 lanes with it clear. Nothing else in the
# instruction moved. ledger/g17-compare-source-lifetime.toml
#
# It was a template constant until then, and the template carried it SET - which is why this
# compiler modelled cmp.src as an unconditionally destructive read and why the first branching
# kernel it ever compiled end-to-end was wrong. Apple clears it in 785 of its 854 six-byte
# compares; the 50 it sets are the compares whose source is dead afterwards.
#
# This is the fifth inherited-lifetime defect. memory g17-modifier-operand-lifetimes
OWNED_CMP_IMM = {0: 0x18, 1: 0x7E, 2: 0x1F, 3: 0x00}

def decode_cmp_imm(u):
    rel = {v: k for k, v in CMP_REL.items()}.get(u[2] & 0x0F)
    return dict(imm=(((u[0] >> 4) & 1) << 7) | (u[1] & 0x7E) | ((u[2] >> 4) & 1), rel=rel,
                keep=not (u[0] >> 3) & 1)

def encode_cmp_imm(imm, rel, template, keep):
    """`keep` is not optional: a compare that releases a live register is a silent wrong answer."""
    if not 0 <= imm <= 0xFF:
        raise ValueError("cmp.imm takes an 8-bit immediate; %d is out of range" % imm)
    if rel not in CMP_REL:
        raise ValueError("cmp.imm relation %r not recovered; only %s are"
                         % (rel, "/".join(sorted(CMP_REL))))
    u = bytearray(template)
    u[0] = (u[0] & ~0x18) | (((imm >> 7) & 1) << 4) | (0 if keep else 0x08)
    u[1] = (u[1] & ~0x7E) | (imm & 0x7E)
    u[2] = (u[2] & ~0x1F) | ((imm & 1) << 4) | CMP_REL[rel]
    return bytes(u)

def roundtrip_cmp_imm(u):
    d = decode_cmp_imm(u)
    return d["rel"] is not None and encode_cmp_imm(template=u, **d) == bytes(u)


def roundtrip_cmp_pair(u):
    return encode_cmp_pair(template=u, **decode_cmp_pair(u)) == bytes(u)


# --- mov.imm.short: the 2-byte immediate move ---------------------------------------------
# isa/g17-scalar-isa.toml mov.imm.short, confidence causal: "imm = byte1. Small immediates only;
# 200 already needs the wide form."
#
# IDENTIFIED BY byte0 == 0x04 EXACTLY, not by the class nibble. class4.2 holds 1135 instructions
# in Apple's corpus whose byte0 high nibble varies (0x04, 0xa4, 0xc4, 0x84, 0x44 ...), so the
# nibble is carrying something and only one sub-form is this instruction. Modelling the whole
# family would score a real field model against instructions it cannot reach - the same error
# avoided when branch.4 was split out of classe.4.
OWNED_MOVSHORT = {1: 0xFF}

def is_mov_short(u):
    return len(u) == 2 and u[0] == 0x04

def decode_mov_short(u):
    return dict(imm=u[1])

def encode_mov_short(imm, template):
    if not 0 <= imm <= 0xFF:
        raise ValueError("mov.imm.short takes an 8-bit immediate; %d needs mov.imm.wide" % imm)
    u = bytearray(template)
    u[1] = imm & 0xFF
    return bytes(u)


# ---- the block-operand add: dest = src1 + block[const] (handoff 10aa) ----------------------------
# op10282, 12 bytes, Apple's consumer of a uniform preload (S1/M1/M4: r105 = r105 + block[16 | 12]).
# Fields located by a decoder sweep of every bit of S1's instruction (each bit flipped alone, decoded
# alone; retained as results/g17-s23-whole-program-v1/sweeps/alu_block_S1.json by tools/g17fieldsweep.py) and checked against the members that
# vary the constant (S1 16, M1 16, M4 12 in this operand order). Registers are eight-bit fields whose
# low three bits coincide with alu.add.imm's recorded dest and src1 - the same family. Bits that change
# the opcode or the operand mode are left as the template holds them; no hardware has run this form.
ALU_BLOCK_TEMPLATE = bytes.fromhex("270404" "1a21" "00a302" "2884" "0300")     # S1 main +20: dest r105, src1 r105, block[16]
ALU_BLOCK_DEST = ((0, 4), (7, 3), (7, 4), (0, 7), (7, 5), (2, 7), (2, 3), (2, 4))    # (byte, bit) of dest bit 0..7
ALU_BLOCK_SRC1 = ((1, 1), (3, 5), (3, 6), (3, 7), (5, 7), (8, 0), (8, 1), (8, 2))
ALU_BLOCK_CONST = ((8, 6), (8, 7), (9, 0), (9, 1), (9, 2), (9, 3), (9, 4), (9, 5), (9, 6))   # const bits 0..8, in BYTES
ALU_BLOCK_SRC1_LIFETIME = (1, 7)       # the 'imm:16' after src1: 16 = release (S1's fetch register dies here); byte1 bit7 flips it to 48 - left as the template's


# THE INDEXED 4-COMPONENT LOAD AND STORE (handoff 10ae; integration's 65c31193): what Apple selects for
# `out[gid.x] = in[gid.x] + c` on device uint4 buffers - op12709/8 loads [descriptor + index] into a 4-tuple,
# op17256/8 stores a 4-tuple to [descriptor + index (+ displacement)]. Carriers located by the decoder sweep
# of the V0/V4/V6 witnesses (results/g17-op17262-v1/sweeps): the tuple register is STORE_SRC's seven bits,
# the descriptor constant is byte1 (bytes, 4 x rank), the index register byte3[7:1], the displacement
# byte6[7:5] + byte7[4:0] in units of EIGHT bytes (16 -> 2, 32 -> 4), and byte5[2] is the index register's
# lifetime (set = release, the trailing imm:16; clear = keep, imm:0). NOT the corpus op17262/14, which is
# addressed through a REGISTER PAIR (a 64-bit pointer) this compiler has no pipeline for - see g17op17262.
VEC4_LOAD_TEMPLATE = bytes.fromhex("0f040308780a1000")      # V4 +16: tuple r0..r3 <- [desc 4 + r4], keep index; the index came through an ALU
# THE 14-BYTE LOAD when the index register comes STRAIGHT from a special-register read (V0 +4: read_sr r4 then the
# load): byte5[5] and byte7[7] set and a six-byte tail - the same shape as the prologue loads' composite (10ac).
# Inherited from the witness whole; the carriers in bytes 0..7 are the 8-byte form's (decoder round trip checks it).
VEC4_LOAD14_TEMPLATE = bytes.fromhex("0f040308782a10804100800000" "00")
VEC4_STORE_TEMPLATE = bytes.fromhex("0f000308610e1000")     # V0 +66: [desc 0 + r4] <- tuple r0..r3, release index
VEC4_TUPLE = STORE_SRC
VEC4_DESC = tuple((1, b) for b in range(8))
VEC4_INDEX = tuple((3, b) for b in range(1, 8))
VEC4_DISP = ((6, 5), (6, 6), (6, 7), (7, 0), (7, 1), (7, 2), (7, 3), (7, 4))
VEC4_INDEX_RELEASE = (5, 2)


# THE COMPONENT COUNT IS byte4[6:5] = n-1, the SAME field position as the stores' (handoff 10am). Measured on
# four retained members whose only difference is the count: results/g17-vec2load-compiles-v1 gives byte4 = 0x18,
# 0x38, 0x58, 0x78 for one, two, three and four components, which is (0,0), (0,1), (1,0), (1,1) in those two bits.
# The opcode NUMBER follows it in steps of nine - 12682, 12691, 12700, 12709 - which is the same step the store
# families take, and the delivery records that lattice with the axes it does and does not account for.
VEC_COUNT = ((4, 5), (4, 6))


# THE ACCESS SIZE IS byte7[6:5] AND IT IS NOT INDEPENDENT OF THE COUNT. The decoder renders it as the
# trailing operand - an undescribed one, absent from g17auth.fields - in BYTES, and the two bits are a
# code rather than a number: 0x20 -> 1, 0x40 -> 4, 0x60 -> 8, 0x00 -> 16. Measured by moving the field
# on one encoding and reading it back, and confirmed against Apple's corpus, where the value is exactly
# the access rounded up to a power of two:
#
#     op12682/8  n=1   4 bytes   0x40   999 of 1019 distinct encodings
#     op12691/8  n=2   8 bytes   0x60   211 of  218
#     op12700/8  n=3  16 bytes   0x00    11 of   11
#     op12709/8  n=4  16 bytes   0x00   118 of  128
#
# (the 0x20 minority is a ONE-BYTE access, a width this encoder does not author.)
#
# THIS WAS THE BUG, and the docstring below already named it for n=1 without noticing it applied one
# field over. VEC4_LOAD_TEMPLATE is a FOUR-component witness, so every n reached by writing VEC_COUNT
# alone inherited the four-component access size. n=3 and n=4 are both 0x00 and were right by accident;
# n=2 emitted 16 bytes for a two-component load - an encoding that appears in ZERO of Apple's 218
# distinct op12691/8 encodings. memory:flip-a-field-not-a-bit, memory:an-undeclared-operand-is-inherited.
VEC_SIZE = ((7, 5), (7, 6))
VEC_SIZE_CODE = {1: 0b01, 4: 0b10, 8: 0b11, 16: 0b00}


def vec_access_bytes(n):
    """The access size Apple pairs with `n` components: 4*n rounded up to a power of two."""
    want = 4 * n
    return next(s for s in (1, 4, 8, 16) if s >= want)


def encode_vec4(kind, tuple_base, index, desc_const, disp=0, release_index=True, wait_sr=False, n=4,
                access_bytes=None):
    """kind 'load' (op12709/8, or /14 with wait_sr: the index straight from a special-register read) or 'store'
    (op17256/8). Registers in this compiler's numbering (105 + r).

    `n` is the component count, 2..4, and defaults to FOUR so that every byte the vector-memory delivery
    (handoff 10ae) authored is unchanged. n=1 is refused: the one-component load is op12682, which has its own
    encoder and its own address fields, and reaching it by clearing this field would author a form whose other
    fields this function does not write."""
    if kind not in ("load", "store"): raise ValueError(kind)
    if n not in (2, 3, 4):
        raise ValueError("the vector forms carry two, three or four components (byte4[6:5] = n-1); n=%r. One "
                         "component is op12682 for a load and op17229 for a store, each with its own encoder" % (n,))
    if wait_sr and kind != "load": raise ValueError("wait_sr is the load's 14-byte form only")
    if not 0 <= tuple_base <= 127 or not 0 <= index <= 127: raise ValueError("vec4 registers are seven-bit fields (got tuple %d, index %d)" % (tuple_base, index))
    if desc_const % 4 or not 0 <= desc_const <= 255: raise ValueError("vec4 descriptor constant %d: 4 x rank, one byte" % desc_const)
    # UNITS, CORRECTED FOR LOADS (MM 25.141.17): the field value F adds F BYTES to a vector LOAD's address, measured 10 of
    # 10 on hardware in the 8- and 14-byte forms. This function still takes `disp` as 8 x field, the store's original
    # reading (witnessed only at 0/16/32 and not re-measured), so a load caller passes 8 x its byte offset.
    if disp % 8 or not 0 <= disp < 8 * 256: raise ValueError("vec4 displacement %d: units of eight bytes, eight bits" % disp)
    u = bytearray((VEC4_LOAD14_TEMPLATE if wait_sr else VEC4_LOAD_TEMPLATE) if kind == "load" else VEC4_STORE_TEMPLATE)
    _bits_put(u, VEC4_TUPLE, tuple_base); _bits_put(u, VEC4_DESC, desc_const); _bits_put(u, VEC4_INDEX, index); _bits_put(u, VEC4_DISP, disp // 8)
    _bits_put(u, VEC_COUNT, n - 1)
    # WRITTEN FROM n, NOT INHERITED. Without this the template's four-component size travels to every
    # count (see VEC_SIZE above). n=3 and n=4 are unchanged by it, which is what keeps the vector-memory
    # delivery's retained bytes identical; n=2 is corrected from 16 bytes to 8.
    # access_bytes overrides the pairing for a narrower element: the packed half2 load (op12655) moves
    # two 16-bit halves, 4 bytes, through the same two-component encoding
    _bits_put(u, VEC_SIZE, VEC_SIZE_CODE[access_bytes or vec_access_bytes(n)])
    b, bit = VEC4_INDEX_RELEASE; u[b] = (u[b] & ~(1 << bit)) | ((1 if release_index else 0) << bit)
    return bytes(u)


def decode_vec4(u):
    b, bit = VEC4_INDEX_RELEASE
    # `n` IS ALWAYS REPORTED, and this is the one place in the merge where I kept my side over
    # integration's. Theirs elided the key at n == 4 to preserve the established four-component
    # report shape. Two reasons not to: a consumer writing d["n"] then raises KeyError on the
    # commonest case, which is where a located field gets quietly lost; and the report-shape cost
    # it was avoiding has already been paid - results/g17-vector-memory-v1 is regenerated with
    # every program byte proved identical, so nothing downstream still expects the old shape.
    # VEC_COUNT is byte4[6:5] and the decoder reads it; a decoder that hides a field it read at the
    # field's most common value is the shape of half the defects in this tree. Reversible if
    # integration wants the elision back - it is one conditional here and one expectation in
    # g17cc._DEC.
    return dict(tuple_base=_bits_get(u, VEC4_TUPLE), desc_const=_bits_get(u, VEC4_DESC), index=_bits_get(u, VEC4_INDEX), disp=8 * _bits_get(u, VEC4_DISP), release_index=bool((u[b] >> bit) & 1), n=_bits_get(u, VEC_COUNT) + 1)


def roundtrip_vec4(u, kind):
    d = decode_vec4(u)
    return encode_vec4(kind, tuple_base=d["tuple_base"], index=d["index"], desc_const=d["desc_const"],
                       disp=d["disp"], release_index=d["release_index"], n=d.get("n", 4),
                       wait_sr=(kind == "load" and len(u) == 14)) == bytes(u)


def encode_alu_block(dest, src1, const, template=ALU_BLOCK_TEMPLATE):
    """dest = src1 + block[const]. Registers in this compiler's numbering (the decoder names them 105 + r)."""
    # the decoder accepts 0..143 in each register field and refuses 144 and above (a sweep of every value; the
    # high combinations collide with something the sweep did not name) - the encoder stops where the decoder does
    if not 0 <= dest <= 143 or not 0 <= src1 <= 143: raise ValueError("alu.block registers: the decoder validates 0..143 only (got dest %d, src1 %d)" % (dest, src1))
    if not 0 <= const < 512: raise ValueError("alu.block constant %d exceeds the nine-bit field" % const)
    u = bytearray(template); _bits_put(u, ALU_BLOCK_DEST, dest); _bits_put(u, ALU_BLOCK_SRC1, src1); _bits_put(u, ALU_BLOCK_CONST, const)
    return bytes(u)


def decode_alu_block(u):
    return dict(dest=_bits_get(u, ALU_BLOCK_DEST), src1=_bits_get(u, ALU_BLOCK_SRC1), const=_bits_get(u, ALU_BLOCK_CONST))


# --- the delivered-byte liveness check --------------------------------------------------------
# WHY THIS EXISTS, AND WHAT IT REPLACED.
#
# `halfzero` (results/g17-halfzero-runtime-v1) dispatched and returned the fill value on lanes
# 1..31. The refusal written from it said the cause was a half store consuming a value whose SSA
# ancestry contains a half LOAD, and refused that whole family. Re-reading the retained bytes says
# otherwise: the failing store's value came from a WORD load (op12682 at +0030), and the actual
# defect is two instructions earlier - the store at +0016 RELEASES its index register reg:110
# (operand 6 = 16) and the very next instruction reads reg:110 to compute the second index. For
# lane 0 the index is zero either way, which is exactly the one lane that came back correct.
#
# Today's compiler writes that lifetime from liveness, so the same IR now emits the one byte at
# +0x1b as a keep, and root's hardware receipt for those bytes passed (10 queries, 2 workers, 640
# words exact). The rule is therefore not "no half loads": it is that a register must not be read
# after an instruction released it. This function checks that on the DELIVERED bytes, which is a
# different instrument from the liveness pass that wrote them - an encoder that puts the bit in the
# wrong slot (op14391 becoming op14392 once did exactly that) is invisible to the pass and visible
# here.
#
# THE OPERAND TABLE IS EXPLICIT, PER MEASURED FORM, because a heuristic "a register followed by an
# immediate" would read a DESTINATION as a source on op17193, whose operand 0 is the stored value.
# A form absent from the table is REPORTED as unmodelled rather than silently passing: silence
# would make this check say "clean" about a program it never looked at.
#
#   opcode: (length, ((register operand index, its lifetime operand index), ...))
LIFETIME_OPERANDS = {
    1004:  (12, ((2, 3),)),                  # widen f16->f32           32 keeps, 16 releases
    1016:  (12, ((2, 3),)),                  # narrow f32->f16
    10279: (12, ((3, 4),)),                  # add reg,imm
    10282: (12, ((2, 3), (4, 5))),           # add reg,reg
    10288: (12, ((3, 4),)),                  # alu reg,imm
    10289: (12, ((3, 4),)),                  # alu reg,imm
    14391: (14, ((3, 4),)),                  # shl reg,imm
    2190:  (16, ((2, 3), (4, 5), (6, 7))),   # fma, three sources
    424:   (4,  ((2, 3), (4, 5))),           # bitwise reg,reg
    13575: (4,  ((2, 3), (4, 5))),           # bitwise reg,reg
    12646: (14, ((5, 6),)),                  # half load: the INDEX register, 0 keeps, 16 releases
    12682: (14, ((5, 6),)),                  # word load: same
    17193: (14, ((0, 1), (5, 6))),           # half store: the stored VALUE and the index
    # FORMS WITH NO REGISTER SOURCE AT ALL are listed with an empty operand set rather than left
    # out, so `unmodelled` names only forms whose lifetimes genuinely are not modelled. op14059
    # reads a SPECIAL register (an SR index, not a GPR) and op555/op11842/op684 read nothing.
    14059: (4,  ()),                         # read_sr
    555:   (4,  ()),                         # sixteen-bit zero move
    11842: (8,  ()),                         # movimm32
    684:   (4,  ()),                         # end
}
RELEASE_VALUE = 16          # the low byte that frees the operand; 0 and 32 both keep it alive

# EVERY LIFETIME OPERAND IS CHECKED, AND THE FIRST VERSION OF THIS GOT THAT WRONG TWICE.
#
# Version 1 refused every register read after any release. That refused four programs whose bytes
# have executed correctly, so I restricted it to a memory instruction's INDEX operand on the
# grounds that the other releases were harmless. Root's review showed both steps were wrong:
#
#   * THE FOUR WERE ALIASING, NOT HARMLESS RELEASES. In syn-s7f595f1cd1's sibling syn379, +0x56
#     releases r433 and +0x17c reads r433 - but +0x62 WRITES r113, which is the same physical
#     register: the 32-bit file is based at 105 and the sixteen-bit file at 425, so 105+i and 425+i
#     are one register. The value released at +0x56 is gone by +0x17c, and that read is a different
#     value. So those programs are not evidence that a conversion release preserves anything.
#   * AND THE RESTRICTION CONTRADICTS AN EXECUTED PAIR. syn-s7f595f1cd1's repair changed a
#     CONVERSION source lifetime (op1004 operand 3, +0x56) and two store VALUE lifetimes (op17193
#     operand 1, +0x21e and +0x26c). Four bytes. The pre-repair program 5cf5f15a FAILED on hardware
#     and the repaired 103e5b09 PASSED (10 queries, two workers, 480 halves). Value and conversion
#     lifetimes therefore matter exactly as much as the index one halfzero showed.
#
# So: all modelled lifetime operands, register ALIASES resolved, and reads examined BEFORE a write
# ends the question - a form that reads and writes the same register would otherwise hide its read
# of the old value behind its own destination.
#
# THE REGISTER FILES, WHAT ALIASES WHAT, AND HOW MUCH OF IT A WRITE COVERS.
#
#   105 + i   the 32-bit register i        - the WHOLE register, both halves
#   425 + i   a sixteen-bit register i     - ONE half (the file the half forms name)
#   281 + i   the other sixteen-bit file i - the OTHER half (asm's MOVIMM16_FILE_BIT)
#
# THE FIRST VERSION OF THIS COLLAPSED ALL THREE and let ANY write end a release, which root's
# review showed is wrong in a way that reads as clean: release the whole of r110, write the
# sixteen-bit r430, then read r110 again, and the half the write did not touch still carries the
# released value - yet the search stopped at the write. Four such shapes came back "conclusive,
# clean" when they are not.
#
# So a register operand carries a MASK of the halves it covers, and a write subtracts only its own
# halves from what is still outstanding. WHICH sixteen-bit file is the low half and which the high
# is NOT established in this project's tables - asm's own note says only that byte3[0] selects "the
# OTHER" one - so the two are modelled as distinct halves A and B without claiming an order. That
# is enough for every question here: what matters is whether a write covers the half a later read
# needs, not which end of the word it sits at.
REG_FILES = (105, 281, 425)
HALF_A, HALF_B = 1, 2                     # the two sixteen-bit files, order deliberately unnamed
WHOLE = HALF_A | HALF_B
_FILE_MASK = {105: WHOLE, 425: HALF_A, 281: HALF_B}


def alias_key(reg):
    """The physical register a decoded register number names, or the number itself if unknown.

    Returns ("phys", index) for a number in a modelled file, so the three files alias correctly,
    and ("raw", reg) otherwise - never confused with an aliased one. The EXTENT is a separate
    question; see alias_mask.
    """
    for base in sorted(REG_FILES, reverse=True):
        if reg >= base and reg - base < 128:
            return ("phys", reg - base)
    return ("raw", reg)


def alias_mask(reg):
    """Which halves of the physical register this number covers: WHOLE, HALF_A or HALF_B."""
    for base in sorted(REG_FILES, reverse=True):
        if reg >= base and reg - base < 128:
            return _FILE_MASK[base]
    return WHOLE


# THE DESTINATION OPERAND, PER FORM, and nothing inferred for a form that is not here.
#
# Reading "whatever register token 0 names" as the destination is what version 1 did, and it is
# wrong twice over: op17193's operand 0 is the stored VALUE, a source; and for a form whose
# lifetimes are not modelled at all, inventing a write from its first token manufactures the very
# fact that ends a search. An unmodelled form's registers are therefore INCONCLUSIVE - the
# released value's fate passes through an instruction this table cannot read - and are reported as
# a coverage gap instead of being answered either way.
WRITES_OPERAND_0 = frozenset({1004, 1016, 10279, 10282, 10288, 10289, 14391, 2190, 424, 13575,
                              12646, 12682, 14059, 555, 11842})
CONTROL_FLOW = frozenset({462, 458, 582, 578, 579, 450})


def _lifetime_reads(rows):
    """Per instruction: registers read, which of those it released, what it wrote, and coverage."""
    out = []
    for _off, length, opcode, toks in rows:
        reads, released, wrote, unmodelled = {}, {}, None, None
        entry = LIFETIME_OPERANDS.get(opcode)
        if entry is None or entry[0] != length:
            unmodelled = (opcode, length)
            # every register this instruction MENTIONS is then uncertain - it may read it, write it,
            # or both, and this table cannot say which
            mentions = {alias_key(int(t[4:])) for t in toks
                        if isinstance(t, str) and t.startswith("reg:")}
            out.append(({}, {}, None, unmodelled, mentions))
            continue
        for reg_at, life_at in entry[1]:
            if len(toks) <= max(reg_at, life_at):
                continue
            r, life = toks[reg_at], toks[life_at]
            if not (isinstance(r, str) and r.startswith("reg:")):
                continue
            if not (isinstance(life, str) and life.startswith("imm:")):
                continue
            n = int(r[4:])
            key, mask = alias_key(n), alias_mask(n)
            reads[key] = reads.get(key, 0) | mask
            if (int(life[4:]) & 0xFF) == RELEASE_VALUE:
                released[key] = released.get(key, 0) | mask
        t0 = toks[0] if toks else None
        if opcode in WRITES_OPERAND_0 and isinstance(t0, str) and t0.startswith("reg:"):
            n = int(t0[4:])
            wrote = (alias_key(n), alias_mask(n))
        out.append((reads, released, wrote, None, set()))
    return out


def read_after_release(code, decode=None):
    """Registers read after an instruction released them, from the delivered bytes.

    Returns a dict. `findings` are releases read again afterwards - the defect two executed pairs
    show, as (released_at, (file, index), read_at). `unmodelled` lists forms whose lifetimes this
    table cannot read; `inconclusive` lists released registers whose fate reaches one of those
    forms; `control_flow` lists branch opcodes, because with a back edge the last mention is not
    the last read and this analysis does not apply. `conclusive` is true only when none of those
    three is non-empty - so a program this cannot answer never reads as clean.

    A register WRITTEN after its release starts a new value, so the search for a later reader of a
    released register stops at the next instruction that writes it. Straight-line programs only:
    with a back edge the last mention is not the last read, so a program containing one is reported
    as unmodelled rather than answered.
    """
    # THE DECODER IS THE CALLER'S, and this function will not put anything on sys.path to find one.
    #
    # It used to insert the repo's tools/ directory into sys.path so it could import the packed
    # checker itself. test_g17librarycompat caught that: the package must never mutate the import
    # path, because a library that does decides what its host imports. The compiler does not call
    # this at all (that would also put a subprocess in code generation), so every caller is a test
    # or the review harness, and both already have the decoder importable.
    if decode is None:
        try:
            import g17packedcheck as decode
        except ImportError as exc:
            raise ImportError(
                "read_after_release needs a decoder: pass decode=<module with .decode(bytes)>, or "
                "import it from a context where g17packedcheck is importable. This function will "
                "not modify sys.path to find one") from exc
    rows = decode.decode(code)
    per = _lifetime_reads(rows)
    unmodelled = sorted({u for _r, _rel, _w, u, _m in per if u is not None})
    control = sorted({op for _o, _l, op, _t in rows if op in CONTROL_FLOW})
    findings, inconclusive = [], []
    for i, (_reads, released, _wrote, _u, _m) in enumerate(per):
        for key, mask in sorted(released.items()):
            outstanding = mask
            for k in range(i + 1, len(rows)):
                reads_k, _rel_k, wrote_k, unmod_k, mentions_k = per[k]
                # READ FIRST, and only where it touches a half still outstanding: an instruction
                # that reads and writes the same register would otherwise hide its read.
                if reads_k.get(key, 0) & outstanding:
                    findings.append((rows[i][0], key, rows[k][0]))
                    break
                if unmod_k is not None and key in mentions_k:
                    inconclusive.append((rows[i][0], key, rows[k][0], unmod_k))
                    break
                if wrote_k is not None and wrote_k[0] == key:
                    # A WRITE COVERS ONLY ITS OWN HALVES. What it does not cover still carries the
                    # released value, so the search continues on the remainder instead of stopping.
                    outstanding &= ~wrote_k[1]
                    if not outstanding:
                        break               # fully rewritten: a different value from here on
    return dict(findings=findings, unmodelled=unmodelled, inconclusive=inconclusive,
                control_flow=control,
                conclusive=not (unmodelled or inconclusive or control))
