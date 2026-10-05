#!/usr/bin/env python3
"""Strict G17 instruction-stream parser: it walks a kernel or FAILS LOUDLY.

There is no fallback stride. A parser that guesses a length when it does not recognise a form
produces plausible desynchronised garbage, and that is worse than no parser: an earlier probe
in this project mutated an instruction 456 bytes from its intended target because a walk had
silently desynced, and the resulting null result was reported as an architectural finding.

So: every instruction must be framed by a rule that was MEASURED, or walk() raises Desync with
the offset and bytes that defeated it. A parse that "consumes __text exactly" proves nothing on
its own - any fallback stride that divides the section length will do that - so callers should
also check that the walk lands on independently known instruction starts.

Lengths here are ground truth, from three sources: strides between N repetitions of one
operation located by differential compilation; instructions of that size authored and executed
correctly; and forms whose next instruction is well-formed at the implied boundary across many
sites. None of it rests on tiling, which was retracted (see ledger/g17-length-rule.toml).

    python3 tools/g17dis.py <applegpu-object>      walk _agc.main and report
"""
import os, sys

class Desync(Exception):
    def __init__(self, off, b):
        self.off, self.bytes = off, b
        super().__init__("no measured framing rule at +0x%03x: %s" % (off, b[off:off+16].hex(" ")))

def is_mac(u):
    """tensor.mac, 10 bytes; signature validated across every catalogued issue."""
    return (len(u) >= 10 and (u[4] & 0xF7) == 0x22 and (u[6] & 0xFB) == 0xA0
            and (u[8] & 0xEF) == 0x00 and (u[7] & 0x1F) == 0x02)

# Set by length() for the instruction it just framed: True when the length came from a rule
# that was never measured, only assumed. The DECODE capability the mission defines requires no
# guessed lengths, so the guessed fraction has to be countable, not merely commented on.
LAST_GUESSED = False

# Three-way provenance for the length just returned. "walks clean" has now failed to discriminate
# on every framing question put to it - class-b 2 vs 4, class-7/f 12 vs 16, the tensor issue 10 vs
# 16, class-1 2 vs 6, and the 6-byte polymorphic form - so it is a health check, not progress. What
# counts is how much of the corpus is framed by a rule that was proven by execution.
#   causal      NOP-coherence, transplant, or additivity at a load-bearing site
#   structural  a compiler differential or an inspection argument, never executed
#   guessed     never measured at all
LAST_PROV = "structural"
CAUSAL_RULES = set()

# THE SIGNATURE GUARD. A framing must not consume the start of an instruction whose signature is
# strong enough that finding it by scanning is not a coincidence. Three qualify:
#
#   four-byte class-e   class nibble in byte0, 0x0E in byte3      found via branch targets
#   tensor.mac          five constraints, ~1 in 8.4M              found by scanning
#   barrier             six exact bytes, and byte1 = 0x51 occurs in 0.02% of the corpus, so the
#                       pair 27 51 is rare - the marginal estimate predicts ~0 chance hits in
#                       135k bytes and 100 were found
#
# ENTROPY WAS CHECKED BEFORE THE BARRIER WAS TRUSTED. Its last four bytes are 00 06 00 00 and
# `06 00` is the compiler's filler, so the signature looked low-entropy; the rare byte is 0x51.
# A signature whose bytes are common would produce coincidences, and the rule would then be
# shortening correct framings. ledger/g17-signature-consistency.toml
# EACH SIGNATURE GETS THE RANGE ITS EVIDENCE SUPPORTS, not the widest range that parses.
# A first version scanned every even k up to the framing length and cost 1.5 points of causal
# framing: the class-e test is only twelve bits (a nibble plus a byte), about 1 in 4096, so at
# eight distinct offsets over thousands of instructions it produces false hits. The mac and
# barrier signatures are far stronger and can be looked for further in.
# CORRECTED 2026-09-04. This range was (2, 4, 6), justified in the comment above as "offsets
# confirmed by branch targets" - collected while the forward PC base was wrong by 4. Re-derived
# with the base reverted to 0 AND this guard switched off, so the branch decode and the framing
# were independent, base+0 targets confirm 8 and 10 far more often than anything else, and confirm
# 6 not once:
#
#     offset into container    2    4    6    8   10
#     base+0 confirmations     1    1    0   12  139
#
# Each confirmation is a twelve-bit class-e signature sitting exactly where an independently
# decoded displacement points: 1 in 4096 per site, and it happened at 119 sites the old range
# could not reach. The heads this produces are 8- and 10-byte class-7/f, lengths already attested
# 1662 and 2460 times elsewhere in the corpus, so the split invents no new form.
# ledger/g17-signature-range-rederived-under-base0.toml
_SIG_RANGE = {"class_e": (2, 4, 8, 10),            # re-derived under base+0, guard disabled
              "mac":     (2, 4, 6, 8, 10),          # offsets seen in the signature scan
              "barrier": (2, 4, 6, 8, 10)}
BARRIER_SIG = bytes.fromhex("275100060000")
# alu.bitwise.imm's operation selector, causal in isa/g17-scalar-isa.toml: the (byte0, byte2,
# byte3) triple distinguishes and/or/xor and was AUTHORED IN BOTH DIRECTIONS - from an AND host,
# re-encoding gave 1, 15 and 14 for the three operations, so the observable names the operation.
# A known instruction's signature, which is what makes it usable as a framing guard.
# alu.bitwise.imm. WIDENED 2026-09-04: this was three exact 4-byte words, all with byte1 == 0x80,
# and byte1 is a FIELD - kernels with several live bitwise operations emit 0x82 there and every
# one of those was invisible to both the signature guard and the family classifier. Found by
# looking for the `and` in a kernel that plainly contained six of them and finding none.
#
# byte0 selects xor from and/or and byte2/byte3 select the operation, so the test is still
# ~1 in 8M by chance: two byte0 values out of 256, and two exact bytes.
# ledger/g17-bitwise-signature-too-narrow.toml
_BITWISE_OPS = {(0x30, 0x30), (0x32, 0x38)}

def is_bitwise(u):
    return (len(u) >= 4 and u[0] in (0x06, 0x07) and (u[2], u[3]) in _BITWISE_OPS)

BITWISE_SIGS = (bytes.fromhex("07803030"), bytes.fromhex("07803238"),
                bytes.fromhex("06803238"))
EXEC_RESTORE_SIG = bytes.fromhex("3e03400e")
# The EXEC MASK WRITE, and the second exec.restore variant, both found by looking at what the
# compiler emits for a NESTED conditional (spike/accel/re/prednest.py):
#
#   1e 00 00 0e   writes the exec mask from the predicate - measured, not inferred
#                 ledger/g17-exec-mask-is-the-conditional.toml
#   be 03 40 0e   exec.restore with bit7 of byte0 set; it appears as the INNER restore of a
#                 nested region while 3e 03 40 0e closes the outer one
#
# Both are 32 exact bits, the same strength as EXEC_RESTORE_SIG, so both may be looked for at any
# offset. They had to be: a three-level nest showed `22 84 04 22 1e 00` framed as ONE six-byte
# class-2 instruction, swallowing the mask write whole, and a compare pair buried inside a 12-byte
# class-7 span. Neither desyncs, so nothing downstream could see it.
EXEC_MASK_SIG = bytes.fromhex("1e00000e")
def _is_exec_restore(u):
    return len(u) >= 4 and (u[0] & 0x7F) == 0x3E and u[1] == 0x03 and u[2] == 0x40 and u[3] == 0x0E
# read_sr.direct: byte1 is the special-register index, measured - 0x9c/9d/9e threadgroup position
# x/y/z, 0xa0/a1/a2 thread position x/y/z. isa/g17-scalar-isa.toml, causal.
SR_INDEX = frozenset((0x9C, 0x9D, 0x9E, 0xA0, 0xA1, 0xA2))
_BOUND_OPC = {0x37, 0x27}   # tensor.bound.a / .b opcode_byte0, isa/tensor-isa.toml

def _swallows_signature(b, p, n):
    """The smallest k in (0, n) at which a recognisable instruction starts, or None."""
    best = None
    for k in _SIG_RANGE["class_e"]:
        if k < n and p + k + 4 <= len(b) and (b[p+k] & 0xF) == 0xE and b[p+k+3] == 0x0E:
            best = k; break
    for k in _SIG_RANGE["mac"]:
        if k < n and is_mac(b[p+k:p+k+10]) and (best is None or k < best): best = k; break
    for k in _SIG_RANGE["barrier"]:
        if k < n and bytes(b[p+k:p+k+6]) == BARRIER_SIG and (best is None or k < best):
            best = k; break
    return best


# class-b length, keyed on byte3. Mined from Apple's decoder over the FULL class-b population:
# 2533 of 2533 instructions have their length determined by byte3, with no exceptions and no
# ambiguous value. The key is inside the shortest length in the class (4), as it must be.
# ledger/g17-length-rules-mined-from-the-oracle.toml
_CLASSB_LEN = {0x00: 4, 0x02: 8, 0x08: 4, 0x2A: 8, 0x80: 10}

def _classb_len(b, p):
    """class-b: byte3 selects the length. Unseen byte3 values fall through to the older rules
    rather than being guessed at - the table covers what the oracle has shown, and no more."""
    if (b[p] & 0xF) != 0xB or p + 4 > len(b): return None
    n = _CLASSB_LEN.get(b[p+3])
    return n if n is not None and p + n <= len(b) else None

# class-3 with byte1 == 0x00: the length needs TWO bytes. byte2 alone determines 291 of 326;
# adding byte3 makes it exceptionless over the full population, 326/326 in nine combinations.
# byte2 == 0x07 is always 10 whatever byte3 holds; byte2 == 0x06 splits 8 or 10 on byte3.
# ledger/g17-length-rules-mined-from-the-oracle.toml
_CLASS3_00_LEN = {(0x00, 0x01): 4, (0x06, 0x02): 8, (0x06, 0x80): 10, (0x07, 0x00): 10,
                  (0x07, 0x02): 10, (0x07, 0x80): 10, (0x0E, 0x02): 8, (0x20, 0x00): 4,
                  (0x20, 0x01): 4}

def _class4_len(b, p):
    """Classes 4 and c - byte0 low THREE bits 100; bit 3 is not part of the length - mined from
    Apple's decoder, 2026-09-22. op14061 is encoded as both 0x24 and 0x1c, which is how class c
    was found to share the rule.

    Byte 1 bit 7 clear is the 2-byte short form; set, the base is 4 bytes, and byte 2's low two
    bits extend it - one set is 8, both set is 10:

        b1.7  b2.0  b2.1     probe corpus (1,745)       vendor corpus (91,423, never fitted on)
         0     *     *       2 bytes: 194               agrees
         1     0     0       4 bytes: 896               agrees
         1     1|0   0|1     8 bytes: 643               agrees
         1     1     1       10 bytes: 12               agrees

    Class c follows it too (4,629 probe instances, all b1.7 set). Both classes together: 186,700
    vendor instances, zero disagreements.

    Unique among every 1-, 2- and 3-bit subset of bytes 0..3, found on the probe corpus and then
    checked with zero disagreements on the vendor corpus. It replaces a guessed 2-byte default
    that framed op14061 (read_sr, 8 bytes, 56,641 vendor instances) as 2 and desynced three
    constant programs that tools/g17gate.py reports as MISLAND now that its failures reach the
    exit status - and two narrower class-4 rules, one of which it corrects (see below)."""
    if p + 3 > len(b) or b[p] & 7 != 4:
        return None
    if not (b[p + 1] >> 7) & 1:
        return 2
    return {0: 4, 1: 8, 2: 8, 3: 10}[b[p + 2] & 3]


def _class3_00_len(b, p):
    """class-3, byte1 == 0x00: (byte2, byte3) selects the length. Combinations the oracle has not
    shown fall through rather than being guessed."""
    if (b[p] & 0xF) != 3 or p + 4 > len(b) or b[p+1] != 0x00: return None
    n = _CLASS3_00_LEN.get((b[p+2], b[p+3]))
    return n if n is not None and p + n <= len(b) else None

def _class3_02_len(b, p):
    """class-3 with byte1 == 0x02 is TEN bytes, exceptionless.

    Mined from Apple's decoder over the FULL population of that family, not the subset where
    g17dis happened to disagree: 233 of 233 instructions have length 10. g17dis framed them as 6,
    which is 101 of the largest single disagreement in a 250-object sample.

    THE KEY HAD TO BE CONSTRAINED. A first pass mined the disagreement subset and found byte14 and
    byte15 "determining" the length 100% of the time - impossible, since a 10-byte instruction has
    no byte14. Those keys were reading the FOLLOWING instruction, which correlates because code is
    not random. A key that sizes an instruction must lie inside the SHORTEST candidate length; the
    same alignment discipline that caught the descriptor's f1 artefact.
    ledger/g17-length-rules-mined-from-the-oracle.toml
    """
    return 10 if (b[p] & 0xF) == 3 and p + 2 <= len(b) and b[p+1] == 0x02 else None

# class-7/f with byte1 == 0: byte6 selects the form, and within one group of forms a single bit
# extends the length. Both halves mined from Apple's decoder over the full population of 4201
# instructions, and together they cover 4079 of them (97.1%) with ZERO exceptions.
#
#   byte6 0x60 0x68 0x70 0x78   length = 12 + 4 * (byte10 bit 0)      2509 instructions, 0 wrong
#   byte6 in _WIDE7_CONST        a constant length                     1570 instructions
#   byte6 0x10 0xa0 0xa1 0xa2    still mixed - 122 instructions, left to the older rules
#
# The extension bit is the same structure the per-opcode census found independently: instruction
# length is a base plus an optional suffix, selected by one bit inside the shortest form.
# ledger/g17-length-is-base-plus-extension-bit.toml
_WIDE7_CONST = {0x83: 12, 0xA3: 12, 0xA4: 10, 0xA9: 12, 0xAA: 10, 0xAB: 10, 0xAE: 10, 0xAF: 10}
_WIDE7_EXT   = {0x60, 0x68, 0x70, 0x78}

def _wide7_len(b, p):
    """class-7/f, byte1 == 0: a constant length per byte6, or 12 + 4 * byte10[0] for one group."""
    if p + 12 > len(b): return None
    if (b[p] & 0xF) not in (7, 0xF) or b[p+1] != 0: return None
    k = b[p+6]
    if k in _WIDE7_CONST:
        n = _WIDE7_CONST[k]
        return n if p + n <= len(b) else None
    if k in _WIDE7_EXT:
        n = 12 + 4 * (b[p+10] & 1)
        return n if p + n <= len(b) else None
    return None

def _branch_len(b, p):
    """A branch is TEN bytes: the four-byte word plus six more.

    g17dis framed it as a 4-byte branch followed by a separate "six zero bytes are one
    instruction", and that split is wrong. Apple's decoder gives length 10 for all 106 branch
    sites in a 242-object sample, under two opcodes (462 and 458).

    Confirmed by EXECUTION, not only by the oracle: the same program with a 4-byte branch faults
    with a GPU address fault when the branch is taken, and with the full 10-byte instruction it
    completes and the join runs. Every branch this compiler authored before today was truncated.
    ledger/g17-branch-is-ten-bytes.toml
    """
    from agxforge.g17 import asm as g17asm
    return 10 if p + 10 <= len(b) and g17asm.is_branch(b[p:p+4]) else None

def _bitwise_len(b, p):
    """alu.bitwise.imm is EIGHT bytes, not the twelve the class-7 rule gives it.

    isa/g17-scalar-isa.toml carried size = "unresolved" for this form: the walk framed 12 while the
    tail looked like the head of the following store, and NOP coherence read the region as
    6+2+2+6. Settled by differential compilation - build the same `x & 15` with the result stored
    to C[8], and the four bytes the 12-byte framing swallows decode as a complete store.8 with
    slot = 8, which is exactly what the source writes:

        +0x060  07 80 30 30 a3 02 c4 83     alu.bitwise.imm, and, imm 15
        +0x068  0f 04 03 00 21 0c 10 24     store.8  src=r0 n=2 slot=8

    The old framing produced `wide.12.opc0` there and then mis-framed the remainder as `21 0c`
    and `10 24`. ledger/g17-bitwise-size-resolved.toml
    """
    return 8 if is_bitwise(b[p:p+4]) else None

def _class7_wide(b, p):
    """True where the class-7/f rules below would frame 12 OR 16 bytes - the spans wide enough to
    hide a four-byte class-e instruction. The 14-byte form (byte11 == 0x34) is excluded because it
    was established causally at two sites with the partial-overwrite signature.

    Covering 16 as well as 12 was not an afterthought: the last two branch targets that still
    landed mid-instruction were inside SIXTEEN-byte spans carrying the same embedded class-e at
    offset 6, and excluding them left exactly those two misses."""
    if p + 10 >= len(b): return False
    if p + 12 <= len(b) and b[p+11] == 0x34: return False
    return True


def _length_raw(b, p):
    """Measured framing only. None means 'not recovered' - never a guess."""
    global LAST_GUESSED, LAST_PROV
    LAST_GUESSED = False
    LAST_PROV = "structural"
    if p >= len(b): return None
    n = _classb_len(b, p)
    if n is not None:
        LAST_PROV = "causal"      # mined from Apple's decoder, 2533/2533
        return n
    n = _class3_00_len(b, p)
    if n is not None:
        LAST_PROV = "causal"      # mined from Apple's decoder, 326/326
        return n
    n = _class4_len(b, p)
    if n is not None and p + n <= len(b):
        LAST_PROV = "causal"      # mined from Apple's decoder, 1,745/1,745 fitted, 91,423/91,423 held out
        return n
    n = _class3_02_len(b, p)
    if n is not None and p + n <= len(b):
        LAST_PROV = "causal"      # mined from Apple's decoder, 233/233
        return n
    n = _wide7_len(b, p)
    if n is not None:
        LAST_PROV = "causal"      # mined from Apple's decoder, 952/952
        return n
    n = _branch_len(b, p)
    if n is not None:
        LAST_PROV = "causal"      # Apple's own decoder, 106/106, plus an executed A/B
        return n
    n = _bitwise_len(b, p)
    if n is not None:
        LAST_PROV = "causal"      # differential compilation + the decoded following store
        return n
    c = b[p] & 0xF
    # b1/b2 are read lazily: classes with a constant length must not touch them, or the
    # self-containment audit in tools/g17consist.py cannot tell a real context dependence
    # from an unused eager read.
    if c in (0, 1, 4, 5, 6, 8, 0xa, 0xb, 0xd, 0xe):
        if c == 0xe: return 4 if p + 4 <= len(b) else None
        if c == 5:   return 2 if p + 2 <= len(b) else None
        if c == 0xa: return 2 if p + 2 <= len(b) else None
        if c == 8:   return 2 if p + 2 <= len(b) else None
        if c == 0xb:
            # FOUR bytes, measured 2026-09-04. Was covered by the unmeasured "return 2" default
            # below, which split 13882 corpus instructions into a class-b pair plus a class-0
            # pair and made class-0 look like the largest family in the corpus.
            #
            # The discriminator is a logical one, and it is the test the 16-byte retraction says
            # to use. At four sites in fwd6, two different tail variants:
            #
            #   +0x0d4  9b 00 | 00 00     NOP all 4: tile 32768 UNCHANGED, fully coherent
            #                             NOP head only: 16384, degraded
            #                             NOP tail only: total fault, every observable zero
            #   +0x056  8b 00 | 80 00     NOP all 4: 32768 unchanged
            #                             NOP head only: total fault
            #                             NOP tail only: 8192, degraded
            #   +0x0d8, +0x0dc            NOP all 4: 32768 unchanged, reproducing +0x0d4
            #
            # Removing BOTH halves is harmless while removing EITHER half alone is harmful. Two
            # independent instructions cannot behave that way: deleting one of them cannot be
            # worse than deleting both. One 4-byte instruction whose partial overwrite is
            # malformed explains every cell.
            #
            # As always here, walk-clean cannot decide it: 2 and 4 both walk all 643 kernels with
            # zero desyncs. ledger/g17-classb-four-bytes.toml
            LAST_PROV = "causal"; return 4 if p + 4 <= len(b) else None
        if c == 0xd:                                         # NOT determined; {2,4,6,8} tie, 16 excluded
            LAST_GUESSED = True; LAST_PROV = "guessed"
            return 2 if p + 2 <= len(b) else None
        # TWO CLASS-4 RULES STOOD HERE and are superseded by _class4_len, which runs first and was
        # mined from Apple's decoder over classes 4 and c. The byte1 == 0x82 / byte2 == 0x10 form
        # (ledger/g17-class4-second-four-byte-form.toml) agrees with it: four bytes. The byte2 ==
        # 0x01 rule said FOUR bytes from a NOP-coherence probe; Apple's decoder reads all 379 such
        # instances as op14060 at EIGHT bytes, and the corrected ledger says so
        # (ledger/g17-class4-read-sr-four-bytes.toml).
        if c == 0 and p + 6 <= len(b) and b[p:p+6] == b"\x00" * 6:
            # SIX zero bytes are ONE instruction, not three 2-byte class-0. This is the largest
            # single guessed form in the decoder: class-0 with byte0 == 0x00 was 7424 instructions,
            # 42.8% of everything still framed by an unmeasured default.
            #
            # It is executable code, not padding. ac2-32x32x64 +0x2fe, following a class-e at
            # +0x2fa, baseline tile 65536:
            #
            #   NOP the 4-byte class-e at +0x2fa        0   total fault
            #   NOP the six zero bytes at +0x2fe        0   total fault   <- load-bearing
            #   NOP all ten                         65536   completely unchanged
            #
            # NOPping the six alone is fatal, so they are not filler. NOPping all ten is harmless
            # while either part alone is fatal, so the region is coupled.
            #
            # RETRACTED ON THE WAY: those three cells first read as ONE 10-byte instruction. They
            # are not. 1022 of 2085 corpus class-e of this shape are followed by NO zero bytes at
            # all - ac2-32x32x64 +0x4ae is '3e 03 40 0e' followed by a class-7 - so the class-e is
            # genuinely 4 bytes and the six are separate. The non-additive signature came from
            # control-flow coupling, not from a single instruction.
            #
            # What makes the six ONE unit rather than three is the corpus distribution: after a
            # class-e of this shape the zero-byte count is 0 (1022 sites) or exactly 6 (1063
            # sites), and NEVER 1, 2, 3, 4, 5 or more than 6. Three independent 2-byte
            # instructions would produce runs of one and two as well, and those exist in
            # abundance elsewhere - 1613 runs of a single '00 00' in the corpus.
            #
            # Scoped to six consecutive zero bytes, so a lone '00 00' is untouched and still
            # guessed. ledger/g17-six-zero-bytes-one-instruction.toml
            return 6
        if c == 0 and p + 6 <= len(b) and b[p] == 0x30 and (b[p+1] & 0x30) == 0x30:
            # SIX bytes, and the reason this rule exists is bigger than the rule: byte0's LOW
            # NIBBLE IS NOT UNIVERSALLY AN INSTRUCTION-CLASS TAG. This form appears with low
            # nibble 2 ('or', framed 6 correctly by the class-2 rule) and low nibble 0 ('and',
            # shattered into three by the 2-byte default) - the nibble is part of the opcode.
            #
            # Proven by WHOLE-FORM TRANSPLANT, not bit fuzzing. Two kernels identical except this
            # operator are byte-identical except bytes +0x06a..+0x06b, same length, and both have
            # their next instruction at +0x070. Operands loaded at runtime, a=12 b=10, so every
            # candidate operation gives a distinct answer; out[100] is the operation and out[110]
            # is a LATER instruction's result. All predictions were registered before running.
            #
            #   baseline OR                 out[100]=14  out[110]=22   predicted 14 / 22
            #   transplant AND (2 bytes)    out[100]= 8  out[110]=22   predicted  8 / 22
            #   transplant XOR (1 byte)     out[100]= 6  out[110]=22   predicted  6 / 22
            #
            # Semantic transplants execute correctly and the successor is untouched, so the six
            # bytes are one coherent unit. The framing controls, on the AND image:
            #
            #   NOP all 6                   out[100]= 0  out[110]=22   clean; successor INTACT
            #   NOP bytes 4-5               out[100]= 8  out[110]=12   partial: corrupts successor
            #   NOP bytes 0-1               out[100]=DEADBEEF, out[110]=DEADBEEF - TOTAL FAULT
            #
            # NOPping exactly six is clean and the successor still runs, which fixes +0x070 as the
            # next boundary. NOPping either part is harmful, which makes the six indivisible. The
            # head cell alone refutes the three-instruction reading: removing a supposed standalone
            # 2-byte class-0 left the program unable to complete either store.
            #
            # SCOPED to byte0 == 0x30, the value measured. The wider signature - byte0 high nibble
            # 3 with byte1 bits 4 and 5 set - matches 813 corpus instructions across low nibbles
            # 8, a and 5, which are classes measured causally as 2 bytes at single sites. That
            # conflict is unresolved and is NOT assumed away here.
            # ledger/g17-byte0-low-nibble-not-a-class-tag.toml
            LAST_PROV = "causal"; return 6
        if c == 1 and p + 6 <= len(b) and b[p] == 0x01 and b[p+1] == 0x00 \
                and b[p+2] == 0x00 and b[p+3] == 0x00 and b[p+4] == 0x00:
            # SIX bytes. 1009 corpus sites have exactly this shape, '01 00 00 00 00 XX', and the
            # 2-byte default split every one of them into a class-1 plus two class-0 - which is
            # how class-0 came to look like the second largest family in the corpus.
            #
            # Measured at ac2-32x32x64 +0x520 ('01 00 00 00 00 20'), baseline 65536. Damage by
            # which bytes are replaced with the compiler's own 06 00 filler:
            #
            #   bytes 0-1   8192      bytes 0-3  32768      bytes 0-5   8192
            #   bytes 2-3      0      bytes 2-5   8192
            #   bytes 4-5   8192
            #
            # Three independent 2-byte instructions is refuted outright: removing the first two
            # costs 32768 while removing the first costs 8192 and the second costs nothing, and
            # deleting two of three cannot be worse than deleting all three. One 6-byte
            # instruction explains every cell - any overwrite touching a live field voids its
            # 8192 contribution, bytes 2-3 are an inert field, and NOPping 0-3 leaves '00 20' as
            # a residue that is decoded afresh and does 24576 of additional damage.
            #
            # Scoped to the measured shape. The 47 other class-1 '01 00' sites, whose bytes 2..4
            # are not zero, stay on the unmeasured 2-byte default below and are still a guess.
            # ledger/g17-class1-six-bytes.toml
            LAST_PROV = "causal"; return 6
        LAST_GUESSED = True; LAST_PROV = "guessed"   # classes 0, 6, other class-4/1 forms
        return 2 if p + 2 <= len(b) else None
    # Everything below reads b1/b2, so those bytes must exist. The guard used to demand three
    # bytes for EVERY class, which framed the last two bytes of a kernel as unparseable; Apple's
    # own driver shaders end that way and four of them reported a false desync at len-2.
    if p + 3 > len(b): return None
    b1, b2 = b[p+1], b[p+2]
    if c in (7, 0xf):
        if b1 == 0x02 and b2 == 0x20 and (b[p] & 0x10): return 4
        # A 4-byte class-7 sub-form. Found by inspection: instructions like 17 02 20 14 were
        # being framed as 16 bytes because byte10 of the FOLLOWING textbook ALU happened to
        # satisfy the 0x81 extension test. Adding it raises corpus conformance from 2143 to
        # 2154 conforming ALUs while keeping 20/20 causal anchors and every kernel walking.
        if b1 in (0x51, 0x69) and b2 == 0x00:
            # BARRIER, 6 bytes. Identified by a single-variable differential over memory scopes
            # (ledger/g17-barrier-identified.toml): threadgroup is 27 51 00, device 0f 69 00,
            # mem_none 07 51 00, and requesting BOTH scopes emits the two back to back in 12
            # bytes - which is what shows the form is 6 and not 12. Framed as 12 the
            # threadgroup case swallowed the following 8-byte instruction, an instruction that
            # appears intact in the barrier-free and both-barriers kernels.
            # Discrimination is exact in the corpus: of class-7/f with byte2 == 0x00, all 227
            # with byte1 == 0x51 are followed by 06 00 00 and none of the other 2500-plus are.
            return 6
        if b1 == 0x08 and (b2 & 0x3F) == 0x07:
            # TEN bytes, for byte2 in {0x07, 0x47, 0x87, 0xc7}. byte2's top two bits are an
            # INDEX, not a length selector: the corpus has all four values in near-equal numbers
            # (269/275/280/269 for the 'Xf 08 ?7 .. .. .. 61' shape), exactly as byte0's top two
            # bits do. Keying length on them gave one form three different lengths - 0x07 -> 10,
            # 0x47 -> 16, and 0x87/0xc7 falling through to the ALU-immediate test - which is the
            # same defect already retracted for the load/store classifier, where byte1 was the
            # BASE REGISTER (ledger/g17-loadstore-classifier-defect.toml).
            #
            # Measured causally in ac2-32x32x64, where the region +0x4fc..+0x54f tiles as
            # 10,10,6,10,6,10,6,10,6 with the 6-byte class-1 form below, and the 16-byte reading
            # cannot tile it at all: +0x4fc and +0x506 are two of these back to back with no
            # 6-byte unit between them. Baseline tile sum 65536, every cell 64:
            #
            #   NOP 10 at +0x4fc            57344   128 cells zeroed, one localized rectangle
            #   NOP 10 at +0x506            24576   640 cells
            #   NOP 20 at +0x4fc            16384   768 cells
            #
            # The removals are EXACTLY additive - 49152 = 8192 + 40960 - which is the signature of
            # two independent instructions with a boundary at +0x506. Framed as 16, +0x4fc..+0x50b
            # would be one instruction and NOPping its first 10 bytes would be a partial
            # overwrite; every partial overwrite measured in this project is malformed, yet this
            # one is the cleanest, smallest, most localized removal in the whole probe.
            # ledger/g17-tensor-issue-ten-bytes.toml
            LAST_PROV = "causal"; return 10 if p + 10 <= len(b) else None
        if b1 == 0x08 and b2 == 0x03: return 8                           # store.device
        if b2 == 0x03:                                                   # load / store form
            # NOT keyed on byte1 any more. byte1 is the BASE REGISTER
            # (ledger/g17-base-register-map.toml), so keying length on it framed the same load
            # as 8 bytes reading base 1 or 2 and 12 reading base 0 - a length that depended on
            # which buffer was addressed. Measured at four per-chain sites in the loadlen probe:
            # NOP 8 removes exactly one chain and leaves the ADDRESS in the destination (8
            # instead of 8001, 10 instead of 10001), while NOP 12 always damages a second chain.
            LAST_PROV = "causal"; return 14 if b[p+7] & 0x80 else 8
        # 16 when byte10 & 0x81 == 0x81, else 12. RESTORED 2026-09-04 after a wrong correction
        # the same day; the retraction is in ledger/g17-class7-12-not-16.toml.
        #
        # The evidence that matters is a three-cell test at a LOAD-BEARING site, fwd6 +0x1cc,
        # worth 4096 of the 32768 tile sum:
        #   12 bytes intact,  tail byte15 0x08 -> 0x60   tile 28672   tail breaks the instruction
        #   first 12 NOPed,   tail byte15 0x08           tile 28672
        #   first 12 NOPed,   tail byte15 0x60           tile 28672   tail does NOTHING alone
        # The tail changes behaviour only while the preceding twelve bytes are present, so it is
        # a field of that instruction and not a separate one. A 2-byte class-0 instruction would
        # still act after its neighbour was removed.
        #
        # What misled the first attempt: NOP-first-12 equals NOP-all-16, and NOPping the last 4
        # is inert. Both are equally true of a 16-byte instruction whose trailing field simply
        # does not matter for the values tested (0x00, 0x08, 0x18 are all inert; 0x60 is not).
        # Inertness is not separability. Note also that walk-clean cannot decide this either way:
        # both 12 and 16 walk all 635 kernels with zero desyncs.
        # A 12-BYTE FRAMING THAT SWALLOWS A FOUR-BYTE CLASS-E INSTRUCTION IS WRONG.
        #
        # Three separate patterns turned out to be one statement. Branch targets landed
        # mid-instruction at +6, +10 and +8 inside 12-byte spans, and in every case a class-e
        # instruction was sitting inside the span:
        #
        #   87 02 | 3e 03 40 0e | ...        exec.restore at offset 2   target +6
        #   87 02 82 06 | 5e 63 40 0e | ...  class-e at offset 4        target +8
        #   0f 52 2a a0 a5 32 | 9e 60 00 0e  class-e at offset 6        target +10
        #
        # A four-byte class-e has its class nibble in byte0 and 0x0E in byte3, so the test is the
        # same at every offset. Taking the smallest k that matches:
        #
        #   branch-target consistency   96.5% -> 100.00%, 773 of 773
        #   branches exposed            723 -> 773; fifty had been hidden inside 12-byte spans
        #   kernels walking clean       942, zero desyncs
        #
        # THE EVIDENCE AND THE APPLICATION DIFFER IN SIZE. Twenty-five sites are confirmed by a
        # branch target; the rule fires far more often than that, on the strength of a pattern
        # whose two conditions co-occur 21.7x more than chance and 97.6% of the time in one
        # direction. The framings it produces are STRUCTURAL, not causal.
        # ledger/g17-exec-restore-swallowed.toml
        # A WIDE FRAMING MUST NOT SWALLOW A RECOGNISABLE INSTRUCTION. Two signatures are strong
        # enough to assert this against a 12- or 16-byte length: a four-byte class-e (nibble in
        # byte0, 0x0E in byte3) and a tensor.mac (five constraints, ~1 in 8.4M by chance).
        #
        # The class-e case was found by branch targets; the tensor.mac case by scanning for the
        # signature and asking where the walk had put boundaries. Both are the same error:
        #
        #   mac on-boundary   5897 -> 6008,  mid-instruction 116 -> 5
        #   branch targets    100.0%, 825 of 825
        #   942 kernels clean, zero desyncs
        #
        # tensor.mac at 10 bytes is causal (ledger/g17-tensor-issue-ten-bytes.toml) and walk()
        # already tests it FIRST - so a mac is only missed when a PRECEDING wide framing consumed
        # its start, which is exactly what this repairs.
        # The four-byte class-e test is only twelve bits, so it stays HERE - inside the
        # class-7/f path, at the three offsets branch targets confirmed - rather than in the
        # global wrapper. Applying it globally, or at more offsets, produces false hits.
        if _class7_wide(b, p):
            for _k in _SIG_RANGE["class_e"]:
                if p + _k + 4 <= len(b) and (b[p+_k] & 0xF) == 0xE and b[p+_k+3] == 0x0E:
                    LAST_PROV = "structural"
                    return _k
        if p + 10 < len(b):
            if (b[p+10] & 0x81) == 0x81: LAST_PROV = "causal"; return 16
            if p + 12 <= len(b) and b[p+11] == 0x34:
                # FOURTEEN bytes. byte11 == 0x34 predicts a trailing pair almost perfectly in the
                # corpus: 2246 of 2247 such instructions were followed by a phantom class-0, and
                # the single exception is not this form at all (it is the ten-byte tensor issue
                # with byte1 == 0x34 rather than 0x08).
                #
                # Measured causally at TWO sites in ac2-32x32x64, both ALU-shaped (byte6 = 0xa1),
                # baseline tile sum 65536:
                #
                #   +0x5a  37 88 00 5a 2a 00 a1 0a b0 80 00 34 | 00 00
                #     NOP first 12          0, all 1024 cells zero - TOTAL FAULT
                #     NOP all 14        16384, 768 cells        - coherent, graded
                #     NOP trailing 2    65536, no change        - inert
                #     byte13 00 -> 20   32768, 512 cells        - coherent, scattered
                #   +0x68  37 80 00 5a 2a 00 a1 02 70 80 00 34 | 00 20
                #     NOP first 12          0, all 1024 cells zero - TOTAL FAULT
                #     NOP all 14        16384, 768 cells        - coherent, graded
                #     byte13 20 -> 00   48384, 192 cells        - coherent, graded
                #
                # Removing MORE is far less damaging than removing part: the partial-overwrite
                # signature. This is not the dependency-edge confound recorded in
                # ledger/g17-tensor-issue-ten-bytes.toml, and the trailing pair is what rules it
                # out. If those two bytes were an independent instruction, NOPping them alone
                # shows they contribute nothing at all - so removing the first 12 and removing
                # all 14 would have to agree. They do not: one is a total fault and the other is
                # a clean graded removal.
                #
                # byte13 is a live field: set to the value the compiler emits at the sibling site
                # it produces a coherent non-fatal change, which is a field, not corruption.
                # byte12 is inert at both sites. ledger/g17-alu-fourteen-bytes.toml
                return 14
            LAST_PROV = "causal"; return 12
        return None
    if c == 0xc:
        # Special-register read: byte1 IS the register index, and the whitelist was incomplete.
        # threadgroup_position_in_grid is 0x9c/0x9d/0x9e but thread_position_in_grid is
        # 0xa0/0xa1/0xa2, and those framed as 8 bytes, swallowing the following 12-byte load.
        # The differential is airtight: the tg.x and gp.x kernels are byte-identical apart from
        # this one byte, and tg.x frames correctly as 4 + 12.
        # Measured in spike/accel/re/prologue.py.
        if b1 in (0x9c, 0x9d, 0x9e, 0xa0, 0xa1, 0xa2):
            # The special-register read is 4 bytes - EXCEPT the byte2 == 0x02, byte3 == 0x00
            # sub-form, which is 8. Those are different instructions and the old rule conflated
            # them, which is why ledger/g17-classc-eight-bytes-undecided.toml could not resolve.
            #
            # The separation is total. Of class-c framed as 4:
            #     byte2 = 0x10 (any byte3)   701 sites, 701 followed by a real instruction (100%)
            #     byte2 = 0x02, byte3 = 0x00 247 sites,   0 followed by a real instruction (0%)
            # The 4-byte length was measured causally on '0c 9c 10 06' - byte2 = 0x10 - so that
            # measurement was never in conflict with this; it simply did not cover this sub-form.
            #
            # Causal, ac2-32x32x64 +0x150 '7c a0 02 00 | 60 20 00 00', baseline tile 65536:
            #     NOP the first 4        98304      NOP all 8              98304
            #     NOP the trailing 4     32768      NOP trailing bytes 0-1 32768
            #                                      NOP trailing bytes 2-3 32768
            # The trailer is load-bearing, and removing either half costs exactly what removing
            # both costs - non-additive, so it is not two independent 2-byte instructions.
            #
            # Its VALUE is inert, though: transplanting the compiler-valid variants seen elsewhere
            # ('60 00 00 00' at 29 sites, '00 20 00 00' at 2) leaves the tile at 65536 exactly.
            # Must be present and well-formed, but does not control - the sixth field in this
            # project that varies systematically without controlling anything.
            # ledger/g17-classc-eight-byte-subform.toml
            return 4 if not (b2 == 0x02 and b[p+3] == 0x00) else 8
        return 8
    if c == 0xe: return 4                                                 # end
    if c == 3:                                                            # and/or/xor immediate
        # Ground truth from NOP coherence (ledger/g17-nop-boundary-scan.toml): the operation is
        # TWO instructions, 6 bytes then 4. The old universal "class 3 = 10" conflated them and
        # was the single reason every framing search returned zero consistent assignments.
        # The "else 4" default was never measured and is now falsified at FOUR sites. Using a
        # non-destructive extent map first (spike/accel/re/c3extent.py) to find class-3
        # instructions that feed exactly ONE output, then NOP coherence at those sites only, gives
        # the categorical signal the method needs: at +0x062, +0x082 and +0x0a4 of the c3x_base
        # probe, NOP 2 zeroes exactly one output while NOP 4 invents values in OTHER chains - the
        # partial-overwrite signature. divlane +0x054 is a fourth. No site has ever measured 4.
        # CAVEAT, recorded rather than hidden: this reads byte2 to return 2, so it is NOT
        # self-contained and tools/g17consist.py flags it. Causal measurement outranks that
        # expectation, but it means the real discriminator is still unknown - plausibly these are
        # 2+2 pairs, the encoding shape this ISA uses repeatedly.
        return 6 if b[p+2] == 0x07 else 2
    if c == 0xd:
        # Length NOT determined: 2, 4, 6 and 8 all parse every corpus kernel identically because
        # class-0 fillers absorb the difference. 16 IS excluded - it desyncs rr_mulrr, missing a
        # textbook ALU at +0x5ae that every smaller value lands on. 2 chosen as the smallest
        # consistent value. ledger/g17-classd-bounded.toml
        return 2
    if c == 0xa:
        # NOP coherence in the bit probe kernel at +0x4f2: lengths 2, 4 and 6 each change exactly
        # one output, 597 -> 500 - the XOR dropped with its constant intact - so boundaries fall
        # at +2, +4 and +6 and the form is 2 bytes. Measured in SCALAR code; the class-a form in
        # tensor setup may differ and is not covered by this.
        return 2
    if c == 5:
        # NOP coherence in the minmax probe kernel at +0x504: length 2 changes exactly one
        # output, 405 -> 407, which is 7 + 400 - the min dropped leaving i, a plausible partial
        # result rather than an invented value. Longer lengths remove further whole chains.
        # The corpus does NOT discriminate: 2, 4, 6 and 8 all give 157 clean scalar tails.
        # ledger/g17-class5-two-bytes.toml
        return 2
    if c == 9:
        # NOP coherence in the madd probe kernel at +0x4f2: length 8 damages exactly one output
        # (356 -> 321, the *5 removed with +21+300 intact), 2/4/6 destroy all six and 10 yields
        # invented values. The scalar corpus does NOT discriminate the length - 2, 4, 6 and 8 all
        # give 119 clean - because class-0 fillers absorb the difference, so this rests on the
        # causal probe alone. ledger/g17-class9-eight-bytes.toml
        return 8
    if c == 2:
        # NOP coherence in chain 3 of the minmax probe kernel: at +0x500 lengths 2 and 4 destroy
        # all six outputs (partial removal), while 6 damages exactly one (405 -> 400). Six bytes
        # is the whole instruction: 22 81 26 84 15 02.
        LAST_PROV = "causal"; return 6
    if c == 8:
        # Measured causally by NOP coherence at a probe site feeding exactly one observable:
        # NOP of 2 or 4 bytes at +0x5d0 of the c8c kernel damages only out[103] (326 -> 26,
        # the +300 cleanly removed) while every other length zeroes all six outputs.
        # ledger/g17-class8-two-bytes.toml
        LAST_PROV = "causal"; return 2
    if c in (0, 1, 4, 6, 0xb): return 2   # class 4 and b were first inferred as 4 bytes from a
                                       # walk; a DFS required to frame entry..first-mac exactly
                                       # determines 2, uniquely and in every kernel tested.
    return None

def walk(text, start, limit=None):
    """Yield (offset, length, kind) from start. Raises Desync at the first unframed byte."""
    p, end = start, len(text) if limit is None else limit
    while p < end:
        if is_mac(text[p:p+10]):
            # Counted as causal: the 10-byte signature is validated across every catalogued issue.
            # It used to bypass length() entirely and so was invisible to the provenance metric,
            # which silently omitted 5560 instructions from the denominator.
            walk.prov["causal"] = walk.prov.get("causal", 0) + 1
            walk.provbytes["causal"] = walk.provbytes.get("causal", 0) + 10
            walk.framed += 1
            yield p, 10, "tensor.mac"; p += 10; continue
        n = length(text, p)
        if n is None: raise Desync(p, text)
        walk.guessed += LAST_GUESSED
        walk.framed += 1
        walk.prov[LAST_PROV] = walk.prov.get(LAST_PROV, 0) + 1
        walk.provbytes[LAST_PROV] = walk.provbytes.get(LAST_PROV, 0) + n
        yield p, n, "class %x" % (text[p] & 0xF)
        p += n
walk.guessed = walk.framed = 0
walk.prov = {}
walk.provbytes = {}

def main():
    from agxforge.g17 import machobj, agxdis
    obj = sys.argv[1]
    d = os.path.dirname(obj)
    loc = machobj.locate(obj, os.path.join(d, "out", "object", "0-0"))
    f, sz = agxdis.sections(loc["obj"]); text = loc["obj"][f:f+sz]
    entry = loc["syms"]["_agc.main"]
    print("__text %d bytes, _agc.main +0x%03x" % (sz, entry))
    n = 0
    try:
        for off, ln, kind in walk(text, entry):
            n += 1
    except Desync as e:
        print("DESYNC after %d instruction(s): %s" % (n, e)); return 1
    print("walked %d instructions to the end cleanly" % n); return 0

if __name__ == "__main__":
    sys.exit(main())


def _strong_signature_at(b, q):
    """Does a strongly-signatured instruction start at q?

    Each of these was checked for ENTROPY before being trusted, because a signature made of common
    bytes produces coincidences and the guard would then shorten correct framings:

        tensor.mac     five constraints                            ~1 in 8.4M
        barrier        six exact bytes; the rare one is 0x51 at 0.02% of the corpus
        tensor.bound   byte0 in the opcode set, then 21 20 a1      marginals give ~0 expected

    DELIBERATELY EXCLUDED after measuring them: `end` (0e 00 00 00) is three zero bytes and its
    marginals predict ~42 chance hits per 135k bytes against 9 observed - BELOW chance, so its
    apparent mid-instruction hits are noise. The six-zero form is worse. The four-byte class-e is
    twelve bits and stays inside the class-7/f path at three confirmed offsets.
    """
    if is_mac(b[q:q+10]): return True
    if bytes(b[q:q+6]) == BARRIER_SIG: return True
    if is_bitwise(b[q:q+4]): return True
    # exec.restore as an EXACT four-byte match - 32 bits, far stronger than the generic class-e
    # test (twelve bits), which is why this one may be looked for at any offset while that one
    # stays confined to the class-7/f path. 45 of these were being swallowed, mostly at offset 8,
    # where the generic test deliberately does not reach.
    if _is_exec_restore(b[q:q+4]): return True
    if bytes(b[q:q+4]) == EXEC_MASK_SIG: return True
    if (len(b) >= q + 4 and (b[q] & 0xF) == 0xC and b[q+1] in SR_INDEX
            and b[q+2] == 0x10 and b[q+3] == 0x06): return True          # read_sr.direct
    if (len(b) >= q + 8 and (b[q] & 0xF) == 0xC and (b[q+1] & 0x80)
            and b[q+2] == 0x02): return True                             # mov.imm.wide
    u = b[q:q+7]
    if (len(u) >= 7 and u[0] in _BOUND_OPC and u[4] == 0x21 and u[5] == 0x20 and u[6] == 0xA1):
        return True
    return False


def length(b, p):
    """The framing rule, with the SIGNATURE GUARD applied to every path.

    A framing must not consume the start of an instruction whose signature is strong enough that
    finding it by scanning cannot be coincidence. Applied as a wrapper rather than inside one
    class's branch because the misframings were not confined to one class: barriers were being
    swallowed by 6-byte class-2 framings as well as by 12- and 16-byte class-7 ones.

        tensor.mac   five constraints, ~1 in 8.4M by chance
        barrier      six exact bytes; byte1 = 0x51 occurs in 0.02% of the corpus, so the pair
                     27 51 is rare and the marginal estimate predicts ~0 chance hits in 135k bytes

    Only these two are applied globally. The four-byte class-e test is twelve bits, about 1 in
    4096, which is weak enough to produce false hits at many offsets, so it stays inside the
    class-7/f path at the offsets its evidence supports - see _SIG_RANGE, re-derived 2026-09-04
    under the corrected PC base. Scanning EVERY even offset cost 1.5 points of causal framing.

        barrier signatures mid-instruction   100 -> 0
        tensor.mac signatures mid-instruction 116 -> 0
        constant programs landing exactly     1000/1000, unchanged  (CORRECTED 2026-09-22: landing
            does not test framing - ledger/g17-constant-program-landing-is-weak.toml)
        1000 kernels clean, zero desyncs

    The "branch targets on a boundary, unchanged" line that used to sit here has been REMOVED, not
    updated. _SIG_RANGE is now chosen using branch targets, so that statistic can no longer
    corroborate it - it would be measuring its own input. What still can, and does, is that no
    strongly-signatured instruction (mac, barrier, exec.restore) is left mid-instruction anywhere
    in the corpus, and that nothing desyncs.

    ledger/g17-signature-consistency.toml
    ledger/g17-signature-range-rederived-under-base0.toml
    """
    global LAST_GUESSED, LAST_PROV
    n = _length_raw(b, p)
    if n and n > 2:
        for k in range(2, min(n, 12), 2):
            if _strong_signature_at(b, p + k):
                LAST_GUESSED = False
                LAST_PROV = "structural"
                return k
    return n
