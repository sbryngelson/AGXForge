#!/usr/bin/env python3
"""The G17 compiler: IR in, native instruction stream out.

    g17ir.Function
      -> select()     semantic instruction selection: IR op -> G17 form -> fields (virtual regs)
      -> allocate()   register allocation over the one scalar namespace
      -> emit()       fields -> bytes, via the canonical per-form templates in g17forms
      -> G17Program   code + the container dependencies still inherited, named explicitly

WHAT MAKES THIS A COMPILER AND NOT THE PATCHER IT REPLACES. spike/accel/re/kern1.py chose bytes
for a FIXED LIST of Apple instruction addresses and needed an Apple instruction of the right form
at each one. Here the stream is built from the IR, laid out by us, allocated by us, and the only
thing taken from Apple is one canonical template per form - a dependency that is counted in
isa/g17-forms.toml and meant to go to zero.

SELECTION IS RESTRICTED TO WHAT IS CAUSALLY RECOVERED. Every form used here has confidence
"causal" in isa/g17-scalar-isa.toml. Notably ABSENT: and/or/xor, because those are NOT alu.12
opcodes - they are a separate form selected by (byte0, byte2, byte3) whose length is still
unresolved (alu.bitwise.imm, "SIZE IS NOT ESTABLISHED"). Emitting them as an alu.12 opcode would
have produced plausible bytes with no basis. They raise Unsupported instead, which is the whole
point of a form registry: the compiler knows what it cannot yet build.
"""
import collections
import os, re, sys
import types
# THE SIBLINGS AND THE OBJECT READERS ARE PACKAGE MODULES NOW, so nothing here joins sys.path.
# These two inserts put tools/ and spike/accel/re on the path so this file could import them by
# bare name; every one of those - the IR, the assembler, the form encoders, agxdis and machobj -
# reached the package over the preceding batches, which is what made moving the compiler possible
# rather than a facade that reaches back into tools/.
_T = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_T))
from agxforge.g17 import ir as ir, asm as g17asm, assembler as g17as, auth as g17auth, forms as g17forms, tensor as g17tensor, agxdis as agxdis, cf as g17cf, registerdomain

# Buffer binding slot -> base REGISTER index. Not identity, and not yet recovered in general;
# this is the mapping observed in the three-buffer hosts used for execution tests.
BUFFER_BASE_REG = {1: 0}



# THE PER-LANE ATOMIC RETURN BOUNDARY IS OPERATION-SPECIFIC. op10090 updates memory and returns
# the old value. Compiler-generated ADD, AND, OR, SUB and XOR returns now have below-Metal whole-program
# evidence: g17purecompiledreturnaudit.py checks ADD's direct store and ALU old+1 consumer;
# g17purexorreturnaudit.py checks XOR's direct store and a byte-identical authored counterpart.
# Each tested arm passed two fresh pure IOGPU Submits on the 12-byte slot-7 form. This admits
# one direct store for ADD/AND/OR/SUB/XOR or one ordinary ADD consumer for ADD/XOR. Other shapes
# remain refused until the corresponding emitted form and wait have hardware evidence.
# Per-lane compare-exchange is a distinct opcode. Its compiler-generated direct-return program
# passed ten pure Submits over index placements R0, R1, R5, R7 and R12
# (g17purecompiledcmpxchgaudit.py). The assembler admits only the measured operation constant
# and authors R0..R12; the other values in that range are decoder-checked predictions.
# Other operation values, R13 and other consumer shapes refuse.
#
#   * THE MEMORY UPDATE HAS SCOPED WHOLE-PROGRAM HARDWARE EVIDENCE. tools/g17endtoend.py `_atomicadd_ir` dispatches
#     a per-lane atomic whose result NOTHING READS, reads the location back with an ordinary load
#     and agrees with `_atomicadd_py` (A[t] gains t+1). That program keeps compiling.
#
#   * THE CONSUMED ADD/AND/OR/SUB/XOR PATHS ARE COMPILER-GENERATED AND DISPATCHED. The emitted atomic publishes on
#     slot 7; its first ALU consumer waits on slot 7. The paired hand-authored slot-0 arm, even
#     with 64 independent intervening instructions, updated memory but returned zero.
#
# So an ADD, AND, OR, SUB or XOR program with one tested consumer shape compiles. Other consumed per-lane fetch
# operations and untested consumer shapes still refuse. Three things
# this deliberately is not:
#
#   * not a ban on the atomic TYPE - a buffer declared `metal::_atomic` with no atomic operation in
#     the program keeps its code (syn-s5186cb7b7d in SAME197 is exactly that shape, and a type ban
#     would drop it for a defect it cannot reach);
#   * not a ban on the atomic OPERATION - the update-only shape above is supported and dispatched;
#   * not keyed on a source tag, a declaration, or a reachability judgement. It reads the built IR's
#     producer/use relation, and a use is counted WHEREVER it appears, including in a block some
#     analysis might call dead. Discounting a use would require a reachability claim this compiler
#     has not established, and the conservative direction for an unvalidated read is to refuse.
#
# The uniform forms are deliberately OUT of scope and keep compiling: op10094 and op11765 both have
# dispatched programs that consume their returned value against a reference (`_wavebcast_ir` with
# `_wavebcast_py`, and `_tgatomic_ir`'s difference of two rounds). Their return path therefore has
# compiler-generated execution evidence. The remaining refusal is narrow rather than family-wide.
_RESULT_UNVALIDATED_ATOMIC_KINDS = frozenset({"atomic_add", "atomic_cmpxchg"})


def _consumed_atomic_results(fn):
    """[(producer, [consumer, ...])] for every per-lane atomic whose result another op reads.

    `is` identity, not equality: two distinct Values must never be conflated, and Value defines no
    __eq__ that would make `in` safe here.
    """
    ops = [o for blk in fn.blocks for o in blk.ops]
    found = []
    for op in ops:
        if op.kind not in _RESULT_UNVALIDATED_ATOMIC_KINDS or op.dest is None:
            continue
        users = [u for u in ops if u is not op and any(a is op.dest for a in u.args)]
        if users:
            found.append((op, users))
    return found


def _refuse_unvalidated_atomic_result(fn):
    for op, users in _consumed_atomic_results(fn):
        if op.kind == "atomic_cmpxchg":
            if len(users) == 1 and users[0].kind == "store_at":
                op.attrs["is_load"] = True
                op.attrs["consumed_result"] = True
                continue
            raise Unsupported("per-lane compare-exchange returned old value currently needs one "
                              "direct indexed-store consumer; other compiler-generated consumer "
                              "shapes lack below-Metal execution evidence")
        aop = op.attrs.get("aop", "add")
        if (len(users) == 1 and
                ((aop in ("add", "xor") and users[0].kind in ("add", "store_at")) or
                 (aop in ("and", "or", "sub") and users[0].kind == "store_at"))):
            op.attrs["is_load"] = True
            op.attrs["consumed_result"] = True
            continue
        raise Unsupported(
            "an atomic %s whose returned old value is read by %s: the compiler-generated "
            "per-lane slot-7 return path has below-Metal hardware evidence for a direct indexed "
            "store of ADD, AND, OR, SUB or XOR, or an ordinary add of ADD/XOR. Other operations and "
            "consumer shapes remain refused until their emitted form and wait are "
            "validated together"
            % (op.attrs.get("aop", "rmw"), ", ".join(sorted({u.kind for u in users}))))


def _atomic_base(slot):
    """The atomic forms' base field, which is NOT the load's and has no discriminating probe.

    The indexed pair that showed the load's base is the rank is a pair of load/store kernels; no
    probe varies an ATOMIC's buffer away from rank 0, so "it is the same law" is a guess. Where
    the two readings agree - every validated kernel, which atomics only ever on the lowest-ranked
    buffer - this emits what it always did. Where they would differ it refuses, because a wrong
    base here addresses another allocation's descriptor.
    """
    ident = BUFFER_BASE_REG.get(slot, slot)
    rank = _buf_rank(slot)
    if ident != rank:
        raise Unsupported(
            "an atomic on buffer slot %d: the identity table says base %d and the binding rank "
            "says %d, and no probe varies an atomic's buffer away from rank 0, so which the form "
            "takes is unmeasured. The load's base was measured to be the rank (indices [1,2,3] "
            "and [2,4,6] emit identical bases); do the same one-variable pair for an atomic "
            "before emitting this" % (slot, ident, rank))
    return ident


def _buf_rank(slot):
    """This function's rank for a bound slot. A slot it never bound is an error, not rank 0.

    _BUF_RANK[0] holds the ranks of the function being compiled - select() sets it - so reading it
    HERE, during lowering, is reading the current program. Reading it after the compile is what
    made abi() report another program's offsets; that is snapshot into G17Program instead.
    """
    ranks = _BUF_RANK[0]
    if slot not in ranks:
        raise Unsupported("buffer slot %d is addressed but not bound by this function (bound: %s)"
                          % (slot, sorted(ranks)))
    return ranks[slot]

# THE IMAGEBLOCK ELEMENT WIDTH, read out of slot 2 of every access: 16 for 8-bit, 17 for 16-bit,
# 18 for 32-bit, 241 for half4. It is a DECODE fact, not an authoring one - the field is
# degenerate in the operand map, and the width is selected by choosing an opcode off the
# read/write ladder instead. Kept here because it is what identifies which rung a form is on.
IB_WIDTH_CODE = {8: 16, 16: 17, 32: 18, 64: 241}

_TREG = None
def _tensor_registry():
    global _TREG
    if _TREG is None: _TREG = g17tensor.build()
    return _TREG

class Unsupported(Exception):
    """An IR op with no causally recovered lowering. Raised, never guessed around."""

# --- the machine IR -----------------------------------------------------------------------
class MInst:
    __slots__ = ("form", "fields", "defs", "uses", "size", "note")
    def __init__(self, form, size, fields, defs=(), uses=(), note=""):
        self.form = form; self.size = size; self.fields = dict(fields)
        self.defs = list(defs); self.uses = list(uses); self.note = note
    def __repr__(self):
        f = " ".join("%s=%s" % kv for kv in sorted(self.fields.items()) if kv[0] != "template")
        return "%-11s %-46s defs=%s uses=%s" % (self.form, f, self.defs, self.uses)

# ALU opcode map, byte6 = 0xA0 | op. ledger/g17-alu-opcode-map.toml - measured by sweeping thirty
# Metal expressions with runtime-loaded operands, NOT inferred from mnemonics.
ALU_OP = {"mul": 1, "sub": 2, "add": 3}
# PER-OPCODE ALU TEMPLATES. isa/g17-opmap.toml maps each IR operation to the opcode Apple's own
# compiler chooses for it, read from kernels whose single arithmetic operation is stated by the
# Metal source. tools/g17optmpl.py then harvests the modal instance of that opcode.
#
# byte6 IS a real opcode field - sub's opcode 11666 does carry 0xa2, exactly what ALU_OP writes.
# What was wrong is everything else: swapping byte6 on ADD's template leaves the other bytes
# belonging to add, and the result is bytes Apple's decoder rejects
# (ledger/g17-oracle-authority-boundary.toml). A template has to come from the right instruction,
# not from a neighbour with one byte changed.
#
# Only `add` is wired through today. sub and mul are recorded here and NOT yet substituted,
# because their templates carry mode and operand bits this compiler does not model, and swapping
# a template it cannot fully author would trade a known-wrong encoding for an unexamined one.
ALU_TEMPLATE_BY_OPCODE = {
    10279: bytes.fromhex("2700049a2d00a31a208321 00".replace(" ", "")),   # add reg,imm   len 12
    10282: bytes.fromhex("3702043a2100a30a289420 00".replace(" ", "")),   # add reg,reg   len 12
    # The two sub templates are the instructions Apple's compiler emitted for `x - 5u` and
    # `x - y` in a kernel written for the purpose (spike/accel/re/opmap.py), not the modal corpus
    # instance: their operands are known because the source states them, so every bit outside the
    # authored fields is inherited from an instruction whose meaning is known rather than from
    # whichever instance was most common.
    11666: bytes.fromhex("2f00041a2500a2024801 0b00".replace(" ", "")),   # sub reg,imm   len 12
    11667: bytes.fromhex("2f00041a2500a2028880 2100".replace(" ", "")),   # sub reg,reg   len 12
    # THE MUL FORMS ARE FOURTEEN BYTES. Templates from `x * 100u` and `x * y`, whose operands the
    # Metal source states. Apple has no mul-immediate below 256 at all: it strength-reduces, and
    # the fused-shift scale in the add forms is how - `x*3` is one add with scale 2, `x*9` one
    # add with scale 8, `x*6` an add-with-scale followed by another. SCALE_CODE predicts all of
    # their byte10/byte11 values, which is an independent confirmation of that table.
    10822: bytes.fromhex("2f00041a2100a10228190a00 0102".replace(" ", "")), # mul reg,imm len 14
    10825: bytes.fromhex("2f02041a2100a102288020000102".replace(" ", "")), # mul reg,reg len 14
    # Shifts, from `x << 13u`, `x >> 3u`, and the two register-operand kernels in opmap.py.
    14391: bytes.fromhex("2f00001a2600a002788300340000"),                  # shl reg,imm len 14
    14392: bytes.fromhex("2704001a2600a002888020340000"),                  # shl reg,reg len 14
    17013: bytes.fromhex("2f00001a2600a102f88000340000"),                  # shr reg,imm len 14
    17014: bytes.fromhex("2704001a2600a102888020340000"),                  # shr reg,reg len 14
}
# ONE-OPERAND INTEGER OPS. The opcodes are the peer's isolation sweep - one Metal operation per
# kernel, differenced against a load/store baseline - and the templates are the modal corpus
# instance of each, from which the encoder overwrites the destination, the source and the hazard.
UNARY_OPCODE = {"not": 11190, "msb": 9986, "reverse": 14047}
# op11179 is NOT in UNARY_OPCODE: its lowering below passes hazard=None, because
# writing a hazard of 0 clears byte4[3], which the witnessed template has SET and
# whose role is unmeasured. The unary path hardcodes hazard=0.
CVT_U32_F32_OPCODE = 11179
UNARY_TEMPLATE = {
    11179: g17asm.CVT_I2F_TEMPLATE,
    9986: bytes.fromhex("2702005a2a00ab020800"),    # clz,     modal of 133
    11190: bytes.fromhex("a300269a2908a802"),       # not,     modal of 30
    14047: bytes.fromhex("2700007a2e00a91a0800"),   # reverse, modal of 38
}

# THE FLOAT UNARY OPS. One ten-byte shape, ten operations, and this backend selects nine of them:
# the transcendental unit's byte6 code and, under code a0, the rounding mode at byte8[6:7]. `trig`
# is deliberately absent - it is one instruction of a twelve-instruction sine and what it computes
# alone is not established. Each template is Apple's own code for that operation, taken from the
# single-instruction lowering in the sw-f_* corpus, so nothing is transplanted between operations.
# ledger/g17-the-float-half-of-the-machine.toml
# `sqrt` IS NOT HERE, because it is not one instruction: op3978, which Apple's sqrt lowering uses,
# returns the RECIPROCAL square root and the multiply after it is the algorithm rather than a
# rounding step. It is offered under the name its execution earned, rsqrt2, and a real sqrt waits
# on a floating-point multiply this backend does not yet select.
FLOAT_UNARY_OPCODE = {"recip": 3658, "rint": 3770, "floor": 3786, "ceil": 3802, "trunc": 3818,
                      "rsqrt": 3850, "rsqrt2": 3978, "log2": 2570, "exp2": 1272}
FLOAT_UNARY_TEMPLATE = {
    3658: bytes.fromhex("37020c582a20a13a1000"),    # recip,  sw-f_recip
    3770: bytes.fromhex("2f0004182220a0023000"),    # rint,   sw-f_rint
    3786: bytes.fromhex("2f0004182220a0027000"),    # floor,  sw-f_floor
    3802: bytes.fromhex("2f0004182220a002b000"),    # ceil,   sw-f_ceil
    3818: bytes.fromhex("270004382a20a00af000"),    # trunc,  sw-f_trunc
    3850: bytes.fromhex("278004582a20a30a1000"),    # rsqrt,  sw-f_rsqrt
    3978: bytes.fromhex("a78284582a20a2021000"),    # rsqrt2, chain_sqrt
    2570: bytes.fromhex("2f00040a2a20a4023000"),    # log2,   sw-f_log2
    1272: bytes.fromhex("2700040a2220a5023000"),    # exp2,   sw-f_exp2
}

# TWO-SOURCE FLOAT, selected through the generic form: no encoder is written for either, the
# fields come from the recovered map and the source lifetimes from real liveness. Named by what
# Apple selects for `p + q` and `p * q` and confirmed by execution - 3.5 and 1.25 give 4.75 and
# 4.375, neither of which is either operand or any of the other four candidate answers.
# NAMED BY EXECUTION, not by isolation - spike/accel/re/opsem.py fed each one six input tuples and
# compared against a candidate library whose control is the 84 opcodes isolation had already named.
# Every one of these matched a single candidate on all six, carries no immediate operand the
# witness fixes. CORRECTED 2026-09-22: this continued "a register-to-register operation Metal has no
# construct for: Apple's compiler never emits nand, nor, orn or xnor". It emits all four, at 10 bytes,
# for ~(x&y), ~(x|y), x|~y and ~(x^y) on int/uint/long/int2; the construct sweeps never wrote them.
# ledger/g17-execution-names-the-opcode.toml
# THE THREE-SOURCE PAIR. Named by isolation and confirmed by execution through the general sweep:
# op1934 = a*b+c in float (2.5, 1.25, 0.75 -> 3.875, and a NaN input propagates), op10826 = a*b+c
# in integers on nine tuples. Both went through the register-ladder fix first - two of op1934's
# three sources step one register per unit, and before that was measured this backend authored them
# as slots and read registers nobody had written. ledger/g17-the-ladder-is-per-operand.toml
# THE OPCODE APPLE ITSELF SELECTS for fma(a,b,c) is op2190, not op1934 - they are the same
# instruction one bit apart (byte 15 bit 7) and the table names both ffma, but op1934's witness is
# a walked one and op2190's is Apple's own, taken from a kernel written for the purpose
# (spike/accel/re/fmaexec.py compiles `fma(a,b,c)` and reads back what the compiler emitted).
# op1934 executes correctly most of the time and returns zero on about one dispatch in six, which
# is a reason to prefer the encoding that occurs in real code even before that is understood.
# BYTE0 BIT 3 IS THE LOAD-WAIT ON THE GENERIC AUTHORING PATH TOO - measured, then bounded.
#
# The end-to-end float kernel returned zero for every thread: `fmul` straight off a load read its
# operand before the load landed. The auth path never set any hazard at all, so this covers every
# opcode reached through it, not the five ALU families fixed earlier.
#
# MEASURED on silicon for op3290: the same kernel with byte0 bit 3 set on the fmul and nothing else
# changed returns 1.0, 2.25, 4.0, 6.25, 9.0 where it returned zeros. That is one opcode.
#
# BOUNDED for the rest by two things that are not execution. (1) Apple sets the bit in most distinct
# patterns of fmul, fadd and ffma - 169 of 192, 209 of 221, 304 of 325 - and in ALL of the
# threadgroup pair's. (2) Setting it keeps the instruction decoding as itself for everything listed
# below. Where it does NOT: op11372 icmp and op11375 csel change instruction entirely - 11 and 32
# patterns refuse - so byte0[3] is an OPCODE bit for them and they are refused rather than guessed.
#
# op17229's witnesses already carry byte0[3] and its store after a load STILL returned zero, which
# is the standing caveat on this table: the bit is the load-wait for ALU-shaped forms, and that a
# form carries it does not mean the form waits.
AUTH_LOAD_WAIT = frozenset({
    998, 2190, 3290, 10826,          # fadd, ffma, fmul, madd  - Apple sets it, decode-safe
    767, 775, 1000,                  # the float add-immediate family
    1062, 13460, 13521, 13548, 16806,  # fsat, nand, nor, orn, sarv
    13488, 17744,                    # andn, xnor - their witnesses already carry it
})
# byte0[3] is an opcode bit here: setting it makes the bytes a different instruction.
AUTH_NO_WAIT = frozenset({11372, 11375})

# THE FLOAT OPERATIONS WHOSE SOURCES CARRY NEGATE (+2) AND ABSOLUTE VALUE (+4) MODIFIERS, and so
# can absorb an IR fneg/fabs (ledger/g17-float-source-modifiers.toml; Apple: x - y is op998 with 18
# on the second source, fabs(x) + y is op998 with 20 on the first, x * -y is op3290 with 18).
MODIFIER_CONSUMERS = ("fadd", "fmul", "fma")


def _fold_modifiers(args):
    """(sources, [(negate, absolute)]) with each fneg/fabs chain folded into its consumer's source.

    Outermost first: fneg(fabs(x)) is -|x| (abs then negate, the modifier's own order), and an
    fabs outside any fneg makes the inner negations irrelevant."""
    out, mods = [], []
    for a in args:
        neg = absol = False
        while isinstance(a, ir.Value) and a.op is not None and a.op.kind in ("fneg", "fabs"):
            if a.op.kind == "fabs":
                absol = True
            elif not absol:
                neg = not neg
            a = a.op.args[0]
        out.append(a); mods.append((neg, absol))
    return out, mods


MACHINE_OPCODE = {"fma": 2190, "madd": 10826,
                  "fadd": 998, "fmul": 3290,
                  # op13488 IS `~a & b`, MEASURED - and the table called it nandn, which is ~(a & ~b) and a
                  # different function. Fourteen candidate operations were fitted against four input
                  # pairs and exactly one survives: andn2, ~a & b. So the IR op is named for what it
                  # computes. THE OTHER FOUR NAMES IN THIS LINE COME FROM THE SAME UNVERIFIED SOURCE:
                  # their opcodes are delivery-proven, which says operands reach them and NOT that
                  # they compute what they are called. ledger/g17-a-name-is-not-a-measurement.toml
                  "nand": 13460, "andn": 13488, "nor": 13521, "orn": 13548, "xnor": 17744,
                  "sarv": 16806}
# fsat WAS op903 AND op903 IS NOT A ONE-SOURCE INSTRUCTION. It has a GPR32 and a GPR16 source, and
# Apple's own instruction table does not put it in the saturate family - bit 7 of the flags word
# covers 64 opcodes, op903 is not among them and op1062 is. op1062 is also the one this backend
# MEASURED: clamp(-x, 0, 1) with its source modifier clear, so fsat is that opcode with the modifier's
# negate bit set, which returns clamp(x, 0, 1) on 2.5, -1, 0.7, -0.3, 3.9 and 0.5.
# ledger/g17-sched-63-is-saturate.toml
# WITNESSES WITH THEIR DEAD BITS CLEARED, one dispatch each.
#
# These four opcodes have ZERO instances in Apple's corpus, so the invariance census that settles
# every other form has nothing to compare against and every unwritten 1-bit fell through to
# "inherited debt" - 76 bits, more than half the scalar backend's total. Calling them inherited from
# Apple was wrong on its face: there is nothing of Apple's here to inherit. They come from a repair
# walk.
#
# Apple's own DECODER settled most of them for free, with no dispatch: clearing 15 of the 76 makes
# the instruction undecodable and clearing 27 more turns it into a different opcode or length, so
# those 42 are forced by the encoding. The remaining 34 leave a well-formed instruction of the same
# opcode and length, and one batch cleared all 34 and got byte-identical answers back on every case
# of all four - the arithmetic shift, the saturate, ~a & b and x + 1/16 all unchanged.
#
# So the compiler emits zero for them. A bit this project writes as zero carries nothing forward;
# a bit it copies from a walked witness is a dependency on a walk nobody has justified.
# ledger/g17-the-decoder-settles-what-no-corpus-can.toml
CLEARED_WITNESS = {
    998    : bytes.fromhex("0102040a0220a00204081200"),   # 3 dead bits
    1000   : bytes.fromhex("110284fa0220000802060200"),   # 7 dead bits
    1062   : bytes.fromhex("1702001a03200a101010"),   # 9 dead bits
    2190   : bytes.fromhex("3180070a2020a3021801800006000080"),   # 2 dead bits
    3290   : bytes.fromhex("210205100280a0021c0020000010"),   # 2 dead bits
    10826  : bytes.fromhex("27008c7a0100a1120b8151008800"),   # 4 dead bits
    11375  : bytes.fromhex("2200470a0300ac02828000000000"),   # 6 dead bits
    13460  : bytes.fromhex("330006803338a4028480"),   # 2 dead bits
    13488  : bytes.fromhex("13000680103804008480"),   # 7 dead bits
    13521  : bytes.fromhex("230006803170a0028080"),   # 3 dead bits
    13548  : bytes.fromhex("230007803370a0028080"),   # 3 dead bits
    16806  : bytes.fromhex("0702009a0200030088830000"),   # 11 dead bits
    17744  : bytes.fromhex("230207803130a4020480"),   # 1 dead bits
}

# fsat IS NOT op1062. op1062 is a CROSS-LANE operation - within each lane pair (2k, 2k+1) both lanes
# get clamp(x[2k] - x[2k+1], 0, 1) under source modifier 2 - and every earlier measurement of it ran
# ONE lane, where the absent neighbour reads as 0 and the result looks like clamp(x). Executed
# 2026-09-23: 1 thread 0.7 -> 0.7; 2 threads (0.7, 0.3) -> 0.4 on both; 32 lanes of 0.7 -> 0 on all.
# Apple's compiler emits op904/12 (fadd.imm.sat.f32, x + -0.0 saturated) for both saturate(x) and
# clamp(x, 0, 1), so that is what fsat lowers to (ledger/g17-fsat-is-op904-and-op1062-is-cross-lane.toml).
MACHINE_UNARY = {}
FSAT_OPCODE = 904
FSAT_APPLE = bytes.fromhex("2900040a2220a00284164200")      # Apple's saturate(x), registers r0/r0
FSAT_IMM_MINUS_ZERO = 128                                   # g17asm.float_imm_value(128) == -0.0
# per-opcode immediates the selection writes with the opcode: fsat's source negate.
MACHINE_IMMS = {1062: {3: 2}}
# THE ONE-SOURCE FLOAT FORMS: an add whose second operand is an eight-bit float immediate. Within a
# block the offset selects the operand form - 774 is reg+reg, 775 is reg+imm - and the BLOCK selects
# the function, which is how the clamp gets in: op767 is op774's offset in the adjacent block and
# returns clamp(x + imm, 0, 1) where op775 returns x + imm. Both measured on this machine over six
# inputs each, with op774 dispatched alongside as the unclamped control (4.0 and -1.5 come back
# unclamped). op1000 is the same offset in the f32 block, measured on six immediates.
# There is no entry for a saturating f32 add: its block is not identified, and inventing one from
# the pattern is the inference-from-selection this pair of measurements exists to replace.
# ledger/g17-the-one-source-float-form.toml
FADD_IMM_OPCODE = {("f16", False): 775, ("f16", True): 767, ("f32", False): 1000}
FADD_IMM_OPERAND = 4

# THE INDEXED STORE, from Apple's own `f[tg.x] = p * q;` - eight bytes, the value in operand 0 and
# the index in operand 5, and the buffer carried by operand 3's expression. This template writes
# buffer 2, the same buffer the slot store's template writes, so a program can mix the two.
# THE COMPARISON THAT YIELDS A VALUE, and the branch-free select. Both are the generic form with
# the relation in the condition-code operand, and both condition codes were read off EXECUTION: the
# same instruction on (10,20), (20,10) and (10,10) leaves exactly one relation consistent with all
# three answers. Only the codes the field map's bits can express are offered - op11372's operand 2
# has value bits 0 and 2, op11375's only bit 0 - so this is two relations each and not a guess at
# the rest of the code space. ledger/g17-a-comparison-that-is-a-value.toml
# THE THREE CODES op11372 CAN WRITE, each measured on (10,20), (20,10), (10,10) and again on
# (-5, 3) - which is what separates the signed comparison from the unsigned one, since they agree on
# every non-negative pair. Code 4 behaves as code 0 because equality does not depend on signedness.
#
#     cc 0 equal      cc 1 unsigned less-than      cc 5 signed less-than
#
# The other seven relations are DERIVED rather than guessed: greater-than is less-than with the
# operands swapped, and the four "or equal" forms and not-equal are the complement of one of those.
# The complement costs one xor with 1, which is exact because the comparison yields 0 or 1.
ICMP_OPCODE, ICMP_CC = 11372, {"eq": 0, "ult": 1, "slt": 5}
# `lt` was this backend's name for the relation before the signedness bit was measured, and what it
# named is the UNSIGNED one. Kept as an alias rather than renamed away, because a silent change of
# meaning is worse than a spelling.
ICMP_ALIAS = {"lt": "ult", "gt": "ugt", "le": "ule", "ge": "uge"}
ICMP_SWAP = {"ugt": "ult", "sgt": "slt"}
ICMP_NEG = {"ne": "eq", "uge": "ult", "sge": "slt", "ule": "ugt", "sle": "sgt"}
# op11375's CODE 1 IS A BIT TEST, NOT EQUALITY: (a & b) != 0 ? x : y. Measured 2026-09-23 on 32
# lanes, x = 0..31: against 16 it selected exactly x >= 16 in BOTH operand orders, against 0 never,
# and against 5 exactly the lanes with x & 5 != 0 - which equality (only 5) and >= (5..31) do not
# give. The earlier reading "1 selects x when a = b" came from cases where the equal pair was
# nonzero, and equal nonzero values always share a bit: a degenerate test. It shipped as "eq".
# ledger/g17-csel-code-one-is-a-bit-test.toml
CSEL_OPCODE, CSEL_CC = 11375, {"gt": 0, "test": 1}
# THE FLOAT SELECT, op9700, is what Apple's compiler emits for max(float, float) and
# min(float, float): sw-f_max and sw-f_min are `load a ; load b ; op9700 d, a, b, a, b ; store`
# and differ in exactly one operand, slot 2, at 7 for max and 3 for min (byte2[6] and byte11[2]).
# The value is taken from those witnesses, not decoded as a relation: under the float
# condition-code decode 7 reads "equal or unordered" and 3 "not equal", which is not a reading of
# a max, so slot 2 is the select's own operation field here and its meaning is Apple's selection
# of it for max()/min(). Slot 1 is 0x80000000 on every max/min instance and is stated rather than
# inherited. ledger/g17-float-source-modifiers.toml has op9700 as fmin with a negated source.
# THE OPERATION IS AUTHORED FROM APPLE'S OWN INSTRUCTION, NOT FROM THE FIELD MAP. The map knows two
# bits of slot 2 and Apple's decoder reads 7 and 3 there, so writing the value through the map is
# refused ("value 7 does not fit its 2 mapped bits") and writing part of it would be the op11666
# mistake (ledger: write the whole operand). Each template is the op9700 from the single-
# instruction corpus kernel for that operation; only the registers and the source lifetimes are
# filled, and selfcheck reads the result back through Apple's decoder.
# op10283's RETAINED FORM, for the zero-first widening. a + (b & 0xFFFF) with a 32-bit first source
# and a SIXTEEN-BIT view of the second, so a = 0 gives zext16(b).
#
# MEASURED WITH a = 0: root's results/g17-integer16-zero-first-v1 dispatched both configurations,
# six queries, 2208 output words, and returned b & 0xFFFF for b across
# [0, 1, 65535, 65536, 32768, 2147483648, 4294967295] - the endpoints, including both halves of the
# sign boundary and the all-ones word. Every one of its seven sites used THESE bytes and THESE
# immediates with only the materialised constants differing. So this is the witness's template,
# not a new encoder.
#
# AND THE HALF-LOAD READINESS IS NOW MEASURED TOO, which is what that evidence's scope excluded.
# root's results/g17-integer16-half-load-v1 (frozen 2088caa7) dispatched the pair this route needs:
# two 440-byte programs whose only difference is seven single bits - byte0[3] of each op10283, at
# offsets 30/88/146/204/262/320/378 - over a real 368-element ushort source, seven op12646 half
# loads at byte offsets 0..12 feeding op10283 as 0 + low16.
#
#     wait bit SET     [0, 1, 32767, 32768, 65535, 4660, 43981] on all five queries, two workers,
#                      and the whole 184-word payload matches root's digest in every one of them
#     wait bit CLEAR   wrong AND unstable: [0, 0, 1, 32767, 65535, 0, 43981] in two queries and
#                      [0, 0, 1, 32767, 32768, 65535, 43981] in the other three
#
# Verified here from the retained bytes and .npz payloads rather than from root's summary: the byte
# diff, both program digests, the per-query values and both review digests. So the load-wait bit is
# what makes a half load ready for op10283 - measured, with its negative control falsified - and
# this route sets it when its operand comes from a load. The receipts are FAILED overall because
# the campaign's repeat predicate stopped on the CONTROL's instability; that is the control doing
# its job, not two passed runs, and nothing here claims opposite-half preservation, arbitrary
# scheduling or the whole uint domain.
WIDEN_U16_OPCODE = 10283
WIDEN_U16_TEMPLATE = bytes.fromhex("3702045a2900a312a8032100")
WIDEN_U16_IMMS = {1: 32, 3: 16, 5: 16}          # the tuple [reg, imm32, reg, imm16, reg, imm16]
# The wait is byte0[3], applied after the operand writes; the decoder then prints the first
# modifier as 2147483680 = 32 | 1 << 31, which is byte-for-byte root's executed candidate.
WIDEN_U16_WAIT_MODIFIER = 2147483680

FSELECT_OPCODE = 9700
FSELECT_TEMPLATE = {"fmax": bytes.fromhex("2200070b2300a0029400a1640040"),   # sw-f_max +0x66
                    "fmin": bytes.fromhex("2200470b2300a0029400a1600040")}   # sw-f_min +0x66

STOREI_OPCODE = 17229
# THE CAPABILITY-OFF ARM FOR THE WIDE BUFFER RANK. True restores the state before the rank
# measurement: a const above rank 7 refuses, which is the measurement the gain is scored
# against. Ranks 8..15 are measured on op17229/8 only - see the gate in the auth path and
# results/g17-buffer-rank-assessment-v1/.
_NO_WIDE_BUFFER_RANK = False
# op11765's OPERATION, which lives in operand 2 rather than in the opcode. Apple's own compilations
# of all seven at length 12; the encoding is a four-bit table key - byte4[5], byte5[3], byte6[3],
# byte11[4] - not a linear field, and g17as places it from the map.
TG_ATOMIC_OP2 = {"add": 262656, "sub": 262657, "min": 262660, "max": 262662,
                 "and": 262664, "or": 262665, "xor": 262666}
STOREI_TEMPLATE = bytes.fromhex("0f04030201061040")

# IR op -> (immediate-operand opcode, register-operand opcode). Both 14 bytes.
SHIFT_OPCODE = {"shl": (14391, 14392), "shr": (17013, 17014)}
# Saturating add and subtract: register-register only, because that is the form the corpus shows
# and a register-immediate variant would be a guess.
SAT_OPCODE = {"addsat": 10239, "subsat": 11624, "sar": 16805}
SAT_TEMPLATE = {
    10239: bytes.fromhex("2f00041a2100a302b8802000"),   # addsat, modal of 3
    11624: bytes.fromhex("2700041a2d00a20298820000"),   # subsat, modal of 110
    16805: bytes.fromhex("2700001a2e00a30278804000"),   # sar, modal of 2
}
# and/or/xor are alu.bitwise.imm, a DIFFERENT form. Its size (8 bytes) and its split immediate are
# recovered and its operation selector was authored in both directions - but its operands are not
# in the instruction at all: six source expressions and two destinations leave the eight bytes
# unchanged, and the 2-byte selector before it, which does vary, is unread. Without that no
# register can be allocated, so the op is refused with the reason rather than lowered onto an
# alu.12 opcode that means something else. ledger/g17-bitwise-size-resolved.toml
BITWISE_OPS = {"and", "or", "xor"}
# Opcode per operation, and the template is the instruction Apple emitted for a source expression
# written here: `x & 15u`, `x | 60u`, `x ^ 60u`. TEN bytes, class b.
BITWISE_OPCODE = {"and": 423, "or": 13574, "xor": 17770}
# The register-register forms, four bytes. Templates are corpus instances whose operand ROLES come
# from tools/g17fields.py over 1070/245/117 instances with zero unexplained bits - so unlike the
# immediate forms, no kernel written here produced them and their semantics rest on correlation
# until executed. spike/accel/re/bwregexec.py is that execution.
BITWISE_REG_OPCODE = {"and": 424, "or": 13575, "xor": 17771}
# One REAL corpus instance per opcode. The first version of this table gave `and` the xor
# template, and since the operation selector was inherited the instruction performed an xor - see
# BW_R_OP. The selector is authored now, so the template supplies only the modifier bits, but a
# per-opcode instance is still the honest starting point.
# ENCODING DATA, HELD IN THE LIBRARY. Re-exported here so every existing reader keeps working;
# g17const.inventory() used to reach into this module for it, which was its only edge back to the
# compiler and the reason its production half could not enter the package.
from agxforge.g17.formenc import BITWISE_REG_TEMPLATE   # noqa: E402
BITWISE_TEMPLATE = {
    423:   bytes.fromhex("2b00078030 30a302c483"),
    13574: bytes.fromhex("2b00078032 38a302048f".replace(" ", "")),
    17770: bytes.fromhex("2b00068032 38a302048f".replace(" ", "")),
}
# read.sr byte1. isa/g17-scalar-isa.toml read_sr.direct, measured.
# 0xa4 is thread_position_in_THREADGROUP, read off a6-tgidx, whose Metal source declares exactly
# that and whose own read_sr carries 0xa4. It matters more than a third name: a kernel appears to
# get only the special registers its metadata declares, so a program patched into that host and
# reading 0xa0 - thread_position_in_grid, which that kernel does not declare - got ZERO in every
# lane, and a 32-lane threadgroup exchange read its own slot back.
# THREADGROUP MEMORY, from a6-tgidx - one kernel, so the store and the load address the same place
# by construction. The base is the template's const(4) and is NOT authored; what this backend writes
# is the value, the index and the destination. isa/g17-threadgroup-pair.txt
TG_STORE_OPCODE, TG_LOAD_OPCODE = 13288, 12364

# THE FOUR-BYTE REGISTER MOVE, op586. It is the single biggest thing this backend could not emit:
# 2,156 of Apple's 6,594 programs contain one and 4,707 instructions are it (tools/g17frontier.py).
# Until now every copy this compiler needed was an `add reg, #0` at twelve bytes, which is three
# times the size and a different instruction from the one Apple writes.
#
# ITS SOURCE LIFETIME IS OPERAND 3, NOT OPERAND 1, and that distinction is the whole risk. Four
# inherited-lifetime defects in this ISA presented as "an authored program silently reads zero",
# and a COPY that releases the value it copies is exactly that bug: the copy exists because the
# original is read again. Apple's own allocator settles which operand it is, over 290 op586/4
# instances sampled with a stride across the corpus, asking whether the SOURCE register appears as
# a source again before anything overwrites it:
#
#     op3 == 32 iff the source is read again    270 of 290   (93.1%)
#     op1 == 32 iff the source is read again    204 of 290   (70.3%)
#
# and op3 takes exactly {32, 16, 0} - the same 32-keep/16-release pair the twelve-byte ALU carries
# (ledger/g17-operand-b-lifetime.toml). Every one of the 51 instances with op3=32 has its source
# read again; 67 of the 70 with op3=16 do not. op3=0 is a third state this backend does not write.
MOV_KEEP, MOV_RELEASE = 32, 16

# THE OLD HALF-LOAD ANCESTRY REFUSAL, KEPT AS A CAPABILITY-OFF ARM.
#
# True restores the refusal exactly as it stood: any value whose SSA ancestry contains a sixteen-bit
# device load is refused as the source of an indexed op17193 half store. It is False because the
# evidence that refusal was written from is disproved - see the block at the store's lowering - and
# because what actually failed on hardware is now checked on the delivered bytes by
# asm.read_after_release. The arm stays so the historical behaviour is one assignment away and the
# test that measured it still measures something.
_HALF_LOAD_ANCESTRY_REFUSAL = False

# op10090's OPERAND 1 IS PER LENGTH, and pinning one value for both is how the twelve-byte atomic
# came to rest on a solver's luck. At ten bytes the map is a ramp based at 1048576 and writing that
# is the base; at twelve the map is VACUOUS - verdict `verified` with no positions, no base and no
# step - so the request fell through to the quadratic solver, which returned a model reading back
# as 2148532224 rather than the 1048576 it was asked for. That was refused, correctly, and the only
# reason it had ever worked is that a solve from an older run sat in the on-disk cache; the cache's
# stamp includes g17as.py's mtime, so editing that file for anything at all discarded it.
#
# Apple settles the ordinary form: over the whole corpus its twelve-byte op10090 instances carry operand 1 =
# 2148532224, 7 of 7, and that is exactly the value the vacuous form already reads back. So the
# twelve-byte form now WRITES what Apple writes, the solver is never reached, and nothing depends
# on a cache. Proven of an encoding, not an opcode - the same rule that made (opcode, length) the
# unit of specification in the first place.
ATOMIC_IDX_OP1 = {10: 1048576, 12: 2148532224} # consumed ADD/AND/OR/SUB/XOR use slot 7 instead of the 12-byte slot-0 value


TG_STORE_TEMPLATE = bytes.fromhex("0f040302030611c00020")
# THREE BITS CLEARED, 2026-09-07, each dispatched. Six of this template's bits are covered by no
# operand in op12364's field map. Clearing them one at a time in the round trip splits them:
#   b4[3] b4[4] b6[4]  the round trip returns 0 - REQUIRED, and now recorded as required
#   b5[5] b8[6] b10[7] the constant still comes back - inert, and cleared here rather than carried
# That also refutes the reading on file that all six were operand 4, source 3's lifetime: a
# lifetime would not be required, and these six do not behave alike.
TG_LOAD_TEMPLATE = bytes.fromhex("0f0403021a0610c0010000000000")
# THE FIELD IS A HARDWARE NUMBER, and byte1 bits 0-6 hold it in the four-byte form. The peer
# recovered the numbering by requiring every bit to be constant across all instances printing the
# same register and to differ across registers, then checking the result is a bijection
# (isa/g17-special-register-numbers.txt). It agrees with this backend where the two overlap:
# 0xa4 & 0x7f is 36, which is SR_LOCAL_X - the register whose per-lane behaviour this project
# measured. Bit 7 is set in every witness on both sides and is NOT part of the number.
#
# Only the registers on that list may be emitted. The mapping is observed, not derived, and nothing
# in it predicts the number of a register nobody has read.
SR_NUMBER = {"SR_SIMD_ELEM": 2, "SR_SIMD_GRP": 5, "SR_PVQUAD": 20, "SR_PVSIMD": 21,
             "SR_TVQUAD": 22, "SR_TVSIMD": 23, "SR_TG_X_SIZE": 24, "SR_TG_Y_SIZE": 25,
             "SR_TG_Z_SIZE": 26, "SR_LOCAL_X": 36, "SR_LOCAL_Y": 37, "SR_LIN_ID": 39,
             "SR_TG_DISP_X_SIZE": 40, "SR_TG_DISP_Z_SIZE": 42, "SR_LIBXDIM": 44,
             "SR_LIBYDIM": 45, "SR_VRID_SET": 106}
SR = {"threadgroup_position_in_grid": 0x9c, "thread_position_in_grid": 0xa0,
      "thread_position_in_threadgroup": 0xa4}
SR.update({k: 0x80 | v for k, v in SR_NUMBER.items()})
# The source builtin and the recovered system-register ledger use different names for the same
# measured lane index.  Keep the public builtin spelling available to the IR instead of forcing a
# quantization primitive to manufacture an authored read_sr row.
SR["thread_index_in_simdgroup"] = SR["SR_SIMD_ELEM"]
# the simdgroup's index in its threadgroup: the register tlower's row split reads (SR_SIMD_GRP, SR133). tlower
# masks it (& 3 up to four simdgroups), and a scalar user should too (MM 25.144.8)
SR["simdgroup_index_in_threadgroup"] = SR["SR_SIMD_GRP"]
SR_AXIS = {"x": 0, "y": 1, "z": 2}

# --- WHAT THE OBJECT NEEDS FROM THE COMPILER --------------------------------------------------
#
# g17authorobj.py authors all five metadata sections and REFUSES rather than defaulting on the
# inputs it cannot derive from bytes. Two of them are this side's, and both are answered here.
#
# __GPU_ARCH_LD_MD IS ONE BOOLEAN AND IT MEANS "EVERY THREAD IN THIS KERNEL IS INDEPENDENT". Set
# in 6,234 of 21,001 corpus objects. It was called underivable because every fact tried against it
# was an OPCODE fact, and read_sr is in almost every program - the discriminator is the SR INDEX,
# an operand value. Forty-seven one-variable probes settle it (tools/g17corpus.py builds them as
# uf-*, uf2-*, uf3-* and uf4-*; each holds the whole kernel fixed and moves one thing):
#
#     SET - the thread stands alone            clear - the thread is coupled to its threadgroup
#     reads no position at all                 threadgroup_position_in_grid  .x .y .z
#     thread_position_in_grid  .x .y .z        thread_position_in_threadgroup
#     threads_per_grid                         thread_index_in_threadgroup
#     a texture read                           threads_per_threadgroup, threadgroups_per_grid
#     a device atomic                          thread_index_in_simdgroup
#     threadgroup MEMORY with no barrier       simdgroup_index_in_threadgroup
#     simdgroup_multiply                       any barrier, either scope
#                                              simd_sum, simd_prefix, simd_broadcast_first,
#                                              simd_is_first, quad_sum
#                                              a THREADGROUP atomic
#
# Two surprises are load-bearing. Threadgroup memory does NOT clear it - `threadgroup uint g[32]`
# read and written leaves the flag SET, and the round-three kernel that appeared to clear it was
# cleared by its barrier. And it is decided after dead-code elimination: `uint i = tp.x;` with `i`
# unused leaves the flag SET, which is why a source-text reading of the corpus has 490 exceptions
# where a reading of the emitted program has far fewer.
#
# Most of the clearing constructs emit no read_sr at all, which is why no fact derivable from
# __text alone can decide them, and why the input belongs to this side.
# ledger/g17-the-arch-flag-is-one-thread-standing-alone.toml
ARCH_CLEARING_SR = frozenset(
    [SR["threadgroup_position_in_grid"] + a for a in SR_AXIS.values()] +
    [SR["thread_position_in_threadgroup"] + a for a in SR_AXIS.values()] +
    [SR["SR_PVSIMD"], SR["SR_TVSIMD"], SR["SR_SIMD_ELEM"], SR["SR_SIMD_GRP"]])

# The forms that couple a thread to its threadgroup without reading one of those registers.
ARCH_CLEARING_FORMS = frozenset(["barrier", "simd.broadcast.10", "atomic.tg.uniform.12"])
# Apple's atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device): op14156 (0, 186)
FENCE_DEVICE_BYTES = bytes.fromhex("0f2b00060000")

# THE PER-KERNEL BOOLEANS. Every one of these slots carries the value 1 in every corpus object
# that has it - 20,800 of 20,800 for slot 15, 2,020 of 2,020 for slot 18 - so PRESENCE is the whole
# content and the linker needs a boolean, not a value.
#
# AND THERE ARE THREE OF THEM, NOT ONE. Over 21,131 corpus sections the joint distribution has
# exactly four cells and no exceptions:
#
#     15  16  17
#      -   -   -    286   the kernel writes nothing
#      Y   -   Y     70   it writes a TEXTURE and no buffer
#      Y   Y   -  20,757  it writes a BUFFER
#      Y   Y   Y     18   both
#
# so slot 15 is `16 or 17` exactly, slot 16 is a written buffer and slot 17 a written texture. The
# peer built six wrong sections by taking has_stores and emitting slot 15 alone; the population is
# what says slot 16 travels with it.
#
# A THREADGROUP STORE IS NOT A BUFFER WRITE and neither is an imageblock write: of 93 corpus
# kernels whose source mentions imageblock, 53 carry none of the three, and the 40 that do also
# write a device buffer. So the threadgroup and imageblock forms are deliberately absent from the
# set below - including them would set slot 16 on a kernel Apple leaves it clear on.
# THE TENSOR STORE OPCODES. A tensor row is classified by OPCODE and not by form name or phase
# string: the registry path's readout row happens to carry phase="readout", but the row-spliced
# general lowering carries the same stores with a different phase, and a predicate keyed on the
# phase silently reported has_stores=False for a stream containing six op17258 range stores. The
# consequence was not a wrong count but an unauthorable image: ProgramABI.contract() refuses the
# binding/write inconsistency, so 17x19x16, 17x19x19 and 50x37x80 could not be authored at all.
# op17258 is the masked form and is NOT in TENSOR_OPCODES, which is why that set could not be
# reused here.
# op17202 is the one-word store of tlower's fp8 quantize-out (Set A item 9b): the same buffer write,
# four bytes per lane per row segment. No other tensor body emits it.
TENSOR_STORE_OPCODES = frozenset({17257, 17258, 17202})
# BYTES PER ELEMENT, and the default row strides are now derived from it rather than assuming
# halves. The defaults were `K * 2` and `N * 2` for A and B regardless of the declared operand
# types - right for half and bfloat, wrong for float (four-byte) and int8/uint8 (one-byte). A
# 32x32x64 int8 matmul with no explicit strides compiled to a body 28 bytes different from the
# lowerer's and disagreed with the reference on 1024 of 1024 output elements over three dispatch
# trials; the same shape with strideA=64, strideB=32 emits the lowerer's body byte for byte.
# C has no dtype - it is always float - so strideC stays N * 4. An unrecognised dtype raises
# KeyError here, which is the disposition this IR already had for one: an invalid fixture.
TENSOR_ELEMENT_BYTES = {'half': 2, 'bfloat': 2, 'float': 4, 'int8': 1, 'uint8': 1,
                        'fp8e4m3': 1, 'fp8e5m2': 1}      # fp8: one byte, unpacked to bf16 by tlower

BUFFER_WRITING_FORMS = frozenset(["store.8", "store.14", "store.half.14", "store.halfvec.14",
                                  "store.byte.14",
                                  "store.half1.8", "store.half1.10", "store.half1.14",
                                  "store.elem1.8", "store.elem1.10", "store.elem1.14", "store.vec4.8", "atomic.uniform.10",
                                  "atomic.add.10", "atomic.add.12"])
WRITING_FORMS = frozenset(["store.8", "store.14", "store.half.14", "store.halfvec.14",
                           "store.byte.14",
                           "store.half1.8", "store.half1.10", "store.half1.14",
                           "store.elem1.8", "store.elem1.10", "store.elem1.14", "store.vec4.8", "store.ib.32",
                           "atomic.uniform.10", "atomic.tg.uniform.12",
                           "atomic.add.10", "atomic.add.12"])
THREADGROUP_FORMS = frozenset(["atomic.tg.uniform.12"])
THREADGROUP_AUTH = frozenset([TG_STORE_OPCODE, TG_LOAD_OPCODE])
# alu.shiftadd.imm scale table: a lookup, not an arithmetic shift field.
# CORRECTED 2026-09-04. Derived from the ISA's own byte pairs rather than transcribed:
#   scale_code = 8*byte10[2] + 4*byte10[1] + 2*byte10[0] + byte11[0]
#   x1 (0x11,0x00) -> 2    x2 (0x14,0x00) -> 8    x4 (0x15,0x00) -> 10
#   x8 (0x10,0x01) -> 1    x16 (0x10,0x00) -> 0
# The previous table was shifted by one entry - it mapped x1 to 8, which is x2 - so every plain
# add that wrote the scale silently DOUBLED its operand, and every add that did not write it
# inherited whatever the template carried. Caught by execution: load B[2] then add 0x11 returned
# B[2]*2 + 0x11. ledger/g17-inherited-scale-multiplied-the-operand.toml
SCALE_CODE = {1: 2, 2: 8, 4: 10, 8: 1, 16: 0}

END = bytes.fromhex("0e000000")          # isa/g17-scalar-isa.toml "end", causal
# THE BARRIER IS NOT A CONSTANT - byte1 IS ITS SCOPE. This comment used to say the whole
# instruction was the field, on the evidence that all 225 instances in Apple's driver shaders are
# identical. They are, and the inference was still wrong: those shaders all use the THREADGROUP
# scope, so the constancy was a property of the sample. A kernel asking for a device barrier emits
# byte1 = 0x69, and one asking for both emits exactly one of each (g17asm.BARRIER_SCOPE).
#
# The old claim - "the whole instruction is the field, so emitting it is authoring it" - is the
# same error as giving opcode 424 the xor template: a selector that never varies in the sample
# reads as safe when it means unexamined.
# ORIGINAL NOTE, kept because the observation was right and only the conclusion was not:
# so unlike every other form it needs no template and has no unresolved bits: the whole
# instruction is the field. Emitting it is authoring it.
BARRIER = bytes.fromhex("275100060000")
# Control-flow templates, taken from the recovered instances in isa/g17-scalar-isa.toml. These
# are LITERAL forms: their examples are real instructions, and the fields we author into them are
# only the ones proven authorable - the branch displacement (authored and executed already) and
# the compare's immediate. Everything else in them is inherited and declared as such.
# TEN BYTES, not four. Apple's decoder gives length 10 for all 106 branch sites in a 242-object
# sample, and an executed A/B settles it: the same program with a 4-byte branch faults with a GPU
# address fault when the branch is taken, and completes with the full instruction. Every branch
# this compiler authored before today was truncated, which is the whole of
# ledger/g17-taken-branch-faults.toml. ledger/g17-branch-is-ten-bytes.toml
# THE TAIL DIFFERS BY DIRECTION, exceptionless over the corpus: 31/31 forward branches (opcode
# 462) carry six zero bytes, and 31/31 backward ones (opcode 458) carry 1f 00 ff c7 ff 7f. The
# compiler emitted zeros for both, so every back edge it could have produced was not merely
# truncated but wrong in its second half. ledger/g17-back-branch-tail.toml
BRANCH_TAIL_FWD  = bytes(6)
BRANCH_TAIL_BACK = bytes.fromhex("1f00ffc7ff7f")
BRANCH_FWD  = bytes.fromhex("3e005b0e")          # branch.cond.fwd
BRANCH_BACK = bytes.fromhex("de6ff31e")          # branch.cond.back - isa/g17-scalar-isa.toml,
# identified causally in the Collatz kernel: NOPping exactly these four bytes made the loop fall
# through after one pass and out[100] became 1, the value preregistered before the run.
#
# SAFETY NOTE, recorded because it changes a property the project has relied on. Every branch the
# compiler emitted before this was FORWARD, and a forward branch only moves the PC forward, so no
# authored displacement could produce a non-terminating loop. That argument is what made
# branch.cond.fwd safe to author and safe to dispatch. It DOES NOT hold for a back edge: a
# generated loop whose condition is wrong runs forever, which is exactly the failure that wedged
# the GPU during a NOP sweep earlier today (ledger/g17-hang-poisons-the-run.toml).
EXEC_JOIN   = bytes.fromhex("3e03400e")          # exec.restore - the reconvergence point
# THE PREDICATE SEQUENCE. Recovered by differential compilation and validated by execution -
# 8 variants x 16 threadgroup widths, 128/128, each predicting the exact width at which the
# branch starts being taken. ledger/g17-compare-immediate-executed.toml
#
# Two of the four pieces are AUTHORED and two are INHERITED, and the split is the honest one:
#
#   cmp.src       0a 03          AUTHORED: the compared REGISTER, (reg << 1) | 1
#   cmp.imm       2a 84 04 22    AUTHORED: the immediate and the relation. byte3 is 0x22 when the
#                                operand came from read_sr and 0x02 when it came from an ALU
#                                result; it is NOT the register (constant across five of them) and
#                                is not decoded, so it is inherited.
#   exec.mask     1e 00 00 0e    writes the EXEC MASK from the predicate - measured, not assumed
#
# THE OPERAND HAS A LOAD-USE HAZARD. Pointing cmp.src at a register whose load had not landed made
# the compare read the register's PREVIOUS contents - it saw a stale read_sr value and behaved as
# though the operand were永 zero, for every data value tried. The same hazard the 14-byte store
# needs its wait bit for, on the compare's operand. Until the compare's wait is located, comparing
# a load result is refused rather than emitted.
# byte5[5] CLEARED, 2026-09-07. It was the template's and it was the WRONG value for anything this
# compiler emits. Across Apple's 854 six-byte compares the bit separates on where the compared value
# CAME FROM: all 33 instances that set it name a source register no instruction in that program ever
# writes - a value delivered from outside - while of the 821 that clear it, 279 name a register an
# instruction in the same program produced. (The other 542 have no producer this reader can find,
# which is a limit of decoding destinations, not evidence; the asymmetry is that 33 of 33 land on
# one side of it and 0 of 33 on the other.) Every compare this backend emits reads a value its own
# program computed - a read_sr or an ALU result - so the value it wants is 0.
#
# Executed before it was changed: flipping this bit in the branching kernel moves nothing, all 32
# lanes either way. So this is not a bug fix; it is an inherited bit becoming a chosen one.
CMP_IMM     = bytes.fromhex("2a840402")
EXEC_MASK   = bytes.fromhex("1e00000e")     # opcode 582, precedes a FORWARD branch
LOOP_FLAG   = bytes.fromhex("fe00020e")     # opcode 579, precedes a BACK EDGE
CMP_RELATIONS = ("gt", "lt")      # the two the encoding is recovered for

def _cmp_bound_or_refuse(cond):
    """The compare's bound as an integer, or a NAMED refusal saying which form is missing.

    THE HOLE THIS CLOSES. Both compare emitters read `cond.op.args[1].v` directly, which assumes an
    Imm. A loop whose trip count is a RUNTIME value - `for (i = 0; i < n; ++i)` with n loaded -
    reaches that line with an ir.Value and raises AttributeError, or reaches _canon_cmp and raises
    TypeError comparing an int to a Value. Either way the compiler crashes where it should refuse,
    and a crash does not tell a caller what is missing.

    WHAT IS MISSING IS A FORM, AND IT IS NAMED. This backend's loop compare is op10369/6 - `cmp.src`
    plus CMP_IMM - and `_canon_cmp` holds its immediate to eight bits, which is the recorded
    trip-count ceiling. Apple compiles a register-bounded loop to op10369/**10**: every probe
    variant in results/g17-cmp10-mode-source-v1 used a register bound and produced /10, never /6.
    So /10 is the form a runtime trip count needs, and this backend does not have it.

    A RUNTIME BOUND NO LONGER REACHES HERE. _lower_runtime_bounds rewrites `i < n` before
    selection into icmp.ult values and `cmp.6 p > 0`, which are measured, so the refusal below
    now fires only for a compare that pass did not rewrite. What stays true is that op10369/10
    itself is not emitted - the runtime bound goes around it, not through it.

    WHY IT IS NOT EMITTED HERE. Only ONE of /10's operands is measured - operand 1, the mode, which
    marks loop nesting depth zero (2^31) against nested (0), established from a source contrast with
    a held-out discriminator. Which bits carry the compared registers, the relation, or the length
    itself are NOT recovered. Emitting would be choosing fields from a decoder rather than from a
    measurement, which is the one thing integration's dispatch forbids.
    """
    bound = _imm_of(cond.op.args[1]) if cond.op.args and len(cond.op.args) > 1 else None
    if bound is None:
        # THE MESSAGE STARTS ON THIS LINE, AND NAMES THE FORM IN ITS FIRST FRAGMENT, DELIBERATELY.
        # g17capcompiler's refusal inventory reads the message off the raising line itself, so a
        # refusal whose text begins on the NEXT line is catalogued with an empty message: it
        # records that a refusal exists and not what it says. Line 46 has had that shape for
        # longer than this one, so the extractor's gap outlives this workaround. The first draft
        # of this comment also quoted the raising construct verbatim, which the same extractor
        # picked up as a PHANTOM refusal at the comment's own line - a scanner that reads source
        # text will read what you write ABOUT it too.
        raise Unsupported("op10369/10 is needed for a loop or branch bound that is not a "
            "compile-time constant, and this backend does not emit it, "
            "this backend does not emit. Its loop compare is op10369/6 with an eight-bit immediate, "
            "so a RUNTIME trip count has no form here. op10369/10's mode operand is measured (loop "
            "nesting depth) and the rest of its field map is not, so emitting it would be guessing "
            "fields rather than placing measured ones")
    return bound


def _canon_cmp(rel, imm):
    """Normalise a relation to one the encoding covers, exactly as Apple's compiler does.

    It never emits >= or <=: `x >= K` comes out as `x > K-1` and `x <= K` as `x < K+1`, which is
    why the relation field only ever holds two values. Doing the same here is not a workaround -
    it is the compiler behaviour the corpus documents.
    """
    if rel == "ge": rel, imm = "gt", imm - 1
    elif rel == "le": rel, imm = "lt", imm + 1
    if rel not in CMP_RELATIONS:
        raise Unsupported("cmp relation %r is not recovered; only %s are, and >=/<= reduce to "
                          "them" % (rel, "/".join(CMP_RELATIONS)))
    if not 0 <= imm <= 0xFF:
        raise Unsupported("cmp immediate %d is outside the recovered 8-bit field" % imm)
    return rel, imm

# CMP.SRC PERFORMS A DESTRUCTIVE READ. Measured: a nested conditional whose two compares name the
# SAME register in byte-identical `0a 11` instructions has the outer one read tp.x correctly and
# the inner one read ZERO. Every observation fits it - `0 > 4` false, `0 > 8` false, `0 < 100`
# true - and the discriminating test settles it: with the inner test set to `t < 1`, "reads zero"
# predicts the region runs whenever the outer does and "reads tp.x" predicts it never runs, since
# lane 0 is outside the outer mask. It ran, 11/11. Giving the second compare its own read_sr makes
# the whole nest gate correctly, 11/11.
#
# So a value consumed by a compare is DEAD, and the compiler must re-materialise it. This is the
# "causal lifetime / destructive-read constraint" the register allocator is required to encode.
# ledger/g17-compare-destructive-read.toml
_CONSUMED = set()

def _rematerialise(out, v):
    """Re-emit the producer of a value a previous compare consumed, or refuse."""
    if v.op is None or v.op.kind != "builtin":
        raise Unsupported("value %r was consumed by an earlier compare - cmp.src reads "
                          "destructively - and only builtins can be re-materialised, so this "
                          "one cannot be compared twice" % v)
    which = v.op.attrs["which"]; axis = v.op.attrs.get("axis", "x")
    out.append(MInst("read_sr.4", 4, dict(sr=SR[which] + SR_AXIS[axis], seq=0), defs=[v],
                     note="re-materialised: an earlier cmp.src consumed this value"))

def _materialise_sr(out, val):
    """A store's SOURCE cannot be a special-register read directly. Copy it through an ALU first.

    MEASURED, with the two operands separated, because `C[t] = t` puts the SAME register in both:

        C[t] = t        value and address both straight off read_sr   every lane stored 0
        C[t] = t + 0    one alu.12 between the read and the store     correct, and == Apple
        C[t] = 7        ADDRESS straight off read_sr, value a const   correct, and == Apple

    CORRECTED 2026-09-24 (MM 25.117): not a late value. In C[t] = t the value and the index are ONE
    register, and the store's index slot released it before the value was read; distance and a wait
    change nothing, a different index register stores t at distance 0, and O[u] = u with no special
    register stores 0 the same way. The copy works because it separates the registers; the lifetime
    pass now also keeps a store's doubly-named value (_STORE_OPCODES_DUP_KEEP). The copy stays: it
    is measured correct and moves no retained program.

    So it is the value operand, not the address, and one intervening ALU is enough - the same
    shape as the load-use hazard two branches down, and the same fix twenty-four end-to-end
    kernels already apply by hand. None of them found it, because none of them stores a
    special register UNMODIFIED: every one puts it through arithmetic on the way.

    The copy KEEPS its source when anything reads the value again - `C[t] = t` reads it as the
    address in the very next instruction - which is the same test the loop latch's compare uses.
    """
    if not (isinstance(val, ir.Value) and val.op is not None and val.op.kind == "builtin"):
        return val
    copy = ir.Value(getattr(val, "type", ir.I32), "%s_sr" % (getattr(val, "name", "v") or "v"))
    out.append(MInst("alu.12", 12,
                     dict(op=3, mode=1, imm=0, src1_w=1, dest_w=1, srcb_w=1,
                          scale_code=2, load_wait=0, b4_5=None, b0_5=None,
                          keep=1 if val in _MULTI_USE else 0),
                     defs=[copy], uses=[val],
                     note="copy: a store's source cannot come straight from read_sr"))
    return copy


def _atomic_fill_slot7(op1):
    """Operand 1 of a uniform atomic (op10094, op11765) with its return published on slot 7.

    Bits 20-23 of operand 1 are the scoreboard slot the returned old value lands on, plus one, as
    for a load: Apple's corpus fills slots 0 and 1 and every first consumer waits on exactly that
    slot (tools/g17waitlaw.py; 34 of 34). The witnesses cc copied fill slot 0, but every wait cc
    can emit for a late value - alu.12's load_wait, the stores' byte9[5] - is the SLOT-7 bit,
    because cc's loads fill slot 7. So #171's "mark the atomic late" routed consumers through a
    wait on a slot the atomic never touches: the residency probe's arrival records read the
    register's old contents at S = 256 and 512 (results/g17-tensor-resid-v3), and
    tensorview.hazards flags the program once it models atomics. Publishing on slot 7 makes the
    existing waits cover it. Slot 7 on these forms is NOT in Apple's corpus - the field is read
    as a slot from slots 0 and 1 - so it rests on a hardware check: the residency probe at
    S = 256, 3 rounds each way, arrivals 1..256 with slot 7 and all 1 with slot 0, the two
    programs differing in these four bytes only (isa/g17-execution-atomic-slot-results.json).
    op11765 takes the same field and was not dispatched here."""
    return (op1 & ~(0xF << 20)) | (8 << 20)


def _wait_for_load(out, val):
    """A range store's member that came from a LOAD is copied through an ALU that WAITS.

    THE FIRST READ-BINDING CONTROL RETURNED ZERO FOR EIGHT OF NINE WORDS (integration 854014a,
    results/g17-rangeread-runtime-v1): nine op12682 loads fed three range stores directly, and
    the selector for `store_range` had no load detection at all - the ordinary `store` widens to
    the 14-byte form and sets byte9[5] for a loaded value, the range path set nothing, so the
    stores read their registers before the loads landed (the same zero the load-use hazard
    comment describes). The sequential interpreter cannot see it: it completes loads at once.

    The wait used here is the one measured CAUSALLY: alu.12's byte0[3] (ledger/g17-alu-load-
    use-wait.toml - with it set the add reads the loaded value, cleared it reads the register's
    prior contents), one copy per loaded member, generic over the component count and both
    slot widths. The wide range form's byte9[5] is measured only for the 2-component op17244
    (the ordinary store's widening); on op17253 and op17262 it is the same byte in the encoder
    and an UNMEASURED bit on a different opcode (proven-of-an-encoding-not-an-opcode), so it is
    not used for them. One extra 12-byte instruction per loaded member is the price of a rule
    that rests on a measurement."""
    if not _is_load_value(val):
        return val
    copy = ir.Value(getattr(val, "type", ir.I32), "%s_waited" % (getattr(val, "name", "v") or "v"))
    out.append(MInst("alu.12", 12,
                     dict(op=3, mode=1, imm=0, src1_w=1, dest_w=1, srcb_w=1,
                          scale_code=2, load_wait=1, b4_5=None, b0_5=None,
                          keep=1 if val in _MULTI_USE else 0),
                     defs=[copy], uses=[val],
                     note="waits on the load: a range store reads its members at once"))
    return copy


# OPERAND ISOLATION FOR THE FOUR-BYTE REGISTER BITWISE. Off means the refusals below fire, which is
# what the negative control uses: the refusals are not removed, they are made unreachable for
# operands this pass has isolated, and an unsafe direct emission must still be refused.
_NO_BITWISE_ISOLATION = False
# The paired FMA waiting control's option. Default OFF: every delivered program's bytes are what
# they were. See the MACHINE_OPCODE selection below for what ON does and why that direction.
# CALLERS DO NOT ASSIGN THIS. `compile_function(..., fma_always_load_wait=True)` sets and restores it
# in a finally, so a delivery can RECORD the option it asked for and a refused compile cannot leave
# the selection state changed for whatever compiles next.
_FMA_ALWAYS_LOAD_WAIT = False
FMA_OPCODE = 2190
# Set to measure the indexed store as it was before the load-use copy above; see that comment.
_NO_STORE_AT_LOADWAIT = False
# The same, for the comparison that yields a value: set to measure op11372/op11375 as they were
# before the load-use copy, so the refusal that named this repair stays reproducible.
_NO_ICMP_LOADWAIT = False
# The same, for a value named in two slots of one generic-form instruction (O[u] = u): set to
# rebuild the store that released u through its index slot and stored 0 (MM 25.117).
_NO_DUPLICATE_OPERAND_KEEP = False
# the store opcodes whose value and index may be one register (tensorview.STORES plus the half and
# one-element word stores)
_STORE_OPCODES_DUP_KEEP = frozenset((17229, 17235, 17256, 17257, 17258, 13075, 17193, 17199))
# THE ONE-OPERAND INTEGER OPS AND low16 READ A LOADED SOURCE AFTER A WAIT, through the measured
# alu.12 waiting copy, as the float unary ops do. They did not: not/msb/reverse were emitted with
# hazard=0 and op590 (low16) with no wait, so a loaded source was read before it landed. Found by
# tools/g17ccfuzz.py (tensorview.hazards) and MEASURED by tools/g17unarywait.py
# (isa/g17-execution-unary-wait-results.json, MM 25.133): with the load's destination poisoned,
# all four ops returned op(POISON) on every lane - 1 and 32 threadgroups, 3 rounds - and with the
# copy every lane was right. True rebuilds the old bytes.
_NO_UNARY_LOAD_WAIT = False
# The same, for a load whose INDEX was loaded (`a[b[t]]`): set to rebuild the program
# tools/g17indirectload.py dispatched before the load branch waited on its index.
_NO_LOAD_INDEX_LOADWAIT = False


# VALUES THAT ARRIVE LATE: a consumer must wait for them. The list was ("load", "load_tg") at nine
# separate sites, so an imageblock read - also a memory read - fed its store with no wait, and a
# program compiled here that wrote the imageblock and read it back stored 0 on all 32 lanes while
# Apple's own kernel, through the same linker path, round-tripped (Piece C, 2026-09-23). One list,
# so the next late-arriving kind is added once.
# The vector load's lanes too: executed 2026-09-23, load_vec_at's second lane read 0 in every form
# (op12691 and op12709 at 8 and 14 bytes) because its first consumer, a multiply, did not wait -
# lane 0 was right only because it was consumed later.
LATE_KINDS = ("load", "load_tg", "imageblock_read", "load_vec_at", "vec_lane", "load_half2", "load_hi16")


# and16's templates: Apple's own instances (the qmv oracle, results/g17-qmv-v1/apple_b4_n2048_k2048), source
# lifetime 32 (keep). op426: R19 <- R43H & 15; op428: R19 <- R43L & (R50L).
AND16_TEMPLATE = {426: bytes.fromhex("330307123870a32ac283"), 428: bytes.fromhex("330607123870a12a0203")}
_AND16_WAITED = {}
# The negative control for and16(direct=True): True reads EVERY direct lane in place, the first consumer included,
# so nothing waits on the weight load (MM 25.144.4). Never set outside that control.
_AND16_DIRECT_UNWAITED = False
# COMPACT REGISTERS (MM 25.139.7), opt-in per function (`fn.compact_registers = True`): every value takes the
# LOWEST register that fits. The default gives an uncapped value the HIGHEST free wide register (to spare the
# low ones capped operands need), and register_count is the highest index named plus one - so a 61-register
# qmv declared 253 and ran at the occupancy of a 253-register kernel. A compact attempt that cannot place a
# capped value falls back to the default policy.
_COMPACT = [False]


def _and16_encoder(template, defs, uses, *, half=None, mask=None):
    raise AssertionError("bound per instruction")


def _and16_encode(opc, template, fields, defs, uses):
    """Write dest, the half-register source (slot 2r for L, 2r+1 for H) and the mask (imm, or the mask
    register's L half) into Apple's template; every other bit is the template's."""
    from agxforge.g17 import auth as _A
    a = fields["and16"]
    vals = {0: _A.field_value(opc, 0, defs[0]), 2: 2 * uses[0] + (1 if a["half"] == "H" else 0)}
    # op428's operand 4 is a UNIFORM half-register index (the template's type byte), MM 25.141.16: the pool form
    # writes the uniform the mask is preloaded into; the register form writes 2r, which names uniform 2r, not r
    vals[4] = a["mask"] if opc == 426 else (a["uniform"] if a.get("uniform") is not None else 2 * uses[1])
    return bytes(_A.encode(opc, vals, template=template, trusted=(2, 4)))[:10]


def _is_load_value(val):
    return (isinstance(val, ir.Value) and val.op is not None
            and (val.op.kind in LATE_KINDS or bool(val.op.attrs.get("is_load"))))


def _half_store_targets(fn, value):
    """The declared element of every buffer this value is half-STORED into, and nothing else.

    Returns (elements, only_half_stores). `elements` is the set of declared buffer elements the
    value reaches through an indexed half store; `only_half_stores` says whether every use of the
    value is one of those. A sixteen-bit constant is a HALF FLOAT only if its destination buffer
    says so - the IR types both `half` and `ushort` literals as I16, so the type cannot tell them
    apart and the declaration is what can.
    """
    elements, only = set(), True
    for blk in fn.blocks:
        for o in blk.ops:
            if not any(x is value for x in o.args):
                continue
            if (o.kind == "store_at" and o.attrs.get("width") == "half"
                    and len(o.args) > 2 and o.args[2] is value):
                elements.add(getattr(o.args[0], "elem", None))
            else:
                only = False
    return elements, only


def _f32_bits_of_half(bits):
    """The binary32 encoding of a binary16 pattern, EXACTLY, for the normal exponents only.

    Every finite binary16 value is exactly representable in binary32, so this is a widening with
    no rounding: sign through, exponent rebiased by 127-15, mantissa left-shifted 13. Returning
    None says the pattern is one this refuses - see the caller for which and why.
    """
    sign, exp, man = (bits >> 15) & 1, (bits >> 10) & 0x1F, bits & 0x3FF
    if exp == 0 or exp == 0x1F:          # zero and subnormal; infinity and NaN
        return None
    return (sign << 31) | ((exp - 15 + 127) << 23) | (man << 13)


def _isolate_bitwise_operand(out, val, *, force=False, keep_source=False,
                             why="operand isolation: the four-byte bitwise releases what it reads"):
    """Give the four-byte bitwise a copy of an operand it must not release, or may not read raw.

    The form cannot express either property. It has nowhere to put the source lifetime - 32 keep,
    16 release, measured elsewhere in this ISA - so it releases what it reads; and it does not wait
    for a load, measured both ways (`x & y` straight off two loads returns zero on every lane, the
    same kernel with one `add 0` on each operand returns 2, 0, 16, 24, 6). Its own refusal named
    the repair: "Put the operands through an ALU op first; those carry the wait."

    So that is what this emits - the same alu.12 copy `_wait_for_load` and `_materialise_sr`
    already use, with the same two measured bits: load_wait set when the operand came from a load
    (byte0[3], ledger/g17-alu-load-use-wait.toml), and keep set when the ORIGINAL still has a
    reader, so the copy does not free the value its own later reader needs.

    ALIASED OPERANDS GET TWO COPIES. `x & x` would otherwise put one register in both slots, and
    what is measured about this form is that two of them reading the SAME registers give a wrong
    answer while disjoint ones match Apple exactly. One instruction with aliased slots is not that
    measurement, so it is not assumed to be safe either: two distinct registers is the state the
    evidence covers. The first copy keeps its source because the second copy still has to read it.
    """
    if not isinstance(val, ir.Value):
        return val
    from_load = _is_load_value(val)
    if not (force or from_load or val in _MULTI_USE):
        return val
    copy = ir.Value(getattr(val, "type", ir.I32), "%s_iso" % (getattr(val, "name", "v") or "v"))
    out.append(MInst("alu.12", 12,
                     dict(op=3, mode=1, imm=0, src1_w=1, dest_w=1, srcb_w=1,
                          scale_code=2, load_wait=1 if from_load else 0, b4_5=None, b0_5=None,
                          keep=1 if (keep_source or val in _MULTI_USE) else 0),
                     defs=[copy], uses=[val],
                     note=why))
    return copy



# THE CAPABILITY-OFF ARM FOR THE PHI-INTERFERENCE REPAIR. True restores the state this repair
# replaced: phi coalescing pre-colours the phi, its entry value AND its latch value to one
# register unconditionally, so a phi that is still READ after its latch value is defined has those
# reads answered with the latch value. Off is the shape that emits a wrong program, which is why
# it exists only so the old failure stays reproducible in a test.
_NO_PHI_INTERFERENCE_COPY = False


def _phi_interference_copies(insts):
    """Break the phi/latch register equality where the phi is still read after the latch is defined.

    THE DEFECT THIS REPAIRS. `Alloc.run`'s PHI COALESCING pre-colours every member of a phi group -
    the phi, its entry value and its latch value - to ONE register, with no interference test. That
    is correct only when the phi is dead at the latch value's definition. `_wide_add_words` breaks
    it: the carry needs the OLD low word after the new sum exists, so the phi has two reads after
    its latch value is defined, and both are answered with the sum. The carry then collapses to
    `addend_low >> 31` - constant, independent of whether the addition overflowed. A SCALAR loop
    does the same thing whenever the latch value is computed before a remaining read of the phi, so
    this is a phi-interference defect and not anything about multi-word values.

    THE REPAIR IS THE CLASSICAL ONE, with no new opcode. Where the phi interferes with its latch
    value, the latch value keeps its own register and an `alu.12` add-zero copy - the same copy
    `_isolate_bitwise_operand`, `_wait_for_load` and `_materialise_sr` already emit - moves it into
    the phi's register at the END of the body, after the phi's last read. The phi group is rewritten
    to hold that copy instead of the latch value, so the loop-carried equality is restored at the
    edge rather than imposed across the body.

    A NON-INTERFERING GROUP IS LEFT ALONE. That is not a claim that every program keeps its bytes -
    this pass deliberately changes the programs that were wrong and adds two refusals. What was
    actually measured is the inspected domain: the 197-source census (125 compiled, no back edges,
    every code hash identical) and the four retained loop programs `ladder:counted_loop`,
    `g17endtoend._divgemv_ir`, `_dotloop_ir` and `_gemvloop_ir`, all four byte-identical. Programs
    outside those two populations were not examined.

    Returns a list of records describing what was inserted, for the handoff and the tests.
    """
    if _NO_PHI_INTERFERENCE_COPY:
        return []
    groups = [(i, m) for i, m in enumerate(insts) if m.fields.get("phi_group")]
    if not groups:
        return []

    def def_index(value):
        for i, m in enumerate(insts):
            if any(d is value for d in m.defs):
                return i
        return None

    def use_indices(value):
        return [i for i, m in enumerate(insts) if any(u is value for u in m.uses)]

    # PHYSICAL EQUALITIES FROM THE UNION, WHICH A PER-GROUP VIEW CANNOT SEE. The coalescing block
    # unions any two groups that share a value and then pre-colours the whole union to ONE
    # register. Two phis that merely share an entry value therefore end up in the same register
    # even though they carry different loop values, and no copy placed inside one group can undo
    # that: the equality is imposed on the union, not on a group. Refused rather than emitted,
    # because proving two loop-carried values are never simultaneously live is exactly the
    # interference question this pass exists because the allocator does not ask.
    union = []
    for _i, m in groups:
        g = {id(v): v for v in m.fields["phi_group"] if isinstance(v, ir.Value)}
        phis = [m.fields["phi_group"][0]] if m.fields["phi_group"] else []
        hit = [u for u in union if set(u[0]) & set(g)]
        for u in hit:
            g.update(u[0]); phis += u[1]; union.remove(u)
        union.append((g, phis))
    for _members, phis in union:
        if len(phis) > 1:
            raise Unsupported(
                "%d loop phis share one pre-coloured register because their groups intersect "
                "(%s): the coalescing union imposes that equality on every member at once, so no "
                "per-group copy can separate them, and this pass will not assume two "
                "loop-carried values are never live together"
                % (len(phis), ", ".join(sorted(str(getattr(v, "name", v)) for v in phis))))

    inserted = []

    # ENTRY MEMBERS ARE NOT AUTOMATICALLY DEAD, and assuming they were is what this phase fixes.
    # An entry value is pre-coloured to the phi's register, so the body's write to that register
    # destroys it. That is only safe if nothing reads the entry value again - and it is not an IR
    # invariant that nothing does. Root's counterexample stores `entry_a + 1` AFTER the loop and
    # read 23 instead of 8, because r16 held the final accumulator by then; a read INSIDE the body
    # is wrong for the same reason from the second iteration onwards. So where an entry value has
    # any real use, it keeps its own register and an `alu.12` add-zero copy carries it into the
    # phi's register before the loop starts. Where it has none, today's coalescing is correct and
    # the program keeps its bytes.
    for pass_no in range(len(groups) + 1):
        changed = False
        for m in [mm for _i, mm in groups]:
            marker = next(i for i, x in enumerate(insts) if x is m)
            members = [v for v in m.fields["phi_group"] if isinstance(v, ir.Value)]
            if not members:
                continue
            phi, rest = members[0], members[1:]
            for entry in rest:
                entry_def = def_index(entry)
                if entry_def is None or entry_def >= marker:
                    continue                  # not an entry member of this group
                # ONLY A READ THAT CAN OBSERVE THE CLOBBER COUNTS. The body's write to the
                # phi register happens at or after the marker, so a read BEFORE the marker - a
                # pre-header guard's compare, say - sees the entry value intact and is safe.
                # `g17endtoend._divgemv_ir` is exactly that shape: its `x0` feeds the guard
                # `cmp(x0, 8, lt)` before the loop, and gating on "any use at all" inserted a
                # copy there and changed a dispatched program's bytes for nothing.
                uses = [u for u in use_indices(entry) if u > marker]
                if not uses:
                    continue                  # no read can see the clobber: coalescing is correct
                copy = ir.Value(getattr(entry, "type", ir.I32),
                                "%s_entry" % (getattr(entry, "name", "v") or "v"))
                insts.insert(entry_def + 1,
                             MInst("alu.12", 12,
                                   dict(op=3, mode=1, imm=0, src1_w=1, dest_w=1, srcb_w=1,
                                        scale_code=2,
                                        load_wait=1 if _is_load_value(entry) else 0,
                                        b4_5=None, b0_5=None, keep=1),
                                   defs=[copy], uses=[entry],
                                   note="phi-interference copy: this entry value is read again, "
                                        "so it may not share the phi's register"))
                m.fields["phi_group"] = [x if x is not entry else copy
                                         for x in m.fields["phi_group"]]
                inserted.append(dict(kind="entry", phi=getattr(phi, "name", None),
                                     entry=getattr(entry, "name", None),
                                     copy=getattr(copy, "name", None),
                                     at=entry_def + 1, entry_reads=uses))
                changed = True
                break
            if changed:
                break
        if not changed:
            break

    for marker, m in groups:
        marker = next(i for i, x in enumerate(insts) if x is m)
        members = [v for v in m.fields["phi_group"] if isinstance(v, ir.Value)]
        if not members:
            continue
        phi, rest = members[0], members[1:]
        for latch in rest:
            latch_def = def_index(latch)
            if latch_def is None:
                continue                      # an entry value defined outside this list
            # Entry-member interference was handled by the preceding phase. This phase handles
            # members defined inside the body; it must not treat a pre-header entry copy as a
            # latch or repeat the entry-member analysis.
            if latch_def < marker:
                continue
            after = [u for u in use_indices(phi) if u > latch_def]
            if not after:
                continue                      # no interference: today's coalescing is correct
            # THE BACK EDGE IS CONDITIONAL, SO A COPY BEFORE IT RUNS ON EXIT TOO. The copy has to
            # sit inside the body, before the branch; a position derived from the phi's last read
            # would land outside the loop entirely if that read is in the exit block, and the
            # loop-carried update would then never happen on the edge.
            edge = next((k for k in range(marker, len(insts))
                         if insts[k].form == "branch.cond.back"), None)
            if edge is None:
                continue                      # no back edge: nothing is carried
            outside = [u for u in after if u >= edge]
            if outside:
                # Preserving the phi's OLD value for a read in the branch condition or the exit
                # block needs the copy on the back EDGE alone rather than on the body's
                # fall-through, which this machinery has no way to express - a predicated or
                # edge-placed copy would be a new encoding. Refused rather than guessed.
                raise Unsupported(
                    "a loop phi %r is read at or after the conditional back edge (instruction "
                    "%d) while interfering with its latch value %r: a copy placed before the "
                    "edge would also execute when the loop exits, and placing one on the edge "
                    "alone is not a form this backend has, so this shape refuses rather than "
                    "being emitted" % (getattr(phi, "name", phi), min(outside),
                                       getattr(latch, "name", latch)))
            after = [u for u in after if u < edge]
            if not after:
                continue
            # the copy goes after the phi's last read and before the loop's control sequence
            last_read = max(after)
            at = min(last_read + 1, edge)     # after the phi's final body read, before the edge
            keep_latch = any(u > last_read for u in use_indices(latch))
            copy = ir.Value(getattr(latch, "type", ir.I32),
                            "%s_phi" % (getattr(latch, "name", "v") or "v"))
            insts.insert(at, MInst("alu.12", 12,
                                   dict(op=3, mode=1, imm=0, src1_w=1, dest_w=1, srcb_w=1,
                                        scale_code=2,
                                        load_wait=1 if _is_load_value(latch) else 0,
                                        b4_5=None, b0_5=None,
                                        keep=1 if keep_latch else 0),
                                   defs=[copy], uses=[latch],
                                   note="phi-interference copy: the phi is read after this latch "
                                        "value is defined, so they may not share a register"))
            m.fields["phi_group"] = [x if x is not latch else copy
                                     for x in m.fields["phi_group"]]
            inserted.append(dict(phi=getattr(phi, "name", None),
                                 latch=getattr(latch, "name", None),
                                 copy=getattr(copy, "name", None),
                                 at=at, latch_def=latch_def, phi_reads_after=after,
                                 latch_kept=keep_latch))
    return inserted


def _emit_cmp(out, cond):
    """The three instructions that produce a predicate, with the hazard checked.

    NESTS, to the measured depth. The mask is a LIFO stack: `1e 00 00 0e` pushes one level and
    `3e 03 40 0e` pops one, confirmed by three preregistered structural predictions over Apple's
    own output (ledger/g17-predication-nests.toml). This compiler emits ONE restore per region and
    so never needs `be 03 40 0e`, the pop-two form, which Apple uses only where two regions end
    together - the shape whose balance those predictions verified.

    Depth beyond MAX_PRED_DEPTH is refused: 4 is the deepest Apple was observed to emit, not a
    known hardware limit.
    """
    depth = sum(m.form == "exec.mask" for m in out) - sum(m.form == "exec.restore" for m in out)
    if depth >= ir.MAX_PRED_DEPTH:
        raise Unsupported("predication would nest %d deep; %d is the deepest Apple's compiler was "
                          "observed to emit and the mask stack is unmeasured beyond it"
                          % (depth + 1, ir.MAX_PRED_DEPTH))
    src = cond.op.args[0]
    if src.op is not None and (src.op.kind in LATE_KINDS
                               or src.op.attrs.get("is_load")):
        raise Unsupported("the compared value comes straight from a load, and the compare's "
                          "operand has a load-use hazard: with the load still in flight the "
                          "compare reads the register's previous contents. The compare's wait "
                          "mechanism is not recovered, so this is refused rather than emitted")
    rel, imm = _canon_cmp(cond.op.attrs.get("pred"), _cmp_bound_or_refuse(cond))
    if src in _CONSUMED: _rematerialise(out, src)
    # THE COMPARE'S SOURCE LIFETIME IS AUTHORED, not inherited. Released, the register is gone for
    # everything after the compare - which is how the first branching kernel this compiler compiled
    # sent every guarded lane's store to slot 0, the index having been the compare's source.
    keep = _cmp_keeps(src, cond.op)
    if not keep: _CONSUMED.add(src)
    out.append(MInst("cmp.6", 6, dict(imm=imm, rel=rel, keep=keep, source_modifier=cond.op.attrs.get('source_modifier',0)), uses=[src],
                     note="the compared register, authored; %s"
                          % ("KEPT: it has a later reader" if keep
                             else "released - nothing reads it again")))
    out.append(MInst("exec.mask", 4, {}, note="exec mask <- predicate; the EFFECT is measured, the fields are not"))

def _emit_cmp_for_loop(out, cond):
    """A loop latch: the compare, then the LOOP flag instruction rather than the forward one."""
    src = cond.op.args[0]
    if src.op is not None and (src.op.kind in LATE_KINDS
                               or src.op.attrs.get("is_load")):
        raise Unsupported("the compared value comes straight from a load, and the compare's "
                          "operand has a load-use hazard")
    # POP BEFORE THE COMPARE, so the loop mask is REBUILT each iteration rather than narrowed.
    #
    # op582 ANDs with the current mask and pushes, so a loop that only pushes narrows monotonically
    # and grows a level per iteration. That is correct INSIDE the body and leaves a partial mask
    # afterwards that the code following the loop cannot survive: measured, a divergent loop
    # lowered that way faults (cb=-1, GPU Address Fault) where the uniform one is right.
    #
    # A pop first restores the pre-loop mask; the compare then runs under it and the push applies
    # THIS iteration's predicate to it. Depth stays at one however many iterations run, and the pop after the loop leaves the
    # pre-loop mask rather than a partial one. Measured straight-line, no branch involved, in
    # tools/g17maskseq.py:
    #
    #     push(30)   pop cmp push(20)   pop cmp push(12)   pop
    #     0..29      0..19              0..11              0..31   <- the PRE-LOOP mask
    #
    # push(30) there is the PREHEADER push, and this lowering did not emit one: the first trip's pop
    # found nothing of the loop's own. At a kernel's top level that is harmless; inside a guarded
    # region it popped the REGION's level and re-enabled the lanes the guard had switched off for
    # everything after the loop (MM 25.139.9). _emit_loop_entry_push now emits the preheader push for
    # loops nested in a region; top-level loops keep their bytes.
    #
    # The pop goes before the COMPARE, which is both what was measured and what the flag
    # discipline requires: a compare's flag must be consumed by the very next instruction.
    #
    # A lane that has already left keeps its own compare's last result, which was false, so the
    # AND keeps it out - it does not re-enter. The push must stay IMMEDIATELY before the branch,
    # because op582 is what gates op458.
    out.append(MInst("exec.restore", 4, dict(label="looptop_%d" % (id(out) & 0xFFFF)),
                     note="rebuilds the loop mask: restores the pre-loop mask so the push below "
                          "applies this iteration's predicate to it rather than to last "
                          "iteration's"))
    rel, imm = _canon_cmp(cond.op.attrs.get("pred"), _cmp_bound_or_refuse(cond))
    if src in _CONSUMED: _rematerialise(out, src)
    # A LATCH CANNOT USE POSITION. The back edge puts the body's readers after this compare in
    # execution order however they sit in block order, so any other reader at all keeps the source.
    keep = src in _MULTI_USE
    if not keep: _CONSUMED.add(src)
    out.append(MInst("cmp.6", 6, dict(imm=imm, rel=rel, keep=keep, source_modifier=cond.op.attrs.get('source_modifier',0)), uses=[src],
                     note="the compared register, immediate, relation and source lifetime authored"))
    _emit_loop_flag(out)

# WHICH VALUES DIFFER BETWEEN LANES. thread_position_in_grid and thread_position_in_threadgroup
# are the roots; threadgroup_position_in_grid is the same for every lane of a threadgroup and is
# not one. A load is lane-varying exactly when its address is - lanes reading the same address get
# the same value - and everything else inherits it from its operands.
LANE_VARYING_SR = {"thread_position_in_grid", "thread_position_in_threadgroup"}


def _lane_varying(v, seen=None):
    if not isinstance(v, ir.Value) or v.op is None:
        return False
    seen = seen if seen is not None else set()
    if id(v) in seen:
        return False                       # a phi cycle: decided by its other inputs
    seen.add(id(v))
    op = v.op
    if op.kind == "builtin":
        return op.attrs.get("which") in LANE_VARYING_SR
    return any(_lane_varying(a, seen) for a in op.args)


def _emit_loop_flag(out):
    """The flag instruction a BACK EDGE needs, which is not the one a forward region uses.

    Apple's decoder, over 200 objects: a forward branch (opcode 462) is preceded by opcode 582,
    `1e 00 00 0e`, in 29 of 31 cases; a backward branch (opcode 458) is preceded by opcode 579,
    `fe 00 02 0e`, in 27 of 31. Different opcode, different encoding, and the decoder's last
    immediate is 1 for the forward one and 2 for the loop one.
    ledger/g17-control-flow-vocabulary.toml
    """
    # MEASURED, and it does NOT do what a loop needs: with op579 in front of a forward branch the
    # then-block runs at every threadgroup width, including those where the compare is false for
    # every lane, while op582 correctly skips it. op579 does not gate on the predicate.
    # ledger/g17-control-flow-vocabulary.toml [what_op579_does_measured_not_inferred]
    # op582, NOT op579, AND THIS IS MEASURED. Apple pairs its back edges with a `while` exec -
    # op579 in 490 of 579, op578 in 89 - and copying that would have produced a kernel that never
    # exits. The back edge was executed safely with each candidate in front of it, in a program
    # with NO REACHABLE CYCLE (the branch's target is a pre-entry field of 2-byte nops ending in
    # `end`, so taken or not the program stops in one step):
    #
    #     exec before op458      predicate TRUE     predicate FALSE
    #     op579 while+invert     taken              taken
    #     op578 while            taken              taken
    #     op582 if               taken              FELL THROUGH
    #     none                   taken              taken
    #
    # op458 repeats WHILE LANES ARE ACTIVE and reads the mask op582 sets. It is the only one of
    # the four that gates it, and it is the instruction this compiler already emits for a
    # conditional - execution-proven since the exec-mask ladder. Whatever the `while` execs do for
    # Apple, it is not what makes their back edge fall through.
    # ledger/g17-the-back-edge-gates-on-op582.toml
    out.append(MInst("exec.mask", 4, {}, note="op582 sets the mask from the compare; op458 "
                                              "repeats while any lane is active"))

# The IR's integer types. A float type reaching an integer ALU form is a malformed program, not
# a compiler defect, and the two must be told apart - see _width.
_FLOAT_TYPES = ("f16", "f32")


def _width(t):
    """The integer operand width. ir.WIDTH covers i16 and i32 and nothing else.

    A FLOAT TYPE ARRIVING HERE IS A MALFORMED PROGRAM AND GETS A NAMED REFUSAL. The IR keeps the
    integer and float operations separate - `add`, `mul`, `sub`, `shl` are integer; `fadd`,
    `fmul`, `fmin`, `fmax`, `fma` are float - so a float value reaching an integer form means the
    caller built the wrong operation. Before this, ir.WIDTH raised KeyError('f32') from inside the
    form's field dict, which reaches the caller as an unexplained compiler crash several frames
    from the cause: a GEMM-plus-residual function using `add` on two f32 loads failed exactly that
    way.

    ANY OTHER UNKNOWN TYPE STILL RAISES KeyError, deliberately. That case is a selector routing a
    type nobody planned for, which is a defect in this compiler, and reporting it to the caller as
    "your program is wrong" would hide it. Only the types known to be floats are refused by name.
    """
    if t in _FLOAT_TYPES:
        raise Unsupported(
            "integer ALU form selected for a %s value: the IR's add/sub/mul/shl family is "
            "INTEGER-ONLY (ir.WIDTH covers %s). Use the float operation instead - fadd, fmul, "
            "fmin, fmax, fma, fsat - or convert with f32_to_f16_rte / f16_to_f32 first. A "
            "tensor result consumed by a scalar residual needs fadd, not add."
            % (t, ", ".join(sorted(ir.WIDTH))))
    return ir.WIDTH[t]

def _b45():
    """byte4[5], UNRESOLVED. Inherited at 1, which is what 90% of Apple's adds carry and what every
    program executed here has used. G17_FORCE_B45 authors it in the other direction so the bit can
    be tested rather than assumed inert."""
    v = os.environ.get("G17_FORCE_B45")
    return None if v is None else int(v)


def _flag(which):
    """Which FLAG a compare writes and its exec-mask reads. One allocator-wide value for now: the
    mask stack carries the region state, so a compare is dead the instant the mask is pushed and
    reuse is safe. G17_FLAG selects a different one, and G17_FLAG_CMP / G17_FLAG_EXEC set them
    SEPARATELY - a mismatch is the control that shows the selector is read rather than inert."""
    v = os.environ.get("G17_FLAG_" + which)
    return int(v if v is not None else os.environ.get("G17_FLAG", "0"))


def _b05():
    """byte0[5] and byte6[5], written TOGETHER because Apple never separates them. Unresolved;
    G17_FORCE_B05 authors the other value so the pair can be tested."""
    v = os.environ.get("G17_FORCE_B05")
    return None if v is None else int(v)


def _all_uses_are_16_bit(fn, value):
    """Every instruction that reads `value` reads it as SIXTEEN bits. The witnessed consumer is the half store
    (store_at width=half, witness W0); a half-typed ALU destination whose every register operand is half counts
    too, because the ALU's width bit says so. A value with NO use does not qualify: there is no witness for an
    unread sixteen-bit zero, and the 32-bit form is the safe one."""
    users = [o for blk in fn.blocks for o in blk.ops if any(x is value for x in o.args)]
    if not users:
        return False
    for o in users:
        if o.kind == "store_at" and o.attrs.get("width") == "half":
            continue
        if getattr(o.dest, "type", None) == ir.I16 and all(getattr(a, "type", ir.I16) == ir.I16 for a in o.args if isinstance(a, ir.Value)):
            continue
        return False
    return True


def _feeds_a_half_store_value(fn, value):
    """`value` is the VALUE operand of a half store somewhere in `fn` - not its index, which is an ordinary
    thirty-two-bit address register. The value operand is the one that names the 425-based file."""
    return any(o.kind == "store_at" and o.attrs.get("width") == "half" and len(o.args) > 2 and o.args[2] is value
               for blk in fn.blocks for o in blk.ops)


def _depends_on_half_load(value, seen=None):
    """Whether a value has a sixteen-bit device load in its SSA ancestry.

    The indexed half-store consumer has no recovered dependency mechanism for this producer
    family.  The first guard covered only an immediate add; the bounded family showed that two
    waited ALUs, a half-to-float round trip, and two independent FMAs all leave odd half elements
    unwritten.  Track the actual source width through the value graph so a short-looking wrapper
    cannot evade the refusal.  Word loads remain a separate, executed path.
    """
    if not isinstance(value, ir.Value) or value.op is None:
        return False
    if seen is None:
        seen = set()
    marker = id(value)
    if marker in seen:
        return False
    seen.add(marker)
    op = value.op
    if op.kind in ("load", "load_tg"):
        return op.attrs.get("width") == "half" or getattr(value, "type", None) is ir.I16
    if op.attrs.get("is_load") and getattr(value, "type", None) is ir.I16:
        return True
    return any(_depends_on_half_load(arg, seen) for arg in getattr(op, "args", ()))


def _load_wait(args):
    """byte0[3]: set when an operand comes from a LOAD.

    Recovered by controlled compiler differential - it is the one property out of eleven that
    moves the bit, and the corpus agrees (set in 19.7% of adds following a load, 0.0% of those
    following an ALU). ledger/g17-alu-load-use-wait.toml.

    G17_FORCE_LOAD_WAIT overrides it so the negative cell of the execution test goes through this
    same encoder rather than a hand-built one.
    """
    forced = os.environ.get("G17_FORCE_LOAD_WAIT")
    if forced is not None:
        return int(forced)
    # A LOAD AUTHORED THROUGH THE GENERIC FORM IS STILL A LOAD. ir.machine(..., is_load=True) says
    # the result arrives late, and every place that asks "did this come from a load" has to accept
    # it - a threadgroup load's round trip returned zero until it did.
    # A TEXTURE FETCH IS NOT ON THIS LIST, AND THAT IS A REVERT RATHER THAN AN OMISSION. The
    # consumer of a texture read returns the value its destination register held BEFORE the fetch -
    # `x`, on all 32 lanes, exactly - which is what reading a register whose write has not landed
    # looks like, so byte0[3] was set here on that hypothesis. It changed nothing: the answer was
    # `x` with the bit and `x` without it.
    #
    # That experiment CANNOT distinguish the two cases, which is why the bit came back out. If the
    # fetch never writes its destination at all, no wait on the consumer can help, and the result
    # is identical either way. Leaving the bit set would have been a guess dressed as a fix - the
    # same shape as the ALU control that "ruled out" a hazard it could never have separated.
    # A VECTOR LOAD SETS IT ON ONE CONSUMER (handoff 10ae, measured on Apple's V0/V1/V2/V4/V5/V6): the add that reads
    # the tuple's FIRST register - the load's own destination value - carries byte0[3]; the adds reading lanes 1..3
    # carry it clear, on every witness. So the wait is a property of the load's value, exactly as the scalar rule
    # says, and `vec_lane` values are not that value.
    #
    # THE RULE IS "THE FIRST CONSUMER WAITS", and Apple's lane-0 add is simply first in every witness.
    # Executed 2026-09-23: a program whose FIRST consumer was lane 1 (a multiply) read lane 1 as 0 in
    # op12691 and op12709 at both lengths - nothing had waited yet. So whichever consumer of a vector
    # load comes first waits, lane 0's consumer always does (Apple's bytes unchanged), and later lane
    # consumers stay clear.
    vec = [a for a in args if isinstance(a, ir.Value) and a.op is not None
           and a.op.kind in ("load_vec_at", "vec_lane")]
    if vec:
        loads = [a if a.op.kind == "load_vec_at" else a.op.args[0] for a in vec]
        first = any(l not in _VEC_WAITED for l in loads)
        _VEC_WAITED.update(loads)
        if any(a.op.kind == "load_vec_at" for a in vec) or first:
            return 1
        return 0
    return int(any(isinstance(v, ir.Value) and v.op is not None
                   and (v.op.kind in LATE_KINDS or v.op.attrs.get("is_load"))
                   for v in args))

# --- phase 1: instruction selection -------------------------------------------------------

def _hz(args):
    """The hazard word for the ALU families that do NOT go through the alu.12 emit path.

    alu.12 derives its hazard from `load_wait` at emit time; every other ALU family - mul, sub, the
    shifts, the bitwise pair, the saturating pair - passes `hazard` straight into encode_alu_form
    and none of their lowerings ever set it. So `mul(load(...), w)` read its operand before the load
    landed and returned ZERO, silently, at status 0.

    Found by the end-to-end proof and not by anything before it: every rung that loads happens to
    feed an `add`, which is on the alu.12 path and did wait. A four-tap dot product multiplies, and
    the whole sum came back as exactly the bias.
    ledger/g17-only-one-alu-family-waited-for-a-load.toml
    """
    return (1 << 31) if _load_wait(args) else 0

# THE MASK STACK DOES NOT BOUND A LOOP'S TRIP COUNT HERE, and this comment used to say it did.
# The depth is measured - 256 nested pushes, tools/g17maskdepth.py - and it was read as "one
# level per iteration, so a loop must run fewer than 256 times". That describes a loop lowered
# push-only. This compiler's lowering is `exec.restore ; cmp ; exec.mask ; op458`: the POP comes
# before the compare, the push applies this iteration's predicate to the pre-loop mask, and the
# depth is the same at every iteration - one above the enclosing region - however many run.
# Measured, not inferred: ledger/g17-the-loop-mask-is-rebuilt-not-narrowed.toml executed divergent
# loops with per-lane trip counts 0..31 whose trailing store saw every lane, which a growing
# stack would have made impossible. So what the stack bounds is NESTING - how many regions are
# open at once - and that is what _prove_terminates checks against it now. The trip count is
# still proven, because termination is a separate question from depth.
MASK_STACK_DEPTH = 256


def _imm_of(v):
    """The compile-time value of `v`, or None. An Imm, or a `const` that holds one."""
    if isinstance(v, ir.Imm):
        return v.v
    if (isinstance(v, ir.Value) and v.op is not None and v.op.kind == "const"
            and v.op.args and isinstance(v.op.args[0], ir.Imm)):
        return v.op.args[0].v
    return None


RUNTIME_CAP_MAX = 1 << 24      # the same "beyond anything this backend dispatches" line as a constant bound


def _lower_runtime_bounds(fn):
    """Rewrite each branch compare against a RUNTIME bound into measured forms, in place.

    `i < n` with n a register has no measured compare: this backend's branch compare is op10369/6
    with an eight-bit immediate, and Apple's register-bounded form op10369/10 is refused in
    _cmp_bound_or_refuse for reasons its campaigns record (two operand layouts share the length;
    the relation cell is unsettled). But the VALUE compare is measured - op11372, `icmp`, 0 or 1,
    its ult code read off execution - and so is `cmp.6 x > 0`. So the predicate is computed as a
    value and branched on:

        t = icmp.ult(i, n)               p = t                 (a forward guard)
        u = icmp.ult(i, const cap)       p = and(t, u)         (a loop latch; cap is REQUIRED)
        cmp p > 0

    THE CAP IS WHAT MAKES A LOOP DISPATCHABLE. With n a register nothing bounds the trip count
    short of 2^32/step, and a non-terminating back edge hangs the GPU (it has rebooted this
    machine). The author states the most n can be; the compiled predicate enforces it, so the proof
    in _prove_terminates runs against a constant and a broken promise truncates the loop instead
    of hanging it. It is a promise the program states, never one the compiler invents: a latch
    without a cap is refused.

    Only `lt` is rewritten - the one relation whose unsigned value compare is measured and whose
    termination argument is written. The original operands stay on the compare as `runtime`, so
    the prover reads the program's comparison rather than the lowering's."""
    latches = set()
    for bi, blk in enumerate(fn.blocks):
        t = blk.ops[-1] if blk.ops else None
        if t is not None and t.kind == "br_cond":
            if any(t.args[1] is b for b in fn.blocks[:bi + 1]):
                latches.add(t.args[0])
    for blk in fn.blocks:
        new = []
        for o in blk.ops:
            if o.kind == "cmp" and len(o.args) > 1 and isinstance(o.args[1], ir.Value) \
                    and _imm_of(o.args[1]) is None and "runtime" not in o.attrs:
                a, n = o.args[0], o.args[1]
                cap = o.attrs.get("cap")
                if o.attrs.get("pred") != "lt":
                    raise Unsupported("a compare against a runtime bound is lowered for `lt` only "
                                      "(icmp.ult then cmp > 0); %r is not" % o.attrs.get("pred"))
                if o.dest in latches and cap is None:
                    raise Unsupported("a loop latch against a runtime bound must state cap=: nothing "
                                      "else bounds its trip count, and a back edge that does not end "
                                      "hangs the GPU. cmp(i, n, cap=K) compiles `i < n && i < K`")
                if cap is not None and not 0 < cap <= RUNTIME_CAP_MAX:
                    raise Unsupported("cap %r is outside 1..%d" % (cap, RUNTIME_CAP_MAX))
                nm = o.dest.name
                t = ir.Value(name=nm + "_lt")
                new.append(ir.Op("icmp", t, [a, n], rel="ult"))
                pv = t
                if cap is not None:
                    kc, u, pv = (ir.Value(name=nm + "_capc"), ir.Value(name=nm + "_ltcap"),
                                 ir.Value(name=nm + "_p"))
                    new.append(ir.Op("const", kc, [ir.Imm(cap)]))
                    new.append(ir.Op("icmp", u, [a, kc], rel="ult"))
                    new.append(ir.Op("and", pv, [t, u]))
                o.attrs = dict(pred="gt", source_modifier=0, runtime=(a, n, cap))
                o.args = [pv, ir.Imm(0)]
            new.append(o)
        blk.ops[:] = new


def _counted(cond):
    """(phi, step, bound) when the latch predicate is `phi advanced by a positive constant < a
    constant`, else None.

    SSA IS DOING THE HARD PART. A value has exactly one definition, so proving the induction
    variable is only ever advanced by this one instruction needs no scan of the body: if the phi's
    latch argument IS the value being compared, nothing else can have written it.
    """
    op = cond.op
    if op.kind == "cmp" and op.attrs.get("runtime"):
        # the lowered predicate is `p > 0`; the program's comparison is `nxt < n && nxt < cap`
        nxt, _n, bound = op.attrs["runtime"]
        if bound is None:
            return None
    else:
        if op.kind != "cmp" or op.attrs.get("pred") != "lt":
            return None
        bound = _imm_of(op.args[1])
        if bound is None:
            return None
        nxt = op.args[0]
    if not (isinstance(nxt, ir.Value) and nxt.op is not None and nxt.op.kind == "add"):
        return None
    a, b = nxt.op.args
    step, phi = _imm_of(b), a
    if step is None:
        step, phi = _imm_of(a), b
    if step is None or step <= 0:
        return None
    if not (isinstance(phi, ir.Value) and phi.op is not None and phi.op.kind == "phi"):
        return None
    if len(phi.op.args) < 2 or phi.op.args[1] is not nxt:
        return None
    return phi, step, bound


def _prove_terminates(blocks, header, cond):
    """(trip count, why) for a loop this compiler can prove ends, or (None, why not).

    THE FAILURE MODE HERE IS NOT A WRONG ANSWER, IT IS A HANG. A forward branch can only move the
    PC forward, which is what made every other authored branch safe to dispatch; a back edge whose
    predicate never goes false wedges the GPU, and that has already killed WindowServer once on
    this machine. So the loop lowering used to refuse outright and take the caller's word through
    G17_ALLOW_LOOP=1 - a human's promise standing in for an analysis.

    This is the analysis. It proves two shapes and refuses everything else BY NAME:

        a constant start    the recurrence is fully determined, so it is simulated to the
                            predicate's first false and the exact trip count comes out
        a guarded start     the start is lane-varying - `x = t`, which is what a triangular loop
                            looks like - but the header is only reachable through a br_cond on
                            `start < the same bound`. Entry therefore has start < bound, the step
                            is positive, and the values are unsigned, so at most ceil(bound/step)
                            iterations run whatever the lane's start was.

    The trip count is NOT checked against the mask-stack depth any more: the lowering pops before
    the compare, so the depth is invariant across iterations (see MASK_STACK_DEPTH). A loop proven
    to end after 10,000 iterations does not overflow anything; it is merely long.
    """
    got = _counted(cond)
    if got is None:
        return None, ("the latch predicate is not a counted induction. This proves termination for "
                      "`phi + positive constant < constant` and nothing else, so a loop whose bound "
                      "or step is computed needs G17_ALLOW_LOOP=1 and a human who has bounded it")
    phi, step, bound = got
    if cond.op.attrs.get("runtime"):
        # A CAPPED RUNTIME BOUND NEEDS NO GUARD ON ENTRY. The latch continues only while
        # nxt < cap, so every iteration after the first starts below cap; with cap + step <= 2^32
        # the increment cannot wrap, so the counter rises strictly and passes cap within
        # ceil(cap/step) more. That holds for ANY start - lane-varying included.
        if bound + step > 1 << 32:
            return None, ("cap %d plus step %d reaches 2^32, so the increment can wrap below the "
                          "cap and the count is not bounded" % (bound, step))
        n = -(-bound // step) + 1
        if n > RUNTIME_CAP_MAX:
            return None, ("cap %d at step +%d allows %d iterations, beyond anything this backend "
                          "dispatches without a stated reason" % (bound, step, n))
        return n, ("the latch compiles `nxt < n && nxt < %d` with step +%d, so at most %d "
                   "iterations run whatever n and the start are" % (bound, step, n))
    init = _imm_of(phi.op.args[0])
    if init is not None:
        n, v = 0, init
        while True:
            v += step
            n += 1
            if not (v < bound):
                break
            if n > 1 << 24:
                return None, ("the induction runs from %d by +%d against %d, which is %d or more "
                              "iterations - proven finite, but beyond anything this backend will "
                              "dispatch without a stated reason" % (init, step, bound, n))
        return n, ("the induction starts at %d, advances by +%d and is compared against %d, so the "
                   "predicate is false forever after %d iterations" % (init, step, bound, n))
    for blk in blocks:
        t = blk.term
        if t is None or t.kind != "br_cond" or t.args[1] is not header:
            continue
        g = t.args[0]
        if (isinstance(g, ir.Value) and g.op is not None and g.op.kind == "cmp"
                and g.op.attrs.get("pred") == "lt"
                and g.op.args[0] is phi.op.args[0]
                and _imm_of(g.op.args[1]) == bound):
            n = -(-bound // step)
            return n, ("the start is lane-varying, but the only edge into the header is guarded by "
                       "`%s < %d` - the same bound the latch tests - and the step is +%d, so at "
                       "most %d iterations run whatever the lane started at"
                       % (getattr(phi.op.args[0], "name", "start"), bound, step, n))
    return None, ("the induction starts at %s, which is not a constant, and no edge into the loop "
                  "header is guarded by a compare of it against the same bound %d - so nothing here "
                  "bounds where it starts" % (getattr(phi.op.args[0], "name", "?"), bound))


def _skip_regions():
    """Whether guarded regions get a skip branch: the function's `skip_regions`, overridden by G17_SKIP_REGIONS."""
    env = os.environ.get("G17_SKIP_REGIONS")
    if env is not None:
        return env == "1"
    return bool(getattr(_CUR_FN[0], "skip_regions", False))


def _emit_loop_entry_push(out):
    """A LOOP INSIDE A GUARDED REGION PUSHES ONE MASK LEVEL ON ENTRY (an always-true compare, 0 < 1, then op582).

    The latch is `pop, cmp, push, back edge` (_emit_cmp_for_loop) and the exit one more pop: the first trip's pop
    has nothing of the loop's own to remove. At a kernel's top level that pop is harmless, which is why every
    top-level loop runs right. Inside a guarded region it pops the REGION's level: the lanes the guard switched off
    come back on the first latch and run the rest of the region (measured: a masked-off qmv arm of a two-op program
    stored threadgroup 0's rows, its unwritten address registers reading 0, MM 25.139.9). The entry push gives the
    first pop its own level; op582 ANDs 0 < 1 with the current mask, so the level IS the region's mask, and the
    exit pop leaves the stack as the loop found it: the preheader push of tools/g17maskseq.py's measured
    loop.shape.pop.then.push, which the lowering had left out. Emitted only for loops nested in a region, so every top-level
    loop keeps its bytes."""
    z = ir.Value(ir.I32, "loop_entry_zero")
    _select_op(ir.Op("const", z, [ir.Imm(0)]), out)
    rel, imm = _canon_cmp("lt", 1)
    out.append(MInst("cmp.6", 6, dict(imm=imm, rel=rel, keep=False, source_modifier=0), uses=[z],
                     note="0 < 1: the loop-entry push's always-true predicate"))
    out.append(MInst("exec.mask", 4, {}, note="op582: the loop's own mask level, the latch's first pop removes it"))


# How many predicated (br_cond) regions enclose the ops being selected right now; 0 is uniform code.
_REGION_DEPTH = [0]


def _lower_blocks(blocks, out, exit_block, all_blocks=None):
    """Lower an ordered, structured region of blocks, recursing on nested conditionals.

    THE RECONVERGENCE STACK IS THE RECURSION. Each br_cond opens a region that ends at its join;
    the region is lowered by the same function, so a nested conditional's exec.restore is emitted
    before the enclosing one, and the enclosing branch's displacement is computed over the whole
    nested body. That ordering is not cosmetic: G17 control flow is execution-mask state, so
    reconvergence must nest exactly the way the conditionals do.

    Structured order is REQUIRED, not inferred - the then-block must immediately follow its
    br_cond, and the join must come after the whole then-region. Anything else raises, because a
    graph we cannot prove structured is one whose mask discipline we cannot emit.
    """
    i = 0
    emitted = {}          # block -> label, for back edges
    while i < len(blocks):
        blk = blocks[i]; t = blk.term
        label = "blk_%s" % blk.label
        if exit_block is not None and any(
                b_.term is not None and b_.term.kind == "br_cond" and b_.term.args[1] is blk
                for b_ in blocks[i:]):
            _emit_loop_entry_push(out)
        emitted[blk] = label
        out.append(MInst("label", 0, dict(label=label)))
        if t.kind == "br_cond" and t.args[1] in emitted:
            # LOOP LATCH: the then-target is a block already emitted, so this is a back edge and
            # the block is the loop's closing test. The false side falls through to the next
            # block, which is the loop exit.
            cond = t.args[0]
            if cond.op.kind != "cmp":
                raise Unsupported("loop latch predicate must come directly from a cmp")
            # PROVED BEFORE ANYTHING IS EMITTED. Selection rewrites the compare on its way out -
            # _canon_cmp folds >= and <= into < with an adjusted immediate - so an analysis that
            # runs afterwards is reading the lowering's arithmetic rather than the program's.
            trips, why = _prove_terminates(all_blocks or blocks, t.args[1], cond)
            _select_ops(blk.ops[:-1], out)
            _emit_cmp_for_loop(out, cond)
            # THE BLOCKER MOVED, 2026-09-04. Authored taken branches no longer fault - they were
            # truncated, and the full ten-byte form executes (ledger/g17-branch-is-ten-bytes.toml).
            # What is still missing is the BACK EDGE's semantics, and it is not a detail:
            #
            # The forward branch fires when NO LANE IS ACTIVE - it is a skip-ahead over a region
            # the exec mask has already emptied. A loop needs the opposite: repeat WHILE lanes are
            # active.
            #
            # THE THREE QUESTIONS THIS USED TO CALL UNMEASURED ARE MEASURED NOW, decode-side over
            # the corpus (ledger/g17-the-back-edge-measured-on-apples-own-loops.toml):
            #   - the same instruction does NOT do both. op458 is backward 581 of 581 times and
            #     op462 forward 1,804 of 1,804, under the ten-byte displacement decoder
            #     (g17asm.decode_branch10). The "6 backward op462 / 2 forward op458" this comment
            #     and the ledger used to carry were the 12-bit HEAD decoder truncating eight
            #     ~20 KB displacements in two ds_setup_indirect_update_mapping kernels and
            #     flipping their sign (ledger/g17-back-edge-direction-is-the-opcode.toml).
            #   - so the back form is not the forward one with the condition inverted.
            #   - and the mask: ALL 579 of Apple's backward branches are immediately preceded by an
            #     exec, op579 (490) or op578 (89). A back edge never stands alone.
            # Apple's loop exec is 0xfe.. - the `while` kind with the count code's high bit set,
            # which g17cf.encode_exec cannot currently produce (it gives 7e for while). This
            # compiler emits op582, the `if` kind, which is right for a conditional and wrong for
            # a loop.
            #
            # WHAT IS STILL MISSING IS NOT THE SEMANTICS, IT IS A SAFE WAY TO PROVE TERMINATION.
            #
            # Emitting it would be guessing at the one construct whose failure mode is
            # NON-TERMINATION. A forward branch can only move the PC forward, which is what made
            # every previous authored branch safe to dispatch; a wrong back edge runs forever and
            # wedges the GPU, which has already killed WindowServer once on this machine
            # (ledger/g17-hang-poisons-the-run.toml). So this stays refused until the back edge is
            # measured on Apple's own loops, not authored on the strength of the forward one.
            # COMPILE-ONLY OPT-IN. G17_ALLOW_LOOP=1 lets the bytes be produced so they can be
            # checked against Apple's decoder, which is a static check and cannot hang anything.
            # It is deliberately NOT the default: the refusal is about DISPATCH, and a rung that
            # silently becomes runnable is how a wrong back edge reaches the GPU.
            # THE SEMANTICS ARE MEASURED NOW, and the refusal is about something else.
            # op582 before op458 gates the branch (16 bounded dispatches), the mask stack is an
            # exact LIFO to 27 levels so a push per iteration is affordable, and counted loops of
            # 1, 3, 4, 5, 8 and 17 iterations execute and return the right value.
            # ledger/g17-a-counted-loop-executes.toml
            #
            # What is still not provable HERE is TERMINATION. This compiler does no trip-count
            # analysis: given a predicate that never goes false it will emit a kernel that runs
            # forever, and that failure mode is a GPU hang rather than a wrong answer. So the gate
            # stays, and what it now guards is the caller's obligation rather than an unknown -
            # tools/g17loop.py states the four conditions it checks before dispatching one.
            # AND THE TRIP COUNT IS PROVED HERE, not promised by the caller. What used to stand
            # in this place was G17_ALLOW_LOOP=1 - the compiler refusing every loop and a human
            # asserting the one in front of them was bounded. _prove_terminates does the analysis
            # for the two shapes it can, and the environment variable survives only as an OVERRIDE
            # for the loops it cannot: a proof where there is one, an explicit human warrant where
            # there is not, and never silence.
            if trips is None:
                if os.environ.get("G17_ALLOW_LOOP") != "1":
                    raise Unsupported(
                        "this loop's termination cannot be proved: %s. The back edge itself is "
                        "measured and executes - op582 gates op458, the mask stack is an exact "
                        "LIFO, and counted loops to 17 iterations return the right value "
                        "(ledger/g17-a-counted-loop-executes.toml) - so what is refused is not the "
                        "encoding but the hang. Set G17_ALLOW_LOOP=1 to take responsibility for "
                        "the bound yourself; tools/g17loop.py shows what it then checks" % why)
                why = "UNPROVEN, emitted under G17_ALLOW_LOOP=1: " + why
            out.append(MInst("branch.cond.back", 10, dict(target=emitted[t.args[1]]),
                             note=("at most %d iterations - %s" % (trips, why)) if trips
                                  else why))
            # THE POP APPLE ALWAYS EMITS. 581 of 581 backward branches in the corpus are
            # IMMEDIATELY followed by op577, with no exceptions - the while-exec pushes a mask
            # level on entry to the loop and this is where it comes off. Without it the mask stack
            # is unbalanced for everything after the loop, which is the same LIFO discipline the
            # forward regions already keep. This lowering emitted no pop at all.
            out.append(MInst("exec.restore", 4, dict(label="loopjoin_%s" % blk.label),
                             note="op577; 581 of 581 of Apple's back edges are followed by one"))
            i += 1
            continue
        if t.kind == "br_cond":
            _, then_b, join_b = t.args
            if i + 1 >= len(blocks) or blocks[i + 1] is not then_b:
                raise Unsupported("the then-block must immediately follow its br_cond "
                                  "(block %r)" % blk.label)
            try:
                j = blocks.index(join_b)
            except ValueError:
                raise Unsupported("join block %r is outside the region opened at %r"
                                  % (join_b.label, blk.label))
            if j <= i:
                raise Unsupported("join %r precedes its branch - not a structured region"
                                  % join_b.label)
            cond = t.args[0]
            if cond.op.kind != "cmp":
                raise Unsupported("br_cond predicate must come directly from a cmp")
            # LOWERED AS PREDICATION, NOT AS A BRANCH. `1e 00 00 0e` writes the exec mask from
            # the predicate and `3e 03 40 0e` restores it, so the then-block is MASKED OFF rather
            # than jumped over - measured directly: with the predicate false the following store
            # does not execute, with it true the store does, and after exec.restore it executes
            # again. ledger/g17-exec-mask-is-the-conditional.toml
            #
            # The branch that used to be emitted here is a separate "skip ahead when no lane is
            # active" optimisation, and every authored one that was actually TAKEN faulted the
            # GPU - at eight displacements, including targets that are valid instruction starts
            # under either PC base. It is omitted rather than emitted, which costs nothing but
            # the skip: the region is correct without it.
            label = "join_%s" % join_b.label
            _select_ops(blk.ops[:-1], out)
            _emit_cmp(out, cond)
            # THE SKIP BRANCH, opt-in (fn.skip_regions or G17_SKIP_REGIONS=1): Apple's op462 right after the
            # region's op582, taken when no lane of the simdgroup is active, to the region's exec.restore
            # (base+0; ledger/g17-branch-is-ten-bytes.toml: a taken ten-byte branch executes, and must land on
            # the restore). Without it a region no lane runs is still WALKED, instruction by instruction and one
            # masked pass per loop (MM 25.139.9). Load waits are per consumer, so a use after the join still
            # waits for a load whose first consumer was skipped.
            skip = _skip_regions()
            if skip:
                out.append(MInst("branch.cond.fwd", 10, dict(target="skip_" + label),
                                 note="op462: skip the region when no lane is active"))
            # THE WHOLE FUNCTION'S BLOCKS TRAVEL DOWN. A region is lowered from a
            # SLICE, so a loop nested inside a guarded region cannot see the guard
            # that bounds it - which is exactly the shape a triangular loop has, and
            # it made the termination proof refuse the one kernel it was written for.
            _REGION_DEPTH[0] += 1                     # selection inside a predicated region knows it is
            try:
                _lower_blocks(blocks[i + 1:j], out, join_b, all_blocks or blocks)
            finally:
                _REGION_DEPTH[0] -= 1
            if skip:
                out.append(MInst("label", 0, dict(label="skip_" + label)))
            out.append(MInst("exec.restore", 4, dict(label=label)))
            i = j
        elif t.kind == "br":
            tgt = t.args[0]
            if tgt in emitted:
                # A BACK EDGE: a structured loop. The branch is conditional in hardware, so the
                # loop's continuation test must already have set the predicate; the IR expresses
                # that by putting the cmp in the block that closes the loop.
                _select_ops(blk.ops[:-1], out)
                out.append(MInst("branch.cond.back", 10, dict(target=emitted[tgt])))
                i += 1
                continue
            nxt = blocks[i + 1] if i + 1 < len(blocks) else exit_block
            if tgt is not nxt and tgt is not exit_block:
                raise Unsupported("br from %r goes to %r, which is neither the next block nor "
                                  "the region exit; only structured fallthrough is lowered"
                                  % (blk.label, tgt.label))
            _select_ops(blk.ops[:-1], out)
            i += 1
        elif t.kind == "ret":
            _select_ops(blk.ops, out, skip_terms=False)
            i += 1
        else:
            raise Unsupported("terminator %r" % t.kind)

def _select_ops(ops, out, skip_terms=True):
    for op in ops:
        if skip_terms and op.kind in ("br", "br_cond"): continue
        _select_op(op, out)

# FORMS THAT DESTROY THEIR SOURCE, and the copy that makes them safe.
#
# Two forms read a register and leave it dead, and this project has no bit that says otherwise.
# For alu.shift.imm the byte that carries the lifetime in the add forms - byte8[5] - SELECTS THE
# OPCODE instead, and byte10[5] was tested in both polarities with no effect. bitwise.imm is a
# ten-byte form with no byte10 at all and no lifetime parameter.
#
# Measured, not assumed. Two orderings of the same two reads of one value:
#
#     v = t+109 ; a = v & 96 ; b = v << 1      b reads 0, a is correct
#     v = t+109 ; b = v << 1 ; a = v & 96      b is correct, a reads 0
#
# Whichever of the two reads first gets the value and the other gets zero, in both directions, so
# each form destroys its source rather than one form destroying the other's. A randomly generated
# chain found it; every hand-written kernel here happened to read such a value exactly once.
#
# The compiler's answer is a copy: `t = v + 0` through alu.12, whose source lifetime IS authored,
# then feed the copy to the destructive form. Costs one register and one instruction, only where a
# value is genuinely live afterwards.
# EMPTY, AND THE REASON MATTERS. These two forms did destroy their source, and the copy was the
# right response to a form with no recovered lifetime bit - but the destruction was Apple's
# TEMPLATE, not the form. The peer's dataflow refuted the hardware reading outright: Apple reads
# the source again after op423 in 1381 of 1444 instances, and the exact pattern that failed here
# occurs 806 times in the corpus. The bits are byte8[2] for the bitwise forms and byte4[2] for the
# subtract and the shifts, both found by authoring every candidate in both polarities on one
# program. With them authored the copy is unnecessary, and a copy that is unnecessary is a wrong
# answer about the machine even when the program it produces is correct.
# op10094 (the uniform atomic) RELEASES ITS VALUE: operand 9 is the value's lifetime and only 16 is
# representable in its ten-byte form, so a value read again - a second atomic sharing a constant - read 0
# on hardware (fuzz seed 5012: A[0] = 0 where the reference gives 24; MM 25.144.6). It takes a copy.
DESTRUCTIVE_FORMS = ("atomic.uniform.10",)


def copy_before_destructive(insts):
    """Insert a copy for every value a destructive form would consume while it is still live.

    G17_NO_COPY=1 disables the pass, so the destruction can be re-tested directly once something
    else changes - which is how it was shown to be an artefact of the unauthored hazard word
    rather than a property of the form.
    """
    if os.environ.get("G17_NO_COPY") == "1":
        return list(insts)
    last = {}
    for i, m in enumerate(insts):
        for v in m.uses: last[v] = i
    out = []
    for i, m in enumerate(insts):
        if m.form in DESTRUCTIVE_FORMS and m.uses and last.get(m.uses[0], -1) > i:
            src = m.uses[0]
            copy = ir.Value(getattr(src, "type", ir.I32), "%s_keep" % getattr(src, "name", "v"))
            # A MOVE, NOT AN ADD OF ZERO. This was `add src, #0` at twelve bytes because op586 was
            # not lowered; it is now the four-byte move Apple writes here, and it KEEPS its source
            # - the copy exists precisely because the original is read again.
            out.append(MInst("mov.4", 4, dict(keep_src=True),
                             defs=[copy], uses=[src],
                             note="copy: %s is read again after a destructive %s"
                                  % (getattr(src, "name", "v"), m.form)))
            m.uses[0] = copy
        out.append(m)
    return out


# TWO-ADDRESS FORMS: the destination IS operand uses[0], re-printed. op2190/4 encodes d = a*b + d and
# has no field for a separate accumulator (ledger/g17-ffma-at-four-bytes-has-two-modes.toml), so the
# allocator must give the accumulator and the destination ONE register. It could not be asked to,
# and the encoder refused every program whose allocation happened not to coincide.
TIED_FORMS = ("alu.ffma.4",)
# THE CONTROL ARM for op2190/6's load wait: True emits its loaded sources unwaited, as before
# 2026-09-23, so one run can say whether the form needs the copy.
_NO_FFMA6_LOADWAIT = False
TIED_REG_MAX = 63


def copy_before_tied(insts):
    """Give each tied form an accumulator it may overwrite.

    The tie makes the destination's write land on the accumulator's register, so the accumulator's
    value is gone afterwards. That is correct only if nothing reads it again - and "again" includes
    the NEXT ITERATION of a loop, which program order cannot see (memory: linear liveness is wrong
    for loops). So the accumulator is copied unless it provably dies here: no later reader in
    program order AND not a loop-carried value - except the one loop case the tie is for, where the
    accumulator and the destination are the same phi group and the overwrite IS the loop update."""
    last = {}
    for i, m in enumerate(insts):
        for v in m.uses: last[v] = i
    groups = [set(m.fields.get("phi_group") or ()) for m in insts if m.fields.get("phi_group")]
    out = []
    for i, m in enumerate(insts):
        tu = m.fields.get("tie_use", 0 if m.form in TIED_FORMS else None)
        if tu is not None and m.uses and m.defs:
            acc, dst = m.uses[tu], m.defs[0]
            same_group = any(acc in g and dst in g for g in groups)
            carried = any(acc in g for g in groups)
            if not same_group and (last.get(acc, -1) > i or carried):
                copy = ir.Value(getattr(acc, "type", ir.I32), "%s_acc" % getattr(acc, "name", "v"))
                out.append(MInst("mov.4", 4, dict(keep_src=True), defs=[copy], uses=[acc],
                                 note="copy: %s is the accumulator of a two-address %s and is "
                                      "still needed" % (getattr(acc, "name", "v"), m.form)))
                m.uses[tu] = copy
            m.fields["tied"] = True
        out.append(m)
    return out


_MULTI_USE = set()
_VEC_WAITED = set()          # vector loads whose first consumer has been emitted (reset per select)
_ORDER = {}
_LAST_READ = {}


# THE CAPABILITY-OFF ARM FOR LOOP-AWARE READS. True restores the linear tables, under which a value
# defined before a loop and read once in its body looks read once: the four-byte register bitwise
# (op424/op13575/op17771) then released it on the first trip and every later trip read a cleared
# register (found by tools/g17ccfuzz.py, loop seeds 99/202/302; MM 25.116's class for an invariant).
_NO_LOOP_AWARE_READS = False


def _extend_reads_over_back_edges(fn):
    """Make _MULTI_USE and _LAST_READ see the BACK EDGE, as Alloc._cfg_uses does for liveness.

    _LAST_READ is the last MENTION in block order, which is the last READ only in straight-line
    code. For each back edge - a terminator in block j naming block i <= j - the loop is blocks
    i..j, and the back edge reads (a) every value the loop body reads but does not define: it is
    read again on the next trip; and (b) the latch value of each header phi. Each such value's last
    read moves to the back edge and it becomes multi-use, so every keep decision these tables drive
    (the bitwise isolation, keep_src, keep_value/keep_index, the SR and load-wait copies) keeps it.
    """
    blocks = list(fn.blocks)
    index = {id(b): n for n, b in enumerate(blocks)}
    for j, blk in enumerate(blocks):
        term = blk.term
        if term is None:
            continue
        for tgt in term.args:
            if not isinstance(tgt, ir.Block) or id(tgt) not in index or index[id(tgt)] > j:
                continue
            body = blocks[index[id(tgt)]:j + 1]
            defined = {o.dest for b in body for o in b.ops if o.dest is not None}
            carried = set()
            for b in body:
                for o in b.ops:
                    if o.kind == "phi":
                        carried.update(a for a in o.args[1:] if isinstance(a, ir.Value))
                        continue
                    carried.update(a for a in o.args if isinstance(a, ir.Value) and a not in defined)
            at = _ORDER.get(term, -1)
            for v in carried:
                _MULTI_USE.add(v)
                _LAST_READ[v] = max(_LAST_READ.get(v, -1), at)


def _fma16_sources(op, out):
    """op798's sources, each a register: this form's load-wait field is not located, so a source straight from a load
    is first copied through the alu.12 whose wait bit IS measured (_wait_for_load), once per loaded value."""
    waited = {}
    for x in op.args:
        if not isinstance(x, ir.Value):
            raise Unsupported("fma16 takes register sources; the form has no immediate carrier")
        if _is_load_value(x) and x not in waited:
            waited[x] = _wait_for_load(out, x)
    return [waited.get(x, x) for x in op.args]


def _has_later_reader(src, op):
    """Does anything read `src` AFTER `op`? The condition every lifetime operand actually wants.

    NOT `src in _MULTI_USE`, which is a WHOLE-PROGRAM use count: a value read three times whose
    third read is this instruction is multi-use and still safe to release here. Root's review of
    35805149 made that distinction, and it is the difference between a conservative keep and a
    correct one.
    """
    return _LAST_READ.get(src, -1) > _ORDER.get(op, -1)


def _cmp_keeps(src, cmpop):
    """Does anything read `src` AFTER this compare?

    The compare's source lifetime is an operand (g17asm.OWNED_CMP_IMM byte0[3]), so this decides a
    bit rather than a refusal. Program order is block order, which is execution order for the
    forward regions this compiler emits; a latch compare is handled by its caller, which cannot use
    position because the back edge puts every reader in the body after it.
    """
    return _LAST_READ.get(src, -1) > _ORDER.get(cmpop, -1)


_IB_COORD = [None]        # one imageblock coordinate register per compiled function
_PRELOADS = [[]]          # the uniform preloads this function needs, recorded at selection (handoff 10aa)
_CUR_FN = [None]          # the function being selected, for the preload's use scan
_FETCH_DERIVED = set()    # values an alu.block produced FROM a fetch: Apple's S1 stores such a value through sub-form 01
_HAS_TG_ACCESS = [False]  # does this function contain an ordinary threadgroup load or store?
_BUF_RANK = [{}]          # {buffer slot: its rank among the buffers this function binds}
_RANK_BASE = [0]          # how many INTERNAL binding records precede the user buffers
_TENSOR_COMPOSED_ROWS = {}  # id(tensor_matmul op) -> prelowered rows for the narrow two-GEMM slice
_TENSOR_INDEX_INIT_ROWS = {}  # id(tensor_index_init op) -> its movimm row (MM 25.114.5)
# THE REGISTER ACCUMULATORS (MM 25.144.8): name -> {(mi, ni): first register of the tile's 8-register group}.
# Fixed physical registers, reserved from every tensor body and published in the occupied set, so no scalar
# value and no other body touches them. They are taken from the TOP, below the stream index registers
# (R124, R125): tlower's own scratch needs low registers (its lane read, the SR read op14060, and a K loop's
# counter take R0..R63), and it allocates lowest first. A group at R64 or above is written by the bitwise
# OR with immediate 0 (op13574, ten bytes, a seven-bit destination) - a bit-exact copy; below R64 by the
# plain move (op586, four bytes, a six-bit destination).
_TENSOR_ACC_REGS = {}
_TENSOR_ACC_HALF = set()           # the accumulators tensor_acc_fma16 writes (MM 25.196)
# THE TENSOR REGISTERS THAT HOLD SOMETHING BETWEEN BODIES (MM 25.144.8): the accumulators, the stream index
# registers, a body's D handed to the next body, and the imageblock coordinate. Everything else a body names is
# its own working set, dead once the body ends - which is what lets the allocator's last attempt (body_share)
# give those registers to scalar values that live between two bodies and across neither.
_TENSOR_PERSISTENT = set()
# the stream index registers THIS program's bodies name (a subset of TENSOR_STREAM_INDEX_REGISTERS, in its
# order): the only ones the loop's latch check may call carried. An unnamed one is an ordinary scalar register,
# and a scalar value the allocator put there was refused as a "carried register ... not a self-add" (M6's fuzz)
_TENSOR_INDEX_USED = []
# the hoisted prologues (MM 25.144.8): rows placed with the tensor_index_init rows, before the loop
_TENSOR_HOIST_ROWS = []
TENSOR_ACC_GROUPS = (112, 104, 96, 88, 80, 72, 64, 56)
# Two more groups below them (MM 25.163), handed out only when a program names more than eight accumulator tiles -
# attention's S beside its eight O tiles - so every program within eight keeps the registers it had.
TENSOR_ACC_EXTRA_GROUPS = (48, 40)
# Measured same-binding B-weight transport.  These are byte offsets inside public binding 2,
# not descriptor pointer offsets.  The Apple differential and repository-authored dispatch both
# cover this finite set; an unmeasured offset remains a refusal rather than an inferred rule.
# Same-binding B offsets measured by the focused transport campaign. This remains a finite
# allowlist; an offset outside it is refused until its own image/runtime class is measured.
TENSOR_WEIGHT_OFFSET_BYTES = (4096, 4352, 8192, 12288, 16384, 20480)
# The positional six-body campaign measured the same 16x32x64 half/half and 16x32x32
# float/half forms at every even B byte offset from 2 through 47104, including the
# displacement/base-register transition.  This is intentionally separate from the older
# two-body runtime allowlist above: the compiler may use this predicate only for the measured
# six-body stream, while the image/runtime contract remains finite until it has its own proof.
TENSOR_STREAM_OFFSET_MIN = 2
TENSOR_STREAM_OFFSET_MAX = 47104
# THE MEMORY STREAM'S B INDEX REGISTERS (P7 key blocks, machine model 25.114.3): physical registers a
# stream body adds to its B index, named by tensor_matmul(offsetB_register=...). The top of the
# allocatable file, so no released program's allocation can have used them; they are reserved from
# every body and published as occupied, which keeps scalar code off them for the whole function.
TENSOR_STREAM_INDEX_REGISTERS = (125, 124)
_TENSOR_B_ELEMENT_BYTES = {"half": 2, "bfloat": 2, "float": 4}


def _measured_tensor_stream_offset(value):
    value = int(value)
    return value == 0 or (TENSOR_STREAM_OFFSET_MIN <= value <= TENSOR_STREAM_OFFSET_MAX and
                          value % 2 == 0)


# THE COUNTED KEY-BLOCK LOOP (P7 past 16 blocks, machine model 25.114.5): a memory stream whose later
# bodies sit in ONE counted loop, so one QK -> row stage -> PV body serves every key block and the
# register-held B offsets (TENSOR_STREAM_INDEX_REGISTERS) carry the block. A looping kernel is the
# class that has hung and rebooted this machine (25.116), so the shape is the narrowest that serves:
#   * exactly one back edge, a single-block loop (the header is its own latch), not nested;
#   * a COMPILE-TIME trip count, `phi + 1 < TRIPS` from a constant start, proved by
#     _prove_terminates, at most TENSOR_LOOP_MAX_TRIPS (the compare's 8-bit immediate). The tensor
#     unit needs all 32 lanes, and a constant bound is uniform by construction and checkable from
#     the bytes; a runtime bound is refused here by name;
#   * no other conditional branch anywhere (a tensor body under a partial mask is not measured);
#   * every named B index register set by exactly one tensor_index_init before the loop, since a
#     first body's own zeroing inside the loop would reset it every trip.
# After compilation the emitted loop must also pass tensorlife.counted_loop_check (the decoded latch
# check); compile_function refuses the program otherwise.
TENSOR_LOOP_MAX_TRIPS = 255


def _tensor_op_positions(fn, ops):
    """[(block index, op index)] of each op, in the order given."""
    where = {}
    for bi, blk in enumerate(fn.blocks):
        for oi, o in enumerate(blk.ops):
            where[id(o)] = (bi, oi)
    return [where[id(o)] for o in ops]


def _ir_back_edges(fn):
    """[(header block index, latch block index)] of every IR back edge: a terminator targeting its
    own block or an earlier one."""
    index = {id(b): i for i, b in enumerate(fn.blocks)}
    out = []
    for i, blk in enumerate(fn.blocks):
        t = blk.ops[-1] if blk.ops else None
        if t is None or t.kind not in ("br", "br_cond"):
            continue
        for tgt in (t.args[1:] if t.kind == "br_cond" else t.args[:1]):
            j = index.get(id(tgt))
            if j is not None and j <= i:
                out.append((j, i))
    return out


def tensor_loop_route(fn):
    """None when no tensor body sits inside a loop; else dict(header=block index, trips=N) for the
    admitted counted key-block loop. Raises Unsupported, by name, for every other loop around a
    tensor body (the rules are above TENSOR_LOOP_MAX_TRIPS)."""
    tensor_ops = [o for blk in fn.blocks for o in blk.ops if o.kind == "tensor_matmul"]
    edges = _ir_back_edges(fn)
    pos = _tensor_op_positions(fn, tensor_ops)
    looped = [p for p in pos if any(h <= p[0] <= t for h, t in edges)]
    if not looped:
        return None
    if len(edges) != 1:
        raise Unsupported("tensor loop: %d back edges; the counted key-block loop has exactly one" % len(edges))
    (h, t), = edges
    if h != t:
        raise Unsupported("tensor loop: the loop spans blocks %d..%d; the counted key-block loop is one "
                          "block, its own latch" % (h, t))
    hdr = fn.blocks[h]
    term = hdr.ops[-1]
    if term.kind != "br_cond" or term.args[1] is not hdr or h + 1 >= len(fn.blocks) or term.args[2] is not fn.blocks[h + 1]:
        raise Unsupported("tensor loop: the latch must be br_cond(cmp, header, the next block)")
    for i, blk in enumerate(fn.blocks):
        if i == h:
            continue
        last = blk.ops[-1]
        straight = last.kind == "ret" or (last.kind == "br" and i + 1 < len(fn.blocks) and
                                          last.args[0] is fn.blocks[i + 1])
        if not straight:
            raise Unsupported("tensor loop: block %r branches conditionally or out of order; a tensor body "
                              "under a partial exec mask is not measured, so the only conditional branch is "
                              "the loop's latch" % blk.label)
    cond = term.args[0]
    if not isinstance(cond, ir.Value) or cond.op is None or cond.op.kind != "cmp":
        raise Unsupported("tensor loop: the latch predicate is not a cmp")
    rt = cond.op.attrs.get("runtime")
    if rt is not None or isinstance(cond.op.args[1], ir.Value) or cond.op.attrs.get("cap") is not None:
        return _tensor_runtime_loop(fn, h, hdr, cond)
    trips, why = _prove_terminates(fn.blocks, hdr, cond)
    if trips is None:
        raise Unsupported("tensor loop: termination is not proved: %s" % why)
    got = _counted(cond)
    if got is None or got[1] != 1 or _imm_of(got[0].op.args[0]) != 0:
        raise Unsupported("tensor loop: the counter must start at 0 and step by 1 (`i + 1 < trips`)")
    if not 1 <= trips <= TENSOR_LOOP_MAX_TRIPS:
        raise Unsupported("tensor loop: %d trips; the trip compare's immediate is 8 bits, so at most %d "
                          "(re-index on a fresh counter for more)" % (trips, TENSOR_LOOP_MAX_TRIPS))
    return dict(header=h, trips=trips)


# THE CAPPED RUNTIME KEY-BLOCK LOOP (MM 25.144.8). A causal prefill needs each query block to stop at
# its own causal edge, so its trip count is a register. The tensor unit needs all 32 lanes, so the
# bound must be the same for every lane of the SIMDGROUP: then every lane leaves the loop on the same
# trip and no tensor body runs under a partial exec mask. That is proved from the IR (below), not
# assumed. The cap is the compile-time most, as in every runtime latch (_lower_runtime_bounds): the
# emitted predicate is `nxt < n && nxt < cap`, so a wrong n truncates the loop instead of hanging it.
# The decoded check is tensorlife.counted_loop_check(runtime=True).
SIMDGROUP_UNIFORM_SR = {"threadgroup_position_in_grid", "threadgroups_per_grid", "grid_size",
                        "threads_per_threadgroup", "simdgroup_index_in_threadgroup"}


def _simdgroup_uniform(v, seen=None):
    """True when v is provably the same in every lane of a simdgroup: built only from immediates,
    constants, threadgroup-level builtins, and loads whose address is itself uniform. Conservative:
    any other builtin (thread or lane position included), any phi, any atomic or threadgroup load is
    not uniform."""
    if isinstance(v, ir.Imm):
        return True
    if not isinstance(v, ir.Value) or v.op is None:
        return False
    seen = seen if seen is not None else set()
    if id(v) in seen:
        return False
    seen.add(id(v))
    op = v.op
    if op.kind in ("const",):
        return True
    if op.kind == "builtin":
        return op.attrs.get("which") in SIMDGROUP_UNIFORM_SR
    if op.kind in ("phi", "load_tg", "atomic", "machine") or op.kind.startswith(("simd", "tensor")):
        return False
    args = [a for a in op.args if isinstance(a, (ir.Value, ir.Imm))]
    return all(_simdgroup_uniform(a, seen) for a in args)


def _tensor_runtime_loop(fn, h, hdr, cond):
    """The capped runtime latch around tensor bodies: admitted when the bound is simdgroup-uniform and
    the cap is 1..tensorlife.TENSOR_LOOP_MAX_RUNTIME_TRIPS; the counter starts at 0 and steps by 1. -> dict(header, trips=cap,
    runtime=True). Accepts the latch before and after _lower_runtime_bounds."""
    rt = cond.op.attrs.get("runtime")
    if rt is not None:
        nxt, n, cap = rt
    else:
        if cond.op.attrs.get("pred") != "lt" or cond.op.attrs.get("cap") is None:
            raise Unsupported("tensor loop: a runtime latch around tensor bodies must be cmp(i, n, 'lt', cap=K)")
        nxt, n, cap = cond.op.args[0], cond.op.args[1], cond.op.attrs.get("cap")
    from agxforge.g17 import tensorlife
    if cap is None or not 1 <= cap <= tensorlife.TENSOR_LOOP_MAX_RUNTIME_TRIPS:
        raise Unsupported("tensor loop: runtime cap %r; the tensor loop's cap is 1..%d" %
                          (cap, tensorlife.TENSOR_LOOP_MAX_RUNTIME_TRIPS))
    if not _simdgroup_uniform(n):
        raise Unsupported("tensor loop: the runtime trip count is not provably simdgroup-uniform; the tensor "
                          "unit needs all 32 lanes, so every lane must leave the loop on the same trip (build "
                          "the bound from constants, threadgroup_position_in_grid and uniform loads)")
    if not (isinstance(nxt, ir.Value) and nxt.op is not None and nxt.op.kind == "add"):
        raise Unsupported("tensor loop: the runtime latch must compare the counter's increment")
    a, b = nxt.op.args
    step, phi = _imm_of(b), a
    if step is None:
        step, phi = _imm_of(a), b
    if step != 1 or not (isinstance(phi, ir.Value) and phi.op is not None and phi.op.kind == "phi") \
            or _imm_of(phi.op.args[0]) != 0 or phi.op.args[1] is not nxt:
        raise Unsupported("tensor loop: the counter must start at 0 and step by 1 (`i + 1 < n, cap`)")
    return dict(header=h, trips=cap, runtime=True)


# An ordinary texture costs TWO ranks: internals 44 and 48, constant over twelve controlled probes
# varying access count 1..4, distinct count 1..4 and five texture types. texture_buffer adds 46 and
# would cost three - not emitted here, so not modelled. Every probe had one texture and no sampler,
# and 4,456 Apple sections carry [44, 45, 48], so this is the count for the cell it was measured in.
TEXTURE_INTERNALS = 2
# AND THEIR INDICES, which is what g17resource.layout actually wants. It ranks
# `ints + users` and drops any declared index that appears in `internal`, so passing a COUNT as
# range(n) fabricates indices 0..n-1 - and index 1 is a real user buffer in almost every kernel
# here. It was swallowed: a texture kernel declaring buffers 1 and 2 got ranks {0:0, 1:1, 2:2},
# putting its store at rank 2 where the binding list has it at 3, which writes into internal 48.
# The identity fallback below happened to be right, so this only appeared when g17resource was
# importable. These are the measured indices from the same twelve probes as the count above.
from agxforge.g17.abi import POOL_WITNESSED_MAX
TEXTURE_INTERNAL_INDICES = (44, 48)


def _uses_texture(fn):
    return any(o.kind == "texture_read" for blk in fn.blocks for o in blk.ops)


# REGISTER-DIRECT TENSOR FEED (recon sections 125, 129, 132, 136). A D tile is the fp32 A tuple
# of a float x half MMA slot for slot (A_eff[r][k] = D[r][k]), so when one tensor body's C is the
# very next body's A the accumulator registers ARE the operand: no conversion, no store, no load.
# The switch exists so the memory bridge can still be built from the same IR as a control.
TENSOR_REGISTER_FEED = True

# THE 16x32x64 SINGLE-GEMM PIN (Set A item 3). The whole-kernel route refuses this one shape because a
# delivered contract (g17regress._tensor_contract_delivered) asserts its registry refusal ("no
# witness"). The general lowering serves it, and the same body is body 1 of every released
# transformer class. False lets a verification bundle be built; the default moves only with
# hardware evidence (docs/archive/g17-tensor-generic.md).
# LIFTED: the general lowering's 16x32x64 ran bit-exact on hardware (gemm_generic run 1, Set C,
# 2026-09-23: M16N32K64unpinned, results/g17-tensor-generic-v1/run1-setc). True restores the
# old refusal, as a control.
TENSOR_PIN_16X32X64 = False


def _feed_form(at):
    """A tensor body's form as the feed table spells it: operand types, a conversion as
    `half<float`, then every modifier the body carries (+acc, +epi(...), +split, +sat, +kloop,
    +reduce, +staged:..., +strided, +seqexp)."""
    def side(t, conv):
        return at.get(t, "half") + ("<" + str(at[conv]) if at.get(conv) else "")
    form = side("a_dtype", "a_converted_from") + "." + side("b_dtype", "b_converted_from")
    mods = []
    if at.get("accumulate"):
        mods.append("acc")
    if at.get("epilogue"):
        mods.append("epi(%s)" % ",".join(str(step[0]) for step in at["epilogue"]))
    for flag, name in (("split_fp32", "split"), ("saturate", "sat"), ("kloop", "kloop"),
                       ("reduce", "reduce"), ("sequence_experiment", "seqexp")):
        if at.get(flag):
            mods.append(name)
    if at.get("a_staged"):
        mods.append("staged:" + str(at["a_staged"]))
    if any(k in at for k in ("strideA", "strideB", "strideC")):
        mods.append("strided")
    return form + "".join("+" + m for m in mods)


def _feed_key(fn, producer, consumer):
    """The feed table's key for handing `producer`'s D to `consumer`, or None when the pair is not a
    hand-off at all (not adjacent in one block, or the consumer reads nothing of the producer's C).

    FeedKey(producer, consumer, role, transpose, grid, locality):
      producer, consumer  each body's form (_feed_form)
      role        "A" or "B" (the consumer's operand is the producer's C; IR `feed` names the mode) or
                  "C" (no `feed`, the consumer accumulates onto the producer's C and reads neither
                  operand from it)
      transpose   "N", or "T" for the At/Bt feed; "+ir" when an IR transA/transB is also set
      grid        (M1, N1, K1, M2, N2, K2), then ("off", ...) naming any nonzero byte offset on the
                  handed tile (the producer's C, and the consumer's fed operand or C)
      locality    ((simdgroups, threadgroups) of the producer, the same of the consumer)"""
    block = next((b for b in fn.blocks if producer in b.ops), None)
    if block is None or consumer not in block.ops:
        return None
    ops = block.ops
    if ops.index(consumer) != ops.index(producer) + 1:
        return None                       # anything in between may read or write C
    pa, ca = producer.attrs, consumer.attrs
    if len(producer.args) < 3 or len(consumer.args) < 3:
        return None
    mode = ca.get("feed")
    handed = producer.args[2]
    if mode in (None, "A", "At") and consumer.args[0] is handed:
        role, fed = "A", "offsetA"
    elif mode in ("B", "Bt") and consumer.args[1] is handed:
        role, fed = "B", "offsetB"
    elif mode is None and consumer.args[2] is handed and ca.get("accumulate"):
        role, fed = "C", "offsetC"
    else:
        return None
    transpose = "T" if mode in ("At", "Bt") else "N"
    if any(at.get(k) for at in (pa, ca) for k in ("transA", "transB")):
        transpose += "+ir"
    grid = tuple(at.get(k) for at in (pa, ca) for k in ("M", "N", "K"))
    offsets = tuple((name, int(at.get(k, 0) or 0)) for name, at, k in
                    (("producerC", pa, "offsetC"), ("consumer" + fed[-1], ca, fed),
                     ("consumerC", ca, "offsetC")) if int(at.get(k, 0) or 0))
    if offsets:
        grid = grid + (("off",) + offsets,)
    locality = tuple((int(at.get("simdgroups", 1)), int(at.get("threadgroups", 1))) for at in (pa, ca))
    return FeedKey(_feed_form(pa), _feed_form(ca), role, transpose, grid, locality)


FeedKey = collections.namedtuple("FeedKey", "producer consumer role transpose grid locality")


def _feed_entry(disposition, receipt):
    return dict(disposition=disposition, receipt=receipt)


_ONE = ((1, 1), (1, 1))            # one simdgroup and one threadgroup on both sides
_SQ = (32, 32, 64, 32, 32, 32)     # the feed-mode arms' square stage

# THE FEED TABLE (production row P2, machine model 25.125). A register hand-off is emitted ONLY for a
# key listed here, and each entry is a pairing with a HARDWARE RECEIPT: a dispatch of this compiler's
# bytes for exactly that key, bit-exact (or output-identical to its own memory bridge), beside a
# control that failed. Every other key - another dtype, conversion, grid, role, transpose, simdgroup
# or threadgroup split, offset on the handed tile, or any modifier - is ABSENT, and an absent key
# compiles exactly as with TENSOR_REGISTER_FEED off: the memory bridge, or, for a consumer that
# declares a register-only conversion (a_converted_from / b_converted_from) or imageblock staging, that
# form's named refusal. The disposition is part of the receipt: "elide" (the producer's store dropped,
# the consumer overwriting all of C) or "keep" (D stored as well). test_g17tensorfeedtable checks the
# table, the fallback for absent keys, and the present keys' bytes.
FEED_TABLE = {
    # mode A, fp32 consumer operand
    FeedKey("half.half", "float.half", "A", "N", _SQ, _ONE): _feed_entry(
        "elide", "results/g17-register-chain-v1/chain_register (3/3, output = chain_memory's; "
                 "chain_register_swapped fails)"),
    FeedKey("half.half+epi(bias,scale,relu)", "float.half", "A", "N", _SQ, _ONE): _feed_entry(
        "elide", "results/g17-tensor-epilogue-v1/epilogue_register (3/3, output = epilogue_memory's; "
                 "epilogue_register_shifted fails)"),
    FeedKey("half.half", "float.half+acc", "A", "N", (16, 32, 64, 16, 32, 32), _ONE): _feed_entry(
        "keep", "results/g17-register-chain-v1/released transformer_layer, transformer_two_layer, "
                "transformer_layer_weight_offset (register arm output hash = memory arm's)"),
    FeedKey("float.half", "float.half+acc", "A", "N", (16, 32, 32, 16, 32, 32), _ONE): _feed_entry(
        "keep", "results/g17-register-chain-v1/released transformer_two_layer, "
                "transformer_continuation_weight_offset (register arm output hash = memory arm's)"),
    FeedKey("half.half", "float.half", "A", "N", (16, 32, 32, 16, 48, 32), _ONE): _feed_entry(
        "keep", "results/g17-tensor-generic-v2/chain_M16N32K32_4832float1648float (bit-exact)"),
    FeedKey("float.half", "float.half", "A", "N", (16, 48, 32, 16, 16, 48), _ONE): _feed_entry(
        "keep", "results/g17-tensor-generic-v2/chain_M16N32K32_4832float1648float (bit-exact)"),
    FeedKey("half.half", "float.half", "A", "N", (32, 64, 64, 32, 16, 64), _ONE): _feed_entry(
        "keep", "results/g17-tensor-generic-v2/chain_M32N64K64_1664float (bit-exact; "
                "neg_chain_claims_half fails)"),
    FeedKey("half.half", "float.half", "A", "N", (16, 32, 32, 16, 16, 32), _ONE): _feed_entry(
        "keep", "results/g17-tensor-ibfragment-v2/M16_register (bit-exact, = M16_memory's reference)"),
    FeedKey("half.half", "float.half", "A", "N", (64, 32, 32, 64, 16, 32), _ONE): _feed_entry(
        "keep", "results/g17-tensor-ibfragment-v2/M64_register (bit-exact, = M64_memory's reference)"),
    # mode A, half consumer narrowed in registers (op1016)
    FeedKey("half.half", "half<float.half", "A", "N", _SQ, _ONE): _feed_entry(
        "elide", "results/g17-tensor-generic-v3/chain_M32N32K64_3232half_fixed (bit-exact)"),
    FeedKey("half.half", "half<float.half", "A", "N", (16, 64, 64, 16, 32, 64), _ONE): _feed_entry(
        "keep", "results/g17-tensor-generic-v2/chain_M16N64K64_3264half (bit-exact)"),
    # modes At, B, Bt (the logical operand; section 132's relabeled reference fails)
    FeedKey("half.half", "half<float.half", "A", "T", _SQ, _ONE): _feed_entry(
        "elide", "results/g17-tensor-feedmodes-v1/neg_At_identity (bit-exact; neg_At_claims_A fails)"),
    FeedKey("half.half", "half.half<float", "B", "N", _SQ, _ONE): _feed_entry(
        "elide", "results/g17-tensor-feedmodes-v1/neg_B_identity (bit-exact; neg_B_claims_A fails)"),
    FeedKey("half.half", "half.half<float", "B", "T", _SQ, _ONE): _feed_entry(
        "elide", "results/g17-tensor-feedmodes-v1/neg_Bt_identity (bit-exact; neg_Bt_claims_A fails)"),
    FeedKey("half.half", "float.half", "A", "T", _SQ, _ONE): _feed_entry(
        "elide", "results/g17-tensor-feedmodes-v1/feed_At_float/mismatch-q1.npz (query 1's GPU output, "
                 "bit-exact against the identity reference offline; the relabeled reference fails)"),
    FeedKey("half.half", "half.float", "B", "N", _SQ, _ONE): _feed_entry(
        "elide", "results/g17-tensor-feedmodes-v1/feed_B_float/mismatch-q1.npz (query 1's GPU output, "
                 "bit-exact against the identity reference offline; the relabeled reference fails)"),
    FeedKey("half.half", "half.float", "B", "T", _SQ, _ONE): _feed_entry(
        "elide", "results/g17-tensor-feedmodes-v1/feed_Bt_float/mismatch-q1.npz (query 1's GPU output, "
                 "bit-exact against the identity reference offline; the relabeled reference fails)"),
    # the D fragment through the imageblock (Set A item 10b). The noread keys are the failing
    # controls' own programs, receipted as failing; a_staged="imageblock_noread" exists only as that
    # control, and without its key the control could not be built.
    FeedKey("half.half", "float.half+staged:imageblock", "A", "N", (16, 32, 32, 16, 16, 32), _ONE): _feed_entry(
        "keep", "results/g17-tensor-ibfragment-v2/M16_imageblock (bit-exact)"),
    FeedKey("half.half", "float.half+staged:imageblock", "A", "N", (64, 32, 32, 64, 16, 32), _ONE): _feed_entry(
        "keep", "results/g17-tensor-ibfragment-v2/M64_imageblock (bit-exact)"),
    FeedKey("half.half", "float.half+staged:imageblock_noread", "A", "N", (16, 32, 32, 16, 16, 32), _ONE): _feed_entry(
        "keep", "results/g17-tensor-ibfragment-v2/M16_imageblock_noread (the failing control, FAILED as predicted)"),
    FeedKey("half.half", "float.half+staged:imageblock_noread", "A", "N", (64, 32, 32, 64, 16, 32), _ONE): _feed_entry(
        "keep", "results/g17-tensor-ibfragment-v2/M64_imageblock_noread (the failing control, FAILED as predicted)"),
    # THE ACCUMULATOR (C) FEED (machine model 25.125): the consumer accumulates onto the kept D with
    # its own A and B from memory (tlower c_regs). Admitted after its preregistered hardware check.
    FeedKey("half.half", "half.half+acc", "C", "N", (32, 32, 64, 32, 32, 64), _ONE): _feed_entry(
        "elide", "results/g17-tensor-cfeed-v1/M32_register (3/3 bit-exact, output = M32_memory's; "
                 "M32_swapped and M32_claims_no_c fail 1024/1024)"),
    FeedKey("half.half", "half.half+acc", "C", "N", (16, 64, 32, 16, 64, 32), _ONE): _feed_entry(
        "elide", "results/g17-tensor-cfeed-v1/M16_register (3/3 bit-exact, output = M16_memory's; "
                 "M16_swapped fails 1024/1024)"),
    # THE INT8 FEED (production row P6, machine model 25.130): the producer's requantized int8 bytes,
    # packed four per word in acc+0/acc+1 by tlower's requant epilogue, ARE the consumer's int8 A tuple.
    # Only this grid; every other int8 pairing stays the memory bridge (cc._requant_int8_chain).
    FeedKey("int8.int8+epi(requant)", "int8.int8", "A", "N", (32, 32, 64, 32, 32, 32), _ONE): _feed_entry(
        "elide", "results/g17-tensor-requant-v1/feed_register (3/3 bit-exact, output = feed_memory's; "
                 "neg_feed_swapped fails 2926 of 4096 bytes)"),
}


def _tensor_feed_plan(fn, producer, consumer):
    """(disposition, key) when `producer` hands its D tiles to `consumer` in registers, else None.

    The key is looked up in FEED_TABLE; an absent key, or a present key whose disposition differs
    from the receipted one, is None - the memory bridge. The disposition is "elide" when the consumer
    overwrites every element of C (same buffer, same M and N, and either no accumulate or the C role,
    whose old C arrives in registers), else "keep"."""
    if not TENSOR_REGISTER_FEED:
        return None
    key = _feed_key(fn, producer, consumer)
    entry = FEED_TABLE.get(key) if key is not None else None
    if entry is None:
        return None
    pa, ca = producer.attrs, consumer.attrs
    overwritten = (consumer.args[2] is producer.args[2] and (ca.get("M"), ca.get("N")) == (pa.get("M"), pa.get("N"))
                   and (not ca.get("accumulate") or key.role == "C"))
    disposition = "elide" if overwritten else "keep"
    if disposition != entry["disposition"]:
        return None
    return disposition, key


def _tensor_register_feed(fn, producer, consumer):
    """None, or how `producer` hands its D tiles to `consumer` in registers: "keep" (store D as
    well, because something still reads it) or "elide" (the consumer overwrites every element).
    A FEED_TABLE lookup (_tensor_feed_plan); everything absent keeps the memory bridge."""
    plan = _tensor_feed_plan(fn, producer, consumer)
    return plan[0] if plan else None


def _tensor_epilogue(attrs, op):
    """The IR's register epilogue in tlower's terms: a bias buffer becomes its binding rank.

    The composition route binds its three public buffers at slots 1, 2 and 3, which are ranks
    0, 1 and 2 (the same mapping its `binds` use). A bias in any other buffer is refused."""
    steps = []
    for step in attrs.get("epilogue", ()):
        if step[0] == "bias":
            buffer = step[1]
            rank = {getattr(b, "slot", None): r for r, b in enumerate(op.args[:3])}.get(getattr(buffer, "slot", None))
            if rank is None or getattr(buffer, "slot", None) not in (1, 2, 3):
                raise Unsupported("tensor epilogue bias must live in one of the three bound tensor buffers")
            steps.append(("bias", {1: 0, 2: 1, 3: 2}[buffer.slot], int(step[2])))
        elif step[0] == "requant":
            # THE REQUANTIZATION PRIMITIVE (P6, MM 25.130): the policy's runtime vectors (scale, bias)
            # must live in one of the three bound tensor buffers, as a bias does
            policy = step[1]
            if not isinstance(policy, ir.RequantPolicy) or len(step) != 2:
                raise Unsupported("tensor epilogue requant takes one ir.RequantPolicy")

            def _rank(buffer, op=op):
                if getattr(buffer, "slot", None) not in (1, 2, 3) or buffer not in op.args[:3]:
                    raise Unsupported("a requantization vector must live in one of the three bound tensor buffers")
                return {1: 0, 2: 1, 3: 2}[buffer.slot]
            steps.append(policy.tlower_step(_rank))
        else:
            steps.append(tuple(step))
    return tuple(steps)


def _requant_int8_chain(tensor_ops):
    """True for the quantized chain of production row P6 (MM 25.130): exactly two whole-tile int8 x
    int8 bodies, one simdgroup and threadgroup, no transposes or strides. The first reads buffers
    1/2 into 3 and is requantized by its ONLY epilogue step (int8 or uint8 bytes, N per row, at C);
    the second reads those bytes as its int8 A (buffer 3, K equal to the first's N, feed A) with B
    from buffer 2 at an even byte offset, and writes int32 C. The int8 output is the consumer's
    operand; a uint8 one is not (the int8 MMA reads signed bytes), so a uint8 producer is refused
    here and the program takes the ordinary refusal."""
    if len(tensor_ops) != 2:
        return False
    p, q = (op.attrs for op in tensor_ops)
    epi = p.get("epilogue") or ()
    if (len(epi) != 1 or epi[0][0] != "requant" or not isinstance(epi[0][1], ir.RequantPolicy)
            or epi[0][1].out != "int8" or q.get("epilogue")):
        return False
    for at in (p, q):
        if ((at.get("a_dtype"), at.get("b_dtype")) != ("int8", "int8") or
                any(not isinstance(at.get(k), int) or at.get(k) % 16 for k in ("M", "N", "K")) or
                at.get("transA") or at.get("transB") or at.get("accumulate") or at.get("saturate") or
                at.get("kloop") or at.get("reduce") or at.get("sequence_experiment") or
                int(at.get("simdgroups", 1)) != 1 or int(at.get("threadgroups", 1)) != 1 or
                any(k in at for k in ("strideA", "strideB", "strideC")) or
                any(int(at.get(k, 0)) for k in ("offsetA", "offsetC"))):
            return False
    if int(p.get("offsetB", 0)) or not _measured_tensor_stream_offset(q.get("offsetB", 0)):
        return False
    if q.get("feed", "A") != "A" or q.get("M") != p.get("M") or q.get("K") != p.get("N"):
        return False
    slots = [tuple(getattr(b, "slot", None) for b in op.args[:3]) for op in tensor_ops]
    return slots == [(1, 2, 3), (3, 2, 3)]


def _adjacent_tensor_chain(fn, tensor_ops, allow_between=False):
    """True for a GEMM chain of any measured-class shapes: two or more tensor bodies with nothing
    between them, the first half x half from buffers 1/2 into 3, each later one reading the
    previous C (buffer 3) as its fp32 A with B from buffer 2, M shared and K equal to the previous
    N, every extent a multiple of 16, one simdgroup, no transposes or strides, and B offsets only
    inside the measured even transport domain. Such a chain needs no scalar allocation between
    bodies (the part the shape tables below protect), and every boundary is a register feed."""
    if len(tensor_ops) < 2:
        return False
    block = next((b for b in fn.blocks if tensor_ops[0] in b.ops), None)
    if block is None or any(op not in block.ops for op in tensor_ops):
        return False
    first = block.ops.index(tensor_ops[0])
    positions = [block.ops.index(op) for op in tensor_ops]
    # allow_between: SCALAR WORK BETWEEN BODIES (Set A fusion, item 6). The bodies stay in order in
    # one block with only non-tensor ops between them; every such boundary is the memory bridge
    # (_tensor_register_feed returns None when anything sits between), so the scalar code reads and
    # writes C in memory, exactly as the released FFN class's GELU does between its two GEMMs.
    if allow_between:
        if positions != sorted(positions):
            return False
    elif positions != list(range(first, first + len(tensor_ops))):
        return False
    if _requant_int8_chain(tensor_ops):
        return True
    M = tensor_ops[0].attrs.get("M")
    for n, op in enumerate(tensor_ops):
        at = op.attrs
        if any(not isinstance(at.get(k), int) or at.get(k) % 16 for k in ("M", "N", "K")):
            return False
        if (at.get("transA") or at.get("transB") or int(at.get("simdgroups", 1)) != 1 or
                int(at.get("threadgroups", 1)) != 1 or at.get("sequence_experiment") or
                at.get("epilogue") or any(k in at for k in ("strideA", "strideB", "strideC")) or
                any(int(at.get(k, 0)) for k in ("offsetA", "offsetC")) or
                not _measured_tensor_stream_offset(at.get("offsetB", 0))):
            return False
        if len(op.args) < 3 or any(getattr(b, "slot", None) is None for b in op.args[:3]):
            return False
        slots = tuple(b.slot for b in op.args[:3])
        types = (at.get("a_dtype", "half"), at.get("b_dtype", "half"))
        if n == 0:
            if slots != (1, 2, 3) or types != ("half", "half"):
                return False
        else:
            prev = tensor_ops[n - 1].attrs
            mode = at.get("feed", "A")
            if mode == "A":
                if (slots != (3, 2, 3) or (types != ("float", "half") and not (
                        types == ("half", "half") and at.get("a_converted_from") == "float")) or at.get("M") != M or
                        at.get("K") != prev.get("N")):
                    return False
            else:
                # B, At, Bt: two bodies only (their outputs are relabeled or shaped differently, so a
                # third body would have to know that), the fed side from C, the other from buffer 2
                fed_a = mode == "At"
                if (len(tensor_ops) != 2 or slots != ((3, 2, 3) if fed_a else (2, 3, 3)) or
                        types != (("float", "half") if fed_a else ("half", "float")) and not (
                            types == ("half", "half") and
                            at.get("a_converted_from" if fed_a else "b_converted_from") == "float")):
                    return False
    return True


def _composition_tables_admit(tensor_ops, expected, expected_types, weight_offset_route, _weight_offset,
                              _transformer_offset, _transformer_cont_offset, _transformer2_offsets):
    """The measured shape tables' verdict on a multi-body program, as a predicate: the checks the
    composition route used to make inline, returning False where it returned (refused)."""
    for index, op in enumerate(tensor_ops):
        at = op.attrs
        if index >= len(expected):
            return False            # the inline loop indexed expected[index] and would have raised
        if (((at.get("M"), at.get("N"), at.get("K")) != expected[index]) or
                ((at.get("a_dtype", "half"), at.get("b_dtype", "half")),
                 bool(at.get("accumulate"))) != expected_types[index] or
                at.get("transA", False) or at.get("transB", False) or
                int(at.get("simdgroups", 1)) != 1 or at.get("sequence_experiment", False)):
            return False
        # Explicitly avoid silently composing a strided variant before its own image/runtime
        # contract is measured.
        if any(k in at for k in ("strideA", "strideB", "strideC", "offsetA", "offsetC")):
            return False
        if at.get("offsetB", 0) and not (weight_offset_route and
                                         ((_weight_offset and index == 1) or
                                          _transformer_offset or _transformer_cont_offset or
                                          _transformer2_offsets)):
            return False
        if index == 0 and at.get("offsetB", 0) and not (_transformer_offset or
                                                         _transformer_cont_offset or
                                                         _transformer2_offsets):
            return False
        if len(op.args) < 3 or any(getattr(b, "slot", None) not in _BUF_RANK[0] for b in op.args[:3]):
            return False
    return True


def _independent_tensor_group(fn, tensor_ops):
    """True for two or more adjacent GEMMs that SHARE their input buffers and write disjoint parts of
    C: every body reads A from buffer 1 and B from buffer 2 and writes buffer 3 at its own byte
    offset, half x half, whole 16x16 tiles, one simdgroup, no transposes, strides or epilogue, with
    A offsets whole rows of A, B offsets inside the measured even transport domain, and C offsets
    fp32-aligned and pairwise disjoint. No body reads another's output, so there is no register feed and no scalar region.
    A and C offsets were outside every measured route; the preregistered bundle in
    machine model 25.102.2 is their first measurement (Set A fusion, item 6)."""
    if len(tensor_ops) < 2:
        return False
    block = next((b for b in fn.blocks if tensor_ops[0] in b.ops), None)
    if block is None or any(op not in block.ops for op in tensor_ops):
        return False
    first = block.ops.index(tensor_ops[0])
    if [block.ops.index(op) for op in tensor_ops] != list(range(first, first + len(tensor_ops))):
        return False
    spans = []
    for op in tensor_ops:
        at = op.attrs
        if any(not isinstance(at.get(k), int) or at.get(k) % 16 for k in ("M", "N", "K")):
            return False
        if (at.get("transA") or at.get("transB") or int(at.get("simdgroups", 1)) != 1 or
                int(at.get("threadgroups", 1)) != 1 or at.get("sequence_experiment") or at.get("epilogue") or
                at.get("accumulate") or at.get("a_converted_from") or at.get("split_fp32") or at.get("saturate") or
                any(k in at for k in ("strideA", "strideB", "strideC")) or int(at.get("offsetA", 0)) < 0 or
                int(at.get("offsetA", 0)) % (2 * at["K"]) or not _measured_tensor_stream_offset(at.get("offsetB", 0))):
            return False
        if len(op.args) < 3 or tuple(getattr(b, "slot", None) for b in op.args[:3]) != (1, 2, 3):
            return False
        if (at.get("a_dtype", "half"), at.get("b_dtype", "half")) != ("half", "half"):
            return False
        off_c = int(at.get("offsetC", 0))
        if off_c < 0 or off_c % 4:
            return False
        spans.append((off_c, off_c + 4 * at["M"] * at["N"]))
    spans.sort()
    if any(spans[i][1] > spans[i + 1][0] for i in range(len(spans) - 1)):
        return False
    return len({s for s in spans}) == len(spans) and any(s[0] for s in spans)


def _memory_stream_group(fn, tensor_ops, check_overlap=True):
    """True for a MEMORY-BRIDGED STREAM of tensor bodies (the online-softmax attention, goal item 6):
    two or more bodies in order in one block, scalar work allowed between them, every boundary a
    memory bridge (no register feed). Each body reads A from buffer 1 (half) or buffer 3 (fp32, a
    previous body's stored output), B from buffer 2, and writes buffer 3; A, B and C carry byte
    offsets (A offsets fp32- or whole-row-aligned, B offsets in the measured even transport domain,
    C offsets fp32-aligned); a body may accumulate into its C region. Whole 16x16 tiles, one
    simdgroup, no transposes, strides, epilogue, split, conversion or saturation.

    It is taken only where every older route refuses (a later body reading A from buffer 1, or an
    A or C offset outside the independent group, is something no released class carries), so every
    existing program keeps its route and its bytes."""
    if len(tensor_ops) < 2 and not any(op.attrs.get("acc") for op in tensor_ops):
        return False
    block = next((b for b in fn.blocks if tensor_ops[0] in b.ops), None)
    if block is None:
        return False
    if any(op not in block.ops for op in tensor_ops):
        # ACROSS IR BLOCKS only as the counted key-block loop (MM 25.114.5): the bodies in program
        # order, some inside one counted loop. tensor_loop_route refuses every other shape by name
        # when the composition is prepared; here a stream across blocks with no loop is not one.
        positions = _tensor_op_positions(fn, tensor_ops)
        if positions != sorted(positions) or not any(h <= p[0] <= t for h, t in _ir_back_edges(fn) for p in positions):
            return False
    else:
        positions = [block.ops.index(op) for op in tensor_ops]
        if positions != sorted(positions):
            return False
    novel = False
    for n, op in enumerate(tensor_ops):
        at = op.attrs
        if any(not isinstance(at.get(k), int) or at.get(k) % 16 for k in ("M", "N", "K")):
            return False
        # THE FUSED ATTENTION PATH (production row P7, machine model 25.129) adds two forms, and only
        # these: a PROJECTION body (A from buffer 1, B from buffer 2) whose only epilogue is the final
        # ("half_kv",) narrowing, so it writes K or V as halves into a buffer-3 cache region; and a
        # CONSUMER body that reads its half B from that buffer-3 region, under transB for QK (the
        # stored K is keys x head, B^T of QK's B). Everything else stays refused as before.
        half_out = tuple(tuple(e) for e in at.get("epilogue", ())) == (("half_kv",),)
        # SEVERAL SIMDGROUPS (MM 25.144.8): admitted when every body of the stream has the same count, so each
        # simdgroup runs the same bodies over its own rows and all of them read the same B
        if int(at.get("simdgroups", 1)) != int(tensor_ops[0].attrs.get("simdgroups", 1)):
            return False
        # A FROM AN ACCUMULATOR NARROWED TO HALF (MM 25.166): the one a_converted_from the stream admits
        a_acc_half = bool(at.get("a_acc")) and at.get("a_converted_from") == "float" and at.get("a_dtype", "half") == "half"
        # ... or a sixteen-bit accumulator read as the half A tuple directly (MM 25.196)
        a_acc_half = a_acc_half or (bool(at.get("a_acc")) and not at.get("a_converted_from") and at.get("a_dtype") == "half")
        if (at.get("transA") or int(at.get("simdgroups", 1)) not in (1, 2, 4) or
                int(at.get("threadgroups", 1)) != 1 or at.get("sequence_experiment") or
                (at.get("epilogue") and not half_out) or
                (at.get("a_converted_from") and not a_acc_half) or at.get("b_converted_from") or at.get("split_fp32") or
                at.get("saturate") or at.get("feed") or at.get("reduce") or at.get("a_staged") or
                at.get("kloop") or any(k in at for k in ("strideA", "strideC")) or
                # a transB B's row stride (MM 25.171, the batched decode's x rows [16][K]): only beside an A from an
                # accumulator, whose B^T rows are longer than the body's K
                ("strideB" in at and not (at.get("a_acc") and at.get("transB")))):
            return False
        if len(op.args) < 3 or any(getattr(b, "slot", None) is None for b in op.args[:3]):
            return False
        slots = tuple(b.slot for b in op.args[:3])
        types = (at.get("a_dtype", "half"), at.get("b_dtype", "half"))
        if slots not in ((1, 2, 3), (3, 2, 3), (1, 3, 3), (3, 3, 3)) or types != (
                ("half", "half") if (slots[0] == 1 or a_acc_half) else ("float", "half")):
            return False
        if (half_out and (slots != (1, 2, 3) or at.get("accumulate"))) or (
                at.get("transB") and slots != (1, 3, 3) and not (at.get("a_acc") and slots == (3, 2, 3))):
            return False
        if half_out or slots[1] == 3:
            novel = True
        off_a, off_c = int(at.get("offsetA", 0)), int(at.get("offsetC", 0))
        if (off_a < 0 or off_c < 0 or off_c % 4 or
                (off_a % (2 * at["K"]) if slots[0] == 1 else off_a % 4) or
                not _measured_tensor_stream_offset(at.get("offsetB", 0))):
            return False
        if at.get("acc") is not None:
            # a register-accumulator body (MM 25.144.8) stores nothing: its C offset would be a dropped fact
            if off_c or not at.get("accumulate") or half_out:
                return False
            novel = True
        if (n > 0 and slots[0] == 1) or off_a or off_c:
            novel = True
    if novel and check_overlap and _stream_c_overlap(tensor_ops):
        return False
    return novel


def _stream_c_overlap(tensor_ops):
    """None, or why two bodies' fp32 C regions overlap. The one admitted sharing is the measured one: a
    body that ACCUMULATES into exactly the region an earlier body wrote (the online-softmax O, 25.114).
    A partial overlap, or a second plain write over a region, was admitted by the memory-stream route
    until P1's planner found it (MM 25.124.4); no retained stream program has one."""
    # A ("half_kv",) body writes its C as HALVES (P7's K/V cache, 25.129): two bytes per element, not
    # four. Sizing it as fp32 refused P7's receipted programs, whose cache regions abut exactly.
    # Only spans in the SAME buffer can overlap; C is buffer 3 on every admitted stream body.
    def span(at):
        es = 2 if tuple(tuple(e) for e in at.get("epilogue", ())) == (("half_kv",),) else 4
        lo = int(at.get("offsetC", 0))
        return lo, lo + es * at["M"] * at["N"]

    def reads(op, lo, hi):
        at, slots = op.attrs, tuple(getattr(b, "slot", None) for b in op.args[:2])
        return ((slots[0] == 3 and lo <= int(at.get("offsetA", 0)) < hi) or
                (slots[1] == 3 and lo <= int(at.get("offsetB", 0)) < hi))

    spans = []
    for n, op in enumerate(tensor_ops):
        at = op.attrs
        if at.get("acc") is not None:
            continue                # a register accumulator writes no memory (MM 25.144.8)
        lo, hi = span(at)
        for m, (plo, phi) in enumerate(spans):
            if not (lo < phi and plo < hi):
                continue
            if (lo, hi) == (plo, phi) and at.get("accumulate"):
                continue            # the online-softmax O accumulate (25.114)
            if (lo, hi) == (plo, phi) and any(reads(tensor_ops[k], lo, hi) for k in range(m + 1, n)):
                continue            # the region is REUSED after a body in between read it: P7's per-block
                #                     score tile S (25.129, results/g17-tensor-attnfused-v1)
            return ("tensor bodies %d and %d write overlapping C regions [%d, %d) and [%d, %d); the "
                    "memory stream admits an accumulate into exactly an earlier body's region, or a rewrite "
                    "of exactly a region a body in between has read" % (m, n, plo, phi, lo, hi))
        spans.append((lo, hi))
    return None


def _plan_tensor_accumulators(fn, tensor_ops, stream):
    """Give every named register accumulator its fixed registers (MM 25.144.8), or refuse by name."""
    names = {}
    for op in tensor_ops:
        name = op.attrs.get("acc")
        if name is None:
            continue
        rows = op.attrs["M"] // 16 // int(op.attrs.get("simdgroups", 1))      # per simdgroup (the registers are)
        tiles = {(mi, ni) for mi in range(rows) for ni in range(op.attrs["N"] // 16)}
        if names.setdefault(name, tiles) != tiles:
            raise Unsupported("tensor acc=%r: every body accumulating into it must have the same M x N" % name)
    scalar = [o for blk in fn.blocks for o in blk.ops
              if o.kind in ("tensor_acc_read", "tensor_acc_write", "tensor_acc_scale", "tensor_acc_fma16")]
    # A SCALAR-WRITTEN OPERAND (MM 25.166): an accumulator that no body accumulates into but a body reads as A (a_acc)
    # is defined by the scalar code that writes it - the batched decode's dequantized weights. Its tiles are the tiles
    # those writes name, and the a_acc check below holds it to exactly the operand's.
    for op in tensor_ops:
        src = op.attrs.get("a_acc")
        if src is not None and src not in names:
            tiles = {tuple(o.attrs["tile"]) for o in scalar
                     if o.kind in ("tensor_acc_write", "tensor_acc_fma16") and o.attrs["acc"] == src}
            if tiles:
                names[src] = tiles
    if names and not stream:
        raise Unsupported("tensor acc (a register accumulator) is admitted only on the memory-stream route")
    for o in scalar:
        if o.attrs["acc"] not in names:
            raise Unsupported("%s %r: no tensor body accumulates into that register accumulator"
                              % (o.kind, o.attrs["acc"]))
        if tuple(o.attrs["tile"]) not in names[o.attrs["acc"]]:
            raise Unsupported("%s %r tile %r is outside its %d tiles" % (o.kind, o.attrs["acc"], o.attrs["tile"],
                                                                     len(names[o.attrs["acc"]])))
    # A FROM AN ACCUMULATOR (MM 25.163): a_acc names an accumulator some body fills, and the A tiles (M/16 x K/16) are
    # exactly its tiles
    for op in tensor_ops:
        src = op.attrs.get("a_acc")
        if src is None:
            continue
        want = {(mi, k) for mi in range(op.attrs["M"] // 16) for k in range(op.attrs["K"] // 16)}
        if src not in names:
            raise Unsupported("tensor a_acc=%r: no tensor body accumulates into that register accumulator" % src)
        if names[src] != want:
            raise Unsupported("tensor a_acc=%r: A is %d x %d tiles, the accumulator holds %s"
                              % (src, op.attrs["M"] // 16, op.attrs["K"] // 16, sorted(names[src])))
    # A SIXTEEN-BIT A ACCUMULATOR (MM 25.196): written only by tensor_acc_fma16 (half slot i is half i % 2 of
    # register base + i // 2) and read only as a half A with no conversion. Its registers hold halves, so an fp32
    # reader, write, read or scale of it - or a half-A reader of an fp32 accumulator, which would read two fp32
    # words' halves as eight halves - is a wrong program, refused here
    halfacc = {o.attrs["acc"] for o in scalar if o.kind == "tensor_acc_fma16"}
    for o in scalar:
        if (o.kind == "tensor_acc_fma16") != (o.attrs["acc"] in halfacc):
            raise Unsupported("%s %r: a half accumulator is written only by tensor_acc_fma16, and an fp32 one never is"
                              % (o.kind, o.attrs["acc"]))
    for op in tensor_ops:
        at = op.attrs
        if at.get("acc") in halfacc:
            raise Unsupported("tensor acc=%r: a half accumulator is an A operand, not a body's C" % at["acc"])
        src = at.get("a_acc")
        if src is not None and (src in halfacc) != (at.get("a_dtype") == "half" and not at.get("a_converted_from")):
            raise Unsupported("tensor a_acc=%r: a half accumulator is read as a_dtype \"half\" with no conversion, and "
                              "only a half accumulator is" % src)
    _TENSOR_ACC_HALF.clear(); _TENSOR_ACC_HALF.update(halfacc)
    groups = list(TENSOR_ACC_GROUPS)
    if sum(len(t) for t in names.values()) > len(groups):
        groups += list(TENSOR_ACC_EXTRA_GROUPS)
    if sum(len(t) for t in names.values()) > len(groups):
        raise Unsupported("tensor acc: %d accumulator tiles requested, %d register groups are set aside"
                          % (sum(len(t) for t in names.values()), len(groups)))
    for name in sorted(names):
        _TENSOR_ACC_REGS[name] = {tile: groups.pop(0) for tile in sorted(names[name])}


def _plan_tensor_hoists(fn, tensor_ops, loop, init_by_name, index_regs):
    """{id(body): {role: fixed register}} for the bodies whose prologue is hoisted (MM 25.144.8): bodies that
    ask (hoist_prologue) and sit in the counted loop, whose index registers are set before it. The registers
    come up from R16, below anything else the program fixes (the accumulators and index registers sit at the
    top), and the lane read's sixteen-bit destination stays low."""
    asked = [op for op in tensor_ops if op.attrs.get("hoist_prologue")]
    if not asked:
        return {}
    if loop is None or not init_by_name:
        raise Unsupported("tensor hoist_prologue: the body must sit in the counted key-block loop, whose "
                          "index registers tensor_index_init sets before it")
    edges = _ir_back_edges(fn)
    taken = set(index_regs.values()) | {r + i for regs in _TENSOR_ACC_REGS.values() for r in regs.values()
                                        for i in range(8)}
    free = [r for r in range(16, 124) if r not in taken]
    out = {}
    for op in asked:
        (bi, _oi), = _tensor_op_positions(fn, [op])
        if not any(h <= bi <= t for h, t in edges):
            raise Unsupported("tensor hoist_prologue: the body is not inside the loop")
        roles = ["lane", "idxA", "idxB0", "idxA2"] + ([] if op.attrs.get("acc") else ["idxC", "idxC2"])
        if len(free) < len(roles):
            raise Unsupported("tensor hoist_prologue: no registers left to keep the prologue in")
        out[id(op)] = {role: free.pop(0) for role in roles}
    return out


class TensorRouteRefusal(str):
    """A refusal from `tensor_route`: the string is the reason. Falsy-looking names are avoided on
    purpose; test with isinstance."""


# The rule-based multi-body routes, in the order `_prepare_tensor_composition` tries them.
TENSOR_RULE_ROUTES = ("adjacent_chain", "independent_group", "chain_with_between", "memory_stream")


def tensor_route(fn):
    """The public name of the tensor route cc would take for `fn`, or a TensorRouteRefusal.

    A THIN READ-ONLY WRAPPER over the predicates `_prepare_tensor_composition` already uses (P1's
    planner calls it; no behavior change). It evaluates them in cc's own order:
      "single_body"         one tensor body (the ordinary single-GEMM path decides the rest);
      "adjacent_chain"      `_adjacent_tensor_chain` (bodies back to back, register feed where
                            `_tensor_register_feed` admits it);
      "independent_group"   `_independent_tensor_group` (shared A/B, disjoint fp32 C regions);
      "chain_with_between"  `_adjacent_tensor_chain(allow_between=True)`: memory bridges with scalar
                            work between. cc takes it only where the measured shape tables refuse the
                            program; where they admit it the table class compiles it instead, and
                            both are measured routes;
      "memory_stream"       `_memory_stream_group` (memory-bridged stream with A/B/C offsets).
    Anything else with two or more bodies is a refusal here: only the released shape tables could
    still admit it, and they are decided inside compilation, not by a rule."""
    ops = [o for blk in fn.blocks for o in blk.ops if o.kind == "tensor_matmul"]
    if not ops:
        return TensorRouteRefusal("no tensor body")
    if len(ops) == 1 and ops[0].attrs.get("acc") is None:
        at = ops[0].attrs
        if any(int(at.get(k, 0)) for k in ("offsetA", "offsetB", "offsetC")):
            # the single-body path does not apply offsetA/B/C; compilation now refuses it by name too
            # (the selector's offset refusal, MM 25.124.4)
            return TensorRouteRefusal("a single tensor body with a nonzero offset: the single-body path "
                                      "does not apply offsetA/B/C, and cc refuses it")
        return "single_body"
    if _adjacent_tensor_chain(fn, ops):
        return "adjacent_chain"
    if _independent_tensor_group(fn, ops):
        return "independent_group"
    if _adjacent_tensor_chain(fn, ops, allow_between=True):
        return "chain_with_between"
    if _memory_stream_group(fn, ops):
        return "memory_stream"
    if _memory_stream_group(fn, ops, check_overlap=False):
        return TensorRouteRefusal(_stream_c_overlap(ops))
    return TensorRouteRefusal("no rule-based tensor route admits these %d bodies (adjacent chain, "
                              "independent group, chain with scalar work between, memory stream); only "
                              "the released shape tables could, at compile" % len(ops))


def _prepare_tensor_composition(fn):
    """Prepare the measured memory-mediated tensor composition before ordinary selection.

    The measured route supports the released two-body chain and one new three-body extension:
    `32x32x64 half/half -> 32x32x32 float/half -> 32x32x32 float/half`. Every boundary is the
    measured tensor store/load bridge. Tensor registers are reusable after each body's stores,
    while the union remains reserved from surrounding scalar allocation. The six-body transformer
    stream additionally admits the measured even B-offset transport domain (zero or 2..47104),
    including body 4 and later. Other shapes, dtypes, strides, sequence experiments and direct
    register forwarding stay refused.
    """
    global _TENSOR_COMPOSED_ROWS
    _TENSOR_COMPOSED_ROWS = {}
    _TENSOR_INDEX_INIT_ROWS.clear()
    _TENSOR_ACC_REGS.clear()
    _TENSOR_ACC_HALF.clear()
    _TENSOR_PERSISTENT.clear()
    _TENSOR_INDEX_USED.clear()
    _TENSOR_HOIST_ROWS.clear()
    tensor_ops = [o for blk in fn.blocks for o in blk.ops if o.kind == "tensor_matmul"]
    # ANY ADJACENT CHAIN skips the measured shape tables (it has no scalar region between bodies);
    # everything else keeps them exactly.
    _chain = _adjacent_tensor_chain(fn, tensor_ops)
    # INDEPENDENT GEMMS OVER SHARED BUFFERS: bodies that share A and B and write disjoint C regions.
    # Like a chain they skip the shape tables; unlike one they hand nothing on.
    _indep = not _chain and _independent_tensor_group(fn, tensor_ops)
    _chain = _chain or _indep
    # A CHAIN WITH SCALAR WORK BETWEEN BODIES, taken only when the measured shape tables would refuse
    # the program: every released class keeps its measured path and its bytes.
    _between = (not _chain and len(tensor_ops) >= 2 and
                _adjacent_tensor_chain(fn, tensor_ops, allow_between=True))
    # A MEMORY-BRIDGED STREAM with offsets (online-softmax attention), taken only where the routes
    # above refuse; like the independent group it hands nothing on in registers and skips the tables
    _stream = not _chain and not _between and _memory_stream_group(fn, tensor_ops)
    if (not _chain and not _between and not _stream and
            _memory_stream_group(fn, tensor_ops, check_overlap=False)):
        # the memory-stream shape with overlapping C regions: refused by name (MM 25.124.4)
        raise Unsupported(_stream_c_overlap(tensor_ops))
    _indep = _indep or _stream
    _chain = _chain or _stream
    _plan_tensor_accumulators(fn, tensor_ops, _stream)
    # REGISTER-HELD B OFFSETS (P7 key blocks, machine model 25.114.3) exist only on the memory-stream
    # route: anywhere else the attribute would be dropped and every body would read offset 0.
    index_names = []
    for op in tensor_ops:
        name = op.attrs.get("offsetB_register")
        if name is not None and name not in index_names:
            index_names.append(name)
    if index_names and not _stream:
        raise Unsupported("tensor offsetB_register is admitted only on the memory-stream route "
                          "(two or more bodies with offsets, scalar work between)")
    # THE HEAD GRID (MM 25.135) exists only on the memory-stream route, like the register-held B offset:
    # anywhere else the stride would be dropped and every threadgroup would read head 0.
    if any(op.attrs.get("head_stride") is not None for op in tensor_ops) and not _stream:
        raise Unsupported("tensor head_stride is admitted only on the memory-stream route "
                          "(two or more bodies with offsets, scalar work between)")
    if len(index_names) > len(TENSOR_STREAM_INDEX_REGISTERS):
        raise Unsupported("tensor stream: %d B index registers named, %d exist"
                          % (len(index_names), len(TENSOR_STREAM_INDEX_REGISTERS)))
    index_regs = dict(zip(index_names, TENSOR_STREAM_INDEX_REGISTERS))
    _TENSOR_INDEX_USED[:] = [r for r in TENSOR_STREAM_INDEX_REGISTERS if r in index_regs.values()]
    # THE COUNTED KEY-BLOCK LOOP and the explicit register starts (MM 25.114.5)
    loop = tensor_loop_route(fn)
    if loop is not None and not _stream:
        raise Unsupported("tensor loop: only the memory-stream route runs inside the counted key-block loop")
    init_ops = [o for blk in fn.blocks for o in blk.ops if o.kind == "tensor_index_init"]
    if init_ops and not _stream:
        raise Unsupported("tensor_index_init sets a memory stream's B index register; this program takes "
                          "no memory-stream route")
    init_by_name = {}
    for o in init_ops:
        name = o.attrs["register"]
        if name not in index_regs:
            raise Unsupported("tensor_index_init %r: no tensor body names that offsetB_register" % name)
        if name in init_by_name:
            raise Unsupported("tensor_index_init %r twice; a stream's index register is set once" % name)
        init_by_name[name] = o
    if init_by_name and set(init_by_name) != set(index_regs):
        raise Unsupported("tensor_index_init sets %s but the stream names %s: set every index register "
                          "or none" % (sorted(init_by_name), sorted(index_regs)))
    if loop is not None and index_regs and not init_by_name:
        raise Unsupported("tensor loop: the B index registers need tensor_index_init before the loop; the "
                          "first body's own zeroing would sit inside it and reset them every trip")
    edges = _ir_back_edges(fn)
    for name, o in init_by_name.items():
        from agxforge.g17 import epienc as g17epienc      # only here: its import reads the register model
        (ib, io), = _tensor_op_positions(fn, [o])
        if any(h <= ib <= t for h, t in edges):
            raise Unsupported("tensor_index_init %r inside a loop would reset the register every trip" % name)
        users = [op for op in tensor_ops if op.attrs.get("offsetB_register") == name]
        if any(p <= (ib, io) for p in _tensor_op_positions(fn, users)):
            raise Unsupported("tensor_index_init %r must precede every body that reads the register" % name)
        widths = {_TENSOR_B_ELEMENT_BYTES.get(op.attrs.get("b_dtype", "half")) for op in users}
        if len(widths) != 1 or None in widths or o.attrs["start"] % next(iter(widths)):
            raise Unsupported("tensor_index_init %r start %d is not a whole number of the bodies' B elements"
                              % (name, o.attrs["start"]))
        _TENSOR_INDEX_INIT_ROWS[id(o)] = [MInst(
            "tensor.wholekernel", 8,
            dict(opcode=11842, bytes=bytes(g17epienc.movimm(index_regs[name], o.attrs["start"] // next(iter(widths)))),
                 phase="tensor stream index init", _defs=[], _uses=[]),
            note="stream B index register %r = %d bytes, set once before the loop" % (name, o.attrs["start"]))]
    for op in tensor_ops:
        at = op.attrs
        if at.get("offsetB_register") is None:
            continue
        # a transposed register-offset B is admitted only as the attention class's K-cache read: B from
        # buffer 3 under transB (MM 25.114.4); the stream route admits no other transB body anyway
        # ... and as the batched decode's x rows beside an A from an accumulator (MM 25.171): B from buffer 2 under
        # transB with its row stride, the register adding a column offset
        slots_ = tuple(getattr(x, "slot", None) for x in op.args[:3])
        if at.get("transB") and slots_ != (1, 3, 3) and not (at.get("a_acc") and slots_ == (3, 2, 3) and "strideB" in at):
            raise Unsupported("tensor offsetB_register under transB is admitted only for B read from buffer 3")
        step, width = int(at.get("offsetB_step", 0)), _TENSOR_B_ELEMENT_BYTES.get(at.get("b_dtype", "half"))
        if width is None or step % width:
            raise Unsupported("tensor offsetB_step %d is not a whole number of %s elements"
                              % (step, at.get("b_dtype", "half")))
    if not _chain and not _between and len(tensor_ops) not in (2, 3, 6):
        return
    # This route has two separately measured classes.  The released chain keeps its exact
    # 32x32x64 -> 32x32x32 bodies.  The FFN slice is deliberately a smaller one-SIMDgroup
    # 16x16 tile: its second body accumulates into the GELU result, which is the residual
    # connection before the compiler-owned row normalization.  Every other shape remains on
    # the existing named refusal path.
    _ffn = (len(tensor_ops) == 2 and
            (tensor_ops[0].attrs.get("M"), tensor_ops[0].attrs.get("N"), tensor_ops[0].attrs.get("K"),
             tensor_ops[0].attrs.get("a_dtype", "half"), tensor_ops[0].attrs.get("b_dtype", "half"),
             bool(tensor_ops[0].attrs.get("accumulate"))) == (16, 16, 64, "half", "half", False) and
            (tensor_ops[1].attrs.get("M"), tensor_ops[1].attrs.get("N"), tensor_ops[1].attrs.get("K"),
             tensor_ops[1].attrs.get("a_dtype", "half"), tensor_ops[1].attrs.get("b_dtype", "half"),
             bool(tensor_ops[1].attrs.get("accumulate"))) == (16, 16, 16, "float", "half", True))
    _ffnwide = (len(tensor_ops) == 2 and
                (tensor_ops[0].attrs.get("M"), tensor_ops[0].attrs.get("N"), tensor_ops[0].attrs.get("K"),
                 tensor_ops[0].attrs.get("a_dtype", "half"), tensor_ops[0].attrs.get("b_dtype", "half"),
                 bool(tensor_ops[0].attrs.get("accumulate"))) == (16, 32, 64, "half", "half", False) and
                (tensor_ops[1].attrs.get("M"), tensor_ops[1].attrs.get("N"), tensor_ops[1].attrs.get("K"),
                 tensor_ops[1].attrs.get("a_dtype", "half"), tensor_ops[1].attrs.get("b_dtype", "half"),
                 bool(tensor_ops[1].attrs.get("accumulate"))) == (16, 32, 32, "float", "half", True))
    _weight_offset = (len(tensor_ops) == 2 and
                      (tensor_ops[0].attrs.get("M"), tensor_ops[0].attrs.get("N"), tensor_ops[0].attrs.get("K"),
                       tensor_ops[0].attrs.get("a_dtype", "half"), tensor_ops[0].attrs.get("b_dtype", "half"),
                       bool(tensor_ops[0].attrs.get("accumulate"))) == (16, 32, 64, "half", "half", False) and
                      (tensor_ops[1].attrs.get("M"), tensor_ops[1].attrs.get("N"), tensor_ops[1].attrs.get("K"),
                       tensor_ops[1].attrs.get("a_dtype", "half"), tensor_ops[1].attrs.get("b_dtype", "half"),
                       bool(tensor_ops[1].attrs.get("accumulate"))) == (16, 32, 32, "float", "half", False) and
                      int(tensor_ops[1].attrs.get("offsetB", 0)) in TENSOR_WEIGHT_OFFSET_BYTES)
    _transformer_cont = (len(tensor_ops) == 3 and
                         (tensor_ops[0].attrs.get("M"), tensor_ops[0].attrs.get("N"),
                          tensor_ops[0].attrs.get("K"), tensor_ops[0].attrs.get("a_dtype", "half"),
                          tensor_ops[0].attrs.get("b_dtype", "half"),
                          bool(tensor_ops[0].attrs.get("accumulate"))) ==
                         (16, 32, 32, "float", "half", False) and
                         all((op.attrs.get("M"), op.attrs.get("N"), op.attrs.get("K"),
                              op.attrs.get("a_dtype", "half"), op.attrs.get("b_dtype", "half"),
                              bool(op.attrs.get("accumulate"))) ==
                             (16, 32, 32, "float", "half", True)
                             for op in tensor_ops[1:]))
    _transformer = (len(tensor_ops) == 3 and
                    (tensor_ops[0].attrs.get("M"), tensor_ops[0].attrs.get("N"),
                     tensor_ops[0].attrs.get("K"), tensor_ops[0].attrs.get("a_dtype", "half"),
                     tensor_ops[0].attrs.get("b_dtype", "half"),
                     bool(tensor_ops[0].attrs.get("accumulate"))) ==
                    (16, 32, 64, "half", "half", False) and
                    all((op.attrs.get("M"), op.attrs.get("N"), op.attrs.get("K"),
                         op.attrs.get("a_dtype", "half"), op.attrs.get("b_dtype", "half"),
                         bool(op.attrs.get("accumulate"))) ==
                        (16, 32, 32, "float", "half", True)
                        for op in tensor_ops[1:]))
    _transformer2 = (len(tensor_ops) == 6 and
                     [(op.attrs.get("M"), op.attrs.get("N"), op.attrs.get("K"),
                       op.attrs.get("a_dtype", "half"), op.attrs.get("b_dtype", "half"),
                       bool(op.attrs.get("accumulate"))) for op in tensor_ops] ==
                     [(16, 32, 64, "half", "half", False),
                      (16, 32, 32, "float", "half", True),
                      (16, 32, 32, "float", "half", True),
                      (16, 32, 32, "float", "half", False),
                      (16, 32, 32, "float", "half", True),
                      (16, 32, 32, "float", "half", True)])
    _transformer2_offsets = (_transformer2 and
                            all(_measured_tensor_stream_offset(op.attrs.get("offsetB", 0))
                                for op in tensor_ops) and
                            any(int(op.attrs.get("offsetB", 0)) for op in tensor_ops))
    # The first measured three-region transformer extension uses the same public binding-2
    # allocation with three distinct, already-measured byte positions.  Keep the tuple exact:
    # this is evidence for one layer's three B regions, not a general offset mechanism.
    _transformer_offset = (len(tensor_ops) == 3 and _transformer and
                           tuple(int(op.attrs.get("offsetB", 0)) for op in tensor_ops) ==
                           (0, 4096, 8192))
    _transformer_cont_offset = (len(tensor_ops) == 3 and _transformer_cont and
                                tuple(int(op.attrs.get("offsetB", 0)) for op in tensor_ops) ==
                                (0, 4096, 8192))
    expected = ([(16, 16, 64), (16, 16, 16)] if _ffn else
                [(16, 32, 64), (16, 32, 32)] if _weight_offset else
                [(16, 32, 64), (16, 32, 32)] if _ffnwide else
                [(16, 32, 32), (16, 32, 32), (16, 32, 32)] if _transformer_cont else
                [(16, 32, 64), (16, 32, 32), (16, 32, 32)] if _transformer else
                [(16, 32, 64), (16, 32, 32), (16, 32, 32),
                 (16, 32, 32), (16, 32, 32), (16, 32, 32)] if _transformer2 else
                [(32, 32, 64), (32, 32, 32), (32, 32, 32)])
    expected_types = ([(("float", "half"), False), (("float", "half"), True),
                       (("float", "half"), True)] if _transformer_cont else
                      [(("half", "half"), False), (("float", "half"), False)] if _weight_offset else
                      [(("half", "half"), False), (("float", "half"), True)] if _ffn or _ffnwide else
                      [(("half", "half"), False), (("float", "half"), True),
                       (("float", "half"), True)] if _transformer else
                      [(("half", "half"), False), (("float", "half"), True),
                       (("float", "half"), True), (("float", "half"), False),
                       (("float", "half"), True), (("float", "half"), True)] if _transformer2 else
                      [(("half", "half"), False), (("float", "half"), False),
                       (("float", "half"), False)])
    # The two-body offset arm keeps the released 16x32x64 -> 16x32x32 shape and public binding
    # class, but moves GEMM2's B reads inside the same binding-2 allocation. The six-body arm is
    # separate: its positional campaign measured the same B forms on every body in the measured
    # stream. A/C offsets remain outside both routes.
    weight_offset = int(tensor_ops[1].attrs.get("offsetB", 0)) if len(tensor_ops) == 2 else 0
    weight_offset_route = bool(_weight_offset or _transformer_offset or _transformer_cont_offset or
                               _transformer2_offsets)
    if not _chain and any(int(op.attrs.get(key, 0)) for op in tensor_ops
           for key in ("offsetA", "offsetC")) or not _chain and any(
               int(op.attrs.get("offsetB", 0)) for index, op in enumerate(tensor_ops)
               if index != 1 and not (_transformer_offset or _transformer_cont_offset or
                                      _transformer2_offsets)):
        raise Unsupported("tensor offset route measures only the established B positions")
    if not _chain and weight_offset and not weight_offset_route:
        raise Unsupported("tensor B weight offset is outside the measured same-binding class")
    if not _chain and _weight_offset and weight_offset not in TENSOR_WEIGHT_OFFSET_BYTES:
        return
    if (_transformer_offset or _transformer_cont_offset) and tuple(int(op.attrs.get("offsetB", 0)) for op in tensor_ops) != (0, 4096, 8192):
        return
    if _transformer2 and not _transformer2_offsets and any(int(op.attrs.get("offsetB", 0))
                                                           for op in tensor_ops):
        raise Unsupported("six-body tensor B offsets require even measured positions 2..47104")
    def _tables_admit():
        return _composition_tables_admit(tensor_ops, expected, expected_types, weight_offset_route,
                                         _weight_offset, _transformer_offset, _transformer_cont_offset,
                                         _transformer2_offsets)
    if not _chain:
        if not _tables_admit():
            if not _between:
                return
            _chain = True          # the scalar-work chain: memory bridges, no shape table
    from agxforge.g17 import model as _g17model
    all_occupied = set()
    lowered_by_op = {}
    # THE FEED TABLE decides every boundary (_tensor_feed_plan). An independent group hands nothing
    # on. A memory stream consults the table only for the ACCUMULATOR (C) role: its A/B boundaries
    # stay the memory bridge they were measured as, so every existing stream keeps its bytes.
    plans = {}
    for n in range(len(tensor_ops) - 1):
        plan = None if (_indep and not _stream) else _tensor_feed_plan(fn, tensor_ops[n], tensor_ops[n + 1])
        if plan is not None and _stream and plan[1].role != "C":
            plan = None
        if tensor_ops[n].attrs.get("acc") is not None or tensor_ops[n + 1].attrs.get("acc") is not None:
            plan = None             # a register accumulator is its own hand-off (MM 25.144.8)
        plans[n] = plan
    feeds = {n: (plan[0] if plan else None) for n, plan in plans.items()}
    hoist_of = _plan_tensor_hoists(fn, tensor_ops, loop, init_by_name, index_regs)
    handed = None          # the previous body's D tile -> register map, when it feeds this one
    handed_role = None     # the role the handed tiles play in this body: "A", "B" or "C"
    staged_rows = {}       # producer id -> decoded imageblock staging instructions after its body
    for number, op in enumerate(tensor_ops):
        at = op.attrs
        # This measured common-runtime class has public buffers 1/2/3.  The tensor encoder's
        # bind values are its descriptor-base fields (0/1/2 for A/B/C, and 2/1/2 when the second
        # body reads C as A); they are not the metadata pointer offsets (0/2/4).  Keeping this
        # explicit prevents an apparently reasonable rank*2 conversion from addressing B/C past
        # the worker's public binding indices.
        mode = at.get("feed", "A") if number and not _indep else None
        slots_here = tuple(b.slot for b in op.args[:3])
        if slots_here != ((slots_here if _stream and slots_here in ((1, 2, 3), (3, 2, 3), (1, 3, 3), (3, 3, 3)) else (1, 2, 3))
                          if number == 0 or _indep else
                          (3, 2, 3) if mode in ("A", "At") else (2, 3, 3)):
            _TENSOR_COMPOSED_ROWS = {}
            return
        # descriptor bases: A-side fed modes read B from rank 1; B-side fed modes read A from rank 1;
        # a stream body reads A from its own buffer's rank (0 for buffer 1, 2 for buffer 3)
        binds = ((0 if slots_here[0] == 1 else 2, 1 if slots_here[1] == 2 else 2, 2) if _stream else
                 (0, 1, 2) if number == 0 or _indep else (2, 1, 2) if mode in ("A", "At") else (1, 2, 2))
        # read_sr has a measured low-register requirement (R0..R15). Keep that ABI scratch
        # window free from both tensor bodies. The first body's registers are reusable after its
        # stores, so they are not passed as a reservation to the second body; the union is still
        # published below and remains unavailable to scalar values for the whole selected stream.
        reserved = tuple(range(16))
        # the stream's B index registers are reserved from EVERY body, so none is overwritten between
        # the body that advances it and the next that reads it (scalar code is kept off them because
        # they are in the published occupied set)
        reserved = tuple(sorted(set(reserved) | set(index_regs.values())))
        # every register accumulator's registers are reserved from EVERY body (the one that accumulates
        # into them takes them as c_regs, which tlower requires reserved too)
        acc_all = {r + i for regs in _TENSOR_ACC_REGS.values() for r in regs.values() for i in range(8)}
        hoist_all = {r for h in hoist_of.values() for r in h.values()}
        reserved = tuple(sorted(set(reserved) | acc_all | hoist_all))
        index_kw = {}
        if index_regs and number == 0 and not init_by_name:
            index_kw["index_init"] = tuple(index_regs.values())
        if at.get("offsetB_register") is not None:
            index_kw["b_index"] = (index_regs[at["offsetB_register"]],
                                   int(at.get("offsetB_step", 0)) // _TENSOR_B_ELEMENT_BYTES[at.get("b_dtype", "half")])
        if at.get("head_stride") is not None:
            # bytes -> elements of each operand's own type (C is fp32); a stride that is not a whole
            # number of elements would split an element between heads
            widths = (_TENSOR_B_ELEMENT_BYTES.get(at.get("a_dtype", "half")),
                      _TENSOR_B_ELEMENT_BYTES.get(at.get("b_dtype", "half")), 4)
            hs = tuple(int(v) for v in at["head_stride"])
            if any(w is None or v % w for v, w in zip(hs, widths)):
                raise Unsupported("tensor head_stride %r is not a whole number of elements of %r"
                                  % (hs, (at.get("a_dtype", "half"), at.get("b_dtype", "half"), "float")))
            index_kw["head_index"] = tuple(v // w for v, w in zip(hs, widths))
            if at.get("head_slices", 1) != 1 or at.get("slice_stride") is not None:
                # THE KV SPLIT (MM 25.114.6): the head is t >> log2 S and the slice t & (S - 1)
                ss = tuple(int(v) for v in at.get("slice_stride", (0, 0, 0)))
                if any(w is None or v % w for v, w in zip(ss, widths)):
                    raise Unsupported("tensor slice_stride %r is not a whole number of elements" % (ss,))
                index_kw["head_slices"] = (int(at.get("head_slices", 1)), tuple(v // w for v, w in zip(ss, widths)))
        feed = feeds.get(number)                  # this body hands its D tiles to the next one
        a_regs = b_regs = c_regs = None
        if handed is not None and handed_role == "C":
            # the accumulator feed: C tile (mi, ni) is the kept D tile (mi, ni), added after the chain
            c_regs = dict(handed)
            reserved = tuple(sorted(set(reserved) | {reg + i for reg in handed.values() for i in range(8)}))
        elif handed is not None:
            # the fed tile map, part 3: A (mi,k)=D(mi,k); At (a,k)=D(k,a); B (k,n)=D(k,n); Bt (k,i)=D(i,k)
            if mode == "At":
                a_regs = {(a, k): reg for (k, a), reg in handed.items()}
            elif mode == "B":
                b_regs = {(k, n): reg for (k, n), reg in handed.items()}
            elif mode == "Bt":
                b_regs = {(k, i): reg for (i, k), reg in handed.items()}
            else:
                a_regs = {(mi, k): reg for (mi, k), reg in handed.items()}
            reserved = tuple(sorted(set(reserved) | {reg + i for reg in handed.values() for i in range(8)}))
        acc_kw = {}
        if at.get("acc") is not None:
            # D = A @ B + the kept registers, written back into them (tlower's c_inplace); nothing stored
            if c_regs is not None or a_regs is not None or b_regs is not None or feed is not None:
                raise Unsupported("tensor acc=%r: a register-accumulator body takes no other register feed"
                                  % at["acc"])
            c_regs = dict(_TENSOR_ACC_REGS[at["acc"]])
            acc_kw = dict(c_inplace=True, store=False)
        if at.get("a_acc") is not None:
            # A is the named accumulator's registers (MM 25.163): tile (mi, k) of A is its tile (mi, k)
            if a_regs is not None or b_regs is not None or feed is not None or at.get("acc") is None:
                raise Unsupported("tensor a_acc=%r: A from an accumulator takes no other register feed, and its "
                                  "C is a register accumulator" % at["a_acc"])
            a_regs = dict(_TENSOR_ACC_REGS[at["a_acc"]])
            if at["a_acc"] in _TENSOR_ACC_HALF:
                # a half accumulator's tuple is its group's upper four registers (tensor_acc_fma16)
                a_regs = {key: r + 4 for key, r in a_regs.items()}
            # a later body reading the same accumulator as A (MM 25.178: the batch halves over one dequantized W): this
            # body must not release its registers; the last reader does
            if any(o2.attrs.get("a_acc") == at["a_acc"] for o2 in tensor_ops[number + 1:]):
                acc_kw = dict(acc_kw, a_keep=True)
        a_type = at.get("a_dtype", "half")
        b_type = at.get("b_dtype", "half")
        if at.get("reduce"):
            # the composition route does not pass a reduction; dropping it silently would store D
            raise Unsupported("tensor matmul reduce=%r is implemented for a single GEMM, not inside a "
                              "composed chain" % (at.get("reduce"),))
        if at.get("a_converted_from") and a_regs is None:
            raise Unsupported("tensor matmul with a_converted_from=%r needs its A handed over in registers "
                              "from the adjacent producer: memory holds the float, not the narrowed operand"
                              % at.get("a_converted_from"))
        if at.get("b_converted_from") and b_regs is None:
            # the B-side twin (P2): without it an absent B-half key loaded C's fp32 words as halves
            raise Unsupported("tensor matmul with b_converted_from=%r needs its B handed over in registers "
                              "from the adjacent producer: memory holds the float, not the narrowed operand"
                              % at.get("b_converted_from"))
        try:
            low = g17tensor.emit_gemm(
                at["M"], at["N"], at["K"],
                # a stream consumer's transB (P7's QK over the keys x head K cache) stores B^T, N x K
                lda=at["K"],
                ldb=((at["strideB"] // TENSOR_ELEMENT_BYTES[at.get("b_dtype", "half")]) if (_stream and at.get("transB") and
                                                                                          "strideB" in at)
                     else at["K"] if (_stream and at.get("transB")) else at["N"]), ldc=at["N"],
                a_type=a_type, b_type=b_type, accumulate=bool(at.get("accumulate")),
                transA=mode == "At", transB=mode == "Bt" or bool(_stream and at.get("transB")),
                simdgroups=int(at.get("simdgroups", 1)) if _stream else 1, registers=126,
                reserved=reserved, binds=binds,
                offsets=((int(at.get("offsetA", 0)) if _indep else 0), int(at.get("offsetB", 0)),
                         int(at.get("offsetC", 0)) if _indep else 0), end=False,
                keep=feed is not None, **({} if acc_kw else dict(store=feed != "elide")), a_regs=a_regs,
                a_convert=("half" if a_regs is not None and at.get("a_converted_from") == "float" else None),
                b_regs=b_regs,
                b_convert=("half" if b_regs is not None and at.get("b_converted_from") == "float" else None),
                grid_n=int(at.get("grid_n", 1)), split_k=int(at.get("split_k", 1)), kloop_unroll=int(at.get("kloop_unroll", 1)),
                **({} if at.get("kloop_chunk") is None else dict(kloop_chunk=at["kloop_chunk"])),
                epilogue=_tensor_epilogue(at, op), **({} if c_regs is None else dict(c_regs=c_regs)),
                **index_kw, **acc_kw, **({} if not at.get("fold_offsets") else dict(fold_offsets=True)),
                **({} if id(op) not in hoist_of else dict(hoist=hoist_of[id(op)])))
        except ValueError as why:
            if str(why).startswith("refused: "):
                _TENSOR_COMPOSED_ROWS = {}
                return
            raise
        decoded = [i for i in _g17model.decode(bytes(low.body), 0) if i.opcode]
        if not decoded or any(i.opcode.id == 684 for i in decoded):
            raise Unsupported("tensor composition: a lowered body unexpectedly contains END")
        names = _g17model.registers()
        occupied = set()
        for inst in decoded:
            for kind, value in inst.values:
                if kind == "reg":
                    occupied.update(registerdomain.registers_in_name_checked(names.get(value, "")))
        # The three-body route also reserves scratch registers consumed by encoded operands whose
        # decoder exposes them only as register-class immediates. Its plan is the authoritative
        # second source for those physicals (the three mask registers are the measured example).
        # Reserve the measured encoded scratch registers as well as decoder-visible operands. The
        # lowering's tuple fields are not all exposed as ordinary scalar operands, but the current
        # three-body scalar route has a measured allocation that reuses its constants; reserving
        # the whole physical span would push those scalar values into an unmeasured high-register
        # class. Keep this reservation narrow and refuse/widen only with a new measurement.
        if len(tensor_ops) >= 3:
            for value in (low.plan.get("scratch", {}) or {}).values():
                values = value if isinstance(value, (list, tuple)) else (value,)
                for register in values:
                    if isinstance(register, int):
                        occupied.add(register)
        if not occupied:
            raise Unsupported("tensor composition: lowered body has no decoded register operands")
        all_occupied.update(occupied)
        all_occupied.update(index_regs.values())
        all_occupied.update(acc_all)
        _TENSOR_PERSISTENT.update(index_regs.values())
        _TENSOR_PERSISTENT.update(acc_all)
        _TENSOR_PERSISTENT.update(hoist_all)
        all_occupied.update(hoist_all)
        if low.plan.get("prologue"):
            for inst in _g17model.decode(bytes(low.plan["prologue"]), 0):
                if not inst.opcode:
                    continue
                fields = dict(opcode=inst.opcode.id, bytes=bytes(inst.raw), phase="hoisted tensor prologue",
                              _defs=[], _uses=[])
                if inst.opcode.id in (14059, 14060):
                    fields["sr"] = g17asm.decode_sr(bytes(inst.raw))["sr"]
                _TENSOR_HOIST_ROWS.append(MInst("tensor.wholekernel", len(inst.raw), fields,
                                                note="hoisted prologue of tensor body %d" % (number + 1)))
        lowered_by_op[id(op)] = decoded
        # A BODY'S UPWARD-EXPOSED REGISTERS STAY RESERVED (MM 25.144.8): tlower's QK body writes R27's low half
        # (a 16-bit system-register read) and then reads all of R27, so R27's high half enters the body from
        # outside. With nothing else ever writing it that was harmless; a scalar value sharing it between bodies
        # would feed the next trip's body its own bits (tensorlife.loop_carried_releases refused exactly that).
        from agxforge.g17 import tensorview as _tv
        _bv, _seen = _tv.view(b"".join(bytes(i.raw) for i in decoded)), set()
        for _x in _bv:
            _TENSOR_PERSISTENT.update(h // 2 for h in (_x.uses - _seen))
            _seen |= _x.defs
        handed = low.plan["acc"] if feed is not None else None
        if handed:
            _TENSOR_PERSISTENT.update(reg + i for reg in handed.values() for i in range(8))
        handed_role = plans[number][1].role if feed is not None else None
        # THE D FRAGMENT THROUGH THE IMAGEBLOCK (Set A item 10b): when the consumer asks for it, the
        # handed registers are written to the lane's imageblock element, clobbered, and read back
        # (agxforge.g17.ibstage). The coordinate lives in R2, inside the R0..R15 window every tensor body
        # leaves free (read_sr writes only there); no scalar value spans the chain's bodies.
        nxt = tensor_ops[number + 1] if number + 1 < len(tensor_ops) else None
        if nxt is not None and nxt.attrs.get("a_staged"):
            if feed is None:
                raise Unsupported("a_staged needs the adjacent register feed: the producer must hand its D tiles over")
            from agxforge.g17 import ibstage
            stage_code, _layout = ibstage.emit(handed, 2, read=nxt.attrs["a_staged"] == "imageblock")
            staged_rows[id(op)] = [i for i in _g17model.decode(stage_code, 0) if i.opcode]
            all_occupied.update({2})
            _TENSOR_PERSISTENT.add(2)
    for number, op in enumerate(tensor_ops):
        rows = []
        for index, inst in enumerate(lowered_by_op[id(op)]):
            fields = dict(opcode=inst.opcode.id, bytes=bytes(inst.raw),
                          phase="composed tensor lowering", _defs=[], _uses=[])
            if number == 0 and index == 0:
                fields["_occupies"] = sorted(all_occupied)
            if inst.opcode.id in (14059, 14060):
                fields["sr"] = g17asm.decode_sr(bytes(inst.raw))["sr"]
            rows.append(MInst("tensor.wholekernel", len(inst.raw), fields,
                              note="composed tensor body %d/%d" % (number + 1, len(tensor_ops))))
        for inst in staged_rows.get(id(op), ()):
            fields = dict(opcode=inst.opcode.id, bytes=bytes(inst.raw),
                          phase="imageblock fragment staging", _defs=[], _uses=[])
            if inst.opcode.id in (14059, 14060):
                fields["sr"] = g17asm.decode_sr(bytes(inst.raw))["sr"]
            if inst.opcode.id in (13075, 12151):
                fields["ib_member"] = [v for _k, v in inst.values][3]
            rows.append(MInst("tensor.wholekernel", len(inst.raw), fields,
                              note="D fragment staged through the imageblock after body %d" % (number + 1)))
        _TENSOR_COMPOSED_ROWS[id(op)] = rows

# ROOT'S MEASURED DOMAIN FOR A DEVICE LOAD IN A TEXTURE KERNEL, and the boundary of it.
#
# The load's base was already the RANK here (measured from Apple's indexed pair at public indices
# [1,2,3] and [2,4,6]); what was NOT measured was whether that still holds once INTERNAL texture
# bindings shift the ranks, and a load resolving to internal 44 reads descriptor state. Root settled
# it with four Apple-compiled sources, decoder output and objects retained at
# results/g17-source-admission-v1/texture-indexed-load-base-measurement.json (no dispatch):
#
#     textures  user buffer   Apple's indexed op12682/8 AND /14 base
#     0         index 0       expr:bin(op0,const(0),8)
#     1         index 0       expr:bin(op0,const(8),8)
#     2         index 0       expr:bin(op0,const(8),8)
#     2         index 7       expr:bin(op0,const(8),8)
#
# const(8) = 4 * 2, and TEXTURE_INTERNALS is 2 for ONE texture as much as for two - so the existing
# rank calculation reproduces all four without a new public-index table, which is what root asked
# for. Note the shape of the evidence: the base does not depend on the public index (0 and 7 agree)
# and does not scale with the texture COUNT (1 and 2 agree). Both are facts, not conveniences.
#
# WHAT STAYS REFUSED, because nothing above measures it:
#   * a SECOND user buffer. All four sources bind one, so rank 3 -> const(12) is unmeasured, and a
#     wrong base here reads another allocation.
#   * a load that is not a WORD load. The sources are `device uint *`; the half and narrow forms
#     have their own base fields and none of them appears above.
#   * a CONSTANT-index load. Root's initial constant-index volatile controls selected a different
#     opcode (op12688) and root retains them separately, "not conflated with" the indexed forms.
def _texture_load_domain(fn):
    """-> None if every device load in this texture kernel is inside the measured base domain,
    else a sentence naming the first thing that is outside it."""
    users = sorted({b.slot for b in fn.buffers})
    if len(users) != 1:
        return ("it binds %d user buffers (%s) and the measurement covers exactly one, so the rank "
                "of a second one with internal bindings ahead of it is unmeasured" % (len(users), users))
    loads = [o for blk in fn.blocks for o in blk.ops if o.kind == "load"]
    for o in loads:
        if o.attrs.get("width", "word") != "word":
            return ("a %r load: the measurement is of word loads (`device uint *`), and the narrow "
                    "forms carry their own base field" % o.attrs.get("width"))
        if len(o.args) < 2 or isinstance(o.args[1], ir.Imm):
            return ("a constant-index load: root's constant-index controls selected op12688 and are "
                    "retained separately from the indexed op12682 forms measured above")
    return None



def _resource_layout(layout, ranks, abi_bindings):
    """ABI v6's resources block for a program whose layout carries texture forms; None otherwise."""
    reads = [(m.fields.get("tex"), m.form, m.fields.get("element", "uint32"))
             for _, _, m in layout if m.form.startswith("texture.read")]
    if not reads:
        return None
    for tex, form, _el in reads:
        if form != "texture.read.32":
            raise Unsupported("texture form %s has no element width this ABI states; only the 32-bit read is declared" % form)
    # THE ELEMENT COMES FROM THE DECLARATION AND NOTHING ELSE. It used to be filled in
    # unconditionally as uint32, which made g17ir.texture_read's `type` a parameter the backend
    # accepted and dropped: a float32 read compiled to the same bytes AND the same declaration.
    # The bytes being the same is correct and measured - the linker's three Apple controls show
    # texture2d<uint> and a bitcast texture2d<float> producing identical program and metadata bytes,
    # while a CONVERTING read differs by an explicit instruction. The declaration being the same was
    # the defect: a format that is not in the image can only come from the source saying so.
    _ELEMENTS = ("uint32", "float32")
    for _tex, _form, el in reads:
        if el not in _ELEMENTS:
            raise Unsupported("texture element %r: this ABI states %s. An unknown element is "
                              "refused rather than defaulted, because a manifest that declares one "
                              "format while the image holds another is the defect a declaration "
                              "exists to prevent" % (el, " or ".join(_ELEMENTS)))
    # A DENSE INDEX WITH TWO ELEMENTS IS REFUSED BY NAME. One texture cannot be both, and choosing
    # either would be a guess; the program has simply said two incompatible things about one object.
    _by_index = {}
    for tex, _form, el in reads:
        _by_index.setdefault(tex, set()).add(el)
    _mixed = sorted(t for t, els in _by_index.items() if len(els) > 1)
    if _mixed:
        raise Unsupported("texture %d is read as %s in the same program: one texture has one "
                          "element, and this side will not choose between two declarations"
                          % (_mixed[0], " and ".join(sorted(_by_index[_mixed[0]]))))
    # the dense indices the layout NAMES, as named: a program reading texture 1 alone (the op7
    # selection case) states [1], and the contract's validator asks for uniqueness, not 0..n-1
    dense = sorted({t for t, _f, _e in reads})
    internal = []
    for i, idx in enumerate(TEXTURE_INTERNAL_INDICES):
        if ranks.get(idx) != i:
            raise Unsupported("internal record %d is at rank %s, not %d: the binding offsets and the resource block would disagree" % (idx, ranks.get(idx), i))
        internal.append(dict(rank=i, apple_index=idx, kind="texture_internal"))
    # A USER BINDING IS READ when a device load names its base, and the load's base is 4 * rank
    # (load.14 selection: base=4 * _buf_rank(slot)) - not 2 * rank, which is the ABI's descriptor
    # OFFSET and a different unit (integration's 6b6c9dda caught the first cut mixing them).
    # THIS IS NOW A FACT ABOUT EMITTED LOADS RATHER THAN A PREDICATE THAT CANNOT FIRE. It used to be
    # checked below only to AGREE with select's blanket refusal of a device load in a texture
    # kernel; select now admits that combination inside root's measured base domain
    # (_texture_load_domain), so a `read` here describes real coordinate loads. The refusal below is
    # narrowed to the same domain rather than deleted: a read fact from a load whose base is NOT the
    # single user record's 4 * rank is still a defect, because that is the only base measured.
    loaded = {m.fields.get("base") for _, _, m in layout if m.form.startswith("load")}
    # a record's accesses are UNIFORM when every one is a constant-slot store (store.8 / store.14,
    # the fetch consumer included); an indexed store (store.idx, op17229) or an indexed load is
    # lane-addressed and makes the record divergent. Recorded per rank constant (4 * rank).
    # A LOAD'S RANK CONSTANT IS `base`, NOT `const`, AND ITS INDEX IS NOT A FIELD AT ALL. The load
    # arm of this expression read `m.fields["const"]` (a load has no such field, so every load
    # contributed None) and was gated on `m.fields["index_reg"] is not None` (the load forms do not
    # carry that field either - encode_load takes the index from m.uses[0] at emit time, so the test
    # was always False). Two mistakes that cancelled into silence: the arm could not fire, and
    # nothing noticed because select refused a device load in a texture kernel outright, so no
    # program with a load ever reached here. Narrowing that refusal to root's measured domain ran it
    # for the first time, and the first coordinate program declared its lane-addressed coordinate
    # read UNIFORM.
    #
    # EVERY LOAD THIS BACKEND EMITS IS INDEX-REGISTER ADDRESSED - op12682 takes its address from a
    # base plus an index register, which is not the "constant-slot store" this field is documented
    # to mean - so a record with a load in it is divergent, by the form and not by the value in the
    # register. Recorded at the load's rank constant, which is `base`.
    divergent = {m.fields.get("const") if m.form in ("store.idx",) else m.fields.get("base")
                 for _, _, m in layout
                 if m.form in ("store.idx",) or m.form.startswith("load")}
    access = [dict(record=idx, kind="internal", written=False, read=None, uniform=None) for idx in TEXTURE_INTERNAL_INDICES]
    for slot, off, written, _t, _b in abi_bindings:
        access.append(dict(record=slot, kind="user", written=bool(written), read=4 * ranks[slot] in loaded, uniform=4 * ranks[slot] not in divergent))
    # the publishes in address units: publish.coord.x -> [op4 + 0], publish.coord.y -> [op4 + 8],
    # four bytes each (g17cc's texture lowering: "the coordinate slots are [op4+0*4] and [op4+2*4]")
    # the publish CONSTANTS as encoded - x at const 0, y at const 2 (the measured pair) - with the byte
    # conversion NOT stated: whether const 2 is byte 2 (16-bit components, one 4-byte pair) or byte 8
    # (4-byte units, two slots) is unmeasured and decides slot 38 for this program (handoff 10w). The
    # earlier target_offset_bytes = 8 for y was the compiler comment's "[op4 + 2 * 4]", not a measurement.
    pubs = sorted({(m.form, {"publish.coord.x": 0, "publish.coord.y": 2}[m.form]) for _, _, m in layout if m.form in ("publish.coord.x", "publish.coord.y")}, key=lambda t: t[1])
    if not pubs:
        raise Unsupported("a texture read with no coordinate published")
    publications = [dict(form=f, target_constant=c, operand_code=4, byte_offset_basis="unmeasured: byte 2 if coordinate components are 16-bit, byte 8 if the constant is in 4-byte units") for f, c in pubs]
    # THE READ FACT NOW DESCRIBES REAL LOADS, AND THERE IS NO SECOND GUARD HERE. `select` used to
    # refuse a device load in any texture kernel, so `read` could only ever be False and the old
    # refusal below existed to say "a read fact here is a defect, not a fact". select now admits
    # that combination inside root's measured base domain (_texture_load_domain), so `read` is a
    # fact - and the three guards tried in this spot were each unreachable, which is worth writing
    # down rather than shipping as evidence:
    #
    #   * "is the read uniform?" - a record holding a load is divergent by the form (op12682 is
    #     base-plus-index addressed and not a constant-slot store), so the answer is always no.
    #   * "does it read more than one user binding?" - select refuses a second user buffer first.
    #   * "is any load's base outside 4 * the record's rank?" - `read` is DEFINED as
    #     `4 * ranks[slot] in loaded`, so a load from a stray base makes the record read=False and
    #     skips the question entirely. The guard can only run once the base already matches.
    #
    # THAT LAST ONE IS A REAL LIMITATION AND NOT A CLOSED QUESTION: a load whose base is neither a
    # user record's 4 * rank nor an internal's would be recorded here as "not read" rather than
    # refused. What actually stands between that and an image is select - _texture_load_domain for
    # the shapes measured, and the "ranked slots ... an internal index has collided with a user one"
    # check for the rank list itself. A guard in this function would need a base computed
    # independently of `ranks` to have anything to compare, and there is no second computation.
    return dict(internal=internal,
                textures=[dict(dense_index=t, access="read", dimension="2d",
                               element=_by_index[t].copy().pop(),
                               coordinates="publish.coord.x/y", rank=None) for t in dense],
                samplers=(), spill_bytes=0, spill_basis="no_spill_form", access=access, coordinate_publications=publications,
                not_stated=("slot27_contents", "slot2_resource_record", "slot2_kind9_record"))


def _argument_bytes(access, pool_bytes):
    """THE ARGUMENT BUFFER'S BYTE SIZE (per-kernel slot 1), stated inside the witnessed class (handoff 10af;
    results/g17-texture-argument-bytes-v1): 8 + 4 * ceil(u / 2) + 4 * (pool // 64), where u counts the user
    records that are UNIFORMLY READ and not written. Measured over 31 Apple members in four preregistered
    families: 29 below the class boundary agree, the first rounding was refuted by a held-out member (H2) and
    the amendment was confirmed on four more (C1-C4). Above a 288-byte pool slot 1 COLLAPSES to the baseline
    (H3 at 768, P5 at 1024) and the boundary is not bracketed, so this returns None there and the caller names
    the field in not_stated rather than extrapolating."""
    if pool_bytes > POOL_WITNESSED_MAX:
        return None
    u = sum(1 for a in access if a["kind"] == "user" and not a["written"] and a["read"] is False and a["uniform"])
    return 8 + 4 * -(-u // 2) + 4 * (pool_bytes // 64)

_PHI_MEMBERS = set()      # every phi, its entry value and its latch value - one register each


def select(fn):
    _CONSUMED.clear()
    _IB_COORD[0] = None
    _PRELOADS[0] = []; _CUR_FN[0] = fn; _FETCH_DERIVED.clear(); _fold_uniform_chains(fn)
    _lower_runtime_bounds(fn)
    _PHI_MEMBERS.clear(); _VEC_WAITED.clear()
    for blk in fn.blocks:
        for o in blk.ops:
            if o.kind == "phi":
                _PHI_MEMBERS.add(o.dest)
                _PHI_MEMBERS.update(a for a in o.args if isinstance(a, ir.Value))
    # THE THREADGROUP REGION IS ALLOCATED BY THE LOAD/STORE PATH, NOT BY THE ATOMIC. Measured by
    # removing one threadgroup store from a working kernel and putting it back: without it the
    # threadgroup atomic writes nothing and every lane reads the same value, with it the counter
    # moves. So an atomic is not on its own enough to get a region to be atomic ON.
    _HAS_TG_ACCESS[0] = any(o.kind in ("store_tg", "load_tg")
                            for blk in fn.blocks for o in blk.ops)
    # THE BUFFER CONST IS 4 x THE RANK, and the rank is a property of the FUNCTION, not of the
    # slot number - a kernel binding only C makes C rank 0 however high its slot is. Computed once
    # here so every addressing path reads the same answer.
    # INTERNAL BINDINGS COME FIRST, AND THEY SHIFT EVERY USER RANK. A texture section carries
    # internal binding records the compiler never asked for - [44, 48] for an ordinary texture,
    # measured constant across access count, distinct count and five types - and the rank law puts
    # internals first ascending. So in a texture kernel the first USER buffer is at rank 2, not 0.
    #
    # This is the cross-layer edge the resource ABI was written for, and it has teeth: the machine
    # code INDEXES the binding list, so a resource the linker adds for its own reasons renumbers
    # the resources the compiler already bound. Emitting rank 0 and 1 here against a correct
    # texture section would make the store write into internal 48 - descriptor state - which is
    # worse than the hang that started this.
    # ledger/g17-the-resource-abi-between-the-two-halves.toml
    _RANK_BASE[0] = TEXTURE_INTERNALS if _uses_texture(fn) else 0
    # ONE AUTHORITY FOR THE RANK, when the resource layout is available. tools/g17resource.py
    # (the linker's, on codex/native-scan-integration) reproduces Apple's recorded binding list
    # exactly - same members, SAME ORDER - in 10,964 of 12,044 corpus kernels, and its
    # "same set, different order" bucket is EMPTY: internals first ascending, then users ascending,
    # is right every time the membership is right. The whole residual is WHICH buffers the image
    # declares, which is this side's fact and is why `internal` is a parameter rather than a
    # heuristic inside it.
    #
    # It also REFUSES a store to a buffer the list does not contain - which is the worst failure
    # this compiler has, a store ranking into a list the image never declared, returning the fill
    # value at status 0 and looking exactly like a broken opcode. Caught at compile time with the
    # list printed instead of on the GPU with nothing written.
    #
    # Falling back to the local computation when the module is absent keeps main buildable while
    # that branch is unmerged; the two agree on every kernel this project compiles today.
    ranks = None
    try:
        from agxforge.g17 import resource as g17resource
        ranks = g17resource.from_declaration(
            [b.slot for b in fn.buffers],
            internal=TEXTURE_INTERNAL_INDICES[:_RANK_BASE[0]] if _RANK_BASE[0] else ())[1]
    except ImportError:
        ranks = None
    except Exception as ex:
        raise Unsupported("the resource layout will not rank this kernel's buffers: %s" % ex)
    # THE ANSWER MUST COVER THIS FUNCTION'S BUFFERS AND NOTHING ELSE. A rank list containing a
    # slot the function never bound means an internal index collided with a user one, and the
    # store then ranks into a binding that belongs to the driver - it RUNS, and corrupts.
    if ranks is not None and set(ranks) - {b.slot for b in fn.buffers} - set(
            TEXTURE_INTERNAL_INDICES[:_RANK_BASE[0]]):
        raise Unsupported(
            "the resource layout ranked slots %s for a function binding %s; an internal index has "
            "collided with a user one and a store would rank into the driver's binding"
            % (sorted(ranks), sorted(b.slot for b in fn.buffers)))
    _BUF_RANK[0] = ranks if ranks is not None else {
        b.slot: i + _RANK_BASE[0]
        for i, b in enumerate(sorted(fn.buffers, key=lambda x: x.slot))}
    _prepare_tensor_composition(fn)
    # A DEVICE LOAD IN A TEXTURE KERNEL IS ADMITTED ONLY INSIDE THE MEASURED DOMAIN.
    #
    # THE OLD REFUSAL'S PREMISE WAS ALREADY STALE, which is worth saying rather than quietly
    # dropping: it said the load "takes a BASE REGISTER index from BUFFER_BASE_REG", and that
    # stopped being true when the base was measured to be the RANK (the indexed pair at public
    # indices [1,2,3] and [2,4,6], two hundred lines below). What remained genuinely open was
    # whether the rank law survives INTERNAL bindings, and root measured exactly that - see
    # _texture_load_domain for the four sources and their decoder output. Inside that domain the
    # existing rank calculation reproduces Apple byte for byte; outside it, this still refuses, and
    # the reason names which of the three boundaries was crossed.
    if _RANK_BASE[0] and any(o.kind == "load" for blk in fn.blocks for o in blk.ops):
        _outside = _texture_load_domain(fn)
        if _outside is not None:
            raise Unsupported(
                "a device load in a texture kernel outside the measured base domain: %s. Apple's "
                "four retained sources fix the indexed word load's base at 4 * rank with one user "
                "buffer and one or two read textures (results/g17-source-admission-v1/"
                "texture-indexed-load-base-measurement.json); a base outside that reads another "
                "allocation or descriptor state" % _outside)
    # VALUES READ MORE THAN ONCE. The four-byte register-register bitwise RELEASES its sources and
    # has no room to say otherwise, so a value it reads is gone for anything after it. Knowing which
    # values have a later reader is the only way to refuse that case instead of emitting it.
    _MULTI_USE.clear(); _ORDER.clear(); _LAST_READ.clear()
    _seen = set()
    for _blk in fn.blocks:
        for _o in _blk.ops:
            _ORDER[_o] = len(_ORDER)
            for _a in _o.args:
                if isinstance(_a, ir.Value):
                    (_MULTI_USE if _a in _seen else _seen).add(_a)
                    _LAST_READ[_a] = _ORDER[_o]
    if not _NO_LOOP_AWARE_READS:
        _extend_reads_over_back_edges(fn)
    """IR -> MInst list with VIRTUAL registers (the ir.Value objects themselves)."""
    out = []
    if len(fn.blocks) == 1:
        for op in fn.blocks[0].ops: _select_op(op, out)
    else:
        _lower_blocks(list(fn.blocks), out, None, list(fn.blocks))
    # cmp placeholders exist so a cmp used for anything but a branch is caught rather than
    # silently dropped; a cmp consumed by a br_cond has been replaced by the compound form.
    kept = [m for m in out if m.form != "cmp.placeholder"]
    # AND IT WAS CAUGHT ONLY AS A BARE KeyError. Dropping the placeholder also drops the only
    # definition of the compare's result, so a program that READS that result died in the
    # allocator as `use before def of %c` - naming neither the cmp nor the reason. That is the
    # exact complaint g17ir's own IMM_OPERANDS comment makes about a constant in a register slot,
    # and the fix is the same: say it here, where the reason is still known.
    #
    # Apple's instruction for this is op11462/10 - the register-register integer compare whose
    # result is a VALUE rather than a predicate. tools/g17cmp11462.py places all 38 of its varying
    # bits and rebuilds 1,637 of 1,637 corpus instances byte-exact, so what is missing is not the
    # encoding: it writes a 16-bit HALF register (425+n low, 281+n high) and this IR has no
    # half-width value for a cmp to define. 542 of 1,704 corpus destinations never have their
    # companion half defined anywhere in the program, so widening the result to a word is not
    # what Apple does either.
    _predicates = {v for m in out if m.form == "cmp.placeholder" for v in m.defs}
    if _predicates:
        _read = [v for m in kept for v in m.uses if v in _predicates]
        if _read:
            raise Unsupported(
                "the result of a cmp is read as a value by %d later instruction(s), and this "
                "compiler's cmp produces a BRANCH PREDICATE, not a value: its placeholder is "
                "dropped here and the read would reach the allocator as `use before def`. "
                "Apple's form for a value-producing register-register compare is op11462/10, "
                "which is fully placed (tools/g17cmp11462.py) and refused for a reason about the "
                "TYPE rather than the encoding - it defines a 16-bit half register and this IR "
                "has no half-width value a cmp could define" % len(_read))
    return copy_before_tied(copy_before_destructive(kept))

# The largest explicit tensor strides the general lowering was measured exact at, in bytes (recon section 47,
# stride_sweep.json): A to 1 MiB, B to 512 KiB. A registry six-bit stride refusal routes to the general lowering
# only within these.
STRIDE_MEASURED_MAX = (("strideA", 1 << 20), ("strideB", 1 << 19))


def _select_op(op, out):
    for op in (op,):
        k = op.kind
        if k == "phi":
            # A PHI EMITS NO INSTRUCTION. It is a register-coalescing constraint: the value it
            # defines, the value arriving on the entry edge and the value arriving on the back
            # edge are one register, so the loop body writes the register the header reads on the
            # next iteration. Apple's own induction variables look exactly like that - the
            # increment in ds_setup_indirect_update_mapping is `add R92, 1, R92`, one register as
            # both source and destination.
            #
            # Carried as a zero-size placeholder rather than as defs/uses, because the allocator
            # is a linear scan and the latch value is defined BELOW this point: presenting it as
            # an ordinary use would trip the use-before-def check that is otherwise load-bearing.
            out.append(MInst("phi", 0, dict(phi_group=[op.dest] + list(op.args))))
        elif k == "const" and op.attrs.get("length") == 2:
            # op11842/2, THE TWO-BYTE IMMEDIATE MOVE (tools/g17movimm2.py, results/g17-movimm2-v1).
            # Eleven varying bits over 194 corpus instances, all 194 rebuilt byte for byte, one
            # residue base, and the whole form is a four-bit destination and a seven-bit immediate.
            #
            # REQUESTED, NOT PREFERRED, as alu.fadd.4, alu.ffma.4/6 and alu.fmul.4 are: `length=2`
            # on the IR op selects it. movimm.8 stays this path's default, so no retained
            # delivery's bytes move.
            #
            # THE SEVEN-BIT BOUND IS THE DECODER'S, NOT THE CORPUS'S. b1[7] is the obvious eighth
            # immediate bit and the corpus never sets it - but silence is the shape of a hard
            # limit and of an unexercised one alike, so it was asked: Apple's decoder REFUSES the
            # instruction with that bit set, on 16 of 16 distinct encodings. Hence 0..127 and no
            # wider.
            #
            # THE DESTINATION BOUND IS AN ALLOCATION CONSTRAINT and is deliberately NOT checked
            # here. The field is four bits - r105..r120 - and which register this value gets is
            # decided after selection, so the encoder refuses an out-of-range destination by name
            # rather than truncating it into somebody else's register. Selecting here and refusing
            # there is the same split op2190/4's accumulator tie needed.
            if not isinstance(op.args[0], ir.Imm):
                raise Unsupported("op11842/2 moves an IMMEDIATE; its operand is %r" % (op.args[0],))
            if not 0 <= op.args[0].v <= 127:
                raise Unsupported("op11842/2 carries a SEVEN-bit immediate (0..127) and %d does "
                                  "not fit. The eighth bit is not a wider field: Apple's decoder "
                                  "refuses the instruction with b1[7] set" % op.args[0].v)
            out.append(MInst("movimm.2", 2, dict(imm=op.args[0].v), defs=[op.dest]))
        elif k == "const":
            if op.attrs.get("requant_stage_const"):
                # The measured six-byte clamp templates carry their signed/unsigned bounds in
                # inherited control fields.  The IR constants remain in the semantic graph for
                # verification, but they are not materialised as registers in this exact Apple
                # class (doing so would select a different clamp form).
                return out
            # A SIXTEEN-BIT ZERO TAKES THE FOUR-BYTE FORM (handoff 10ag; integration's d076d651): op555/4 is
            # Apple's move of zero into a sixteen-bit register, and the source-faithful witness is a half store
            # of 0.0h (results/g17-movimm4-roundB-compiles-v1/W0, whose instruction this reproduces byte for
            # byte). It is taken only when EVERY consumer reads the value as sixteen bits - the half store is the
            # witnessed one - because the form writes one half and leaves the other as it was, so a 32-bit reader
            # of the same value would read whatever the high half held. Everything else keeps movimm.8.
            if op.args[0].v == 0 and getattr(op.dest, "type", None) == ir.I16 and _all_uses_are_16_bit(_CUR_FN[0], op.dest):
                # NO `opcode=` HERE ON PURPOSE: an emitter that names its own opcode can only confirm
                # itself (the linker's correction on the vector forms), so the checked-in registry names
                # this one, harvested from a program by tools/g17formops.py --refresh.
                out.append(MInst("movimm16.zero.4", 4, {}, defs=[op.dest],
                                 note="r<425 + dest> <- 0, the sixteen-bit zero move"))
            elif _feeds_a_half_store_value(_CUR_FN[0], op.dest):
                # A NON-ZERO SIXTEEN-BIT CONSTANT FEEDING A HALF STORE, BUILT FROM TWO MEASURED FORMS
                # RATHER THAN FROM A THIRD THAT WOULD HAVE TO BE GUESSED.
                #
                # movimm.8 alone is wrong here and that is what this used to refuse: the half store's
                # value operand names the 425-based file (Builder.store_at) while movimm.8 writes the
                # 32-bit file at the same INDEX, so the pair compiled and the store read a different
                # register. Apple emits op11843/8 (W3: `14 80 02 00 10 0c 01 00`, imm 0x3C00 = 1.0h),
                # whose one retained instance is single-valued in BOTH its register and its immediate -
                # so inverting the decoder for it would be a round trip that cannot fail, and the field
                # split would be a guess.
                #
                # THE COMPOSITION INSTEAD: materialise the value's binary32 encoding with movimm.8,
                # then narrow it with op1016/12, the measured f32->f16 conversion this backend already
                # emits and whose bytes have executed. Why it is exact, and not merely close:
                #
                #   * every finite binary16 value is exactly representable in binary32 - 11 bits of
                #     significand into 24, and the exponent range is strictly wider - so the immediate
                #     carries the value itself, not an approximation of it;
                #   * narrowing an exactly-representable value performs NO ROUNDING, so the result does
                #     not depend on the rounding mode at all. This does not rest on op1016 being
                #     round-to-nearest-even (which the executed fill receipts exercise in only six
                #     values, none of them a tie);
                #   * verified over the whole domain: all 61,440 admitted patterns widen and narrow back
                #     to themselves, 0 wrong (test_g17halfconversions).
                #
                # WHAT IS REFUSED, BY NAME, because each would need a fact nothing here has measured:
                #
                #   exponent 0   zero and subnormal halves. A subnormal binary16 is a NORMAL binary32,
                #                so the widening is still exact, but narrowing it back needs op1016 to
                #                produce a subnormal rather than flush to zero, and that is unmeasured.
                #                (A zero VALUE never reaches here: op555/4 above takes it.)
                #   exponent 31  infinity and NaN. Infinity is exactly representable both ways but
                #                op1016's behaviour on it is unmeasured, and for NaN the payload and
                #                the quiet bit are a second unmeasured fact on top of that.
                #
                # And it applies only where the destination buffer is declared `half`. The IR types a
                # `ushort` literal and a `half` literal identically (both I16), so the DECLARATION is
                # the only thing that can say whether these sixteen bits are a float at all; a 16-bit
                # integer constant narrowed as a float would be silently wrong.
                bits = op.args[0].v & 0xFFFF
                elements, only_half_stores = _half_store_targets(_CUR_FN[0], op.dest)
                f32 = _f32_bits_of_half(bits)
                if not only_half_stores or elements != {ir.F16}:
                    raise Unsupported(
                        "a non-zero sixteen-bit constant (%d) feeding a half store is materialised as "
                        "its binary32 encoding plus op1016/12, which is a FLOAT composition, and the "
                        "destination declares %s: the IR types `half` and `ushort` literals alike, so "
                        "the declaration is the only thing that says these bits are a float"
                        % (bits, ", ".join(sorted(str(e) for e in elements)) or "no half store"))
                if f32 is None:
                    exp = (bits >> 10) & 0x1F
                    raise Unsupported(
                        "the half constant 0x%04x is %s, and composing it from its binary32 encoding "
                        "plus op1016/12 needs a fact this backend has not measured: %s"
                        % (bits, "zero or subnormal" if exp == 0 else "infinite or NaN",
                           "whether op1016 produces a subnormal half or flushes it to zero" if exp == 0
                           else "op1016's behaviour on a non-finite value, and for NaN its payload and "
                                "quiet bit as well"))
                wide = ir.Value(ir.I32, "h%04x_f32" % bits)
                out.append(MInst("movimm.8", 8, dict(imm=f32), defs=[wide],
                                 note="the half constant 0x%04x as its exact binary32 encoding" % bits))
                out.append(MInst("cvt.f32.f16", 12, dict(keep_src=False), defs=[op.dest], uses=[wide],
                                 note="narrow it back: exactly representable, so no rounding occurs"))
            else:
                out.append(MInst("movimm.8", 8, dict(imm=op.args[0].v), defs=[op.dest]))
        elif k == "builtin":
            which = op.attrs["which"]; axis = op.attrs.get("axis", "x")
            if which not in SR: raise Unsupported("builtin %r" % which)
            fields = dict(sr=SR[which] + SR_AXIS[axis], seq=0)
            if op.attrs.get("requant_stage"):
                from agxforge.g17 import requantenc
                if op.attrs.get("requant_stage_step") != "read_sr":
                    raise Unsupported("requantization stage has an unexpected builtin step")
                fields["raw"] = requantenc.stage_bytes(bool(op.attrs.get("requant_signed")), "read_sr")
                fields["requant_stage"] = True
            _m = MInst("read_sr.4", 4, fields, defs=[op.dest])
            if fields.get("requant_stage"):
                _m.fields["requant_fixed_defs"] = (1,)
            out.append(_m)
        elif k in SAT_OPCODE:
            opc = SAT_OPCODE[k]
            a, b_ = op.args
            name, ra, rb = g17asm.ALU_FORM[opc]
            if rb == "imm":
                if not isinstance(b_, ir.Imm):
                    raise Unsupported("%s takes an immediate shift amount" % k)
                out.append(MInst("alu.sat", 12, dict(opcode=opc, hazard=_hz(op.args), imm=b_.v),
                                 defs=[op.dest], uses=[a]))
            else:
                if isinstance(a, ir.Imm) or isinstance(b_, ir.Imm):
                    raise Unsupported("%s takes two registers; the corpus shows no immediate form" % k)
                out.append(MInst("alu.sat", 12, dict(opcode=opc, hazard=_hz(op.args)),
                                 defs=[op.dest], uses=[a, b_]))
        elif k == "f32_to_f16_rte":
            # THE NARROWING IS A REAL INSTRUCTION, unlike the widening. op1016 cvt.f32.f16: a
            # 16-bit destination in the file based at 425 and a 32-bit source based at 105.
            #
            # THE SOURCE LIFETIME IS NOT UNCONDITIONALLY RELEASE, and it used to be. See the
            # widening below for the measurement that found it: a value with a LATER reader must be
            # kept, and `keep_src` is operand 3 of this form, MOV_KEEP against MOV_RELEASE.
            (x,) = op.args
            out.append(MInst("cvt.f32.f16", 12, dict(keep_src=_has_later_reader(x, op)),
                             defs=[op.dest], uses=[x]))
        elif k == "fma16":
            # THE BINARY16 FUSED MULTIPLY-ADD, op798/12 (MM 25.196). Three sixteen-bit sources and a
            # sixteen-bit destination in the 425-based file; a source may be either half of a 32-bit
            # value (281+n reads the high half), which is how integer code's packed halves are read
            # without a move. The source lifetimes are written from liveness in Alloc._lifetimes.
            out.append(MInst("ffma.f16", 12, dict(halves=tuple(op.attrs["halves"])), defs=[op.dest],
                             uses=_fma16_sources(op, out)))
        elif k == "f16_to_f32":
            # THE WIDENING IS AN ALU READ, NOT A CONVERSION. op1004 is an fadd.imm of zero whose
            # source operand is SIXTEEN BITS: its destination is the 32-bit file based at 105 and
            # its source the 16-bit file based at 425, and authoring it reproduces Apple's own
            # instruction for `(float)h[i]` byte for byte. Operand 1 carries the zero immediate and
            # operand 3 is the source lifetime, and it WAS unconditionally 16 RELEASE with this
            # reason: "the widened value is what the program keeps and the half is not read again".
            #
            # THAT PREMISE STOPPED BEING TRUE and the receipt found it. When the only producer of
            # half values was a half load feeding one widening, no half was read twice. Composed
            # half arithmetic (front end, batch 4) makes half values with SEVERAL consumers, and
            # releasing one at its first consumer frees a register a later one still reads.
            #
            # MEASURED, on hardware, in results/g17-source-half-add-extended-runtime-v1: the first
            # query of unchanged syn-s7f595f1cd1 returned half24 = 0.0 where 7.0 was expected,
            # half41 = 2.0 where 2054.0 was, half43 = 4.0 where 4100.0 was, and the other 45 halves
            # were exact. Substituting ZERO for exactly the operand whose register had been
            # released reproduces all three: (half)2051 + 2 is 2052 + 2 = 2054, and with the first
            # addend read as zero it is 2.
            #
            # THE CLAIM IS BOUNDED TO THOSE THREE SITES. An earlier version of this comment ended
            # "a released half register reads as zero", which is a semantic law one receipt cannot
            # establish - root's review is right that zero-like results at three sites in one
            # program are not a general statement about what a released register reads.
            (x,) = op.args
            out.append(MInst("cvt.f16.f32", 12, dict(keep_src=_has_later_reader(x, op)),
                             defs=[op.dest], uses=[x]))
        elif k == "faddi":
            ty, sat = op.attrs["ty"], op.attrs["sat"]
            opc = FADD_IMM_OPCODE.get((ty, sat))
            if opc is None:
                raise Unsupported("no measured %s float add with an immediate%s"
                                  % (ty, " that saturates" if sat else ""))
            code = g17asm.float_imm(op.attrs["imm"])
            if code is None:
                raise Unsupported("the eight-bit float immediate cannot spell %r exactly; the "
                                  "nearest it reaches is %r"
                                  % (op.attrs["imm"],
                                     g17asm.float_imm_value(g17asm.float_imm(op.attrs["imm"], True))))
            out.append(MInst("auth", g17auth.length(opc),
                             dict(opcode=opc, imms={FADD_IMM_OPERAND: code}),
                             defs=[op.dest], uses=[op.args[0]]))
        elif k == "fadd" and op.attrs.get("length") == 6:
            # op998/6, THE SIX-BYTE FLOAT ADD (tools/g17fadd6.py, results/g17-fadd6-v1). The
            # largest ordinary lowering gap on the frontier - 366 blocked programs - and what
            # refused it was one operand read as one number.
            #
            # OPERAND 1 IS TWO FIELDS. g17faddselector refused this form because "operand 1 takes
            # ten values leaving 29 bits unlocated". Those ten values are {0, 32} x {no index,
            # 2^24, 2^25, 2^26, 2^27, 2^28}: a DESTINATION LIFETIME bit at b2[5] and a three-bit
            # INDEX CODE at b5[5..7]. No single-bit fit of a 40-bit value can separate a lifetime
            # from a code sharing its operand, which is why the population looked uninformative
            # when it was not. All 30 varying bits are placed and all 1,875 corpus instances
            # reconstruct operand for operand from one residue base.
            #
            # THE INDEX IS WRITTEN AS ZERO and a non-zero one is refused by the encoder. It is
            # ascending-distinct within a program - the shape handoff 10dn found on op2190/8 - so
            # it is a slot the scheduler allocates rather than something the source chooses. Apple
            # leaves it zero on 1,695 of 1,875 rows.
            #
            # A LENGTH-6 REQUEST USED TO VANISH: before this branch it fell through to the generic
            # auth path and emitted op998/12 with no signal, while length=4 was honoured.
            #
            # REQUESTED, NOT PREFERRED: op998's default on this path stays the twelve-byte route.
            if len(op.args) != 2:
                raise Unsupported("op998/6 takes two sources; %d were given" % len(op.args))
            for a in op.args:
                if isinstance(a, ir.Imm):
                    raise Unsupported("op998/6's sources are both REGISTERS - two seven-bit "
                                      "105-based fields and their lifetimes - and no immediate "
                                      "carrier is located at this length")
            out.append(MInst("alu.fadd.6", 6, dict(opcode=998, dest_life=32),
                             defs=[op.dest], uses=list(op.args)))
        elif k == "fadd" and op.attrs.get("length") == 4:
            # op998/4, THE FOUR-BYTE FLOAT ADD (tools/g17fadd4.py, results/g17-fadd4-v1). Every
            # field located over 1,405 corpus instances, all 1,405 rebuilt byte for byte, one
            # residue base, and ZERO instances dropped as expression operands.
            #
            # THIS IS THE LOWERING 10cz SAID WAS BLOCKED, and the blocker it named does not apply
            # at this length. That section says `fadd` goes through the generic `auth` route where
            # the source lifetime is INHERITED from the authoring witness - four programs in
            # memory:g17-modifier-operand-lifetimes read zero that way - and the dedicated-encoder
            # route was then refused because operand 1 looked like the unplaced modifier word that
            # refuses op998/6. AT /4 OPERAND 1 IS THE DESTINATION LIFETIME: two values, 32 and 0,
            # carried at b2[5], the same bit and meaning as op2190/6's. Nothing is unplaced here.
            # /6 and /12 remain refused and are a different question.
            #
            # THREE-ADDRESS, so no tie for the allocator: 344 of 1,405 rows write a destination
            # equal to neither source. op2190/4's post-allocation accumulator check has nothing to
            # do here and a straight-line witness selects the form.
            #
            # REQUESTED, NOT PREFERRED, exactly as alu.ffma.4, alu.ffma.6 and alu.fmul.4 are:
            # `length=4` on the IR op selects it. op998's default on this path is the generic
            # route's TWELVE-byte form and every retained delivery's bytes were taken with that
            # choice, so making /4 the default would rewrite them.
            if len(op.args) != 2:
                raise Unsupported("op998/4 takes two sources; %d were given" % len(op.args))
            for a in op.args:
                if isinstance(a, ir.Imm):
                    raise Unsupported("op998/4's sources are both REGISTERS - two six-bit "
                                      "105-based fields and their lifetimes - and no immediate "
                                      "carrier is located at this length")
            out.append(MInst("alu.fadd.4", 4, dict(opcode=998, dest_life=32),
                             defs=[op.dest], uses=list(op.args)))
        elif k == "fma" and op.attrs.get("length") == 6:
            # op2190/6, THE SIX-BYTE FUSED MULTIPLY-ADD (tools/g17ffma6.py, results/g17-ffma6-v1).
            # Every field located over 1,179 corpus instances, all 1,179 rebuilt byte for byte from
            # one base, 31 of 31 varying bits accounted.
            #
            # THREE-ADDRESS, WHICH /4 IS NOT, and that is the whole difference in difficulty. Zero
            # of 1,179 rows alias the destination into a source, so all four registers are
            # independent and there is no tie for the allocator to satisfy. /4's accumulator check -
            # verified after allocation and refused when untied - has nothing to do here.
            #
            # REQUESTED, NOT PREFERRED, exactly as /4 and op3290/4 are: `length=6` on the IR op
            # selects it. op2190's default on this path is sixteen bytes and every retained float
            # delivery's bytes were taken with that choice.
            #
            # AND IT IS NOT THE BIT-24 FAMILY. op2190/8 and op3290/6 are refused because their
            # source controls reach only the sub-population whose operand 1 has bit 24 set while
            # Apple's common case is elsewhere. Here operand 1 takes exactly TWO values across the
            # whole corpus - 32 and 0, with bit 24 in none of them - and four controls already in
            # the tree span both. There is no minority for an emitter to be stranded in.
            if len(op.args) != 3:
                raise Unsupported("op2190/6 takes three sources; %d were given" % len(op.args))
            for a in op.args:
                if isinstance(a, ir.Imm):
                    raise Unsupported("op2190/6's sources are all REGISTERS - four six-bit "
                                      "105-based fields and their lifetimes, and no immediate "
                                      "carrier is located at this length")
            # A LOADED SOURCE GOES THROUGH THE WAITING COPY, as on /4, whose bytes read 0 on every
            # lane when fed straight from loads. Whether /6 waits on its own is not measured; the
            # copy costs one instruction per loaded source and never a value, and
            # _NO_FFMA6_LOADWAIT is the control arm that removes it (tools/g17ffma4receipt.py).
            args = list(op.args) if _NO_FFMA6_LOADWAIT else [_wait_for_load(out, a) for a in op.args]
            out.append(MInst("alu.ffma.6", 6, dict(opcode=2190, dest_life=32),
                             defs=[op.dest], uses=args))
        elif k == "fma" and op.attrs.get("length") == 4:
            # op2190/4, THE FOUR-BYTE FUSED MULTIPLY-ADD (tools/g17ffma4.py, results/g17-ffma4-v1).
            # Every field located over 653 corpus instances, all 653 rebuilt byte for byte.
            #
            # REQUESTED, NOT PREFERRED, exactly as op3290/4 is: `length=4` on the IR op selects it
            # and nothing else does. op2190's default on this path is sixteen bytes and every
            # retained float delivery's bytes were taken with that choice; making /4 the default
            # would rewrite them, which integration's dispatch says not to do.
            #
            # THE FORM IS TWO-ADDRESS AND THAT IS THE WHOLE DIFFICULTY. The accumulator IS the
            # destination - the decoder prints it again in operand 4 or 6, and a mode says which -
            # so this instruction can only express `d = a*b + d`. alu.fmul.4 is three-address and
            # its shape does not carry here.
            #
            # THE TIE IS REQUESTED, AND STILL CHECKED AFTER ALLOCATION. In SSA the accumulator
            # operand and the destination are distinct values. They used to end in one register only
            # when a phi coalesced them, so everything but an accumulator loop was refused. Now
            # TIED_FORMS asks for it: copy_before_tied copies an accumulator that is still needed,
            # and Alloc pre-colours accumulator and destination to one register (2026-09-23,
            # ledger/g17-ffma4-tie-and-one-rounding.toml). The check in the encoder below stays as
            # the backstop - it refuses if the allocator did not produce the tie. Emitting bytes that claim a tie the registers do
            # not have would be a wrong program, and inserting a copy to force one would be an
            # allocation decision the caller did not ask for.
            if len(op.args) != 3:
                raise Unsupported("op2190/4 takes three sources; %d were given" % len(op.args))
            for a in op.args:
                if isinstance(a, ir.Imm):
                    raise Unsupported("op2190/4's sources are all REGISTERS - the located fields "
                                      "are two six-bit register fields plus the destination "
                                      "re-printed, and no immediate carrier exists at this length")
            # A LOADED SOURCE IS READ BEFORE IT LANDS. Executed 2026-09-23: fed straight from three
            # loads, op2190/4 returned 0 on 32 of 32 lanes; fed from `x + 0` copies, the same bytes
            # (590d3a0f) returned the single-rounded a*b+c on 32 of 32. The base's byte0[3] - the
            # bit that is the load-wait on op3290/4 - does not make this form wait. So every loaded
            # source goes through the measured waiting copy (alu.12 byte0[3]) first.
            args = [_wait_for_load(out, a) for a in op.args]
            # uses[0] is the ACCUMULATOR, which must come back as the destination's register.
            out.append(MInst("alu.ffma.4", 4, dict(opcode=2190, dest_life=32),
                             defs=[op.dest], uses=[args[2], args[0], args[1]]))
        elif k == "requant_fmul":
            # The retained scalar requantization witness uses op3290/6's specialized expression
            # form.  Its generic six-byte table is a different encoding, so this row carries the
            # measured template codec and is admitted only for the explicit IR primitive.
            from agxforge.g17 import requantenc
            if len(op.args) != 1:
                raise Unsupported("requantization scale multiply takes one source")
            _fields = dict(opcode=3290, template=requantenc.FMUL6_TEMPLATE,
                            encoder=requantenc.encode_fmul6, requant_fmul=True)
            if op.attrs.get("requant_stage"):
                if op.attrs.get("requant_stage_step") != "fmul":
                    raise Unsupported("requantization stage has an unexpected scale step")
                _fields.update(raw=requantenc.stage_bytes(bool(op.attrs.get("requant_signed")), "fmul"),
                               requant_stage=True)
            _m = MInst("auth", 6, _fields, defs=[op.dest], uses=[op.args[0]])
            if _fields.get("requant_stage"):
                _m.fields.update(requant_fixed_defs=(0,), requant_fixed_uses=(0,))
            out.append(_m)
        elif k == "fmul" and op.attrs.get("length") == 4:
            # op3290/4, THE FOUR-BYTE REGISTER-REGISTER FORM (handoff 10bb; tools/g17fmul4.py,
            # results/g17-fmul4-v1). Every field located over 2,675 corpus instances, all 2,675
            # reconstructed byte for byte from one base.
            #
            # IT IS REQUESTED, NOT PREFERRED, and that is deliberate rather than timid. op3290's
            # default length on this path is 14 through the authoring table, and every retained
            # float delivery's bytes were taken with that choice. Making /4 the default would
            # rewrite all of them - integration's dispatch says to preserve existing validated
            # bytes, and a length change that silently moved a hundred retained programs would be
            # the opposite of that. So `length=4` on the IR op selects it and nothing else does,
            # which is also exactly what was asked for: a source-owned witness SELECTING the form.
            #
            # THE SOURCE LIFETIMES ARE WRITTEN FROM LIVENESS, never inherited - four separate
            # occasions in memory:g17-modifier-operand-lifetimes are an authored program reading
            # zero because a template's lifetime was carried. The destination lifetime is 32,
            # which is what Apple writes in 2,281 of its 2,675 instances; what it MEANS is not
            # measured, so it is stated as inherited-with-a-count rather than derived, and the
            # third value the field can hold (0, in 394 of them) is expressible by the encoder and
            # never written here - the same standing g17asm gives the mov lifetime's 0.
            #
            # The load-wait is not a decision: byte0[3] is set in the base on all 2,675 instances,
            # and it is the bit whose silicon measurement on THIS opcode took the float kernel
            # from zeros to 1.0, 2.25, 4.0, 6.25, 9.0.
            from agxforge.g17.formenc import Fmul4 as g17fmul4
            if len(op.args) != 2:
                raise Unsupported("op3290/4 takes two register sources; %d were given" % len(op.args))
            for a in op.args:
                if isinstance(a, ir.Imm) or getattr(a, "op", None) is None and not isinstance(a, ir.Value):
                    raise Unsupported("op3290/4's sources are both REGISTERS - the six located "
                                      "fields are three register/lifetime pairs and no immediate "
                                      "carrier is located in this length")
            # [CORRECTED 2026-09-22: the premise above is too strong for two loaded operands.
            # results/g17-formreceipt-v1: with BOTH sources straight from loads, op3290/4 returned 0
            # on every one of 32 lanes in two workers (fmul4_unwaited_failed); with each source first
            # copied through an add of 0 that carries the wait, it was bit-exact on 192 words in two
            # workers (fmul4_waited). byte0[3] in the base does not cover this shape. So an operand
            # that comes straight from a load is copied through the waiting alu.12 add first - the
            # repair op9700 already uses - and a source that is not a load's result is untouched.]
            args = []
            for v in op.args:
                if _load_wait([v]):
                    copy = ir.Value(getattr(v, "type", ir.F32), "%s_w" % (getattr(v, "name", "v") or "v"))
                    out.append(MInst("alu.12", 12,
                                     dict(op=3, mode=1, imm=0, src1_w=1, dest_w=1, srcb_w=1,
                                          scale_code=2, load_wait=1, b4_5=None, b0_5=None,
                                          keep=1 if v in _MULTI_USE else 0),
                                     defs=[copy], uses=[v],
                                     note="copy: op3290/4 does not wait for two loaded sources; the add does"))
                    v = copy
                args.append(v)
            out.append(MInst("alu.fmul.4", 4, dict(opcode=3290, dest_life=32),
                             defs=[op.dest], uses=args))
        elif k == "fsat":
            (x,) = op.args
            if isinstance(x, ir.Imm):
                raise Unsupported("fsat of an immediate: materialise it first")
            x = _wait_for_load(out, x)
            out.append(MInst("auth", len(FSAT_APPLE),
                             dict(opcode=FSAT_OPCODE, imms={FADD_IMM_OPERAND: FSAT_IMM_MINUS_ZERO},
                                  template=FSAT_APPLE),
                             defs=[op.dest], uses=[x]))
        elif k in ("fneg", "fabs"):
            # A MODIFIER, NOT AN INSTRUCTION. Every consumer must be a float add, multiply or fma,
            # which reads the source through its modifier word; anything else would need the value
            # itself, which Apple makes with an integer sign-bit operation that is not lowered here.
            users = [o for blk in _CUR_FN[0].blocks for o in blk.ops
                     if any(a is op.dest for a in o.args)]
            # a chain (fneg of fabs) is fine: each link is checked where it is selected, and the
            # outermost must reach a consumer that folds it
            bad = [o.kind for o in users
                   if o.kind not in MODIFIER_CONSUMERS + ("fneg", "fabs") or o.attrs.get("length")]
            if bad or not users:
                raise Unsupported("%s is a source modifier: it folds into %s, and %s reads it directly"
                                  % (k, "/".join(sorted(MODIFIER_CONSUMERS)), bad or "nothing"))
        elif k in MACHINE_OPCODE or k in MACHINE_UNARY:
            opc = MACHINE_OPCODE.get(k) or MACHINE_UNARY[k]
            args, mods = (_fold_modifiers(op.args) if k in MODIFIER_CONSUMERS
                          else (list(op.args), None))
            lw = _load_wait(args)
            # THE PAIRED CONTROL'S ONE DEGREE OF FREEDOM, and nothing else about the program moves.
            # The arithmetic checker admits op2190/16's lead modifier only at 10737418240 - byte0[3]
            # SET - and the retained sl32-u148 image has that on its first FMA, whose operand comes
            # from a load, and 8589934592 on the 147 that chain from the previous FMA. The only
            # receipt carrying the second value is a FAILED one whose cause is confounded with a
            # wrong immediate, so whether the bit matters here is unproven, and this option is how
            # the pair that would prove it gets built: OFF is the default and emits what this
            # compiler has always emitted; ON conservatively requests the ALREADY-ADMITTED state for
            # every FMA, so the two programs differ in byte0[3] of 147 instructions and in nothing
            # else at all.
            #
            # It is a REQUEST for a state the checker already admits, not a new one: setting a bit
            # Apple sets on 670 retained instances, on instructions that currently clear it. That
            # this direction can only make an instruction wait LONGER than it needs to is a
            # HYPOTHESIS supported by that existing usage - it is not an isolated measurement of
            # this 147-instruction configuration, and nothing here has measured what the bit does
            # when the operand did not come from a load.
            # op2190 ALONE, and the first version of this was broader. `opc in AUTH_LOAD_WAIT`
            # also caught fadd, fmul, madd, fsat and the bitwise operations that share this
            # selection path, so the option changed instructions the assignment never mentioned -
            # and my no-FMA bounding test only STORED A CONSTANT, so it contained none of them and
            # could not fail. A bounding test that does not contain the construct it bounds is not
            # a bound; the repaired one chains a wait-capable operation and is checked BOTH ways.
            if _FMA_ALWAYS_LOAD_WAIT and opc == FMA_OPCODE:
                lw = 1
            if lw and opc not in AUTH_LOAD_WAIT:
                raise Unsupported("%s takes an operand straight from a load and op%d has no known "
                                  "wait bit - byte0[3] is the one every other form on this path "
                                  "uses and it is an OPCODE bit here. Put the value through an ALU "
                                  "op first; those carry the wait" % (k, opc))
            _fields = dict(opcode=opc, imms=dict(MACHINE_IMMS.get(opc) or {}), load_wait=lw)
            if op.attrs.get("requant_stage"):
                from agxforge.g17 import requantenc
                if op.attrs.get("requant_stage_step") != "narrow" or opc != 9320:
                    raise Unsupported("requantization stage has an unexpected narrowing step")
                _fields.update(raw=requantenc.stage_bytes(bool(op.attrs.get("requant_signed")), "narrow"),
                               requant_stage=True)
            if mods and any(n or a for n, a in mods):
                _fields["mods"] = mods
            out.append(MInst("auth", g17auth.length(opc),
                             _fields,
                             defs=[op.dest], uses=args))
        elif k == "and16":
            # HALF-REGISTER FIELD EXTRACTION (MM 25.138): op426 (imm) / op428 (reg) from Apple's own
            # instances, only the register fields written; the source lifetime stays the template's 32
            # (keep), so one word serves every field without a copy. Its load wait is unmeasured, so a
            # loaded source gets ONE alu.12 copy with the wait bit, shared by every field of that word.
            src = op.args[0]
            # direct (ir.and16(direct=True), MM 25.144.4): a vector-load lane whose load an earlier consumer
            # has already waited on is read in place - the rule every later lane consumer already follows
            waited = (op.attrs.get("direct") and isinstance(src, ir.Value) and src.op is not None
                      and src.op.kind in ("load_vec_at", "vec_lane")
                      and ((src if src.op.kind == "load_vec_at" else src.op.args[0]) in _VEC_WAITED
                           or _AND16_DIRECT_UNWAITED))
            if _is_load_value(src) and not waited:
                vload = (src if src.op.kind == "load_vec_at" else src.op.args[0]) if (
                    op.attrs.get("direct") and src.op.kind in ("load_vec_at", "vec_lane")) else None
                src = _AND16_WAITED.get(id(src)) or _AND16_WAITED.setdefault(
                    id(src), _isolate_bitwise_operand(out, src, force=True, keep_source=True,
                                                      why="and16: op426/op428's load wait is unmeasured"))
                if vload is not None:
                    _VEC_WAITED.add(vload)       # the copy waited on the whole vector load
            mask, half = op.attrs["mask"], op.attrs["half"]
            opc = 426 if mask <= 0xFF else 428
            pooled = op.attrs.get("uniform") is not None
            uses = [src] if (opc == 426 or pooled) else [src, op.args[1]]
            a16 = dict(half=half, mask=mask)
            if pooled:
                a16.update(uniform=op.attrs["uniform"], pool_h=op.attrs["pool_h"])
            out.append(MInst("auth", 10, dict(opcode=opc, template=AND16_TEMPLATE[opc], encoder=_and16_encoder,
                                              and16=a16),
                             defs=[op.dest], uses=uses,
                             note="and16 %s & 0x%x%s" % (half, mask, " (pool u%d)" % op.attrs["uniform"] if pooled else "")))
        elif k == "machine":
            # THE GENERIC FORM. No hand-written encoder: the length, the operand roles and every
            # field come from the authoring table, so any certified opcode can be selected. The
            # bits no operand names are the witness's - that is inheritance, and it is why this
            # form is for probing an instruction's meaning rather than for shipping one.
            opc = op.attrs["opcode"]
            # A SUPPLIED TEMPLATE CARRIES ITS OWN LENGTH. The same opcode has encodings of
            # different lengths - the condition-code table reaches cc 2 and cc 14 through six-byte
            # forms where the rest are twelve - so the instruction's size comes from the template
            # when one is given and from the table's witness otherwise.
            _t = op.attrs.get("template")
            _fields = dict(opcode=opc, imms=dict(op.attrs.get("imms") or {}), template=_t,
                           encoder=op.attrs.get("encoder"))
            if op.attrs.get("requant_stage"):
                from agxforge.g17 import requantenc
                if op.attrs.get("requant_stage_step") != "narrow" or opc != 9320:
                    raise Unsupported("requantization stage has an unexpected narrowing step")
                _fields.update(raw=requantenc.stage_bytes(bool(op.attrs.get("requant_signed")), "narrow"),
                               requant_stage=True)
            _m = MInst("auth", len(_fields["raw"]) if "raw" in _fields else (len(_t) if _t else g17auth.length(opc)),
                       _fields, defs=[op.dest] if op.dest is not None else [], uses=list(op.args))
            if _fields.get("requant_stage"):
                _m.fields.update(requant_fixed_defs=(0,), requant_fixed_uses=(0,))
            out.append(_m)
        elif k in ("f32_to_u32", "f32_to_i32"):
            # FP32 TO INT: op9320/10, the plain cast (mode operand 1), code 4 unsigned and 5 signed, Apple's own
            # instruction byte for byte (MM 25.196). Its wait and its source lifetime are unmeasured, as op11179's are,
            # and handled the same way: a loaded or re-read operand is copied through the alu.12 whose bits ARE
            # measured, and the source is released as Apple's instances release it (16).
            (x,) = op.args
            if not isinstance(x, ir.Value):
                raise Unsupported("%s of an immediate; fold it at the source instead" % k)
            if not _NO_BITWISE_ISOLATION:
                x = _isolate_bitwise_operand(
                    out, x, force=_is_load_value(x),
                    why="operand isolation: op9320's wait and source lifetime are unmeasured")
            if _is_load_value(x) or x in _MULTI_USE:
                raise Unsupported("%s of a loaded or re-read value: op9320's wait and source lifetime are "
                                  "unmeasured" % k)
            out.append(MInst("cvt.f2i", 10, dict(code=4 if k == "f32_to_u32" else 5), defs=[op.dest], uses=[x]))
        elif k in ("u32_to_f32", "i32_to_f32"):
            # INT TO FP32: op11179 cvt.i2f, Apple's own instruction, from a WITNESSED template
            # (g17asm.CVT_I2F_WITNESS). The two registers and operand 2, the signedness, are
            # written; nothing else.
            #
            # TWO PROPERTIES OF THIS FORM ARE UNMEASURED, and neither is guessed here:
            #   * operand 5 is two located bits with three witnessed values, in the position the
            #     lifetime family occupies elsewhere - so whether this instruction keeps or
            #     releases its source is unknown, and a value with a later reader cannot be handed
            #     to it.
            #   * whether it waits for a load is unknown too. hazard is passed as None rather than
            #     0 for a concrete reason: encode_unary's hazard write clears byte4[3], which the
            #     witness has SET, so asking for "no hazard" would mutate a bit nobody has measured.
            #
            # Both are handled the way the four-byte bitwise handles the same two unknowns - with a
            # COPY through the alu.12 whose load-wait and keep bits ARE measured - rather than by
            # writing this form's unmeasured bits or refusing the programs outright.
            (x,) = op.args
            if not isinstance(x, ir.Value):
                raise Unsupported("%s of an immediate; fold it at the source instead" % k)
            if not op.attrs.get("requant_stage") and not _NO_BITWISE_ISOLATION:
                x = _isolate_bitwise_operand(
                    out, x, force=_is_load_value(x),
                    why="operand isolation: op11179's wait and source lifetime are unmeasured")
            if _is_load_value(x) and not op.attrs.get("requant_stage"):
                raise Unsupported("%s takes its operand straight from a load, and whether " % k +
                                  "op11179 waits is unmeasured; put it through an ALU first")
            if x in _MULTI_USE and not op.attrs.get("requant_stage"):
                raise Unsupported("%s would hand op11179 a value read again later, and " % k +
                                  "whether this form releases its source is unmeasured (operand 5 "
                                  "is two located bits with three witnessed values)")
            # operand 2 is SIGNEDNESS (g17asm.CVT_I2F_SIGNED_BIT): written for both directions
            # rather than inherited, so i32_to_f32 cannot silently convert unsigned
            _fields = dict(opcode=CVT_U32_F32_OPCODE, hazard=None, signed=(k == "i32_to_f32"))
            if op.attrs.get("requant_stage"):
                from agxforge.g17 import requantenc
                if op.attrs.get("requant_stage_step") != "i32_to_f32":
                    raise Unsupported("requantization stage has an unexpected conversion step")
                _fields.update(raw=requantenc.stage_bytes(bool(op.attrs.get("requant_signed")), "i32_to_f32"),
                               requant_stage=True)
            _m = MInst("unary", 10, _fields, defs=[op.dest], uses=[x])
            if _fields.get("requant_stage"):
                _m.fields.update(requant_fixed_defs=(0,), requant_fixed_uses=(0,))
            out.append(_m)
        elif k in FLOAT_UNARY_OPCODE:
            # THE SOURCE IS NOT DECLARED DEAD. This form has no lifetime bit this session can
            # write - on the sibling op9986 the candidate bit turns the source operand into an
            # expression rather than freeing it - so the source stays live and a later reader of
            # the same value is safe. Conservative and measured: fpexec.py reads one value through
            # four of these operations in a row and gets four right answers.
            _fields = dict(opcode=FLOAT_UNARY_OPCODE[k])
            if op.attrs.get("requant_stage"):
                from agxforge.g17 import requantenc
                if op.attrs.get("requant_stage_step") != "rint" or k != "rint":
                    raise Unsupported("requantization stage has an unexpected rounding step")
                _fields.update(raw=requantenc.stage_bytes(bool(op.attrs.get("requant_signed")), "rint"),
                               requant_stage=True)
            # A LOADED SOURCE WAITS. This form carries no load-wait of its own: recip straight from a
            # load read the register before the load landed (online softmax's 1/l,
            # results/g17-tensor-stream-v1), so the source goes through the measured waiting copy.
            src = op.args[0] if _fields.get("requant_stage") else _wait_for_load(out, op.args[0])
            _m = MInst("float.unary", 10, _fields, defs=[op.dest], uses=[src])
            if _fields.get("requant_stage"):
                _m.fields.update(requant_fixed_defs=(0,), requant_fixed_uses=(0,))
            out.append(_m)
        elif k in UNARY_OPCODE:
            # ONE-OPERAND INTEGER OPS. Named by the peer's isolation sweep and encoded through
            # g17asm.encode_unary, whose destination is ALU_DEST and whose source is a per-form
            # field - the eight-byte `not` puts its top two source bits in byte6 where the
            # ten-byte forms use byte8, so the map is per opcode and never shared.
            opc = UNARY_OPCODE[k]
            src = op.args[0] if _NO_UNARY_LOAD_WAIT else _wait_for_load(out, op.args[0])
            out.append(MInst("unary", g17asm.UNARY_FORM[opc][1], dict(opcode=opc, hazard=0),
                             defs=[op.dest], uses=[src]))
        elif k in BITWISE_OPS:
            # RESOLVED 2026-09-04. The form that had no operands was the wrong form: chosen by
            # family name, 8 bytes, and not what Apple emits for `x & 15u` at all. Apple emits a
            # TEN-byte class-b instruction whose opcode names the operation - 423 and, 13574 or,
            # 17770 xor - and whose destination, source register and 8-bit immediate are all
            # located. ledger/g17-alu-slot-model.toml, isa/g17-opmap.toml.
            a, b = op.args
            if not isinstance(a, ir.Value):
                raise Unsupported("%s with an immediate first operand" % k)
            if not isinstance(b, ir.Imm):
                # CERTIFIED NOW. decode_bitwise_reg existed all along; selfcheck was routing
                # this form to decode_alu_form, which has no entry for op424/op13575/op17771 and
                # raises, so every register-register bitwise failed its round trip.
                #
                # AND IT DOES NOT WAIT FOR A LOAD, measured both ways: `x & y` straight off two
                # loads returns zero for every thread, and the same kernel with one `add 0` on each
                # operand returns 2, 0, 16, 24, 6 - which is also the first execution evidence that
                # the four-byte operand layout (BW_R_DEST/SRCA/SRCB) is right. byte0[3] is the wait
                # on every other form here and patching it on this one changes nothing, so its wait
                # bit is unknown rather than unset, and guessing at a four-byte form's spare bits is
                # how an instruction becomes a different instruction.
                # IT RELEASES ITS SOURCES AND CANNOT SAY OTHERWISE. Measured: two of these reading
                # the SAME two registers gives a wrong answer, and two reading DISJOINT registers
                # match Apple exactly on every element. The source lifetime is an operand elsewhere
                # in this ISA - 32 keep, 16 release - and this four-byte form has nowhere to put it,
                # so a value with a later reader cannot go through it.
                # This is the fifth time an inherited lifetime has made an authored program read
                # something that was already gone. memory g17-modifier-operand-lifetimes
                # ISOLATE FIRST, then check. The checks below read the operands that are
                # actually emitted, so an isolated program passes them on its merits rather than
                # by having the check skipped.
                if not _NO_BITWISE_ISOLATION:
                    alias = isinstance(a, ir.Value) and a is b
                    a = _isolate_bitwise_operand(out, a, force=alias, keep_source=alias)
                    b = _isolate_bitwise_operand(out, b, force=alias)
                reread = [v for v in (a, b) if isinstance(v, ir.Value) and v in _MULTI_USE]
                if reread:
                    raise Unsupported("%s of two registers would release %s, which %s read again "
                                      "later: the four-byte op%d has no lifetime operand, so the "
                                      "next reader gets a register this instruction freed. "
                                      "Measured - two of these on the same pair give a wrong "
                                      "answer, two on disjoint pairs match Apple exactly"
                                      % (k, ", ".join(str(v) for v in reread),
                                         "is" if len(reread) == 1 else "are",
                                         BITWISE_REG_OPCODE[k]))
                if _load_wait([a, b]):
                    raise Unsupported("%s of two registers takes an operand straight from a load, "
                                      "and the four-byte op%d has no known wait bit - byte0[3] is "
                                      "the one every other authored form uses and it was measured "
                                      "not to work here. Put the operands through an ALU op first; "
                                      "those carry the wait" % (k, BITWISE_REG_OPCODE[k]))
                out.append(MInst("bitwise.reg", 4,
                                 dict(opcode=BITWISE_REG_OPCODE[k], hazard=_hz([a, b])),
                                 defs=[op.dest], uses=[a, b]))
                continue
            if not 0 <= b.v <= 0xFF:
                raise Unsupported("%s immediate %d exceeds the 8-bit slot" % (k, b.v))
            out.append(MInst("bitwise.imm", 10, dict(opcode=BITWISE_OPCODE[k], imm=b.v, hazard=_hz(op.args)),
                             defs=[op.dest], uses=[a]))
        elif k == "sub":
            # SUB IS A SLOT ORDER, not a byte6 value. Every measured form of this family computes
            # dest = slotA - slotB, so which slot holds the immediate IS the direction, and Apple
            # gives the two directions different opcodes:
            #     11666   reg - imm      11664   imm - reg      11667   reg - reg
            # The old lowering wrote add's byte6 as 0xa2 and left the operands where add puts
            # them - immediate in slot A, register in slot B - so it computed K - x. Executed,
            # (t+10)-3 returned -10, which is 0 - 10. g17asm.ALU_FORM, isa/g17-opmap.toml.
            a, b = op.args
            if not isinstance(a, ir.Value):
                raise Unsupported("sub with an immediate minuend is opcode 11664, whose template "
                                  "is recovered but which the IR has no way to reach yet")
            if isinstance(b, ir.Imm):
                if not 0 <= b.v <= 0xFF:
                    raise Unsupported("sub immediate %d exceeds the 8-bit slot" % b.v)
                out.append(MInst("alu.sub.imm", 12, dict(opcode=11666, imm=b.v, hazard=_hz(op.args)),
                                 defs=[op.dest], uses=[a]))
            else:
                out.append(MInst("alu.sub.reg", 12, dict(opcode=11667, hazard=_hz(op.args)),
                                 defs=[op.dest], uses=[a, b]))
        elif k in SHIFT_OPCODE:
            a, b = op.args
            if not isinstance(a, ir.Value):
                raise Unsupported("%s of an immediate by a register" % k)
            imm_op, reg_op = SHIFT_OPCODE[k]
            if isinstance(b, ir.Imm):
                if not 0 <= b.v <= 0xFF:
                    raise Unsupported("%s amount %d exceeds the 8-bit slot" % (k, b.v))
                out.append(MInst("alu.shift.imm", 14, dict(opcode=imm_op, imm=b.v, hazard=_hz(op.args)),
                                 defs=[op.dest], uses=[a]))
            else:
                out.append(MInst("alu.shift.reg", 14, dict(opcode=reg_op, hazard=_hz(op.args)),
                                 defs=[op.dest], uses=[a, b]))
        elif k == "mulhi":
            # THE UNSIGNED WIDENING MULTIPLY, op10793/12, and the first ALU result in this backend
            # whose destination is a REGISTER PAIR. `GPR32tup2` means one instruction writes the
            # full 64-bit product across two consecutive registers, so the lowering reserves a pair
            # the way a vector load does - `tuple_group` with two defs - rather than a register.
            #
            # THE PAIR INDEX IS THE FIRST REGISTER, NOT TWICE IT. Measured over Apple's 654
            # instances: read the destination as registers (105+k, 105+k+1) and only 20 of 654 have
            # neither register used anywhere in their program; read it as (105+2k, 105+2k+1) and
            # 338 of 654 have neither. The field itself carries 2k, which `_caps` already halves
            # for a `slot` domain, so nothing here scales it a second time.
            #
            # WHICH LANE IS THE HIGH WORD IS NOT DECIDED HERE. ir.Builder.mulhi returns both, the
            # evidence for lane 1 is recorded there, and a program that stores both settles it on
            # silicon. This arm places the pair; it does not name a half.
            a, b = op.args
            for v in (a, b):
                if isinstance(v, ir.Imm):
                    raise Unsupported(
                        "op10793/12 takes two REGISTER sources: its operand 4 has an expression "
                        "arm witnessed on 4 of 649 rows and refuted there, and no immediate "
                        "carrier is located at this length. Materialise the constant first")
            # _CUR_FN rather than `fn`: this arm sits above the point where _select_op binds a
            # local of that name, so the bare name is unbound here and bound 780 lines below.
            lanes = [o for blk in _CUR_FN[0].blocks for o in blk.ops
                     if o.kind == "vec_lane" and o.args[0] is op.dest]
            if len(lanes) != 1:
                raise Unsupported(
                    "op10793/12 defines a register PAIR and this mulhi has %d companion lane(s), "
                    "not one. ir.Builder.mulhi makes the pair; a bare `mulhi` op does not describe "
                    "the second register and the allocator would leave it unreserved"
                    % len(lanes))
            out.append(MInst("auth", 12, dict(opcode=10793, tuple_group=True),
                             defs=[op.dest, lanes[0].dest], uses=[a, b]))
        elif k == "mul":
            # MUL HAS ITS OWN OPCODES AND ITS OWN LENGTH: 10822 reg,imm and 10825 reg,reg, both
            # FOURTEEN bytes. The old lowering wrote byte6 = 0xa1 into a 12-byte add template,
            # which Apple's decoder reads as a truncated instruction - it consumed two bytes of
            # whatever followed. `t*1` then `+10` stored nothing at all under it.
            a, b = op.args
            if not isinstance(a, ir.Value):
                raise Unsupported("mul with an immediate in slot A")
            if isinstance(b, ir.Imm):
                if b.v == 1:
                    # Apple folds `x * 1` away entirely. The IR still wants a definition, so it
                    # becomes the identity add, a lowering that is already executed.
                    out.append(MInst("alu.12", 12,
                                     dict(op=ALU_OP["add"], mode=g17asm.MODE_IMM, imm=0, srcb_w=1,
                                          src1_w=_width(a.type), dest_w=_width(op.dest.type),
                                          scale_code=SCALE_CODE[1],
                                          load_wait=_load_wait(op.args)),
                                     defs=[op.dest], uses=[a]))
                elif 0 <= b.v <= 0xFF:
                    out.append(MInst("alu.mul.imm", 14, dict(opcode=10822, imm=b.v, hazard=_hz(op.args)),
                                     defs=[op.dest], uses=[a]))
                else:
                    raise Unsupported("mul immediate %d exceeds the 8-bit slot; a wider constant "
                                      "needs materialising into a register first" % b.v)
            else:
                out.append(MInst("alu.mul.reg", 14, dict(opcode=10825, hazard=_hz(op.args)),
                                 defs=[op.dest], uses=[a, b]))
        elif k == "uniform_load":
            # THE PRELOAD (handoff 10aa, folded in 10ab): recorded for the constant program, emitting nothing in
            # main - main reads the published word as the block operand of ONE add (alu.block). Apple folds
            # a[0] + b[0] + c[0] into one published word (S2/S3): every term is element 0 of a distinct buffer,
            # consumed by one add of a linear chain (_fold_uniform_chains); which declarations the argument-table
            # entry law is witnessed for is the constant program's refusal (g17uniformpreload.constant_program_folded).
            buf, idx = op.args; fn = _CUR_FN[0]
            if not _uses_texture(fn):
                raise Unsupported("uniform_load outside a texture kernel: the argument-table entry law (units 8/10 for the first preload) is witnessed in texture kernels only")
            if not isinstance(idx, ir.Imm) or idx.v != 0:
                raise Unsupported("uniform_load at element %r: the prologue load's offset field is not located (S1 carries 0, M4 carries 1604 - M4 is the reproducer)" % (idx.v if isinstance(idx, ir.Imm) else idx,))
            if any(p["slot"] == buf.slot for p in _PRELOADS[0]):
                raise Unsupported("uniform_load of buffer %d twice: one term per buffer (each term is one argument-table entry read once)" % buf.slot)
            if op.dest not in _FOLD[0]["terms"]:
                raise Unsupported("uniform_load's value exists only as a term of ONE folded add chain (a[0] + b[0] + c[0] with the fetch); this one is not")
            _PRELOADS[0].append(dict(slot=buf.slot, element=0, value=op.dest))
        elif k in ALU_OP and (op in _FOLD[0]["skip"] or op in _FOLD[0]["root"]):
            # dest = src1 + block[const]: the add whose block operand is the published word (op10282, S1 main +20);
            # const = 4 x the declared binding records (internals first), the block byte the prologue publishes to.
            # Of a folded chain only the ROOT emits, with the chain's base register as src1; the intermediate adds
            # are the constant program's work and emit nothing (their values exist only inside the chain).
            if op in _FOLD[0]["skip"]: return out
            reg = _FOLD[0]["root"][op]
            if reg in _PHI_MEMBERS: raise Unsupported("alu.block src1 is a phi-group member: copy it through an ALU first")
            records = max(_BUF_RANK[0].values()) + 1        # the declared records, internals included: the rank map already carries them (S1: {44:0, 48:1, 0:2, 1:3} -> 4)
            out.append(MInst("alu.block", 12, dict(const=4 * records, opcode=10282, keep=1 if reg in _MULTI_USE else 0), defs=[op.dest], uses=[reg],
                             note="dest = src1 + block[%d]: the preloaded word at 4 x %d declared records" % (4 * records, records)))
            if reg.op is not None and reg.op.kind == "texture_read": _FETCH_DERIVED.add(op.dest)
        elif k in ALU_OP:
            a, b = op.args
            if not isinstance(a, ir.Value): raise Unsupported("%s with immediate src1" % k)
            # CLEAR THE INHERITED SHIFT-ADD SCALE. alu.shiftadd.imm shares this form, and its
            # scale lives in byte10/byte11 as a TABLE (x1 = 0b1000, x2 = 0b1010, x4 = 0b1011,
            # x8 = 0b0001, x16 = 0b0000). A template harvested from a host that computed x*n+m is
            # a strength-reduced multiply and carries a non-x1 scale; a plain add that does not
            # write the field inherits it and silently multiplies.
            #
            # Measured: load B[2] then add 0x11 returned 0x81BC0015, which is B[2]*2 + 0x11, and
            # B[3] + 0x2A returned B[3]*2 + 0x2A. The template's scale was x2.
            # ledger/g17-inherited-scale-multiplied-the-operand.toml
            f = dict(op=ALU_OP[k], src1_w=_width(a.type), dest_w=_width(op.dest.type),
                     scale_code=SCALE_CODE[1], load_wait=_load_wait(op.args),
                     b4_5=_b45(), b0_5=_b05())
            uses = [a]
            if isinstance(b, ir.Imm) and not 0 <= b.v <= 0xFF:
                # A LITERAL THE FORM CANNOT HOLD IS MATERIALISED, NOT REFUSED. The immediate slot
                # is eight bits; `base + 256` - the LayerNorm workload indexing the last 128 of
                # its 384 columns - is one past it. Refusing here made the IR author's choice of
                # spelling (a literal against a b.const) decide whether a program compiles, which
                # is the selector's job and not the author's. This emits exactly what b.const()
                # would - a movimm.8 into a fresh value, then the register form - so the two
                # spellings produce identical bytes, and the diagnostic --materialized-control
                # becomes the ordinary path rather than a workaround.
                lit = ir.Value(ir.I32, "%s_lit%d" % (getattr(op.dest, "name", "v") or "v", b.v))
                out.append(MInst("movimm.8", 8, dict(imm=b.v), defs=[lit]))
                b = lit
            if isinstance(b, ir.Imm):
                f.update(mode=g17asm.MODE_IMM, imm=b.v, srcb_w=1)
            else:
                f.update(mode=g17asm.MODE_REG, srcb_w=_width(b.type)); uses.append(b)
            out.append(MInst("alu.12", 12, f, defs=[op.dest], uses=uses))
        elif k == "shiftadd":
            a, b = op.args; s = op.attrs["scale"]
            if s not in SCALE_CODE: raise Unsupported("shiftadd scale %r (table has %s)"
                                                      % (s, sorted(SCALE_CODE)))
            out.append(MInst("alu.12", 12,
                             dict(op=ALU_OP["add"], mode=g17asm.MODE_REG, scale_code=SCALE_CODE[s],
                                  src1_w=_width(a.type), srcb_w=_width(b.type),
                                  dest_w=_width(op.dest.type),
                                  load_wait=_load_wait(op.args)),
                             defs=[op.dest], uses=[a, b]))
        elif k == "load_hi16":
            # A HALF LOAD INTO A HIGH HALF (ir.load_hi16): op12646 with hi16 = 1 writes RdH and leaves RdL, so
            # the destination is TIED to prev (uses[1]) and prev's low half is the result's. uses[0] stays the
            # index, which is what the load emitter and its read-back take.
            buf, idx, prev = op.args
            if not isinstance(idx, ir.Value): raise Unsupported("load_hi16 with immediate index")
            if not _NO_LOAD_INDEX_LOADWAIT:
                idx = _wait_for_load(out, idx)
            out.append(MInst("load.14", 14,
                             dict(base=4 * _buf_rank(buf.slot), offset=0, disp2=0, index_scale=1,
                                  narrow=0, half=1, hi16=1, tie_use=1, tie_max=g17asm.LOAD_DEST_MAX),
                             defs=[op.dest], uses=[idx, prev], note="index_reg; high-half destination tied to prev"))
        elif k == "load":
            buf, idx = op.args
            if not isinstance(idx, ir.Value): raise Unsupported("load with immediate index")
            # AN INDEX THAT IS ITSELF LOADED is a late value like any other: `a[b[t]]` read the
            # second load's index before the first had landed (tools/g17indirectload.py). The
            # measured waiting copy, as for every ALU and store consumer.
            if not _NO_LOAD_INDEX_LOADWAIT:
                idx = _wait_for_load(out, idx)
            at = op.attrs
            if at.get("scale", 1) not in (1, 2):
                raise Unsupported("load index scale %r; byte7[5] encodes x1 and x2 only"
                                  % at.get("scale"))
            if not 0 <= at.get("disp", 0) <= 3:
                raise Unsupported("load disp %r; byte4[6:5] holds 0..3" % at.get("disp"))
            # BUFFER SLOT IS NOT THE BASE REGISTER. base is 4 * a BASE REGISTER INDEX
            # (isa load.base), and which register holds which buffer's address is set up before
            # the entry point by the descriptors - it is not the buffer's binding slot. The host
            # reads buffer 1 through base register 0. Until the mapping is recovered the compiler
            # takes it from the caller rather than assuming identity.
            # ledger/g17-buffer-slot-is-not-base-register.toml
            # THE LOAD'S BASE IS THE RANK, MEASURED. It used to be BUFFER_BASE_REG with an
            # IDENTITY FALLBACK, whose one entry was slot 1 -> 0 - and slot 1 was rank 0 in every
            # kernel that had it, so the measurement could not tell "the slot number" from "the
            # rank" apart. It survived because no validated kernel ever loaded from a SECOND
            # buffer: all 29 with buffer loads read base 0 only. A three-buffer FP16 program is
            # the first to ask, and the fallback sent its query load to base 8 where the binding
            # list has it at 4 - a silent read of the wrong allocation.
            #
            # Two of Apple's own kernels settle it, one variable apart: the same program at public
            # indices [1,2,3] and at [2,4,6] emits byte-identical bases - loads at const 0 and 4,
            # store at const 8. The base tracks the RANK and not the public index.
            # results/g17-abi-class-probes-indexed in the integration branch holds both objects.
            requested_length = at.get("form_length")
            if requested_length == 8:
                if at.get("width", "word") != "word":
                    raise Unsupported("load form_length=8 is only measured for word loads; width %r needs a witnessed form" % at.get("width"))
                if at.get("shift16"):
                    raise Unsupported("load form_length=8 has no byte4 hi16 field; use the measured 14-byte form")
                if not 0 <= at.get("offset", 0) < 64:
                    raise Unsupported("load form_length=8 offset %r needs the 14-byte form's high displacement byte" % at.get("offset"))
                out.append(MInst("load.8", 8,
                                 dict(base=4 * _buf_rank(buf.slot),
                                      offset=at.get("offset", 0),
                                      disp2=at.get("disp", 0), index_scale=at.get("scale", 1),
                                      narrow=0, half=0, hi16=0),
                                 defs=[op.dest], uses=[idx], note="index_reg; measured short form"))
            elif requested_length == 10:
                if at.get("width") != "half":
                    raise Unsupported("load form_length=10 is measured only for half loads; width %r" % at.get("width"))
                if at.get("narrow") or at.get("shift16"):
                    raise Unsupported("load form_length=10 has no measured narrow/hi16 control")
                if at.get("scale", 1) != 1:
                    raise Unsupported("load form_length=10 has no measured index-scale control")
                if not 0 <= at.get("offset", 0) < 64:
                    raise Unsupported("load form_length=10 offset %r needs the measured low-displacement range" % at.get("offset"))
                out.append(MInst("load.10", 10,
                                 dict(base=4 * _buf_rank(buf.slot),
                                      offset=at.get("offset", 0),
                                      disp2=at.get("disp", 0), index_scale=at.get("scale", 1),
                                      narrow=0, half=1, hi16=0),
                                 defs=[op.dest], uses=[idx], note="index_reg; measured half short form"))
            else:
                _fields = dict(base=4 * _buf_rank(buf.slot),
                                      offset=at.get("offset", 0),
                                      disp2=at.get("disp", 0), index_scale=at.get("scale", 1),
                                      narrow=1 if at.get("width") == "byte" else 0,
                                      half=1 if at.get("width") == "half" else 0,
                                      hi16=1 if at.get("shift16") else 0)
                if op.attrs.get("requant_stage"):
                    from agxforge.g17 import requantenc
                    if op.attrs.get("requant_stage_step") != "load":
                        raise Unsupported("requantization stage has an unexpected load step")
                    _fields.update(raw=requantenc.stage_bytes(bool(op.attrs.get("requant_signed")), "load"),
                                   requant_stage=True)
                _m = MInst("load.14", 14, _fields,
                           defs=[op.dest], uses=[idx], note="index_reg")
                if _fields.get("requant_stage"):
                    _m.fields.update(requant_fixed_defs=(0,), requant_fixed_uses=(1,))
                out.append(_m)
        elif k == "simd_vote_pair":
            # THE SAME TWO-HALF PROLOGUE THE IMAGEBLOCK USES, on a different pair of registers:
            # SR_PVSIMD into the half-register file based at 425 and SR_TVSIMD into the one at 281,
            # both at the index this value is allocated to. If those files are the low and high
            # halves of one 32-bit register, this value reads (TVSIMD << 16) | PVSIMD.
            out.append(MInst("read_sr.4", 4, dict(sr=SR["SR_PVSIMD"], seq=0, half=0),
                             defs=[op.dest]))
            out.append(MInst("read_sr.4", 4, dict(sr=SR["SR_TVSIMD"], seq=0, half=1),
                             defs=[op.dest], uses=[op.dest]))
        elif k == "texture_read":
            # A TEXTURE READ IS FIVE INSTRUCTIONS, AND THE COORDINATE TRAVELS THROUGH A RELAY. The
            # read itself takes NO coordinate operand - two reads of one texture at different
            # coordinates compile to byte-identical instructions (tu-rep4, first and third) - so
            # the texture unit takes its coordinate from wherever the publishes put it.
            #
            # THE COORDINATE SLOTS ARE [op4+0*4] AND [op4+2*4], and getting a register into them
            # took three wrong answers. Apple's ty-2d publishes a register into [op0+12*4] and
            # [op0+14*4] from its CONSTANT program and then relays those two slots into op4 from
            # its main program - but its coordinate is a uniform, u[0] and u[1]. Copying that
            # relay dispatches and returns Apple's texel, not this compiler's: a main program's
            # publish into [op0+12*4] does not land, measured against Apple's own constant program
            # writing 3 and 2 into the same slots.
            #
            # A PER-LANE COORDINATE IS NOT PUBLISHED AT ALL. Apple's own per-lane read - probe
            # ty-2d-perlane-slot, built for this question - has NO publish: the `and` and the
            # shift WRITE THEIR DESTINATION straight into [op4+0*4] and [op4+2*4], through
            # opcodes 444 and 17070, which are the expr-destination siblings of the register-
            # destination 423 and 17013 this compiler emits. Those two opcodes appear nowhere in
            # the operand maps, so authoring them is real work and this does something smaller:
            # op592 WILL take a register source with an op4 destination. The first attempt at that
            # was hand-patched behind g17as and wrote an inconsistent instruction - operand 2's
            # pinned imm is 16777216 for an op0 destination and 1048576 for an op4 one, and the
            # patch kept the op0 value - so the fetch ran and every lane read texel (0, 0). With
            # operand 2 right it authors through g17as and Apple's decoder reads back exactly
            # `[op4+0*4] <- reg`. Apple never emits it, so it is CONSTRUCTED, and the dispatch
            # below is what makes it a fact rather than a decode.
            xv, yv = op.args
            for v in (xv, yv):
                if isinstance(v, ir.Imm):
                    raise Unsupported("texture_read coordinate as an immediate: publish takes its "
                                      "source in a register, so materialise the literal first")
            # op592/4 has no measured load-wait field. Loaded coordinates must
            # first pass through the same causally measured ALU wait used by
            # range-store members. Otherwise v3's nonzero coordinates arrived
            # at the following texture read; the zero-coordinate cases hid it.
            if xv is yv:
                xv = yv = _wait_for_load(out, xv)
            else:
                xv = _wait_for_load(out, xv)
                yv = _wait_for_load(out, yv)
            out.append(MInst("publish.coord.x", 4, dict(), uses=[xv],
                             note="publishes the x coordinate the texture unit reads"))
            out.append(MInst("publish.coord.y", 4, dict(), uses=[yv],
                             note="publishes the y coordinate"))
            out.append(MInst("texture.read.32", 8, dict(tex=op.attrs.get("tex", 0),
                                                        element=op.attrs.get("element", "uint32")),
                             defs=[op.dest],
                             note="op7 is a DENSE index over the textures this function uses, "
                                  "not the Metal binding index"))
        elif k in ("imageblock_write", "imageblock_read"):
            # THE COORDINATE REGISTER IS BUILT, NOT BORROWED. Every imageblock function Apple
            # emits opens with two read_sr writing the two 16-bit halves of one register, and the
            # access names that same register - 47 of 47. So the prologue is emitted here, once
            # per function, and the access uses its value; nothing is inherited and no register
            # number is assumed.
            coord = _IB_COORD[0]
            if k == "imageblock_read" and op.attrs.get("explicit"):
                # AN EXPLICIT COLUMN, row 0: the value is the packed coordinate (x low, y = 0 high).
                # Apple's own neighbour reads, tensor or not, compute it and read with no offset.
                (coord,) = op.args
                coord = _wait_for_load(out, coord)   # a loaded coordinate is late (the "load" branch)
                f = dict(member=op.attrs.get("member", 0), dx=0, dy=0, explicit=True)
                out.append(MInst("load.ib.32", 14, f, defs=[op.dest], uses=[coord]))
                continue                 # the body is `for op in (op,)`, which returns `out` after it
            if coord is None:
                coord = _IB_COORD[0] = ir.Value(ir.I32, "ibcoord")
                out.append(MInst("read_sr.4", 4,
                                 dict(sr=SR["thread_position_in_threadgroup"] + SR_AXIS["x"],
                                      seq=0, half=0), defs=[coord]))
                out.append(MInst("read_sr.4", 4,
                                 dict(sr=SR["thread_position_in_threadgroup"] + SR_AXIS["y"],
                                      seq=0, half=1), defs=[coord], uses=[coord]))
            f = dict(member=op.attrs.get("member", 0), dx=op.attrs.get("dx", 0),
                     dy=op.attrs.get("dy", 0))
            if k == "imageblock_write":
                (val,) = op.args
                if isinstance(val, ir.Imm):
                    raise Unsupported("imageblock_write of an immediate: the store takes its "
                                      "value in a register and Apple materialises a literal first")
                # A LOADED VALUE IS STORED ONLY AFTER IT LANDS. op13075's operand 1 bits 24-31 are the
                # store's WAIT MASK (bit 24+s = wait on slot s), the consumer mask general stores
                # carry - MEASURED 2026-09-23 by Piece B on Apple's own kernel: waiting on the stored
                # value's load slot 32/32, no wait 0/32, a wrong slot 0/32 (isa/g17-execution-ib-store-
                # wait-results.json). This store named only slot 0 (the coordinate's read_sr), so a
                # value straight from a load was written before it arrived. It goes through the same
                # measured waiting copy every other consumer of a late value uses.
                val = _wait_for_load(out, val)
                out.append(MInst("store.ib.32", 14, f, uses=[val, coord]))
            else:
                out.append(MInst("load.ib.32", 14, f, defs=[op.dest], uses=[coord]))
        elif k == "atomic_uniform":
            # OPERAND 6 IS THE ADDRESS'S BYTE OFFSET, NOT A SOURCE LIFETIME (CORRECTED 2026-09-25, MM
            # 25.140.3). This comment used to say op6=16 "does not write memory" and op6=0 works. A
            # sentinel window read back after the atomic shows where it goes: op6 = 4, 8, 16 add to
            # words 1, 2, 4 (1 and 2 round down to word 0), so op6=16 wrote word 4 and the old probe
            # watched only word 0. Pinning op6 to 0 remains right for an atomic on word 0; `slot6`
            # (ir.atomic_uniform) writes other offsets, read back through Apple's decoder. The same
            # probe measured that this "uniform" form runs once PER LANE (+32, 32 distinct old values).
            buf, val = op.args
            if isinstance(val, ir.Imm):
                raise Unsupported("atomic_uniform has no immediate addend - materialise it first")
            aop = op.attrs.get("aop", "add")
            if aop not in ir.Builder.ATOMIC_OPS:
                raise Unsupported("atomic operation %r" % aop)
            code = ir.Builder.ATOMIC_OPS[aop]
            if code not in (0, 1):
                raise Unsupported("op10094 is lowered at ten bytes, which carries add and and "
                                  "only - the length is part of the operation encoding")
            # THE RETURNED OLD VALUE ARRIVES LATE, LIKE A LOAD'S. cc read it at once, and a residency
            # probe's arrival records came back 1 for every simdgroup while its departure records
            # showed all 256 present (results/g17-tensor-resid-v2): the consumer read the register's
            # previous contents. Apple's compiler makes the first consumer of op10094's result wait
            # on slot 0 (operand-1 bit 24) in 120 of 120 corpus instances. Marking the value late
            # routes every consumer through cc's load-use wait (_hz / alu.12 load_wait) - WHICH IS
            # THE SLOT-7 BIT, and the witness's operand 1 published this atomic on slot 0, so that
            # wait did not cover it: the arrival records stayed 1 at S = 256 (resid-v3). The
            # encoder now publishes on slot 7 (_atomic_fill_slot7); measured 2026-09-24, 3 of 3
            # rounds at S = 256 each way: slot 7 arrivals 1..256, slot 0 arrivals all 1.
            op.attrs["is_load"] = True
            out.append(MInst("atomic.uniform.10", 10,
                             dict(base=_atomic_base(buf.slot),
                                  offset=op.attrs.get("offset", 0), aop=code, slot6=op.attrs.get("slot6", 0)),
                             defs=[op.dest], uses=[val]))
        elif k == "atomic_tg_uniform":
            if not _HAS_TG_ACCESS[0]:
                raise Unsupported("a threadgroup atomic needs the threadgroup region that an "
                                  "ordinary threadgroup load or store allocates - without one it "
                                  "authors correctly and writes nothing. Measured by removing a "
                                  "store from a working kernel and putting it back")
            (val,) = op.args
            if isinstance(val, ir.Imm):
                raise Unsupported("atomic_tg_uniform has no immediate addend")
            # THE OPERATION IS A FIELD AFTER ALL, and this comment used to say it was not.
            # CORRECTED 2026-09-08 by compiling all seven from source. Apple emits every one of
            # them as op11765 AT LENGTH 12 - there is no 16-byte form and no second opcode - and
            # they differ in OPERAND 2:
            #
            #     add 262656   sub 262657   min 262660   max 262662
            #     and 262664   or  262665   xor 262666
            #
            # So "the device family's three-bit field lands on an opcode bit here" was true about
            # the bits it named and wrong about the conclusion: the operation is selected somewhere
            # else in the same instruction.
            #
            # THE KEY WAS ONE BIT SHORT. Operand 2's table key was byte4[5], byte6[3], byte11[3],
            # byte11[4], and those four do not separate the seven: min and xor differ in NOTHING
            # they look at, so both key 1101. byte11[3] is 0 in all seven witnesses - a dead bit
            # occupying a slot in a four-wide key that therefore only carried three.
            #
            # The missing bit is byte5[3], and it was findable only by changing the instrument.
            # The seven shipped witnesses differ in their destination register and their source
            # offset as well as their operation, so a bit that tracks the operation across seven
            # programs may be tracking the program; fitting from them is what produced the dead
            # bit twice. Flipping one bit at a time in ONE witness and asking Apple's decoder gives
            # the field directly - and all seven operations then reconstruct from that single base,
            # each constructed key equal to its own witness's. The device atomic op10018 at length
            # 12 already carried this exact key, which is corroboration from a population this one
            # was never compared against. isa/g17-operand-maps-atomics.jsonl.
            aop = op.attrs.get("aop", "add")
            # THE RETURNED OLD VALUE ARRIVES LATE, as op10094's does (see atomic_uniform): Apple's
            # first consumer of op11765 waits on the atomic's slot in 10 of 10 corpus instances, and
            # cc's never waited - tgatomic's op11667 read both returns with an empty wait mask
            # (tensorview.hazards, once it modelled atomics). Marked late, consumers take cc's
            # load-use wait, which covers slot 7, where _atomic_fill_slot7 now publishes.
            op.attrs["is_load"] = True
            out.append(MInst("atomic.tg.uniform.12", 12,
                             dict(aop=ir.Builder.ATOMIC_OPS[aop], op2=TG_ATOMIC_OP2[aop]),
                             defs=[op.dest], uses=[val]))
        elif k == "simd_broadcast":
            (val,) = op.args
            # op14157 IS simd_broadcast(x, CONSTANT lane), not broadcast_first (MM 25.144.6). Measured on a
            # region-mode fuzz program: inside a region whose lane 0 was inactive, every active lane got 0
            # (fma(x, x, broadcast(x)) came back exactly a*b, 11 of 11 mismatched lanes). Apple compiles
            # simd_broadcast_first in divergent code as a shuffle BY REGISTER LANE: op14060 reads SR 38 (the
            # active-lane mask), op14214 finds its lowest set lane, op9989 narrows it, and op14158 shuffles
            # from that lane. That sequence is not lowered here, so a broadcast inside a predicated region is
            # refused rather than silently reading lane 0; uniform code (lane 0 active) keeps op14157.
            if _REGION_DEPTH[0] > 0:
                raise Unsupported("simd_broadcast_first inside a predicated region: op14157 broadcasts lane 0, "
                                  "which may be inactive there (returns 0). The first-active form (SR 38 -> "
                                  "op14214 -> op9989 -> op14158) is not lowered; hoist the broadcast out of the "
                                  "region (MM 25.144.6)")
            # a LATE operand waits, like every other consumer's - see simd_shuffle_xor below
            val = _wait_for_load(out, val)
            out.append(MInst("simd.broadcast.10", 10, {}, defs=[op.dest], uses=[val]))
        elif k == "simd_shuffle_xor":
            (val,) = op.args
            mask = op.attrs.get("lane_mask")
            if mask not in (1, 2, 4, 8, 16):
                raise Unsupported("simd_shuffle_xor lane mask %r is outside the measured one-bit "
                                   "butterfly masks (1, 2, 4, 8, 16)" % (mask,))
            if getattr(op.dest, "type", None) is not ir.F32:
                raise Unsupported("simd_shuffle_xor is measured only for FP32 values")
            # op14169's operand map is complete: destination slot 0, source slot 2, fixed
            # control words 1 and 3, and the XOR lane mask in slot 4.  The latter is the
            # compiler-owned degree of freedom; all other values are the measured witness
            # controls, not defaults inferred from tuple width.
            # A LOADED OPERAND IS READ BEFORE IT LANDS, and op14169 has no wait of its own that this
            # side writes. MEASURED 2026-09-23 on Piece A's simdgroup-combine probe (64 and 128
            # threads): a load straight into the first shuffle gave every lane one wrong value - its
            # max came out below every simdgroup's own maximum - and the same program with the load
            # through a waiting copy was exact on 64/64 and 128/128 in all four arms. So a late
            # operand goes through the measured waiting copy (alu.12 byte0[3]) first.
            val = _wait_for_load(out, val)
            out.append(MInst("auth", 10,
                             dict(opcode=14169, imms={1: 32, 3: 32, 4: mask}),
                                  defs=[op.dest], uses=[val]))
        elif k == "atomic_add":
            buf, idx, val = op.args
            if not isinstance(idx, ir.Value):
                raise Unsupported("atomic_add needs its index in a register: op10090 is the "
                                  "PER-LANE form and a uniform address selects op10094 instead")
            if isinstance(val, ir.Imm):
                raise Unsupported("atomic_add has no immediate addend - materialise it first; "
                                  "a literal also sets operand 1 bit 2^24, which the operand map "
                                  "cannot yet tell from 2^25 (ledger/g17-operand-1-has-three-bits"
                                  "-nobody-mapped.toml)")
            aop = op.attrs.get("aop", "add")
            if aop not in ir.Builder.ATOMIC_OPS:
                raise Unsupported("atomic operation %r" % aop)
            # THE LENGTH IS PART OF THE OPERATION ENCODING, and the decoder enforces it: authored
            # bytes with aop=7 at ten bytes are rejected outright, and aop=0 at twelve are too.
            # Measured by assembling each operation at each length and asking Apple's decoder:
            #
            #     add, and          ten bytes for discarded returns; the validated consumed ADD
            #                       uses a twelve-byte slot-7 form (pure IOGPU paired trial)
            #     max, min, or       either length decodes
            #     sub, xor           TWELVE bytes
            #
            # sub is the one the DECODER does not separate: aop=3 at ten bytes decodes as op10090
            # and computes the wrong thing - 12 sub 10 returned 0 rather than 2 - while at twelve
            # it is correct. Apple only ever emits integer sub at twelve bytes, which the factorial
            # showed and this reproduces from the other side. Decoding is not meaning.
            #
            # which is the same fact the corpus census showed as "the short encoding does not exist
            # for exchange, store or compare-exchange", seen from the emitting side.
            # ledger/g17-the-atomic-operation-is-a-field-and-a-length.toml
            code = ir.Builder.ATOMIC_OPS[aop]
            size = 12 if code in (3, 7) or op.attrs.get("consumed_result") else 10
            out.append(MInst("atomic.add.%d" % size, size,
                             dict(base=_atomic_base(buf.slot),
                                  offset=op.attrs.get("offset", 0), aop=code,
                                  consumed_result=bool(op.attrs.get("consumed_result"))),
                             defs=[op.dest], uses=[idx, val]))
        elif k == "atomic_cmpxchg":
            buf, idx, desired, expected = op.args
            if not op.attrs.get("consumed_result"):
                raise Unsupported("per-lane compare-exchange compiler path requires a direct old-value store")
            if not all(isinstance(x, ir.Value) for x in (idx, desired, expected)):
                raise Unsupported("per-lane compare-exchange needs index, desired and expected in registers")
            if _atomic_base(buf.slot) != 0:
                raise Unsupported("per-lane compare-exchange has only executed at atomic base 0")
            out.append(MInst("atomic.cmpxchg.12", 12, dict(base=0),
                             defs=[op.dest], uses=[idx, desired, expected]))
        elif k == "store":
            buf, idx, val = op.args
            if not isinstance(idx, ir.Imm):
                raise Unsupported("store to a computed index: the recovered store carries a "
                                  "SLOT, not an index register (isa store.device)")
            # TWO PRE-COLOURINGS CANNOT BOTH HOLD ONE VALUE. This store is a two-component RANGE
            # group - the value and a companion zero in consecutive registers - and a value that is
            # also a phi-group member is already pinned to its loop register by coalescing. When
            # the loop's exit stores its induction's final value (`k + U`, the latch of phi k),
            # the range group's base and the phi's register disagree and emit refuses with a
            # negative base slot. Exposed by re-indexing a 384-trip loop, latent before it: the
            # 255-trip reproducer stores the same kind of value and happened to fit. The value is
            # copied through an ALU first, the same way a special-register read is before a store
            # (_materialise_sr), so the range group owns a fresh register with no other claim.
            if val in _PHI_MEMBERS:
                copy = ir.Value(getattr(val, "type", ir.I32), "%s_st" % (getattr(val, "name", "v") or "v"))
                out.append(MInst("alu.12", 12,
                                 dict(op=3, mode=1, imm=0, src1_w=1, dest_w=1, srcb_w=1,
                                      scale_code=2, load_wait=0, b4_5=None, b0_5=None,
                                      keep=1 if val in _MULTI_USE else 0),
                                 defs=[copy], uses=[val],
                                 note="copy: a phi-group member cannot also be a range-store member"))
                val = copy
            # FORM SELECTION BY OPERAND RANGE. The 8-byte store has no byte13, so its slot
            # field is only 6 bits; slot 64 and above need the 14-byte form. Always choosing
            # store.8 made the mixed scalar+tensor rung raise "slot 64 needs the 14-byte store
            # form" - a real selection defect the ladder caught, not a bad test.
            # DO NOT LOWER A SCALAR STORE THROUGH byte4[6:5]=0.
            #
            # That encoding writes r<src> to slot k and r<src+1> to slot k+3 - measured by
            # executing generated code (ledger/g17-store-has-no-single-slot-form.toml). Emitting
            # it for a one-value store would corrupt whatever sits at k+3 with whatever happens
            # to be in the next register: two things the caller did not choose.
            #
            # Lower through the 2-component encoding instead, which IS confirmed to write exactly
            # k and k+1 from r<src> and r<src+1>, with a compiler-materialised ZERO as the
            # companion. The write to k+1 is then deterministic, adjacent, and stated - the
            # caller reserves one extra slot instead of losing an arbitrary one.
            if op.attrs.get("width") == "half":
                # ONE SIXTEEN-BIT ELEMENT AT AN IMMEDIATE ADDRESS: op17199, sub-form 01, and n=1 here is a
                # GENUINE one-element store rather than the word form's k/k+3 encoding - every one of the
                # seven retained witnesses writes exactly one half element (results/g17-halfslot-compiles-v1).
                # So this path needs no companion zero, which the word path below does.
                #
                # THE ADDRESS IS A BYTE DISPLACEMENT and `index` counts half elements, so an ODD element is
                # expressible: the decoder sweep locates the field completely at byte6[5:7] + byte7[0:4]
                # (+ byte13 at fourteen bytes) with bit 0 worth one byte. The word store's slot field is the
                # same bits from byte6[7] up at a four-byte scale, which cannot say "two bytes into word 3".
                disp = 2 * idx.v
                # A TEXTURE FETCH CANNOT REACH HERE AND THERE IS NO GUARD FOR IT, deliberately: a fetch is a
                # thirty-two-bit value, so Builder.store's own type check refuses it one level up. A second
                # check here would read as a measured refusal of something this path can see, and it cannot -
                # an unreachable guard is a claim about a case that does not exist.
                if disp > g17asm.HALFSLOT_MAX_DISP:
                    # THE FIELD IS SIGNED (see g17asm.HALFSLOT_MAX_DISP): a larger displacement would be read
                    # as negative and write before the buffer base.
                    raise Unsupported("half element %d is %d bytes out; op17199's displacement is SIGNED "
                                      "sixteen bits, so the largest expressible element is %d"
                                      % (idx.v, disp, g17asm.HALFSLOT_MAX_DISP // 2))
                val = _materialise_sr(out, val)
                from_load_h = _is_load_value(val)
                ln = 14 if disp > 0xFF else (10 if from_load_h else 8)
                out.append(MInst("store.half1.%d" % ln, ln,
                                 dict(disp=disp, wait_load=1 if from_load_h else 0,
                                      const=4 * _BUF_RANK[0].get(buf.slot, 0)), uses=[val],
                                 note="out[%d..%d] <- one half from r<425+src>%s"
                                      % (disp, disp + 1, ", waiting on the load" if from_load_h else "")))
                continue
            if not op.attrs.get("reserve_companion", True):
                # ONE THIRTY-TWO-BIT ELEMENT, RESERVING NOTHING: op17235, sub-form 01, the width twin of
                # op17199 (handoff 10ak; integration's f49d43fb). The default path below keeps the
                # two-component encoding and its companion zero, byte for byte, because flipping it would
                # move every constant-slot store this project has executed.
                disp = 4 * idx.v
                if isinstance(val, ir.Value) and val.op is not None and val.op.kind == "texture_read":
                    # op17235 IS the form measured to read a fetch (ledger/g17-only-one-consumer-can-read-a-
                    # texture-fetch.toml), but the admitted texture program stores through op17244 and the
                    # question of which is true on hardware is the one results/g17-texture-consumer-pair-v1
                    # exists to settle. Quietly switching the texel store here would answer it by fiat.
                    #
                    # AND BOTH CONSUMERS HAVE NOW BEEN ASKED. op17235 executed twice - coordinate [1,2]
                    # returning texel 201 and [3,1] returning 103, preregistered, guards preserved
                    # (results/g17-texture-query-runtime-v1). And op17244/14 was dispatched at (1,2) and
                    # READ THE FETCH: its receipt's stderr is `value=201 expected=201`, with the
                    # companion word 401 written 0. The run is recorded `failed` because the query
                    # protocol counted 401 as a boundary guard, not because the texel was wrong -
                    # gpu_dispatched is true and the diagnostics are unchanged
                    # (results/g17-op17244-receipt-v1).
                    #
                    # THAT IS ONE COORDINATE, ONE 4x4 R32Uint class, one thread, one form. It is NOT a
                    # general store rule and NOT a texture rule, and it says nothing about op17229,
                    # which is a different opcode and the subject of a separate tension (handoff
                    # 10cy) whose scope the linker narrowed at 4c0ca7ef: what was dispatched there
                    # is the alu-fed shape, and Apple emits the direct one. The
                    # refusal below stays: the one-element sub-form is still a different encoding from
                    # the two-component one that ran, and switching it here would still be by fiat.
                    raise Unsupported("a one-element store of a texture fetch: op17235 is the form measured "
                                      "to read a fetch, but the admitted texture program uses op17244 and "
                                      "results/g17-texture-consumer-pair-v1 is the preregistered pair that "
                                      "decides between them - use store_fetch, which states that choice")
                if disp > g17asm.HALFSLOT_MAX_DISP:
                    raise Unsupported("slot %d is %d bytes out; op17235's displacement is SIGNED sixteen "
                                      "bits, so the largest expressible slot is %d"
                                      % (idx.v, disp, g17asm.HALFSLOT_MAX_DISP // 4))
                val = _materialise_sr(out, val)
                from_load_w = _is_load_value(val)
                ln = 14 if disp > 0xFF else (10 if from_load_w else 8)
                out.append(MInst("store.elem1.%d" % ln, ln,
                                 dict(disp=disp, wait_load=1 if from_load_w else 0,
                                      const=4 * _BUF_RANK[0].get(buf.slot, 0)), uses=[val],
                                 note="out[%d] <- one word from r<105+src>, reserving NOTHING%s"
                                      % (idx.v, ", waiting on the load" if from_load_w else "")))
                continue
            zero = ir.Value(ir.I32, "storepad")
            out.append(MInst("movimm.8", 8, dict(imm=0), defs=[zero],
                             note="companion zero for the 2-component store"))
            # LOAD-USE HAZARD. A store reads its source register immediately; a load's result is
            # not there yet. The 14-byte store has byte9[5] - "wait for a pending LOAD into the
            # source register before storing" - and the 8-byte form does NOT have byte9 at all.
            # So a value that came from a load must be stored through the WIDE form with the wait
            # bit set. Measured: an 8-byte store of a freshly loaded value returns 0 for every
            # offset, while the host's own code has ALU work between its load and its store.
            # ledger/g17-load-use-hazard.toml
            # THE WIDE FORM'S WAIT IS A *LOAD* WAIT. byte9[5] is "wait for a pending load
            # into the source register"; whether it also covers a special-register read has never
            # been measured, so an SR source is materialised through an ALU here rather than
            # trusted to it. Same mechanism as the indexed store two branches down.
            val = _materialise_sr(out, val)
            # A TEXTURE FETCH WIDENS THIS STORE TOO. The 8-byte slot store is the same shape as
            # op17229.
            #
            # "op17229 IS MEASURED NOT TO READ A FETCH" IS RE-SCOPED, 2026-09-12. That is what this
            # comment said, and the linker's 4c0ca7ef retracts the scope it rested on: the eight
            # dispatches measured the ALU-FED shape - store value from an alu.12, one instruction
            # after the fetch - and APPLE EMITS THE OTHER ONE. Apple's own tex33r-noloop object
            # decodes to op15813/8 TEXTURE READ -> r106 then op17229/8 STORE value r106, with ZERO
            # instructions between. So the measurement covers a shape, not the opcode, and three
            # readings are alive: Apple stores stale too; op17229 reads under a condition the
            # authored image did not meet; or the difference is something neither side isolated.
            # The two encodings differ in five bits, four of them register numbers and one not -
            # byte5 bit 5, which Apple sets and this side does not. A candidate, not the condition.
            #
            # THE REFUSAL BELOW STANDS, and is now conservative for a stated reason rather than a
            # settled one: emitting this form for a texel would be betting on reading 1 of the 3.
            # It also has a COST worth naming, because the linker hit it - g17cc refusing
            # store_at of a texture fetch is what stops that side authoring Apple's direct shape,
            # so the discriminating experiment currently needs Apple's own object dispatched
            # instead. handoff 10cy, ledger/g17-only-one-consumer-can-read-a-texture-fetch.toml Widening gets the 14-byte store, which is
            # the FAMILY Apple uses for a texel - though not the same opcode: this emits op17244
            # and Apple's texture kernels use op17235. op17244/14 has read a fetch ONCE, at one
            # coordinate in one class (results/g17-op17244-receipt-v1, quoted at the slot store above);
            # nothing wider is measured. ledger/g17-only-one-consumer-can-read-a-texture-fetch.toml
            from_tex = (isinstance(val, ir.Value) and val.op is not None
                        and val.op.kind == "texture_read")
            from_load = _is_load_value(val)
            wide = idx.v >= 64 or from_load or from_tex
            fields = dict(n=2, slot=idx.v, range_group=True,
                          const=4 * _BUF_RANK[0].get(buf.slot, 0))
            if wide: fields["wait_load"] = 1 if from_load else 0
            out.append(MInst("store.14" if wide else "store.8", 14 if wide else 8,
                             fields, uses=[val, zero],
                             note="writes slot %d, RESERVES slot %d (=0)%s"
                                  % (idx.v, idx.v + 1, ", waits on the load" if from_load else "")))
        elif k == "store_fetch":
            # THE FETCH CONSUMER (g17ir.Builder.store_fetch; handoff 10t): the 14-byte store in sub-form
            # 01 - Apple's op17235, the only consumer measured to read a texture fetch - with its
            # source naming the fetch's destination register, exactly the relation Apple's texture
            # kernels have and the one the bisection kept when it moved both together (ledger/g17-
            # only-one-consumer-can-read-a-texture-fetch.toml). The value MUST be a fetch: the same
            # sub-form was measured not to read a changed general source. The 0 component encoding is
            # Apple's; whether it writes slot k + 3 from r<src+1> here as sub-form 11 does is
            # UNMEASURED, so a companion is stated and allocated next to the fetch, and the control's
            # prediction carries that word as conditional. The opcode is named outright from the
            # decoder's numbering of the sub-form (g17asm.STORE_SUBFORM_OPCODES), so abi() carries it
            # without a table row - the same route the per-opcode ALU forms take.
            buf, idx, val, companion = op.args
            components = int(op.attrs.get("components", 1))
            if components not in (1, 2):
                raise Unsupported("store_fetch components %r: 1 (Apple's encoding of the texel store) or 2 (the delivered store's component field, one byte from it)" % (components,))
            if not isinstance(idx, ir.Imm):
                raise Unsupported("store_fetch to a computed index: the slot store carries a SLOT")
            if not (isinstance(val, ir.Value) and ((val.op is not None and val.op.kind == "texture_read") or val in _FETCH_DERIVED)):
                raise Unsupported("store_fetch of a value that is not a texture fetch (nor the block-add of one): sub-form 01 was "
                                  "measured to read a fetch and NOT a changed general source; Apple's S1 stores its block-add result "
                                  "(fetch + preload) through it, which is the one derived case admitted - an experimental hypothesis, "
                                  "not a measurement; any other value goes through `store`")
            if companion in _PHI_MEMBERS:
                raise Unsupported("store_fetch companion is a phi-group member: copy it through an ALU first")
            companion = _wait_for_load(out, _materialise_sr(out, companion))
            out.append(MInst("store.14", 14,
                             dict(n=components, slot=idx.v, range_group=True, const=4 * _BUF_RANK[0].get(buf.slot, 0), wait_load=0,
                                  subform=1, opcode=g17asm.STORE_SUBFORM_OPCODES[1]),
                             uses=[val, companion],
                             note="fetch consumer op17235 (sub-form 01, %d component%s): writes slot %d from the fetch; slot %d from the stated companion is CONDITIONAL (the companion write is measured for sub-form 11 only)"
                                  % (components, "" if components == 1 else "s", idx.v, idx.v + (3 if components == 1 else 1))))
        elif k in FSELECT_TEMPLATE:
            # fmax / fmin: op9700 with the two operands as both the compared pair and the choices,
            # exactly as Apple emits it. op9700 shares op11375's shape and, like it, has no
            # load-wait this session can write (byte0[3] is an opcode bit on the select family and
            # Apple's own max after two loads carries byte0 = 0x22), so an operand that comes
            # straight from a load is copied through an ALU add-0 first, which does carry the wait.
            # Conservative: the copy costs one instruction per loaded operand and never a value.
            args = []
            for v in op.args:
                if _load_wait([v]):
                    copy = ir.Value(getattr(v, "type", ir.I32), "%s_w" % (getattr(v, "name", "v") or "v"))
                    out.append(MInst("alu.12", 12,
                                     dict(op=3, mode=1, imm=0, src1_w=1, dest_w=1, srcb_w=1,
                                          scale_code=2, load_wait=1, b4_5=None, b0_5=None,
                                          keep=1 if v in _MULTI_USE else 0),
                                     defs=[copy], uses=[v],
                                     note="copy: op9700 cannot wait for a load; the add does"))
                    v = copy
                args.append(v)
            a, b = args
            out.append(MInst("auth", len(FSELECT_TEMPLATE[k]),
                             dict(opcode=FSELECT_OPCODE, imms={}, template=FSELECT_TEMPLATE[k]),
                             defs=[op.dest], uses=[a, b, a, b]))
        elif k == "u16_to_u32":
            # ZEXT16 AS op10283 WITH A MATERIALISED ZERO, waiting for the load when the value is one.
            #
            # A sixteen-bit load result is exactly what this route has to widen, and what makes it
            # ready is MEASURED: root's results/g17-integer16-half-load-v1 pair (see the template
            # above) differs only in op10283's byte0[3], and the set arm returns the source's low
            # halfword on every query while the clear arm returns wrong, query-to-query unstable
            # values. So a loaded operand is consumed directly with the wait bit set. There is no
            # intermediate copy: the earlier version of this arm inserted a half-to-half op10289
            # `add(h, 0)` because the readiness was then unmeasured, and that instruction now has
            # nothing to establish - the waiting widen does the job the copy was standing in for.
            #
            # A NON-LOADED OPERAND KEEPS THE NO-WAIT FORM, which is the configuration the zero-first
            # evidence dispatched. Waiting where there is no outstanding load would be a modifier
            # neither run measured, so each arm emits the encoding its own receipt covers.
            (x,) = op.args
            if getattr(x, "type", None) is not ir.I16:
                raise Unsupported("u16_to_u32 takes a sixteen-bit value; got %r"
                                  % (getattr(x, "type", None),))
            src = x
            zero = ir.Value(ir.I32, "wz")
            out.append(MInst("movimm.8", 8, dict(imm=0), defs=[zero],
                             note="the zero first operand op10283's measured endpoint used"))
            out.append(MInst("auth", len(WIDEN_U16_TEMPLATE),
                             dict(opcode=WIDEN_U16_OPCODE, imms=dict(WIDEN_U16_IMMS),
                                  template=WIDEN_U16_TEMPLATE, encoder=None,
                                  load_wait=_is_load_value(src)),
                             defs=[op.dest], uses=[zero, src],
                             note="zext16: 0 + (b & 0xFFFF) on op10283's retained form%s"
                                  % (", waiting for the half load" if _is_load_value(src) else "")))
        elif k == "low16":
            # TRUNCATION TO SIXTEEN BITS, through the half move this backend already emits.
            #
            # op590/4 moves a word source into a HALF destination and its two file bits select
            # which half on each side independently - MOVHALF_DEST_FILE = byte3[0] and
            # MOVHALF_SRC_FILE = byte1[0], with all four combinations authored and read back. The
            # half-vector packing path uses exactly this instruction to put component values into
            # halves (`pack: component %d into the %s half`), and the packed layout it produces has
            # a validating hardware receipt.
            #
            # dest_281 is FALSE here, so the destination is the 425-based file: the LOW half. That
            # is the half a `(short)` or `(ushort)` cast keeps, and which half 425+n names is not
            # assumed - the imageblock prologue witnesses it and root's retained op10283 measurement
            # confirms the read view (a + low16(b), against a + high16(b) and a + b, on three
            # discriminating pairs).
            #
            # WHAT THIS DOES NOT CLAIM: that the other half of the destination word is preserved,
            # or anything about a sixteen-bit LOAD's effect on it. Root's evidence is explicit that
            # the read view does not settle either, so nothing here reads a partially written word.
            (x,) = op.args
            if not _NO_UNARY_LOAD_WAIT and _is_load_value(x):
                x = _wait_for_load(out, x)          # the copy waits; op590 has no wait field modelled
            out.append(MInst("mov.half.4", 4,
                             dict(keep_src=_has_later_reader(x, op), dest_281=False, half_index=0),
                             defs=[op.dest], uses=[x],
                             note="truncate: the low half of r<105+n> as r<425+n>"))
        elif k in ("icmp", "csel"):
            opc = ICMP_OPCODE if k == "icmp" else CSEL_OPCODE
            # THIS FORM CANNOT WAIT FOR A LOAD, AND ITS OWN REFUSAL NAMED THE REPAIR: "Put the
            # value through an ALU op first". byte0[3] is the load-wait on every other authored
            # form and on op11372/op11375 it is an OPCODE bit - setting it decodes as a different
            # instruction - so the wait has to come from an intervening instruction that has one.
            #
            # That is the same repair `store_at` and the four-byte bitwise already take, through
            # the same measured alu.12 copy: load_wait set because the operand came from a load
            # (byte0[3], ledger/g17-alu-load-use-wait.toml), keep set when the original still has a
            # reader so the copy does not free a value its own later reader needs. Nothing new is
            # measured here; an existing measured repair is applied at a third site.
            #
            # WHY IT MATTERS FOR THE COMPARISON SPECIFICALLY: a source that compares a loaded value
            # is the ordinary case, not an exotic one - `if (b[i] != 9)` - so refusing it refuses
            # the construct rather than an edge of it. cmp_ne is exactly that shape.
            if _load_wait(op.args):
                if _NO_ICMP_LOADWAIT:
                    raise Unsupported("%s takes an operand straight from a load, and op%d cannot "
                                      "say so: byte0[3] is the load-wait on every other authored "
                                      "form and setting it here decodes as a DIFFERENT "
                                      "instruction, so it is an opcode bit. Put the value through "
                                      "an ALU op first" % (k, opc))
                op = ir.Op(k, op.dest, [_isolate_bitwise_operand(
                               out, v,
                               why="%s cannot carry the load-use wait: op%d's byte0[3] is an "
                                   "opcode bit, so the wait comes from this copy" % (k, opc))
                           if _is_load_value(v) else v for v in op.args], **op.attrs)
            table = ICMP_CC if k == "icmp" else CSEL_CC
            rel = op.attrs.get("rel")
            args = list(op.args)
            neg = False
            if k == "icmp":
                rel = ICMP_ALIAS.get(rel, rel)
                if rel in ICMP_NEG: rel, neg = ICMP_NEG[rel], True
                if rel in ICMP_SWAP:
                    rel = ICMP_SWAP[rel]; args[0], args[1] = args[1], args[0]
            if k == "csel" and rel == "eq":
                raise Unsupported("csel with `eq`: op11375's code 1 is a BIT TEST, (a & b) != 0, "
                                  "not equality (ledger/g17-csel-code-one-is-a-bit-test.toml). For "
                                  "equality select on icmp's 0/1 value with `gt` against zero")
            if rel not in table:
                raise Unsupported("%s relation %r is not one this opcode's condition-code field "
                                  "can express; %s are" % (k, rel, "/".join(sorted(table))))
            dest = op.dest
            if neg:
                dest = ir.Value(name="%s_raw" % (op.dest.name if op.dest else "c"))
            if op.attrs.get("requant_stage"):
                # The two measured six-byte clamp forms carry their bound and select operands in
                # the retained control word.  Keep one semantic value as the allocator's tied
                # placeholder for each row; the skipped bound constants never become registers.
                args = ([op.args[1]] * 4 if op.attrs.get("requant_stage_step") == "clamp_low"
                        else [op.args[0]] * 4)
            _fields = dict(opcode=opc, imms={2: table[rel]}, template=None)
            if op.attrs.get("requant_stage"):
                from agxforge.g17 import requantenc
                if k != "csel" or op.attrs.get("requant_stage_step") not in ("clamp_low", "clamp_high"):
                    raise Unsupported("requantization stage has an unexpected clamp step")
                _fields.update(raw=requantenc.stage_bytes(bool(op.attrs.get("requant_signed")),
                                                           op.attrs["requant_stage_step"]),
                               requant_stage=True)
            _m = MInst("auth", len(_fields["raw"]) if "raw" in _fields else g17auth.length(opc),
                       _fields, defs=[dest], uses=args)
            if _fields.get("requant_stage"):
                _m.fields.update(requant_fixed_defs=(0,), requant_fixed_uses=(0, 0, 0, 0))
            out.append(_m)
            if neg:
                # the comparison yields 0 or 1, so its complement is exactly xor with 1
                out.append(MInst("bitwise.imm", 10,
                                 dict(opcode=17770, imm=1, hazard=0, keep_src=False),
                                 defs=[op.dest], uses=[dest]))
        elif k == "store_at":
            # THE INDEXED STORE. op17229 takes its index in a register, so a thread writes its own
            # element instead of every thread writing one slot. The buffer is still the template's:
            # operand 3 is an expression Apple's decoder does not resolve to a register, so which
            # buffer is written comes from the witness, exactly as it does for the slot store.
            buf, idx, val = op.args[0], op.args[1], op.args[2]
            if isinstance(idx, ir.Imm):
                raise Unsupported("store_at with a constant index; use store, which carries a slot")
            _buf_const = 4 * _BUF_RANK[0].get(buf.slot, 0)
            if op.attrs.get("requantized"):
                # The retained requantization probes declare int32_t *Out.  The logical narrowed
                # value is therefore carried in a 32-bit word and reaches memory through the
                # ordinary indexed word store (op17229), not through a guessed byte store.
                if op.attrs.get("width") != "word" or getattr(buf, "elem", None) != ir.I32:
                    raise Unsupported("requantized store is measured only for a physical i32 word destination")
                if getattr(val, "type", None) != ir.I32:
                    raise Unsupported("requantized word store requires a clamped i32 value")
                # Fall through to the measured indexed-word path below.  Its load-use handling
                # and lifetime fields are the same ones used by Apple's int32 output probe.
            # A LOAD'S RESULT ARRIVES LATE AND THIS FORM CANNOT WAIT FOR IT. The slot store handles
            # the same hazard by switching to the FOURTEEN-byte form, whose byte9 bit5 is the wait;
            # op17229's eight-byte encoding has no byte9 and no wait bit, so a value taken straight
            # from a load is stored before it lands. Measured, not inferred: the same kernel with a
            # single `add 0` between the load and the store returns 0xBB000000.. per thread and
            # without it returns zero for every thread, at status 0 with no fault - the store runs,
            # it just stores nothing yet. The ALU forms escape this because they DO carry a
            # load-wait (byte0[3], _load_wait), which is why one intervening instruction is enough.
            # ledger/g17-load-use-hazard.toml, ledger/g17-the-indexed-store-cannot-wait.toml
            # A TEXTURE FETCH IS NOT READABLE BY THIS STORE, and that is measured rather than
            # inferred from the load case it resembles. Deforming Apple's reading program one
            # change at a time: its leading instructions, its fetch destination register and its
            # publish form can all be changed and it still reads; swap ONLY its store for op17229
            # and it stops. Storing the fetch through op17229 returns the destination register's
            # PRIOR contents, and so does an alu.12 reading it - both consumers see the register as
            # it was before the fetch.
            #
            # IT IS NOT A WAIT. Six unrelated instructions between the fetch and the store change
            # nothing, so the result never lands in the register and no distance or wait bit
            # reaches it. op17235 - the 14-byte slot store - is the only consumer shown to read a
            # fetch. ledger/g17-only-one-consumer-can-read-a-texture-fetch.toml
            # NARROWED 2026-09-12 (user-directed, not a root assignment), AND THE REFUSAL IS
            # STILL THE DEFAULT. The bisection above recorded its op17229 arm as "src r105, idx
            # r109" rather than as bytes, so whether that arm carried the one non-register bit
            # Apple sets CANNOT BE CHECKED from what is committed (the linker's 4c0ca7ef).
            # Measured here against Apple's own tex33r-noloop store, 1f08030001261040: of the five
            # bits by which this side's op17229 differed, b0[6] and b3[3] are the two register
            # numbers the allocator picks, b1[2] and b1[3] the description does not read, and
            # b5[5] carries OPERAND 1's BIT 24 - Apple 16777232 where this side emits 16.
            #
            # AND OPERAND 1 IS NOT A STRANGER: g17auth.lifetime_operand(17229, 0) IS operand 1.
            # It is the VALUE'S KEEP/RELEASE MODIFIER, whose two values are 32 and 16, and this
            # backend already writes it. Apple's 16777232 is 16 - release - plus a bit 2^24 that
            # sits in the same operand, above the lifetime's values, and that
            # g17auth.carriers(17229, 1) does not list. So the bit is not an independent flag and
            # not an unnamed inherited operand; it is a high bit of a field whose low bits this
            # side does write (memory:flip-a-field-not-a-bit,
            # memory:a-binary-question-cannot-find-a-categorical-field), of the same operand-1
            # high-bit family already found on op2190/8, op998/6 and op14060/8. The consequence is
            # measured, not assumed: with the bit set, certify_on refuses op17229's source-0
            # lifetime, because the certification asks operand 1 to read exactly 32 and it reads
            # 16777248 - so in Apple's shape a fetched value CANNOT be used again after the store.
            #
            # An ordinary program therefore still gets the refusal - emitting this for a texel
            # would bet on one of three unresolved readings, the silent wrong answer the refusal
            # exists to prevent. A program that ASKS, by setting `direct_fetch_store` on the store,
            # gets Apple's shape instead, with that bit as a named parameter so one dispatch can
            # vary exactly one bit. With the index and the fetch destination in the registers
            # Apple's compiler used, True reproduces Apple's store BYTE FOR BYTE and False is the
            # same instruction with bit 24 clear (tools/g17fetchstoredirect.py). This is a CONTROL,
            # not a lowering: nothing selects it by accident, it adds no form to the frontier, and
            # a refusal lifted to enable the measurement that would justify it would be circular.
            _direct = op.attrs.get("direct_fetch_store")
            if (isinstance(val, ir.Value) and val.op is not None
                    and val.op.kind == "texture_read" and _direct is None):
                raise Unsupported(
                    "store_at of a texture fetch: op17229 cannot read a fetch result and neither "
                    "can an alu.12 - both return the destination register's prior contents, and "
                    "six instructions of distance do not help, so it is not a wait. The only "
                    "consumer measured to read a fetch is op17235, the 14-byte SLOT store. Store "
                    "the texel to a constant slot rather than a computed index. To author Apple's "
                    "direct shape as a CONTROL rather than as a lowering, set "
                    "direct_fetch_store=True (Apple's encoding, operand 1 bit 24 set) or "
                    "direct_fetch_store=False (the same shape without it) - that path is for the "
                    "measurement that would settle this and is not selected by any ordinary "
                    "program")
            if _direct is not None:
                if not (isinstance(val, ir.Value) and val.op is not None
                        and val.op.kind == "texture_read"):
                    raise Unsupported("direct_fetch_store asks for Apple's fetch-store shape, but "
                                      "this store's value does not come from a texture_read")
                if not isinstance(_direct, bool):
                    # `_direct not in (True, False)` ACCEPTED 1 and 0, because 1 == True in Python.
                    # A control whose parameter silently takes an integer is a control that can be
                    # set by an expression nobody read as a flag.
                    raise Unsupported("direct_fetch_store is %r; it takes True (operand 1 bit 24 "
                                      "set, Apple's encoding) or False (the same shape without "
                                      "it), and nothing else - 1 and 0 are refused on purpose"
                                      % (_direct,))
                if op.attrs.get("width") != "word":
                    # THE SHAPE IS op17229'S AND op17229 IS THE WORD FORM. A half store is op17193,
                    # a different template with a different register file, and the measured bit is
                    # byte 5 of THIS one. Silently giving a half store the word shape would store
                    # 32 bits where the program asked for 16.
                    raise Unsupported("direct_fetch_store is the op17229 WORD shape; this store "
                                      "asks for width=%r, whose form is op17193 and whose bit 24 "
                                      "has not been measured" % (op.attrs.get("width"),))
                _tmpl = bytearray(STOREI_TEMPLATE)
                if _direct:
                    _tmpl[5] |= 1 << 5
                out.append(MInst("auth", len(STOREI_TEMPLATE),
                                 dict(opcode=STOREI_OPCODE, imms={}, srcmap=[0, 5],
                                      buf_const=_buf_const, direct_fetch_store=bool(_direct),
                                      template=bytes(_tmpl)), uses=[val, idx]))
                continue
            if _is_load_value(val):
                # THE REFUSAL NAMED ITS OWN REPAIR, AND THE REPAIR IS MEASURED. "Put the value
                # through an ALU op first - those carry the wait" is exactly the alu.12 copy
                # `_isolate_bitwise_operand` emits, with load_wait set (byte0[3],
                # ledger/g17-alu-load-use-wait.toml). And the measurement is this very hazard: the
                # same kernel with a single `add 0` between the load and the store returns
                # 0xBB000000.. per thread, and without it returns zero for every thread at status 0
                # with no fault (ledger/g17-the-indexed-store-cannot-wait.toml).
                #
                # NO EXISTING PROGRAM'S BYTES CAN MOVE. The line this replaces was an unconditional
                # refusal, so nothing that compiles today reaches here: this converts a refusal into
                # code and can only add programs. It became load-bearing when the front end started
                # routing constant-address scalar stores through store_at - which is the correctness
                # fix for the extra word it used to write - because that took eleven of Apple's
                # sources to this refusal. MEASURED on the pinned 197-tag list, three arms in one
                # process (the flags below are what makes them comparable):
                #
                #     A  the old mapping, legacy `store`         56 compiled, 6 backend
                #     B  store_at, this refusal standing         50 compiled, 12 backend
                #     C  store_at + this copy                    61 compiled, 1 backend
                #
                # (a fourth arm followed: preserving the DECLARED buffer element type, which the
                # front end had been defaulting to i32, takes C's 61 to 52 by refusing nine kernels
                # that declare a ulong/long/atomic buffer - one of which was compiling 32-bit
                # arithmetic for 64-bit elements. Same tag list; docs/g17-conversion-admission-
                # inspection.md carries it.)
                #
                # so the correctness fix ALONE costs six programs and the copy pays that back and
                # more: five tags move backend -> compiled against A (gap32-33, gap32c-6-0,
                # syn-s4e69e1c302, syn-sbf352471d6, syn-sf1cc51860e) and none regresses. A one-element
                # store needs neither two consecutive registers nor the pair's source lifetime, so
                # both "no 2 consecutive registers free for a range store" refusals and all three
                # "op17244 stores a value this program reads again" refusals go away with it.
                if _NO_STORE_AT_LOADWAIT:
                    raise Unsupported("store_at of a value that comes straight from a load: "
                                      "op17229's eight-byte form has no load-wait bit, so the store "
                                      "would write before the load lands and every thread would "
                                      "store zero. Put the value through an ALU op first - those "
                                      "carry the wait - or use `store`, whose fourteen-byte form "
                                      "has one")
                val = _isolate_bitwise_operand(
                    out, val, force=True,
                    why="load-use wait: op17229 has no load-wait bit, so the stored value goes "
                        "through the measured alu.12 copy that does")
            if op.attrs.get("width") == "half":
                # A SIXTEEN-BIT ELEMENT IS ANOTHER FORM. op17193 at fourteen bytes, whose value
                # operand names the 16-bit register file based at 425 while its index stays in the
                # 32-bit one based at 105. op17193 at TEN bytes - the length Apple picks when it
                # can - derives to None, so the fourteen-byte form is the one emitted, and it is
                # what Apple's own huL_h_cos uses for the same operation.
                # THE VALUE-SIDE DEPENDENCY REFUSAL WAS WRITTEN FROM EVIDENCE THAT IS DISPROVED,
                # and this is the correction, at the claim.
                #
                # WHAT IT SAID: that `halfzero` showed an add with a load-wait before op17193
                # leaving the store reading its old source register for every lane but zero, and
                # that the retained 32-thread family showed the same for two waited ALUs, a
                # half-to-float round trip, and two independent FMAs. Every part of that is wrong
                # about its own receipts:
                #
                #   * add_load_zero, two_alu and narrow_roundtrip
                #     (results/g17-halfstore-wait-runtime-v1) each mismatched their reference in
                #     16 words. The REFERENCE was wrong. The input's uint words 0x1000, 0x1001 ...
                #     are the contiguous half stream 0x1000, 0, 0x1001, 0 ..., and the expected
                #     array packed each word's LOW half consecutively, discarding the zero high
                #     halves. Rebuilt from the fill plus the program's semantics on the ACTUAL half
                #     stream: 0 mismatches for all three, with the old packing as a null arm that
                #     still fails at 16. two_alu adds 1 to every actual half, 0 -> 1 included.
                #   * half_long returned zero for every element and was called a failure. It
                #     computes zero: its two movimm32 write the BIT PATTERNS 1 and 0 into the FMA's
                #     multiplier and addend, so the multiplier is 1.4e-45, not 1.0, and
                #     fma(v, 1.4e-45, 0) underflows to zero for every input half. The hardware
                #     returned exactly that. op2190/16 is the FMA, which the fill programs settle:
                #     151 source fma() calls to 151 instances, 254 to 254.
                #   * halfzero (results/g17-halfzero-runtime-v1) is a REAL failure - lanes 1..31
                #     came back as fill - but not this one. The value its failing store writes came
                #     from a WORD load (op12682 at +0030). The defect is that the store at +0016
                #     RELEASES its index register reg:110 (operand 6 = 16) and the next instruction
                #     reads reg:110 to compute the second index; lane 0 has index zero either way,
                #     which is precisely the one lane that came back right.
                #
                # WHAT ACTUALLY FIXES IT is the lifetime rule this file already computes from
                # liveness: recompiling halfzero's own IR today moves ONE byte, +0x1b, from release
                # to keep - 0a4dcc573fe4 becomes 9b1b8798b6d7 - and root's hardware receipt for
                # those bytes passed (10 queries, two workers, 640 words exact). narrow_roundtrip's
                # 60 bytes re-passed in v3 (640 halves, two workers). So the condition is not "no
                # half loads in the ancestry": it is that no register may be read after an
                # instruction released it, which asm.read_after_release now checks on the DELIVERED
                # bytes, with those same retained failing bytes as its negative control.
                if (_HALF_LOAD_ANCESTRY_REFUSAL
                        and (_depends_on_half_load(val)
                             or (isinstance(val, ir.Value) and val.op is not None
                                 and val.op.kind in ("add", "fadd")
                                 and _load_wait(getattr(val.op, "args", []))))
                        and not (getattr(_CUR_FN[0], "allow_unvalidated_half_store", False)
                                 or getattr(_CUR_FN[0], "allow_validated_half_store", False))):
                    raise Unsupported(
                        "op17193 indexed half store consumes a value derived from a half load. "
                        "This is the HISTORICAL refusal, restored by "
                        "_HALF_LOAD_ANCESTRY_REFUSAL; the evidence it was written from is "
                        "disproved (three wrong references, one program that computes zero, and "
                        "one real failure whose value came from a WORD load and whose cause was an "
                        "index register released before its next read). The condition that "
                        "replaced it is asm.read_after_release on the delivered bytes.")
                # THE STORE'S TWO LIFETIMES ARE STATED, NOT INHERITED, and the rule is measured.
                #
                # My first repair refused a shared value here and said the operand was unmeasured.
                # It was not: ledger/g17-half-store-displacement-refusal.toml records it, answered
                # by this side with paired one-variable controls - "operand 1 is the stored value's
                # lifetime and operand 6 is the index register's, 0 keeps and 16 releases". I
                # declared a fact missing that my own ledger held, and the refusal would have
                # discarded newly supported sources to avoid a mechanism that was already known.
                #
                # NOTE THE POLARITY: here 0 KEEPS and 16 RELEASES, which is the opposite of the
                # conversions' MOV_KEEP=32 / MOV_RELEASE=16. Two forms, two conventions; reading
                # one off the other would be a guess.
                #
                # The condition is LATER LIVENESS, not whole-program use count: the half-scan's
                # store is the last use of both the value and the index, so 16/16 is right there
                # and this change must leave those bytes alone.
                out.append(MInst("store.half.14", 14,
                                 dict(buf_const=4 * _BUF_RANK[0].get(buf.slot, 0),
                                      keep_value=_has_later_reader(val, op),
                                      keep_index=_has_later_reader(idx, op)),
                                 uses=[val, idx]))
                continue
            val = _materialise_sr(out, val)
            _store_fields = dict(opcode=STOREI_OPCODE, imms={}, srcmap=[0, 5],
                                 buf_const=_buf_const,
                                 template=STOREI_TEMPLATE)
            if op.attrs.get("requant_stage"):
                from agxforge.g17 import requantenc
                if op.attrs.get("requant_stage_step") != "store":
                    raise Unsupported("requantization stage has an unexpected store step")
                _store_fields.update(raw=requantenc.stage_bytes(bool(op.attrs.get("requant_signed")), "store"),
                                     requant_stage=True)
            _m = MInst("auth", len(STOREI_TEMPLATE), _store_fields, uses=[val, idx])
            if _store_fields.get("requant_stage"):
                _m.fields["requant_fixed_uses"] = (0, 1)
            out.append(_m)
        elif k == "load_vec_at":
            # THE INDEXED 4-COMPONENT LOAD (handoff 10ae): op12709/8, tuple r<t>..r<t+3> <- [descriptor + index]. What
            # Apple selects for `in[gid.x]` on a device uint4 buffer (V0). The lanes are the vec_lane values on this
            # dest; they are DEFINED by this instruction (the allocator gives them the tuple's consecutive registers)
            # and emit nothing. One vector load per program: Apple's second one takes the 14-byte form whose trailing
            # composite (10ac) is not authored.
            buf, idx = op.args; fn = _CUR_FN[0]
            idx = _wait_for_load(out, idx)       # a loaded index is late (see the "load" branch)
            n_comp = op.attrs.get("n", 4)
            if not isinstance(idx, ir.Value): raise Unsupported("load_vec_at with an immediate index: the form takes its index in a register")
            lanes = [o for blk in fn.blocks for o in blk.ops if o.kind == "vec_lane" and o.args[0] is op.dest]
            if [o.args[1].v for o in lanes] != list(range(1, n_comp)):
                raise Unsupported("load_vec_at at n=%d needs exactly its lanes 1..%d in order, got %s"
                                  % (n_comp, n_comp - 1, [o.args[1].v for o in lanes]))
            # THE CONFOUND, NAMED (handoff 10ae): Apple selects the load's LENGTH by what is in flight, not by the
            # address. V0's single load follows a special-register read and is 14 bytes; V4's first load follows an
            # ALU and is 8; V4's SECOND load follows a store and is 14. Reproducer:
            # results/g17-vector-store-compiles-A2-v1/V4 (+16 and +80).
            # A SECOND VECTOR LOAD TAKES THE 14-BYTE FORM, WHICH NEEDS NO GUESS (MM 25.139.2): bytes 4..13 of
            # VEC4_LOAD14_TEMPLATE are EXACTLY the scalar load.14's (op12682/14, emitted here after ALUs, stores and
            # loads in every program that loads) with byte5 bit3 set and the count byte4[6:5] and access size
            # byte7[6:5] written - Apple's decoder reads the scalar template so flipped as op12691/op12709 with every
            # other operand unchanged. So the long form is the scalar load's own encoding at n components, and the
            # 8-byte form, whose selection is the confound, is kept only for the first load where it is witnessed.
            # A program with ONE vector load emits exactly the bytes it did; a program with more takes the long form
            # for every one of them (a loop's first load follows the previous trip's loads, not a witnessed position).
            later = sum(o.kind == "load_vec_at" for blk in fn.blocks for o in blk.ops) > 1
            wait_sr = later or (idx.op is not None and idx.op.kind == "builtin")
            # THE DISPLACEMENT IS A BYTE COUNT (MM 25.141.17, measured): field value F adds F bytes to the indexed
            # address, in the 8- and 14-byte forms alike (Apple's x loads: 16/32/48 for x[k+4], x[k+8], x[k+12]).
            # encode_vec4 takes its disp in "eight-byte units" (field = disp // 8), so the byte count is passed x8.
            vdisp = 8 * op.attrs.get("offset_bytes", 0)
            out.append(MInst("load.vec%d.%d" % (n_comp, 14 if wait_sr else 8), 14 if wait_sr else 8, dict(desc=4 * _buf_rank(buf.slot), disp=vdisp, tuple_group=True, release_index=False, wait_sr=wait_sr, n=n_comp), defs=[op.dest] + [o.dest for o in lanes], uses=[idx],
                             note="tuple <- [descriptor %d + index]%s" % (4 * _buf_rank(buf.slot), " (index straight from a special-register read: the 14-byte form)" if wait_sr else "")))
        elif k == "load_half2":
            # op12655/14, THE PACKED HALF2 LOAD: the two-component vector load with byte0[3] set -
            # asked of Apple's decoder, that one flip turns op12691/14 into op12655/14 with every
            # operand unchanged but the destination's class (GPR32, not a tuple) and the mask - and a
            # 4-byte access. Only the witnessed 14-byte position: the index straight from a
            # special-register read, as for load_vec_at's 14-byte form.
            buf, idx = op.args
            if not (isinstance(idx, ir.Value) and idx.op is not None and idx.op.kind == "builtin"):
                raise Unsupported("load_half2_at takes its index straight from a position builtin: only "
                                  "the 14-byte form, in load_vec_at's witnessed position, is authored")
            if any(x.form.startswith("load.vec") for x in out):
                raise Unsupported("a second vector-family load: the same length confound as load_vec_at")
            out.append(MInst("load.vec2h.14", 14, dict(desc=4 * _buf_rank(buf.slot), disp=0, tuple_group=False,
                                                       release_index=False, wait_sr=True, n=2, half2=True),
                             defs=[op.dest], uses=[idx], note="packed half2 <- [descriptor + index]"))
        elif k == "vec_lane":
            pass                                # defined by the load's tuple; no instruction
        elif k == "store_vec4_at":
            # THE INDEXED 4-COMPONENT STORE (handoff 10ae): op17256/8, [descriptor + index + disp] <- tuple. The four
            # values are pre-coloured consecutive by the range-store machinery (range_n names the members; the index is
            # the fifth use). A value straight from a load or a fetch refuses: the witnesses store ALU results.
            buf, idx = op.args[0], op.args[1]; vals = list(op.args[2:])
            if not isinstance(idx, ir.Value): raise Unsupported("store_vec4_at with an immediate index: the form takes its index in a register; use store_range for a slot")
            if len(vals) != 4 or not all(isinstance(v, ir.Value) for v in vals): raise Unsupported("store_vec4_at stores four register values")
            for v in vals:
                if v.op is not None and v.op.kind in ("load", "load_vec_at", "vec_lane", "texture_read"):
                    raise Unsupported("store_vec4_at of a value straight from a %s: the 8-byte form carries no load-wait and no witness stores an unmodified load or fetch; put it through an ALU op first" % v.op.kind)
            disp = op.attrs.get("disp", 0)
            if disp not in (0, 16, 32): raise Unsupported("store_vec4_at displacement %r: witnessed at 0, 16 and 32 (V0/V4/V6); the field is eight bits in units of eight bytes but no other value is witnessed" % disp)
            out.append(MInst("store.vec4.8", 8, dict(n=4, desc=4 * _buf_rank(buf.slot), disp=disp, range_group=True, range_n=4, release_index=True), uses=vals + [idx],
                             note="[descriptor %d + index + %d] <- tuple" % (4 * _buf_rank(buf.slot), disp)))
        elif k == "store_range":
            buf, idx = op.args[0], op.args[1]; vals = op.args[2:]
            if not isinstance(idx, ir.Imm):
                if op.attrs.get("width") == "half":
                    # MEASURED, not assumed: Apple selects op17220 (n=4), op17211 (n=3) and op17202 (n=2)
                    # for a half vector at a per-thread index - round A's eight members, every one of them -
                    # and those forms differ from the ones lowered here by byte5[1] alone. They are a
                    # separate lowering with their own address operand, and naming them is the refusal.
                    raise Unsupported("a half range store at a COMPUTED index: the lowered forms "
                                      "(op17208/17217/17226) carry an immediate slot. Apple uses "
                                      "op17202/17211/17220 there - one bit away at byte5[1], the address "
                                      "mode - and that family is not lowered")
                raise Unsupported("store to a computed index: the recovered store carries a SLOT")
            if not 2 <= len(vals) <= 4:
                # ONE VALUE IS NOT A RANGE. n=1 is byte4[6:5]=0, which this form's measured
                # behaviour makes a MASK selecting components 0 and 3: it writes slot k from
                # r<src> AND slot k+3 from r<src+1> - a neighbour the caller did not name, from a
                # register the caller did not choose (ledger/g17-store-has-no-single-slot-form).
                # `store` carries a one-value write with its companion zero stated.
                raise Unsupported("store_range of %d value(s); the form's component field takes 2..4 "
                                  "(n=1 is the k/k+3 encoding); a single value goes through `store`, "
                                  "which states its companion write" % len(vals))
            # THE BINDING RANK IS STATED, as the ordinary store states it. It was not: byte1
            # (const = 4 * rank) was inherited from the template, 0x04 = rank 1, whatever buffer
            # the IR named - a range store to rank 0 or 2 wrote rank 1's buffer. The delivered
            # constant-only kernel writes rank 1 and was right by inheritance; this states it
            # and leaves those bytes unchanged (the no-change control is asserted).
            for v in vals:
                if isinstance(v, ir.Value) and v.op is not None and v.op.kind == "texture_read":
                    raise Unsupported("a range store member from a texture fetch: whether the range "
                                      "forms read a fetch is unmeasured (the two-component op17244/14 "
                                      "read one once, at one coordinate - results/g17-op17244-receipt-v1 "
                                      "- and no range form has been asked; Apple uses op17235)")
            vals = [_wait_for_load(out, _materialise_sr(out, v)) for v in vals]
            # A VALUE NAMED TWICE NEEDS TWO REGISTERS, and until this commit it silently got one. The form
            # writes r<src> .. r<src+n-1>, so a caller asking to store the same value in two components got
            # the value in the first and WHATEVER THE ALLOCATOR LEFT in the second - a wrong value with no
            # refusal, at both widths (store_range(C, 14, [v, v]) emitted src=6 n=2 and only r6 held v).
            # Each repeat now gets its own copy, which is a register move: op590 for a sixteen-bit value and
            # op586 for a thirty-two-bit one. The FIRST occurrence keeps the original register, so a program
            # with no repeat is byte-identical to before.
            seen_once, deduped = set(), []
            for v in vals:
                if v in seen_once:
                    half_v = op.attrs.get("width") == "half"
                    copy = ir.Value(getattr(v, "type", ir.I32), "%s_dup" % getattr(v, "name", "v"))
                    out.append(MInst("mov.half.4" if half_v else "mov.word.4", 4, dict(keep_src=True),
                                     defs=[copy], uses=[v],
                                     note="copy: %s is stored in more than one component, and the form "
                                          "writes consecutive registers"
                                          % getattr(v, "name", "v")))
                    deduped.append(copy)
                else:
                    seen_once.add(v); deduped.append(v)
            vals = deduped
            if op.attrs.get("width") == "half":
                # THE HALF-VECTOR SLOT STORES (handoff 10ah; integration's 66cef0b5): op17208 at n=2,
                # op17217 at n=3, op17226 at n=4 - the same slot store as the word forms with byte0[3]
                # clear, and the component count is part of Apple's opcode number exactly as it is there.
                # Witnessed by results/g17-halfvec-roundB-compiles-v1: Apple selects these three for a
                # CONSTANT slot index (R0/R1/R5) and the indexed family op17220/17202/17211 for a computed
                # one (round A), which is the address mode the sweep located at byte5[1].
                #
                # ONE LENGTH ONLY. The fourteen-byte member is what the corpus carries (658 instances of
                # op17226/14 in 633 programs, 572 of op17208/14 in 572) and the one g17as.derive_form can
                # author; the ten-byte member Apple picks for these small kernels derives to None, so it is
                # not emitted rather than guessed at.
                if idx.v >= 64 * 256:
                    raise Unsupported("half range store at slot %d: the fourteen-byte form's slot field "
                                      "is byte6[7] + byte7[4:0] + byte13, which reaches 64*255+63" % idx.v)
                # THE VALUES ARE NOT PACKED, AND THE HARDWARE SAID SO. integration dispatched this
                # lowering (results/g17-halfvec-runtime-negative-v1): status 0, five mismatched
                # output words, and every one of them keeps only the LOW half of its word.
                #
                # The store instruction is not the defect. Field for field it is Apple's own
                # fourteen-byte witness - the same modifier operands, the same address expression,
                # the same trailing 1 - differing only in the source register and the slot. What
                # differs is what those registers HOLD. A half-vector store reads n consecutive
                # HALF registers, which is two per word register; Apple's own program emits two
                # op586/4 word moves into r105 and r106 and then stores four halves out of them,
                # while this backend puts each half value in a separate word register's low half
                # (r110..r113) and the store then reads value, unwritten, value, unwritten.
                #
                # So the guard that passed was measuring the wrong thing: selfcheck asserted the
                # USES are consecutive value indices, which they are, and never that they occupy
                # consecutive half REGISTERS, which they do not. Refused rather than repaired,
                # because the repair is a register-packing change and integration's condition is a
                # new receipt before the lowering moves. results/g17-halfvec-packing-v1 carries the
                # diagnosis with Apple's packing witness beside this program.
                # AND THE REFUSAL NAMES THE ALLOCATION, NOT THIS PROGRAM'S REGISTERS. It used to say
                # "the values are in %d separate word registers" as though it had looked; it had not,
                # and it cannot - selection runs before allocation, so `vals` are value indices here
                # and no register exists to inspect. The refusal fires for EVERY half range store, so
                # a message phrased as a per-program measurement would be a claim this site is not in
                # a position to make. What it is in a position to state is the backend-wide fact that
                # forces it, and that fact IS measured, in results/g17-halfvec-packing-v1: Apple packs
                # four halves into two word registers with op586/4 moves while this backend emits four
                # op10288/12 into four separate ones, and that delivery's checker refuses if the
                # allocation ever stops doing so. So the condition is falsifiable somewhere, the
                # refusal points at where, and it lifts through a packing pass and a fresh receipt
                # rather than by being deleted.
                # THE PACKING PASS, which is what the refusal above was waiting for.
                #
                # The refusal was never about the store instruction - field for field it is Apple's
                # own fourteen-byte witness, five of its eight operands identical and the three that
                # differ being the source register, the descriptor constant and the slot. It was
                # about the ALLOCATION: this backend gave every half value its own word register's
                # low half, and the form reads n consecutive HALF registers, two per word, so it
                # read value, unwritten, value, unwritten. Integration measured exactly that
                # (results/g17-halfvec-runtime-negative-v1: status 0, five wrong output words, every
                # one keeping only its low half).
                #
                # THE LEVER WAS ALREADY BUILT AND ALREADY ROUND-TRIPPED, which is why this is a
                # placement change and not a new encoding. 425+n is the LOW half of word register n
                # and 281+n is the HIGH half of the SAME word - witnessed by the imageblock
                # prologue, whose two read_sr halves decode 425 and 281, a fact the regression has
                # been gating in a case about something else. op590/4 selects the file independently
                # on both operands (MOVHALF_DEST_FILE = byte3[0], MOVHALF_SRC_FILE = byte1[0]) and
                # ALL FOUR combinations were authored and read back, because whoever located them
                # refused to assume the two bits shared a sense. So each component is moved into its
                # half and the group needs ceil(n/2) words instead of n.
                #
                # THIS IS NOT APPLE'S INSTRUCTION SEQUENCE AND THAT IS STATED, NOT SMOOTHED OVER.
                # Apple packs with two op586/4 WORD moves into r105 and r106, moving words that
                # already hold two halves. This side uses op590/4 half moves into the two files,
                # because that is the form whose half-file selection this project has located and
                # round-tripped, and Apple's is one it has not. Same PLACEMENT, different
                # instruction to reach it - so byte-for-byte agreement with Apple's witness holds
                # for the STORE and not for the moves.
                #
                # AND INTEGRATION HAS NOW RUN IT. This comment said "nothing hardware is
                # claimed - the receipt that refuted the old layout is not a receipt for this one;
                # integration runs that, and until it does this is a compile-time placement backed
                # by located fields and a selfcheck". They ran it: 7a0beb2c retains
                # results/g17-halfvec-runtime-packed-v1, three cases, 4,096 words checked each,
                # ZERO wrong words, repeat identical - where the unpacked layout measured five
                # wrong words, low halves only.
                #
                # The receipt's loader pins programs/halfvec/program.bin, and that sha256 equals
                # what this compiler builds for the halfvec kernel byte for byte - checked against
                # the git object rather than taken from the summary, because a receipt whose
                # subject is not pinned to a product is exactly the shape that let a comment
                # invalidate a different receipt earlier today.
                #
                # STILL NOT CLAIMED: the receipt covers the DISPATCHED program - three stores at
                # n=2, 3 and 4 at that kernel's slots, with component order and the surrounding
                # program held fixed. A measured function belongs to the form dispatched, so this
                # is not a claim about every shape the packing can produce.
                if getattr(_CUR_FN[0], "allow_unpacked_halfvec", False):
                    # THE RETAINED BYTES KEEP BUILDING, UNPACKED, because deliveries whose evidence
                    # is those exact bytes have to be comparable with the receipt that refuted part
                    # of them - g17halfmove._opt_in says so at length. This path is unchanged, so
                    # every retained instance is byte-identical to before, and it is NOT the path
                    # ordinary compilation takes.
                    out.append(MInst("store.halfvec.14", 14,
                                     dict(n=len(vals), slot=idx.v, range_group=True,
                                          const=4 * _BUF_RANK[0].get(buf.slot, 0)), uses=list(vals)))
                    continue
                packed, half_ix = [], []
                for _i, _v in enumerate(vals):
                    _dst = ir.Value(ir.I16, "%s_h%d" % (getattr(_v, "name", "v"), _i))
                    out.append(MInst("mov.half.4", 4,
                                     dict(keep_src=True, dest_281=bool(_i % 2), half_index=_i),
                                     defs=[_dst], uses=[_v],
                                     note="pack: component %d into the %s half of the group's word %d"
                                          % (_i, "high" if _i % 2 else "low", _i // 2)))
                    packed.append(_dst); half_ix.append(_i)
                out.append(MInst("store.halfvec.14", 14,
                                 dict(n=len(packed), slot=idx.v, range_group=True, half_pack=True,
                                      half_indices=half_ix,
                                      const=4 * _BUF_RANK[0].get(buf.slot, 0)), uses=list(packed)))
                continue
            wide = idx.v >= 64
            out.append(MInst("store.14" if wide else "store.8", 14 if wide else 8,
                             dict(n=len(vals), slot=idx.v, range_group=True,
                                  const=4 * _BUF_RANK[0].get(buf.slot, 0)), uses=list(vals)))
        elif k == "tensor_acc_fma16":
            # op798 INTO A HALF ACCUMULATOR'S FIXED REGISTER (MM 25.196): half slot i is half i % 2 of
            # register base + i // 2, the half A tuple the body reads with no narrowing
            regs = _TENSOR_ACC_REGS.get(op.attrs["acc"])
            if regs is None or op.attrs["acc"] not in _TENSOR_ACC_HALF:
                raise Unsupported("tensor_acc_fma16 %r: no tensor body reads that half accumulator" % op.attrs["acc"])
            srcs = _fma16_sources(op, out)
            # the UPPER four registers of the tile's group: op798's low-half destination holes sit at 2 and 3 mod 8, and
            # every accumulator group is eight-aligned, so base + 4 .. base + 7 author both halves
            reg = regs[tuple(op.attrs["tile"])] + 4 + op.attrs["slot"] // 2
            out.append(MInst("acc.ffma.f16", 12, dict(acc_reg=reg, acc_hi=op.attrs["slot"] % 2,
                                                      halves=tuple(op.attrs["halves"])), uses=srcs,
                             note="%s tile %r half %d (R%d%s) = fma16" % (op.attrs["acc"], op.attrs["tile"],
                                                                         op.attrs["slot"], reg, "LH"[op.attrs["slot"] % 2])))
        elif k == "tensor_acc_scale":
            regs = _TENSOR_ACC_REGS.get(op.attrs["acc"])
            if regs is None:
                raise Unsupported("tensor_acc_scale %r: no tensor body accumulates into that register accumulator"
                                  % op.attrs["acc"])
            reg = regs[tuple(op.attrs["tile"])] + op.attrs["slot"]
            # R<acc> = R<acc> * factor: the fp32 multiply (op3290/14) with the register on both ends, both sources
            # kept; a loaded factor carries the slot-7 load-use wait, as cc's own op3290 does (MM 25.144.8)
            out.append(MInst("acc.scale.14", 14, dict(acc_reg=reg, opcode=3290, load_wait=_load_wait([op.args[0]])),
                             uses=[op.args[0]],
                             note="%s tile %r slot %d (R%d) *= factor" % (op.attrs["acc"], op.attrs["tile"],
                                                                         op.attrs["slot"], reg)))
        elif k in ("tensor_acc_read", "tensor_acc_write"):
            regs = _TENSOR_ACC_REGS.get(op.attrs["acc"])
            if regs is None:
                raise Unsupported("%s %r: no tensor body accumulates into that register accumulator"
                                  % (k, op.attrs["acc"]))
            reg = regs[tuple(op.attrs["tile"])] + op.attrs["slot"]
            if k == "tensor_acc_read":
                out.append(MInst("acc.read.4", 4, dict(acc_reg=reg, opcode=586), defs=[op.dest],
                                 note="%s tile %r slot %d (R%d, kept)" % (op.attrs["acc"], op.attrs["tile"],
                                                                          op.attrs["slot"], reg)))
            elif reg >= 64 or _load_wait([op.args[0]]):
                # the wide write: OR with 0 into the fixed register, which also carries a load wait (the
                # move has no wait field, so a loaded value goes this way whatever its register)
                out.append(MInst("acc.write.10", 10, dict(acc_reg=reg, opcode=BITWISE_OPCODE["or"],
                                                          hazard=_hz([op.args[0]])), uses=[op.args[0]],
                                 note="%s tile %r slot %d (R%d) = value | 0" % (op.attrs["acc"], op.attrs["tile"],
                                                                                op.attrs["slot"], reg)))
            else:
                out.append(MInst("acc.write.4", 4, dict(acc_reg=reg, opcode=586), uses=[op.args[0]],
                                 note="%s tile %r slot %d (R%d) = value" % (op.attrs["acc"], op.attrs["tile"],
                                                                            op.attrs["slot"], reg)))
        elif k == "tensor_index_init":
            rows = _TENSOR_INDEX_INIT_ROWS.get(id(op))
            _inits = [o for blk in _CUR_FN[0].blocks for o in blk.ops if o.kind == "tensor_index_init"]
            if rows is not None and _TENSOR_HOIST_ROWS and _inits and _inits[-1] is op:
                rows = list(rows) + list(_TENSOR_HOIST_ROWS)
            if rows is None:
                raise Unsupported("tensor_index_init %r: the memory-stream route did not take this program, "
                                  "so nothing reads the register" % op.attrs.get("register"))
            out.extend(rows)
        elif k == "tensor_matmul":
            at = op.attrs
            shape = (at["M"], at["N"], at["K"])
            _composed = _TENSOR_COMPOSED_ROWS.get(id(op))
            if _composed is not None:
                out.extend(_composed)
                return
            # NO PATH BELOW APPLIES offsetA/B/C: the witness stream, tlower's general lowering (called
            # without offsets) and the sequence experiment all address the buffers from byte 0. A
            # single 16x32x32 body compiled to the same bytes at offsets 0, 1,024 and 2,048
            # (d87f982e3641, MM 25.124.4), so the offset was dropped in silence. Only the composition
            # routes above (_TENSOR_COMPOSED_ROWS) apply offsets; anything else with one is refused.
            _offs = [key for key in ("offsetA", "offsetB", "offsetC") if int(at.get(key, 0) or 0)]
            if _offs:
                raise Unsupported("tensor matmul %dx%dx%d: %s on a path that does not apply offsets (only "
                                  "the composition routes do: adjacent chain, independent group, memory "
                                  "stream, shape tables); refused rather than emitted at offset 0"
                                  % (shape + (", ".join("%s=%d" % (k, int(at[k])) for k in _offs),)))
            if at.get("offsetB_register") is not None:
                # only the memory-stream route lowers a register-held B offset; any other lowering
                # would drop it and read offset 0 for every key block
                raise Unsupported("tensor offsetB_register %r: the memory-stream route refused this "
                                  "program, and no other lowering reads the register"
                                  % at["offsetB_register"])
            if at.get("head_stride") is not None:
                # likewise the head grid (MM 25.135): elsewhere every threadgroup would read head 0
                raise Unsupported("tensor head_stride %r: the memory-stream route refused this program, "
                                  "and no other lowering applies it" % (at["head_stride"],))
            if not at.get("sequence_experiment"):
                # THE DELIVERED CONTRACT for the witnessed shape (docs/g17-architectural-compiler-
                # handoff.md 9c/9d): the zeroing, operand loads, MACs and readout are AUTHORED from
                # (M, N, K, strides, binding offsets, the allocation, the token protocol); the
                # lane/address setup and the loop are INHERITED from Apple's witness byte for byte
                # and declared as such on the program. Strides default to contiguous row-major
                # (A: K a_dtype elements, B: N b_dtype elements, C: N floats) - this said "halves" and
                # the compiler computed halves, which is the int8/float disagreement;
                # the binding offsets are the buffers'
                # positions. Any other shape or dtype falls to the precise refusal below.
                from agxforge.g17 import tensorlower as g17tensorlower
                                       # made it a local for the whole selector (UnboundLocalError)
                # AND THE STRIDE BASIS IS NOT THE BINDING'S element_bytes, which a consumer
                # will otherwise assume. The IR refuses a 1-byte buffer for ANY op - "this
                # backend accesses 2- and 4-byte scalars only" - so an int8 or uint8 tensor
                # operand has to be declared in a 2- or 4-byte buffer, and its binding reports
                # `half`/2 while its stride basis is 1. The two describe different things: the
                # binding is the BUFFER's declared scalar type, `a_dtype` is how the tensor unit
                # READS those bytes. Nothing but this comment and its test said so.
                _abytes = TENSOR_ELEMENT_BYTES[at.get("a_dtype", "half")]
                _bbytes = TENSOR_ELEMENT_BYTES[at.get("b_dtype", "half")]
                strides = dict(strideA=at.get("strideA", at["K"] * _abytes),
                               strideB=at.get("strideB", at["N"] * _bbytes),
                               strideC=at.get("strideC", at["N"] * 4))
                bufs = op.args[:3]; ranks = _BUF_RANK[0]
                bindings = tuple(2 * ranks.get(b.slot, 0) for b in bufs)
                try:
                    # THE INHERITED SETUP ENCODES THE WITNESS'S STRIDES. The stride witnesses show
                    # the address phase moving with every stride (B and C each move an address
                    # add, A moves fourteen and the instruction mix), so a stream whose loads and
                    # readout follow a different stride while its setup stays the witness's would
                    # be a wrong program that happens to match on the authored phases. Refused
                    # until the setup is generated from the strides (integration's review of
                    # d11a6905).
                    # (g17tensorlower.stream refuses them, naming the witness's strides)
                    # THE WITNESS STREAM CARRIES A PLAIN GEMM ONLY. It encodes shape, strides, dtypes and
                    # bindings; an epilogue, a grid or simdgroup split, split-fp32, saturation, a
                    # converted operand, an accumulate, a transpose, a reduction, a feed mode or a K
                    # loop has no place in it, and serving the shape from the stream dropped them in
                    # silence (found three times on 2026-09-23 by Set A: an exp2 epilogue and a column
                    # reduction on 32x32x64 each compiled to a bare GEMM, and a kloop request would
                    # have dispatched Apple's own loop). Such a GEMM declines the stream and takes the
                    # general lowering (tlower), which implements each of them or refuses by name.
                    _plain = [key for key in ("epilogue", "split_fp32", "saturate", "a_converted_from",
                                              "b_converted_from", "accumulate", "transA", "transB",
                                              "reduce", "feed", "kloop", "kloop_chunk") if at.get(key)] + \
                             [key for key in ("threadgroups", "simdgroups") if int(at.get(key, 1)) != 1]
                    if _plain:
                        raise ValueError("the witness stream carries a plain GEMM; this one has %s"
                                         % ", ".join(_plain))
                    rows = g17tensorlower.stream(at["M"], at["N"], at["K"], strides, bindings, at["a_dtype"], at["b_dtype"])
                except ValueError as why:
                    rows = None; reason = str(why)
                if rows is not None:
                    # the register ceiling of the FAMILY the stream came from (the multiply program
                    # is allocated differently from the baseline)
                    _fam = g17tensorlower.assignment_of(*g17tensorlower.family_of(strides)[1])
                    hi = max([f["dest"] + 1 for f in _fam["loads"]] + [f["value"] + 3 for f in _fam["stores"]])
                    for i, r in enumerate(rows):
                        fields = dict(opcode=r["opcode"], bytes=r["bytes"], phase=r["phase"], _defs=list(r["regs"]) + ([hi] if i == 0 else []), _uses=[])
                        if r["opcode"] == 14060:
                            # the ABI's system register is the ENCODED selector (g17asm.decode_sr:
                            # 130 = SR_SIMD_ELEM), not the decoder's printed MCRegister id 45 -
                            # ledger/g17-the-special-register-numbering-executes.toml; integration's
                            # review of d11a6905 caught the substitution
                            fields["sr"] = g17asm.decode_sr(r["bytes"])["sr"]
                        out.append(MInst("tensor.inherited" if r["inherited"] else "tensor.authored", r["length"], fields,
                                         note=("INHERITED from the 32x32x64 witness %s: %s" % (g17tensorlower.WD.WITNESS_OBJECT_SHA256[:16], r["phase"])) if r["inherited"] else ("authored: " + r["phase"])))
                    return
                # THE PRECISE REFUSAL (integration 132d34d4 asked for it in place of the opaque
                # "no opcode recorded for tensor.seq|172"): the native tensor lowering is
                # INCOMPLETE, and the cached sequence is not one instruction and not a kernel.
                # Measured against Apple's complete 32x32x64 witness (results/g17-tensor-common-
                # witness-v1, 128 instructions) the kernel has six phases and this compiler emits
                # none of them from the IR: (1) lane/address setup - op14060 reads the lane,
                # op17016 x2 and op423/426/10283/10286/586 form the per-lane tile addresses
                # (the two exact op17016 configurations are measured on lane inputs 0..31:
                # x>>2 and (x>>1)&3; see g17tensorsetupmodel. It is byte-identical when a buffer is
                # bound before A, so it is not an argument-position load - the binding is carried
                # by the tensor loads' and stores' BASE field, which reads the binding offset:
                # 0/2/4 for A/B/C, 2/4/6 with a pad bound first; renumbering 1/2/3 -> 2/4/6
                # changes no instruction); (2) accumulator zeroing -
                # op554 x 8mn+1 four-byte immediates; (3) tensor operand loads - op12674/12675,
                # 8(m+n) of them into aligned 4-register tiles, whose operand-7 offsets are
                # k_block * stride_bytes (the B-stride witness scales exactly those six offsets
                # 512->768, 1024->1536, 1536->2304 and two add immediates); (4) the MACs - op5106
                # x 4mn on aligned 8-register accumulators (register model measured); (5) the K
                # loop - two trips at K=64 through the counted-loop shape this compiler already
                # emits; (6) readout - op17257 range stores of the accumulators, a form this
                # compiler has never emitted. The lowering below consumes none of A, B, C.
                # ------------------------------------------------------------------------
                _ELEM = TENSOR_ELEMENT_BYTES
                # THE WHOLE-KERNEL ROUTE. The registry refused, so before the refusal below try
                # the general lowering (agxforge/g17/tensorgemm.py via tensor.emit_gemm).
                #
                # WHY WHOLE-KERNEL AND NOT ROW-SPLICING INTO AN ARBITRARY FUNCTION. The lowering
                # returns a COMPLETE kernel - it ends in END and allocates its own registers, up
                # to R83 on 32x32x64 - so its rows can only be emitted where the GEMM IS the whole
                # program. Splicing it into a function that also computes other things would put
                # an END mid-program and would need the lowering's allocation reconciled with this
                # compiler's. So the route is taken ONLY for a function whose sole work is this
                # matmul, and any other function gets the named refusal below with the reason.
                #
                # The register ceiling is declared on the first row exactly as the registry path
                # above does it, which is what reserves the lowering's range from the allocator.
                _fn = _CUR_FN[0]
                # THE NARROW SLICE. A SECOND TENSOR OPERATION still refuses: two complete kernels
                # cannot both be the whole program, and splicing either one would place an END
                # mid-program. Ordinary scalar/control/memory work beside ONE tensor body is now
                # composed, because the only thing that made it unsafe was the register plan, and
                # the allocator now knows what the body occupies.
                _tensor_others = [o for blk in _fn.blocks for o in blk.ops
                                  if o is not op and o.kind == "tensor_matmul"]
                _others = list(_tensor_others)
                # A STRIDE THAT IS NOT A WHOLE NUMBER OF ELEMENTS IS REFUSED, NOT TRUNCATED.
                # The conversion below is an integer division, so strideA = 3 bytes on halves
                # would have become lda = 1 and lowered a smaller program in silence. Caught by
                # the route's own test, which expected a refusal and got a program.
                _ragged = [n for n, w in (("strideA", _ELEM[at.get("a_dtype", "half")]),
                                          ("strideB", _ELEM[at.get("b_dtype", "half")]),
                                          ("strideC", 4)) if strides[n] % w]
                # THE PRESERVED REFUSALS ARE AN EXPLICIT GUARD, derived from the pinned cases
                # in g17regress._tensor_contract_delivered. That function asserts three refusals
                # and they do NOT share one discriminator, which is why this is two rules and not
                # a message match:
                #
                #   A at 18 tile rows (strideA=576)  message carries "multiply immediate"/"six-bit"
                #   B at 64 tile rows (strideB=2048) message carries "six-bit"
                #   16x32x64                          message carries "no witness"
                #
                # "six-bit" is specific to the two stride cases and is used. "no witness" is NOT:
                # every shape the registry lacks a witness for says it, including 17x19x16,
                # 50x37x80 and 16x16x16, so matching it would decline the whole route. That case
                # is therefore pinned BY SHAPE.
                #
                # THE STRIDE REFUSALS ARE LIFTED, BOUNDED BY WHAT WAS MEASURED (root's call,
                # 2026-09-28). They were encoding limits of the registry's multiply program (the
                # immediate's six bits), not of the general lowering. Its stride handling was
                # measured exact on hardware (recon section 47: stride_evidence.json, 3/3 at both
                # refused points; stride_sweep.json, 2/2 across tlower's DISP_MAX = 32,767-byte fold
                # and at 64 KiB and 1 MiB for A, 512 KiB for B, all at 32x32x64 half/half on the
                # campaign harness). So the route serves strideA <= 1 MiB and strideB <= 512 KiB,
                # and a registry refusal beyond that stands, naming the range: the folded
                # address lands in a 32-bit register, so a real ceiling exists somewhere above,
                # untested. 16x32x64 is pinned by shape only if TENSOR_PIN_16X32X64 is restored.
                # Everything else the registry refused for INCOMPLETE LOWERING is what the route
                # now serves.
                #
                # An opt-in flag was implemented first and removed: making ordinary tensor IR
                # require G17_TENSOR_WHOLE_KERNEL=1 preserves the contract but defeats the point
                # of routing, so the guard is a reviewed denylist and the route is on by default.
                _PINNED_SHAPES = {(16, 32, 64)} if TENSOR_PIN_16X32X64 else set()
                if (at["M"], at["N"], at["K"]) in _PINNED_SHAPES:
                    _others = None          # the original refusal stands, wording untouched
                # THE ROUTE SERVES ONLY THE MEASURED STRIDE RANGE, whatever the registry's reason: the
                # registry refuses 576 and 2048 as six-bit but a far larger stride for another reason,
                # and the general lowering was measured exact to STRIDE_MEASURED_MAX and no further.
                _beyond = [n for n, cap in STRIDE_MEASURED_MAX if strides[n] > cap]
                if _beyond and _others == []:
                    reason = ("%s; general lowering: %s beyond the measured stride range (%s)"
                              % (reason, " and ".join("%s=%d" % (n, strides[n]) for n in _beyond),
                                 ", ".join("%s <= %d bytes" % (n, cap) for n, cap in STRIDE_MEASURED_MAX)))
                    _others = None
                _ragged = [n for n, w in (("strideA", _ELEM[at.get("a_dtype", "half")]),
                                          ("strideB", _ELEM[at.get("b_dtype", "half")]),
                                          ("strideC", 4)) if strides[n] % w]
                if _ragged and _others == []:
                    reason = ("%s; general lowering: %s not a whole number of elements (%s)"
                              % (reason, " and ".join(_ragged),
                                 ", ".join("%s=%d" % (n, strides[n]) for n in _ragged)))
                    _others = None          # fall through to the refusal without the decline text
                if _others == []:
                    # g17tensor is already imported at module scope; model is local, in this
                    # file's own style for the tensor path.
                    from agxforge.g17 import model as _g17model
                    try:
                        _low = g17tensor.emit_gemm(
                            at["M"], at["N"], at["K"],
                            # THE IR CARRIES STRIDES IN BYTES (strideA/B/C, defaulted above);
                            # the lowering takes LEADING DIMENSIONS IN ELEMENTS. Reading
                            # at["lda"] would have found nothing and silently lowered every
                            # strided GEMM as contiguous - the same defect class as
                            # ownimage.build recomputing them, which this candidate also fixes.
                            lda=strides["strideA"] // _ELEM[at.get("a_dtype", "half")],
                            ldb=strides["strideB"] // _ELEM[at.get("b_dtype", "half")],
                            ldc=strides["strideC"] // 4,
                            a_type=at.get("a_dtype", "half"), b_type=at.get("b_dtype", "half"),
                            accumulate=bool(at.get("accumulate")),
                            transA=bool(at.get("transA")), transB=bool(at.get("transB")),
                            simdgroups=int(at.get("simdgroups", 1)),
                            grid=int(at.get("threadgroups", 1)),
                            grid_n=int(at.get("grid_n", 1)),
                            split_k=int(at.get("split_k", 1)),
                            split_fp32=bool(at.get("split_fp32")),
                            # THE EPILOGUE WAS SILENTLY DROPPED HERE: wired only into the composition
                            # route, so a single GEMM's scale/relu never ran (Set C's gemm_generic run,
                            # 2026-09-23: raw GEMM where relu(0.75 GEMM) was predicted).
                            epilogue=_tensor_epilogue(at, op),
                            saturate=bool(at.get("saturate")), kloop=bool(at.get("kloop")),
                            kloop_unroll=int(at.get("kloop_unroll", 1)),
                            **({} if at.get("kloop_chunk") is None else dict(kloop_chunk=at["kloop_chunk"])),
                            reduce=at.get("reduce"))
                    except ValueError as _why:
                        # ONLY THE LOWERING'S DOCUMENTED REFUSAL IS ABSORBED. tensorgemm raises
                        # ValueError('refused: ...') when it declines a shape; that is a result
                        # and becomes the reason on the refusal below. Anything else propagates -
                        # a KeyError, an AttributeError, a malformed lowering, or even a bare
                        # ValueError from a defect - because converting a bug into a named
                        # refusal makes every workload failure untrustworthy: the caller would
                        # read "this shape is not supported" where the truth is "this compiler is
                        # broken". The prefix is the discriminator, not the exception type.
                        if not str(_why).startswith("refused: "):
                            raise
                        _low, _lowreason = None, str(_why)
                    else:
                        _lowreason = None
                    if _low is not None:
                        _ins = [i for i in _g17model.decode(bytes(_low.body), 0) if i.opcode]
                        if not _ins or _ins[-1].opcode.id != 684:
                            raise Unsupported("tensor matmul %dx%dx%d: the general lowering's body "
                                              "does not end in END, so it is not a whole kernel" % shape)
                        # this compiler emits its own END for `ret`; drop the lowering's
                        _body = _ins[:-1]
                        # THE PRECISE OCCUPIED SET, not a ceiling. Declaring only
                        # plan["registers"] + 1 as a def reserved a RANGE and told the allocator
                        # nothing about which registers inside it are actually written, so a
                        # surrounding scalar value could still be coloured into a hole the body
                        # uses. Read the physical registers out of the body's own decoded
                        # operands and publish exactly those.
                        _names = _g17model.registers()
                        _occ = set()
                        for _i2 in _body:
                            for _k2, _v2 in _i2.values:
                                if _k2 == "reg":
                                    _occ.update(registerdomain.registers_in_name_checked(_names.get(_v2, "")))
                        if not _occ:
                            raise Unsupported(
                                "tensor row splice: the lowered body decodes to no register "
                                "operands, so the allocator cannot be told what it occupies")
                        _hi = sorted(_occ)
                        for _i, _inst in enumerate(_body):
                            _f = dict(opcode=_inst.opcode.id, bytes=bytes(_inst.raw),
                                      phase="general lowering", _defs=[], _uses=[])
                            if _i == 0:
                                _f["_occupies"] = _hi
                            if _inst.opcode.id in (14059, 14060):
                                # THE ABI NEEDS THE SYSTEM REGISTER DECLARED. The body reads one,
                                # and a program that reads an undeclared system register is what
                                # authorobj refuses by name ("no measured slot-29 entry for system
                                # register N"). The registry path declares it the same way.
                                _f["sr"] = g17asm.decode_sr(bytes(_inst.raw))["sr"]
                            out.append(MInst("tensor.wholekernel", len(_inst.raw), _f,
                                             note=("authored whole-kernel by tensorgemm.lower_gemm: "
                                                   "%dx%dx%d %s.%s%s" % (at["M"], at["N"], at["K"],
                                                   at.get("a_dtype", "half"), at.get("b_dtype", "half"),
                                                   " acc" if at.get("accumulate") else ""))))
                        return
                    reason = "%s; general lowering: %s" % (reason, _lowreason)
                elif _others:
                    reason = ("%s; row-splice route declined: the function contains %d other "
                              "tensor operation%s - two complete kernels cannot both be the whole "
                              "program, and splicing either would place an END mid-program"
                              % (reason, len(_others), "" if len(_others) == 1 else "s"))
                raise Unsupported(
                    "tensor matmul %dx%dx%d: the native lowering is incomplete - the IR's operands A, B, C "
                    "are not consumed, and none of the witness kernel's six phases (lane/address "
                    "setup op14060/op17016/op423/op426/op10283/op10286/op586, accumulator zeroing op554, "
                    "tensor operand loads op12674/op12675, MACs op5106, the K loop, readout op17257) is "
                    "emitted from the IR. The cached MAC-block sequence is an EXPERIMENT (pass "
                    "sequence_experiment=True to author it as one), not an executable kernel. "
                    "docs/archive/g17-architectural-compiler-handoff.md section 9 states what is measured, what "
                    "is missing and the discriminating controls. (%s)" % (shape + (reason,)))
            reg = _tensor_registry()
            seq = reg.get(shape)
            # K RETARGETING WAS TRIED AND WITHDRAWN. Three bits of tensor.bound.b track K/32
            # across six shapes, two of them preregistered and exact - but that is a DECODE
            # correlation, not the encoding. Authoring those bits alone changes nothing at
            # runtime: the patched kernel's output is bit-identical to K=64, while Apple's native
            # 32x32x160 differs completely. K also lives in at least four bytes of scalar setup
            # outside the tensor units. ledger/g17-tensor-k-field-retracted.toml
            if seq is None:
                raise Unsupported(
                    "tensor matmul %dx%dx%d: no reference sequence. isa/tensor-isa.toml records "
                    "that composing a novel tensor instruction SEQUENCE is unsolved - M, K, mode "
                    "and transpose are not located in the 12-byte units - so a shape with no "
                    "reference cannot be emitted. Available: %s"
                    % (shape + (", ".join("%dx%dx%d" % s for s in sorted(reg)),)))
            out.append(MInst("tensor.seq", sum(len(b) for _, b, *_ in seq.bounds)
                             + agxdis.MAC_UNIT * len(seq.macs),
                             dict(shape=shape, a_dtype=at["a_dtype"], b_dtype=at["b_dtype"]),
                             note="mac COUNT and ORDER inherited from %s" % seq.source))
        elif k == "exec_mask":
            # REFUSED: NOTHING EMITS THE COMPARE. op582 reads the FLAG a cmp (op10369) writes, and
            # the IR's cmp lowers only at a branch; an icmp is a 0/1 VALUE (op11372), not a flag.
            # This op emitted op582 alone, the mask came up empty whatever the condition, and the
            # guarded store never ran: Set C's round 4, 2026-09-23 (the tail stored with no exec
            # pair and was dropped with one, even under a condition true on every lane).
            raise Unsupported("exec_mask emits op582 with no compare in front of it, so the mask is "
                              "empty whatever the condition. Write the region as br_cond(cmp(...)) "
                              "into a block: the measured cmp -> op582 -> region -> op577 "
                              "(ledger/g17-the-mask-suppresses-a-store.toml)")
        elif k == "exec_restore":
            out.append(MInst("exec.restore", 4, dict(label="pred_%d" % id(op))))
        elif k == "cmp":
            # A placeholder: the real lowering is the compound cmp.pair emitted by select() at
            # the branch. It exists so that a cmp used for anything OTHER than a branch is
            # caught rather than silently dropped.
            out.append(MInst("cmp.placeholder", 0, {}, defs=[op.dest], uses=[op.args[0]]))
        elif k in ("store_tg", "load_tg"):
            # THREADGROUP MEMORY. Templates from a6-tgidx, one kernel, so the store's and the
            # load's address expressions agree by construction.
            #
            # THE BASE IS A REGISTER, IT IS READ, AND IT USED TO BE INHERITED. Operand 3 of both
            # forms was left at whatever register g17auth.encode put there - register 2 - and the
            # round trip worked because nothing in these programs writes register 2, so it reads
            # zero, which is the right base. Measured 2026-09-07 by writing the operand in the
            # program that proves the round trip: at registers 20 and 60 (both untouched, both
            # reading zero) the constant comes back; at register 0 it does not.
            #
            # That is a latent defect rather than a bug: correct today, wrong the first time the
            # allocator puts something in the named register. So the base is MATERIALISED now -
            # one zero per threadgroup access, chosen rather than inherited.
            #
            # ONE ZERO PER ACCESS, not one shared. The lifetime of operand 3 would live in operand
            # 4, which is absent from op12364's field map, so this backend cannot say "keep" about
            # the base at all. A shared zero read by two accesses would need one, and would be
            # refused; a private zero is read once and released.
            if k == "store_tg":
                val, idx = op.args
                # ONE REGISTER IN BOTH OPERANDS DOES NOT INDEX. Measured: every lane then writes
                # the base. Apple's own allocator never does it - zero of 866 corpus instances -
                # so the copy here is what its register allocator achieves by construction.
                if val is idx:
                    copy = ir.Value(getattr(val, "type", ir.I32),
                                    "%s_idx" % getattr(val, "name", "v"))
                    out.append(MInst("mov.4", 4, dict(keep_src=True),
                                     defs=[copy], uses=[val],
                                     note="copy: a threadgroup store cannot take one register as "
                                          "both its value and its index"))
                    idx = copy
                base = ir.Value(ir.I32, "tgbase_%d" % (id(op) & 0xFFFF))
                out.append(MInst("movimm.8", 8, dict(imm=0), defs=[base],
                                 note="the threadgroup base, materialised rather than inherited"))
                out.append(MInst("auth", len(TG_STORE_TEMPLATE),
                                 dict(opcode=TG_STORE_OPCODE, imms={}, srcmap=[0, 3, 5],
                                      template=TG_STORE_TEMPLATE), uses=[val, base, idx]))
            else:
                base = ir.Value(ir.I32, "tgbase_%d" % (id(op) & 0xFFFF))
                out.append(MInst("movimm.8", 8, dict(imm=0), defs=[base],
                                 note="the threadgroup base, materialised rather than inherited"))
                out.append(MInst("auth", len(TG_LOAD_TEMPLATE),
                                 dict(opcode=TG_LOAD_OPCODE, imms={}, srcmap=[3, 5],
                                      template=TG_LOAD_TEMPLATE),
                                 defs=[op.dest], uses=[base, op.args[0]]))
        elif k == "barrier":
            out.append(MInst("barrier", 6, dict(scope=op.attrs.get("scope", "threadgroup"))))
        elif k == "ret":
            out.append(MInst("end", 4, {}))
        else:
            raise Unsupported("IR op %r" % k)
    return out

# --- phase 2: register allocation ---------------------------------------------------------
# WHICH REGISTERS EACH FORM CAN NAME. Measured from the decoders, and it is not uniform:
#
#   alu.12/14  dest  FIVE bits   r0..r31      byte0[4], byte7[3,4], byte0[7], byte7[5]
#   alu.12/14  src1  EIGHT bits  r0..r255     2*(byte9 & 0x7F) + byte8[7]
#   alu.12/14  op B  FOUR bits   r0..r15      plus the half-register selector byte1[0]
#   load/store/read_sr/classb/movimm          FOUR bits, r0..r15, all in byte0[7:4]
#
# So a value may live above r15 ONLY if it is defined by an ALU and used only as an ALU src1.
# The register file is 32 (the ALU dest saturates at 31 across kernels needing 32 to 64 live
# values - ledger/g17-two-byte-pressure-threshold.toml), and the allocator was using twelve.
NARROW_MAX = 15
# THE WIDE POOL IS r16..r125, MEASURED. It was r16..r31, on the belief that the ALU's operand
# fields were 4 and 5 bits; they are 7 (ledger/g17-alu-slot-model.toml). A peer session then
# searched all four metadata sections of 924 objects and found nothing that declares register
# usage - the sections are the same size for a shader using R1 and one using R125 - and 39% of
# Apple's own shaders exceed r31, its driver shaders reaching R112.
#
# So the ceiling was asked of the hardware, one value pinned into one high register and read back
# (spike/accel/re/hireg.py, one dispatch per process, 107 preregistered every time):
#
#     r20   107   correct   <- control, inside the old ceiling
#     r40   107   correct       r100  107  correct
#     r64   107   correct       r120  107  correct
#     r124  107   correct       r125  107  correct
#     r126+  squashed as a whole instruction (no register or memory side effect)
#
# Section 135 corrected the original zero-filled-buffer interpretation: these operands are not
# hardwired zero.  Any instruction naming R126 or above is simply not executed, so the compiler
# rejects the namespace rather than emitting a silent no-op.
#
# The two failures are what make the rest mean something: a test where every register "worked"
# would be measuring a truncated field rather than a register file. And the edge is Apple's own -
# the highest register in 924 objects is R125.
WIDE_MAX = registerdomain.ALLOCATABLE_MAX
REGISTER_COUNT = registerdomain.REGISTER_COUNT
FP32_ACCUMULATOR_GROUP_WIDTH = registerdomain.FP32_ACCUMULATOR_GROUP_WIDTH
MAX_FP32_ACCUMULATOR_GROUPS = registerdomain.MAX_FP32_ACCUMULATOR_GROUPS


def _check_physical_registers(insts, context="compiler stream"):
    """Reject virtual/physical fields that name the squashed R126+ namespace."""
    values = []
    for m in insts:
        fields = getattr(m, "fields", {}) or {}
        for key in ("_defs", "_uses", "_occupies"):
            values.extend(fields.get(key, ()) or ())
    bad = registerdomain.invalid_registers(values)
    if bad:
        raise Unsupported("%s names unsupported register(s) %s; only R0..R125 are allocatable" %
                          (context, ", ".join("R%d" % r for r in bad)))

def _tensor_row_registers(m):
    """The physical registers a spliced tensor row names (decoded from its pre-encoded bytes)."""
    from agxforge.g17 import model as _g17model
    names = _g17model.registers()
    out = set()
    for inst in _g17model.decode(bytes(m.fields["bytes"]), 0):
        for kind, value in inst.values:
            if kind == "reg":
                out.update(registerdomain.registers_in_name_checked(names.get(value, "")))
    return out


def _check_tensor_body_clobbers(insts, regs_of=None):
    """REFUSE A SCALAR VALUE LIVE ACROSS A TENSOR BODY IN A REGISTER THAT BODY NAMES.

    A spliced tensor row carries fixed physical registers the allocator did not assign. Today the
    pool excludes their union for the whole program (Alloc.run), so this cannot fire; it is the
    check that makes any narrower reservation safe - a body modelled as a clobber at its position,
    say - because it reads the FINAL physical list, not the allocator's reasoning. Liveness is the
    allocator's own fixed point over the real control-flow graph, on physical registers."""
    regs_of = regs_of or _tensor_row_registers
    shim = [MInst(m.form, m.size, m.fields, defs=list(m.fields.get("_defs") or ()),
                  uses=list(m.fields.get("_uses") or ())) for m in insts]
    live_in, live_out = Alloc._cfg_live(shim)
    cache = {}
    for i, m in enumerate(insts):
        if m.form != "tensor.wholekernel":
            continue
        across = live_in[i] & live_out[i]
        if not across:
            continue
        key = bytes(m.fields["bytes"])
        named = cache.get(key)
        if named is None:
            named = cache[key] = regs_of(m)
        hit = sorted(across & named)
        if hit:
            raise Unsupported("tensor body row %d (%s) names r%s while a scalar value lives across it "
                              "in that register" % (i, m.note, ", r".join(str(r) for r in hit)))


def _clone_minst(m):
    """A copy of a selected instruction that allocation can mutate without touching the original:
    its own fields dict (list values copied, since allocation appends to them) and defs/uses lists,
    the same ir.Value objects."""
    c = MInst(m.form, m.size, {k: (list(v) if isinstance(v, list) else v) for k, v in m.fields.items()},
              m.defs, m.uses, m.note)
    return c


class Alloc:
    """Deliberately dumb and deliberately correct: linear scan over one namespace, no spilling.

    THIS CLASS DOES NOT SPILL. THE COMPILER DOES, one pass above it. `_spill_pressure(fn, budget)`
    keeps at most `budget` values live by spilling to scratch and reloading at each use, runs after
    rematerialisation to pick up what remat cannot, and is called from the compile driver before
    allocation. `integration.automatic_spill` carries passing hardware receipts for it -
    `results/g17-automatic-spill-runtime-v2`, n91 at one spill group and 16 scratch bytes per
    thread, n200 at 28 groups and 448 bytes. So Alloc's own "no spilling" is a statement about
    Alloc, and the program as a whole may well have spilled before reaching it.

    THIS DOCSTRING SAID "SPILLING IS NOT IMPLEMENTED" UNTIL 2026-09-19 AND THAT HAD STOPPED BEING
    TRUE. It is the second time: it previously carried a reason - that a spill needs a store to
    scratch with a COMPUTED address and the recovered store carries a slot - which the vector
    memory delivery (handoff 10ae) closed and nobody came back to edit, and it recorded that fact
    about itself. What closed it: `g17asm.encode_vec4` takes an INDEX REGISTER in both directions,
    `encode_vec4("store", index=7)` and `index=9` differ in byte 3 and decode back as 7 and 9, and
    the matching load reads through the same field. That evidence stays here because
    `test/test_g17spillgap.py` asserts it does - the correction belongs at the point of the claim,
    and a test that pins the identifier rather than the prose is what kept it there when this
    docstring was rewritten. It then listed three things that "actually block a spill now": a declared scratch
    binding and an ABI, the four-register granularity, and "the work - spill-slot layout, lifetime
    handling across the store, and a reload that the scan understands - none of it written". All
    three are closed. The work is `_spill_pressure`, `_emit_spill`, `_spill_candidates`,
    `_spill_address` and `_restride_spill`; the scratch binding is what the receipts above measure
    in bytes per thread. A docstring that records having gone stale once, and then goes stale the
    same way, is the argument for a check rather than a careful reader.

    WHAT REMAINS TRUE OF THIS CLASS: it is a linear scan over one namespace, it does not spill, and
    it raises `Unsupported` when it runs out - which now means the spill pass above it did not
    relieve enough, not that no spill exists. The ceiling `tools/g17spillgap.py` measures, 118
    simultaneously-live values with 119 refusing, is Alloc's own with the spill pass disabled.

    THE REFUSAL BELOW STILL SAYS "no spill form recovered" AND IS DELIBERATELY LEFT ALONE. It is
    the wrong phrase - a spill form exists and the compiler uses it - but `tools/g17spillgap.py`
    matches on that exact string as its detector and `test/test_g17spillgap.py` asserts it, so
    rewording it here would silently change what another lane's measurement detects and take its
    test red for a cosmetic gain. Whoever owns that tool should retire the phrase in both places at
    once; it is not a thing to fix from this end.
    """
    def __init__(self, regs=None, wide=None):
        # narrow pool: usable by every form. wide pool: ALU-only values.
        self.regs = list(regs if regs is not None else range(4, NARROW_MAX + 1))
        self.wide = list(wide if wide is not None else range(NARROW_MAX + 1, WIDE_MAX + 1))
        self.sr_compact = False          # set only for the retry after a refusal (run)
        self.hole_aware = False          # set only for the third attempt (run)
        self.atomic_wide = False         # set only for the fourth attempt (run)
        bad = registerdomain.invalid_registers(self.regs + self.wide)
        if bad:
            raise Unsupported("allocator pool names unsupported register(s) %s; only R0..R125 are allocatable" %
                               ", ".join("R%d" % r for r in bad))

    # Ops whose operands may be exchanged without changing the result. sub is NOT here.
    COMMUTATIVE = {3, 1}          # ALU_OP values for add and mul

    @classmethod
    def commute(cls, insts):
        """OPERAND COMMUTING TO RELIEVE REGISTER PRESSURE.

        Operand B is a FOUR-bit field, so any value placed there must live in r0..r15, while src1
        is eight bits and can reach the wide half of the file. For a commutative op the compiler
        chooses which operand goes where, and the choice decides how many values are confined to
        the narrow pool.

        The accumulate pattern makes this concrete. In acc = add(acc, v) repeated N times, putting
        each v in operand B confines ALL N values to r0..r15 and the allocator runs out at twelve.
        Putting the ACCUMULATOR there confines only the accumulator, which is one live value, and
        every v becomes eligible for r16..r31.

        The rule: place in operand B whichever operand was defined MORE RECENTLY - the running
        accumulator - and leave the longer-lived operand in src1.
        """
        defined_at = {}
        for i, m in enumerate(insts):
            for v in m.defs: defined_at[v] = i
        swapped = 0
        for i, m in enumerate(insts):
            if m.form not in ("alu.12", "alu.14") or len(m.uses) != 2: continue
            if m.fields.get("op") not in cls.COMMUTATIVE: continue
            a, b = m.uses
            if defined_at.get(a, -1) > defined_at.get(b, -1):
                m.uses[0], m.uses[1] = b, a          # the fresher operand goes to operand B
                swapped += 1
        return swapped

    @staticmethod
    def _needs_narrow(insts, wide_ok=()):
        """A value needs a narrow register unless every form touching it can NAME a wide one.

        This used to confine anything outside alu.12 to r0..r15, which was right when the store's
        source was read as a four-bit field and the bitwise and shift destinations as seven. They
        are seven and eight bits now - byte2[6], byte2[7] and byte5[4] extend the store's source,
        byte2[4] the ALU destination, byte2[3]/byte2[4]/byte2[6]/byte2[7] the movimm's - so only
        the forms with genuinely small fields still constrain, and read_sr's four-bit destination
        is the one that does.

        AND `auth` IS NOT ON THIS LIST BY OMISSION, IT IS OFF IT BY MEASUREMENT. Two mechanisms
        were answering the same question and disagreeing: this hand-maintained set, and _caps,
        which computes each value's highest legal register from the FIELD WIDTHS in the authoring
        table. The generic path's operands were being forced into the twelve narrow registers by
        the list while _caps said 127, so a kernel with eleven live values ran out at an indexed
        store whose index field is seven bits (op17229/8 operand 5, verified over 98 instances,
        with Apple emitting r89 in operand 0). The field widths are the authority; the list is a
        statement about forms that do NOT go through them.
        """
        need = set()
        for m in insts:
            if m.form == "read_sr.4":
                need.update(m.defs)
            elif m.form not in ("alu.12", "alu.14", "store.8", "store.14", "movimm.8",
                                # bitwise.reg JOINED ITS IMMEDIATE TWIN 2026-09-12. It was the only
                                # member of the pair on this list, so a reg-reg bitwise held every
                                # value it touched to the twelve narrow registers while its own
                                # measured fields hold sixty-four - and that confinement is what
                                # hid a field bug for as long as it lasted: BW_R_DEST was five bits
                                # where the form's is six, unreachable while nothing allocated a
                                # bitwise destination above r15. _caps bounds these values from the
                                # field widths now, which is what this function's docstring says
                                # the authority is.
                                "bitwise.reg",
                                "bitwise.imm", "alu.sub.imm", "alu.sub.reg", "alu.mul.imm",
                                "alu.mul.reg", "alu.shift.imm", "alu.shift.reg",
                                "load.10", "load.14", "mov.4", "acc.read.4", "acc.write.4", "acc.write.10",
                                "acc.scale.14", "auth") + tuple(wide_ok):
                need.update(m.defs); need.update(m.uses)
            elif len(m.uses) > 1:
                # SLOT A IS SEVEN BITS, not four (g17asm.SLOT_A - byte5[7] and byte8[1:0] are its
                # top three register bits, recovered 2026-09-04). So a value used as an ALU
                # operand B is no longer confined to the narrow pool. The pool ITSELF stays at
                # r0..r31: the other forms have their own smaller fields, and how many registers
                # a program may use without redeclaring them in the host's descriptors is not
                # established, so the field being wide is not licence to allocate wider.
                pass
        return need

    @staticmethod
    def _narrow_by_float_unary_only(insts):
        """Values that _needs_narrow confines ONLY because a float.unary touches them.

        THE FLOAT UNARY'S FIELDS ARE NOT NARROW. Its destination is seven bits (g17asm.TRANS_DEST)
        and its source eight (TRANS_SRC_MAX 143), and Apple uses them: in the corpus 26 of 410
        exp2, 438 of 1,244 recip, 40 of 276 log2 and 4 of 112 rsqrt write r16 or above, to r96.
        _needs_narrow still lists the form among those confined to r0..r15, and every program that
        executed was allocated that way - the LayerNorm's rsqrt sits at r6 - so the confinement is
        kept as the PREFERENCE and lifted only when the narrow pool is empty: the attention softmax
        keeps 32 exp2 results live against twelve narrow registers, which no preference can fit.
        A value confined by any other form is not lifted. Programs that fit are allocated exactly
        as before, which is what keeps the validated hashes where they are.
        """
        by_other, by_unary = set(), set()
        for m in insts:
            if m.form == "float.unary":
                by_unary.update(m.defs); by_unary.update(m.uses)
            elif m.form == "read_sr.4" or m.form not in (
                    "alu.12", "alu.14", "store.8", "store.14", "movimm.8", "bitwise.imm",
                    "alu.sub.imm", "alu.sub.reg", "alu.mul.imm", "alu.mul.reg", "alu.shift.imm",
                    "alu.shift.reg", "load.10", "load.14", "mov.4", "acc.read.4", "acc.write.4", "acc.write.10",
                    "acc.scale.14", "auth"):
                by_other.update(m.defs); by_other.update(m.uses)
        return by_unary - by_other

    @staticmethod
    def _caps(insts):
        """{value: the highest register it may be given}, from the FIELD WIDTHS themselves.

        A register operand's field is as wide as it is, and 642 opcodes in the authoring table are
        NARROW - the form reaches only part of its register class. Allocating r12 into a field that
        holds three bits is not a wrong number, it is an encoding that cannot be written at all,
        and the failure arrives at emit time as a ValueError from the encoder rather than as
        anything the allocator could have explained.

        So the cap is computed before allocation and the linear scan honours it: a value takes the
        lowest free register at or below the smallest cap any instruction touching it imposes.
        Without this the generic form is unusable in any program long enough to need r16.
        """
        cap = {}
        def put(v, c):
            cap[v] = min(cap.get(v, 1 << 30), c)
        for m in insts:
            if m.form == "float.unary":
                for v in m.defs: put(v, 127)
                for v in m.uses: put(v, g17asm.TRANS_SRC_MAX)
            elif m.form == "bitwise.reg":
                # FROM THE FIELD WIDTHS, as this function's docstring requires. The four-byte form
                # holds six bits in each of its three register operands; the ten-byte form holds
                # more (ALU_DEST 8, BW_SRC 7, BW_IMM 8 carrying register << 1), and the narrowest
                # of those is what bounds a value that might land in either.
                for v in m.defs:
                    put(v, (1 << len(g17asm.ALU_DEST)) - 1)
                for v in m.uses:
                    put(v, (1 << len(g17asm.BW_SRC)) - 1)
            elif m.form == "acc.read.4":
                for v in m.defs: put(v, 63)          # op586's four-byte destination is six bits
            elif m.form == "acc.scale.14":
                for v in m.uses: put(v, 125)         # op3290/14's source field names any allocatable register
            elif m.form == "auth":
                opc = m.fields["opcode"]
                dsts, srcs = g17auth.register_operands(opc)
                classes = g17auth.record(opc)["operands"]
                fm = g17auth.fields(opc)
                for i, v in list(zip(dsts, m.defs)) + list(zip(srcs, m.uses)):
                    top = (1 << (max(j for j, _, _, _ in fm[i][1]) + 1)) - 1
                    put(v, top // 2 if fm[i][0] == "slot" else top)
        return cap

    @staticmethod
    def _cfg_live(insts):
        """live-in and live-out per instruction, by fixed point over the real control-flow graph.

        THE ONE PLACE LIVENESS IS DECIDED. Register freeing and every lifetime bit - the ALU's
        keep and src2_keep, the auth path's keeps, keep_a/keep_b, keep_src - all read the table
        this produces, so they cannot disagree with each other. They did: the scan took the last
        MENTION of a value in the instruction list as its last read, which is true of straight-line
        code and false the moment there is a back edge, and one wrong order produced two different
        silent wrong answers - a register handed away inside a loop AND a source released while
        still needed.

        Successors are fall-through plus the target of any branch. Predication is NOT control flow:
        exec.mask and exec.restore change which lanes execute, not which instruction runs next, so
        they fall through like anything else.

        The standard backward equations, to a fixed point:

            live_out[i] = union of live_in[s] over successors s
            live_in[i]  = uses[i] + (live_out[i] - defs[i])
        """
        n = len(insts)
        labels = {m.fields["label"]: i for i, m in enumerate(insts)
                  if m.form == "label" and "label" in m.fields}
        succ = []
        for i, m in enumerate(insts):
            s = [i + 1] if i + 1 < n else []
            if m.form in ("branch.cond.back", "branch.cond.fwd"):
                t = labels.get(m.fields.get("target"))
                if t is None:
                    raise Unsupported("branch to unplaced label %r" % m.fields.get("target"))
                s.append(t)
            succ.append(s)
        uses = Alloc._cfg_uses(insts, labels)
        defs = [set(m.defs) for m in insts]
        live_in = [set() for _ in range(n)]
        live_out = [set() for _ in range(n)]
        changed = True
        while changed:
            changed = False
            for i in range(n - 1, -1, -1):
                lo = set()
                for j in succ[i]:
                    lo |= live_in[j]
                li = uses[i] | (lo - defs[i])
                if lo != live_out[i] or li != live_in[i]:
                    live_out[i] = lo; live_in[i] = li; changed = True
        return live_in, live_out

    @staticmethod
    def _cfg_uses(insts, labels=None):
        """The values each instruction READS for liveness: its operands, and at a back edge the
        latch values of its header's phis. The equations in _cfg_live run over this, and so must
        anything that checks them (g17regress's fixed-point case).

        A LATCH VALUE IS READ BY THE BACK EDGE. The phi marker carries no uses, so the value the
        phi takes next trip (i + 1) looked dead after its last mention in the body. A runtime-bound
        latch mentions it twice (i < n, then the cap i < K), and the cap compare RELEASED the
        counter: per section 25.95 the lane's copy reads 0 next trip, i sticks at 1 and the loop
        never exits. It ran away on the GPU on 2026-09-24 (tools/g17releasereuse.py, 8,192
        threadgroups). Each back edge to a header now uses the latch members of that header's phis.
        """
        n = len(insts)
        if labels is None:
            labels = {m.fields["label"]: i for i, m in enumerate(insts)
                      if m.form == "label" and "label" in m.fields}
        uses = [set(m.uses) for m in insts]
        # AN ENTRY VALUE IS READ BY THE EDGE INTO THE LOOP. The phi marker carries no uses, and when the phi
        # takes its entry value's register the entry copy disappears, so the entry value looked dead after
        # its last mention BEFORE the loop - and that mention RELEASED it (lifetime 16), so the loop's first
        # trip read 0. Measured: build_rmsnorm_wide's scale loop started from t, t's last mention was the
        # second x load, and all 1,024 threads stored element 0 (MM 25.141.4). The instruction that falls
        # into a header's label now uses the entry members of that header's phis.
        for h, m in enumerate(insts):
            if m.form != "label" or h == 0 or h + 1 >= n or insts[h + 1].form != "phi":
                continue
            for j in range(h + 1, n):
                if insts[j].form != "phi":
                    break
                g = insts[j].fields.get("phi_group", ())
                if len(g) > 1 and isinstance(g[1], ir.Value):
                    uses[h - 1].add(g[1])
        for i, m in enumerate(insts):
            if m.form != "branch.cond.back":
                continue
            h = labels[m.fields["target"]]
            for j in range(h + 1, n):
                if insts[j].form != "phi":
                    break
                uses[i] |= {v for v in insts[j].fields.get("phi_group", ())[2:] if isinstance(v, ir.Value)}
        return uses

    def run(self, insts):
        _check_physical_registers(insts, "compiler stream")
        # A ROW-SPLICED TENSOR BODY OCCUPIES PHYSICAL REGISTERS THIS ALLOCATOR DID NOT ASSIGN.
        # Its rows carry pre-encoded bytes with fixed register numbers, so they have no virtual
        # defs for the allocator to colour, and without this the allocator would hand the same
        # physicals to surrounding scalar values - two writers, one register, no diagnostic. The
        # route publishes the set as `_occupies`; the pool excludes it for this run, which is what
        # makes ONE allocator cover the whole function rather than two plans over one file.
        occupied = set()
        for _m in insts:
            occupied |= set((getattr(_m, "fields", None) or {}).get("_occupies", ()))
        _saved = (self.regs, self.wide)
        if occupied:
            # THE REFUSAL MUST NOT LEAVE THE POOL MUTATED. This raised while self.regs and
            # self.wide were already narrowed and before the try below, so the finally never ran
            # and an Alloc instance reused after a refusal carried a permanently shrunken pool -
            # a later allocation would fail for a reason belonging to an earlier program. Root
            # found it (c0ad2c32) and it is a real defect in my slice: the restoration was
            # correct for the success path and absent for the refusal path, which is the half a
            # passing test would not reach.
            self.regs = [r for r in self.regs if r not in occupied]
            self.wide = [r for r in self.wide if r not in occupied]
            if not self.regs:
                self.regs, self.wide = _saved
                raise Unsupported(
                    "tensor row splice: the lowered body occupies every register in the "
                    "allocator's pool (%d registers), leaving none for the %d surrounding "
                    "operations. Raise the pool or lower the tensor body with a smaller register "
                    "budget." % (len(occupied), len(insts)))
        # A SECOND ATTEMPT ONLY AFTER A REFUSAL, with compact read_sr placement: an earlier read_sr
        # register whose value is dead is reused BEFORE a fresh narrow register is taken, so the
        # twelve narrow registers are left for the values that need them. It runs on a clone of the
        # selected instructions (the first attempt mutates them) and only when the first attempt
        # refused, so every program that compiled before is allocated exactly as it was. One-dispatch
        # attention needed it: 13 read_sr values held all twelve narrow registers from four softmax
        # rows up, although no more than a few were ever live (Set A item 6).
        snapshot = [_clone_minst(m) for m in insts]
        # A THIRD ATTEMPT, also only after refusals: HOLE-AWARE placement. A form whose destination
        # field cannot author some registers (cvt.f16.f32 cannot write d = 14, 15, 30, 31, ...;
        # _encodable_dests) found only those holes free in a looped RMSNorm (MM 25.136.6), because
        # values with no such limit had taken the authorable ones first. Here an unconstrained value
        # takes a hole register first, leaving authorable registers to the forms that need them.
        snapshot2 = [_clone_minst(m) for m in insts]
        # THEN THE SAME THREE AGAIN, also only after refusals, with several uniform atomics' destinations from the
        # WIDE pool (bit 1 clear), which the narrow pool of six cannot hold beside a program's other narrow values
        # (the last-threadgroup norm and the fused attention merge, MM 25.140.4)
        pristine = [_clone_minst(m) for m in insts]
        try:
            try:
                return self._run(insts)
            except Unsupported as first:
                self.sr_compact = True
                try:
                    return self._run(snapshot)
                except Unsupported:
                    self.hole_aware = True
                    try:
                        return self._run(snapshot2)
                    except Unsupported:
                        self.atomic_wide = True
                        try:
                            for k in range(3):
                                self.sr_compact, self.hole_aware = k >= 1, k >= 2
                                try:
                                    return self._run([_clone_minst(m) for m in pristine])
                                except Unsupported:
                                    pass
                            # THE LAST ATTEMPT (MM 25.144.8): a tensor body's working registers for scalar
                            # values that live across no body naming them. Only after every other attempt
                            # refused, so every program that compiled before keeps its bytes. The body rows'
                            # registers stay out of the pool; a value takes one only through _run's interval
                            # test, and _check_tensor_body_clobbers re-checks the final list.
                            self._body_pool = sorted(r for r in occupied - _TENSOR_PERSISTENT
                                                     if r > NARROW_MAX and r in _saved[1])
                            why_share = None
                            if self._body_pool:
                                self.body_share = True
                                try:
                                    for k in range(3):
                                        self.sr_compact, self.hole_aware = k >= 1, k >= 2
                                        try:
                                            return self._run([_clone_minst(m) for m in pristine])
                                        except Unsupported as e:
                                            why_share = why_share or e
                                finally:
                                    self.body_share = False
                            # THE INTERVAL ATTEMPT (MM 25.144.12), after every attempt above refused, so every
                            # program that compiled before keeps its bytes. Two reservations that outlived their
                            # need are released to their live intervals:
                            #   * a tensor body's NARROW working registers join the shared pool (the attempt above
                            #     shares only the wide ones), under the same test: no row naming the register lies
                            #     in the value's interval;
                            #   * a pre-coloured register (a loop phi group, a read_sr destination, a range or tuple
                            #     group) is held from the earliest definition of its values to their last live index
                            #     (CFG liveness: a loop-carried value lives to its back edge), not for the whole
                            #     program; outside that span another value may borrow it.
                            # Beside a K-looped tensor body the narrow pool was 5 registers (r4..r15 less the
                            # body's), and a scalar tail's read_sr values and loop counter held 4 of them for the
                            # whole program, so a counted-loop tail could not allocate (g17swigluqmm).
                            # SCOPED TO PROGRAMS WITH A TENSOR BODY. On scalar programs the same rule lifts the
                            # measured ceiling tools/g17spillgap.py records (118 live values; its 119-value kernel
                            # compiles clash-free under it, the thread-id read_sr register being the 119th), which is
                            # that lane's number to move with its report, not a side effect of this one.
                            if not occupied:
                                if self._body_pool:
                                    raise Unsupported("%s; and with %d tensor-body working registers shared between "
                                                      "bodies: %s" % (first, len(self._body_pool), why_share))
                                raise first
                            self._body_pool = sorted(r for r in occupied - _TENSOR_PERSISTENT
                                                     if (r > NARROW_MAX and r in _saved[1]) or
                                                     (r <= NARROW_MAX and r in _saved[0]))
                            self.body_share = self.interval_pre = True
                            why_iv = None
                            try:
                                for k in range(3):
                                    self.sr_compact, self.hole_aware = k >= 1, k >= 2
                                    try:
                                        return self._run([_clone_minst(m) for m in pristine])
                                    except Unsupported as e:
                                        why_iv = why_iv or e
                            finally:
                                self.body_share = self.interval_pre = False
                            raise Unsupported("%s; and with %d tensor-body working registers shared between "
                                              "bodies: %s; and with pre-coloured registers held only over their "
                                              "intervals: %s" % (first, len(self._body_pool), why_share, why_iv))
                        finally:
                            self.atomic_wide = False
                    finally:
                        self.hole_aware = False
                finally:
                    self.sr_compact = False
        finally:
            self.regs, self.wide = _saved

    def _run(self, insts):
        # BEFORE LIVENESS, because inserting an instruction shifts every index it computes.
        self.phi_copies = _phi_interference_copies(insts)
        self.commute(insts)
        # `last` is now DERIVED from liveness rather than from program order: the last index at
        # which a value is live at all. For straight-line code that is its last use, unchanged. For
        # a value carried around a back edge it is the branch, because the branch's successor is
        # the loop top where the value is live-in - which is the fact a linear scan cannot see.
        live_in, live_out = self._cfg_live(insts)
        last = {}
        for i in range(len(insts)):
            for v in live_in[i] | live_out[i]:
                if last.get(v, -1) < i:
                    last[v] = i
        _phi_entries_live_to_header(insts, last)
        self._lifetimes(insts, last)
        # REGISTER-RANGE STORES. A store writes r<src>..r<src+n-1> to n consecutive slots, so
        # the values it stores must land in CONSECUTIVE registers. A general allocator would
        # coalesce; this one pre-colours, which is enough because a range group's members are
        # always defined before the store and never overlap another group.
        pre = {}
        # READ_SR HAS A FOUR-BIT DESTINATION. Every other form this backend emits can name r0..r125
        # now, so the special register read is the one hard constraint left, and it has to be
        # pre-coloured before the range stores take the low registers - otherwise a program whose
        # pool starts high fails to allocate at all rather than for any semantic reason.
        # the hole-aware attempt also lets cvt.f16.f32 name wide registers: its measured fields do
        # (_encodable_dests authors destinations to r61, holes excepted, and _caps bounds its source
        # from the field width); only the first two attempts keep it narrow, so every program that
        # compiled before keeps its bytes
        narrow_needed = self._needs_narrow(insts, ("cvt.f16.f32", "cvt.f32.f16") if getattr(self, "hole_aware", False) else ())
        narrow_soft = self._narrow_by_float_unary_only(insts)
        cap = self._caps(insts)
        # LOW REGISTERS ARE THE SCARCE ONES when any form has a genuinely narrow field, so
        # everything unconstrained is pushed to the top of the file and the narrow operands keep
        # the bottom. Without this a program allocates r0..r9 to its stores and then cannot place
        # a source that must sit at or below r7 - which is a scheduling failure reported as an
        # out-of-registers error, and the wrong diagnosis for a file that is nearly empty.
        #
        # The reordering only happens when a cap actually binds, so every program that does not
        # use a narrow form allocates exactly as it did before.
        tight = bool(cap) and min(cap.values()) < max(self.regs)
        SR_MAX = 15
        cursor = list(self.regs)
        defidx = {}
        for _i, _m in enumerate(insts):
            for _v in _m.defs:
                defidx[_v] = _i
        busy = {}                       # register -> instruction index it stays occupied through (tuple loads and range stores)
        # The measured constant-preload requantization class has a fixed two-register body:
        # printed r105 carries the scalar value (logical r0) and printed r106 carries the
        # thread index (logical r1).  This is a class constraint, not a general allocator rule;
        # selection marks only the exact retained stage rows.  Pre-colour both definitions and
        # uses before the ordinary read_sr/range machinery so the raw measured bytes cannot be
        # paired with a different register plan.
        for m in insts:
            for key in ("requant_fixed_defs", "requant_fixed_uses"):
                if key not in m.fields:
                    continue
                requested = m.fields[key]
                vals = m.defs if key.endswith("defs") else m.uses
                if len(requested) != len(vals):
                    raise Unsupported("measured requantization row has %s for %d values, not %d"
                                      % (key, len(requested), len(vals)))
                for value, reg in zip(vals, requested):
                    if value in pre and pre[value] != reg:
                        raise Unsupported("measured requantization row needs value %r in both r%d and r%d"
                                          % (value, pre[value], reg))
                    pre[value] = reg
        # A TUPLE LOAD DEFINES CONSECUTIVE REGISTERS (handoff 10ae): the vector load's four lanes take a run of the
        # cursor, busy from the load through the last use of any lane - the range store's reservation, on defs.
        # MANY TUPLE LOADS (MM 25.139.2): a run whose last lane use is behind this load is REUSED - its registers left
        # the general pool, so only tuples ever hold them, and a tuple defined and consumed within one block (a loop
        # body's) is dead before the next tuple's load. And when a narrow cap binds, runs are taken from ABOVE it:
        # the lowest run is Apple's placement for its one tuple, but a dozen of them took every narrow register.
        _tuple_regs = []
        _tuple_count = sum(1 for m in insts if m.fields.get("tuple_group"))
        _floor = min(cap.values()) if (tight and _tuple_count > 1) else -1
        # and MANY tuples come from the WIDE pool (the tuple field is seven bits: r0..r127): the narrow pool is
        # twelve registers, and four two-word tuples took eight of them
        _tcursor = sorted(r for r in self.wide if r <= 127 and r not in set(pre.values())) if _tuple_count > 1 else cursor
        for _at_i, m in enumerate(insts):
            if not m.fields.get("tuple_group"): continue
            n = len(m.defs)
            _end = max([j for j, x in enumerate(insts) for v in m.defs if v in x.uses] or [_at_i])
            _old = sorted(r for r in _tuple_regs if busy.get(r, -1) < _at_i)
            reuse = [j for j in range(len(_old) - n + 1) if _old[j:j+n] == list(range(_old[j], _old[j] + n))] if _tuple_count > 1 else []
            if reuse:
                regs_taken = _old[reuse[0]:reuse[0] + n]
            else:
                tc = _tcursor
                runs = [j for j in range(len(tc) - n + 1) if tc[j:j+n] == list(range(tc[j], tc[j] + n)) and all(busy.get(r, -1) < _at_i for r in tc[j:j+n])]
                if not runs: raise Unsupported("no %d consecutive registers free for a tuple load at +%d" % (n, _at_i))
                high = [j for j in runs if tc[j] > _floor]
                j = min(high or runs, key=lambda j: tc[j])         # the lowest run: Apple's tuple sits at r0..r3
                regs_taken = tc[j:j+n]
                for r in regs_taken: tc.remove(r)
                _tuple_regs.extend(regs_taken)
            for off, v in enumerate(m.defs): pre[v] = regs_taken[off]
            for r in regs_taken: busy[r] = _end
        sr_owner = {}                   # read_sr register -> (last live index of its value, value)
        at_index = {id(x): j for j, x in enumerate(insts)}
        for m in insts:
            # THE UNIFORM ATOMIC'S DESTINATION HAS HOLES. Sweeping r105..r136 through the
            # encoder, only indices with bit 1 CLEAR author - 0,1,4,5,8,9,... - because the map
            # claims byte0[5] as the field's weight-1 carrier and the decoder does not give it
            # back. The read-back check refuses the rest rather than emitting a register nobody
            # asked for, so the allocator has to hand it one that exists.
            if m.form == "atomic.uniform.10":
                # under the tied probe the VALUE takes the same register as the destination, which
                # means it too has to come from the encodable half of the field
                for v in m.defs:
                    if v in pre:
                        continue
                    # A WIDE register with bit 1 clear when the program has several (the encoder's read-back
                    # refuses any the field cannot author): each held one is gone for the whole program,
                    # and the narrow pool has only six. The FOURTH attempt only (run), after the narrow
                    # placement refused, so every program that compiled before keeps its bytes: taking wide
                    # registers first cost a 125-register residency probe its 126th (test_g17residency).
                    taken = set(pre.values())
                    n_atomic = sum(1 for x in insts if x.form == "atomic.uniform.10")
                    r = None
                    if n_atomic > 1 and self.atomic_wide:
                        r = next((x for x in self.wide if not (x & 2) and x not in taken and x <= 127), None)
                    if r is None:
                        r = next((x for x in cursor if not (x & 2)), None)
                        if r is not None:
                            cursor.remove(r)
                    if r is None:
                        raise Unsupported("op10094's destination needs a register whose index has "
                                          "bit 1 clear; the pool offers none")
                    pre[v] = r
            if m.form == "read_sr.4":
                for v in m.defs:
                    # THE IMAGEBLOCK COORDINATE IS DEFINED TWICE, once per 16-bit half, and both
                    # halves are the same register by construction. Without this the second write
                    # would claim a second register and the two halves would land apart.
                    if v in pre:
                        continue
                    r = next((x for x in cursor if x <= SR_MAX), None)
                    if self.sr_compact and r is not None:
                        _dead = next((reg for reg, (end, owner) in sorted(sr_owner.items())
                                      if end < at_index[id(m)]), None)
                        if _dead is not None:
                            r = None               # take the reuse branch below instead
                    if r is None:
                        # REUSE A read_sr REGISTER WHOSE VALUE IS DEAD. Pre-colouring takes each
                        # read_sr destination out of the pool for the whole program, so a program
                        # with more than twelve read_sr values refused however short their lives
                        # were: one-dispatch attention fitted two softmax rows and refused four
                        # (Set A item 6, machine model 25.110). Only when no fresh narrow
                        # register is left - so every program that compiled before keeps its
                        # bytes - an earlier read_sr register is taken whose value's last live
                        # index precedes this definition. Only read_sr values ever share one: the
                        # register never returns to the general cursor.
                        here = at_index[id(m)]
                        r = next((reg for reg, (end, owner) in sorted(sr_owner.items())
                                  if end < here), None)
                        if r is None:
                            raise Unsupported("read_sr needs a register at or below r%d; the pool "
                                              "offers none, and no earlier read_sr value is dead "
                                              "here" % SR_MAX)
                        pre[v] = r
                        sr_owner[r] = (last.get(v, here), v)
                        continue
                    pre[v] = r; cursor.remove(r)
                    sr_owner[r] = (last.get(v, at_index[id(m)]), v)
        # THE REQUANTIZATION SCALE MULTIPLY IS TWO-ADDRESS.  The measured op3290/6 template
        # prints its destination and source at the same decoder register; make that tie explicit
        # before the linear scan rather than letting a generic three-address allocation emit bytes
        # the hardware would interpret as a different operation.
        for m in insts:
            if not m.fields.get("requant_fmul"):
                continue
            if len(m.defs) != 1 or len(m.uses) != 1:
                raise Unsupported("requantization scale multiply requires one destination and one source")
            dst, src = m.defs[0], m.uses[0]
            chosen = pre.get(src, pre.get(dst))
            if src in pre and dst in pre and pre[src] != pre[dst]:
                raise Unsupported("requantization scale multiply's destination and source cannot be coalesced")
            if chosen is None:
                chosen = next((r for r in cursor if r <= 127), None)
                if chosen is None:
                    raise Unsupported("no register is available for the requantization scale multiply")
                cursor.remove(chosen)
            pre[src] = chosen
            pre[dst] = chosen
        blocks = [(_i, _m) for _i, _m in enumerate(insts)
                  if _m.fields.get("range_group") and len(_m.uses) > 1]
        for _at_i, m in blocks:
            members = m.uses[:m.fields["range_n"]] if m.fields.get("range_n") else m.uses
            n = len(members)
            # the interval: earliest definition among the group's values, through the store itself
            _start = min([defidx.get(v, 0) for v in members] or [0])
            # THE PRE-COLOURING HAS TO HONOUR THE CAPS TOO. A range store assigns consecutive
            # registers before the linear scan runs, and it was ignoring the field widths the scan
            # respects - so an instruction whose destination is six bits wide got register 96
            # because a store happened to pre-colour it. Nine opcodes failed to compile at all for
            # that reason, and each took nine batchmates with it.
            # IN PLACE ON A TUPLE (handoff 10ae): when every member is the ALU result of the matching lane of one
            # tuple load and that lane dies at the ALU, the members take the lanes' registers (Apple's V0: the adds
            # write r0..r3 back over the loaded tuple and the store reads them). Otherwise a fresh run.
            def _lane_of(v):
                d = insts[defidx[v]] if v in defidx else None
                if d is None or not d.form.startswith("alu.") or len(d.uses) != 1: return None
                src = d.uses[0]
                if src not in pre or sum(1 for x in insts if src in x.uses) != 1: return None
                return pre[src]
            lanes = [_lane_of(v) for v in members]
            if all(r is not None for r in lanes) and lanes == list(range(lanes[0], lanes[0] + n)):
                for v, r in zip(members, lanes): pre[v] = r
                for r in lanes: busy[r] = _at_i
                continue
            # A HALF-PACKED GROUP NEEDS HALF THE REGISTERS, and this is the one general change the
            # packing costs: two values may share a word register when their half indices differ.
            # Everywhere else a register holds one value, and that assumption is why the old
            # layout was the natural one - n values asked for n registers and got them. The form
            # reads HALVES, so n components occupy ceil(n/2) words and component i takes word
            # base + i//2, low half for even i and high for odd. The file bit on each packing move
            # is what distinguishes the two occupants, and selfcheck asserts the placement.
            if m.fields.get("half_pack"):
                words = (n + 1) // 2
                _lim = min([cap.get(v, 1 << 30) for v in members] or [1 << 30])
                runs = [j for j in range(len(cursor) - words + 1)
                        if cursor[j:j+words] == list(range(cursor[j], cursor[j] + words))
                        and cursor[j] + words - 1 <= _lim
                        and all(busy.get(r, -1) < _start for r in cursor[j:j+words])]
                base = (runs[-1] if (tight and not _COMPACT[0]) and runs else (runs[0] if runs else None))
                if base is None:
                    raise Unsupported("no %d consecutive registers free for a %d-component half "
                                      "range store at +%d; the components pack two per word, so "
                                      "this needs %d and not %d"
                                      % (words, n, _at_i, words, n))
                taken = cursor[base:base+words]
                for _i, v in enumerate(members):
                    pre[v] = taken[_i // 2]
                for r in taken: busy[r] = _at_i; cursor.remove(r)
                continue
            _lim = min([cap.get(v, 1 << 30) for v in members] or [1 << 30])
            runs = [j for j in range(len(cursor) - n + 1)
                    if cursor[j:j+n] == list(range(cursor[j], cursor[j] + n))
                    and cursor[j] + n - 1 <= _lim
                    and all(busy.get(r, -1) < _start for r in cursor[j:j+n])]
            base = (runs[-1] if (tight and not _COMPACT[0]) and runs else (runs[0] if runs else None))
            if base is None:
                raise Unsupported("no %d consecutive registers free for a range store at +%d; "
                                  "%d of the pool are held by overlapping range stores"
                                  % (n, _at_i, len(busy)))
            for off, v in enumerate(members): pre[v] = cursor[base] + off
            for r in cursor[base:base+n]: busy[r] = _at_i
            # IN PLACE, AS APPLE'S S1 IS (handoff 10aa): when the group's first value is a block-operand add of a
            # texture fetch that dies there, the fetch takes the SAME register, so the chain is fetch -> rK,
            # rK = rK + block[c], store rK - Apple's r105 throughout. The store sub-form 01 was measured not to
            # read a changed general source (ledger/g17-stage3-full-chain.toml) and is read as naming the fetch's
            # destination; with the add in place, the register the store names IS the fetch's destination, so
            # the program does not depend on which reading is right.
            _d = insts[defidx[m.uses[0]]] if m.uses[0] in defidx else None
            if _d is not None and _d.form == "alu.block" and len(_d.uses) == 1 and _d.uses[0] not in pre:
                _src = _d.uses[0]
                if sum(1 for _x in insts if _src in _x.uses) == 1:
                    pre[_src] = cursor[base]
        # A RANGE STORE'S REGISTERS ARE ONLY BUSY WHILE ITS VALUES ARE LIVE. Every range store used
        # to take its consecutive run out of `cursor` permanently, so two stores that never coexist
        # still got disjoint runs and a program was capped at len(regs)//2 stores whatever else it
        # did - six, with the default pool. full_bitwise (two loads, five bitwise ops, three live
        # values) ran out at its SECOND instruction.
        #
        # The run has to be reserved, because the values must be DEFINED into consecutive registers
        # and that cannot be arranged after the fact - which is why releasing them at the store was
        # tried first and did nothing: the reservation happens before the instruction that fails.
        # What was missing is the interval. A group is busy from the definition of its earliest
        # value to the store that consumes them, and two groups whose intervals do not overlap can
        # have the same run. Each store carries its own companion zero, defined immediately before
        # it, so in practice consecutive stores never overlap and one pair serves all of them.
        # PHI COALESCING. Every member of a phi group - the phi, its entry value and its latch
        # value - is pre-coloured to ONE register, which is what makes the loop body's write
        # visible to the header's read on the next iteration. Groups are unioned first, because
        # two phis can share a value.
        groups, phi_seed = [], {}
        for m in insts:
            g = m.fields.get("phi_group")
            if not g: continue
            g = set(g)
            merged = [x for x in groups if x & g]
            for x in merged: g |= x; groups.remove(x)
            groups.append(g)
        for g in groups:
            members = [v for v in g if isinstance(v, ir.Value)]
            # A group takes a NARROW register if any member needs one. Its register is never
            # released: a loop-carried value is live across the back edge, and the linear scan's
            # "last use" is meaningless for it - the last use in program order is followed by
            # another read on the next iteration.
            pool = self.regs if any(v in narrow_needed for v in members) else self.wide
            avail = [r for r in pool if r not in set(pre.values())]
            if not avail:
                raise Unsupported("no register free for a loop-carried value (%d members)"
                                  % len(members))
            r = avail[0]
            for v in members: pre[v] = r
            phi_seed.update({v: r for v in members})
        # THE TWO-ADDRESS TIE, after phi coalescing so an accumulator loop keeps its loop register.
        # copy_before_tied has already made the accumulator safe to overwrite; here the accumulator
        # and the destination get one register, inside the form's field (Ffma4: r0..r63).
        for m in insts:
            if not m.fields.get("tied") or not m.uses or not m.defs:
                continue
            acc, dst = m.uses[m.fields.get("tie_use", 0)], m.defs[0]
            tmax = m.fields.get("tie_max", TIED_REG_MAX)
            if acc in pre and dst in pre and pre[acc] != pre[dst]:
                raise Unsupported("%s's accumulator and destination are pinned to r%d and r%d; the "
                                  "form has one field for both" % (m.form, pre[acc], pre[dst]))
            chosen = pre.get(acc, pre.get(dst))
            if chosen is None:
                taken = set(pre.values())
                chosen = next((r for r in self.regs if r <= tmax and r not in taken), None)
                if chosen is None:
                    raise Unsupported("no register r0..r%d is free for %s's tied accumulator"
                                      % (tmax, m.form))
            if chosen > tmax:
                raise Unsupported("%s's tied register r%d is outside its six-bit field"
                                  % (m.form, chosen))
            pre[acc] = pre[dst] = chosen
        free = [r for r in self.regs if r not in set(pre.values())]
        free_wide = [r for r in self.wide if r not in set(pre.values())]
        # DESTINATIONS WITH HOLES, by the encoder's own read-back (see _encodable_dests).
        holes = {m.defs[0]: _encodable_dests(m.form) for m in insts
                 if m.form in _HOLED_DEST_FORMS and m.defs}
        # ... and op798's SOURCE fields have holes of their own (a low half of r17 in operand 4, of r18 in operand 6;
        # _encodable_fma16_sources): a value read there avoids them wherever it is defined
        for m in insts:
            if m.form in ("ffma.f16", "acc.ffma.f16"):
                for j, (v, h) in enumerate(zip(m.uses, m.fields["halves"])):
                    ok = _encodable_fma16_sources(j, h)
                    holes[v] = ok if holes.get(v) is None else holes[v] & ok
        # the registers some holed form cannot author: an unconstrained value takes these first when
        # hole-aware (the third attempt in run)
        hole_regs = set()
        if getattr(self, "hole_aware", False):
            for _ok in holes.values():
                hole_regs |= {r for r in list(self.regs) + list(self.wide) if r not in _ok}
        # The phi's own value is defined by no instruction, so it is seeded here; without this
        # the header's first read of it is a use-before-def.
        amap = dict(phi_seed)
        # interval_pre (the interval attempt in run): a pre-coloured register is held over [lo, hi], the earliest
        # definition of its values to their last live index; a value whose own interval misses it may borrow it
        interval = getattr(self, "interval_pre", False)
        pre_span, lent, borrowed = {}, {}, {}
        if interval:
            for _v, _r in pre.items():
                _lo = defidx.get(_v, 0)
                _hi = last.get(_v, _lo)
                a0, b0 = pre_span.get(_r, (_lo, _hi))
                pre_span[_r] = (min(a0, _lo), max(b0, _hi))
        # body_share (the last attempt in run): register r -> the indices of the tensor rows that name it
        share = getattr(self, "body_share", False)
        body_free = list(getattr(self, "_body_pool", ())) if share else []
        body_set = set(body_free)
        rows_of = {}
        if share:
            for j, x in enumerate(insts):
                if x.form == "tensor.wholekernel":
                    for r in _tensor_row_registers(x) & body_set:
                        rows_of.setdefault(r, []).append(j)
        for i, m in enumerate(insts):
            for v in m.uses:
                if v not in amap: raise KeyError("use before def of %r" % v)
                m.fields.setdefault("_uses", []).append(amap[v])
            if m.size == 0:
                continue                       # placeholder: holds no register
            for v in m.defs:
                if v in pre:
                    amap[v] = pre[v]; m.fields.setdefault("_defs", []).append(pre[v]); continue
                # A value that only ever feeds an ALU src1 can live above r15; and whatever pool
                # it comes from, it must fit the narrowest field that will hold it.
                lim = cap.get(v, 1 << 30)
                _ok = holes.get(v)
                fits = (lambda r, _l=lim, _o=_ok: r <= _l and (_o is None or r in _o))
                # a capped value takes the LOWEST register that fits; an uncapped one takes the
                # highest, so it does not consume a register some other operand cannot do without
                order = (lambda xs: xs) if (v in cap or not tight or _COMPACT[0]) else reversed
                if hole_regs and _ok is None:
                    _base = order
                    order = (lambda xs, _b=_base: [r for r in _b(xs) if r in hole_regs] +
                             [r for r in _b(xs) if r not in hole_regs])
                pick = None
                if v not in narrow_needed:
                    pick = next((r for r in order(free_wide) if fits(r)), None)
                    if pick is not None: free_wide.remove(pick)
                if pick is None and share and v not in narrow_needed:
                    # under body_share the shared pool comes BEFORE the narrow registers, which are kept for the
                    # values that can live nowhere else (MM 25.144.8)
                    _end = last.get(v, i)
                    pick = next((r for r in sorted(body_free, key=lambda r: r <= NARROW_MAX) if fits(r) and
                                 not any(i <= j <= _end for j in rows_of.get(r, ()))), None)
                    if pick is not None: body_free.remove(pick)
                if pick is None:
                    pick = next((r for r in order(free) if fits(r)), None)
                    if pick is not None: free.remove(pick)
                if pick is None and interval:
                    # the interval attempt: a body register (narrow ones included) whose rows miss this value's
                    # interval, then a pre-coloured register whose held span misses it and no borrower holds
                    _end = last.get(v, i)
                    _narrow = v in narrow_needed
                    pick = next((r for r in body_free if fits(r) and (not _narrow or r <= NARROW_MAX) and
                                 not any(i <= j <= _end for j in rows_of.get(r, ()))), None)
                    if pick is not None:
                        body_free.remove(pick)
                    else:
                        pick = next((r for r in sorted(pre_span) if fits(r) and (not _narrow or r <= NARROW_MAX)
                                     and r not in lent and (_end < pre_span[r][0] or i > pre_span[r][1])), None)
                        if pick is not None:
                            lent[pick] = _end
                            borrowed[v] = pick
                if pick is None and v in narrow_soft:
                    # the narrow pool is empty and only a float.unary wanted it: the form's fields
                    # reach the wide pool (_narrow_by_float_unary_only)
                    pick = next((r for r in order(free_wide) if fits(r)), None)
                    if pick is not None: free_wide.remove(pick)
                if pick is None and share and v not in narrow_needed:
                    # a tensor body's working register, when no row naming it lies in the value's live
                    # interval [this definition, its last live index] (MM 25.144.8)
                    _end = last.get(v, i)
                    pick = next((r for r in sorted(body_free, key=lambda r: r <= NARROW_MAX) if fits(r) and
                                 not any(i <= j <= _end for j in rows_of.get(r, ()))), None)
                    if pick is not None: body_free.remove(pick)
                if pick is None:
                    # NAME WHAT ACTUALLY RAN OUT. "Out of registers" reads as pressure from live
                    # values, and for the first program to hit this it was not: five stores had
                    # reserved ten of the twelve narrow registers before the second instruction
                    # executed. Every `store` is lowered to a two-wide RANGE store - the value plus
                    # a zero holding the next slot - and a range store pre-colours consecutive
                    # registers that are never released, because `pre` also holds loop-carried
                    # values, which must not be. So a program is capped at len(regs)/2 stores
                    # whatever else it does, and the message has to say so or the next person
                    # reading it goes looking for a spill form they do not need.
                    held = sorted(set(pre.values()) & set(self.regs))
                    nrg = sum(1 for x in insts if x.fields.get("range_group"))
                    unencodable = sorted(r for r in (free if v in narrow_needed else free + free_wide)
                                         if r <= lim and _ok is not None and r not in _ok)
                    if unencodable:
                        raise Unsupported("%s's destination field cannot author r%s (the encoder "
                                          "reads a different register back; _encodable_dests), and "
                                          "no other register fits at %r"
                                          % (m.form, ", r".join(str(105 + r) for r in unencodable), m))
                    raise Unsupported("out of registers at %r; %d narrow + %d wide available%s, no "
                                      "spill form recovered. %d narrow are pre-coloured and held "
                                      "for the whole program: %d range-group store%s reserving "
                                      "consecutive pairs%s"
                                      % (m, len(self.regs), len(self.wide),
                                         "" if lim >= (1 << 30) else " at or below r%d" % lim,
                                         len(held), nrg, "" if nrg == 1 else "s",
                                         ", so this program is capped at %d stores"
                                         % (len(self.regs) // 2) if nrg else ""))
                amap[v] = pick
                m.fields.setdefault("_defs", []).append(amap[v])
            for v in list(amap):
                if last.get(v, -1) <= i and v not in m.defs and v not in pre:
                    r = amap.pop(v)
                    if borrowed.get(v) == r:
                        del borrowed[v]
                        lent.pop(r, None)                          # back to its owner's interval only
                        continue
                    if r in body_set:
                        body_free.append(r); body_free.sort()     # back to the interval-tested pool only
                        continue
                    (free_wide if r > NARROW_MAX else free).append(r)
                    free.sort(); free_wide.sort()
        return insts

    @staticmethod
    def _lifetimes(insts, last):
        """Set the two causal lifetime bits from real liveness.

        Reading a register as an ALU operand does NOT destroy it by definition; two separate bits
        decide, both recovered causally and both authored in both directions:

            byte10[5]  src1       1 releases after the read, 0 keeps  (g17-register-lifetime-resolved)
            byte8[5]   operand B  1 releases after the read, 0 keeps  (g17-operand-b-lifetime)

        NOTE THE INVERTED NAMES. encode_alu's parameter `keep` writes byte10[5] directly, so
        keep=1 RELEASES; its `src2_keep=True` clears byte8[5] and KEEPS. The compiler was passing
        the defaults - keep=0, src2_keep=False - which KEEPS src1 and RELEASES operand B on every
        single ALU instruction. Any program whose operand B is used again was silently wrong: the
        value would be read correctly and then vanish, which is exactly the failure that led to
        byte8[5] being recovered in the first place.

        Apple sets these exactly when the value is dead afterwards, so matching that is both
        correct and what the corpus does.
        """
        for i, m in enumerate(insts):
            if m.form == "alu.12":
                src1 = m.uses[0] if m.uses else None
                srcb = m.uses[1] if len(m.uses) > 1 else None
                if src1 is not None:
                    m.fields["keep"] = 0 if last.get(src1, -1) > i else 1
                if srcb is not None:
                    m.fields["src2_keep"] = last.get(srcb, -1) > i
            elif m.form == "auth":
                # The generic form expresses a source lifetime wherever the field map shows the
                # modifier operand that carries one - the same 32-keeps/16-releases convention the
                # float unary and the float add both use. Sources whose form has no such operand
                # are left alone rather than guessed at.
                #
                # A VALUE NAMED IN TWO SLOTS OF ONE STORE IS KEPT IN BOTH. `O[u] = u` stores
                # u as value and index; u is dead after, so both slots released, and the INDEX
                # slot's release (op17229 operand 6) cleared the register before the value was
                # read - every lane stored 0, at 1 and 8,192 threadgroups, 3 of 3. Clearing ONLY
                # operand 6 stored u; clearing only operand 1 did not
                # (isa/g17-execution-sr-latency-results.json, MM 25.117). Apple's same-register
                # op17229 stores release neither slot. Keeping is safe whatever the slot order.
                # STORES ONLY: an ALU form naming one register twice with both slots released is
                # measured correct (op9700 in four bit-exact tensor bodies, e.g. R40 = f(R40, R34,
                # R40, R34), both R40 slots 16), so only the store, whose index-slot release was
                # measured to clear the value, is changed.
                _dup = (m.fields.get("opcode") in _STORE_OPCODES_DUP_KEEP
                        and not _NO_DUPLICATE_OPERAND_KEEP)
                m.fields["keeps"] = [last.get(u, -1) > i or (_dup and m.uses.count(u) > 1)
                                     for u in m.uses]
                # AND A LIFETIME THE SELECTION ALSO NAMED MUST NOT CONTRADICT THIS PASS.
                #
                # A template's immediate map states the witness's modifiers, lifetimes included, so
                # a form authored from a witness whose value was dead carried a hardcoded RELEASE.
                # When liveness then says keep, put_modifier writes 32 into the bytes and the map
                # still claims 16: the emitted instruction is right and selfcheck reports it as
                # wrong, which is how root's review found it (two widenings of one loaded half).
                # The pass that reads liveness owns the lifetime, so the stated immediate is
                # corrected here rather than the check being relaxed to accept either value - an
                # unchecked lifetime is exactly how a value that is read again gets released.
                imms = m.fields.get("imms")
                if imms:
                    opc = m.fields.get("opcode")
                    srcs = list(m.fields.get("srcmap")
                                or g17auth.register_operands(opc)[1]) if opc else []
                    for j, src_operand in enumerate(srcs[:len(m.fields["keeps"])]):
                        carrier = g17auth.lifetime_operand(opc, src_operand)
                        if carrier is not None and carrier in imms and imms[carrier] in (16, 32):
                            imms[carrier] = 32 if m.fields["keeps"][j] else 16
                            mod = (m.fields.get("mods") or [])[j] if j < len(m.fields.get("mods") or []) else (False, False)
                            imms[carrier] += 2 * bool(mod[0]) + 4 * bool(mod[1])
            elif m.form in ("ffma.f16", "acc.ffma.f16"):
                # op798's three source lifetimes are operands 3, 5 and 7 (32 keeps, 16 releases), from
                # liveness; one register in two slots releasing in both is the measured-correct ALU case
                m.fields["keeps"] = [last.get(u, -1) > i for u in m.uses]
            elif m.form == "float.unary":
                # The float unary form carries its source lifetime as an OPERAND - 32 keeps, 16
                # releases - so it is written from liveness like every other form's, not left at
                # whatever the donor kernel happened to need. g17asm.TRANS_KEEP.
                if m.uses:
                    m.fields["keep_src"] = last.get(m.uses[0], -1) > i
            elif m.form == "simd.broadcast.10":
                # THE BROADCAST RELEASED ITS SOURCE WHATEVER LIVENESS SAID (MM 25.144.6). Its encoder
                # pinned operand 1 only, so operand 3 - the source lifetime (auth.lifetime_operand(14157,
                # 2) == 3) - kept the template's 16. `y = broadcast(x); z = fma(x, x, y)` then read a
                # released x: found by the compile-time release guard on a region-mode fuzz program
                # (seed 37), and on hardware the lanes that read x after the broadcast stored 0.
                if m.uses:
                    m.fields["keep_src"] = last.get(m.uses[0], -1) > i
            elif m.form == "alu.fadd.6":
                # TWO SOURCES FROM LIVENESS. 32 keeps, 16 releases. Source 1 additionally carries
                # a negate bit at b5[3]; nothing in this compiler asks for it yet, so liveness
                # writes the lifetime alone.
                m.fields["src0_life"] = 32 if last.get(m.uses[0], -1) > i else 16
                m.fields["src1_life"] = 32 if last.get(m.uses[1], -1) > i else 16
            elif m.form == "alu.fadd.4":
                # TWO SOURCES, EACH FROM ITS OWN LIVENESS. 32 keeps, 16 releases; a source read
                # again after this instruction must not be freed by it. Writing these rather than
                # inheriting them is the whole reason this form has a dedicated encoder.
                m.fields["src0_life"] = 32 if last.get(m.uses[0], -1) > i else 16
                m.fields["src1_life"] = 32 if last.get(m.uses[1], -1) > i else 16
            elif m.form == "alu.ffma.6":
                # THREE SOURCES, EACH FROM ITS OWN LIVENESS. 32 keeps, 16 releases; a source read
                # again after this instruction must not be freed by it.
                m.fields["src0_life"] = 32 if last.get(m.uses[0], -1) > i else 16
                m.fields["src1_life"] = 32 if last.get(m.uses[1], -1) > i else 16
                m.fields["src2_life"] = 32 if last.get(m.uses[2], -1) > i else 16
            elif m.form == "alu.ffma.4":
                # THE TWO INDEPENDENT SOURCES, EACH FROM ITS OWN LIVENESS. uses[0] is the
                # accumulator, whose lifetime is the constant 16 on all 653 corpus instances and is
                # not written from liveness because it is not a field - see g17ffma4.
                m.fields["src0_life"] = 32 if last.get(m.uses[1], -1) > i else 16
                m.fields["other_life"] = 32 if last.get(m.uses[2], -1) > i else 16
            elif m.form == "alu.fmul.4":
                # BOTH SOURCES, EACH FROM ITS OWN LIVENESS. 32 keeps, 16 releases; a source read
                # again after this instruction must not be freed by it.
                m.fields["src0_life"] = 32 if last.get(m.uses[0], -1) > i else 16
                m.fields["src1_life"] = 32 if last.get(m.uses[1], -1) > i else 16
            elif m.form in ("bitwise.imm", "unary"):
                # The ten-byte bitwise forms release their source at byte8[2] unless told not to,
                # and Apple's templates disagree with each other about it: op423's releases and
                # op13574's does not, because their donor kernels differed.
                if m.uses:
                    m.fields["keep_src"] = last.get(m.uses[0], -1) > i
            elif m.form.startswith(("store.elem1.", "store.half1.")):
                # THE ONE-ELEMENT STORES HAD THE RANGE STORE'S DEFECT AND NOT ITS REPAIR. Every
                # template carries 16 (release) on operand 1, and a store whose value is read again
                # inherited it. MEASURED 2026-09-22 (isa/g17-execution-storereread-results.json): at
                # all six forms, op17235 and op17199 at 8, 10 and 14 bytes, `store v; w = v + 1`
                # gave w = 1 - the released source reads 0. With operand 1 written 0 or 32 at 14
                # bytes it gave v + 1 (isa/g17-execution-storelifetime-results.json). So the source
                # lifetime is written from liveness here too, and a dead value's bytes are unchanged.
                m.fields["keep_members"] = bool(m.uses) and last.get(m.uses[0], -1) > i
            elif m.form in ("store.8", "store.14", "store.halfvec.14"):
                # A STORE RELEASES ITS SOURCE MEMBERS UNLESS TOLD NOT TO, and the compiler never
                # told it. The range forms carry a source lifetime operand (op17244: byte6[4] reads
                # 16 = release, byte2[5] reads 32 = keep, through Apple's decoder on the emitted
                # bytes themselves), and every template this backend holds carries 16 - Apple's
                # value, because Apple stores a value last. A member the program reads AGAIN after
                # the store inherited that release. Integration measured the consequence
                # (results/g17-tensor-reload-runtime-v1): a movimm, a range store of it, then an
                # exact op426 reading the same register returned 0 on all 40 inputs; restoring the
                # register right before the read returned x&1 on all 40; writing the neighbour
                # register did not. So the stored value's STATE after the store is what failed.
                # From here the store's source lifetime is written from liveness like every ALU
                # source's: 32 (keep) when any member is live after the store, the template's 16
                # otherwise - which leaves every store of a dead value byte-identical. MEASURED
                # (integration, results/g17-store-lifetime-runtime-v1, replayed by
                # g17storelifetimeanalysis): with only the 40 stores' modifier changed, keep32 and
                # keep0 both return x&1 on all 40 inputs where release16 returned 0. Scope of that
                # evidence: two-member op17244 stores of movimm values read next by op426, one
                # thread; the three- and four-member forms (same carrier bytes, other opcodes) and
                # loaded members are written the same way and are UNMEASURED (handoff 9f).
                m.fields["keep_members"] = any(last.get(u, -1) > i for u in m.uses)
            elif m.form == "store.ib.32":
                # THE IMAGEBLOCK STORE'S COORDINATE LIFETIME (slot 6) FROM LIVENESS. It was hard-coded
                # 16 (release), and a read after the barrier reuses the coordinate register: MEASURED
                # 2026-09-23 on Apple's own working kernel (ledger/g17-imageblock-store-released-its-
                # coordinate.toml), with only the write's slot 6 changed from 0 to 16 every lane read
                # element 0 - the collapse this compiler's imageblock programs showed. Keep (0) when
                # the coordinate is read again; release (16) otherwise, which leaves a store whose
                # coordinate is dead byte-identical.
                m.fields["keep_coord"] = last.get(m.uses[1], -1) > i
                # THE STORED VALUE HAS THE SAME EXPOSURE through slot 1's low bits (16 = release in
                # every instance this side emits) and no keep value for it is measured, so a value
                # read again after the store is refused rather than guessed.
                if last.get(m.uses[0], -1) > i:
                    raise Unsupported("an imageblock store whose VALUE is read again: slot 1 releases it "
                                      "(16) and no keep encoding for that operand is measured; copy the "
                                      "value through an ALU op before storing it")
            elif m.form.startswith(("load.vec", "store.vec")):
                # the index register's lifetime operand: released (imm:16) when this is its last read, kept (imm:0)
                # otherwise - V0's load keeps r4 for the store, V0's store releases it; V4's first store keeps it
                m.fields["release_index"] = not (last.get(m.uses[-1], -1) > i)
                # THE STORED TUPLE IS RELEASED TOO (MM 25.144.6): op17256's operand 1 is the tuple's lifetime
                # (auth.lifetime_operand(17256, 0) == 1) and every witness carries 16. A tuple member read after
                # the store read 0 on hardware - 0 of 32 lanes right, found by the compile-time release guard on
                # test_g17spillalloc's mixed program. Apple never needs a keep form: it schedules the reads
                # BEFORE the store (a float4 stored then summed compiles as sum-then-store), so no keep value is
                # measured, and a member read again is refused, as the imageblock store's value is.
                if m.form.startswith("store.vec"):
                    n = m.fields.get("range_n") or m.fields.get("n") or 4
                    again = [k for k, u in enumerate(m.uses[:n]) if last.get(u, -1) > i]
                    if again:
                        raise Unsupported("store_vec4_at whose stored value(s) %s are read again after the store: "
                                          "op17256 releases its tuple (operand 1 = 16) and no keep encoding is "
                                          "measured; read them before the store, or store copies (an ALU op) "
                                          "(MM 25.144.6)" % again)
            elif m.form.startswith("alu.") and m.form != "alu.14":
                # THE FORM-BASED OPS NEED THIS TOO, and did not have it. sub, mul and the shifts
                # were added with per-opcode templates and no lifetime handling, so they inherited
                # whatever their template carried - and the mul template releases its source,
                # because it came from a kernel where the multiplicand was dead. The first program
                # here that used a value twice read zero the second time.
                #
                # uses[0] is slot A and uses[1] is slot B, which is the order the forms declare.
                if m.uses:
                    m.fields["keep_a"] = last.get(m.uses[0], -1) > i
                if len(m.uses) > 1:
                    m.fields["keep_b"] = last.get(m.uses[1], -1) > i


# --- phase 3: emission --------------------------------------------------------------------
# THE ONE PLACE A TEMPLATE ENTERS THE EMITTER.
#
# Every encoder in g17asm starts from a template and overwrites the fields it knows, so the bits it
# does NOT overwrite come from whichever Apple instruction the template was cut from. The mission
# calls those inherited bits blockers. Routing every template through one resolver makes them
# measurable - hand it an all-zeros and an all-ones template and the bits that still agree are the
# ones this compiler actually chose - and, once measured, replaceable by a constants table with
# provenance (tools/g17const.py).
TEMPLATE_HOOK = None

# A LOAD OR STORE MUST NOT ADDRESS THREADGROUP MEMORY. The device and threadgroup accesses share a
# family - same length, same encoder, same harvest - so whichever member a registry hands over is
# the one that gets emitted, and a threadgroup store into a device program is silent: it compiles,
# the decoder reads it back, and the buffer the host reads is never written. This backend's IR has
# only device buffers, so the six opcodes the peer isolated as threadgroup are always wrong here.
# g17forms.build() now prefers a device member; this refuses the ones that get through.
def _memory_gate(form, t):
    if not (form or "").startswith(("load", "store")): return t
    o = g17forms._opcode_of(bytes(t))
    if o in g17forms.TG_MEMORY_OPCODES:
        raise Unsupported("form %s resolved to op%d, which addresses THREADGROUP memory; this IR's "
                          "buffers are device memory and the write would go nowhere the host can "
                          "read" % (form, o))
    return t

# THE HALF LOAD. A sixteen-bit element is a different FORM, not a field: op12646 where a 32-bit
# element takes op12682, measured one variable at a time in tools/g17halfprobe.py over eleven
# kernels. This is Apple's own instance from hw-halfboth, `h[400] = h[1u + tp.x]`.
#
# ITS DESTINATION IS A 16-BIT REGISTER. The same destination field decodes as 105+n on op12682 and
# 425+n on op12646, and 425 is the base of the low-half register file - the imageblock prologue's
# two halves are 425 and 281, low and high of one 32-bit register. So the encoder needs no change
# and the ALLOCATOR does: a half value occupies the low half of a whole 32-bit register here.
#
# "Nothing packs two halves into one until something measures that it may" is what this said, and
# something has. The measurement was already in the tree when the sentence was written: op590/4
# selects the register FILE independently on both operands (byte3[0] for the destination, byte1[0]
# for the source) with all four combinations authored and read back. A half range store's
# components are now packed two per word through that field - see the selection path - so the
# condition this refusal named has been met by measurement rather than by relaxing the refusal.
# An ordinary half value still occupies a low half; it is the STORE's members that pack.
#
# `narrow` IS NOT THIS. byte9[2] is the BYTE load and Apple's half load leaves it CLEAR on the same
# opcode; the two were conflated, which is why asking for width='half' used to produce a
# byte-flavoured word load. ledger/g17-the-half-width-is-the-form.toml
# The ONE position this backend still inherits, named rather than left as a number: b5[5] of
# op12646's operand 1. Apple splits 405/235 on it across 640 instances, so it means something, and
# nothing here writes it - it comes from this template.
#
# IT IS NOT THE ELEMENT SIZE, which is what two instances suggested: the pair that differ in b5[5]
# also decode imm:2 against imm:4 at operand 8. Tested causally instead of taken - flipping b5[5]
# alone leaves the element size at imm:2, and a sweep of all 112 bits finds NO single bit that
# changes it while keeping op12646 at fourteen bytes. That agrees with what the half probes already
# established: the width is the FORM, op12646 against op12682, not a field inside one. Operand 8
# has no positions in the operand map at all, which is the same statement from the other side.
LOAD14_HALF = bytes.fromhex("0700030018021080c10080000000")

# THE EIGHT-BYTE DEVICE LOAD. This is Apple's exact op12682/8 witness
# (`0f00030018221040`), retained from the 960-program short-load population
# rather than made by truncating the fourteen-byte form. The short form carries
# the same destination, rank/base, index and low address fields, but no byte9 or
# byte13 field; the selector therefore refuses narrow, shift16 and offsets >=64.
# isa/g17-forms.jsonl and docs/archive/g17-surface-next-frontier.md.
LOAD8_TEMPLATE = bytes.fromhex("0f00030018221040")

# An Apple instance of op12646 at fourteen bytes with byte9 bit2 SET - the narrow-capable load.
# Taken from the corpus rather than constructed, and probed: it retargets cleanly across dest,
# index register and narrow, and passes g17forms.retargetable("load.14").
LOAD14_NARROW = bytes.fromhex("07000300180210a7410480000000")

# The ten-byte half-load family is a distinct measured form, not a truncation of LOAD14_HALF.
# This witness is in isa/g17-forms.jsonl and decodes as op12646/10. Its address field, index
# register, and destination all retarget through encode_load; narrow and hi16 remain refused here.
LOAD10_HALF = bytes.fromhex("17000302180210808000")



def _decode_tokens(raw, expr=False):
    """Apple's decoder's operand tokens for one instruction, or [] if it will not decode.

    An independent reader for the selfcheck: it shares nothing with the maps that authored the
    bytes, so agreement between them is evidence rather than a tautology.

    WITHOUT `expr` AN ADDRESS OPERAND PRINTS AS A NUMBER - the decoder resolves the expression
    against a heap pointer, so `[op4+0*4]` comes out as something like `expr:49710932000` and two
    different destinations can print the same. Pass expr=True to get `expr:bin(op4,const(0),4)`,
    which is the reading a destination has to be checked against. The default is unchanged because
    the callers that predate this compare whole token vectors against recorded ones.
    """
    from agxforge.g17 import ref as g17ref
    import subprocess, tempfile
    try:
        with tempfile.NamedTemporaryFile(suffix=".bin") as fh:
            fh.write(raw); fh.flush()
            r = subprocess.run([g17ref.binary(), fh.name, "0", str(len(raw)), "--pc", "0"]
                               + (["--expr"] if expr else []),
                               capture_output=True, text=True)
        parts = r.stdout.strip().split()
        return parts[3:] if len(parts) > 3 and parts[1] != "bad" else []
    except Exception:
        return []


def _modal_operand(opcode, length, idx):
    """Apple's most common value at one operand, from the operand map that declared the slot.

    Used where a form declares a slot this compiler has no role for. Taking the value from the SAME
    record that says the slot exists is what lets the emit path survive a re-fit that changes the
    operand count - the alternative is a positional signature that silently becomes wrong.
    """
    for kind in ("imm", "reg"):
        r = g17as.maps().get((opcode, length, idx, kind))
        if r is not None and r.get("modal") is not None:
            return r["modal"]
    raise Unsupported("op%d operand %d at %d bytes has no modal value to fall back on"
                      % (opcode, idx, length))


# FORMS WHOSE DESTINATION FIELD HAS HOLES. op1004 (fadd.imm.l12, the half-to-float widening) cannot
# author destination r105+d for d in {14, 15, 30, 31, 46, 47} or d >= 62: the assembler's read-back
# returns r105+d+2 ("asked for 119, reads back as 121") or rejects a forced bit. Found by
# tools/g17ccfuzz.py (seed 6: a compile crash, never a wrong byte - the read-back refused it). Which
# indices author is the ENCODER'S answer, swept once, not a pattern typed in here; the allocator
# then gives such a destination only a register that exists, as it does for op10094's.
_HOLED_DEST_FORMS = {"cvt.f16.f32": lambda d: _as_line(
    "fadd.imm.l12", 1004, 12, {0: "r%d" % (d + 105), 2: "r425"}, pinned={1: 2147483648, 3: MOV_KEEP}),
    # op798's LOW-HALF destination cannot author r18, r19, r26, r27, r50, r51, r58 or r59 (the encoder reads another
    # register back); its high half and its other fields have fewer holes (_encodable_fma16_sources)
    "ffma.f16": lambda d: _as_line("ffma.f16.l12", 798, 12, {0: "r%d" % (d + 425), 2: "r426", 4: "r427", 6: "r428"},
                                   pinned={1: 2147483648, 3: MOV_RELEASE, 5: MOV_RELEASE, 7: MOV_RELEASE})}
_ENCODABLE_DESTS = {}
_ENCODABLE_FMA16_SOURCES = {}


def _encodable_fma16_sources(j, half, top=128):
    """frozenset of registers op798's source j (operand 2, 4 or 6) authors and reads back as that half."""
    if (j, half) not in _ENCODABLE_FMA16_SOURCES:
        ok = set()
        for r in range(top):
            roles = {0: "r426", 2: "r427", 4: "r428", 6: "r429"}
            roles[2 + 2 * j] = "r%d" % ((281 if half == "hi" else 425) + r)
            try:
                g17as.assemble(_as_line("ffma.f16.l12", 798, 12, roles,
                                        pinned={1: 2147483648, 3: MOV_RELEASE, 5: MOV_RELEASE, 7: MOV_RELEASE}))
            except Exception:
                continue
            ok.add(r)
        _ENCODABLE_FMA16_SOURCES[(j, half)] = frozenset(ok)
    return _ENCODABLE_FMA16_SOURCES[(j, half)]


def _encodable_dests(form, top=128):
    """frozenset of destination indices the form's encoder authors and reads back exactly."""
    if form not in _ENCODABLE_DESTS:
        ok = set()
        for d in range(top):
            try:
                g17as.assemble(_HOLED_DEST_FORMS[form](d))
            except Exception:
                continue
            ok.add(d)
        _ENCODABLE_DESTS[form] = frozenset(ok)
    return _ENCODABLE_DESTS[form]


def _as_line(mnem, opcode, size, roles, pinned=None, controls=""):
    """One g17as source line naming EVERY slot the form declares, by index.

    A positional signature is a bet that the operand map will keep saying the same number of
    operands, and a re-baseline is exactly the event that changes it - two forms broke authoring
    that way in one afternoon, each with a message ("takes 6 operands, got 6") that says nothing
    about which operand moved. Roles the compiler owns come from selection, values it has measured
    are pinned, and any other slot takes Apple's MODAL value from the same record that declares the
    slot exists.
    """
    pinned = dict(pinned or {})
    named = {}
    for i, _kind in g17as.forms_table()[mnem]["slots"]:
        if i in roles:
            named[i] = roles[i]
        elif i in pinned:
            named[i] = "#%d" % pinned[i]
        else:
            named[i] = "#%d" % _modal_operand(opcode, size, i)
    for i, v in pinned.items():
        named.setdefault(i, "#%d" % v)
    return "%s %s%s" % (mnem, ", ".join("op%d=%s" % (i, named[i]) for i in sorted(named)), controls)

def _tmpl(default, form=None, opcode=None):
    t = default if TEMPLATE_HOOK is None else TEMPLATE_HOOK(form, opcode, len(default), default)
    return _memory_gate(form, t)


def _const_forms():
    """The form registry whose templates have PROVENANCE, plus its hook.

    g17const carries, per (form, opcode, length), the constant this compiler chose and where the
    value came from; g17forms harvests the modal member of a family out of Apple's kernels. Both
    are templates, but only the first is the one every executed probe in this project actually
    dispatched, and the difference is not cosmetic: on endtoend's own kernel the harvested registry
    produces a program whose store leaves the device buffer at its fill value (G17_DEFAULT_FORMS=1
    reproduces it) while this one returns the right answer. So this is the default now, and
    g17forms.build() is what a caller asks for deliberately.
    """
    from agxforge.g17 import const as g17const
    T = g17const.load(); out = {}
    for (form, opc, ln), e in T.items():
        if opc is None: out.setdefault(form, dict(name=form, length=ln, template=e["value"]))
    return out, g17const.hook_from(T)


# THE LIFETIME FIELDS, BY NAME. Alloc._lifetimes writes most of them; the three selection-time
# ones (the conversions' keep_src and the half store's keep_value/keep_index) come from
# _has_later_reader and are checked against the same recomputed liveness below.
_LIFETIME_FIELDS = ("keep", "src2_keep", "keeps", "keep_src", "src0_life", "src1_life",
                    "src2_life", "other_life", "keep_a", "keep_b", "keep_members",
                    "release_index")
_SELECTION_LIFETIMES = {"cvt.f32.f16": (("keep_src", 0),), "cvt.f16.f32": (("keep_src", 0),),
                        "store.half.14": (("keep_value", 0), ("keep_index", 1)),
                        "store.byte.14": (("keep_value", 0), ("keep_index", 1))}


def _phi_entries_live_to_header(insts, last):
    """A LOOP PHI'S ENTRY VALUE IS READ ON THE EDGE INTO THE HEADER - after the instruction that falls into it.

    _cfg_uses charges the entry value to insts[h-1] so liveness keeps it to the loop (f453622af). But when
    insts[h-1] ITSELF reads that value, it was the last reader by index and released it, and the phi - which
    took the entry value's register - read a released register on the first trip. Found by the compile-time
    release guard on a loop-mode fuzz program (MM 25.144.6; minimal: `c = 0; x = c + 1; loop i = phi(c, i + 1)`,
    the add releasing c); 29 of 150 loop-mode programs had it, all with a phi starting at constant 0, which a
    released register also reads - right by luck, as the f453622af note predicted. The entry value's last
    point is therefore the header LABEL (h), not h - 1: no instruction before the loop releases it, and the live
    sets - so coalescing and interference - are unchanged."""
    n = len(insts)
    for h, m in enumerate(insts):
        if m.form != "label" or h == 0 or h + 1 >= n or insts[h + 1].form != "phi":
            continue
        for j in range(h + 1, n):
            if insts[j].form != "phi":
                break
            g = insts[j].fields.get("phi_group", ())
            if len(g) > 1 and isinstance(g[1], ir.Value) and last.get(g[1], -1) < h:
                last[g[1]] = h
    return last


def _final_liveness(insts):
    """The last index at which each value is live, over the FINAL instruction list.

    The same derivation Alloc uses - a fixed point over the real control-flow graph, not a linear
    scan - because with a back edge the last MENTION is not the last read.
    """
    live_in, live_out = Alloc._cfg_live(insts)
    last = {}
    for i in range(len(insts)):
        for v in live_in[i] | live_out[i]:
            if last.get(v, -1) < i:
                last[v] = i
    return _phi_entries_live_to_header(insts, last)


def _stale_lifetimes(insts):
    """Lifetime fields that disagree with liveness over the final list.

    Re-runs the very code that wrote them rather than re-deriving each field's polarity, because
    the polarities genuinely differ between forms - `keep=1` RELEASES on alu.12 while
    `keep_src=True` KEEPS on the conversions - and a second model of that would be a second place
    to get it wrong.
    """
    before = [{k: m.fields.get(k) for k in _LIFETIME_FIELDS if k in m.fields} for m in insts]
    last = _final_liveness(insts)
    Alloc._lifetimes(insts, last)
    out = []
    for i, m in enumerate(insts):
        for k in sorted(set(before[i]) | {k for k in _LIFETIME_FIELDS if k in m.fields}):
            if before[i].get(k) != m.fields.get(k):
                out.append((i, m.form, k, before[i].get(k), m.fields.get(k)))
        for field, use_at in _SELECTION_LIFETIMES.get(m.form, ()):
            if field not in m.fields or len(m.uses) <= use_at:
                continue
            want = last.get(m.uses[use_at], -1) > i
            if bool(m.fields[field]) != want:
                out.append((i, m.form, field, m.fields[field], want))
    return out


def _promote_wide_bitwise(insts):
    """A reg-reg bitwise whose allocated registers exceed the FOUR-byte fields takes the TEN-byte
    form. Runs after allocation, because the length depends on the registers and nothing before
    allocation knows them; before emit, because `_emit` computes every branch offset from `m.size`.

    THE FOUR-BYTE FORM IS UNTOUCHED WHERE IT FITS. Six bits in each of its three operands, so a
    program whose bitwise stays at or below r63 emits exactly the bytes it did before - which is
    what keeps this from being a byte movement on every program that contains one.
    """
    for m in insts:
        if m.form != "bitwise.reg":
            continue
        regs = [m.fields.get("_defs", [])[i] for i in range(len(m.fields.get("_defs", [])))] + \
               [m.fields.get("_uses", [])[i] for i in range(len(m.fields.get("_uses", [])))]
        if not regs:
            continue
        four = (1 << len(g17asm.BW_R_DEST)) - 1
        if max(regs) <= four:
            continue
        if m.fields["opcode"] not in g17asm.BITWISE_REG10_TEMPLATE:
            raise Unsupported("op%d needs the ten-byte bitwise at r%d and no template for it is "
                              "recovered" % (m.fields["opcode"], max(regs)))
        m.form = "bitwise.reg.10"
        m.size = 10



def _keep_element_store_source(b, opcode):
    """Write a one-element store's source lifetime (operand 1) as KEEP, the range stores' way.

    ONLY THE LIFETIME CARRIERS MOVE. Operand 1 also carries the load wait - the 10-byte forms read
    0x80000010, wait + release - and the assembler's per-width maps do not even hold operand 1's
    positions at op17235/14 or op17199/10, so a first draft that wrote the whole operand through them
    dropped the wait at 10 bytes and could not reach two forms at all. g17auth.put_modifier writes
    exactly the keep/release carriers, certified on THIS template through Apple's decoder by
    certify_on (both polarities authored and decoded, the opcode and every register operand
    unchanged). An uncertified or unwritable lifetime refuses rather than releasing a live value."""
    try:
        choice = g17auth.certify_on(opcode, b, 0)
    except KeyError as ex:
        raise Unsupported("op%d stores a value this program reads again, and its source lifetime is "
                          "not certified on this template (%s); run tools/g17authtables.py to certify "
                          "it" % (opcode, str(ex).split(",")[0][:80]))
    if choice is None:
        raise Unsupported("op%d/%d's source lifetime does not survive Apple's decoder on this template, "
                          "so a value it stores cannot be read again" % (opcode, len(b)))
    got = g17auth.put_modifier(opcode, b, 0, True, choice=choice)
    if not got:
        raise Unsupported("op%d/%d: the source lifetime could not be written" % (opcode, len(b)))
    return bytes(got)[:len(b)]


def emit(insts, forms=None):
    """MInst -> bytes, from the canonical per-form template."""
    global TEMPLATE_HOOK
    if forms is None and TEMPLATE_HOOK is None:
        forms, hook = _const_forms()
        TEMPLATE_HOOK = hook
        try: return _emit(insts, forms)
        finally: TEMPLATE_HOOK = None
    return _emit(insts, forms)


# THE IMAGEBLOCK ACCESS'S OPERAND 1. For the store, bits 24-31 are its WAIT MASK (bit 24+s waits on
# slot s; Piece B, 2026-09-23, 32/32 against 0/32 for no wait and for a wrong slot): 0x01000010 waits on
# slot 0, where the coordinate's two read_sr fill. A loaded stored value is made ready by a waiting
# copy at selection (above), so the store itself never needs another slot.
# CORRECTION (2026-09-23): a per-program IB_STORE_OP1_SHARED = 0x80000010 stood here, read as "bit 31
# selects shared storage" from item 12's rounds 8-9. It was a wait on slot 7 - where Apple's kernel
# (and ours) loaded the stored value - so it only happened to cover that one producer. Removed.
IB_STORE_OP1 = 16777232
IB_LOAD_OP1 = 25165824


def _ib_op1(raw):
    """Operand 1 of an emitted imageblock access, as Apple's decoder reads it."""
    from agxforge.g17 import model as _model
    return [v for _k, v in next(iter(_model.decode(bytes(raw), 0))).values][1]


def _emit(insts, forms=None):
    """MInst -> bytes, from the canonical per-form template.

    Two passes, because a forward branch's displacement is not known until the join has been
    placed. Pass 1 fixes every offset (all forms have a static size); pass 2 encodes, resolving
    labels against those offsets. target = branch_offset + displacement, and the target is the
    JOIN INSTRUCTION itself, not the instruction after it - the ISA entry for exec.restore is
    explicit that reconvergence is a place the branch lands on.
    """
    _check_physical_registers(insts, "allocated compiler stream")
    forms = forms or g17forms.build()
    off, offsets, labels = 0, [], {}
    for m in insts:
        offsets.append(off)
        if m.form in ("exec.restore", "label"): labels[m.fields["label"]] = off
        off += m.size
    code = bytearray(); layout = []; tensor_blobs = []
    for m, at in zip(insts, offsets):
        if m.size == 0: continue
        # A measured class may carry a complete instruction byte string whose unexposed fields are
        # intentionally inherited from one Apple witness.  The row still has an ordinary form,
        # opcode, defs and uses for ABI/liveness accounting; this override only prevents a generic
        # encoder from rewriting the measured six/ten/fourteen-byte residue.
        if m.fields.get("raw") is not None:
            if not m.fields.get("requant_stage"):
                raise Unsupported("raw instruction bytes are reserved for the measured requantization stage")
            b = bytes(m.fields["raw"])
        elif m.form == "end":            b = END
        elif m.form == "barrier" and m.fields["scope"] == "fence_device":
            # THE DEVICE-SCOPE FENCE (MM 25.140.5): Apple's atomic_thread_fence(mem_device, seq_cst,
            # thread_scope_device) is op14156 (0, 186), six bytes and no registers - NOT op447, the
            # threadgroup_barrier(mem_device) cc emits for scope "device", which leaves another core's stores
            # invisible to this core's loads (the last-threadgroup probe summed stale markers). Apple's bytes.
            b = FENCE_DEVICE_BYTES
        elif m.form == "barrier":
            b = g17asm.encode_barrier(m.fields["scope"], BARRIER)
        elif m.form == "exec.restore": b = g17cf.encode_exec("pop")
        elif m.form == "cmp.6":
            # THE COMPARE IS ONE INSTRUCTION. It was modelled as two - a 2-byte operand word and
            # a 4-byte relation word - and the bytes were right, but the BOUNDARY was not: Apple's
            # decoder reads all six as one instruction, opcode 10369. Two encoders still author
            # the two halves; what changed is that the compiler no longer claims a boundary in the
            # middle of an instruction, which is the same class of error as the 12-byte mul.
            if m.fields['_uses'][0] >= 64:
                raise Unsupported('cmp.6 source allocation exceeds its six-bit register field; modifier32 is not register bit6')
            b = (g17asm.encode_cmp_src(m.fields["_uses"][0],source_modifier=m.fields.get('source_modifier',0))
                 + g17asm.encode_cmp_imm(m.fields["imm"], m.fields["rel"], CMP_IMM,
                                         keep=m.fields["keep"]))
            # THE FLAG IS AUTHORED NOW, not left at whatever the template carried. The selector is
            # a compact 3-bit index (g17asm.FLAG_BITS, from the ISA agent); until it was located,
            # every compare here went to FLAG0 and _check_flag_discipline had to refuse any program
            # that did not consume it immediately.
            b = g17asm.encode_flag(10369, b, _flag("CMP"))
        elif m.form == "exec.mask":
            # CONSTRUCTED, NOT COPIED. These were emitted as the literals EXEC_MASK 1e00000e and
            # EXEC_JOIN 3e03400e, and the program declared the word "emitted verbatim" - but
            # g17cf.encode_exec builds both from its field map and reproduces them exactly:
            # encode_exec("if") is 1e00000e and encode_exec("pop") is 3e03400e. The base bytes are
            # the ones mutation proved forced and every other bit comes from kind, count, aux and
            # the predicate. Emitting them through the encoder puts the derivation in the path
            # rather than in a sibling module, so 256 bits stop being inherited by assertion.
            b = g17asm.encode_flag(582, g17cf.encode_exec("if"), _flag("EXEC"))
        elif m.form == "loop.flag":
            b = g17cf.encode_exec("while", count=2, invert=True)
            if b != LOOP_FLAG:
                raise Unsupported("the constructed loop exec %s is not op579's %s"
                                  % (b.hex(), LOOP_FLAG.hex()))

        elif m.form == "tensor.seq":
            seq = _tensor_registry()[m.fields["shape"]]
            # Units are emitted in their ORIGINAL PROGRAM ORDER. Their original INTERLEAVING with
            # scalar setup is not reproduced, and that is an inherited structural fact, not a
            # choice - declared in G17Program.inherited rather than silently assumed harmless.
            units = sorted([(o, by) for o, by, *_ in seq.bounds] + list(seq.macs))
            chunk = bytearray()
            for _, by in units:
                if agxdis.is_mac(by):
                    u = bytearray(by)
                    b6, b7, b3 = agxdis.encode_mac(u, m.fields["a_dtype"], m.fields["b_dtype"],
                                                   enable=1)
                    u[6], u[7], u[3] = b6, b7, b3
                    chunk += bytes(u)
                else:
                    chunk += by
            b = bytes(chunk)
        elif m.form in ("branch.cond.fwd", "branch.cond.back"):
            tgt = labels.get(m.fields["target"])
            if tgt is None: raise Unsupported("branch to unplaced label %r" % m.fields["target"])
            # THE TWO FORMS USE DIFFERENT PC BASES. Over Apple's 723 branches, a forward target
            # lands on an instruction boundary 97.9% of the time when computed from the END of
            # the branch (base+4) and 67.5% from its start; a backward target lands 88.1% of the
            # time from the START (base+0) and 29.7% from the end. All 622 forward displacements
            # are positive and all 101 backward ones negative, so the two forms are cleanly
            # separated. ledger/g17-branch-pc-base-differs-by-form.toml
            # REVERTED to base+0 for BOTH forms on 2026-09-04. The base+4 change came from a
            # BOUNDARY test - 97.9% of forward targets land on some instruction at base+4 against
            # 67.5% at base+0 - and a boundary hit is not a semantic hit. The join is a four-byte
            # exec.restore, so base+4 lands JUST PAST it, on the next instruction, which is a
            # boundary every time. The semantic test asks whether the target IS the join:
            #     probe corpus  base+0 -> 434 land on exec.restore, base+4 -> 0
            #     apple corpus  base+0 -> 156,                      base+4 -> 47
            # and a compiler-emitted conditional confirms it directly: a branch at +0x060 with
            # disp +220 reaches the exec.restore at +0x13c only at base+0.
            # ledger/g17-pc-base-reverted-boundary-is-not-semantic.toml
            back = m.form == "branch.cond.back"
            base = 0
            # G17_FWD_BASE overrides the forward base, so a deliberately wrong base can be
            # dispatched as a positive control on the right one.
            if not back and os.environ.get("G17_FWD_BASE") is not None:
                base = int(os.environ["G17_FWD_BASE"])
            d = tgt - at - base
            if back and d >= 0:
                raise Unsupported("back edge with a non-negative displacement %+d" % d)
            if not back and d <= 0:
                raise Unsupported("forward branch with displacement %+d" % d)
            m.fields["_disp"] = d
            # THE WHOLE TEN BYTES, not four plus a literal tail. The displacement field spans
            # bytes 0-9 (g17cf.DISP_BITS, 47 bits with the last as sign), and this used to write
            # only the four-byte head through encode_branch - a TWELVE-bit encoder - and append
            # the tail as a constant. For a small displacement the sign extension makes that tail
            # correct by accident; for anything past +-2046 it is simply wrong, and 16 of Apple's
            # 581 back edges are outside that range.
            #
            # encode_branch10 writes every displacement bit into a ten-byte template. Measured
            # against Apple: from ONE template it constructs 1,763 of 1,804 forward branches and
            # 575 of 581 backward ones exactly, and every single miss differs only in AUX - zero
            # displacement disagreements in 2,385 instances.
            b = g17asm.encode_branch10(
                (BRANCH_BACK + BRANCH_TAIL_BACK) if back else (BRANCH_FWD + BRANCH_TAIL_FWD), d)
        elif m.form == "alu.sat":
            f = dict(m.fields)
            defs = f.pop("_defs", []); uses = f.pop("_uses", [])
            slot_b = ("imm", f["imm"]) if "imm" in f else ("reg", uses[1], 0)
            b = g17asm.encode_alu_form(f["opcode"], defs[0], ("reg", uses[0], 0), slot_b,
                                       _tmpl(SAT_TEMPLATE[f["opcode"]], m.form, f["opcode"]),
                                       keep_a=f.get("keep_a"), keep_b=f.get("keep_b"),
                                       hazard=f.get("hazard", 0))
        elif m.form == "auth":
            f = dict(m.fields)
            defs = f.pop("_defs", []); uses = f.pop("_uses", [])
            opc = f["opcode"]
            dsts, srcs = g17auth.register_operands(opc)
            # WHICH OPERAND DOES EACH USE GO TO. Positionally, by default - but the table's idea of
            # which operands are registers CHANGES as witnesses are repaired, and when op17229's
            # address operand became a register operand the indexed store started writing its index
            # into the buffer address. A caller that knows the roles states them.
            srcs = list(f.get("srcmap") or srcs)
            # A TUPLE DESTINATION HAS ONE ENCODED REGISTER AND N-1 PLACED COMPANIONS. op10793/12
            # writes a GPR32tup2: the field names the pair once and the allocator reserves the
            # consecutive run, so defs beyond the encoded destinations are PLACED, not encoded.
            # They are still checked for adjacency, because a companion that is not the next
            # register means the instruction writes one nobody reserved - and that would be a
            # silent clobber rather than a refusal.
            if f.get("tuple_group") and len(defs) > len(dsts):
                placed = list(defs)          # already allocated register numbers at this point
                if placed != list(range(placed[0], placed[0] + len(placed))):
                    raise Unsupported(
                        "op%d writes a %d-register tuple and the allocator placed %s, which is not "
                        "a consecutive run. The second half would land in a register nothing "
                        "reserved" % (opc, len(defs), placed))
                defs = defs[:len(dsts)]
            if len(defs) > len(dsts) or len(uses) > len(srcs):
                raise Unsupported("op%d takes %d register destinations and %d sources; the IR "
                                  "gave %d and %d" % (opc, len(dsts), len(srcs), len(defs), len(uses)))
            # AN ENCODER HOOK, for an encoding the authoring table does not describe. The same
            # opcode can have several encodings with DIFFERENT field layouts - op999's six-byte form
            # puts its operands nowhere near where its twelve-byte form does - so a caller that has
            # measured one layout can supply it rather than being refused. The hook receives the
            # template and the allocated registers and returns the bytes.
            # AN UNFILLED SOURCE IS A SILENT WRONG REGISTER. If the IR supplies fewer values than
            # the opcode has source operands, every operand it does not name must be set some other
            # way - an immediate the caller states, or an operand the caller has deliberately left
            # to the witness by naming it in srcmap. Anything else inherits a register number from
            # whatever instruction the witness came from.
            if f.get("encoder") is None and len(uses) < len(srcs):
                loose = [i for i in srcs[len(uses):] if i not in (f.get("imms") or {})]
                if loose:
                    raise Unsupported("op%d has source operands %s that the IR does not fill and "
                                      "no immediate names; they would keep the witness's registers"
                                      % (opc, loose))
            if opc == 11452 and m.size == 10 and f.get("encoder") is None:
                from agxforge.g17 import predicateform as g17predicateform
                b = g17predicateform.encode(defs, uses, f.get("imms") or {}, f.get("keeps"))
            elif f.get("and16") is not None:
                b = _and16_encode(opc, f.get("template"), f, defs, uses)
            elif f.get("encoder") is not None:
                b = bytes(f["encoder"](f.get("template"), defs, uses))
            else:
                vals = {}
                for i, r in list(zip(dsts, defs)) + list(zip(srcs, uses)):
                    # WHAT VALUE NAMES REGISTER r IS A PER-OPERAND FACT, measured through Apple's
                    # decoder: a slot operand takes 2*r, a register-numbered one takes r, and the
                    # same opcode can have both (op1934's ffma has one of each). Deriving it from
                    # the domain alone named register 2r for every operand of the second kind.
                    vals[i] = g17auth.field_value(opc, i, r)
                vals.update(f.get("imms") or {})
                # A TEMPLATE OVERRIDE authors from a chosen witness rather than the table's. The
                # table's witness comes from a mutation walk as often as from Apple's code, and a
                # walked one can sit in a corner of the encoding no real instruction occupies - op612's
                # is such a case, rare twice over in its own corpus distribution.
                # values the IR states explicitly are trusted; values the allocator derived are not
                b = g17auth.encode(opc, vals,
                                   template=f.get("template") or CLEARED_WITNESS.get(opc),
                                   trusted=tuple(f.get("imms") or ()))
                if f.get("load_wait"):
                    # the measured bit, applied after encode so it survives the operand writes
                    b = bytes([b[0] | 0x08]) + b[1:]
                if f.get("buf_const") is not None and f["buf_const"] > 28:
                    # THE MEASURED RANGE IS NOW RANKS 0..15, AND ONLY ON op17229/8.
                    #
                    # The corpus reason for stopping at 7 was silence, not a field limit: across
                    # 15,138 of Apple's sections with a store the const takes 0, 4, ... 28, and its
                    # only higher value is 84, in nine ray-tracing `intersect` kernels binding
                    # buffers, textures, a sampler and acceleration structures - where 84 is not
                    # 4 x any buffer rank, so the rank reading does not describe them at all.
                    # Silence about ranks 8..15 is a question Apple's compiler answers directly,
                    # so it was asked: sixteen bindings declared and LIVE, one store, and the
                    # emitted byte1[6:2] equals the binding rank at 0, 1, 7, 8, 12 and 15.
                    # Scrambling the store order leaves the consts following the RANKS rather than
                    # the store positions, which is what rules out an allocation counter.
                    # results/g17-buffer-rank-assessment-v1/ retains each arm's source, text
                    # section and full disassembly.
                    #
                    # TWO THINGS STAY REFUSED, because neither was measured. Above rank 15 the
                    # five-bit carrier could still hold the value, but nothing has been seen there.
                    # And this allowance is for THIS FORM: the fourteen-byte half store spells its
                    # binding as `[op0+N*8]` through the assembler rather than this byte patch, and
                    # its behaviour above rank 7 is not part of the measurement, so any other form
                    # arriving here with a wide const refuses by name rather than inheriting a
                    # result measured elsewhere.
                    if _NO_WIDE_BUFFER_RANK:
                        raise Unsupported(
                            "buffer rank %d is past the range the address const was measured on "
                            "(0..7): the wide-rank capability is switched off here, which is the "
                            "state the gain is measured against" % (f["buf_const"] // 4))
                    # THE MEASUREMENT WAS DEVICE-BUFFER-ONLY, AND THE GATE MUST SAY SO.
                    #
                    # Root found this: checking the opcode and length is not checking the DOMAIN.
                    # Every arm of the rank measurement bound device buffers and nothing else - no
                    # texture, no sampler, no internal binding records - which is exactly the
                    # configuration Apple's const-84 ray-tracing sections are NOT, and the reason
                    # those sections tell us nothing about ranks. A kernel that reads a texture
                    # carries internal records at indices 44 and 48 ahead of its user buffers, so
                    # its ranks are shifted by a base this measurement never varied, and admitting
                    # a wide const there would be extrapolating across the one axis the whole
                    # assessment held fixed. I documented the device-only scope and then did not
                    # enforce it, which is the same shape of error as stating a scope in prose.
                    if _RANK_BASE[0]:
                        raise Unsupported(
                            "buffer rank %d in a kernel that also binds a texture: ranks 8..15 "
                            "were measured on device buffers ONLY, with no internal binding "
                            "records ahead of the user buffers, and this kernel's ranks are "
                            "shifted by %d internal records - a domain the measurement never "
                            "varied" % (f["buf_const"] // 4, _RANK_BASE[0]))
                    if f["buf_const"] > 60:
                        raise Unsupported(
                            "buffer rank %d is past the range the address const was measured on "
                            "(0..15); the five-bit carrier could hold it, but no instance has been "
                            "seen and an unvalidated const is how a law becomes a silent wrong "
                            "answer" % (f["buf_const"] // 4))
                    if not (opc == STOREI_OPCODE and m.size == 8):
                        raise Unsupported(
                            "buffer rank %d on op%d/%d: ranks 8..15 are measured on the op%d/8 "
                            "store only, and this form's const has never been read above rank 7, "
                            "so it refuses rather than borrowing another form's measurement"
                            % (f["buf_const"] // 4, opc, m.size, STOREI_OPCODE))
                if f.get("buf_const") is not None:
                    # THE BUFFER SELECTOR, applied after encode for the same reason. op17229's
                    # address expression is not resolved to a register by Apple's decoder, so
                    # which buffer a store writes used to come from the witness - fixing every
                    # store this compiler emits at const 4. That is correct only for a kernel
                    # whose target is the second of two buffers, which is every kernel in this
                    # corpus, which is why twenty of them passed with it wrong.
                    #
                    # The carrier is byte1[6:2] and it is worth 4 a unit, measured by flipping
                    # each bit of an emitted store and reading the address expression back:
                    # bit2 -> 4, bit3 -> 8, bit4 -> 16, bit5 -> 32, bit6 -> 64.
                    q = f["buf_const"] // 4
                    if f["buf_const"] % 4 or not 0 <= q < 32:
                        raise Unsupported("buffer const %d is not 4 x a rank the field can hold"
                                          % f["buf_const"])
                    b = bytes(b[:1]) + bytes([(b[1] & ~0x7C) | (q << 2)]) + bytes(b[2:])
                certified = g17auth.lifetime_certified(opc)
                for i, keep in zip(srcs, f.get("keeps") or []):
                    _m = (f.get("mods") or [])
                    _k = srcs.index(i)
                    if _k < len(_m) and any(_m[_k]) and (
                            g17auth.lifetime_operand(opc, i) is None
                            or (not certified and not f.get("template"))):
                        raise Unsupported("op%d operand %d: a folded negate/abs needs the source's "
                                          "modifier word, and this form's is not certified" % (opc, i))
                    if g17auth.lifetime_operand(opc, i) is None:
                        if keep:
                            raise Unsupported("op%d has no source lifetime for operand %d, so a value "
                                              "it reads cannot be used again" % (opc, i))
                        continue
                    choice = g17auth.lifetime_choice(opc, i) if certified else None
                    if not certified:
                        # The table's verdict is about the TABLE's witness. When the caller supplies a
                        # template - Apple's own eight-byte indexed store, say, where the table's is
                        # fourteen - certify against that instead of refusing on a verdict about a
                        # different encoding.
                        choice = g17auth.certify_on(opc, b, i) if f.get("template") else None
                        if choice is None:
                            # Only a KEEP has to be written. A value read once is released or kept by
                            # the witness's own bits and either is harmless.
                            if keep:
                                raise Unsupported("op%d's source lifetime does not survive Apple's "
                                                  "decoder, so a value it reads cannot be used again"
                                                  % opc)
                            continue
                    k_src = srcs.index(i)
                    neg, absol = (f.get("mods") or [])[k_src] if k_src < len(f.get("mods") or []) else (False, False)
                    if neg or absol:
                        # A MODIFIER THE FORM CANNOT CARRY IS REFUSED, NOT DROPPED. put_modifier skips a
                        # missing negate/abs carrier silently, which would compile a - b as a + b.
                        car = g17auth.carriers(opc, g17auth.lifetime_operand(opc, i))
                        if (neg and g17auth.MOD_NEG not in car) or (absol and g17auth.MOD_ABS not in car):
                            raise Unsupported("op%d operand %d has no %s carrier; the folded modifier "
                                              "cannot be written" % (opc, i, "negate" if neg else "abs"))
                    got = g17auth.put_modifier(opc, b, i, keep, negate=neg, absolute=absol, choice=choice)
                    if got: b = got
                b = b[:m.size]
        elif m.form == "float.unary":
            f = dict(m.fields)
            defs = f.pop("_defs", []); uses = f.pop("_uses", [])
            b = g17asm.encode_trans(f["opcode"], defs[0], uses[0],
                                    _tmpl(FLOAT_UNARY_TEMPLATE[f["opcode"]], m.form, f["opcode"]),
                                    keep=f.get("keep_src"))
        elif m.form in ("publish.coord.x", "publish.coord.y"):
            # AUTHORED THROUGH g17as, WHICH IS THE POINT. An earlier version of this branch
            # assembled Apple's memory-source publish and then patched byte3's reg/expr selector
            # and byte1's register in behind g17as's back. That produced an instruction with an
            # op4 destination and operand 2 still carrying the op0 destination's pinned imm - a
            # combination g17as would have refused, and the refusal was right. Supplying operand 2
            # as 1048576 makes the whole instruction consistent and it assembles with no patching;
            # Apple's decoder reads `expr:bin(op4,const(0),4) ... reg:112` back out of it.
            #
            # THIS ENCODING IS CONSTRUCTED. All 104 witnesses of op592 operand 3's register form
            # write op0 destinations, because a per-lane coordinate in Apple's own code is written
            # by the ALU instead (opcodes 444 and 17070, expr-destination siblings that no map
            # here has yet). What licenses it is a dispatch, not the map.
            #
            # NO `/ wait=3`, AND THAT IS A FIX RATHER THAN AN OMISSION. The wait control writes
            # the SAME byte-1 bits as the memory source's coordinate const, so `wait=3` pins that
            # const to 12 and every other value is refused by the round trip. Dropping it is byte
            # IDENTICAL at const 12 - 0b0c210c either way, because wait=3's bits ARE the const's.
            # The renderer prints both readings of the same bits, which is how the two fields came
            # to overlap in the map at all.
            #
            # THE POPULATION'S LIMIT: registers 105..137 are witnessed, enc 0..32. The field is
            # six bits and holds 0..63, and g17as will encode r138 and above because its own
            # decode agrees with its own encode. That is self-consistency, not a witness, so this
            # refuses above r137 rather than trusting the field's width.
            f = dict(m.fields); uses = f.pop("_uses", [])
            reg = uses[0]
            if not 0 <= reg <= 32:
                raise Unsupported("publish source r%d: registers r105..r137 are witnessed for "
                                  "op592 operand 3 and this is outside them. The field is six "
                                  "bits wide, which is not the same as six bits exercised"
                                  % (reg + 105))
            dst = 0 if m.form.endswith("x") else 2
            b = g17as.assemble("publish.l4 [op4+%d*4], r%d, op2=#1048576"
                               % (dst, reg + 105)).text
        elif m.form == "texture.read.32":
            # EVERY OPERAND NAMED. op1 carries a lifetime bit at 2^33 - the field class that has
            # produced a silent zero five times here - and the value written is the one Apple emits
            # for a read whose result is consumed once, chosen rather than inherited. op5 and op8
            # are the two constant offsets, pinned in the map with no register. op11's width is
            # pinned at (5, 8) by the map; only its const is free and 0 is the first coordinate
            # slot. op12 is the per-fetch counter's first value.
            f = dict(m.fields); defs = f.pop("_defs", [])
            line = ("texture.read.l8 r%d, #8607760384, [op0+0*8], [op0+4*8], [op5+0*8], "
                    "#1048592, op7=#%d / wait=1" % (defs[0] + 105, f["tex"]))
            b = g17as.assemble(line).text
        elif m.form in ("store.ib.32", "load.ib.32"):
            # AUTHORED THROUGH g17as, like the atomic, so no bit is inherited from a witness.
            # Slot 5 is the coordinate register the two read_sr above just assembled - NOT a base
            # address and not a live-in, which is what a first reading of this family claimed.
            # op3 is the member's byte offset inside the imageblock struct; op7/op8 are constant
            # offsets from this thread's own tile coordinate. See g17ir.Builder for the map and
            # the 47-of-47 that identifies slot 5.
            f = dict(m.fields)
            uses = f.pop("_uses", []); defs = f.pop("_defs", [])
            # EVERY OPERAND BY INDEX. The store and the load derive different positional
            # signatures from the same nine slots - the load leaves the y offset a modifier - so
            # naming each slot explicitly is what makes one emit path serve both.
            # OPERAND 1 IS THE PROVENANCE AND WAIT FIELD and its encodable range is not the same
            # on the two forms - the store's constant is not representable in the load's. Each
            # takes the value Apple's own probes carry for that direction. The 2^37 bit some
            # instances add is NOT set here: it appears on the last access of a sequence, the same
            # position that carries op6=16, and it has not been named, so it is left clear rather
            # than copied across.
            if m.form == "store.ib.32":
                val, coord = uses[0], uses[1]
                op1 = IB_STORE_OP1
                slots = {0: "r%d" % (val + 105), 1: "#%d" % op1, 6: "#0" if f.get("keep_coord") else "#16"}
            else:
                coord = uses[0]
                op1 = IB_LOAD_OP1
                slots = {0: "r%d" % (defs[0] + 105), 1: "#%d" % op1, 6: "#0"}
            m.fields["op1"] = op1          # what selection asked, so the selfcheck reads it back
            # OP2 IS NOT AN AUTHORING KNOB. The width code reads out of every instance - 16 for
            # 8-bit, 17 for 16-bit, 18 for 32-bit, 241 for half4 - but it is degenerate in the map
            # on both forms, because the element width is chosen by picking the OPCODE off the
            # read/write ladder rather than by setting a field. So it is not written here, and
            # `store.ib.32`/`load.ib.32` carry 32-bit by construction.
            slots.update({3: "#%d" % f["member"], 5: "r%d" % (coord + 105),
                          7: "#%d" % f["dx"], 8: "#%d" % f["dy"]})
            line = "%s %s" % (m.form, ", ".join("op%d=%s" % (i, slots[i])
                                                for i in sorted(slots)))
            b = g17as.assemble(line).text
        elif m.form == "atomic.uniform.10":
            # op10094, THE UNIFORM FORM. Authored through g17as like the per-lane one. Its slots
            # are NOT the per-lane form's: there is no index register, the address is slot 3, and
            # the value sits at slot 8 rather than slot 9.
            f = dict(m.fields)
            uses = f.pop("_uses", []); defs = f.pop("_defs", [])
            line = _as_line("atomic.l10", 10094, 10,
                            {0: "r%d" % (defs[0] + 105),
                             3: "[op%d+%d*8]" % (f["base"], f["offset"]),
                             8: "r%d" % (uses[0] + 105)},
                            # operand 9 is the VALUE's lifetime and only 16 (release) is representable here, so a
                            # value read again reaches this form through a copy (DESTRUCTIVE_FORMS, MM 25.144.6)
                            pinned={1: _atomic_fill_slot7(17825792), 6: 0, 9: 16},
                            controls=" / waitload aop=%d" % f.get("aop", 0))
            b = g17as.assemble(line).text
            if f.get("slot6"):
                # EXPERIMENTAL: operand 6 written directly into bits 53..60, read back by APPLE'S decoder (g17as's
                # render does not map the field there)
                u = int.from_bytes(bytes(b), "little")
                u = (u & ~(0xFF << 53)) | ((f["slot6"] & 0xFF) << 53)
                b = u.to_bytes(len(b), "little")
                from . import model as _model
                _vals = list(_model.decode(bytes(b), 0))[0].values
                if _vals[6] != ("imm", f["slot6"] & 0xFF):
                    raise Unsupported("op10094 operand 6 written %d decodes as %r" % (f["slot6"], _vals[6]))
        elif m.form == "atomic.tg.uniform.12":
            # op11765. OPERAND 6 IS THE FIELD TODAY'S LESSON IS ABOUT: a two-bit table taking
            # 16, 1, 4 or 8, refuted as a linear ramp in the main map and corrected as a table in
            # the atomics overlay, which is what made this form authorable at all. Apple emits 1
            # in every instance, so 1 is what is written - deliberately, not inherited.
            # EVERY OPERAND SET DELIBERATELY, which is the habit today's uniform-atomic bug
            # bought. op1 is the provenance and wait field: the form's default carries an extra
            # 2^20 over what Apple emits here, and that bit is not this compiler's to leave to a
            # witness.
            f = dict(m.fields)
            uses = f.pop("_uses", []); defs = f.pop("_defs", [])
            # AND THE OPERATION, which a first version of this line left unset - the form's
            # default is not `add`, and an atomic that quietly subtracts is the same class of
            # defect as one that quietly does nothing.
            # OP2 IS STATED, NOT LEFT TO THE MODAL. It was the one operand this compiler wrote
            # from "whatever Apple usually puts there", which is a field governed by something the
            # compiler cannot say - the shape the ISA peer found on their own side the same day.
            # Asking what varies with it: 20 of Apple's 21 instances carry 262656 and every one of
            # those is a fetch-family RMW; the single instance carrying 262658 is an exchange
            # probe. This backend emits the fetch family and never exchange, so 262656 is written
            # for a reason it can state - 20 of 20 on the operation it actually emits.
            #
            # The exchange correlation is ONE WITNESS and is recorded as an observation, not a law:
            # it says where to look if an exchange is ever lowered, not what the bit means.
            line = _as_line("atomic.tg", 11765, 12,
                            {0: "r%d" % (defs[0] + 105), 7: "r%d" % (uses[0] + 105)},
                            pinned={1: _atomic_fill_slot7(2199041081376), 2: f["op2"], 6: 1})
            b = g17as.assemble(line).text
        elif m.form == "mov.word.4":
            f = dict(m.fields)
            uses = f.pop("_uses", []); defs = f.pop("_defs", [])
            # A PURE ENCODER, unlike mov.4's line-assembler path, so a program containing this copy can be
            # part of a decoder-free audited build. Same fields, one width up.
            b = g17asm.encode_movword(dest=defs[0], src=uses[0], keep_src=f.get("keep_src", True))
        elif m.form == "alu.fadd.6":
            f = dict(m.fields)
            uses = f.pop("_uses", []); defs = f.pop("_defs", [])
            from agxforge.g17.formenc import Fadd6 as g17fadd6
            b = g17fadd6.encode(defs[0], uses[0], uses[1],
                                dest_life=f.get("dest_life", 32),
                                src0_life=f.get("src0_life", 16),
                                src1_life=f.get("src1_life", 16))
        elif m.form == "alu.fadd.4":
            f = dict(m.fields)
            uses = f.pop("_uses", []); defs = f.pop("_defs", [])
            from agxforge.g17.formenc import Fadd4 as g17fadd4
            b = g17fadd4.encode(defs[0], uses[0], uses[1],
                                dest_life=f.get("dest_life", 32),
                                src0_life=f.get("src0_life", 16),
                                src1_life=f.get("src1_life", 16))
        elif m.form == "movimm.2":
            f = dict(m.fields)
            defs = f.pop("_defs", []); f.pop("_uses", None)
            from agxforge.g17.formenc import Movimm2 as g17movimm2
            b = g17movimm2.encode(defs[0], f["imm"])
        elif m.form == "alu.ffma.6":
            f = dict(m.fields)
            uses = f.pop("_uses", []); defs = f.pop("_defs", [])
            from agxforge.g17.formenc import Ffma6 as g17ffma6
            b = g17ffma6.encode(defs[0], uses[0], uses[1], uses[2],
                                dest_life=f.get("dest_life", 32),
                                src0_life=f.get("src0_life", 16),
                                src1_life=f.get("src1_life", 16),
                                src2_life=f.get("src2_life", 16))
        elif m.form == "alu.ffma.4":
            f = dict(m.fields)
            uses = f.pop("_uses", []); defs = f.pop("_defs", [])
            from agxforge.g17.formenc import Ffma4 as g17ffma4
            # THE TIE, CHECKED WHERE THE REGISTERS ARE KNOWN. uses[0] is the accumulator and the
            # form encodes it as the destination re-printed; if allocation did not give them one
            # register there is no encoding for what was asked, and a refusal is the only honest
            # answer. This is the "unrepresented accumulator placement" integration's assignment
            # asks to refuse precisely.
            if uses[0] != defs[0]:
                raise Unsupported("op2190/4 is two-address: its accumulator is the destination "
                                  "re-printed, and the allocator gave the accumulator r%d against "
                                  "destination r%d. Nothing at this length encodes a separate "
                                  "accumulator register." % (uses[0], defs[0]))
            b = g17ffma4.encode(defs[0], uses[1], uses[2],
                                accumulator_printed_at_6=True,
                                dest_life=f.get("dest_life", 32),
                                src0_life=f.get("src0_life", 16),
                                other_life=f.get("other_life", 16))
        elif m.form == "alu.fmul.4":
            f = dict(m.fields)
            uses = f.pop("_uses", []); defs = f.pop("_defs", [])
            from agxforge.g17.formenc import Fmul4 as g17fmul4
            b = g17fmul4.encode(defs[0], uses[0], uses[1], dest_life=f.get("dest_life", 32),
                                src0_life=f.get("src0_life", 16), src1_life=f.get("src1_life", 16))
        elif m.form == "mov.half.4":
            f = dict(m.fields)
            uses = f.pop("_uses", []); defs = f.pop("_defs", [])
            # THE LIFETIME IS WRITTEN FROM LIVENESS, not inherited: this copy exists because the source is
            # read again, so it keeps its source. memory:g17-modifier-operand-lifetimes.
            # THE DESTINATION FILE IS THE MOVE'S OWN FIELD NOW. A packing move for an odd
            # component writes the HIGH half of the group's word, which is the 281-based file; the
            # duplicate-copy moves and every previously emitted instance pass no field and stay in
            # 425, so their bytes are unchanged. The SOURCE is always 425: the value being packed
            # was produced into a word's low half, and the encoder's own default for src_281 is
            # True, so this has to be written rather than left.
            b = g17asm.encode_movhalf(dest=defs[0], src=uses[0], keep_src=f.get("keep_src", True),
                                      dest_281=bool(f.get("dest_281", False)), src_281=False)
        elif m.form == "acc.scale.14":
            f = dict(m.fields)
            uses = f.pop("_uses", []); f.pop("_defs", None)
            from agxforge.g17 import epienc as _epienc
            b = _epienc.fmul(f["acc_reg"], f["acc_reg"], uses[0], keep_a=True, keep_b=True)
            if f.get("load_wait"):
                b = bytes([b[0] | 0x08]) + b[1:]     # the slot-7 load-use wait (operand 1 bit 31), as cc's op3290
        elif m.form == "acc.write.10":
            # THE WIDE ACCUMULATOR WRITE (MM 25.144.8): R<acc> = value | 0, the ten-byte bitwise OR with an
            # immediate, whose destination field is seven bits; the source kept, the load wait authored
            f = dict(m.fields)
            uses = f.pop("_uses", []); f.pop("_defs", None)
            b = g17asm.encode_bitwise_imm(f["opcode"], f["acc_reg"], uses[0], 0,
                                          _tmpl(BITWISE_TEMPLATE[f["opcode"]], "bitwise.imm", f["opcode"]),
                                          hazard=f.get("hazard", 0), keep=True)
        elif m.form in ("acc.read.4", "acc.write.4"):
            # A MOVE WITH ONE FIXED END (MM 25.144.8): the register accumulator's physical register on one
            # side, an allocated value on the other; mov.4's encoding, the source always kept.
            f = dict(m.fields)
            uses = f.pop("_uses", []); defs = f.pop("_defs", [])
            f.pop("opcode", None)            # the plain move (op586), named for the ABI's form table
            dst, src = ((defs[0], f["acc_reg"]) if m.form == "acc.read.4" else (f["acc_reg"], uses[0]))
            line = _as_line("mov.l4", 586, 4, {0: "r%d" % (dst + 105), 2: "r%d" % (src + 105)},
                            pinned={1: 0, 3: MOV_KEEP})
            b = g17as.assemble(line).text
        elif m.form == "mov.4":
            f = dict(m.fields)
            uses = f.pop("_uses", []); defs = f.pop("_defs", [])
            # op1 IS NOT THE LIFETIME and is written 0 for a reason: it is 0 in 4,002 of Apple's
            # 4,707 instances, and the 680 carrying 32 are not the ones whose source survives.
            line = _as_line("mov.l4", 586, 4,
                            {0: "r%d" % (defs[0] + 105), 2: "r%d" % (uses[0] + 105)},
                            pinned={1: 0,
                                    3: MOV_KEEP if f.get("keep_src", True) else MOV_RELEASE})
            b = g17as.assemble(line).text
        elif m.form == "store.half.14":
            f = dict(m.fields)
            uses = f.pop("_uses", []); defs = f.pop("_defs", [])
            # OPERAND 7 IS THE BYTE DISPLACEMENT AND IT IS STATED, NOT INHERITED. Left unnamed
            # it is a modifier, and a modifier is filled from the template - which came from a
            # probe writing h[400u + tp.x], so every store this compiler emitted carried that
            # kernel's 800. This program addresses its row through the index register, so the
            # displacement is zero, and zero is a value the form encodes: see OVERLAY3 in
            # g17as.py for why asking for it used to clear an instruction-length bit instead.
            line = _as_line("store@17193", 17193, 14,
                            {0: "r%d" % (uses[0] + 425),          # the 16-bit value
                             3: "[op0+%d*8]" % f["buf_const"],    # the binding's rank, as op17229
                             5: "r%d" % (uses[1] + 105)},         # the 32-bit index
                            # OPERAND 1 IS THE VALUE'S LIFETIME AND 6 IS THE INDEX'S, 0 keeps and
                            # 16 releases (ledger/g17-half-store-displacement-refusal.toml, paired
                            # one-variable controls). Stated rather than inherited, so a value with
                            # a later reader is not freed under it.
                            pinned={7: 0,                         # the byte displacement
                                    1: 0 if f.get("keep_value") else 16,
                                    6: 0 if f.get("keep_index") else 16})
            b = g17as.assemble(line).text
        elif m.form == "store.byte.14":
            # The measured requantization writer is the same 14-byte narrowing form as Apple's
            # int8_t store.  Its source is the low half of the clamped word register and its
            # destination declaration is uchar; unlike an ordinary half store this branch does
            # not admit arbitrary I16 buffers or generic byte stores.
            f = dict(m.fields)
            uses = f.pop("_uses", []); defs = f.pop("_defs", [])
            line = _as_line("store@17193", 17193, 14,
                            {0: "r%d" % (uses[0] + 425),
                             3: "[op0+%d*8]" % f["buf_const"],
                             5: "r%d" % (uses[1] + 105)},
                            pinned={7: 0,
                                    1: 0 if f.get("keep_value") else 16,
                                    6: 0 if f.get("keep_index") else 16})
            b = g17as.assemble(line).text
        elif m.form == "cvt.f2i":
            f = dict(m.fields)
            uses = f.pop("_uses", []); defs = f.pop("_defs", [])
            # Apple's (uint)x is 2f00001a2200ae02b003 and (int)x 2f00001a2200ae027001 (both at r105 <- r105)
            line = _as_line("cvt.f2i@9320", 9320, 10, {0: "r%d" % (defs[0] + 105), 4: "r%d" % (uses[0] + 105)},
                            pinned={1: 2147483648, 2: f["code"], 3: 1, 5: MOV_RELEASE})
            b = g17as.assemble(line).text
        elif m.form in ("ffma.f16", "acc.ffma.f16"):
            # op798/12: 0 destination, 2 / 4 / 6 the sources, 1 / 3 / 5 / 7 their modifiers (operand 1 Apple's
            # 2147483648, the rest the lifetime). Apple's own fma(half, half, half) is 3800060a2320a40245018000.
            f = dict(m.fields)
            uses = f.pop("_uses", []); defs = f.pop("_defs", [])
            dest = ((281 if f["acc_hi"] else 425) + f["acc_reg"] if m.form == "acc.ffma.f16"
                    else 425 + defs[0])
            src = ["r%d" % ((281 if h == "hi" else 425) + u) for u, h in zip(uses, f["halves"])]
            keeps = f.get("keeps") or [False] * 3
            line = _as_line("ffma.f16.l12", 798, 12, {0: "r%d" % dest, 2: src[0], 4: src[1], 6: src[2]},
                            pinned={1: 2147483648, 3: MOV_KEEP if keeps[0] else MOV_RELEASE,
                                    5: MOV_KEEP if keeps[1] else MOV_RELEASE,
                                    7: MOV_KEEP if keeps[2] else MOV_RELEASE})
            b = g17as.assemble(line).text
        elif m.form == "cvt.f32.f16":
            f = dict(m.fields)
            uses = f.pop("_uses", []); defs = f.pop("_defs", [])
            line = _as_line("cvt.f32.f16.l12", 1016, 12,
                            {0: "r%d" % (defs[0] + 425),      # the 16-bit destination
                             2: "r%d" % (uses[0] + 105)},     # the 32-bit source
                            pinned={1: 2147483648,
                                    3: MOV_KEEP if f.get("keep_src") else MOV_RELEASE})
            b = g17as.assemble(line).text
        elif m.form == "cvt.f16.f32":
            f = dict(m.fields)
            uses = f.pop("_uses", []); defs = f.pop("_defs", [])
            line = _as_line("fadd.imm.l12", 1004, 12,
                            {0: "r%d" % (defs[0] + 105),      # the 32-bit destination
                             2: "r%d" % (uses[0] + 425)},     # the 16-bit source
                            pinned={1: 2147483648,
                                    3: MOV_KEEP if f.get("keep_src") else MOV_RELEASE})
            b = g17as.assemble(line).text
        elif m.form == "simd.broadcast.10":
            f = dict(m.fields)
            uses = f.pop("_uses", []); defs = f.pop("_defs", [])
            # operand 3 is the source lifetime, written from liveness (32 keeps, 16 releases)
            line = _as_line("simd.shuffle@14157", 14157, 10,
                            {0: "r%d" % (defs[0] + 105), 2: "r%d" % (uses[0] + 105)},
                            pinned={1: 16777248, 3: 32 if f.get("keep_src") else 16})
            b = g17as.assemble(line).text
        elif m.form in ("atomic.add.10", "atomic.add.12"):
            # AUTHORED THROUGH g17as, NOT FROM A TEMPLATE. Every other form here starts from a
            # modal witness and overwrites the fields it knows, which leaves the rest inherited.
            # g17as builds the instruction from the operand maps, so the only bits that appear are
            # ones something measured. That is the whole point of the atomic being the first form
            # lowered this way; ledger/g17-the-atomic-forms-are-authorable.toml.
            #
            # printed = logical + 105. The maps decode this operand class with base 105, and this
            # compiler's logical r4 reads back as reg:109 through the reference disassembler.
            #
            # op1=#1048576 is operand 1 with only its 2^20 bit set - the COMPUTED-addend case. A
            # literal addend sets 2^24 as well and is refused in selection, because the map cannot
            # distinguish 2^24 from 2^25 and would emit a different instruction.
            f = dict(m.fields)
            defs = f.pop("_defs", []); uses = f.pop("_uses", [])
            # EVERY SLOT THE FORM DECLARES, NAMED BY INDEX. Four positional operands is a bet
            # that the map will keep saying four, and it does not: under a re-baseline op10090's
            # twelve-byte form derives SIX slots and authoring breaks outright - which was the one
            # blocker left on adopting that re-baseline. The roles this compiler owns are written
            # from selection, the two it has measured are pinned, and any slot it has no opinion
            # about takes Apple's MODAL value from the same map that declared the slot.
            line = _as_line("atomic.idx.l%d@10090" % m.size, 10090, m.size,
                            {0: "r%d" % (defs[0] + 105),
                             3: "[op%d+%d*8]" % (f["base"], f["offset"]),
                             5: "r%d" % (uses[0] + 105),
                             9: "r%d" % (uses[1] + 105)},
                            pinned={1: 0x80800000 if f.get("consumed_result") else ATOMIC_IDX_OP1[m.size],
                                    6: 0},
                            controls=" / waitload aop=%d" % f.get("aop", 0))
            b = g17as.assemble(line).text
            if f.get("consumed_result"):
                from agxforge.g17 import model as _g17model
                if m.size != 12 or f.get("aop", 0) not in (0, 1, 3, 6, 7):
                    raise Unsupported("consumed per-lane atomic result is encoded only for 12-byte ADD/AND/OR/SUB/XOR")
                if f.get("aop", 0) == 0:
                    # g17as's wide ADD operand map leaves three control bits set. Clearing
                    # precisely these yields the hardware-tested form. The XOR form already
                    # decodes with the required control bits, and needs no such correction.
                    b = bytearray(b)
                    b[1] &= ~0x04
                    b[8] &= ~0x80
                    b[11] &= ~0x10
                    b = bytes(b)
                elif f.get("aop", 0) in (1, 6):
                    # In op10090/l12, the assembler's three-bit aop=6 leaves
                    # byte11[4] set. A pure Submit showed exchange semantics;
                    # clearing just that bit gave OR. aop=1 does not decode
                    # until this bit clears; the resulting AND form updated
                    # memory and returned old values in two pure Submits.
                    b = bytearray(b)
                    b[11] &= ~0x10
                    b = bytes(b)
                decoded = list(_g17model.decode(b, 0))
                if len(decoded) != 1 or decoded[0].opcode.id != 10090 or decoded[0].size != 12:
                    raise Unsupported("consumed per-lane ADD/AND/OR/SUB/XOR does not decode as 12-byte op10090")
                v = decoded[0].values
                if (v[0] != ("reg", defs[0] + 105) or v[1] != ("imm", 0x80800000) or
                        v[2] != ("imm", {0:262656,1:262664,3:262657,6:262665,7:262666}[f.get("aop",0)]) or
                        v[5] != ("reg", uses[0] + 105) or
                        v[9] != ("reg", uses[1] + 105)):
                    raise Unsupported("consumed per-lane ADD/AND/OR/SUB/XOR decoded operands differ from selection")
        elif m.form == "atomic.cmpxchg.12":
            from agxforge.g17 import model as _g17model
            defs = m.fields.pop("_defs", [])
            uses = m.fields.pop("_uses", [])
            if m.fields.get("base") != 0:
                raise Unsupported("per-lane compare-exchange has only executed at atomic base 0")
            k = uses[0] if len(uses) == 3 else -1
            if not 0 <= k <= 12 or uses != [k, k + 1, k + 2] or defs != [k + 3]:
                raise Unsupported("per-lane compare-exchange needs the measured consecutive "
                                  "index, desired/expected and destination placement with index "
                                  "R0..R12; beyond that the destination crosses this tested field")
            # The assembler's narrow overlay admits only the observed compare-exchange
            # operation constant. R0, R1, R5, R7 and R12 passed two pure Submits each;
            # other R0..R12 placements are decode-checked predictions of these fields.
            line = _as_line("atomic.idx@10091", 10091, 12,
                            {0: "r%d" % (defs[0] + 105), 3: "[op0+0*8]",
                             5: "r%d" % (uses[0] + 105),
                             9: "r%d" % (2451 + uses[1])},
                            pinned={1: 0x80800000, 2: 262659, 6: 0})
            b = g17as.assemble(line).text
            decoded = list(_g17model.decode(b, 0))
            if (len(decoded) != 1 or decoded[0].opcode.id != 10091 or
                    decoded[0].size != 12 or decoded[0].values[0] != ("reg", defs[0] + 105) or
                    decoded[0].values[1] != ("imm", 0x80800000) or
                    decoded[0].values[2] != ("imm", 262659) or
                    decoded[0].values[5] != ("reg", uses[0] + 105) or
                    decoded[0].values[9] != ("reg", 2451 + uses[1])):
                raise Unsupported("per-lane compare-exchange template decode differs from execution")
        elif m.form == "unary":
            f = dict(m.fields)
            defs = f.pop("_defs", []); uses = f.pop("_uses", [])
            b = g17asm.encode_unary(f["opcode"], defs[0], uses[0],
                                    _tmpl(UNARY_TEMPLATE[f["opcode"]], m.form, f["opcode"]),
                                    hazard=f.get("hazard", 0), keep=f.get("keep_src"),
                                    signed=f.get("signed"))
        elif m.form == "bitwise.reg":
            f = dict(m.fields)
            defs = f.pop("_defs", []); uses = f.pop("_uses", [])
            b = g17asm.encode_bitwise_reg(f["opcode"], defs[0], uses[0], uses[1],
                                          _tmpl(BITWISE_REG_TEMPLATE[f["opcode"]], m.form, f["opcode"]))
        elif m.form == "bitwise.reg.10":
            f = dict(m.fields)
            defs = f.pop("_defs", []); uses = f.pop("_uses", [])
            b = g17asm.encode_bitwise_reg10(f["opcode"], defs[0], uses[0], uses[1])
        elif m.form == "bitwise.imm":
            f = dict(m.fields)
            defs = f.pop("_defs", []); uses = f.pop("_uses", [])
            # THE HAZARD WORD IS AUTHORED, NOT INHERITED. It is zero here: no load is
            # outstanding, so there is nothing to wait on and no tag to carry.
            b = g17asm.encode_bitwise_imm(f["opcode"], defs[0], uses[0], f["imm"],
                                          _tmpl(BITWISE_TEMPLATE[f["opcode"]], m.form, f["opcode"]),
                                          hazard=f.get("hazard", 0), keep=f.get("keep_src"))
        elif m.form in ("alu.sub.imm", "alu.sub.reg", "alu.mul.imm", "alu.mul.reg",
                        "alu.shift.imm", "alu.shift.reg"):
            # PER-OPCODE TEMPLATE, so this form does not go through the family table at all:
            # 11666 and 10279 are the same family and differ in eight bytes.
            f = dict(m.fields)
            defs = f.pop("_defs", []); uses = f.pop("_uses", [])
            opc = f["opcode"]
            slot_b = ("imm", f["imm"]) if m.form.endswith(".imm") else ("reg", uses[1], 0)
            # THE ADDEND IS SEMANTIC AND ZERO HERE. op10822 computes A*B + addend; the IR's
            # `mul` has no addend, so the compiler must write 0 rather than inherit whatever the
            # template's kernel multiplied and added. g17asm.ALU_ADDEND holds the measured field.
            b = g17asm.encode_alu_form(opc, defs[0], ("reg", uses[0], 0), slot_b,
                                       _tmpl(ALU_TEMPLATE_BY_OPCODE[opc], m.form, opc),
                                       keep_a=f.get("keep_a"), keep_b=f.get("keep_b"),
                                       addend=0 if opc in g17asm.ALU_ADDEND else None,
                                       hazard=f.get("hazard", 0))
        elif m.form == "alu.block":
            f = dict(m.fields); defs = f.pop("_defs", []); uses = f.pop("_uses", [])
            b = g17asm.encode_alu_block(dest=defs[0], src1=uses[0], const=f["const"])
        elif m.form.startswith(("load.vec", "store.vec")):
            f = dict(m.fields); defs = f.pop("_defs", []); uses = f.pop("_uses", [])
            tup = defs[0] if m.form.startswith("load") else uses[0]
            b = g17asm.encode_vec4("load" if m.form.startswith("load") else "store", tuple_base=tup, index=uses[-1], desc_const=f["desc"], disp=f["disp"], release_index=f["release_index"], wait_sr=f.get("wait_sr", False), n=f.get("n", 4),
                                   access_bytes=4 if f.get("half2") else None)
            if f.get("half2"):
                # op12691 -> op12655: byte0[3] CLEARED (0x4f -> 0x47), found by asking Apple's decoder
                # which single flip moves the opcode and leaves every operand; selfcheck decodes it
                b = bytearray(b); b[0] &= ~0x08 & 0xFF; b = bytes(b)
        elif m.form in ("tensor.inherited", "tensor.authored", "tensor.wholekernel"):
            # the tensor stream's bytes are authored (g17tensorlower), inherited (declared), or
            # authored whole-kernel (tensorgemm.lower_gemm); no template applies - the bytes ARE
            # the instruction. A form absent from this tuple reaches the template lookup and is
            # refused there with "no canonical template", which is how the route first failed.
            b = bytes(m.fields["bytes"])
        else:
            f = dict(m.fields)
            if m.form == "movimm16.zero.4":
                # Selected from a retained Apple witness (W0) whose four bytes ARE the template, so there is no
                # registry entry to look up and no default that could silently promote it to another length.
                t = g17asm.MOVIMM16_ZERO_TEMPLATE
            elif m.form.startswith("store.elem1."):
                t = _memory_gate(m.form, {8: g17asm.WORDSLOT_TEMPLATE_8, 10: g17asm.WORDSLOT_TEMPLATE_10,
                                          14: g17asm.WORDSLOT_TEMPLATE_14}[m.size])
            elif m.form.startswith("store.half1."):
                # The three templates ARE the retained witnesses, one per length, so there is no registry
                # entry that could promote this to another length and no canonical template to look up.
                t = _memory_gate(m.form, {8: g17asm.HALFSLOT_TEMPLATE_8, 10: g17asm.HALFSLOT_TEMPLATE_10,
                                          14: g17asm.HALFSLOT_TEMPLATE_14}[m.size])
            elif m.form == "store.halfvec.14":
                # THE TEMPLATE IS THE RETAINED FOURTEEN-BYTE CORPUS INSTANCE, chosen by component count -
                # the count is part of Apple's opcode number, so there is one template per opcode and no
                # registry entry that could promote this to another length.
                t = _memory_gate(m.form, g17asm.HALFVEC_SLOT_TEMPLATES[f["n"]])
            elif m.form == "load.8":
                # This form is selected explicitly from a checked-in Apple witness. It has no
                # byte9/byte13 fields, so use the fixed eight-byte template and the same device
                # memory gate as the long form. No default registry entry may silently promote it.
                t = _memory_gate(m.form, LOAD8_TEMPLATE)
            elif m.form == "load.10":
                t = _memory_gate(m.form, LOAD10_HALF)
            else:
                F = forms.get(m.form)
                if F is None: raise Unsupported("no canonical template for form %r" % m.form)
                t = _tmpl(F["template"], m.form)
            # A LOAD THAT USES A FIELD NEEDS A FORM THAT HAS IT. Neither `narrow` (byte9 bit2) nor
            # `hi16` (byte4 bit2) is EVER set on op12682: 0 of 1,359 Apple instances for each, and
            # setting either emits bytes Apple's decoder rejects outright. op12646 carries both -
            # narrow in 1 of 586, hi16 in 208 of 586 - and retargets cleanly 24 of 24 over
            # dest x index x narrow. So a load that needs either field selects it, and a plain one
            # keeps the form the rest of the backend is proven on.
            if m.form == "load.14" and f.get("half"):
                # A SIXTEEN-BIT ELEMENT SELECTS ANOTHER FORM. Not through _tmpl for the same reason
                # the narrow load is not: the hook would answer with the registry's op12682
                # constant and discard the choice.
                t = _memory_gate(m.form, LOAD14_HALF)
            elif m.form == "load.14" and (f.get("narrow") or f.get("hi16")):
                # NOT THROUGH _tmpl: its TEMPLATE_HOOK answers with the registry's load.14
                # constant and would discard this one, which is the whole point of choosing it.
                t = _memory_gate(m.form, LOAD14_NARROW)
            defs = f.pop("_defs", []); uses = f.pop("_uses", []); f.pop("range_group", None)
            if m.form == "alu.12":
                # The hazard word: zero unless this ALU must wait on a pending load, and then
                # only bit 31. Written whole rather than one bit at a time.
                f.setdefault("hazard", (1 << 31) if f.get("load_wait") else 0)
                b = g17asm.encode_alu(dest=defs[0], src1=uses[0], template=t,
                                      src2=uses[1] if len(uses) > 1 else None, **f)
            elif m.form == "movimm16.zero.4":
                b = g17asm.encode_movimm16_zero(dest=defs[0])
            elif m.form == "movimm.8":
                b = g17asm.encode_movimm(imm=f["imm"], template=t, dest=defs[0])
            elif m.form == "read_sr.4":
                b = g17asm.encode_sr(dest=defs[0], sr=f["sr"], seq=f["seq"], template=t,
                                     half=f.get("half"))
            elif m.form in ("load.8", "load.10", "load.14"):
                b = g17asm.encode_load(dest=defs[0], base=f["base"], offset=f["offset"],
                                       template=t, index_reg=uses[0], disp2=f.get("disp2", 0),
                                       index_scale=f.get("index_scale", 1),
                                       narrow=f.get("narrow", 0), hi16=f.get("hi16", 0))
            elif m.form.startswith(("store.half1.", "store.elem1.")):
                b = g17asm.encode_element_store(src=uses[0], disp=f["disp"], half=m.form.startswith("store.half1."),
                                                const=f.get("const"), wait_load=f.get("wait_load", 0), template=t)
                if f.get("keep_members"):
                    b = _keep_element_store_source(b, 17199 if m.form.startswith("store.half1.") else 17235)
            elif m.form in ("store.8", "store.14", "store.halfvec.14"):
                # src is the FIRST register of the range; the hardware writes src..src+n-1 to
                # slots slot..slot+n-1. IN REGISTER ORDER, WHICH NEED NOT BE USES ORDER.
                #
                # The range pre-colouring assigns the group consecutive registers in uses order,
                # so uses[0] - the value the IR actually asked to store - is normally the lowest
                # and lands on the requested slot. PHI COALESCING RUNS AFTERWARDS and pins a
                # loop-carried value to its own register, which can leave uses[0] above its
                # partner: the counted loop stored acc at r6 and its reserved zero at r5, so the
                # hardware wrote the zero to the requested slot and acc to the next one. A silent
                # wrong slot, found by running the first counted loop.
                #
                # The slot is shifted by the value's position in the run, so uses[0] lands where
                # it was asked for. Offset 0 is the normal case and leaves every byte unchanged.
                _off = uses[0] - min(uses)
                _slot = f["slot"] - _off
                if _off and f["n"] > 2 and list(uses) != sorted(uses):
                    raise Unsupported("a %d-value range store's registers are out of uses order "
                                      "(%s); one shift cannot place them all" % (f["n"], list(uses)))
                if _slot < 0:
                    raise Unsupported("range store at slot %d would need base slot %d"
                                      % (f["slot"], _slot))
                m.fields["_slot_emitted"] = _slot
                b = g17asm.encode_store(src=min(uses), n=f["n"], slot=_slot, template=t, subform=f.get("subform"),
                                        const=f.get("const"),
                                        wait_load=f.get("wait_load", 0))
                if f.get("keep_members"):
                    # a member is read again after this store: write KEEP (32) on the source
                    # operand, certified on this very template through Apple's decoder
                    # (isa/g17-auth-on-templates.json; tools/g17authtables.py harvests it) -
                    # see _lifetimes. A form whose lifetime cannot be written here is refused
                    # rather than left releasing a live value.
                    # THE OPCODE IS THE FORM'S, NOT THE WORD FAMILY'S. The half-vector stores are
                    # different opcode numbers (17208/17217/17226), and asking the certification table
                    # about op17262 while emitting op17226 would certify a lifetime on another form -
                    # the confound memory:proven-of-an-encoding-not-an-opcode names. If the half form's
                    # source lifetime does not survive Apple's decoder, this refuses below by name.
                    opc = ({2: 17208, 3: 17217, 4: 17226} if m.form == "store.halfvec.14"
                           else {2: 17244, 3: 17253, 4: 17262})[f["n"]]
                    try:
                        choice = g17auth.certify_on(opc, b, 0)
                    except KeyError as ex:
                        # AN UNCERTIFIED LIFETIME REFUSES BY NAME rather than raising the table's KeyError
                        # at the caller. The half-vector forms are new here and nothing has certified that
                        # their source-lifetime operand survives Apple's decoder on this template, so a
                        # program that reads a stored half again is refused - not emitted with a released
                        # register, which is the defect integration measured on the word forms
                        # (results/g17-tensor-reload-runtime-v1).
                        raise Unsupported("op%d stores a value this program reads again, and its source "
                                          "lifetime is not certified on this template (%s) - so the store "
                                          "would release a live register. Run tools/g17authtables.py to "
                                          "certify it, or keep the value in another register"
                                          % (opc, str(ex).split(",")[0][:80]))
                    if choice is None:
                        raise Unsupported("op%d's source lifetime does not survive Apple's decoder on "
                                          "this template, so a value it stores cannot be read again"
                                          % opc)
                    got = g17auth.put_modifier(opc, b, 0, True, choice=choice)
                    if not got:
                        raise Unsupported("op%d: the source lifetime could not be written" % opc)
                    b = bytes(got)[:m.size]
            else:
                raise Unsupported("no emitter for form %r" % m.form)
        if len(b) != m.size:
            raise AssertionError("%s emitted %d bytes, expected %d" % (m.form, len(b), m.size))
        if m.form.startswith("tensor."):
            tensor_blobs.append((m.form, bytes(b)))
        layout.append((at, bytes(b), m)); code += b
    # Tensor rows carry physical registers in pre-encoded bytes, outside the ordinary MInst
    # defs/uses fields. Decode them once after emission so a copied or authored body cannot
    # silently name the squashed R126+ namespace.
    if tensor_blobs:
        from agxforge.g17 import model as _g17model
        names = _g17model.registers()
        for form, blob in tensor_blobs:
            # A malformed tensor stream is a compiler defect and must remain loud.  Only the
            # measured register-domain violation below is converted to a named refusal.
            decoded = _g17model.decode(blob, 0)
            for ins in decoded:
                for kind, value in ins.values:
                    if kind != "reg":
                        continue
                    name = names.get(value, "")
                    bad = [n for n in registerdomain.registers_in_name_checked(name)
                           if n > registerdomain.ALLOCATABLE_MAX]
                    if bad:
                        raise Unsupported("%s encoded operand %s is outside the allocatable R0..R125 namespace" %
                                          (form, name or "R%d" % value))
    return bytes(code), layout

# --- the program image --------------------------------------------------------------------
class Component:
    """One part of a native program image, with where its bytes came from.

    provenance is the whole point: GENERATED means the compiler authored every byte from
    semantics; INHERITED means they come from an Apple-built container and the dependency is still
    outstanding. Anything in between is not allowed a label - it is INHERITED until it is not.
    """
    __slots__ = ("name", "nbytes", "provenance", "where", "note")
    def __init__(self, name, nbytes, provenance, where="", note=""):
        self.name = name; self.nbytes = nbytes; self.provenance = provenance
        self.where = where; self.note = note

class G17Program:
    """THE PROGRAM IMAGE MODEL: code, constant program, descriptors, launch metadata, tensor
    setup and relocations, each named with its own provenance.

    The mission requires this to exist "even if some fields initially come from an Apple-derived
    container", and then requires those dependencies to be RETIRED one at a time. That is only
    possible if they are enumerated and measured, so this model refuses to describe the container
    dependency in prose - every component carries a byte count and a provenance, and
    `owned_fraction` is the number that has to go to 1.0.

    The container sections were located 2026-09-04 and are no longer "somewhere in the metallib":

        __TEXT,__text              the constant program, then the main program
        __GPU_METADATA,__compute   descriptors
        __GPU_LD_MD,__compute      launch metadata
        __GPU_ARCH_LD_MD,__compute launch metadata, architecture-specific
        __GPU_STATS_MD,__compute   statistics
    """
    SECTION_ROLE = {"__GPU_METADATA,__compute": "descriptors",
                    "__GPU_LD_MD,__compute": "launch_metadata",
                    "__GPU_ARCH_LD_MD,__compute": "launch_metadata_arch",
                    "__GPU_STATS_MD,__compute": "statistics",
                    "__GPU_REMARKS_MD,__compute": "remarks"}

    def __init__(self, name, code, layout, inherited, reference=None, buffers=(), written=(),
                 binding_ranks=None, ranks=None, threadgroup=None, preloads=None, resolved_layout=None,
                 requantization=None):
        # THE KEYWORD IS binding_ranks. `ranks` is the spelling this constructor shipped with and
        # is accepted so a caller written against either tree works: a constructor keyword that
        # differs between two copies of one file raises TypeError at the call site, which reads as
        # "the compiler is broken" rather than as a rename. Both, disagreeing, is a caller bug.
        if binding_ranks is not None and ranks is not None and dict(binding_ranks) != dict(ranks):
            raise Unsupported("binding_ranks and ranks were both given and disagree; "
                              "binding_ranks is the keyword, ranks is its accepted alias")
        binding_ranks = binding_ranks if binding_ranks is not None else ranks
        self.name = name; self.code = code; self.layout = layout
        # The binding ranks THIS function was compiled with, frozen. See compile_function.
        self._abi_cache = {}
        self.reference = reference
        # the function's own signature, which the ABI reports and the bytes cannot recover
        self.buffers = list(buffers); self.written = set(written)
        # Slots written as a side effect of a form that cannot write fewer. Part of the program's
        # contract: a caller that puts live data at a clobbered slot loses it.
        self.reserved = sorted({m.fields["slot"] + 1 for _, _, m in layout
                                if m.form.startswith("store") and m.fields.get("n") == 2
                                and "RESERVES" in (m.note or "")})
        self.inherited = dict(inherited)     # per-FIELD dependencies inside the code itself
        # Freeze ABI inputs now. Buffer objects and layout entries remain mutable
        # for existing diagnostic tooling; ABI queries never consult them again.
        import copy
        ranks = dict(binding_ranks) if binding_ranks is not None else {
            b.slot: i for i, b in enumerate(sorted(buffers, key=lambda b: b.slot))}
        self._ranks = types.MappingProxyType(ranks)
        # the program's DECLARED threadgroup requirement (Function.declare_threadgroup), frozen with
        # the other ABI inputs; None when the program declares none
        self._abi_threadgroup = dict(threadgroup) if threadgroup else None
        # Captured from the explicit IR primitive, never inferred from a writable uchar binding or
        # from the presence of opcode 17193 alone.  The latter would make an authored ordinary
        # byte sequence indistinguishable from the measured narrowing boundary.
        self._abi_requantization = copy.deepcopy(requantization) if requantization else None
        # CAPTURED HERE, with the other ABI inputs, not read off the public layout at ABI time:
        # `layout` is mutable, and a contract for unchanged bytes must not change because someone
        # edited the layout before the first abi() call (Codex's review of the first cut, 4c3e381).
        self._abi_constant_pool = _constant_pool(layout)
        # ABI v5: the execution requirement, captured from the layout the same way. Stated ONLY by
        # a lowering that knows it - the 32x32x64 tensor lowering, whose address setup is a
        # function of the 32-lane index (the SR130 read; op17016 measured on lanes 0..31) - and
        # None for every other program (handoff 9e; the linker's amendments b846283a: no
        # simdgroups, since that is a fact of the witness's source and not of the bytes).
        self._abi_execution = _execution_requirement(layout)
        # THE IMAGEBLOCK DECLARATION, stated by the compiler that emitted the accesses. The linker
        # declares it (scanlink, agxforge.g17.imageblock) only when the ABI says so, and without it the
        # pipeline allocates no tile and every read returns 0. Nothing stated it before Set A item
        # 12. The element is the struct the accesses span: every member is a 32-bit word at a byte
        # offset, so its size is the highest member + 4 (Apple's byte 32 equals sizeof(struct) in 7 of 7).
        _ib = [m.fields["member"] for _o, _b, m in layout if m.form in ("store.ib.32", "load.ib.32")]
        # and the staging rows a tensor composition emits between bodies (ibstage), which carry
        # their member so the imageblock is declared exactly as for a selected access
        _ib += [m.fields["ib_member"] for _o, _b, m in layout
                if m.form == "tensor.wholekernel" and m.fields.get("ib_member") is not None]
        self._abi_imageblock = dict(layout="explicit", element_bytes=max(_ib) + 4) if _ib else None
        self._abi_bindings = tuple((b.slot, 2 * ranks[b.slot], b.slot in self.written,
                                   ir.ELEM_NAMES[b.elem], ir.ELEM_BYTES[b.elem])
                                  for b in sorted(buffers, key=lambda b: b.slot))
        # ABI v6: THE RESOURCE LAYOUT of a program that reads a texture, captured here with the other
        # ABI inputs. The internal records are the measured pair (TEXTURE_INTERNAL_INDICES) whose
        # ranks g17resource put ahead of the user buffers - checked here against the same `ranks`
        # the bindings' offsets came from, so the two cannot disagree - and the textures are the
        # dense indices the texture.read forms name. What the section needs beyond this and this
        # side cannot state (slot 27's contents, the slot-2 resource record) is NAMED as not stated.
        self._abi_resources = _resource_layout(layout, ranks, self._abi_bindings)
        if self._abi_resources is not None:
            # argument_bytes is stated inside the witnessed class and NAMED in not_stated above it (handoff 10af)
            _ab = _argument_bytes(self._abi_resources["access"], len(self._abi_constant_pool or ()))
            self._abi_resources = dict(self._abi_resources, argument_bytes=_ab,
                                       not_stated=tuple(self._abi_resources["not_stated"]) + (() if _ab is not None else ("argument_bytes",)))
        # ABI v7 (handoff 10aa): the uniform preloads and the CONSTANT PROGRAM that performs them. The
        # prologue stops being `end` plus filler: it reads the argument pointer, loads the word and
        # publishes it into the block at 4 x the declared records - the constant main's alu.block carries.
        self._abi_preloads = None; self._constant_program = None; self._resolved_layout = None
        if preloads:
            from agxforge.g17 import uniformpreload as g17uniformpreload
            if not resolved_layout: raise Unsupported("a preload without a resolved layout: the block constant is encoded against the linker's resolution, never assumed")
            # CAPTURED IMMUTABLY (integration's 83d189e8: an encoding input that never has to serialise is frozen,
            # not merely detected): the layout both uses were encoded against cannot be altered under the program
            self._resolved_layout = self._freeze(copy.deepcopy(dict(resolved_layout))); block = self._resolved_layout["block_bytes"]
            if block != 4 * len(ranks): raise Unsupported("the resolved layout's block (%d bytes) and this program's rank map (%d records) disagree" % (block, len(ranks)))
            consts = {m.fields["const"] for _, _, m in layout if m.form == "alu.block"}
            if consts != {block}: raise Unsupported("main's alu.block constant %s and the resolved block %d disagree: both uses must encode the same layout" % (sorted(consts), block))
            # THE ARGUMENT-TABLE ENTRY LAW IS WITNESSED FOR TWO DECLARATION SHAPES ONLY: the S-shape (a written buffer 0
            # and the terms as buffers 1..n, entries at units 8/10, 12/14, 16/18 - S1, S2, S3) and M4's (one declared
            # buffer, the term; entry 8/10). Any other declaration refuses here by name rather than guess an entry.
            declared = sorted(b.slot for b in buffers); slots = [p["slot"] for p in preloads]
            n = len(slots); s_shape = declared == list(range(n + 1)) and sorted(slots) == list(range(1, n + 1)); m4_shape = n == 1 and declared == [0] and slots == [0]
            if not (s_shape or m4_shape): raise Unsupported("preload terms %s with declared buffers %s: the argument-table entry law is witnessed for buffers 1..n beside a written buffer 0 (S1/S2/S3) and for a lone buffer 0 (M4) only" % (slots, declared))
            preloads = sorted(preloads, key=lambda p: p["slot"])
            try: self._constant_program = g17uniformpreload.constant_program_folded(n, block) if n > 1 else g17uniformpreload.constant_program(pair=105, arg_offset=8, publish_const=block)
            except g17uniformpreload.Unsupported as e: raise Unsupported("the constant program for a %d-term preload into a %d-byte block cannot be encoded: %s" % (n, block, e)) from e
            self.ENTRY = len(self._constant_program)          # S3's constant program needs the 128-byte entry
            self._abi_preloads = (dict(terms=tuple(dict(binding=p["slot"], element=p["element"]) for p in preloads), element_type="uint", lifetime="release", consumer=dict(opcode=10282, form="alu.block", block_constant=block)),)
        self._abi_semantics = copy.deepcopy(self._derive_abi_inputs())
        # The measured requantization class has a real 64-byte constant program, but it is not
        # the generic uniform-preload/resource class.  Install it only for the explicit marker and
        # validate the exact binding/system-register witness before any ABI is exposed.
        if (self._abi_requantization is not None and
                self._abi_requantization.get("scale_addressing") == "constant_program_preload"):
            from agxforge.g17 import requantpreload
            requantpreload.validate_request(
                tuple((b[0], b[1], b[2]) for b in self._abi_bindings),
                self._abi_semantics.get("system_registers"))
            self._constant_program = requantpreload.build()
        from agxforge.g17 import formops as g17formops
        self._abi_opcode_table = types.MappingProxyType(dict(g17formops.load()))
        # Snapshot only fields that select the opcode. Copying an entire MInst
        # follows its source SSA graph and is both unnecessary and unbounded.
        self._abi_selection = tuple(
            (off, len(data), m.form, types.MappingProxyType({
                key: m.fields[key] for key in ("opcode",) + g17formops.SELECTORS.get(m.form, ())
                if key in m.fields})) for off, data, m in layout)
        self._abi_instruction_cache = None
        self._abi_code = bytes(code)
        self._abi_name = name

    @property
    def _abi_instructions(self):
        """Use the owner's strict lookup on captured selection, never a None form."""
        from agxforge.g17 import formops as g17formops
        if self._abi_instruction_cache is None:
            self._abi_instruction_cache = tuple(
                (off, size, (g17formops.opcode_of(types.SimpleNamespace(form=form, fields=fields),
                                               size, self._abi_opcode_table), size))
                for off, size, form, fields in self._abi_selection)
        return self._abi_instruction_cache

    def components(self, reference=None):
        """Every part of the image, measured. Needs a reference container for the parts that are
        still inherited, because their SIZE is a fact about that container."""
        ref = reference or self.reference
        out = [Component("code", len(self.code), "GENERATED", "__TEXT,__text",
                         "%d instructions, every field authored or listed in .inherited"
                         % len(self.layout))]
        if ref is None:
            out.append(Component("(rest of the image)", 0, "INHERITED", "",
                                 "no reference container given, so nothing else can be measured"))
            return out
        from agxforge.g17 import image as g17image, machobj as machobj
        loc = machobj.locate(ref + "/s.arc.metallib", ref + "/out/object/0-0")
        img = g17image.G17Image(ref, self.name)
        try:
            probe = g17image.G17Image(ref, self.name); probe.generate_constant_program()
            head = probe.cprog_inherited_head
            out.append(Component("constant_program", img.entry - head, "GENERATED", "__TEXT,__text",
                                 "authored: the terminating `end` and the filler after it, "
                                 "byte-for-byte what this container holds, emitted not copied"))
            if head:
                out.append(Component("constant_program_head", head, "INHERITED", "__TEXT,__text",
                                     "the 8-byte buffer-binding preamble 849 of 1083 catalogued "
                                     "kernels share byte for byte; unmodelled, emitted verbatim"))
        except ValueError as ex:
            out.append(Component("constant_program", img.entry, "INHERITED", "__TEXT,__text",
                                 "cannot be authored here: %s" % str(ex)[:80]))
        tail = len(img.text) - img.entry - len(self.code)
        if tail > 0:
            out.append(Component("code_tail", tail, "INHERITED", "__TEXT,__text",
                                 "container space after the generated program; NOP-filled by "
                                 "pad_to, which is chosen padding rather than authored content"))
        for sect, (off, size) in sorted(loc["sects"].items()):
            role = self.SECTION_ROLE.get(sect)
            if role and size:
                note = "located, not yet modelled"
                if role == "descriptors":
                    note = ("a small FlatBuffers header - root +16, 4 fields, parses in "
                            "1159/1159 sections - followed by a payload ~3x its size in an "
                            "unknown format, which is where the buffer-count growth happens. "
                            "Two bytes decoded (buffer slot, highest bound slot).")
                out.append(Component(role, size, "INHERITED", sect, note))
        used = sum(c.nbytes for c in out)
        rest = len(loc["fat"]) - used
        if rest > 0:
            out.append(Component("container_framing", rest, "INHERITED", "metallib",
                                 "MTLB header, NAME/TYPE/HASH/OFFT/VERS/RLST/UUID records and "
                                 "the Mach-O wrapper around the object"))
        out.append(Component("relocations", 0, "GENERATED", "",
                             "none needed: every branch and store offset the compiler emits is "
                             "PC-relative or an immediate slot, so nothing requires fixing up"))
        return out

    def owned_fraction(self, reference=None):
        c = self.components(reference)
        tot = sum(x.nbytes for x in c)
        return (sum(x.nbytes for x in c if x.provenance == "GENERATED") / tot) if tot else 0.0

    def image_report(self, reference=None):
        c = self.components(reference)
        tot = sum(x.nbytes for x in c) or 1
        out = ["G17Program %s - image components" % self.name,
               "  %-22s %8s  %-10s %s" % ("component", "bytes", "provenance", "where")]
        for x in c:
            out.append("  %-22s %8d  %-10s %s" % (x.name, x.nbytes, x.provenance, x.where))
            if x.note: out.append("  %-22s %8s  %s" % ("", "", x.note))
        gen = sum(x.nbytes for x in c if x.provenance == "GENERATED")
        out.append("  OWNED: %d of %d bytes (%.1f%%). The mission retires the INHERITED rows one "
                   "at a time." % (gen, tot, 100.0 * gen / tot))
        return "\n".join(out)

    def _derive_abi_inputs(self):
        """The compiler-owned inputs a linker cannot derive from the bytes it is handed.

        g17authorobj.author() takes these in `abi`. It refuses without them on purpose - a
        constant measured on buffer kernels is what put a 224-byte __GPU_LD_MD into a texture
        image - so this is the answer, not a default.

            arch_flag         __GPU_ARCH_LD_MD's one boolean: True when every thread in the
                              program stands alone - no barrier, no cross-lane operation, no
                              threadgroup atomic and no threadgroup-relative coordinate
                              (see ARCH_CLEARING_SR and ARCH_CLEARING_FORMS)
            system_registers  sorted read_sr indices captured from instruction selection
            register_count    highest 32-bit register index named in _agc.main, plus one,
                              EXCLUDING the constant program/prologue - per-kernel slot 0's
                              measured law, 2,022 of 2,022 corpus objects over main alone
            uses_threadgroup  per-kernel slot 18, value 1
            has_stores        per-kernel slot 15, value 1 - the kernel writes memory. The 286
                              corpus objects without it write nothing at all, e.g.
                              atm-f-load-dev-void-1-r0, whose body is a discarded atomic load.
            writes_buffer     per-kernel slot 16, and slot 15 is `16 or 17` with no exceptions
            writes_texture    per-kernel slot 17 - always False here, this backend has no
                              texture write

        pk_values carries the slots themselves, so a caller hands it straight to g17authorobj.
        """
        srs = {m.fields["sr"] for _, _, m in self.layout
               if m.form == "read_sr.4"
               or (m.form in ("tensor.inherited", "tensor.wholekernel") and "sr" in m.fields)}
        coupled = bool(srs & ARCH_CLEARING_SR) or any(
            m.form in ARCH_CLEARING_FORMS for _, _, m in self.layout)
        tg = _uses_threadgroup_memory(self.layout)
        buf = any(m.form in BUFFER_WRITING_FORMS
                  or (m.form == "auth" and m.fields.get("opcode") == STOREI_OPCODE)
                  or (m.form == "tensor.authored" and m.fields.get("phase") == "readout")
                  # ANY tensor row carrying a tensor store writes a buffer, whichever route
                  # produced it. The phase clause above is kept so the registry path's captured
                  # facts are bit-for-bit what they were; this clause is what makes the general
                  # lowering's stores visible.
                  or (m.form.startswith("tensor.")
                      and m.fields.get("opcode") in TENSOR_STORE_OPCODES)
                  for _, _, m in self.layout)
        # This backend has no texture WRITE - it reads textures and stores the result to a buffer -
        # so slot 17 is False by construction rather than by measurement, and slot 15 is slot 16.
        tex = False
        wr = buf or tex
        pk = {}
        if wr: pk[15] = 1
        if buf: pk[16] = 1
        if tex: pk[17] = 1
        if tg: pk[18] = 1
        # THE REGISTER COUNT, delivered rather than decoded. __GPU_METADATA per-kernel slot 0
        # declares a register count, and the linker measured its law over the whole corpus: the
        # highest 32-bit register index named in _agc.main, plus one - 2,022 of 2,022 objects once
        # the count is taken over MAIN ALONE. Over main plus the constant program it was 1,942:
        # the 80 exceptions were all under-declared, never over, and that one-sidedness pointed at
        # the instrument, not the objects. So the definition is exact and it excludes the prologue,
        # which this side authors as `end` plus filler and which names no register at all.
        #
        # The class had been emitting a per-witness CONSTANT here - 5 for the four-buffer class
        # against a real 28 for the 1x4 and 99 for the 32x384. Every hardware-correct image this
        # side knows violates the law that way and every one ran, so the field is declared but
        # measured inert; stating it correctly is still owed, and stated from the compiler's own
        # allocation rather than read back off the bytes, because "which register does this
        # operand name" is instruction semantics the linker is told not to rediscover. The
        # linker's decoded value stays an INDEPENDENT check of this one, never its source.
        #
        # 32-bit file only, as the law is stated. A 16-bit value occupies a whole 32-bit register
        # here (the allocator packs nothing), so its index is the same index; half operands add
        # no register the 32-bit view does not already name.
        # `_occupies` COUNTS TOO. A row-spliced tensor body carries pre-encoded bytes with fixed
        # registers, so it has no virtual defs or uses for the allocator to colour and published
        # its register set as `_occupies` instead. Reading only _defs/_uses gave hi = -1 and a
        # declared register_count of 0 for a program naming 60 registers - which is exactly the
        # under-declaration the law above says never happens in a correct object. The definition
        # is unchanged: the highest 32-bit register index named in main, plus one. This only adds
        # the rows that name registers without the allocator assigning them.
        hi = max((r for _, _, m in self.layout
                  for r in ((m.fields.get("_defs") or []) + (m.fields.get("_uses") or [])
                            + (m.fields.get("_occupies") or []))),
                 default=-1)
        return {"arch_flag": not coupled, "system_registers": tuple(sorted(srs)),
                "register_count": hi + 1,
                "uses_threadgroup": tg, "has_stores": wr,
                "writes_buffer": buf, "writes_texture": tex,
                "pk_values": pk, "pk_extra": tuple(sorted(pk))}

    def abi_inputs(self):
        """Return a detached view of the semantic facts captured at compilation."""
        import copy
        return copy.deepcopy(self._abi_semantics)

    def contract(self):
        """The immutable v1 compiler/linker ABI, built only from captured facts."""
        import hashlib
        from agxforge.g17.abi import (Binding, Instruction, ProgramABI, ThreadgroupABI, ExecutionABI,
                                   ResourcesABI, InternalResource, TextureResource, AccessFact,
                                   CoordinatePublication, PreloadABI, PreloadTerm, PreloadConsumer,
                                   ResolvedLayoutABI, SpillState,
                                   RequantizationABI,
                                   allow_requantized_binding as abi_allow_requantized_binding)
        pre = getattr(self, "_abi_preloads", None)
        preloads = tuple(PreloadABI(terms=tuple(PreloadTerm(**t) for t in p["terms"]), element_type=p["element_type"], lifetime=p["lifetime"], consumer=PreloadConsumer(**p["consumer"])) for p in pre) if pre else None
        res = getattr(self, "_abi_resources", None)
        if pre and not res: raise Unsupported("a preload on a program without a resource layout: the preload's block is the resource block")
        resources = ResourcesABI(internal=tuple(InternalResource(**r) for r in res["internal"]), textures=tuple(TextureResource(**t) for t in res["textures"]),
                                 samplers=(), spill_bytes=0, spill_basis="no_spill_form", access=tuple(AccessFact(**a) for a in res["access"]),
                                 coordinate_publications=tuple(CoordinatePublication(**c) for c in res["coordinate_publications"]), not_stated=tuple(res["not_stated"]), argument_bytes=res.get("argument_bytes"),
                                 preloads=preloads, resolved_layout=(ResolvedLayoutABI(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in self.abi_plain(self._resolved_layout).items()}) if pre else None)) if res else None
        if self.code != self._abi_code:
            raise Unsupported("program code changed after compilation; its captured ABI is stale")
        facts = self._abi_semantics
        _requant_ranks = {m.fields.get("buf_const") for _off, _size, m in self.layout
                          if m.form == "store.byte.14"}
        _requant_indices = {b[0] for b in self._abi_bindings
                            if b[1] * 2 in _requant_ranks}
        from contextlib import nullcontext
        # A byte-store opcode is not itself the semantic permission.  The IR primitive carries the
        # measured marker; without it, even a hand-constructed selector row must remain a strict
        # declaration-only uchar binding and fail closed.
        _binding_context = (abi_allow_requantized_binding(_requant_indices)
                            if self._abi_requantization and _requant_indices else nullcontext())
        with _binding_context:
            _bindings = tuple(Binding(*b) for b in self._abi_bindings)
        return ProgramABI(version=1, name=self._abi_name,
                          code_sha256=hashlib.sha256(self._abi_code).hexdigest(),
                          code_size=len(self._abi_code), entry=self.ENTRY, prologue=self.prologue(),
                          bindings=_bindings,
                          instructions=tuple(Instruction(off, size, form[0])
                                             for off, size, form in self._abi_instructions),
                          arch_flag=facts["arch_flag"], uses_threadgroup=facts["uses_threadgroup"],
                          writes_buffer=facts["writes_buffer"], writes_texture=facts["writes_texture"],
                          exact_grid_required=True,
                          threadgroup=(ThreadgroupABI(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in self._abi_threadgroup.items()})
                                       if getattr(self, "_abi_threadgroup", None) else None),
                          constant_pool=(self._abi_constant_pool if (getattr(self, "_abi_threadgroup", None) or getattr(self, "_abi_execution", None) or res) else None),
                          execution=(ExecutionABI(**self._abi_execution) if getattr(self, "_abi_execution", None) else None),
                          resources=resources,
                          constant_program_sha256=(hashlib.sha256(self._constant_program).hexdigest()
                                                  if (pre or (self._abi_requantization and self._constant_program))
                                                  else None),
                          argument_state=self._argument_state(resources),
                          spill_state=self._spill_state(),
                          resource_projection=getattr(self, "_abi_projection", None),
                          requantization=(RequantizationABI(**self._abi_requantization)
                                           if self._abi_requantization else None))

    def _spill_state(self):
        """ABI v10: the scratch the allocator's spill needs, per thread, from the emitted stores.

        Root's requirement: "Scratch is deliberately not guessed from a buffer name or observed
        stores. The spilled delivery must supply its allocation extent through its ABI." The
        binding says where the scratch is bound; this says how much of it each thread needs, which
        is the fact a caller has to allocate from.
        """
        from agxforge.g17.abi import SpillState
        # ONLY WHAT THE ALLOCATOR INTRODUCED. A count of op17256 in the emitted code is not this
        # number: a program may author a vector store itself, and root's review caught exactly that
        # - g17spillsource.kernel(4, True) emits one ordinary store and its contract refused.
        # An ordinary store is not scratch and must not inflate the extent, so the provenance is
        # carried from the pass that created the slot rather than recovered from the bytes.
        slot, groups = getattr(self, "_spill_slot", None), getattr(self, "_spill_groups", 0)
        if slot is None or not groups:
            return None
        return SpillState(binding_index=slot, groups=groups, words_per_group=SPILL_GROUP,
                          words_per_thread=SPILL_GROUP * groups,
                          bytes_per_thread=4 * SPILL_GROUP * groups,
                          launch_index=SPILL_INDEX_BUILTIN, launch_axis=SPILL_INDEX_AXIS,
                          alignment_bytes=4 * SPILL_GROUP,
                          basis="emitted_vector_stores")

    def _argument_state(self, resources):
        """ABI v9: the argument buffer THIS COMPILER stages, read off its own binding offsets.

        The linker's six-buffer author is blocked because a device-buffer contract cannot state
        per-kernel slot 1, so the class inherits FIVE's and emits 12 - right for one Apple arm of
        seven. This states what the compiler actually emits instead: every bound buffer's pointer is
        read at word `2 * rank` and a pointer is two words, so the block is `2 * n` words.

        SLOT 1 ITSELF IS NAMED, NOT STATED. It is not a function of the binding shape - seven arms
        share this one and carry 12 or 16 - and the number this side could supply is an extent it
        stages, not a witness it measured. The distinction is the whole point of the field.

        THE TEXTURE ROUTE KEEPS ITS MEANING. It states slot 1 through resources.argument_bytes from
        a rule measured on 31 Apple members, so a contract carrying that does not carry this.
        """
        from agxforge.g17.abi import ArgumentState
        if resources is not None:
            return None
        offsets = tuple((index, offset) for index, offset, *_ in self._abi_bindings)
        if not offsets:
            return None
        indices = [index for index, _ in offsets]
        return ArgumentState(
            pointer_offsets=offsets, pointer_words=2, block_words=2 * len(offsets),
            block_bytes=8 * len(offsets), basis="emitted_pointer_offsets", offset_rule="2 * rank",
            # Apple's measured rule is 2 * (index - lowest bound index). It agrees with this one
            # exactly when the bound indices are contiguous, so the contract says which case it is
            # rather than leaving a reader to assume the program that came first.
            indices_contiguous=(indices == list(range(indices[0], indices[0] + len(indices)))),
            not_stated=("per_kernel_slot_1",))

    # THE CONSTANT PROGRAM THIS COMPILER AUTHORS. `end` and then NOP filler up to the entry -
    # byte for byte what Apple's container holds for a kernel with no constant work, emitted rather
    # than copied (see components(), which measures it as GENERATED). The linker needs the bytes
    # and the entry offset and will not infer either, so they are stated here.
    PROLOGUE_END = bytes.fromhex("0e000000")
    PROLOGUE_FILLER = bytes.fromhex("0600")
    ENTRY = 64

    def prologue(self, entry=None):
        entry = self.ENTRY if entry is None else entry
        cp = getattr(self, "_constant_program", None)
        if cp is not None:
            if len(cp) != entry: raise Unsupported("the constant program is %d bytes and the entry is %d" % (len(cp), entry))
            return cp
        if entry < len(self.PROLOGUE_END) or (entry - len(self.PROLOGUE_END)) % 2:
            raise Unsupported("entry %d cannot be reached by `end` plus two-byte filler" % entry)
        return self.PROLOGUE_END + self.PROLOGUE_FILLER * ((entry - len(self.PROLOGUE_END)) // 2)

    ABI_VERSION = 3          # programs that declare threadgroup storage carry version 4, programs executing tensor forms version 5 (see abi())

    @staticmethod
    def abi_plain(abi):
        """The frozen ABI as ordinary JSON types, for serialising it. Read-only is for CALLERS.

        The ABI is a read-only mapping so that nobody can alter one program's contract in place;
        json cannot serialise that, and the right answer is an explicit conversion at the boundary
        rather than handing out a mutable structure and hoping. bytes become hex, because that is
        how the linker reads the prologue back.
        """
        if isinstance(abi, (types.MappingProxyType, dict)):
            return {k: G17Program.abi_plain(v) for k, v in abi.items()}
        if isinstance(abi, (list, tuple)):
            return [G17Program.abi_plain(v) for v in abi]
        if isinstance(abi, (bytes, bytearray)):
            return bytes(abi).hex()
        return abi


    @staticmethod
    def _freeze(v):
        """A value no consumer can alter: mappings become read-only, sequences become tuples."""
        if isinstance(v, dict):
            return types.MappingProxyType(
                {k: G17Program._freeze(x) for k, x in sorted(v.items(), key=lambda kv: str(kv[0]))})
        if isinstance(v, (list, tuple)):
            return tuple(G17Program._freeze(x) for x in v)
        if isinstance(v, (set, frozenset)):
            return tuple(sorted(v))
        return v

    def abi(self, entry=None, profile=None):
        """THE ONE DOCUMENTED ABI between this compiler and the linker. Immutable and versioned.

        docs/archive/g17-scan-acceptance.md asks the compiler to "supply the resource, entry/prologue, and
        launch facts required by the linker through one documented ABI". This is it. Every field is
        something this side owns; nothing here is a default borrowed from a calibration population,
        and a fact this side cannot state is absent rather than guessed.

            abi_version    this structure's version, so a linker can refuse one it does not know
            profile        a LABEL for the architectural class, carried so an execution claim can
                           be attributed to one. It is not a routing key and must not select an
                           authoring path: a per-kernel short-circuit keyed on this string is the
                           thing one authoring path exists to remove
            entry          where _agc.main begins
            prologue       the constant program's bytes: `end` then filler to the entry
            bindings       per binding: index, pointer-block offset, written, element_type
                           - offset is 2*rank and rank comes from g17resource when it is available
            forms          every (opcode, length) the program emits, so a profile that restricts
                           the instruction set can REFUSE a widened one instead of shipping it
            launch         exact_grid_required stays true until the compiler supplies an actual
                           index-bound proof; merely finding a compare is not such a proof
            main_instruction_count
                           how many instructions _agc.main holds, MAIN ONLY - the constant program
                           is `prologue` and is not counted. Additive at v3 and optional in the
                           consumer model, exactly as register_count is: retained contracts predate
                           it and must still load
            arch_flag,     the compiler-owned launch booleans, unchanged from abi_inputs()
            uses_threadgroup, has_stores, writes_buffer, writes_texture, pk_values, pk_extra

        WHAT IS DELIBERATELY ABSENT, AND constant_pool_emptiness IS NOW ONE OF THEM. The linker
        asked for a field saying which shape an empty constant pool takes - a zero-length slot-13
        vector or an eight-byte zero vector. It is not a compiler fact, and their own authoring
        code is the evidence: g17authorobj's `unswept_two_buffer` class writes "slot 13 vector
        length 0" for a constant-free program, while the narrow texture class needs the eight-byte
        zero form. SAME PROGRAM, TWO CLASSES, TWO SHAPES - so the program does not determine it and
        this structure describes the program. What this side owns is `constant_pool`, already
        carried: `()` means there are no constants. Which empty representation a metadata class
        uses is a class constant, and belongs where the class census measured it.

        WHAT IS ALSO DELIBERATELY ABSENT: ld_md_slots, ld_md_values and per-kernel slot 1. This side has
        no measurement that determines them - slot 1 is a small integer that is not the binding
        count (1.8%), not register usage (0%) and not the program's position - so they are the
        linker's to supply or to refuse, and inventing them here is exactly the borrowed default
        the acceptance document forbids.
        """
        # BOTH ABI VIEWS REJECT CHANGED CODE, not just contract(). Freezing the contract stops a
        # CONSUMER altering it and says nothing about the PROGRAM moving underneath it; the cache
        # then answers for bytes that no longer exist, which looks authoritative and is false.
        if self.code != self._abi_code:
            raise Unsupported(
                "this program's code changed after it was compiled (%d bytes now, %d when its ABI "
                "was captured), so any contract derived from it describes bytes that no longer "
                "exist. Compile again rather than re-reading a frozen answer"
                % (len(self.code), len(self._abi_code)))
        key = (entry, profile)
        if key not in self._abi_cache:
            self._abi_cache[key] = self._build_abi(entry, profile)
        return self._abi_cache[key]

    def _build_abi(self, entry, profile):
        import hashlib
        facts = dict(self.abi_inputs())
        bindings = [dict(zip(("index", "offset", "written", "element_type", "element_bytes"), b))
                    for b in self._abi_bindings]
        forms = sorted({form for _, _, form in self._abi_instructions})
        # A branch's presence is not a proof that it guards a buffer index.
        # No index-bound proof is currently emitted by this compiler.
        guarded = False
        facts.update(abi_version=self.ABI_VERSION, profile=profile,
                     entry=self.ENTRY if entry is None else entry,
                     prologue=self.prologue(entry), bindings=bindings, forms=forms,
                     # MAIN'S INSTRUCTION COUNT, which a metadata class needs and was otherwise
                     # derived by its consumer from bytes it does not own. It is len(layout) and
                     # not a decode: this side selected those instructions, so counting them here
                     # costs nothing and stops a consumer resolving instruction boundaries to get
                     # a number this side already has. MAIN ONLY - the constant program is the
                     # `prologue` field above and is not in self.code - which is the same scope
                     # register_count uses, and the scope is the whole difficulty: a count taken
                     # over __text including the 64-byte prologue is a different quantity with
                     # the same name, and reading one as the other produced four false
                     # counterexamples on the linker's side.
                     main_instruction_count=len(self._abi_selection),
                     launch={"bounds_checked": guarded,
                             "exact_grid_required": not guarded})
        if self._abi_requantization is not None:
            facts["requantization"] = dict(self._abi_requantization)
        # ABI v4 = v3 plus the threadgroup block, carried ONLY by a program that uses threadgroup
        # memory and declared it (Function.declare_threadgroup). Every v3 image stays v3, byte for
        # byte; the block is the agreed shape of docs/archive/g17-cooperative-integration-feedback.md.
        tg = getattr(self, "_abi_threadgroup", None)
        if facts["uses_threadgroup"]:
            if tg is None:
                raise Unsupported("uses_threadgroup without a declaration; the compiler refuses this at compile")
            # ABI v4 ALSO states the constant pool (slot 13) explicitly, because the linker's
            # measured cooperative class has an empty pool and an empty pool is right only for a
            # program with no external constants - which is a fact about the program, so the
            # program says it. v3 stays byte-identical: the key is carried only by v4.
            facts.update(abi_version=4, threadgroup=dict(tg), constant_pool=self._abi_constant_pool)
        # ABI v5 = the execution requirement, carried only by a program that executes tensor forms
        # (g17abi.ExecutionABI; refused the other way round too). Every other image keeps its version.
        ex = getattr(self, "_abi_execution", None)
        if ex is not None:
            # v5 states its constant pool as v4 does: captured at construction from the layout,
            # an explicit empty tuple for the accepted tensor programs (none reads slot 13)
            facts.update(abi_version=5, execution=dict(ex), constant_pool=self._abi_constant_pool)
        if getattr(self, "_abi_imageblock", None) is not None:
            facts["imageblock"] = dict(self._abi_imageblock)
        # ABI v6 = the resource layout, carried only by a program that emits texture forms (handoff
        # 10s; the linker's answers d4e77d79/ec67258b). States its pool and its spill as the section
        # does, and names the section facts this side does not state.
        res = getattr(self, "_abi_resources", None)
        if res is not None:
            facts.update(abi_version=6, resources=dict(res), constant_pool=self._abi_constant_pool, spill_bytes=0)
        # ABI v7 = the preloads and the constant program's identity, carried only by a program with a preload
        pre = getattr(self, "_abi_preloads", None)
        if pre:
            from agxforge.g17 import uniformpreload as g17uniformpreload
            cp = self._constant_program
            # the preloads and the layout sit INSIDE resources: that is where the linker's plan reads them. The constant
            # program's forms are stated BY TERM COUNT (S2's carry op10306/12 and op12688/8, S3's op10282/12 too; a
            # first cut declared S1's forms for every member - integration's 50e750df caught it with the delivered-
            # declarations check, and test_g17preloadfold now decodes the serialised prefix against this list)
            facts.update(abi_version=7, constant_program=dict(sha256=hashlib.sha256(cp).hexdigest(), bytes=cp, forms=list(g17uniformpreload.CONSTANT_PROGRAM_FORMS_BY_TERMS[len(pre[0]["terms"])])),
                         resources=dict(facts["resources"], preloads=[dict(p) for p in pre], resolved_layout=self.abi_plain(self._resolved_layout)))
        elif self._abi_requantization is not None and self._constant_program is not None:
            facts["constant_program_sha256"] = hashlib.sha256(self._constant_program).hexdigest()
        return self._freeze(facts)

    def to_image(self, reference, at, pad_to=None, own_constant_program=True):
        """Install this program in a reference container and return the image.

        own_constant_program authors the constant program instead of inheriting it. It is ON by
        default: the compiler should emit every byte it knows how to emit, and a container whose
        constant program is not the trivial form raises rather than being silently inherited.
        """
        from agxforge.g17 import image as g17image
        img = g17image.G17Image(reference, self.name)
        if own_constant_program: img.generate_constant_program()
        img.place(self.code, at)
        if pad_to is not None: img.pad_to(pad_to)
        return img

    def __repr__(self):
        s = ["G17Program %s: %d bytes, %d instructions" % (self.name, len(self.code), len(self.layout))]
        for off, b, m in self.layout:
            s.append("  +0x%03x  %-42s %s" % (off, b.hex(" "), m.form))
        if self.reserved:
            s.append("  RESERVED slots, set to 0 (this store form has no single-slot encoding): %s"
                     % ", ".join(str(c) for c in self.reserved))
        s.append("  INHERITED FIELDS inside the code: " + (", ".join(sorted(self.inherited)) or "none"))
        return "\n".join(s)

# --- verification: decode every emitted instruction back ----------------------------------
_DEC = {"alu.12": "decode_alu", "alu.block": "decode_alu_block", "load.vec4.8": "decode_vec4", "load.vec4.14": "decode_vec4", "load.vec2.8": "decode_vec4", "load.vec2.14": "decode_vec4", "load.vec3.8": "decode_vec4", "load.vec3.14": "decode_vec4", "load.vec2h.14": "decode_vec4", "store.vec4.8": "decode_vec4", "movimm.8": "decode_movimm", "read_sr.4": "decode_sr",
        "load.8": "decode_load", "load.10": "decode_load", "load.14": "decode_load", "store.8": "decode_store", "store.14": "decode_store", "store.halfvec.14": "decode_store",
        "store.byte.14": "decode_element_store",
        "store.half1.8": "decode_element_store", "store.half1.10": "decode_element_store", "store.half1.14": "decode_element_store",
        "store.elem1.8": "decode_element_store", "store.elem1.10": "decode_element_store", "store.elem1.14": "decode_element_store",
        "cmp.6": None,     # decoded by its two halves in selfcheck
        "branch.cond.fwd": "decode_branch",
        "branch.cond.back": "decode_branch"}

# FORMS SELFCHECK VERIFIES WITH AN INLINE BRANCH rather than through a decoder in _DEC. Without
# this the scorecard counted them as "not reached" - a check that runs and a check that is counted
# are two different things, and the second is the one that tells you whether coverage is real.
_INLINE_CHECKED = {"barrier", "store.ib.32", "load.ib.32",
                   "atomic.add.10", "atomic.add.12",
                   "atomic.uniform.10", "simd.broadcast.10",
                   "atomic.tg.uniform.12"}

_CONSTRUCTED = {
    "end":          lambda m: g17cf.encode_end(),
    "exec.restore": lambda m: g17cf.encode_exec("pop"),
    # exec.mask carries a flag index the constructor does not take, so it is checked against the
    # base word with any flag: what must hold is that the rest of the instruction is constructed.
    "exec.mask":    lambda m: None,
}


def selfcheck(layout):
    """Decode each emitted instruction and confirm it says what selection asked for.

    This is the byte-exact round-trip discipline applied to GENERATED code rather than to corpus
    code. It is the difference between "the encoder ran without raising" and "the bytes mean what
    the IR said", and it is cheap enough to run on every compile - so it does, and compile_function
    raises rather than returning a program that failed it.
    """
    bad = []
    for off, b, m in layout:
        if m.fields.get("raw") is not None:
            if not m.fields.get("requant_stage"):
                bad.append("raw instruction bytes are not marked as the measured requantization stage")
                continue
            want = bytes(m.fields["raw"])
            if bytes(b) != want:
                bad.append("measured raw %s at +0x%x changed from %s to %s"
                           % (m.form, off, want.hex(), bytes(b).hex()))
            continue
        # FORMS BUILT WHOLE BY A CONSTRUCTOR ARE CHECKED BY RE-RUNNING IT. They have no decoder,
        # so the loop below skipped them silently and 35 of 177 emitted instructions were never
        # verified at all. Re-running the constructor and requiring the bytes to match is the same
        # both-directions discipline, available wherever the form takes no template.
        # FORMS WHOSE DECODER TAKES THE OPCODE. decode_alu_form and decode_bitwise_imm need it to
        # know the slot roles, so they cannot go in _DEC, which maps a form to a one-argument
        # decoder. They were therefore never checked - alu.mul.imm, alu.sub.imm and bitwise.imm,
        # emitted and unverified.
        if m.form in ("float.unary", "unary"):
            # THE SAME SILENT SKIP, TWO FORMS FURTHER ON. Widening the ladder to the unary families
            # - floor, ceil, trunc, rint, recip, rsqrt, exp2, log2, not, msb, reverse - put eleven
            # instructions into the output that no decoder was reached for, and they went by
            # unchecked exactly as the generic form did. Both have decoders; both take the OPCODE,
            # which is why they could not go in _DEC and so went nowhere.
            f2 = dict(m.fields); opc = f2.get("opcode")
            d2 = f2.pop("_defs", []); u2 = f2.pop("_uses", [])
            try:
                got2 = (g17asm.decode_trans if m.form == "float.unary"
                        else g17asm.decode_unary)(b, opc)
            except Exception as e:
                bad.append("%s op%s at +0x%x: cannot decode (%s)"
                           % (m.form, opc, off, str(e)[:40])); continue
            if d2 and got2.get("dest") != d2[0]:
                bad.append("%s op%s at +0x%x: dest decoded %s, selection asked %s"
                           % (m.form, opc, off, got2.get("dest"), d2[0]))
            if u2 and got2.get("src") != u2[0]:
                bad.append("%s op%s at +0x%x: src decoded %s, selection asked %s"
                           % (m.form, opc, off, got2.get("src"), u2[0]))
            continue
        if m.form == "bitwise.reg.10":
            # THE TEN-BYTE FORM GETS ITS OWN ARM RATHER THAN JOINING THE LINE BELOW, because it has
            # a different field map and decode_bitwise_reg would read the four-byte one and report
            # plausible wrong registers. It also has to be HERE at all: this function skips forms it
            # has no decoder for SILENTLY, which is how 35 of 177 instructions once went unverified,
            # and a newly added emitted form is exactly the case that walks into it.
            f2 = dict(m.fields); opc = f2.get("opcode")
            d2 = f2.pop("_defs", []); u2 = f2.pop("_uses", [])
            try:
                got2 = g17asm.decode_bitwise_reg10(b)
            except Exception as exc:
                bad.append("%s at +0x%x: cannot decode (%s)" % (m.form, off, str(exc)[:40]))
                continue
            # THE OPCODE IS CHECKED FROM THE SOURCE'S OWN FIELD MAP, NOT BY FORKING A DECODER.
            # This arm used to call g17auth._decode_many, which runs Apple's disassembler in a
            # subprocess - so a compile under the build audit raised "build attempted an external
            # process" and no audited ten-byte bitwise could be authored at all. The question it
            # was asking is answerable here: the opcode lives in the bits no operand field owns,
            # so requiring those bits to be the registered template's catches a wrong template and
            # an encoder that clobbered a bit it does not own. Whether the TABLE is labelled right
            # is a different question with a different instrument, and it is asked post-build by
            # test_g17bw424lower.test_the_opcode_is_read_back_from_the_bytes.
            if len(b) != 10:
                bad.append("%s at +0x%x: emitted %d bytes, selection asked 10"
                           % (m.form, off, len(b)))
            else:
                moved = g17asm.bitwise_reg10_departures(opc, bytes(b))
                if moved:
                    bad.append("%s at +0x%x: op%s's template does not explain bit%s %s; the "
                               "emitted bytes are not that form"
                               % (m.form, off, opc, "" if len(moved) == 1 else "s",
                                  ", ".join("b%d[%d]" % p for p in moved[:6])))
            # BW_IMM carries 2*b, so an odd field would floor in the decoder and read back as a
            # register the selection never asked for.
            if g17asm._slot_get(bytearray(b), g17asm.BW_IMM) & 1:
                bad.append("%s at +0x%x: BW_IMM is odd, so source B does not round-trip"
                           % (m.form, off))
            for lbl, want in (("dest", d2[0] if d2 else None), ("a", u2[0] if u2 else None),
                              ("b", u2[1] if len(u2) > 1 else None)):
                if want is not None and got2.get(lbl) != want:
                    bad.append("%s at +0x%x: %s decoded %s, selection asked %s"
                               % (m.form, off, lbl, got2.get(lbl), want))
            continue
        if m.form in ("alu.mul.imm", "alu.sub.imm", "alu.mul.reg", "alu.sub.reg",
                      "alu.shift.imm", "alu.shift.reg", "alu.sat", "bitwise.imm", "bitwise.reg"):
            f2 = dict(m.fields); opc = f2.get("opcode")
            d2 = f2.pop("_defs", []); u2 = f2.pop("_uses", [])
            try:
                # bitwise.reg HAS ITS OWN DECODER AND WAS NEVER GIVEN IT. The four-byte
                # register-register form is modelled - BITWISE_REG_FORM covers op424, op13575 and
                # op17771, with encode_bitwise_reg and decode_bitwise_reg either side - but this
                # routed it to decode_alu_form, which has no entry for those opcodes and raises.
                # So every `x & y` between two registers failed its round trip and the compiler
                # refused to emit the plainest bitwise operation there is.
                fn = (g17asm.decode_bitwise_imm if m.form == "bitwise.imm"
                      else g17asm.decode_bitwise_reg if m.form == "bitwise.reg"
                      else g17asm.decode_alu_form)
                got2 = fn(opc, b)
            except Exception as e:
                bad.append("%s at +0x%x: cannot decode (%s)" % (m.form, off, str(e)[:40])); continue
            if d2 and got2.get("dest") != d2[0]:
                bad.append("%s at +0x%x: dest decoded %s, selection asked %s"
                           % (m.form, off, got2.get("dest"), d2[0]))
            if m.form == "bitwise.reg":
                # both sources are checked here, which decode_alu_form's shape never allowed
                for lbl, want in (("a", u2[0] if u2 else None), ("b", u2[1] if len(u2) > 1 else None)):
                    if want is not None and got2.get(lbl) != want:
                        bad.append("%s at +0x%x: src %s decoded %s, selection asked %s"
                                   % (m.form, off, lbl, got2.get(lbl), want))
                if got2.get("op") != {424: "and", 13575: "or", 17771: "xor"}.get(opc):
                    bad.append("%s at +0x%x: the bytes say %r, selection asked op%d"
                               % (m.form, off, got2.get("op"), opc))
            if "imm" in f2 and "imm" in got2 and got2["imm"] != f2["imm"]:
                bad.append("%s at +0x%x: imm decoded %s, selection asked %s"
                           % (m.form, off, got2["imm"], f2["imm"]))
            continue
        if m.form == "auth":
            # THE GENERIC PATH WAS THE UNCHECKED PATH. Seven of the compiler's forms - the indexed
            # store, the float add-immediate, madd, icmp, csel and both threadgroup accesses - are
            # authored from the field map with no hand-written encoder, so `_DEC` has no entry and
            # the loop below skipped every one of them in silence. Widening the program set is what
            # exposed it: nineteen rungs reached the generic form three times, twenty-nine reach it
            # ten, and the check that says "the bytes mean what selection asked" was running on
            # none of them.
            #
            # The map that wrote the bits reads them back, which is the same both-directions
            # discipline the templated forms get. Two things are checked and the second is the one
            # that matters: that each operand decodes to the register selection allocated, AND that
            # every bit of that operand lies INSIDE the instruction's own length. encode() works on
            # a 16-byte buffer and the caller's template decides how much of it is emitted, so an
            # operand whose bits sit past that length is written and then truncated away - the
            # register is silently the witness's, not the one asked for.
            f2 = dict(m.fields)
            d2 = f2.pop("_defs", []); u2 = f2.pop("_uses", [])
            opc = f2.get("opcode")
            if opc == 11452 and len(b) == 10 and f2.get("encoder") is None:
                from agxforge.g17 import predicateform as g17predicateform
                try: g17predicateform.check(b, d2, u2, f2.get("imms") or {})
                except ValueError as error: bad.append("auth op11452/10 at +0x%x: %s" % (off, error))
                continue
            if f2.get("requant_fmul"):
                from agxforge.g17 import requantenc
                try:
                    got = requantenc.decode_fmul6(bytes(b))
                    if len(d2) != 1 or len(u2) != 1 or d2[0] != u2[0]:
                        bad.append("auth op3290/6 at +0x%x: selection is not a tied one-source row" % off)
                    elif got["register"] != d2[0] + 105:
                        bad.append("auth op3290/6 at +0x%x: decoded decoder register %s, selection asked %s"
                                   % (off, got["register"], d2[0] + 105))
                except Exception as error:
                    bad.append("auth op3290/6 at +0x%x: measured requant codec rejected bytes (%s)"
                               % (off, str(error)[:80]))
                continue
            if f2.get("and16") is not None:
                # CERTIFIED BY APPLE'S DECODER: the opcode, the destination, the half-register source and
                # the mask (immediate, or the mask register's L half) must read back as selected
                from agxforge.g17 import model as _model
                a = f2["and16"]
                try:
                    (ins,) = list(_model.decode(bytes(b), 0))
                    vals = ins.values
                    want_src = 2 * u2[0] + (1 if a["half"] == "H" else 0)
                    ok = (ins.opcode.id == opc and ins.size == 10 and vals[0] == ("reg", 105 + d2[0]) and
                          vals[2] == ("reg", (281 if a["half"] == "H" else 425) + u2[0]))
                    if opc == 426:
                        ok = ok and vals[4] == ("imm", a["mask"])
                    elif a.get("uniform") is not None:
                        # the pool form: operand 4 names the uniform the mask is preloaded into, read back by
                        # the table AND by the byte law Apple's own op425/op428 instances follow (MM 25.141.16:
                        # uniform = byte 9 << 2 | byte 8 >> 6, uniform type 0xa1 in byte 6)
                        from agxforge.g17 import auth as _A
                        bb = bytes(b)
                        ok = (ok and _A.decode(opc, bb).get(4) == a["uniform"]
                              and ((bb[9] << 2) | (bb[8] >> 6)) == a["uniform"] and bb[6] == 0xA1)
                    else:
                        from agxforge.g17 import auth as _A
                        ok = ok and _A.decode(opc, bytes(b)).get(4) == 2 * u2[1]
                    if not ok:
                        bad.append("and16 op%d at +0x%x: decodes as %s, selection asked dest R%d src R%d%s mask 0x%x"
                                   % (opc, off, str(ins)[30:110], d2[0], u2[0], a["half"], a["mask"]))
                except Exception as error:
                    bad.append("and16 op%d at +0x%x: %s" % (opc, off, str(error)[:120]))
                continue
            if f2.get("encoder") is not None:
                # A caller-supplied encoder means a layout the table does not describe; decoding it
                # with the table's map would check the wrong bits. Named, not skipped.
                bad.append("auth op%s at +0x%x: authored by a caller's encoder, which the field "
                           "map cannot certify" % (opc, off))
                continue
            try:
                dsts, srcs = g17auth.register_operands(opc)
                srcs = list(f2.get("srcmap") or srcs)
                fm = g17auth.fields(opc)
                got = g17auth.decode(opc, b)
            except Exception as e:
                bad.append("auth op%s at +0x%x: cannot decode (%s)" % (opc, off, str(e)[:60]))
                continue
            want = [(i, g17auth.field_value(opc, i, r), "r%d" % r)
                    for i, r in list(zip(dsts, d2)) + list(zip(srcs, u2))]
            want += [(i, v, "imm %d" % v) for i, v in sorted((f2.get("imms") or {}).items())]
            # THE LOAD-WAIT IS A BIT OF AN OPERAND THE SELECTION ALSO NAMES. It is applied to
            # byte0[3] after the operand writes, so the decoder reads the modifier back as
            # `asked | that operand's bit` - 32 becomes 2147483680 on op10283 - and comparing
            # against the selection's raw immediate called root's own executed encoding broken.
            # Which operand and which of its bits live at byte0[3] is read from the SAME field map
            # the encoder wrote through, not assumed to be bit 31 of operand 1.
            wait_bit = {}
            if f2.get("load_wait"):
                for i, spec in fm.items():
                    for j, by, bi, _inv in spec[1]:
                        if by == 0 and bi == 3:
                            wait_bit[i] = j
                if not b[0] & 0x08:
                    bad.append("auth op%s at +0x%x: selection asked for the load wait and byte0[3] "
                               "is clear - the instruction does not wait" % (opc, off))
            for i, v, what in want:
                if i in wait_bit:
                    v |= 1 << wait_bit[i]
                    what = "%s with the load wait" % what
                if i not in fm:
                    bad.append("auth op%s at +0x%x: operand %d has no field map" % (opc, off, i))
                    continue
                # A BIT PAST THE END ONLY MATTERS WHEN IT IS SET. op17229's index operand carries
                # its bit 8 in byte 8 and its template is eight bytes long, so every allocation
                # this backend makes truncates that bit - harmlessly, because 2*r has bit 8 clear
                # for every register below 128. Reporting the position alone called a correct
                # instruction broken. What is a defect is a SET bit falling off the end: the field
                # then names a different register than the one asked for, silently.
                lost = sorted(j for j, by, _bi, _inv in fm[i][1] if by >= len(b) and (v >> j) & 1)
                if lost:
                    bad.append("auth op%s at +0x%x: operand %d (%s) needs value bit%s %s, which "
                               "the map puts past the %d bytes emitted - the field names a "
                               "different register"
                               % (opc, off, i, what, "s" if len(lost) > 1 else "",
                                  ",".join(str(x) for x in lost), len(b)))
                elif got.get(i) != v:
                    bad.append("auth op%s at +0x%x: operand %d decoded %s, selection asked %s (%s)"
                               % (opc, off, i, got.get(i), v, what))
            continue
        if m.form == "barrier" and m.fields.get("scope") == "fence_device":
            from . import model as _model
            _d = list(_model.decode(bytes(b), 0))[0]
            if "op14156" not in str(_d.opcode) or list(_d.values)[:2] != [("imm", 0), ("imm", 186)]:
                bad.append("fence at +0x%x decodes as %s" % (off, _d))
            continue
        if m.form == "barrier":
            got2 = g17asm.decode_barrier(b)
            if got2.get("scope") != m.fields.get("scope"):
                bad.append("barrier at +0x%x: scope decoded %r, selection asked %r"
                           % (off, got2.get("scope"), m.fields.get("scope")))
            continue
        # THE INSTRUCTION'S OWN FIELDS, AND ITS OWN REGISTERS. Three names were leaking here.
        #
        # `want`, `defs` and `uses` are all read by the mov branches below and none of them was
        # bound for those branches: two earlier loops in this function use `for lbl, want in (...)`,
        # and `defs`/`uses` are bound only in the branches AFTER these - so all three held whatever
        # a previous iteration left behind. Python leaks loop variables, and the mov branches were
        # reading that leak. It stayed invisible while nothing emitted a mov.half.4 in a program
        # that also carried one of those other forms; truncation and the sixteen-bit widening both
        # do. Bound here the way every branch below binds them, from a COPY so popping the private
        # keys cannot mutate the instruction.
        want = dict(m.fields)
        defs = want.pop("_defs", []); uses = want.pop("_uses", [])
        if m.form == "mov.word.4":
            got = g17asm.decode_movword(b); exp = dict(dest=defs[0], src=uses[0], keep_src=want.get("keep_src", True))
        elif m.form == "mov.half.4":
            got = g17asm.decode_movhalf(b); exp = dict(dest=defs[0], src=uses[0], keep_src=want.get("keep_src", True),
                                                       dest_281=bool(want.get("dest_281", False)), src_281=False)
        elif m.form == "mov.4":
            # BOTH REGISTERS AND THE LIFETIME. Checking only the registers would pass a move that
            # releases the value it was inserted to preserve, which is the one way this form can
            # be wrong without being wrong-looking.
            toks = _decode_tokens(bytes(b))
            if not toks:
                bad.append("mov.4 at +0x%x: emitted bytes do not decode" % off)
                continue
            regs = [int(t[4:]) for t in toks if t.startswith("reg:")]
            imms = [int(t[4:]) for t in toks if t.startswith("imm:")]
            want = dict(m.fields)
            defs = want.pop("_defs", []); uses = want.pop("_uses", [])
            if regs[:2] != [defs[0] + 105, uses[0] + 105]:
                bad.append("mov.4 at +0x%x: registers decoded %s, selection asked [%d, %d]"
                           % (off, regs[:2], defs[0] + 105, uses[0] + 105))
            keep = MOV_KEEP if want.get("keep_src", True) else MOV_RELEASE
            if len(imms) < 2 or imms[1] != keep:
                bad.append("mov.4 at +0x%x: source lifetime decoded %s, selection asked %d"
                           % (off, imms[1:2], keep))
            continue
        if m.form == "alu.fadd.6":
            from agxforge.g17.formenc import Fadd6 as g17fadd6
            want = dict(m.fields)
            defs = want.pop("_defs", []); uses = want.pop("_uses", [])
            got = g17fadd6.decode(bytes(b))
            exp = dict(dest=defs[0], src0=uses[0], src1=uses[1], index=0,
                       dest_life=want.get("dest_life", 32),
                       src0_life=want.get("src0_life", 16),
                       src1_life=want.get("src1_life", 16), residue=g17fadd6.BASE)
            for key, value in exp.items():
                if got.get(key) != value:
                    bad.append("alu.fadd.6 at +0x%x: %s decoded %r, selection asked %r"
                               % (off, key, got.get(key), value))
            continue
        if m.form == "alu.fadd.4":
            from agxforge.g17.formenc import Fadd4 as g17fadd4
            want = dict(m.fields)
            defs = want.pop("_defs", []); uses = want.pop("_uses", [])
            got = g17fadd4.decode(bytes(b))
            exp = dict(dest=defs[0], src0=uses[0], src1=uses[1],
                       dest_life=want.get("dest_life", 32),
                       src0_life=want.get("src0_life", 16),
                       src1_life=want.get("src1_life", 16), residue=g17fadd4.BASE)
            for key, value in exp.items():
                if got.get(key) != value:
                    bad.append("alu.fadd.4 at +0x%x: %s decoded %r, selection asked %r"
                               % (off, key, got.get(key), value))
            continue
        if m.form == "alu.ffma.6":
            from agxforge.g17.formenc import Ffma6 as g17ffma6
            want = dict(m.fields)
            defs = want.pop("_defs", []); uses = want.pop("_uses", [])
            got = g17ffma6.decode(bytes(b))
            exp = dict(dest=defs[0], src0=uses[0], src1=uses[1], src2=uses[2],
                       dest_life=want.get("dest_life", 32),
                       src0_life=want.get("src0_life", 16),
                       src1_life=want.get("src1_life", 16),
                       src2_life=want.get("src2_life", 16), residue=g17ffma6.BASE)
            for key, value in exp.items():
                if got.get(key) != value:
                    bad.append("alu.ffma.6 at +0x%x: %s decoded %r, selection asked %r"
                               % (off, key, got.get(key), value))
            continue
        if m.form == "alu.ffma.4":
            from agxforge.g17.formenc import Ffma4 as g17ffma4
            want = dict(m.fields)
            defs = want.pop("_defs", []); uses = want.pop("_uses", [])
            got = g17ffma4.decode(bytes(b))
            exp = dict(dest=defs[0], src0=uses[1], other=uses[2],
                       accumulator_printed_at_6=True,
                       dest_life=want.get("dest_life", 32),
                       src0_life=want.get("src0_life", 16),
                       other_life=want.get("other_life", 16), residue=g17ffma4.BASE)
            for key, value in exp.items():
                if got.get(key) != value:
                    bad.append("alu.ffma.4 at +0x%x: %s decoded %r, selection asked %r"
                               % (off, key, got.get(key), value))
            continue
        if m.form == "alu.fmul.4":
            # READ BACK THROUGH THE LOCATED FIELDS, INCLUDING THE RESIDUE. An instruction whose
            # non-field bits are not Apple's base is not this form however its operands read -
            # 2,675 of 2,675 corpus instances share one residue, so a different one means a
            # different instruction wearing the same operands.
            #
            # This block extracts its own defs/uses rather than joining the decoder-table chain
            # below, and the first version of it did not: placed among the `auth` branches, it
            # read `defs` and `uses` from the PREVIOUS iteration, because those names are only
            # bound further down. It raised IndexError on a two-source instruction whose uses were
            # [7, 8] - a check that looked correct and was reading another instruction's operands.
            from agxforge.g17.formenc import Fmul4 as g17fmul4
            want = dict(m.fields)
            defs = want.pop("_defs", []); uses = want.pop("_uses", [])
            got = g17fmul4.decode(bytes(b))
            exp = dict(dest=defs[0], src0=uses[0], src1=uses[1],
                       dest_life=want.get("dest_life", 32), src0_life=want.get("src0_life", 16),
                       src1_life=want.get("src1_life", 16), residue=g17fmul4.BASE)
            for k, v in exp.items():
                if got.get(k) != v:
                    bad.append("alu.fmul.4 at +0x%x: %s decoded %r, selection asked %r"
                               % (off, k, got.get(k), v))
            continue
        if m.form == "atomic.tg.uniform.12":
            # READ BACK THROUGH APPLE'S DECODER, not through render - render declines to offer a
            # mnemonic for this form even though the assembler can build it, and a check that
            # depends on my own renderer agreeing with my own assembler is weaker anyway. The
            # decoder is the independent reader. What is checked is the pair that decides what
            # this instruction writes: the destination and the value register.
            toks = _decode_tokens(bytes(b))
            if not toks:
                bad.append("%s at +0x%x: emitted bytes do not decode" % (m.form, off))
                continue
            want = dict(m.fields)
            defs = want.pop("_defs", []); uses = want.pop("_uses", [])
            regs = [int(t[4:]) for t in toks if t.startswith("reg:")]
            if regs[:2] != [defs[0] + 105, uses[0] + 105]:
                bad.append("%s at +0x%x: registers decoded %s, selection asked [%d, %d]"
                           % (m.form, off, regs[:2], defs[0] + 105, uses[0] + 105))
            continue
        if m.form in ("atomic.uniform.10", "simd.broadcast.10"):
            # THE LIFETIME IS WHAT THIS CHECKS FOR. op6 at 16 makes the uniform atomic author
            # byte-exactly and write nothing, so the field that voids the instruction is the field
            # read back here - not just the registers.
            import re as _re
            opc = 10094 if m.form == "atomic.uniform.10" else 14157
            txt = [x.strip() for x in
                   g17as.render(bytes(b), [(0, len(b), opc)]).splitlines()
                   if x.strip() and not x.startswith(".")]
            if not txt:
                bad.append("%s at +0x%x: emitted bytes do not render" % (m.form, off))
                continue
            want = dict(m.fields)
            defs = want.pop("_defs", []); uses = want.pop("_uses", [])
            head = txt[0].split(" / ")[0]
            regs = [int(x) for x in _re.findall(r"\br(\d+)\b", head)]
            mods = dict((int(a), int(v)) for a, v in _re.findall(r"op(\d+)=#(-?\d+)", head))
            if regs[:1] != [defs[0] + 105]:
                bad.append("%s at +0x%x: destination decoded r%s, selection asked r%d"
                           % (m.form, off, regs[0] if regs else None, defs[0] + 105))
            elif regs[-1] != uses[0] + 105:
                bad.append("%s at +0x%x: source decoded r%s, selection asked r%d"
                           % (m.form, off, regs[-1], uses[0] + 105))
            elif m.form == "atomic.uniform.10" and not want.get("slot6") and mods.get(6, 16) != 0:
                bad.append("%s at +0x%x: operand 6 decoded %s - a source lifetime of 16 authors "
                           "byte-exactly and writes nothing"
                           % (m.form, off, mods.get(6)))
            elif m.form == "simd.broadcast.10" and 3 in mods and mods[3] != (32 if want.get("keep_src") else 16):
                bad.append("%s at +0x%x: source lifetime (operand 3) decoded %s, liveness asked %s"
                           % (m.form, off, mods[3], 32 if want.get("keep_src") else 16))
            continue
        if m.form in ("atomic.add.10", "atomic.add.12"):
            # THE ATOMIC HAD NO SELFCHECK AT ALL, which the scorecard's coverage row is what
            # noticed - it was emitted, dispatched, and shipped as an end-to-end kernel without
            # anything ever decoding it back to see whether the bytes say what selection asked.
            # Execution proves the answer is right for the values tested; it does not prove the
            # operation field says `add` rather than something that happens to agree on them.
            import re as _re
            txt = [x.strip() for x in
                   g17as.render(bytes(b), [(0, len(b), 10090)]).splitlines() if "@10090" in x]
            if not txt:
                bad.append("%s at +0x%x: emitted bytes do not render, so nothing read them back"
                           % (m.form, off))
                continue
            want = dict(m.fields)
            defs = want.pop("_defs", []); uses = want.pop("_uses", [])
            head = txt[0].split(" / ")[0]
            regs = [int(x) for x in _re.findall(r"\br(\d+)\b", head)]
            aop = _re.search(r"aop=(\d+)", txt[0])
            got_aop = int(aop.group(1)) if aop else 0
            exp_regs = [defs[0] + 105, uses[0] + 105, uses[1] + 105]
            if regs[:1] + regs[-2:] != exp_regs:
                bad.append("%s at +0x%x: registers decoded %s, selection asked %s"
                           % (m.form, off, regs, exp_regs))
            elif got_aop != want.get("aop", 0):
                bad.append("%s at +0x%x: OPERATION decoded %d, selection asked %d - the three-bit "
                           "field is byte4[5]|byte5[3]<<1|byte6[3]<<2"
                           % (m.form, off, got_aop, want.get("aop", 0)))
            continue
        if m.form in ("store.ib.32", "load.ib.32"):
            # BOTH DIRECTIONS, THROUGH AN INDEPENDENT READER. There is no hand-written decoder for
            # the imageblock forms, so the check reads the emitted bytes back with g17as - which
            # decodes through the operand maps, not through the table that authored them - and
            # compares each slot against what selection asked. A field written to the wrong bits
            # comes back as a different number here rather than as a byte-exactness surprise later.
            import re as _re
            opc = 13075 if m.form == "store.ib.32" else 12151
            txt = [x.strip() for x in
                   g17as.render(bytes(b), [(0, len(b), opc)]).splitlines()
                   if x.strip() and not x.startswith(".")]
            if not txt:
                bad.append("%s at +0x%x: emitted bytes do not render, so nothing read them back"
                           % (m.form, off))
                continue
            want = dict(m.fields)
            defs = want.pop("_defs", []); uses = want.pop("_uses", [])
            head = txt[0].split(" / ")[0]
            regs = [int(x) for x in _re.findall(r"\br(\d+)\b", head)]
            # POSITIONAL immediates only: a named modifier such as `op3=#4` (a nonzero member) was
            # matched too and read back as the y offset - hidden while every imageblock access used
            # member 0 (Set A item 10b, the first program to read members 4..60)
            imms = [int(x) for x in _re.findall(r"(?<!=)#(-?\d+)", head)]
            mods = dict((int(a), int(v)) for a, v in _re.findall(r"op(\d+)=#(-?\d+)", head))
            coord = uses[1] if m.form == "store.ib.32" else uses[0]
            val = uses[0] if m.form == "store.ib.32" else defs[0]
            # the member offset is slot 3, printed only when it differs from the form's default
            member = mods.get(3, 12)
            if regs[:2] != [val + 105, coord + 105]:
                bad.append("%s at +0x%x: registers decoded %s, selection asked value r%d coord r%d"
                           % (m.form, off, regs[:2], val + 105, coord + 105))
            elif member != want.get("member"):
                bad.append("%s at +0x%x: member offset decoded %d, selection asked %d"
                           % (m.form, off, member, want.get("member")))
            elif _ib_op1(b) != want.get("op1"):
                # THE WHOLE OPERAND 1, read back through Apple's decoder: a template value copied
                # from another context made every lane's write private (item 12, rounds 5 to 9)
                bad.append("%s at +0x%x: operand 1 decoded %#x, selection asked %#x"
                           % (m.form, off, _ib_op1(b), want.get("op1") or 0))
            else:
                # the trailing positional immediates are the x and y offsets; the load leaves y a
                # modifier, so it is read from whichever place this form put it
                dx = imms[1] if len(imms) > 1 else mods.get(7, 0)
                dy = (imms[2] if len(imms) > 2 else mods.get(8, 0))
                if (dx, dy) != (want.get("dx"), want.get("dy")):
                    bad.append("%s at +0x%x: coordinate offset decoded (%s,%s), selection asked "
                               "(%s,%s)" % (m.form, off, dx, dy, want.get("dx"), want.get("dy")))
            continue
        if m.form in _CONSTRUCTED:
            want = _CONSTRUCTED[m.form](m)
            if want is not None and bytes(b) != bytes(want):
                bad.append("%s at +0x%x: constructor gives %s, emitted %s"
                           % (m.form, off, bytes(want).hex(), bytes(b).hex()))
            continue
        dec = _DEC.get(m.form)
        if dec is None: continue
        got = getattr(g17asm, dec)(b)
        want = dict(m.fields)
        defs = want.pop("_defs", []); uses = want.pop("_uses", []); want.pop("range_group", None)
        exp = {}
        if m.form == "alu.12":
            exp = dict(dest=defs[0], src1=uses[0], op=want["op"], mode=want["mode"])
            if want["mode"] == g17asm.MODE_REG and len(uses) > 1: exp["src2"] = uses[1]
            elif "imm" in want: exp["imm"] = want["imm"]
        elif m.form == "alu.block": exp = dict(dest=defs[0], src1=uses[0], const=want["const"])
        elif m.form.startswith(("load.vec", "store.vec")): exp = dict(tuple_base=defs[0] if m.form.startswith("load") else uses[0], desc_const=want["desc"], index=uses[-1], disp=want["disp"], release_index=want["release_index"], n=want.get("n", 4))
        elif m.form == "movimm.2":
            from agxforge.g17.formenc import Movimm2 as g17movimm2
            got = g17movimm2.decode(bytes(b))
            for key, value in dict(dest=defs[0], imm=want["imm"],
                                   residue=g17movimm2.BASE).items():
                if got.get(key) != value:
                    bad.append("movimm.2 at +0x%x: %s decoded %r, selection asked %r"
                               % (off, key, got.get(key), value))
            continue
        elif m.form == "movimm.8":  exp = dict(dest=defs[0], imm=want["imm"])
        elif m.form == "movimm16.zero.4": exp = dict(dest=defs[0], file_281=False)
        elif m.form == "read_sr.4":
            exp = dict(dest=defs[0], sr=want["sr"], seq=want["seq"])
            if want.get("half") is not None: exp["half"] = want["half"]
        elif m.form in ("load.8", "load.10", "load.14"):
            exp = dict(dest=defs[0], base=want["base"], offset=want["offset"], index_reg=uses[0],
                       disp2=want.get("disp2", 0), index_scale=want.get("index_scale", 1),
                       narrow=want.get("narrow", 0), hi16=want.get("hi16", 0))
        elif m.form.startswith(("store.half1.", "store.elem1.")):
            # THIS FORM'S ADDRESS IS A DISPLACEMENT, so the check asks the displacement decoder what it
            # wrote rather than the slot decoder - reading a byte displacement through the word store's
            # slot field is exactly the misreading the sweep caught (14 bytes came back as "slot 3").
            exp = dict(src=uses[0], disp=want["disp"], wait_load=want.get("wait_load", 0), subform=1,
                       half=m.form.startswith("store.half1."))
        elif m.form == "store.byte.14":
            # Decode the measured narrowing writer as an element store.  The decoder names its
            # form half because the source register is the 425-based low half; the source IR
            # declaration is nevertheless uchar and the specialist's retained Apple sequence
            # establishes the one-byte destination semantics.
            exp = dict(src=uses[0], disp=0, wait_load=0, subform=1, half=True)
        elif m.form.startswith("store"):
            # EVERY COMPONENT MUST HAVE ITS OWN REGISTER IN THE RUN THE FORM WRITES. The forms write
            # r<src> .. r<src+n-1>, so this asserts that the values the selection named occupy exactly that
            # run - no repeat, no gap. It is here because a repeat used to pass everything else: the store
            # decoded as src=6 n=2 and the selection had asked for [6, 6], so a per-field comparison agreed
            # while the second component read whatever r7 held. A wrong value with every field correct is
            # what a guard on the fields alone cannot see (handoff 10al).
            if want.get("half_pack"):
                # THE GUARD THIS FILE SAID WAS MISSING, now that there is something to guard.
                # A half-vector store reads n consecutive HALF registers - two per word - so the
                # run it writes is ceil(n/2) WORDS and component i belongs to word base + i//2,
                # in the low half for even i and the high half for odd. The word-register rule
                # below is the wrong question for this form: it would demand n distinct words,
                # which is exactly the unpacked layout the hardware refuted, and it PASSED that
                # layout for a session (results/g17-halfvec-runtime-negative-v1: five wrong output
                # words, low halves only). Asserting value indices were consecutive was never the
                # property; this is.
                n = len(uses)
                base = min(uses)
                expect = [base + i // 2 for i in range(n)]
                if list(uses) != expect:
                    bad.append("+0x%03x %s: %d half components must occupy words r%d..r%d two per "
                               "word (%s) and they are in %s - an odd component would read an "
                               "unwritten high half"
                               % (off, m.form, n, base, base + (n - 1) // 2, expect, list(uses)))
                ix = want.get("half_indices")
                if ix != list(range(n)):
                    bad.append("+0x%03x %s: the packed half indices are %s, not 0..%d - the "
                               "components would be stored out of order"
                               % (off, m.form, ix, n - 1))
            elif want.get("range_group") and len(uses) > 1:
                run = list(range(min(uses), min(uses) + len(uses)))
                if sorted(uses) != run:
                    bad.append("+0x%03x %s: the form writes r%d..r%d but the components are in %s - a "
                               "component would read a register the program did not name"
                               % (off, m.form, min(uses), min(uses) + len(uses) - 1, list(uses)))
            # the slot the ENCODER wrote, which is the requested one shifted by the value's
            # position in the register run - see the emit path
            exp = dict(src=min(uses), n=want["n"], **({"subform": want["subform"]} if want.get("subform") is not None else {}),
                       slot=want.get("_slot_emitted", want["slot"]))
        elif m.form == "cmp.6":
            # Two decoders over one instruction's two halves, so both authored parts are checked.
            g = g17asm.decode_cmp_src(b[:2]); g.update(g17asm.decode_cmp_imm(b[2:]))
            got = g; exp = dict(reg=uses[0], source_modifier=want.get('source_modifier',0), imm=want["imm"], rel=want["rel"],
                                keep=want["keep"])
        elif m.form in ("branch.cond.fwd", "branch.cond.back"):
            # THE TEN-BYTE BRANCH IS READ WITH ITS OWN DECODER. decode_branch is the four-byte
            # form's 12-bit field, so any displacement past +-2046 came back wrapped (-2074 read as
            # -26) and this check refused a correct program: a 128-op loop body at 2 KiB. The
            # emitter writes all 47 bits through encode_branch10; decode_branch10 reads them.
            if len(b) == 10:
                got = dict(disp=g17asm.decode_branch10(bytes(b)))
            exp = dict(disp=want["_disp"]) if "_disp" in want else {}
        for k, v in exp.items():
            if got.get(k) != v:
                bad.append("+0x%03x %s: %s decoded %r, selection asked %r"
                           % (off, m.form, k, got.get(k), v))
    return bad

def _check_flag_discipline(layout):
    """A compare's FLAG must be consumed by the very next instruction.

    THERE ARE SEVEN FLAG REGISTERS - FLAG0..FLAG6, all read by the compare and exec-mask opcodes,
    with FLAG0 used 1168 times in the corpus against 176 for FLAG1 (isa/g17-special-registers.toml,
    from the ISA agent). So flags are an allocatable resource with pressure, and this compiler
    allocates none: every compare it emits writes FLAG0, including both levels of a nested
    conditional.

    That is safe ONLY because the flag is dead immediately - the exec-mask instruction consumes it
    on the next instruction, and it is the MASK STACK, not the flag, that carries state across the
    region. Nested predication executes 11/11 for exactly that reason.

    It stops being safe the moment anything separates a compare from its consumer, which is what a
    scheduler would do. So the invariant is checked rather than assumed: if this fires, flag
    allocation is no longer optional and FLAG1..FLAG6 are where to put the other live ones.
    """
    forms = [m.form for _, _, m in layout]
    bad = []
    for i, f in enumerate(forms):
        if not f.startswith("cmp"):
            continue
        nxt = forms[i + 1] if i + 1 < len(forms) else None
        if nxt not in ("exec.mask", "loop.flag"):
            bad.append("+%d: %s is followed by %r, not the instruction that consumes its flag - "
                       "with one flag allocated that is a clobber" % (i, f, nxt))
    return bad


import re as _re
_BITPOS = _re.compile(r"b(\d+)\[(\d+)\]")


class InheritedBitsUnreadable(RuntimeError):
    """The inherited-bit audit could not read its table: never reported as "no inherited bits"."""


def _inherited_bits(layout):
    """The bits THIS program emits that nobody authored, measured. -> {form: [(byte, bit), ...]}

    THIS USED TO BE A LITERAL. Five keys - constant_program, descriptors, launch_metadata,
    prologue, form_templates - were set unconditionally on every program, so "compiles with zero
    inherited bits" was a constant and no work could move it. Three of the five were also false by
    then: the compiler AUTHORS the constant program and the prologue (see G17Program.prologue),
    and descriptors and launch metadata are the linker's sections, which its own generator now
    emits rather than copying from an artefact. The dict also popped two keys nothing ever added
    and carried an `if False` branch.

    What is actually inherited is a BIT: one this backend's encoder does not write and that
    isa/g17-form-constants.toml does not settle as a form constant, so it comes out of whatever
    template the form was harvested from. That is per (form, opcode, length) and therefore per
    program, which is the whole point - a program that emits only settled forms owes nothing, and
    saying so is only meaningful if a program that emits an unsettled one is told apart from it.

    WHAT IT COUNTS, exactly: a position that (a) the encoder's written mask does not cover, (b) the
    constants record does not class as none/opcode/rejected - g17scorecard.bit_provenance calls
    those DERIVED, "the form requires it and the value is not a choice" - and (c) this program
    emits differently from the settled value. All three filters are needed; without the third it
    reports Apple's corpus minorities rather than anything about this program.

    IT IS NOT g17debt's NUMBER AND I AM NOT CLAIMING IT IS. That tool is the backend-wide census
    over its own program set and currently reports six bits on auth/op12364; this reports ten
    positions on alu and load forms over the end-to-end kernels plus the scan. The two sets are
    DISJOINT, which means they are answering different questions - most likely different program
    populations - and that is an open question worth naming rather than a discrepancy to average
    away. Use g17debt for the backend census; use this for what one compiled program owes.
    """
    from agxforge.g17 import const as g17const
    # AN UNREADABLE TABLE IS AN ERROR, NOT "NOTHING INHERITED". This returned {} on any exception, so every
    # `p.inherited == {}` check (g17regress) passed VACUOUSLY whenever isa/g17-form-constants.toml could not be read -
    # the audit reported a clean program because it had audited nothing (ledger
    # g17-the-inherited-bit-audit-reads-an-unreadable-table-as-none). The table is committed; failing to read it
    # means the checkout is broken, and the compile says so.
    try:
        table = g17const.load()
    except Exception as why:
        raise InheritedBitsUnreadable("the inherited-bit audit cannot read the form-constants table "
                                      "(isa/g17-form-constants.toml): %s: %s - refusing to report no inherited "
                                      "bits for a program it has not audited" % (type(why).__name__, why)) from why
    # A FAMILY ROW DESCRIBES ITS OWN OPCODES AND NOT A NEIGHBOUR'S. g17const's rows keyed
    # (form, None, length) cover a named set of corpus opcodes, and its docstring is explicit that
    # substituting a same-length entry of the same form name "hands one opcode's form-defining bits
    # to another". The half load is exactly that case: the scan emits op12646 while the load.14 row
    # was probed with the WORD load and covers [12682, 12674]. Comparing the half load's bytes
    # against the word load's constants reported two positions as inherited that are simply a
    # different instruction. An opcode no row covers is reported as UNCOVERED rather than measured
    # against the wrong row.
    from agxforge.g17 import formops as g17formops
    fam = {}
    # the same vacuity one level down: an unreadable inventory emptied `fam`, and with it the UNCOVERED check below
    # (an opcode no row covers is then silently measured against the wrong row)
    try:
        for f, o, l, corpus_ops, _e in g17const.inventory():
            fam[(f, o, l)] = set(corpus_ops or ())
    except Exception as why:
        raise InheritedBitsUnreadable("the inherited-bit audit cannot read the form inventory (g17const.inventory): "
                                      "%s: %s" % (type(why).__name__, why)) from why
    optable = g17formops.load()
    out = {}
    for _off, raw, m in layout:
        try:
            emitted_op = g17formops.opcode_of(m, len(raw), optable)
        except Exception:
            emitted_op = None
        # THE EMITTED OPCODE PICKS THE ROW, not the form name. A form name plus a length can name
        # two different instructions - load.14 is op12682 for a word and op12646 for a half - and
        # the row that describes one says nothing correct about the other.
        key = (m.form, emitted_op, len(raw))
        row = table.get(key)
        rowkey = key
        if row is None:
            rowkey = (m.form, m.fields.get("opcode"), len(raw))
            row = table.get(rowkey)
        if row is None:
            rowkey = (m.form, None, len(raw))
            row = table.get(rowkey)
        if row is None:
            continue
        covers = fam.get(rowkey)
        if emitted_op is not None and covers and emitted_op not in covers:
            out.setdefault("%s/%s UNCOVERED" % (m.form, emitted_op), set()).add("no form-constant row")
            continue
        written = row.get("written") or b""
        # A BIT THE FORM REQUIRES IS NOT DEBT EITHER. g17scorecard.bit_provenance puts it exactly:
        # "a bit that is an opcode bit, or one the constants record classes as rejected, is DERIVED
        # - the form requires it and the value is not a choice. The endpoint's debt is the bits
        # that are neither written nor explained." So roles none/opcode/rejected are explained.
        derived = set()
        for spec in row.get("roles") or []:
            if str(spec).split(":")[0] in ("none", "opcode", "rejected"):
                derived |= {(int(a), int(b)) for a, b in _BITPOS.findall(str(spec))}
        for note in row.get("unresolved") or []:
            pos = str(note).split()[0]
            try:
                byi, bi = pos.rstrip("]").split("[")
                byi, bi = int(byi.lstrip("b")), int(bi)
            except ValueError:
                continue
            # A BIT THE ENCODER WRITES IS NOT INHERITED, whatever the corpus minority says about
            # it. `unresolved` in the table marks an operand whose observed value has a minority -
            # a fact about Apple's programs - and most of those bits are ones this backend computes
            # and writes. Only a bit outside the written mask comes from the template.
            if byi < len(written) and (written[byi] >> bi) & 1:
                continue
            if (byi, bi) in derived:
                continue
            # AND THE EMITTED VALUE DECIDES. `unresolved` marks a position whose OBSERVED value has
            # a minority in Apple's corpus; that is a fact about Apple's programs, not about this
            # one. The bit is only inherited here if what this program emits differs from the
            # settled constant - otherwise the template's value and the settled value agree and
            # nothing was inherited that anyone could have chosen differently.
            val = row.get("value") or b""
            if byi < len(raw) and byi < len(val) and \
                    ((raw[byi] >> bi) & 1) == ((val[byi] >> bi) & 1):
                continue
            out.setdefault("%s/%s" % (m.form, key[1] if key[1] is not None else ""), set()).add(pos)
    return {k: sorted(v) for k, v in sorted(out.items())}



# IR OP KINDS THAT MAY BE RECOMPUTED. A producer is pure when running it twice yields the same value
# and nothing else changes. `load` is pure ONLY when this function never writes its buffer, which
# is decided per call. Left out on purpose: every store and atomic (side effects), cmp and icmp
# (cmp.src reads destructively, see _rematerialise), csel (reads predicate state), phi (a merge is
# not a computation), machine (its meaning is whatever the table says), texture and imageblock
# reads (resource state), the simd ops (cross-lane), and the threadgroup forms.
_REMAT_PURE = frozenset({
    # `load` is here CONDITIONALLY: pure() also requires that this function never writes the
    # buffer. Leaving it off the set made every load `impure` before that check could run - the
    # one kind this pass exists to recompute - and 96 columns refused with 91 loads live.
    "load", "const", "builtin", "add", "mul", "sub", "shl", "shr", "sar", "sarv", "and", "or", "xor", "not",
    "nand", "nor", "andn", "orn", "xnor", "shiftadd", "madd", "addsat", "subsat", "fadd", "faddi",
    "fmul", "fma", "fsat", "f16_to_f32", "f32_to_f16_rte", "floor", "ceil", "trunc", "rint",
    "exp2", "log2", "recip", "rsqrt", "rsqrt2", "msb", "reverse"})
_WRITES_BUFFER = ("store", "store_at", "store_range", "store_vec4_at", "atomic_rmw", "atomic_add", "atomic_uniform")
REMAT_HEADROOM = 32      # registers left for the recomputation chains' own temporaries
# HOW FAR BELOW BUDGET EACH ROUND EVICTS. Evicting exactly the excess made a hot point of every
# instruction in an over-budget region - one full rescan per instruction, thousands of rounds at
# 384 columns. Evicting down to budget minus this slack makes hot points rare and the extra
# evictions are the cheapest chains anyway.
REMAT_SLACK = 16
# A dead leaf whose own recomputation is at most this many ops is recomputed; deeper ones are
# referenced and kept live instead (one register each).
REMAT_LEAF_MAX = 3


SPILL_GROUP = 4                 # the vec4 store moves four consecutive registers, so victims come in fours
SPILL_MAX_ROUNDS = 512          # a BOUND ON COMPILE TIME, and it is not a property of the machine
# WHAT THIS NUMBER IS AND IS NOT. Each round spills one group of four, so 512 covers 2,048 spilled
# values - the in-place pressure family reaches n=2,128 under it. Raising it does not hit an
# architectural wall: with the bound effectively removed the same family still compiles at n=6,000,
# and what grows is COMPILE TIME (4.5s at 2,129, 22.9s at 4,000, 61.8s at 6,000). So no ceiling was
# found below this cap, and the refusal below says so rather than reporting a bound as a limit.
#
# The earlier value of 64 is why this is spelled out: it stopped the family at n=278 and the
# refusal read like a measurement. A bound mistaken for a fact about the hardware is worse than a
# slow compile.


def _spill_pressure(fn, budget):
    """Keep at most `budget` values live by SPILLING them to scratch and reloading at each use.

    -> the number of groups spilled. A NO-OP on any function already under budget, so a program
    that fits keeps its bytes exactly; that is asserted by a digest, not assumed.

    THIS RUNS AFTER REMATERIALISATION AND PICKS UP WHAT IT CANNOT. Remat recomputes a value whose
    producer is pure, and a load from a buffer the program WRITES is not pure - the value at that
    address need not still be there. So an in-place update defeats remat completely: every live
    value is impure, the pass runs out of candidates rather than out of room, and the allocator
    refuses. That program is the one this pass exists for (tools/g17spillpressure.py).

    THE SHAPE IS THE ONE THE SOURCE-OWNED DELIVERY MEASURED (docs/archive/g17-spill-source-handoff.md),
    because three others were tried there and each failed differently:

      * a value STRAIGHT FROM A LOAD cannot be spilled - the 8-byte store carries no load-wait -
        so a victim must have been through an ALU. That is a constraint on victim policy, and it
        is why `_spill_candidates` excludes a raw load.
      * the RELOAD IS SCALAR, one value at its own use. A vector reload restores four registers
        whether or not the uses are there and measured NET NEGATIVE against no spill at all.
      * SPILLING AFTER THE COMPUTATION relieves nothing, because the peak is set by the last value
        computed. That shape compiles, which is what makes it dangerous, so the store goes at the
        last definition IN the group and the group is chosen from what is live at the hot point.

    ADDRESSES ARE PER-THREAD BY INDEX REGISTER, not by displacement: slot `t * groups + g`, whose
    four words are `4 * slot ..`. The displacement is witnessed only at 0, 16 and 32, so it could
    never have scaled; the index register is what makes a thread's slots its own.
    """
    if len(fn.blocks) != 1:
        return 0                                   # a value live across a block is the CFG's business
    if SPILL_MAX_ROUNDS <= 0:
        return 0                                   # an off switch, so the refusal underneath stays readable
    blk = fn.blocks[0]
    scratch = thread = base = None
    spilled = 0
    # THE PASS MUST NOT SPILL ITS OWN ADDRESS REGISTERS. `spill_base` is live from the top of the
    # program to the last reload and is neither a load nor a builtin, so it is the single best
    # victim this policy can see - and spilling it means computing the address of the spill from a
    # value that has been spilled. Measured before it was excluded: every round after the first
    # chose it again, so four "different" groups all carried the same value and the tuples they
    # needed overlapped until the narrow pool ran out.
    machinery = set()
    for _round in range(SPILL_MAX_ROUNDS):
        hot, live, defat, uses = _pressure_point(blk, budget)
        if hot is None:
            return spilled
        victims = _spill_candidates(live, defat, uses, hot, machinery, blk.ops)
        if len(victims) < SPILL_GROUP:
            return spilled                         # let the allocator refuse, and say what is live
        if scratch is None:
            thread = _spill_index_source(blk, live, budget)
            scratch = ir.Buffer("SPILL", max(b.slot for b in fn.buffers) + 1)
            fn.buffers.append(scratch)
            fn.spill_slot = scratch.slot        # the ABI states the extent against this binding
            # ONE ADDRESS COMPUTATION FOR THE WHOLE PROGRAM, NOT ONE PER GROUP. A first cut built
            # the stride and the base inside each group's store: at 21 groups that was 21 redundant
            # `thread * stride` values, each live from the top of the program to its own last
            # reload, so the pass spent about forty registers competing with itself for the
            # pressure it was there to relieve.
            address = _spill_address(fn, blk, thread)
            machinery.update(o.dest for o in address if o.dest is not None)
            base = next(o.dest for o in address if getattr(o.dest, "name", "") == "spill_base")
            # THE INDICES JUST MOVED. defat and uses were measured before those two ops were
            # spliced in, so every position above them is stale by two - and a store placed from a
            # stale index lands before the values it stores are defined. Recompute rather than
            # adjust: an offset correction is the kind of thing that is right until someone inserts
            # a third op.
            continue
        group = victims[:SPILL_GROUP]
        _emit_spill(fn, blk, scratch, base, group, spilled, defat, uses, machinery)
        spilled += 1
        # PROVENANCE, RECORDED BY THE PASS THAT CREATED IT. Counting op17256 in the emitted code
        # would count a SOURCE-AUTHORED vector store too, and a program that writes one by hand -
        # g17spillsource.kernel(4, True) - would then have its ordinary store read as scratch.
        fn.spill_groups = spilled
        _restride_spill(blk, spilled)
    raise Unsupported(
        "register pressure still exceeds the budget after %d spilled groups (%d values). This is "
        "g17cc.SPILL_MAX_ROUNDS, a compile-time bound and NOT a limit of the machine: the same "
        "family compiles at n=6,000 with it raised, at 61.8s. Raise it if a program needs more, "
        "and do not read this number as a ceiling"
        % (SPILL_MAX_ROUNDS, SPILL_GROUP * SPILL_MAX_ROUNDS))


def _pressure_point(blk, budget):
    """(hot, live, defat, uses) - the first instruction after which more than `budget` are live."""
    ops = blk.ops
    defat, uses = {}, {}
    for i, o in enumerate(ops):
        if o.dest is not None:
            defat[o.dest] = i
        for a in o.args:
            if isinstance(a, ir.Value):
                uses.setdefault(a, []).append(i)
    last = {v: us[-1] for v, us in uses.items()}
    live = set()
    for i, o in enumerate(ops):
        for a in o.args:
            if isinstance(a, ir.Value) and last.get(a, -1) <= i:
                live.discard(a)
        if o.dest is not None and last.get(o.dest, -1) > i:
            live.add(o.dest)
        if len(live) > budget:
            return i, set(live), defat, uses
    return None, set(), defat, uses


# A RELOAD IS A LOAD, AND SEVERAL CONSUMERS REFUSE ONE. Each of these forms lacks a load-wait bit
# this session can write - on most of them byte0[3] is an opcode bit - so an operand straight from
# a load is refused at selection with "straight from a load". A spilled value whose later use is
# one of them comes back as a reload and turns a program that compiled into one that does not.
#
# THE LIST IS THE SELECTOR'S, READ OFF ITS OWN REFUSALS, not a guess at which forms are delicate:
# the bitwise pair (op424 family), icmp/csel, store_at (op17229's 8-byte form), store_vec4_at, the
# predication compare, and the machine forms. Two of these were found by tests rather than by
# reading - the op424 allocation pair, and then the mixed source-store control, which is the whole
# reason a control that combines an authored store with an automatic spill exists.
#
# The alternative is the laundering the fselect path already does - copy through an ALU add-0,
# which carries the wait - and it is NOT used here: at the IR level a `+ 0` is the folded-arm case
# and this compiler folds it away, so the launder would vanish and the refusal would come back
# somewhere harder to read.
_SPILL_REFUSING_CONSUMERS = (frozenset(BITWISE_REG_OPCODE) | frozenset(MACHINE_OPCODE)
                             | frozenset(MACHINE_UNARY)
                             | {"icmp", "csel", "cmp", "store_at", "store_vec4_at"})


def _spill_candidates(live, defat, uses, hot, machinery=(), ops=()):
    """Live values worth spilling, Belady order - farthest next use first.

    A RAW LOAD IS EXCLUDED because store_vec4_at refuses a value straight from a load, and a
    `builtin` because it lives in a four-bit destination among twelve narrow registers: spilling
    the thread id would cost the address arithmetic its own operand. Both exclusions are the
    source-owned delivery's findings, not caution.
    """
    def next_use(v):
        return next((u for u in uses.get(v, []) if u > hot), None)

    def consumers_accept_a_reload(v):
        return all(ops[u].kind not in _SPILL_REFUSING_CONSUMERS
                   for u in uses.get(v, []) if u > hot)

    out = [v for v in live
           if v not in machinery
           and getattr(v, "op", None) is not None
           and v.op.kind not in ("load", "builtin", "const")
           and next_use(v) is not None
           and defat.get(v, 10 ** 9) < hot
           and consumers_accept_a_reload(v)]
    out.sort(key=lambda v: (-next_use(v), defat[v]))
    # GROUPED BY DEFINITION ORDER once chosen: the four go to one register tuple, and values
    # defined near each other are the ones an allocator can place consecutively.
    out.sort(key=lambda v: defat[v])
    return out


def _clone_straightline(fn):
    """A copy of a single-block function, cloned ITERATIVELY.

    copy.deepcopy recurses through Value -> Op -> args -> Value, so a long dependency chain - which
    is precisely what a pressure program is - raises RecursionError before the copy is made. This
    walks the op list once instead. Multi-block functions are returned as-is; neither pass touches
    them.
    """
    if len(fn.blocks) != 1:
        return fn
    clone = ir.Function(fn.name, list(fn.buffers))
    clone.blocks = [ir.Block(fn.blocks[0].label)]
    clone.threadgroup = fn.threadgroup          # a declaration; neither pass mutates it
    mapped = {}
    for op in fn.blocks[0].ops:
        args = [mapped.get(a, a) if isinstance(a, ir.Value) else a for a in op.args]
        dest = None
        if op.dest is not None:
            dest = ir.Value(op.dest.type, op.dest.name)
            mapped[op.dest] = dest
        clone.blocks[0].ops.append(ir.Op(op.kind, dest, args, **dict(op.attrs)))
    return clone


# THE ONLY LAUNCH INDEX A SPILL SLOT MAY BE ADDRESSED FROM. A slot is per thread, and "per thread"
# means over the whole GRID: `thread_position_in_threadgroup` repeats in every group, so two threads
# in different groups would share a slot and silently overwrite each other. The y and z components
# are a different domain again. The first cut of this pass took whatever builtin came first in the
# program - root's review named it - which is right only because every program that had reached it
# happened to read grid.x first.
SPILL_INDEX_BUILTIN, SPILL_INDEX_AXIS = "thread_position_in_grid", "x"


def _spill_index_source(blk, live, budget):
    """The value a spill address is formed from, or a refusal naming the domain it needs."""
    builtins = [o for o in blk.ops if o.kind == "builtin"]
    for o in builtins:
        if (o.attrs.get("which") == SPILL_INDEX_BUILTIN
                and o.attrs.get("axis") == SPILL_INDEX_AXIS):
            return o.dest
    read = sorted({"%s.%s" % (o.attrs.get("which"), o.attrs.get("axis")) for o in builtins})
    raise Unsupported(
        "this program needs %d values live at once against a budget of %d and cannot be spilled: a "
        "spill slot is addressed per thread over the whole grid, which needs %s.%s, and this "
        "function reads %s. Another builtin's domain is not a substitute - "
        "thread_position_in_threadgroup repeats in every group, so two threads would share a slot"
        % (len(live), budget, SPILL_INDEX_BUILTIN, SPILL_INDEX_AXIS,
           ", ".join(read) if read else "no position builtin"))


def _spill_address(fn, blk, thread):
    """Build `base = thread * stride` once, immediately after the builtin it reads. -> its ops.

    THE STRIDE IS A REGISTER, NOT AN IMMEDIATE: mul's immediate slot is eight bits, so a program
    with 256 groups would refuse on the CONSTANT - address arithmetic, and it would read as a spill
    ceiling if it were not separated.
    """
    made = _build(fn, lambda b: b.mul(thread, b.const(1, name="spill_stride"), name="spill_base"))
    at = next(i for i, o in enumerate(blk.ops) if o.dest is thread)
    blk.ops[at + 1:at + 1] = made
    return made


def _build(fn, make):
    """Ops built into a scratch block, for splicing: Op objects reference Values, not positions."""
    tmp = ir.Block("spill")
    make(ir.Builder(fn, tmp))
    return tmp.ops


def _emit_spill(fn, blk, scratch, base, group, index, defat, uses, machinery=None):
    """Store `group` to scratch slot `index` and reload each value at every later use."""
    ops = blk.ops
    after = max(defat[v] for v in group)

    def store(b):
        b.store_vec4_at(scratch, b.add(base, ir.Imm(index), name="spill_slot%d" % index),
                        list(group))

    inserts = {after: _build(fn, store)}
    repoint = {}
    for j, v in enumerate(group):
        for u in uses.get(v, []):
            if u <= after:
                continue                            # a use before the spill still reads the register

            def reload(b, j=j, u=u, v=v):
                # THE RELOAD'S ADDRESS IS REBUILT AT ITS USE, so it is short-lived by construction
                # rather than another value held across the program.
                word = b.mul(b.add(base, ir.Imm(index), name="spill_ri"),
                             ir.Imm(SPILL_GROUP), name="spill_rw")
                repoint[(u, v)] = b.load(scratch, word, offset=j,
                                         name="spill_reload%d_%d" % (index, j))

            inserts.setdefault(u - 1, []).extend(_build(fn, reload))

    if machinery is not None:
        for made in list(inserts.values()):
            machinery.update(o.dest for o in made if o.dest is not None)
    out = []
    for i, o in enumerate(ops):
        for (u, v), fresh in repoint.items():
            if u == i:
                o.args = [fresh if a is v else a for a in o.args]
        out.append(o)
        out.extend(inserts.get(i, []))
    blk.ops = out


def _restride_spill(blk, groups):
    """Each thread owns `groups` slots, so the stride constant is rewritten as they accumulate.

    WITHOUT THIS EVERY THREAD BUT ZERO OVERLAPS ITS NEIGHBOUR once a second group is spilled: the
    base is `thread * stride`, and a stride of one with two groups per thread puts thread 1's first
    slot on thread 0's second. That is the `collide` defect the source-owned delivery's own address
    checker failed to see, and it is a rewrite rather than a late computation because the stride has
    to be right for every group already emitted, not only the new one.
    """
    for o in blk.ops:
        if o.dest is not None and getattr(o.dest, "name", "") == "spill_stride":
            o.args = [ir.Imm(groups)]


def _rematerialise_pressure(fn, budget):
    """Keep at most `budget` values live at once by RECOMPUTING pure values at their later uses.

    -> the number of recomputations inserted. A no-op on any function already under budget, which
    is every kernel this backend had compiled before the LayerNorm workload: their bytes do not
    move, and that is asserted by the regression rather than assumed.

    THE PRESSURE IS REAL AND NOT THE ALLOCATOR'S FAULT. The width-384 LayerNorm keeps every loaded
    value live through the mean reduction so it can be re-read for the variance and again for the
    output - 384 values, then 384 shifted ones, against 110 wide registers. No assignment of
    registers fits that. The workload's own docstring says so and says the failure belongs to the
    backend; this is the backend owning it. There is no spill form (Alloc's docstring says why), so
    the alternative to refusing is to trade loads for registers: a value whose producer is pure can
    be recomputed where it is next needed instead of held, and a load from a buffer this function
    never writes is pure. That is the three-pass LayerNorm a person would write by hand, produced by
    the compiler from the one-pass IR the author wrote.

    ONLY STRAIGHT-LINE FUNCTIONS. A value live across a block boundary is the CFG's business and a
    loop-carried one cannot be recomputed at all, so a function with more than one block is left
    exactly as it was. Rewrites are SSA-preserving: each recomputation defines a FRESH value and
    only the one use being served is repointed at it, so the lifetimes, the load-use waits and the
    keep/release modifiers are then computed by select() over ordinary IR rather than special-cased.
    Belady's choice picks the victim: the live value whose next use is farthest away.
    """
    if len(fn.blocks) != 1:
        return 0
    blk = fn.blocks[0]
    written = {a.slot for o in blk.ops if o.kind in _WRITES_BUFFER
               for a in o.args[:1] if isinstance(a, ir.Buffer)}

    def pure(v):
        o = getattr(v, "op", None)
        if o is None or o.kind not in _REMAT_PURE:
            return False
        if o.kind == "load":
            return isinstance(o.args[0], ir.Buffer) and o.args[0].slot not in written
        return True

    inserted = 0
    for _round in range(100000):
        ops = blk.ops
        defat, uses = {}, {}
        for i, o in enumerate(ops):
            if o.dest is not None:
                defat[o.dest] = i
            for a in o.args:
                if isinstance(a, ir.Value):
                    uses.setdefault(a, []).append(i)
        # the first point where more than `budget` values are live AFTER instruction i
        last = {v: us[-1] for v, us in uses.items()}
        live = set()
        hot = None
        for i, o in enumerate(ops):
            for a in o.args:
                if isinstance(a, ir.Value) and last.get(a, -1) <= i:
                    live.discard(a)
            if o.dest is not None and last.get(o.dest, -1) > i:
                live.add(o.dest)
            if len(live) > budget:
                hot = i
                break
        if hot is None:
            return inserted

        def next_use(v, after):
            return next((u for u in uses.get(v, []) if u > after), None)

        # WHAT A RECOMPUTATION COSTS, counted in ops and in whether it cascades. A leaf that is
        # live at `at` is free; a dead one is cloned, recursively. The first version of this pass
        # chose victims by Belady alone - farthest next use - and that picked `base`, the row
        # index: one op to recompute, used by every address the program forms. Each eviction of an
        # index value then wrote a NEW use of `base` into the program, `base` was evicted again for
        # each, and every one of those re-derived the thread-id read and the width constant: 11,156
        # recomputations of the thread id in forty seconds, and no convergence. Cheap, widely used
        # values are the ones to KEEP; the values worth recomputing are the ones a register holds
        # for a long time and a short chain rebuilds - a load from an unwritten buffer, or one or
        # two float ops on top of it.
        # A DEAD LEAF IS RECOMPUTED ONLY WHEN THAT IS CHEAP; OTHERWISE IT IS REFERENCED AND KEPT.
        # In straight-line SSA a value defined before the use may simply be named there - the cost
        # is one register held from its last use to this one - and the first version of this pass
        # did not know that: it recomputed every dead leaf, so `centered_k = fadd(shifted_k,
        # negative_mean)` at its late use found negative_mean dead (its last use was the final
        # centered fadd), cloned it, and that meant cloning mean, and the whole 96-deep reduction
        # under it, until the depth limit refused. negative_mean is one register. A load or an
        # index add is one op. The rule below recomputes a dead leaf whose own chain is at most
        # REMAT_LEAF_MAX ops and references anything deeper, so the row's scalars stay in registers
        # across the passes and the per-column values are what get recomputed.
        def is_live(a, at):
            return defat.get(a, 10**9) < at and last.get(a, -1) >= at

        def cheap_leaf(a, at, depth):
            """Whether a dead leaf is RECOMPUTED (True) or REFERENCED and kept live (False).

            Two things make a leaf one to keep rather than rebuild, and the 32x384 LayerNorm
            found both. A `builtin` reads a special register into a FOUR-BIT destination - one of
            twelve narrow registers - so recomputing the thread id per use put 1,006 read_sr into
            one program and the allocator ran out at instruction zero; a producer that needs a
            constrained register class is never cheap. And a leaf read by many instructions -
            `base`, dead after the first output store because indices[0] IS base - costs one
            register to keep and hundreds of ops to rebuild everywhere; it is a root and stays.
            """
            o = getattr(a, "op", None)
            if o is None or o.kind == "builtin" or len(uses.get(a, ())) > 4:
                return False
            c = chain_cost(a, at, depth + 1) if depth < 8 else None
            return c is not None and c <= REMAT_LEAF_MAX

        def chain_cost(v, at, depth=0):
            """Ops a recomputation of v at `at` would emit, or None if v's own producer is impure.
            Never None because of a leaf: a leaf is either cloned (cheap) or referenced (free)."""
            if not pure(v):
                return None
            cost = 1
            for a in getattr(v, "op").args:
                if isinstance(a, ir.Value) and not is_live(a, at) and cheap_leaf(a, at, depth):
                    cost += chain_cost(a, at, depth + 1)
            return cost

        excess = len(live) - (budget - REMAT_SLACK)
        scored = []
        for v in live:
            later = [u for u in uses.get(v, []) if u > hot]
            if not later or later[0] <= hot + 1:
                continue
            # a value read by many later instructions is a ROOT - base, the row's scalars - and
            # recomputing it per use multiplies the program; it stays in its register.
            if len(later) > 4:
                continue
            costs = [chain_cost(v, j) for j in later]
            if any(c is None for c in costs):
                continue
            scored.append((sum(costs), -later[0], v, later))
        scored.sort(key=lambda t: (t[0], t[1]))
        victims = scored[:excess]
        if os.environ.get("G17_REMAT_TRACE"):
            print("round %d: hot=%d live=%d excess=%d victims=%s" % (
                _round, hot, len(live), excess,
                [(getattr(v, "name", "?"), later, "cost %d" % c) for c, _n, v, later in victims]))
            if _round >= int(os.environ["G17_REMAT_TRACE"]):
                raise Unsupported("trace cap")
        if not victims:
            # SAY WHAT IS LIVE AND WHY EACH ONE WAS PASSED OVER, or the message is a count.
            why = []
            for v in sorted(live, key=lambda x: defat.get(x, 0)):
                later = [u for u in uses.get(v, []) if u > hot]
                o = getattr(v, "op", None)
                why.append("%s=%s next%s uses%d %s" % (
                    getattr(v, "name", "?"), o.kind if o else "?",
                    ("+%d" % (later[0] - hot)) if later else "-", len(later),
                    "root" if len(later) > 4 else "impure" if not pure(v) else
                    "next-is-adjacent" if later and later[0] <= hot + 1 else "?"))
            raise Unsupported(
                "%d values live after instruction %d (%r) against a budget of %d, and none can be "
                "recomputed. Live: %s%s. No spill form is recovered, so this program cannot be "
                "allocated" % (len(live), hot, ops[hot], budget, "; ".join(why[:16]),
                               " ..." if len(why) > 16 else ""))
        # EVERY LATER USE OF EVERY VICTIM IS REPOINTED IN THIS ROUND, and ALL of them in one
        # globally descending order of position. Inserting a chain before use j shifts every index
        # above j, so a second victim's use indices - computed before the first insertion - are
        # stale the moment they lie above it: `ops[j]` is then the wrong instruction, nothing in it
        # `is v`, the repoint silently does nothing, the value stays live across the hot point, and
        # the next round evicts it again into a program that grows without bound. Processing the
        # pairs latest-first across victims keeps every remaining target below every insertion.
        # GROUPED BY TARGET. Two victims read by the same instruction - or one victim read twice,
        # fmul(c, c) - share a target index; inserting the first chain moves the target and the
        # second repoint finds nothing to replace. So every victim of one target is cloned first
        # and inserted together, once, and targets are processed latest-first so no remaining
        # index lies above an insertion.
        by_target = {}
        for _c, _n, v, later in victims:
            for j in set(later):
                by_target.setdefault(j, []).append(v)
        for j in sorted(by_target, reverse=True):
            def clone(w, at, out, depth=0):
                o = w.op
                args = []
                for a in o.args:
                    if isinstance(a, ir.Value) and not is_live(a, at) and cheap_leaf(a, at, depth):
                        a = clone(a, at, out, depth + 1)
                    # else: reference the original - legal, it is defined before `at`
                    args.append(a)
                fresh = ir.Value(getattr(w, "type", ir.I32),
                                 "%s_re%d" % (getattr(w, "name", "v") or "v", inserted + len(out)))
                out.append(ir.Op(o.kind, fresh, args, **dict(o.attrs)))
                return fresh
            new_ops, repl = [], []
            target = ops[j]
            for v in by_target[j]:
                if not any(a is v for a in target.args):
                    raise AssertionError("rematerialisation lost track of a use of %r at %d" % (v, j))
                repl.append((v, clone(v, j, new_ops)))
            args = list(target.args)
            for v, fresh in repl:
                args = [fresh if (a is v) else a for a in args]
            target.args = args
            blk.ops[j:j] = new_ops
            inserted += len(new_ops)
        if _round > 20 * len(blk.ops):
            raise Unsupported("rematerialisation did not converge after %d rounds" % _round)
    raise Unsupported("rematerialisation did not converge")

def _eliminate_dead_pure(fn):
    """Remove pure ops whose destination nothing reads, to a fixpoint. -> ops removed.

    REMATERIALISATION LEFT THE ORIGINALS BEHIND, AND THE HARDWARE NOTICED. Repointing a value's
    uses to a recomputation leaves its original producer in the stream with no reader; the 32x384
    LayerNorm carried 1,257 such ops, 330 of them loads. A dead load still writes its destination
    register - asynchronously - and the allocator, seeing a value with no uses, hands that register
    to the next value, which for 203 of them was another load. On the CPU model the later write
    wins and every output matched FP64; on the GPU the first query returned 12,261 wrong outputs
    and the fitted pattern says the EARLIER write won: column 61 came back holding column 59. That
    is a write-after-write on a load destination whose completion order nothing guaranteed, and it
    exists only because a dead instruction was emitted. Deleting dead pure ops is the generic fix;
    the emit-time guard below is what rejects the old behaviour if anything ever recreates it.
    """
    removed = 0
    if len(fn.blocks) != 1:
        return 0
    blk = fn.blocks[0]
    while True:
        used = {a for o in blk.ops for a in o.args if isinstance(a, ir.Value)}
        dead = [o for o in blk.ops
                if o.dest is not None and o.dest not in used and o.kind in _REMAT_PURE
                and not (o.kind == "load" and o.args and isinstance(o.args[0], ir.Buffer)
                         and o.args[0].slot in {a.slot for q in blk.ops if q.kind in _WRITES_BUFFER
                                                for a in q.args[:1] if isinstance(a, ir.Buffer)})]
        if not dead:
            return removed
        ids = {id(o) for o in dead}
        blk.ops = [o for o in blk.ops if id(o) not in ids]
        removed += len(dead)


def unread_load_destination_reuse(layout):
    """[(earlier load offset, later writer offset, register)]: a load whose destination register is
    written again before the loaded value was ever read. The hardware hazard behind the first
    full-width LayerNorm failure; zero is the only acceptable count.

    ANY WRITER, NOT ONLY A LOAD. The first version counted only a second load, and an ALU write is
    the same race: a dead load (its value never read) lets the allocator reuse its destination at
    once, and the load, still in flight, lands on the new value. Online-softmax attention read m and
    l with four-word row loads and used one word; the three dead loads' registers were reused for
    the next address computations, and on hardware rows 2 and 9 of 16 came back wrong (P2 columns
    left unexponentiated, both stats unmoved, O exactly zero; results/g17-tensor-stream-v1)."""
    # Unread loads never reach emission in the first place (_drop_unread_loads); this is the
    # backstop for any that another pass creates after the IR.
    out, pending = [], {}
    for off, _raw, m in layout:
        for r in m.fields.get("_uses") or []:
            pending.pop(r, None)
        for r in m.fields.get("_defs") or []:
            if r in pending:
                out.append((pending[r], off, r))
            if m.form in ("load.8", "load.10", "load.14"):
                pending[r] = off
            else:
                pending.pop(r, None)
    return out


CMP_IMM_MAX = 0xFF


def _strip_mine_for_compare_width(fn):
    """Re-index a counted loop whose bound exceeds the compare immediate. -> unroll factor, or 0.

    THE COMPARE IMMEDIATE IS THE BOUND, NOT THE TRIP COUNT. The latch of a counted loop is
    `phi + step < bound`, and cmp.6 holds an eight-bit immediate, so a 384-column reduction refuses
    at the compare however many iterations it would run. Unrolling alone does not help: after two
    copies the latch is still `k + 2 < 384`. What clears it is a NEW counter that counts
    iterations - `i + 1 < trips` with trips <= 255 - while the original induction `k` keeps
    stepping by the unroll factor as a second, uncompared loop-carried value. Body copy j reads
    `k + j`, so every load, product and add of the 384-term reduction happens in exactly the order
    the IR wrote, and the exit sees the same final values it saw before.

    This is what Apple's compiler does NOT do: at bound 384 it switches to op10370, a register-
    register compare, measured on `for (k = 0; k < 384; ++k)` against the same loop at 200 which
    keeps the immediate form. That form is derived here (g17as: six slots) but has never been
    authored or executed by this backend, so the lowering that uses only executed forms is the one
    chosen, and the wider compare is recorded as Apple's measured alternative rather than adopted.

    WHAT IT REFUSES BY NAME: a bound that does not divide into <= 255 trips by any factor up to 16
    (a remainder epilogue is not implemented), and any loop whose header is not the single-block
    counted shape select() proves. A loop already within the immediate is untouched, which is every
    loop this backend had compiled before: asserted by regression over the executed loop kernels.
    """
    done = 0
    for blk in list(fn.blocks):
        t = blk.term
        if t is None or t.kind != "br_cond" or t.args[1] is not blk:
            continue                                   # not a single-block self loop
        cond = t.args[0]
        got = _counted(cond) if isinstance(cond, ir.Value) and cond.op is not None else None
        if got is None:
            continue
        phi_k, step, bound = got
        if bound <= CMP_IMM_MAX:
            continue
        init = _imm_of(phi_k.op.args[0])
        if init is None or step != 1:
            raise Unsupported("a counted loop bound of %d exceeds the compare's 8-bit immediate and "
                              "this re-indexing handles only `phi(constant) + 1 < bound`; the start "
                              "is %r and the step %d" % (bound, phi_k.op.args[0], step))
        trips = bound - init
        U = next((u for u in range(2, 17) if trips % u == 0 and trips // u <= CMP_IMM_MAX), None)
        if U is None:
            raise Unsupported("a counted loop of %d iterations cannot be re-indexed under the compare's "
                              "8-bit immediate by any factor up to 16 without a remainder, and no "
                              "remainder epilogue is implemented" % trips)
        ops = blk.ops
        phis = [o for o in ops if o.kind == "phi"]
        cmp_op = cond.op
        next_k = cmp_op.args[0]
        body = [o for o in ops if o.kind not in ("phi", "br_cond") and o is not cmp_op]
        # every phi's latch value, and the loop-defined values the exit may read
        latch_of = {p.dest: p.args[1] for p in phis}
        defined = {o.dest for o in body if o.dest is not None}

        def fresh(v, tag):
            return ir.Value(getattr(v, "type", ir.I32), "%s_u%s" % (getattr(v, "name", "v") or "v", tag))

        new_body = []
        # copy 0 is the original body, with next_k left in place but no longer compared
        prev = {v: v for v in defined}
        k_vals = {0: phi_k}
        for j in range(1, U):
            # this copy's induction value is k + j, a fresh add off the phi
            kj = fresh(phi_k, "%d" % j)
            new_body.append(ir.Op("add", kj, [phi_k, ir.Imm(j)]))
            k_vals[j] = kj
            ren = {}
            for pv, latch in latch_of.items():
                # a loop-carried value in copy j is what copy j-1 produced for its latch
                ren[pv] = kj if pv is phi_k else prev.get(latch, latch)
            cur = {}
            for o in body:
                if o is next_k.op:
                    continue                           # the counter's own increment: replaced below
                # A USE RESOLVES THROUGH THIS COPY FIRST. An argument defined earlier in the same
                # copy (the index this copy computed, the value this copy loaded) is `cur`'s; only
                # a value this copy has not defined yet is copy j-1's. Resolving through `prev`
                # alone made every copy's loads read copy 0's index and every copy's fma read
                # copy 0's operands - a program that decoded and matched Apple's boundaries and
                # returned the bias alone (g17queryloweringcheck.py, 2026-09-09).
                args = [ren.get(a, cur.get(a, prev.get(a, a))) if isinstance(a, ir.Value) else a
                        for a in o.args]
                d = fresh(o.dest, "%d" % j) if o.dest is not None else None
                new_body.append(ir.Op(o.kind, d, args, **dict(o.attrs)))
                if d is not None:
                    cur[o.dest] = d
            prev = {v: cur.get(v, prev[v]) for v in defined}
        # THE TWO INDUCTIONS. k advances by U; a new counter i advances by 1 and is the one compared.
        k_next = fresh(phi_k, "next")
        new_body.append(ir.Op("add", k_next, [phi_k, ir.Imm(U)]))
        # A PHI IS SEEDED BY A VALUE, not a literal: verify() refuses `phi 0`. The seed is a
        # const materialised in the loop's predecessor, before its terminator.
        preds = [b for b in fn.blocks if b is not blk and b.term is not None
                 and any(a is blk for a in b.term.args)]
        if len(preds) != 1:
            raise Unsupported("the loop at %r has %d predecessor blocks; re-indexing needs exactly "
                              "one to seed its trip counter in" % (blk.label, len(preds)))
        seed = ir.Value(ir.I32, "trip0")
        preds[0].ops.insert(len(preds[0].ops) - 1, ir.Op("const", seed, [ir.Imm(0)]))
        i_phi = ir.Value(ir.I32, "trip")
        i_next = ir.Value(ir.I32, "trip_next")
        i_op = ir.Op("phi", i_phi, [seed])
        new_body.append(ir.Op("add", i_next, [i_phi, ir.Imm(1)]))
        more = ir.Value(ir.I32, getattr(cond, "name", "more") or "more")
        new_cmp = ir.Op("cmp", more, [i_next, ir.Imm(trips // U)], pred="lt")
        # rebuild the latches: every phi takes copy U-1's value; k takes k_next; i takes i_next
        for p in phis:
            p.args = [p.args[0], k_next if p.dest is phi_k else prev[latch_of[p.dest]]]
        i_op.args.append(i_next)
        # the exit's view of loop-defined values is the LAST copy's
        exit_map = {v: prev[v] for v in defined if prev[v] is not v}
        exit_map[next_k] = k_next
        for other in fn.blocks:
            if other is blk:
                continue
            for o in other.ops:
                o.args = [exit_map.get(a, a) if isinstance(a, ir.Value) else a for a in o.args]
        term = ir.Op("br_cond", None, [more, blk, t.args[2]])
        blk.ops = phis + [i_op] + body + new_body + [new_cmp, term]
        done = U
    return done


# THE CONSTANT POOL (metadata slot 13) IS THE PROGRAM'S, NOT THE CLASS'S. Codex's one-variable
# controls (results/g17-cooperative-pool-controls-v1) move slot-13 bytes with one source literal
# while __TEXT stays identical, so what an image's pool must hold is whatever the PROGRAM reads
# from constant memory. This compiler reads none: every literal is materialised by movimm
# (op11842, `mov.imm`), every memory read names a bound buffer, a texture binding, an imageblock or
# the declared scratchpad. So the pool is derived EMPTY - and derived, not assumed: the set below
# is the forms that would read constant memory, and a program carrying one is refused here until
# its bytes are captured, rather than shipped with an empty pool and hope. It is empty because no
# such lowering exists yet; a future one registers its form here and lands the bytes with it.
CONSTANT_READING_FORMS = frozenset()
# the pool readers Apple's tensor programs use where an immediate no longer fits: the constant-pool
# multiply pair (op10828/op10829, A at 18+ rows) and op615 (the K = 256 program). Classified by
# opcode on the tensor rows because those rows are Apple's instructions, not this backend's forms.
POOL_READING_OPCODES = frozenset({10828, 10829, 615})


# = g17abi.TENSOR_OPCODES (test_g17abi pins the equality). Stated here rather than imported: g17abi
# imports pydantic, and importing it during compilation made the input audit (g17inputs --check)
# see platform.mac_ver() read /System/Library/CoreServices/SystemVersion.plist from outside the repo.
TENSOR_OPCODES = frozenset({5106, 12674, 12675, 17257, 10384, 10385})   # = abi.TENSOR_OPCODES (int8 MACs: MM 25.130)


def _execution_requirement(layout):
    """{"simd_width": 32, "tensor": True} when the layout executes a tensor form, else None."""
    for _off, _size, m in layout:
        if m.form.startswith("tensor.") and m.fields.get("opcode") in TENSOR_OPCODES:
            return {"simd_width": 32, "tensor": True}
    return None


def _constant_pool(layout):
    """The external constant bytes the program reads, in slot-13 order. Empty for every program
    this compiler emits (see CONSTANT_READING_FORMS); a constant-reading form is refused by name.

    THE LIMIT OF THIS CHECK, stated: an empty denylist proves that no form the compiler emits
    TODAY reads constant memory - a claim about the enumerated form set, each of whose members
    names its source (a binding, a texture, an imageblock, the scratchpad or an immediate). It
    does not, by itself, prove anything about a form that does not exist yet; that form's author
    must classify it, and the regression that counts the cooperative program's movimm literals
    and asserts its form set is what turns a silent omission into a failing case."""
    hits = sorted({m.form for _o, _r, m in layout if m.form in CONSTANT_READING_FORMS})
    if hits:
        raise Unsupported("the program reads constant memory through %s and the compiler does not yet "
                          "capture the pool bytes; the ABI would state an empty pool that is false" % hits)
    # THE TENSOR ROWS ARE CLASSIFIED BY OPCODE, not by form name: a tensor.inherited or
    # tensor.authored row carries Apple's opcode, and Apple's programs at 18 or more A rows read
    # their multiplier from slot 13 through op10828/op10829 (the retained a-rows-18/32/63/64
    # objects; the linker measured the pool word there, 16 x rows) - and op615 reads a pool word
    # for K = 256. A stream carrying one of those must not become an empty declaration.
    pooled = sorted({m.fields.get("opcode") for _o, _r, m in layout
                     if m.form.startswith("tensor.") and m.fields.get("opcode") in POOL_READING_OPCODES})
    if pooled:
        raise Unsupported("the tensor stream reads the constant pool (slot 13) through op%s and the "
                          "compiler does not author pool bytes; the ABI would state an empty pool that "
                          "is false" % "/op".join(str(o) for o in pooled))
    # THE ONE POOL THIS COMPILER AUTHORS: and16's pool masks (ir.and16(pool=True), MM 25.141.16). Apple's
    # three-binding cooperative witnesses carry them as an 8-byte vector, halfword 0 zero and the masks at 1-3;
    # the instruction names uniform 4 x buffers + h, so the bytes stated here are what that uniform receives.
    masks = {}
    for _o, _r, m in layout:
        a = m.fields.get("and16") if m.form == "auth" else None
        if a and a.get("pool_h") is not None:
            if masks.get(a["pool_h"], a["mask"]) != a["mask"]:
                raise Unsupported("and16 pool halfword %d holds two masks" % a["pool_h"])
            masks[a["pool_h"]] = a["mask"]
    if masks:
        if sorted(masks) != list(range(1, len(masks) + 1)) or len(masks) > 3:
            raise Unsupported("and16 pool halfwords %s: the measured vector is halfword 0 zero and masks at 1..3"
                              % sorted(masks))
        halves = [0] + [masks[h] for h in sorted(masks)] + [0] * (3 - len(masks))
        return tuple(b for hw in halves for b in (hw & 0xFF, hw >> 8))
    return ()


class CooperativeSharingRefused(Unsupported, ValueError):
    """MM P12's production refusal, raised at compile: a tensor program that uses or declares
    threadgroup memory is a cooperative-sharing request. A ValueError too, so the runtime's callers
    and the compiler's catch the same named refusal (agxforge.g17.runtime.COOPERATIVE_SHARING_REFUSAL)."""


def refuse_tensor_sharing(fn, layout):
    """Refuse, by name, a program that executes a tensor form AND uses threadgroup memory or declares
    it (Function.declare_threadgroup). Production's multi-simdgroup tensor class is a per-simdgroup
    tile partition with no threadgroup memory; sharing is refused (MM P12, second branch)."""
    if _execution_requirement(layout) is None:
        return
    uses, declared = _uses_threadgroup_memory(layout), getattr(fn, "threadgroup", None) is not None
    if uses or declared:
        from agxforge.g17.runtime import COOPERATIVE_SHARING_REFUSAL
        raise CooperativeSharingRefused(
            COOPERATIVE_SHARING_REFUSAL + "; the tensor program %s threadgroup memory"
            % ("uses and declares" if uses and declared else "uses" if uses else "declares"))


def _uses_threadgroup_memory(layout):
    """ONE PREDICATE for slot 18: the threadgroup forms, whether a named form or an auth opcode.
    abi_inputs() reports it and compile_function() refuses a program that has it without a
    declaration, so the two cannot disagree."""
    return any(m.form in THREADGROUP_FORMS or (m.form == "auth" and m.fields.get("opcode") in THREADGROUP_AUTH)
               for _o, _r, m in layout)


def resolved_layout(fn):
    """THE LINKER'S RESOLVER, called explicitly (integration's 29644d9c, two-phase): g17teximage.resolve is the
    one resolver - this side does not keep a copy of its arithmetic - and returns the layout both preload
    uses are encoded against (record order and ranks, 4 bytes per record, the descriptor offsets it writes,
    an identity over the canonical form, and the derived preload offset). The program carries it whole in
    resources.resolved_layout, and the contract runs the linker's check_resolved on it."""
    from agxforge.g17 import teximage as g17teximage
    declared = [(b.slot, b.slot in _written_slots(fn)) for b in fn.buffers]
    internal = [(rank, idx) for rank, idx in enumerate(sorted(TEXTURE_INTERNAL_INDICES))] if _uses_texture(fn) else []
    terms = sum(1 for blk in fn.blocks for o in blk.ops if o.kind == "uniform_load")
    return g17teximage.resolve(internal, declared, terms)


_FOLD = [dict(terms=(), skip=frozenset(), root={})]


def _fold_uniform_chains(fn):
    """THE FOLD (handoff 10ab): every uniform_load value must be consumed by exactly one `add`, and those adds must
    form ONE linear chain - base + t1, (+ t2), (+ t3) - with exactly one non-term register operand (the base; Apple's
    fetch). The chain's last add is the root (the one alu.block in main); the others emit nothing. Anything else
    refuses at the uniform_load or the add by name. Sets _FOLD[0] = dict(terms, skip, root={root op: base value})."""
    ops = [o for blk in fn.blocks for o in blk.ops]
    terms = [o.dest for o in ops if o.kind == "uniform_load"]
    _FOLD[0] = dict(terms=tuple(terms), skip=frozenset(), root={})
    if not terms: return
    def is_term(x): return isinstance(x, ir.Value) and x in terms
    adds = [o for o in ops if o.kind in ALU_OP and any(is_term(x) for x in o.args)]
    if any(o.kind != "add" for o in adds): raise Unsupported("%s of a uniform load: only `add` consumes a preloaded term (Apple's S1/S2/S3)" % [o.kind for o in adds if o.kind != "add"][0])
    for t in terms:
        users = [o for o in ops if any(x is t for x in o.args)]
        if len(users) != 1 or users[0].kind != "add": raise Unsupported("a uniform term exists only as an operand of ONE add; %s has %d use(s)%s" % (t.name, len(users), "" if not users else " (%s)" % ",".join(o.kind for o in users)))
    if any(all(is_term(x) for x in o.args) for o in adds): raise Unsupported("an add of two uniform terms with no register: the fold's base is the fetch register, every term is added to it in one chain")
    partial = {o.dest: o for o in adds}
    chain_of = {}
    for o in adds:
        other = [x for x in o.args if not is_term(x)][0]
        if isinstance(other, ir.Value) and other in partial: chain_of[o] = partial[other]      # continues the chain from that add
        elif isinstance(other, ir.Value): chain_of[o] = None                                 # the base register
        else: raise Unsupported("add of a uniform term to an immediate does not lower")
    firsts = [o for o, prev in chain_of.items() if prev is None]
    if len(firsts) != 1: raise Unsupported("uniform terms in %d separate add chains: Apple folds every term into ONE published word" % len(firsts))
    order = [firsts[0]]
    while True:
        nxt = [o for o, prev in chain_of.items() if prev is order[-1]]
        if not nxt: break
        if len(nxt) != 1: raise Unsupported("a folded partial sum is read by two adds; the chain must be linear")
        order.append(nxt[0])
    if len(order) != len(adds): raise Unsupported("the uniform adds do not form one linear chain")
    for o in order[:-1]:
        users = [u for u in ops if any(x is o.dest for x in u.args)]
        if len(users) != 1: raise Unsupported("a folded partial sum (%s) is used outside the chain; only the final sum exists in main" % o.dest.name)
    base = [x for x in firsts[0].args if not is_term(x)][0]
    _FOLD[0] = dict(terms=tuple(terms), skip=frozenset(order[:-1]), root={order[-1]: base})


def _written_slots(fn):
    return {o.args[0].slot for blk in fn.blocks for o in blk.ops if o.kind in ("store", "store_fetch", "store_vec4_at") and isinstance(o.args[0], ir.Buffer)}


def compile_function(fn, regs=range(4, 16), name=None, resolved=None,
                     fma_always_load_wait=False, resource_projection=None, guards=None):
    """Compile `fn`, then run the compile-time GUARDS (agxforge/g17/guards.py, MM 25.144.6) on the result: no
    unread operation in the IR, and no register read after its release on the emitted bytes. `guards` is
    "refuse" / "warn" / "off" (default: G17_CC_GUARDS, else guards.DEFAULT_MODE); the guards never change
    the emitted bytes. A refusal is an Unsupported.

    `fma_always_load_wait` is the paired FMA waiting control's one degree of freedom."""
    program = _compile_function_unguarded(fn, regs=regs, name=name, resolved=resolved,
                                          fma_always_load_wait=fma_always_load_wait,
                                          resource_projection=resource_projection)
    from agxforge.g17 import guards as _guards
    try:
        _guards.check(fn, program, guards)
    except _guards.GuardRefused as why:
        raise Unsupported("compile guard: %s" % why)
    return program


def _compile_function_unguarded(fn, regs=range(4, 16), name=None, resolved=None,
                                fma_always_load_wait=False, resource_projection=None):
    """`fma_always_load_wait` is the paired FMA waiting control's one degree of freedom.

    Default false, so every delivered program's bytes are unchanged. True requests op2190's
    ALREADY-ADMITTED load-wait state (byte0[3] set) on every FMA, including the chained ones that
    normally clear it, so an experiment can hold a program fixed and move only that field. It is a
    keyword rather than a module global the caller assigns, for two reasons root named: a delivery
    should be able to record the option it requested, and the selection state must be restored even
    when the compile REFUSES - otherwise one Unsupported leaves the next compile silently altered.
    """
    _AND16_WAITED.clear()     # a waited copy belongs to one function's selection
    if resource_projection is not None:
        if resource_projection != "unused-device-buffer-v1":
            raise Unsupported("unknown resource projection mode %r" % (resource_projection,))
        from .projection import compile_compact
        return compile_compact(fn, regs=regs, name=name, resolved=resolved,
                               fma_always_load_wait=fma_always_load_wait)
    global _FMA_ALWAYS_LOAD_WAIT
    _saved_fma_wait = _FMA_ALWAYS_LOAD_WAIT
    _FMA_ALWAYS_LOAD_WAIT = bool(fma_always_load_wait)
    # THE COUNTED KEY-BLOCK LOOP (MM 25.114.5) is admitted only if its emitted bytes pass the static
    # latch check: decoded, not reasoned about. A loop that fails it is refused here, before any
    # caller can author or dispatch it (a runaway loop has rebooted this machine, 25.116).
    loop = (tensor_loop_route(fn) if any(o.kind == "tensor_matmul" for blk in fn.blocks for o in blk.ops)
            else None)
    try:
        program = _compile_function(fn, regs=regs, name=name, resolved=resolved)
    finally:
        _FMA_ALWAYS_LOAD_WAIT = _saved_fma_wait
    if loop is not None:
        from agxforge.g17 import tensorlife as _tensorlife
        try:
            program._tensor_loop = _tensorlife.counted_loop_check(bytes(program.code), loop["trips"],
                                                                  carried=tuple(_TENSOR_INDEX_USED),
                                                                  runtime=bool(loop.get("runtime")))
        except ValueError as why:
            raise Unsupported("tensor loop: the emitted loop fails the static latch check: %s" % why)
    return program


DEAD_LOAD_KINDS = ("load",)


def _drop_unread_loads(fn):
    """Remove every device `load` whose result nothing reads. -> how many were removed.

    AN UNREAD LOAD IS A RACE, NOT A NO-OP. It still writes its destination asynchronously, the
    allocator frees that register at once, and the load lands on whatever took it: online-softmax
    attention's four-word stat reads used one word, and rows 2 and 9 of 16 came back wrong on
    hardware (results/g17-tensor-stream-v1). A device load has no side effect (an out-of-range one
    returns 0 and never faults), so the load is removed, not waited for. Iterated, because an index
    computation may feed only dead loads. unread_load_destination_reuse still refuses whatever
    reaches emission this way."""
    removed = 0
    while True:
        used = {id(a) for blk in fn.blocks for o in blk.ops for a in o.args}
        used |= {id(a) for blk in fn.blocks if blk.term is not None for a in blk.term.args}
        dead = [(blk, o) for blk in fn.blocks for o in blk.ops
                if o.kind in DEAD_LOAD_KINDS and o.dest is not None and id(o.dest) not in used]
        if not dead:
            return removed
        for blk, o in dead:
            blk.ops.remove(o)
        removed += len(dead)


def _compile_function(fn, regs=range(4, 16), name=None, resolved=None):
    ir.verify(fn)
    _refuse_unvalidated_atomic_result(fn)
    ir.expand(fn)          # IR-level operations (erf) become selectable ops; a no-op otherwise
    ir.verify(fn)
    if _drop_unread_loads(fn):
        ir.verify(fn)
    if _strip_mine_for_compare_width(fn):
        ir.verify(fn)
    # THE REGISTER BUDGET IS THE ALLOCATOR'S POOLS LESS HEADROOM, computed from the same constants
    # Alloc uses so the two cannot drift apart. Recomputation chains need registers of their own.
    budget = len(list(regs)) + (WIDE_MAX - NARROW_MAX) - REMAT_HEADROOM
    # REMATERIALISATION FIRST, SPILLING ONLY WHERE IT RUNS OUT OF CANDIDATES. Recomputing costs no
    # memory traffic and is strictly better where it applies, so a program remat can handle must
    # keep the bytes it has.
    #
    # AND IT IS TRIED ON A COPY, because it does not fail cleanly. When it exhausts it has already
    # rewritten - and on the in-place pressure program its rewrites make things WORSE: it clones
    # each `add` to its far use, which does not shorten anything and instead holds the impure LOAD
    # under it live to that point. 91 live adds become 91 live loads, and the loads are the one
    # thing a spill may not take as a victim. Spilling has to start from the function as written.
    trial = _clone_straightline(fn)
    try:
        moved = _rematerialise_pressure(trial, budget)
    except Unsupported as exhausted:
        if "none can be recomputed" not in str(exhausted):
            raise
        # Remat had no candidate. Spill from the ORIGINAL, then let remat work on what is left.
        if _spill_pressure(fn, budget):
            ir.verify(fn)
            if _rematerialise_pressure(fn, budget):
                _eliminate_dead_pure(fn)
                ir.verify(fn)
        else:
            raise
    else:
        fn.blocks, fn.buffers = trial.blocks, trial.buffers
        if moved:
            _eliminate_dead_pure(fn)
            ir.verify(fn)
        # Under budget after remat for every program that ever compiled; a no-op there.
        if _spill_pressure(fn, budget):
            ir.verify(fn)
    selected = select(fn)
    binding_ranks = dict(_BUF_RANK[0])
    if getattr(fn, "compact_registers", False):
        import copy as _copy
        _COMPACT[0] = True
        try:
            insts = Alloc(regs).run(_copy.deepcopy(selected))
        except Unsupported:
            _COMPACT[0] = False
            insts = Alloc(regs).run(selected)
        finally:
            _COMPACT[0] = False
    else:
        insts = Alloc(regs).run(selected)
    _promote_wide_bitwise(insts)
    _check_tensor_body_clobbers(insts)
    # THE LIFETIME OPERANDS ARE CHECKED AGAINST THE FINAL INSTRUCTION LIST, SOURCE-OWNED.
    #
    # Two executed pairs say what is at stake. halfzero's store released its index register and the
    # next instruction read it: 31 of 32 lanes returned the fill value, and keeping it (one byte)
    # passed. syn-s7f595f1cd1's repair changed a CONVERSION source lifetime and two store VALUE
    # lifetimes - four bytes, 5cf5f15a failed on hardware and 103e5b09 passed. So a lifetime that
    # does not match liveness is a wrong program, not a style question.
    #
    # Liveness runs inside Alloc, and passes run AFTER it - _promote_wide_bitwise here, and
    # rematerialisation and spilling earlier - so the risk is a lifetime that was right when it was
    # written and is stale by emission. This recomputes liveness over the final list and compares.
    #
    # IT USES NO DECODER AND SPAWNS NO PROCESS. An earlier version of this check called the packed
    # checker, which shells out to the vendor reference binary - ordinary compilation must not do
    # that, and root's review said so. The byte-level check with its register-alias model lives in
    # g17asm.read_after_release for the review harness and the tests, where a subprocess is fine.
    stale = _stale_lifetimes(insts)
    if stale:
        i, form, field, was, now = stale[0]
        raise Unsupported(
            "instruction %d (%s) carries lifetime %s=%r but liveness over the final instruction "
            "list gives %r%s. A pass after register allocation changed who reads what; emitting "
            "this would release a register another instruction still reads, which is the defect "
            "results/g17-halfzero-runtime-v1 and the syn-s7f595f1cd1 negative both executed."
            % (i, form, field, was, now, "" if len(stale) == 1 else " (and %d more)" % (len(stale) - 1)))
    code, layout = emit(insts)
    waw = unread_load_destination_reuse(layout)
    if waw:
        raise Unsupported("%d load(s) write a destination register another instruction redefines before "
                          "the loaded value is read - e.g. the load at +0x%x and the write at +0x%x into r%d. "
                          "A dead load's asynchronous write races the next one (12,261 wrong outputs at "
                          "32x384; online-softmax rows 2 and 9). Nothing may emit an unread load"
                          % (len(waw), waw[0][0], waw[0][1], waw[0][2]))
    refuse_tensor_sharing(fn, layout)          # MM P12, named; before the undeclared-scratchpad check
    if _uses_threadgroup_memory(layout) and getattr(fn, "threadgroup", None) is None:
        raise Unsupported("the program uses threadgroup memory and declares no scratchpad: call "
                          "Function.declare_threadgroup(words, size) - the compiler cannot bound a "
                          "register-indexed scratchpad from the program, so the requirement is stated")
    bad = selfcheck(layout) + _check_flag_discipline(layout)
    if bad: raise AssertionError("generated code does not decode back:\n  " + "\n  ".join(bad))
    inherited = _inherited_bits(layout)
    for _at, _bb, m in layout:
        if m.form == "tensor.inherited":      # every bit of an inherited instruction is inherited, and says so
            inherited["tensor.inherited.op%d.%d" % (m.fields["opcode"], m.size)] = [(b, i) for b in range(m.size) for i in range(8)]
    # THE BUFFERS TRAVEL WITH THE PROGRAM. The linker needs each binding's index, pointer-block
    # offset, access flag and ELEMENT TYPE, and every one of those is a fact about the function
    # rather than about the bytes - a program that has forgotten its own signature cannot state
    # them, which is why the ABI could not be built until they were carried.
    written = {b.slot for blk in fn.blocks for o in blk.ops
               if o.kind in ("store", "store_at", "store_range", "store_fetch", "store_vec4_at", "atomic_rmw", "atomic_add",
                             "atomic_uniform")
               for b in o.args[:1] if isinstance(b, ir.Buffer)}
    written |= {o.args[2].slot for blk in fn.blocks for o in blk.ops
                if o.kind == "tensor_matmul" and isinstance(o.args[2], ir.Buffer)}    # C is the matmul's written buffer
    _requant_facts = [o.attrs.get("requantization") for blk in fn.blocks for o in blk.ops
                      if o.kind == "store_at" and o.attrs.get("requantization") is not None]
    if len(_requant_facts) > 1 and any(f != _requant_facts[0] for f in _requant_facts[1:]):
        raise Unsupported("a function contains multiple incompatible requantization contracts")
    _requantization = dict(_requant_facts[0]) if _requant_facts else None
    # THE RANKS ARE SNAPSHOT HERE, NOT READ LATER. _BUF_RANK[0] is set by select(fn) and holds
    # only the function compiled most recently, so an abi() that read it produced binding offsets
    # belonging to whatever was compiled last. Two programs alive at once is all it takes.
    # THE RESOLVED LAYOUT IS CONSUMED, NOT ASSUMED: with a preload, the layout the caller passed (the linker's, or
    # this call's own resolution when none was passed) must be the layout of THIS declaration - a stale one, resolved
    # for another binding list, refuses before any byte is trusted (integration's 29644d9c control)
    # THE RESOLVED LAYOUT IS CONSUMED, NOT ASSUMED, AND CHECKED BY ITS OWNER: the layout the caller passed (the
    # linker's, or this call's own resolution when none was passed) is carried whole into the contract, whose
    # construction runs the linker's check_resolved against the delivered binding list - a stale one, resolved for
    # another declaration, refuses before any byte is trusted (integration's 29644d9c control). No field- or
    # digest-comparison of this side's own is kept beside it (c54a7466).
    resolved = (resolved_layout(fn) if resolved is None else dict(resolved)) if _PRELOADS[0] else None
    _spill_slot = getattr(fn, "spill_slot", None)
    _spill_groups = getattr(fn, "spill_groups", 0)
    program = G17Program(name or fn.name, code, layout, inherited=inherited, threadgroup=getattr(fn, "threadgroup", None), preloads=list(_PRELOADS[0]), resolved_layout=resolved,
                         buffers=list(fn.buffers), written=written, binding_ranks=binding_ranks,
                         requantization=_requantization)
    program._spill_slot, program._spill_groups = _spill_slot, _spill_groups
    if resolved is not None:
        try: program.contract()
        except ValueError as e: raise Unsupported("stale or tampered resolved layout: %s" % e) from e
    return program
