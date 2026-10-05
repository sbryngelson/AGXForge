#!/usr/bin/env python3
"""THE END-TO-END PROOF: one kernel, two compilers, the same numbers.

Every other measure in this project is assembled - 54 opcodes proven one at a time, forms certified
both directions, an image whose provenance census says nothing is inherited. This is the one number
that does not need assembling: take a kernel someone would actually write, compile it with THIS
backend and with Apple's, run both on identical inputs, and compare every output element.

    python3 tools/g17endtoend.py             every kernel, both compilers, diffed
    python3 tools/g17endtoend.py <kernel>    one kernel
    python3 tools/g17endtoend.py g17|metal <kernel> <out>   one side (each needs its own process)
    python3 tools/g17endtoend.py --compiled-now [--write|--check]   today's bytes, NO dispatch
    python3 tools/g17endtoend.py --record <out.json> <kernel>...    partial re-run, bytes retained

WHY MORE THAN ONE KERNEL. The first one found that only alu.12 ever waited for a load; the fix
touched twelve lowerings across five ALU families and exactly ONE of them - mul after a load - was
verified on silicon. The rest were fixed by inspection. A hazard is a property of a PAIR of
instructions, so the only way to test the other four families is a kernel that pairs each of them
with a load.

ONE PIPELINE PER PROCESS. ac_pipeline_from_archive refuses a second, because Metal would hand back
the cached unpatched one - so the two sides run as children and the parent compares their output.

THE KERNEL is a fused linear-plus-threshold, one output element per thread: four strided loads, four
multiplies by constant weights, a summation tree, a bias, and a branch-free threshold. That is the
inner shape of an inference layer, it is loop-free, and every opcode in it is execution-proven.

WHAT IS DELIBERATELY ABSENT, and why, so the list of kernels is not mistaken for the list of things
that could be tested:

    a LOOP / BACK EDGE      Two reasons and either is enough. The back branch's semantics are not
                            recovered - it is the one ladder program of 45 that does not compile -
                            and dispatching Apple's own looping kernels with foreign buffer
                            contents wedged this GPU and rebooted the machine on 2026-09-05, so
                            back edges are refused before dispatch by standing rule. A loop kernel
                            is the single most valuable one missing, and it is blocked on the ISA
                            side, not on this file.

    MIXED SCALAR + TENSOR   tensor.seq carries 5,696 bits the compiler declares it inherits whole -
                            Apple's MAC sequence, its order and its accumulator one-hot. A kernel
                            mixing it with scalar work would be comparing two compilers on bytes
                            one of them copied from the other, which is not the comparison this
                            file exists to make.

    TRUE 64-BIT ADDRESSING  no buffer here is large enough to carry an index past 2^32. See
                            _wideindex_ir for what is tested instead, and what that does not cover.
"""
import ctypes, json, os, subprocess, sys
import numpy as np

T = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(T, "tools"))
sys.path.insert(0, os.path.join(T, "spike", "accel", "re"))

N = 64                       # buffers are N*N elements, as everything else in this harness assumes
THREADS = 32
PRESSURE_N = 20        # live values in the `pressure` kernel; the narrow pool holds 12
W = (3, 5, 7, 11)            # weights, folded into the multiply immediates
BIAS = 100
THRESH = 5000
SCRATCH = os.path.expanduser("~/.cache/agxforge/agx/e2e")

_HDR = """
#include <metal_stdlib>
using namespace metal;
kernel void k(device const uint *A [[buffer(0)]], device const uint *B [[buffer(1)]],
              device uint *C [[buffer(2)]], uint t [[thread_position_in_grid]],
              uint tt [[thread_position_in_threadgroup]],
              uint tgp [[threadgroup_position_in_grid]]) {
%s
}
"""


# THE ATOMIC SIGNATURE. The default header declares A as `device const uint *`, which cannot be
# the target of an atomic. Same buffers, same order, A retyped.
_HDR_ATOMIC = """
#include <metal_stdlib>
using namespace metal;
kernel void k(device atomic_uint *A [[buffer(0)]], device const uint *B [[buffer(1)]],
              device uint *C [[buffer(2)]], uint t [[thread_position_in_grid]],
              uint tt [[thread_position_in_threadgroup]],
              uint tgp [[threadgroup_position_in_grid]]) {
%s
}
"""
ATOMIC_KERNELS = {"atomicadd", "waveagg", "wavebcast"}

# KERNELS DECLARED NOT WORKING, with the reason, rather than left to fail. Empty, and the entry
# that used to sit here is worth keeping as a warning: atomicadd was declared INERT - "it writes
# neither memory nor its destination register" - and that was wrong twice over. The atomic worked
# the whole time. What was broken was the OBSERVABLE: ac_run_ps_es uploads buffer 0 and never
# copies it back, so a kernel judged by reading buffer 0 reads its fill value forever. And the
# second reading, that a per-lane form at a uniform address is inert, was also wrong - it does not
# SERIALISE, so every lane legitimately reads 0.
#
# ledger/g17-the-atomic-works-and-the-observable-did-not.toml, and memory:check-the-observable-exists
PENDING = set()

# WHICH LANE GETS WHICH OLD VALUE IS A RACE, and that is the point rather than a defect: thirty-two
# lanes incrementing one counter must between them observe every prior value exactly once. The
# ORDER is not part of the contract and no compiler owes it, so this kernel is compared as a
# multiset. Every other kernel stays order-sensitive.
UNORDERED = set()


def _atomicadd_ir():
    """AN ATOMIC READ-MODIFY-WRITE THAT IS ACTUALLY OBSERVABLE.

    A[t] += t+1 on a PER-LANE address, then read A[t] back and store it to C. C is the only buffer
    the harness copies back - ac_run_ps_es uploads buffer 0 and never returns it - so a kernel whose
    only effect is on A cannot be checked at all, and the first version of this kernel was scored
    against an array that had never been read. See
    ledger/g17-the-atomic-works-and-the-observable-did-not.toml.

    WHAT THIS PROVES AND WHAT IT DOES NOT. C[t] == t+1 proves the atomic wrote MEMORY, at the right
    address, with the right addend, per lane. It does NOT prove serialisation: every lane touches
    its own slot, so nothing contends. The counter test - 32 lanes on ONE address, returns exactly
    {0..31} - needs the lane election Apple emits (op10372, op582, op10094, op577, then op14157 and
    op10283 to derive each lane's value) and is the next capability, not this one.

    The addend is computed rather than literal: a literal sets operand 1's 2^24 bit, which the
    operand map cannot tell from 2^25, and selection refuses it.
    """
    import g17ir as ir
    f = ir.Function("atomic_counter", [ir.Buffer("A", 0), ir.Buffer("C", 2)])
    b = ir.Builder(f, f.block("entry"))
    t = b.builtin("thread_position_in_grid", name="t")
    inc = b.add(b.mul(t, ir.Imm(1), name="m"), ir.Imm(1), name="inc")
    b.atomic_add(f.buffers[0], t, inc, name="old")
    v = b.load(f.buffers[0], t, name="v")
    b.store_at(f.buffers[1], t, b.add(v, ir.Imm(0), name="w"))
    b.ret()
    return f


def _atomicadd_py(B):
    """A[t] starts at zero and gains t+1, and C[t] is what A[t] then holds."""
    return [t + 1 for t in range(THREADS)]


def _waveagg_ir():
    """THE WAVE-AGGREGATED ATOMIC, Apple's idiom: elect one lane, add the group's total, redistribute.

        total  = TVSIMD                 active lanes, from the high half of the vote pair
        if PVSIMD == 0:  atomic(A, total)    ONE lane applies the whole group's contribution
        C[t]   = A[0] + PVSIMD          each lane's own share, from the low half

    Thirty-two lanes, one counter, A[0] ends at 32 and C[t] is exactly t. The election is not an
    optimisation: without it every lane adds `total` and the counter reaches 1024, which is what
    this kernel's ungated form measured.
    """
    import g17ir as ir
    f = ir.Function("wave_agg", [ir.Buffer("A", 0), ir.Buffer("C", 2)])
    e = f.block("entry"); one = f.block("one"); join = f.block("join")
    b = ir.Builder(f, e)
    t = b.builtin("thread_position_in_grid", name="t")
    w = b.simd_vote_pair(name="w")
    total = b.shr(w, ir.Imm(16), name="total")
    prefix = b.shr(b.shl(w, ir.Imm(16), name="lo"), ir.Imm(16), name="prefix")
    b.br_cond(b.cmp(prefix, 1, "lt", name="p"), one, join)
    b.at(one)
    b.atomic_uniform("add", f.buffers[0], total, name="unused")
    b.br(join)
    b.at(join)
    z = b.mul(t, ir.Imm(0), name="z")
    v = b.load(f.buffers[0], z, name="v")
    b.store_at(f.buffers[1], t, b.add(v, prefix, name="mine"))
    b.ret()
    return f


def _waveagg_py(B):
    """The elected lane adds THREADS; every lane reads it back and adds its own prefix."""
    return [THREADS + t for t in range(THREADS)]


def _wavebcast_ir():
    """APPLE'S IDIOM WHOLE, including the broadcast: the elected lane's OLD VALUE reaches every lane.

    The aggregation kernel reads the counter back through a load, which proves memory changed but
    not that the atomic's RESULT is delivered. This takes the other path - op14157, simd shuffle -
    so the returned old value travels from the elected lane to all of them.

    TWO ROUNDS, because one round on a zeroed counter gives old = 0, which is also what a register
    nobody wrote holds. The second round sees 32, so a lane that genuinely received the broadcast
    stores 32 + t and a lane holding an unwritten register stores t: thirty-two apart instead of
    identical. A[0] ends at 64.
    """
    import g17ir as ir
    f = ir.Function("wave_bcast", [ir.Buffer("A", 0), ir.Buffer("C", 2)])
    e = f.block("entry"); one = f.block("one"); join = f.block("join")
    b = ir.Builder(f, e)
    t = b.builtin("thread_position_in_grid", name="t")
    w = b.simd_vote_pair(name="w")
    total = b.shr(w, ir.Imm(16), name="total")
    prefix = b.shr(b.shl(w, ir.Imm(16), name="lo"), ir.Imm(16), name="prefix")
    b.br_cond(b.cmp(prefix, 1, "lt", name="p"), one, join)
    b.at(one)
    b.atomic_uniform("add", f.buffers[0], total, name="first")
    old = b.atomic_uniform("add", f.buffers[0], total, name="old")
    b.br(join)
    b.at(join)
    bc = b.simd_broadcast_first(old, name="bc")
    b.store_at(f.buffers[1], t, b.add(bc, prefix, name="mine"))
    b.ret()
    return f


def _wavebcast_py(B):
    """Second round sees the counter at THREADS, so every lane stores THREADS + its prefix."""
    return [THREADS + t for t in range(THREADS)]


def _tgatomic_ir():
    """A THREADGROUP ATOMIC COUNTER - op11765, and the difference of two rounds so the initial
    contents of threadgroup memory cannot matter.

    op11765 carries NO address operand: a destination, a modifier and a value. It names one
    implicit location per threadgroup, which is what a threadgroup counter is - and it is the
    twin of the device uniform form, so the same election applies.

    THREADGROUP MEMORY IS NOT ZERO AT ENTRY, and nothing promises it is. So this stores
    `second - first` rather than `first`: the difference is exactly the group total whatever the
    location held, which turns a kernel that depends on an unpromised initial value into one that
    does not. C[t] is THREADS + t.

    THIS KERNEL DOES NOT TEST WHICH OPERATION op11765 PERFORMS, measured 2026-09-08 and recorded
    here because it looked like it did for weeks. Substituting `and`'s operand-2 code for `sub`'s
    leaves its output bit-identical at 32 + t, and so does substituting `add`'s - three operations,
    one answer. Whatever the difference of two rounds is measuring here, it is not the operation,
    and the earlier reading "what the form does is SUBTRACT" rested on this kernel. `tgops` is the
    one that discriminates: forcing every operation to max's code moves it from 88 to 200.

    The elected lane does both rounds; op14157 carries the difference to every lane.
    """
    import g17ir as ir
    # TWO BUFFERS, C SECOND, like every other kernel here - C alone gets rank 0 and the address
    # const that goes with it, and the store lands on a binding the harness never reads back.
    f = ir.Function("tg_atomic", [ir.Buffer("B", 1), ir.Buffer("C", 2)])
    f.declare_threadgroup(THREADS, size=(THREADS, 1, 1))      # executed with one group of 32 lanes; declared now (ABI v4)
    e = f.block("entry"); one = f.block("one"); join = f.block("join")
    b = ir.Builder(f, e)
    t = b.builtin("thread_position_in_grid", name="t")
    w = b.simd_vote_pair(name="w")
    total = b.shr(w, ir.Imm(16), name="total")
    prefix = b.shr(b.shl(w, ir.Imm(16), name="lo"), ir.Imm(16), name="prefix")
    # A REAL THREADGROUP STORE ALONGSIDE, to test whether the ALLOCATION is what is missing: the
    # atomic form names an implicit location, and if the threadgroup region is sized from the
    # load/store opcodes then an atomic-only kernel has nowhere to write.
    b.store_tg(total, t)
    b.barrier()
    b.br_cond(b.cmp(prefix, 1, "lt", name="p"), one, join)
    b.at(one)
    o1 = b.atomic_tg_uniform("sub", total, name="o1")
    o2 = b.atomic_tg_uniform("sub", total, name="o2")
    d = b.sub(o1, o2, name="d")
    b.br(join)
    b.at(join)
    b.store_at(f.buffers[1], t, b.add(b.simd_broadcast_first(d, name="bc"), prefix, name="mine"))
    b.ret()
    return f


def _tgops_ir():
    """ALL SEVEN THREADGROUP ATOMIC OPERATIONS IN ONE KERNEL, chained so no single operation
    repeated seven times can reach the same answer.

    op11765's operation is operand 2, a four-bit table key - byte4[5], byte5[3], byte6[3],
    byte11[4] - recovered by flipping one bit at a time in Apple's `add` witness. This kernel is
    the execution side of that claim, and the constants are chosen so that it can FAIL: forcing
    every operation to one code has to change the number, or the kernel is not testing the field.

    THE FIRST VERSION DID NOT TEST IT. Its chain was and 0, or 5, xor 3, max 9, min 7, add 100,
    sub 7 and it landed on 100 - which is also exactly what SEVEN MAXES land on, because the chain
    was monotone and 100 was its largest constant. Forcing every operation to max's code passed it
    unchanged. A kernel whose answer a degenerate hypothesis reproduces is not evidence.

    THREADGROUP MEMORY IS NOT ZERO AT ENTRY, so the chain opens by ANDing zero in - the only step
    whose result does not depend on what was there. After that:

        and 0    -> 0        add 200 -> 200     or 7   -> 207     xor 200 -> 7
        max 100  -> 100      min 99  -> 99      sub 11 -> 88      add 0 returns 88

    and the seven constant-operation hypotheses give 0, 617, 239, 11, 200, 0 and a wrap - none of
    them 88. The constants stay under 256 because the ALU form that materialises them carries an
    eight-bit immediate. The trailing `add 0` is there because an atomic returns the value it FOUND, so the last
    operation in a chain never checks itself.
    """
    import g17ir as ir
    f = ir.Function("tg_ops", [ir.Buffer("B", 1), ir.Buffer("C", 2)])
    f.declare_threadgroup(THREADS, size=(THREADS, 1, 1))      # executed with one group of 32 lanes; declared now (ABI v4)
    e = f.block("entry"); one = f.block("one"); join = f.block("join")
    b = ir.Builder(f, e)
    t = b.builtin("thread_position_in_grid", name="t")
    w = b.simd_vote_pair(name="w")
    total = b.shr(w, ir.Imm(16), name="total")
    prefix = b.shr(b.shl(w, ir.Imm(16), name="lo"), ir.Imm(16), name="prefix")
    b.store_tg(total, t)
    b.barrier()
    # THE ADDENDS IN REGISTERS. op11765 has no immediate addend, and each constant gets its own
    # register rather than sharing one: the source lifetime is an OPERAND on this family and a
    # register released after its first read is how an authored G17 program silently reads zero.
    z = getattr(b, "and")(t, ir.Imm(0), name="z")
    ks = [b.add(z, ir.Imm(k), name="k%d_%d" % (i, k))
          for i, k in enumerate((0, 200, 7, 200, 100, 99, 11, 0))]
    b.br_cond(b.cmp(prefix, 1, "lt", name="p"), one, join)
    b.at(one)
    for op, k in zip(("and", "add", "or", "xor", "max", "min", "sub"), ks):
        b.atomic_tg_uniform(op, k, name="o_" + op)
    d = b.atomic_tg_uniform("add", ks[7], name="d")
    b.br(join)
    b.at(join)
    b.store_at(f.buffers[1], t, b.add(b.simd_broadcast_first(d, name="bc"), prefix, name="mine"))
    b.ret()
    return f


def _tgops_py(B):
    """The chain lands on 88 whatever the threadgroup location held, so lane t stores 88 + t."""
    return [88 + t for t in range(THREADS)]


def _tgatomic_py(B):
    """The difference of two rounds is the group total, so lane t stores THREADS + t."""
    return [THREADS + t for t in range(THREADS)]


def _votepair_ir():
    """SR_PVSIMD and SR_TVSIMD read into the two 16-bit halves of ONE register.

    THIS EXISTS TO TEST AN INFERENCE. The imageblock work established that read_sr can write a
    16-bit half and that Apple uses two half-register files, based at 425 and 281, at one index -
    for the imageblock coordinate and again for a wave-aggregated atomic's two vote registers.
    That those files are the LOW and HIGH halves of the 32-bit register at that index is a reading
    of the encoding and has never been executed. If it holds, C[t] is (active << 16) | prefix.
    """
    import g17ir as ir
    # TWO BUFFERS, C SECOND, like every other kernel here. Declaring C alone gives it rank 0 and
    # the address const that goes with rank 0, so the store lands on a different binding than the
    # one the harness reads back - which presents exactly as a kernel that ran, faulted nothing,
    # and wrote nothing.
    f = ir.Function("votepair", [ir.Buffer("B", 1), ir.Buffer("C", 2)])
    b = ir.Builder(f, f.block("entry"))
    t = b.builtin("thread_position_in_grid", name="t")
    b.store_at(f.buffers[1], t, b.simd_vote_pair(name="v"))
    b.ret()
    return f


def _votepair_py(B):
    """All THREADS lanes active in one simdgroup: prefix is the lane, total is THREADS."""
    return [(THREADS << 16) | t for t in range(THREADS)]


def _linear_ir():
    """A fused linear-plus-threshold: four strided loads, weighted multiplies, bias, threshold."""
    import g17ir as ir
    f = ir.Function("linear_threshold", [ir.Buffer("B", 1), ir.Buffer("C", 2)])
    b = ir.Builder(f, f.block("entry"))
    t = b.builtin("thread_position_in_grid", name="t")
    base = b.mul(t, ir.Imm(4), name="base")
    acc = None
    for j, w in enumerate(W):
        idx = base if j == 0 else b.add(base, ir.Imm(j), name="i%d" % j)
        v = b.load(f.buffers[0], idx, name="a%d" % j)
        # the multiply is also what makes the later store legal: an ALU op carries the load-wait
        # that op17229's eight-byte form does not have
        m = b.mul(v, ir.Imm(w), name="m%d" % j)
        acc = m if acc is None else b.add(acc, m, name="s%d" % j)
    s = b.add(acc, ir.Imm(BIAS), name="s")
    r = b.csel(s, b.const(THRESH, name="k"), s, b.const(0, name="z"), rel="gt", name="r")
    b.store_at(f.buffers[1], t, r)
    b.ret()
    return f


def _linear_py(B):
    out = []
    for t in range(THREADS):
        s = (sum(int(B[4 * t + j]) * W[j] for j in range(4)) + BIAS) & 0xFFFFFFFF
        out.append(s if s > THRESH else 0)
    return out


def _bitops_ir():
    """EACH OF THE FOUR UNVERIFIED FAMILIES CONSUMING A LOAD DIRECTLY.

    The load-wait fix touched mul, sub, the shifts, the bitwise pair and the saturating pair, and
    only mul was checked on silicon. Here the shift, the bitwise-and and the subtract each take the
    loaded value as their own operand, so any family still not waiting shows as a wrong answer.
    """
    import g17ir as ir
    f = ir.Function("bitops", [ir.Buffer("B", 1), ir.Buffer("C", 2)])
    b = ir.Builder(f, f.block("entry"))
    t = b.builtin("thread_position_in_grid", name="t")
    v = b.load(f.buffers[0], t, name="v")
    sh = b.shl(v, ir.Imm(3), name="sh")                 # alu.shift.imm <- straight off the load
    an = getattr(b, "and")(v, ir.Imm(0xFF), name="an")  # bitwise.imm   <- straight off the load
    su = b.sub(v, ir.Imm(5), name="su")                 # alu.sub.imm   <- straight off the load
    mu = b.mul(v, ir.Imm(3), name="mu")                 # alu.mul.imm   <- the one already proven
    # combined with adds (alu.12); a register-register xor would select op17771, which cannot be
    # certified - see bitwise_registers in the ladder
    r = b.add(b.add(sh, an, name="p"), b.add(su, mu, name="q"), name="r")
    b.store_at(f.buffers[1], t, r)
    b.ret()
    return f


def _bitops_py(B):
    out = []
    for t in range(THREADS):
        v = int(B[t])
        r = (((v << 3) & 0xFFFFFFFF) + (v & 0xFF)
             + ((v - 5) & 0xFFFFFFFF) + ((v * 3) & 0xFFFFFFFF)) & 0xFFFFFFFF
        out.append(r)
    return out


def _satops_ir():
    """The saturating family taking a load directly - the fifth family the fix touched.

    addsat/subsat are the one branch of the load-wait fix that bitops does not reach, and alu.sat
    is a different encoder path again (encode_alu_form with an immediate shift amount).
    """
    import g17ir as ir
    f = ir.Function("satops", [ir.Buffer("B", 1), ir.Buffer("C", 2)])
    b = ir.Builder(f, f.block("entry"))
    t = b.builtin("thread_position_in_grid", name="t")
    v = b.load(f.buffers[0], t, name="v")
    big = b.const(0xFFFFFF00, name="big")
    a = b.addsat(v, big, name="a")          # saturates for all but the smallest inputs
    c = b.subsat(v, b.const(4000, name="k"), name="c")
    r = b.add(a, c, name="r")
    b.store_at(f.buffers[1], t, r)
    b.ret()
    return f


def _satops_py(B):
    out = []
    for t in range(THREADS):
        v = int(B[t])
        a = min(v + 0xFFFFFF00, 0xFFFFFFFF)
        c = max(v - 4000, 0)
        out.append((a + c) & 0xFFFFFFFF)
    return out


def _floatmath_ir():
    """FLOAT ARITHMETIC STRAIGHT OFF A LOAD, including the three-source ffma.

    op998 fadd, op3290 fmul and op2190 ffma all have measured functions but none had ever been run
    inside a real kernel against Apple's own compilation of the same arithmetic. ffma also goes
    through the GENERIC AUTHORING path rather than any ALU family, so whether it carries a load-wait
    is a separate question from the twelve lowerings that were fixed.
    """
    import g17ir as ir
    f = ir.Function("floatmath", [ir.Buffer("B", 1), ir.Buffer("C", 2)])
    b = ir.Builder(f, f.block("entry"))
    t = b.builtin("thread_position_in_grid", name="t")
    v = b.load(f.buffers[0], t, name="v")
    m = b.fmul(v, v, name="m")            # op3290, straight off the load
    a = b.fadd(m, v, name="a")            # op998
    r = b.fma(v, v, a, name="r")          # op2190, three registers, generic authoring path
    b.store_at(f.buffers[1], t, r)
    b.ret()
    return f


def _floatmath_py(B):
    import struct
    out = []
    for t in range(THREADS):
        x = struct.unpack("<f", struct.pack("<I", int(B[t])))[0]
        m = np.float32(x) * np.float32(x)
        a = np.float32(m) + np.float32(x)
        r = np.float32(np.float32(x) * np.float32(x) + np.float32(a))
        out.append(struct.unpack("<I", struct.pack("<f", float(r)))[0])
    return out


def _bitreg_ir():
    """THE REGISTER-REGISTER BITWISE FORM, straight off two loads.

    The ISA peer recovered a ten-byte layout for op424/op13575/op17771 and asked for the half they
    cannot do - running one against Apple's compiler. This is the FOUR-byte form, which this project
    already modelled (BITWISE_REG_FORM, encode_bitwise_reg, decode_bitwise_reg) and could not emit
    because selfcheck routed it to the wrong decoder. Both sources come straight from loads, so the
    form's hazard behaviour is under test as well as its operand layout.
    """
    import g17ir as ir
    f = ir.Function("bitreg", [ir.Buffer("B", 1), ir.Buffer("C", 2)])
    b = ir.Builder(f, f.block("entry"))
    t = b.builtin("thread_position_in_grid", name="t")
    # THE ALU HOP IS NOT DECORATION. The four-byte form does not wait for a load - measured: this
    # kernel without the two adds returns zero for every thread, and with them returns the right
    # answer, which is also the first execution evidence that its operand layout is right.
    # DISJOINT OPERANDS PER OPERATION, and that is the finding rather than a workaround: this form
    # releases its sources, so two of these reading the same pair gives a wrong answer while two
    # reading different pairs match Apple exactly. Six loads, three operations, no register shared.
    def v(i, n):
        idx = t if i == 0 else b.add(t, ir.Imm(i), name="t%d" % i)
        return b.add(b.load(f.buffers[0], idx, name=n + "0"), ir.Imm(0), name=n)
    a = getattr(b, "and")(v(0, "x"), v(1, "y"), name="a")
    o = getattr(b, "or")(v(2, "u"), v(3, "w"), name="o")
    e = b.xor(v(4, "p"), v(5, "q"), name="e")
    r = b.add(b.add(a, o, name="s"), e, name="r")
    b.store_at(f.buffers[1], t, r)
    b.ret()
    return f


def _bitreg_py(B):
    out = []
    for t in range(THREADS):
        g = [int(B[t + j]) for j in range(6)]
        out.append(((g[0] & g[1]) + (g[2] | g[3]) + (g[4] ^ g[5])) & 0xFFFFFFFF)
    return out


def _branch_ir():
    """A CONDITIONAL WITH A PER-THREAD OBSERVABLE, which is what the ladder's rungs do not give.

    exec.mask suppressing a store was proved with three hand-built programs; a branching kernel has
    never been diffed against Apple's compilation of the same source. Every thread writes a default
    and the guarded threads overwrite it, so the output distinguishes taken from not-taken PER LANE
    rather than by one slot's presence. The relation must be `gt` - cmp.pair.imm's relation encoding
    is unresolved, so conditional_lt is a BLOCKED rung and this kernel would be too.
    """
    import g17ir as ir
    f = ir.Function("branch", [ir.Buffer("B", 1), ir.Buffer("C", 2)])
    e = f.block("entry"); th = f.block("then"); jn = f.block("join")
    b = ir.Builder(f, e)
    t = b.builtin("thread_position_in_grid", name="t")
    v = b.add(b.load(f.buffers[0], t, name="v0"), ir.Imm(0), name="v")
    b.store_at(f.buffers[1], t, b.add(v, ir.Imm(100), name="dflt"))
    b.br_cond(b.cmp(t, 8, "gt", name="p"), th, jn)
    b.at(th)
    b.store_at(f.buffers[1], t, b.mul(v, ir.Imm(3), name="hi"))
    b.br(jn)
    b.at(jn); b.ret()
    return f


def _branch_py(B):
    return [(int(B[t]) * 3 if t > 8 else int(B[t]) + 100) & 0xFFFFFFFF for t in range(THREADS)]


def _narrow_ir():
    """i16 THROUGH THE ALU, where the width actually matters.

    The three widths - src1, operand B, dest - are INDEPENDENT operands on this ISA, so a kernel
    whose 16-bit result must truncate is the one that says whether the backend writes all three or
    inherits any. The shift is chosen so the 16-bit value overflows: the inputs run to 4095 and a
    left shift of five carries them past 65535.
    """
    import g17ir as ir
    f = ir.Function("narrow", [ir.Buffer("B", 1), ir.Buffer("C", 2)])
    b = ir.Builder(f, f.block("entry"))
    t = b.builtin("thread_position_in_grid", name="t")
    v = b.add(b.load(f.buffers[0], t, name="v0"), ir.Imm(0), name="v", type=ir.I16)
    sh = b.shl(v, ir.Imm(5), type=ir.I16, name="sh")      # truncates at 16 bits
    w = b.add(sh, ir.Imm(1), type=ir.I32, name="w")       # then widened
    b.store_at(f.buffers[1], t, w)
    b.ret()
    return f


def _narrow_py(B):
    return [(((int(B[t]) << 5) & 0xFFFF) + 1) & 0xFFFFFFFF for t in range(THREADS)]


def _threadgroup_ir():
    """A CROSS-LANE EXCHANGE, which is the only shape that can catch a broken barrier.

    Every lane writes its own slot and then reads its NEIGHBOUR'S, so a lane's answer depends on an
    instruction another lane executed. The ladder's threadgroup rung sends a constant through one
    slot: that proves the store and the load, and it is deliberately lane-independent so the single
    output cannot depend on which lane wrote last. It therefore cannot see a barrier that does not
    order anything, or a threadgroup store that lands in the wrong lane's slot - both of which are
    still consistent with a constant coming back out.

    The neighbour is (t+1) & 31, so the exchange WRAPS: lane 31 reads lane 0. A missing barrier
    would leave the high lanes reading a slot their neighbour had not written, and the wrap makes
    lane 31 the loudest case rather than a boundary nobody looks at.
    """
    import g17ir as ir
    f = ir.Function("threadgroup", [ir.Buffer("B", 1), ir.Buffer("C", 2)])
    f.declare_threadgroup(THREADS, size=(THREADS, 1, 1))      # executed with one group of 32 lanes; declared now (ABI v4)
    b = ir.Builder(f, f.block("entry"))
    t = b.builtin("thread_position_in_grid", name="t")
    mine = b.add(b.load(f.buffers[0], t, name="v0"), ir.Imm(1), name="mine")
    b.store_tg(mine, t)
    b.barrier()
    nb = getattr(b, "and")(b.add(t, ir.Imm(1), name="t1"), ir.Imm(THREADS - 1), name="nb")
    got = b.load_tg(nb, name="got")
    b.store_at(f.buffers[1], t, b.mul(got, ir.Imm(2), name="out"))
    b.ret()
    return f


def _pressure_ir():
    """TWENTY VALUES LIVE AT ONCE - the wide register file, executed rather than compiled.

    Every load is issued before any of them is consumed, so at the last load twenty values are
    live and the allocator must reach past r15. Until the load's destination was widened from four
    bits to its measured seven, this kernel could not be built at all: the backend ran out at FOUR
    simultaneous live values with 110 wide registers sitting unused.

    COMPILING IS NOT EXECUTING, which is the whole reason this exists. A register field written
    wrong produces bytes that decode - the selfcheck sweep already proves they decode - and the
    question a dispatch answers is whether the hardware reads the register the encoding names.
    A wrong high bit reads a DIFFERENT register, which on this shape gives a sum missing one term
    or carrying a stale one, and both are numbers Apple's own compilation does not produce.
    """
    import g17ir as ir
    f = ir.Function("pressure", [ir.Buffer("B", 1), ir.Buffer("C", 2)])
    b = ir.Builder(f, f.block("entry"))
    t = b.builtin("thread_position_in_grid", name="t")
    vals = [b.load(f.buffers[0], getattr(b, "and")(b.add(t, ir.Imm(i), name="i%d" % i),
                                                   ir.Imm(THREADS - 1), name="j%d" % i),
                   name="v%d" % i)
            for i in range(PRESSURE_N)]
    acc = vals[0]
    for k, v in enumerate(vals[1:]):
        acc = b.add(acc, v, name="a%d" % k)
    b.store_at(f.buffers[1], t, acc)
    b.ret()
    return f


def _pressure_py(B):
    return [sum(int(B[(t + i) % THREADS]) for i in range(PRESSURE_N)) & 0xFFFFFFFF
            for t in range(THREADS)]


def _tgself_ir():
    """ONE REGISTER AS BOTH THE VALUE AND THE INDEX, which is the shape that needs a MOVE.

    `tg[t] = t` hands the threadgroup store the same register twice, and that does not index: every
    lane then writes the base, measured, and Apple's allocator never does it - zero of 866 corpus
    instances. So the backend copies one of them first. That copy used to be `add reg, #0` at
    twelve bytes because op586 was not lowered; it is now the four-byte move Apple writes, and this
    kernel is what exercises it.

    THE EXCHANGE IS STILL WHAT IS CHECKED. Every lane writes its own slot with its own id and reads
    its neighbour's, wrapping, so a copy that dropped the value, released it, or wrote the base
    would change the answer in a way a constant could not hide.
    """
    import g17ir as ir
    f = ir.Function("tgself", [ir.Buffer("B", 1), ir.Buffer("C", 2)])
    f.declare_threadgroup(THREADS, size=(THREADS, 1, 1))      # executed with one group of 32 lanes; declared now (ABI v4)
    b = ir.Builder(f, f.block("entry"))
    t = b.builtin("thread_position_in_grid", name="t")
    b.store_tg(t, t)
    b.barrier()
    nb = getattr(b, "and")(b.add(t, ir.Imm(1), name="t1"), ir.Imm(THREADS - 1), name="nb")
    got = b.load_tg(nb, name="got")
    b.store_at(f.buffers[1], t, b.add(got, ir.Imm(100), name="out"))
    b.ret()
    return f


def _tgself_py(B):
    return [((t + 1) % THREADS + 100) & 0xFFFFFFFF for t in range(THREADS)]


def _threadgroup_py(B):
    return [((int(B[(t + 1) % THREADS]) + 1) * 2) & 0xFFFFFFFF for t in range(THREADS)]


# --- COMPOSITION, not coverage ----------------------------------------------------------------
#
# The first seven kernels each exercised a FAMILY. These exercise the joins between them, which is
# where every defect this project has found actually lived: the compare that released the store's
# index, the ALU families that did not wait for a load, the four-byte bitwise that freed its own
# sources. None of those is visible in one instruction, and none was found by adding an opcode.


def _ifelse_ir():
    """AN ELSE, AND A NEST INSIDE IT - two exec regions in sequence and one inside another.

    This backend does not lower a two-armed br_cond: `_lower_blocks` refuses a branch to a block
    that is neither the next one nor the region exit, so an else has to be a SECOND guarded region
    with the complementary predicate. That is worth exercising precisely because it doubles the
    number of mask pushes and pops, and the mask is a LIFO stack whose balance is the thing that
    goes wrong. Three regions, one of them nested, and every lane lands in exactly one of the four
    outcomes - so a mask that popped the wrong number of levels shows up as lanes in two of them.
    """
    import g17ir as ir
    f = ir.Function("ifelse", [ir.Buffer("B", 1), ir.Buffer("C", 2)])
    # THE BLOCKS ARE CREATED IN THE ORDER THEY MUST APPEAR. This backend lowers structured
    # fallthrough only - a then-block must immediately follow its br_cond - so a nested region's
    # blocks interleave with its parent's rather than being appended after them.
    e = f.block("entry")
    hi = f.block("hi")
    deep, deepj = f.block("deep"), f.block("deepj")
    hj = f.block("hij")
    lo, loj = f.block("lo"), f.block("loj")
    b = ir.Builder(f, e)
    t = b.builtin("thread_position_in_grid", name="t")
    v = b.add(b.load(f.buffers[0], t, name="v0"), ir.Imm(0), name="v")
    b.store_at(f.buffers[1], t, b.add(v, ir.Imm(1), name="base"))
    b.br_cond(b.cmp(t, 8, "gt", name="p"), hi, hj)
    b.at(hi)
    b.store_at(f.buffers[1], t, b.mul(v, ir.Imm(5), name="five"))
    b.br_cond(b.cmp(t, 20, "gt", name="q"), deep, deepj)
    b.at(deep); b.store_at(f.buffers[1], t, b.mul(v, ir.Imm(3), name="three")); b.br(deepj)
    b.at(deepj); b.br(hj)
    b.at(hj)
    b.br_cond(b.cmp(t, 9, "lt", name="r"), lo, loj)
    b.at(lo); b.store_at(f.buffers[1], t, b.add(v, ir.Imm(100), name="lo1")); b.br(loj)
    b.at(loj); b.ret()
    return f


def _ifelse_py(B):
    out = []
    for t in range(THREADS):
        v = int(B[t])
        r = v + 1
        if t > 8:
            r = v * 5
            if t > 20:
                r = v * 3
        if t < 9:
            r = v + 100
        out.append(r & 0xFFFFFFFF)
    return out


def _liveidx_ir():
    """THREE INDICES LIVE AT ONCE, each used twice and none of them last.

    The compare's source lifetime was found because ONE index had to survive ONE instruction. This
    is the general case: three computed indices, each read by a load and then again by the
    arithmetic, so every one of them has to survive its first reader. A backend that releases any
    source on any of those reads gets a zero into the sum, and the sum is checked against Apple's.
    """
    import g17ir as ir
    f = ir.Function("liveidx", [ir.Buffer("B", 1), ir.Buffer("C", 2)])
    b = ir.Builder(f, f.block("entry"))
    t = b.builtin("thread_position_in_grid", name="t")
    m = ir.Imm(THREADS - 1)
    i = getattr(b, "and")(b.add(t, ir.Imm(1), name="i0"), m, name="i")
    j = getattr(b, "and")(b.add(t, ir.Imm(2), name="j0"), m, name="j")
    k = getattr(b, "and")(b.add(t, ir.Imm(3), name="k0"), m, name="k")
    x = b.add(b.load(f.buffers[0], i, name="bi"), ir.Imm(0), name="x")
    y = b.add(b.load(f.buffers[0], j, name="bj"), ir.Imm(0), name="y")
    z = b.add(b.load(f.buffers[0], k, name="bk"), ir.Imm(0), name="z")
    # every index is read AGAIN here, after its load
    s1 = b.add(b.add(x, y, name="xy"), z, name="xyz")
    s2 = b.add(b.add(i, j, name="ij"), k, name="ijk")
    b.store_at(f.buffers[1], t, b.add(s1, s2, name="r"))
    b.ret()
    return f


def _liveidx_py(B):
    out = []
    for t in range(THREADS):
        i, j, k = (t + 1) % THREADS, (t + 2) % THREADS, (t + 3) % THREADS
        out.append((int(B[i]) + int(B[j]) + int(B[k]) + i + j + k) & 0xFFFFFFFF)
    return out


def _loadbranch_ir():
    """load -> ALU -> compare -> branch -> LOAD -> store, which is the chain nothing else runs.

    The compare cannot take a value straight from a load - the compare's wait is unrecovered and
    the backend refuses it - so the ALU hop is required rather than decorative. What this adds over
    the branch kernel is a load INSIDE the guarded region: its result feeds a store in the same
    region, so the load-use wait has to hold under a mask as well as outside one.
    """
    import g17ir as ir
    f = ir.Function("loadbranch", [ir.Buffer("B", 1), ir.Buffer("C", 2)])
    e = f.block("entry"); th = f.block("then"); jn = f.block("join")
    b = ir.Builder(f, e)
    t = b.builtin("thread_position_in_grid", name="t")
    v = b.add(b.load(f.buffers[0], t, name="v0"), ir.Imm(0), name="v")
    g = getattr(b, "and")(v, ir.Imm(7), name="g")
    b.store_at(f.buffers[1], t, v)
    b.br_cond(b.cmp(g, 3, "gt", name="p"), th, jn)
    b.at(th)
    far = getattr(b, "and")(b.add(t, ir.Imm(3), name="f0"), ir.Imm(THREADS - 1), name="far")
    w = b.add(b.load(f.buffers[0], far, name="w0"), ir.Imm(200), name="w")
    b.store_at(f.buffers[1], t, w)
    b.br(jn)
    b.at(jn); b.ret()
    return f


def _loadbranch_py(B):
    out = []
    for t in range(THREADS):
        v = int(B[t])
        out.append(((int(B[(t + 3) % THREADS]) + 200) if (v & 7) > 3 else v) & 0xFFFFFFFF)
    return out


def _tgbarrier2_ir():
    """TWO exchanges and TWO barriers, the second reading what the first wrote.

    One barrier can be a no-op that happens to be followed by a load that is late anyway. Two, with
    a dependent exchange between them, cannot: the second read depends on a value another lane
    computed FROM a value a third lane wrote, so a barrier that does not order anything gives a
    different answer rather than the same one.
    """
    import g17ir as ir
    f = ir.Function("tgbarrier2", [ir.Buffer("B", 1), ir.Buffer("C", 2)])
    f.declare_threadgroup(THREADS, size=(THREADS, 1, 1))      # executed with one group of 32 lanes; declared now (ABI v4)
    b = ir.Builder(f, f.block("entry"))
    t = b.builtin("thread_position_in_grid", name="t")
    m = ir.Imm(THREADS - 1)
    b.store_tg(b.add(b.load(f.buffers[0], t, name="v0"), ir.Imm(1), name="mine"), t)
    b.barrier()
    n1 = getattr(b, "and")(b.add(t, ir.Imm(1), name="n10"), m, name="n1")
    a = b.mul(b.load_tg(n1, name="got1"), ir.Imm(2), name="a")
    b.barrier()
    b.store_tg(a, t)
    b.barrier()
    n2 = getattr(b, "and")(b.add(t, ir.Imm(2), name="n20"), m, name="n2")
    b.store_at(f.buffers[1], t, b.add(b.load_tg(n2, name="got2"), ir.Imm(0), name="out"))
    b.ret()
    return f


def _tgbarrier2_py(B):
    first = [(int(B[i]) + 1) & 0xFFFFFFFF for i in range(THREADS)]
    second = [(first[(i + 1) % THREADS] * 2) & 0xFFFFFFFF for i in range(THREADS)]
    return [second[(t + 2) % THREADS] for t in range(THREADS)]


def _identity_ir():
    """THE KEYSTONE KERNEL: C[t] = t, and nothing else at all.

    It exists to be SMALL, not to exercise anything. Every other kernel here contains at least
    three opcodes whose function has never been measured, so a whole-program result - however
    exactly it matches - constrains them only jointly and pins none of them. This one emits the
    special-register read, the store and the end and NOTHING more, so with the store and the end
    already entailed by the identity control (tools/g17scorecard.py:proven_by_the_harness), the
    special-register read is the single remaining unknown and its function follows.

    That is the whole point: it is the first link of a chain. Once the SR read is pinned, floatmath
    has one unknown left and pins the load; then branch pins the multiply, liveidx the add, and so
    on. tools/g17entail.py runs the propagation and names the next kernel worth writing.

    A LANE-VARYING OUTPUT IS WHAT MAKES IT DISCRIMINATING. 32 distinct values, one per lane, in
    lane order. A special register returning a constant, the threadgroup index, or the lane id of
    some other lane each gives a different answer, and a store that wrote the wrong slot scrambles
    the order rather than the values.
    """
    import g17ir as ir
    f = ir.Function("identity", [ir.Buffer("B", 1), ir.Buffer("C", 2)])
    b = ir.Builder(f, f.block("entry"))
    t = b.builtin("thread_position_in_grid", name="t")
    b.store_at(f.buffers[1], t, t)
    b.ret()
    return f


def _identity_py(B):
    return [t for t in range(THREADS)]


# THE SPLITTERS. Eight kernels that exist to contain exactly ONE opcode whose function has never
# been measured, so that a passing whole-program result pins that opcode rather than constraining a
# handful of them jointly.
#
# WHY THEY ARE NEEDED AT ALL. Every kernel written before these carries three to eight unmeasured
# opcodes, so the strongest evidence this project has - 25 programs matching both an independent
# reference and Apple's compilation of the same source - entailed the function of NOT ONE opcode.
# The chain has to start somewhere and then it runs: srconst and identity pin the special-register
# read, floatmath then has only the load left, ifelse then only the multiply, and so on.
# tools/g17entail.py runs that propagation and prints the residual set per kernel, which is how
# each of these was chosen: the probe below was written against the cluster it splits, and the
# compile-side check that it isolates one opcode came before the dispatch, not after.
#
# THE ALU HOP AFTER EACH LOAD is the four-byte bitwise form's measured load-use hazard, the same
# one bitreg documents. It is not decoration and removing it returns zero for every thread.


def _hop(b, f, t, i, n):
    """B[t+i] through an ALU, which is what the register-register forms need to see it."""
    import g17ir as ir
    idx = t if i == 0 else b.add(t, ir.Imm(i), name="t%d" % i)
    return b.add(b.load(f.buffers[0], idx, name=n + "0"), ir.Imm(0), name=n)


def _splitter(name, build):
    def make():
        import g17ir as ir
        f = ir.Function(name, [ir.Buffer("B", 1), ir.Buffer("C", 2)])
        b = ir.Builder(f, f.block("entry"))
        t = b.builtin("thread_position_in_grid", name="t")
        b.store_at(f.buffers[1], t, build(b, f, t))
        b.ret()
        return f
    return make


def _shiftr_ir():
    """op17013, the right shift - the only unmeasured opcode here. wideindex already pins the LEFT
    shift, and the two are different opcodes, so neither stands for the other."""
    import g17ir as ir
    return _splitter("shiftr", lambda b, f, t: b.shr(_hop(b, f, t, 0, "v"), ir.Imm(3), name="r"))()


def _addsatk_ir():
    """op10239, saturating add. satops emits it beside subsat and pins neither."""
    import g17ir as ir
    return _splitter("addsatk", lambda b, f, t: b.addsat(
        _hop(b, f, t, 0, "v"), b.const(0xFFFFFF00, name="k"), name="r"))()


def _subsatk_ir():
    """op11624, saturating subtract - the other half of satops' pair."""
    import g17ir as ir
    return _splitter("subsatk", lambda b, f, t: b.subsat(
        _hop(b, f, t, 0, "v"), b.const(4000, name="k"), name="r"))()


def _andreg_ir():
    """op424, the four-byte register-register AND. bitreg emits all three of this family at once."""
    return _splitter("andreg", lambda b, f, t: getattr(b, "and")(
        _hop(b, f, t, 0, "x"), _hop(b, f, t, 1, "y"), name="r"))()


def _orreg_ir():
    """op13575, register-register OR."""
    return _splitter("orreg", lambda b, f, t: getattr(b, "or")(
        _hop(b, f, t, 0, "x"), _hop(b, f, t, 1, "y"), name="r"))()


def _xorreg_ir():
    """op17771, register-register XOR."""
    return _splitter("xorreg", lambda b, f, t: b.xor(
        _hop(b, f, t, 0, "x"), _hop(b, f, t, 1, "y"), name="r"))()


def _subreg_ir():
    """op11667, register-register subtract. tgatomic is the only other kernel that emits it, and it
    carries the threadgroup atomic too - so pinning it here is what leaves that atomic alone."""
    return _splitter("subreg", lambda b, f, t: b.sub(
        _hop(b, f, t, 0, "x"), _hop(b, f, t, 1, "y"), name="r"))()


def _halfzero_ir():
    """op555/4, THE SIXTEEN-BIT ZERO MOVE, and the sixteen-bit store that reads it (handoff 10ag).

    Every other kernel here materialises a zero into a thirty-two-bit register. This one writes a
    SIXTEEN-bit element, which is the only shape in which Apple emits the four-byte move: the value
    must live in the 425-based half register file, and op555/4 is what puts a zero there. The
    source-faithful witnesses are results/g17-movimm4-roundB-compiles-v1/W0 (a `half` store) and
    results/g17-movimm4-roundC-compiles-v1/X0 (the same store through a `ushort` pointer, whose main
    program is byte-identical to W0's - which is how the form is known to be about sixteen bits
    rather than about half-FLOAT).

    THE OBSERVABLE IS THE POINT, and a kernel that stores only zeroes has none: its output is the
    fill value's twin and it passes while doing nothing (memory:check-the-observable-exists). So the
    zero goes in the LOW half of each word and a loaded value in the HIGH half:

        C[t] == (B[t] & 0xffff) << 16

    the low half zero says the first sixteen-bit store landed at element 2t, and the high half says
    the second landed at 2t+1 with the loaded value - so a miss on either element is visible, and a
    store that ignored the element scaling entirely would put both writes in one place.

    WHERE APPLE DIVERGES, recorded because this kernel is not a reproduction of Apple's lowering:
    Apple MERGES the adjacent pair into one two-component sixteen-bit store (op17202/10, round C's
    X1) and still emits op555/4 for the zero. This compiler emits the two stores separately
    (op17193/14 each, a form Apple's corpus carries 144 times). The zero move is identical; the
    stores are a merge this backend does not do.
    """
    import g17ir as ir
    f = ir.Function("halfzero", [ir.Buffer("B", 1), ir.Buffer("C", 2)])
    b = ir.Builder(f, f.block("entry"))
    t = b.builtin("thread_position_in_grid", name="t")
    lo = b.shl(t, ir.Imm(1), name="lo")
    b.store_at(f.buffers[1], lo, b.const(0, type=ir.I16, name="z"), width="half")
    hi = b.add(lo, ir.Imm(1), name="hi")
    b.store_at(f.buffers[1], hi,
               b.add(b.load(f.buffers[0], t, name="v0"), ir.Imm(0), name="v", type=ir.I16), width="half")
    b.ret()
    return f


def _halfzero_py(B): return [(int(B[t]) & 0xFFFF) << 16 for t in range(THREADS)]


def _ushort16_ir():
    """op10288, the 16-bit-destination add. narrow emits it beside op10280, the widening one, and
    truncation alone reaches only this one - which then leaves narrow with a single unknown."""
    import g17ir as ir
    f = ir.Function("ushort16", [ir.Buffer("B", 1), ir.Buffer("C", 2)])
    b = ir.Builder(f, f.block("entry"))
    t = b.builtin("thread_position_in_grid", name="t")
    b.store_at(f.buffers[1], t,
               b.add(b.load(f.buffers[0], t, name="v0"), ir.Imm(0), name="v", type=ir.I16))
    b.ret()
    return f


def _shiftr_py(B):   return [(int(B[t]) >> 3) & 0xFFFFFFFF for t in range(THREADS)]
def _addsatk_py(B):  return [min(int(B[t]) + 0xFFFFFF00, 0xFFFFFFFF) for t in range(THREADS)]
def _subsatk_py(B):  return [max(int(B[t]) - 4000, 0) for t in range(THREADS)]
def _andreg_py(B):   return [int(B[t]) & int(B[t + 1]) for t in range(THREADS)]
def _orreg_py(B):    return [int(B[t]) | int(B[t + 1]) for t in range(THREADS)]
def _xorreg_py(B):   return [int(B[t]) ^ int(B[t + 1]) for t in range(THREADS)]
def _subreg_py(B):   return [(int(B[t]) - int(B[t + 1])) & 0xFFFFFFFF for t in range(THREADS)]
def _ushort16_py(B): return [int(B[t]) & 0xFFFF for t in range(THREADS)]


def _srconst_ir():
    """SEPARATES THE TWO OPERANDS OF THE INDEXED STORE: address from the SR read, value from a
    constant.

    identity failed with the SAME register in both the value and the address slot, so it cannot say
    which one arrived late. This one puts a materialised constant in the value slot and leaves only
    the ADDRESS coming straight off the special-register read.

        the address is fine   every lane writes its own slot     C = [7] * 32
        the address is late   every lane computes slot 0         C[0] = 7 and the rest untouched

    Thirty-two identical values are usually a weak expectation. Here they are the strong one: the
    failure mode is not a wrong value, it is thirty-one lanes not writing at all.
    """
    import g17ir as ir
    f = ir.Function("srconst", [ir.Buffer("B", 1), ir.Buffer("C", 2)])
    b = ir.Builder(f, f.block("entry"))
    t = b.builtin("thread_position_in_grid", name="t")
    b.store_at(f.buffers[1], t, b.const(7, name="seven"))
    b.ret()
    return f


def _srconst_py(B):
    return [7] * THREADS


def _srindex_ir():
    """AN INDEX BUILT FROM TWO SPECIAL REGISTERS rather than from thread_position alone.

    threadgroup_position_in_grid and thread_position_in_threadgroup are separate SR reads with
    separate encodings, and this is the only kernel here that reads either. One threadgroup of 32
    makes the group index 0, so the arithmetic is checkable - and a special register that returned
    something other than 0 would move every lane, not one.
    """
    import g17ir as ir
    f = ir.Function("srindex", [ir.Buffer("B", 1), ir.Buffer("C", 2)])
    b = ir.Builder(f, f.block("entry"))
    g = b.builtin("threadgroup_position_in_grid", name="g")
    tt = b.builtin("thread_position_in_threadgroup", name="tt")
    idx = b.add(b.mul(g, ir.Imm(THREADS), name="gbase"), tt, name="idx")
    v = b.add(b.load(f.buffers[0], idx, name="v0"), ir.Imm(0), name="v")
    b.store_at(f.buffers[1], idx, b.add(b.mul(v, ir.Imm(2), name="d"), tt, name="r"))
    b.ret()
    return f


def _srindex_py(B):
    return [(int(B[t]) * 2 + t) & 0xFFFFFFFF for t in range(THREADS)]


def _wideindex_ir():
    """AN INDEX FAR OUTSIDE THE IMMEDIATE'S RANGE, computed into a register and then addressed.

    Every other kernel here indexes within a few of the thread id. This one reaches word 2084 of a
    4096-word buffer, so the address arithmetic is exercised at a magnitude no immediate field can
    hold - the index has to travel as a register and be scaled at the load.

    WHAT THIS IS NOT: a test of 64-bit addressing. No buffer here is large enough to carry an index
    past 2^32, and allocating one to find out is not a thing to do casually on a machine whose GPU
    this project has already wedged twice. What it tests is that an index well beyond a byte, and
    beyond the 8-bit immediate the ALU forms carry, survives the shift, the add, the load's scaling
    and the store - checked against Apple's compilation of the same source.
    """
    import g17ir as ir
    f = ir.Function("wideindex", [ir.Buffer("B", 1), ir.Buffer("C", 2)])
    b = ir.Builder(f, f.block("entry"))
    t = b.builtin("thread_position_in_grid", name="t")
    i = b.add(b.shl(t, ir.Imm(6), name="hi"), ir.Imm(100), name="i")
    v = b.add(b.load(f.buffers[0], i, name="v0"), ir.Imm(0), name="v")
    b.store_at(f.buffers[1], t, b.add(v, i, name="r"))
    b.ret()
    return f


def _wideindex_py(B):
    return [(int(B[(t << 6) + 100]) + ((t << 6) + 100)) & 0xFFFFFFFF for t in range(THREADS)]


LOOP_K = 8
VEC_BASE = 200
LOOP_KERNELS = {"gemvloop", "dotloop", "divgemv"}


def _gemvloop_ir():
    """A LOOP THAT DOES MEMORY WORK - the shape every workload this project exists for actually has.

    The counted loop that came before this one counts. No loads, no stores in the body, one uniform
    trip count: the smallest possible loop, and it leaves the interesting joins untested. This one
    is `for k: acc += B[t*K + k]`, which puts three things in the loop that have never been there:

      A LOAD IN THE BODY. The load-use wait is authored from liveness over the linear instruction
        list; across a back edge the consumer of iteration n+1 sits EARLIER in the text than the
        producer of iteration n. Its failure mode is reading a pending load, which looks like a
        plausible wrong number rather than a crash.
      A LOOP-CARRIED ACCUMULATOR, which must survive the whole body and the back edge - a longer
        obligation than the induction variable's self-referencing update.
      A LOOP-INVARIANT INDEX BASE, computed before the loop and read every iteration. This is the
        one that broke: `base` is textually last used at the top of the body, so the allocator gave
        its register to the load two instructions later.

    Uniform trip count on purpose. op582 pushes per iteration and one pop follows, so the mask
    after the loop is the second-to-last push's - harmless while every push carries the same mask,
    wrong the moment lanes exit at different iterations. Divergent loops are a separate kernel and
    a separate fix.
    """
    import g17ir as ir
    f = ir.Function("gemvloop", [ir.Buffer("B", 1), ir.Buffer("C", 2)])
    pre = f.block("pre"); hdr = f.block("header"); ex = f.block("exit")
    b = ir.Builder(f, pre)
    t = b.builtin("thread_position_in_grid", name="t")
    base = b.mul(t, ir.Imm(LOOP_K), name="base")
    z = b.const(0, name="z"); k0 = b.const(0, name="k0")
    b.br(hdr)
    b.at(hdr)
    acc = b.phi(z, name="acc"); k = b.phi(k0, name="k")
    idx = b.add(base, k, name="idx")
    v = b.load(f.buffers[0], idx, name="v")
    accn = b.add(acc, v, name="accn")
    kn = b.add(k, ir.Imm(1), name="kn")
    ir.Builder.phi_latch(acc, accn); ir.Builder.phi_latch(k, kn)
    b.br_cond(b.cmp(kn, LOOP_K, "lt", name="p"), hdr, ex)
    b.at(ex); b.store_at(f.buffers[1], t, accn); b.ret()
    return f


def _gemvloop_py(B):
    return [sum(int(B[t * LOOP_K + k]) for k in range(LOOP_K)) & 0xFFFFFFFF
            for t in range(THREADS)]


def _dotloop_ir():
    """THE INNER PRODUCT: two loads and a multiply in the body, which is GEMV's actual inner loop.

    gemvloop proved a load in a loop. This adds the rest of the shape - a second load, a
    register-register multiply consuming BOTH of them, and an accumulator over the product. Every
    iteration now has two values that arrive late and one instruction that must wait for both.
    """
    import g17ir as ir
    f = ir.Function("dotloop", [ir.Buffer("B", 1), ir.Buffer("C", 2)])
    pre = f.block("pre"); hdr = f.block("header"); ex = f.block("exit")
    b = ir.Builder(f, pre)
    t = b.builtin("thread_position_in_grid", name="t")
    row = b.mul(t, ir.Imm(LOOP_K), name="row")
    vec = b.const(VEC_BASE, name="vec")
    z = b.const(0, name="z"); k0 = b.const(0, name="k0")
    b.br(hdr)
    b.at(hdr)
    acc = b.phi(z, name="acc"); k = b.phi(k0, name="k")
    a = b.load(f.buffers[0], b.add(row, k, name="ia"), name="a")
    c = b.load(f.buffers[0], b.add(vec, k, name="ib"), name="c")
    prod = b.mul(a, c, name="prod")
    accn = b.add(acc, prod, name="accn")
    kn = b.add(k, ir.Imm(1), name="kn")
    ir.Builder.phi_latch(acc, accn); ir.Builder.phi_latch(k, kn)
    b.br_cond(b.cmp(kn, LOOP_K, "lt", name="p"), hdr, ex)
    b.at(ex); b.store_at(f.buffers[1], t, accn); b.ret()
    return f


def _dotloop_py(B):
    return [sum(int(B[t * LOOP_K + k]) * int(B[VEC_BASE + k]) for k in range(LOOP_K)) & 0xFFFFFFFF
            for t in range(THREADS)]


DIV_LIM = 8                       # the divergent bound: lane t runs 8-t iterations, 0 past it


def _divgemv_ir():
    """A DIVERGENT LOOP THAT DOES MEMORY WORK - the intersection neither loop kernel covers.

    Two things were proven separately and never together. gemvloop put a LOAD in a loop, which is
    what breaks a load-use wait authored from linear liveness: across a back edge the consumer of
    iteration n+1 sits earlier in the text than the producer of iteration n. divloop made lanes
    EXIT AT DIFFERENT ITERATIONS, which is what breaks a mask built by pushing: it faulted, cb=-1,
    until the latch learned to pop before its compare (ledger/g17-the-loop-mask-is-rebuilt-not-
    narrowed.toml). Both loops that carry a load are uniform; the divergent loop's body is pure
    arithmetic with no memory access at all.

    This is both at once, and it is the shape a variable-length row reduction actually has:

        row = t * 8;  x = t;  acc = 0
        if (x < 8) { do { acc += B[row + x]; x += 1 } while (x < 8) }
        C[t] = acc

    A TRIANGLE. Lane t reads B[t*8 + t .. t*8 + 7] - a different base AND a different length per
    lane - so the address is loop-carried, lane-varying, and derived from a loop-invariant base
    computed before the guard. That last one is what broke gemvloop: `row` is textually last used
    at the top of the body, so a position-based allocator hands its register to the load two
    instructions later. Under divergence the same value must also survive lanes leaving.

    WHAT EACH WRONG ANSWER LOOKS LIKE, which is why the expectation discriminates:
        a uniform trip count      every lane sums 8 elements; lane 7 and lane 0 stop agreeing
        an always-taken loop      lanes 8..31 store a sum of B[64..] instead of 0
        an off-by-one mask        each lane's total is short or long by exactly one element
        a mask left partial       the trailing store does not run for every lane - that FAULTED
        a pending load read       a plausible wrong number, which is why Apple checks it too

    BOUNDED BY CONSTRUCTION, which is the standing precondition for dispatching a back edge: x
    starts at t >= 0, advances by a constant +1 in one self-referencing instruction, and is
    compared against the constant 8, so the predicate is false forever after at most 8 iterations.
    """
    import g17ir as ir
    f = ir.Function("divgemv", [ir.Buffer("B", 1), ir.Buffer("C", 2)])
    pre = f.block("pre"); body = f.block("body"); ex = f.block("exit")
    b = ir.Builder(f, pre)
    t = b.builtin("thread_position_in_grid", name="t")
    row = b.mul(t, ir.Imm(DIV_LIM), name="row")
    x0 = b.add(t, ir.Imm(0), name="x0")
    a0 = b.const(0, name="a0")
    # THE GUARD IS WHAT MAKES ZERO ITERATIONS REACHABLE. A do-while runs its body once; lanes that
    # must run none are masked off here and still have to reach the store afterwards, which is the
    # property that used to fault.
    b.br_cond(b.cmp(x0, DIV_LIM, "lt", name="g"), body, ex)
    b.at(body)
    x = b.phi(x0, name="x"); acc = b.phi(a0, name="acc")
    v = b.load(f.buffers[0], b.add(row, x, name="ix"), name="v")
    accn = b.add(acc, v, name="accn")
    xn = b.add(x, ir.Imm(1), name="xn")
    ir.Builder.phi_latch(x, xn); ir.Builder.phi_latch(acc, accn)
    b.br_cond(b.cmp(xn, DIV_LIM, "lt", name="p"), body, ex)
    b.at(ex); b.store_at(f.buffers[1], t, accn); b.ret()
    return f


def _divgemv_py(B):
    out = []
    for t in range(THREADS):
        x, a = t, 0
        if x < DIV_LIM:
            while True:
                a = (a + int(B[t * DIV_LIM + x])) & 0xFFFFFFFF
                x += 1
                if not x < DIV_LIM:
                    break
        out.append(a)
    return out


RED_K, RED_VEC = 64, 200          # a 64-wide row reduced by 32 lanes, two products each


def _gemvreduce_ir():
    """A REAL GEMV: a full row, reduced across the threadgroup, then read back by every lane.

    Every kernel before this one is a lane-local computation with a store at the end. This is the
    first that is a WORKLOAD - the shape agxforge exists to run - and it composes almost everything
    the backend has: two strided loads and a multiply per lane, a threadgroup store, SIX barriers,
    five guarded regions whose predicates are lane-varying, threadgroup loads at a computed index
    inside those regions, and a final threadgroup read that every lane consumes.

    The reduction is UNROLLED because the compare takes an immediate: `t < d` for a register d is
    not expressible, so the tree is five explicit steps at 16, 8, 4, 2, 1. That is not a workaround
    here - it is what the ISA's compare offers, and the unrolled form is what Apple emits for a
    fixed-width reduction anyway.

    THE BARRIERS SIT OUTSIDE THE GUARDED REGIONS. A barrier reached by only some lanes is
    undefined on every GPU that has one, so each step's barrier is in the JOIN block, after the
    region's exec.restore has put every lane back.

    The output is s[0] + t rather than s[0] so the lanes differ: 32 identical values would satisfy
    the arithmetic check while also being exactly what a broken reduction returns.
    """
    import g17ir as ir
    f = ir.Function("gemvreduce", [ir.Buffer("B", 1), ir.Buffer("C", 2)])
    f.declare_threadgroup(THREADS, size=(THREADS, 1, 1))      # executed with one group of 32 lanes; declared now (ABI v4)
    steps = [16, 8, 4, 2, 1]
    entry = f.block("entry")
    body = {}; join = {}
    for d in steps:
        body[d] = f.block("body%d" % d); join[d] = f.block("join%d" % d)
    b = ir.Builder(f, entry)
    t = b.builtin("thread_position_in_grid", name="t")

    def product(off):
        i1 = b.add(t, ir.Imm(off), name="ia%d" % off) if off else t
        i2 = b.add(t, ir.Imm(RED_VEC + off), name="ib%d" % off)
        x = b.add(b.load(f.buffers[0], i1, name="x0%d" % off), ir.Imm(0), name="x%d" % off)
        y = b.add(b.load(f.buffers[0], i2, name="y0%d" % off), ir.Imm(0), name="y%d" % off)
        return b.mul(x, y, name="p%d" % off)

    b.store_tg(b.add(product(0), product(THREADS), name="acc"), t)
    b.barrier()
    for d in steps:
        b.br_cond(b.cmp(t, d, "lt", name="pr%d" % d), body[d], join[d])
        b.at(body[d])
        mine = b.add(b.load_tg(t, name="m%d" % d), ir.Imm(0), name="mv%d" % d)
        far = b.add(b.load_tg(b.add(t, ir.Imm(d), name="fi%d" % d), name="f%d" % d),
                    ir.Imm(0), name="fv%d" % d)
        b.store_tg(b.add(mine, far, name="sum%d" % d), t)
        b.br(join[d])
        b.at(join[d]); b.barrier()
    tot = b.add(b.load_tg(b.const(0, name="zero"), name="tot0"), ir.Imm(0), name="tot")
    b.store_at(f.buffers[1], t, b.add(tot, t, name="out"))
    b.ret()
    return f


def _gemvreduce_py(B):
    total = sum(int(B[k]) * int(B[RED_VEC + k]) for k in range(RED_K)) & 0xFFFFFFFF
    return [(total + t) & 0xFFFFFFFF for t in range(THREADS)]


def _rangestore_ir():
    """THE RANGE-STORE FORMS, EVERY COMPONENT COUNT, BOTH SLOT WIDTHS - the forms Apple's corpus
    uses most and no kernel here had ever emitted: four values at slot 4 (op17262, 8 bytes), three
    at slot 70 (op17253, 14 bytes), two at slot 80 (op17244, 14 bytes). Slot stores are the same
    for every thread, so this probe launches one invocation and checks the buffer's
    fill everywhere else. op17262/14 blocks 1,550 of 6,594 corpus programs and had been emittable,
    unexecuted, since the range store landed (docs/archive/g17-surface-compiler-handoff.md).
    """
    import g17ir as ir
    f = ir.Function("rangestore", [ir.Buffer("B", 1), ir.Buffer("C", 2)])
    b = ir.Builder(f, f.block("entry"))
    b.builtin("thread_position_in_grid", name="t")
    b.store_range(f.buffers[1], ir.Imm(4), [b.const(v, name="c%d" % v) for v in (1, 2, 3, 4)])
    b.store_range(f.buffers[1], ir.Imm(70), [b.const(v, name="c%d" % v) for v in (5, 6, 7)])
    b.store_range(f.buffers[1], ir.Imm(80), [b.const(v, name="c%d" % v) for v in (8, 9)])
    b.ret()
    return f


def _rangestore_py(B):
    out = [0xDEADBEEF] * 92                 # the harness's fill: unwritten slots stay as they were
    for slot, v in zip((4, 5, 6, 7, 70, 71, 72, 80, 81), (1, 2, 3, 4, 5, 6, 7, 8, 9)):
        out[slot] = v
    return out


def _vec2load_ir():
    """THE TWO- AND THREE-COMPONENT TUPLE LOADS, op12691 and op12700 (handoff 10am).

    The component count is byte4[6:5] = n-1 and the opcode follows it in steps of NINE: op12682 at one
    component, op12691 at two, op12700 at three, op12709 at four. The four-component member was lowered by the
    vector-memory delivery (10ae) and its encoder wrote the count unconditionally, because every witness it
    held had four; this kernel is the other two counts.

    THE OBSERVABLE IS PER LANE. Each lane goes to its own slot, so a load that fetched the wrong number of
    components, or placed them in the wrong registers, shows up as a wrong word rather than as a crash:

        C[4], C[5]        <- in[t].x + 1, in[t].y + 2          (two components)
        C[8], C[9], C[10] <- in2[t].x + 1, .y + 2, .z + 3      (three components)

    ONE VECTOR LOAD PER PROGRAM is the standing refusal from 10ae - Apple selects the load's LENGTH by what is
    in flight through an unrecovered composite - so the two loads here are in SEPARATE functions and this
    kernel carries the two-component one. The three-component form is exercised by the harvest program in
    g17formops and by the delivery's own witness.
    """
    import g17ir as ir
    f = ir.Function("vec2load", [ir.Buffer("B", 1), ir.Buffer("C", 2)])
    b = ir.Builder(f, f.block("entry"))
    t = b.builtin("thread_position_in_grid", name="t")
    lanes = b.load_vec_at(f.buffers[0], t, 2, name="q")
    for k, lane in enumerate(lanes):
        b.store(f.buffers[1], ir.Imm(4 + k), b.add(lane, ir.Imm(k + 1), name="s%d" % k))
    b.ret()
    return f


def _vec2load_py(B):
    out = [0xDEADBEEF] * 92
    out[4] = (int(B[0]) + 1) & 0xFFFFFFFF
    out[5] = (int(B[1]) + 2) & 0xFFFFFFFF
    return out


def _duprange_ir():
    """A VALUE STORED IN TWO COMPONENTS, which until handoff 10al silently stored garbage in the second.

    The range forms write r<src> .. r<src+n-1>, so a caller naming one value twice used to get the value in
    the first component and whatever the allocator had left in the second - at both widths, with no refusal.
    Each repeat now gets its own register through a MOVE: op590 for a sixteen-bit value, op586 for a
    thirty-two-bit one.

    THE OBSERVABLE IS THE SECOND COMPONENT. Both halves of C[14] must hold the same sixteen-bit value, and
    C[20] and C[21] must hold the same word. Before the fix the second of each pair was an unrelated
    register, which is a wrong value rather than a crash - so this kernel is the regression test for a
    defect that no earlier kernel could see.

    ONE INVOCATION, like rangestore: every lane writes the same slots.
    """
    import g17ir as ir
    f = ir.Function("duprange", [ir.Buffer("B", 1), ir.Buffer("C", 2)])
    b = ir.Builder(f, f.block("entry"))
    t = b.builtin("thread_position_in_grid", name="t")
    w = b.load(f.buffers[0], t, name="w")
    h = b.add(w, ir.Imm(1), name="h", type=ir.I16)
    b.store_range(f.buffers[1], ir.Imm(14), [h, h], width="half")
    q = b.add(w, ir.Imm(2), name="q")
    b.store_range(f.buffers[1], ir.Imm(20), [q, q])
    b.ret()
    return f


def _duprange_py(B):
    out = [0xDEADBEEF] * 92
    h = (int(B[0]) + 1) & 0xFFFF
    out[14] = h | (h << 16)                 # both halves of one word, from ONE named value
    out[20] = out[21] = (int(B[0]) + 2) & 0xFFFFFFFF
    return out


def _wordslot_ir():
    """op17235, ONE thirty-two-bit element at an immediate address, reserving NOTHING (handoff 10ak).

    The witnesses are results/g17-wordslot-compiles-v1, and this kernel exercises what they measured at all
    three lengths, including the boundary T4 found:

        C[7]  (28 bytes)    an ALU value, nothing to wait for      ->  8 bytes
        C[8]  (32 bytes)    straight from a load                   -> 10 bytes
        C[63] (252 bytes)   the last displacement inside 8 bits    ->  8 bytes
        C[70] (280 bytes)   the first one outside it                -> 14 bytes

    THE OBSERVABLE IS THE UNTOUCHED NEIGHBOUR, which is what this form buys. The default constant-slot
    store writes slot k from the value and RESERVES k+1 with a materialised zero, so a program wanting one
    word loses the next one. Here C[64] and C[71] must still hold the harness's fill, and C[8] holds a
    stored value rather than a companion zero - which the reserving form could not produce, because its
    write to C[8] would be the zero it reserved for C[7].

    ONE INVOCATION, like rangestore: every lane writes the same slots.
    """
    import g17ir as ir
    f = ir.Function("wordslot", [ir.Buffer("B", 1), ir.Buffer("C", 2)])
    b = ir.Builder(f, f.block("entry"))
    t = b.builtin("thread_position_in_grid", name="t")
    loads = [b.load(f.buffers[0], t, offset=k, name="b%d" % k) for k in range(4)]
    b.store(f.buffers[1], ir.Imm(7), b.add(loads[0], ir.Imm(1), name="q0"), reserve_companion=False)
    b.store(f.buffers[1], ir.Imm(8), loads[1], reserve_companion=False)
    b.store(f.buffers[1], ir.Imm(63), b.add(loads[2], ir.Imm(3), name="q2"), reserve_companion=False)
    b.store(f.buffers[1], ir.Imm(70), b.add(loads[3], ir.Imm(4), name="q3"), reserve_companion=False)
    b.ret()
    return f


def _wordslot_py(B):
    out = [0xDEADBEEF] * 92                 # the harness's fill: an unwritten slot stays as it was
    out[7] = (int(B[0]) + 1) & 0xFFFFFFFF
    out[8] = int(B[1]) & 0xFFFFFFFF         # a STORED value where the reserving form would write its zero
    out[63] = (int(B[2]) + 3) & 0xFFFFFFFF
    out[70] = (int(B[3]) + 4) & 0xFFFFFFFF
    return out


def _halfslot_ir():
    """op17199, ONE sixteen-bit element at an immediate address, at all three of its lengths (handoff 10aj).

    The form is op17235's sixteen-bit twin - one bit at byte0[3] under the decoder - and its address is a
    BYTE DISPLACEMENT rather than a slot, which is the whole reason an ODD half element is expressible. The
    seven retained witnesses are results/g17-halfslot-compiles-v1; this kernel exercises what they measured:

        element 6  (byte 12, the low half of word 3)    an ALU value, no load to wait for   ->  8 bytes
        element 7  (byte 14, the HIGH half of word 3)   an ALU value                        ->  8 bytes
        element 8  (byte 16, the low half of word 4)    straight from a half LOAD           -> 10 bytes
        element 200 (byte 400)                          a displacement past eight bits      -> 14 bytes

    THE ODD ELEMENT IS THE POINT. Element 7 lands two bytes into word 3, which no slot field can address -
    the word store's slot is the same physical bits at a four-byte scale. C[3] therefore carries two
    different sixteen-bit values, one per half, and a store that ignored the displacement's low bit would
    put both in the same place.

    ONE INVOCATION, as `rangestore_read` is: every lane writes the same elements, so the values must not
    depend on the lane for the result to be defined. They come from B at fixed offsets.
    """
    import g17ir as ir
    f = ir.Function("halfslot", [ir.Buffer("B", 1), ir.Buffer("C", 2)])
    b = ir.Builder(f, f.block("entry"))
    t = b.builtin("thread_position_in_grid", name="t")
    v = b.load(f.buffers[0], t, name="v")
    b.store(f.buffers[1], ir.Imm(6), b.add(v, ir.Imm(1), name="h0", type=ir.I16), width="half")
    b.store(f.buffers[1], ir.Imm(7), b.add(v, ir.Imm(2), name="h1", type=ir.I16), width="half")
    b.store(f.buffers[1], ir.Imm(8), b.load(f.buffers[0], t, name="hl", type=ir.I16, width="half"), width="half")
    b.store(f.buffers[1], ir.Imm(200), b.add(v, ir.Imm(3), name="h2", type=ir.I16), width="half")
    b.ret()
    return f


def _halfslot_py(B):
    out = [0xDEADBEEF] * 256                # the harness's fill: unwritten halves stay as they were
    lo = lambda x: int(x) & 0xFFFF
    out[3] = lo(int(B[0]) + 1) | (lo(int(B[0]) + 2) << 16)
    out[4] = (0xDEADBEEF & 0xFFFF0000) | lo(B[0])          # element 8: the low half only
    out[100] = (0xDEADBEEF & 0xFFFF0000) | lo(int(B[0]) + 3)
    return out


def _halfvec_ir():
    """THE HALF-VECTOR SLOT STORES: op17226 (four sixteen-bit components), op17217 (three) and op17208 (two),
    at fourteen bytes (handoff 10ah; integration's 66cef0b5).

    The same shape as `rangestore_read` - four loads, an add on each so no store reads a pending load, then
    three range stores - but SIXTEEN bits per element. Apple selects these three forms for a CONSTANT slot
    index (results/g17-halfvec-roundB-compiles-v1: R0, R1, R5) and a different family, op17220/17202/17211,
    for a per-thread index; the sweep located the address mode at byte5[1], and round A's eight members are
    the retained evidence that a computed index does not reach these opcodes.

    WHY THE THREE-COMPONENT STORE IS THE INTERESTING ONE. Its six bytes cross a word boundary: at slot 20 it
    writes bytes 80..85, so C[20] is fully written and only the LOW half of C[21] is. The reference says the
    high half of C[21] keeps the harness's fill, which is an observable that a store writing eight bytes
    instead of six would fail and a store writing four would fail differently. The two- and four-component
    stores are word-aligned and say nothing about extent.

        C[14], C[15]            <- (B[k] + k + 1) & 0xffff, packed two per word
        C[20], C[21] low half   <- (B[k] + k + 5) & 0xffff, three of them
        C[24]                   <- (B[k] + k + 8) & 0xffff, two of them
    """
    import g17ir as ir
    f = ir.Function("halfvec", [ir.Buffer("B", 1), ir.Buffer("C", 2)])
    b = ir.Builder(f, f.block("entry"))
    t = b.builtin("thread_position_in_grid", name="t")
    loads = [b.load(f.buffers[0], t, offset=k, name="b%d" % k) for k in range(4)]
    b.store_range(f.buffers[1], ir.Imm(14), [b.add(loads[k], ir.Imm(k + 1), name="q%d" % k, type=ir.I16) for k in range(4)], width="half")
    b.store_range(f.buffers[1], ir.Imm(20), [b.add(loads[k], ir.Imm(k + 5), name="r%d" % k, type=ir.I16) for k in range(3)], width="half")
    b.store_range(f.buffers[1], ir.Imm(24), [b.add(loads[k], ir.Imm(k + 8), name="p%d" % k, type=ir.I16) for k in range(2)], width="half")
    b.ret()
    return f


def _halfvec_py(B):
    out = [0xDEADBEEF] * 92                 # the harness's fill: unwritten halves stay as they were
    h = lambda k, add: (int(B[k]) + add) & 0xFFFF
    out[14] = h(0, 1) | (h(1, 2) << 16)
    out[15] = h(2, 3) | (h(3, 4) << 16)
    out[20] = h(0, 5) | (h(1, 6) << 16)
    out[21] = (0xDEADBEEF & 0xFFFF0000) | h(2, 7)      # only the LOW half is written
    out[24] = h(0, 8) | (h(1, 9) << 16)
    return out


def _rangestore_read_ir():
    """THE READ BINDING EXERCISED. The delivered range-store kernel reads nothing, so its image's
    read binding record carried no execution evidence and could not (linker review 836510f): the
    cases varied the source and produced one output. This variant loads four words of B and
    stores values derived from them through the same three range forms, so the output is a
    function of the input - two sources, two outputs - and every load goes through an add before
    the store, because a store's source straight from a load is refused (load-wait). One
    invocation, same slots, same bindings, same forms plus the 14-byte load."""
    import g17ir as ir
    f = ir.Function("rangestore_read", [ir.Buffer("B", 1), ir.Buffer("C", 2)])
    b = ir.Builder(f, f.block("entry"))
    t = b.builtin("thread_position_in_grid", name="t")       # 0: one invocation
    loads = [b.load(f.buffers[0], t, offset=k, name="b%d" % k) for k in range(4)]
    b.store_range(f.buffers[1], ir.Imm(4), [b.add(loads[k], ir.Imm(k + 1), name="q%d" % k) for k in range(4)])
    b.store_range(f.buffers[1], ir.Imm(70), [b.add(loads[k], ir.Imm(k + 5), name="r%d" % k) for k in range(3)])
    b.store_range(f.buffers[1], ir.Imm(80), [b.add(loads[k], ir.Imm(k + 8), name="p%d" % k) for k in range(2)])
    b.ret()
    return f


def _rangestore_read_py(B):
    out = [0xDEADBEEF] * 92
    for k in range(4): out[4 + k] = (int(B[k]) + k + 1) & 0xFFFFFFFF
    for k in range(3): out[70 + k] = (int(B[k]) + k + 5) & 0xFFFFFFFF
    for k in range(2): out[80 + k] = (int(B[k]) + k + 8) & 0xFFFFFFFF
    return out


KERNELS = {
 "rangestore": (_rangestore_ir, _rangestore_py,
                "  C[4] = 1u; C[5] = 2u; C[6] = 3u; C[7] = 4u;\n"
                "  C[70] = 5u; C[71] = 6u; C[72] = 7u;\n"
                "  C[80] = 8u; C[81] = 9u;"),
 "rangestore_read": (_rangestore_read_ir, _rangestore_read_py,
                     "  C[4] = B[0] + 1u; C[5] = B[1] + 2u; C[6] = B[2] + 3u; C[7] = B[3] + 4u;\n"
                     "  C[70] = B[0] + 5u; C[71] = B[1] + 6u; C[72] = B[2] + 7u;\n"
                     "  C[80] = B[0] + 8u; C[81] = B[1] + 9u;"),
 "vec2load": (_vec2load_ir, _vec2load_py,
              "  uint2 v = ((device const uint2 *)B)[t];\n"
              "  C[4] = v.x + 1u; C[5] = v.y + 2u;"),
 "duprange": (_duprange_ir, _duprange_py,
              "  device ushort *H = (device ushort *)C;\n"
              "  ushort h = (ushort)(B[t] + 1u); H[28] = h; H[29] = h;\n"
              "  uint q = B[t] + 2u; C[20] = q; C[21] = q;"),
 "wordslot": (_wordslot_ir, _wordslot_py,
              "  C[7] = B[0] + 1u; C[8] = B[1];\n"
              "  C[63] = B[2] + 3u; C[70] = B[3] + 4u;"),
 "halfslot": (_halfslot_ir, _halfslot_py,
              "  device ushort *H = (device ushort *)C;\n"
              "  H[6] = (ushort)(B[t] + 1u); H[7] = (ushort)(B[t] + 2u);\n"
              "  H[8] = H[2u * t]; H[200] = (ushort)(B[t] + 3u);"),
 "halfvec": (_halfvec_ir, _halfvec_py,
             "  device ushort4 *H4 = (device ushort4 *)C;\n"
             "  device ushort *H = (device ushort *)C;\n"
             "  H4[7] = ushort4((ushort)(B[0]+1u), (ushort)(B[1]+2u), (ushort)(B[2]+3u), (ushort)(B[3]+4u));\n"
             "  H[40] = (ushort)(B[0]+5u); H[41] = (ushort)(B[1]+6u); H[42] = (ushort)(B[2]+7u);\n"
             "  H[48] = (ushort)(B[0]+8u); H[49] = (ushort)(B[1]+9u);"),
 "atomicadd": (_atomicadd_ir, _atomicadd_py,
               "  atomic_fetch_add_explicit(&A[t], t + 1u, memory_order_relaxed);\n"
               "  C[t] = atomic_load_explicit(&A[t], memory_order_relaxed);"),
 "wavebcast": (_wavebcast_ir, _wavebcast_py,
               "  uint tv = simd_sum(1u);\n"
               "  uint pv = simd_prefix_exclusive_sum(1u);\n"
               "  uint old = 0u;\n"
               "  if (pv == 0u) {\n"
               "    atomic_fetch_add_explicit(&A[0], tv, memory_order_relaxed);\n"
               "    old = atomic_fetch_add_explicit(&A[0], tv, memory_order_relaxed);\n"
               "  }\n"
               "  C[t] = simd_broadcast_first(old) + pv;"),
 "waveagg": (_waveagg_ir, _waveagg_py,
             "  uint tv = simd_sum(1u);\n"
             "  uint pv = simd_prefix_exclusive_sum(1u);\n"
             "  if (pv == 0u) atomic_fetch_add_explicit(&A[0], tv, memory_order_relaxed);\n"
             "  threadgroup_barrier(mem_flags::mem_device);\n"
             "  C[t] = atomic_load_explicit(&A[0], memory_order_relaxed) + pv;"),
 "tgops": (_tgops_ir, _tgops_py,
           "  threadgroup atomic_uint counter;\n"
           "  uint tv = simd_sum(1u);\n"
           "  uint pv = simd_prefix_exclusive_sum(1u);\n"
           "  uint d = 0u;\n"
           "  if (pv == 0u) {\n"
           "    atomic_fetch_and_explicit(&counter, 0u, memory_order_relaxed);\n"
           "    atomic_fetch_add_explicit(&counter, 200u, memory_order_relaxed);\n"
           "    atomic_fetch_or_explicit(&counter, 7u, memory_order_relaxed);\n"
           "    atomic_fetch_xor_explicit(&counter, 200u, memory_order_relaxed);\n"
           "    atomic_fetch_max_explicit(&counter, 100u, memory_order_relaxed);\n"
           "    atomic_fetch_min_explicit(&counter, 99u, memory_order_relaxed);\n"
           "    atomic_fetch_sub_explicit(&counter, 11u, memory_order_relaxed);\n"
           "    d = atomic_fetch_add_explicit(&counter, 0u, memory_order_relaxed);\n"
           "  }\n"
           "  C[t] = simd_broadcast_first(d) + pv;"),
 "tgatomic": (_tgatomic_ir, _tgatomic_py,
              "  threadgroup atomic_uint counter;\n"
              "  uint tv = simd_sum(1u);\n"
              "  uint pv = simd_prefix_exclusive_sum(1u);\n"
              "  uint d = 0u;\n"
              "  if (pv == 0u) {\n"
              "    uint o1 = atomic_fetch_sub_explicit(&counter, tv, memory_order_relaxed);\n"
              "    uint o2 = atomic_fetch_sub_explicit(&counter, tv, memory_order_relaxed);\n"
              "    d = o1 - o2;\n"
              "  }\n"
              "  C[t] = simd_broadcast_first(d) + pv;"),
 "votepair": (_votepair_ir, _votepair_py,
              "  uint pv = simd_prefix_exclusive_sum(1u);\n"
              "  uint tv = simd_sum(1u);\n"
              "  C[t] = (tv << 16) | pv;"),
 "linear": (_linear_ir, _linear_py,
            "  uint s = B[4*t]*%du + B[4*t+1]*%du + B[4*t+2]*%du + B[4*t+3]*%du + %du;\n"
            "  C[t] = (s > %du) ? s : 0u;" % (W[0], W[1], W[2], W[3], BIAS, THRESH)),
 "floatmath": (_floatmath_ir, _floatmath_py,
               "  float x = as_type<float>(B[t]);\n"
               "  float m = x * x;\n"
               "  float a = m + x;\n"
               "  C[t] = as_type<uint>(fma(x, x, a));", "float"),
 "branch": (_branch_ir, _branch_py,
            "  uint v = B[t];\n"
            "  C[t] = v + 100u;\n"
            "  if (t > 8u) C[t] = v * 3u;"),
 "narrow": (_narrow_ir, _narrow_py,
            "  ushort s = (ushort)(((ushort)B[t]) << 5);\n"
            "  C[t] = (uint)s + 1u;"),
 "bitreg": (_bitreg_ir, _bitreg_py,
            "  C[t] = (B[t] & B[t+1]) + (B[t+2] | B[t+3]) + (B[t+4] ^ B[t+5]);"),
 "satops": (_satops_ir, _satops_py,
            "  uint v = B[t];\n"
            "  C[t] = addsat(v, 0xFFFFFF00u) + subsat(v, 4000u);"),
 "pressure": (_pressure_ir, _pressure_py,
              "  uint " + ", ".join("v%d = B[(t + %du) & %du]" % (i, i, THREADS - 1)
                                    for i in range(PRESSURE_N)) + ";\n"
              "  C[t] = " + " + ".join("v%d" % i for i in range(PRESSURE_N)) + ";"),
 "tgself": (_tgself_ir, _tgself_py,
            "  threadgroup uint tg[%d];\n"
            "  tg[t] = t;\n"
            "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            "  C[t] = tg[(t + 1u) & %du] + 100u;" % (THREADS, THREADS - 1)),
 "threadgroup": (_threadgroup_ir, _threadgroup_py,
                "  threadgroup uint tg[%d];\n"
                "  tg[t] = B[t] + 1u;\n"
                "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
                "  C[t] = tg[(t + 1u) & %du] * 2u;" % (THREADS, THREADS - 1)),
 "ifelse": (_ifelse_ir, _ifelse_py,
            "  uint v = B[t];\n"
            "  C[t] = v + 1u;\n"
            "  if (t > 8u) { C[t] = v * 5u; if (t > 20u) C[t] = v * 3u; }\n"
            "  if (t < 9u) C[t] = v + 100u;"),
 "liveidx": (_liveidx_ir, _liveidx_py,
             "  uint i = (t + 1u) & %du, j = (t + 2u) & %du, k = (t + 3u) & %du;\n"
             "  C[t] = B[i] + B[j] + B[k] + i + j + k;"
             % (THREADS - 1, THREADS - 1, THREADS - 1)),
 "loadbranch": (_loadbranch_ir, _loadbranch_py,
                "  uint v = B[t];\n"
                "  C[t] = v;\n"
                "  if ((v & 7u) > 3u) C[t] = B[(t + 3u) & %du] + 200u;" % (THREADS - 1,)),
 "tgbarrier2": (_tgbarrier2_ir, _tgbarrier2_py,
                "  threadgroup uint s[%d];\n"
                "  s[t] = B[t] + 1u;\n"
                "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
                "  uint a = s[(t + 1u) & %du] * 2u;\n"
                "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
                "  s[t] = a;\n"
                "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
                "  C[t] = s[(t + 2u) & %du];" % (THREADS, THREADS - 1, THREADS - 1)),
 "identity": (_identity_ir, _identity_py,
              "  C[t] = t;"),
 "shiftr": (_shiftr_ir, _shiftr_py,        "  C[t] = B[t] >> 3u;"),
 "addsatk": (_addsatk_ir, _addsatk_py,    "  C[t] = addsat(B[t], 0xFFFFFF00u);"),
 "subsatk": (_subsatk_ir, _subsatk_py,    "  C[t] = subsat(B[t], 4000u);"),
 "andreg": (_andreg_ir, _andreg_py,       "  C[t] = B[t] & B[t+1];"),
 "orreg": (_orreg_ir, _orreg_py,          "  C[t] = B[t] | B[t+1];"),
 "xorreg": (_xorreg_ir, _xorreg_py,       "  C[t] = B[t] ^ B[t+1];"),
 "subreg": (_subreg_ir, _subreg_py,       "  C[t] = B[t] - B[t+1];"),
 "ushort16": (_ushort16_ir, _ushort16_py, "  C[t] = (uint)(ushort)B[t];"),
 "halfzero": (_halfzero_ir, _halfzero_py,
              "  device ushort *H = (device ushort *)C;\n"
              "  H[2u * t] = 0;\n"
              "  H[2u * t + 1u] = (ushort)B[t];"),
 "srconst": (_srconst_ir, _srconst_py,
             "  C[t] = 7u;"),
 "srindex": (_srindex_ir, _srindex_py,
             "  uint idx = tgp * %du + tt;\n"
             "  C[idx] = B[idx] * 2u + tt;" % (THREADS,)),
 "wideindex": (_wideindex_ir, _wideindex_py,
               "  uint i = (t << 6) + 100u;\n"
               "  C[t] = B[i] + i;"),
 "gemvloop": (_gemvloop_ir, _gemvloop_py,
              "  uint acc = 0;\n"
              "  for (uint k = 0; k < %du; k++) acc += B[t * %du + k];\n"
              "  C[t] = acc;" % (LOOP_K, LOOP_K)),
 "dotloop": (_dotloop_ir, _dotloop_py,
             "  uint acc = 0;\n"
             "  for (uint k = 0; k < %du; k++) acc += B[t * %du + k] * B[%du + k];\n"
             "  C[t] = acc;" % (LOOP_K, LOOP_K, VEC_BASE)),
 "divgemv": (_divgemv_ir, _divgemv_py,
             "  uint row = t * %du;\n"
             "  uint x = t; uint acc = 0;\n"
             "  if (x < %du) { do { acc += B[row + x]; x += 1u; } while (x < %du); }\n"
             "  C[t] = acc;" % (DIV_LIM, DIV_LIM, DIV_LIM)),
 "gemvreduce": (_gemvreduce_ir, _gemvreduce_py,
                '  threadgroup uint s[%d];\n  s[t] = B[t] * B[%du + t] + B[t + %du] * B[%du + t];\n  threadgroup_barrier(mem_flags::mem_threadgroup);\n  if (t < 16u) s[t] += s[t + 16u];\n  threadgroup_barrier(mem_flags::mem_threadgroup);\n  if (t <  8u) s[t] += s[t +  8u];\n  threadgroup_barrier(mem_flags::mem_threadgroup);\n  if (t <  4u) s[t] += s[t +  4u];\n  threadgroup_barrier(mem_flags::mem_threadgroup);\n  if (t <  2u) s[t] += s[t +  2u];\n  threadgroup_barrier(mem_flags::mem_threadgroup);\n  if (t <  1u) s[t] += s[t +  1u];\n  threadgroup_barrier(mem_flags::mem_threadgroup);\n  C[t] = s[0] + t;' % (THREADS, RED_VEC, THREADS, RED_VEC + THREADS)),
 "bitops": (_bitops_ir, _bitops_py,
            "  uint v = B[t];\n"
            "  C[t] = (v << 3) + (v & 0xFFu) + (v - 5u) + v*3u;"),
}


def inputs(name=None):
    """Integer ramp by default. A kernel may name its own pattern - a float kernel needs float BIT
    PATTERNS in the buffer, and the integer ramp read as floats is a field of denormals, which is
    the one place two compilers are entitled to disagree."""
    A = np.zeros(N * N, np.uint32)
    if name and KERNELS.get(name) and len(KERNELS[name]) > 3 and KERNELS[name][3] == "float":
        # 1.0, 1.5, 2.0, ... - every value exactly representable, so the comparison is bit-exact
        B = (1.0 + 0.5 * np.arange(N * N, dtype=np.float32)).astype(np.float32).view(np.uint32)
    else:
        B = ((np.arange(N * N, dtype=np.uint64) * 7 + 3) % 4096).astype(np.uint32)
    C = np.full(N * N, 0xDEADBEEF, np.uint32)
    return A, B, C


def expected(B):
    """What the kernel says, in Python. A third opinion, computed from neither compiler."""
    out = np.full(THREADS, 0, np.uint64)
    for t in range(THREADS):
        s = sum(int(B[4 * t + j]) * W[j] for j in range(4)) + BIAS
        s &= 0xFFFFFFFF
        out[t] = s if s > THRESH else 0
    return out.astype(np.uint32)


def _lib():
    L = ctypes.CDLL(os.path.join(T, "spike", "accel", "libaccel.dylib"))
    assert L.ac_init() == 0
    L.ac_lib_from_data.argtypes = [ctypes.c_char_p]
    L.ac_compile.argtypes = [ctypes.c_char_p]
    L.ac_archive.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    L.ac_pipeline_from_archive.restype = ctypes.c_void_p
    L.ac_pipeline_from_archive.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    L.ac_run_ps_es.restype = ctypes.c_int
    L.ac_run_ps_es.argtypes = [ctypes.c_void_p] + [ctypes.c_void_p] * 3 + [ctypes.c_uint] * 5
    return L


def dispatch_threads(name):
    # Constant slot stores need one invocation: identical writes by multiple
    # lanes are still overlapping writes, not a useful memory-semantics test.
    # halfslot and halfvec have the same contract, which integration's 67746b09 spotted for the first of
    # them: every store names a fixed byte displacement or slot - the odd half element included - so a
    # multi-lane dispatch would race the observation rather than test the encoding. halfvec is here for the
    # same reason and was missed by that commit, not excluded by it.
    return 1 if name in ("rangestore", "rangestore_read", "halfslot", "halfvec", "wordslot", "duprange", "vec2load") else THREADS


def output_count(name, B):
    count = max(THREADS, len(KERNELS[name][1](B)))
    if count > N * N:
        raise ValueError("kernel expectation exceeds the output allocation")
    return count


def run_g17(name, out_path):
    # THE LOOP OPT-IN, per kernel. g17cc refuses a back edge unless the caller asserts the loop is
    # bounded - it has no trip-count analysis and an unbounded loop is a GPU hang. These kernels
    # are counted with a constant bound, which is that assertion.
    if name in LOOP_KERNELS:
        os.environ["G17_ALLOW_LOOP"] = "1"
    import g17cc, g17program, g17oracle
    import g17imgconst_scalar as K
    os.makedirs(SCRATCH, exist_ok=True)
    L = _lib()
    # G17_FRONT=1 BUILDS THE IR FROM THE KERNEL'S OWN METAL SOURCE instead of its Python builder,
    # through tools/g17front.py. That makes this harness a test of the FRONT END too: the same
    # source goes to Apple's compiler and to ours, and the comparison is unchanged. It is off by
    # default because the two paths do not produce the same program - AIR has already folded work
    # the hand-built IR spells out - and a byte comparison is not the check that matters. Agreement
    # with Apple and with the arithmetic is.
    if os.environ.get("G17_FRONT") == "1":
        import tempfile, g17front
        src = (_HDR_ATOMIC if name in ATOMIC_KERNELS else _HDR) % KERNELS[name][2]
        with tempfile.NamedTemporaryFile("w", suffix=".metal", delete=False) as fh:
            fh.write(src)
            mp = fh.name
        try:
            fn_used = g17front.from_metal(mp)
        finally:
            os.unlink(mp)
        p = g17cc.compile_function(fn_used)
    else:
        fn_used = KERNELS[name][0]()
        p = g17cc.compile_function(fn_used)
    text = bytes.fromhex("0e000000") + g17oracle.FILLER * ((g17oracle.ENTRY - 4) // 2) + p.code
    if len(text) % 16:
        text += g17oracle.FILLER * ((16 - len(text) % 16) // 2)
    # THE BINDING COMES FROM THE EMITTED OPCODES, not from a flag beside the kernel. A threadgroup
    # store needs a threadgroup binding in the image AND a threadgroup length at dispatch, and
    # missing either returns zeros that look exactly like a broken opcode - which is what the
    # ladder's threadgroup rung did until the binding class was found. Deriving it from the code
    # means a kernel that starts using threadgroup memory cannot forget to ask for it.
    #
    # AND THAT IS THE GENERAL RULE, not a local convenience. The metadata records OPERATIONS, not
    # resources, and the ISA peer's naming has now arrived there five times independently:
    #
    #     slot 9   a threadgroup argument USED          slot 18  an argument or an allocation
    #     slot 19  an imageblock WRITTEN                slot 30  a uniform-address fetch RMW
    #     slot 44  a tensor_ops matmul, not a tensor
    #
    # Declaring a thing is never what gets recorded. This backend composes its metadata from the
    # SIGNATURE, which is a declaration - so every one of those slots is a program fact the
    # signature cannot supply, and the line above is the one place that is already handled.
    #
    # WHAT THAT MEANS FOR ANYTHING ADDED LATER: an emitter that starts producing an imageblock
    # write or an atomic at a uniform address must set slot 19 or slot 30 from the emitted code the
    # way the threadgroup binding is set here, and will get no help from the signature. Neither is
    # emittable today, so this is a constraint rather than a defect.
    import g17ref
    spans = [(a, l, op) for a, l, op in g17ref.walk(text, g17oracle.ENTRY)]
    ops = {op for _a, _l, op in spans}
    tg = bool(ops & g17oracle.TG_OPCODES)
    # THE BUFFER LIST IS THE KERNEL'S, NOT A CONSTANT. Every kernel until the atomic used B and C
    # only, so [1, 2] was right by accident; atomicadd targets buffer 0 and binding it is not
    # optional - an unbound buffer returns zeros that look exactly like a broken opcode, which is
    # the same failure the threadgroup binding had.
    # THE BINDINGS COME FROM THE FUNCTION THAT WAS COMPILED, not from the Python builder. Under
    # G17_FRONT=1 the code comes from the kernel's Metal source and its buffer list can differ from
    # the hand-built one - and building the image for one list while the code was compiled against
    # the other is a store at a rank the image does not declare, which returns the fill value at
    # status 0 and looks exactly like a broken opcode.
    slots = sorted({bb.slot for bb in fn_used.buffers})
    P = g17program.G17Program(text=text, entry=g17oracle.ENTRY,
                              buffers=(slots + [0]) if tg else slots,
                              binding_kinds=(["device_buffer"] * len(slots) + ["threadgroup"]
                                             if tg else None),
                              # THE CLASS IS CHOSEN BY WHAT THE PROGRAM DOES. g17mdgen keys a
                              # recorded metadata class on the memory opcodes emitted, and computes
                              # the atomic reduction identity from the operation code in the bytes -
                              # neither reachable without these two. Passing them is how an atomic
                              # kernel can get a class carrying slot 30, the uniform-address RMW.
                              spans=spans, memory_opcodes=sorted(ops),
                              stats_md=K.STATS_MD)
    arc, lib = SCRATCH + "/g17.arc.metallib", SCRATCH + "/g17.lib.metallib"
    open(arc, "wb").write(P.image()); open(lib, "wb").write(P.library())
    assert L.ac_lib_from_data(lib.encode()) == 0
    ps = L.ac_pipeline_from_archive(arc.encode(), b"k")
    assert ps, "no pipeline from the generated image"
    A, B, C = inputs(name)
    threads = dispatch_threads(name)
    if tg:
        L.ac_run_ps_abc.restype = ctypes.c_int
        L.ac_run_ps_abc.argtypes = [ctypes.c_void_p] + [ctypes.c_void_p] * 3 + [ctypes.c_uint] * 6
        st = L.ac_run_ps_abc(ctypes.c_void_p(ps), A.ctypes.data, B.ctypes.data, C.ctypes.data,
                             N, 4, threads, 1, 1, THREADS * 4)
    else:
        st = L.ac_run_ps_es(ctypes.c_void_p(ps), A.ctypes.data, B.ctypes.data, C.ctypes.data,
                            N, 4, threads, 1, 1)
    json.dump({"status": int(st), "C": [int(x) for x in C[:output_count(name, B)]], "threadgroup": tg,
               "dispatched_threads": threads,
               "instructions": len(p.layout), "bytes": len(p.code),
               "inherited": P.census().get("inherited", 0),
               # THE EXECUTED BYTES, RETAINED. Every run before this recorded outputs and two agreement
               # flags but no code, so what executed could only be inferred from a later compile
               # (gap compiler.gap.evidence.e2e-code-identity). These are the bytes this process built
               # the pipeline from, and the image digest is of the archive it loaded.
               **retained_code(p.code), "image_sha256": _sha256(P.image())}, open(out_path, "w"))


def _sha256(data):
    import hashlib
    return hashlib.sha256(bytes(data)).hexdigest()


def retained_code(code):
    """The fields that make a g17 record carry its own program: the bytes and their digest."""
    return {"code_hex": bytes(code).hex(), "code_sha256": _sha256(code)}


LOOP_OPCODES = frozenset((450, 458, 578, 579))


def compile_now(name):
    """Compile one kernel TODAY, exactly as run_g17 does without G17_FRONT, dispatching nothing.
    The loop opt-in is the same per-kernel assertion run_g17 makes, and is restored afterwards."""
    import g17cc, g17ref
    prior = os.environ.get("G17_ALLOW_LOOP")
    try:
        if name in LOOP_KERNELS:
            os.environ["G17_ALLOW_LOOP"] = "1"
        p = g17cc.compile_function(KERNELS[name][0]())
    finally:
        if prior is None:
            os.environ.pop("G17_ALLOW_LOOP", None)
        else:
            os.environ["G17_ALLOW_LOOP"] = prior
    ops = [op for _a, _l, op in g17ref.walk(p.code, 0)]
    return p, sorted(set(ops) & LOOP_OPCODES)


RESULTS = os.path.join(T, "isa", "g17-endtoend-results.json")
COMPILED_NOW = os.path.join(T, "isa", "g17-endtoend-compiled-now.json")

# THE COMMIT THAT LAST WROTE isa/g17-endtoend-results.json (a full run writes it; a partial one cannot).
# Measured 2026-09-23 by compiling every kernel from `git archive 155a8452` with a fresh interpreter:
# these four compiled to DIFFERENT bytes there than today (the four "changed legacy programs" that
# results/g17-bitwise-runtime-v2 then executed), and the other 32 to the same bytes. That is the only
# comparison available with what the recorded run executed, and it is not proof in either direction
# for the 32: the run need not have been from a clean tree at that commit.
RECORDING_COMMIT = "155a8452"
DIFFERENT_AT_RECORDING_COMMIT = {
    "andreg": "ab3d5c13b1c4", "orreg": "e751a20d8ba5", "xorreg": "6e5a15cf5a53", "bitreg": "3a8b057eff95",
}

# TODAY'S BYTES EXECUTED ELSEWHERE: a retained receipt whose file map pins program.bin to the digest.
# That proves these bytes ran on the GPU in THAT receipt's image and harness - not that they are what
# the end-to-end run executed (for these four they are provably not; see above).
EXECUTED_ELSEWHERE = {
    k: ("results/g17-bitwise-runtime-v2/%s/first-worker/execution.json" % k, "programs/%s/program.bin" % k)
    for k in ("andreg", "orreg", "xorreg", "bitreg")
}


def _receipt_digest(path, member):
    """The program.bin digest a retained receipt pins, or None if the receipt is absent or silent."""
    full = os.path.join(T, path)
    if not os.path.exists(full):
        return None
    with open(full) as stream:
        doc = json.load(stream)
    if doc.get("status") != "passed" or doc.get("gpu_dispatched") is not True:
        return None
    return (doc.get("files") or {}).get(member)


def identity(name, sha, size, instructions, recorded):
    """How far today's bytes for one kernel can be tied to what its recorded run executed.

    `proven_executed` only when the recorded run itself retained a digest equal to today's (none did
    before retention); `contradicted` when something the run recorded rules today's bytes out;
    `changed_since_recording` when the committed compiler that wrote the record emits other bytes;
    otherwise `unproven`. Bytes executed under ANOTHER receipt are reported beside it, never as this identity."""
    g = (recorded or {}).get("g17") or {}
    out = {"executed_elsewhere": None}
    if name in EXECUTED_ELSEWHERE:
        path, member = EXECUTED_ELSEWHERE[name]
        pinned = _receipt_digest(path, member)
        out["executed_elsewhere"] = {"receipt": path, "pinned": pinned, "equal": pinned == sha}
    if g.get("code_sha256"):
        out["status"] = "proven_executed" if g["code_sha256"] == sha else "contradicted"
        out["reason"] = "the recorded run retained code_sha256 %s" % g["code_sha256"][:12]
    elif g and (g.get("bytes") != size or g.get("instructions") != instructions):
        out["status"] = "contradicted"
        out["reason"] = "recorded %s bytes/%s instructions, today %d/%d" % (
            g.get("bytes"), g.get("instructions"), size, instructions)
    elif name in DIFFERENT_AT_RECORDING_COMMIT:
        # Not `contradicted`: that needs something the run itself recorded. This is a compile of the
        # committed tree the record was written from, and the change landed four days later (f05ce4d8).
        out["status"] = "changed_since_recording"
        out["reason"] = ("the compiler at %s, which wrote the record, emits %s... - not today's bytes"
                         % (RECORDING_COMMIT, DIFFERENT_AT_RECORDING_COMMIT[name]))
    else:
        out["status"] = "unproven"
        out["reason"] = ("no digest was retained; size and instruction count match the record and the "
                         "compiler at %s emits the same bytes, which is consistent, not proof" % RECORDING_COMMIT)
    return out


def compiled_now(names=None):
    """Today's bytes for every recorded kernel, labelled as NOT the executed bytes. No dispatch."""
    with open(RESULTS) as stream:
        recorded = json.load(stream)["kernels"]
    names = sorted(recorded) if names is None else list(names)
    kernels = {}
    for name in names:
        p, loops = compile_now(name)
        sha = _sha256(p.code)
        kernels[name] = {
            "compiled_now": dict(retained_code(p.code), bytes=len(p.code), instructions=len(p.layout)),
            "loop_opcodes": loops,
            "loop_bounded_by_harness": name in LOOP_KERNELS,
            "identity": identity(name, sha, len(p.code), len(p.layout), recorded.get(name)),
        }
    counts = {}
    for k in kernels.values():
        counts[k["identity"]["status"]] = counts.get(k["identity"]["status"], 0) + 1
    return {"label": "compiled_now: bytes today's compiler emits for each recorded end-to-end kernel. "
                     "These are NOT the executed bytes; the recorded run retained none.",
            "source": "tools/g17endtoend.py --compiled-now",
            "recording_commit": RECORDING_COMMIT,
            "identity_counts": dict(sorted(counts.items())),
            "kernels": kernels}


def dispatch_plan(names):
    """The kernels a retained re-run may dispatch, or ValueError naming the first it must refuse.
    A kernel whose bytes carry a loop opcode is refused unless this harness already runs it bounded
    (LOOP_KERNELS: a constant trip count asserted through G17_ALLOW_LOOP)."""
    plan = []
    for name in names:
        if name not in KERNELS:
            raise ValueError("unknown kernel %r" % name)
        _p, loops = compile_now(name)
        if loops and name not in LOOP_KERNELS:
            raise ValueError("%s carries loop opcodes %r and this harness does not run it bounded" % (name, loops))
        plan.append((name, loops))
    return plan


def record(out_path, names):
    """Run the named kernels through one() - both sides, one process each - retaining the g17 bytes,
    and write a PARTIAL record to out_path. Never the shared results file: a partial run must not
    replace a complete one (see main)."""
    if os.path.abspath(out_path) == os.path.abspath(RESULTS):
        raise ValueError("refusing to write a partial record over %s" % RESULTS)
    plan = dispatch_plan(names)
    here = os.path.abspath(__file__)
    out, allok = {}, True
    for name, loops in plan:
        ok, rec = one(name, here)
        allok &= ok
        if "error" not in rec:
            p, _loops = compile_now(name)
            rec["compiled_now_sha256"] = _sha256(p.code)
            rec["executed_equals_compiled_now"] = rec["g17"].get("code_sha256") == rec["compiled_now_sha256"]
        rec["loop_opcodes"] = loops
        out[name] = rec
    with open(out_path, "w") as stream:
        json.dump({"threads": THREADS, "weights": list(W), "bias": BIAS, "threshold": THRESH,
                   "partial": True, "kernels": out}, stream, indent=1)
    return 0 if allok else 1


def run_metal(name, out_path):
    os.makedirs(SCRATCH, exist_ok=True)
    L = _lib()
    src = (_HDR_ATOMIC if name in ATOMIC_KERNELS else _HDR) % KERNELS[name][2]
    assert L.ac_compile(src.encode()) == 0, "Metal would not compile the reference kernel"
    arc = SCRATCH + "/metal.arc.metallib"
    assert L.ac_archive(b"k", arc.encode()) == 0
    ps = L.ac_pipeline_from_archive(arc.encode(), b"k")
    assert ps, "no pipeline from Apple's archive"
    A, B, C = inputs(name)
    st = L.ac_run_ps_es(ctypes.c_void_p(ps), A.ctypes.data, B.ctypes.data, C.ctypes.data,
                        N, 4, dispatch_threads(name), 1, 1)
    # AS MANY ELEMENTS AS THE EXPECTATION NAMES. A slot store above 63 selects the 14-byte forms,
    # and a comparison cut at THREADS could never see them; every earlier kernel's expectation is
    # THREADS long, so this records exactly what it did for them.
    n_out = output_count(name, B)
    json.dump({"status": int(st), "C": [int(x) for x in C[:n_out]]}, open(out_path, "w"))


def one(name, here):
    """Run both sides of one kernel in their own processes and compare. -> (ok, record)."""
    res = {}
    for side in ("g17", "metal"):
        f = "%s/%s.%s.json" % (SCRATCH, name, side)
        if os.path.exists(f): os.unlink(f)
        r = subprocess.run([sys.executable, here, side, name, f], capture_output=True, timeout=900)
        if r.returncode != 0 or not os.path.exists(f):
            return False, {"error": r.stderr.decode()[-600:]}
        with open(f) as stream:
            res[side] = json.load(stream)
        if type(res[side].get("status")) is not int or res[side]["status"] != 0:
            return False, {"error": "%s dispatch did not complete successfully: %r" %
                           (side, res[side].get("status"))}
    _A, B, _C = inputs(name)
    want = KERNELS[name][1](B)
    g, m = res["g17"], res["metal"]
    key = sorted if name in UNORDERED else (lambda x: x)
    rec = {"g17": g, "metal": m, "expected": want,
           "agree_with_apple": key(g["C"]) == key(m["C"]),
           "agree_with_arithmetic": key(g["C"]) == key(want)}
    return rec["agree_with_apple"] and rec["agree_with_arithmetic"], rec


def main():
    here = os.path.abspath(__file__)
    if len(sys.argv) > 3 and sys.argv[1] in ("g17", "metal"):
        return (run_g17 if sys.argv[1] == "g17" else run_metal)(sys.argv[2], sys.argv[3]) or 0
    # --compiled-now [--write|--check]: today's bytes per recorded kernel, no dispatch.
    if len(sys.argv) > 1 and sys.argv[1] == "--compiled-now":
        doc = compiled_now()
        text = json.dumps(doc, indent=1, sort_keys=True) + "\n"
        if "--write" in sys.argv:
            with open(COMPILED_NOW, "w") as stream:
                stream.write(text)
        elif "--check" in sys.argv:
            with open(COMPILED_NOW) as stream:
                if stream.read() != text:
                    print("%s is stale: today's compile differs from it" % COMPILED_NOW)
                    return 1
        print(json.dumps(doc["identity_counts"]))
        return 0
    # --record OUT kernel...: a retained partial re-run (dispatches both sides of each kernel).
    if len(sys.argv) > 2 and sys.argv[1] == "--record":
        return record(sys.argv[2], sys.argv[3:])
    names = [sys.argv[1]] if len(sys.argv) > 1 and sys.argv[1] in KERNELS else list(KERNELS)
    os.makedirs(SCRATCH, exist_ok=True)
    out, allok = {}, True
    print("ONE KERNEL, TWO COMPILERS - %d kernel%s\n" % (len(names), "" if len(names) == 1 else "s"))
    for nm in names:
        ok, rec = one(nm, here)
        allok &= ok
        if "error" in rec:
            print("  %-10s BUILD/RUN FAILED\n%s" % (nm, rec["error"])); continue
        g = rec["g17"]
        print("  %-10s %2d instr  %3d bytes  %d inherited  |  == Apple: %-3s  == arithmetic: %s"
              % (nm, g["instructions"], g["bytes"], g["inherited"],
                 "YES" if rec["agree_with_apple"] else "NO",
                 "YES" if rec["agree_with_arithmetic"] else "NO"))
        if not (rec["agree_with_apple"] and rec["agree_with_arithmetic"]):
            for i in range(min(6, THREADS)):
                print("       t=%-3d g17=%-12d metal=%-12d expected=%d"
                      % (i, g["C"][i], rec["metal"]["C"][i], rec["expected"][i]))
        out[nm] = rec
    # THE RESULTS ARE WRITTEN DOWN, so the regression can hold them without dispatching - but ONLY
    # BY A FULL RUN. A single-kernel invocation used to overwrite the shared record with one entry,
    # which is how `python3 tools/g17endtoend.py tgatomic` silently deleted seventeen kernels'
    # results and fired the case that requires the atomic to be present. A partial measurement must
    # not be able to replace a complete one; running one kernel is a debugging convenience and
    # leaves the record alone.
    if set(out) >= set(KERNELS):
        json.dump({"threads": THREADS, "weights": list(W), "bias": BIAS, "threshold": THRESH,
                   "kernels": out}, open(os.path.join(T, "isa", "g17-endtoend-results.json"), "w"),
                  indent=1)
    else:
        print("\n  partial run (%d of %d kernels) - the results file is left alone"
              % (len(out), len(KERNELS)))
    print("\n  %s" % ("all kernels agree with Apple and with the arithmetic" if allok
                       else "SOME KERNEL DISAGREES - see above"))
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
