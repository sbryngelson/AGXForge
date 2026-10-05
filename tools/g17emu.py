#!/usr/bin/env python3
"""g17emu: run a delivered G17 program FROM ITS BYTES on the CPU, with no GPU dispatch (MM 25.145).

It reproduces tools/g17bundlerun.m, the hardware runner behind every g17deliver check: the bundle's a.f16, b.f16 and
c.f32 each sit at byte 128 of a buffer pre-filled with 0xA5; base 1 binds a, b, c at slots 1, 2, 3 and base 0 binds
c, a, b at 0, 1, 2; the grid is threads / group threadgroups of `group` threads; the output is buffer c's c.f32-sized
window after the program has run.

THE RULES.
  - The program is decoded from its bytes by Apple's decoder (g17packedcheck.decode, the tokens tools/g17halftrace.py
    reads), never taken from the IR that produced it: a checker that replays the compiler's intent proves only that
    the compiler agrees with itself.
  - Every opcode's handler names its evidence (EVIDENCE below: the execution receipts in isa/, the addressing rules,
    the sections of the machine model). An opcode with no handler, a mode word no receipt executed, a register class
    or special register not measured, is REFUSED by name. Nothing is guessed.
  - Two tiers. "strict" admits only semantics an isolated receipt executed; "wp" also admits the named semantics
    that only whole programs verified on hardware (index scale 4, the 10-byte op13575, ...) and records each use in
    Machine.admitted.
  - Registers are one file per thread, R0..R143, resolved by NAME from Apple's register table
    (agxforge.g17.model.registers(), through agxforge.g17.registerdomain): Rn, its halves RnL / RnH, 32-bit tuples
    Rn_Rn+1.., FLAGn, SR_*. Any other class (the IR registers, mixed-half tuples) is refused. Every register read
    carries a modifier operand: bit 1 negate, bit 2 absolute value, bit 4 RELEASE, bit 5 keep. A release takes effect
    at the end of its instruction, a write to the register cancels it, and a released register may read zero or its
    old value on the hardware, so a later read is admitted only where the old value is zero (both outcomes agree).
  - Every thread of the grid runs each instruction in program order (lockstep), under the measured exec-mask stack
    (push-and-mask, pop, a back edge taken iff any lane is active, a skip taken iff none is). tensor.mac runs per
    simdgroup through the measured fragment maps and g17tensorcommonruntime's pinned MMA arithmetic.

    python3 tools/g17emu.py BUNDLE_DIR --threads T --group G [--base 1] [--tier wp] [--out FILE]
    python3 tools/g17emu.py --receipts [--tier wp]      replay every whole-program hardware receipt
"""
import argparse
import os
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

FILL, PAD = 0xA5, 128                 # g17bundlerun: inputs at byte 128 of a 0xA5-filled buffer
R32 = 105                              # R0's register id (Apple's register table: agxforge.g17.model.registers())
NREG = 144                             # R0..R143 in Apple's register table
MOD_NEG, MOD_ABS, MOD_RELEASE, MOD_KEEP = 2, 4, 16, 32

# The receipts each handler rests on. A handler with no entry here does not exist.
EVIDENCE = {
    14059: "read_sr: isa/g17-execution-readsr, -srsweep; SR numbers isa/g17-sr-number-prediction.txt, cc.SR",
    11842: "movimm: isa/g17-execution-requestedforms",
    10825: "mul: isa/g17-execution-delivery, -lanes, -sweep",
    10282: "add (reg + reg): isa/g17-execution-delivery, -laneforms, -sweep; isa/g17-addressing-rules.json",
    12682: "load 32-bit: isa/g17-execution-memforms, -constspace; isa/g17-addressing-rules.json",
    12646: "load 16-bit: isa/g17-execution-memforms; isa/g17-addressing-rules.json",
    1004: "fadd.imm (f16 source widened to f32): isa/g17-execution-arity1split, -conv12lowhalf, -residue",
    998: "fadd f32: isa/g17-execution-float, -floatmods, -functions (executed and checked, isa/g17.yaml)",
    1016: "cvt.f32.f16 (round to nearest even): isa/g17-execution-sweep, -regate",
    798: "ffma.f16, binary16 a*b + c with one rounding: MM 25.196 (Apple's fma(half) kernel, 0 of 65,536 random "
         "triples off the exact rounding, 9,890 off multiply-then-add; our compiled kernel against it, "
         "evidence/g17-ffma16-v1)",
    17229: "store 32-bit: isa/g17-execution-memforms, -storebit2; isa/g17-addressing-rules.json",
    17193: "store 16-bit: isa/g17-execution-memforms; isa/g17-addressing-rules.json",
    684: "end: isa/g17-execution-controlflow",
    3658: "recip: within one ulp of 1/x, not correctly rounded (MM op table, 25.136: 931 of 1,024 probes) - so it is "
          "emulated BOTH ways (the two floats bracketing 1/x) and a program is vouched for only if its output agrees",
    3850: "rsqrt seed: an exact function tabulated in isa/g17-rsqrt-seed.npz (MM op table, 25.141.16), "
          "g17decodestep.rsqrt_seed",
}

EVIDENCE[14169] = ("simd.shuffle_xor: dst, mod, src, mod, mask - lane k reads src from lane k ^ mask of its 32-lane "
                   "simdgroup (MM op table: the final immediate is the XOR lane mask; executed per lane in "
                   "isa/g17-execution-lanes, -lanemask, -perlane)")

EVIDENCE[13288] = ("store.tg: value, mod, width, base, 0, index, index mod, displacement, element bytes - threadgroup "
                   "memory at base + index x element + displacement (cc TG_STORE_TEMPLATE, agxforge/g17/forms.py "
                   "TG_MEMORY_OPCODES; the base operand, printed bin(op0,const(2n),2), is general register rn: the "
                   "delivered programs write it with movimm just before, MM 25.145)")
EVIDENCE[12364] = "load.tg: dst, mode, width, base, 0, index, index mod, displacement, element bytes - as store.tg"
TG_BYTES = 32768                        # threadgroup memory per threadgroup: the measured limit (MM table 0.x)

EVIDENCE.update({
    14060: "read_sr16: whole-program; the lane value executed through op14059 (notes.toml:7106). It writes ONLY its half: "
           "with the register live and its high half nonzero, the high half survives (isa/g17-execution-halfwrite-"
           "results.json, 3 runs identical; MM 25.147's probe)",
    10279: "add-imm, source scaled by smod bits 8-10: confound receipts, ledger other.toml:1709-1750",
    11667: "sub (wraps): isa/g17-execution receipts g11667, m11667, d11667",
    586: "mov: isa/g17-execution-moves",
    423: "and-imm: confound3 op423.operand4 receipts", 426: "and-imm on a 16-bit source: receipt u426",
    424: "and: shiftpairs t424; isa/g17-execution-fits.json", 13575: "or: shiftpairs t13575",
    428: "and16 (register mask only): shiftpairs t428", 13574: "or-imm: receipt u13574",
    14391: "shl-imm: receipt u14391", 17013: "shr-imm (logical): receipt u17013",
    17016: "shr16-imm with a result width: receipt u17016, results/g17-tensor-setup-runtime-v1",
    14392: "shl by register (k = amt & 0x7F, k >= 32 -> 0): satwrap, marginrepair receipts",
    17014: "shr by register: as op14392",
    3290: "fmul: isa/g17-execution-functions, -sweep, -floatmods", 2190: "ffma, single rounding: claims.toml:4030",
    11179: "cvt.i2f (4 unsigned, 5 signed, round to nearest even): claims.toml:3757",
    10369: "cmp: printed cc = 8 | field (isa/g17-condition-codes.toml); isa/g17-execution-exec; notes.toml:6680",
    11372: "csel.reg -> 1/0: notes.toml:694 (40/40, incl. (-5, 3)); the predicates receipt",
    11375: "select: confounds, newforms, union, switch receipts; cc 15 bit test (claims.toml:3725)",
    9700: "fselect: lane.fsel_code0..7 (claims.toml:4657)",
    582: "exec push-and-mask: isa/g17-execution-exec, -maskdepth, -maskseq; isa/g17-exec-mask.txt",
    577: "exec pop: as op582", 458: "back edge, taken iff any lane is active: compiler.toml:6330",
    462: "skip, taken iff no lane is active: MM 16602-16616", 447: "threadgroup barrier: cf.barrier.op447",
    14156: "device fence: MM 16465-16480",
})
EVIDENCE[10822] = "mul-imm (low 32 bits): isa/g17-execution-delivery, -marginpairs, -residue, -sweep"
EVIDENCE[17770] = "xor-imm: isa/g17-execution-sweep"
EVIDENCE[12709] = "load 128-bit (int4) into a 4-register tuple: isa/g17-addressing-rules.json"
EVIDENCE[12691] = "load 64-bit (long) into a register pair: isa/g17-addressing-rules.json"
# The lanes receipts pick one semantics among candidates, all 32 lanes matching (isa/g17-execution-lanes-results.json,
# field "fits"). Replaying those receipts is therefore a consistency check for these six, not independent evidence.
EVIDENCE.update({
    10826: "madd: a*b+c mod 2^32 (lane.int_madd fits)",
    10239: "add, unsigned saturating (lane.int_addsat: fits unsigned saturate, not signed or wrap)",
    11624: "sub, unsigned saturating (lane.int_subsat: fits unsigned saturate, not signed or wrap)",
    11190: "not: ~x (lane.int_not fits)",
    9986: "msb: index of the highest set bit, 0 -> 0xFFFFFFFF (lane.int_msb fits, not clz or 0 -> 0)",
    14047: "bit reverse (lane.int_reverse fits)",
})

EVIDENCE[12674] = ("tile/vector load: dst tuple, mode, width, binding, 0, index, index mod, byte displacement, element "
                   "bytes, component mask - popcount(mask) consecutive elements into the tuple, lowest address in "
                   "the lowest half (the tensor campaign's tlower loads; the qmm GEMMs, hardware bit-exact)")
EVIDENCE[17257] = "tile/vector store: value tuple, mod, width, binding, 0, index, index mod, byte displacement, " \
                  "element bytes, component mask - as op12674"
EVIDENCE[5107] = ("tensor.mac.init D = A B (fp16 A, B; fp32 D): the pinned MMA arithmetic "
                  "(g17tensorcommonruntime._mma16 / _gemm_mma_fast, bit-exact on hardware) on the measured fragment "
                  "maps (A and D: pos, B: pos_b; recon section 144, tools/g17layerbench.py)")
EVIDENCE[5106] = "tensor.mac D = A B + C: as op5107, the C-first chain of _mma16 (acc = C, then + Q0..Q3)"
# bf16 A/B (type code 3): the same arithmetic on exact products (recon section 136 part 2, 79,500 tiles); checked here
# against the p13 sg8_grid2 and wide_n192_split bundles' GPU output. Transposed A (code + 32): PERM_AT.
EVIDENCE[5101] = ("tensor.mac.init with an fp32 A (8 registers, the A/D layout): A's mantissa cut to 10 bits "
                  "(& 0xFFFFE000) before exact products - _mma16 truncate_a, as register-O attention's P.V "
                  "(g17prefillmma.mma_prefill_reference) and the relaxed-fp32 finding")
EVIDENCE.update({
    # the formwidths receipts (isa/g17-execution-formwidths*) replay these; each ALSO rests on the record named
    1006: "fadd.imm a + K8 (float immediate, isa/g17-float-immediate.toml): D1006.l12 sweep (+1.0 on 1, 2, 4, 8)",
    904: "sat(a + K8), K = +-0 only: fsat.op904_12 (32 lanes, negatives, -0 -> +0, > 1; isa/g17-execution-fsat)",
    1014: "half(a32 + b32): D1014.l12, z1014 (overflow -> inf), t1014 (NaN -> 0x7e00, tiny negative -> -0)",
    1015: "half(a32 + b16), half subnormals kept: m1015 unique fit f32_plus_f16_to_f16 (arity2split)",
    3306: "half(a32 * b32): m3306 fit hmul_f32_operands; the width receipt's 0x3e00 rules out its bf16 rival",
    2198: "a K8 + c: t2198 (16-byte form: a x -0.5, NaN -> 0x7fc00000, subnormal inputs flushed)",
    2200: "a K1 + K2: u2200 (marginpairs, 31 cases: 0.5 - 0.5 a, subnormals flushed, NaN -> 0x7fc00000)",
    17771: "xor: m17771 / g17771 unique fits (bitwise_6), op17771.at4 (shortforms2), delivery and sweep records",
    13588: "or16 lo16(a) | lo16(b): t13588 (shiftpairs, 17 cases); the peer's 0x0f0f | 0x3333 (slice.py:2562)",
    17784: "xor16 lo16(a) ^ lo16(b): t17784 (shiftpairs); the peer's 0x3c3c (slice.py:2563)",
    554: "movimm, immediate 0 only (every receipt and 1228/1231 corpus uses; isa/g17-imageblock-execution.txt:36)",
    11666: "a - imm (wraps): D11666.l12 / d11666.l12.s1 (4096 -> 4081, imm 15 decoded from 270004ba2d00a202c8030b00)",
    10283: "add a32 + zext(b16): t10283 (shiftpairs), g10283 (marginrepair) unique fits add_b16 (the margin re-probe "
           "against sign extension)",
    590: "mov16 dst = low 16 of src: move.half (isa/g17-execution-moves, mode 0); w590.truncwidth / g590 trunc16 fits",
    3818: "ftrunc: op3818.trunc (2.5 -> 2, -0.3 -> -0, -1.5 -> -1), q3818 unique ftrunc fit (arity1split)",
    16805: "asr by an immediate, source modifier 24 only: op16805.operand4_3 / _5 (confound3, asr3 / asr5), "
           "lane.form_sar5 (laneforms, 32 lanes); g17isamap preregistered 16/16",
    2192: "fp32 FMA with a float-immediate addend, a b + K, rounded ONCE: fma(a, b, 1.0f) exact over 32 lanes (MM 25.84); "
          "fused vs twice separated on 3,186 of 4,096 lanes of Apple's code on the GPU, fused on all (MM 25.184)",
    11182: "u32 -> half, RNE, overflow to inf: 4,096 lanes of Apple's (half)u on the GPU, 2,992 past the range "
           "(isa/g17-rounding-results.json, MM 25.184)",
    17229: EVIDENCE[17229] + "; lanes of ONE simdgroup storing different values to one word: the lowest active lane "
           "wins (isa/g17-execution-storewinner-results.json: all, half, odd and value-reversed arms, 5 runs each)",
    17235: "word store at an immediate address (index + displacement) x element: store.store_word_8 / _10 / _14 "
           "(isa/g17-execution-onestore: C[20], C[70], C[40] written, the neighbour untouched, 3 runs each)",
    12688: "one-word load at an immediate address (the op17235 order, mask 0x812): reproduces Apple's GPU output on "
           "the sources of isa/g17-apple-oracle-receipts.json (MM 25.177)",
    12706: "three-word load at an immediate address (mask 0x872): the constant program's form of op12715 (MM 25.179)",
    14061: "binding-table word: binding K's pointer at halfwords 8 + K / 10 + K, modelled as a tagged pointer that "
           "only a register-pair load base accepts (MM 25.179: Apple's constant programs on the CPU against its GPU)",
    11185: "cvt.i2f into a uniform (op11179's operands, a uniform destination): the constant program's publish (MM 25.179)",
    592: "mov into a uniform (op586 with a uniform destination and one extra mode operand): MM 25.179",
    1038: "fadd.imm into a uniform (op1006 with a uniform destination and one extra mode operand): MM 25.179",
    10289: "16-bit add of an immediate, wrapping: MM 25.195 (Apple's GPU, 4,096 lanes)",
    776: "half + a float immediate, rounded once to half, subnormals kept, NaN -> 0x7e00: MM 25.195 (Apple's GPU, "
         "3 x 4,096 lanes; h + 1/64 gives subnormal results, and flushing differs on 965 lanes)",
    11180: "16-bit integer to fp32, unsigned (code 2) or signed (code 3), exact: MM 25.195 (Apple's GPU, 4,096 lanes)",
    17199: "16-bit store at an immediate address (mask 2065): MM 25.195 (Apple's GPU, full-width census programs)",
    11462: "compare to a 16-bit 1/0 (op11372 into a half): MM 25.195 (Apple's GPU, 4,096 lanes)",
    10285: "zext16(a) + b32: MM 25.195 (Apple's GPU, full-width census programs)",
    17253: "three-word store at an immediate address (mask 0x872), op12706's store: MM 25.192 (Apple's GPU, full-width)",
    435: "16-bit AND with an immediate, dst16 = src16 & imm: MM 25.192 (Apple's GPU, full-width)",
    12697: "two-word load at an immediate address (mask 0x832), op17244's load: MM 25.184 (Apple's GPU, full-width)",
    555: "16-bit move immediate into a half, immediate 0 (a nonzero immediate is written as unknown bits): MM 25.184",
    12715: "four-word load at an immediate address (mask 0x8f2, the lowest register first): as op12688 (MM 25.177)",
    17244: "two-component store at an immediate address (mask 0x832; the two-component store executed at MM 25.62): "
           "as op12688 (MM 25.177)",
    17262: "four-word store at an immediate address (mask 0x8f2): as op12688 (MM 25.177)",
    14157: "simd.shuffle from constant lane 0: lane.shuffle.default (-perlane, every lane reads lane 0's 4096) and "
           "lane.broadcast_first; only with every lane of the simdgroup active (lane 0 vs first-active unseparated)",
})
EVIDENCE.update({
    12656: "int8 tensor load, one word: as op12674 with element 1 and component mask 15, four int8 into one register "
           "(the int8 tensor bundles, hardware bit-exact: MM 25.34, P13 int8_grid4 / int8_sg2_saturate)",
    10384: "tensor.mac int8 -> int32, D = C + A B per 16-product issue, exact; D code 9 wraps mod 2^32, D code 41 clips "
           "C + the issue sum once (recon section 136 parts 5.1 and 6: 2,400 of 2,400 tiles, the per-product clip "
           "refuted); fragments two registers a lane, byte j in register j / 4, on the fp16 slot maps (recon 147)",
    10385: "tensor.mac int8 -> int32 with no C: the exact 16-product sum (fits int32)",
    3770: "rint, round half to even (Metal rint; the requant epilogue's rounding, MM 25.130: the 30 requant arms "
          "bit-exact on hardware, including int8_i2f_ties)",
    9320: "fp32 to int32 (code 5) or uint32 (code 4). Mode operand 1 (the plain cast): truncate, saturate, NaN -> 0, "
          "4,096 lanes of Apple's casts on the GPU each (MM 25.186). Other modes ONLY on integral, finite, in-range "
          "values, where no rounding or saturation mode can move them (MM 25.130.1)",
    17202: "word store at a word index: as op17229 (the packed int8 and half epilogue store, MM 25.105)",
})
EVIDENCE.update({op: "tensor.mac with fp32 operands (%s): the fp32 operand cut to 10 mantissa bits before exact products, "
                     "the C-first chain (_mma16 truncate_a / truncate_b; recon section 136, relaxed fp32)" % what
                 for op, what in ((5100, "fp32 A, fp16 B, + C"), (5104, "fp16 A, fp32 B, + C"),
                                  (5105, "fp16 A, fp32 B"), (5098, "fp32 A and B, + C"), (5099, "fp32 A and B"))})
EVIDENCE[13618] = ("fp8 pack of two fp32 into a 16-bit register, RNE without saturation, NaN input -> canonical NaN: "
                   "measured over all 2^32 fp32 patterns (recon section 138, MM 0.11), negative e4m3fn overflow 0xff "
                   "471 of 471 (results/g17-tensor-lowprec-v1)")
EVIDENCE.update({
    10286: "d32 = zext(lo16 a) + zext(lo16 b): t10286 (shiftpairs, 17 cases), z10286 (residue), add@10286 (widen: "
           "0xFFFF + 1 = 65536, the 16-bit-wrap prediction refuted), D10286.l12",
    13075: "imageblock store, 32-bit, at the lane's (x, y) = coordinate low / high halves and a member byte offset "
           "(MM 25.104, 25.96; isa/g17-imageblock-execution.txt: lanes reading (0, 0) all get lane 0's value; "
           "the ibfragment-v2 and imageblock-v8..v10 bundles bit-exact on hardware)",
    12151: "imageblock load, 32-bit: as op13075; an undeclared image reads 0 (isa/g17-imageblock-receipt.json "
           "control; v10 ib_x_neighbour_undeclared_sized, whole program)",
    579: "exec while n, inverting (releasecheck._step_mask; whole programs, tier wp)",
})
EVIDENCE[1272] = ("hardware exp2, from the SPARSE MEASURED TABLE isa/g17-exp2-op1272-table.npz: the GPU's own op1272 "
                  "output for each input (tools/g17exp2oracle.py, g17decodeops probe, dispatched twice and identical); "
                  "deterministic (MM 25.144.2: 0 of 496,131 differ across runs)")
EVIDENCE[17642] = ("fp8 -> two bf16 (format 97 e4m3fn, 98 e5m2): recon section 137 part 5, Apple's own fp8 lowering "
                   "(seta-fp8-*, agxforge/g17/fp8enc.py); the fp8-v1 bundle bit-exact on hardware (receipt sha256)")
TENSOR_MODE_EVIDENCE = {17642, 12674, 17257, 5107, 5106, 5101, 12656, 10384, 10385, 5100, 5104, 5105, 5098, 5099}   # no scalar receipt encodes them; their mode words carry only the
#                                                     wait tokens (bits 24-28) and lifetime bits, masked below


def _frag_maps():
    """(A/D perm, B perm): perm[lane * 8 + slot] = row * 16 + col (tools/g17layerbench.py, recon section 144)."""
    def pos(r, c):
        return 16 * (r >> 3) + 8 * (c >> 3) + 2 * ((r >> 1) & 3) + ((c >> 2) & 1), 4 * (r & 1) + (c & 3)

    def pos_b(k, c):
        return 16 * ((k >> 2) & 1) + 8 * (c >> 3) + 2 * (k & 3) + ((c >> 2) & 1), 4 * (k >> 3) + (c & 3)
    out = []
    for fn in (pos, pos_b):
        p = np.empty(256, np.int64)
        for r in range(16):
            for c in range(16):
                ln, sl = fn(r, c)
                p[ln * 8 + sl] = r * 16 + c
        assert sorted(p) == list(range(256))
        out.append(p)
    return out


PERM_AD, PERM_B = _frag_maps()


def _perm_bt():
    """B under the transpose bit (type code + 32): B[k][c] sits where the A/D map puts (rotl1(c), k) - recon section
    132 part 3's mode Bt (tlower.py). The same law gives the measured pos_b: pos(rotl1(k), c) == pos_b(k, c)."""
    def rotl1(x):
        return ((x << 1) | (x >> 3)) & 15
    p = np.empty(256, np.int64)
    inv = np.empty(256, np.int64)
    inv[PERM_AD] = np.arange(256)                      # (row * 16 + col) -> lane * 8 + slot, the A/D map
    for k in range(16):
        for c in range(16):
            p[inv[rotl1(c) * 16 + k]] = k * 16 + c
    assert sorted(p) == list(range(256))
    return p


PERM_BT = _perm_bt()


def _perm_at():
    """A under the transpose bit (type code + 32): slot (lane l, slot j) holds A[row][k] with
    k = 4 (l >> 4) + ((l >> 1) & 3) + 8 (j >> 2) and row = 2 (j & 3) + 8 (l & 1) + ((l >> 3) & 1) - the closed form
    measured by one-hot probes (MM section 3, 6,651 dispatches), in the same convention that gives PERM_AD and
    PERM_B exactly. Checked here against the p13 trans_a and trans_ab_split bundles' GPU output, 0 words differing."""
    p = np.empty(256, np.int64)
    for ln in range(32):
        for j in range(8):
            k = 4 * (ln >> 4) + ((ln >> 1) & 3) + 8 * (j >> 2)
            row = 2 * (j & 3) + 8 * (ln & 1) + ((ln >> 3) & 1)
            p[ln * 8 + j] = row * 16 + k
    assert sorted(p) == list(range(256))
    return p


PERM_AT = _perm_at()
TENSOR_16BIT = {2: np.float16, 3: "bf16"}       # the A and B type codes of the 16-bit operands: fp16, bf16

# Instructions whose exact result is NOT a measured function, only a bound: each is run under every admissible
# choice, and the program is vouched for only if the outputs agree (run_bundle).
BRACKETED = {3658: ("down", "up")}

# special registers by Apple's register id (the read_sr source token), each tied to the builtin a compiled kernel
# read (isa/g17-special-registers.json, tools/g17sr.py): the value g17emu gives it is Metal's definition of that
# builtin over the dispatch's 3-D grid (Machine.__init__). threadgroups_per_grid .y/.z have no witnessed id.
SR_BUILTIN = {25: "local_linear", 26: "local_x", 27: "local_y", 28: "local_z", 45: "lane", 46: "simdgroup",
              51: "tgs_x", 54: "tg_x", 55: "tg_size_x", 56: "tg_y", 57: "tg_size_y", 58: "tg_z", 59: "tg_size_z",
              61: "grid_x", 62: "grid_y", 63: "grid_z"}
NFLAG = 16


# The destination's raw operand (operand 1 of an ALU form) mixes fields. Two kinds of bits cannot change the value
# written, and both are measured: bits 20-23 (load fill slot) and 24-31 (scoreboard wait mask) are SCHEDULING (MM
# table 0.8; "a wait names a slot"), and bits 4-5 are LIFETIME (release/keep; the destination keep flag is
# hardware-inert, MM sections 27, 94 and the destination-keep row of table 0.x). Every other bit must match a mode the
# hardware has EXECUTED for that opcode: MODES is built from the encoded instructions in isa/g17-execution-*-results.
INERT_MODE_BITS = 0xFFF00000 | 0x30
# THE LOW-BITS OPS: integer forms whose result's low 16 bits depend only on their sources' low 16 bits (add, subtract,
# multiply, multiply-add, the bitwise forms, move, shift LEFT - carries and products only move upward). An unknown
# source high half may flow through one of these into the result's high half (Machine.unk); every other op refuses it.
LOWBITS_OPS = {10282, 10279, 10283, 11667, 11666, 10825, 10822, 10826, 424, 423, 13575, 13574, 17771, 17770, 11190,
               586, 14391, 14392}
NO_MODE_CHECK = {14059, 14060, 684,       # read_sr (operand 1 is its fixed form word), end
                 458, 462, 577, 582, 447, 14156,
                 10090,   # control forms, and the atomic against its measured encodings: checked by the handler
                 554,     # movimm's operand 1 is its immediate, not a mode word: the handler admits only 0
                 13075, 12151, 579}   # imageblock store/load (operand 1: waits, fill slot, value release); exec


def receipt_modes(root=None):
    """{opcode: {masked mode}} over every instruction the execution receipts record as dispatched. Cached per content
    hash of the receipt files, so a new receipt changes the answer (MM 25.145)."""
    import glob, hashlib, json, re
    root = Path(root or HERE.parent)
    files = sorted(glob.glob(str(root / "isa" / "g17-execution-*results*.json")))
    h = hashlib.sha256()
    for f in files:
        h.update(Path(f).read_bytes())
    cache = Path.home() / ".cache" / "agxforge" / ("g17emu-modes-%s.json" % h.hexdigest()[:16])
    if cache.exists():
        return {int(k): set(v) for k, v in json.loads(cache.read_text()).items()}
    hexes = set()

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k in ("encoded", "bytes", "hex", "raw"):
                    for e in (v if isinstance(v, list) else [v]):
                        if isinstance(e, str) and re.fullmatch(r"[0-9a-f]{4,40}", e) and len(e) % 4 == 0:
                            hexes.add(e)
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    for f in files:
        try:
            walk(json.loads(Path(f).read_text()))
        except ValueError:
            continue
    modes = {}
    for e in sorted(hexes):
        try:
            rows = decode(bytes.fromhex(e) + END)
        except _undecodable():
            continue
        for _off, _size, op, toks in rows[:-1]:
            if len(toks) > 1 and toks[1].startswith("imm:"):
                modes.setdefault(op, set()).add(int(toks[1][4:]) & ~INERT_MODE_BITS)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({str(k): sorted(v) for k, v in modes.items()}))
    return modes


EVIDENCE[10090] = ("atomic RMW at a per-lane device address: memory takes the result and the lane gets the OLD value; "
                   "the operation is the one the measured encoding of the same length and operation operand "
                   "performed (isa/g17-atomic-semantics.json: vendor kernels executed, memory and return read back "
                   "separately, one thread per address; min and max unsigned; tools/g17atomicprobe.py)")
_ATOMIC = {}


def _atomic_table():
    """{(instruction length, operation operand token): (operation, {masked mode})} from the measured records: each
    record whose memory matched exactly one operation and whose return was the old value, its encoding decoded
    here. The 12-byte form also carries the 4-bit operation field (MM, the atomic facts); the 10-byte form has no
    byte 11, so the decoder's operation operand is the key both share."""
    if not _ATOMIC:
        import ast, json
        recs = json.loads((HERE.parent / "isa/g17-atomic-semantics.json").read_text())["records"]
        for r in recs.values():
            ops = r.get("operations_matching_the_memory")
            ops = ops if isinstance(ops, list) else ast.literal_eval(ops or "[]")
            forms = r.get("forms") or []
            forms = forms if isinstance(forms, list) else ast.literal_eval(forms)
            if len(ops) != 1 or str(r.get("return_is_the_old_value")) != "True" or r.get("status") != "ok":
                continue
            for f in forms:
                if not f["form"].startswith("10090/"):
                    continue
                raw = bytes.fromhex(f["bytes"])
                toks = decode(raw + END)[0][3]
                name, modes = _ATOMIC.setdefault((len(raw), toks[2]), (ops[0], set()))
                if name != ops[0]:
                    raise ValueError("two operations measured for one atomic encoding: %s, %s" % (name, ops[0]))
                modes.add(int(toks[1][4:]) & ~INERT_MODE_BITS)
    return _ATOMIC


ATOMIC_FN = {"add": lambda o, v: (o + v) & 0xFFFFFFFF, "sub": lambda o, v: (o - v) & 0xFFFFFFFF,
             "and": lambda o, v: o & v, "or": lambda o, v: o | v, "xor": lambda o, v: o ^ v,
             "max": max, "min": min, "exchange": lambda o, v: v}


def load_hazards(code):
    """[(index, offset, what)]: every read of a register a LOAD or ATOMIC filled before any instruction waited on the
    fill's scoreboard slot, across loop back edges (agxforge.g17.tensorview.hazards: the fill slot is operand 1 bits
    20-23 minus 1, a wait is a bit of operand 1 bits 24-31). g17emu completes a load at once, so without this a program
    missing a wait would run correctly here and not on the GPU: the indirect-load receipt's as-compiled arm got 1 of
    32 lanes right on the hardware and is refused here; its waited arm (32 of 32) runs.

    Special-register reads (op14059, op14060) also publish a slot, but are not producers here: all 636 delivered
    programs of the q4 and q8 cap-2048 roots read them unwaited (6,604 reads) and are bit-exact on the hardware,
    while with them counted 280 of those programs would be refused. With loads and atomics only, 0 of 636 are."""
    from agxforge.g17 import tensorview as TV
    v = TV.view(code)
    rows = None
    for i in v:
        if i.kind == "sr":
            i.fills = None
        elif i.opcode == 12151:
            # THE IMAGEBLOCK LOAD fills a slot as a device load does (operand 1 bits 20-23, MM 25.96: 0x01800000 fills
            # slot 7 and waits on slot 0), and cc's history has the unwaited consumer failing (cc.py, the wait fix);
            # tensorview.LOADS leaves it out, so it is added here
            rows = rows or decode(code)
            w = int(rows[i.index][3][1][4:])
            i.kind = "load"
            i.fills = ((w >> 20) & 0xF) - 1 if (w >> 20) & 0xF else None
            i.waits = (w >> 24) & 0xFF
    return [(idx, v[idx].offset, what) for idx, what in TV.hazards(v)]


END = bytes.fromhex("0e000000")        # op684 end, appended so one receipt instruction decodes as a program


@__import__("functools").lru_cache(maxsize=None)
def reg_name(tok):
    import agxforge.g17.model as MODEL
    if not tok.startswith("reg:") or not tok[4:].isdigit():
        return None
    return MODEL.registers().get(int(tok[4:]))


def reg_parts(tok):
    kind, regs = _reg_parts(tok)
    return kind, (list(regs) if isinstance(regs, list) else regs)       # a copy: the cached list is shared


@__import__("functools").lru_cache(maxsize=None)
def _reg_parts(tok):
    """A register token -> (kind, indices) from Apple's register names: ('r32', [n, n+1, ..]) for Rn and 32-bit
    tuples Rn_Rn+1.., ('lo'|'hi', [n]) for RnL / RnH, ('flag', [n]) for FLAGn, ('sr', name) for SR_*. Anything
    else (IR registers, mixed-half tuples) is refused."""
    from agxforge.g17 import registerdomain
    name = reg_name(tok)
    if name is None:
        raise Refused("operand %s where a register is expected" % tok)
    try:
        regs = registerdomain.registers_in_name_checked(name)
    except ValueError:
        raise Refused("register %s (%s): a register name registerdomain does not model" % (tok, name))
    if regs and name[-1] in "LH":
        if name != "R%d%s" % (regs[0], name[-1]):
            raise Refused("register %s (%s): mixed-half tuples have no measured model" % (tok, name))
        return ("lo" if name[-1] == "L" else "hi"), regs
    if regs:
        if (name != "_".join("R%d" % r for r in regs) or regs != list(range(regs[0], regs[0] + len(regs)))
                or regs[-1] >= NREG):
            raise Refused("register tuple %s is not a run of consecutive 32-bit registers" % name)
        return "r32", regs
    if name.startswith("FLAG") and name[4:].isdigit():
        return "flag", [int(name[4:])]
    if name.startswith("SR_"):
        return "sr", name
    raise Refused("register %s (%s): no measured model of that register class" % (tok, name))


HALVES = {None: (0, 1), "lo": (0,), "hi": (1,)}


def _half_bits(v, h):
    """The 16 bits of half h (0 low, 1 high) of u32 register values."""
    return (v & np.uint32(0xFFFF)) if h == 0 else (v >> np.uint32(16))


class Refused(Exception):
    """The program uses something g17emu has no measured semantics for. The message names it."""


def decode(code):
    import g17packedcheck as PC
    return PC.decode(code)


def _undecodable():
    import subprocess
    return (ValueError, RuntimeError, subprocess.CalledProcessError)


def fma16_exact(a, b, c):
    """binary16(a*b + c) with ONE rounding, the measured op798 (MM 25.196), from binary16-valued float64 arrays. The
    product is exact in binary64 (22 significant bits); the sum's error comes from TwoSum and is folded in by rounding
    to odd, which makes the binary64 -> binary16 rounding the correct one (53 >= 2 * 11 + 2)."""
    a, b, c = (np.asarray(v, np.float64) for v in (a, b, c))
    with np.errstate(invalid="ignore", over="ignore"):
        p = a * b
        s = p + c
        bb = s - p
        e = (p - (s - bb)) + (c - bb)
        fix = np.isfinite(s) & np.isfinite(e) & (e != 0) & ((s.view(np.uint64) & np.uint64(1)) == 0)
        s = np.where(fix, np.nextafter(s, np.where(e > 0, np.inf, -np.inf)), s)
        return s.astype(np.float16).view(np.uint16)


def _regs(toks):
    return [t for t in toks if t.startswith("reg:")]


_INV = {}


def _inv_perm(perm):
    """The inverse of a fragment permutation (cached by identity): v[:, inv] places what t[:, perm] = v places."""
    key = id(perm)
    if key not in _INV:
        inv = np.empty(len(perm), np.int64)
        inv[np.asarray(perm)] = np.arange(len(perm))
        _INV[key] = (perm, inv)
    return _INV[key][1]


class Machine:
    """One dispatch: registers per thread, the bound buffers, the grid."""

    def __init__(self, code, buffers, threads, group, choice=None, tier="strict", views=False, table=None):
        """buffers: {slot: bytes}, copied, each holding its data at byte PAD (g17bundlerun's placement); or, with
        views=True, {slot: numpy uint8 array} used IN PLACE with the binding at byte 0 (a graph's arena slices, so one
        dispatch's stores are the next one's inputs).

        threads and group are thread counts of a 1-D dispatch, or (x, y, z) extents: the grid in THREADS (a whole
        number of threadgroups on every axis) and the threadgroup. Threads are numbered threadgroup by threadgroup,
        and within one by thread_index_in_threadgroup = x + y X + z X Y, so each simdgroup is 32 consecutive threads.

        table: {K: slot}, the binding table - a load's bin(op0,const(K),8) names buffer table[K]. The compiler's ABI
        gives it (binding offset o is K = 2 o, run_program); without it entry K / 4 is the K/4-th slot in slot order,
        which is the bundles' dense table."""
        grid3 = tuple(threads) if isinstance(threads, (tuple, list)) else (int(threads), 1, 1)
        tg3 = tuple(group) if isinstance(group, (tuple, list)) else (int(group), 1, 1)
        if len(grid3) != 3 or len(tg3) != 3 or any(g % t for g, t in zip(grid3, tg3)) or min(tg3) < 1:
            raise Refused("grid %s is not a whole number of %s threadgroups" % (grid3, tg3))
        threads, group = int(np.prod(grid3)), int(np.prod(tg3))
        self.grid3, self.tg3 = grid3, tg3
        self.code, self.threads, self.group = bytes(code), threads, group
        self.table = dict(table) if table is not None else None
        self.tier = tier                                       # "strict": receipts only; "wp": also whole-program
        self.uniforms = {}                                     # halfword -> value the constant program published
        self.choice = choice or {}                             # BRACKETED opcode -> the admissible result used
        self.pad = 0 if views else PAD
        self.mem = (dict(buffers) if views else
                    {s: np.frombuffer(bytearray(b), np.uint8).copy() for s, b in buffers.items()})
        self.slots = sorted(self.mem)                          # the binding table, in slot order
        # column-major: each register's lanes are contiguous, so R[:, r] (every read and write) is not strided
        self.R = np.zeros((threads, NREG), np.uint32, order="F")
        # per lane and per 16-bit HALF (0 low, 1 high): a 16-bit write makes only its half written and live, so a
        # full read after it still sees the other half's history (tlower reads all of a register whose low half a
        # 16-bit system-register read just wrote, MM 25.141: the high half enters from outside)
        self.written = np.zeros((threads, NREG, 2), bool, order="F")      # written at least once by this program
        self.poison = np.zeros((threads, NREG, 2), bool, order="F")
        # UNKNOWN BITS: a half whose value the hardware leaves undetermined (released and nonzero, or never written),
        # carried forward instead of refused where only a LOW-BITS op read it (LOWBITS_OPS) - such an op's low result
        # bits depend only on its sources' low bits, so an unknown source high half makes only the result's high half
        # unknown. Any other reader of an unknown half is refused: the program is judged where the bits are OBSERVED.
        self.unk = np.zeros((threads, NREG, 2), bool, order="F")
        # SPEED (MM 25.197): clean[r, h] - known that EVERY lane's half h of r is written, live and fully known. Set
        # only by a full-lane write or a full check that found it so; cleared by any release, partial write or unknown
        # bit. It lets the common read skip three reductions over every thread; it never admits what the check refuses.
        self.clean = np.zeros((NREG, 2), bool)
        self._lowbits, self._taint = False, None      # released and not written since
        self.active = np.ones(threads, bool)                   # the exec mask: a masked-off lane writes nothing
        self.tg = None                                         # threadgroup memory, [threadgroups, bytes], on first use
        # THE RACE CHECK (MM 25.197): lockstep runs every thread instruction by instruction, one legal schedule; a
        # program whose result depends on the schedule is refused. Per written byte: the last writer (thread, epoch);
        # per read byte of a buffer the program also writes: the reader's threadgroup and simdgroup (or -2, several)
        self.race = {}                                         # space -> tracking arrays
        self.ep_tg = self.ep_dev = 0                           # barriers seen (lockstep: every thread's count)
        self.written_slots = None                              # the device slots some store names, on first use
        self.ib = None                                          # the imageblock tile, on first use (op13075 / op12151)
        tid = np.arange(threads, dtype=np.uint32)
        lin, tgl = tid % np.uint32(group), tid // np.uint32(group)     # thread_index_in_threadgroup, threadgroup
        ntg = [g // t for g, t in zip(grid3, tg3)]
        loc = (lin % tg3[0], lin // tg3[0] % tg3[1], lin // (tg3[0] * tg3[1]))
        tgp = (tgl % ntg[0], tgl // ntg[0] % ntg[1], tgl // (ntg[0] * ntg[1]))
        full = lambda v: np.full(threads, v, np.uint32)
        self.sr = {"lane": lin % 32, "simdgroup": lin // 32, "local_linear": lin, "tgs_x": full(ntg[0])}
        for a, ax in enumerate("xyz"):
            self.sr["local_" + ax] = loc[a].astype(np.uint32)
            self.sr["tg_" + ax] = tgp[a].astype(np.uint32)
            self.sr["tg_size_" + ax] = full(tg3[a])
            self.sr["grid_" + ax] = (tgp[a] * tg3[a] + loc[a]).astype(np.uint32)
        self.F = np.zeros((threads, NFLAG), bool)                # flag registers FLAG0.. (compare results)
        self.fwritten = np.zeros(NFLAG, bool)
        # THE EXEC STATE IS A PER-LANE NESTING COUNTER, active iff 0 (Asahi/Mesa's r0l; agxforge/g17/releasecheck.py's
        # _step_mask, which balances all 6,594 corpus programs, 338 of them a top-level `while 2 / back edge / pop 2`
        # that a push/pop stack cannot read). For the push-and-mask / one-level-pop pairs every receipt ran, it is the
        # stack exactly.
        self.ctr = np.zeros(threads, np.int64)
        self.pending = []                                        # releases of the current instruction
        self.used = {}                                          # opcode -> executed count
        self.admitted = {}                                       # whole-program-only semantics used (tier "wp")

    # ---- registers
    def _where(self, tok):
        """A scalar register operand -> (index, None | 'lo' | 'hi'), by Apple's register table."""
        kind, regs = reg_parts(tok)
        if kind == "r32" and len(regs) == 1:
            return regs[0], None
        if kind in ("lo", "hi"):
            return regs[0], kind
        raise Refused("operand %s (%s) where a scalar register is expected" % (tok, reg_name(tok)))

    def read_u(self, tok, mod, significant=None):
        """The raw bits of a source register (u32 for a 32-bit register, u16 for a half), after its modifier's
        lifetime bits. Negate/abs are applied by the float readers, never to integers. `significant` (a u32 mask,
        per lane or one for all) names the source bits that can reach the result; a released half is judged on those
        bits only (a shift left by 16 discards the high half, so its release cannot matter)."""
        if tok.startswith("expr:bin(op0,const("):
            return self._uniform(tok)
        r, half = self._where(tok)
        act = self.active
        hs = HALVES[half]
        v = self.R[:, r]
        sig = None if significant is None else np.broadcast_to(np.asarray(significant, np.uint64) & 0xFFFFFFFF,
                                                               (self.threads,)).astype(np.uint32)
        # Judged per half, and only on the bits that can reach the result: a half written since a release is live
        # again, and a half none of whose bits are significant (shifted out) can be unwritten or released freely.
        # A RELEASED half reads zero OR its old content, run to run (memory: a released register reads 0 only
        # sometimes), so a read of one is admitted only where its significant old bits ARE zero: both outcomes agree.
        full = act.all()
        for h in hs:
            # the common case, every lane active and every lane's half written and live: nothing to judge
            if full and (sig is None or half is not None):
                if self.clean[r, h]:
                    continue
                if self.written[:, r, h].all() and not self.poison[:, r, h].any() and not self.unk[:, r, h].any():
                    self.clean[r, h] = True
                    continue
            need = act if sig is None or half is not None else act & (_half_bits(sig, h) != 0)
            carry = self._lowbits and h == 1 and half is None     # the unknown high half can only reach a high half
            unk = need & self.unk[:, r, h]
            if np.any(unk):
                if not carry:
                    raise Refused("read of r%d%s whose bits are unknown (an earlier released or unwritten half "
                                  "carried into it), where they reach the result" % (r, "LH"[h] if half is None else ""))
                self._taint = unk if self._taint is None else self._taint | unk
            if not np.all(self.written[need, r, h]):
                # Untouched registers read zero on the hardware (R2, R20, R60), R0 does not: a whole-program
                # measurement (ledger/g17-the-threadgroup-base-was-an-unassigned-register), so tier wp only. tlower
                # relies on it: it writes R27L and reads all of R27 (MM 25.141), and nothing writes R27H.
                blank = need & ~self.written[:, r, h]
                if carry:
                    self._taint = blank if self._taint is None else self._taint | blank
                elif self.tier != "wp" or r == 0 or np.any(self.R[blank, r] >> (16 * h) & 0xFFFF):
                    raise Refused("read of r%d%s before any write" % (r, "LH"[h] if half is None else ""))
                else:
                    self.wp("a never-written register reads zero (untouched R2, R20, R60 read zero on hardware; "
                            "R0 does not)")
            rel = need & self.poison[:, r, h]
            if np.any(rel):
                bad = rel & (_half_bits(v if sig is None else v & sig, h) != 0)
                if np.any(bad):
                    if not carry:
                        raise Refused("read of r%d%s after its release, where its old value is not zero: the "
                                      "hardware returns zero or the old value" % (r, "LH"[h] if half is None else ""))
                    self._taint = bad if self._taint is None else self._taint | bad
        if half == "lo":
            v = (v & 0xFFFF).astype(np.uint16)
        elif half == "hi":
            v = (v >> 16).astype(np.uint16)
        if mod & MOD_RELEASE:                     # takes effect when the instruction has read all its operands
            self.pending.append((act.copy(), r, hs))
        return v

    def _retire(self):
        """The end of an instruction: every operand has been read, so its releases take effect now."""
        self._taint = None
        for lanes, reg, hs in self.pending:
            for h in hs:
                self.clean[reg, h] = False
                if lanes.all():                                    # a slice: the same lanes, without the mask
                    self.poison[:, reg, h] = True
                else:
                    self.poison[lanes, reg, h] = True
        self.pending = []

    def write_u(self, tok, v):
        """Write the active lanes only (a masked-off lane keeps its register)."""
        if tok.startswith("expr:bin(op0,const("):
            return self._write_uniform(tok, v)
        r, half = self._where(tok)
        act = self.active
        v = np.broadcast_to(np.asarray(v, np.uint32), (self.threads,))
        old = self.R[:, r]
        if half is None:
            new = v
        elif half == "lo":
            new = (old & np.uint32(0xFFFF0000)) | (v & np.uint32(0xFFFF))
        else:
            new = (old & np.uint32(0x0000FFFF)) | ((v & np.uint32(0xFFFF)) << np.uint32(16))
        hs = HALVES[half]
        if act.all():                                               # every lane active: slices, not masks
            self.R[:, r] = new
            for h in hs:
                self.written[:, r, h], self.poison[:, r, h], self.unk[:, r, h] = True, False, False
                self.clean[r, h] = True
        else:
            self.R[:, r] = np.where(act, new, old)
            for h in hs:
                self.written[act, r, h], self.poison[act, r, h], self.unk[act, r, h] = True, False, False
                self.clean[r, h] = False
        if self._taint is not None and half is None:                # a low-bits op's result: its high half unknown
            self.unk[act & self._taint, r, 1] = True
            self.clean[r, 1] = False
        # a register this instruction both reads-with-release and writes holds the NEW value: the release frees
        # the old value before the write lands (an in-place accumulate, D = A B + C with C released, is the case).
        # Only the halves written are spared; a released half the write does not touch stays released.
        if not self.pending:
            return
        pending = []
        for lanes, reg, phs in self.pending:
            if reg != r:
                pending.append((lanes, reg, phs))
                continue
            pending.append((lanes & ~act, reg, phs))
            rest = tuple(h for h in phs if h not in hs)
            if rest:
                pending.append((lanes & act, reg, rest))
        self.pending = pending

    UNIFORM = __import__("re").compile(r"expr:bin\(op0,const\((\d+)\),(2|4)\)")

    def _uniform(self, tok):
        """A SOURCE operand in the uniform file, bin(op0,const(K),W): 16 bits at halfword K (W 2) or 32 at halfwords
        K, K + 1 (W 4). Past the binding pointers (4 halfwords a binding) it is the container's slot-13 constant
        pool, preloaded before the program runs, at pool halfword K - 4 x bindings (the and16 pool form, MM
        25.141.16). The layout was read with Apple's compiler as the oracle; the preload is verified only by whole
        programs (the q4 qmv chain), so tier wp. A binding pointer read as a value is refused."""
        mt = self.UNIFORM.fullmatch(tok)
        pool = getattr(self, "pool", None)
        if not mt or pool is None:
            raise Refused("uniform operand %s with no container pool read" % tok)
        nb = len(self.table) if self.table is not None else len(self.slots)
        k, w = int(mt.group(1)), int(mt.group(2))
        h = k - 4 * nb
        if h < 0:
            raise Refused("uniform %s is a binding pointer, read as a value" % tok)
        hw = set(range(k, k + w // 2))
        if hw & set(getattr(self, "published", ())) and hw <= set(self.uniforms):
            # the constant program ran first (run_program(constant=...)) and published these halfwords
            if any(self.uniforms[k + i] is None for i in range(w // 2)):
                raise Refused("uniform %s: the constant program published unknown bits there" % tok)
            v = sum(self.uniforms[k + i] << (16 * i) for i in range(w // 2))
            return np.full(self.threads, v, np.uint16 if w == 2 else np.uint32)
        if hw & set(getattr(self, "published", ())):
            # a value the CONSTANT PROGRAM computed and published before main (Apple folds uniform work there, e.g.
            # u[0] * u[1]); it is not the pool, and g17emu does not run the constant program
            raise Refused("uniform %s is published by the constant program, which g17emu does not run" % tok)
        if 2 * h + w > len(pool):
            raise Refused("uniform %s is past the %d-byte constant pool" % (tok, len(pool)))
        self.wp("a source operand from the container's slot-13 constant pool (uniform 4 x bindings + h)")
        v = int.from_bytes(pool[2 * h:2 * h + w], "little")
        return np.full(self.threads, v, np.uint16 if w == 2 else np.uint32)

    def read_f(self, tok, mod):
        """A float source: f32 from a 32-bit register, f16 widened exactly from a half; then abs, then negate."""
        if tok.startswith("expr:bin(op0,const("):
            u = self._uniform(tok)
            f = u.view(np.float16).astype(np.float32) if u.dtype == np.uint16 else u.view(np.float32).copy()
            f = np.abs(f) if mod & MOD_ABS else f
            return -f if mod & MOD_NEG else f
        r, half = self._where(tok)
        u = self.read_u(tok, mod)
        f = u.view(np.float32).astype(np.float32) if half is None else u.view(np.float16).astype(np.float32)
        if mod & MOD_ABS:
            f = np.abs(f)
        if mod & MOD_NEG:
            f = -f
        return f

    def _write_uniform(self, tok, v, unknown=()):
        """A DESTINATION in the uniform file (the constant program's publish, MM 25.179): bin(op0,const(K),W) takes the
        low W bytes of the value, at halfwords K.. . The constant program runs one thread, so the value is one."""
        mt = self.UNIFORM.fullmatch(tok)
        if not mt:
            raise Refused("uniform destination %s of an unmodelled form" % tok)
        k, w = int(mt.group(1)), int(mt.group(2))
        vals = np.asarray(v, np.uint32)[self.active] if np.ndim(v) else np.asarray([v], np.uint32)
        if vals.size == 0 or np.any(vals != vals[0]):
            raise Refused("uniform destination %s written with differing lane values" % tok)
        x = int(vals[0]) & ((1 << (8 * w)) - 1)
        for i in range(w // 2):
            self.uniforms[k + i] = None if i in unknown else (x >> (16 * i)) & 0xFFFF

    def write_f32(self, tok, f):
        if tok.startswith("expr:bin(op0,const("):
            return self._write_uniform(tok, np.asarray(f, np.float32).view(np.uint32))
        r, half = self._where(tok)
        if half is not None:
            raise Refused("f32 result into a 16-bit register %s" % tok)
        self.write_u(tok, np.asarray(f, np.float32).view(np.uint32))

    # ---- memory
    def _binding(self, expr):
        """expr:bin(op0,const(K),8): the binding table entry K / 4, in slot order (memforms)."""
        if not (expr.startswith("expr:bin(op0,const(") and expr.endswith("),8)")):
            raise Refused("binding expression %s is not the measured bin(op0,const(K),8) form" % expr)
        k = int(expr[len("expr:bin(op0,const("):-len("),8)")])
        if self.table is not None:
            if k not in self.table or self.table[k] not in self.mem:
                raise Refused("binding const(%d) names no bound buffer (table: %s)" % (k, self.table))
            return self.table[k]
        if k % 4 or not 0 <= k // 4 < len(self.slots):
            raise Refused("binding const(%d) names no bound buffer (bound: %s)" % (k, self.slots))
        return self.slots[k // 4]

    def load(self, slot, index, elem, disp):
        m = self.mem[slot]
        addr = self.pad + index.astype(np.int64) * elem + disp
        if self.RACE_CHECK:
            act = self.active
            self._track_read(("dev", slot), len(m), addr[act], np.nonzero(act)[0], elem)
        out = np.zeros(len(index), np.uint32)
        ok = (addr >= 0) & (addr + elem <= len(m))              # an out-of-range device read returns zero (measured)
        a = addr[ok]
        v = np.zeros(len(a), np.uint32)
        for b in range(elem):
            v |= m[a + b].astype(np.uint32) << np.uint32(8 * b)
        out[ok] = v
        return out

    RACE_CHECK = True
    STORE_OPS = (17193, 17199, 17202, 17229, 17235, 17244, 17253, 17262, 17257)

    def _units(self, lanes):
        """(threadgroup, global simdgroup) of each thread: a simdgroup is 32 consecutive threads of one threadgroup."""
        lanes = np.asarray(lanes, np.int64)
        if self.group % 32 == 0:                                  # simdgroups never straddle threadgroups
            return lanes // self.group, lanes >> 5
        tg = lanes // self.group
        per = -(-self.group // 32)
        return tg, tg * per + (lanes % self.group) // 32

    def _writes_slot(self, slot):
        if self.written_slots is None:
            ws, anyreg = set(), False
            for _o, _s, op, t in decode(self.code):
                if op in self.STORE_OPS and len(t) > 3:
                    if t[3].startswith("expr:bin(op0,const("):
                        try:
                            ws.add(self._binding(t[3]))
                        except Refused:
                            anyreg = True
                    else:
                        anyreg = True
            self.written_slots = None if anyreg else ws
            if anyreg:
                return True
        return slot in self.written_slots

    def _space(self, key, n):
        sp = self.race.get(key)
        if sp is None:
            sp = self.race[key] = dict(w_thr=np.full(n, -1, np.int32), w_ep=np.zeros(n, np.int32),
                                       r_tgmin=np.full(n, np.iinfo(np.int32).max, np.int32),
                                       r_tgmax=np.full(n, -1, np.int32),
                                       r_sgmin=np.full(n, np.iinfo(np.int32).max, np.int32),
                                       r_sgmax=np.full(n, -1, np.int32),
                                       r_ep=np.full(n, -1, np.int32), gen=np.zeros(n, np.int32),
                                       mw_b=np.zeros(0, np.int64), mw_t=np.zeros(0, np.int64),
                                       mw_e=np.zeros(0, np.int64), mw_g=np.zeros(0, np.int64), mw_sorted=None,
                                       mw_new=[], wr_tgmin=np.full(n, np.iinfo(np.int32).max, np.int32),
                                       wr_tgmax=np.full(n, -1, np.int32),
                                       wr_sgmin=np.full(n, np.iinfo(np.int32).max, np.int32),
                                       wr_sgmax=np.full(n, -1, np.int32), wr_ep=np.full(n, -1, np.int32))
        return sp

    def _race_on(self):
        return self.RACE_CHECK and self.threads > 32 or (self.RACE_CHECK and self.threads // self.group > 1)

    def _epoch(self, key):
        return self.ep_tg if key[0] == "tg" else self.ep_dev

    def _ordered(self, w_thr, w_ep, tg, sg, ep):
        """Is the write (thread w_thr, epoch w_ep) ordered before an access by (tg, sg) at epoch ep? The same simdgroup is
        program order; another simdgroup of the threadgroup needs a barrier of the space's scope between; another
        threadgroup never is."""
        wtg, wsg = self._units(w_thr)
        return (wtg == tg) & ((wsg == sg) | (w_ep < ep))

    def _mw_compact(self, sp):
        """Fold the buffered new writer records into the arrays, drop records whose byte has changed since (a stale
        generation) and duplicate (byte, thread) pairs (keeping the earliest epoch): the record set stays the size of
        the CURRENT writer sets, so a loop of rewrites is not quadratic."""
        if sp["mw_new"]:
            parts = [np.concatenate([p[i] for p in sp["mw_new"]]) for i in range(4)]
            for k, v in zip(("mw_b", "mw_t", "mw_e", "mw_g"), parts):
                sp[k] = np.concatenate([sp[k], v.astype(np.int64)])
            sp["mw_new"] = []
        rb, rt, re_, rg = (sp[k] for k in ("mw_b", "mw_t", "mw_e", "mw_g"))
        cur = rg == sp["gen"][rb]
        rb, rt, re_, rg = rb[cur], rt[cur], re_[cur], rg[cur]
        if rb.size:
            o = np.lexsort((re_, rt, rb))
            rb, rt, re_, rg = rb[o], rt[o], re_[o], rg[o]
            first = np.r_[True, (rb[1:] != rb[:-1]) | (rt[1:] != rt[:-1])]
            rb, rt, re_, rg = rb[first], rt[first], re_[first], rg[first]
        sp["mw_b"], sp["mw_t"], sp["mw_e"], sp["mw_g"] = rb, rt, re_, rg

    def _multi_expand(self, sp, a):
        """The writer-set records of bytes a (w_thr == -2): (query index, thread, epoch) of every current record."""
        if sp["mw_sorted"] is None:
            self._mw_compact(sp)
            sp["mw_index"] = None                                 # the summary must see the folded tail too
            order = np.argsort(sp["mw_b"], kind="stable")
            sp["mw_sorted"] = tuple(np.asarray(sp[k], np.int64)[order] for k in ("mw_b", "mw_t", "mw_e", "mw_g"))
        rb, rt, re_, rg = sp["mw_sorted"]
        lo = np.searchsorted(rb, a, "left"); hi = np.searchsorted(rb, a, "right")
        cnt = hi - lo
        q = np.repeat(np.arange(a.size), cnt)
        pos = lo[q] + (np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt))
        cur = rg[pos] == sp["gen"][a[q]]
        return q[cur], rt[pos[cur]], re_[pos[cur]]

    def _unit_counts(self):
        per = -(-self.group // 32)
        return (self.threads // self.group) * per, self.threads // self.group

    def _mw_merge(self, sp):
        """Fold the tail into the main records (compacting them) and rebuild the main summary: (byte, simdgroup) keys
        and (byte, threadgroup) keys with the earliest epoch, each carrying the byte's generation at the merge."""
        self._mw_compact(sp)
        rb, rt, re_, rg = sp["mw_b"], sp["mw_t"], sp["mw_e"], sp["mw_g"]
        nsg, ntg = self._unit_counts()
        tg, sgg = self._units(rt)
        ksg = rb * nsg + sgg
        o = np.argsort(ksg, kind="stable")
        ksg, gsg = ksg[o], rg[o]
        first = np.r_[True, ksg[1:] != ksg[:-1]] if ksg.size else np.zeros(0, bool)
        ktg = rb * ntg + tg
        o = np.lexsort((re_, ktg))
        ktg, etg, gtg = ktg[o], re_[o], rg[o]
        f2 = np.r_[True, ktg[1:] != ktg[:-1]] if ktg.size else np.zeros(0, bool)
        sp["mw_index"] = (nsg, ntg, ksg[first], gsg[first], ktg[f2], etg[f2], gtg[f2])
        sp["mw_tail_n"] = 0

    def _multi_index(self, sp):
        """The main summary, merged when the tail has grown past a quarter of the main records (amortized)."""
        tail = sum(p[0].size for p in sp["mw_new"])
        if sp.get("mw_index") is None or tail > max(4096, sp["mw_b"].size // 4):
            self._mw_merge(sp)
        return sp["mw_index"]

    def _tail_ordered(self, sp, b, t_, s_, ep):
        """Per query (byte b, threadgroup t_, simdgroup s_): some ordered writer among the tail records?"""
        if not sp["mw_new"]:
            return np.zeros(b.size, bool)
        tb, tt, te, tgn = (np.concatenate([p[i] for p in sp["mw_new"]]) for i in range(4))
        cur = tgn == sp["gen"][tb]
        tb, tt, te = tb[cur], tt[cur], te[cur]
        o = np.argsort(tb, kind="stable")
        tb, tt, te = tb[o], tt[o], te[o]
        lo = np.searchsorted(tb, b, "left"); hi = np.searchsorted(tb, b, "right")
        cnt = hi - lo
        if not cnt.any():
            return np.zeros(b.size, bool)
        q = np.repeat(np.arange(b.size), cnt)
        pos = lo[q] + (np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt))
        wtg, wsg = self._units(tt[pos])
        o_ = (wtg == t_[q]) & ((wsg == s_[q]) | (te[pos] < ep))
        return np.bincount(q[o_], minlength=b.size) > 0

    @staticmethod
    def _has(keys, q):
        i = np.searchsorted(keys, q)
        i = np.minimum(i, max(keys.size - 1, 0))
        return (keys.size > 0) & (keys[i] == q) if keys.size else np.zeros(q.size, bool)

    def _writers_ordered(self, sp, a, tg, sg, ep, need_all=False):
        """Per accessed byte: is SOME write of its current value ordered before this access (need_all: EVERY write)?
        Several units storing the same value leave a writer set, the mw_* records."""
        wt = sp["w_thr"][a]
        ok = np.zeros(a.size, bool)
        one = wt >= 0
        if one.any():
            ok[one] = self._ordered(wt[one], sp["w_ep"][a[one]], tg[one], sg[one], ep)
        mult = np.nonzero(wt == -2)[0]
        if mult.size and not need_all:
            nsg, ntg, ksg, gsg, ktg, kep, gtg = self._multi_index(sp)
            b, t_, s_ = a[mult].astype(np.int64), tg[mult].astype(np.int64), sg[mult].astype(np.int64)
            gb = sp["gen"][b]
            q1 = b * nsg + s_
            j1 = np.minimum(np.searchsorted(ksg, q1), max(ksg.size - 1, 0))
            same_sg = (ksg.size > 0) & (ksg[j1] == q1) & (gsg[j1] == gb) if ksg.size else np.zeros(b.size, bool)
            q2 = b * ntg + t_
            j2 = np.minimum(np.searchsorted(ktg, q2), max(ktg.size - 1, 0))
            earlier = ((ktg.size > 0) & (ktg[j2] == q2) & (gtg[j2] == gb) & (kep[j2] < ep)) if ktg.size \
                else np.zeros(b.size, bool)
            ok[mult] = same_sg | earlier | self._tail_ordered(sp, b, t_, s_, ep)
        elif mult.size:
            q, t, e = self._multi_expand(sp, a[mult])
            o = self._ordered(t, e, tg[mult][q], sg[mult][q], ep)
            if need_all:
                bad = np.bincount(q[~o], minlength=mult.size) > 0
                ok[mult] = ~bad
            else:
                ok[mult] = np.bincount(q[o], minlength=mult.size) > 0
        return ok, wt != -1

    def _track_read(self, key, n, addr, lanes, width):
        """addr: the first byte of each access (int64), lanes: its thread; width bytes each. A read races when the byte
        has been written in this dispatch and no write of its current value is ordered before the read."""
        if not self._race_on() or addr.size == 0:
            return
        if key[0] == "dev" and not self._writes_slot(key[1]):
            return
        sp = self._space(key, n)
        a = (addr[:, None] + np.arange(width)).ravel()
        tg, sg = self._units(np.repeat(lanes, width))
        ok = (a >= 0) & (a < n)
        a, tg, sg = a[ok], tg[ok], sg[ok]
        ep = self._epoch(key)
        ordered, written = self._writers_ordered(sp, a, tg, sg, ep)
        bad = written & ~ordered
        if np.any(bad):
            j = np.nonzero(bad)[0][0]
            raise Refused("a race: byte %d of %s is read by thread-group %d / simdgroup %d, and no write of its value "
                          "is ordered before the read (another threadgroup, or no barrier of that scope between; MM "
                          "25.197)" % (a[j], key, tg[j], sg[j]))
        # record the readers as per-byte ranges (no sort): every threadgroup that read it in this dispatch, and the
        # simdgroups that read it in the CURRENT epoch (a read before a barrier is ordered before a write after it)
        stale = sp["r_ep"][a] < ep
        if stale.any():
            sp["r_sgmin"][a[stale]] = np.iinfo(np.int32).max
            sp["r_sgmax"][a[stale]] = -1
        np.minimum.at(sp["r_tgmin"], a, tg.astype(np.int32))
        np.maximum.at(sp["r_tgmax"], a, tg.astype(np.int32))
        np.minimum.at(sp["r_sgmin"], a, sg.astype(np.int32))
        np.maximum.at(sp["r_sgmax"], a, sg.astype(np.int32))
        sp["r_ep"][a] = ep

    def _track_write(self, key, n, addr, lanes, old, new):
        """addr: every written BYTE (int64), lanes: its thread; old/new: the byte before and after this instruction.
        A write that CHANGES the byte must be ordered after every other write and read of it; a write of the value
        already there changes nothing any schedule can see, and joins the byte's writer set."""
        if not self._race_on() or addr.size == 0:
            return
        sp = self._space(key, n)
        ep = self._epoch(key)
        eff = old != new
        if eff.any():
            a = addr[eff]
            tg, sg = self._units(lanes[eff])
            wt = sp["w_thr"][a]
            has = wt != -1
            if has.any():
                # a value change needs EVERY earlier write of the byte ordered before it: all its writers in this
                # threadgroup, and those of the latest epoch in this simdgroup (the per-byte ranges, no sort)
                ah, th, shh = a[has], tg[has], sg[has]
                ordered = ((sp["wr_tgmin"][ah] == th) & (sp["wr_tgmax"][ah] == th) &
                           ((sp["wr_ep"][ah] < ep) | ((sp["wr_sgmin"][ah] == shh) & (sp["wr_sgmax"][ah] == shh))))
                if not np.all(ordered):
                    j = np.nonzero(~ordered)[0][0]
                    raise Refused("a race: byte %d of %s, written earlier, is changed again by thread-group %d / "
                                  "simdgroup %d with no order between the two (MM 25.197)"
                                  % (a[has][j], key, tg[has][j], sg[has][j]))
            tmin, tmax = sp["r_tgmin"][a], sp["r_tgmax"][a]
            smin, smax, rep = sp["r_sgmin"][a], sp["r_sgmax"][a], sp["r_ep"][a]
            read = tmax >= 0
            # a read by another threadgroup (any epoch), or by another simdgroup in this epoch, is unordered with
            # this change
            bad = read & ((tmin != tg) | (tmax != tg) | ((rep == ep) & (smax >= 0) & ((smin != sg) | (smax != sg))))
            if np.any(bad):
                j = np.nonzero(bad)[0][0]
                raise Refused("a race: byte %d of %s, read earlier by another thread-group or unbarriered simdgroup, is "
                              "changed by thread-group %d / simdgroup %d (MM 25.197)" % (a[j], key, tg[j], sg[j]))
        # the writer ranges (for the change check): reset where the byte changes, extended where a lane rewrites a
        # value that already has writers
        la = lanes.astype(np.int64)
        prior_any = sp["w_thr"][addr] != -1
        lt, ls = self._units(la)
        upd = eff | prior_any
        if upd.any():
            IMAX = np.iinfo(np.int32).max
            ch = addr[eff]
            sp["wr_tgmin"][ch] = IMAX; sp["wr_tgmax"][ch] = -1
            sp["wr_sgmin"][ch] = IMAX; sp["wr_sgmax"][ch] = -1
            sp["wr_ep"][ch] = ep
            ua, ut, us = addr[upd], lt[upd].astype(np.int32), ls[upd].astype(np.int32)
            newer = sp["wr_ep"][ua] < ep                        # a later epoch: its simdgroup range starts over
            if newer.any():
                sp["wr_sgmin"][ua[newer]] = IMAX; sp["wr_sgmax"][ua[newer]] = -1
                sp["wr_ep"][ua[newer]] = ep
            np.minimum.at(sp["wr_tgmin"], ua, ut); np.maximum.at(sp["wr_tgmax"], ua, ut)
            np.minimum.at(sp["wr_sgmin"], ua, us); np.maximum.at(sp["wr_sgmax"], ua, us)
        # the writer sets: every lane writing a byte's (new) value in this instruction, plus same-value rewrites
        order = np.argsort(addr, kind="stable")
        aa, ll, ee = addr[order], lanes[order].astype(np.int64), eff[order]
        first = np.r_[True, aa[1:] != aa[:-1]]
        starts = np.nonzero(first)[0]
        counts = np.diff(np.r_[starts, aa.size])
        byte = aa[starts]
        changed = np.maximum.reduceat(ee.astype(np.int8), starts).astype(bool)
        prior = sp["w_thr"][byte].astype(np.int64)
        prior_ep = sp["w_ep"][byte].astype(np.int64)
        # an unchanged byte with no writer yet: no schedule can see this write, so it is not recorded
        keep = changed | (prior != -1)
        if not keep.any():
            return
        sp["gen"][byte[changed]] += 1                             # a changed byte's old writer set is void
        grp = np.repeat(np.arange(byte.size), counts)             # each lane's byte group
        single = keep & (counts == 1) & changed
        sp["w_thr"][byte[single]] = ll[starts[single]].astype(np.int32)
        sp["w_ep"][byte[single]] = ep
        multi = keep & ~single                                    # the byte ends with a set of writers
        if multi.any():
            recs_b, recs_t, recs_e = [], [], []
            # a single earlier writer of an unchanged byte joins the set
            conv = multi & ~changed & (prior >= 0)
            recs_b.append(byte[conv]); recs_t.append(prior[conv]); recs_e.append(prior_ep[conv])
            lane_in = multi[grp]
            recs_b.append(aa[lane_in]); recs_t.append(ll[lane_in]); recs_e.append(np.full(int(lane_in.sum()), ep))
            nb = np.concatenate(recs_b).astype(np.int64)
            sp["mw_new"].append((nb, np.concatenate(recs_t).astype(np.int64), np.concatenate(recs_e).astype(np.int64),
                                 sp["gen"][nb].astype(np.int64)))
            sp["mw_sorted"] = None
            sp["w_thr"][byte[multi]] = -2

    @staticmethod
    def _same_value_or_refuse(addr, value, elem, where):
        """Several lanes storing to one address: the order is unmeasured, so it is admitted only when they all
        store the same bytes (then no order can change the result)."""
        u, first, inv = np.unique(addr, return_index=True, return_inverse=True)
        if len(u) == len(addr):
            return
        mask = np.uint32((1 << (8 * elem)) - 1) if elem < 4 else np.uint32(0xFFFFFFFF)
        v = np.asarray(value, np.uint32) & mask
        if np.any(v != v[first][inv]):
            raise Refused("two lanes store different values to one address in %s: the winner is unmeasured" % where)

    def store(self, slot, index, elem, disp, value, width=None, lowest_lane_wins=False):
        """Active lanes only. Two active lanes storing DIFFERENT values to one address is refused - except, with
        lowest_lane_wins, when they are in ONE simdgroup: then the lowest active lane's value lands (measured for the
        word store op17229, isa/g17-execution-storewinner-results.json). Across simdgroups the order is unmeasured.
        width: bytes written when it differs from the addressing element (op17235 addresses in bytes, stores words)."""
        m = self.mem[slot]
        act = self.active
        lanes = np.nonzero(act)[0]
        addr = (self.pad + index.astype(np.int64) * elem + disp)[act]
        value = value[act]
        width = elem if width is None else width
        if np.any(addr < 0) or np.any(addr + width > len(m)):
            raise Refused("store outside buffer %d (%d bytes)" % (slot, len(m)))
        u, first, inv = np.unique(addr, return_index=True, return_inverse=True)
        if lowest_lane_wins and len(u) != len(addr):
            sg = lanes // 32                                        # lanes are in thread order: first = lowest lane
            if np.any(sg != sg[first][inv]):
                # across simdgroups the order is unmeasured: admitted only when every lane at such an address stores
                # the same bytes, so no order changes the result
                cross = np.zeros(len(u), bool)
                np.logical_or.at(cross, inv, sg != sg[first][inv])
                mask = np.uint32((1 << (8 * width)) - 1) if width < 4 else np.uint32(0xFFFFFFFF)
                v = np.asarray(value, np.uint32) & mask
                if np.any(cross[inv] & (v != v[first][inv])):
                    raise Refused("lanes of different simdgroups store different values to one address in buffer %d: "
                                  "their order is unmeasured" % slot)
            addr, value, lanes = addr[first], value[first], lanes[first]
        self._same_value_or_refuse(addr, value, width, "buffer %d" % slot)
        if self.RACE_CHECK:
            nb = np.asarray(value, np.uint32)
            ab = (addr[:, None] + np.arange(width)).ravel()
            newb = ((nb[:, None] >> (8 * np.arange(width, dtype=np.uint32))) & 0xFF).astype(np.uint8).ravel()
            self._track_write(("dev", slot), len(m), ab, np.repeat(lanes, width), m[ab], newb)
        for b in range(width):
            m[addr + b] = ((value >> np.uint32(8 * b)) & 0xFF).astype(np.uint8)

    # ---- execution
    def run(self, modes=None, max_steps=50_000_000):
        for idx, off, what in load_hazards(self.code):
            raise Refused("instruction %d at +0x%x %s: a load's destination read before a wait on its scoreboard "
                          "slot reads zeros or stale data on the hardware, and g17emu completes loads at once"
                          % (idx, off, what))
        insts = decode(self.code)
        at = {off: i for i, (off, _s, _o, _t) in enumerate(insts)}
        modes = receipt_modes() if modes is None else modes
        i, steps = 0, 0
        while i < len(insts):
            off, size, op, toks = insts[i]
            steps += 1
            if steps > max_steps:
                raise Refused("more than %d instructions executed: a loop that does not end" % max_steps)
            if op not in EVIDENCE:
                raise Refused("op%d at +0x%x has no measured semantics in g17emu" % (op, off))
            if op not in NO_MODE_CHECK and op not in TENSOR_MODE_EVIDENCE and len(toks) > 1 and toks[1].startswith("imm:"):
                m = int(toks[1][4:]) & ~INERT_MODE_BITS
                if m not in modes.get(op, set()):
                    if self.tier != "wp":
                        raise Refused("op%d at +0x%x: mode 0x%x (inert bits masked) was never executed in a receipt"
                                      % (op, off, m))
                    self.wp("op%d mode 0x%x (unexecuted; the executed mode's semantics assumed)" % (op, m))
            self.used[op] = self.used.get(op, 0) + 1
            self._lowbits = op in LOWBITS_OPS
            try:
                r = getattr(self, "op%d" % op)(off, size, toks)
            except Refused as e:
                if str(e).startswith("op"):
                    raise
                raise Refused("op%d at +0x%x: %s" % (op, off, e)) from None
            self._retire()
            if r == "end":
                break
            if isinstance(r, tuple) and r[0] == "jump":
                if r[1] not in at:
                    raise Refused("branch at +0x%x to +0x%x, not an instruction boundary" % (off, r[1]))
                i = at[r[1]]
                continue
            i += 1
        return self

    def _sr(self, tok, off):
        rid = int(tok[4:]) if tok.startswith("reg:") and tok[4:].isdigit() else None
        if reg_name(tok) in ("SR_PVSIMD", "SR_TVSIMD"):
            # the active lanes before this one / in all (agxforge/g17/ir.py:725-731): lane.vote_pair and the endtoend
            # votepair kernel read (32 << 16) | k, Apple's simd_prefix_exclusive_sum(1) / simd_sum(1). Every lane was
            # active in both, so only a fully active simdgroup is admitted: there the count IS the lane index (and 32)
            if self.group % 32:
                raise Refused("%s in a threadgroup that is not whole simdgroups" % reg_name(tok))
            sg = self.active.reshape(-1, 32)
            if np.any(sg.any(1) & ~sg.all(1)):
                raise Refused("%s (+0x%x) in a partially active simdgroup: unmeasured" % (reg_name(tok), off))
            return self.sr["lane"] if reg_name(tok) == "SR_PVSIMD" else np.full(self.threads, 32, np.uint32)
        name = SR_BUILTIN.get(rid)
        if name is None:
            raise Refused("read_sr of %s (%s, +0x%x): no witnessed builtin" % (tok, reg_name(tok), off))
        # Only 1-D dispatches ran on hardware: a y or z builtin's value (and the linear index's, which equals x only
        # when y and z are 1) is Metal's definition, witnessed at compile time only (isa/g17-special-registers.json).
        # On an axis of extent 1 every definition agrees (position 0, size 1); beyond that it is admitted in tier wp.
        ntg = [g // t for g, t in zip(self.grid3, self.tg3)]
        extent = {"local_linear": self.tg3[1] * self.tg3[2]}                  # the axes beyond x that name reads
        for a in (1, 2):
            ax = "xyz"[a]
            extent.update({"local_" + ax: self.tg3[a], "tg_size_" + ax: self.tg3[a], "tg_" + ax: ntg[a],
                           "grid_" + ax: self.grid3[a]})
        if extent.get(name, 1) > 1:
            self.wp("%s on a multi-axis dispatch (compile-witnessed builtin, never dispatched beyond 1-D)" % name)
        return self.sr[name]

    def wp(self, what):
        """Admit a semantics backed only by whole programs verified on hardware - in the 'wp' tier, recorded."""
        if self.tier != "wp":
            raise Refused(what + " - whole-program evidence only (run with tier 'wp' to admit it, reported)")
        self.admitted[what] = self.admitted.get(what, 0) + 1

    def op14059(self, off, size, toks):                           # read_sr: dst32, raw, SR, 0
        self.write_u(toks[0], self._sr(toks[2], off))

    def op14060(self, off, size, toks):                           # read_sr16: dst16, raw, SR, 0
        v = self._sr(toks[2], off)
        if np.any(v[self.active] > 0xFFFF):
            raise Refused("read_sr16 of a value above 16 bits: truncation unmeasured")
        self.write_u(toks[0], v)

    def op11842(self, off, size, toks):                           # movimm: the constant is the LAST operand
        self.write_u(toks[0], np.full(self.threads, int(toks[-1][4:]) & 0xFFFFFFFF, np.uint32))

    def op10825(self, off, size, toks):                           # mul: dst, 0, a, mod, b, mod, 0
        a = self.read_u(toks[2], int(toks[3][4:]))
        b = self.read_u(toks[4], int(toks[5][4:]))
        self.write_u(toks[0], (a.astype(np.uint64) * b.astype(np.uint64)) & 0xFFFFFFFF)

    def _imod(self, tok, scale=False):
        """An integer source modifier: lifetime bits only, plus the shift scale (bits 8-10) where it was measured
        (the add forms). Any other bit is refused: bit 3 is op10239's signedness (16/32 unsigned, 24/40 signed; the
        opcode glossary), so an integer modifier bit cannot be assumed inert. No delivered program sets one."""
        m = int(tok[4:])
        if m & ~(MOD_RELEASE | MOD_KEEP | (0x700 if scale else 0)):
            raise Refused("integer source modifier %d: bits beyond lifetime%s have no measured meaning here"
                          % (m, " and scale" if scale else ""))
        return m

    def op10282(self, off, size, toks):                           # add: dst, raw, a, mod, b, mod; b scaled by bmod
        a = self.read_u(toks[2], self._imod(toks[3]))
        bm = self._imod(toks[5], scale=True)
        b = self.read_u(toks[4], bm).astype(np.uint64) << np.uint64((bm >> 8) & 7)
        self.write_u(toks[0], (a.astype(np.uint64) + b) & 0xFFFFFFFF)

    def op10279(self, off, size, toks):                           # add-imm: dst, raw, imm, src, mod (src scaled)
        sm = self._imod(toks[4], scale=True)
        v = self.read_u(toks[3], sm).astype(np.uint64) << np.uint64((sm >> 8) & 7)
        self.write_u(toks[0], (v + np.uint64(int(toks[2][4:]) & 0xFFFFFFFF)) & 0xFFFFFFFF)

    def op11667(self, off, size, toks):                           # sub: dst, raw, a, mod, b, mod (wraps)
        a = self.read_u(toks[2], self._imod(toks[3])).astype(np.int64)
        b = self.read_u(toks[4], self._imod(toks[5])).astype(np.int64)
        self.write_u(toks[0], (a - b) & 0xFFFFFFFF)

    def _u64(self, toks, i):
        return self.read_u(toks[i], self._imod(toks[i + 1])).astype(np.uint64)

    def op10826(self, off, size, toks):                           # madd: dst, raw, a, mod, b, mod, c, mod
        a, b, c = self._u64(toks, 2), self._u64(toks, 4), self._u64(toks, 6)
        self.write_u(toks[0], (a * b + c) & 0xFFFFFFFF)

    def op10239(self, off, size, toks):                           # add.usat: dst, raw, a, mod, b, mod
        self.write_u(toks[0], np.minimum(self._u64(toks, 2) + self._u64(toks, 4), 0xFFFFFFFF))

    def op11624(self, off, size, toks):                           # sub.usat: dst, raw, a, mod, b, mod
        a, b = self._u64(toks, 2), self._u64(toks, 4)
        self.write_u(toks[0], np.where(a > b, a - b, 0))

    def op11190(self, off, size, toks):                           # not: dst, raw, src, mod
        self.write_u(toks[0], ~self._u64(toks, 2) & 0xFFFFFFFF)

    def op9986(self, off, size, toks):                            # msb: dst, raw, src, mod
        x = self._u64(toks, 2)
        v, r = x.copy(), np.zeros_like(x)
        for sh in (16, 8, 4, 2, 1):
            hi = v >= (np.uint64(1) << np.uint64(sh))
            r += np.where(hi, sh, 0).astype(np.uint64)
            v = np.where(hi, v >> np.uint64(sh), v)
        self.write_u(toks[0], np.where(x == 0, 0xFFFFFFFF, r))

    def op14047(self, off, size, toks):                           # bit reverse: dst, raw, src, mod
        v = self._u64(toks, 2)
        for sh, m in ((1, 0x55555555), (2, 0x33333333), (4, 0x0F0F0F0F), (8, 0x00FF00FF), (16, 0x0000FFFF)):
            sh, m = np.uint64(sh), np.uint64(m)
            v = ((v >> sh) & m) | ((v & m) << sh)
        self.write_u(toks[0], v & 0xFFFFFFFF)

    def op586(self, off, size, toks):                             # mov: dst, raw, src, mod
        self.write_u(toks[0], self.read_u(toks[2], self._imod(toks[3])))

    def _bitop(self, toks, fn):                                   # dst, raw, a, mod, b, mod
        a = self.read_u(toks[2], self._imod(toks[3])).astype(np.uint32)
        b = self.read_u(toks[4], self._imod(toks[5])).astype(np.uint32)
        self.write_u(toks[0], fn(a, b))

    def op424(self, off, size, toks):
        self._bitop(toks, lambda a, b: a & b)

    def op13575(self, off, size, toks):
        self._bitop(toks, lambda a, b: a | b)

    def op428(self, off, size, toks):                             # and16: zext(a16 & b16)
        if toks[4].startswith("reg:"):
            self._bitop(toks, lambda a, b: (a & b) & np.uint32(0xFFFF))
            return
        # THE POOL FORM (MM 25.141.16): the mask is uniform halfword K, and the container preloads its slot-13 constant
        # pool after the binding pointers, at uniform 4 x buffers + h. Apple's compiler is the oracle for that layout;
        # the preload itself is verified only by whole programs (the delivered q4 qmv chain), so tier wp only.
        import re
        mt = re.fullmatch(r"expr:bin\(op0,const\((\d+)\),2\)", toks[4])
        pool = getattr(self, "pool", None)
        if not mt or pool is None:
            raise Refused("op428 with its mask from the constant pool (%s) and no container pool read" % toks[4])
        h = int(mt.group(1)) - 4 * len(self.slots)
        if not 0 <= h < len(pool) // 2:
            raise Refused("op428 pool operand %s names halfword %d of a %d-byte pool" % (toks[4], h, len(pool)))
        if self.tier != "wp":
            raise Refused("op428 with its mask from the container's constant pool: the preload is verified only by "
                          "whole programs")
        self.wp("op428 mask from the container's slot-13 constant pool (uniform 4 x buffers + h)")
        mask = np.uint32(pool[2 * h] | pool[2 * h + 1] << 8)
        a = self.read_u(toks[2], self._imod(toks[3])).astype(np.uint32)
        self.write_u(toks[0], a & mask & np.uint32(0xFFFF))

    def op423(self, off, size, toks):                             # and-imm: dst, raw, src, mod, imm
        k = int(toks[4][4:]) & 0xFFFFFFFF                         # only the mask's bits of the source reach the result
        self.write_u(toks[0], self.read_u(toks[2], self._imod(toks[3]), k).astype(np.uint32) & np.uint32(k))

    def op426(self, off, size, toks):                             # and-imm on a 16-bit source (zero-extended)
        v = self.read_u(toks[2], self._imod(toks[3])).astype(np.uint32) & np.uint32(0xFFFF)
        self.write_u(toks[0], v & np.uint32(int(toks[4][4:])))

    def op10822(self, off, size, toks):                           # mul-imm: dst, raw, src, mod, imm, 0
        if toks[5] != "imm:0":
            raise Refused("mul-imm operand 5 = %s: unmeasured" % toks[5])
        v = self.read_u(toks[2], self._imod(toks[3])).astype(np.uint64)
        self.write_u(toks[0], (v * np.uint64(int(toks[4][4:]) & 0xFFFFFFFF)) & 0xFFFFFFFF)

    def op17770(self, off, size, toks):                           # xor-imm: dst, raw, src, mod, imm
        self.write_u(toks[0], self.read_u(toks[2], self._imod(toks[3])).astype(np.uint32) ^ np.uint32(int(toks[4][4:])))

    def op13574(self, off, size, toks):                           # or-imm: dst, raw, src, mod, imm
        k = int(toks[4][4:]) & 0xFFFFFFFF                         # a bit the immediate sets hides the source's
        self.write_u(toks[0], self.read_u(toks[2], self._imod(toks[3]), ~k & 0xFFFFFFFF).astype(np.uint32) | np.uint32(k))

    def _shift_imm(self, toks, left, src16=False):                # dst, raw, 0, src, mod, n, width
        if toks[2] != "imm:0":
            raise Refused("shift with operand 2 = %s: unmeasured" % toks[2])
        n, width = int(toks[5][4:]), int(toks[6][4:])
        if n >= 32 or not 1 <= width <= 32 or (width < 32 and not src16):
            raise Refused("shift by %d with width %d: unmeasured" % (n, width))
        keep = (1 << width) - 1                                   # the source bits that reach the result
        sig = (keep >> n) if left else ((keep << n) & 0xFFFFFFFF)
        v = self.read_u(toks[3], self._imod(toks[4]), sig & (0xFFFF if src16 else 0xFFFFFFFF)).astype(np.uint64)
        if src16:
            v &= np.uint64(0xFFFF)
        r = (v << np.uint64(n)) if left else (v >> np.uint64(n))
        self.write_u(toks[0], r & np.uint64((1 << width) - 1))

    def op14391(self, off, size, toks):
        self._shift_imm(toks, True)

    def op17013(self, off, size, toks):
        self._shift_imm(toks, False)

    def op17016(self, off, size, toks):
        self._shift_imm(toks, False, src16=True)

    def _shift_reg(self, toks, left):                             # dst, raw, 0, src, mod, amt, amod, width
        if toks[2] != "imm:0" or toks[7] != "imm:32":
            raise Refused("register shift with operands %s / %s: unmeasured" % (toks[2], toks[7]))
        k = self.read_u(toks[5], self._imod(toks[6])).astype(np.uint64) & np.uint64(0x7F)
        big = k >= 32
        ones = np.uint64(0xFFFFFFFF)
        kk0 = np.where(big, 0, k)
        sig = np.where(big, np.uint64(0), (ones >> kk0) if left else ((ones << kk0) & ones))
        v = self.read_u(toks[3], self._imod(toks[4]), sig).astype(np.uint64)
        kk = np.where(big, 0, k)
        r = (v << kk) if left else (v >> kk)
        self.write_u(toks[0], np.where(big, np.uint64(0), r) & np.uint64(0xFFFFFFFF))

    def op14392(self, off, size, toks):
        self._shift_reg(toks, True)

    def op17014(self, off, size, toks):
        self._shift_reg(toks, False)

    def _mem_operands(self, toks):
        """load/store: [value, mode|mod, width, bin expr, 0, index, index mod, displacement, element bytes]"""
        slot = self._binding(toks[3])
        index = self.read_u(toks[5], int(toks[6][4:]))
        disp, elem = int(toks[7][4:]), int(toks[8][4:])
        return slot, index, disp, elem

    def _load(self, toks, elem_want):
        slot, index, disp, elem = self._mem_operands(toks)
        if elem != elem_want:
            raise Refused("load element size %d on a %d-byte form" % (elem, elem_want))
        self.write_u(toks[0], self.load(slot, index, elem, disp))

    def _load_wide(self, toks, elem_want):
        """A 64- or 128-bit load into a register tuple, words little-endian from the lowest register
        (isa/g17-addressing-rules.json: long op12691 / int4 op12709 are one instruction per access)."""
        kind, regs = reg_parts(toks[0])
        slot, index, disp, scale = self._mem_operands(toks)
        if kind != "r32" or len(regs) * 4 != elem_want:
            raise Refused("wide load into %s" % reg_name(toks[0]))
        if scale not in (8, 16):
            if scale != 4:
                raise Refused("wide load with index scale %d" % scale)
            self.wp("op%d with index scale 4" % (12709 if elem_want == 16 else 12691))
        m = self.mem[slot]
        addr = self.pad + index.astype(np.int64) * scale + disp
        ok = (addr >= 0) & (addr + elem_want <= len(m))
        if self.RACE_CHECK:
            act = self.active
            self._track_read(("dev", slot), len(m), addr[act], np.nonzero(act)[0], elem_want)
        for w, r in enumerate(regs):
            v = np.zeros(self.threads, np.uint32)
            a = addr[ok] + 4 * w
            for b in range(4):
                v[ok] |= m[a + b].astype(np.uint32) << np.uint32(8 * b)
            self.write_u("reg:%d" % (R32 + r), v)

    def op12709(self, off, size, toks):
        self._load_wide(toks, 16)

    def op12691(self, off, size, toks):
        self._load_wide(toks, 8)

    def op12682(self, off, size, toks):
        self._load(toks, 4)

    def op10090(self, off, size, toks):
        """dst, mode, operation, bin expr, 0, index, index mod, displacement, element bytes, value, value mod. Active
        lanes are applied one at a time in thread order. For a contended cell that is one of the orders the hardware
        may pick: add, and, or, xor, min and max leave the same memory in any order, but the returns (and sub and
        exchange) depend on it, so a contended atomic is admitted only in tier wp and reported."""
        known = _atomic_table().get((size, toks[2]))
        if known is None:
            raise Refused("atomic: no measured encoding of %d bytes with operation operand %s" % (size, toks[2]))
        name, modes = known
        if (int(toks[1][4:]) & ~INERT_MODE_BITS) not in modes:
            raise Refused("atomic %s: mode %s differs from the measured encodings'" % (name, toks[1]))
        if len(toks) < 11 or not toks[9].startswith("reg:"):
            raise Refused("atomic %s with a non-register value (the immediate form is not modelled)" % name)
        slot, index, disp, elem = self._mem_operands(toks)
        if elem != 4:
            raise Refused("atomic on a %d-byte element" % elem)
        value = self.read_u(toks[9], self._imod(toks[10]))
        m, act = self.mem[slot], self.active
        addr = self.pad + index.astype(np.int64) * 4 + disp
        lanes = np.nonzero(act)[0]
        if np.any(addr[lanes] < 0) or np.any(addr[lanes] + 4 > len(m)):
            raise Refused("atomic outside buffer %d" % slot)
        if len(np.unique(addr[lanes])) < len(lanes):
            if self.tier != "wp":
                raise Refused("atomic %s: %d lanes share a cell, and the order the hardware applies them in is "
                              "unmeasured" % (name, len(lanes) - len(np.unique(addr[lanes]))))
            self.wp("op10090 %s on a contended cell (lanes applied in thread order)" % name)
        fn, old = ATOMIC_FN[name], np.zeros(self.threads, np.uint32)
        for t in lanes:
            a = int(addr[t])
            o = int(m[a]) | int(m[a + 1]) << 8 | int(m[a + 2]) << 16 | int(m[a + 3]) << 24
            n = fn(o, int(value[t]))
            m[a:a + 4] = np.frombuffer(n.to_bytes(4, "little"), np.uint8)
            old[t] = o
        self.write_u(toks[0], old)

    def op12646(self, off, size, toks):
        self._load(toks, 2)

    def _store(self, toks, elem_want, lowest_lane_wins=False):
        value = self.read_u(toks[0], int(toks[1][4:]))
        slot, index, disp, elem = self._mem_operands(toks)
        if elem != elem_want:
            raise Refused("store element size %d on a %d-byte form" % (elem, elem_want))
        self.store(slot, index, elem, disp, value.astype(np.uint32), lowest_lane_wins=lowest_lane_wins)

    def op17235(self, off, size, toks):
        """store sub-form 01, IMMEDIATE address: value, mod, width, binding, 0, index, displacement, element. The word
        at binding + (index + displacement) x element (isa/g17-addressing-rules.json's rule; the onestore receipts
        land exactly on the predicted word and leave its neighbour)."""
        if len(toks) != 8 or toks[2] != "imm:2066" or not toks[5].startswith("imm:"):
            raise Refused("op17235 form %s: only the word store with an immediate index is measured" % " ".join(toks[2:]))
        value = self.read_u(toks[0], int(toks[1][4:])).astype(np.uint32)
        slot = self._binding(toks[3])
        elem = int(toks[7][4:])
        byte = (int(toks[5][4:]) + int(toks[6][4:])) * elem
        self.store(slot, np.zeros(self.threads, np.int64), 1, byte, value, width=4)

    # THE IMMEDIATE-ADDRESS VECTOR FORMS (MM 25.177): value/destination, mod, component mask, binding, 0, index,
    # displacement, element - the op17235 operand order, at the word (index + displacement) x element of the
    # binding. The mask field (operand 2) is a component mask in bits 4-7 above 0x802: 2066 one word, 2098 two,
    # 2290 four, from the lowest register of the tuple up. Only those three masks are modelled.
    IMM_MASK = {2066: 1, 2098: 2, 2162: 3, 2290: 4}
    PTR_TAG = 0xB0DE0000                                       # a binding pointer's high word: tag | slot (op14061)

    def _imm_address(self, toks, what):
        if len(toks) != 8 or toks[2][4:].isdigit() is False or int(toks[2][4:]) not in self.IMM_MASK \
                or toks[4] not in (("imm:0", "imm:16") if toks[3].startswith("reg:") else ("imm:0",)) or not (toks[5].startswith("imm:") and toks[6].startswith("imm:")
                                              and toks[7].startswith("imm:")):
            raise Refused("op%s immediate-address form %s: only the component masks 0x812/0x832/0x872/0x8f2 with an "
                          "immediate index are modelled" % (what, " ".join(toks[2:])))
        words = self.IMM_MASK[int(toks[2][4:])]
        kind, regs = reg_parts(toks[0])
        if kind != "r32" or len(regs) != words:
            raise Refused("op%s: %d components into %s" % (what, words, reg_name(toks[0])))
        byte = (int(toks[5][4:]) + int(toks[6][4:])) * int(toks[7][4:])
        if toks[3].startswith("reg:"):
            return (*self._pointer(toks[3], what, byte), regs)
        return self._binding(toks[3]), byte, regs

    def _pointer(self, tok, what, byte):
        """A register-pair BASE (the constant program's form): the pair must hold a binding pointer op14061 read,
        (offset, PTR_TAG | slot); anything else is refused. The address is the pointer's offset plus `byte`."""
        kind, regs = reg_parts(tok)
        if kind != "r32" or len(regs) != 2:
            raise Refused("op%s base %s is not a register pair" % (what, reg_name(tok)))
        lo = self.read_u("reg:%d" % (R32 + regs[0]), 0)[self.active]
        hi = self.read_u("reg:%d" % (R32 + regs[1]), 0)[self.active]
        if lo.size == 0 or np.any(hi != hi[0]) or np.any(lo != lo[0]) or (int(hi[0]) & 0xFFFF0000) != self.PTR_TAG:
            raise Refused("op%s base %s does not hold a binding pointer" % (what, reg_name(tok)))
        slot = int(hi[0]) & 0xFFFF
        if slot not in self.mem:
            raise Refused("op%s base names slot %d, which is not bound" % (what, slot))
        return slot, int(lo[0]) + byte

    def _imm_load(self, toks, what):
        slot, byte, regs = self._imm_address(toks, what)
        zero = np.zeros(self.threads, np.int64)
        for w, r in enumerate(regs):
            self.write_u("reg:%d" % (R32 + r), self.load(slot, zero, 4, byte + 4 * w))   # 4 bytes a word

    def _imm_store(self, toks, what):
        slot, byte, regs = self._imm_address(toks, what)
        zero = np.zeros(self.threads, np.int64)
        for w, r in enumerate(regs):
            self.store(slot, zero, 1, byte + 4 * w, self.read_u("reg:%d" % (R32 + r), int(toks[1][4:])).astype(np.uint32),
                       width=4)

    def op12688(self, off, size, toks):
        self._imm_load(toks, 12688)

    def op12706(self, off, size, toks):
        self._imm_load(toks, 12706)

    # ---- MM 25.195: six census blockers (tools/g17sixprobe.py; each against Apple's own code on the GPU)
    F16_ADD = "once"                     # op776 rounds once; through fp32 is the same function on every half (25.195)

    def op10289(self, off, size, toks):                           # add16 imm: dst16, mode, imm, src16, mod
        a = self.read_u(toks[3], self._imod(toks[4])).astype(np.uint32)
        self.write_u(toks[0], (a + np.uint32(int(toks[2][4:]) & 0xFFFF)) & np.uint32(0xFFFF))

    def op776(self, off, size, toks):                             # fadd.imm.f16: dst16, mode, K8, a16, amod: a + K
        if reg_parts(toks[0])[0] not in ("lo", "hi") or reg_parts(toks[3])[0] not in ("lo", "hi"):
            raise Refused("op776 operands %s, %s: only register halves are modelled" % (toks[0], toks[3]))
        a = self.read_f(toks[3], int(toks[4][4:])).astype(np.float64)       # a half, widened exactly
        k = np.float64(self.fimm8(toks[2][4:]))
        if self.F16_ADD == "once":
            r = self._round_sum(a, k, np.float16)
        else:                                                     # "twice": through fp32
            r = self._round_sum(a, k, np.float32).astype(np.float16)
        # a NaN input gives the canonical quiet NaN 0x7e00, never its payload (the GPU on all 194 NaN lanes of the
        # probe; op1014's t1014 receipt found the same)
        bits = np.asarray(r, np.float16).view(np.uint16)
        self.write_u(toks[0], np.where(np.isnan(a), np.uint16(0x7E00), bits).astype(np.uint16))

    def op11180(self, off, size, toks):                           # cvt.i2f from 16 bits: dst32, lead, code, 0, src16
        code = int(toks[2][4:])
        if code not in (2, 3):
            raise Refused("op11180 code %d: only 2 (unsigned 16) and 3 (signed 16) ran" % code)
        v = self.read_u(toks[4], self._imod(toks[5])).astype(np.uint16)
        f = (v.view(np.int16) if code == 3 else v).astype(np.float32)          # exact: 16 bits fit fp32
        self.write_f32(toks[0], f)

    def op17199(self, off, size, toks):
        """16-bit store at an immediate address: value16, mod, mask 2065, binding, 0, index, displacement, element -
        the halfword at (index + displacement) x element (25.177's order)."""
        if len(toks) != 8 or toks[2] != "imm:2065" or toks[4] != "imm:0" or not all(
                t.startswith("imm:") for t in toks[5:8]):
            raise Refused("op17199 form %s" % " ".join(toks[2:]))
        if reg_parts(toks[0])[0] not in ("lo", "hi"):
            raise Refused("op17199 value %s: a half is stored" % reg_name(toks[0]))
        byte = (int(toks[5][4:]) + int(toks[6][4:])) * int(toks[7][4:])
        v = self.read_u(toks[0], int(toks[1][4:])).astype(np.uint32)
        self.store(self._binding(toks[3]), np.zeros(self.threads, np.int64), 1, byte, v, width=2)

    def op11462(self, off, size, toks):                           # csel into a half: dst16, raw, cc, a, ma, b, mb, 1, 0
        if (toks[7], toks[8]) != ("imm:1", "imm:0"):
            raise Refused("op11462 arms %s/%s: only 1/0 ran on hardware" % (toks[7], toks[8]))
        a = self.read_u(toks[3], self._imod(toks[4]))
        b = self.read_u(toks[5], self._imod(toks[6]))
        self.write_u(toks[0], self._rel(int(toks[2][4:]), a, b, off).astype(np.uint32))

    def op10285(self, off, size, toks):                           # add: dst32, raw, a16, mod, b32, mod: zext(a) + b
        if reg_parts(toks[2])[0] not in ("lo", "hi"):
            raise Refused("op10285 first source %s: a half is modelled" % reg_name(toks[2]))
        a = self.read_u(toks[2], self._imod(toks[3])).astype(np.uint64)
        b = self.read_u(toks[4], self._imod(toks[5])).astype(np.uint64)
        self.write_u(toks[0], ((a + b) & 0xFFFFFFFF).astype(np.uint32))

    def op17253(self, off, size, toks):                           # three-word store (mask 2162): op12706's store
        self._imm_store(toks, 17253)

    def op435(self, off, size, toks):
        """16-bit AND with an immediate: dst16, mode, src16, mod, imm -> dst = src & imm (the glossary's `a & 0xf` on
        ushort; MM 25.192). Both operands are register halves."""
        if len(toks) != 5 or not toks[4].startswith("imm:"):
            raise Refused("op435 form %s" % " ".join(toks))
        for t in (toks[0], toks[2]):
            if reg_parts(t)[0] not in ("lo", "hi"):
                raise Refused("op435 operand %s: only register halves are modelled" % reg_name(t))
        a = self.read_u(toks[2], self._imod(toks[3])).astype(np.uint32)
        self.write_u(toks[0], (a & np.uint32(int(toks[4][4:]) & 0xFFFF)).astype(np.uint32))

    def op12697(self, off, size, toks):                           # two-word load (mask 2098): op17244's load
        self._imm_load(toks, 12697)

    def op555(self, off, size, toks):
        """dst16, imm: a 16-bit move immediate into a register half (byte 2 is the immediate, MM 25.184). Main writes
        0, clearing a half before the register is read as a 32-bit index; that is the form receipted. A nonzero
        immediate (32, only in constant programs, into a half no output reads) is written as UNKNOWN bits, so a
        read that could reach an output is refused."""
        if len(toks) != 2 or not toks[1].startswith("imm:"):
            raise Refused("op555 form %s" % " ".join(toks))
        kind, regs = reg_parts(toks[0])
        if kind not in ("lo", "hi"):
            raise Refused("op555 into %s: only a register half is modelled" % reg_name(toks[0]))
        v = int(toks[1][4:])
        self.write_u(toks[0], v)
        if v:
            self.unk[self.active, regs[0], 0 if kind == "lo" else 1] = True
            self.clean[regs[0], 0 if kind == "lo" else 1] = False

    def op14061(self, off, size, toks):
        """dst, mode, H, 0, 16: a word of the binding table - binding K's 64-bit pointer at halfwords 8 + K (low
        word) and 10 + K (high word), K the binding table key (MM 25.179). It is modelled as (0, PTR_TAG | slot),
        a pointer only the register-pair loads accept."""
        if len(toks) != 5 or toks[3] != "imm:0" or toks[4] != "imm:16" or not toks[2].startswith("imm:"):
            raise Refused("op14061 form %s: only the binding-table word read is modelled" % " ".join(toks[1:]))
        rel = int(toks[2][4:]) - 8
        k, part = rel - rel % 4, rel % 4
        if rel < 0 or part not in (0, 2) or self.table is None or k not in self.table or self.table[k] not in self.mem:
            raise Refused("op14061 halfword %s names no bound binding (table %s)" % (toks[2], self.table))
        self.write_u(toks[0], 0 if part == 0 else self.PTR_TAG | self.table[k])

    def op11185(self, off, size, toks):                           # cvt.i2f into a uniform: dst, mode, then op11179's
        self.op11179(off, size, [toks[0]] + toks[2:])

    def op592(self, off, size, toks):                             # mov into a uniform: op586's operands
        """A 32-bit source half with UNKNOWN bits (op555's nonzero immediate) is published as an unknown halfword: a
        main read of it is refused, a read of the other half is not."""
        kind, regs = reg_parts(toks[2 + 1]) if toks[3].startswith("reg:") else (None, None)
        if kind == "r32" and len(regs) == 1 and self.UNIFORM.fullmatch(toks[0]):
            r, act = regs[0], self.active
            unknown = {h for h in (0, 1) if self.unk[act, r, h].any()}
            if unknown:
                if not all(self.written[act, r, h].all() and not self.poison[act, r, h].any() for h in (0, 1)):
                    raise Refused("op592 source R%d is unwritten or released" % r)
                return self._write_uniform(toks[0], self.R[:, r], unknown=unknown)
        self.op586(off, size, [toks[0], toks[1], toks[3], toks[4]])

    def op1038(self, off, size, toks):                            # fadd.imm into a uniform: op1006 + a mode operand
        self.op1006(off, size, [toks[0], toks[1], toks[3], toks[4], toks[5]])

    def op12715(self, off, size, toks):
        self._imm_load(toks, 12715)

    def op17244(self, off, size, toks):
        self._imm_store(toks, 17244)

    def op17262(self, off, size, toks):
        self._imm_store(toks, 17262)

    def op17229(self, off, size, toks):
        self._store(toks, 4, lowest_lane_wins=True)

    def op17193(self, off, size, toks):
        self._store(toks, 2)

    def _ftz(self, f):
        """f32 arithmetic flushes subnormal inputs and outputs to a signed zero (recon 5455, executed)."""
        f = np.asarray(f, np.float32)
        return np.where(np.abs(f) < np.finfo(np.float32).tiny, np.copysign(np.float32(0), f), f).astype(np.float32)

    def op998(self, off, size, toks):                             # fadd: dst, mode, a, mod, b, mod
        a = self._ftz(self.read_f(toks[2], int(toks[3][4:])))
        b = self._ftz(self.read_f(toks[4], int(toks[5][4:])))
        with np.errstate(over="ignore", invalid="ignore"):      # inf and NaN are values here
            r = a + b
        self.write_f32(toks[0], self._ftz(r))

    def op3290(self, off, size, toks):                            # fmul: dst, mode, a, mod, b, mod
        a = self._ftz(self.read_f(toks[2], int(toks[3][4:])))
        b = self._ftz(self.read_f(toks[4], int(toks[5][4:])))
        with np.errstate(over="ignore", invalid="ignore"):
            r = a * b
        self.write_f32(toks[0], self._ftz(r))

    @staticmethod
    def _round_sum(p, c, dtype=np.float32):
        """p + c rounded ONCE to dtype (nearest even), for float64 p and c that are exact (a product of two floats of
        at most 53 significant bits together, a float32, a float16)."""
        p, c = np.asarray(p, np.float64), np.asarray(c, np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            s_ = p + c
            bb = s_ - p
            e = (p - (s_ - bb)) + (c - bb)                        # TwoSum: p + c == s_ + e exactly
            r = s_.astype(dtype)
            # the double rounding f64 -> dtype is wrong only when s_ is exactly a dtype midpoint and e is not zero
            lo = np.where(r.astype(np.float64) > s_, np.nextafter(r, dtype(-np.inf)), r)
            hi = np.where(r.astype(np.float64) < s_, np.nextafter(r, dtype(np.inf)), r)
            mid = (lo.astype(np.float64) + hi.astype(np.float64)) / 2
        tie = (s_ == mid) & (lo != hi) & (e != 0) & np.isfinite(e)
        return np.where(tie & (e > 0), hi, np.where(tie & (e < 0), lo, r)).astype(dtype)

    def op2190(self, off, size, toks):                            # ffma: a b + c, ONE rounding (claims.toml:4030)
        a = self._ftz(self.read_f(toks[2], int(toks[3][4:]))).astype(np.float64)
        b = self._ftz(self.read_f(toks[4], int(toks[5][4:]))).astype(np.float64)
        c = self._ftz(self.read_f(toks[6], int(toks[7][4:]))).astype(np.float64)
        self.write_f32(toks[0], self._ftz(self._round_sum(a * b, c)))  # a * b is exact: 24 x 24 bits

    @staticmethod
    def fimm8(k):
        """The 8-bit float immediate (isa/g17-float-immediate.toml, swept on hardware and matched to Apple's
        constants): sign, 3-bit exponent, 4-bit mantissa, value = (m + 16 (e != 0)) 2^(e - 1 if e else 0) / 64."""
        k = int(k)
        if not 0 <= k < 256:
            raise Refused("float immediate %d is not 8 bits" % k)
        e, m = (k >> 4) & 7, k & 15
        v = (m + 16 * (e != 0)) * 2.0 ** (e - 1 if e else 0) / 64
        return -v if k & 128 else v

    def _fma_imm(self, a, k, c, off):
        """a k + c for a float immediate k: one rounding. a k is exact in float32 whenever it is a normal float (k has
        at most 5 significant bits), so the fused and unfused results can differ only where a k leaves the normal
        range - which is refused (fused versus unfused is unmeasured on these forms)."""
        p = a.astype(np.float64) * k
        fused = self._round_sum(p, c.astype(np.float64))
        with np.errstate(over="ignore"):
            pf = self._ftz(p.astype(np.float32))
            unfused = self._ftz(pf + c)
        act = self.active
        if np.any((fused.view(np.uint32) != unfused.view(np.uint32))[act] & ~np.isnan(fused[act])):
            raise Refused("fma with an immediate (+0x%x): the fused and unfused results differ (the product leaves "
                          "the normal range) and which one the hardware computes is unmeasured" % off)
        return self._ftz(fused)

    def op2198(self, off, size, toks):                            # dst, mode, a, amod, K8, c, cmod: a K + c
        a = self._ftz(self.read_f(toks[2], int(toks[3][4:])))
        c = self._ftz(self.read_f(toks[5], int(toks[6][4:])))
        self.write_f32(toks[0], self._fma_imm(a, self.fimm8(toks[4][4:]), c, off))

    def op2192(self, off, size, toks):                            # dst, mode, a, amod, b, bmod, K8: a b + K
        a = self._ftz(self.read_f(toks[2], int(toks[3][4:])))
        b = self._ftz(self.read_f(toks[4], int(toks[5][4:])))
        k = np.float64(self.fimm8(toks[6][4:]))
        # ROUNDED ONCE (fused): isa/g17-rounding-results.json, 4,096 lanes of Apple's fma(a, b, 1.0f) on the GPU, 3,186
        # of them where once and twice differ - fused matches every lane, twice misses all 3,186 (MM 25.184)
        fused = self._round_sum(a.astype(np.float64) * b.astype(np.float64), k)
        self.write_f32(toks[0], self._ftz(fused))

    def op11182(self, off, size, toks):                           # cvt.u2h: dst16, lead, sign, 0, src, mod
        """u32 -> half, round to nearest even, past half's range -> inf (isa/g17-rounding-results.json: 4,096 lanes of
        Apple's (half)u on the GPU; 2,992 past the range give inf, not 65504; below 2^24 the u32 is exact in fp32, so
        rounding once or through fp32 cannot differ - MM 25.184). Only the unsigned code (4) ran."""
        if toks[2] != "imm:4":
            raise Refused("op11182 sign code %s: only the unsigned code 4 is measured" % toks[2])
        kind, _regs = reg_parts(toks[0])
        if kind not in ("lo", "hi"):
            raise Refused("op11182 into %s: the result is a half" % reg_name(toks[0]))
        v = self.read_u(toks[4], self._imod(toks[5])).astype(np.uint32)
        with np.errstate(over="ignore"):
            h = v.astype(np.float64).astype(np.float16)
        self.write_u(toks[0], h.view(np.uint16))

    def op2200(self, off, size, toks):                            # dst, mode, a, amod, K1, K2: a K1 + K2
        a = self._ftz(self.read_f(toks[2], int(toks[3][4:])))
        c = np.full(self.threads, self.fimm8(toks[5][4:]), np.float32)
        self.write_f32(toks[0], self._fma_imm(a, self.fimm8(toks[4][4:]), c, off))

    def op1006(self, off, size, toks):                            # fadd.imm: dst, mode, K8, a, amod: a + K
        a = self._ftz(self.read_f(toks[3], int(toks[4][4:])))
        self.write_f32(toks[0], self._ftz(self._round_sum(a, np.float64(self.fimm8(toks[2][4:])))))

    def op904(self, off, size, toks):                             # fadd.imm.sat: dst, mode, a, amod, K8: sat(a + K)
        k = self.fimm8(toks[4][4:])
        if k != 0:
            raise Refused("op904 with a nonzero immediate (%s): only +-0 ran on hardware" % toks[4])
        a = self._ftz(self.read_f(toks[2], int(toks[3][4:])))
        if np.any(np.isnan(a[self.active])):
            raise Refused("op904 (+0x%x) of a NaN: saturate's NaN result is unmeasured" % off)
        self.write_f32(toks[0], np.clip(a + np.float32(0.0), 0.0, 1.0).astype(np.float32) + np.float32(0.0))

    def _to_f16(self, toks, off, x, y, mul):
        """An f32 operation with a 16-bit result: rounded once to half (nearest even), half subnormals kept (op1015's
        width receipt: 10, 17, 24). Whether the hardware rounds once or through float32 first is unmeasured, so a
        lane where the two differ is refused."""
        x, y = x.astype(np.float64), y.astype(np.float64)
        once = (self._round_sum(x * y, np.float64(0.0), np.float16) if mul else self._round_sum(x, y, np.float16))
        with np.errstate(over="ignore", invalid="ignore"):
            twice = (self._round_sum(x * y, np.float64(0.0)) if mul else self._round_sum(x, y)).astype(np.float16)
        act = self.active
        if np.any((once.view(np.uint16) != twice.view(np.uint16))[act] & ~np.isnan(once[act])):
            raise Refused("op%s (+0x%x): rounding once to half and through float32 differ here; which the hardware "
                          "does is unmeasured" % (toks and "", off))
        self.write_u(toks[0], once.view(np.uint16))

    def op1014(self, off, size, toks):                            # dst16, mode, a32, amod, b32, bmod: half(a + b)
        a = self._ftz(self.read_f(toks[2], int(toks[3][4:])))
        b = self._ftz(self.read_f(toks[4], int(toks[5][4:])))
        self._to_f16(toks, off, a, b, mul=False)

    def op1015(self, off, size, toks):                            # dst16, mode, a32, amod, b16, bmod: half(a + b)
        a = self._ftz(self.read_f(toks[2], int(toks[3][4:])))
        b = self.read_f(toks[4], int(toks[5][4:]))
        self._to_f16(toks, off, a, b, mul=False)

    def op3306(self, off, size, toks):                            # dst16, mode, a32, amod, b32, bmod: half(a b)
        a = self._ftz(self.read_f(toks[2], int(toks[3][4:])))
        b = self._ftz(self.read_f(toks[4], int(toks[5][4:])))
        self._to_f16(toks, off, a, b, mul=True)

    def op10283(self, off, size, toks):                           # add a32 + zext(b16): dst, raw, a, mod, b16, mod
        a = self.read_u(toks[2], self._imod(toks[3])).astype(np.uint64)
        b = self.read_u(toks[4], self._imod(toks[5])).astype(np.uint64) & np.uint64(0xFFFF)
        self.write_u(toks[0], (a + b) & 0xFFFFFFFF)

    def op590(self, off, size, toks):                             # mov16: dst16, raw, src16, mod
        self.write_u(toks[0], self.read_u(toks[2], self._imod(toks[3])).astype(np.uint32) & np.uint32(0xFFFF))

    def op3818(self, off, size, toks):                            # ftrunc f32: dst, mode, src, mod
        x = self._ftz(self.read_f(toks[2], int(toks[3][4:])))
        if np.any(np.isnan(x[self.active])):
            raise Refused("op3818 (+0x%x) of a NaN: its NaN result is unmeasured" % off)
        self.write_f32(toks[0], np.trunc(x).astype(np.float32))

    def op16805(self, off, size, toks):                           # asr-imm: dst, raw, src, mod, n
        m = int(toks[3][4:])
        if m & ~(MOD_RELEASE | MOD_KEEP) != 8:
            raise Refused("op16805 source modifier %d: only 24 (bit 3 set) ran on hardware" % m)
        n = int(toks[4][4:])
        if not 0 <= n < 32:
            raise Refused("op16805 shift %d: unmeasured" % n)
        v = self.read_u(toks[2], m & (MOD_RELEASE | MOD_KEEP)).astype(np.uint32).view(np.int32)
        self.write_u(toks[0], (v >> np.int32(n)).view(np.uint32))

    def op17771(self, off, size, toks):                           # xor: dst, raw, a, mod, b, mod
        self._bitop(toks, lambda a, b: a ^ b)

    def op13588(self, off, size, toks):                           # or16: dst16, raw, a16, mod, b16, mod
        self._bitop(toks, lambda a, b: (a | b) & np.uint32(0xFFFF))

    def op17784(self, off, size, toks):                           # xor16: dst16, raw, a16, mod, b16, mod
        self._bitop(toks, lambda a, b: (a ^ b) & np.uint32(0xFFFF))

    def op554(self, off, size, toks):                             # movimm: dst, imm - only 0 ever ran
        if toks[1] != "imm:0":
            raise Refused("op554 immediate %s: only 0 ran on hardware" % toks[1])
        self.write_u(toks[0], np.zeros(self.threads, np.uint32))

    def op11666(self, off, size, toks):                           # sub-imm: dst, raw, a, mod, imm (wraps)
        a = self.read_u(toks[2], self._imod(toks[3])).astype(np.int64)
        self.write_u(toks[0], (a - (int(toks[4][4:]) & 0xFFFFFFFF)) & 0xFFFFFFFF)

    def op14157(self, off, size, toks):                           # simd.shuffle from a CONSTANT lane: dst, mode, src, mod, lane
        lane_k = int(toks[4][4:])
        if lane_k != 0:
            raise Refused("shuffle from constant lane %d: only lane 0 ran on hardware" % lane_k)
        if self.group % 32:
            raise Refused("shuffle in a threadgroup that is not whole simdgroups")
        sg = self.active.reshape(-1, 32)
        if np.any(sg.any(1) & ~sg.all(1)):
            raise Refused("shuffle from lane 0 (+0x%x) in a partially active simdgroup: a constant lane and the first "
                          "active lane are not separated by any receipt" % off)
        r, half = self._where(toks[2])
        src = (np.arange(self.threads) // 32) * 32 + lane_k
        act = self.active
        for h in HALVES[half]:
            if not np.all(self.written[src[act], r, h]):
                raise Refused("shuffle (+0x%x) reads r%d unwritten in its source lane" % (off, r))
            rel = self.poison[src[act], r, h]
            if np.any(rel) and np.any(_half_bits(self.R[src[act], r][rel], h) != 0):
                raise Refused("shuffle (+0x%x) reads r%d released, nonzero, in its source lane" % (off, r))
        if any(np.any(self.unk[src[self.active], r, h]) for h in HALVES[half]):
            raise Refused("a shuffle reads r%d with unknown bits in its source lane" % r)
        v = self.R[src, r]
        v = v & np.uint32(0xFFFF) if half == "lo" else (v >> np.uint32(16) if half == "hi" else v)
        if int(toks[3][4:]) & MOD_RELEASE:
            self.pending.append((act.copy(), r, HALVES[half]))
        self.write_u(toks[0], v)

    def op11179(self, off, size, toks):                           # cvt.i2f: dst, lead, sign, 0, src, mod
        code = int(toks[2][4:])
        v = self.read_u(toks[4], self._imod(toks[5]))
        if code == 4:
            f = v.astype(np.uint32).astype(np.float64).astype(np.float32)   # exact in f64, one rounding to f32
        elif code == 5:
            f = v.astype(np.uint32).view(np.int32).astype(np.float64).astype(np.float32)
        else:
            raise Refused("cvt.i2f sign code %d: unmeasured" % code)
        self.write_f32(toks[0], f)

    def op1004(self, off, size, toks):                            # f16 -> f32: dst, mode, src16, mod, imm
        if toks[-1] != "imm:128":
            raise Refused("op1004 immediate %s: only the widening form (imm 128) is measured" % toks[-1])
        self.write_f32(toks[0], self.read_f(toks[2], int(toks[3][4:])))

    def op1016(self, off, size, toks):                            # f32 -> f16, round to nearest even
        if toks[-1] != "imm:128":
            raise Refused("op1016 immediate %s: only the plain narrowing (imm 128) is measured" % toks[-1])
        f = self.read_f(toks[2], int(toks[3][4:]))
        self.write_u(toks[0], f.astype(np.float16).view(np.uint16))

    def op798(self, off, size, toks):                             # ffma.f16: dst, mod, a, mod, b, mod, c, mod
        if toks[1] != "imm:2147483648":
            raise Refused("op798 operand 1 %s: only Apple's 2147483648 is measured" % toks[1])
        mods = [int(t[4:]) for t in toks[3:8:2]]
        if any(m & ~(MOD_RELEASE | MOD_KEEP) for m in mods):
            raise Refused("op798 source modifier %s: only the lifetimes are measured" % mods)
        src = []
        for tok, m in zip(toks[2:8:2], mods):
            if self._where(tok)[1] is None:
                raise Refused("op798 source %s is a 32-bit register: the form's sources are halves" % tok)
            src.append(self.read_f(tok, m).astype(np.float64))
        self.write_u(toks[0], fma16_exact(*src))

    def op684(self, off, size, toks):
        return "end"

    # integer relations by the printed code, which is 8 | field (isa/g17-condition-codes.toml, measured on op11313;
    # tools/g17normcheck.py). 14 (signed >) never ran on inputs where signedness matters: guarded below
    ICMP = {8: "eq", 9: "ult", 10: "ugt", 12: "eq", 13: "slt", 14: "sgt"}

    def _rel(self, code, a, b, off):
        rel = self.ICMP.get(code)
        if rel is None:
            raise Refused("integer condition code %d (+0x%x): unmeasured" % (code, off))
        a = np.asarray(a, np.uint64) & np.uint64(0xFFFFFFFF)
        b = np.asarray(b, np.uint64) & np.uint64(0xFFFFFFFF)
        sa = np.where(a >= 2 ** 31, a.astype(np.int64) - 2 ** 32, a.astype(np.int64))
        sb = np.where(b >= 2 ** 31, b.astype(np.int64) - 2 ** 32, b.astype(np.int64))
        return {"eq": a == b, "ult": a < b, "ugt": a > b, "slt": sa < sb, "sgt": sa > sb}[rel]

    def op10369(self, off, size, toks):                           # cmp: flag, raw, cc, src, mod, K (8-bit)
        kind, idx = reg_parts(toks[0])
        if kind != "flag":
            raise Refused("cmp destination %s" % reg_name(toks[0]))
        v = self.read_u(toks[3], self._imod(toks[4]))
        if np.any(v[self.active] >= 2 ** 31):
            raise Refused("cmp (+0x%x) of a source >= 2^31: unmeasured on this opcode" % off)
        f = self._rel(int(toks[2][4:]), v, np.uint64(int(toks[5][4:])), off)
        self.F[:, idx[0]] = np.where(self.active, f, self.F[:, idx[0]])
        self.fwritten[idx[0]] = True

    def op11372(self, off, size, toks):                           # csel.reg: dst, raw, cc, a, ma, b, mb, 1, 0
        if (toks[7], toks[8]) != ("imm:1", "imm:0"):
            raise Refused("csel.reg arms %s/%s: only 1/0 ran on hardware" % (toks[7], toks[8]))
        a = self.read_u(toks[3], self._imod(toks[4]))
        b = self.read_u(toks[5], self._imod(toks[6]))
        self.write_u(toks[0], self._rel(int(toks[2][4:]), a, b, off).astype(np.uint32))

    def op11375(self, off, size, toks):                           # select: dst, lead, cc, a, ma, b, mb, x, mx, y, my
        code = int(toks[2][4:])
        a = self.read_u(toks[3], self._imod(toks[4]))
        b = self.read_u(toks[5], self._imod(toks[6]))
        if code == 15:                                            # the bit test (claims.toml:3725)
            c = (a & b) != 0
        elif code == 14:
            act = self.active
            if np.any((a[act] >= 2 ** 31) | (b[act] >= 2 ** 31)):
                raise Refused("select cc 14 (+0x%x) on an input >= 2^31: its signedness is unmeasured" % off)
            c = self._rel(14, a, b, off)
        else:
            c = self._rel(code, a, b, off)
        x = self.read_u(toks[7], self._imod(toks[8]))
        y = self.read_u(toks[9], self._imod(toks[10]))
        self.write_u(toks[0], np.where(c, x, y))

    def op9700(self, off, size, toks):                            # fselect: dst, lead, cc, a, ma, b, mb, x, mx, y, my
        code = int(toks[2][4:])
        mods = [int(toks[i][4:]) for i in (4, 6, 8, 10)]
        if any(m & (MOD_NEG | MOD_ABS) for m in mods):
            raise Refused("fselect with negate/abs modifiers: unmeasured")
        a = self.read_f(toks[3], mods[0])
        b = self.read_f(toks[5], mods[1])
        xu = self.read_u(toks[7], mods[2])
        yu = self.read_u(toks[9], mods[3])
        with np.errstate(invalid="ignore"):
            rel = {0: a == b, 1: a < b, 2: a > b, 5: a >= b, 6: a <= b, 4: a != b,
                   # codes 3 and 7: a NaN a selects y FIRST, then a NaN b selects x (receipt lane.fsel_code3/7: with
                   # both NaN the hardware returns y; an earlier b-first order disagreed on lanes 7 and 23)
                   3: np.where(np.isnan(a), False, np.where(np.isnan(b), True, a < b)),
                   7: np.where(np.isnan(a), False, np.where(np.isnan(b), True, a > b))}.get(code)
        if rel is None:
            raise Refused("fselect code %d: unmeasured" % code)
        self.write_u(toks[0], np.where(rel, xu, yu))

    def _exec_flag(self, tok):
        kind, idx = reg_parts(tok)
        if kind != "flag":
            raise Refused("exec on %s: only a FLAG register is modelled" % reg_name(tok))
        if not self.fwritten[idx[0]]:
            raise Refused("exec on FLAG%d before any compare wrote it" % idx[0])
        return self.F[:, idx[0]]

    def op582(self, off, size, toks):                             # if 1: 0, FLAGn, 1 -> active &= flag
        if (toks[0], toks[2]) != ("imm:0", "imm:1"):
            raise Refused("exec form %s: only the push-and-mask form is modelled" % " ".join(toks))
        cond = self._exec_flag(toks[1])
        on = self.ctr == 0
        self.ctr = np.where(on, np.where(cond, 0, 1), self.ctr + 1)
        self.active = self.ctr == 0

    def op577(self, off, size, toks):                             # pop n: counter -= n, floored at 0
        n = int(toks[1][4:])
        if toks[0] != "imm:2345052143616" or n not in (1, 2):
            raise Refused("exec form %s: only the one- and two-level pops are modelled" % " ".join(toks))
        if n == 2:
            self.wp("op577 pop 2 (the nesting-counter model: whole programs, the imageblock K loops)")
        self.ctr = np.maximum(0, self.ctr - n)
        self.active = self.ctr == 0

    def op579(self, off, size, toks):
        """while n, INVERTING: an active lane whose flag is TRUE (the inverted condition fails) goes to n and waits
        for its pop n; every other lane is unchanged (releasecheck._step_mask; ledger/compiler.toml straight-line).
        Whole programs only - and isa/g17-execution-backedge's reading of the back edge after it differs, though
        both readings give the same outputs on these kernels (the extra pass runs on no lane) - so tier wp."""
        n = int(toks[2][4:])
        if toks[0] != "imm:0" or n != 2:
            raise Refused("exec form %s: only the inverting while 2 is modelled" % " ".join(toks))
        cond = self._exec_flag(toks[1])
        self.wp("op579 while 2, inverting (the nesting-counter model: whole programs, the imageblock K loops)")
        self.ctr = np.where((self.ctr == 0) & cond, n, self.ctr)
        self.active = self.ctr == 0

    def op458(self, off, size, toks):                             # back edge: taken iff any lane is active
        if toks[0] != "imm:0":
            raise Refused("branch mode %s" % toks[0])
        if np.any(self.active):
            return ("jump", off + int(toks[1][4:]))

    def op462(self, off, size, toks):                             # skip: taken iff no lane is active
        if toks[0] != "imm:0":
            raise Refused("branch mode %s" % toks[0])
        if not np.any(self.active):
            return ("jump", off + int(toks[1][4:]))

    def op447(self, off, size, toks):                             # threadgroup barrier (lockstep satisfies it)
        if toks[0] != "imm:0" or toks[1] not in ("imm:276", "imm:154", "imm:532"):
            raise Refused("barrier form %s" % " ".join(toks))
        # 532 is the imageblock barrier (isa/g17-barrier-scopes.txt); one simdgroup in lockstep satisfies it, and the
        # imageblock handlers refuse more than one simdgroup (none was ever witnessed)
        if toks[1] == "imm:154":
            # threadgroup_barrier(mem_device) (isa/g17-barrier-scopes.txt): scopes 16, 528 and 1138 in place of 276
            # each carried a threadgroup-memory round trip (confounds2), 154 itself never ran around one
            self.wp("op447 scope 154 (mem_device) orders threadgroup memory as 276 does (other non-276 scopes "
                    "measured; 154 not)")
        if not np.all(self.active):
            raise Refused("barrier inside an exec region: unmodelled")
        # every barrier orders threadgroup memory; only the device scope (154, mem_device) also orders device memory
        # between the threadgroup's simdgroups (a threadgroup barrier is not a device fence)
        self.ep_tg += 1
        if toks[1] == "imm:154":
            self.ep_dev += 1

    def op14156(self, off, size, toks):                           # device fence: a no-op when programs run serially
        if (toks[0], toks[1]) != ("imm:0", "imm:186"):
            raise Refused("fence form %s" % " ".join(toks))

    def op14169(self, off, size, toks):                           # simd.shuffle_xor: dst, mod, src, mod, mask
        mask = int(toks[4][4:])
        if not 0 <= mask < 32:
            raise Refused("shuffle_xor mask %d (+0x%x) outside a 32-lane simdgroup" % (mask, off))
        if self.group % 32:
            raise Refused("shuffle_xor in a threadgroup that is not whole simdgroups (a partial simdgroup's lanes are "
                          "unmeasured)")
        r, half = self._where(toks[2])
        lane = np.arange(self.threads) % 32
        src = np.arange(self.threads) - lane + (lane ^ mask)         # the source lane of each lane
        act = self.active
        if np.any(act & ~self.active[src]):
            raise Refused("shuffle_xor (+0x%x) reads an inactive lane: unmeasured" % off)
        hs = HALVES[half]
        if not all(np.all(self.written[src[act], r, h]) for h in hs):
            raise Refused("shuffle_xor (+0x%x) reads r%d unwritten in its source lane" % (off, r))
        for h in hs:
            rel = self.poison[src[act], r, h]
            if np.any(rel) and np.any(_half_bits(self.R[src[act], r][rel], h) != 0):
                raise Refused("shuffle_xor (+0x%x) reads r%d released, nonzero, in its source lane" % (off, r))
        if any(np.any(self.unk[src[self.active], r, h]) for h in HALVES[half]):
            raise Refused("a shuffle reads r%d with unknown bits in its source lane" % r)
        v = self.R[src, r]
        if half == "lo":
            v = v & np.uint32(0xFFFF)
        elif half == "hi":
            v = v >> np.uint32(16)
        if int(toks[3][4:]) & MOD_RELEASE:                            # released in the reading lanes
            self.pending.append((act.copy(), r, hs))
        self.write_u(toks[0], v)

    def _tg_address(self, toks):
        """[.., width, base expr, 0, index, index mod, displacement, element bytes] -> (tg rows, byte addresses)."""
        import re
        mt = re.fullmatch(r"expr:bin\(op0,const\((\d+)\),2\)", toks[3])
        if not mt or int(mt.group(1)) % 2:
            raise Refused("threadgroup base operand %s is not a general register (bin(op0,const(2n),2))" % toks[3])
        base = self.read_u("reg:%d" % (R32 + int(mt.group(1)) // 2), 0).astype(np.int64)
        index = self.read_u(toks[5], int(toks[6][4:])).astype(np.int64)
        disp, elem = int(toks[7][4:]), int(toks[8][4:])
        if elem not in (2, 4):
            raise Refused("threadgroup access of %d-byte elements" % elem)
        addr = base + index * elem + disp
        act = self.active
        if np.any((addr[act] < 0) | (addr[act] + elem > TG_BYTES)):
            raise Refused("threadgroup access outside %d bytes" % TG_BYTES)
        if self.tg is None:
            self.tg = np.zeros((self.threads // self.group, TG_BYTES), np.uint8)
        return np.arange(self.threads) // self.group, addr, elem

    def op13288(self, off, size, toks):                           # store.tg
        value = self.read_u(toks[0], int(toks[1][4:])).astype(np.uint32)
        row, addr, elem = self._tg_address(toks)
        act = self.active
        self._same_value_or_refuse(row[act] * TG_BYTES + addr[act], value[act], elem, "threadgroup memory")
        if self.RACE_CHECK:
            flat = (row[act] * TG_BYTES + addr[act]).astype(np.int64)
            ab = (flat[:, None] + np.arange(elem)).ravel()
            newb = ((value[act].astype(np.uint32)[:, None] >> (8 * np.arange(elem, dtype=np.uint32))) & 0xFF).astype(
                np.uint8).ravel()
            self._track_write(("tg",), self.tg.size, ab, np.repeat(np.nonzero(act)[0], elem), self.tg.ravel()[ab], newb)
        for b in range(elem):
            self.tg[row[act], addr[act] + b] = ((value[act] >> np.uint32(8 * b)) & 0xFF).astype(np.uint8)

    def op12364(self, off, size, toks):                           # load.tg
        row, addr, elem = self._tg_address(toks)
        act = self.active
        if self.RACE_CHECK:
            self._track_read(("tg",), self.tg.size, (row[act] * TG_BYTES + addr[act]).astype(np.int64),
                             np.nonzero(act)[0], elem)
        v = np.zeros(self.threads, np.uint32)
        for b in range(elem):
            v[act] |= self.tg[row[act], addr[act] + b].astype(np.uint32) << np.uint32(8 * b)
        self.write_u(toks[0], v)

    # ---- the tensor path: vector tile loads/stores and the 16x16x16 MMA on register fragments
    def _vec_operands(self, toks):
        slot = self._binding(toks[3])
        index = self.read_u(toks[5], int(toks[6][4:]))
        disp, elem, mask = int(toks[7][4:]), int(toks[8][4:]), int(toks[9][4:])
        if mask not in (1, 3, 7, 15):
            raise Refused("vector access with component mask %d: only contiguous low masks are modelled" % mask)
        return slot, index, disp, elem, bin(mask).count("1")

    def op12674(self, off, size, toks):
        kind, regs = reg_parts(toks[0])
        slot, index, disp, elem, n = self._vec_operands(toks)
        nbytes = elem * n
        if kind != "r32" or len(regs) * 4 != nbytes:
            raise Refused("vector load of %d x %d bytes into %s" % (n, elem, reg_name(toks[0])))
        m = self.mem[slot]
        addr = self.pad + index.astype(np.int64) * elem + disp
        ok = (addr >= 0) & (addr + nbytes <= len(m))
        if self.RACE_CHECK:
            act = self.active
            self._track_read(("dev", slot), len(m), addr[act], np.nonzero(act)[0], nbytes)
        for w, r in enumerate(regs):
            v = np.zeros(self.threads, np.uint32)
            for b in range(4):
                v[ok] |= m[addr[ok] + 4 * w + b].astype(np.uint32) << np.uint32(8 * b)
            self.write_u("reg:%d" % (R32 + r), v)

    def op12656(self, off, size, toks):                           # int8 tensor load: one word, four bytes
        if toks[8] != "imm:1" or toks[9] != "imm:15":
            raise Refused("op12656 element %s mask %s: only four bytes, element 1, ran" % (toks[8], toks[9]))
        self.op12674(off, size, toks)

    def op3770(self, off, size, toks):                            # rint: dst, mode, src, mod (half to even)
        x = self._ftz(self.read_f(toks[2], int(toks[3][4:])))
        if np.any(np.isnan(x[self.active])):
            raise Refused("op3770 (+0x%x) of a NaN: unmeasured" % off)
        self.write_f32(toks[0], np.rint(x).astype(np.float32))

    def op9320(self, off, size, toks):                            # f2i: dst, mode, sign code, mode operand, src, mod
        code = int(toks[2][4:])
        if code not in (4, 5):
            raise Refused("op9320 sign code %d: unmeasured" % code)
        x = self._ftz(self.read_f(toks[4], int(toks[5][4:]))).astype(np.float64)
        xa = x[self.active]
        lo, hi = (0, 2 ** 32 - 1) if code == 4 else (-2 ** 31, 2 ** 31 - 1)
        if toks[3] == "imm:1":
            # MODE 1, the plain (uint)x / (int)x cast: truncate toward zero, saturate to the range, NaN -> 0
            # (isa/g17-rounding-results.json: 4,096 lanes each of Apple's casts on the GPU - fractions, negatives, past
            # 2^31 and 2^32, NaN, +-inf; rounding misses 326 / 653 lanes, wrapping 1,974; MM 25.186)
            out = np.zeros(self.threads, np.int64)
            out[self.active] = np.clip(np.trunc(np.where(np.isnan(xa), 0.0, xa)), lo, hi).astype(np.int64)
            return self.write_u(toks[0], (out & 0xFFFFFFFF).astype(np.uint32))
        # another mode operand (rounding, saturation) is unmeasured (MM 25.130.1), so only values it cannot reach are
        # admitted: integral, finite, in range - where every rounding and saturation mode agrees
        if np.any(~np.isfinite(xa)) or np.any(xa != np.trunc(xa)) or np.any((xa < lo) | (xa > hi)):
            raise Refused("op9320 (+0x%x) of a non-integral, non-finite or out-of-range value: its rounding and "
                          "saturation mode is unmeasured" % off)
        out = np.zeros(self.threads, np.int64)
        out[self.active] = xa.astype(np.int64)
        self.write_u(toks[0], (out & 0xFFFFFFFF).astype(np.uint32))

    FP8 = {97: "float8_e4m3fn", 98: "float8_e5m2"}

    def op17642(self, off, size, toks):
        """fp8 unpack: dst32, mode, format (97 e4m3fn, 98 e5m2), src16, mod. The 16-bit source holds two fp8 bytes;
        byte b becomes the bf16 in half b of the destination (tlower: bf16 slot s is fp8 byte s). Every finite fp8
        value is exact in bf16, so the conversion has no rounding; a nonfinite code is refused (its bf16 is
        unmeasured here)."""
        import ml_dtypes
        fmt = self.FP8.get(int(toks[2][4:]))
        if fmt is None:
            raise Refused("op17642 format %s: unmeasured" % toks[2])
        _r, half = self._where(toks[3])
        if half is None:
            raise Refused("op17642 source %s is not a 16-bit half" % reg_name(toks[3]))
        v = self.read_u(toks[3], int(toks[4][4:])).astype(np.uint16)
        pair = np.stack([v & 0xFF, v >> 8], 1).astype(np.uint8)                 # [threads, 2]: low byte, high byte
        f = pair.view(getattr(ml_dtypes, fmt)).astype(np.float32)
        if np.any(~np.isfinite(f[self.active])):
            raise Refused("op17642 (+0x%x) of a nonfinite fp8 code: its bf16 is unmeasured" % off)
        bf = (f.view(np.uint32) >> np.uint32(16)).astype(np.uint32)             # exact: the top half of the fp32
        self.write_u(toks[0], bf[:, 0] | (bf[:, 1] << np.uint32(16)))

    def op13618(self, off, size, toks):
        """fp8 pack: dst16, mode, format (97 e4m3fn, 98 e5m2), a32, mod, b32, mod -> byte 0 fp8(a), byte 1 fp8(b).
        Round to nearest even, NO saturation (an overflow keeps its sign: NaN in e4m3fn, infinity in e5m2), every
        NaN input the positive canonical NaN - the function recon section 138 measured over all 2^32 fp32 patterns
        (g17tensorcommonruntime.fp8_quantize; negative e4m3fn overflow 0xff, 471 of 471 on hardware)."""
        import ml_dtypes
        fmt = int(toks[2][4:])
        kind, nan = {97: (ml_dtypes.float8_e4m3fn, 0x7F), 98: (ml_dtypes.float8_e5m2, 0x7E)}.get(fmt, (None, None))
        if kind is None:
            raise Refused("op13618 format %d: unmeasured" % fmt)
        codes = []
        for i in (3, 5):
            _r, half = self._where(toks[i])
            if half is not None:
                raise Refused("op13618 from a 16-bit source %s: only the fp32 form is modelled" % reg_name(toks[i]))
            x = self.read_u(toks[i], self._imod(toks[i + 1])).view(np.float32)
            with np.errstate(over="ignore", invalid="ignore"):
                c = x.astype(kind).view(np.uint8).copy()
            c[np.isnan(x)] = nan
            codes.append(c.astype(np.uint32))
        self.write_u(toks[0], codes[0] | (codes[1] << np.uint32(8)))

    def op10286(self, off, size, toks):                           # d32 = zext(lo16 a) + zext(lo16 b)
        a = self.read_u(toks[2], self._imod(toks[3])).astype(np.uint64) & np.uint64(0xFFFF)
        b = self.read_u(toks[4], self._imod(toks[5])).astype(np.uint64) & np.uint64(0xFFFF)
        self.write_u(toks[0], a + b)

    # ---- the IMAGEBLOCK: the per-core tile memory the tensor path stages fragments through (agxforge/g17/ibstage.py)
    def _ib(self, toks, what):
        """[value/dst, op1, width, member, 1, coord, coord mod, dx, dy] -> (x, y, member). One tile a threadgroup,
        indexed by (x, y) = the coordinate register's low and high 16 bits, element_bytes a pixel."""
        elem = getattr(self, "ib_elem", None)
        if elem is None:
            raise Refused("%s with no imageblock declared in the ABI" % what)
        if toks[2] != "imm:18" or toks[4] != "imm:1" or (toks[7], toks[8]) != ("imm:0", "imm:0"):
            raise Refused("%s form %s: only the 32-bit, zero-offset form is measured" % (what, " ".join(toks[2:])))
        if self.group > 32 or self.threads != self.group:
            raise Refused("%s with more than one simdgroup or threadgroup: never witnessed" % what)
        member = int(toks[3][4:])
        if member % 4 or member + 4 > elem:
            raise Refused("%s member %d outside a %d-byte element" % (what, member, elem))
        c = self.read_u(toks[5], int(toks[6][4:])).astype(np.int64)
        x, y = c & 0xFFFF, c >> 16
        act = self.active
        if np.any((x[act] >= self.tg3[0]) | (y[act] >= self.tg3[1])):
            raise Refused("%s coordinate outside the %dx%d tile" % (what, self.tg3[0], self.tg3[1]))
        if self.ib is None:
            self.ib = np.zeros((self.tg3[1], self.tg3[0], elem), np.uint8)
            self.ib_written = np.zeros((self.tg3[1], self.tg3[0], elem // 4), bool)
            self.ib_owner = np.full((self.tg3[1], self.tg3[0], elem // 4), -1, np.int64)   # the lane of a bit-31-clear write
        return x, y, member

    def op13075(self, off, size, toks):                           # imageblock store, 32-bit
        op1 = int(toks[1][4:])
        value = self.read_u(toks[0], op1 & (MOD_RELEASE | MOD_KEEP)).astype(np.uint32)   # bit 4: the value's release
        x, y, member = self._ib(toks, "op13075")
        if not getattr(self, "ib_declared", True):
            return                                                # undeclared: a store lands nowhere a read sees
        act = self.active
        # OPERAND 1 BIT 31 selects COORDINATE-ADDRESSED staging (Set C round 9's 2 x 2, results/g17-tensor-imageblock-
        # v8/setc-round9-receipt.json: with it set a neighbour read passes on all 32 lanes; clear, the read FAILS to
        # see the neighbour). A clear-bit store is admitted only where each lane addresses its OWN pixel - the ibfragment
        # kernels, bit-exact on hardware - in tier wp, and a read of that word from another lane is refused below.
        own = (x == self.sr["local_x"]) & (y == self.sr["local_y"])
        if not op1 & (1 << 31):
            if np.any(act & ~own):
                raise Refused("op13075 (+0x%x) without operand 1 bit 31 at a coordinate that is not the lane's own "
                              "pixel: that staging is not coordinate-addressed (round 9)" % off)
            self.wp("op13075 without operand 1 bit 31, each lane at its own pixel (whole programs: ibfragment)")
        self._same_value_or_refuse(y[act] * 65536 + x[act], value[act], 4, "the imageblock")
        w = np.ascontiguousarray(value[act]).view(np.uint8).reshape(-1, 4)
        for b in range(4):
            self.ib[y[act], x[act], member + b] = w[:, b]
        self.ib_written[y[act], x[act], member // 4] = True
        self.ib_owner[y[act], x[act], member // 4] = -1 if op1 & (1 << 31) else np.arange(self.threads)[act]

    def op12151(self, off, size, toks):                           # imageblock load, 32-bit
        x, y, member = self._ib(toks, "op12151")
        act = self.active
        out = np.zeros(self.threads, np.uint32)
        if getattr(self, "ib_declared", True):
            if not np.all(self.ib_written[y[act], x[act], member // 4]):
                raise Refused("op12151 (+0x%x) reads an imageblock word no lane wrote: unmeasured" % off)
            owner = self.ib_owner[y[act], x[act], member // 4]
            if np.any((owner >= 0) & (owner != np.arange(self.threads)[act])):
                raise Refused("op12151 (+0x%x) reads another lane's word that a store without operand 1 bit 31 "
                              "wrote: not coordinate-addressed (round 9)" % off)
            out[act] = np.ascontiguousarray(self.ib[y[act], x[act], member:member + 4]).view(np.uint32).ravel()
        else:
            self.wp("an undeclared imageblock reads zero (isolated for the unsized control; the sized read in "
                    "whole programs)")
        self.write_u(toks[0], out)

    def op1272(self, off, size, toks):
        """HARDWARE EXP2: dst, mode, src, mod. op1272 is a deterministic function within one ulp of 2^x but NOT a
        known one - no closed form, and its exponent-shift law fails (MM 25.144.2), so no small table covers it.
        It is emulated from a SPARSE MEASURED TABLE (isa/g17-exp2-op1272-table.npz, tools/g17exp2oracle.py): the
        GPU's own output for exactly the inputs the programs we check feed it. An input the table does not hold is
        refused - or, in COLLECT mode (EXP2_COLLECT a set), recorded, and the run marked so it can never count as
        agreeing (Machine.exp2_placeholder)."""
        x = self.read_f(toks[2], int(toks[3][4:]))
        bits = x.view(np.uint32)
        act = self.active
        keys, vals = exp2_table()
        want = bits[act]
        pos = np.searchsorted(keys, want)
        pos_c = np.minimum(pos, max(len(keys) - 1, 0))
        hit = (len(keys) > 0) & (keys[pos_c] == want) if len(keys) else np.zeros(want.size, bool)
        out = np.zeros(self.threads, np.uint32)
        got = np.zeros(want.size, np.uint32)
        got[hit] = vals[pos_c[hit]]
        if not np.all(hit):
            if EXP2_COLLECT is None:
                raise Refused("op1272 (+0x%x): %d input(s) not in the measured exp2 table (tools/g17exp2oracle.py)"
                              % (off, int((~hit).sum())))
            EXP2_COLLECT.update(int(v) for v in np.unique(want[~hit]))
            self.exp2_placeholder = True
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                got[~hit] = np.exp2(want[~hit].view(np.float32).astype(np.float64)).astype(np.float32).view(np.uint32)
        out[act] = got
        self.write_u(toks[0], out)

    def op17202(self, off, size, toks):                           # word store at a word index
        self._store(toks, 4)

    def _mma_int8(self, off, toks, with_c):
        """op10384 / op10385: int8 x int8 -> int32 on a whole simdgroup, one 16-product issue, exact sum."""
        if self.group % 32:
            raise Refused("tensor.mac int8 (+0x%x) in a threadgroup that is not whole simdgroups" % off)
        sg_act = self.active.reshape(-1, 32)
        if np.any(sg_act.any(1) & ~sg_act.all(1)):
            raise Refused("tensor.mac int8 (+0x%x) in a partially active simdgroup: unmodelled" % off)
        d_code, a_code, b_code = int(toks[2][4:]), int(toks[5][4:]), int(toks[8][4:])
        if d_code not in (9, 41) or a_code not in (11, 75) or b_code not in (11, 75):
            raise Refused("tensor.mac int8 (+0x%x) type codes D %d A %d B %d: unmodelled" % (off, d_code, a_code, b_code))
        nsg = self.threads // 32

        def frag8(tok, mod, signed, perm):
            kind, regs = reg_parts(tok)
            if kind != "r32" or len(regs) != 2:
                raise Refused("int8 tensor operand %s is not a two-register fragment" % reg_name(tok))
            words = np.stack(self._read_tuple(regs, mod), 1)                     # [threads, 2]
            b = words.view(np.uint8).reshape(self.threads, 8)                    # byte j = register j / 4, byte j % 4
            # a gather through the inverse permutation (t[:, perm] = v is the same tile, and slower as a scatter)
            v = (b.view(np.int8) if signed else b).reshape(nsg, 256)
            return np.take(v, _inv_perm(perm), axis=1).astype(np.float64).reshape(nsg, 16, 16)
        A = frag8(toks[3], int(toks[4][4:]), a_code == 75, PERM_AD)
        B = frag8(toks[6], int(toks[7][4:]), b_code == 75, PERM_B)
        # the exact issue sum, through float64 matmul (MM 25.197, speed): |a b| <= 255 x 255 and 16 terms stay below
        # 2^21, far inside float64's 53 exact bits, so every partial sum is an exact integer
        acc = np.rint(np.matmul(A, B)).astype(np.int64)
        if with_c:
            if toks[11] != "imm:9":
                raise Refused("tensor.mac int8 (+0x%x) C type code %s" % (off, toks[11]))
            kind, regs = reg_parts(toks[9])
            cw = np.stack(self._read_tuple(regs, int(toks[10][4:])), 1).view(np.int32).astype(np.int64)
            ct = np.empty((nsg, 256), np.int64)
            ct[:, PERM_AD] = cw.reshape(nsg, 256)
            acc = acc + ct.reshape(nsg, 16, 16)
        if d_code == 41:
            acc = np.clip(acc, -2 ** 31, 2 ** 31 - 1)
        acc = acc.astype(np.int32)                              # wraps modulo 2^32, as the accumulator does
        kind, regs = reg_parts(toks[0])
        if kind != "r32" or len(regs) != 8:
            raise Refused("tensor.mac int8 destination %s" % reg_name(toks[0]))
        flat = np.take(acc.reshape(-1, 256), PERM_AD, axis=1).reshape(self.threads, 8).view(np.uint32)
        for sl, r in enumerate(regs):
            self.write_u("reg:%d" % (R32 + r), flat[:, sl])

    def op10384(self, off, size, toks):
        self._mma_int8(off, toks, with_c=True)

    def op10385(self, off, size, toks):
        self._mma_int8(off, toks, with_c=False)

    def op17257(self, off, size, toks):
        kind, regs = reg_parts(toks[0])
        slot, index, disp, elem, n = self._vec_operands(toks)
        if kind != "r32" or len(regs) * 4 != elem * n:
            raise Refused("vector store of %d x %d bytes from %s" % (n, elem, reg_name(toks[0])))
        vals = self._read_tuple(regs, int(toks[1][4:]))
        act = self.active
        m = self.mem[slot]
        addr = (self.pad + index.astype(np.int64) * elem + disp)[act]
        if np.any(addr < 0) or np.any(addr + 4 * len(regs) > len(m)):
            raise Refused("vector store outside buffer %d" % slot)
        lanes = np.nonzero(act)[0]
        for w, v in enumerate(vals):
            a = addr + 4 * w
            # two lanes storing different words to one address: the winner is unmeasured (as store() refuses)
            self._same_value_or_refuse(a, v[act], 4, "buffer %d" % slot)
            if self.RACE_CHECK:
                ab = (a[:, None] + np.arange(4)).ravel()
                newb = ((v[act].astype(np.uint32)[:, None] >> (8 * np.arange(4, dtype=np.uint32))) & 0xFF).astype(
                    np.uint8).ravel()
                self._track_write(("dev", slot), len(m), ab, np.repeat(lanes, 4), m[ab], newb)
            for b in range(4):
                m[a + b] = ((v[act] >> np.uint32(8 * b)) & 0xFF).astype(np.uint8)

    def _read_tuple(self, regs, mod):
        """A tuple operand: every register read before any is released (a tuple's modifier covers the whole tuple)."""
        return [self.read_u("reg:%d" % (R32 + r), mod) for r in regs]

    def _frag(self, tok, mod, dtype_code, perm):
        """A register fragment -> tiles [simdgroups, 16, 16] (float32): fp16 slots packed two per register (slot s in
        register s // 2, low half first), fp32 slots one per register."""
        kind, regs = reg_parts(tok)
        nsg = self.threads // 32
        if dtype_code in TENSOR_16BIT and len(regs) == 4:    # fp16 / bf16 A and B, two per register
            words = np.stack(self._read_tuple(regs, mod), 1)
            halves = np.stack([words & np.uint32(0xFFFF), words >> np.uint32(16)], 2).reshape(self.threads, 8)
            if TENSOR_16BIT[dtype_code] == "bf16":           # bf16: the top half of an fp32, exactly
                vals = (halves.astype(np.uint32) << np.uint32(16)).view(np.float32)
            else:
                vals = halves.astype(np.uint16).view(np.float16).astype(np.float32)
        elif dtype_code == 1 and len(regs) == 8:             # fp32 C / D, and fp32 A / B: one slot a register
            vals = np.stack(self._read_tuple(regs, mod), 1)
            vals = vals.view(np.float32)
        else:
            raise Refused("tensor operand %s with dtype code %d is not a modelled fragment" % (reg_name(tok), dtype_code))
        # a gather through the inverse permutation: the same tile as tiles[:, perm] = vals, without the slow scatter
        return np.take(vals.reshape(nsg, 256), _inv_perm(perm), axis=1).astype(np.float32).reshape(nsg, 16, 16)

    def _mma(self, off, toks, with_c, a_fp32=False, b_fp32=False):
        """Cooperative per simdgroup: a fully active simdgroup multiplies, a fully masked one does not (its D is left
        as it was by write_u's mask); a simdgroup with some lanes masked is refused."""
        if self.group % 32:
            raise Refused("tensor.mac (+0x%x) in a threadgroup that is not whole simdgroups" % off)
        sg_act = self.active.reshape(-1, 32)
        if np.any(sg_act.any(1) & ~sg_act.all(1)):
            raise Refused("tensor.mac (+0x%x) in a partially active simdgroup: unmodelled" % off)
        a_code, b_code = int(toks[5][4:]), int(toks[8][4:])
        # type codes (agxforge/g17/mmaenc.py): 1 fp32, 2 fp16, 3 bf16, + 32 transposed; D code 1 is fp32
        # the OPCODE fixes which operands are fp32 (the MM operand table: 5106/7 16x16, 5104/5 16 x fp32 B, 5100/1
        # fp32 A x 16, 5098/9 fp32 x fp32); the type codes must agree with it
        a_ok = (a_code & ~32) == 1 if a_fp32 else (a_code & ~32) in TENSOR_16BIT
        b_ok = (b_code & ~32) == 1 if b_fp32 else (b_code & ~32) in TENSOR_16BIT
        if toks[2] != "imm:1" or not a_ok or not b_ok:
            raise Refused("tensor.mac (+0x%x) type codes %s: unmodelled" % (off, toks[2:9:3]))
        cut = lambda t: (np.ascontiguousarray(t).view(np.uint32) & np.uint32(0xFFFFE000)).view(np.float32)
        A = self._frag(toks[3], int(toks[4][4:]), a_code & ~32, PERM_AT if a_code & 32 else PERM_AD)
        if a_fp32:                                            # truncated to 10 mantissa bits: products stay exact
            A = cut(A)
        B = self._frag(toks[6], int(toks[7][4:]), b_code & ~32, PERM_BT if b_code & 32 else PERM_B)
        if b_fp32:
            B = cut(B)
        # [sg, m, k, n] products are exact in f32; P_i = p_2i + p_2i+1, computed from the even and odd k halves directly
        # (the same f32 operations in the same order as forming every product first: MM 25.197, speed)
        P = A[:, :, 0::2, None] * B[:, None, 0::2, :]
        P += A[:, :, 1::2, None] * B[:, None, 1::2, :]
        Q = P[:, :, 0:4, :] + P[:, :, 4:8, :]                 # Q_j = P_j + P_j+4
        if with_c:
            if toks[11] != "imm:1":
                raise Refused("tensor.mac (+0x%x) C type code %s" % (off, toks[11]))
            acc = self._frag(toks[9], int(toks[10][4:]), 1, PERM_AD)
            for j in range(4):
                acc = acc + Q[:, :, j, :]
        else:
            acc = Q[:, :, 0, :].copy()
            for j in (1, 2, 3):
                acc = acc + Q[:, :, j, :]
        kind, regs = reg_parts(toks[0])
        if kind != "r32" or len(regs) != 8:
            raise Refused("tensor.mac destination %s" % reg_name(toks[0]))
        flat = np.take(acc.reshape(-1, 256), PERM_AD, axis=1).reshape(self.threads, 8).astype(np.float32)
        for sl, r in enumerate(regs):
            self.write_u("reg:%d" % (R32 + r), flat[:, sl].view(np.uint32))

    def op5107(self, off, size, toks):
        self._mma(off, toks, with_c=False)

    def op5106(self, off, size, toks):
        self._mma(off, toks, with_c=True)

    def op5101(self, off, size, toks):                            # D = A(fp32) B(fp16), no C
        self._mma(off, toks, with_c=False, a_fp32=True)

    def op5100(self, off, size, toks):                            # D = A(fp32) B(fp16) + C
        self._mma(off, toks, with_c=True, a_fp32=True)

    def op5104(self, off, size, toks):                            # D = A(fp16) B(fp32) + C
        self._mma(off, toks, with_c=True, b_fp32=True)

    def op5105(self, off, size, toks):                            # D = A(fp16) B(fp32)
        self._mma(off, toks, with_c=False, b_fp32=True)

    def op5098(self, off, size, toks):                            # D = A(fp32) B(fp32) + C
        self._mma(off, toks, with_c=True, a_fp32=True, b_fp32=True)

    def op5099(self, off, size, toks):                            # D = A(fp32) B(fp32)
        self._mma(off, toks, with_c=False, a_fp32=True, b_fp32=True)

    def op3658(self, off, size, toks):                            # recip: dst, mode, src, mod
        """The GPU's own answer from the sparse measured table (isa/g17-recip-op3658-table.npz,
        tools/g17exp2oracle.py) where it holds the input; otherwise the two admissible roundings, bracketed, over
        the probed normal range. In COLLECT mode a missing input is recorded and the run can never agree."""
        x = self.read_f(toks[2], int(toks[3][4:]))
        a = self.active
        keys, vals = recip_table()
        hit, tv = _lookup(keys, vals, x[a].view(np.uint32))
        if np.all(hit):
            out = np.zeros(self.threads, np.uint32)
            out[a] = tv
            self.write_u(toks[0], out)
            return
        if RECIP_COLLECT is not None:
            RECIP_COLLECT.update(int(v) for v in np.unique(x[a].view(np.uint32)[~hit]))
            self.exp2_placeholder = True
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                ph = (1.0 / x[a].astype(np.float64)).astype(np.float32).view(np.uint32)
            out = np.zeros(self.threads, np.uint32)
            out[a] = np.where(hit, tv, ph)
            self.write_u(toks[0], out)
            return
        xa = x[a].astype(np.float64)
        tiny = np.float64(np.finfo(np.float32).tiny)
        with np.errstate(divide="ignore"):
            outside = ~np.isfinite(xa) | (np.abs(xa) < tiny) | (np.abs(1.0 / xa) < tiny)
        if np.any(outside & ~hit):
            raise Refused("recip (+0x%x) of zero, a denormal, an infinity or NaN, or with a denormal result: outside "
                          "the probed normal range" % off)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            f = (1.0 / xa).astype(np.float32)              # the nearest float to 1/x; which side of 1/x is it on?
            fx = f.astype(np.float64) * xa                 # exact in float64 (24 x 24 bits)
        gt = np.where(xa > 0, fx > 1.0, fx < 1.0)          # f > 1/x
        lt = np.where(xa > 0, fx < 1.0, fx > 1.0)          # f < 1/x
        if self.choice.get(3658, "down") == "down":        # the largest float <= 1/x
            f = np.where(gt, np.nextafter(f, np.float32(-np.inf)), f)
        else:                                              # the smallest float >= 1/x
            f = np.where(lt, np.nextafter(f, np.float32(np.inf)), f)
        f = np.where(hit, tv.view(np.float32), f)          # measured inputs take the GPU's answer
        out = np.zeros(self.threads, np.float32)
        out[a] = f
        self.write_f32(toks[0], out)

    def op3850(self, off, size, toks):                            # rsqrt seed: dst, mode, src, mod
        import g17decodestep as DS
        x = self.read_f(toks[2], int(toks[3][4:]))
        xa = x[self.active]
        if np.any(~np.isfinite(xa)) or np.any(xa < np.finfo(np.float32).tiny):
            raise Refused("rsqrt seed (+0x%x) of a non-positive, denormal or non-finite value: the table covers "
                          "positive normal inputs only" % off)
        out = np.zeros(self.threads, np.float32)
        out[self.active] = DS.rsqrt_seed(xa)
        self.write_f32(toks[0], out)


EXP2_TABLE_PATH = HERE.parent / "isa" / "g17-exp2-op1272-table.npz"
RECIP_TABLE_PATH = HERE.parent / "isa" / "g17-recip-op3658-table.npz"
EXP2_COLLECT = None          # a set: collect op1272 inputs missing from the table instead of refusing (the oracle)
RECIP_COLLECT = None         # the same for op3658 (hardware reciprocal)
_EXP2 = {}


def _measured(path, key):
    """(sorted uint32 inputs, uint32 outputs) of a measured-function table, or empty arrays."""
    if key not in _EXP2:
        if path.exists():
            z = np.load(path)
            _EXP2[key] = (z["x"].astype(np.uint32), z["y"].astype(np.uint32))
        else:
            _EXP2[key] = (np.zeros(0, np.uint32), np.zeros(0, np.uint32))
    return _EXP2[key]


def recip_table():
    return _measured(RECIP_TABLE_PATH, "recip")


def _lookup(keys, vals, want):
    """(hit mask, values) of `want` in a sorted table."""
    if not len(keys):
        return np.zeros(want.size, bool), np.zeros(want.size, np.uint32)
    pos = np.minimum(np.searchsorted(keys, want), len(keys) - 1)
    hit = keys[pos] == want
    got = np.zeros(want.size, np.uint32)
    got[hit] = vals[pos[hit]]
    return hit, got


def _exp2_on():
    """Replay hardware-exp2 programs once a measured table exists (or while the oracle collects)."""
    return EXP2_COLLECT is not None or EXP2_TABLE_PATH.exists()


def _recip_on():
    return RECIP_COLLECT is not None or RECIP_TABLE_PATH.exists()


def exp2_table():
    """(sorted uint32 inputs, uint32 outputs) of the measured op1272 table, or empty arrays when there is none."""
    return _measured(EXP2_TABLE_PATH, "exp2")


def abi_table(abi):
    """The binding table from the compiler's ABI (g17cc Program.abi()["bindings"], or a manifest's copy of it): the
    binding at table offset o is the load operand bin(op0,const(2 o),8) (the bundles' dense table, o = 2 i, is the
    same rule)."""
    return {2 * int(b["offset"]): int(b["index"]) for b in abi["bindings"]}


def published_uniforms(constant_program):
    """The uniform halfwords a constant program writes: every instruction whose FIRST operand is a uniform expression
    (agxforge.g17.constprog.uniform_writes' rule), 16 bits for width 2, 32 for width 4."""
    out = set()
    for _o, _s, _op, toks in decode(bytes(constant_program) + END):
        mt = toks and Machine.UNIFORM.fullmatch(toks[0])
        if mt:
            k, w = int(mt.group(1)), int(mt.group(2))
            out.update(range(k, k + w // 2))
        elif toks and toks[0].startswith("expr:bin(op0,"):
            raise Refused("constant program writes uniform operand %s of an unmodelled width" % toks[0])
    return out


def run_program(code, buffers, grid, group, table=None, tier="strict", pool=None, published=(), constant=None):
    """THE GENERAL ENTRY: run any program over the caller's buffers, with no harness layout assumed.

    buffers {index: bytes}: each is the WHOLE buffer the binding points at (byte 0 is the binding's address; a device
    load past its end reads zero, as measured). grid: the dispatch in threads, an int or (x, y, z); group: the
    threadgroup, an int or (x, y, z). table: {K: index} (abi_table), or None for the dense slot-order table.
    Programs using a BRACKETED instruction run under every admissible choice and must agree. Returns
    ({index: bytes after the run}, the machine)."""
    code = bytes(code)
    ops = {op for _o, _s, op, _t in decode(code)}
    bracket = [op for op in BRACKETED if op in ops]
    outs, m = [], None
    for choice in ([{}] if not bracket else [{op: ch for op in bracket} for ch in ("down", "up")]):
        mem = {int(k): np.frombuffer(bytearray(v), np.uint8).copy() for k, v in buffers.items()}
        uniforms = {}
        if constant is not None and published:
            # THE CONSTANT PROGRAM (MM 25.179): Apple's per-dispatch program, run once (one thread) over the same
            # buffers before main; what it writes to the uniform file is what main reads as published uniforms
            c = Machine(bytes(constant) + END, mem, 1, 1, choice, tier, views=True, table=table)
            c.pool = pool
            c.wp("the constant program run as one thread before main")
            c.run()
            uniforms = dict(c.uniforms)
        m = Machine(code, mem, grid, group, choice, tier, views=True, table=table)
        m.pool, m.published, m.uniforms = pool, set(published), uniforms
        m.run()
        outs.append({k: bytes(v) for k, v in mem.items()})
    if any(o != outs[0] for o in outs):
        raise Refused("the output depends on the unmeasured rounding of %s" % ", ".join("op%d" % op for op in bracket))
    return outs[0], m


def run_bundle(bundle, threads, group, base=1, tier="strict"):
    """Emulate one bundle as g17bundlerun would dispatch it; returns buffer c's c.f32-sized window (bytes)."""
    d = Path(bundle)
    a, b, c = ((d / n).read_bytes() for n in ("a.f16", "b.f16", "c.f32"))
    bufs = {}
    for data, slot in zip((a, b, c), (1, 2, 3) if base == 1 else (1, 2, 0)):
        buf = bytearray([FILL]) * (PAD + len(data) + PAD)
        buf[PAD:PAD + len(data)] = data
        bufs[slot] = buf
    code = (d / "program.bin").read_bytes()
    pool = container_pool(d)
    ib_elem, ib_declared = bundle_imageblock(d)
    cslot = 3 if base == 1 else 0
    ops = {op for _o, _s, op, _t in decode(code)}
    bracket = [op for op in BRACKETED if op in ops]
    runs = [{}] if not bracket else [{op: ch for op in bracket} for ch in ("down", "up")]
    outs = []
    for choice in runs:
        m = Machine(code, {k: bytearray(v) for k, v in bufs.items()}, threads, group, choice, tier)
        m.pool, m.ib_elem, m.ib_declared = pool, ib_elem, ib_declared
        m.run()
        if getattr(m, "exp2_placeholder", False):
            raise Refused("op1272 inputs collected for the exp2 oracle, not measured: this run proves nothing")
        outs.append(bytes(m.mem[cslot][PAD:PAD + len(c)]))
    if len(set(outs)) != 1:
        raise Refused("the output depends on the unmeasured rounding of %s: the two admissible results give "
                      "different outputs" % ", ".join("op%d" % op for op in bracket))
    return outs[0], m


def bundle_imageblock(bundle):
    """(element_bytes or None, declared): the imageblock a bundle carries. The element size is the ABI's
    (manifest.json abi.imageblock); whether it is DECLARED is the image's own per-kernel slot 19 in scan.o
    (imageblock.pk_declared), not the manifest - v10's undeclared_sized arm keeps a manifest that says declared."""
    import json
    d = Path(bundle)
    try:
        ib = (json.loads((d / "manifest.json").read_text()).get("abi") or {}).get("imageblock")
    except (OSError, ValueError):
        ib = None
    if not ib or not ib.get("element_bytes"):
        return None, False
    declared = True
    obj = d / "scan.o"
    if obj.exists():
        from agxforge.g17 import machobj, imageblock
        o = obj.read_bytes()
        sects, _syms = machobj.parse(o)
        if "__GPU_METADATA,__compute" in sects:
            off, size = sects["__GPU_METADATA,__compute"]
            declared = imageblock.pk_declared(o[off:off + size])
    return int(ib["element_bytes"]), declared


def container_pool(bundle):
    """The slot-13 constant pool the bundle's own container carries (scan.o, __GPU_METADATA,__compute), as bytes, or
    None when there is no object or no pool. Read from the delivered bytes, not from the manifest's contract."""
    obj = Path(bundle) / "scan.o"
    if not obj.exists():
        return None
    from agxforge.g17 import machobj, cooperativemetadata
    o = obj.read_bytes()
    sects, _syms = machobj.parse(o)
    if "__GPU_METADATA,__compute" not in sects:
        return None
    off, size = sects["__GPU_METADATA,__compute"]
    try:
        return bytes(cooperativemetadata._witness_pool(o[off:off + size]))
    except (struct.error, TypeError, IndexError):
        return None


def run_job(bundle, threads, group, base=1, tier="strict"):
    """run_bundle for a worker process: (output bytes, whole-program semantics admitted, seconds). A refusal raises."""
    import time
    t = time.time()
    out, m = run_bundle(bundle, threads, group, base, tier)
    return out, dict(m.admitted), time.time() - t


def run(bundle, buffers, grid, tier="strict"):
    """Run a bundle's program.bin over caller-supplied buffers: buffers = {slot: bytes}, grid = {"threadgroups": n,
    "threads_per_group": g}. Each buffer is placed at byte 128 of a 0xA5-filled buffer, as g17bundlerun places its
    inputs. Returns {slot: bytes} after the run (each buffer's own window). Programs using a BRACKETED instruction
    run under every admissible choice and must agree, as in run_bundle."""
    d = Path(bundle)
    code = (d / "program.bin").read_bytes()
    bufs = {}
    for slot, data in buffers.items():
        b = bytearray([FILL]) * (PAD + len(data) + PAD)
        b[PAD:PAD + len(data)] = data
        bufs[slot] = b
    tpg = int(grid["threads_per_group"])
    threads = int(grid["threadgroups"]) * tpg
    ops = {op for _o, _s, op, _t in decode(code)}
    bracket = [op for op in BRACKETED if op in ops]
    outs = []
    for choice in ([{}] if not bracket else [{op: ch for op in bracket} for ch in ("down", "up")]):
        m = Machine(code, {k: bytearray(v) for k, v in bufs.items()}, threads, tpg, choice, tier).run()
        outs.append({s_: bytes(m.mem[s_][PAD:PAD + len(buffers[s_])]) for s_ in buffers})
    if any(o != outs[0] for o in outs):
        raise Refused("the output depends on the unmeasured rounding of %s" % ", ".join("op%d" % op for op in bracket))
    return outs[0]


def _harness(code, arrays, out_slot, tier, threads=32):
    """A receipt tool's dispatch on the CPU: `arrays` = {slot: u32 words} (64 x 64 each, as the tools bind them), one
    threadgroup of `threads`; returns (the output slot's first `threads` words, how many of its later words are no
    longer the 0xDEADBEEF sentinel, the machine)."""
    bufs = {}
    for slot, arr in arrays.items():
        b = bytearray([FILL]) * (PAD + arr.nbytes + PAD)
        b[PAD:PAD + arr.nbytes] = arr.tobytes()
        bufs[slot] = b
    ops = {op for _o, _s, op, _t in decode(code)}
    bracket = [op for op in BRACKETED if op in ops]
    outs = []
    for choice in ([{}] if not bracket else [{op: ch for op in bracket} for ch in ("down", "up")]):
        m = Machine(code, {k: bytearray(v) for k, v in bufs.items()}, threads, threads, choice, tier).run()
        c = np.frombuffer(bytes(m.mem[out_slot][PAD:PAD + arrays[out_slot].nbytes]), np.uint32)
        outs.append((c[:threads].tolist(), int((c[threads:] != 0xDEADBEEF).sum())))
    if any(o != outs[0] for o in outs):
        raise Refused("the output depends on the unmeasured rounding of %s" % bracket)
    return outs[0][0], outs[0][1], m


def _formlower_run(code, words, tier):
    """tools/g17formlowerrun.py's dispatch harness on the CPU: the program declares buffers 1 and 2; buffer 1 holds
    the caller's u32 words (then zeros), buffer 2 is 0xDEADBEEF-filled; 64 x 64 words each; one threadgroup of 32
    threads; the output is buffer 2's first 32 words."""
    n = 64 * 64
    B = np.zeros(n, np.uint32)
    B[:len(words)] = np.asarray(words, np.uint64).astype(np.uint32)
    out, _beyond, m = _harness(code, {1: B, 2: np.full(n, 0xDEADBEEF, np.uint32)}, 2, tier)
    return out, m


def receipts(root=None, tier="strict"):
    """GOAL ITEM 4 (MM 25.145): replay every hardware execution receipt that records a WHOLE program with its inputs
    and observed outputs, and compare g17emu's output with what the GPU returned, word for word. Families:
    isa/g17-execution-lanes (32 lanes, per-lane inputs), -vecload (the tuple loads), -rtloops (runtime trip counts)
    and -pairs (two-instruction chains), all dispatched by tools/g17formlowerrun.py. Returns rows (family, id, verdict, detail): agree, DISAGREE, or
    refused (g17emu has no measured semantics for something in it - not covered)."""
    import json
    root = Path(root or HERE.parent)
    rows = []
    families = (("lanes", lambda p: [w for lane in p["cases"] for w in lane]),       # 32 lanes, per-lane words
                ("vecload", lambda p: [w for lane in p["cases"] for w in lane]),     # the op12691 / op12709 loads
                ("rtloops", lambda p: list(p["n"])))                                 # runtime trip counts per lane
    for fam, words_of in families:
        plan = {r["id"]: r for r in json.loads((root / ("isa/g17-execution-%s.json" % fam)).read_text())}
        for r in json.loads((root / ("isa/g17-execution-%s-results.json" % fam)).read_text()):
            p = plan.get(r["id"])
            if not p or not p.get("program") or r.get("cb_status") != 0 or not isinstance(r.get("values"), list):
                continue
            rows.append(_replay(fam, r["id"], bytes.fromhex(p["program"]), words_of(p) or [0], r["values"], tier))
    import g17pairhazard as PH
    for r in json.loads((root / "isa/g17-execution-pairs-results.json").read_text())["rows"]:
        if r.get("cb_status") != 0 or not r.get("program"):
            continue
        a = r["pair"][0]
        ins = PH.FLOAT_IN if a in PH.FLOATS else PH.INT_IN
        rows.append(_replay("pairs", "%s+%s" % tuple(r["pair"]), bytes.fromhex(r["program"]), ins, r["values"], tier))
    # The same harness, from tools that record their inputs in the result row or fix them in code.
    for r in json.loads((root / "isa/g17-execution-cvt-sign-results.json").read_text()):
        if r.get("cb_status") == 0:
            rows.append(_replay("cvtsign", r["id"], bytes.fromhex(r["program"]), r["inputs"], r["values"], tier))
    for r in json.loads((root / "isa/g17-execution-release-mask-results.json").read_text())["arms"]:
        if r.get("cb_status") == 0:                   # tools/g17releasemask.py: inputs all zero; it also counts
            rows.append(_replay("relmask", r["id"], bytes.fromhex(r["program"]), [0] * 32, r["values"], tier,
                                beyond=r["region_words_written"]))      # words written past the first 32
    for r in json.loads((root / "isa/g17-execution-indirect-load-results.json").read_text())["arms"]:
        if all(st == 0 for st in r["status"]):          # tools/g17indirectload.py: word i = (7i + 3) % 64, 3 runs
            rows.append(_replay("indload", r["arm"], bytes.fromhex(r["program"]), [(7 * i + 3) % 64 for i in range(64)],
                                r["values"], tier, runs=True))
    plan = {r["id"]: r for r in json.loads((root / "isa/g17-execution-formwidths.json").read_text())["records"]}
    n = 64 * 64
    pattern = {"int": ((np.arange(n, dtype=np.uint64) * 7 + 3) % 4096).astype(np.uint32).tolist(),     # :190
               "float": (1.0 + 0.5 * np.arange(n, dtype=np.float32)).view(np.uint32).tolist()}
    for r in json.loads((root / "isa/g17-execution-formwidths-results.json").read_text())["records"]:
        for pat, arms in sorted(r["cases"].items()):
            for which, key in (("default", "default_code"), ("form", "form_code")):
                got = arms.get(which) or {}
                if got.get("status") == 0 and isinstance(got.get("out"), list):
                    rows.append(_replay("widths", "%s %s %s" % (r["id"], pat, which), bytes.fromhex(plan[r["id"]][key]),
                                        pattern[pat], got["out"], tier, beyond=got.get("untouched")))
    # Three front-end receipts bind slots 0, 1, 2 (input, 0xDEADBEEF output, second input): their own inputs().
    import g17callreceipt, g17constspacereceipt, g17switchreceipt
    for fam, tool in (("callinline", g17callreceipt), ("constspace", g17constspacereceipt), ("switch", g17switchreceipt)):
        for r in json.loads((root / ("isa/g17-execution-%s-results.json" % fam)).read_text()):
            if r.get("cb_status") != 0:
                continue
            a, c = tool.inputs()
            arrays = {0: np.asarray(a).view(np.uint32), 1: np.full(n, 0xDEADBEEF, np.uint32), 2: np.asarray(c).view(np.uint32)}
            code, observed = bytes.fromhex(r["program"]), r["values"]
            try:
                out, _b, _m = _harness(code, arrays, 1, tier)
                rows.append(_verdict(fam, r["id"], out, observed))
            except Refused as e:
                rows.append((fam, r["id"], "refused", str(e)[:160]))
    rows.extend(graph_receipts(root, tier))   # the compiled Metal sources' worker receipts, query by query
    rows.extend(tensor_receipts(root, tier))  # the in-core accelerator classes with the GPU's own output kept
    return rows


def tensor_receipts(root=None, tier="strict", classes=None):
    """THE IN-CORE ACCELERATOR RECEIPTS (MM P13): every tensor class bundle under results/g17-p13-classes-v1 that
    kept the GPU's output (output-q1.npz, `got_u32`) - fp16, bf16, transposed A and B, int8 wrapping and saturating,
    half output - run from its program.bin as g17bundlerun dispatches it (the receipt's grid and threadgroup), and
    compared word for word with what the GPU returned."""
    import json
    root = Path(root or HERE.parent)
    rows = []
    for d in sorted((root / "results/g17-p13-classes-v1").glob("*/")):
        if not (d / "output-q1.npz").exists() or (classes and d.name not in classes):
            continue
        shape = json.loads((d / "receipt.json").read_text())["shape"]
        got = np.load(d / "output-q1.npz")["got_u32"].ravel()
        try:
            out, _m = run_bundle(d, int(np.prod(shape["grid"])), int(np.prod(shape["threadgroup"])), 1, tier)
        except Refused as e:
            rows.append(("tensor", d.name, "refused", str(e)[:160]))
            continue
        rows.append(_verdict("tensor", d.name, np.frombuffer(out[:got.nbytes], "<u4").tolist(), got.tolist()))
    # EVERY OTHER TENSOR RECEIPT that records the dispatch shape and the GPU output's sha256 (register chains,
    # epilogues, grids, fused attention cuts, ...): the same run, compared by hash over the receipt's `bytes`. A
    # program with hardware exp2 (op1272) is left out by name - it is not a known function (MM 25.136).
    import hashlib
    seen = {d.resolve() for d in (root / "results/g17-p13-classes-v1").glob("*/")}
    for f in sorted((root / "results").glob("**/receipt.json")):
        d = f.parent
        if d.resolve() in seen or (classes and d.name not in classes):
            continue
        if not all((d / n).exists() for n in ("program.bin", "a.f16", "b.f16", "c.f32")):
            continue
        try:
            rec = json.loads(f.read_text())
        except ValueError:
            continue
        sh = rec.get("shape") if isinstance(rec, dict) else None
        q = [x for x in (rec.get("queries") or []) if isinstance(x, dict) and x.get("output_sha256")] \
            if isinstance(rec, dict) else []
        if not (isinstance(sh, dict) and sh.get("grid") and sh.get("threadgroup") and q):
            continue
        if 1272 in {op for _o, _s, op, _t in decode((d / "program.bin").read_bytes())} and not _exp2_on():
            continue
        rid = str(d.relative_to(root / "results"))
        try:
            out, _m = run_bundle(d, int(np.prod(sh["grid"])), int(np.prod(sh["threadgroup"])), 1, tier)
        except Refused as e:
            rows.append(("tensorhash", rid, "refused", str(e)[:160]))
            continue
        n = q[0].get("bytes")
        same = hashlib.sha256(out[:n] if n else out).hexdigest() == q[0]["output_sha256"]
        rows.append(("tensorhash", rid, "agree" if same else "DISAGREE",
                     "" if same else "output sha256 differs from the GPU's"))
    # EVERY KEPT RAW GPU OUTPUT of a tensor bundle (output-q1.npz or mismatch-q1.npz: a run that disagreed with
    # some MODEL still returned the GPU's truth in `got`): run as its manifest's tensor contract dispatches it, and
    # compared word for word. Hardware exp2 (op1272) is left out by name, as above.
    done = {r[1] for r in rows}
    for f in sorted(set((root / "results").glob("**/mismatch-q1.npz")) | set((root / "results").glob("**/output-q1.npz"))):
        d = f.parent
        rid = str(d.relative_to(root / "results"))
        if rid in done or "p13-classes" in rid or "requant" in rid or (classes and d.name not in classes):
            continue
        if not all((d / n).exists() for n in ("program.bin", "a.f16", "b.f16", "c.f32", "manifest.json")):
            continue
        z = np.load(f)
        if "got" not in z.files:
            continue
        try:
            t = json.loads((d / "manifest.json").read_text()).get("tensor") or {}
        except ValueError:
            continue
        if not (t.get("grid") and t.get("threadgroup")):
            continue
        if 1272 in {op for _o, _s, op, _t in decode((d / "program.bin").read_bytes())} and not _exp2_on():
            continue
        got = np.ascontiguousarray(z["got_u32"] if "got_u32" in z.files else z["got"]).view(np.uint8).ravel()
        try:
            out, _m = run_bundle(d, int(np.prod(t["grid"])), int(np.prod(t["threadgroup"])), 1, tier)
        except Refused as e:
            rows.append(("tensorgot", rid, "refused", str(e)[:160]))
            continue
        rows.append(_verdict("tensorgot", rid, np.frombuffer(out[:got.size], "<u4").tolist(), got.view("<u4").tolist()))
    # THE IMAGEBLOCK SET C ROUNDS (MM 25.104): results/g17-tensor-imageblock-v*/setc-round*-receipt.json records each
    # arm's GPU output sha256 over `bytes`; the arm's bundle sits beside the receipt.
    for f in sorted((root / "results").glob("g17-tensor-imageblock-v*/setc-round*-receipt.json")):
        for arm in json.loads(f.read_text()).get("arms", {}).values():
            d = f.parent / arm["arm"]
            if not (d / "program.bin").exists() or (classes and d.name not in classes) or not arm.get("output_sha256"):
                continue
            t = json.loads((d / "manifest.json").read_text()).get("tensor") or {}
            rid = "%s %s" % (f.parent.name, arm["arm"])
            try:
                out, _m = run_bundle(d, int(np.prod(t["grid"])), int(np.prod(t["threadgroup"])), 1, tier)
            except Refused as e:
                rows.append(("imageblock", rid, "refused", str(e)[:160]))
                continue
            same = hashlib.sha256(out[:arm["bytes"]]).hexdigest()[:len(arm["output_sha256"])] == arm["output_sha256"]
            rows.append(("imageblock", rid, "agree" if same else "DISAGREE",
                         "" if same else "output sha256 differs from the GPU's"))
    # EVERY ARM A dispatch-receipt.json RECORDS AS "passed": the GPU's output was bit-identical to the independent CPU
    # reference (g17tensorcommonruntime.generic_reference) for that exact program (program_sha256_16 checked), so the
    # reference IS the GPU's output. Hardware exp2 is left out by name, as above.
    ref_done = {r[1] for r in rows}
    for f in sorted((root / "results").glob("**/dispatch-receipt.json")):
        try:
            rec = json.loads(f.read_text())
        except ValueError:
            continue
        if not isinstance(rec, dict):
            continue
        for arm, v in sorted(rec.items()):
            d = f.parent / arm
            rid = str(d.relative_to(root / "results"))
            if not isinstance(v, dict) or v.get("status") != "passed" or rid in ref_done or \
                    (classes and arm not in classes):
                continue
            if not all((d / n).exists() for n in ("program.bin", "generic.json", "a.f16", "b.f16", "c.f32",
                                                  "manifest.json")):
                continue
            code = (d / "program.bin").read_bytes()
            if v.get("program_sha256_16") and hashlib.sha256(code).hexdigest()[:16] != v["program_sha256_16"]:
                continue
            if {1272, 3658} & {op for _o, _s, op, _t in decode(code)}:
                # a hardware exp2 or reciprocal program "passed" against a correctly-rounded model WITHIN A BOUND,
                # not bit for bit (both ops are within one ulp and not correctly rounded, MM 25.136): its reference is
                # not the GPU's output, so this family cannot judge it (fusion-v1's exp2 / exp arms: 1-ulp words)
                continue
            t = json.loads((d / "manifest.json").read_text()).get("tensor") or {}
            if not (t.get("grid") and t.get("threadgroup")):
                continue
            import g17tensorcommonruntime as TR
            try:
                out, _m = run_bundle(d, int(np.prod(t["grid"])), int(np.prod(t["threadgroup"])), 1, tier)
            except Refused as e:
                rows.append(("tensorref", rid, "refused", str(e)[:160]))
                continue
            exp = np.ascontiguousarray(TR.generic_reference(d, TR.generic_spec(json.loads(
                (d / "generic.json").read_text())))).view(np.uint8).ravel()
            rows.append(_verdict("tensorref", rid, np.frombuffer(out[:exp.size], "<u4").tolist(),
                                 exp.view("<u4").tolist()))
    # THE INT8 REQUANT ARMS (MM 25.130): an int8 GEMM requantized in registers (bias, i2f, scale, rint, clamp, f2i,
    # zero point, packed byte stores), each with the GPU's whole C buffer kept as `got` bytes - the negative arms too
    for d in sorted((root / "results/g17-tensor-requant-v1").glob("*/")):
        if not (d / "output-q1.npz").exists() or (classes and d.name not in classes):
            continue
        t = json.loads((d / "manifest.json").read_text())["tensor"]
        got = np.load(d / "output-q1.npz")["got"]
        try:
            out, _m = run_bundle(d, int(np.prod(t["grid"])), int(np.prod(t["threadgroup"])), 1, tier)
        except Refused as e:
            rows.append(("requant", d.name, "refused", str(e)[:160]))
            continue
        rows.append(_verdict("requant", d.name, list(out[:got.size]), got.tolist()))
    return rows


ATTN_GUARD = 128                                   # G17_ATTN_GUARD: a binding's payload starts 128 bytes in


def graph_receipts(root=None, tier="strict"):
    """THE SOURCE RECEIPTS (MM 25.149): the compiled Metal sources that ran on hardware through the attention worker
    (tools/g17attentionexecutor.m), each an execution.json beside its graph.json, manifest.json and programs/, with
    every query's input and observed output kept as query-N.npz. isa/g17-source-execution.json lists them
    ("retained_receipts"). Each query is replayed as the worker ran it: every allocation filled with 0xA5 ^ its
    position (the binding trace gives positions), each payload with the sentinel 0x7fc01234 (0x7e12 for 2-byte
    elements), inputs copied in (intermediate_initialization, output_initialization_offset), then every stage in
    order with dispatchThreads over the stage's resolved grid; the output allocation's payload is compared word for
    word with what the GPU returned. The program run is the retained program.bin (its sha256 checked against the
    receipt), bound by its manifest's ABI table."""
    import hashlib, json
    root = Path(root or HERE.parent)
    rows = []
    listed = json.loads((root / "isa/g17-source-execution.json").read_text())["retained_receipts"]
    for tag, files in sorted(listed.items()):
        for f in files:
            f = root / f
            rid = "%s %s" % (tag, f.parent.name)
            try:
                rows.extend(_graph_receipt(f, rid, tier))
            except Refused as e:
                rows.append(("graph", rid, "refused", str(e)[:160]))
            except (OSError, KeyError, ValueError) as e:
                rows.append(("graph", rid, "refused", "receipt not replayable here: %s: %s" % (type(e).__name__, e)))
    return rows


def _graph_receipt(execution, rid, tier):
    import hashlib, json
    e = json.loads(Path(execution).read_text())
    d = Path(execution).parent.parent
    g = json.loads((d / "graph.json").read_text())
    man = json.loads((d / "manifest.json").read_text())
    if g.get("format") != "g17-attention-graph-v1" or g.get("write_policy") not in (None, "multiple-v1"):
        raise Refused("graph format %s / %s is not modelled" % (g.get("format"), g.get("write_policy")))
    # no write_policy is the legacy single-write mode (g17attentionschedule.h:54-156): one written binding a stage,
    # no intermediate initialization; reset and output initialization are the same (g17attentionstorage.h)
    if g.get("binding_windows") or g.get("textures"):
        raise Refused("a graph with binding windows or textures is not modelled")
    allocs = g["allocations"]
    pos = {name: i for i, name in enumerate(sorted(allocs))}     # positions are the sorted names (worker.m:59)
    for row in e["queries"][0]["header"].get("binding_trace", []):   # and the worker's own trace agrees
        b = [b for b in g["stages"][row[0]]["bindings"] if b["index"] == row[1]][0]
        if pos[b["allocation"]] != row[2]:
            raise Refused("allocation %s at position %d, the trace says %d" % (b["allocation"], pos[b["allocation"]], row[2]))
    codes = {}
    for name, h in e["files"].items():
        if name.startswith("programs/") and name.endswith("/program.bin"):
            prog = name.split("/")[1]
            code = (d / name).read_bytes()
            if hashlib.sha256(code).hexdigest() != h:
                raise Refused("%s is not the program the receipt ran" % name)
            codes[prog] = code
    if any(a["role"] not in ("input", "parameter", "intermediate", "output") for a in allocs.values()):
        raise Refused("an allocation role outside input, parameter, intermediate, output")
    source = [n for n, a in allocs.items() if a["role"] == "input"]
    output = [n for n, a in allocs.items() if a["role"] == "output"]
    if len(source) != 1 or len(output) != 1:
        raise Refused("a graph without exactly one input and one output allocation")
    source, output = source[0], output[0]
    launches = {l["stage"]: l for l in e["identity"]["resolved_launches"]}
    width = lambda a: {"half": 2, "ushort": 2}.get(allocs[a]["element_type"], 4)    # worker.m:101
    rows = []
    qfiles = sorted(Path(execution).parent.glob("query-*.npz"), key=lambda q: int(q.stem.split("-")[1]))
    if not qfiles:
        raise Refused("no retained queries")
    for qf in qfiles:
        q = np.load(qf)
        src = q["source"].astype("<u4").tobytes() if q["source"].dtype.itemsize == 4 else q["source"].tobytes()
        mem = {}
        for name, a in allocs.items():
            buf = np.full(int(a["payload_bytes"]) + 2 * ATTN_GUARD, 0xA5 ^ pos[name], np.uint8)  # storage.h:33
            sent = (b"\x12\x7e" if width(name) == 2 else (0x7fc01234).to_bytes(4, "little"))
            n = int(a["payload_bytes"])
            buf[ATTN_GUARD:ATTN_GUARD + n] = np.frombuffer((sent * (n // len(sent) + 1))[:n], np.uint8)
            mem[name] = buf
        mem[source][ATTN_GUARD:ATTN_GUARD + len(src)] = np.frombuffer(src, np.uint8)
        for name, a in allocs.items():                          # parameters: their snapshot, loaded once (Prepare)
            if a["role"] == "parameter":
                snap = [k for k in e["files"] if k.startswith("inputs/%s." % name)]
                if len(snap) != 1:
                    raise Refused("parameter %s has no retained snapshot" % name)
                data = (d / snap[0]).read_bytes()
                if hashlib.sha256(data).hexdigest() != e["files"][snap[0]] or len(data) != int(a["payload_bytes"]):
                    raise Refused("%s is not the snapshot the receipt ran" % snap[0])
                mem[name][ATTN_GUARD:ATTN_GUARD + len(data)] = np.frombuffer(data, np.uint8)
        oi = g.get("output_initialization")
        if oi is not None and oi not in ("copy-input-v1", "copy-input-slice-v1", "copy-input-bytes-v1"):
            raise Refused("output initialization %s" % oi)
        if oi:                                                  # every mode is one memcpy (storage.h:201-206)
            o, n = int(g.get("output_initialization_offset", 0)), int(allocs[output]["payload_bytes"])
            mem[output][ATTN_GUARD:ATTN_GUARD + n] = np.frombuffer(src[o:o + n], np.uint8)
        for ini in g.get("intermediate_initialization", []):
            if ini["mode"] != "copy-input-bytes-v1":
                raise Refused("intermediate initialization %s" % ini["mode"])
            o, n = int(ini["source_offset"]), int(allocs[ini["allocation"]]["payload_bytes"])
            mem[ini["allocation"]][ATTN_GUARD:ATTN_GUARD + n] = np.frombuffer(src[o:o + n], np.uint8)
        for st in g["stages"]:
            l = launches[st["name"]]
            abi = man["programs"][st["program"]]["abi"]
            views = {}
            for b in st["bindings"]:
                views[int(b["index"])] = mem[b["allocation"]][int(b["offset"]):]
            m = Machine(codes[st["program"]], views, tuple(l["grid"]), tuple(l["threadgroup"]), None, tier,
                        views=True, table=abi_table(abi))
            if BRACKETED.keys() & {op for _o, _s, op, _t in decode(m.code)}:
                raise Refused("a bracketed instruction in a graph stage")
            m.run()
        el = "<u2" if width(output) == 2 else "<u4"             # compared in the output's own elements
        got = np.frombuffer(bytes(mem[output][ATTN_GUARD:ATTN_GUARD + int(allocs[output]["payload_bytes"])]), el)
        obs = np.frombuffer(np.ascontiguousarray(q["output"]).tobytes(), el)
        rows.append(_verdict("graph", "%s %s" % (rid, qf.stem), got.tolist(), obs.tolist()))
    return rows


def _verdict(family, rid, got, observed):
    obs = [int(v) & 0xFFFFFFFF for v in observed]
    bad = [i for i, (g, o) in enumerate(zip(got, obs)) if g != o]
    return (family, rid, "agree" if not bad and len(got) >= len(obs) else "DISAGREE",
            "" if not bad else "%d of %d words differ, first lane %d: emulated 0x%08x, hardware 0x%08x"
            % (len(bad), len(obs), bad[0], got[bad[0]], obs[bad[0]]))


def _replay(family, rid, code, words, observed, tier, beyond=None, runs=False):
    """One _formlower_run receipt: `observed` is the 32 words (or, with runs, a list of repeat runs, each compared);
    `beyond`, where the receipt counted them, is how many output words past the first 32 the kernel wrote."""
    try:
        n = 64 * 64
        B = np.zeros(n, np.uint32)
        B[:len(words)] = np.asarray(words, np.uint64).astype(np.uint32)
        out, written, _m = _harness(code, {1: B, 2: np.full(n, 0xDEADBEEF, np.uint32)}, 2, tier)
    except Refused as e:
        return (family, rid, "refused", str(e)[:160])
    for obs in (observed if runs else [observed]):
        v = _verdict(family, rid, out, obs)
        if v[2] != "agree":
            return v
    if beyond is not None and written != beyond:
        return (family, rid, "DISAGREE", "%d words written past the first 32; the hardware wrote %d" % (written, beyond))
    return (family, rid, "agree", "")


def run_graph(graph_path, positions=1, tier="strict", log=None):
    """THE STRETCH (MM 25.145): a model graph's decode steps on the CPU, dispatch by dispatch, as decodegen's pipelined
    mode runs them - every arena allocated, arena_init files loaded, zero_init regions zeroed, then per position every
    dispatch in order, its bindings live slices of the arenas (so each dispatch's stores feed the next). Returns
    (arenas, per-position logits as float32 arrays read from the graph's logits_readback)."""
    import json, time
    g = json.loads(Path(graph_path).read_text())
    arenas = {name: np.zeros(int(n), np.uint8) for name, n in g["arenas"].items()}
    def load(ini):
        data = np.frombuffer(Path(ini["file"]).read_bytes(), np.uint8)
        arenas[ini["arena"]][int(ini["offset"]):int(ini["offset"]) + len(data)] = data
    # decodegen's order: every arena_init file, then reinit - zero_init regions zeroed, then R_init.bin loaded over
    # them again (the first token's embedding and the generation state live there)
    for ini in g.get("arena_init", []):
        load(ini)
    for z in g.get("zero_init", []):
        arenas[z["arena"]][int(z["offset"]):int(z["offset"]) + int(z["bytes"])] = 0
    for ini in g.get("arena_init", []):
        if Path(ini["file"]).name == "R_init.bin":
            load(ini)
    if g.get("per_token_writes"):
        raise Refused("a graph with per-token host writes: only the pipelined (device-resident) form is modelled")
    codes = {}
    lr = g["logits_readback"]
    logits = []
    admitted = {}                                   # whole-program-only semantics used (tier wp), with counts
    for pos in range(positions):
        t0 = time.time()
        for n, d in enumerate(g["dispatches"]):
            code = codes.get(d["bundle"]) or codes.setdefault(d["bundle"], (Path(d["bundle"]) / "program.bin").read_bytes())
            views = {int(slot): arenas[b["arena"]][int(b["offset"]):int(b["offset"]) + int(b["bytes"])]
                     for slot, b in d["binds"].items()}
            ops = {op for _o, _s, op, _t in decode(code)}
            bracket = [op for op in BRACKETED if op in ops]
            try:
                if not bracket:
                    m = Machine(code, views, int(d["threads"]), int(d["group"]), None, tier, views=True).run()
                    for k, v in m.admitted.items():
                        admitted[k] = admitted.get(k, 0) + v
                else:                                   # both admissible results, each from the same memory
                    outs = []
                    for ch in ("down", "up"):
                        copies = {k: v.copy() for k, v in views.items()}
                        m = Machine(code, copies, int(d["threads"]), int(d["group"]), {op: ch for op in bracket},
                                    tier, views=True).run()
                        outs.append(copies)
                    for k, v in m.admitted.items():
                        admitted[k] = admitted.get(k, 0) + v
                    if any(not np.array_equal(outs[0][k], outs[1][k]) for k in views):
                        raise Refused("output depends on the unmeasured rounding of %s" % bracket)
                    for k in views:
                        views[k][:] = outs[0][k]
            except Refused as e:
                raise Refused("position %d, dispatch %d (%s): %s" % (pos, n, d["name"], e)) from None
            if log:
                log("pos %d  %3d/%d  %-24s %.0fs" % (pos, n + 1, len(g["dispatches"]), d["name"], time.time() - t0))
        a = arenas[lr["arena"]][int(lr["offset"]):int(lr["offset"]) + int(lr["bytes"])]
        logits.append(a.view(np.float32).copy())
    run_graph.admitted = admitted
    return arenas, logits


def control(bundle, threads, group, base, check, limit=400, tier="strict", skip_ops=(14059, 14060), stats=None,
            operands=("reg",), seed=None):
    """THE CONTROL (MM 25.145): a deliberately wrong program must FAIL the check. Search the program for a single-bit
    flip that keeps every instruction's framing and opcode and changes exactly one SOURCE register token of one
    instruction; run it; return the first mutation whose output the check rejects, as (offset, byte, bit, before,
    after, mismatches). Refused mutants do not count - a refusal is g17emu's safety, not the check's.

    operands=("reg", "imm") also flips SOURCE immediates (operand 2 onward; operand 1 is the destination's mode word,
    which the receipt-mode rule already polices). seed shuffles the instruction order, so a limit samples the whole
    program instead of its first instructions."""
    import shutil, tempfile
    d = Path(bundle)
    code = (d / "program.bin").read_bytes()
    base_rows = decode(code)
    frame = [(o, s_, op) for o, s_, op, _t in base_rows]
    tried = 0
    stats = stats if stats is not None else {}
    try:                                     # deform a WORKING program: the original must run and pass first
        base_out, _m = run_bundle(d, threads, group, base, tier)
    except Refused as e:
        raise Refused("control: the unmutated program is itself refused (%s), so no mutant can be judged" % e)
    if check(base_out):
        raise ValueError("control: the unmutated program already fails its check (%d)" % check(base_out))
    order = list(base_rows)
    if seed is not None:
        __import__("random").Random(seed).shuffle(order)
    for off, size, op, toks in order:
        if op not in EVIDENCE or op in (684,) or op in skip_ops or len(toks) < 3:
            continue
        for byte in range(off, off + size):
            for bit in range(8):
                m = bytearray(code)
                m[byte] ^= 1 << bit
                try:
                    rows = decode(bytes(m))
                except _undecodable():
                    continue
                if [(o, s_, p) for o, s_, p, _t in rows] != frame:
                    continue
                new = [t for o, _s, _p, t in rows if o == off][0]
                diff = [i for i, (a, b) in enumerate(zip(toks, new)) if a != b]
                if len(diff) != 1 or diff[0] == 0:
                    continue
                was, now = toks[diff[0]], new[diff[0]]
                if not (("reg" in operands and was.startswith("reg:") and now.startswith("reg:")) or
                        ("imm" in operands and diff[0] >= 2 and was.startswith("imm:") and now.startswith("imm:"))):
                    continue
                tried += 1
                stats["tried"] = tried
                if tried > limit:
                    return None
                with tempfile.TemporaryDirectory() as t:
                    md = Path(t) / "mut"
                    shutil.copytree(d, md)
                    (md / "program.bin").write_bytes(bytes(m))
                    try:
                        out, _m = run_bundle(md, threads, group, base, tier)
                    except Refused as e:
                        stats["refused"] = stats.get("refused", 0) + 1
                        why = stats.setdefault("refused_by", {})
                        key = __import__("re").sub(r"(instruction \d+ )?at \+0x[0-9a-f]+ ?", "", str(e))[:60]
                        why[key] = why.get(key, 0) + 1
                        continue
                bad = check(out)
                if not bad:
                    stats["passed"] = stats.get("passed", 0) + 1
                if bad:
                    return dict(offset=off, byte=byte - off, bit=bit, before=toks[diff[0]], after=new[diff[0]],
                                mismatches=int(bad))
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("bundle", nargs="?")
    ap.add_argument("--threads", type=int)
    ap.add_argument("--group", help="threads per threadgroup: G, or X,Y,Z with --program")
    ap.add_argument("--base", type=int, default=1)
    ap.add_argument("--tier", choices=("strict", "wp"), default="strict")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--receipts", action="store_true", help="replay the hardware receipts and report agreement")
    ap.add_argument("--check-coverage", action="store_true",
                    help="replay the receipts and fail unless isa/g17-emu-coverage.json's receipts section is exactly "
                         "what they return now (the gate: the coverage file cannot go stale)")
    ap.add_argument("--write-coverage", action="store_true", help="rewrite that section from the replay")
    ap.add_argument("--program", type=Path, help="run this program.bin over --buffer files (run_program)")
    ap.add_argument("--buffer", action="append", default=[], metavar="INDEX=FILE",
                    help="with --program: bind FILE's bytes as buffer INDEX (repeatable); a missing FILE is a "
                         "zero-filled buffer of the size given as INDEX=@BYTES")
    ap.add_argument("--grid", help="with --program: the dispatch in threads, X or X,Y,Z")
    ap.add_argument("--abi", type=Path, help="with --program: a JSON file holding the compiler's ABI (a bindings list, "
                                             "or an object with 'abi' or 'bindings'); without it the table is dense")
    ap.add_argument("--outdir", type=Path, help="with --program: write each buffer after the run as bufINDEX.bin")
    a = ap.parse_args(argv)
    if a.check_coverage or a.write_coverage:
        import json
        path = HERE.parent / "isa/g17-emu-coverage.json"
        doc = json.loads(path.read_text())
        rows = receipts(tier=a.tier)
        fams = {}
        for fam, _rid, verdict, _d in rows:
            fams.setdefault(fam, {}).setdefault(verdict, 0)
            fams[fam][verdict] += 1
        live = dict(by_family=fams, rows=[dict(family=f, id=i, verdict=v, detail=d) for f, i, v, d in rows])
        bad = [r for r in rows if r[2] not in ("agree", "refused")]
        if a.write_coverage:
            doc["receipts"] = live
            path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
            print("wrote %s: %d receipts, %d disagree" % (path.relative_to(HERE.parent), len(rows), len(bad)))
            return 1 if bad else 0
        if bad or doc.get("receipts") != live:
            print("g17emu coverage: %d receipt rows disagree; the recorded section %s the replay (run "
                  "tools/g17emu.py --write-coverage)" % (len(bad), "matches" if doc.get("receipts") == live else
                                                          "differs from"))
            return 1
        print("g17emu coverage: %d receipt rows as recorded, none disagree" % len(rows))
        return 0
    if a.program:
        import json
        axes = lambda t: tuple(int(x) for x in t.split(",")) + (1,) * (3 - len(t.split(",")))
        if not (a.grid and a.group):
            ap.error("--program needs --grid and --group")
        bufs = {}
        for spec in a.buffer:
            idx, _eq, src = spec.partition("=")
            bufs[int(idx)] = bytes(int(src[1:])) if src.startswith("@") else Path(src).read_bytes()
        table = None
        if a.abi:
            doc = json.loads(a.abi.read_text())
            doc = doc.get("abi", doc) if isinstance(doc, dict) else doc
            table = abi_table({"bindings": doc["bindings"] if isinstance(doc, dict) else doc})
        try:
            out, m = run_program(a.program.read_bytes(), bufs, axes(a.grid), axes(str(a.group)), table, a.tier)
        except Refused as e:
            print("REFUSED:", e)
            return 2
        print("ran %d instructions' worth; opcodes %s" % (sum(m.used.values()), dict(sorted(m.used.items()))))
        if m.admitted:
            print("whole-program semantics admitted: %s" % dict(m.admitted))
        if a.outdir:
            a.outdir.mkdir(parents=True, exist_ok=True)
            for idx, data in out.items():
                (a.outdir / ("buf%d.bin" % idx)).write_bytes(data)
        return 0
    if a.receipts:
        rows = receipts(tier=a.tier)
        for fam, rid, verdict, detail in rows:
            print("%-8s %-40s %-8s %s" % (fam, rid, verdict, detail))
        covered = [r for r in rows if r[2] != "refused"]
        bad = [r for r in covered if r[2] != "agree"]
        print("%d of %d covered receipts agree; %d refused (not covered)" % (len(covered) - len(bad), len(covered),
                                                                           len(rows) - len(covered)))
        return 1 if bad else 0
    if not (a.bundle and a.threads and a.group):
        ap.error("a bundle needs --threads and --group")
    try:
        out, m = run_bundle(a.bundle, a.threads, int(a.group), a.base, a.tier)
    except Refused as e:
        print("REFUSED:", e)
        return 2
    print("ran %d instructions' worth; opcodes %s" % (sum(m.used.values()), dict(sorted(m.used.items()))))
    if m.admitted:
        print("whole-program semantics admitted: %s" % dict(m.admitted))
    if a.out:
        a.out.write_bytes(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
