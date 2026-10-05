#!/usr/bin/env python3
"""THE DELIVERED-BYTE INTERPRETER FOR THE FP32 LAYERNORM: registers, four buffers, addresses, FP32.

tools/g17halfcheck.py interprets the half scan's delivered bytes. The LayerNorm program is FP32 with
FOUR bindings and computed addresses, and the first version of this checker could not read it: it
refused op14059 - the thread-id read at offset zero - because it modelled only the float opcodes and
never the integer ones that compute every address; it took `allocation_words` and did not use it, so
a four-column program passed with zero words of allocation; its float add was Python's double, so
16777216 + 1 came back 16777217 where FP32 gives 16777216; and its bit-exact comparison was `==`,
which calls +0.0 and -0.0 the same bits. Each of those is fixed here and each has a control in
tools/g17layernormadmission.py that fails without the fix.

WHAT THIS IS. A register machine over the DECODED operands of the delivered instructions - Apple's
decoder gives the tokens, this reads them. Integer opcodes compute addresses; a load resolves its
binding RANK from the address expression and its element from an index register plus a byte
displacement, and is refused if the element lies outside that binding's allocation; every FP32
result is rounded through numpy.float32; the reciprocal square root goes through interpret(), which
carries the measured domain with the number. The output is checked element by element against the
application's own FP64 reference by whoever calls this.

WHAT THE EVIDENCE IS, kept in two classes because they are two different measurements:

    execution     an opcode named by spike/accel/re/opsem.py: authored alone, fed known inputs, run,
                  matched against candidate meanings. ~/.cache/agxforge/opsem.json. Here: 998 fadd,
                  3290 fmul, 3850 rsqrt (and 3978).

    silicon       an opcode this compiler emitted inside an end-to-end kernel whose every output
    agreement     matched Apple's compilation AND the arithmetic on the GPU, isa/g17-endtoend-
                  results.json. Here: the thread-id read, movimm, the integer add and multiply, the
                  word load and the indexed word store. That is strong evidence about the whole
                  instruction in context and weaker evidence about any one of its bits than the
                  first class is; it is reported under its own name and never as "executed".

A result is `executed` only when every arithmetic opcode has the first class AND every structural
one has the second. An EMPTY evidence set gives `unverified`, never `executed` - the checker must not
report confidence it was not handed. And bit-exact means the same 32 bits, which +0.0 and -0.0 are
not.

WHAT THIS DOES NOT SAY: that the hardware computes what these opcodes are named for on inputs nobody
has run. op3850's rounding on every normal input is not established by four probes, and that limit
travels in the confidence string rather than being smoothed over by a tolerance.

    python3 tools/g17normcheck.py                  report what can and cannot be interpreted
"""
import json
import functools
import math
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# THE EVIDENCE IS COMMITTED, NOT READ FROM ONE MACHINE'S CACHE. This read ~/.cache/agxforge/opsem.json,
# the output of spike/accel/re/opsem.py's 2026-09-09 execution sweep, and returned an EMPTY set where
# the cache was absent - so on any other machine 452 opcodes the sweep ran alone lost their
# "executed" evidence without a word (Piece A, 2026-09-22). isa/g17-opsem.json is that cache,
# committed 2026-09-22; opsem.py still writes the cache, and a new sweep is committed by copying it.
OPSEM = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "isa", "g17-opsem.json")
E2E = os.path.join(ROOT, "isa", "g17-endtoend-results.json")


def executed_opcodes(path=OPSEM):
    """The opcodes named BY RUNNING THEM ALONE, as integers. A missing file REFUSES: an empty set
    here silently turns every executed verdict into an unexecuted one."""
    with open(path) as fh:
        doc = json.load(fh)
    return frozenset(int(str(k).split("|")[0]) for k in doc if str(k).split("|")[0].isdigit())


_E2E_CACHE = {}


def silicon_agreement_opcodes(path=E2E):
    """The opcodes emitted by end-to-end kernels whose outputs matched Apple AND the arithmetic.

    Derived, not listed: the results file names the kernels and the two agreement flags, and the
    kernels are recompiled here to see which opcodes they contain. A kernel that disagreed on either
    axis contributes nothing.
    """
    if path in _E2E_CACHE:
        return _E2E_CACHE[path]
    out = set()
    try:
        with open(path) as fh:
            doc = json.load(fh)
        import g17cc, g17endtoend, g17ref
        for name, rec in (doc.get("kernels") or {}).items():
            if not (rec.get("agree_with_apple") and rec.get("agree_with_arithmetic")):
                continue
            ent = g17endtoend.KERNELS.get(name)
            if ent is None:
                continue
            prog = g17cc.compile_function(ent[0]())
            out |= {op for _a, _l, op in g17ref.walk(prog.code, 0)}
    except Exception:
        out = set()
    _E2E_CACHE[path] = frozenset(out)
    return _E2E_CACHE[path]


# ---------------------------------------------------------------- the reciprocal square root
# MEASURED BY EXECUTION 2026-09-09, four inputs, both opcodes, op998 as the control in the same
# session. It closes "what still separates op3978 from op3850 is not established":
#
#     input                     op3850     op3978     1/sqrt(x)
#     2.5                       0.632456   0.632456   0.632456
#     -3.75                     NaN        NaN        NaN
#     2.10e-44  (denormal)      +inf       1.0        6.90e+21
#     1.40e-45  (denormal)      +inf       1.0        2.67e+22
#
# op3850 FLUSHES A DENORMAL INPUT TO ZERO and returns +inf; op3978 returns exactly 1.0. So "rsqrt"
# describes neither completely. A tolerance band cannot express that - the disagreement is infinity
# against 6.9e+21, not rounding - so the checker states a DOMAIN instead.
FLOAT32_MIN_NORMAL = 2.0 ** -126


def _rsqrt_3850(x):
    """op3850: a denormal or zero input flushes to zero, so the reciprocal is +inf."""
    x = float(x)
    if x != x:
        return x
    if x < 0.0:
        return float("nan")
    if x < FLOAT32_MIN_NORMAL:
        return float("inf")
    return 1.0 / math.sqrt(x)


def _rsqrt_3978(x):
    """op3978: a denormal input returns exactly 1.0, measured on two independent denormals."""
    x = float(x)
    if x != x:
        return x
    if x < 0.0:
        return float("nan")
    if x < FLOAT32_MIN_NORMAL:
        return 1.0
    return 1.0 / math.sqrt(x)


def _exp2_1272(x):
    """2**x in float32, correctly rounded - WHICH op1272 IS NOT. EXECUTED BIT-EXACTLY ON EXACT POWERS:
    the sweep ran op1272 on 1, 2, 4, 8 and read 2, 4, 16, 256 back (isa/g17-execution-sweep-results
    .json D1272.l10), so interpret() reports "executed" only for an integer input.

    CORRECTED 2026-09-22: what stood here said exp2's rounding on a fractional input "is not
    measured" and needed the exp2_fractional separator run alone. It WAS measured, alone, on
    2026-09-17: marginpairs u1272 ran op1272 by itself on eight fractional inputs (+-0.5, +-1.5,
    +-2.5, +-3.25), and isolated_rounding(1272) reads them against the correctly-rounded value -
    seven of those eight were one ulp high. THAT WAS EIGHT INPUTS, AND THE DENSE SWEEP CORRECTS IT
    (isa/g17-execution-transsweep, 2026-09-23, 256 fractional inputs in [-10, 10], each run alone):
    over all 264 isolated fractional inputs exp2 is EXACT on 172, one ulp high on 89 and one ulp LOW
    on 3 - never more than one ulp, but neither always high nor never low. The control is inside the same batch:
    exp2(0.5) and rsqrt(0.5) are both sqrt(2); op3850 returns 0x3FB504F3, the correctly-rounded
    bits, and op1272 returns 0x3FB504F4. So this model is wrong by one ulp on about a third of fractional
    inputs, the verdict off integers stays a bounded comparison, and the softmax's <=4-5 ulp
    against this model (~5,200-5,600 of 12,288 outputs, per element, results/g17-attention-softmax
    -execution-v1) is the chain - fmul, exp2, the fadd sum, recip, fmul - carrying exp2's own
    <=1 ulp, not exp2's rounding. The separator probe now only adds the softmax's exact inputs."""
    import numpy as np
    with np.errstate(over="ignore"):
        return float(np.float32(2.0) ** np.float32(x))


def _recip_3658(x):
    """1/x in float32. The sweep ran op3658 on 1, 2, 4, 8 and read 1, .5, .25, .125 back
    (D3658.l10) - exact reciprocals of powers of two; rounding elsewhere is not measured."""
    import numpy as np
    with np.errstate(divide="ignore"):
        return float(np.float32(1.0) / np.float32(x))


# rsqrt: NOT correctly rounded, and that is now an ISOLATED measurement rather than an inference
# from a LayerNorm row: marginpairs u3850/u3978 ran each opcode alone at 1.5 and both return
# 0x3F5105EB where the correctly-rounded 1/sqrt(1.5) is 0x3F5105EC - one ulp low. The dense sweep
# (isa/g17-execution-transsweep, 256 inputs 1e-3..1e4) puts op3850 at 251 exact, 14 one ulp high and
# 3 one ulp low over 268 positive normals: never more than one ulp, in both directions. The resident LayerNorm's per-row
# factor within 1.0e-7 of this model was right; the "bit-exact on the normal positive domain"
# generalisation from four inputs was the instrument error (see confidence())
LOG2_MEASURED_INPUTS = (1.0, 2.0, 4.0, 8.0, 0.5, 0.25, 1024.0, 2.0 ** -10)


def _log2_2570(x):
    """log2 in float32; op2570 was run alone on LOG2_MEASURED_INPUTS and returned exactly this."""
    import math
    return float(__import__("numpy").float32(math.log2(float(x))))


SEMANTICS = {3850: ("rsqrt", _rsqrt_3850, "the inputs 2.5, 1, 2, 4, 8 and the NaN/denormal cases"),
             2570: ("log2", _log2_2570, "the powers of two 1, 2, 4, 8, 0.5, 0.25, 1024 and 2^-10"),
             3978: ("rsqrt2", _rsqrt_3978, "the inputs 2.5, 1, 2, 4, 8 and the NaN/denormal cases"),
             1272: ("exp2", _exp2_1272, "executed on integer inputs"),
             3658: ("recip", _recip_3658, "executed on powers of two")}
# the input domain on which an opcode's execution record covers the value bit for bit
# The encode session (2.5, -3.75, two denormals), the sweep D3850.l10 / D3978.l10 (1, 2, 4, 8) and
# the isolated marginpairs cases on which the GPU word EQUALS this model's (0.5, 3.25 and the two
# extreme normals). 1.5 is measured too and deliberately absent: the GPU is one ulp low there, so
# listing it would make interpret() call a wrong value "executed". test_g17normcheck checks every
# listed input against the retained GPU word, which is what fails if 1.5 is ever added.
RSQRT_MEASURED_INPUTS = (2.5, 1.0, 2.0, 4.0, 8.0, 0.5, 3.25,
                         struct.unpack("<f", struct.pack("<I", 0x0ABCDEF1))[0],
                         struct.unpack("<f", struct.pack("<I", 0x5A5A5A5A))[0])


_EXACT_FROM_RECORDS = None


def _rsqrt_exact_inputs():
    """Every positive normal whose ISOLATED retained word equals this model, read from the records
    of op3850. Computed, not typed, so a new batch widens the domain and a wrong input cannot enter."""
    global _EXACT_FROM_RECORDS
    if _EXACT_FROM_RECORDS is None:
        _EXACT_FROM_RECORDS = frozenset(r[2] for r in isolated_rounding(3850) if r[5] == 0)
    return _EXACT_FROM_RECORDS


def _rsqrt_measured(x):
    x = float(x)
    return (x in RSQRT_MEASURED_INPUTS or x in _rsqrt_exact_inputs() or x != x or x < 0
            or (0 < x < FLOAT32_MIN_NORMAL))


# THE ISOLATED RECORDS, READ AS NUMBERS. The capability inventory cited u1272 as evidence for
# exp2 while its own gap said exp2 had never run alone; nothing had read the record's values
# against a reference. This does, for the three opcodes above, from the retained plan and result
# files. Only non-trivial inputs are kept (exp2: finite non-integers; rsqrt: positive normals), so
# a denominator of cases where every candidate is trivially exact cannot inflate the agreement.
ISOLATED_BATCHES = ("marginpairs", "residue", "sweep", "transsweep")


def _f32_bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def _correctly_rounded_bits(ref):
    """The nearest float32 to a float64 reference, or None when the reference is too close to a
    float32 midpoint for float64 to decide (the answer would then be a guess)."""
    import numpy as np
    c = np.float32(ref)
    cands = [np.nextafter(c, np.float32(-np.inf)), c, np.nextafter(c, np.float32(np.inf))]
    dist = sorted((abs(float(k) - ref), _f32_bits(float(k))) for k in cands)
    if dist[1][0] - dist[0][0] <= 4 * abs(ref) * 2.0 ** -52:
        return None
    return dist[0][1]


def _ordinal(bits):
    return bits if bits < 0x80000000 else -(bits & 0x7FFFFFFF)


def isolated_rounding(op, root=ROOT):
    """[(batch, record id, x, gpu bits, correctly-rounded bits, ulp)] for op1272/op3850/op3978."""
    rows = []
    for batch in ISOLATED_BATCHES:
        plan_path = os.path.join(root, "isa", "g17-execution-%s.json" % batch)
        with open(plan_path) as fh:
            plan = json.load(fh)
        with open(os.path.join(root, "isa", "g17-execution-%s-results.json" % batch)) as fh:
            results = {r["id"]: r for r in json.load(fh)}
        for p in plan:
            r = results.get(p.get("id"))
            if p.get("op") != op or not r or r.get("status") != "ok":
                continue
            for case, word in zip(p["cases"], r["values"]):
                x = struct.unpack("<f", struct.pack("<I", case[0] & 0xFFFFFFFF))[0]
                if not math.isfinite(x):
                    continue
                if op == 1272:
                    if x == int(x) or abs(x) < 2.0 ** -20:
                        continue
                    ref = 2.0 ** x
                else:
                    if not x >= FLOAT32_MIN_NORMAL:
                        continue
                    ref = 1.0 / math.sqrt(x)
                cr = _correctly_rounded_bits(ref)
                if cr is None:
                    continue
                rows.append((batch, p["id"], x, word, cr, _ordinal(word) - _ordinal(cr)))
    return rows


# A BOUND IS NOT A REFUSAL. Off its measured domain an executed opcode's value is not refuted, it is
# unmeasured; where a whole-program run has MEASURED how far the model can be from the hardware,
# that magnitude is reported as confidence "bounded" and an admission gate can compare it with its
# budget. Without such a measurement the verdict stays "isolation" and a gate refuses. The first
# version of the reconciliation reported every off-domain input as unverified, and the LayerNorm
# gate turned that into a refusal of results/g17-layernorm-hardware-1x4 - eight GPU queries behind
# it - because real rows never have variance exactly 4 (the linker's finding, 2026-09-10).
MEASURED_BOUND = {3850: (1.0e-7, "per-row relative deviation of the delivered LayerNorm's output from "
                                  "this model, measured on the resident attention block's 32 rows "
                                  "(results/g17-attention-resident-32x384-v1) and on the five-buffer "
                                  "run (results/g17-residualnorm-runtime-v1, 9.6e-8)")}
MEASURED_DOMAIN = {2570: lambda x: float(x) in LOG2_MEASURED_INPUTS,
                   1272: lambda x: float(x) == int(x) and 0 <= x <= 8,
                   3658: lambda x: x > 0 and (float(x) == 2 ** int(round(__import__("math").log2(x)))) and 1 <= x <= 8,
                   # CORRECTED 2026-09-10: "bit-exact on the normal positive domain" was the claim,
                   # and the resident LayerNorm's per-row factor differs from this model by up to
                   # 1.0e-7 on some rows - so bit-exact holds on the inputs actually run and the
                   # denormal/negative cases, and the rest of the domain is bounded, not exact.
                   3850: _rsqrt_measured, 3978: _rsqrt_measured}
RSQRT_OPCODES = {op: name for op, (name, _f, _s) in SEMANTICS.items()}
# A CONVERSION IS NOT A MEMBER OF SEMANTICS, AND KEEPING IT OUT IS DELIBERATE. Three sets are
# derived from SEMANTICS - RSQRT_OPCODES above, ARITH_FP below, and the Machine arm that reads its
# source with `self.rf` AS A FLOAT - so adding op11179 there would (a) call it a reciprocal square
# root in the report, (b) enroll it in the float-arithmetic set, and (c) read a uint32 bit pattern
# as a float, which is a silently wrong answer rather than a refusal. Its source is an INTEGER, so
# it gets its own table and its own arm, reading raw bits with `self.rd` exactly as BITWISE_REG does.
#
# WHAT THE MODEL RESTS ON, AND WHAT IT DOES NOT. Apple's decoder names op11179 cvt.i2f; Apple's own
# compiler emits it for `air.convert.f.f32.u.i32`; it appears 7,903 times in the build cache. All
# three are facts about ENCODING and SELECTION. None of them measures what the instruction computes,
# and op11179 has never been dispatched by this project - so `interpret_conversion` returns the
# confidence "candidate" and a caveat that says so, and `compare` treats a candidate as a BIT-EXACT
# question rather than a tolerance. That distinction is not cosmetic: at 2^24+3 a TRUNCATING
# conversion differs from a rounding one by 4 in 16,777,220, which is 1.2e-7 relative - inside the
# 1e-6 tolerance the isolation path would have applied, so the tolerance would have accepted the
# wrong arithmetic. A candidate model that cannot be refuted by its own comparison is not a model.
CONVERSIONS = {11179: ("cvt.u32.f32", lambda u: _to_f(_to_bits(float(u & 0xFFFFFFFF))),
                       "Apple's decoder's name, Apple's selection for air.convert.f.f32.u.i32, and "
                       "7,903 build-cache instances - all encoding, none of it execution")}
CONVERSION_LENGTHS = (10,)
CANDIDATE_CAVEAT = ("op%d (%s) is a CANDIDATE here: %s. This input is outside the measured set, so "
                    "a match would be the first evidence for it and a mismatch would refute the "
                    "model rather than the image; neither is assumed")
# THE INPUTS op11179 HAS ACTUALLY RUN ON, AND NOTHING WIDER. Root dispatched the unchanged Apple
# source syn-se8dbb86316 as this compiler's 86-byte program (sha 49e6a59c..., byte-identical to the
# prediction committed before the run) in two workers, 20 queries, 240 words checked, cross-worker
# identical: results/g17-source-conversion-runtime-v1 in the evidence archive. Every one of the nine
# frozen inputs produced exactly the predicted word.
#
# THE SCOPE IS THE FORM AND THE OPERAND, NOT THE OPCODE. The measured instances all carry operand 5
# = 16, so the key is (opcode, that operand): the same opcode with operand 5 = 0 or 32 has not run,
# and "proven of an encoding, not an opcode" is a mistake this project has already paid for. Nothing
# here measures what operands 1 or 5 MEAN either, because the program never reads the conversion's
# source again and a release would be invisible.
#
# FIVE INPUTS, NOT THE NINE THAT RAN, AND ROOT'S COUNTEREXAMPLE IS WHY. The stored word is the
# conversion followed by an op998 ADD, so a case whose addend is nonzero does not identify the
# conversion's intermediate bits - the add can absorb a wrong one. Root's example: float32's
# predecessor of 1.0 is 0.99999994039535522, and 0.99999994 + 0.5 rounds to the same 1.5 as
# 1.0 + 0.5, so the receipt for input 1 cannot certify that the conversion produced 1.0. Input 0 is
# out for a second reason: -0.0 + +0.0 is +0.0, so the receipt cannot establish the intermediate's
# SIGN. What survives is the five POSITIVE inputs whose addend is +0.0, and they survive only on the
# stated assumption that a binary32 add of +0.0 is the identity on a positive finite value.
#
# THE NARROWING COSTS NOTHING THAT WAS CLAIMED. 2^31 and UINT32_MAX (unsignedness) and 2^24+3
# (rounding, not truncation) are all inside the five. And an independent scan finds that the two
# large nonzero-addend cases are not actually degenerate at +-1 or +-2 ulp of their intermediate -
# so the exclusion of 2^24-1 and 2^24 is CONSERVATIVE rather than forced, and restoring them would
# still need an argument about the add rather than a scan of neighbours
# (test_g17conversioninterpret pins both halves of that).
CONVERSION_MEASURED = {
    (11179, 16): frozenset({2 ** 24 + 1, 2 ** 24 + 3, 2 ** 31 - 1, 2 ** 31, 2 ** 32 - 1})}
# the nine that RAN: composed agreement, which is a weaker and separate fact from the five above
CONVERSION_WORKLOAD = frozenset({0, 1, 2 ** 24 - 1, 2 ** 24, 2 ** 24 + 1, 2 ** 24 + 3,
                                 2 ** 31 - 1, 2 ** 31, 2 ** 32 - 1})
CONVERSION_RECEIPTS = ("results/g17-source-conversion-runtime-v1, two workers x 10 queries, 240 "
                       "words, program sha 49e6a59c; this input is one of the five whose addend is "
                       "+0.0 and whose value is positive, so the conversion's own bits follow from "
                       "the word given additive identity")


def interpret_conversion(opcode, bits, lifetime=None, executed=None):
    """(value, confidence, why) for an integer-sourced conversion.

    "executed" ONLY for an input this opcode has been run on in the form it was run in; "candidate"
    everywhere else, which is where a comparison can still refute the model.
    """
    executed = executed_opcodes() if executed is None else executed
    entry = CONVERSIONS.get(opcode)
    if entry is None:
        raise Unexecuted("op%d is not a conversion this checker models" % opcode)
    name, fn, source = entry
    value = fn(bits)
    measured = CONVERSION_MEASURED.get((opcode, lifetime))
    if measured is not None and (bits & 0xFFFFFFFF) in measured:
        return value, "executed", ("op%d (%s) named by RUNNING it on this input, operand 5 = %s "
                                   "(%s); the value model is this input's measured answer and "
                                   "nothing wider" % (opcode, name, lifetime, CONVERSION_RECEIPTS))
    if opcode in executed:
        return value, "executed", "op%d (%s) named by running it" % (opcode, name)
    return value, "candidate", CANDIDATE_CAVEAT % (opcode, name, source)

ISOLATION_CAVEAT = ("op%d is named by ISOLATION - by which opcode Apple selects for a construct - "
                    "and not by running it; %d opcodes are execution-named and this is not one. "
                    "Its rounding is unestablished and no bit-exact claim is available")


def in_normal_domain(x):
    x = float(x)
    return x == x and x >= FLOAT32_MIN_NORMAL


class Unexecuted(ValueError):
    """An opcode this checker has no measured meaning for."""


def interpret(opcode, *args, executed=None):
    """(value, confidence, why) - the value AND what it rests on, so a loader cannot mistake an
    isolation name for a measurement."""
    executed = executed_opcodes() if executed is None else executed
    entry = SEMANTICS.get(opcode)
    if entry is None:
        raise Unexecuted("op%d has no candidate meaning in this checker" % opcode)
    name, fn, source = entry
    value = fn(*args)
    covered = MEASURED_DOMAIN.get(opcode)
    if opcode in executed and covered is not None and not all(covered(a) for a in args):
        if opcode in MEASURED_BOUND:
            mag, how = MEASURED_BOUND[opcode]
            return value, "bounded", ("op%d (%s) executed only on %s; this input is outside that "
                                      "domain, and the model is within %.1e of the hardware: %s"
                                      % (opcode, name, source, mag, how))
        return value, "isolation", ("op%d (%s) executed only on %s; this input is outside that "
                                    "domain and its rounding is not measured" % (opcode, name, source))
    if opcode in executed:
        domain = ("normal positive input: both reciprocal-square-root opcodes agree with 1/sqrt(x) "
                  "here, so a bit-exact comparison is available"
                  if all(in_normal_domain(a) for a in args) else
                  "OUTSIDE THE NORMAL RANGE: op3850 flushes the input to zero and returns +inf "
                  "where op3978 returns 1.0, so which opcode the program selected decides the "
                  "answer and it cannot be assumed")
        return value, "executed", "op%d (%s) named by running it; %s" % (opcode, name, domain)
    return value, "isolation", ISOLATION_CAVEAT % (opcode, len(executed))


def arithmetic_status(opcode, executed=None):
    executed = executed_opcodes() if executed is None else executed
    if opcode not in SEMANTICS:
        return False, "op%d has no candidate meaning in this checker" % opcode
    if opcode in executed:
        return True, "op%d (%s) named by execution" % (opcode, SEMANTICS[opcode][0])
    return False, ISOLATION_CAVEAT % (opcode, len(executed))


def _bits(v):
    return struct.pack("<f", float(v))


def compare(got, want, confidence, *, rel=1e-6, domain_ok=True):
    """Bit-exact for an executed opcode; a bounded comparison for an isolation-named one.

    BIT-EXACT MEANS THE SAME THIRTY-TWO BITS. `==` calls +0.0 and -0.0 equal and two NaNs unequal;
    neither is what "the hardware produced these bits" means, so the comparison is on the packed
    float32 representation. Demanding bit-exactness from an instruction whose rounding nobody has
    measured would fail a correct image, and granting it would pass a wrong one - so an isolation
    confidence gets a stated bound instead, and the claim string says which was used.
    """
    if confidence == "candidate":
        # BIT-EXACT, AND THE CLAIM SAYS WHY. A conversion's candidate model is exact by
        # construction on every input - uint32 to binary32 round-to-nearest-ties-to-even is fully
        # specified - so what is open is whether the HARDWARE implements it, and only the same
        # thirty-two bits can answer that. A tolerance here would accept truncation.
        return (_bits(got) == _bits(want),
                "bit-exact against a CANDIDATE model that has never been run; a mismatch refutes "
                "the model rather than the image")
    if confidence == "executed":
        same = _bits(got) == _bits(want)
        if domain_ok:
            return same, "bit-exact on the measured domain (same float32 bits)"
        return same, ("bit-exact (same float32 bits), but the input is outside the normal range "
                      "where the two reciprocal-square-root opcodes DISAGREE - +inf against 1.0")
    g, w = float(got), float(want)
    if w == 0:
        return abs(g) <= rel, "within %g absolute (rounding unestablished)" % rel
    return abs(g - w) <= rel * abs(w), "within %g relative (rounding unestablished)" % rel


# ---------------------------------------------------------------- the register machine
FADD, FMUL = 998, 3290
LOAD32, STORE32 = 12682, 17229
READ_SR, MOVIMM, END = 14059, 11842, 684
IADD = {10279, 10280, 10282, 10288}          # add: imm and reg forms, and the width variants

# THE REGISTER-REGISTER BITWISE, FOUR-BYTE FORM. The delivered programs in
# results/g17-bitwise-common-v1 stop at their bitwise instruction with "not interpretable", which is
# what blocks their acceptance; these three opcodes are what they use. The operation comes from the
# OPCODE - Apple's decoder reads it out of the bytes, and g17asm's bitwise map agrees - so `step`
# does not need the raw bytes it is not given.
#
# UINT32 THROUGHOUT, AND THAT IS THE POINT OF THE RETAINED INPUTS. `alternating` carries 0xFFFFFFFF,
# 0xAAAAAAAA and 0x55555555, and the consumer seeds its output with the NaN payload 0x7FC01234.
# These arms read raw bits with self.rd and write a whole word, so no value passes through a float
# operation and no NaN payload is canonicalised on the way through.
BITWISE_REG = {424: lambda a, b: a & b, 13575: lambda a, b: a | b, 17771: lambda a, b: a ^ b}

# OPCODES WHOSE FUNCTION AN ISOLATED RECORD DETERMINED, admitted at exactly what was measured.
# (fit name in g17fitfromexecution, sources, length, lead modifiers seen on the determining records).
# The function is taken FROM the fit library - one definition, so a correction there reaches here -
# and every other length, modifier or source lifetime refuses, as the other arms do. Added
# 2026-09-22 for the form-receipt campaigns (tools/g17formreceipt.py), whose arithmetic admission
# otherwise stopped at "is not interpretable by this checker".
MEASURED_ALU = {
    13460: ("bitwise_7", 2, 10, frozenset({0, 2147483680})),
    13488: ("bitwise_2", 2, 10, frozenset({143082540498944, 2147483680})),
    13521: ("bitwise_1", 2, 10, frozenset({0, 32})),
    13548: ("bitwise_D", 2, 10, frozenset({0, 32})),
    17744: ("bitwise_9", 2, 10, frozenset({0, 2147483648})),
    16806: ("sar", 2, 12, frozenset({143082540498944})),
    3770: ("frint", 1, 10, frozenset({2147483648})),
    3786: ("ffloor_denormal_to_zero", 1, 10, frozenset({2147483648})),
    3802: ("fceil", 1, 10, frozenset({2147483648})),
    # op1062 WAS HERE as fsat, admitted on compiled.op1062 - a ONE-THREAD record. op1062 is
    # cross-lane (each lane pair gets clamp(x[2k] - x[2k+1], 0, 1)), so a per-lane model of it is
    # wrong on every multi-lane program and it now refuses. fsat is op904 (FSAT_904 below).
    # ledger/g17-fsat-is-op904-and-op1062-is-cross-lane.toml
}
MEASURED_ALU_SOURCE_MODIFIERS = {}
MEASURED_ALU_EVIDENCE = {}
# op3818 is NOT here: an earlier arm already interprets it as a scoped candidate (finite values exactly
# representable as binary16), and two models of one opcode would disagree silently. Its receipt uses
# inputs inside that scope instead.
MEASURED_ALU_SOURCE_LIFETIMES = frozenset({16})

# THE FLOAT ADD-IMMEDIATE, op1000 at 12 bytes: [dst, lead, src, life, code]. Measured alone at
# immediate code 4 (+0.0625) with both leads below, on 1, 2, 4 and 8 (isa/g17-execution-debtbits,
# -debtbits2 and -union results: 1.0625, 2.0625, 4.0625, 8.0625). Every other code refuses: the
# eight-bit immediate spells 256 values and one was run.
FADDI_LEADS = frozenset({143082540498944, 68719476736})
FADDI_CODES = frozenset({4})
# op904/12, THE SATURATING FLOAT ADD-IMMEDIATE Apple emits for saturate(x) and clamp(x, 0, 1):
# [dst, lead, src, life, code]. Interpreted only as Apple writes it - lead 2^31, code 128 (-0.0) -
# which is fsat, and which isa/g17-execution-fsat-results.json ran on 32 distinct lanes.
FSAT_904_LEADS = frozenset({2147483648})
FSAT_904_CODES = frozenset({128})

# THE LAYOUT AND THE MODIFIERS ARE CENSUSED, NOT ASSUMED. Over Apple's 6,594-program corpus:
#
#     four-byte    2,188 instances, ALL of them three register/immediate pairs, dest first
#     ten-byte       910 instances, of which 345 are three register/immediate pairs
#
# and across all 2,533 of those, the LOW BYTE of each immediate falls in these sets and nowhere
# else (four-byte counts + ten-byte counts):
#
#     position 0 (destination)   {0: 860+291, 32: 1328+54}
#     position 1 (first source)  {0: 80+1,    16: 1581+159, 32: 527+185}
#     position 2 (second source) {0: 203+1,   16: 1792+322, 32: 193+22}
#
# 16 is release and 32 is keep (see the lifetime-operand family); they are liveness, not arithmetic,
# so the value ignores them. What matters is what is ABSENT: negate (2) and absolute value (4)
# appear on none of the 2,533, and either WOULD change the result. So a low byte outside the
# censused set is refused rather than ignored - the alternative is a checker that quietly computes
# a & b for an instruction that means a & ~b.
BITWISE_REG_MODIFIERS = (frozenset({0, 32}), frozenset({0, 16, 32}), frozenset({0, 16, 32}))

# THE TEN-BYTE FORM PUTS SOMETHING ELSE IN THE DESTINATION OPERAND'S HIGH BITS. 296 of its 345
# three-register instances carry one bit above the low byte, always exactly one, always from
# {24..31, 36}, and always on the DESTINATION: the two source operands carry nothing above their
# low byte in any of the 345. That is the per-instruction index family, which the lifetime operand
# is already known to share high bits with. It rides on the write target, so it cannot change the
# value being computed, and accepting it is what takes this arm from 49 of Apple's 345 to all 345.
# A source carrying the same bit is a combination never measured and is refused - the two cases
# differ, and treating them alike in either direction would be guessing.
BITWISE_REG_INDEX_BITS = frozenset({24, 25, 26, 27, 28, 29, 30, 31, 36})

# THE MAJORITY OF THE TEN-BYTE POPULATION IS NOT INTERPRETED HERE, and it says so rather than
# reading around it: 565 of the 910 take an `expr` where this arm needs a register - 502 at source
# B, 61 at source A, 2 at both. An expression operand is a different operand kind, not a register
# this checker can read, and no delivered program with an independent reference uses one.
BITWISE_REG_LENGTHS = (4, 10)

# THE VECTOR STORE THROUGH AN INDEX REGISTER (op17256/8), which is what an automatic spill emits.
# Unlike the range store, whose address is a fixed displacement, this one addresses
# `4 * <index register> + disp/4 + i` - that index register is the whole reason a spill can have a
# per-thread slot at all (handoff 10ae; g17asm.encode_vec4 takes it in both directions).
#
# THE IMMEDIATES ARE NEARLY CONSTANT ACROSS APPLE'S EIGHT INSTANCES, all eight bytes and all one
# shape: imm0=16, imm1=2290, imm2=0, imm3=16, imm4=0, imm5 in {4, 16}. A small population is a weak
# one, so this arm refuses anything outside it rather than treating the fields as free - eight
# instances cannot tell us what a ninth value would mean.
# THE WIDENING MULTIPLY (op10793/12), whose destination is a register PAIR. One instruction writes
# the full 64-bit product across two consecutive registers, so this arm defines BOTH.
#
# WHICH REGISTER HOLDS THE HIGH WORD IS NOT ESTABLISHED ON SILICON, and this checker does not get to
# decide it - it implements the reading the retained evidence favours and says so. Across Apple's
# 654 corpus instances, when only one register of the pair is read afterwards it is the SECOND in
# 94 cases against the first in 28, and the consumers differ in kind: the second feeds add, madd,
# store and sub, while all 28 of the first's are a single `mov`. So: first = low, second = high.
# A program that stores both halves is what would settle it, and g17nextform.control_program() is that program.
MULHI = 10793
MULHI_HALVES = "first=low, second=high (favoured by 94 vs 28 single-half uses; NOT silicon)"
# THE INDEXED VECTOR LOADS (op12691 two words, op12709 four, at 8 and 14 bytes), cc's load_vec_at:
# tuple lane i <- word n * index + disp/4 + i of the binding, n the component count (ir.load_vec_at).
# The opcode fixes n and the mask must agree with it (RANGE_MASK). n = 3 (op12700) is refused: a
# uint3 element's stride is not measured. Interpreted only with the immediates cc emits: lead
# 8388608 at eight bytes and 137464119296 at fourteen (the form cc uses when the index comes straight
# from read_sr), then 0, the index lifetime, 0, and the element stride in bytes (4n).
VEC_LOADS = {12691: 2, 12700: 3, 12709: 4}


def _vec_access_bytes(n):
    """4n bytes rounded up to a power of two - asm.vec_access_bytes' rule, restated here so this
    tool does not import the package (test_g17librarycompat's anchor rule)."""
    return next(s for s in (1, 4, 8, 16) if s >= 4 * n)
VEC_LOAD_LEADS = {8: frozenset({8388608}), 14: frozenset({137464119296})}
VEC_STORE = 17256
VEC_STORE_IMMS = (frozenset({16}), frozenset({2290}), frozenset({0}), frozenset({16}),
                  frozenset({0}), frozenset({4, 16}))
IMUL_REG, IMUL_IMM = 10825, 10822
SHL_IMM = 14391                              # shift left by an immediate: [dst, 0, 0, src, life, amount, 32]
SAR_IMM = 16805                              # arithmetic shift right by an immediate (lane.form_sar5)
SHR_IMM = 17013                              # shift right by an immediate, the same layout (executed: shiftr kernel)
ISUB_REG = 11667                             # dst = a - b: [dst, 0, a, life, b, life] (executed: subreg kernel)
FSELECT, FMA = 9700, 2190
# op9700's slot 2 is Apple's operation field for max()/min() (sw-f_max 7, sw-f_min 3): NAMED BY
# APPLE'S SELECTION of the instruction for those constructs, never executed by this backend.
# What it returns on a NaN operand is not established, so a NaN operand is refused rather than read.
FSELECT_OPS = {7: ("fmax", max), 3: ("fmin", min)}
# A FLOAT SOURCE CARRIES A MODIFIER, NOT JUST A LIFETIME. Apple's decoder prints it as the
# immediate after the register: 16 releases and 32 keeps; +2 negates and +4 takes the absolute
# value (ledger/g17-float-source-modifiers.toml: f[x] - f[y] is op998 with 18 on its second
# source). An interpreter that reads the register and ignores the immediate computes a + b for a
# - b. Only 16 and 32 are accepted; anything else is refused by name, because no program with
# hardware evidence carries another value.
SOURCE_LIFETIMES = {16, 32}
# THE LEADING IMMEDIATE OF A FLOAT INSTRUCTION is its operation modifier (rounding, saturation,
# the wait bit), and a value is accepted only where a program carrying it ran and its results were
# checked: the LayerNorm (56655bb5, eight-query receipts) carries both fadd and both fmul values,
# the query campaign the fma value, the softmax run the fselect, exp2 and recip values.
LEAD_MODIFIERS = {FADD: {148176371712, 146028888064, 32}, FMUL: {137438953472, 139586437120, 32},
                  FMA: {10737418240, 8589934592, 32}, FSELECT: {2147483648}, 1272: {32}, 3658: {32},
                  3850: {2147483648}, 3978: {2147483648, 32}, 2570: {2147483680}}
# WHY EACH ADMITTED VALUE IS ADMITTED, as data rather than as a comment, because the two FMA values
# are admitted on very different strengths of evidence and a set cannot say so. An opcode with no
# entry here is admitted by the older receipts described above.
#
# AN ENCODING COUNT IS NOT AN EXECUTION COUNT, and this table keeps them in separate fields. The
# 670/43 figure for 10737418240 came from DECODING 399 retained program*.bin images; it says how
# often the value is emitted, not that those programs ran or that anything checked their output.
# What makes that value executed evidence is a receipt, and the cleanest one is the control arm of
# the paired run below, where all 148 FMAs carry it.
#
# FMA 8589934592 WAS REFUSED UNTIL 2026-09-13 AND THE REFUSAL WAS RIGHT AT THE TIME. The only
# retained program carrying it was g17-halfstore-wait-runtime-v1/half_long, whose finding reads "all
# observed output words are zero rather than the expected half values" - a failure, and a confounded
# one, because that program also emitted fma(x, 1.4e-45, 0.0) for fma(x, 1.0, 0.0). THAT RECEIPT IS
# NOT SUPERSEDED AND IS NOT REINTERPRETED: it is retained, it is still a failure, and half_long's
# cause is still unknown. What changed is that it is no longer the only receipt carrying the value -
# a new paired execution carries it in an arm whose every checked word came back correct.
LEAD_MODIFIER_EVIDENCE = {
    (3978, 32): dict(
        strength="isolated execution",
        execution_evidence="isa/g17-execution-compiledunary-results.json compiled.op3978 - the compiled "
                           "rsqrt2 instance (lead 32), registers only rewritten: 1/sqrt(x) bit-exact on "
                           "1, 4, 16, 0.25, 1/16, 64, 1/64, 256; rivals recip, sqrt, identity rejected.",
        scope="op3978 at 10 bytes, those inputs; rounding elsewhere is governed by MEASURED_DOMAIN.",
        confound_resolved=None),
    (2570, 2147483680): dict(
        strength="isolated execution",
        execution_evidence="isa/g17-execution-compiledunary-results.json compiled.op2570 - the compiled "
                           "log2 instance: log2 exact on 1, 2, 4, 8, 0.5, 0.25, 1024, 2^-10; rivals ln, "
                           "log10, exp2 rejected.",
        scope="op2570 at 10 bytes, exactly those inputs (LOG2_MEASURED_INPUTS).",
        confound_resolved=None),
    # 32 ON THE FOUR-BYTE FORMS, admitted 2026-09-22 from isolated execution. At 4 bytes operand 1 is
    # the destination lifetime (tools/g17fmul4.py: "dest lifetime 32 at b2[5]"), and 32 - keep - is
    # what Apple's dominant 4-byte instances carry. The admission table is keyed by OPCODE, so this
    # also admits 32 at the other lengths, where no record measured it; that is stated rather than
    # hidden, and a 14-byte instance carrying 32 has not been seen in anything this compiler emits.
    (FMUL, 32): dict(
        strength="isolated execution",
        execution_evidence="isa/g17-execution-shortforms2-results.json op3290.at4 - Apple's own 4-byte "
                           "instance, operand 1 = 32, registers written by the assembler author; "
                           "f32 multiply bit-exact on 6 cases x 3 runs, rivals (fadd, f16 variants) "
                           "rejected, beside a same-run 14-byte baseline.",
        scope="op3290 at 4 bytes only; the value's meaning at 14 bytes is not measured.",
        confound_resolved=None),
    (FADD, 32): dict(
        strength="isolated execution",
        execution_evidence="isa/g17-execution-shortforms2-results.json op998.at4 - operand 1 = 32, "
                           "f32 add bit-exact on 6 cases x 3 runs, beside a same-run baseline.",
        scope="op998 at 4 bytes only; the value's meaning at 6 and 12 bytes is not measured.",
        confound_resolved=None),
    # 32 ON THE FOUR-BYTE FMA, admitted 2026-09-23. At 4 bytes operand 1 is the destination
    # lifetime, as on op3290/4 and op998/4, and op2190/4 is TWO-ADDRESS: the accumulator is the
    # destination printed again. The interpreter admits 32 only at 4 bytes and only with the
    # accumulator printed at operand 6 - the one shape executed - and refuses it anywhere else.
    (FMA, 32): dict(
        strength="isolated execution",
        execution_evidence="isa/g17-execution-ffma4-results.json ffma4.op2190_4 - op2190/4 as cc emits it "
                           "(590d3a0f, accumulator tied to the destination), 32 lanes single-rounded "
                           "a*b+c, 0 of 16 rounding-witness lanes double-rounded "
                           "(ledger/g17-ffma4-tie-and-one-rounding.toml).",
        scope="op2190 at 4 bytes with the accumulator at operand 6, and at 6 bytes three-address: "
              "isa/g17-execution-ffma6-results.json ffma6.op2190_6 (890b3e0d800e shape), 32 lanes "
              "single-rounded through the waiting copy, the unwaited control 0 of 32.",
        confound_resolved=None),
    (FMA, 10737418240): dict(
        strength="executed receipts",
        execution_evidence="results/g17-source-literals-runtime-v2/control - all 148 op2190/16 "
                           "instructions of the 4758-byte control 3c2aece5 carry this value; 10 "
                           "queries over 2 workers, 320 words, every one bit-exact against the "
                           "independent reference. The earlier query campaign also carries it.",
        encoding_inventory=dict(
            instances=670, experiments=43, decoded_program_files=399,
            note="obtained by DECODING retained program*.bin images, not by reading execution "
                 "receipts; an emission count, which is why it is not the execution field"),
        scope="byte0[3] SET, which is what an FMA consuming a load carries.",
        confound_resolved=None),
    (FMA, 8589934592): dict(
        strength="one paired execution",
        campaign="results/g17-source-literals-runtime-v2",
        candidate_code_sha256="a4a0e700ca8b9f72eb83510b7133244811383b34029104d6ecd3c247b22f0188",
        control_code_sha256="3c2aece5534c57969534c4a56b2a163970e0ed8b8d48642331f2f87832f5213c",
        program_bytes=4758, instances=147, of_fma_instructions=148,
        arms=2, workers_per_arm=2, queries_per_arm=10, checked_words=640,
        cases=("ramp", "alternating", "wide", "fractional"),
        ordering="control-before-candidate is an enforced and recorded PRECONDITION, not a "
                 "chronology: each candidate receipt carries the sha256 of both completed control "
                 "receipts (experimental_prerequisite.control_receipt_sha256) and the committed "
                 "runner computes that prerequisite before it constructs the worker. There are no "
                 "wall-clock query timestamps, and worker pid magnitude establishes nothing.",
        prior_failed_receipt="results/g17-halfstore-wait-runtime-v1/half_long",
        prior_failed_receipt_explained=False,
        confound_resolved=False,
        scope="The retained Metal source sl32-u148 as this compiler's 4758-byte program a4a0e700, "
              "147 of its 148 FMAs carrying this value, run against an otherwise byte-identical "
              "control (3c2aece5) differing only in byte0[3] of those 147. Every one of 640 output "
              "words is bit-exact against a reference computed by exact rational fused arithmetic, "
              "independent of AIR, the IR and this decoder; the repeated case matches across "
              "workers. This is an executed CONFIGURATION - a pure FMA dependency chain after one "
              "waited load, at these inputs, this launch and this device. It is NOT an arithmetic "
              "or scheduling law: nothing here measures what byte0[3] does for an operand that did "
              "come from a load, for a differently shaped chain, or at another length. The bit's "
              "meaning is causally measured only on the alu.12 family "
              "(ledger/g17-alu-load-use-wait.toml), and "
              "ledger/g17-only-one-alu-family-waited-for-a-load.toml records that one family."),
}


def _float_operands(opcode, toks, offset, sources):
    """The `sources` registers of a float instruction, after refusing an unmodelled modifier."""
    regs = _regs(toks)
    # THE SYNTHETIC LAYOUT CARRIES NO MODIFIERS. The admission tool's controls write float ops as
    # [dst, imm:0, a, b] - no lifetime after a source - and are told apart by that shape, as the
    # memory forms' two layouts are; the delivered layout has an immediate after every source.
    positions = [i for i, t in enumerate(toks) if t.startswith("reg:")]
    delivered = all(i + 1 < len(toks) and toks[i + 1].startswith("imm:") for i in positions[1:])
    if not delivered:
        if len(regs) < sources + 1:
            raise Unexecuted("op%d at +%#x names %d registers; %d expected" % (opcode, offset, len(regs), sources + 1))
        return regs
    lead = toks[1] if len(toks) > 1 and toks[1].startswith("imm:") else None
    if opcode in LEAD_MODIFIERS and (lead is None or int(lead[4:]) not in LEAD_MODIFIERS[opcode]):
        raise Unexecuted("op%d at +%#x carries operation modifier %s, which no executed program "
                         "carries; its meaning is not modelled. Admitted here: %s"
                         % (opcode, offset, lead,
                            ", ".join(str(v) for v in sorted(LEAD_MODIFIERS[opcode]))))
    for i in positions[1:]:
        t, nxt = toks[i], toks[i + 1]
        if True:
            if int(nxt[4:]) not in SOURCE_LIFETIMES:
                raise Unexecuted("op%d at +%#x source %s carries modifier %s - a negation or an "
                                 "absolute value, or a lifetime this checker does not know - and "
                                 "reading the register alone would compute the wrong operation"
                                 % (opcode, offset, t, nxt))
    if len(regs) < sources + 1:
        raise Unexecuted("op%d at +%#x names %d registers; %d expected" % (opcode, offset, len(regs), sources + 1))
    return regs
STORES32 = {17229, 17235}
# The word twin of the half-element slot store, and the element width it writes. Only its EIGHT
# byte length is modelled: g17asm records ten-byte (the value waits on a load) and fourteen-byte
# (the displacement does not fit in eight bits) forms of the same opcode, and neither has been
# measured here, so both still refuse by name.
WORD_SLOT_STORE = 17235
WORD_BYTES = 4
# THE RANGE STORES: n consecutive 32-bit registers to n consecutive slots. The decoder prints the
# source as ONE register of a tuple file - pair 2451+r, triple 2594+r, quad 2736+r, r the first
# 32-bit register - measured by compiling range stores at a known first register (r5 -> 2456,
# 2599, 2741). The mask word names n (2098, 2162, 2290); the displacement is the slot in bytes.
RANGE_STORES = {17244: 2, 17253: 3, 17262: 4}
TUPLE_BASE = {2: 2451, 3: 2594, 4: 2736}
RANGE_MASK = {2: 2098, 3: 2162, 4: 2290}
# THREADGROUP MEMORY AND THE BARRIER, as the executed kernels carry them (tgbarrier2, threadgroup,
# tgself: isa/g17-endtoend-results.json): the store op13288 [value, life, mode, expr(base), 0,
# index, life, 0, elem], the load op12364 [dest, wait, mode, expr(base), 0, index, life, 0, elem],
# the index in the 16-bit file - printed 425+n, the low half of the 32-bit register printed 105+n
# (ledger: the imageblock prologue) - and op447 the threadgroup barrier [0, 276].
TG_STORE, TG_LOAD, BARRIER = 13288, 12364, 447
R16 = 425
ARITH_FP = {FADD, FMUL, FSELECT, FMA, 1004, 1016, 11372, 11375, 14392, 17014, 17770, 14047, 9986} | set(SEMANTICS)
# the single-lane control flow the run loop interprets: compare-immediate, mask, back edge, restore
CMP_IMM, EXEC_MASK, BACK_EDGE, EXEC_RESTORE = 10369, 582, 458, 577
CONTROL = {CMP_IMM, EXEC_MASK, BACK_EDGE, EXEC_RESTORE}
TENSOR_SETUP_SHIFT = 17016
TENSOR_SETUP_LOGIC = {423, 426}
R16H = 281                                   # printed base of the HIGH 16-bit file (imageblock ledger: 281+n is r(105+n).hi)
# THE TENSOR LANE-ADDRESS PREFIX (integration's assignment, handoff 9h): the 4-byte read_sr of
# SR130 (the lane within its SIMD group), the two-view adds op10286 (u16 + u16, measured wide on
# the widen record) and op10283 (32-bit + u16 view: executed on small values, the view's width
# above 65535 unmeasured - the probe g17tensorwidthprobe is for it), the four-byte tensor init
# op554, the move op586, and the fused shift-add SCALE of every add, which the decoder prints in
# bits 8..10 of the add's last immediate (token >> 8 = 0 x1, 1 x2, 2 x4, 3 x8, 4 x16 - 986 of 986
# adds across the stride witnesses agree with g17asm's causal scale table).
READ_SR_LANE, TENSOR_INIT, MOV32 = 14060, 554, 586
ADD_U16_U16, ADD_32_U16 = 10286, 10283
SCALED_ADDS = {10279, 10282, ADD_U16_U16, ADD_32_U16}
# the seven (a, b) pairs integration executed through the exact +78 op10283 and +66 op10286 forms
# (tools/g17tensorwidthprobe.PAIRS; results/g17-tensor-width-runtime-v1): both forms matched their
# models on every pair in three queries
WIDTH_MEASURED_PAIRS = {(3, 5), (0x12345678, 0x0F0F0F0F), (0xFFFFFFFF, 1), (0x00010000, 0x00010000), (65535, 65535), (0x80000000, 0x0001FFFF), (7, 0x00020003)}
STRUCTURAL = {590, 555, 12646, 17193, READ_SR, MOVIMM, LOAD32, END, IMUL_REG, IMUL_IMM, SHL_IMM, SHR_IMM, SAR_IMM, ISUB_REG, TG_STORE, TG_LOAD, BARRIER, TENSOR_SETUP_SHIFT, READ_SR_LANE, TENSOR_INIT, MOV32, ADD_U16_U16, ADD_32_U16} | TENSOR_SETUP_LOGIC | IADD | STORES32 | set(RANGE_STORES) | CONTROL
R32 = 105                                    # printed base of the 32-bit register file


def _regs(toks):
    return [t for t in toks if t.startswith("reg:")]


def _imms(toks):
    return [int(t[4:]) for t in toks if t.startswith("imm:")]


@functools.cache
def _host_fmaf():
    """Resolve the host's single-rounding operation once, not per instruction."""
    import ctypes
    function=ctypes.CDLL(None).fmaf
    function.argtypes=(ctypes.c_float,)*3
    function.restype=ctypes.c_float
    return function


def _element_mask(toks):
    """The access mask word: the immediate immediately before the expr token that names the rank."""
    for i, t in enumerate(toks):
        if t.startswith("expr:bin(op0,const(") and i and toks[i - 1].startswith("imm:"):
            return int(toks[i - 1][4:])
    return None


def _mask_element(toks, offset, what):
    """Which element of the access window this instruction touches, from bits 4..7 of that word.

    MEASURED, and cross-checked against a number this file already uses. Compiling the same load at
    displacements 0..3 - the range the compiler itself states for the field, "byte4[6:5] holds
    0..3" - moves only that nibble, one-hot:

        disp 0 -> 0x1     disp 1 -> 0x2     disp 2 -> 0x4     disp 3 -> 0x8

    and the range stores' masks already recorded above are the same nibble for n CONSECUTIVE
    elements: 2098 -> 0x3, 2162 -> 0x7, 2290 -> 0xf. So the nibble is the set of elements the
    access touches, and a single-element access names its element by the one bit that is set.

    This is why two loads one element apart both read element 0 here before: the byte displacement
    this checker reads is zero for them, and the element they differ by is in this nibble.
    """
    # ONLY THE DELIVERED, REGISTER-INDEXED LAYOUT CARRIES A MASK. In the four-token control layout
    # - [reg, width, expr, index] - the immediate before the expr is the WIDTH, and reading it as a
    # mask makes a 4-byte element look like an access that names no element at all. That layout
    # reaches the constant-index branch rather than this one, but a helper that is only safe
    # because of where it is called from is one refactor away from being wrong.
    if len([t for t in toks if t.startswith("reg:")]) < 2:
        return 0
    word = _element_mask(toks)
    if word is None:
        return 0                                    # no mask word: the address is the whole story
    nibble = (word >> 4) & 0xF
    if nibble == 0:
        raise Unexecuted("%s at +%#x names no element in its access mask (%#x)"
                         % (what, offset, word))
    if nibble & (nibble - 1):
        raise Unexecuted("%s at +%#x touches %d elements (mask %#x); this checker models the "
                         "single-element access, and the range forms have their own model"
                         % (what, offset, bin(nibble).count("1"), nibble))
    return (nibble & -nibble).bit_length() - 1


def _rank(toks):
    for t in toks:
        if t.startswith("expr:bin(op0,const("):
            return int(t.split("const(")[1].split(")")[0]) // 4
    return None


def _f32(v):
    import numpy as np
    return np.float32(v)


# REGISTERS HOLD THIRTY-TWO BITS AND THE OPCODE DECIDES THE VIEW. The workload materialises its
# float constants with movimm - -1.0 arrives as the integer 0xBF800000 - and a machine that kept
# Python floats read that register as 3,212,836,864.0, drove every shifted value to overflow, and
# reported an output equal to beta alone. So every register is a uint32 bit pattern: an integer
# opcode computes on it as an integer, a float opcode reinterprets it as float32, computes in
# float32, and stores the result's bits back. That is also what makes "bit-exact" a statement
# about bits rather than about Python equality.
class _RawSignalingWord(float):
    """Memory bits whose Python-float conversion would quiet a signaling NaN.

    Integer loads/stores must preserve these bits. Numeric arithmetic still sees
    a float and its result follows the ordinary floating-point conversion path.
    """
    def __new__(cls, value, bits):
        obj=super().__new__(cls,value)
        obj.raw_bits=bits
        return obj


def _to_f(bits):
    bits=int(bits)&0xffffffff
    value=struct.unpack("<f", struct.pack("<I",bits))[0]
    if bits&0x7f800000==0x7f800000 and bits&0x003fffff and not bits&0x00400000:
        return _RawSignalingWord(value,bits)
    return value


def _to_bits(f):
    import numpy as np
    if isinstance(f,_RawSignalingWord):return f.raw_bits
    return struct.unpack("<I", struct.pack("<f", float(np.float32(f))))[0]


class Machine:
    """Runs one thread of a delivered program over concrete buffers. Refuses what it cannot read.

    Two decoded layouts are read for a memory form. The DELIVERED one has an index register and a
    trailing element size: [dst, mode, width, expr, 0, index, life, displacement, elem]. The
    four-token layout the admission tool's synthetic controls use - [reg, width, expr, index] -
    carries the element index as its last immediate and no register; it is accepted so that the
    controls run, and it is told apart by the register count rather than guessed.
    """

    def __init__(self, bindings, buffers, row, executed, silicon, waw_model="in_order"):
        self.outside_domain = {}
        # bindings: [(index, offset, written), ...] in RANK order; buffers: {rank: sequence}
        self.bindings, self.buffers, self.row = list(bindings), buffers, row
        self.executed, self.silicon = executed, silicon
        self.regs, self.stores, self.loads = {}, [], []
        # A THREAD SEES ITS OWN STORES. simulate_threads drains `stores` into the shared buffers
        # only after the thread finishes, so a store followed by a load of the same address inside
        # ONE thread read the buffer's pre-thread contents - and a spill is exactly that sequence.
        # Measured on the automatic spill: the four spilled values reached scratch words 0..3
        # correctly and every reload still read the initial value, so the program was right and the
        # model was not.
        #
        # THE OVERLAY IS PER-THREAD AND CROSS-THREAD VISIBILITY IS UNCHANGED: this makes a thread
        # see what it wrote, and nothing else sees it any sooner than it did before. For a program
        # whose threads never read an address they wrote - every delivered program to date - the
        # overlay is never consulted and no result moves; a test asserts that rather than assuming
        # it.
        self.own_stores = {}                   # (rank, addr) -> value written by THIS thread
        self.defined_halves = {}
        self.range_store_releases = {}
        self.used = set()
        # THE HARDWARE-ORDERING MODEL, for the discriminating control. "in_order" is the CPU
        # machine: a later write wins. "earlier_wins" models the write-after-write the first
        # full-width GPU run exhibited: a load into a register whose previous load was never read
        # leaves the EARLIER value in place. A program with no unread load destinations behaves
        # identically under both, which is what makes the two models a control.
        self.waw_model = waw_model
        self.unread_load = set()          # registers holding a load result nobody has read yet

    def _view(self, tok):
        """(32-bit token, half) for a printed register: 425+n is r(105+n).lo, 281+n its .hi."""
        n = int(tok[4:])
        if R16 <= n < R16 + 128: return "reg:%d" % (R32 + n - R16), "lo"
        if R16H <= n < R16H + 128: return "reg:%d" % (R32 + n - R16H), "hi"
        return tok, None

    def _write_word(self, tok, value):
        """Define a whole register without changing instruction-specific load tracking."""
        self.regs[tok] = value & 0xFFFFFFFF
        self.defined_halves[tok] = {'lo','hi'}

    def wr(self, tok, value):
        """Write a register; a 16-bit view writes its half of the 32-bit register and DEFINES only
        that half (integration's review of e41c28b7: supplying zero for the other half would be an
        initialisation assumption, not measured preservation). A 32-bit write defines both."""
        base, half = self._view(tok)
        halves = self.defined_halves
        if half is None:
            self._write_word(tok,value)
        else:
            defined = (set(halves.get(base, {'lo','hi'})) if base in self.regs else set())
            old = self.regs.get(base, 0)
            self.regs[base] = ((old & 0xFFFF0000) | (value & 0xFFFF)) if half == "lo" else ((old & 0xFFFF) | ((value & 0xFFFF) << 16))
            halves[base] = defined | {half}
        self.unread_load.discard(base)

    def rd(self, tok):
        base, half = self._view(tok)
        halves = self.__dict__.setdefault("defined_halves", {})
        if half is not None:
            if base not in self.regs:                     # the 32-bit path's order: a released register first
                if base in self.range_store_releases:
                    raise Unexecuted('read of %s after potentially releasing range store at +%#x; '
                                     'the synchronous model cannot assume its old value survives'
                                     % (tok, self.range_store_releases[base]))
                raise ValueError("read of uninitialised %s (%s)" % (tok, base))
            if base in halves and half not in halves[base]:
                raise ValueError("read of %s: the %s half of %s was never written" % (tok, half, base))
            self.unread_load.discard(base)
            v = self.regs[base]
            return v & 0xFFFF if half == "lo" else (v >> 16) & 0xFFFF
        if tok in self.regs and tok in halves and halves[tok] != {"lo", "hi"}:
            raise ValueError("32-bit read of %s after only its %s half was written" % (tok, "/".join(sorted(halves[tok]))))
        if tok not in self.regs:
            if tok in self.range_store_releases:
                raise Unexecuted('read of %s after potentially releasing range store at +%#x; '
                                 'the synchronous model cannot assume its old value survives'
                                 % (tok, self.range_store_releases[tok]))
            raise ValueError("read of uninitialised %s" % tok)
        self.unread_load.discard(tok)
        return self.regs[tok]

    def rf(self, tok):
        return _f32(_to_f(self.rd(tok)))

    def _release_view(self, token):
        """Invalidate the consumed view; never predict the bits hardware returns after release."""
        base, half = self._view(token)
        if half is None:
            self.regs.pop(base,None);self.defined_halves.pop(base,None)
        else:
            valid=set(self.defined_halves.get(base,{'lo','hi'}))
            valid.discard(half)
            self.defined_halves[base]=valid
        self.unread_load.discard(base)

    def _access(self, toks, offset, what, *, opcode=None, size=None):
        rank = _rank(toks)
        if rank is None or rank >= len(self.bindings):
            raise ValueError("%s at +%#x names binding rank %s, which is not in the delivered "
                             "contract of %d bindings" % (what, offset, rank, len(self.bindings)))
        regs, imms = _regs(toks), _imms(toks)
        if len(regs) >= 2:                                  # the delivered, register-indexed layout
            elem = imms[-1]
            if elem != 4:
                raise ValueError("%s at +%#x has element size %d, not the FP32 4" % (what, offset, elem))
            disp = imms[-2] if len(imms) >= 2 else 0
            if disp % elem:
                raise ValueError("%s at +%#x has a %d-byte displacement that is not element-aligned"
                                 % (what, offset, disp))
            addr = (int(self.rd(regs[1])) + disp // elem
                    + _mask_element(toks, offset, what))
        elif opcode == WORD_SLOT_STORE and size == 8 and len(toks) == 8:
            # THE WORD SLOT STORE, op17235 at eight bytes: ONE 32-bit element at an immediate BYTE
            # DISPLACEMENT. g17asm calls it the word twin of the half-element store op17199 - "the
            # SAME displacement field and the same length rule", the word store's field being the
            # same physical bits with the low two always zero - and the range stores in this file
            # already read a displacement the same way ("the displacement is the slot in bytes").
            #
            # WHICH TOKEN IS WHICH WAS MEASURED, NOT READ OFF THE SHAPE. The compiler's three
            # compact stores differ in exactly one token across slots 0, 1 and 2:
            #
            #   +0x54  reg:112  imm:16  imm:2066  expr:bin(op0,const(4),8)  imm:0 imm:0 imm:0 imm:1
            #   +0x5c  reg:110  imm:16  imm:2066  expr:bin(op0,const(4),8)  imm:0 imm:0 imm:4 imm:1
            #   +0x64  reg:111  imm:16  imm:2066  expr:bin(op0,const(4),8)  imm:0 imm:0 imm:8 imm:1
            #
            # so imms[-2] is the byte displacement and imms[-1] is the element COUNT. That count is
            # why eight-token layouts were refused wholesale here: S1's fourteen-byte sub-form 01
            # carries eight tokens whose last immediate is a count rather than an index, and
            # reading it as an address would have stored to the wrong place quietly. This branch
            # therefore admits ONE length of ONE opcode and still refuses everything else by name.
            disp, count = imms[-2], imms[-1]
            if count != 1:
                raise Unexecuted("%s at +%#x is a %d-element store; this checker models the single "
                                 "element form, and the range stores have their own model"
                                 % (what, offset, count))
            if disp % WORD_BYTES:
                raise ValueError("%s at +%#x has a %d-byte displacement that is not element-aligned "
                                 "for a %d-byte element" % (what, offset, disp, WORD_BYTES))
            addr = disp // WORD_BYTES
        else:                                               # the four-token layout: a constant index
            if len(toks) != 4:
                # Any other single-register shape (S1's 14-byte sub-form 01 store carries eight tokens whose last
                # immediate is a count, not an index) is not modelled here: refuse by name rather than read a field.
                raise Unexecuted("%s at +%#x: a %d-token single-register layout is not the constant-index form this checker models"
                                 % (what, offset, len(toks)))
            addr = imms[-1]
        buf = self.buffers.get(rank)
        if buf is None:
            raise ValueError("%s at +%#x reads rank %d, which has no allocation" % (what, offset, rank))
        if not 0 <= addr < len(buf):
            raise ValueError("%s at +%#x addresses element %d of binding rank %d (index %d), "
                             "outside its %d-element allocation"
                             % (what, offset, addr, rank, self.bindings[rank][0], len(buf)))
        return rank, addr

    def step(self, offset, size, opcode, toks):
        self.used.add(opcode)
        regs, imms = _regs(toks), _imms(toks)
        M = 0xFFFFFFFF
        if opcode == 10090:
            # Candidate semantics for probe-aq-0-before's ten-byte indexed ADD only.
            # The retained atomic readback ledger establishes per-lane updates, not
            # arbitrary mode fields, nonzero-old return semantics or contention.
            # The forthcoming whole-program comparison can refute this model.
            if not getattr(self, 'single_thread_atomic_model', False):
                raise Unexecuted('op10090 model requires explicit one-thread grid and group')
            kinds = [t.split(':', 1)[0] for t in toks]
            fixed = {1:'imm:1048576', 2:'imm:262656',
                     3:'expr:bin(op0,const(0),8)', 4:'imm:0',
                     6:'imm:0', 7:'imm:0', 8:'imm:4', 10:'imm:0'}
            if (size != 10 or kinds != ['reg','imm','imm','expr','imm','reg',
                                       'imm','imm','imm','reg','imm'] or
                    any(toks[i] != value for i,value in fixed.items()) or
                    any(not R32 <= int(toks[i][4:]) < R32+128 for i in (0,5,9))):
                raise Unexecuted('op10090 outside retained ten-byte indexed-add form')
            rank = _rank(toks)
            if rank is None or not 0 <= rank < len(self.bindings):
                raise ValueError('op10090 addresses an undeclared binding rank')
            if not self.bindings[rank][2]:
                raise ValueError('op10090 writes read-only binding rank %d' % rank)
            addr, addend = self.rd(toks[5]), self.rd(toks[9])
            buf = self.buffers.get(rank)
            if buf is None or not 0 <= addr < len(buf):
                raise ValueError('op10090 addresses outside binding allocation')
            old = self._read_bits(rank, addr)
            self.loads.append((rank, addr))
            self._store(rank, addr, (old + addend) & M)
            self.wr(toks[0], old)
            self.outside_domain[10090] = ('unverified',
                'candidate indexed atomic-add model: modulo32 update and old-word return; '
                'one thread only, no contention or isolated hardware semantics claim')
        elif opcode == 555:
            # movimm16.zero.4: measured source-faithful W0 half-zero witness,
            # results/g17-movimm4-roundB-compiles-v1 and form-opcodes registry.
            # No nonzero immediate or high-half destination is modeled here.
            if (size != 4 or len(toks) != 2 or len(regs) != 1 or
                not toks[0].startswith('reg:') or toks[1] != 'imm:0' or
                not R16 <= int(toks[0][4:]) < R16+128):
                raise Unexecuted('op555 outside measured low-half zero form')
            self.wr(toks[0],0)
            self.outside_domain[555]=('unverified',
                'op555 low-half zero model follows W0 compiled witness and retained halfzero program; '
                'no isolated general hardware semantics claim')
        elif opcode == 590:
            # asm.MOVHALF fields, exact low16 -> low16 form emitted by the
            # two retained integer-narrow sources. No float conversion occurs.
            if (size!=4 or len(toks)!=4 or len(regs)!=2 or
                    toks[1]!='imm:0' or toks[3]!='imm:16' or
                    not toks[0].startswith('reg:') or not toks[2].startswith('reg:') or
                    any(not R16<=int(t[4:])<R16+128 for t in regs)):
                raise Unexecuted('op590 outside scoped low16 move/release form')
            if self._view(regs[1])[0] in self.unread_load:
                raise Unexecuted('op590 cannot consume an outstanding load')
            value=self.rd(regs[1]);self._release_view(regs[1]);self.wr(regs[0],value)
            self.outside_domain[590]=('unverified',
                'op590 low16 move candidate follows asm.MOVHALF encoding; '
                'opposite-half preservation is a model assumption, not new hardware evidence')
        elif opcode in (1004, 1016):
            # Exact operand forms retained in g17-half-runtime-executed-33/scan.o
            # and interpreted by g17halfcheck: widen adds +0; narrow rounds to f16.
            # Those workload receipts do not isolate general rounding/NaN semantics.
            if (size != 12 or len(toks) != 5 or toks[1] != 'imm:2147483648'
                    or toks[3] not in ('imm:16','imm:32') or toks[4] != 'imm:128'
                    or len(regs) != 2):
                raise Unexecuted('op%d at +%#x: unsupported half-conversion operands' % (opcode, offset))
            dst, src = regs
            dst_base, dst_half = self._view(dst)
            src_base, src_half = self._view(src)
            if (not R32 <= int(dst_base[4:]) < R32+128 or not R32 <= int(src_base[4:]) < R32+128
                    or (opcode == 1004 and (dst_half is not None or src_half is None))
                    or (opcode == 1016 and (dst_half is None or src_half is not None))):
                raise Unexecuted('op%d: half-conversion register widths differ' % opcode)
            import numpy as np
            if opcode == 1004:
                value = np.asarray(self.rd(src), dtype=np.uint16).view(np.float16)
                if not np.isfinite(value): raise Unexecuted('half widening nonfinite input is outside retained model')
                value = np.float32(value) + np.float32(0)
                bits = _to_bits(value)
            else:
                value = self.rf(src)
                if not np.isfinite(value): raise Unexecuted('half narrowing nonfinite input is outside retained model')
                with np.errstate(over='ignore'):
                    bits = int(np.asarray(value, dtype=np.float16).view(np.uint16))
            if toks[3]=='imm:16':self._release_view(src)
            self.wr(dst,bits)
            self.outside_domain[opcode] = ('unverified', 'half-scan conversion model; workload evidence only, no general bit-exact rounding claim')
        elif opcode == READ_SR:
            if imms != [1048576, 0]:
                raise Unexecuted("op14059 at +%#x carries %s; the coordinate read the hardware "
                                 "runs established carries [1048576, 0]" % (offset, imms))
            # reg:61 is SR160 (thread_position_in_grid.x) and reg:62 SR161 (.y), as the query's
            # decoded prologue and the worker's (columns, rows) launch established on hardware;
            # a one-coordinate driver passes (row, 0)
            explicit = getattr(self, "launch_coordinates", None)
            if explicit is not None and regs[1] in explicit:
                self._write_word(regs[0], explicit[regs[1]])
                self.outside_domain[opcode] = ('unverified', 'explicit launch-coordinate model; selector identity is decoder-measured, not new hardware semantics')
            elif regs[1] == "reg:62":
                self._write_word(regs[0], int(getattr(self, "coord_y", 0)) & M)
            elif regs[1] == "reg:61":
                self._write_word(regs[0], int(self.row) & M)
            elif regs[1] == "reg:54":
                # SR156 = threadgroup_position_in_grid.x. THE REGISTER NUMBER IS MEASURED, not
                # inferred from the neighbours: encoding read_sr at sr=156,157,160,161 and reading
                # Apple's decoder back gives reg:54, reg:56, reg:61, reg:62 - so 54 is 156, and the
                # 61 above is 160, which is the pair this arm already modelled.
                #
                # THE VALUE IS NOT the thread index, and conflating them is how a multi-thread launch
                # would silently read the wrong coordinate: every thread of one threadgroup reads the
                # SAME threadgroup position. So this reads an explicit `group` when the caller set
                # one, is 0 for a single thread by the launch, and REFUSES otherwise rather than
                # substituting self.row.
                group = getattr(self, "group", None)
                if group is None:
                    if self.row != 0:
                        raise Unexecuted(
                            "op14059 at +%#x reads the threadgroup position (SR156) in thread %d, "
                            "and this machine was not told which threadgroup that thread is in; a "
                            "launch with more than one thread must set `group` rather than have the "
                            "thread index read as a threadgroup index" % (offset, self.row))
                    group = 0
                self._write_word(regs[0], int(group) & M)
            else:
                raise Unexecuted("op14059 at +%#x reads %s, which is not a grid coordinate this "
                                 "checker models" % (offset, regs[1]))
        elif opcode == READ_SR_LANE:
            # the 4-byte read_sr: [dest, 1048576, reg:45, 0] reads SR130 = the lane within its
            # SIMD group (the ABI's encoded selector 130, g17asm.decode_sr; the decoder prints the
            # register 45) into a 16-bit destination. Modelled as this machine's lane index.
            if imms != [1048576, 0] or regs[1] != "reg:45":
                raise Unexecuted("op14060 at +%#x reads %s %s; only SR130 (the SIMD lane) is modelled" % (offset, regs[1:], imms))
            self.wr(regs[0], int(getattr(self, "lane", self.row)))
        elif opcode == TENSOR_INIT:
            # op554, the four-byte movimm: dest = imm (the regression 'op554 ... encoding r129 and
            # reading it back'); the tensor witness zeroes its accumulators and its k counter with it
            self.wr(regs[0], imms[-1])
        elif opcode == MOV32:
            self.wr(regs[0], self.rd(regs[1]))
        elif opcode == ADD_U16_U16:
            # both sources are 16-bit views (add@10286 widen record: (0xFFFFFFFF, 1) -> 65536,
            # (0x12345678, 0x0F0F0F0F) -> 0x6587): u16 + u16, a 32-bit result
            self.wr(regs[0], (self.rd(regs[1]) & 0xFFFF) + (self.rd(regs[2]) & 0xFFFF))
        elif opcode == ADD_32_U16:
            # Decode the form before interpreting its values. A synchronous
            # register read must not silently consume an asynchronous half load.
            # The waiting form stays unverified for this generic caller: the
            # retained half-load run binds one program and seven inputs.
            if (size != 12 or len(toks) != 6 or len(regs) != 3 or
                    toks[0] != regs[0] or toks[2] != regs[1] or toks[4] != regs[2] or
                    imms not in ([32,16,16], [2147483680,16,16]) or
                    any(not R32 <= int(t[4:]) < R32+128 for t in regs[:2]) or
                    not R16 <= int(regs[2][4:]) < R16+128):
                raise Unexecuted('op10283 outside scoped 32+low16 form/modifiers')
            waiting = bool(imms[0] & (1 << 31))
            if not waiting and any(self._view(t)[0] in self.unread_load for t in regs[1:]):
                raise Unexecuted('op10283 without wait cannot consume an outstanding load')
            if waiting:
                self.outside_domain[ADD_32_U16] = ('unverified',
                    'op10283 waiting form has scoped execution in g17-integer16-half-load-v1; '
                    'this generic interpretation is not bound to that exact seven-input program')
            # first source a 32-bit register, second a 16-bit view: dest = a + (b & 0xFFFF).
            # MEASURED on seven operand pairs including wide ones (results/g17-tensor-width-
            # runtime-v1, integration's run of g17tensorwidthprobe, replayed here:
            # (0x12345678, 0x0F0F0F0F) -> 305423751, (0xFFFFFFFF, 1) -> 0) and on the small-value
            # records (D10283, d10283.l12.s2). The WIDTH is established; the arithmetic is not
            # exhaustive, so a wide input outside those pairs is still named as such.
            a, b = self.rd(regs[1]), self.rd(regs[2])
            base, _half = self._view(regs[2])
            if self.defined_halves.get(base, {'lo','hi'}) != {'lo','hi'}:
                # The instruction reads a defined half. Confidence accounting
                # must not introduce a full-word read that the program lacks.
                self.outside_domain[ADD_32_U16] = ("unverified", "op10283 reads a defined low half whose companion is undefined; interpreted using the measured width model, not classified as a measured full backing-word pair")
            else:
                backing = self.rd(base)
                wide = a > 0xFFFF or backing > 0xFFFF
                if wide and (a, backing) not in WIDTH_MEASURED_PAIRS:
                    self.outside_domain[ADD_32_U16] = ("unverified", "op10283 on a wide input outside the seven measured pairs; the 32 + u16 width is measured, the arithmetic there is not exhaustive (results/g17-tensor-width-runtime-v1)")
            self._release_view(regs[1]); self._release_view(regs[2])
            self.wr(regs[0], a + (b & 0xFFFF))
        elif opcode == MOVIMM and size == 2:
            # THE TWO-BYTE FORM: [dest, 0, value], a seven-bit literal and no modifier word
            # (isa/g17-execution-laneforms-results.json lane.form_movimm2, 32 lanes).
            if len(imms) != 2 or imms[0] != 0 or not 0 <= imms[1] <= 127:
                raise Unexecuted("op11842/2 at +%#x is not the executed [dest, 0, value] shape: %s" % (offset, toks))
            self._write_word(regs[0], imms[1])
            self.unread_load.discard(regs[0])
        elif opcode == MOVIMM:
            if imms[0] == 68719476736:
                # THE TENSOR WITNESS'S VARIANT: bit 24 of the printed modifier clear where every
                # executed movimm has it set (68736253952). Interpreted as the same move, and the
                # program's confidence says so - a movimm carrying this modifier has not executed.
                self.outside_domain[MOVIMM] = ("unverified", "op11842 with modifier 68719476736 (bit 24 clear; every executed movimm carries 68736253952) is interpreted as the same move, unmeasured")
            elif imms[0] != 68736253952:
                raise Unexecuted("op11842 at +%#x carries modifier %d; every executed movimm carries "
                                 "68736253952" % (offset, imms[0]))
            self._write_word(regs[0], imms[-1] & M)
            self.unread_load.discard(regs[0])
        elif opcode in IADD:
            # THE FUSED SHIFT-ADD SCALE: the decoder prints it in bits 8..10 of the last immediate
            # (0 x1, 1 x2, 2 x4, 3 x8, 4 x16; g17asm's causal table, authored x1/x2/x4 onto an x8
            # kernel). It scales SLOT A - the second printed source (D10282: (4097, 4096) -> 12289
            # = 4097 + 4096 << 1) or the immediate form's register (alu.shiftadd.imm: row*S + C).
            # This interpreter read every scaled add as an unscaled one before.
            shift = (imms[-1] >> 8) & 7 if opcode in SCALED_ADDS and imms else 0
            if shift > 4:
                raise Unexecuted("op%d at +%#x carries scale bits %d, outside the measured table" % (opcode, offset, shift))
            if len(regs) >= 3:
                self.wr(regs[0], self.rd(regs[1]) + (self.rd(regs[2]) << shift))
            else:
                self.wr(regs[0], imms[1] + (self.rd(regs[1]) << shift))
        elif opcode == IMUL_REG:
            self._write_word(regs[0], (self.rd(regs[1]) * self.rd(regs[2])) & M)
        elif opcode == IMUL_IMM:
            # THE IMMEDIATE IS imms[2]. The decoder prints the source's LIFETIME modifier (32 keep /
            # 16 release) before the value, and this read imms[1] - the modifier - so `t * 8`
            # interpreted as `t * 32`. Nothing interpreted had used a multiply-immediate with
            # t >= 1 before the loop kernels became interpretable (the LayerNorms multiply by a
            # register); the executed dotloop's Python model is the control that found it.
            if len(imms) < 3:
                raise Unexecuted("op%d at +%#x carries %d immediates; the multiply-immediate layout is lifetime, value" % (opcode, offset, len(imms)))
            self._write_word(regs[0], (self.rd(regs[1]) * imms[2]) & M)
        elif opcode in (12646, 17193):
            # Packed f32 buffer transport: one list element holds two exact halfwords.
            # Keep byte addressing and merge stores, including same-thread reloads.
            load = opcode == 12646
            # THE TEN-BYTE HALF LOAD, admitted 2026-09-22 at exactly the instance measured alone:
            # isa/g17-execution-memforms-results.json mem.load_half_10 (lead 8388608, the compiler's
            # form_length=10 lowering) delivered B's first half, converted, bit-exact. Its token layout
            # is the fourteen-byte one's except that lead, so the same arm reads it.
            ten = load and size == 10 and len(toks) == 9 and toks[1] == 'imm:8388608'
            if ((size != 14 and not ten) or len(toks) != 9 or len(regs) != 2
                    or (not ten and toks[1] not in (('imm:137447342080',) if load else ('imm:0','imm:16')))
                    or toks[2] != 'imm:2065' or toks[4] != 'imm:0'
                    or toks[6] not in (('imm:0',) if load else ('imm:0','imm:16')) or toks[8] != 'imm:2'):
                raise Unexecuted('op%d at +%#x: unsupported half-memory mode/mask' % (opcode, offset))
            rank = _rank(toks)
            if rank is None or not 0 <= rank < len(self.bindings) or toks[3] != 'expr:bin(op0,const(%d),8)' % (4*rank):
                raise ValueError('half memory binding rank/expression differs at +%#x' % offset)
            base, half = self._view(regs[0])
            if (half != 'lo' or not R32 <= int(base[4:]) < R32+128
                    or not R32 <= int(regs[1][4:]) < R32+128):
                raise Unexecuted('half memory requires measured low-half value and word index registers')
            displacement = int(toks[7].removeprefix('imm:'))
            byte_address = 2 * int(self.rd(regs[1])) + displacement
            buf = self.buffers.get(rank)
            if displacement % 2 or byte_address % 2 or buf is None or not 0 <= byte_address <= 4*len(buf)-2:
                raise ValueError('half memory byte address outside or misaligned allocation at +%#x' % offset)
            word, shift = byte_address//4, (byte_address%4)*8
            if not load and not self.bindings[rank][2]:
                raise ValueError('half store writes read-only binding rank %d' % rank)
            old = self._read_bits(rank, word)
            if load:
                self.loads.append((rank, word))
                self.wr(regs[0], (old >> shift) & 0xffff)
            else:
                bits = (old & ~(0xffff << shift)) | ((self.rd(regs[0]) & 0xffff) << shift)
                if _to_bits(_to_f(bits)) != bits:
                    raise Unexecuted('half store packed transport would change adjacent bits')
                self._store(rank, word, bits)
                # The half-store paired controls measure0 keep/16 release,
                # unlike the conversion forms'32 keep/16 release. Both source
                # values and address have been captured before either kill.
                if toks[1]=='imm:16':self._release_view(regs[0])
                if toks[6]=='imm:16':self._release_view(regs[1])
            self.outside_domain[opcode] = ('unverified', 'half-scan packed memory model; no general scheduling claim')
        elif opcode == LOAD32:
            rank, addr = self._access(toks, offset, "FP32 load", opcode=opcode, size=size)
            self.loads.append((rank, addr))
            if self.waw_model == "earlier_wins" and regs[0] in self.unread_load:
                pass                                        # the earlier, unread load's write stands
            else:
                self._write_word(regs[0], self._read_bits(rank, addr))
            self.unread_load.add(regs[0])
        elif opcode in RANGE_STORES and regs and int(regs[0][4:]) >= TUPLE_BASE[2]:
            self._range_store(opcode, offset, regs, imms, toks)
        elif opcode in STORES32:
            rank, addr = self._access(toks, offset, "FP32 store", opcode=opcode, size=size)
            if not self.bindings[rank][2]:
                raise ValueError("FP32 store at +%#x writes binding rank %d (index %d), which the "
                                 "contract declares read-only" % (offset, rank, self.bindings[rank][0]))
            self._store(rank, addr, self.rd(regs[0]))
        elif opcode == FADD:
            regs = _float_operands(opcode, toks, offset, 2)
            self._write_word(regs[0], _to_bits(self.rf(regs[1]) + self.rf(regs[2])))
        elif opcode == 11372:
            # Value compare, not the exec-mask cmp10 family. Executed relation
            # records: ledger/g17-all-ten-relations-from-three-codes.toml and
            # spike/accel/re/relations.py. The integer decoder token is 8|cc.
            # Only this full-word, register-input, 1/0-result layout is modeled.
            if (size != 10 or len(toks) != 9 or len(regs) != 3 or
                [i for i,t in enumerate(toks) if t.startswith('reg:')] != [0,3,5] or
                toks[1] != 'imm:16777248' or toks[2] not in ('imm:8','imm:9','imm:12','imm:13') or
                toks[4] not in ('imm:16','imm:32') or toks[6] not in ('imm:16','imm:32') or
                toks[7:] != ['imm:1','imm:0'] or
                any(not R32 <= int(t[4:]) < R32+128 for t in regs)):
                raise Unexecuted('op11372 at +%#x has an unmodeled compare width, relation, lifetime or result layout' % offset)
            if any(t in self.unread_load for t in regs[1:]):
                raise Unexecuted('op11372 at +%#x cannot directly consume an outstanding load; a wait-capable copy is required' % offset)
            a,b=self.rd(regs[1]),self.rd(regs[2]);relation=int(toks[2][4:])
            signed=lambda v:v-2**32 if v&0x80000000 else v
            value=int(a==b) if relation in (8,12) else int(a<b) if relation==9 else int(signed(a)<signed(b))
            # Capture both inputs before honoring release, including an aliased
            # destination. A later read cannot borrow a released old value.
            for token,life in ((regs[1],toks[4]),(regs[2],toks[6])):
                if life=='imm:16':
                    self.regs.pop(token,None);self.defined_halves.pop(token,None)
            self.wr(regs[0],value)
            self.outside_domain[11372]=('unverified',
                'op11372 relation/0-or-1 model follows ledger/g17-all-ten-relations-from-three-codes.toml; '
                'the old four-pair execution is a ledger report, not a supplied raw execution receipt for this delivery')
        elif opcode == 11375:
            # Four-source select: the raw newforms receipt establishes gt on
            # small positive comparands, not arbitrary signedness or aliases.
            # eq (decoder condition15) and the compiler destination modifier
            # are candidate interpretations; neither inherits that receipt.
            if (size != 14 or len(toks) != 11 or len(regs) != 5 or
                [i for i,t in enumerate(toks) if t.startswith('reg:')] != [0,3,5,7,9] or
                toks[1] not in ('imm:10737418240','imm:146028888064') or
                toks[2] not in ('imm:14','imm:15') or
                any(toks[i] not in ('imm:16','imm:32') for i in (4,6,8,10)) or
                len(set(regs)) != 5 or
                any(not R32 <= int(t[4:]) < R32+128 for t in regs)):
                raise Unexecuted('op11375 has an unmodeled select width, condition, modifier or alias')
            if any(t in self.unread_load for t in regs[1:]):
                raise Unexecuted('op11375 cannot consume an outstanding load')
            a,b,x,y = [self.rd(t) for t in regs[1:]]
            if max(a,b) > (1 if toks[2]=='imm:15' else 255):
                raise Unexecuted('op11375 comparands exceed the bounded byte/boolean domain')
            value = x if (a > b if toks[2]=='imm:14' else a == b) else y
            for token,life in zip(regs[1:],(toks[i] for i in (4,6,8,10))):
                if life=='imm:16':
                    self.regs.pop(token,None);self.defined_halves.pop(token,None)
            self.wr(regs[0],value)
            self.outside_domain[opcode]=('unverified',
                'op11375 gt direction has four small-positive cases in isa/g17-execution-newforms-results.json; '
                'byte-domain generalization, boolean eq and compiler modifier remain candidate models')
        elif opcode in (14392,17014):
            # Encoding and family direction are measured. The retained sweep
            # used only count4096 (identity); it does not establish nonzero
            # register counts or hardware count masking. Restrict the candidate
            # to explicit counts0..31, as emitted by the conversion lowering.
            # cc._hz / g17-only-one-alu-family-waited-for-a-load measures
            # hazard bit31 for these ALU families; no other hazard is admitted.
            if (size != 14 or len(toks) != 8 or len(regs) != 3 or
                [i for i,t in enumerate(toks) if t.startswith('reg:')] != [0,3,5] or
                toks[1] not in ('imm:0','imm:2147483648') or toks[2]!='imm:0' or toks[7] != 'imm:32' or
                toks[4] not in ('imm:16','imm:32') or toks[6] != 'imm:16' or
                len(set(regs)) != 3 or
                any(not R32 <= int(t[4:]) < R32+128 for t in regs)):
                raise Unexecuted('variable shift has an unmodeled width, modifier, lifetime or alias')
            if toks[1]=='imm:0' and any(t in self.unread_load for t in regs[1:]):
                raise Unexecuted('variable shift requires its measured load-wait bit for an outstanding load')
            value,count = self.rd(regs[1]),self.rd(regs[2])
            if count > 31:
                raise Unexecuted('variable shift count outside candidate0..31; hardware masking unestablished')
            result = ((value << count) & M) if opcode==14392 else value >> count
            for token,life in ((regs[1],toks[4]),(regs[2],toks[6])):
                if life=='imm:16':
                    self.regs.pop(token,None);self.defined_halves.pop(token,None)
            self.wr(regs[0],result)
            self.outside_domain[opcode]=('unverified',
                'variable shift candidate for counts0..31; hazard bit31 wait follows cc._hz and '
                'isa/g17-execution-sweep-results.json tests only count4096 identity, not nonzero counts')
        elif opcode in (14047,9986):
            # Exact unary shapes in the unchanged ctz delivery. The retained
            # execution sweep establishes reverse and MSB (not CLZ) on four
            # nonzero inputs; its different mode/lifetime tuple is not imported.
            life='imm:16' if opcode==14047 else 'imm:32'
            if (size!=10 or len(toks)!=4 or len(regs)!=2 or
                [i for i,t in enumerate(toks) if t.startswith('reg:')]!=[0,2] or
                toks[1]!='imm:0' or toks[3]!=life or
                any(not R32<=int(t[4:])<R32+128 for t in regs)):
                raise Unexecuted('op%d at +%#x has an unmodeled unary width, mode or lifetime' % (opcode,offset))
            if regs[1] in self.unread_load:
                raise Unexecuted('unary bit operation cannot consume an outstanding load')
            value=self.rd(regs[1])
            if opcode==9986 and value==0:
                raise Unexecuted('op9986 MSB zero input is outside this model')
            answer=int(format(value,'032b')[::-1],2) if opcode==14047 else value.bit_length()-1
            if life=='imm:16':
                self.regs.pop(regs[1],None);self.defined_halves.pop(regs[1],None)
            self.wr(regs[0],answer)
            self.outside_domain[opcode]=('unverified',
                'reverse32/MSB candidate follows isa/g17-execution-sweep-results.json '
                'D14047.l10/D9986.l10 on inputs4096,4097,4352,8192; '
                'this emitted mode/lifetime tuple and other inputs/endpoints are not isolated hardware evidence')
        elif opcode == 17770:
            # Retain normalized XOR1 unchanged; add only the exact XOR31 tuple
            # used by ctz. Modifier48 remains a dead-source restriction, not a
            # newly established interpretation of its hardware lifetime effect.
            if (size!=10 or len(toks)!=5 or len(regs)!=2 or
                not toks[0].startswith('reg:') or not toks[2].startswith('reg:') or
                toks[1]!='imm:0' or toks[3]!='imm:48' or toks[4] not in ('imm:1','imm:31') or
                any(not R32<=int(t[4:])<R32+128 for t in regs)):
                raise Unexecuted('op17770 at +%#x is outside the retained XOR1/XOR31 forms' % offset)
            if regs[1] in self.unread_load:
                raise Unexecuted('op17770 immediate XOR cannot consume an outstanding load')
            value=self.rd(regs[1]);constant=int(toks[4][4:])
            if constant==1 and value not in (0,1):
                raise Unexecuted('op17770 boolean complement source is not normalized0/1')
            self.regs.pop(regs[1],None);self.defined_halves.pop(regs[1],None)
            self.wr(regs[0],value^constant)
            self.outside_domain[17770]=('unverified',
                ('op17770 exact XOR1 boolean complement follows the40relation ledger cases; '
                 if constant==1 else
                 'op17770 XOR31 candidate follows D17770.l10 execution sweep on4096,4097,4352,8192; '
                 'other inputs and this emitted mode/lifetime tuple are not isolated hardware evidence; ')+
                'source modifier48 remains unestablished, so the checker restricts this form to a dead source')
        elif opcode == FMUL:
            regs = _float_operands(opcode, toks, offset, 2)
            self._write_word(regs[0], _to_bits(self.rf(regs[1]) * self.rf(regs[2])))
        elif opcode in (TG_STORE, TG_LOAD, BARRIER):
            self._threadgroup(opcode, offset, regs, imms)
        elif opcode == SHL_IMM:
            self._write_word(regs[0], (self.rd(regs[1]) << imms[-2]) & M)
        elif opcode == SHR_IMM:
            self._write_word(regs[0], (self.rd(regs[1]) >> imms[-2]) & M)
        elif opcode == SAR_IMM:
            # ARITHMETIC shift right by an immediate, [dest, 0, src, 24, shift]: executed on 32
            # edge-led lanes at shift 5 (isa/g17-execution-laneforms-results.json lane.form_sar5).
            # Only that operand shape; operand 3 = 24 is carried, not interpreted.
            if size != 12 or len(regs) != 2 or imms[:2] != [0, 24] or not 0 <= imms[-1] < 32:
                raise Unexecuted("op16805 at +%#x is not the executed [dest, 0, src, 24, shift] shape: %s"
                                 % (offset, toks))
            v = self.rd(regs[1])
            v = v - (1 << 32) if v & 0x80000000 else v
            self._write_word(regs[0], (v >> imms[-1]) & M)
        elif opcode == TENSOR_SETUP_SHIFT:
            # These two exact configurations were measured on all lane values
            # 0..31 plus eight controls. Neither an opcode name nor the older
            # shift-by-two record licenses other immediate/modifier tuples.
            import g17tensorsetupmodel
            if size != 14 or len(toks) != 7 or len(regs) != 2:
                raise Unexecuted("op17016 at +%#x is not the measured setup form" % offset)
            dest = int(regs[0][4:])
            if not R32 <= dest < R32 + 128:
                raise Unexecuted("op17016 destination is outside the modeled register file")
            value = self._half_index(regs[1], offset)
            try:
                result = g17tensorsetupmodel.interpret(toks, value)
            except ValueError as error:
                raise Unexecuted("op17016 at +%#x: %s" % (offset, error)) from error
            self._write_word(regs[0], result)
            self.unread_load.discard(regs[0])
        elif opcode == 13574:
            # Exact delivered if_bi_or60 tuple. asm.BITWISE_FORM and BW_IMM identify
            # OR and literal60; this is a source-wiring model, not a hardware claim
            # for source-control48 or the load-wait destination control. Older
            # D13574.l10 observations do not establish this operand tuple/domain.
            if (size != 10 or len(toks) != 5
                    or [t[:4] for t in toks] != ['reg:','imm:','reg:','imm:','imm:']
                    or toks[1] != 'imm:2147483648' or toks[3] != 'imm:48'
                    or toks[4] != 'imm:60'
                    or any(not R32 <= int(toks[i][4:]) < R32+128 for i in (0,2))):
                raise Unexecuted('op13574 outside exact OR60 candidate tuple')
            value = self.rd(toks[2])
            self._write_word(toks[0], (value | 60) & M)
            self.outside_domain[opcode] = ('unverified',
                'exact OR60 source-wiring candidate; source-control48 and wait behavior '
                'are not established by the older isolated sweep')
        elif opcode==423 and size==10 and len(toks)==5 and toks[4]=='imm:15':
            # Exact mask15 load-wait tuple in the signed narrowing source.
            # Do not expand tensorlogicmodel's measured40-input classification.
            if (len(regs)!=2 or not toks[0].startswith('reg:') or not toks[2].startswith('reg:')
                    or toks[1]!='imm:2147483648' or toks[3] not in ('imm:0','imm:16')
                    or any(not R32<=int(t[4:])<R32+128 for t in regs)):
                raise Unexecuted('op423 mask15 outside scoped load-wait form')
            value=self.rd(regs[1])
            if toks[3]=='imm:16':self._release_view(regs[1])
            self.wr(regs[0],value&15)
            self.outside_domain[423]=('unverified',
                'mask15 AND candidate with cc._hz bit31 load-wait; '
                'this tuple is outside the retained40-input hardware classification')
        elif opcode in TENSOR_SETUP_LOGIC:
            import g17tensorlogicmodel
            if size != 10 or len(toks) != 5 or len(regs) != 2:
                raise Unexecuted('op%d at +%#x is not the measured logic form' % (opcode,offset))
            dest = int(regs[0][4:])
            if not R32 <= dest < R32+128:
                raise Unexecuted('logic destination is outside the modeled register file')
            if opcode == 426:
                value = self._half_index(regs[1],offset)
            else:
                source = int(regs[1][4:])
                if not R32 <= source < R32+128:
                    raise Unexecuted('logic source is outside the modeled register file')
                value = self.rd(regs[1])
            try:
                result = g17tensorlogicmodel.interpret(opcode,toks,value)
            except ValueError as error:
                raise Unexecuted('op%d at +%#x: %s' % (opcode,offset,error)) from error
            self._write_word(regs[0], result)
            self.unread_load.discard(regs[0])
        elif opcode == ISUB_REG:
            self._write_word(regs[0], (self.rd(regs[1]) - self.rd(regs[2])) & M)
        elif opcode == FMA:
            regs = _float_operands(opcode, toks, offset, 3)
            # THE LEAD 32 IS ADMITTED FOR ONE SHAPE: the four-byte form, accumulator = destination at
            # operand 6. regs[3] is then the destination's value BEFORE this write, which is what a
            # two-address fma reads.
            if toks[1] == "imm:32" and not ((size == 4 and regs[3] == regs[0]) or size == 6):
                raise Unexecuted("op2190 at +%#x carries lead 32 at %d bytes with sources %s; only the "
                                 "four-byte form with the accumulator printed as the destination at "
                                 "operand 6, and the six-byte form, are executed" % (offset, size, regs[1:]))
            # one rounding, as the host's fmaf - the same model the query's GPU outputs matched
            # bit for bit on 49,152 words (docs/archive/g17-query-compiler-release-review.md)
            f = _host_fmaf()
            self._write_word(regs[0], _to_bits(f(float(self.rf(regs[1])), float(self.rf(regs[2])),
                                           float(self.rf(regs[3])))))
        elif opcode == FSELECT:
            regs = _float_operands(opcode, toks, offset, 4)
            name_fn = FSELECT_OPS.get(imms[1])
            if name_fn is None or len(regs) != 5 or regs[1] != regs[3] or regs[2] != regs[4]:
                raise Unexecuted("op9700 at +%#x is not the max/min shape Apple emits (slot 2 %s, "
                                 "sources %s)" % (offset, imms[1:2], regs[1:]))
            a, b = float(self.rf(regs[1])), float(self.rf(regs[2]))
            if a != a or b != b:
                raise Unexecuted("op9700 at +%#x reads a NaN, whose result is not established" % offset)
            self.notes_fselect = True
            self._write_word(regs[0], _to_bits(name_fn[1](a, b)))
        elif opcode == 3818:
            # Scoped candidate for mp-trunc.h-4: Apple's rounding-unit selection
            # and asm.TRANS_FORM identify truncation, not universal execution
            # semantics. The old sweep's four zero outputs do not certify these
            # fractional inputs. Require the exact delivered operand form and a
            # finite value exactly representable as binary16 after widening.
            if (size != 10 or len(toks) != 4 or len(regs) != 2
                    or not toks[0].startswith('reg:') or not toks[2].startswith('reg:')
                    or toks[1] != 'imm:32' or toks[3] != 'imm:16'
                    or any(not R32 <= int(r[4:]) < R32+128 for r in regs)):
                raise Unexecuted('op3818 outside scoped truncation/release form')
            if regs[1] in self.unread_load:
                raise Unexecuted('op3818 cannot consume an outstanding load')
            value = float(self.rf(regs[1]))
            try:
                half = struct.unpack('<e', struct.pack('<e', value))[0]
            except OverflowError:
                half = float('inf')
            if not math.isfinite(value) or half != value:
                raise Unexecuted('op3818 candidate requires a finite widened binary16 value')
            result = math.copysign(float(math.trunc(value)), value)
            self._release_view(regs[1])
            self.wr(regs[0], _to_bits(result))
            self.outside_domain[3818] = ('unverified',
                'op3818/10 lead32 release16 finite widened-half truncation candidate; '
                'source selection and arithmetic model, not general hardware semantics')
        elif opcode in SEMANTICS:
            regs = _float_operands(opcode, toks, offset, 1)
            value, conf, why = interpret(opcode, float(self.rf(regs[1])), executed=self.executed)
            if conf != "executed":
                # THE VERDICT PER INPUT REACHES THE VERDICT PER PROGRAM, with its class: "bounded"
                # carries a measured magnitude, "isolation" carries none. exp2 is executed on 1, 2,
                # 4, 8 and the softmax feeds it fractions; dropping interpret()'s per-call
                # confidence here reported "executed" for a program whose every exp2 was outside
                # the measured domain.
                self.outside_domain[opcode] = (conf, why)
            self._write_word(regs[0], _to_bits(value))
        elif opcode in (612, 11452):
            import g17tensorpredicatemodel
            if len(regs) != 2:
                raise Unexecuted('predicate register layout is not measured')
            value = self.rd(regs[1])
            try:
                result = g17tensorpredicatemodel.interpret(opcode, size, toks, value)
            except ValueError as error:
                raise Unexecuted(str(error)) from error
            if opcode == 11452 or (opcode == 612 and toks[3] == 'imm:16'):
                # The measured configuration releases its source. Do not assume
                # its old value remains available after the operation.
                base, _half = self._view(regs[1])
                self.regs.pop(base, None); self.defined_halves.pop(base, None)
            self.wr(regs[0], result)
        elif opcode in CONVERSIONS:
            # INTEGER IN, FLOAT OUT, so the source is read with `self.rd` (raw bits) and never with
            # `self.rf`. The layout is asserted rather than counted: Apple's decoder prints this form
            # as [dest, imm, imm, imm, src, imm] and reading "whichever registers happen to be
            # present" is what the bitwise arm's own note warns against.
            if size not in CONVERSION_LENGTHS:
                raise Unexecuted("op%d at +%#x is %d bytes; the conversion is interpreted at %s"
                                 % (opcode, offset, size,
                                    " and ".join(str(n) for n in CONVERSION_LENGTHS)))
            if len(toks) != 6 or len(regs) != 2 or not (toks[0].startswith("reg:")
                                                        and toks[4].startswith("reg:")):
                raise Unexecuted("op%d at +%#x is not the [dest, imm, imm, imm, src, imm] layout "
                                 "this checker models: %s" % (opcode, offset, toks))
            # OPERAND 5 IS ONE OF THREE SHIPPED VALUES AND ITS MEANING IS UNMEASURED. {0, 16, 32} is
            # the value set the twelve-byte ALU's source lifetime takes, so 16 may mean "release the
            # source". This checker does NOT model that, and it must not pretend the question is
            # closed: a program that reads the source again after the conversion is REFUSED here,
            # which is the same bound the compiler's lowering enforces at selection time.
            if imms[-1] not in (0, 16, 32):
                raise Unexecuted("op%d at +%#x carries operand 5 = %d; only the three values Apple "
                                 "ships (0, 16, 32) are admitted" % (opcode, offset, imms[-1]))
            value, conf, why = interpret_conversion(opcode, self.rd(regs[1]),
                                                    lifetime=imms[-1], executed=self.executed)
            if conf != "executed":
                self.outside_domain[opcode] = (conf, why)
            else:
                # AN EXECUTED CONVERSION STILL HAS TO SAY WHAT IT RESTS ON. outside_domain is the
                # wrong place - a non-empty one downgrades the whole verdict to "bounded" - so the
                # citation is carried separately and reported beside the verdict. Without it a
                # program reads as "executed" with nothing naming the nine inputs that earned it.
                if not hasattr(self, "conversion_evidence"):
                    self.conversion_evidence = {}
                self.conversion_evidence[opcode] = why
            if imms[-1] == 16:
                self.conversion_released = self.conversion_released | {regs[1]} \
                    if hasattr(self, "conversion_released") else {regs[1]}
            self._write_word(regs[0], _to_bits(value))
        elif opcode in BITWISE_REG:
            if size not in BITWISE_REG_LENGTHS:
                raise Unexecuted("op%d at +%#x is %d bytes; the register-register bitwise is "
                                 "interpreted at %s and this is neither"
                                 % (opcode, offset, size,
                                    " and ".join(str(n) for n in BITWISE_REG_LENGTHS)))
            # Counting three registers and three immediates is NOT checking the layout: a token
            # list that groups them would count the same and still be read as pairs. The positions
            # are asserted - which is also what turns the ten-byte form's `expr` operands into a
            # refusal instead of a read of whichever registers happen to be present.
            if (len(toks) != 6 or [t[:4] for t in toks]
                    != ['reg:', 'imm:', 'reg:', 'imm:', 'reg:', 'imm:']):
                raise Unexecuted("op%d/%d at +%#x decodes as %s; this arm interprets three "
                                 "register/immediate pairs in that order, and an operand that is "
                                 "not a register is a kind it cannot read"
                                 % (opcode, size, offset, toks))
            for i, (m, allowed) in enumerate(zip(imms, BITWISE_REG_MODIFIERS)):
                if m & 0xFF not in allowed:
                    raise Unexecuted("op%d at +%#x carries %d at operand position %d, whose low "
                                     "byte is outside the censused set %s; an unmeasured modifier "
                                     "there can change the result (negate and absolute value "
                                     "would) and this checker will not guess it away"
                                     % (opcode, offset, m, i, sorted(allowed)))
                high = m >> 8
                if not high:
                    continue
                if size == 4:
                    raise Unexecuted("op%d/4 at +%#x carries %#x above the low byte at operand "
                                     "position %d; none of the 2,188 four-byte instances carries "
                                     "anything there. The index family is measured on the ten-byte "
                                     "destination, and a length does not inherit another's census"
                                     % (opcode, offset, high, i))
                if i:
                    raise Unexecuted("op%d at +%#x carries %#x above the low byte at operand "
                                     "position %d; in all 345 measured instances only the "
                                     "destination does, so a source that does is a combination "
                                     "this checker has never seen" % (opcode, offset, high, i))
                bit = (high << 8).bit_length() - 1
                if high & (high - 1) or bit not in BITWISE_REG_INDEX_BITS:
                    raise Unexecuted("op%d at +%#x carries %#x above the destination's low byte; "
                                     "the measured index family is one bit from %s, and anything "
                                     "else is unmeasured"
                                     % (opcode, offset, high, sorted(BITWISE_REG_INDEX_BITS)))
            self._write_word(regs[0], BITWISE_REG[opcode](self.rd(regs[1]) & M,
                                                          self.rd(regs[2]) & M) & M)
        elif opcode in MEASURED_ALU:
            self._measured_alu(opcode, offset, size, regs, imms, toks)
        elif opcode == 1000 and size == 12:
            if ([t[:4] for t in toks] != ["reg:", "imm:", "reg:", "imm:", "imm:"]
                    or imms[0] not in FADDI_LEADS or imms[1] not in MEASURED_ALU_SOURCE_LIFETIMES
                    or imms[2] not in FADDI_CODES):
                raise Unexecuted("op1000/12 at +%#x decodes as %s; interpreted only at lead %s, "
                                 "source lifetime 16 and immediate code %s - what was run alone"
                                 % (offset, toks, sorted(FADDI_LEADS), sorted(FADDI_CODES)))
            import g17asm
            x = struct.unpack('<f', struct.pack('<I', self.rd(regs[1]) & 0xFFFFFFFF))[0]
            import numpy as np
            v = np.float32(np.float32(x) + np.float32(g17asm.float_imm_value(imms[2])))
            self._write_word(regs[0], struct.unpack('<I', struct.pack('<f', float(v)))[0])
        elif opcode == 904 and size == 12:
            if ([t[:4] for t in toks] != ["reg:", "imm:", "reg:", "imm:", "imm:"]
                    or imms[0] not in FSAT_904_LEADS or imms[1] not in MEASURED_ALU_SOURCE_LIFETIMES
                    or imms[2] not in FSAT_904_CODES):
                raise Unexecuted("op904/12 at +%#x decodes as %s; interpreted only as Apple's "
                                 "saturate(x): lead %s, source lifetime 16, immediate code %s"
                                 % (offset, toks, sorted(FSAT_904_LEADS), sorted(FSAT_904_CODES)))
            import numpy as np
            x = np.float32(struct.unpack('<f', struct.pack('<I', self.rd(regs[1]) & 0xFFFFFFFF))[0])
            v = np.float32(min(max(np.float32(x + np.float32(-0.0)), np.float32(0.0)), np.float32(1.0)))
            # THE HARDWARE CANONICALISES ZERO: saturate(-0.0) came back +0.0 on silicon
            # (isa/g17-execution-fsat-results.json), where Python's max keeps -0.0.
            if v == 0:
                v = np.float32(0.0)
            self._write_word(regs[0], struct.unpack('<I', struct.pack('<f', float(v)))[0])
        elif opcode == MULHI:
            if size != 12:
                raise Unexecuted("op%d at +%#x is %d bytes; only the twelve-byte widening multiply "
                                 "is interpreted, and it is the only length in the corpus"
                                 % (opcode, offset, size))
            if len(regs) != 3:
                raise Unexecuted("op%d at +%#x decodes as %s; this arm needs three registers - an "
                                 "expression operand is a kind it cannot read (witnessed on 4 of "
                                 "649 rows and refuted there)" % (opcode, offset, toks))
            pair = int(regs[0][4:])
            if not TUPLE_BASE[2] <= pair < TUPLE_BASE[2] + 128:
                raise Unexecuted("op%d at +%#x: %s is not in the pair file" % (opcode, offset, regs[0]))
            # THE PAIR INDEX IS THE FIRST REGISTER, not twice it - measured over the 654 instances:
            # under this reading 20 of 654 leave both registers unread, under the doubling 338 do.
            first = R32 + (pair - TUPLE_BASE[2])
            product = (self.rd(regs[1]) & M) * (self.rd(regs[2]) & M)
            self._write_word("reg:%d" % first, product & M)                 # low
            self._write_word("reg:%d" % (first + 1), (product >> 32) & M)   # high
        elif opcode == VEC_STORE:
            self._vec_store(opcode, offset, size, regs, imms, toks)
        elif opcode in VEC_LOADS:
            self._vec_load(opcode, offset, size, regs, imms, toks)
        elif opcode == END:
            return False
        else:
            raise Unexecuted("op%d at +%#x is not interpretable by this checker" % (opcode, offset))
        return True

    def _measured_alu(self, opcode, offset, size, regs, imms, toks):
        """One MEASURED_ALU instruction: [dst, lead, src, life(, src, life)] exactly, or refuse."""
        name, nsrc, length, leads = MEASURED_ALU[opcode]
        if size != length:
            raise Unexecuted("op%d at +%#x is %d bytes; it is interpreted at %d, the length its "
                             "function was measured at" % (opcode, offset, size, length))
        want = ["reg:", "imm:"] + ["reg:", "imm:"] * nsrc
        if [t[:4] for t in toks] != want:
            raise Unexecuted("op%d/%d at +%#x decodes as %s; this arm reads %s exactly"
                             % (opcode, size, offset, toks, want))
        if imms[0] not in leads:
            raise Unexecuted("op%d at +%#x carries lead modifier %d; admitted, from the records "
                             "that determined %s: %s" % (opcode, offset, imms[0], name, sorted(leads)))
        allowed = MEASURED_ALU_SOURCE_MODIFIERS.get(opcode, MEASURED_ALU_SOURCE_LIFETIMES)
        if any(m not in allowed for m in imms[1:]):
            raise Unexecuted("op%d at +%#x carries source modifiers %s; only %s was measured"
                             % (opcode, offset, imms[1:], sorted(allowed)))
        import g17fitfromexecution as FF
        fn = (FF.INT2 if nsrc == 2 else FF.FLOAT1)[name]
        got = fn(*[self.rd(r) & 0xFFFFFFFF for r in regs[1:1 + nsrc]])
        if got is None:
            raise Unexecuted("op%d at +%#x: %s has no defined result on these inputs" % (opcode, offset, name))
        self._write_word(regs[0], int(got) & 0xFFFFFFFF)

    def _read_bits(self, rank, addr):
        """The buffer word as THIS thread sees it, AS BITS: its own store, else memory.

        BITS, NOT A FLOAT, AND THAT IS NOT A DETAIL. The shared buffers hold Python floats, and a
        SIGNALLING NaN does not survive that round trip - 0x7F9A6A01 comes back 0x7F9E6A01, quieted
        in bit 22. A uint32 program carrying such a pattern is then modelled wrongly, and the error
        is a plausible-looking number rather than a refusal. Found by spilling a buffer of hashed
        uint32s: every sum was off by exactly 0x400000, which was one quieted NaN and not the
        spill. A thread's own stores are therefore kept as the exact pattern it wrote.
        """
        if (rank, addr) in self.own_stores:
            return self.own_stores[(rank, addr)]
        return _to_bits(self.buffers[rank][addr])

    def _store(self, rank, addr, bits):
        """Record a store and make its exact bits visible to this thread's later loads.

        simulate_threads drains stores after each modeled thread. _to_f retains raw
        signaling-NaN words through its raw-word wrapper; numeric floating operations
        explicitly convert that wrapper. This transport does not model concurrency.
        """
        self.stores.append((rank, addr, _to_f(bits)))
        self.own_stores[(rank, addr)] = bits & 0xFFFFFFFF

    def _vec_load(self, opcode, offset, size, regs, imms, toks):
        """op12709 at 8 or 14 bytes: a register tuple <- consecutive words at 4 * index + disp/4."""
        if size not in VEC_LOAD_LEADS:
            raise Unexecuted("op%d at +%#x is %d bytes; only the 8- and 14-byte forms cc emits are "
                             "interpreted" % (opcode, offset, size))
        kinds = [t.split(":")[0] if ":" in t else t[:4] for t in toks]
        if kinds != ["reg", "imm", "imm", "expr", "imm", "reg", "imm", "imm", "imm"]:
            raise Unexecuted("op%d at +%#x decodes as %s; cc's vector load carries the tuple, the "
                             "lead, the mask, the address expression and the index register"
                             % (opcode, offset, toks))
        lead, mask, *rest = imms
        n = VEC_LOADS[opcode]
        # rest = [0, index lifetime (0 keep / 16 release), 0, element stride in BYTES]. The stride is
        # 4n ROUNDED UP TO A POWER OF TWO (asm.vec_access_bytes), so three components step 16 bytes,
        # not 12: executed 2026-09-23, lane t of a three-word load read words 4t..4t+2
        # (isa/g17-execution-laneforms-results.json lane.form_vec3, ledger
        # g17-three-component-vector-load-steps-sixteen-bytes.toml).
        stride = _vec_access_bytes(n)
        if (lead not in VEC_LOAD_LEADS[size] or mask != RANGE_MASK[n] or len(rest) != 4
                or rest[0] != 0 or rest[1] not in (0, 16) or rest[2] != 0 or rest[3] != stride):
            raise Unexecuted("op%d/%d at +%#x carries %s; interpreted only as cc emits it (lead %s, "
                             "mask %d, then 0, an index lifetime of 0 or 16, 0, and a %d-byte stride)"
                             % (opcode, size, offset, imms, sorted(VEC_LOAD_LEADS[size]),
                                RANGE_MASK[n], stride))
        base = int(regs[0][4:]) - TUPLE_BASE[n]
        if not 0 <= base <= 127:
            raise Unexecuted("op%d at +%#x: %s is not in the %d-tuple file" % (opcode, offset, regs[0], n))
        rank = _rank(toks)
        if rank is None or rank >= len(self.bindings):
            raise ValueError("op%d at +%#x names binding rank %s outside the contract"
                             % (opcode, offset, rank))
        disp = imms[-2]
        index = self.rd(regs[1]) & 0xFFFFFFFF
        buf = self.buffers.get(rank)
        for i in range(n):
            addr = stride // 4 * index + disp // 4 + i
            if buf is None or not 0 <= addr < len(buf):
                raise ValueError("op%d at +%#x reads element %d of rank %d outside its %d-element "
                                 "allocation" % (opcode, offset, addr, rank, 0 if buf is None else len(buf)))
            self._write_word("reg:%d" % (R32 + base + i), self._read_bits(rank, addr))

    def _vec_store(self, opcode, offset, size, regs, imms, toks):
        """op17256/8: four consecutive registers to `4 * index + disp/4 ..`, a spill's store.

        THE ADDRESS IS COMPUTED, NOT A DISPLACEMENT, which is the difference from _range_store and
        the reason this form can give each thread its own slot. A checker that read the
        displacement and ignored the index register would place every thread's spill at one
        address and report a cross-thread collision as clean - the exact defect the source-owned
        spill work found in its own address map (docs/archive/g17-spill-source-handoff.md, the `collide`
        control that did not fire).
        """
        if size != 8:
            raise Unexecuted("op%d at +%#x is %d bytes; all eight Apple instances are eight"
                             % (opcode, offset, size))
        kinds = [t.split(":")[0] if ":" in t else t[:4] for t in toks]
        if kinds != ["reg", "imm", "imm", "expr", "imm", "reg", "imm", "imm", "imm"]:
            raise Unexecuted("op%d at +%#x decodes as %s; all eight Apple instances carry the "
                             "tuple, the address expression and the index register in that order"
                             % (opcode, offset, toks))
        for i, (m, allowed) in enumerate(zip(imms, VEC_STORE_IMMS)):
            if m not in allowed:
                raise Unexecuted("op%d at +%#x carries %d at immediate %d, outside the eight "
                                 "measured instances' %s; a population of eight cannot say what a "
                                 "ninth value means" % (opcode, offset, m, i, sorted(allowed)))
        printed = int(regs[0][4:])
        n = next((k for k, base in TUPLE_BASE.items() if 0 <= printed - base <= 127), None)
        if n is None:
            raise Unexecuted("op%d at +%#x: %s is in no tuple file this checker maps"
                             % (opcode, offset, regs[0]))
        if n != 4:
            raise Unexecuted("op%d at +%#x stores a %d-tuple; only the quad is measured here"
                             % (opcode, offset, n))
        base = printed - TUPLE_BASE[n]
        rank = _rank(toks)
        if rank is None or rank >= len(self.bindings):
            raise ValueError("op%d at +%#x names binding rank %s outside the contract"
                             % (opcode, offset, rank))
        if not self.bindings[rank][2]:
            raise ValueError("op%d at +%#x writes read-only rank %d" % (opcode, offset, rank))
        disp = imms[-2]
        if disp % 4:
            raise ValueError("op%d at +%#x has a %d-byte displacement that is not word-aligned"
                             % (opcode, offset, disp))
        # THE INDEX REGISTER IS READ, and it is what makes the address per-thread.
        # (`M` is a local of step(); this is a separate method, so the mask is written out.)
        index = self.rd(regs[1]) & 0xFFFFFFFF
        buf = self.buffers.get(rank)
        for i in range(n):
            addr = 4 * index + disp // 4 + i
            if buf is None or not 0 <= addr < len(buf):
                raise ValueError("op%d at +%#x addresses element %d of rank %d outside its "
                                 "%d-element allocation"
                                 % (opcode, offset, addr, rank, 0 if buf is None else len(buf)))
            self._store(rank, addr, self.rd("reg:%d" % (R32 + base + i)))

    def _range_store(self, opcode, offset, regs, imms, toks):
        n = RANGE_STORES[opcode]
        base = int(regs[0][4:]) - TUPLE_BASE[n]
        if not 0 <= base <= 127:
            raise Unexecuted("range store at +%#x: %s is not in the %d-tuple file" % (offset, regs[0], n))
        if len(regs) != 1 or len(imms) < 2 or imms[1] != RANGE_MASK[n] or imms[-1] != 1:
            raise Unexecuted("range store at +%#x carries %s; the delivered %d-component layout carries mask %d and a trailing 1"
                             % (offset, imms, n, RANGE_MASK[n]))
        rank = _rank(toks)
        if rank is None or rank >= len(self.bindings):
            raise ValueError("range store at +%#x names binding rank %s outside the contract" % (offset, rank))
        if not self.bindings[rank][2]:
            raise ValueError("range store at +%#x writes read-only rank %d" % (offset, rank))
        disp = imms[-2]
        if disp % 4:
            raise ValueError("range store at +%#x has a %d-byte displacement that is not word-aligned" % (offset, disp))
        buf = self.buffers.get(rank)
        for i in range(n):
            addr = disp // 4 + i
            if buf is None or not 0 <= addr < len(buf):
                raise ValueError("range store at +%#x addresses element %d of rank %d outside its %d-element allocation" % (offset, addr, rank, 0 if buf is None else len(buf)))
            self._store(rank, addr, self.rd("reg:%d" % (R32 + base + i)))
        if imms[0] == 16:
            # Conservative admission, not a prediction of a released register's
            # value. The op17244 probe returned zero on later use; neither that
            # result nor the broader tuple interpretation licenses retaining the
            # pre-store value. A subsequent defining write makes it readable again.
            for i in range(n):
                token = 'reg:%d' % (R32 + base + i)
                self.regs.pop(token, None)
                self.defined_halves.pop(token,None)
                self.range_store_releases[token] = offset

    def _half_index(self, token, offset):
        """A threadgroup index register is printed in the 16-bit file: 425+n is the low half of
        the 32-bit register printed 105+n."""
        n = int(token[4:])
        if not R16 <= n < R16 + 128:
            raise Unexecuted("threadgroup index %s at +%#x is not in the 16-bit file this checker maps" % (token, offset))
        return int(self.rd(token)) & 0xFFFF          # the view read: defined-half checked, the other half untouched

    def _threadgroup(self, opcode, offset, regs, imms):
        """Threadgroup memory under the group driver. Without a group (self.group is None) the
        forms are refused: one thread cannot say what another wrote."""
        g = getattr(self, "group", None)
        if g is None:
            raise Unexecuted("op%d at +%#x needs the threadgroup driver (simulate_group); a single "
                             "thread cannot establish what other lanes wrote" % (opcode, offset))
        if opcode == BARRIER:
            if imms != [0, 276]:
                raise Unexecuted("op447 at +%#x carries %s; the executed threadgroup barrier carries [0, 276]" % (offset, imms))
            g.barrier(self.lane, offset)
            return
        idx = self._half_index(regs[1], offset)
        if not 0 <= idx < g.words:
            raise ValueError("threadgroup access at +%#x: lane %d names word %d of a %d-word scratchpad"
                             % (offset, self.lane, idx, g.words))
        if opcode == TG_STORE:
            g.store(self.lane, idx, self.rd(regs[0]), offset)
        else:
            self._write_word(regs[0], g.load(self.lane, idx, offset))
            self.unread_load.add(regs[0])

    # SINGLE-LANE CONTROL FLOW, the shape this compiler emits and Apple's corpus carries: op10369
    # compares a register with an immediate into a predicate register (the decoder's relation
    # token 9 is the only one this compiler authors for a loop, `lt`, unsigned); op582 sets the
    # lane's mask from that predicate; op458 REPEATS WHILE THE LANE IS ACTIVE (ledger/g17-the-
    # back-edge-gates-on-op582, 16 bounded dispatches) with its displacement from the
    # instruction's own start (base 0, ledger/g17-pc-base-reverted-boundary-is-not-semantic);
    # op577 restores the lane. A masked lane executes nothing until the restore. What this models
    # is ONE lane; divergence between lanes is not a thing here, and the back edge's semantics
    # for a lane that is masked (it falls through) is what the bounded dispatches measured.
    CMP_IMM, EXEC_MASK, BACK_EDGE, EXEC_RESTORE = 10369, 582, 458, 577
    STEP_LIMIT = 50_000_000

    def run(self, instructions):
        instructions = list(instructions)
        cursor = 0
        for offset, size, _opcode, _toks in instructions:
            if offset != cursor:
                raise ValueError("instruction boundaries are not consecutive at +%#x" % offset)
            cursor += size
        by_offset = {offset: i for i, (offset, _s, _o, _t) in enumerate(instructions)}
        preds, active, i, steps = {}, True, 0, 0
        while i < len(instructions):
            offset, size, opcode, toks = instructions[i]
            steps += 1
            if steps > self.STEP_LIMIT:
                raise ValueError("more than %d instruction steps: the loop at +%#x does not terminate under this model" % (self.STEP_LIMIT, offset))
            if opcode == self.CMP_IMM:
                if not active: i += 1; continue
                regs = _regs(toks); imms = [int(t.split(":")[1]) for t in toks if t.startswith("imm:")]
                if len(regs)==2 and len(imms)==4 and imms[2]==32:
                    import g17tensorcomparemodel
                    try:result=g17tensorcomparemodel.interpret(size,toks,self.rd(regs[1]))
                    except ValueError as error:raise Unexecuted(str(error)) from error
                    preds[int(regs[0].split(':')[1])]=result
                    self.used.add(opcode);i+=1;continue
                if len(regs) != 2 or len(imms) != 4 or imms[1] != 9:
                    raise Unexecuted("op10369 at +%#x carries relation token %s; only the loop's `lt` (token 9) is authored and measured" % (offset, imms[1:2]))
                pred = int(regs[0].split(":")[1]); src = int(regs[1].split(":")[1])
                preds[pred] = (self.rd(regs[1]) & 0xFFFFFFFF) < (imms[3] & 0xFFFFFFFF)
                self.used.add(opcode); i += 1; continue
            if opcode == self.EXEC_MASK:
                regs = _regs(toks)
                if not regs or int(regs[0].split(":")[1]) not in preds:
                    raise Unexecuted("op582 at +%#x masks from a predicate nothing wrote" % offset)
                active = active and preds[int(regs[0].split(":")[1])]
                self.used.add(opcode); i += 1; continue
            if opcode == self.BACK_EDGE:
                disp = [int(t.split(":")[1]) for t in toks if t.startswith("imm:")][-1]
                self.used.add(opcode)
                if active:
                    target = offset + disp
                    if target not in by_offset:
                        raise ValueError("op458 at +%#x targets +%#x, not an instruction boundary" % (offset, target))
                    i = by_offset[target]; continue
                i += 1; continue
            if opcode == self.EXEC_RESTORE:
                active = True; self.used.add(opcode); i += 1; continue
            if not active:
                i += 1; continue
            if not self.step(offset, size, opcode, list(toks)):
                break
            i += 1
        return self

    def confidence(self):
        """('executed' | 'unverified', notes). Every arithmetic opcode needs execution evidence and
        every structural one needs silicon agreement; an empty evidence set is never 'executed'."""
        notes = []
        arith = sorted(o for o in self.used if o in ARITH_FP)
        struct_ = sorted(o for o in self.used if o in STRUCTURAL)
        # A CONVERSION IS A THIRD CLASS. Left out, op11179 landed in "opcodes in neither class",
        # which makes the whole program "unverified" - indistinguishable from a program this checker
        # has NO model for. The distinction is what root's preparation needs: unverified means no
        # prediction is available, candidate means a fully specified model that a comparison can
        # refute. It is still not a measurement, and the verdict below never reaches "executed" or
        # "bounded" while a conversion in it has not been run.
        unk = sorted(o for o in self.used if o not in ARITH_FP | STRUCTURAL | set(CONVERSIONS))
        # A CONVERSION WHOSE EVERY INTERPRETATION CAME BACK "executed" IS NOT A CANDIDATE. The
        # per-call confidence is what decides it, recorded in outside_domain by the arm above -
        # reading the opcode alone would have called a measured input a candidate and an unmeasured
        # one executed, which is the mistake exp2's measured domain already cost this file once.
        conv = sorted(o for o in self.used if o in CONVERSIONS and o not in self.executed
                      and getattr(self, "outside_domain", {}).get(o, ("", ""))[0] == "candidate")
        missing_a = [o for o in arith if o not in self.executed]
        missing_s = [o for o in struct_ if o not in self.silicon]
        if missing_a:
            notes.append("arithmetic without execution evidence: %s" % missing_a)
        if missing_s:
            notes.append("structural opcodes without silicon-agreement evidence: %s" % missing_s)
        if unk:
            notes.append("opcodes in neither class: %s" % unk)
        if conv:
            notes.append("conversions interpreted as CANDIDATES, never run: %s" % conv)
        if TENSOR_SETUP_SHIFT in self.used:
            notes.append("op17016 uses the exact setup configurations measured on lane inputs 0..31 "
                         "and eight retained controls; other inputs/configurations refuse")
        if self.used & {READ_SR_LANE, ADD_U16_U16, ADD_32_U16}:
            notes.append("the tensor lane prefix: SR130 modelled as the lane index; op10286 as u16 + u16 and op10283 "
                         "as 32-bit + u16 view, both widths measured on seven pairs (g17-tensor-width-runtime-v1); "
                         "every add's fused scale read from the decoder's token (handoff 9h)")
        if self.used & TENSOR_SETUP_LOGIC:
            # THE DOMAIN'S SIZE COMES FROM THE MODEL, NOT FROM THIS SENTENCE. This note used to read
            # "the retained 40-input domain" and a rewording dropped the number, which is the whole
            # informative part - "exact retained configurations/domains" does not tell a reader
            # WHICH domain the classification rests on, and the receipt is the only place that
            # statement reaches. It also went unnoticed because the number lived in prose in two
            # files at once. Reading len(DOMAIN) means the note cannot drift from the model it
            # describes: change the measured set and the sentence changes with it.
            import g17tensorlogicmodel
            notes.append('op423/op426 measured classifications require exact retained '
                         'configurations on the retained %d-input domain '
                         '(g17tensorlogicmodel.DOMAIN); the projection configurations carry their '
                         'own measured source sets; source availability is separate from their '
                         'value model' % len(g17tensorlogicmodel.DOMAIN))
        for o, (conf_o, why) in sorted(getattr(self, "outside_domain", {}).items()):
            notes.append("outside the measured domain (%s): %s" % (conf_o, why))
        for o, why in sorted(getattr(self, "conversion_evidence", {}).items()):
            notes.append("conversion named by execution, on this input only: %s" % why)
        for o in arith:
            if o in SEMANTICS:
                notes.append("op%d rounding is measured on %s only; bit-exact is claimed on those "
                             "inputs, bounded elsewhere" % (o, SEMANTICS[o][2]))
                if o in (3850, 3978):
                    # CORRECTED AT THE CLAIM. On the resident attention block's LayerNorm stage
                    # (results/g17-attention-resident-32x384-v1, code 56655bb5) the delivered
                    # program's outputs differ from this model in 923 of 12,288 words, by up to 32
                    # ulp of near-zero outputs, with a per-row factor within 1.0e-7 relative - so
                    # op3850 is NOT bit-exact against float32 1/sqrt beyond its measured
                    # inputs - and run alone it is one ulp low at 1.5 (isolated_rounding).
                    notes.append("op%d: on the resident LayerNorm run the per-row factor differs "
                                 "from this model by up to 1.0e-7 relative, and run alone it is "
                                 "one ulp low at 1.5; bit-exact holds on the %d measured inputs "
                                 "only, the rest is bounded" % (o, len(RSQRT_MEASURED_INPUTS)))
        outside = getattr(self, "outside_domain", {})
        if missing_a or missing_s or unk:
            return "unverified", notes
        other = [c for c, _w in outside.values() if c not in ("bounded", "candidate")]
        if other:
            return "unverified", notes
        if conv or any(c == "candidate" for c, _w in outside.values()):
            return "candidate", notes
        return ("bounded" if outside else "executed"), notes


class Threadgroup:
    """One threadgroup's scratchpad under LOCKSTEP execution, and what lockstep cannot see.

    All lanes advance one instruction at a time, so a barrier is a no-op for ordering here; what
    the driver CAN establish is what a correct program must satisfy under any interleaving the
    hardware chooses between barriers: a slot is never read before something wrote it; a lane
    never reads, in one barrier interval, a slot another lane wrote in the same interval (the
    read's order against that write is unestablished without a barrier between them); and a lane
    never overwrites, in one interval, a slot another lane read in the same interval. Those are
    refused as races. The scratchpad's bound is the contract's, in words."""

    def __init__(self, words, lanes=None):
        self.words = words
        self.lanes = lanes      # how many lanes a barrier waits for; None = however many have arrived
        self.mem = {}
        self.interval = 0
        self.written = {}       # idx -> (lane, interval) of the last write
        self.read = {}          # idx -> {lane: interval} of reads
        self.barriers = {}      # lane -> count
        self.race = []
        self.log = []           # (kind, lane, idx, interval, offset) in program order, for hazard reports

    def barrier(self, lane, offset):
        self.log.append(("barrier", lane, None, self.interval, offset))
        self.barriers[lane] = self.barriers.get(lane, 0) + 1
        # THE INTERVAL ADVANCES WHEN EVERY LANE HAS PASSED THIS BARRIER. Written first as "all
        # counts equal and this is the highest lane so far", which at the FIRST barrier is true
        # for every lane as it arrives (the dict grows one lane at a time), so the interval
        # advanced 32 times at one barrier and a three-barrier program reported 35 intervals.
        # The race verdicts did not move - every access after a barrier still shared one
        # interval value - but the count was wrong, and a hazard report that says so is the
        # reason it was found. The scratchpad now knows its lane count.
        arrived = len(self.barriers) if self.lanes is None else self.lanes
        if len(self.barriers) == arrived and len(set(self.barriers.values())) == 1 and lane == max(self.barriers):
            self.interval += 1

    def store(self, lane, idx, bits, offset):
        for other, when in self.read.get(idx, {}).items():
            if other != lane and when == self.interval:
                self.race.append("+%#x: lane %d writes word %d that lane %d read in the same barrier interval" % (offset, lane, idx, other))
        w = self.written.get(idx)
        if w and w[0] != lane and w[1] == self.interval:
            self.race.append("+%#x: lanes %d and %d both write word %d in one barrier interval" % (offset, lane, w[0], idx))
        self.mem[idx] = int(bits) & 0xFFFFFFFF
        self.written[idx] = (lane, self.interval)
        self.log.append(("store", lane, idx, self.interval, offset))

    def load(self, lane, idx, offset):
        if idx not in self.mem:
            raise ValueError("threadgroup read at +%#x: lane %d reads word %d before anything wrote it" % (offset, lane, idx))
        w = self.written[idx]
        if w[0] != lane and w[1] == self.interval:
            self.race.append("+%#x: lane %d reads word %d that lane %d wrote in the same barrier interval" % (offset, lane, idx, w[0]))
        self.read.setdefault(idx, {})[lane] = self.interval
        self.log.append(("load", lane, idx, self.interval, offset))
        return self.mem[idx]


def simulate_group(instructions, buffers, bindings, lanes, groups, words, *, executed=None, silicon=None,
                   coords=lambda group, lane: (lane, group), observer=None):
    """Lockstep threadgroups: `groups` groups of `lanes` lanes, each group with a `words`-word
    scratchpad; lane (x) reads SR160 = coords()[0], SR161 = coords()[1]. Every lane runs every
    instruction (no divergence is modelled: a divergent barrier would be refused by the ordering
    count). Returns (buffers, confidence, notes); a race or an unwritten read raises.
    `observer(group, threadgroup)`, if given, sees each group's scratchpad after its run - the
    access log, barrier counts and race list - so a caller can REPORT hazard facts, not only
    rely on the refusals."""
    executed = executed_opcodes() if executed is None else frozenset(executed)
    silicon = silicon_agreement_opcodes() if silicon is None else frozenset(silicon)
    bindings = [tuple(b)[:3] for b in bindings]
    notes, conf = [], "executed"
    instructions = list(instructions)
    for g in range(groups):
        tg = Threadgroup(words, lanes)
        machines = []
        for lane in range(lanes):
            x, y = coords(g, lane)
            m = Machine(bindings, buffers, x, executed, silicon)
            m.coord_y, m.group, m.lane = y, tg, lane
            machines.append(m)
        cursor = 0
        for offset, size, opcode, toks in instructions:
            if offset != cursor:
                raise ValueError("instruction boundaries are not consecutive at +%#x" % offset)
            cursor += size
            alive = False
            for m in machines:
                alive = m.step(offset, size, opcode, list(toks)) or alive
            if not alive:
                break
        if observer is not None:
            observer(g, tg)
        if len(set(tg.barriers.values())) > 1:
            raise ValueError("group %d: lanes passed different numbers of barriers %s" % (g, sorted(set(tg.barriers.values()))))
        if tg.race:
            raise ValueError("group %d: %d race(s) between barriers; first: %s" % (g, len(tg.race), tg.race[0]))
        for m in machines:
            for rank, addr, val in m.stores:
                if not bindings[rank][2]:
                    raise ValueError("lane %d stored to read-only rank %d" % (m.lane, rank))
                buffers[rank][addr] = val
            c, n = m.confidence()
            conf = weakest(conf, c)
            for x_ in n:
                if x_ not in notes:
                    notes.append(x_)
    notes.append("lockstep group model: barrier ordering is asserted by the race check, not measured; "
                 "asynchronous completion and divergence are not modelled")
    return buffers, conf, notes


# "candidate" sits BELOW isolation. An isolation-named opcode in this file is a float unary whose
# neighbours have been run and which can carry a measured bound; a candidate has never been run at
# all and carries none. The two consumers outside this file both handle a new value safely, which
# was checked rather than assumed: g17layernormimagecheck admits only ("executed", "bounded") and so
# refuses a candidate, and g17attentioninterpret.weakest_confidence raises "unknown interpreter
# confidence" on anything it does not list - loud, not a silent pass.
CONFIDENCE_ORDER = ("executed", "bounded", "isolation", "candidate", "unverified")


def weakest(*verdicts):
    """The program's verdict is its weakest thread's: executed < bounded < isolation < candidate
    < unverified."""
    return max(verdicts, key=CONFIDENCE_ORDER.index)


def _bindings_default(n=4):
    return [(1, 0, False), (2, 2, False), (3, 4, False), (4, 6, True)][:n]


def check_operands(instructions, bindings, allocation_words, *, rows=1):
    """STRUCTURE: every access decodes, names a delivered rank, and lies inside its allocation.

    `allocation_words` is USED - it is the size of every binding (an int) or per rank (a mapping or
    sequence) - and the addresses are computed the way the program computes them, by running its
    integer opcodes for each row. A four-column program checked against three words of allocation
    refuses at its fourth access. Loads read zeros; only the addresses matter here.
    """
    bindings = [tuple(b)[:3] for b in bindings]
    if isinstance(allocation_words, int):
        sizes = {r: allocation_words for r in range(len(bindings))}
    elif isinstance(allocation_words, dict):
        sizes = dict(allocation_words)
    else:
        sizes = {r: n for r, n in enumerate(allocation_words)}
    seen = {"loads": 0, "stores": 0, "rows": rows}
    for row in range(rows):
        bufs = {r: [0.0] * int(sizes.get(r, 0)) for r in range(len(bindings))}
        m = Machine(bindings, bufs, row, frozenset(), frozenset()).run(instructions)
        seen["loads"] += len(m.loads)
        seen["stores"] += len(m.stores)
    return seen


def simulate_grid(instructions, buffers, bindings, grid, *, executed=None, silicon=None):
    """A two-coordinate launch: `grid` is (x extent, y extent); thread (x, y) reads SR160 = x and
    SR161 = y. Otherwise simulate_threads."""
    executed = executed_opcodes() if executed is None else frozenset(executed)
    silicon = silicon_agreement_opcodes() if silicon is None else frozenset(silicon)
    bindings = [tuple(b)[:3] for b in bindings]
    nx, ny = grid
    notes, conf, fsel = [], "executed", False
    for y in range(ny):
        for x in range(nx):
            m = Machine(bindings, buffers, x, executed, silicon)
            m.coord_y = y
            m.run(instructions)
            for rank, addr, val in m.stores:
                if not bindings[rank][2]:
                    raise ValueError("thread (%d, %d) stored to read-only rank %d" % (x, y, rank))
                buffers[rank][addr] = val
            c, n = m.confidence()
            conf = weakest(conf, c)
            fsel = fsel or getattr(m, "notes_fselect", False)
            for x_ in n:
                if x_ not in notes:
                    notes.append(x_)
    if fsel:
        conf = "isolation"
    return buffers, conf, notes


def simulate_threads(instructions, buffers, bindings, threads, *, executed=None, silicon=None, launch=None):
    """ANY straight-line program, one Machine per thread id: `buffers` is {rank: list of float32}
    shared across threads (a thread's stores land before the next thread runs, so a program whose
    threads write disjoint elements - every stage here - reads back what it should). Returns
    (buffers after all threads, confidence, notes). The written ranks come from `bindings`."""
    executed = executed_opcodes() if executed is None else frozenset(executed)
    silicon = silicon_agreement_opcodes() if silicon is None else frozenset(silicon)
    bindings = [tuple(b)[:3] for b in bindings]
    geometry = None
    if launch is not None:
        if not isinstance(launch, dict) or set(launch) != {'grid', 'threadgroup'}:
            raise ValueError('explicit launch needs exact grid and threadgroup fields')
        grid, group = launch['grid'], launch['threadgroup']
        if (any(not isinstance(v, (list, tuple)) or len(v) != 3 or
                any(type(n) is not int or n <= 0 for n in v) for v in (grid, group))
                or type(threads) is not int or grid[0]*grid[1]*grid[2] != threads):
            raise ValueError('explicit launch dimensions differ from thread count')
        geometry = grid, group
    notes, conf, fsel = [], "executed", False
    for t in range(threads):
        m = Machine(bindings, buffers, t, executed, silicon)
        m.single_thread_atomic_model = (threads == 1 and geometry is not None and
            tuple(geometry[0]) == (1,1,1) and tuple(geometry[1]) == (1,1,1))
        if geometry is not None:
            grid, group = geometry
            x, y = t % grid[0], (t // grid[0]) % grid[1]
            z = t // (grid[0]*grid[1])
            # Measured selectors, not neighboring-number inference: SR156/157/164
            # decode as reg54/56/26 in the actual half-add delivery. SR160/161's
            # reg61/62 are already interpreted above. The unchanged syn-s7f595f1cd1
            # delivery at offset208 has bytes6c9e1006: compiler SR158, decoder reg58.
            # No other selector is assumed.
            m.launch_coordinates = {'reg:54': x//group[0], 'reg:56': y//group[1],
                                    'reg:58': z//group[2], 'reg:26': x%group[0],
                                    'reg:61': x, 'reg:62': y}
        m.run(instructions)
        for rank, addr, val in m.stores:
            if not bindings[rank][2]:
                raise ValueError("thread %d stored to read-only rank %d" % (t, rank))
            buffers[rank][addr] = val
        c, n = m.confidence()
        conf = weakest(conf, c)
        fsel = fsel or getattr(m, "notes_fselect", False)
        for x_ in n:
            if x_ not in notes:
                notes.append(x_)
    if fsel:
        conf = "isolation"
        notes.append("op9700 (fselect) is named by Apple's selection of it for max()/min() and has "
                     "never executed on this backend")
    return buffers, conf, notes


def simulate_layernorm(instructions, x, gamma, beta, eps, *, executed=None, bindings=None,
                       silicon=None):
    """Interpret a delivered FP32 LayerNorm from its bytes for ONE ROW. -> (outputs, confidence, notes)

    outputs is {column: float32}. `x`, `gamma` and `beta` are the row and the parameters; the four
    bindings are taken in the delivered rank order source, gamma, beta, output unless `bindings` says
    otherwise. Use simulate_rows() for a whole matrix.
    """
    executed = executed_opcodes() if executed is None else frozenset(executed)
    silicon = silicon_agreement_opcodes() if silicon is None else frozenset(silicon)
    bindings = [tuple(b)[:3] for b in (bindings or _bindings_default())]
    n = len(x)
    bufs = {0: list(x), 1: list(gamma), 2: list(beta), 3: [float("nan")] * n}
    m = Machine(bindings, bufs, 0, executed, silicon).run(instructions)
    outputs = {}
    for rank, addr, val in m.stores:
        if rank != 3:
            raise ValueError("a store landed on rank %d; the output is rank 3" % rank)
        outputs[addr] = float(val)
    conf, notes = m.confidence()
    return outputs, conf, notes


def simulate_rows(instructions, X, gamma, beta, eps, *, executed=None, bindings=None, silicon=None,
                  waw_model="in_order"):
    """The whole matrix: one Machine per row with the thread id set to the row. -> (2-D outputs,
    confidence, notes). A row that stores outside its own slice is refused."""
    import numpy as np
    executed = executed_opcodes() if executed is None else frozenset(executed)
    silicon = silicon_agreement_opcodes() if silicon is None else frozenset(silicon)
    bindings = [tuple(b)[:3] for b in (bindings or _bindings_default())]
    X = np.asarray(X, np.float32)
    rows, cols = X.shape
    out = np.full(X.shape, np.nan, np.float32)
    notes, conf = [], "executed"
    for r in range(rows):
        bufs = {0: X.reshape(-1).tolist(), 1: list(gamma), 2: list(beta), 3: [float("nan")] * (rows * cols)}
        m = Machine(bindings, bufs, r, executed, silicon, waw_model=waw_model).run(instructions)
        for rank, addr, val in m.stores:
            if rank != 3 or not (r * cols <= addr < (r + 1) * cols):
                raise ValueError("row %d stored to rank %d element %d, outside its own slice" % (r, rank, addr))
            out[r, addr - r * cols] = val
        c, n = m.confidence()
        conf = weakest(conf, c)
        for x_ in n:
            if x_ not in notes:
                notes.append(x_)
    return out, conf, notes


def reference_layernorm(x, gamma, beta, eps):
    """The application reference in FP64 over exact FP32 inputs."""
    m = sum(float(v) for v in x) / len(x)
    var = sum((float(v) - m) ** 2 for v in x) / len(x)
    inv = 1.0 / math.sqrt(var + eps)
    return [(float(v) - m) * inv * float(g) + float(b) for v, g, b in zip(x, gamma, beta)]


def report():
    executed = executed_opcodes()
    silicon = silicon_agreement_opcodes()
    print(__doc__.split("\n\n")[0])
    print("\n   opcodes named by execution (opsem):           %d" % len(executed))
    print("   opcodes with silicon agreement (end-to-end):  %d" % len(silicon))
    for op in sorted(RSQRT_OPCODES):
        ok, why = arithmetic_status(op, executed)
        print("\n   op%-6d %-8s arithmetic interpretable: %s\n      %s" % (op, RSQRT_OPCODES[op], ok, why[:150]))
    need = sorted(ARITH_FP | STRUCTURAL)
    print("\n   the LayerNorm program's opcodes and their evidence class:")
    for op in need:
        cls = ("execution" if op in executed else "") + (" silicon-agreement" if op in silicon else "")
        print("      op%-6d %s" % (op, cls.strip() or "NONE"))
    return 0 if all(arithmetic_status(o, executed)[0] for o in RSQRT_OPCODES) else 2


if __name__ == "__main__":
    sys.exit(report())


def range_store_load_hazards(instructions):
    """The consumers that returned zero on silicon: a range store (op17244/17253/17262) whose tuple
    members' last writer is a LOAD (op12682 or the threadgroup load), with nothing in between that
    waited. Returns [(offset, opcode, register)] per offending member. On the retained failed
    read-control bytes (results/g17-rangeread-runtime-v1, 4e8977ec5653f65e) this names nine; on
    the corrected compile, none. The interpreter itself cannot see the hazard, because it
    completes every load synchronously - this is the static rule the failure established."""
    last_writer = {}
    found = []
    for offset, _size, opcode, toks in instructions:
        regs = _regs(toks)
        if opcode in RANGE_STORES:
            n = RANGE_STORES[opcode]
            if regs:
                r = int(regs[0].split(":")[1]) - TUPLE_BASE[n]
                for k in range(n):
                    w = last_writer.get(105 + r + k)
                    if w in (LOAD32, TG_LOAD):
                        found.append((offset, opcode, 105 + r + k))
            continue
        if regs and opcode not in STORES32 and opcode != END:
            try:
                last_writer[int(regs[0].split(":")[1])] = opcode
            except ValueError:
                pass
    return found
